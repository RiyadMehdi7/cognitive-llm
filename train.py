"""
train.py - Self-contained training script for Cognitive LLM ablation experiments.

All configuration lives in the CONFIG dict below.
The research agent modifies only CONFIG between experiments.

Usage:
    python train.py

Outputs:
    val_loss: X.XXXXXX      (printed on success)
    gsm8k_acc: X.XX         (printed on success, % on 50 test samples)
    CRASH: OOM              (printed on torch.cuda.OutOfMemoryError)
    CRASH: NaN              (printed on NaN loss)
    CRASH: <message>        (printed on any other exception)
"""

import sys

# -----------------------------------------------------------------------------
# CONFIG - the agent modifies only this dict between experiments
# -----------------------------------------------------------------------------
CONFIG = {
    # Block flags
    "use_block1": False,
    "use_block2": False,
    "use_block3": False,
    "use_block4": False,
    "use_block5": False,
    "use_block6": False,

    # Training hyperparameters
    "learning_rate": 2e-4,
    "max_steps": 1000,
    "warmup_steps": 100,
    "gradient_accumulation": 4,
    "max_grad_norm": 1.0,
    "batch_size": 4,
    "max_seq_len": 512,

    # Loss weights
    "lambda_surprise": 0.01,
    "lambda_critic": 0.1,
    "lambda_predictive": 0.05,

    # Eval settings
    "gsm8k_eval_samples": 50,
    "n_baseline_runs": 3,

    # Vanilla mode: skip LoRA, eval-only (no training)
    "skip_lora": False,
}
# -----------------------------------------------------------------------------


def _run_training() -> None:
    import gc
    import math
    import re

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from torch.utils.data import DataLoader, random_split
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from cognitive_llm.models.cognitive_model import CognitiveModel
    from cognitive_llm.training.trainer import CognitiveTrainer

    # -- Device / dtype -------------------------------------------------------
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required - no GPU found")
    capability = torch.cuda.get_device_capability(0)
    compute_dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16
    print(f"GPU: {torch.cuda.get_device_name(0)}  dtype: {compute_dtype}")

    # -- Load tokenizer -------------------------------------------------------
    model_id = "HuggingFaceTB/SmolLM3-3B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -- Tokenize GSM8K train split -------------------------------------------
    max_seq_len = CONFIG["max_seq_len"]
    raw_train = load_dataset("gsm8k", "main", split="train")

    def tokenize(examples):
        texts = [
            f"Question: {q}\nAnswer: {a}"
            for q, a in zip(examples["question"], examples["answer"])
        ]
        out = tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_len,
            padding="max_length",
        )
        out["labels"] = [ids.copy() for ids in out["input_ids"]]
        return out

    tokenized = raw_train.map(tokenize, batched=True, remove_columns=raw_train.column_names)
    tokenized.set_format(type="torch")

    # Split 90/10 train/val with fixed seed for reproducibility
    n_val = max(16, int(len(tokenized) * 0.1))
    n_train = len(tokenized) - n_val
    train_ds, val_ds = random_split(
        tokenized, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    batch_size = CONFIG["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # -- Load base model with 4-bit quantization ------------------------------
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=compute_dtype,
    )

    # -- Vanilla mode: skip LoRA, eval-only (no training) ----------------------
    skip_lora = CONFIG.get("skip_lora", False)

    if not skip_lora:
        # -- Apply LoRA (same config as notebooks/colab_debug.ipynb) ----------
        base_model = prepare_model_for_kbit_training(base_model)
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        base_model = get_peft_model(base_model, lora_config)

    # -- Wrap with CognitiveModel ---------------------------------------------
    block_config = {k: CONFIG[k] for k in CONFIG if k.startswith("use_block")}
    model = CognitiveModel(base_model, block_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")
    print(f"Active blocks: {[k for k, v in block_config.items() if v]}")
    print(f"Mode: {'vanilla (eval-only)' if skip_lora else 'LoRA + train'}")

    if skip_lora:
        # Vanilla baseline: just evaluate, no training
        from cognitive_llm.training.trainer import compute_total_loss
        from cognitive_llm.training.device import move_batch_to_device

        lambda_config = {
            "surprise": CONFIG["lambda_surprise"],
            "critic": CONFIG["lambda_critic"],
            "predictive": CONFIG["lambda_predictive"],
        }
        device = next(model.parameters()).device
        model.eval()
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = move_batch_to_device(batch, device)
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    labels=batch.get("labels", batch["input_ids"].clone()),
                )
                total_loss += compute_total_loss(outputs, lambda_config).item()
                n_batches += 1
        val_loss = total_loss / max(n_batches, 1)
    else:
        # -- Lambda config ----------------------------------------------------
        lambda_config = {
            "surprise": CONFIG["lambda_surprise"],
            "critic": CONFIG["lambda_critic"],
            "predictive": CONFIG["lambda_predictive"],
        }

        training_config = {
            "device": "auto",
            "learning_rate": CONFIG["learning_rate"],
            "max_steps": CONFIG["max_steps"],
            "warmup_steps": CONFIG["warmup_steps"],
            "gradient_accumulation": CONFIG["gradient_accumulation"],
            "max_grad_norm": CONFIG["max_grad_norm"],
            "eval_every_n_steps": CONFIG["max_steps"] + 1,
            "save_every_n_steps": CONFIG["max_steps"] + 1,
            "use_wandb": False,
        }

        # -- Train ------------------------------------------------------------
        trainer = CognitiveTrainer(
            model=model,
            train_dataloader=train_loader,
            eval_dataloader=val_loader,
            config=training_config,
            lambda_config=lambda_config,
        )
        losses = trainer.train()

        # Check for NaN in final training loss
        if losses and math.isnan(losses[-1]):
            print("CRASH: NaN")
            sys.exit(1)

        # -- Val loss (on held-out GSM8K train split) -------------------------
        val_loss = trainer._run_assessment()

    if math.isnan(val_loss):
        print("CRASH: NaN")
        sys.exit(1)

    # -- GSM8K accuracy on test set -------------------------------------------
    n_eval = CONFIG["gsm8k_eval_samples"]
    test_raw = load_dataset("gsm8k", "main", split=f"test[:{n_eval}]")
    if not skip_lora:
        device = trainer.device
    else:
        device = next(model.parameters()).device

    model.eval()
    correct = 0
    for example in test_raw:
        prompt = f"Question: {example['question']}\nAnswer:"
        inputs = tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=max_seq_len,
        )
        input_ids = inputs["input_ids"].to(device)
        attn_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            generated_ids = input_ids.clone()
            gen_mask = attn_mask.clone()
            for _ in range(64):
                fwd = model(input_ids=generated_ids, attention_mask=gen_mask)
                next_tok = fwd["logits"][:, -1, :].argmax(dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_tok], dim=1)
                gen_mask = torch.cat([gen_mask, torch.ones_like(next_tok)], dim=1)
                if next_tok.item() == tokenizer.eos_token_id:
                    break

        response = tokenizer.decode(
            generated_ids[0][input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        pred_match = re.search(r"####\s*([-\d,\.]+)", response)
        gold_match = re.search(r"####\s*([-\d,\.]+)", example["answer"])
        if (pred_match and gold_match
                and pred_match.group(1).replace(",", "") == gold_match.group(1).replace(",", "")):
            correct += 1

    gsm8k_acc = correct / n_eval * 100

    # -- Print results (agent extracts these lines) ---------------------------
    print(f"val_loss: {val_loss:.6f}")
    print(f"gsm8k_acc: {gsm8k_acc:.2f}")

    # Cleanup
    del model, base_model
    gc.collect()
    torch.cuda.empty_cache()


def _oom_safe_main() -> None:
    """Wraps _run_training so torch is imported before the OOM except clause."""
    import torch
    try:
        _run_training()
    except torch.cuda.OutOfMemoryError:
        print("CRASH: OOM")
        sys.exit(1)


if __name__ == "__main__":
    try:
        _oom_safe_main()
    except SystemExit:
        raise
    except Exception as exc:
        msg = str(exc)
        if "nan" in msg.lower():
            print("CRASH: NaN")
        else:
            print(f"CRASH: {msg}")
        sys.exit(1)
