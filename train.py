"""
train.py — Self-contained training script for Cognitive LLM ablation experiments.

All configuration lives in the CONFIG dict below.
The research agent modifies only CONFIG between experiments.

Usage:
    python train.py

Outputs:
    val_loss: X.XXXXXX      (printed on success)
    CRASH: OOM              (printed on torch.cuda.OutOfMemoryError)
    CRASH: NaN              (printed on NaN loss)
    CRASH: <message>        (printed on any other exception)
"""

import sys

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — the agent modifies only this dict between experiments
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    # Block flags
    "use_block1": True,
    "use_block2": False,
    "use_block3": False,
    "use_block4": False,   # NEVER enable — Phase 2 only
    "use_block5": False,
    "use_block6": True,    # ALWAYS ON when any other block is ON

    # Training hyperparameters
    "learning_rate": 2e-4,
    "max_steps": 500,
    "warmup_steps": 100,
    "gradient_accumulation": 4,
    "max_grad_norm": 1.0,
    "batch_size": 4,
    "max_seq_len": 512,

    # Loss weights
    "lambda_surprise": 0.01,
    "lambda_critic": 0.1,
    "lambda_predictive": 0.05,
}
# ─────────────────────────────────────────────────────────────────────────────


def _run_training() -> None:
    import gc
    import math

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from torch.utils.data import DataLoader, random_split
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from cognitive_llm.models.cognitive_model import CognitiveModel
    from cognitive_llm.training.trainer import CognitiveTrainer

    # ── Device / dtype ────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    capability = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
    compute_dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16

    # ── Load tokenizer ────────────────────────────────────────────────────────
    model_id = "HuggingFaceTB/SmolLM3-3B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Tokenize GSM8K ────────────────────────────────────────────────────────
    max_seq_len = CONFIG["max_seq_len"]
    raw = load_dataset("gsm8k", "main", split="train")

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

    tokenized = raw.map(tokenize, batched=True, remove_columns=raw.column_names)
    tokenized.set_format(type="torch")

    # Split 90/10 train/val
    n_val = max(1, int(len(tokenized) * 0.1))
    n_train = len(tokenized) - n_val
    train_ds, val_ds = random_split(
        tokenized, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    batch_size = CONFIG["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # ── Load base model with 4-bit quantization ───────────────────────────────
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

    # ── Apply LoRA (same config as notebooks/colab_debug.ipynb) ──────────────
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

    # ── Wrap with CognitiveModel ──────────────────────────────────────────────
    block_config = {k: CONFIG[k] for k in CONFIG if k.startswith("use_block")}
    model = CognitiveModel(base_model, block_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")
    print(f"Active blocks: {[k for k, v in block_config.items() if v]}")

    # ── Build lambda_config for CognitiveTrainer ──────────────────────────────
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
        "eval_every_n_steps": CONFIG["max_steps"] + 1,  # skip mid-training eval
        "save_every_n_steps": CONFIG["max_steps"] + 1,  # skip mid-training checkpoint
        "use_wandb": False,
    }

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = CognitiveTrainer(
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=val_loader,
        config=training_config,
        lambda_config=lambda_config,
    )
    losses = trainer.train()

    # Check for NaN in training losses
    if losses and math.isnan(losses[-1]):
        print("CRASH: NaN")
        sys.exit(1)

    # ── Evaluate val_loss ─────────────────────────────────────────────────────
    val_loss = trainer._run_assessment()

    if math.isnan(val_loss):
        print("CRASH: NaN")
        sys.exit(1)

    print(f"val_loss: {val_loss:.6f}")

    # Cleanup
    del model, trainer, base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    try:
        _run_training()
    except torch.cuda.OutOfMemoryError:
        print("CRASH: OOM")
        sys.exit(1)
    except Exception as exc:
        msg = str(exc)
        if "nan" in msg.lower() or "NaN" in msg:
            print("CRASH: NaN")
        else:
            print(f"CRASH: {msg}")
        sys.exit(1)
