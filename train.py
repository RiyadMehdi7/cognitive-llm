"""
train.py - Self-contained training script for Cognitive LLM ablation experiments.

All configuration lives in the CONFIG dict below.
The research agent modifies only CONFIG between experiments.

Usage:
    python train.py

Outputs:
    val_loss: X.XXXXXX      (printed on success)
    gsm8k_acc: X.XX         (printed on success when GSM8K eval is enabled)
    CRASH: OOM              (printed on torch.cuda.OutOfMemoryError)
    CRASH: NaN              (printed on NaN loss)
    CRASH: <message>        (printed on any other exception)
"""

import argparse
import sys

# -----------------------------------------------------------------------------
# CONFIG - the agent modifies only this dict between experiments
# -----------------------------------------------------------------------------
CONFIG = {
    # Runtime / model selection
    "model_id": "HuggingFaceTB/SmolLM-360M",
    "device": "auto",
    "quantization": "none",
    "dtype": "auto",

    # Block flags
    "use_block1": False,
    "use_block2": False,
    "use_block3": False,
    "use_block4": False,
    "use_block5": False,
    "use_block6": False,

    # Training hyperparameters
    "learning_rate": 2e-4,
    "max_steps": 100,
    "warmup_steps": 10,
    "gradient_accumulation": 4,
    "max_grad_norm": 1.0,
    "batch_size": 4,
    "max_seq_len": 512,
    "gradient_checkpointing": True,

    # Loss weights
    "lambda_surprise": 0.01,
    "lambda_critic": 0.1,
    "lambda_predictive": 0.05,

    # Eval settings
    "gsm8k_eval_samples": 0,
    "n_baseline_runs": 1,
    "val_subset_size": 64,
    "max_generation_tokens": 32,

    # Vanilla mode: skip LoRA, eval-only (no training)
    "skip_lora": False,
}
# -----------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments and merge YAML config into CONFIG."""
    parser = argparse.ArgumentParser(description="Cognitive LLM training script")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file (overrides CONFIG defaults)")
    parser.add_argument("--max_train_samples", type=int, default=0,
                        help="Limit training dataset size (0 = use all)")
    parser.add_argument("--skip_entropy_init", action="store_true",
                        help="Skip entropy-based weight initialization")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable wandb logging")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Wandb run name (auto-generated if not set)")
    parser.add_argument("--eval_benchmarks", action="store_true",
                        help="Run lm-eval-harness benchmarks after training")
    return parser.parse_args()


def _apply_yaml_config(yaml_path: str) -> None:
    """Load a YAML config file and merge into CONFIG."""
    import yaml

    with open(yaml_path) as f:
        yaml_cfg = yaml.safe_load(f)

    if yaml_cfg is None:
        return

    def _coerce_numeric(key, value):
        """YAML safe_load parses '2e-4' as string; coerce to float if CONFIG expects a number."""
        if isinstance(value, str) and key in CONFIG and isinstance(CONFIG[key], (int, float)):
            try:
                return type(CONFIG[key])(float(value))
            except (ValueError, TypeError):
                return value
        return value

    # Flatten nested 'training' and 'lambda_config' sections
    training = yaml_cfg.pop("training", {}) or {}
    lambda_cfg = yaml_cfg.pop("lambda_config", {}) or {}

    # Map lambda_config keys to CONFIG's flat keys
    for key, value in lambda_cfg.items():
        flat_key = f"lambda_{key}"
        CONFIG[flat_key] = _coerce_numeric(flat_key, value)

    # Map training keys directly
    for key, value in training.items():
        CONFIG[key] = _coerce_numeric(key, value)

    # Map top-level keys (model_id, device, use_block*, etc.)
    for key, value in yaml_cfg.items():
        if key == "dataset":
            continue  # dataset is handled by the data loading logic
        CONFIG[key] = _coerce_numeric(key, value)


def _get_torch_dtype(dtype_name, runtime_device):
    import torch

    if dtype_name == "auto":
        if runtime_device.type == "xla":
            return torch.bfloat16
        if runtime_device.type == "cuda":
            capability = torch.cuda.get_device_capability(0)
            return torch.bfloat16 if capability[0] >= 8 else torch.float16
        return torch.float32

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    if dtype_name not in dtype_map:
        raise ValueError(
            f"Unsupported dtype={dtype_name!r}; expected one of "
            f"{sorted(dtype_map)} or 'auto'."
        )
    return dtype_map[dtype_name]


def _resolve_runtime():
    import torch

    from cognitive_llm.training.device import resolve_device

    runtime_device = resolve_device(CONFIG.get("device", "auto"))
    compute_dtype = _get_torch_dtype(CONFIG.get("dtype", "auto"), runtime_device)

    if runtime_device.type == "cuda":
        accelerator = torch.cuda.get_device_name(0)
    elif runtime_device.type == "xla":
        accelerator = "TPU/XLA"
    else:
        accelerator = str(runtime_device)

    print(
        f"Device: {runtime_device}  accelerator: {accelerator}  dtype: {compute_dtype}",
        flush=True,
    )
    return runtime_device, compute_dtype


def _load_base_model(model_id, runtime_device, compute_dtype):
    from transformers import AutoModelForCausalLM

    quantization = str(CONFIG.get("quantization", "none")).lower()
    use_gradient_checkpointing = (
        CONFIG.get("gradient_checkpointing", False)
        and runtime_device.type != "xla"
    )
    model_kwargs = {
        "torch_dtype": compute_dtype,
        "trust_remote_code": True,
    }

    if quantization in {"4bit", "4-bit"}:
        if runtime_device.type != "cuda":
            raise RuntimeError(
                "4-bit quantization requires CUDA; set CONFIG['quantization'] = "
                "'none' for TPU/XLA runs."
            )
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs.update({
            "quantization_config": bnb_config,
            "device_map": "auto",
        })
    elif quantization != "none":
        raise ValueError(
            f"Unsupported quantization={quantization!r}; expected 'none' or '4bit'."
        )
    else:
        model_kwargs["low_cpu_mem_usage"] = True

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    if use_gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    return model


def _set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        import torch_xla.core.xla_model as xm
        xm.set_rng_state(seed)
    except (ImportError, RuntimeError):
        pass


def _run_training() -> None:
    import gc
    import math
    import re

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from torch.utils.data import DataLoader, random_split
    from transformers import AutoTokenizer

    from cognitive_llm.models.cognitive_model import CognitiveModel
    from cognitive_llm.training.device import (
        mark_step,
        move_batch_to_device,
        move_model_to_device,
        wrap_dataloader,
    )
    from cognitive_llm.training.trainer import CognitiveTrainer

    # -- Seed -----------------------------------------------------------------
    seed = int(CONFIG.get("_seed", 42))
    _set_seed(seed)
    print(f"Seed: {seed}", flush=True)

    # -- Device / dtype -------------------------------------------------------
    runtime_device, compute_dtype = _resolve_runtime()

    # -- Load tokenizer -------------------------------------------------------
    model_id = CONFIG["model_id"]
    print("Stage: loading tokenizer", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -- Tokenize GSM8K train split -------------------------------------------
    max_seq_len = CONFIG["max_seq_len"]
    print("Stage: loading GSM8K train split", flush=True)
    max_train_samples = int(CONFIG.get("_max_train_samples", 0))
    if max_train_samples > 0:
        raw_train = load_dataset("gsm8k", "main", split=f"train[:{max_train_samples}]")
    else:
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

    print("Stage: tokenizing GSM8K train split", flush=True)
    tokenized = raw_train.map(tokenize, batched=True, remove_columns=raw_train.column_names)
    tokenized.set_format(type="torch")

    # Split 90/10 train/val with fixed seed for reproducibility
    n_val = max(16, int(len(tokenized) * 0.1))
    n_train = len(tokenized) - n_val
    train_ds, val_ds = random_split(
        tokenized, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    val_subset_size = int(CONFIG.get("val_subset_size", n_val))
    if val_subset_size > 0 and len(val_ds) > val_subset_size:
        _, val_ds = random_split(
            val_ds,
            [len(val_ds) - val_subset_size, val_subset_size],
            generator=torch.Generator().manual_seed(42),
        )
        print(f"Stage: capped validation subset to {len(val_ds)} examples", flush=True)

    batch_size = CONFIG["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # -- Load base model ------------------------------------------------------
    print("Stage: loading base model", flush=True)
    base_model = _load_base_model(model_id, runtime_device, compute_dtype)

    # -- Vanilla mode: skip LoRA, eval-only (no training) ----------------------
    skip_lora = CONFIG.get("skip_lora", False)
    quantization = str(CONFIG.get("quantization", "none")).lower()

    if not skip_lora:
        # Full-precision LoRA works on TPU; k-bit prep stays CUDA-only.
        if quantization in {"4bit", "4-bit"}:
            base_model = prepare_model_for_kbit_training(base_model)
        elif CONFIG.get("gradient_checkpointing", False):
            if hasattr(base_model, "enable_input_require_grads"):
                base_model.enable_input_require_grads()
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
    model = move_model_to_device(model, runtime_device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}", flush=True)
    print(f"Active blocks: {[k for k, v in block_config.items() if v]}", flush=True)
    print(f"Mode: {'vanilla (eval-only)' if skip_lora else 'LoRA + train'}", flush=True)

    if skip_lora:
        # Vanilla baseline: just evaluate, no training
        from cognitive_llm.training.trainer import compute_total_loss

        lambda_config = {
            "surprise": CONFIG["lambda_surprise"],
            "critic": CONFIG["lambda_critic"],
            "predictive": CONFIG["lambda_predictive"],
        }
        device = next(model.parameters()).device
        eval_loader = wrap_dataloader(val_loader, device)
        model.eval()
        total_loss = 0.0
        n_batches = 0
        print("Stage: running vanilla validation", flush=True)
        with torch.no_grad():
            for batch in eval_loader:
                batch = move_batch_to_device(batch, device)
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    labels=batch.get("labels", batch["input_ids"].clone()),
                )
                total_loss += compute_total_loss(outputs, lambda_config).item()
                n_batches += 1
                mark_step(device)
        val_loss = total_loss / max(n_batches, 1)
    else:
        # -- Lambda config ----------------------------------------------------
        lambda_config = {
            "surprise": CONFIG["lambda_surprise"],
            "critic": CONFIG["lambda_critic"],
            "predictive": CONFIG["lambda_predictive"],
        }

        training_config = {
            "device": CONFIG.get("device", "auto"),
            "learning_rate": CONFIG["learning_rate"],
            "max_steps": CONFIG["max_steps"],
            "warmup_steps": CONFIG["warmup_steps"],
            "gradient_accumulation": CONFIG["gradient_accumulation"],
            "max_grad_norm": CONFIG["max_grad_norm"],
            "eval_every_n_steps": CONFIG.get("eval_every_n_steps", CONFIG["max_steps"] + 1),
            "save_every_n_steps": CONFIG.get("save_every_n_steps", CONFIG["max_steps"] + 1),
            "checkpoint_dir": CONFIG.get("checkpoint_dir", "./checkpoints"),
            "use_wandb": CONFIG.get("use_wandb", False),
            "wandb_group": CONFIG.get("wandb_group"),
            "wandb_tags": CONFIG.get("wandb_tags"),
            "run_name": CONFIG.get("_run_name"),
            "seed": CONFIG.get("_seed", 42),
        }

        # -- Train ------------------------------------------------------------
        trainer = CognitiveTrainer(
            model=model,
            train_dataloader=train_loader,
            eval_dataloader=val_loader,
            config=training_config,
            lambda_config=lambda_config,
        )
        print("Stage: starting training loop", flush=True)
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
    n_eval = int(CONFIG["gsm8k_eval_samples"])
    gsm8k_acc = None
    if n_eval > 0:
        print(f"Stage: GSM8K eval on {n_eval} samples", flush=True)
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
                for _ in range(int(CONFIG.get("max_generation_tokens", 64))):
                    fwd = model(input_ids=generated_ids, attention_mask=gen_mask)
                    next_tok = fwd["logits"][:, -1, :].argmax(dim=-1, keepdim=True)
                    generated_ids = torch.cat([generated_ids, next_tok], dim=1)
                    gen_mask = torch.cat([gen_mask, torch.ones_like(next_tok)], dim=1)
                    mark_step(device)
                    if next_tok.item() == tokenizer.eos_token_id:
                        break

            response = tokenizer.decode(
                generated_ids[0][input_ids.shape[1]:].detach().cpu(),
                skip_special_tokens=True,
            )
            pred_match = re.search(r"####\s*([-\d,\.]+)", response)
            gold_match = re.search(r"####\s*([-\d,\.]+)", example["answer"])
            if (pred_match and gold_match
                    and pred_match.group(1).replace(",", "") == gold_match.group(1).replace(",", "")):
                correct += 1

        gsm8k_acc = correct / n_eval * 100
    else:
        print("Stage: GSM8K eval skipped for screening run", flush=True)

    # -- Print results (agent extracts these lines) ---------------------------
    print(f"val_loss: {val_loss:.6f}")
    if gsm8k_acc is not None:
        print(f"gsm8k_acc: {gsm8k_acc:.2f}")

    # -- Benchmark eval (optional, via lm-eval-harness) -----------------------
    if CONFIG.get("_eval_benchmarks", False):
        print("Stage: running lm-eval-harness benchmarks", flush=True)
        from cognitive_llm.evaluation.benchmark import BenchmarkRunner

        # Save model for eval
        checkpoint_dir = CONFIG.get("checkpoint_dir", "./checkpoints")
        eval_model_path = f"{checkpoint_dir}/eval_model"
        try:
            model.base.save_pretrained(eval_model_path)
            tokenizer.save_pretrained(eval_model_path)

            runner = BenchmarkRunner(
                model_path=eval_model_path,
                output_dir=f"{checkpoint_dir}/benchmark_results",
                device=str(runtime_device),
                batch_size=CONFIG.get("batch_size", 8),
            )
            results = runner.run_all(
                tasks=["gsm8k", "arc_challenge", "hellaswag", "mathqa"]
            )
            runner.print_results(results)

            # Print in parseable format
            for r in results:
                print(f"bench_{r.task}_{r.metric}: {r.score:.4f}")
        except Exception as exc:
            print(f"Benchmark eval failed: {exc}", flush=True)

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
    # Parse CLI args and merge YAML config before running
    _cli_args = _parse_args()

    if _cli_args.config:
        _apply_yaml_config(_cli_args.config)

    if _cli_args.no_wandb:
        CONFIG["use_wandb"] = False

    if _cli_args.max_train_samples > 0:
        CONFIG["_max_train_samples"] = _cli_args.max_train_samples

    CONFIG["_skip_entropy_init"] = _cli_args.skip_entropy_init
    CONFIG["_seed"] = _cli_args.seed
    CONFIG["_eval_benchmarks"] = _cli_args.eval_benchmarks

    if _cli_args.run_name:
        CONFIG["_run_name"] = _cli_args.run_name

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
