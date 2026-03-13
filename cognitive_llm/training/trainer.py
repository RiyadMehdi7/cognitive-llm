"""
Custom training loop for Cognitive LLM with multi-loss optimization.

Handles the weighted combination of LM loss and auxiliary cognitive block losses,
gradient clipping, checkpointing, and wandb logging.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from cognitive_llm.training.device import (
    is_primary_process,
    mark_step,
    move_batch_to_device,
    move_model_to_device,
    optimizer_step,
    resolve_device,
    save_checkpoint,
    wrap_dataloader,
)

try:
    import wandb
except ImportError:
    wandb = None


# Recommended lambda starting values
LAMBDA_CONFIG = {
    "surprise": 0.01,  # Block 1 auxiliary
    "critic": 0.1,  # Block 3 actor-critic
    "predictive": 0.05,  # Block 4 pred coding aux
}


def compute_total_loss(
    outputs: dict, lambda_config: dict | None = None
) -> torch.Tensor:
    """
    Compute weighted combination of all block losses.

    Args:
        outputs: Dict from CognitiveModel.forward().
        lambda_config: Loss weights. Uses LAMBDA_CONFIG defaults if None.

    Returns:
        Scalar total loss tensor.
    """
    if lambda_config is None:
        lambda_config = LAMBDA_CONFIG

    loss = outputs["lm_loss"]

    if outputs["surprise_loss"] is not None:
        loss = loss + lambda_config["surprise"] * outputs["surprise_loss"]

    if outputs["critic_losses"]:
        critic_loss = torch.stack(outputs["critic_losses"]).mean()
        loss = loss + lambda_config["critic"] * critic_loss

    if outputs["pred_losses"]:
        valid_pred_losses = [p for p in outputs["pred_losses"] if p.requires_grad]
        if valid_pred_losses:
            pred_loss = torch.stack(valid_pred_losses).mean()
            loss = loss + lambda_config["predictive"] * pred_loss

    return loss


class CognitiveTrainer:
    """
    Training loop for the Cognitive LLM.

    Args:
        model: CognitiveModel instance.
        train_dataloader: DataLoader for training data.
        eval_dataloader: Optional DataLoader for assessment.
        config: Training configuration dict.
        lambda_config: Loss weight configuration.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader | None = None,
        config: dict | None = None,
        lambda_config: dict | None = None,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config or {}
        self.lambda_config = lambda_config or LAMBDA_CONFIG

        # Training hyperparameters
        self.device = resolve_device(self.config.get("device", "auto"), model=model)
        self.lr = self.config.get("learning_rate", 2e-4)
        self.max_steps = self.config.get("max_steps", 500)
        self.warmup_steps = self.config.get("warmup_steps", 100)
        self.grad_accum = self.config.get("gradient_accumulation", 8)
        self.max_grad_norm = self.config.get("max_grad_norm", 1.0)
        self.check_every = self.config.get("eval_every_n_steps", 100)
        self.save_every = self.config.get("save_every_n_steps", 500)
        self.checkpoint_dir = Path(self.config.get("checkpoint_dir", "./checkpoints"))
        self.is_primary = is_primary_process(self.device)
        self.use_wandb = (
            self.config.get("use_wandb", True)
            and wandb is not None
            and self.is_primary
        )

        self.model = move_model_to_device(model, self.device)
        self.train_dataloader = wrap_dataloader(self.train_dataloader, self.device)
        self.eval_dataloader = wrap_dataloader(self.eval_dataloader, self.device)

        # Optimizer — only train non-frozen parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable_params, lr=self.lr, weight_decay=0.01)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.max_steps)

        self.global_step = 0
        self.train_losses = []

    def train(self) -> list[float]:
        """
        Run the full training loop.

        Returns:
            List of training losses per step.
        """
        self.model.train()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if self.use_wandb:
            wandb.init(
                project="cognitive-llm",
                config={**self.config, **self.lambda_config},
            )

        accum_loss = 0.0
        accum_lm_loss = 0.0
        accum_surprise_loss = 0.0
        data_iter = iter(self.train_dataloader)

        for step in range(1, self.max_steps + 1):
            # Get batch (cycle dataloader if needed)
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)

            # Move to device
            batch = move_batch_to_device(batch, self.device)
            input_ids = batch["input_ids"]
            attention_mask = batch.get("attention_mask", None)
            labels = batch.get("labels", input_ids.clone())

            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            # Compute total loss
            loss = compute_total_loss(outputs, self.lambda_config)
            loss = loss / self.grad_accum
            loss.backward()
            accum_loss += loss.item()
            accum_lm_loss += outputs["lm_loss"].item() / self.grad_accum
            if outputs["surprise_loss"] is not None:
                accum_surprise_loss += outputs["surprise_loss"].item() / self.grad_accum

            # Gradient accumulation step
            if step % self.grad_accum == 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                optimizer_step(self.optimizer, self.device)
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                mark_step(self.device)

                self.global_step += 1
                step_loss = accum_loss
                step_lm_loss = accum_lm_loss
                step_surprise_loss = accum_surprise_loss
                accum_loss = 0.0
                accum_lm_loss = 0.0
                accum_surprise_loss = 0.0
                self.train_losses.append(step_loss)

                # Logging
                log_dict = {
                    "train/total_loss": step_loss,
                    "train/lm_loss": step_lm_loss,
                    "train/surprise_loss": step_surprise_loss,
                    "train/lr": self.scheduler.get_last_lr()[0],
                }

                if self.use_wandb:
                    wandb.log(log_dict, step=self.global_step)

                if self.is_primary and self.global_step % 10 == 0:
                    print(
                        f"Step {self.global_step}: loss={step_loss:.4f} "
                        f"lm_loss={step_lm_loss:.4f}"
                    )

            # Run assessment
            if self.eval_dataloader and step % (self.check_every * self.grad_accum) == 0:
                self._run_assessment()

            # Checkpointing
            if step % (self.save_every * self.grad_accum) == 0:
                self._save_checkpoint()

        if self.use_wandb:
            wandb.finish()

        return self.train_losses

    @torch.no_grad()
    def _run_assessment(self) -> float:
        """Run model assessment and return average loss."""
        self.model.train(False)
        total_loss = 0.0
        n_batches = 0

        for batch in self.eval_dataloader:
            batch = move_batch_to_device(batch, self.device)
            input_ids = batch["input_ids"]
            attention_mask = batch.get("attention_mask", None)
            labels = batch.get("labels", input_ids.clone())

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = compute_total_loss(outputs, self.lambda_config)
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        if self.is_primary:
            print(f"  Assessment loss: {avg_loss:.4f}")

        if self.use_wandb:
            wandb.log({"assessment/loss": avg_loss}, step=self.global_step)

        self.model.train(True)
        mark_step(self.device)
        return avg_loss

    def _save_checkpoint(self) -> None:
        """Save model checkpoint with block config in filename."""
        block_flags = "".join(
            str(int(self.model.config.get(f"use_block{i}", False)))
            for i in range(1, 7)
        )
        filename = f"cognitive_step{self.global_step}_blocks{block_flags}.pt"
        path = self.checkpoint_dir / filename

        save_checkpoint(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "global_step": self.global_step,
                "config": self.model.config,
                "lambda_config": self.lambda_config,
                "train_losses": self.train_losses,
            },
            path,
            self.device,
        )
        if self.is_primary:
            print(f"  Checkpoint saved: {path}")
