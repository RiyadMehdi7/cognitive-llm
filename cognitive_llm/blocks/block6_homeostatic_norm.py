"""
Block 6: Homeostatic Regulation — drop-in LayerNorm replacement.

Position: Replaces or augments LayerNorm inside transformer layers.
Purpose: Tracks activation history via EMA. Dynamically adjusts gain/bias
         to prevent chronic over-excitation. Critical for stability when
         running multiple cognitive blocks simultaneously.

Reference: Homeostatic plasticity in biological neural networks maintains
           stable activity levels despite perturbations.
"""

import torch
import torch.nn as nn


class HomeostaticNorm(nn.Module):
    """
    Drop-in LayerNorm replacement with activation history tracking.
    Adapts gain/bias based on running statistics to prevent instability.

    Args:
        d_model: Hidden dimension size.
        tau: EMA decay factor for running statistics (default: 0.99).
        eps: Epsilon for numerical stability (default: 1e-5).
    """

    def __init__(self, d_model: int, tau: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.d_model = d_model
        self.layer_norm = nn.LayerNorm(d_model, eps=eps)
        self.tau = tau
        self.eps = eps

        # Trainable adaptive parameters
        self.adapt_scale = nn.Parameter(torch.ones(d_model))
        self.adapt_bias = nn.Parameter(torch.zeros(d_model))

        # Non-trainable running statistics
        self.register_buffer("running_mean", torch.zeros(d_model))
        self.register_buffer("running_var", torch.ones(d_model))
        self.register_buffer("step_count", torch.tensor(0, dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply homeostatic normalization.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Normalized tensor of same shape.
        """
        # Update running stats (online, not during eval)
        if self.training:
            with torch.no_grad():
                batch_mean = x.detach().mean(dim=[0, 1])
                batch_var = x.detach().var(dim=[0, 1])
                self.running_mean.mul_(self.tau).add_(batch_mean, alpha=1 - self.tau)
                self.running_var.mul_(self.tau).add_(batch_var, alpha=1 - self.tau)
                self.step_count += 1

        # Homeostatic correction factor
        correction = 1.0 / (self.running_var.sqrt() + self.eps)
        correction = correction.clamp(0.1, 10.0)  # Safety clamp

        # Apply corrected normalization
        out = self.layer_norm(x)
        out = out * (self.adapt_scale * correction) + self.adapt_bias

        return out
