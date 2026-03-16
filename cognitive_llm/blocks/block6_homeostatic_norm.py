"""
Block 6: Homeostatic Regulation — drop-in normalization wrapper.

Position: Wraps LayerNorm/RMSNorm-style modules inside transformer layers.
Purpose: Tracks activation history via EMA. Dynamically adjusts gain/bias
         to prevent chronic over-excitation. Critical for stability when
         running multiple cognitive blocks simultaneously.

Reference: Homeostatic plasticity in biological neural networks maintains
           stable activity levels despite perturbations.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


class HomeostaticNorm(nn.Module):
    """
    Drop-in normalization wrapper with activation history tracking.
    Adapts gain/bias based on running statistics to prevent instability.

    Args:
        d_model: Hidden dimension size.
        tau: EMA decay factor for running statistics (default: 0.99).
        eps: Epsilon for numerical stability (default: 1e-5).
        base_norm: Optional existing norm module to wrap.
    """

    def __init__(
        self,
        d_model: int,
        tau: float = 0.99,
        eps: float = 1e-5,
        base_norm: nn.Module | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.tau = tau
        self.eps = eps
        self.base_norm = copy.deepcopy(base_norm) if base_norm is not None else nn.LayerNorm(d_model, eps=eps)

        # Trainable adaptive parameters
        self.adapt_scale = nn.Parameter(torch.ones(d_model))
        self.adapt_bias = nn.Parameter(torch.zeros(d_model))

        # Non-trainable running statistics
        self.register_buffer("running_mean", torch.zeros(d_model))
        self.register_buffer("running_var", torch.ones(d_model))
        self.register_buffer("step_count", torch.tensor(0, dtype=torch.long))

    @classmethod
    def from_norm(
        cls,
        norm_module: nn.Module,
        d_model: int,
        tau: float = 0.99,
    ) -> "HomeostaticNorm":
        """Wrap an existing LayerNorm/RMSNorm-style module."""
        eps = getattr(
            norm_module,
            "eps",
            getattr(norm_module, "variance_epsilon", 1e-5),
        )
        return cls(d_model=d_model, tau=tau, eps=eps, base_norm=norm_module)

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

        # Preserve the wrapped norm's baseline behavior, then adapt it.
        out = self.base_norm(x)
        out = out * (self.adapt_scale * correction) + self.adapt_bias

        return out
