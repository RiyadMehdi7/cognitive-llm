from __future__ import annotations

"""
Block 4: Predictive Coding — inter-layer prediction and error propagation.

Position: Between transformer layers — wraps each layer's input/output.
Purpose: Each layer predicts activations of the next layer. Only residual
         error propagates forward. Biologically inspired — based on
         Rao & Ballard 1999 predictive coding theory.

WARNING: This is the most unstable block. Implement ONLY after Blocks 1-3
         are stable. This block is Phase 2 only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictiveCodingLayer(nn.Module):
    """
    Predictive coding wrapper for transformer layers.
    Each layer predicts next layer activations; passes error residual.

    Args:
        d_model: Hidden dimension size.
        alpha_init: Initial mixing coefficient between error and full signal (default: 0.1).
    """

    def __init__(self, d_model: int, alpha_init: float = 0.1):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

        # Learned mixing: how much error vs full signal
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self, x_current: torch.Tensor, x_prev: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply predictive coding: predict current from previous, pass error.

        Args:
            x_current: Current layer activations (batch, seq_len, d_model).
            x_prev: Previous layer activations, or None for the first layer.

        Returns:
            Tuple of (output, pred_loss) where output has the same shape as
            x_current and pred_loss is a scalar tensor.
        """
        if x_prev is None:
            return x_current, torch.tensor(0.0, device=x_current.device)

        prediction = self.predictor(x_prev)
        error = x_current - prediction.detach()

        # Auxiliary prediction loss
        pred_loss = F.mse_loss(prediction, x_current.detach())

        # Mix full signal with error-emphasis
        alpha_clamped = torch.sigmoid(self.alpha)
        out = self.layer_norm(x_current + alpha_clamped * error)

        return out, pred_loss
