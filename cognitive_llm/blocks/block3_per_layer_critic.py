"""
Block 3: Per-Layer Critic — distributed value estimation.

Position: Attached to every N-th transformer layer (default: every 4 layers).
          Does NOT modify activations — produces auxiliary loss only.
Purpose: Distributed value estimation. Solves credit assignment in RL training
         by providing intermediate reward signals throughout the network depth.

Reference: Inspired by temporal difference learning and multi-step value
           estimation in deep RL.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerCritic(nn.Module):
    """
    Per-layer value head for distributed credit assignment.
    Attached every N layers. Does not modify forward pass.

    Args:
        d_model: Hidden dimension size.
        hidden_dim: Critic MLP hidden dimension (default: 256).
    """

    def __init__(self, d_model: int, hidden_dim: int = 256):
        super().__init__()
        self.value_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Compute value estimate from hidden state.

        Args:
            hidden_state: Tensor of shape (batch, seq_len, d_model).

        Returns:
            Value estimate of shape (batch, 1).
        """
        pooled = hidden_state.mean(dim=1)  # mean pool over sequence
        return self.value_head(pooled)

    def compute_loss(self, hidden_state: torch.Tensor, td_target: torch.Tensor) -> torch.Tensor:
        """
        Compute TD loss between predicted and target values.

        Args:
            hidden_state: Tensor of shape (batch, seq_len, d_model).
            td_target: Target values of shape (batch,).

        Returns:
            Scalar MSE loss.
        """
        v_pred = self(hidden_state)
        return F.mse_loss(v_pred.squeeze(-1), td_target)
