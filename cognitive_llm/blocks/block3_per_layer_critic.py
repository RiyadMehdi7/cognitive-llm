"""
Block 3: Per-Layer Critic — distributed value estimation with TD learning.

Position: Attached to every N-th transformer layer (default: every 4 layers).
          Does NOT modify activations — produces auxiliary loss only.
Purpose: Distributed value estimation across network depth. Each critic
         learns to predict the "remaining difficulty" at its layer using
         bootstrapped TD targets from deeper critics, enabling true
         credit assignment across layers.

Reference: Inspired by temporal difference learning (Sutton 1988) and
           distributed value estimation in deep RL. Neuroscience parallel:
           Wang et al. 2018 showed prefrontal cortex implements distributed
           value signals for credit assignment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerCritic(nn.Module):
    """
    Per-layer value head for distributed credit assignment.
    Attached every N layers. Does not modify forward pass.

    Supports two modes:
    - Supervised: compute_loss(hidden, td_target) regresses against a scalar target.
    - TD bootstrapped: compute_td_loss(hidden, next_value, lm_loss, gamma) uses
      bootstrapped targets from the next deeper critic.

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
        Compute MSE loss between predicted value and a scalar target.

        Args:
            hidden_state: Tensor of shape (batch, seq_len, d_model).
            td_target: Target values of shape (batch,).

        Returns:
            Scalar MSE loss.
        """
        v_pred = self(hidden_state)
        return F.mse_loss(v_pred.squeeze(-1), td_target)

    def compute_td_loss(
        self,
        hidden_state: torch.Tensor,
        next_value: torch.Tensor,
        lm_loss: torch.Tensor,
        gamma: float = 0.95,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute TD(0) loss with bootstrapped target from the next critic.

        TD target: r + gamma * V(next_layer)
        where r = per-example LM loss (the "cost" signal) and V(next_layer)
        is the detached value estimate from the next deeper critic.

        Args:
            hidden_state: Tensor of shape (batch, seq_len, d_model).
            next_value: Detached value from next deeper critic, shape (batch,).
            lm_loss: Per-example LM loss, shape (batch,).
            gamma: Discount factor across layers (default: 0.95).

        Returns:
            Tuple of (td_loss, current_value_detached) where current_value
            is returned detached for the previous critic's bootstrap.
        """
        v_pred = self(hidden_state).squeeze(-1)  # (batch,)
        td_target = lm_loss.detach() + gamma * next_value.detach()
        td_loss = F.mse_loss(v_pred, td_target)
        return td_loss, v_pred.detach()
