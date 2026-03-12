"""
Block 5: RL Gating Policy — learned orchestration of cognitive blocks.

Position: After transformer stack, before output projection.
Purpose: Learned policy that orchestrates all other blocks. Decides per-token:
         write to memory / deepen compute / recall from memory / pass through.
         Trained with PPO.

NOTE: Build this only after Blocks 1 and 2 are working and stable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatingPolicy(nn.Module):
    """
    RL-trained gating policy. Orchestrates all cognitive blocks.
    Actions: 0=pass, 1=write_memory, 2=deepen_compute, 3=recall_memory

    Args:
        d_model: Hidden dimension size.
        hidden_dim: Policy/value MLP hidden dimension (default: 128).
    """

    N_ACTIONS = 4
    ACTION_PASS = 0
    ACTION_WRITE_MEMORY = 1
    ACTION_DEEPEN_COMPUTE = 2
    ACTION_RECALL_MEMORY = 3

    def __init__(self, d_model: int, hidden_dim: int = 128):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.N_ACTIONS),
        )
        self.value_net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_std = nn.Parameter(torch.zeros(self.N_ACTIONS))

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute action probabilities and value estimate.

        Args:
            h: Hidden states of shape (batch, seq_len, d_model).

        Returns:
            Tuple of (action_probs, value) where action_probs is (batch, N_ACTIONS)
            and value is (batch, 1).
        """
        pooled = h.mean(dim=1)  # (batch, d_model)
        logits = self.policy_net(pooled)
        action_probs = F.softmax(logits, dim=-1)
        value = self.value_net(pooled)
        return action_probs, value

    def get_action(
        self, h: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample or select an action.

        Args:
            h: Hidden states of shape (batch, seq_len, d_model).
            deterministic: If True, select argmax action.

        Returns:
            Tuple of (action, probs, value).
        """
        probs, value = self(h)
        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        return action, probs, value
