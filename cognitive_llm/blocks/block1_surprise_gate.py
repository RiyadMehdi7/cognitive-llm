"""
Block 1: Meta-Surprise Gate — adaptive compute routing.

Position: After embedding layer, before transformer Layer 1.
Purpose: Measures how 'surprising' a token is relative to the model's prior.
         Routes token to shallow or deep compute path. RL trains the routing
         threshold.

Reference: Inspired by predictive processing theory — the brain allocates
           more compute to unexpected stimuli.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SurpriseGate(nn.Module):
    """
    Meta-Surprise Gate: adaptive compute routing based on token unexpectedness.
    Placed between embedding and transformer layer 1.

    Args:
        d_model: Hidden dimension size.
        n_buckets: Number of depth buckets for routing (default: 3).
        ema_decay: EMA decay for prior hidden state estimation (default: 0.99).
    """

    def __init__(self, d_model: int, n_buckets: int = 3, ema_decay: float = 0.99):
        super().__init__()
        self.d_model = d_model
        self.n_buckets = n_buckets

        # Prior network: predicts expected hidden state from previous
        self.prior = nn.Linear(d_model, d_model, bias=False)

        # Surprise projection: scalar score from surprise signal
        self.surprise_proj = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # RL-trained thresholds (one per bucket boundary)
        self.rl_thresholds = nn.Parameter(torch.linspace(0.3, 0.7, n_buckets - 1))

        # EMA of hidden states for prior estimation
        self.register_buffer("ema_hidden", torch.zeros(1, d_model))
        self.ema_decay = ema_decay

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute surprise scores and assign depth buckets.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).

        Returns:
            Tuple of (x, depth) where depth is (batch, seq_len) with values
            in [0, n_buckets-1]. Used by CognitiveModel to route layers.
        """
        # Update EMA prior
        with torch.no_grad():
            current_mean = x.mean(dim=[0, 1], keepdim=False).unsqueeze(0)
            self.ema_hidden = self.ema_decay * self.ema_hidden + (1 - self.ema_decay) * current_mean

        # Compute surprise: norm of difference between x and prior prediction
        expected = self.prior(self.ema_hidden).unsqueeze(1).expand_as(x)
        surprise_vec = x - expected
        surprise_score = torch.sigmoid(self.surprise_proj(surprise_vec))  # (B, S, 1)

        # Assign depth bucket: 0=shallow, 1=mid, 2=deep
        thresholds = torch.sigmoid(self.rl_thresholds).sort().values
        depth = torch.zeros(x.shape[0], x.shape[1], dtype=torch.long, device=x.device)
        for i, t in enumerate(thresholds):
            depth += (surprise_score.squeeze(-1) > t).long()

        return x, depth

    def get_surprise_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Auxiliary loss: prior should predict future hidden states.

        Args:
            x: Hidden states of shape (batch, seq_len, d_model).

        Returns:
            Scalar MSE loss between predicted and actual next-token hidden states.
        """
        predicted = self.prior(x[:, :-1])
        target = x[:, 1:].detach()
        return F.mse_loss(predicted, target)
