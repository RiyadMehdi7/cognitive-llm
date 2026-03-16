"""
PPO trainer for the RL Gating Policy (Block 5) and Surprise Gate thresholds (Block 1).

Implements Proximal Policy Optimization to train the gating decisions
that orchestrate cognitive block interactions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class PPOConfig:
    """PPO hyperparameters."""

    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.95
    n_epochs: int = 4
    mini_batch_size: int = 8
    max_grad_norm: float = 0.5


class RolloutBuffer:
    """Stores rollout data for PPO training."""

    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def add(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        log_prob: torch.Tensor,
        reward: float | torch.Tensor,
        value: torch.Tensor,
        done: bool | torch.Tensor,
    ) -> None:
        self.states.append(state.detach())
        self.actions.append(action.detach())
        self.log_probs.append(log_prob.detach())
        reward_tensor = torch.as_tensor(reward, dtype=torch.float32, device=value.device)
        done_tensor = torch.as_tensor(done, dtype=torch.float32, device=value.device)
        if reward_tensor.dim() == 0:
            reward_tensor = reward_tensor.expand_as(value)
        if done_tensor.dim() == 0:
            done_tensor = done_tensor.expand_as(value)
        self.rewards.append(reward_tensor.detach())
        self.values.append(value.detach())
        self.dones.append(done_tensor.detach())

    def compute_returns_and_advantages(
        self, gamma: float, gae_lambda: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and discounted returns."""
        rewards = torch.stack(self.rewards)  # (T, B)
        values = torch.stack(self.values)  # (T, B)
        dones = torch.stack(self.dones)  # (T, B)

        advantages = torch.zeros_like(rewards)
        last_gae = torch.zeros_like(rewards[0])

        for t in reversed(range(len(rewards))):
            next_value = torch.zeros_like(values[t]) if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
            last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return returns.reshape(-1), advantages.reshape(-1)

    def clear(self) -> None:
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()

    def __len__(self) -> int:
        return len(self.rewards)


class PPOTrainer:
    """
    PPO trainer for cognitive block gating policy.

    Args:
        policy: GatingPolicy module.
        config: PPO hyperparameters.
    """

    def __init__(self, policy: nn.Module, config: PPOConfig | None = None):
        self.policy = policy
        self.config = config or PPOConfig()
        self.optimizer = torch.optim.Adam(
            policy.parameters(), lr=3e-4, eps=1e-5
        )
        self.buffer = RolloutBuffer()

    def collect_rollout(
        self,
        hidden_states: torch.Tensor,
        reward: float,
        done: bool,
    ) -> torch.Tensor:
        """
        Collect a single transition for PPO training.

        Args:
            hidden_states: Model hidden states (batch, seq_len, d_model).
            reward: Scalar reward for this step.
            done: Whether this is a terminal state.

        Returns:
            Selected action tensor.
        """
        with torch.no_grad():
            action, probs, value = self.policy.get_action(hidden_states)
            dist = torch.distributions.Categorical(probs)
            log_prob = dist.log_prob(action)

        self.buffer.add(
            state=hidden_states.mean(dim=1),  # pool for storage efficiency
            action=action,
            log_prob=log_prob,
            reward=reward,
            value=value.squeeze(-1),
            done=done,
        )

        return action

    def update(self) -> dict[str, float]:
        """
        Run PPO update on collected rollouts.

        Returns:
            Dict of loss metrics.
        """
        if len(self.buffer) == 0:
            return {}

        cfg = self.config

        returns, advantages = self.buffer.compute_returns_and_advantages(
            cfg.gamma, cfg.gae_lambda
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states = torch.cat(self.buffer.states, dim=0)
        actions = torch.cat(self.buffer.actions, dim=0)
        old_log_probs = torch.cat(self.buffer.log_probs, dim=0)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(cfg.n_epochs):
            probs, values = self.policy(states.unsqueeze(1))

            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            # Policy loss (clipped)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(values.squeeze(-1), returns)

            # Total loss
            loss = (
                policy_loss
                + cfg.value_coef * value_loss
                - cfg.entropy_coef * entropy
            )

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            n_updates += 1

        self.buffer.clear()

        return {
            "ppo/policy_loss": total_policy_loss / max(n_updates, 1),
            "ppo/value_loss": total_value_loss / max(n_updates, 1),
            "ppo/entropy": total_entropy / max(n_updates, 1),
        }
