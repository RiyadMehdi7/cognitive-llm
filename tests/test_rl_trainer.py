"""Regression tests for PPO training utilities."""

import torch

from cognitive_llm.blocks.block5_rl_gating import GatingPolicy
from cognitive_llm.training.rl_trainer import PPOTrainer


def test_ppo_update_handles_batched_rollouts():
    policy = GatingPolicy(d_model=32)
    trainer = PPOTrainer(policy)
    hidden_states = torch.randn(3, 5, 32)

    trainer.collect_rollout(hidden_states, reward=1.0, done=False)
    trainer.collect_rollout(hidden_states * 0.5, reward=0.5, done=True)

    metrics = trainer.update()

    assert set(metrics) == {
        "ppo/policy_loss",
        "ppo/value_loss",
        "ppo/entropy",
    }
    assert len(trainer.buffer) == 0
