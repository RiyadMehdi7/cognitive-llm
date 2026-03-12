"""Unit tests for Block 5: GatingPolicy."""

import torch
import pytest
from cognitive_llm.blocks.block5_rl_gating import GatingPolicy


@pytest.fixture
def policy():
    return GatingPolicy(d_model=128)


@pytest.fixture
def sample_input():
    return torch.randn(2, 16, 128)


class TestGatingPolicy:
    def test_forward_output_shapes(self, policy, sample_input):
        """Forward must return (action_probs, value) with correct shapes."""
        probs, value = policy(sample_input)
        assert probs.shape == (2, GatingPolicy.N_ACTIONS)
        assert value.shape == (2, 1)

    def test_probs_sum_to_one(self, policy, sample_input):
        """Action probabilities must sum to 1."""
        probs, _ = policy(sample_input)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(2), atol=1e-5)

    def test_probs_non_negative(self, policy, sample_input):
        """Action probabilities must be non-negative."""
        probs, _ = policy(sample_input)
        assert (probs >= 0).all()

    def test_get_action_deterministic(self, policy, sample_input):
        """Deterministic action should be argmax of probs."""
        action, probs, value = policy.get_action(sample_input, deterministic=True)
        expected = probs.argmax(dim=-1)
        assert torch.equal(action, expected)

    def test_get_action_stochastic(self, policy, sample_input):
        """Stochastic action should be a valid action index."""
        action, _, _ = policy.get_action(sample_input, deterministic=False)
        assert action.shape == (2,)
        assert (action >= 0).all()
        assert (action < GatingPolicy.N_ACTIONS).all()

    def test_gradient_flow(self, policy, sample_input):
        """Gradients must flow through the policy."""
        sample_input.requires_grad_(True)
        probs, value = policy(sample_input)
        loss = probs.sum() + value.sum()
        loss.backward()
        assert sample_input.grad is not None

    def test_action_constants(self, policy):
        """Action constants should match spec."""
        assert policy.ACTION_PASS == 0
        assert policy.ACTION_WRITE_MEMORY == 1
        assert policy.ACTION_DEEPEN_COMPUTE == 2
        assert policy.ACTION_RECALL_MEMORY == 3
        assert policy.N_ACTIONS == 4
