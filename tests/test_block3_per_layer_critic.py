"""Unit tests for Block 3: LayerCritic."""

import torch
import pytest
from cognitive_llm.blocks.block3_per_layer_critic import LayerCritic


@pytest.fixture
def critic():
    return LayerCritic(d_model=128)


@pytest.fixture
def sample_input():
    return torch.randn(2, 16, 128)


class TestLayerCritic:
    def test_output_shape(self, critic, sample_input):
        """Forward must return (batch, 1) value estimate."""
        out = critic(sample_input)
        assert out.shape == (2, 1)

    def test_compute_loss_scalar(self, critic, sample_input):
        """compute_loss must return a scalar."""
        td_target = torch.randn(2)
        loss = critic.compute_loss(sample_input, td_target)
        assert loss.dim() == 0
        assert loss.item() >= 0.0

    def test_gradient_flow(self, critic, sample_input):
        """Gradients must flow through the critic."""
        sample_input.requires_grad_(True)
        out = critic(sample_input)
        loss = out.sum()
        loss.backward()
        assert sample_input.grad is not None

    def test_no_activation_modification(self, critic, sample_input):
        """Critic should not modify the input tensor."""
        input_clone = sample_input.clone()
        _ = critic(sample_input)
        assert torch.allclose(sample_input, input_clone)

    def test_deterministic(self, critic, sample_input):
        """Same input should give same output."""
        out1 = critic(sample_input)
        out2 = critic(sample_input)
        assert torch.allclose(out1, out2)
