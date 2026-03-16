"""Unit tests for Block 6: HomeostaticNorm."""

import torch
import pytest
from cognitive_llm.blocks.block6_homeostatic_norm import HomeostaticNorm


@pytest.fixture
def norm():
    return HomeostaticNorm(d_model=128)


@pytest.fixture
def sample_input():
    return torch.randn(2, 16, 128)


class TestHomeostaticNorm:
    def test_output_shape(self, norm, sample_input):
        """Output shape must match input shape."""
        out = norm(sample_input)
        assert out.shape == sample_input.shape

    def test_running_stats_update_during_training(self, norm, sample_input):
        """Running stats must update during training mode."""
        norm.train()
        initial_mean = norm.running_mean.clone()
        initial_var = norm.running_var.clone()

        _ = norm(sample_input)

        assert not torch.allclose(norm.running_mean, initial_mean), \
            "running_mean should update during training"
        assert not torch.allclose(norm.running_var, initial_var), \
            "running_var should update during training"
        assert norm.step_count == 1

    def test_running_stats_frozen_during_inference(self, norm, sample_input):
        """Running stats must NOT update during inference mode."""
        norm.train()
        _ = norm(sample_input)  # Initialize stats

        norm.train(False)
        mean_before = norm.running_mean.clone()
        var_before = norm.running_var.clone()
        step_before = norm.step_count.clone()

        _ = norm(sample_input)

        assert torch.allclose(norm.running_mean, mean_before), \
            "running_mean should be frozen during inference"
        assert torch.allclose(norm.running_var, var_before), \
            "running_var should be frozen during inference"
        assert norm.step_count == step_before

    def test_correction_clamping(self, norm):
        """Correction factor must be clamped to [0.1, 10.0]."""
        norm.running_var = torch.zeros(128)  # Would cause division by zero
        x = torch.randn(1, 4, 128)
        # Should not raise, correction is clamped
        out = norm(x)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_gradient_flow(self, norm, sample_input):
        """Gradients must flow through HomeostaticNorm."""
        sample_input.requires_grad_(True)
        out = norm(sample_input)
        loss = out.sum()
        loss.backward()
        assert sample_input.grad is not None
        assert not torch.isnan(sample_input.grad).any()

    def test_multiple_steps(self, norm):
        """Running stats should evolve over multiple steps."""
        norm.train()
        for _ in range(10):
            x = torch.randn(2, 8, 128)
            _ = norm(x)
        assert norm.step_count == 10

    def test_wraps_existing_norm_module(self):
        """Wrapping an existing norm should preserve a valid forward path."""
        wrapped = HomeostaticNorm.from_norm(torch.nn.LayerNorm(128), d_model=128)
        x = torch.randn(2, 8, 128)

        out = wrapped(x)

        assert out.shape == x.shape
        assert isinstance(wrapped.base_norm, torch.nn.LayerNorm)
