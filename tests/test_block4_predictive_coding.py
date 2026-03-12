"""Unit tests for Block 4: PredictiveCodingLayer."""

import torch
import pytest
from cognitive_llm.blocks.block4_predictive_coding import PredictiveCodingLayer


@pytest.fixture
def pc_layer():
    return PredictiveCodingLayer(d_model=128)


@pytest.fixture
def sample_input():
    return torch.randn(2, 16, 128)


class TestPredictiveCodingLayer:
    def test_output_shape_no_prev(self, pc_layer, sample_input):
        """Without previous input, output shape should match input."""
        out, pred_loss = pc_layer(sample_input, x_prev=None)
        assert out.shape == sample_input.shape
        assert pred_loss.item() == 0.0

    def test_output_shape_with_prev(self, pc_layer, sample_input):
        """With previous input, output shape should match input."""
        x_prev = torch.randn_like(sample_input)
        out, pred_loss = pc_layer(sample_input, x_prev)
        assert out.shape == sample_input.shape

    def test_pred_loss_positive(self, pc_layer, sample_input):
        """Prediction loss should be positive when prev is provided."""
        x_prev = torch.randn_like(sample_input)
        _, pred_loss = pc_layer(sample_input, x_prev)
        assert pred_loss.item() > 0.0

    def test_pred_loss_scalar(self, pc_layer, sample_input):
        """Prediction loss should be a scalar tensor."""
        x_prev = torch.randn_like(sample_input)
        _, pred_loss = pc_layer(sample_input, x_prev)
        assert pred_loss.dim() == 0

    def test_gradient_flow(self, pc_layer, sample_input):
        """Gradients must flow through PredictiveCodingLayer."""
        sample_input.requires_grad_(True)
        x_prev = torch.randn_like(sample_input)
        out, _ = pc_layer(sample_input, x_prev)
        loss = out.sum()
        loss.backward()
        assert sample_input.grad is not None

    def test_alpha_is_learnable(self, pc_layer):
        """Alpha parameter should be learnable."""
        assert pc_layer.alpha.requires_grad

    def test_no_nan_output(self, pc_layer, sample_input):
        """Output should not contain NaN values."""
        x_prev = torch.randn_like(sample_input)
        out, pred_loss = pc_layer(sample_input, x_prev)
        assert not torch.isnan(out).any()
        assert not torch.isnan(pred_loss).any()
