"""Unit tests for Block 1: SurpriseGate."""

import torch
import pytest
from cognitive_llm.blocks.block1_surprise_gate import SurpriseGate


@pytest.fixture
def gate():
    return SurpriseGate(d_model=128, n_buckets=3)


@pytest.fixture
def sample_input():
    return torch.randn(2, 16, 128)


class TestSurpriseGate:
    def test_forward_returns_tuple(self, gate, sample_input):
        """Forward must return (hidden, depth) tuple."""
        result = gate(sample_input)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_output_shapes(self, gate, sample_input):
        """Hidden shape unchanged, depth is (batch, seq_len)."""
        hidden, depth = gate(sample_input)
        assert hidden.shape == sample_input.shape
        assert depth.shape == (2, 16)

    def test_depth_values_in_range(self, gate, sample_input):
        """Depth values must be in [0, n_buckets-1]."""
        _, depth = gate(sample_input)
        assert depth.min() >= 0
        assert depth.max() <= gate.n_buckets - 1

    def test_depth_dtype(self, gate, sample_input):
        """Depth must be long (integer) tensor."""
        _, depth = gate(sample_input)
        assert depth.dtype == torch.long

    def test_surprise_loss_is_scalar(self, gate, sample_input):
        """get_surprise_loss must return a scalar tensor."""
        loss = gate.get_surprise_loss(sample_input)
        assert loss.dim() == 0
        assert loss.dtype == torch.float32

    def test_surprise_loss_positive(self, gate, sample_input):
        """MSE loss should be positive."""
        loss = gate.get_surprise_loss(sample_input)
        assert loss.item() >= 0.0

    def test_ema_updates(self, gate, sample_input):
        """EMA hidden state should update after forward pass."""
        initial_ema = gate.ema_hidden.clone()
        _ = gate(sample_input)
        assert not torch.allclose(gate.ema_hidden, initial_ema)

    def test_gradient_flow(self, gate, sample_input):
        """Gradients must flow through SurpriseGate."""
        sample_input.requires_grad_(True)
        hidden, _ = gate(sample_input)
        loss = hidden.sum()
        loss.backward()
        assert sample_input.grad is not None

    def test_different_bucket_counts(self, sample_input):
        """Test with different numbers of buckets."""
        for n_buckets in [2, 3, 5]:
            gate = SurpriseGate(d_model=128, n_buckets=n_buckets)
            _, depth = gate(sample_input)
            assert depth.max() <= n_buckets - 1
