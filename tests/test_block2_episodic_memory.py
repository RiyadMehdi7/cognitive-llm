"""Unit tests for Block 2: EpisodicMemory."""

import torch
import pytest
from cognitive_llm.blocks.block2_episodic_memory import EpisodicMemory


@pytest.fixture
def memory():
    return EpisodicMemory(d_model=128, mem_slots=32, n_heads=4)


@pytest.fixture
def sample_input():
    return torch.randn(2, 16, 128)


class TestEpisodicMemory:
    def test_reset_creates_memory(self, memory):
        """reset() must initialize memory_values to zeros."""
        memory.reset(batch_size=2, device=torch.device("cpu"))
        assert memory.memory_values is not None
        assert memory.memory_values.shape == (2, 32, 128)
        assert (memory.memory_values == 0).all()

    def test_write_updates_memory(self, memory, sample_input):
        """write() must modify memory_values."""
        memory.reset(2, torch.device("cpu"))
        memory.write(sample_input)
        # At least some slots should be non-zero after write
        assert not (memory.memory_values == 0).all()

    def test_read_output_shape(self, memory, sample_input):
        """read() must return tensor of same shape as query."""
        memory.reset(2, torch.device("cpu"))
        memory.write(sample_input)
        out = memory.read(sample_input)
        assert out.shape == sample_input.shape

    def test_full_pipeline(self, memory, sample_input):
        """Full reset -> write -> read pipeline should work."""
        memory.reset(2, torch.device("cpu"))
        memory.write(sample_input)
        out = memory.read(sample_input)
        assert out.shape == sample_input.shape
        assert not torch.isnan(out).any()

    def test_read_without_reset_raises(self, memory, sample_input):
        """read() without reset() should raise AssertionError."""
        memory.memory_values = None
        with pytest.raises(AssertionError):
            memory.read(sample_input)

    def test_write_without_reset_raises(self, memory, sample_input):
        """write() without reset() should raise AssertionError."""
        memory.memory_values = None
        with pytest.raises(AssertionError):
            memory.write(sample_input)

    def test_gradient_flow(self, memory, sample_input):
        """Gradients must flow through read operation."""
        sample_input.requires_grad_(True)
        memory.reset(2, torch.device("cpu"))
        memory.write(sample_input)
        out = memory.read(sample_input)
        loss = out.sum()
        loss.backward()
        assert sample_input.grad is not None

    def test_multiple_writes(self, memory, sample_input):
        """Multiple writes should accumulate in memory."""
        memory.reset(2, torch.device("cpu"))
        memory.write(sample_input)
        state1 = memory.memory_values.clone()
        memory.write(sample_input * 2)
        # Memory should change after second write
        assert not torch.allclose(memory.memory_values, state1)
