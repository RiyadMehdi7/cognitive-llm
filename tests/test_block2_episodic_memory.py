"""Unit tests for Block 2: EpisodicMemory."""

import torch
import pytest
from cognitive_llm.blocks.block2_episodic_memory import EpisodicMemory


@pytest.fixture
def memory():
    return EpisodicMemory(d_model=128, mem_slots=32, bottleneck_dim=32)


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

    def test_read_near_identity_at_init(self, memory, sample_input):
        """At initialization, read() should approximately pass through the query.

        mem_proj is zero-initialized, so gate * proj(mem) = 0 regardless of
        gate value, meaning read output = query + 0 = query.
        """
        memory.reset(2, torch.device("cpu"))
        memory.write(sample_input)
        out = memory.read(sample_input)
        assert torch.allclose(out, sample_input, atol=1e-5)

    def test_write_gate_starts_conservative(self):
        """Write gate should start with low values (sigmoid(-2) ≈ 0.12)."""
        mem = EpisodicMemory(d_model=128, mem_slots=32)
        x = torch.randn(2, 16, 128)
        gate_layer = mem.write_gate
        gate_vals = gate_layer(x)
        # sigmoid(-2) ≈ 0.119, should be well below 0.5
        assert gate_vals.mean().item() < 0.2

    def test_multiple_writes(self, memory, sample_input):
        """Multiple writes should accumulate in memory."""
        memory.reset(2, torch.device("cpu"))
        memory.write(sample_input)
        state1 = memory.memory_values.clone()
        memory.write(sample_input * 2)
        # Memory should change after second write
        assert not torch.allclose(memory.memory_values, state1)

    def test_subset_write_only_updates_selected_rows(self, memory):
        """Selective writes should leave untouched batch rows unchanged."""
        memory.reset(3, torch.device("cpu"))
        initial = memory.memory_values.clone()

        memory.write(torch.randn(1, 4, 128), batch_indices=torch.tensor([1]))

        assert torch.allclose(memory.memory_values[0], initial[0])
        assert not torch.allclose(memory.memory_values[1], initial[1])
        assert torch.allclose(memory.memory_values[2], initial[2])

    def test_param_count_is_low_rank(self):
        """B2 with d_model=2048 should have ~5M params, not 25M."""
        mem = EpisodicMemory(d_model=2048, mem_slots=64, bottleneck_dim=128)
        total = sum(p.numel() for p in mem.parameters())
        # Should be well under 10M (old was 25M)
        assert total < 10_000_000
        # Should be around 5M
        assert total > 1_000_000
