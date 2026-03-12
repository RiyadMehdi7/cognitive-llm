"""
Block 2: Episodic Memory — online key-value buffer.

Position: After Block 1, before transformer stack (write hook).
          After transformer stack (read hook).
Purpose: Online key-value buffer. Model writes 'thoughts' during inference
         and reads them back. Not RAG — this is working memory within a
         single forward pass.

Reference: Inspired by hippocampal episodic memory — rapid one-shot storage
           and retrieval of experiences.
"""

import torch
import torch.nn as nn


class EpisodicMemory(nn.Module):
    """
    Online episodic memory: read/write buffer updated during inference.

    Args:
        d_model: Hidden dimension size.
        mem_slots: Number of memory slots (default: 64).
        n_heads: Number of attention heads for read operation (default: 4).
    """

    def __init__(self, d_model: int, mem_slots: int = 64, n_heads: int = 4):
        super().__init__()
        self.mem_slots = mem_slots
        self.d_model = d_model

        # Learnable memory keys (address space)
        self.memory_keys = nn.Parameter(torch.randn(mem_slots, d_model) * 0.02)

        # Write gate: decides how much to write
        self.write_gate = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )

        # Read attention
        self.read_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        # Combine read output with current hidden
        self.combine = nn.Linear(d_model * 2, d_model)

        # Memory values buffer (set during reset)
        self.memory_values: torch.Tensor | None = None

    def reset(self, batch_size: int, device: torch.device) -> None:
        """
        Reset memory values at start of each sequence.

        Args:
            batch_size: Current batch size.
            device: Device to allocate memory on.
        """
        self.memory_values = torch.zeros(
            batch_size, self.mem_slots, self.d_model, device=device
        )

    def write(self, x: torch.Tensor) -> None:
        """
        Write hidden states to memory slots using soft attention.

        Args:
            x: Hidden states of shape (batch, seq_len, d_model).
        """
        assert self.memory_values is not None, "Call reset() before write()"

        gate = self.write_gate(x)  # (B, S, 1)

        # Find most similar memory slot for each token via soft attention
        keys_exp = self.memory_keys.unsqueeze(0).expand(x.shape[0], -1, -1)
        sim = torch.bmm(x, keys_exp.transpose(1, 2))  # (B, S, mem_slots)
        attn_weights = torch.softmax(sim, dim=-1)  # (B, S, mem_slots)

        # Compute weighted write values: (B, mem_slots, d_model)
        # For each slot, aggregate gated input across all tokens
        write_vals = torch.bmm(
            (attn_weights * gate).transpose(1, 2),  # (B, mem_slots, S)
            x,  # (B, S, d_model)
        )  # (B, mem_slots, d_model)

        # Compute total gate weight per slot for blending
        slot_gate = (attn_weights * gate).sum(dim=1)  # (B, mem_slots)
        slot_gate = slot_gate.clamp(max=1.0).unsqueeze(-1)  # (B, mem_slots, 1)

        # Blend new writes with existing memory
        self.memory_values = (
            slot_gate * write_vals + (1 - slot_gate) * self.memory_values
        )

    def read(self, query: torch.Tensor) -> torch.Tensor:
        """
        Read from memory using attention.

        Args:
            query: Query tensor of shape (batch, seq_len, d_model).

        Returns:
            Combined tensor of shape (batch, seq_len, d_model).
        """
        assert self.memory_values is not None, "Call reset() before read()"

        keys_exp = self.memory_keys.unsqueeze(0).expand(query.shape[0], -1, -1)
        mem_out, _ = self.read_attn(query, keys_exp, self.memory_values)
        combined = torch.cat([query, mem_out], dim=-1)
        return self.combine(combined)
