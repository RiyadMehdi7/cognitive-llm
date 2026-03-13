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

    Uses low-rank bottleneck projections for the read operation instead of
    full-rank MultiheadAttention. With only 64 memory slots, a 128-dim
    bottleneck provides sufficient expressiveness while keeping param count
    proportional to the LoRA baseline (~2M vs 25M).

    Args:
        d_model: Hidden dimension size.
        mem_slots: Number of memory slots (default: 64).
        bottleneck_dim: Dimension for low-rank read projections (default: 128).
    """

    def __init__(self, d_model: int, mem_slots: int = 64, bottleneck_dim: int = 128):
        super().__init__()
        self.mem_slots = mem_slots
        self.d_model = d_model
        self.bottleneck_dim = bottleneck_dim

        # Learnable memory keys (address space)
        self.memory_keys = nn.Parameter(torch.randn(mem_slots, d_model) * 0.02)

        # Write gate: decides how much to write
        # Initialize bias to -2 so sigmoid starts near 0.12 (conservative writes)
        write_linear = nn.Linear(d_model, 1)
        nn.init.zeros_(write_linear.weight)
        nn.init.constant_(write_linear.bias, -2.0)
        self.write_gate = nn.Sequential(
            write_linear,
            nn.Sigmoid(),
        )

        # Low-rank read: project query and keys to bottleneck dim for attention
        self.read_q_proj = nn.Linear(d_model, bottleneck_dim, bias=False)
        self.read_k_proj = nn.Linear(d_model, bottleneck_dim, bias=False)
        self.scale = bottleneck_dim ** -0.5

        # Gated residual: memory output is added with a learnable scalar gate
        # Starting at 0 (sigmoid(gate_param) ≈ 0.5 * tanh(0) = 0) so memory
        # contribution starts as zero — model learns to use it gradually.
        self.mem_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.mem_proj.weight)
        self.mem_gate = nn.Parameter(torch.tensor(0.0))

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

        # Cast to module dtype for computation
        compute_dtype = self.memory_keys.dtype
        x = x.to(dtype=compute_dtype)
        self.memory_values = self.memory_values.to(dtype=compute_dtype)

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
        Read from memory using low-rank bottleneck attention + gated residual.

        Args:
            query: Query tensor of shape (batch, seq_len, d_model).

        Returns:
            Augmented tensor of shape (batch, seq_len, d_model).
        """
        assert self.memory_values is not None, "Call reset() before read()"

        orig_dtype = query.dtype
        compute_dtype = self.memory_keys.dtype
        query = query.to(dtype=compute_dtype)

        # Low-rank attention over memory slots
        q = self.read_q_proj(query)  # (B, S, bottleneck)
        keys_exp = self.memory_keys.unsqueeze(0).expand(query.shape[0], -1, -1)
        k = self.read_k_proj(keys_exp)  # (B, mem_slots, bottleneck)

        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale  # (B, S, mem_slots)
        attn = torch.softmax(attn, dim=-1)

        # Retrieve from memory values
        mem_out = torch.bmm(attn, self.memory_values)  # (B, S, d_model)

        # Gated residual: query + gate * proj(mem_out)
        gate = torch.sigmoid(self.mem_gate)
        result = query + gate * self.mem_proj(mem_out)
        return result.to(dtype=orig_dtype)
