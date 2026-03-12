"""
Cognitive Model Wrapper — injects cognitive blocks into a base HuggingFace model.

Supports SmolLM3, OLMo 3, and any LlamaForCausalLM-compatible model.
All blocks are toggled via config flags to support clean ablation studies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from cognitive_llm.blocks.block1_surprise_gate import SurpriseGate
from cognitive_llm.blocks.block2_episodic_memory import EpisodicMemory
from cognitive_llm.blocks.block3_per_layer_critic import LayerCritic
from cognitive_llm.blocks.block4_predictive_coding import PredictiveCodingLayer
from cognitive_llm.blocks.block5_rl_gating import GatingPolicy
from cognitive_llm.blocks.block6_homeostatic_norm import HomeostaticNorm


class CognitiveModel(nn.Module):
    """
    Wraps a HuggingFace causal LM and injects cognitive blocks.
    Supports SmolLM3, OLMo 3, and any LlamaForCausalLM-compatible model.

    Args:
        base_model: A HuggingFace AutoModelForCausalLM instance.
        config: Dict with keys 'use_block1' through 'use_block6' (bool),
                and optional 'critic_every_n_layers' (int, default 4).
    """

    def __init__(self, base_model: nn.Module, config: dict):
        super().__init__()
        self.base = base_model
        self.config = config
        d_model = base_model.config.hidden_size
        n_layers = base_model.config.num_hidden_layers
        critic_every = config.get("critic_every_n_layers", 4)

        # Instantiate blocks based on config flags
        self.surprise_gate = SurpriseGate(d_model) if config.get("use_block1") else None

        self.episodic_mem = EpisodicMemory(d_model) if config.get("use_block2") else None

        self.critics = nn.ModuleList(
            [
                LayerCritic(d_model)
                if (i % critic_every == 0 and config.get("use_block3"))
                else None
                for i in range(n_layers)
            ]
        )

        self.pred_coding = nn.ModuleList(
            [
                PredictiveCodingLayer(d_model) if config.get("use_block4") else None
                for _ in range(n_layers)
            ]
        )

        self.gating_policy = GatingPolicy(d_model) if config.get("use_block5") else None

        # Block 6 replaces LayerNorm in-place
        self.homeo_norms = None
        if config.get("use_block6"):
            self._replace_layer_norms(d_model)

    def _get_parent(self, name: str) -> nn.Module:
        """Get parent module from a dotted name path."""
        parts = name.split(".")
        module = self.base
        for part in parts[:-1]:
            module = getattr(module, part)
        return module

    def _replace_layer_norms(self, d_model: int) -> None:
        """Replace all LayerNorm modules with HomeostaticNorm."""
        replacements = []
        for name, module in self.base.named_modules():
            if isinstance(module, nn.LayerNorm) and module.normalized_shape == (d_model,):
                replacements.append(name)

        for name in replacements:
            parent = self._get_parent(name)
            attr = name.split(".")[-1]
            setattr(parent, attr, HomeostaticNorm(d_model))

    def _get_model_backbone(self) -> nn.Module:
        """Get the transformer backbone (handles different model architectures)."""
        # SmolLM3, LLaMA, OLMo all use .model
        if hasattr(self.base, "model"):
            return self.base.model
        # Fallback for GPT-style models
        if hasattr(self.base, "transformer"):
            return self.base.transformer
        raise AttributeError("Cannot find transformer backbone in base model")

    def _get_embed_tokens(self) -> nn.Module:
        """Get the embedding layer."""
        backbone = self._get_model_backbone()
        if hasattr(backbone, "embed_tokens"):
            return backbone.embed_tokens
        if hasattr(backbone, "wte"):
            return backbone.wte
        raise AttributeError("Cannot find embedding layer in base model")

    def _get_layers(self) -> nn.ModuleList:
        """Get the transformer layers."""
        backbone = self._get_model_backbone()
        if hasattr(backbone, "layers"):
            return backbone.layers
        if hasattr(backbone, "h"):
            return backbone.h
        raise AttributeError("Cannot find transformer layers in base model")

    def _get_final_norm(self) -> nn.Module:
        """Get the final normalization layer."""
        backbone = self._get_model_backbone()
        if hasattr(backbone, "norm"):
            return backbone.norm
        if hasattr(backbone, "ln_f"):
            return backbone.ln_f
        raise AttributeError("Cannot find final norm in base model")

    def _get_lm_head(self) -> nn.Module:
        """Get the language model head."""
        if hasattr(self.base, "lm_head"):
            return self.base.lm_head
        raise AttributeError("Cannot find lm_head in base model")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | list | None]:
        """
        Forward pass through cognitive-augmented model.

        Args:
            input_ids: Token IDs of shape (batch, seq_len).
            attention_mask: Attention mask of shape (batch, seq_len).
            labels: Target token IDs for loss computation.

        Returns:
            Dict with keys: logits, lm_loss, surprise_loss, critic_losses,
            pred_losses, gating_actions, depth_signal.
        """
        # Step 1: Get embeddings from base model
        hidden = self._get_embed_tokens()(input_ids)

        # Step 2: Block 1 — Surprise gate
        depth_signal = None
        surprise_loss = torch.tensor(0.0, device=hidden.device)
        if self.surprise_gate is not None:
            hidden, depth_signal = self.surprise_gate(hidden)
            surprise_loss = self.surprise_gate.get_surprise_loss(hidden)

        # Step 3: Block 2 — Write to episodic memory
        if self.episodic_mem is not None:
            self.episodic_mem.reset(hidden.shape[0], hidden.device)
            self.episodic_mem.write(hidden)

        # Step 4: Transformer layers with optional Block 3 + 4
        critic_losses = []
        pred_losses = []
        prev_hidden = None
        layers = self._get_layers()

        for i, layer in enumerate(layers):
            # Block 4: predictive coding (modify input)
            if i < len(self.pred_coding) and self.pred_coding[i] is not None:
                hidden, pl = self.pred_coding[i](hidden, prev_hidden)
                pred_losses.append(pl)

            prev_hidden = hidden.detach()

            # Standard transformer layer
            layer_out = layer(hidden, attention_mask=attention_mask)
            # Handle both tuple and tensor returns
            hidden = layer_out[0] if isinstance(layer_out, tuple) else layer_out

            # Block 3: critic (auxiliary loss, no modification)
            if i < len(self.critics) and self.critics[i] is not None:
                critic_losses.append(self.critics[i](hidden))

        # Step 5: Block 2 — Read from episodic memory
        if self.episodic_mem is not None:
            hidden = self.episodic_mem.read(hidden)

        # Step 6: Block 5 — RL gating
        gating_actions = None
        if self.gating_policy is not None:
            gating_actions, _, _ = self.gating_policy.get_action(hidden)

        # Step 7: Output projection
        hidden = self._get_final_norm()(hidden)
        logits = self._get_lm_head()(hidden)

        # Step 8: Compute losses
        lm_loss = None
        if labels is not None:
            lm_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        return {
            "logits": logits,
            "lm_loss": lm_loss,
            "surprise_loss": surprise_loss,
            "critic_losses": critic_losses,
            "pred_losses": pred_losses,
            "gating_actions": gating_actions,
            "depth_signal": depth_signal,
        }
