"""
Cognitive Model Wrapper — injects cognitive blocks into a base HuggingFace model.

Supports SmolLM3, OLMo 3, and any LlamaForCausalLM-compatible model.
All blocks are toggled via config flags to support clean ablation studies.
"""

import importlib

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

    @staticmethod
    def _is_norm_like(module: nn.Module, d_model: int) -> bool:
        """Detect LayerNorm/RMSNorm-style modules by interface, not class name."""
        class_name = module.__class__.__name__.lower()
        if "layernorm" in class_name or "rmsnorm" in class_name:
            return getattr(module, "weight", None) is not None and module.weight.shape == (d_model,)
        return isinstance(module, nn.LayerNorm) and module.normalized_shape == (d_model,)

    @staticmethod
    def _resolve_child(module: nn.Module, part: str) -> nn.Module:
        """Traverse dotted module paths that may contain numeric indices."""
        if part.isdigit():
            return module[int(part)]
        return getattr(module, part)

    def _get_parent(self, name: str) -> nn.Module:
        """Get parent module from a dotted name path."""
        parts = name.split(".")
        module = self.base
        for part in parts[:-1]:
            module = self._resolve_child(module, part)
        return module

    def _replace_layer_norms(self, d_model: int) -> None:
        """Wrap all LayerNorm/RMSNorm-style modules with HomeostaticNorm."""
        replacements = []
        for name, module in self.base.named_modules():
            if self._is_norm_like(module, d_model):
                replacements.append((name, module))

        for name, module in replacements:
            parent = self._get_parent(name)
            attr = name.split(".")[-1]
            replacement = HomeostaticNorm.from_norm(module, d_model)
            if attr.isdigit():
                parent[int(attr)] = replacement
            else:
                setattr(parent, attr, replacement)

    def _get_model_backbone(self) -> nn.Module:
        """Get the transformer backbone (handles PEFT wrappers and different architectures).

        Walks down .model / .base_model attributes until finding a module with
        embed_tokens or wte, which indicates the actual transformer backbone.
        PEFT hierarchy: PeftModel.base_model -> CausalLM.model -> backbone
        """
        module = self.base
        for _ in range(5):  # safety limit
            if hasattr(module, "embed_tokens") or hasattr(module, "wte"):
                return module
            # PEFT wraps via .base_model, HF models use .model
            if hasattr(module, "base_model"):
                module = module.base_model
            elif hasattr(module, "model"):
                module = module.model
            elif hasattr(module, "transformer"):
                return module.transformer
            else:
                break
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
        """Get the language model head (handles PEFT wrappers)."""
        module = self.base
        for _ in range(5):
            if hasattr(module, "lm_head"):
                return module.lm_head
            if hasattr(module, "model"):
                module = module.model
            else:
                break
        raise AttributeError("Cannot find lm_head in base model")

    def _move_blocks_to_match(self, hidden: torch.Tensor) -> None:
        """Move cognitive blocks to the hidden-state device once.

        Keep cognitive block parameters in their native dtype for stability on
        4-bit / fp16 debug runs. The blocks cast activations internally when
        needed, which avoids training them directly in low precision.
        """
        if hasattr(self, "_blocks_moved"):
            return

        target_device = hidden.device
        if self.surprise_gate is not None:
            self.surprise_gate = self.surprise_gate.to(device=target_device)
        if self.episodic_mem is not None:
            self.episodic_mem = self.episodic_mem.to(device=target_device)
        for i, c in enumerate(self.critics):
            if c is not None:
                self.critics[i] = c.to(device=target_device)
        for i, p in enumerate(self.pred_coding):
            if p is not None:
                self.pred_coding[i] = p.to(device=target_device)
        if self.gating_policy is not None:
            self.gating_policy = self.gating_policy.to(device=target_device)
        self._blocks_moved = True

    @staticmethod
    def _should_use_manual_stack(
        surprise_gate: nn.Module | None,
        critics: nn.ModuleList,
        pred_coding: nn.ModuleList,
    ) -> bool:
        """Use an owned decoder loop when blocks need true inter-layer control."""
        return (
            surprise_gate is not None
            or any(c is not None for c in critics)
            or any(p is not None for p in pred_coding)
        )

    def _manual_stack_supported(self) -> bool:
        """Return True for decoder backbones whose layer loop we can own safely."""
        backbone = self._get_model_backbone()
        module = importlib.import_module(backbone.__class__.__module__)
        has_layers = hasattr(backbone, "layers")
        has_norm = hasattr(backbone, "norm")
        has_rotary = hasattr(backbone, "rotary_emb")
        has_mask_builder = hasattr(module, "create_causal_mask")
        return has_layers and has_norm and has_rotary and has_mask_builder

    @staticmethod
    def _compute_lm_loss(
        logits: torch.Tensor,
        labels: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Compute mean and per-example causal LM losses."""
        if labels is None:
            return None, None

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(shift_labels.shape)
        valid_tokens = shift_labels.ne(-100)
        per_example_loss = (token_loss * valid_tokens).sum(dim=1) / valid_tokens.sum(dim=1).clamp_min(1)
        return per_example_loss.mean(), per_example_loss

    def _prepare_decoder_context(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor | dict | None,
    ) -> dict[str, torch.Tensor | dict]:
        """Build the shared decoder-loop context used by Llama-like backbones."""
        backbone = self._get_model_backbone()
        module = importlib.import_module(backbone.__class__.__module__)
        cache_position = torch.arange(hidden.shape[1], device=hidden.device)
        position_ids = cache_position.unsqueeze(0)

        if isinstance(attention_mask, dict):
            mask_mapping = attention_mask
        else:
            mask_kwargs = {
                "config": backbone.config,
                "input_embeds": hidden,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": None,
                "position_ids": position_ids,
            }
            full_mask = module.create_causal_mask(**mask_kwargs)
            mask_mapping = {
                "default": full_mask,
                "full_attention": full_mask,
            }
            if getattr(backbone, "has_sliding_layers", False):
                create_sliding = getattr(module, "create_sliding_window_causal_mask", None)
                if create_sliding is not None:
                    mask_mapping["sliding_attention"] = create_sliding(**mask_kwargs)

        position_embeddings = backbone.rotary_emb(hidden, position_ids)
        return {
            "position_ids": position_ids,
            "cache_position": cache_position,
            "mask_mapping": mask_mapping,
            "position_embeddings": position_embeddings,
        }

    @staticmethod
    def _select_layer_mask(layer: nn.Module, mask_mapping: dict[str, torch.Tensor]) -> torch.Tensor:
        """Pick the right causal mask for the current decoder layer."""
        attention_type = getattr(layer, "attention_type", None)
        if attention_type in mask_mapping:
            return mask_mapping[attention_type]
        if "default" in mask_mapping:
            return mask_mapping["default"]
        return next(iter(mask_mapping.values()))

    def _depth_signal_to_layer_budget(
        self,
        depth_signal: torch.Tensor | None,
        n_layers: int,
    ) -> torch.Tensor | None:
        """Map surprise buckets to how many layers each token is allowed to traverse."""
        if depth_signal is None or self.surprise_gate is None:
            return None

        n_buckets = max(self.surprise_gate.n_buckets, 1)
        layer_budget = ((depth_signal + 1) * n_layers + n_buckets - 1) // n_buckets
        return layer_budget.clamp_(1, n_layers)

    @staticmethod
    def _apply_depth_routing(
        previous_hidden: torch.Tensor,
        layer_output: torch.Tensor,
        layer_idx: int,
        layer_budget: torch.Tensor | None,
    ) -> torch.Tensor:
        """Keep low-surprise tokens on shallower representations in later layers."""
        if layer_budget is None:
            return layer_output
        active_mask = (layer_budget > layer_idx).unsqueeze(-1)
        return torch.where(active_mask, layer_output, previous_hidden)

    def _compute_critic_losses(
        self,
        hidden_states: list[torch.Tensor] | tuple[torch.Tensor, ...],
        per_example_loss: torch.Tensor | None,
        gamma: float = 0.95,
    ) -> list[torch.Tensor]:
        """Train critics using TD(0) bootstrapping across layers.

        The deepest critic bootstraps from the final LM loss. Each earlier
        critic bootstraps from the next deeper critic's value estimate.
        This creates a true temporal-difference learning signal across
        network depth, enabling distributed credit assignment.
        """
        if per_example_loss is None:
            return []

        # Collect active (critic_index, layer_index) pairs in layer order
        active = [
            (i, critic)
            for i, critic in enumerate(self.critics)
            if critic is not None
        ]
        if not active:
            return []

        # Bootstrap backwards: deepest critic first
        critic_losses = [None] * len(active)
        next_value = per_example_loss.detach()  # terminal value = actual LM loss

        for rev_idx, (layer_idx, critic) in enumerate(reversed(active)):
            hidden_idx = min(layer_idx, len(hidden_states) - 1)
            td_loss, current_value = critic.compute_td_loss(
                hidden_states[hidden_idx],
                next_value,
                per_example_loss,
                gamma=gamma,
            )
            critic_losses[len(active) - 1 - rev_idx] = td_loss
            next_value = current_value  # pass detached value to shallower critic

        return critic_losses

    def _apply_gating_actions(
        self,
        hidden: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the gating policy's selected action to each batch element."""
        if self.gating_policy is None:
            return hidden

        result = hidden.clone()

        if self.episodic_mem is not None:
            write_mask = actions == self.gating_policy.ACTION_WRITE_MEMORY
            if write_mask.any():
                self.episodic_mem.write(result[write_mask], batch_indices=write_mask)

            recall_mask = actions == self.gating_policy.ACTION_RECALL_MEMORY
            if recall_mask.any():
                result[recall_mask] = self.episodic_mem.read(
                    result[recall_mask], batch_indices=recall_mask
                )

        deepen_mask = actions == self.gating_policy.ACTION_DEEPEN_COMPUTE
        if deepen_mask.any():
            result[deepen_mask] = self.gating_policy.deepen(result[deepen_mask])

        return result

    def _manual_forward(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor | dict | None,
        labels: torch.Tensor | None,
        depth_signal: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor], list[torch.Tensor], torch.Tensor | None]:
        """Run a true decoder loop where cognitive blocks can alter computation."""
        layers = self._get_layers()
        context = self._prepare_decoder_context(hidden, attention_mask)
        layer_budget = self._depth_signal_to_layer_budget(depth_signal, len(layers))

        current_hidden = hidden
        prev_hidden = None
        layer_hidden_states = []
        pred_losses = []

        for i, layer in enumerate(layers):
            layer_input = current_hidden
            if i < len(self.pred_coding) and self.pred_coding[i] is not None:
                current_hidden, pred_loss = self.pred_coding[i](current_hidden, prev_hidden)
                pred_losses.append(pred_loss)

            layer_output = layer(
                current_hidden,
                attention_mask=self._select_layer_mask(layer, context["mask_mapping"]),
                position_ids=context["position_ids"],
                past_key_values=None,
                use_cache=False,
                cache_position=context["cache_position"],
                position_embeddings=context["position_embeddings"],
            )
            current_hidden = self._apply_depth_routing(
                layer_input,
                layer_output,
                i,
                layer_budget,
            )
            prev_hidden = current_hidden.detach()
            layer_hidden_states.append(current_hidden)

        working_hidden = self._get_final_norm()(current_hidden)
        if self.episodic_mem is not None:
            working_hidden = self.episodic_mem.read(working_hidden)

        gating_actions = None
        if self.gating_policy is not None:
            gating_actions, _, _ = self.gating_policy.get_action(working_hidden)
            working_hidden = self._apply_gating_actions(working_hidden, gating_actions)

        logits = self._get_lm_head()(working_hidden)
        lm_loss, per_example_loss = self._compute_lm_loss(logits, labels)
        critic_losses = self._compute_critic_losses(layer_hidden_states, per_example_loss)
        return logits, lm_loss, critic_losses, pred_losses, gating_actions

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

        # Step 0: Lazily move cognitive blocks to match hidden device & dtype (once)
        self._move_blocks_to_match(hidden)

        # Step 2: Block 1 — Surprise gate
        depth_signal = None
        surprise_loss = torch.tensor(0.0, device=hidden.device)
        if self.surprise_gate is not None:
            hidden, depth_signal = self.surprise_gate(hidden)
            surprise_loss = self.surprise_gate.get_surprise_loss(hidden)

        # Step 3: Block 2 — Write to episodic memory (pre-transformer)
        if self.episodic_mem is not None:
            self.episodic_mem.reset(hidden.shape[0], hidden.device)
            self.episodic_mem.write(hidden)

        critic_losses = []
        pred_losses = []
        gating_actions = None
        if self._should_use_manual_stack(self.surprise_gate, self.critics, self.pred_coding):
            if not self._manual_stack_supported():
                raise NotImplementedError(
                    "True cognitive blocks currently require a Llama/SmolLM-style decoder backbone."
                )
            logits, lm_loss, critic_losses, pred_losses, gating_actions = self._manual_forward(
                hidden,
                attention_mask,
                labels,
                depth_signal,
            )
        else:
            need_hidden = self.episodic_mem is not None or self.gating_policy is not None
            base_out = self.base(
                inputs_embeds=hidden,
                attention_mask=attention_mask,
                labels=labels if not need_hidden else None,
                output_hidden_states=need_hidden,
                return_dict=True,
            )
            logits = base_out.logits
            lm_loss = base_out.loss

            if need_hidden:
                working_hidden = base_out.hidden_states[-1]
                if self.episodic_mem is not None:
                    working_hidden = self.episodic_mem.read(working_hidden)
                if self.gating_policy is not None:
                    gating_actions, _, _ = self.gating_policy.get_action(working_hidden)
                    working_hidden = self._apply_gating_actions(working_hidden, gating_actions)
                logits = self._get_lm_head()(working_hidden)
                lm_loss, _ = self._compute_lm_loss(logits, labels)

        return {
            "logits": logits,
            "lm_loss": lm_loss,
            "surprise_loss": surprise_loss,
            "critic_losses": critic_losses,
            "pred_losses": pred_losses,
            "gating_actions": gating_actions,
            "depth_signal": depth_signal,
        }
