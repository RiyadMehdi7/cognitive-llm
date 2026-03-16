"""Regression tests for the CognitiveModel wrapper."""

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from cognitive_llm.blocks.block6_homeostatic_norm import HomeostaticNorm
from cognitive_llm.models.cognitive_model import CognitiveModel


def make_base_model():
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
    )
    return LlamaForCausalLM(config)


class TestCognitiveModel:
    def test_forward_with_block1_and_block6_uses_base_forward(self):
        """Phase 1 debug path should work on Llama-family models."""
        base_model = make_base_model()
        model = CognitiveModel(
            base_model,
            {
                "use_block1": True,
                "use_block2": False,
                "use_block3": False,
                "use_block4": False,
                "use_block5": False,
                "use_block6": True,
            },
        )
        input_ids = torch.randint(0, 128, (2, 6))

        outputs = model(input_ids)

        assert outputs["logits"].shape == (2, 6, 128)
        assert outputs["depth_signal"] is not None
        assert outputs["critic_losses"] == []
        assert outputs["pred_losses"] == []

    def test_forward_with_labels_returns_lm_loss(self):
        """The delegated base-model path should preserve causal LM loss."""
        base_model = make_base_model()
        model = CognitiveModel(
            base_model,
            {
                "use_block1": True,
                "use_block2": False,
                "use_block3": False,
                "use_block4": False,
                "use_block5": False,
                "use_block6": True,
            },
        )
        input_ids = torch.randint(0, 128, (2, 6))

        outputs = model(input_ids, labels=input_ids)

        assert outputs["lm_loss"] is not None
        assert outputs["lm_loss"].dim() == 0

    def test_block6_wraps_llama_norms(self):
        """Llama-family RMSNorm modules should be wrapped by HomeostaticNorm."""
        base_model = make_base_model()

        _ = CognitiveModel(
            base_model,
            {
                "use_block1": False,
                "use_block2": False,
                "use_block3": False,
                "use_block4": False,
                "use_block5": False,
                "use_block6": True,
            },
        )

        assert any(isinstance(module, HomeostaticNorm) for module in base_model.modules())

    def test_block3_returns_scalar_losses_not_raw_values(self):
        """Block 3 should contribute scalar losses, not raw value predictions."""
        base_model = make_base_model()
        model = CognitiveModel(
            base_model,
            {
                "use_block1": False,
                "use_block2": False,
                "use_block3": True,
                "use_block4": False,
                "use_block5": False,
                "use_block6": False,
            },
        )
        input_ids = torch.randint(0, 128, (2, 6))

        outputs = model(input_ids, labels=input_ids)

        assert outputs["critic_losses"]
        assert all(loss.dim() == 0 for loss in outputs["critic_losses"])
        assert all(loss.item() >= 0.0 for loss in outputs["critic_losses"])

    def test_block1_depth_signal_changes_forward_path(self):
        """Forcing different routing budgets should change the decoder output."""
        base_model = make_base_model()
        model = CognitiveModel(
            base_model,
            {
                "use_block1": True,
                "use_block2": False,
                "use_block3": False,
                "use_block4": False,
                "use_block5": False,
                "use_block6": False,
            },
        )
        input_ids = torch.randint(0, 128, (2, 6))

        model.surprise_gate.forward = lambda x: (x, torch.zeros(x.shape[:2], dtype=torch.long, device=x.device))
        shallow = model(input_ids)["logits"]

        model.surprise_gate.forward = lambda x: (
            x,
            torch.full(x.shape[:2], model.surprise_gate.n_buckets - 1, dtype=torch.long, device=x.device),
        )
        deep = model(input_ids)["logits"]

        assert not torch.allclose(shallow, deep)
