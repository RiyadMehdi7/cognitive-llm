"""Regression tests for the CognitiveModel wrapper."""

import torch
from transformers import LlamaConfig, LlamaForCausalLM

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
