"""Cognitive architecture blocks for transformer augmentation."""

from .block1_surprise_gate import SurpriseGate
from .block2_episodic_memory import EpisodicMemory
from .block3_per_layer_critic import LayerCritic
from .block4_predictive_coding import PredictiveCodingLayer
from .block5_rl_gating import GatingPolicy
from .block6_homeostatic_norm import HomeostaticNorm

__all__ = [
    "SurpriseGate",
    "EpisodicMemory",
    "LayerCritic",
    "PredictiveCodingLayer",
    "GatingPolicy",
    "HomeostaticNorm",
]
