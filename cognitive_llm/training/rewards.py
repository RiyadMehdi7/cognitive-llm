from __future__ import annotations

"""
Reward functions for RL training of cognitive blocks.

Provides reward signals for the gating policy based on benchmark-specific
metrics and general language modeling quality.
"""

import re
import torch


class RewardFunction:
    """
    Computes rewards for RL-trained cognitive blocks.

    Supports benchmark-specific rewards (GSM8K, ARC, MATH) and
    general perplexity-based rewards.

    Args:
        reward_type: One of 'gsm8k', 'arc', 'math', 'perplexity'.
        baseline_ppl: Baseline perplexity for normalized reward (default: 20.0).
    """

    def __init__(self, reward_type: str = "perplexity", baseline_ppl: float = 20.0):
        self.reward_type = reward_type
        self.baseline_ppl = baseline_ppl

    def compute(
        self,
        outputs: dict,
        predictions: list[str] | None = None,
        targets: list[str] | None = None,
    ) -> float:
        """
        Compute reward based on model outputs and optional predictions.

        Args:
            outputs: Dict from CognitiveModel.forward().
            predictions: Model text predictions (for answer-checking).
            targets: Ground truth answers.

        Returns:
            Scalar reward value.
        """
        if self.reward_type == "perplexity":
            return self._perplexity_reward(outputs)
        elif self.reward_type == "gsm8k":
            return self._gsm8k_reward(predictions, targets)
        elif self.reward_type == "arc":
            return self._arc_reward(predictions, targets)
        elif self.reward_type == "math":
            return self._math_reward(predictions, targets)
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")

    def _perplexity_reward(self, outputs: dict) -> float:
        """Reward based on perplexity improvement over baseline."""
        if outputs["lm_loss"] is None:
            return 0.0
        ppl = torch.exp(outputs["lm_loss"]).item()
        # Positive reward when perplexity is below baseline
        return max(0.0, (self.baseline_ppl - ppl) / self.baseline_ppl)

    def _gsm8k_reward(
        self, predictions: list[str] | None, targets: list[str] | None
    ) -> float:
        """
        GSM8K reward: check if final numerical answer matches.
        GSM8K answers are formatted as #### <number>.
        """
        if predictions is None or targets is None:
            return 0.0

        correct = 0
        for pred, target in zip(predictions, targets):
            pred_num = self._extract_gsm8k_answer(pred)
            target_num = self._extract_gsm8k_answer(target)
            if pred_num is not None and pred_num == target_num:
                correct += 1

        return correct / max(len(predictions), 1)

    def _arc_reward(
        self, predictions: list[str] | None, targets: list[str] | None
    ) -> float:
        """ARC reward: exact match on answer letter (A/B/C/D)."""
        if predictions is None or targets is None:
            return 0.0

        correct = 0
        for pred, target in zip(predictions, targets):
            pred_letter = self._extract_letter(pred)
            target_letter = self._extract_letter(target)
            if pred_letter and pred_letter == target_letter:
                correct += 1

        return correct / max(len(predictions), 1)

    def _math_reward(
        self, predictions: list[str] | None, targets: list[str] | None
    ) -> float:
        """MATH reward: check if boxed answer matches."""
        if predictions is None or targets is None:
            return 0.0

        correct = 0
        for pred, target in zip(predictions, targets):
            pred_ans = self._extract_boxed(pred)
            target_ans = self._extract_boxed(target)
            if pred_ans is not None and pred_ans == target_ans:
                correct += 1

        return correct / max(len(predictions), 1)

    @staticmethod
    def _extract_gsm8k_answer(text: str) -> float | None:
        """Extract numerical answer from GSM8K #### format."""
        match = re.search(r"####\s*([-\d,\.]+)", text)
        if match:
            try:
                return float(match.group(1).replace(",", ""))
            except ValueError:
                return None
        # Fallback: try to find the last number in text
        numbers = re.findall(r"[-\d,\.]+", text)
        if numbers:
            try:
                return float(numbers[-1].replace(",", ""))
            except ValueError:
                return None
        return None

    @staticmethod
    def _extract_letter(text: str) -> str | None:
        """Extract answer letter (A-D) from text."""
        match = re.search(r"\b([A-D])\b", text.upper())
        return match.group(1) if match else None

    @staticmethod
    def _extract_boxed(text: str) -> str | None:
        """Extract LaTeX \\boxed{...} answer."""
        match = re.search(r"\\boxed\{([^}]+)\}", text)
        return match.group(1).strip() if match else None
