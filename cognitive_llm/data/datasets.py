"""Dataset loaders for GSM8K, MATH, and mixed reasoning datasets."""

from __future__ import annotations

from typing import List


def load_gsm8k_dataset(split: str = "train", max_samples: int = 0) -> List[dict]:
    """
    Load the GSM8K dataset from HuggingFace.

    Args:
        split: Dataset split ("train" or "test").
        max_samples: Maximum number of samples to return. 0 means all.

    Returns:
        List of dicts with keys: id, question, answer.
    """
    from datasets import load_dataset

    split_str = f"{split}[:{max_samples}]" if max_samples > 0 else split
    raw = load_dataset("gsm8k", "main", split=split_str)

    results = []
    for i, example in enumerate(raw):
        results.append({
            "id": f"gsm8k_{split}_{i}",
            "question": example["question"],
            "answer": example["answer"],
        })
    return results


def load_math_dataset(split: str = "train", max_samples: int = 0) -> List[dict]:
    """
    Load the MATH dataset from HuggingFace.

    Args:
        split: Dataset split ("train" or "test").
        max_samples: Maximum number of samples to return. 0 means all.

    Returns:
        List of dicts with keys: id, question, answer.
    """
    from datasets import load_dataset

    split_str = f"{split}[:{max_samples}]" if max_samples > 0 else split
    raw = load_dataset("hendrycks/competition_math", split=split_str)

    results = []
    for i, example in enumerate(raw):
        results.append({
            "id": f"math_{split}_{i}",
            "question": example["problem"],
            "answer": example["solution"],
        })
    return results


def create_mixed_dataset(
    gsm8k_ratio: float = 0.5,
    math_ratio: float = 0.5,
    total_samples: int = 100,
) -> List[dict]:
    """
    Create a mixed dataset from GSM8K and MATH with specified ratios.

    Args:
        gsm8k_ratio: Fraction of samples from GSM8K (0.0 to 1.0).
        math_ratio: Fraction of samples from MATH (0.0 to 1.0).
        total_samples: Total number of samples in the mixed dataset.

    Returns:
        List of dicts with keys: id, question, answer.
    """
    assert abs(gsm8k_ratio + math_ratio - 1.0) < 1e-6, (
        f"Ratios must sum to 1.0, got {gsm8k_ratio + math_ratio}"
    )

    n_gsm8k = int(total_samples * gsm8k_ratio)
    n_math = total_samples - n_gsm8k

    mixed = []
    if n_gsm8k > 0:
        mixed.extend(load_gsm8k_dataset("train", max_samples=n_gsm8k))
    if n_math > 0:
        mixed.extend(load_math_dataset("train", max_samples=n_math))

    return mixed
