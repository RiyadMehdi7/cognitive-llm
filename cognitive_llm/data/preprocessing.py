"""Prompt formatting and preprocessing for different model types."""

from __future__ import annotations

from typing import List

# Chain-of-thought prompt templates per model family
_TEMPLATES = {
    "qwen": (
        "<|im_start|>user\n"
        "Solve the following step by step.\n\n{question}<|im_end|>\n"
        "<|im_start|>assistant\n"
        "Let me solve this step by step.\n\n{answer}"
    ),
    "llama": (
        "[INST] Solve the following step by step.\n\n{question} [/INST]\n"
        "Let me solve this step by step.\n\n{answer}"
    ),
    "generic": (
        "Question: {question}\n"
        "Answer: Let me solve this step by step.\n{answer}"
    ),
}


def format_dataset(
    dataset: List[dict],
    model_type: str = "generic",
) -> List[dict]:
    """
    Format a dataset with CoT prompts for a specific model type.

    Each item gets a 'formatted_prompt' key added containing the full
    prompt string ready for tokenization.

    Args:
        dataset: List of dicts with 'question' and 'answer' keys.
        model_type: One of 'qwen', 'llama', or 'generic'.

    Returns:
        The same list with 'formatted_prompt' key added to each dict.
    """
    model_type = model_type.lower()

    # Resolve model type from model ID strings
    if "qwen" in model_type:
        template_key = "qwen"
    elif "llama" in model_type or "olmo" in model_type:
        template_key = "llama"
    else:
        template_key = "generic"

    template = _TEMPLATES[template_key]

    for item in dataset:
        item["formatted_prompt"] = template.format(
            question=item["question"],
            answer=item["answer"],
        )

    return dataset
