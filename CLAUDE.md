# Cognitive LLM Architecture — Claude Code Guardrails

## Project Overview
Research project implementing a Cognitive LLM Architecture: a transformer-based language model (SmolLM3 3B / OLMo 3 7B) enhanced with 6 neuroscience-inspired architectural blocks. Goal: demonstrate that a small model with cognitive additions can outperform larger vanilla models on reasoning benchmarks.

## Critical Implementation Rules

### Block Development Order
1. **NEVER implement more than one block at a time** — validate each before proceeding.
2. Implementation order: Block 6 → Block 1 → Block 2 → Block 3 → Block 5 → Block 4 (Phase 2 only).
3. **Block 4 (Predictive Coding) is Phase 2 ONLY** — do not implement in Phase 1.
4. Every block MUST have unit tests in the `tests/` directory before integration.

### Base Model Rules
- **NEVER modify base model weights directly** — all blocks are additions on top of frozen or LoRA-adapted base.
- All blocks must be toggleable via config dict flags (`use_block1`, `use_block2`, etc.) for ablation.
- When adding a block to CognitiveModel, add a corresponding ablation flag in ALL config YAML files.
- Block 6 (HomeostaticNorm) should ALWAYS be ON when any other block is enabled.

### Training Stability
If a block causes training loss to diverge (NaN or explosive growth):
1. First check HomeostaticNorm is enabled (Block 6)
2. Reduce lambda weight for that block's auxiliary loss by 10x
3. Add gradient clipping: `max_norm=1.0`
4. Reduce learning rate by 2x
5. If still unstable — disable the block and report as negative result (this is valid)

### Loss Weights (Starting Values)
```python
LAMBDA_CONFIG = {
    'surprise':    0.01,   # Block 1 auxiliary
    'critic':      0.1,    # Block 3 actor-critic
    'predictive':  0.05,   # Block 4 pred coding aux
}
```

### Hardware Phases
| Phase | Hardware | Model | Purpose |
|-------|----------|-------|---------|
| 1 - Debug | Colab T4 16GB | SmolLM3 3B 4-bit | Block dev & testing |
| 2 - Ablation | Kaggle A100 30h/wk | SmolLM3 3B bf16 | Clean ablation results |
| 3 - Main results | TPU v3/v4 (TRC) | OLMo 3 7B | Paper main results |
| 4 - Scaling | TPU v4 (TRC) | OLMo 3 32B | Scaling experiment |

### Experiment Tracking
- Use **wandb** for ALL experiment tracking — log loss, perplexity, and benchmark scores.
- Save checkpoints every 500 steps with the block configuration in the filename.
- Use `lm-eval-harness` for standardized benchmark evaluation.

### Target Benchmarks
- GSM8K (Accuracy, +5-10%)
- ARC-Challenge (Accuracy, +3-8%)
- MATH (Pass@1, +3-6%)
- HellaSwag (Accuracy, maintained or improved)

## File Structure
```
cognitive_llm/
├── blocks/          # One file per cognitive block
├── models/          # CognitiveModel wrapper
├── training/        # Training loops, PPO, rewards
├── evaluation/      # Benchmarks, ablation runner
├── notebooks/       # Colab & Kaggle notebooks
├── configs/         # YAML configs per hardware phase
└── tests/           # Unit tests per block
```

## Code Style
- Type hints on all function signatures
- Docstrings on all classes and public methods
- PyTorch nn.Module for all blocks
- All blocks must implement `forward()` returning tensors
- Use `torch.no_grad()` for EMA/running stat updates

## Base Models
- SmolLM3 3B: `HuggingFaceTB/SmolLM3-3B` (Apache 2.0)
- OLMo 3 7B: `allenai/OLMo-3-7B` (Apache 2.0)
