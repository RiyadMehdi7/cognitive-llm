# Cognitive LLM Architecture

A transformer-based language model enhanced with six neuroscience-inspired architectural blocks. This research project demonstrates that a small model (3B parameters) with cognitive additions can significantly outperform larger vanilla models on reasoning benchmarks.

## Research Hypothesis

A small LLM (SmolLM3 3B or OLMo 3 7B) augmented with adaptive compute routing, online episodic memory, distributed RL credit assignment, predictive coding, learned gating, and homeostatic regulation will achieve reasoning performance competitive with models 3-5x larger — while being more computationally efficient.

## Cognitive Blocks

| Block | Name | Position | Purpose |
|-------|------|----------|---------|
| B1 | Meta-Surprise Gate | After embedding, before Layer 1 | Adaptive compute routing based on token surprise |
| B2 | Episodic Memory | Before/after transformer stack | Online key-value working memory buffer |
| B3 | Per-Layer Critic | Every 4th transformer layer | Distributed value estimation for RL credit assignment |
| B4 | Predictive Coding | Between transformer layers | Inter-layer prediction and error propagation (Phase 2) |
| B5 | RL Gating Policy | After transformer stack | Learned orchestration of all cognitive blocks |
| B6 | Homeostatic Norm | Replaces LayerNorm | Activation stability via EMA-tracked running statistics |

## Target Benchmarks

| Benchmark | Metric | Target Delta |
|-----------|--------|-------------|
| GSM8K | Accuracy | +5-10% over baseline |
| ARC-Challenge | Accuracy | +3-8% over baseline |
| MATH | Pass@1 | +3-6% over baseline |
| HellaSwag | Accuracy | Maintained or improved |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run unit tests
pytest tests/ -v

# Phase 1: Colab debug (see notebooks/colab_debug.ipynb)
# Phase 2: Kaggle ablation (see notebooks/kaggle_ablation.ipynb)
```

## Project Structure

```
cognitive_llm/
├── blocks/              # Six cognitive architecture blocks
├── models/              # CognitiveModel wrapper
├── training/            # Training loops, PPO, rewards
├── evaluation/          # Benchmarks, ablation runner
notebooks/               # Colab & Kaggle notebooks
configs/                 # YAML configs per hardware phase
tests/                   # Unit tests per block
```

## Hardware Phases

1. **Debug** — Colab T4 16GB, SmolLM3 3B 4-bit
2. **Ablation** — Kaggle A100 30h/wk, SmolLM3 3B bf16
3. **Main results** — TPU v3/v4 (TRC), OLMo 3 7B
4. **Scaling** — TPU v4 (TRC), OLMo 3 32B

## Base Models

- **SmolLM3 3B**: `HuggingFaceTB/SmolLM3-3B` (Apache 2.0)
- **OLMo 3 7B**: `allenai/OLMo-3-7B` (Apache 2.0)

## License

Apache 2.0
