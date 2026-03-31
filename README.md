# Cognitive LLM

<p align="center">
  Neuroscience-inspired architectural blocks for small language models.
</p>

<p align="center">
  We augment a frozen <a href="https://huggingface.co/HuggingFaceTB/SmolLM-360M">SmolLM-360M</a> backbone with six toggleable cognitive modules and run controlled ablations to measure how each block changes reasoning performance.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+"/>
  <img src="https://img.shields.io/badge/backbone-SmolLM--360M-5B8DEF" alt="Backbone SmolLM-360M"/>
  <img src="https://img.shields.io/badge/focus-reasoning%20ablations-1B7F3B" alt="Reasoning ablations"/>
  <img src="https://img.shields.io/badge/license-Apache%202.0-111827" alt="Apache 2.0"/>
</p>

**Paper:** [Memory Dominates Routing: A Controlled Screening of Neuroscience-Inspired Transformer Blocks](paper/main.pdf)

| What this repo studies | Strongest result from Phase 1 | Why it matters |
|---|---|---|
| Six neuroscience-inspired plug-in blocks on top of a frozen transformer | `B1 + B2 + B6` reaches **2.718 val loss** vs **6.299** baseline | Isolates which ideas actually help reasoning, instead of bundling many changes together |

## Architecture

<p align="center">
  <img src="assets/readme/architecture-overview.svg" width="980" alt="Clean overview of the Cognitive LLM architecture with the frozen transformer backbone in the center and six toggleable cognitive blocks arranged around it"/>
</p>

Six plug-in blocks wrap or augment a frozen transformer backbone. Each block is independently toggleable so the architecture stays interpretable during ablation runs.

| Block | Name | Inspiration | Role |
|:-----:|------|-------------|------|
| B1 | **SurpriseGate** | Predictive processing (Friston) | Gates hidden states by prediction-error magnitude |
| B2 | **EpisodicMemory** | Hippocampal rapid learning (Kumaran et al.) | Online key-value memory bank for episodic retrieval |
| B3 | **PerLayerCritic** | Prefrontal value signals (Wang) | Intermediate critic for TD-style training signal |
| B4 | **PredictiveCoding** | Visual cortex (Rao & Ballard) | Top-down error correction between layers |
| B5 | **RLGatingPolicy** | Meta-RL (Hassabis et al.) | Learned policy for block activation routing |
| B6 | **HomeostaticNorm** | Homeostatic plasticity | EMA-based activation stabilization |

## Results

Phase 1 screening on GSM8K with SmolLM-360M + LoRA (8 configurations, ranked by validation loss):

<p align="center">
  <img src="paper/figures/phase1_val_loss_ranking.png" width="780" alt="Phase 1 validation loss ranking"/>
</p>

| Rank | Configuration | Val Loss | vs Baseline |
|:----:|---------------|:--------:|:-----------:|
| 1 | B1 + B2 + B6 | 2.718 | **-56.8%** |
| 2 | B2 + B6 | 2.722 | **-56.8%** |
| 3 | B6 only | 2.743 | **-56.5%** |
| — | Baseline (LoRA only) | 6.299 | — |

**Key finding:** HomeostaticNorm (B6) alone accounts for most of the improvement. EpisodicMemory (B2) adds a small marginal gain. The PerLayerCritic (B3) consistently *hurts* performance, suggesting that intermediate critic signals interfere with the base model's learned representations.

## Repository Structure

```
cognitive_llm/
├── blocks/             Six cognitive block implementations (nn.Module)
├── models/             CognitiveModel wrapper with toggleable block composition
├── training/           Training loop, RL trainer, reward shaping, device abstraction
└── evaluation/         Benchmark runners, ablation framework
configs/                Experiment configuration (YAML)
tests/                  Unit tests for every block and the model wrapper
notebooks/              Phase 1 ablation notebook
paper/                  LaTeX manuscript, figures, and frozen results
train.py                Single-experiment entry point
```

## Quick Start

```bash
pip install -r requirements.txt
python train.py
```

Run the test suite:

```bash
pytest tests/ -v
```

Reproduce the paper figures:

```bash
python paper/scripts/make_figures.py
```

## Design Principles

- **Frozen backbone** — all blocks are additions on top of a LoRA-adapted base; original weights are never modified
- **Toggleable blocks** — each block is controlled by a config flag (`use_block1`, ..., `use_block6`) for clean ablation
- **Stability first** — HomeostaticNorm (B6) is always enabled alongside other blocks to prevent training divergence
- **Architecture-agnostic** — CognitiveModel auto-detects layer structure and works across SmolLM, OLMo, and LLaMA families

## License

[Apache 2.0](LICENSE)
