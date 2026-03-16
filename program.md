---
## GOAL
Determine which combination of the 6 cognitive blocks
improves val_loss on GSM8K over a LoRA-only baseline
on SmolLM3 3B. Find evidence for NeurIPS/ICLR submission.

## EXPERIMENT
file: train.py
command: python train.py
time_budget_minutes: 10

## METRIC
primary: val_loss
direction: minimize
extract_pattern: "val_loss: ([0-9.]+)"

## STOPPING
improvement_threshold: 0.05
min_experiments: 5
max_no_improvement: 15
gpu_budget_usd: 8.00
api_budget_usd: 3.00
checkpoint_every: 10

## HYPOTHESIS_SPACE

HIGH PRIORITY — single block ablations (run first):
  1. Baseline: all blocks OFF (LoRA only)
  2. B6 only: HomeostaticNorm alone
  3. B1+B6: SurpriseGate + HomeostaticNorm
  4. B2+B6: EpisodicMemory + HomeostaticNorm
  5. B3+B6: PerLayerCritic + HomeostaticNorm

MED PRIORITY — pairwise combinations:
  6. B1+B2+B6
  7. B1+B3+B6
  8. B2+B3+B6
  9. B1+B2+B3+B6

MED PRIORITY — hyperparameter sensitivity on best config:
  10. Best config + lr=1e-4
  11. Best config + lr=5e-4
  12. Best config + max_seq_len=1024

LOW PRIORITY — full Phase 1 stack:
  13. B1+B2+B3+B5+B6

NEVER enable B4 — it is Phase 2 only.
B6 must always be ON when any other block is ON.

## KNOWLEDGE

arxiv_queries:
  - "predictive processing surprise adaptive compute transformer 2024 2025"
  - "token importance routing adaptive depth language model"
  - "episodic memory key value buffer transformer working memory"
  - "hippocampus rapid one-shot learning language model 2024 2025"
  - "per layer value estimation credit assignment transformer"
  - "temporal difference learning intermediate reward deep network"
  - "learned gating policy orchestration language model blocks"
  - "mixture of experts dynamic routing transformer 2025 2026"
  - "homeostatic plasticity normalization activation stability"
  - "adaptive layer normalization running statistics transformer"
  - "neuroscience inspired language model architecture 2025 2026"
  - "cognitive architecture small language model reasoning"
  - "biologically plausible transformer attention mechanism 2025"
  - "working memory prefrontal cortex language model reasoning"
  - "GSM8K small model improvement architecture reasoning 2025"
  - "ARC challenge language model cognitive architecture 2025"

foundational_papers:
  - "Rao Ballard 1999 predictive coding visual cortex"
  - "Hassabis 2017 neuroscience inspired artificial intelligence"
  - "Kumaran 2016 complementary learning systems"
  - "Wang 2018 prefrontal cortex meta reinforcement learning"
  - "Baars global workspace theory consciousness"

## VALIDATION

Expected behavior per block if hypothesis is correct:
  B1 SurpriseGate:
    val_loss improves on rare and complex tokens.
    attn_entropy decreases — more selective routing.
  B2 EpisodicMemory:
    val_loss improves on sequences longer than 256 tokens.
    few-shot accuracy improves.
  B3 LayerCritic:
    training stability improves — lower loss variance.
    gradient_norm more consistent across steps.
  B5 GatingPolicy:
    improvement only visible when B1+B2 both active.
    negligible standalone contribution expected.
  B6 HomeostaticNorm:
    prevents loss spikes.
    small standalone improvement but stabilizes all other blocks.

## NOTES
Phase: 1 — debug on Colab T4 16GB
Model: SmolLM3-3B 4-bit quantization
Baseline: not yet measured — agent runs EXP_000 first
Status: 42/42 unit tests passing
Block 4: Phase 2 only — never enable in train.py
---
