---
## GOAL
Validate which cognitive block combinations improve reasoning benchmarks
on OLMo 3 7B (HuggingFaceTB/SmolLM2-1.7B) using TPU v3/v4 (TRC quota).
Phase 1 screening on SmolLM-360M identified B6 (HomeostaticNorm) as the
dominant contributor. Phase 2 must confirm this on a 7B model and produce
paper-quality ablation results across multiple benchmarks.

## EXPERIMENT
file: train.py
command: python train.py --config configs/ablation/<name>.yaml --no_wandb
time_budget_minutes: 60
hardware: TPU v3/v4 (TRC)
model: HuggingFaceTB/SmolLM2-1.7B

## METRIC
primary: val_loss
secondary: gsm8k_acc, arc_challenge_acc, math_acc, hellaswag_acc
direction: minimize (val_loss), maximize (accuracies)
extract_pattern: "val_loss: ([0-9.]+)"

## STOPPING
improvement_threshold: 0.05
min_experiments: 5
max_no_improvement: 10
checkpoint_every: 5

## PHASE 1 FINDINGS (SmolLM-360M, 100 steps, GSM8K val_loss)
| Rank | Config      | val_loss | delta% | Key Finding                    |
|------|-------------|----------|--------|--------------------------------|
| 1    | B1+B2+B6   | 2.7175   | 56.85  | Best overall                   |
| 2    | B2+B6      | 2.7219   | 56.79  | Nearly identical to B1+B2+B6   |
| 3    | B6 only    | 2.7430   | 56.45  | B6 alone captures most gains   |
| 4    | B1+B6      | 2.9898   | 52.53  | B1 alone slightly helps        |
| 5    | B1+B2+B3+B6| 3.4507   | 45.21  | B3 hurts when added            |
| 6    | B2+B3+B6   | 3.4930   | 44.54  | Confirms B3 is detrimental     |
| 7    | B3+B6      | 3.5717   | 43.29  | B3 alone is worst performer    |
| base | LoRA only  | 6.2985   | 0.00   | Baseline                       |

## HYPOTHESIS_SPACE — Phase 2 (OLMo 3 7B, TPU)

HIGH PRIORITY — confirm Phase 1 findings at scale:
  1. Baseline: LoRA only (3 seeds)
  2. B6 only (3 seeds) — does B6 still dominate at 7B?
  3. B2+B6 (3 seeds) — does episodic memory help at larger scale?
  4. B1+B2+B6 (3 seeds) — was this Phase 1's best, confirm or reject

MED PRIORITY — explore what Phase 1 couldn't:
  5. B5+B6: RL gating (needs B1+B2 context, test standalone first)
  6. B1+B2+B5+B6: full routing stack without B3
  7. B2+B6 + lr=5e-5: lower LR may help at 7B scale
  8. B2+B6 + lr=3e-4: higher LR sensitivity check

LOW PRIORITY — scaling experiments:
  9. Best config + max_seq_len=2048 (vs 1024 default)
  10. Best config + longer training (20K steps vs 10K)

NEVER enable B4 — Phase 3 only.
NEVER enable B3 — Phase 1 showed it hurts.
B6 must always be ON when any other block is ON.

## BENCHMARK EVAL
After training, run lm-eval-harness on:
  - gsm8k (exact_match, target: +5-10% over baseline)
  - arc_challenge (acc_norm, target: +3-8%)
  - hellaswag (acc_norm, target: maintained or improved)
  - math (equiv, target: +3-6%)

Each experiment must report all 4 benchmarks for the paper.

## KNOWLEDGE

arxiv_queries:
  - "predictive processing surprise adaptive compute transformer 2024 2025"
  - "episodic memory key value buffer transformer working memory"
  - "homeostatic plasticity normalization activation stability transformer"
  - "neuroscience inspired language model architecture 2025 2026"
  - "cognitive architecture small language model reasoning 2025"
  - "GSM8K small model improvement architecture reasoning 2025"

foundational_papers:
  - "Rao Ballard 1999 predictive coding visual cortex"
  - "Hassabis 2017 neuroscience inspired artificial intelligence"
  - "Kumaran 2016 complementary learning systems"
  - "Wang 2018 prefrontal cortex meta reinforcement learning"

## VALIDATION

Expected behavior at 7B scale:
  B6 HomeostaticNorm:
    Should still be the dominant contributor.
    At 7B, effect may be smaller (larger models are already more stable).
    ANOMALY if B6 has <10% improvement at 7B.
  B2 EpisodicMemory:
    Should show larger improvement at 7B (more capacity to use memory).
    Should help on longer sequences.
  B1 SurpriseGate:
    Marginal at 360M. May shine at 7B with deeper layers.
    ANOMALY if B1 shows >10% standalone improvement.
  B5 GatingPolicy:
    Expected to be negligible standalone.
    Should only help when B1+B2 provide useful signals to gate.

## NOTES
Phase: 2 — TPU v3/v4 (TRC)
Model: HuggingFaceTB/SmolLM2-1.7B (Apache 2.0)
Baseline: not yet measured — first experiment on TPU
Status: 61/61 unit tests passing, smoke test passing
Block 3: DISABLED — Phase 1 showed it hurts performance
Block 4: Phase 3 only — never enable
---
