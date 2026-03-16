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

## RESEARCH AGENT MODE

Activated when human says: START RESEARCH SESSION

### Startup Sequence
1. Read program.md fully
2. Read results.tsv — understand what has been tried
3. Fetch papers:
   - Run each query in program.md KNOWLEDGE.arxiv_queries
   - Fetch up to 10 papers per query, sorted by date descending
   - Score each abstract for relevance to current hypothesis (0.0-1.0)
   - Load full text only for top 3 most relevant papers
   - Summarize each paper to 300 tokens
   - Save summaries to papers/summaries/{arxiv_id}.txt
   - Discard full text after summarizing — never keep in context
   - Also search for foundational_papers by author+year
4. Run baseline (EXP_000):
   - Set all block flags to False in train.py CONFIG
   - Run 3 times, record median val_loss
   - Record in results.tsv
   - Compute TARGET = baseline_val_loss × 0.95
   - Revert train.py after baseline
5. Log: "SESSION STARTED. Baseline=X.XXXX. Target=X.XXXX. Papers=N."
6. Enter main loop

### Main Loop

Repeat until stopping criteria met:

STEP 1 — HYPOTHESIZE
  Pick next untried experiment from HYPOTHESIS_SPACE in priority order.
  Write to lab_notebook.md:

    ## EXP_NNN | timestamp
    Hypothesis: [block combination or config change]
    Motivation: [which paper or pattern from results.tsv]
    Prediction: val_loss will change by ~X% because [reason]
    Change: [exact CONFIG fields to modify]

STEP 2 — IMPLEMENT
  Modify train.py. Two types of changes allowed:

  TYPE A — Hypothesis change (modifying CONFIG):
    Change only the CONFIG dict fields listed in the hypothesis.
    This counts as an experiment. Creates EXP_NNN entry.

  TYPE B — Code fix (experiment crashed with code error):
    Diagnose error from experiments/exp_NNN.log.
    Fix the minimal code change needed to make it run.
    Log under "Code Fix:" in current lab_notebook.md entry.
    Does NOT count as a new experiment.
    Does NOT create a new EXP_NNN entry.
    Rerun the same hypothesis after fix.
    If same code error appears 3 times: log "SKIP: unresolvable",
    move to next hypothesis.

  NEVER refactor, rename, or restructure train.py.
  NEVER modify files outside train.py.
  NEVER enable use_block4.
  ALWAYS keep use_block6 = True when any other block is True.

STEP 3 — RUN
  Execute: python train.py
  Save all stdout and stderr to experiments/exp_NNN.log
  Time the run.
  On CRASH: check log, apply TYPE B fix if code error,
            revert CONFIG if OOM or NaN.

STEP 4 — EVALUATE
  Extract val_loss using extract_pattern from program.md.
  Compute improvement_pct = (baseline - val_loss) / baseline × 100
  Compare result to prediction in lab_notebook.md entry.
  If result contradicts VALIDATION predictions in program.md:
    Set anomaly_score = 0.8, flag "ANOMALY" in lab_notebook.md.
    This is a potential novel finding — note it explicitly.

STEP 5 — COMMIT OR DISCARD
  If val_loss < best_so_far:
    git add train.py
    git commit -m "exp_NNN: +X.X% val_loss | [one line summary]"
    Update best_so_far.
    Reset consecutive_no_improvement = 0.
  Else:
    git checkout HEAD -- train.py
    Increment consecutive_no_improvement.
  Always append to results.tsv regardless of outcome.

STEP 6 — CHECK STOPPING
  Stop if ANY of these are true:
    - val_loss <= TARGET (5% improvement confirmed)
    - consecutive_no_improvement >= 15
    - gpu_cost >= $8.00
    - api_cost >= $3.00
    - all 13 HYPOTHESIS_SPACE experiments attempted
  Never stop before 5 completed experiments.
  Every 10 experiments: write intermediate morning_report.md,
  then continue without stopping.

### When Loop Stops

Write two files then stop completely:

morning_report.md:
  - One line: best result or "no improvement found"
  - Stop reason
  - Best experiment: ID, val_loss, delta%, hypothesis
  - Anomalies flagged (if any)
  - Stats: total / improved / failed / skipped experiments
  - GPU and API cost used
  - Top 3 recommended next steps

draft_paper.md:
  - Title: most interesting finding, or "Ablation Study of
    Neuroscience-Inspired Blocks in SmolLM3 3B"
  - Abstract: 150 words, what we did, found, why it matters
  - Method: which blocks, why (cite hypotheses from lab_notebook.md)
  - Results: table of top 10 experiments from results.tsv
  - Analysis: what worked, what failed, anomalies
  - If no improvement: frame as ablation study (still publishable)
  - Every claim must cite an experiment ID or a paper summary

### Stability Protocol (from existing CLAUDE.md rules)
If train.py produces NaN loss after a block is enabled:
  1. Verify use_block6 is True in CONFIG
  2. Reduce lambda weight for that block by 10x in train.py
  3. Confirm max_grad_norm = 1.0 in CONFIG
  4. Reduce learning_rate by 2x
  5. If still NaN after 2 retries: disable the block,
     log as negative result, continue to next hypothesis

### Context Management
Always keep in context:
  program.md, current train.py, last 5 rows of results.tsv,
  current lab_notebook.md entry, top 3 paper summaries (300 tokens each)

Load on demand then release:
  Full paper text, old experiment logs, full results.tsv

When context exceeds 150K tokens:
  Summarize lab_notebook.md entries older than last 10
  into notebook_archive.md.
  Reduce all paper summaries to 100 tokens.
  Keep only last 3 full experiment logs.
  Log: "CONTEXT COMPACTED at EXP_NNN"
