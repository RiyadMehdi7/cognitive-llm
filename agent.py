"""
agent.py — Research agent for Cognitive LLM ablation experiments.

Reads program.md and results.tsv, then orchestrates experiments
by modifying train.py CONFIG and running `python train.py`.

Usage:
    python agent.py            # print status and exit (safe to call anytime)
    python agent.py --run      # start the full research loop
"""

from __future__ import annotations

import ast
import csv
import re
import subprocess
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent
TRAIN_PY = ROOT / "train.py"
PROGRAM_MD = ROOT / "program.md"
RESULTS_TSV = ROOT / "results.tsv"
LAB_NOTEBOOK = ROOT / "lab_notebook.md"
EXPERIMENTS_DIR = ROOT / "experiments"

# ── Experiment definitions matching program.md HYPOTHESIS_SPACE ──────────────
HYPOTHESIS_SPACE = [
    # (exp_id, label, config_overrides)
    ("EXP_000", "Baseline: LoRA only",
     {"use_block1": False, "use_block2": False, "use_block3": False,
      "use_block4": False, "use_block5": False, "use_block6": False}),

    ("EXP_001", "B6 only: HomeostaticNorm",
     {"use_block1": False, "use_block2": False, "use_block3": False,
      "use_block4": False, "use_block5": False, "use_block6": True}),

    ("EXP_002", "B1+B6: SurpriseGate + HomeostaticNorm",
     {"use_block1": True,  "use_block2": False, "use_block3": False,
      "use_block4": False, "use_block5": False, "use_block6": True}),

    ("EXP_003", "B2+B6: EpisodicMemory + HomeostaticNorm",
     {"use_block1": False, "use_block2": True,  "use_block3": False,
      "use_block4": False, "use_block5": False, "use_block6": True}),

    ("EXP_004", "B3+B6: PerLayerCritic + HomeostaticNorm",
     {"use_block1": False, "use_block2": False, "use_block3": True,
      "use_block4": False, "use_block5": False, "use_block6": True}),

    ("EXP_005", "B1+B2+B6",
     {"use_block1": True,  "use_block2": True,  "use_block3": False,
      "use_block4": False, "use_block5": False, "use_block6": True}),

    ("EXP_006", "B1+B3+B6",
     {"use_block1": True,  "use_block2": False, "use_block3": True,
      "use_block4": False, "use_block5": False, "use_block6": True}),

    ("EXP_007", "B2+B3+B6",
     {"use_block1": False, "use_block2": True,  "use_block3": True,
      "use_block4": False, "use_block5": False, "use_block6": True}),

    ("EXP_008", "B1+B2+B3+B6",
     {"use_block1": True,  "use_block2": True,  "use_block3": True,
      "use_block4": False, "use_block5": False, "use_block6": True}),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def read_results() -> list[dict]:
    """Read all completed experiment rows from results.tsv."""
    if not RESULTS_TSV.exists():
        return []
    with open(RESULTS_TSV, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def completed_exp_ids() -> set[str]:
    return {r["exp_id"] for r in read_results()}


def append_result(row: dict) -> None:
    """Append a result row to results.tsv."""
    fieldnames = [
        "exp_id", "timestamp", "hypothesis", "val_loss",
        "baseline_delta_pct", "is_improvement", "anomaly_score",
        "verdict", "run_minutes", "notes",
    ]
    with open(RESULTS_TSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writerow(row)


def patch_config(overrides: dict) -> str:
    """Patch the CONFIG dict in train.py; return original source for reverting."""
    source = TRAIN_PY.read_text()
    original = source

    for key, value in overrides.items():
        # Match the key inside the CONFIG dict (handles True/False/numbers)
        pattern = rf'("{key}"\s*:\s*)([^\n,]+)'
        replacement = rf'\g<1>{repr(value)}'
        source = re.sub(pattern, replacement, source)

    TRAIN_PY.write_text(source)
    return original


def revert_train(original_source: str) -> None:
    TRAIN_PY.write_text(original_source)


def extract_val_loss(log: str) -> float | None:
    m = re.search(r"val_loss:\s*([0-9]+\.[0-9]+)", log)
    return float(m.group(1)) if m else None


def run_train(exp_id: str) -> tuple[str, float]:
    """Run train.py, save log, return (log_text, elapsed_minutes)."""
    EXPERIMENTS_DIR.mkdir(exist_ok=True)
    log_path = EXPERIMENTS_DIR / f"{exp_id.lower()}.log"

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(TRAIN_PY)],
        capture_output=True, text=True, cwd=ROOT,
    )
    elapsed = (time.time() - t0) / 60.0

    combined = result.stdout + result.stderr
    log_path.write_text(combined)
    return combined, elapsed


def log_hypothesis(exp_id: str, label: str, overrides: dict, motivation: str = "") -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = textwrap.dedent(f"""
    ## {exp_id} | {ts}
    Hypothesis: {label}
    Motivation: {motivation or 'Next in HYPOTHESIS_SPACE priority order'}
    Prediction: val_loss will decrease if block combination is beneficial
    Change: {overrides}

    """)
    with open(LAB_NOTEBOOK, "a") as f:
        f.write(entry)


def log_result_in_notebook(exp_id: str, val_loss: float | None, verdict: str, notes: str = "") -> None:
    with open(LAB_NOTEBOOK, "a") as f:
        f.write(f"Result: val_loss={val_loss}  verdict={verdict}  {notes}\n\n")


# ── Status ────────────────────────────────────────────────────────────────────

def print_status() -> None:
    done = completed_exp_ids()
    total = len(HYPOTHESIS_SPACE)
    print(f"agent.py — Cognitive LLM Research Agent")
    print(f"program.md   : {'found' if PROGRAM_MD.exists() else 'MISSING'}")
    print(f"results.tsv  : {len(done)} / {total} experiments completed")
    print(f"train.py     : {'found' if TRAIN_PY.exists() else 'MISSING'}")
    print(f"")
    if done:
        rows = read_results()
        baseline_row = next((r for r in rows if r["exp_id"] == "EXP_000"), None)
        if baseline_row:
            print(f"Baseline val_loss : {baseline_row['val_loss']}")
        best = min((r for r in rows if r["val_loss"]), key=lambda r: float(r["val_loss"]), default=None)
        if best:
            print(f"Best so far       : {best['exp_id']} val_loss={best['val_loss']} ({best['hypothesis']})")
    else:
        print("No experiments run yet. Use --run to start the research loop.")
    print("")
    print("To start a full research session: python agent.py --run")


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_loop() -> None:
    print("=== RESEARCH SESSION STARTING ===")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    done = completed_exp_ids()
    baseline_val_loss: float | None = None

    # Check if baseline already exists
    existing = read_results()
    baseline_row = next((r for r in existing if r["exp_id"] == "EXP_000"), None)
    if baseline_row and baseline_row["val_loss"]:
        try:
            baseline_val_loss = float(baseline_row["val_loss"])
        except ValueError:
            pass

    best_val_loss = min(
        (float(r["val_loss"]) for r in existing if r["val_loss"]),
        default=float("inf"),
    )
    consecutive_no_improvement = 0

    for exp_id, label, overrides in HYPOTHESIS_SPACE:
        if exp_id in done:
            print(f"Skipping {exp_id} (already done)")
            continue

        print(f"\n{'='*60}")
        print(f"  {exp_id}: {label}")
        print(f"  Config: {overrides}")
        print(f"{'='*60}")

        # Patch train.py
        original_source = patch_config(overrides)
        log_hypothesis(exp_id, label, overrides)

        # Run
        log_text, elapsed = run_train(exp_id)
        val_loss = extract_val_loss(log_text)

        crashed = "CRASH:" in log_text
        verdict = "crash" if crashed else ("ok" if val_loss is not None else "unknown")

        # Determine baseline
        if exp_id == "EXP_000" and val_loss is not None:
            baseline_val_loss = val_loss
            best_val_loss = val_loss
            print(f"Baseline established: val_loss={val_loss:.6f}")

        # Compute improvement
        delta_pct = 0.0
        is_improvement = False
        if val_loss is not None and baseline_val_loss is not None and baseline_val_loss > 0:
            delta_pct = (baseline_val_loss - val_loss) / baseline_val_loss * 100
            is_improvement = val_loss < best_val_loss

        notes = ""
        if crashed:
            crash_line = next((l for l in log_text.splitlines() if "CRASH:" in l), "")
            notes = crash_line

        # Record result
        append_result({
            "exp_id": exp_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "hypothesis": label,
            "val_loss": f"{val_loss:.6f}" if val_loss is not None else "crash",
            "baseline_delta_pct": f"{delta_pct:.2f}",
            "is_improvement": str(is_improvement),
            "anomaly_score": "0.0",
            "verdict": verdict,
            "run_minutes": f"{elapsed:.1f}",
            "notes": notes,
        })

        log_result_in_notebook(exp_id, val_loss, verdict, notes)

        if is_improvement and val_loss is not None:
            best_val_loss = val_loss
            consecutive_no_improvement = 0
            print(f"  NEW BEST: val_loss={val_loss:.6f} (+{delta_pct:.1f}%)")
        else:
            consecutive_no_improvement += 1
            if val_loss is not None:
                print(f"  val_loss={val_loss:.6f} (delta={delta_pct:+.1f}%)")
            else:
                print(f"  CRASHED — see experiments/{exp_id.lower()}.log")
            # Revert if not an improvement
            revert_train(original_source)

        # Stopping check
        if (baseline_val_loss is not None and val_loss is not None
                and val_loss <= baseline_val_loss * 0.95):
            print(f"\nSTOPPING: 5% improvement target reached at {exp_id}!")
            break

        if consecutive_no_improvement >= 15:
            print(f"\nSTOPPING: {consecutive_no_improvement} consecutive no-improvement runs.")
            break

    print("\n=== RESEARCH SESSION COMPLETE ===")
    print_status()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--run" in sys.argv:
        run_loop()
    else:
        print_status()
