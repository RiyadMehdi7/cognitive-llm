"""
agent.py - Research agent for Cognitive LLM ablation experiments.

Reads program.md and results.tsv, then orchestrates experiments
by modifying train.py CONFIG and running `python train.py`.

Usage:
    python agent.py            # print status and exit (safe to call anytime)
    python agent.py --run      # start the full research loop
"""

from __future__ import annotations

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

# -- Experiment definitions (priority-ordered for Phase 1) --------------------
# EXP_V00: Vanilla pretrained (no LoRA, no blocks) — true zero-shot baseline
# EXP_000: LoRA-only baseline
# Then single blocks, best-bet combos, full stack
HYPOTHESIS_SPACE = [
    # (exp_id, label, config_overrides)
    ("EXP_V00", "Vanilla: pretrained SmolLM3 (no LoRA, no blocks)",
     {"use_block1": False, "use_block2": False, "use_block3": False,
      "use_block4": False, "use_block5": False, "use_block6": False,
      "skip_lora": True}),

    ("EXP_000", "Baseline: LoRA only",
     {"use_block1": False, "use_block2": False, "use_block3": False,
      "use_block4": False, "use_block5": False, "use_block6": False}),

    ("EXP_001", "B6 only: HomeostaticNorm",
     {"use_block1": False, "use_block2": False, "use_block3": False,
      "use_block4": False, "use_block5": False, "use_block6": True}),

    ("EXP_002", "B2+B6: EpisodicMemory + HomeostaticNorm",
     {"use_block1": False, "use_block2": True,  "use_block3": False,
      "use_block4": False, "use_block5": False, "use_block6": True}),

    ("EXP_003", "B1+B6: SurpriseGate + HomeostaticNorm",
     {"use_block1": True,  "use_block2": False, "use_block3": False,
      "use_block4": False, "use_block5": False, "use_block6": True}),

    ("EXP_004", "B3+B6: PerLayerCritic (TD) + HomeostaticNorm",
     {"use_block1": False, "use_block2": False, "use_block3": True,
      "use_block4": False, "use_block5": False, "use_block6": True}),

    ("EXP_005", "B1+B2+B6",
     {"use_block1": True,  "use_block2": True,  "use_block3": False,
      "use_block4": False, "use_block5": False, "use_block6": True}),

    ("EXP_006", "B2+B3+B6",
     {"use_block1": False, "use_block2": True,  "use_block3": True,
      "use_block4": False, "use_block5": False, "use_block6": True}),

    ("EXP_007", "B1+B2+B3+B6",
     {"use_block1": True,  "use_block2": True,  "use_block3": True,
      "use_block4": False, "use_block5": False, "use_block6": True}),

    ("EXP_008", "B1+B3+B6",
     {"use_block1": True,  "use_block2": False, "use_block3": True,
      "use_block4": False, "use_block5": False, "use_block6": True}),
]

TSV_FIELDS = [
    "exp_id", "timestamp", "hypothesis", "val_loss", "gsm8k_acc",
    "baseline_delta_pct", "is_improvement", "anomaly_score",
    "verdict", "run_minutes", "notes",
]


# -- Helpers ------------------------------------------------------------------

def read_results() -> list[dict]:
    """Read all completed experiment rows from results.tsv."""
    if not RESULTS_TSV.exists():
        return []
    with open(RESULTS_TSV, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def completed_exp_ids() -> set[str]:
    return {r["exp_id"] for r in read_results()}


def _parse_float(value: str) -> float | None:
    """Parse a float from a TSV cell; return None for 'crash' / empty."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def append_result(row: dict) -> None:
    """Append a result row to results.tsv."""
    with open(RESULTS_TSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TSV_FIELDS, delimiter="\t")
        writer.writerow(row)


def patch_config(overrides: dict) -> str:
    """Patch CONFIG values in train.py; return original source for reverting.

    Uses a tight regex that stops at the first comma, closing brace, or
    end-of-line — so trailing inline comments are preserved correctly.
    """
    source = TRAIN_PY.read_text()
    original = source

    for key, value in overrides.items():
        # Match: "key": <value>  where value ends before ,  }  or newline
        pattern = rf'("{re.escape(key)}"\s*:\s*)([^,}}\n]+)'
        replacement = rf'\g<1>{repr(value)}'
        source = re.sub(pattern, replacement, source)

    TRAIN_PY.write_text(source)
    return original


def revert_train(original_source: str) -> None:
    TRAIN_PY.write_text(original_source)


def extract_metric(log: str, name: str) -> float | None:
    """Extract a printed metric line like 'val_loss: 1.234567' from log."""
    m = re.search(rf"^{re.escape(name)}:\s*([0-9]+\.[0-9]+)", log, re.MULTILINE)
    return float(m.group(1)) if m else None


def run_train(exp_id: str) -> tuple[str, float]:
    """Run train.py, save log, return (combined_log, elapsed_minutes)."""
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


def run_baseline_3x() -> float | None:
    """Run EXP_000 three times and return median val_loss."""
    print("  Running baseline 3x for stable measurement...")
    vals = []
    for i in range(3):
        log, _ = run_train(f"EXP_000_r{i}")
        v = extract_metric(log, "val_loss")
        if v is not None:
            vals.append(v)
            print(f"    run {i+1}/3: val_loss={v:.6f}")
        else:
            print(f"    run {i+1}/3: CRASHED")
    if not vals:
        return None
    vals.sort()
    return vals[len(vals) // 2]


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


def log_result_in_notebook(
    exp_id: str,
    val_loss: float | None,
    gsm8k_acc: float | None,
    verdict: str,
    notes: str = "",
) -> None:
    with open(LAB_NOTEBOOK, "a") as f:
        f.write(
            f"Result: val_loss={val_loss}  gsm8k_acc={gsm8k_acc}%  "
            f"verdict={verdict}  {notes}\n\n"
        )


# -- Status -------------------------------------------------------------------

def print_status() -> None:
    done = completed_exp_ids()
    total = len(HYPOTHESIS_SPACE)
    print("agent.py - Cognitive LLM Research Agent")
    print(f"program.md   : {'found' if PROGRAM_MD.exists() else 'MISSING'}")
    print(f"results.tsv  : {len(done)} / {total} experiments completed")
    print(f"train.py     : {'found' if TRAIN_PY.exists() else 'MISSING'}")
    print("")
    rows = read_results()
    if rows:
        baseline_row = next((r for r in rows if r["exp_id"] == "EXP_000"), None)
        if baseline_row:
            print(f"Baseline val_loss : {baseline_row['val_loss']}")
            print(f"Baseline gsm8k_acc: {baseline_row.get('gsm8k_acc', 'n/a')}%")
        valid = [r for r in rows if _parse_float(r.get("val_loss", "")) is not None]
        if valid:
            best = min(valid, key=lambda r: _parse_float(r["val_loss"]))
            print(f"Best so far       : {best['exp_id']} val_loss={best['val_loss']} ({best['hypothesis']})")
    else:
        print("No experiments run yet. Use --run to start the research loop.")
    print("")
    print("To start a full research session: python agent.py --run")


# -- Main loop ----------------------------------------------------------------

def run_loop() -> None:
    print("=== RESEARCH SESSION STARTING ===")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    done = completed_exp_ids()
    existing = read_results()

    # Recover baseline from prior session if available
    baseline_row = next((r for r in existing if r["exp_id"] == "EXP_000"), None)
    baseline_val_loss: float | None = _parse_float(
        baseline_row["val_loss"] if baseline_row else None
    )

    valid_losses = [_parse_float(r.get("val_loss", "")) for r in existing]
    valid_losses = [v for v in valid_losses if v is not None]
    best_val_loss = min(valid_losses) if valid_losses else float("inf")
    consecutive_no_improvement = 0

    for exp_id, label, overrides in HYPOTHESIS_SPACE:
        if exp_id in done:
            print(f"Skipping {exp_id} (already done)")
            continue

        print(f"\n{'='*60}")
        print(f"  {exp_id}: {label}")
        print(f"  Config: {overrides}")
        print(f"{'='*60}")

        # Vanilla baseline (eval-only, no training)
        if exp_id == "EXP_V00":
            original_source = patch_config(overrides)
            log_hypothesis(exp_id, label, overrides)
            log_text, elapsed = run_train(exp_id)
            val_loss = extract_metric(log_text, "val_loss")
            gsm8k_acc = extract_metric(log_text, "gsm8k_acc")
            crashed = "CRASH:" in log_text
            verdict = "crash" if crashed else "ok"
            notes = "vanilla pretrained, no LoRA, no training"
            if crashed:
                notes = next((l for l in log_text.splitlines() if "CRASH:" in l), "")
            append_result({
                "exp_id": exp_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "hypothesis": label,
                "val_loss": f"{val_loss:.6f}" if val_loss is not None else "crash",
                "gsm8k_acc": f"{gsm8k_acc:.2f}" if gsm8k_acc is not None else "n/a",
                "baseline_delta_pct": "0.00",
                "is_improvement": "False",
                "anomaly_score": "0.0",
                "verdict": verdict,
                "run_minutes": f"{elapsed:.1f}",
                "notes": notes,
            })
            log_result_in_notebook(exp_id, val_loss, gsm8k_acc, verdict, notes)
            if val_loss is not None:
                print(f"  Vanilla pretrained: val_loss={val_loss:.6f} gsm8k_acc={gsm8k_acc}%")
            revert_train(original_source)
            continue

        # LoRA baseline: run 3x for stable measurement
        if exp_id == "EXP_000":
            original_source = patch_config(overrides)
            log_hypothesis(exp_id, label, overrides)
            baseline_val_loss = run_baseline_3x()
            if baseline_val_loss is not None:
                best_val_loss = baseline_val_loss
                print(f"Baseline (median of 3): val_loss={baseline_val_loss:.6f}")
            # Also grab gsm8k_acc from one of the runs for the record
            single_log, single_elapsed = run_train(exp_id)
            val_loss = extract_metric(single_log, "val_loss") or baseline_val_loss
            gsm8k_acc = extract_metric(single_log, "gsm8k_acc")
            crashed = "CRASH:" in single_log
            verdict = "crash" if crashed else "ok"
            notes = ""
            if crashed:
                notes = next((l for l in single_log.splitlines() if "CRASH:" in l), "")
            append_result({
                "exp_id": exp_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "hypothesis": label,
                "val_loss": f"{baseline_val_loss:.6f}" if baseline_val_loss else "crash",
                "gsm8k_acc": f"{gsm8k_acc:.2f}" if gsm8k_acc is not None else "n/a",
                "baseline_delta_pct": "0.00",
                "is_improvement": "False",
                "anomaly_score": "0.0",
                "verdict": verdict,
                "run_minutes": f"{single_elapsed:.1f}",
                "notes": notes,
            })
            log_result_in_notebook(exp_id, baseline_val_loss, gsm8k_acc, verdict, notes)
            revert_train(original_source)
            continue

        # Patch train.py for this experiment
        original_source = patch_config(overrides)
        log_hypothesis(exp_id, label, overrides)

        log_text, elapsed = run_train(exp_id)
        val_loss = extract_metric(log_text, "val_loss")
        gsm8k_acc = extract_metric(log_text, "gsm8k_acc")

        crashed = "CRASH:" in log_text
        verdict = "crash" if crashed else ("ok" if val_loss is not None else "unknown")

        # Compute improvement vs baseline
        delta_pct = 0.0
        is_improvement = False
        if val_loss is not None and baseline_val_loss is not None and baseline_val_loss > 0:
            delta_pct = (baseline_val_loss - val_loss) / baseline_val_loss * 100
            is_improvement = val_loss < best_val_loss

        notes = ""
        if crashed:
            notes = next((l for l in log_text.splitlines() if "CRASH:" in l), "")

        append_result({
            "exp_id": exp_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "hypothesis": label,
            "val_loss": f"{val_loss:.6f}" if val_loss is not None else "crash",
            "gsm8k_acc": f"{gsm8k_acc:.2f}" if gsm8k_acc is not None else "n/a",
            "baseline_delta_pct": f"{delta_pct:.2f}",
            "is_improvement": str(is_improvement),
            "anomaly_score": "0.0",
            "verdict": verdict,
            "run_minutes": f"{elapsed:.1f}",
            "notes": notes,
        })

        log_result_in_notebook(exp_id, val_loss, gsm8k_acc, verdict, notes)

        if is_improvement and val_loss is not None:
            best_val_loss = val_loss
            consecutive_no_improvement = 0
            print(f"  NEW BEST: val_loss={val_loss:.6f} gsm8k_acc={gsm8k_acc}% (+{delta_pct:.1f}%)")
        else:
            consecutive_no_improvement += 1
            if val_loss is not None:
                print(f"  val_loss={val_loss:.6f} gsm8k_acc={gsm8k_acc}% (delta={delta_pct:+.1f}%)")
            else:
                print(f"  CRASHED - see experiments/{exp_id.lower()}.log")
            revert_train(original_source)

        # Stopping checks
        if (baseline_val_loss is not None and val_loss is not None
                and val_loss <= baseline_val_loss * 0.95):
            print(f"\nSTOPPING: 5% improvement target reached at {exp_id}!")
            break

        if consecutive_no_improvement >= 15:
            print(f"\nSTOPPING: {consecutive_no_improvement} consecutive no-improvement runs.")
            break

    print("\n=== RESEARCH SESSION COMPLETE ===")
    print_status()


# -- Entry point --------------------------------------------------------------

if __name__ == "__main__":
    if "--run" in sys.argv:
        run_loop()
    else:
        print_status()
