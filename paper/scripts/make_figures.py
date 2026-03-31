#!/usr/bin/env python3
"""Generate all paper figures from screening data."""
from __future__ import annotations

import csv
from pathlib import Path
import subprocess

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.size"] = 10

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
DATA_PATH = ROOT / "data" / "phase1_screening_results.tsv"
FIG_DIR = ROOT / "figures"
README_FIG_DIR = REPO_ROOT / "assets" / "readme"
ARCHITECTURE_SOURCE = FIG_DIR / "architecture_overview.d2"


def load_rows() -> list[dict[str, str]]:
    with DATA_PATH.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.4)
    ax.set_axisbelow(True)


def save(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────
#  Architecture overview — D2 source rendered to SVG/PNG/PDF
# ──────────────────────────────────────────────────────────────────


def make_architecture_figure() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    README_FIG_DIR.mkdir(parents=True, exist_ok=True)

    svg_path = FIG_DIR / "architecture_overview.svg"
    png_path = FIG_DIR / "architecture_overview.png"
    pdf_path = FIG_DIR / "architecture_overview.pdf"

    d2_cmd = [
        "d2",
        "--layout=elk",
        "--theme=0",
        "--pad=36",
        "--scale=1",
        "--salt=architecture-overview",
        "--omit-version",
        str(ARCHITECTURE_SOURCE),
        str(svg_path),
    ]

    try:
        subprocess.run(d2_cmd, check=True, cwd=REPO_ROOT)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "D2 CLI is required to render architecture_overview.d2. "
            "Install it from https://d2lang.com/."
        ) from exc

    try:
        subprocess.run(
            ["rsvg-convert", str(svg_path), "-o", str(png_path)],
            check=True,
            cwd=REPO_ROOT,
        )
        subprocess.run(
            ["rsvg-convert", "-f", "pdf", str(svg_path), "-o", str(pdf_path)],
            check=True,
            cwd=REPO_ROOT,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "rsvg-convert is required to turn the D2 SVG into PNG/PDF assets. "
            "Install librsvg so rsvg-convert is available on PATH."
        ) from exc

    (README_FIG_DIR / "architecture-overview.svg").write_bytes(svg_path.read_bytes())


# ──────────────────────────────────────────────────────────────────
#  Validation-loss ranking bar chart
# ──────────────────────────────────────────────────────────────────

def make_val_loss_figure(rows: list[dict[str, str]]) -> None:
    ordered = sorted(rows, key=lambda row: float(row["val_loss"]))
    names = [row["hypothesis"] for row in ordered]
    values = [float(row["val_loss"]) for row in ordered]

    colors = []
    for row in ordered:
        if row["exp_id"] == "EXP_000":
            colors.append("#9aa5b1")
        elif row["exp_id"] == "EXP_005":
            colors.append("#157f3b")
        elif row["is_improvement"] == "True":
            colors.append("#56a36c")
        else:
            colors.append("#d97a2b")

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    bars = ax.barh(range(len(names)), values, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    style_axes(ax)
    ax.set_xlabel("Validation loss (lower is better)")
    ax.set_title("Phase 1 screening: validation loss by configuration", fontsize=12)
    ax.invert_yaxis()

    for bar, value in zip(bars, values):
        ax.text(
            value + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            va="center",
            fontsize=8.5,
        )

    save(fig, "phase1_val_loss_ranking")


# ──────────────────────────────────────────────────────────────────
#  Delta vs baseline
# ──────────────────────────────────────────────────────────────────

def make_delta_figure(rows: list[dict[str, str]]) -> None:
    non_baseline = [row for row in rows if row["exp_id"] != "EXP_000"]
    ordered = sorted(non_baseline, key=lambda row: float(row["baseline_delta_pct"]), reverse=True)
    names = [row["hypothesis"] for row in ordered]
    values = [float(row["baseline_delta_pct"]) for row in ordered]

    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    bars = ax.bar(range(len(names)), values, color="#1f5aa6")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    style_axes(ax)
    ax.set_ylabel("Reduction vs. LoRA baseline (%)")
    ax.set_title("Improvement relative to the LoRA-only baseline", fontsize=12)
    ax.axhline(0.0, color="black", linewidth=1.0)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.7,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.5,
        )

    save(fig, "phase1_delta_vs_baseline")


def main() -> None:
    rows = load_rows()
    make_architecture_figure()
    make_val_loss_figure(rows)
    make_delta_figure(rows)
    print(f"Wrote figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
