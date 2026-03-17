#!/usr/bin/env python3
"""Generate all paper figures from screening data."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.size"] = 10

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "phase1_screening_results.tsv"
FIG_DIR = ROOT / "figures"


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
#  Architecture overview — publication-quality vertical pipeline
# ──────────────────────────────────────────────────────────────────

# Refined colour palette (softer, more harmonious)
_SCREENED = "#dcedc8"      # sage green — screened blocks
_DEFERRED = "#ffe0cc"      # peach — deferred blocks
_CORE = "#e3ecf7"          # light steel blue — base transformer
_NEUTRAL = "#f2f3f5"       # near-white grey — input/output
_WHITE = "#ffffff"
_STACK_BG = "#f0f4fa"      # very faint blue for stack interior

_SCREENED_EDGE = "#558b2f"
_DEFERRED_EDGE = "#d84315"
_CORE_EDGE = "#5c7cba"
_NEUTRAL_EDGE = "#9e9e9e"

_ARROW_COLOR = "#546e7a"   # blue-grey for main flow arrows
_TEXT_PRIMARY = "#263238"   # near-black
_TEXT_SECONDARY = "#607d8b" # muted blue-grey for annotations


def _box(
    ax: plt.Axes,
    cx: float, cy: float,
    w: float, h: float,
    label: str,
    face: str,
    edge: str = "#333333",
    fontsize: float = 9,
    fontweight: str = "medium",
    style: str = "round,pad=0.10",
    alpha: float = 1.0,
    linestyle: str = "-",
    linewidth: float = 0.9,
    text_color: str = _TEXT_PRIMARY,
) -> None:
    """Draw a rounded box centred at (cx, cy)."""
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=style,
        linewidth=linewidth,
        edgecolor=edge,
        facecolor=face,
        alpha=alpha,
        linestyle=linestyle,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight,
            color=text_color, zorder=3, linespacing=1.3)


def _arrow(
    ax: plt.Axes,
    x0: float, y0: float,
    x1: float, y1: float,
    color: str = _ARROW_COLOR,
    lw: float = 1.0,
    style: str = "-|>",
    linestyle: str = "-",
) -> None:
    p = FancyArrowPatch(
        (x0, y0), (x1, y1),
        arrowstyle=style, mutation_scale=10,
        linewidth=lw, color=color, zorder=1,
        linestyle=linestyle,
    )
    ax.add_patch(p)


def _side_connector(
    ax: plt.Axes,
    x_from: float, y_from: float,
    x_to: float, y_to: float,
    color: str,
    lw: float = 0.8,
    linestyle: str = "-",
) -> None:
    """Draw an L-shaped connector (horizontal then vertical)."""
    mid_x = (x_from + x_to) / 2
    ax.plot([x_from, mid_x, mid_x, x_to], [y_from, y_from, y_to, y_to],
            color=color, linewidth=lw, linestyle=linestyle, zorder=1,
            solid_capstyle="round", solid_joinstyle="round")
    # Arrowhead
    _arrow(ax, mid_x - 0.01, y_to, x_to, y_to, color=color, lw=lw)


def make_architecture_figure() -> None:
    fig, ax = plt.subplots(figsize=(8.5, 9))
    fig.patch.set_facecolor("white")
    ax.set_xlim(-0.5, 12)
    ax.set_ylim(1.5, 11.8)
    ax.axis("off")
    ax.set_aspect("equal")

    # ── Column centres ──
    C = 4.2        # main pipeline centre
    L = 1.0        # left column (memory write)
    R = 8.2        # right column (critics, B4, B6)

    # ── Vertical positions (top to bottom, evenly spaced) ──
    y_input = 10.8
    y_embed = 9.6
    y_b1 = 8.4
    y_stack_top = 7.2
    y_stack_bot = 4.8
    y_stack_mid = (y_stack_top + y_stack_bot) / 2
    y_b2r = 3.6
    y_lmhead = 2.4

    bw = 2.8     # main pipeline box width
    bh = 0.65    # main pipeline box height
    sw = 2.4     # side box width
    sh = 0.60    # side box height
    gap = 0.12   # arrow gap from box edge

    # ── Layer positions inside stack ──
    ly_top = y_stack_top - 0.58
    ly_bot = y_stack_bot + 0.42
    ly_mid = (ly_top + ly_bot) / 2
    inner_w = 2.0
    inner_h = 0.42

    # B2 write position (left, aligned with stack top)
    y_b2w = ly_top + 0.1

    # ═══════════════════════════════════════
    #  Main vertical pipeline
    # ═══════════════════════════════════════

    _box(ax, C, y_input, 2.0, 0.5, "Input tokens", _NEUTRAL, _NEUTRAL_EDGE,
         fontsize=8.5, fontweight="normal")
    _arrow(ax, C, y_input - 0.25 - gap, C, y_embed + bh / 2 + gap)

    _box(ax, C, y_embed, bw, bh, "Token Embeddings", _CORE, _CORE_EDGE,
         fontsize=9, fontweight="medium")
    _arrow(ax, C, y_embed - bh / 2 - gap, C, y_b1 + bh / 2 + gap)

    _box(ax, C, y_b1, bw, bh, "B1  SurpriseGate", _SCREENED, _SCREENED_EDGE,
         fontsize=9, fontweight="bold")
    _arrow(ax, C, y_b1 - bh / 2 - gap, C, y_stack_top + gap)

    # ── Transformer stack (outer container) ──
    stack_h = y_stack_top - y_stack_bot
    stack_patch = FancyBboxPatch(
        (C - bw / 2, y_stack_bot), bw, stack_h,
        boxstyle="round,pad=0.14",
        linewidth=0.8,
        edgecolor=_CORE_EDGE,
        facecolor=_STACK_BG,
        zorder=1,
    )
    ax.add_patch(stack_patch)

    # Stack label — placed inside the container, below the top edge
    ax.text(C, y_stack_top - 0.08, "Transformer Stack (32 layers)",
            ha="center", va="top",
            fontsize=10, color="#37474f", fontstyle="normal",
            fontweight="bold",
            bbox=dict(facecolor=_STACK_BG, edgecolor="none", pad=1.5))

    # Internal layer boxes
    _box(ax, C, ly_top, inner_w, inner_h, "Layer N", _WHITE, "#bdbdbd",
         fontsize=8, fontweight="normal", linewidth=0.6)
    ax.text(C, ly_mid, "\u22ee", ha="center", va="center",
            fontsize=13, color="#9e9e9e")
    _box(ax, C, ly_bot, inner_w, inner_h, "Layer 1", _WHITE, "#bdbdbd",
         fontsize=8, fontweight="normal", linewidth=0.6)

    _arrow(ax, C, y_stack_bot - gap, C, y_b2r + bh / 2 + gap)

    _box(ax, C, y_b2r, bw, bh, "B2  EpisodicMemory (read)", _SCREENED, _SCREENED_EDGE,
         fontsize=9, fontweight="bold")
    _arrow(ax, C, y_b2r - bh / 2 - gap, C, y_lmhead + 0.25 + gap)

    _box(ax, C, y_lmhead, 2.0, 0.5, "LM Head", _CORE, _CORE_EDGE,
         fontsize=8.5, fontweight="medium")

    # ═══════════════════════════════════════
    #  Left side: B2 write (memory bank)
    # ═══════════════════════════════════════

    _box(ax, L, y_b2w, sw, sh * 1.15,
         "B2  EpisodicMemory\n(write)", _SCREENED, _SCREENED_EDGE,
         fontsize=8, fontweight="bold")

    # L-shaped connector from main pipeline to B2 write
    branch_y = (y_b1 + y_stack_top) / 2
    ax.annotate("", xy=(L + sw / 2 + 0.05, y_b2w),
                xytext=(C - bw / 2 - 0.05, branch_y),
                arrowprops=dict(arrowstyle="-|>", color=_SCREENED_EDGE,
                                lw=0.8, mutation_scale=9,
                                connectionstyle="arc3,rad=0.15"))

    ax.text(L, y_b2w - sh * 0.6 - 0.22, "64-slot memory bank",
            ha="center", va="top", fontsize=7, color=_TEXT_SECONDARY,
            fontstyle="italic")

    # ═══════════════════════════════════════
    #  Right side: B3, B4, B6 (attached to stack layers)
    # ═══════════════════════════════════════

    # B3 — PerLayerCritic (at top of stack)
    y_b3 = ly_top
    _box(ax, R, y_b3, sw, sh, "B3  PerLayerCritic", _SCREENED, _SCREENED_EDGE,
         fontsize=8, fontweight="bold")
    _arrow(ax, C + bw / 2 + gap, y_b3, R - sw / 2 - gap, y_b3,
           color=_SCREENED_EDGE, lw=0.8)
    ax.text(R + sw / 2 + 0.12, y_b3, "every 4th layer",
            ha="left", va="center", fontsize=6.5, color=_TEXT_SECONDARY,
            fontstyle="italic")

    # B4 — PredictiveCoding (deferred, middle of stack)
    y_b4 = ly_mid
    _box(ax, R, y_b4, sw, sh, "B4  PredictiveCoding", _DEFERRED, _DEFERRED_EDGE,
         fontsize=8, fontweight="bold", linestyle=(0, (4, 2.5)), alpha=0.75)
    _arrow(ax, C + bw / 2 + gap, y_b4, R - sw / 2 - gap, y_b4,
           color=_DEFERRED_EDGE, lw=0.7, linestyle="--")
    ax.text(R + sw / 2 + 0.12, y_b4, "Phase 2",
            ha="left", va="center", fontsize=6.5, color=_DEFERRED_EDGE,
            fontstyle="italic")

    # B6 — HomeostaticNorm (bottom of stack)
    y_b6 = ly_bot
    _box(ax, R, y_b6, sw, sh, "B6  HomeostaticNorm", _SCREENED, _SCREENED_EDGE,
         fontsize=8, fontweight="bold")
    _arrow(ax, C + bw / 2 + gap, y_b6, R - sw / 2 - gap, y_b6,
           color=_SCREENED_EDGE, lw=0.8)
    ax.text(R + sw / 2 + 0.12, y_b6, "wraps all norms",
            ha="left", va="center", fontsize=6.5, color=_TEXT_SECONDARY,
            fontstyle="italic")

    # B5 — RLGatingPolicy (deferred, below stack on the right)
    y_b5 = y_b2r
    _box(ax, R, y_b5, sw, sh, "B5  RLGatingPolicy", _DEFERRED, _DEFERRED_EDGE,
         fontsize=8, fontweight="bold", linestyle=(0, (4, 2.5)), alpha=0.75)
    ax.text(R, y_b5 - sh / 2 - 0.12, "after stack (Phase 2)",
            ha="center", va="top", fontsize=6.5, color=_DEFERRED_EDGE,
            fontstyle="italic")

    # ═══════════════════════════════════════
    #  Legend (top, horizontal)
    # ═══════════════════════════════════════

    leg_y = 11.45
    leg_size = 0.22

    legend_items = [
        (_SCREENED, _SCREENED_EDGE, "-", "Phase 1 (active)"),
        (_DEFERRED, _DEFERRED_EDGE, "--", "Phase 2 (deferred)"),
        (_CORE, _CORE_EDGE, "-", "Base transformer"),
    ]

    total_w = sum(len(lbl) * 0.082 + 0.7 for _, _, _, lbl in legend_items)
    start_x = C - total_w / 2

    x_cursor = start_x
    for face, edge, ls, label in legend_items:
        patch = FancyBboxPatch(
            (x_cursor, leg_y - leg_size / 2), leg_size, leg_size,
            boxstyle="round,pad=0.02",
            linewidth=0.7, edgecolor=edge, facecolor=face,
            linestyle=ls, zorder=2,
        )
        ax.add_patch(patch)
        ax.text(x_cursor + leg_size + 0.1, leg_y, label,
                va="center", fontsize=7.5, color=_TEXT_PRIMARY, zorder=3)
        x_cursor += len(label) * 0.082 + 0.7

    fig.tight_layout(pad=0.3)
    save(fig, "architecture_overview")


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
