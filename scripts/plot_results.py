"""Draw results/evaluation.png from results/evaluation_summary.json (one panel per game and seat)."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import ROOT  # noqa: E402

PANELS = [
    ("bargaining/player 1", "Bargaining, proposing first", "discounted share of the pot"),
    ("bargaining/player 2", "Bargaining, responding first", "discounted share of the pot"),
    ("negotiation/player 1", "Negotiation, as seller", "share of the available surplus"),
    ("negotiation/player 2", "Negotiation, as buyer", "share of the available surplus"),
    ("persuasion/seller", "Persuasion, as seller", "sales per round"),
    ("persuasion/buyer", "Persuasion, as buyer", "gain per round (x price)"),
]
LABELS = {"best_response": "This agent", "proxy": "LLM (simulated)", "LLM (real games)": "LLM (real games)",
          "spne": "Equilibrium (SPNE)", "fair": "Always 50/50", "split": "Split the difference",
          "commitment": "Optimal signal (K&G)", "honest": "Always honest", "always": "Always recommend",
          "naive": "Trust every recommendation"}
ACCENT, NEUTRAL, INK, MUTED, GRID = "#2a78d6", "#8a8984", "#0b0b0b", "#52514e", "#e6e5e1"


def main() -> None:
    summary = json.loads((ROOT / "results" / "evaluation_summary.json").read_text())
    fig, axes = plt.subplots(3, 2, figsize=(11, 9.5), facecolor="#fcfcfb")
    for ax, (key, title, unit) in zip(axes.ravel(), PANELS):
        stats = sorted(summary[key].items(), key=lambda kv: kv[1]["mean"])
        names = [LABELS.get(k, k) for k, _ in stats]
        for i, (k, s) in enumerate(stats):
            color = ACCENT if k == "best_response" else NEUTRAL
            ax.errorbar(s["mean"], i, xerr=s["ci95"], fmt="o", ms=7, color=color, ecolor=color, elinewidth=2,
                        capsize=0)
            ax.text(s["mean"] + s["ci95"], i, f"  {s['mean']:.2f}", va="center", fontsize=9, color=INK)
        ax.set_yticks(range(len(names)), names, fontsize=9.5, color=INK)
        for lbl, (k, _) in zip(ax.get_yticklabels(), stats):
            lbl.set_fontweight("bold" if k == "best_response" else "normal")
        ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
        ax.set_xlabel(unit, fontsize=9, color=MUTED)
        ax.set_facecolor("#fcfcfb")
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="x", colors=MUTED, labelsize=8.5)
        ax.tick_params(axis="y", length=0)
        lo = min(s["mean"] - s["ci95"] for _, s in stats)
        hi = max(s["mean"] + s["ci95"] for _, s in stats)
        pad = (hi - lo) * 0.25 + 1e-3
        ax.set_xlim(min(0, lo - pad * 0.3) if lo > 0 else lo - pad * 0.3, hi + pad)
    fig.suptitle("Payoff against simulated LLM opponents on held-out GLEE games (mean, 95% CI)",
                 x=0.01, ha="left", fontsize=12.5, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = ROOT / "results" / "evaluation.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
