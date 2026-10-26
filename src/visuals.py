"""Export low-friction research charts for each held-out horizon."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def save_charts(folder: str | Path, stem: str, y, probs, trades):
    folder = Path(folder)
    pred = np.array([-1, 0, 1])[np.argmax(probs, axis=1)]
    cm = confusion_matrix(y, pred, labels=[-1, 0, 1])
    fig, ax = plt.subplots(figsize=(5.2, 4.3))
    ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=11)
    ax.set(xticks=range(3), xticklabels=["Down", "Flat", "Up"],
           yticks=range(3), yticklabels=["Down", "Flat", "Up"],
           xlabel="Predicted movement", ylabel="Actual movement", title=f"{stem} · Test confusion matrix")
    fig.tight_layout()
    fig.savefig(folder / f"{stem}_confusion.png", dpi=160)
    plt.close(fig)
    if not trades.empty:
        equity = np.r_[0, trades["net_pnl_usd"].cumsum().to_numpy()]
        fig, ax = plt.subplots(figsize=(7.0, 3.6))
        ax.plot(np.arange(len(equity)), equity, lw=1.6)
        ax.axhline(0, ls="--", lw=0.8)
        ax.set(xlabel="Completed trades", ylabel="Cumulative net P&L (USD)",
               title=f"{stem} · Aggressive execution proxy")
        fig.tight_layout()
        fig.savefig(folder / f"{stem}_equity.png", dpi=160)
        plt.close(fig)


def save_summary(folder, results):
    rows = ["# LOB Alpha Research — Experiment Summary", "",
            f"Data mode: **{results['mode']}**; model: **{results['model']}**; feature set: **{results['feature_set']}**.", "",
            "**Important:** Outputs from synthetic data demonstrate software behavior, NOT market predictability.",
            "A single trading session does not provide independent day-level out-of-sample evidence.",
            "The execution simulation assumes immediate full fills at displayed quotes and excludes market impact.", "",
            "| Horizon (events) | Test accuracy | Test macro-F1 | Test log loss | Spearman IC | Trades | Net P&L (USD) |",
            "|---:|---:|---:|---:|---:|---:|---:|"]
    for h, exp in results["experiments"].items():
        m = exp["test"]
        p = exp["aggressive_execution"]
        ic = exp["test_spearman_ic"]
        rows.append(f"| {h} | {m['accuracy']:.3f} | {m['macro_f1']:.3f} | {m['log_loss']:.3f} | "
                    f"{ic:.3f} | {p['trades']} | {p['net_pnl_usd']:.2f} |" if ic is not None else
                    f"| {h} | {m['accuracy']:.3f} | {m['macro_f1']:.3f} | {m['log_loss']:.3f} | N/A | {p['trades']} | {p['net_pnl_usd']:.2f} |")
    rows += ["", "## Audit", "", "See `results.json` for snapshot validation and per-horizon confusion matrices; the trade log contains event indices and actual entry/exit quote prices."]
    (Path(folder) / "summary.md").write_text("\n".join(rows) + "\n")
