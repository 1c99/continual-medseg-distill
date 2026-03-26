#!/usr/bin/env python3
"""Generate paper-ready figures from ablation or multi-task run outputs.

Usage:
    python scripts/plot_results.py outputs/ablations/ablation_<hash>/
    python scripts/plot_results.py outputs/multitask/ --output-dir figures/

Generates:
    - method_comparison.png  (bar chart of dice_mean per method)
    - forgetting_heatmap.png (task × task evaluation matrix)
    - per_task_dice.png      (per-task dice across training steps)

Gracefully skips if matplotlib is not installed.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _float(v: str | None) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def plot_method_comparison(run_dir: Path, out_dir: Path) -> Path | None:
    """Bar chart comparing dice_mean across methods."""
    agg_path = run_dir / "aggregate_metrics.csv"
    rows = _read_csv(agg_path)
    if not rows:
        return None

    methods = []
    dice_vals = []
    for r in rows:
        method = r.get("method", "")
        dice = _float(r.get("val_dice_mean"))
        if method and dice is not None:
            methods.append(method)
            dice_vals.append(dice)

    if not methods:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.Set2.colors
    bars = ax.bar(range(len(methods)), dice_vals, color=colors[:len(methods)])
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel("Dice Mean")
    ax.set_title("Method Comparison — Dice Mean")
    ax.set_ylim(0, max(dice_vals) * 1.15 if dice_vals else 1)
    for bar, val in zip(bars, dice_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out = out_dir / "method_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_forgetting_heatmap(run_dir: Path, out_dir: Path) -> Path | None:
    """Heatmap of task_eval_matrix (trained_on × evaluated_on)."""
    csv_path = run_dir / "task_eval_matrix.csv"
    rows = _read_csv(csv_path)
    if not rows:
        return None

    tasks_trained = []
    tasks_eval = []
    for r in rows:
        t = r.get("trained_on", "")
        e = r.get("evaluated_on", "")
        if t and t not in tasks_trained:
            tasks_trained.append(t)
        if e and e not in tasks_eval:
            tasks_eval.append(e)

    if not tasks_trained or not tasks_eval:
        return None

    import numpy as np
    matrix = np.full((len(tasks_trained), len(tasks_eval)), float("nan"))
    for r in rows:
        t = r.get("trained_on", "")
        e = r.get("evaluated_on", "")
        dice = _float(r.get("val_dice_mean"))
        if t in tasks_trained and e in tasks_eval and dice is not None:
            matrix[tasks_trained.index(t)][tasks_eval.index(e)] = dice

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="YlGn", aspect="auto")
    ax.set_xticks(range(len(tasks_eval)))
    ax.set_xticklabels(tasks_eval, rotation=45, ha="right")
    ax.set_yticks(range(len(tasks_trained)))
    ax.set_yticklabels(tasks_trained)
    ax.set_xlabel("Evaluated On")
    ax.set_ylabel("Trained On")
    ax.set_title("Task Evaluation Matrix (Dice Mean)")
    for i in range(len(tasks_trained)):
        for j in range(len(tasks_eval)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Dice Mean")
    fig.tight_layout()
    out = out_dir / "forgetting_heatmap.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_per_task_dice(run_dir: Path, out_dir: Path) -> Path | None:
    """Per-method dice curves from individual metrics.csv files."""
    method_data = {}
    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue
        metrics_path = child / "metrics.csv"
        rows = _read_csv(metrics_path)
        if not rows:
            continue
        epochs = []
        dices = []
        for r in rows:
            epoch = _float(r.get("epoch"))
            dice = _float(r.get("val_dice_mean"))
            if epoch is not None and dice is not None:
                epochs.append(epoch)
                dices.append(dice)
        if epochs:
            method_data[child.name] = (epochs, dices)

    if not method_data:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    for method, (epochs, dices) in method_data.items():
        ax.plot(epochs, dices, marker="o", label=method, markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dice Mean")
    ax.set_title("Per-Method Dice Curves")
    ax.legend()
    fig.tight_layout()
    out = out_dir / "per_task_dice.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate result figures")
    parser.add_argument("run_dir", help="Ablation or multi-task run directory")
    parser.add_argument("--output-dir", default=None, help="Figure output dir")
    args = parser.parse_args()

    if not HAS_MPL:
        print("[plot_results] matplotlib not installed, skipping figure generation")
        return

    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir) if args.output_dir else run_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    generated = []
    for fn in [plot_method_comparison, plot_forgetting_heatmap, plot_per_task_dice]:
        result = fn(run_dir, out_dir)
        if result:
            generated.append(str(result))
            print(f"[ok] {result}")

    if not generated:
        print("[plot_results] no figures generated (missing data)")
    else:
        print(f"[plot_results] {len(generated)} figures saved to {out_dir}")


if __name__ == "__main__":
    main()
