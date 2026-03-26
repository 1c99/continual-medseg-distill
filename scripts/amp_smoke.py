#!/usr/bin/env python3
"""AMP smoke test: validates mixed precision training path.

Runs two quick training passes (AMP OFF and AMP ON) on synthetic data
and compares runtime and memory usage.

Usage:
    python scripts/amp_smoke.py
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

from src.engine.trainer import train
from src.engine.evaluator import evaluate
from src.models.factory import build_model
from src.methods import create_method
from src.data.registry import create_loaders
from src.utils.reproducibility import set_seed


def run_with_amp(amp_enabled: bool) -> dict:
    """Run a short training pass and return timing/memory stats."""
    set_seed(42)

    cfg = {
        "experiment": {"name": "amp_smoke", "seed": 42},
        "model": {
            "name": "monai_unet",
            "in_channels": 1,
            "out_channels": 3,
            "channels": [16, 32, 64, 128],
            "strides": [2, 2, 2],
            "num_res_units": 2,
        },
        "train": {
            "epochs": 2,
            "lr": 0.001,
            "max_steps_per_epoch": 10,
            "loss_type": "dicece",
            "amp": {"enabled": amp_enabled},
        },
        "data": {
            "source": "synthetic",
            "batch_size": 2,
            "synthetic": {
                "train_samples": 24,
                "val_samples": 8,
                "channels": 1,
                "num_classes": 3,
                "shape": [64, 64, 64],
            },
        },
        "runtime": {"device": "cuda" if torch.cuda.is_available() else "cpu"},
        "output": {
            "dir": f"outputs/amp_smoke/{'amp_on' if amp_enabled else 'amp_off'}",
            "best_metric": "voxel_acc",
        },
        "method": {"name": "finetune"},
    }

    model = build_model(cfg)
    method = create_method(cfg)
    train_loader, val_loader = create_loaders(cfg)
    logger = logging.getLogger(f"amp_{'on' if amp_enabled else 'off'}")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    model = train(
        model, method, train_loader, cfg, logger,
        val_loader=val_loader, evaluate_fn=evaluate,
    )

    elapsed = time.time() - t0
    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    metrics = evaluate(model, val_loader, cfg, logger)

    return {
        "amp": "ON" if amp_enabled else "OFF",
        "elapsed_sec": round(elapsed, 2),
        "peak_gpu_mb": round(peak_mem_mb, 1),
        "val_acc": round(metrics.get("voxel_acc", 0.0), 4),
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    if not torch.cuda.is_available():
        print("SKIP: No CUDA device available")
        sys.exit(1)

    print("=" * 60)
    print("AMP SMOKE TEST")
    print("=" * 60)

    # Run AMP OFF first
    print("\n--- AMP OFF ---")
    off_stats = run_with_amp(False)
    print(f"  Time: {off_stats['elapsed_sec']}s | Peak GPU: {off_stats['peak_gpu_mb']}MB | Val acc: {off_stats['val_acc']}")

    # Run AMP ON
    print("\n--- AMP ON ---")
    on_stats = run_with_amp(True)
    print(f"  Time: {on_stats['elapsed_sec']}s | Peak GPU: {on_stats['peak_gpu_mb']}MB | Val acc: {on_stats['val_acc']}")

    # Summary
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"  {'Metric':<20} {'AMP OFF':>10} {'AMP ON':>10} {'Delta':>10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")

    time_delta = on_stats['elapsed_sec'] - off_stats['elapsed_sec']
    mem_delta = on_stats['peak_gpu_mb'] - off_stats['peak_gpu_mb']

    print(f"  {'Time (s)':<20} {off_stats['elapsed_sec']:>10} {on_stats['elapsed_sec']:>10} {time_delta:>+10.2f}")
    print(f"  {'Peak GPU (MB)':<20} {off_stats['peak_gpu_mb']:>10} {on_stats['peak_gpu_mb']:>10} {mem_delta:>+10.1f}")
    print(f"  {'Val acc':<20} {off_stats['val_acc']:>10} {on_stats['val_acc']:>10}")

    if on_stats['peak_gpu_mb'] < off_stats['peak_gpu_mb']:
        print(f"\n  AMP reduced peak GPU memory by {-mem_delta:.1f}MB ({-mem_delta/off_stats['peak_gpu_mb']*100:.1f}%)")
    print("\nAMP SMOKE TEST: PASS")


if __name__ == "__main__":
    main()
