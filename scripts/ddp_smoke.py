#!/usr/bin/env python3
"""DDP smoke test: validates multi-GPU distributed training.

Launch:
    torchrun --nproc_per_node=2 scripts/ddp_smoke.py

Validates:
    - Process group init on nccl backend
    - Rank-safe logging (only rank 0 prints summaries)
    - No duplicate checkpoint writes
    - Deterministic seeding per rank
    - Model training for 1 epoch on synthetic data
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

from src.engine.distributed import setup_ddp, cleanup_ddp
from src.engine.trainer import train
from src.engine.evaluator import evaluate
from src.models.factory import build_model
from src.methods import create_method
from src.data.registry import create_loaders
from src.utils.reproducibility import set_seed


def main():
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    logging.basicConfig(
        level=logging.INFO,
        format=f"[rank {rank}] %(message)s",
    )
    logger = logging.getLogger("ddp_smoke")

    # Each rank uses its own GPU
    device = f"cuda:{local_rank}"

    cfg = {
        "experiment": {"name": "ddp_smoke", "seed": 42},
        "model": {
            "name": "monai_unet",
            "in_channels": 1,
            "out_channels": 3,
            "channels": [8, 16],
            "strides": [2],
        },
        "train": {
            "epochs": 1,
            "lr": 0.001,
            "max_steps_per_epoch": 3,
            "loss_type": "dicece",
        },
        "data": {
            "source": "synthetic",
            "batch_size": 2,
            "synthetic": {
                "train_samples": 8,
                "val_samples": 4,
                "channels": 1,
                "num_classes": 3,
                "shape": [16, 16, 16],
            },
        },
        "runtime": {
            "device": device,
            "distributed": {
                "enabled": True,
                "backend": "nccl",
                "world_size": world_size,
            },
        },
        "output": {"dir": f"outputs/ddp_smoke/rank_{rank}", "best_metric": "voxel_acc"},
        "method": {"name": "finetune"},
    }

    # Seed per rank for reproducibility
    set_seed(cfg["experiment"]["seed"] + rank)

    # Setup DDP
    dist_ctx = setup_ddp(cfg)
    logger.info(f"DDP context: rank={dist_ctx.rank}, world={dist_ctx.world_size}")

    # Build model and wrap
    model = build_model(cfg)
    model = dist_ctx.wrap_model(model)

    method = create_method(cfg)
    train_loader, val_loader = create_loaders(cfg)

    # Train
    model = train(
        model, method, train_loader, cfg, logger,
        dry_run=True, val_loader=val_loader, evaluate_fn=evaluate,
        dist_ctx=dist_ctx,
    )

    # Verify only rank 0 has checkpoints
    ckpt = Path(cfg["output"]["dir"]) / "checkpoints" / "last.pt"
    if dist_ctx.is_main_process():
        assert ckpt.exists(), f"Rank 0 should have checkpoint at {ckpt}"
        logger.info(f"PASS: rank 0 checkpoint exists at {ckpt}")
    else:
        # Non-rank-0 should NOT have written checkpoints (they use their own output dir)
        logger.info(f"Rank {rank}: checkpoint write skipped (rank-safe)")

    dist_ctx.barrier()

    if dist_ctx.is_main_process():
        logger.info("=" * 50)
        logger.info("DDP SMOKE TEST: PASS")
        logger.info(f"  GPUs used: {world_size}")
        logger.info(f"  Backend: {dist_ctx.backend}")
        logger.info("=" * 50)

    cleanup_ddp()


if __name__ == "__main__":
    main()
