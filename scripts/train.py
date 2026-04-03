#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path so `from src.*` imports work when invoked as a script.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import hashlib
import json

import yaml
import torch

from src.utils.config import load_yaml, merge_dicts
from src.utils.logging import setup_logger
from src.utils.reproducibility import set_seed
from src.data.registry import create_loaders
from src.models.factory import build_model
from src.methods import create_method
from src.engine.trainer import train
from src.engine.evaluator import evaluate
from src.engine.distributed import setup_ddp, cleanup_ddp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--method-config", default=None)
    ap.add_argument("--task-config", default=None)
    ap.add_argument("--dataset-config", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--print-config", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    for p in [args.method_config, args.task_config, args.dataset_config]:
        if p:
            cfg = merge_dicts(cfg, load_yaml(p))

    if args.print_config:
        print(yaml.safe_dump(cfg, sort_keys=False))

    seed = cfg.get("experiment", {}).get("seed", 42)
    set_seed(seed)

    cfg.setdefault("runtime", {})
    cfg["runtime"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    # Distributed (DDP) setup — no-op when runtime.distributed.enabled is false
    dist_ctx = setup_ddp(cfg)

    logger = setup_logger("train")
    logger.info(f"seed={seed}")
    train_loader, val_loader = create_loaders(cfg, dist_ctx=dist_ctx)
    model = build_model(cfg)
    model = dist_ctx.wrap_model(model)
    method = create_method(cfg)

    logger.info(f"method={cfg.get('method', {}).get('name', 'finetune')}")
    logger.info(f"device={cfg['runtime']['device']} train_batches={len(train_loader)}")

    model = train(
        model,
        method,
        train_loader,
        cfg,
        logger,
        dry_run=args.dry_run,
        val_loader=val_loader,
        evaluate_fn=evaluate,
        dist_ctx=dist_ctx,
    )
    metrics = evaluate(model, val_loader, cfg, logger, dist_ctx=dist_ctx)
    logger.info(f"eval={metrics}")

    # Cleanup DDP
    cleanup_ddp()

    # Write run manifest with teacher checkpoint metadata (rank 0 only)
    if not dist_ctx.is_main_process():
        return

    output_dir = Path(cfg.get("output", {}).get("dir", "outputs/runs/default"))
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": seed,
        "method": cfg.get("method", {}).get("name", "finetune"),
        "device": cfg["runtime"]["device"],
        "metrics": metrics if isinstance(metrics, dict) else {},
    }
    teacher_cfg = cfg.get("method", {}).get("kd", {}).get("teacher", {})
    if teacher_cfg:
        teacher_meta = {
            "type": teacher_cfg.get("type"),
            "ckpt_path": teacher_cfg.get("ckpt_path"),
            "lora_path": teacher_cfg.get("lora_path"),
            "source_repo": teacher_cfg.get("source_repo"),
        }
        # Compute checkpoint hashes for provenance
        for key in ("ckpt_path", "lora_path"):
            fpath = teacher_cfg.get(key)
            if fpath and fpath != "auto" and Path(fpath).exists():
                h = hashlib.sha256()
                with open(fpath, "rb") as fh:
                    h.update(fh.read(4096))
                teacher_meta[f"{key}_hash"] = h.hexdigest()[:16]
        manifest["teacher"] = teacher_meta
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    logger.info(f"run_manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
