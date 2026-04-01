#!/usr/bin/env python3
"""Run a continual learning task sequence (A→B→...).

Usage:
    # Synthetic dry run
    python scripts/run_continual.py \
        --base-config configs/base.yaml \
        --task-config configs/tasks/synthetic_2task.yaml \
        --dry-run

    # Real data A→B
    python scripts/run_continual.py \
        --base-config configs/base.yaml \
        --task-config configs/tasks/totalseg_AB_sequence.yaml \
        --dataset-config configs/datasets/totalseg_train_clean.yaml \
        --method-config configs/methods/finetune.yaml

Produces:
    task_eval_matrix.csv, forgetting.json, multi_task_summary.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import yaml

from src.utils.config import load_yaml, merge_dicts
from src.utils.logging import setup_logger
from src.utils.reproducibility import set_seed
from src.models.factory import build_model
from src.methods import create_method
from src.engine.evaluator import evaluate
from src.engine.multi_task_trainer import run_task_sequence
from src.engine.distributed import setup_ddp, cleanup_ddp


def main():
    ap = argparse.ArgumentParser(description="Run continual learning task sequence")
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--task-config", required=True, help="Task sequence YAML")
    ap.add_argument("--method-config", default=None)
    ap.add_argument("--dataset-config", default=None)
    ap.add_argument("--override-config", default=None, help="Extra YAML merged last (e.g. kd weight override)")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--print-config", action="store_true")
    args = ap.parse_args()

    # Build global config
    cfg = load_yaml(args.base_config)
    if args.method_config:
        cfg = merge_dicts(cfg, load_yaml(args.method_config))
    if args.dataset_config:
        cfg = merge_dicts(cfg, load_yaml(args.dataset_config))
    if args.override_config:
        cfg = merge_dicts(cfg, load_yaml(args.override_config))

    # Load task sequence
    task_cfg = load_yaml(args.task_config)
    task_seq = task_cfg.get("task_sequence", {})
    task_configs = task_seq.get("tasks", [])

    if not task_configs:
        print("ERROR: No tasks found in task_sequence.tasks", file=sys.stderr)
        sys.exit(1)

    # Output dir
    seq_name = task_seq.get("name", "continual_run")
    output_dir = Path(args.output_dir) if args.output_dir else Path(
        cfg.get("output", {}).get("dir", f"outputs/{seq_name}")
    )

    if args.print_config:
        print(yaml.safe_dump(cfg, sort_keys=False))
        print(f"Task sequence: {seq_name}")
        for i, tc in enumerate(task_configs):
            print(f"  Task {i}: {tc.get('id', f'task_{i}')}")
        return

    # Setup
    cfg.setdefault("runtime", {})
    cfg["runtime"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    # Distributed (DDP) setup — no-op when runtime.distributed.enabled is false
    dist_ctx = setup_ddp(cfg)

    # Apply model overrides from first task so the model is built with the
    # correct architecture. Multi-head wrapper is created with first task's head.
    first_task_id = task_configs[0].get("id", "task_0")
    first_task_model = task_configs[0].get("model", {})
    if first_task_model:
        cfg = merge_dicts(cfg, {"model": first_task_model})

    seed = cfg.get("experiment", {}).get("seed", 42)
    set_seed(seed)

    logger = setup_logger("continual")
    logger.info(f"Continual sequence: {seq_name}")
    logger.info(f"Tasks: {[tc.get('id', f'task_{i}') for i, tc in enumerate(task_configs)]}")
    logger.info(f"Method: {cfg.get('method', {}).get('name', 'finetune')}")
    logger.info(f"Device: {cfg['runtime']['device']}")
    logger.info(f"Output: {output_dir}")

    # ---- Save run manifest (config locking) ----
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_cfg_str = yaml.safe_dump(cfg, sort_keys=True)
    config_hash = hashlib.sha256(resolved_cfg_str.encode()).hexdigest()[:12]

    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True
        ).strip()
    except Exception:
        commit_hash = "unknown"

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "commit_hash": commit_hash,
        "config_hash": config_hash,
        "seed": seed,
        "sequence_name": seq_name,
        "method": cfg.get("method", {}).get("name", "finetune"),
        "tasks": [tc.get("id", f"task_{i}") for i, tc in enumerate(task_configs)],
        "device": cfg["runtime"]["device"],
        "dry_run": args.dry_run,
        "args": {
            "base_config": args.base_config,
            "task_config": args.task_config,
            "method_config": args.method_config,
            "dataset_config": args.dataset_config,
        },
    }

    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (output_dir / "resolved_config.yaml").write_text(resolved_cfg_str, encoding="utf-8")

    logger.info(f"Config hash: {config_hash}")
    logger.info(f"Commit: {commit_hash}")
    logger.info(f"Seed: {seed}")

    model = build_model(cfg, task_id=first_task_id)
    method = create_method(cfg)

    results = run_task_sequence(
        model=model,
        method=method,
        task_configs=task_configs,
        global_cfg=cfg,
        logger=logger,
        evaluate_fn=evaluate,
        output_dir=output_dir,
        dry_run=args.dry_run,
        resume=args.resume,
        dist_ctx=dist_ctx,
    )

    # Cleanup DDP
    cleanup_ddp()

    # Print summary
    logger.info("=" * 50)
    logger.info("CONTINUAL SEQUENCE COMPLETE")
    logger.info(f"  Tasks: {results['task_order']}")
    logger.info(f"  Mean forgetting: {results['forgetting']['mean']:.4f}")
    logger.info(f"  Mean BWT: {results['forgetting'].get('mean_bwt', float('nan')):.4f}")
    logger.info(f"  Mean FWT: {results['forgetting'].get('mean_fwt', float('nan')):.4f}")
    logger.info(f"  Artifacts: {results['output_dir']}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
