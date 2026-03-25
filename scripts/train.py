#!/usr/bin/env python3
from __future__ import annotations

import argparse
import yaml
import torch

from src.utils.config import load_yaml, merge_dicts
from src.utils.logging import setup_logger
from src.data.registry import create_loaders
from src.models.factory import build_model
from src.methods import create_method
from src.engine.trainer import train
from src.engine.evaluator import evaluate


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

    cfg.setdefault("runtime", {})
    cfg["runtime"]["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    logger = setup_logger("train")
    train_loader, val_loader = create_loaders(cfg)
    model = build_model(cfg)
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
    )
    metrics = evaluate(model, val_loader, cfg, logger)
    logger.info(f"eval={metrics}")


if __name__ == "__main__":
    main()
