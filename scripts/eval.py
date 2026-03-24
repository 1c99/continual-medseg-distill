#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.config import load_experiment_config
from utils.logging import setup_logger
from data import create_loaders
from models import create_model
from engine import evaluate


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate continual medseg scaffold")
    p.add_argument("--config", type=str, default="configs/base.yaml")
    p.add_argument("--synthetic", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("eval")

    cfg = load_experiment_config(ROOT / args.config)
    if args.synthetic:
        cfg.setdefault("data", {})["source"] = "synthetic"

    _, val_loader = create_loaders(cfg)
    model = create_model(cfg)
    evaluate(model, val_loader, cfg, logger)


if __name__ == "__main__":
    main()
