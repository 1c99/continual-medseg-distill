"""Multi-task sequential trainer for continual learning.

Trains a single model on a sequence of tasks, calling post_task_update()
between tasks, evaluating on all seen tasks after each, and computing
forgetting metrics.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List

import torch

from src.data.registry import create_loaders
from src.engine.trainer import train
from src.utils.config import merge_dicts


def _save_task_checkpoint(
    path: Path,
    model: torch.nn.Module,
    method,
    task_idx: int,
    task_id: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "task_idx": task_idx,
            "task_id": task_id,
            "model_state_dict": model.state_dict(),
        },
        path,
    )
    method_state_path = path.parent / f"method_state_{task_id}.pt"
    try:
        method.save_state(method_state_path, model_template=model)
    except TypeError:
        method.save_state(method_state_path)


def compute_forgetting(
    task_eval_history: Dict[str, Dict[str, Dict[str, float]]],
    task_order: List[str],
    metric_key: str = "dice_mean",
) -> Dict[str, Any]:
    """Compute backward transfer (forgetting) from per-task eval history.

    Args:
        task_eval_history: {trained_task_id: {eval_task_id: {metric: value}}}
        task_order: ordered list of task IDs
        metric_key: metric to measure forgetting on

    Returns:
        Dict with 'matrix' (task x task), 'per_task' forgetting, 'mean' forgetting.
    """
    n = len(task_order)
    matrix: Dict[str, Dict[str, float | None]] = {}
    forgetting_per_task: Dict[str, float] = {}

    for trained_id in task_order:
        matrix[trained_id] = {}
        for eval_id in task_order:
            evals = task_eval_history.get(trained_id, {})
            metrics = evals.get(eval_id)
            if metrics is not None:
                matrix[trained_id][eval_id] = metrics.get(metric_key, math.nan)
            else:
                matrix[trained_id][eval_id] = None

    # Forgetting for task_i = perf(task_i, after_task_i) - perf(task_i, after_last_task)
    last_task = task_order[-1]
    for i, task_id in enumerate(task_order[:-1]):
        perf_after_own = matrix[task_id].get(task_id)
        perf_after_last = matrix[last_task].get(task_id)
        if perf_after_own is not None and perf_after_last is not None:
            forgetting_per_task[task_id] = perf_after_own - perf_after_last
        else:
            forgetting_per_task[task_id] = math.nan

    valid = [v for v in forgetting_per_task.values() if not math.isnan(v)]
    mean_forgetting = sum(valid) / len(valid) if valid else math.nan

    return {
        "matrix": matrix,
        "per_task": forgetting_per_task,
        "mean": mean_forgetting,
        "metric_key": metric_key,
    }


def _write_task_results(
    output_dir: Path,
    task_order: List[str],
    task_eval_history: Dict[str, Dict[str, Dict[str, float]]],
    forgetting: Dict[str, Any],
) -> None:
    """Write per-task CSV, forgetting JSON, and summary to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-task evaluation CSV
    rows = []
    for trained_id in task_order:
        for eval_id in task_order:
            evals = task_eval_history.get(trained_id, {}).get(eval_id)
            if evals is None:
                continue
            row = {
                "trained_on": trained_id,
                "evaluated_on": eval_id,
                **{f"val_{k}": v for k, v in evals.items()
                   if not isinstance(v, dict)},
            }
            # Flatten nested dicts (dice_per_class, hd95_per_class)
            for k, v in evals.items():
                if isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        row[f"val_{k}_{sub_k}"] = sub_v
            rows.append(row)

    if rows:
        all_keys = []
        for r in rows:
            for k in r:
                if k not in all_keys:
                    all_keys.append(k)
        csv_path = output_dir / "task_eval_matrix.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=all_keys)
            w.writeheader()
            w.writerows(rows)

    # Forgetting JSON
    forg_path = output_dir / "forgetting.json"
    forg_path.write_text(json.dumps(forgetting, indent=2, default=str), encoding="utf-8")

    # Summary JSON
    summary = {
        "task_order": task_order,
        "num_tasks": len(task_order),
        "mean_forgetting": forgetting["mean"],
        "forgetting_metric": forgetting["metric_key"],
        "per_task_forgetting": forgetting["per_task"],
    }
    (output_dir / "multi_task_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )


def run_task_sequence(
    model: torch.nn.Module,
    method,
    task_configs: List[Dict[str, Any]],
    global_cfg: Dict[str, Any],
    logger,
    evaluate_fn: Callable,
    output_dir: Path,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run a sequence of tasks with continual learning.

    Args:
        model: The model to train.
        method: ContinualMethod instance.
        task_configs: List of per-task config overrides.
        global_cfg: Base configuration.
        logger: Logger instance.
        evaluate_fn: Evaluation function (model, loader, cfg, logger) -> dict.
        output_dir: Root output directory for all task outputs.
        dry_run: If True, limits training steps.

    Returns:
        Dict with per-task results, forgetting metrics, and paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_order = []
    val_loaders: Dict[str, Any] = {}  # task_id -> (val_loader, task_cfg)
    task_eval_history: Dict[str, Dict[str, Dict[str, float]]] = {}

    for task_idx, task_override in enumerate(task_configs):
        task_id = task_override.get("id", f"task_{task_idx}")
        task_order.append(task_id)
        logger.info(f"=== Task {task_idx + 1}/{len(task_configs)}: {task_id} ===")

        # Build task-specific config
        task_cfg = merge_dicts(global_cfg, task_override)
        task_output = output_dir / task_id
        task_cfg.setdefault("output", {})
        task_cfg["output"]["dir"] = str(task_output)

        # Create data loaders for this task
        train_loader, val_loader = create_loaders(task_cfg)
        val_loaders[task_id] = (val_loader, task_cfg)

        # Train on this task
        model = train(
            model,
            method,
            train_loader,
            task_cfg,
            logger,
            dry_run=dry_run,
            val_loader=val_loader,
            evaluate_fn=evaluate_fn,
        )

        # Evaluate on ALL seen tasks
        task_eval_history[task_id] = {}
        for prev_id in task_order:
            prev_loader, prev_cfg = val_loaders[prev_id]
            metrics = evaluate_fn(model, prev_loader, prev_cfg, logger)
            task_eval_history[task_id][prev_id] = metrics
            logger.info(
                f"  eval[{prev_id}] dice={metrics.get('dice_mean', float('nan')):.4f} "
                f"hd95={metrics.get('hd95_mean', float('nan')):.4f}"
            )

        # Save task checkpoint + method state
        _save_task_checkpoint(
            task_output / "checkpoints" / f"after_{task_id}.pt",
            model, method, task_idx, task_id,
        )
        logger.info(f"  checkpoint saved: {task_output / 'checkpoints'}")

    # Compute forgetting
    forgetting = compute_forgetting(task_eval_history, task_order)
    logger.info(f"Mean forgetting (dice_mean): {forgetting['mean']:.4f}")

    # Write evidence outputs
    _write_task_results(output_dir, task_order, task_eval_history, forgetting)
    logger.info(f"Evidence outputs written to {output_dir}")

    return {
        "task_order": task_order,
        "eval_history": task_eval_history,
        "forgetting": forgetting,
        "output_dir": str(output_dir),
    }
