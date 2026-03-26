"""Multi-task sequential trainer for continual learning.

Trains a single model on a sequence of tasks, calling post_task_update()
between tasks, evaluating on all seen tasks after each, and computing
forgetting metrics.

Supports resuming from an interrupted run via task progress index.
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

_PROGRESS_FILE = "task_progress.json"


# ---------------------------------------------------------------------------
# Task checkpoint helpers
# ---------------------------------------------------------------------------

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


def _load_task_checkpoint(
    path: Path,
    model: torch.nn.Module,
    method,
    task_id: str,
) -> None:
    """Restore model and method state from a task checkpoint."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])

    method_state_path = path.parent / f"method_state_{task_id}.pt"
    if method_state_path.exists():
        try:
            method.load_state(method_state_path, model_template=model)
        except TypeError:
            method.load_state(method_state_path)


# ---------------------------------------------------------------------------
# Task progress tracking (for resume)
# ---------------------------------------------------------------------------

def _save_progress(
    output_dir: Path,
    completed_task_idx: int,
    task_id: str,
    task_order: List[str],
    task_eval_history: Dict[str, Dict[str, Dict[str, float]]],
    resume_count: int = 0,
) -> None:
    """Persist progress to allow resume after interruption."""
    progress = {
        "completed_task_idx": completed_task_idx,
        "last_completed_task_id": task_id,
        "task_order_so_far": task_order,
        "task_eval_history": task_eval_history,
        "resume_count": resume_count,
    }
    path = output_dir / _PROGRESS_FILE
    path.write_text(json.dumps(progress, indent=2, default=str), encoding="utf-8")


def _load_progress(output_dir: Path) -> Dict[str, Any] | None:
    """Load progress from a previous interrupted run. Returns None if no progress."""
    path = output_dir / _PROGRESS_FILE
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


# ---------------------------------------------------------------------------
# Forgetting computation
# ---------------------------------------------------------------------------

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

    # BWT (Backward Transfer) = mean of (R_{T,i} - R_{i,i}) for i < T
    # Negative BWT means forgetting; positive means backward improvement
    bwt_per_task: Dict[str, float] = {}
    for i, task_id in enumerate(task_order[:-1]):
        perf_after_own = matrix[task_id].get(task_id)
        perf_after_last = matrix[last_task].get(task_id)
        if perf_after_own is not None and perf_after_last is not None:
            bwt_per_task[task_id] = perf_after_last - perf_after_own
        else:
            bwt_per_task[task_id] = math.nan

    valid_bwt = [v for v in bwt_per_task.values() if not math.isnan(v)]
    mean_bwt = sum(valid_bwt) / len(valid_bwt) if valid_bwt else math.nan

    # FWT (Forward Transfer) = mean of (R_{i-1,i} - random_baseline) for i > 0
    # We approximate random baseline as 0 (no prior knowledge)
    fwt_per_task: Dict[str, float] = {}
    for i, task_id in enumerate(task_order[1:], start=1):
        prev_task = task_order[i - 1]
        perf_before_training = matrix.get(prev_task, {}).get(task_id)
        if perf_before_training is not None:
            fwt_per_task[task_id] = perf_before_training
        else:
            fwt_per_task[task_id] = math.nan

    valid_fwt = [v for v in fwt_per_task.values() if not math.isnan(v)]
    mean_fwt = sum(valid_fwt) / len(valid_fwt) if valid_fwt else math.nan

    return {
        "matrix": matrix,
        "per_task": forgetting_per_task,
        "mean": mean_forgetting,
        "bwt_per_task": bwt_per_task,
        "mean_bwt": mean_bwt,
        "fwt_per_task": fwt_per_task,
        "mean_fwt": mean_fwt,
        "metric_key": metric_key,
    }


# ---------------------------------------------------------------------------
# Evidence output writers
# ---------------------------------------------------------------------------

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
            for k, v in evals.items():
                if isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        row[f"val_{k}_{sub_k}"] = sub_v
            rows.append(row)

    if rows:
        all_keys: List[str] = []
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
    forg_path.write_text(
        json.dumps(forgetting, indent=2, default=str), encoding="utf-8"
    )

    # Summary JSON
    summary = {
        "task_order": task_order,
        "num_tasks": len(task_order),
        "mean_forgetting": forgetting["mean"],
        "mean_bwt": forgetting.get("mean_bwt", math.nan),
        "mean_fwt": forgetting.get("mean_fwt", math.nan),
        "forgetting_metric": forgetting["metric_key"],
        "per_task_forgetting": forgetting["per_task"],
        "bwt_per_task": forgetting.get("bwt_per_task", {}),
        "fwt_per_task": forgetting.get("fwt_per_task", {}),
    }
    (output_dir / "multi_task_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_task_sequence(
    model: torch.nn.Module,
    method,
    task_configs: List[Dict[str, Any]],
    global_cfg: Dict[str, Any],
    logger,
    evaluate_fn: Callable,
    output_dir: Path,
    dry_run: bool = False,
    resume: bool = False,
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
        resume: If True, attempt to resume from last completed task.

    Returns:
        Dict with per-task results, forgetting metrics, and paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_order: List[str] = []
    val_loaders: Dict[str, Any] = {}
    task_eval_history: Dict[str, Dict[str, Dict[str, float]]] = {}
    start_idx = 0
    resume_count = 0

    # ---- resume logic ----
    if resume:
        progress = _load_progress(output_dir)
        if progress is not None:
            completed_idx = progress["completed_task_idx"]
            start_idx = completed_idx + 1
            task_order = progress["task_order_so_far"]
            resume_count = progress.get("resume_count", 0) + 1
            task_eval_history = progress["task_eval_history"]

            if start_idx >= len(task_configs):
                logger.info("All tasks already completed. Nothing to resume.")
                forgetting = compute_forgetting(task_eval_history, task_order)
                return {
                    "task_order": task_order,
                    "eval_history": task_eval_history,
                    "forgetting": forgetting,
                    "output_dir": str(output_dir),
                    "resumed": True,
                }

            # Restore model and method state from last completed task
            last_task_id = progress["last_completed_task_id"]
            ckpt_path = (
                output_dir
                / last_task_id
                / "checkpoints"
                / f"after_{last_task_id}.pt"
            )
            if ckpt_path.exists():
                _load_task_checkpoint(ckpt_path, model, method, last_task_id)
                logger.info(
                    f"Resumed from task {start_idx}/{len(task_configs)} "
                    f"(after {last_task_id})"
                )
            else:
                logger.warning(
                    f"Resume requested but checkpoint {ckpt_path} not found. "
                    "Starting from scratch."
                )
                start_idx = 0
                task_order = []
                task_eval_history = {}

            # Rebuild val_loaders for already-completed tasks
            for prev_idx in range(start_idx):
                prev_override = task_configs[prev_idx]
                prev_task_id = prev_override.get("id", f"task_{prev_idx}")
                prev_cfg = merge_dicts(global_cfg, prev_override)
                _, prev_val_loader = create_loaders(prev_cfg)
                val_loaders[prev_task_id] = (prev_val_loader, prev_cfg)
        else:
            logger.info("No previous progress found. Starting from scratch.")

    # ---- task loop ----
    for task_idx in range(start_idx, len(task_configs)):
        task_override = task_configs[task_idx]
        task_id = task_override.get("id", f"task_{task_idx}")
        task_order.append(task_id)
        logger.info(
            f"=== Task {task_idx + 1}/{len(task_configs)}: {task_id} ==="
        )

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
            model,
            method,
            task_idx,
            task_id,
        )

        # Save progress for resume
        _save_progress(
            output_dir, task_idx, task_id, task_order, task_eval_history,
            resume_count=resume_count,
        )
        logger.info(f"  checkpoint + progress saved: {task_output / 'checkpoints'}")

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
        "resumed": start_idx > 0,
    }
