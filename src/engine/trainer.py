from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Callable

import torch
from tqdm import tqdm

from src.utils.metrics import CSVMetricsLogger


def _save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, global_step: int, best_metric: float | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": best_metric,
        },
        path,
    )


def train(
    model: torch.nn.Module,
    method,
    train_loader,
    cfg: Dict[str, Any],
    logger,
    dry_run: bool = False,
    val_loader=None,
    evaluate_fn: Callable | None = None,
):
    tcfg = cfg.get("train", {})
    device = cfg.get("runtime", {}).get("device", "cpu")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    exp_name = cfg.get("experiment", {}).get("name", "run")
    out_cfg = cfg.get("output", {})
    output_dir = Path(out_cfg.get("dir", f"outputs/{exp_name}"))
    ckpt_dir = output_dir / "checkpoints"
    metrics_logger = CSVMetricsLogger(output_dir / "metrics.csv")

    best_metric_key = out_cfg.get("best_metric", "voxel_acc")
    best_mode = out_cfg.get("best_mode", "max")

    model.to(device)
    model.train()

    lr = tcfg.get("lr", 1e-3)
    epochs = tcfg.get("epochs", 1)
    max_steps = tcfg.get("max_steps_per_epoch", 3 if dry_run else 100)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0
    best_metric = None

    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{epochs}")
        loss_sum = 0.0
        steps = 0

        for i, batch in enumerate(pbar):
            if i >= max_steps:
                break
            optimizer.zero_grad()
            loss = method.training_loss(model, batch, device)
            loss.backward()
            optimizer.step()
            global_step += 1
            loss_val = float(loss.item())
            loss_sum += loss_val
            steps += 1
            pbar.set_postfix(loss=loss_val, step=global_step)

        train_loss = loss_sum / max(steps, 1)
        row = {"epoch": epoch + 1, "train_loss": train_loss}

        eval_metrics = {}
        if evaluate_fn is not None and val_loader is not None:
            eval_metrics = evaluate_fn(model, val_loader, cfg, logger)
            for k, v in eval_metrics.items():
                if isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        row[f"val_{k}_{sub_k}"] = sub_v
                else:
                    row[f"val_{k}"] = v

        metrics_logger.log(row)

        _save_checkpoint(
            ckpt_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch + 1,
            global_step=global_step,
            best_metric=best_metric,
        )

        current = eval_metrics.get(best_metric_key)
        improved = False
        if current is not None:
            if best_metric is None:
                improved = True
            elif best_mode == "min":
                improved = current < best_metric
            else:
                improved = current > best_metric

        if improved:
            best_metric = current
            _save_checkpoint(
                ckpt_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                global_step=global_step,
                best_metric=best_metric,
            )
            logger.info(f"saved best checkpoint ({best_metric_key}={best_metric:.4f})")

        logger.info(f"epoch={epoch+1} done train_loss={train_loss:.4f}")
        if dry_run:
            logger.info("dry-run enabled; stopping after one epoch")
            break

    method.post_task_update(model)
    return model
