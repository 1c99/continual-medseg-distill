from __future__ import annotations

import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Any, Callable

import torch
from tqdm import tqdm

from src.utils.metrics import CSVMetricsLogger

trainer_logger = logging.getLogger(__name__)


def _save_checkpoint(path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, global_step: int, best_metric: float | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Unwrap DDP if needed
    state_model = model.module if hasattr(model, "module") else model
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": state_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": best_metric,
        },
        path,
    )


class EarlyStopper:
    """Tracks a monitored metric and signals when to stop.

    Config keys (under ``train.early_stopping``):
        patience: int  (epochs without improvement, default: 0 = disabled)
        metric: str    (eval metric key, default: dice_mean)
        mode: str      (max | min, default: max)
    """

    def __init__(self, patience: int = 0, metric: str = "dice_mean", mode: str = "max"):
        self.patience = patience
        self.metric = metric
        self.mode = mode
        self._best: float | None = None
        self._wait: int = 0
        self.enabled = patience > 0

    def step(self, eval_metrics: dict) -> bool:
        """Return True if training should stop."""
        if not self.enabled:
            return False
        current = eval_metrics.get(self.metric)
        if current is None:
            return False

        improved = False
        if self._best is None:
            improved = True
        elif self.mode == "min":
            improved = current < self._best
        else:
            improved = current > self._best

        if improved:
            self._best = current
            self._wait = 0
        else:
            self._wait += 1

        return self._wait >= self.patience


def train(
    model: torch.nn.Module,
    method,
    train_loader,
    cfg: Dict[str, Any],
    logger,
    dry_run: bool = False,
    val_loader=None,
    evaluate_fn: Callable | None = None,
    dist_ctx=None,
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

    # Early stopping
    es_cfg = tcfg.get("early_stopping", {})
    early_stopper = EarlyStopper(
        patience=int(es_cfg.get("patience", 0)),
        metric=es_cfg.get("metric", "dice_mean"),
        mode=es_cfg.get("mode", "max"),
    )

    # --- AMP setup ---
    amp_enabled = bool(tcfg.get("amp", {}).get("enabled", False))
    amp_dtype = torch.float16
    use_cuda = device != "cpu" and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_enabled and use_cuda))
    amp_ctx = torch.cuda.amp.autocast(enabled=(amp_enabled and use_cuda), dtype=amp_dtype) if use_cuda else nullcontext()
    logger.info(f"AMP: {'ON' if (amp_enabled and use_cuda) else 'OFF'}")

    # --- Gradient checkpointing ---
    grad_ckpt_enabled = bool(tcfg.get("grad_checkpoint", {}).get("enabled", False))
    if grad_ckpt_enabled:
        raw_model = model.module if hasattr(model, "module") else model
        if hasattr(raw_model, "gradient_checkpointing_enable"):
            raw_model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing: ON (model native)")
        else:
            logger.info("Gradient checkpointing: ON (manual torch.utils.checkpoint)")
    else:
        logger.info("Gradient checkpointing: OFF")

    # --- DDP context ---
    is_main = dist_ctx.is_main_process() if dist_ctx else True
    grad_accum_steps = dist_ctx.grad_accum_steps if dist_ctx else 1

    model.to(device)
    model.train()

    lr = tcfg.get("lr", 1e-3)
    epochs = tcfg.get("epochs", 1)
    max_steps = tcfg.get("max_steps_per_epoch", 3 if dry_run else 100)

    # When student LoRA is enabled, only train LoRA parameters
    lora_cfg = cfg.get("model", {}).get("lora", {})
    if lora_cfg.get("enabled", False):
        from src.models.lora import get_lora_params
        trainable = list(get_lora_params(model))
        trainer_logger.info(f"Optimizer: {len(trainable)} LoRA param groups")
        optimizer = torch.optim.Adam(trainable, lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0
    best_metric = None

    for epoch in range(epochs):
        # Ensure each DDP rank sees a different data shard per epoch
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{epochs}", disable=not is_main)
        loss_sum = 0.0
        loss_seg_sum = 0.0
        loss_kd_sum = 0.0
        steps = 0

        for i, batch in enumerate(pbar):
            if i >= max_steps:
                break

            with amp_ctx:
                loss = method.training_loss(model, batch, device)

            if grad_accum_steps > 1:
                loss = loss / grad_accum_steps

            scaler.scale(loss).backward()

            if (i + 1) % grad_accum_steps == 0 or (i + 1) == max_steps:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            global_step += 1
            loss_val = float(loss.item()) * grad_accum_steps  # undo scaling for logging
            loss_sum += loss_val
            # Track component losses if method exposes them
            if hasattr(method, '_last_loss_seg'):
                loss_seg_sum += method._last_loss_seg
                loss_kd_sum += method._last_loss_kd
            steps += 1
            if is_main:
                pbar.set_postfix(loss=loss_val, step=global_step)

        train_loss = loss_sum / max(steps, 1)
        row = {"epoch": epoch + 1, "train_loss": train_loss}
        if hasattr(method, '_last_loss_seg') and loss_kd_sum > 0:
            avg_seg = loss_seg_sum / max(steps, 1)
            avg_kd = loss_kd_sum / max(steps, 1)
            row["train_loss_seg"] = avg_seg
            row["train_loss_kd"] = avg_kd
            row["train_kd_seg_ratio"] = avg_kd / (avg_seg + 1e-8)

        eval_metrics = {}
        if evaluate_fn is not None and val_loader is not None:
            eval_metrics = evaluate_fn(model, val_loader, cfg, logger, dist_ctx=dist_ctx)
            for k, v in eval_metrics.items():
                if isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        row[f"val_{k}_{sub_k}"] = sub_v
                else:
                    row[f"val_{k}"] = v

        if is_main:
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
            if is_main:
                _save_checkpoint(
                    ckpt_dir / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch + 1,
                    global_step=global_step,
                    best_metric=best_metric,
                )
                logger.info(f"saved best checkpoint ({best_metric_key}={best_metric:.4f})")

        if is_main:
            logger.info(f"epoch={epoch+1} done train_loss={train_loss:.4f}")

        if dry_run:
            if is_main:
                logger.info("dry-run enabled; stopping after one epoch")
            break

        if early_stopper.step(eval_metrics):
            if is_main:
                logger.info(
                    f"Early stopping triggered after {epoch+1} epochs "
                    f"(patience={early_stopper.patience}, metric={early_stopper.metric})"
                )
            break

    method.post_task_update(model, train_loader=train_loader)
    return model
