from __future__ import annotations

from typing import Dict, Any

import torch

from src.engine.distributed import unwrap_model
from src.utils.metrics import segmentation_metrics


def evaluate(model: torch.nn.Module, val_loader, cfg: Dict[str, Any], logger, dist_ctx=None):
    device = cfg.get("runtime", {}).get("device", "cpu")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    num_classes = int(
        cfg.get("model", {}).get(
            "out_channels",
            cfg.get("data", {}).get("synthetic", {}).get("num_classes", 2),
        )
    )

    # Switch to the correct task head for multi-head models
    task_id = cfg.get("task", {}).get("id") or cfg.get("id")
    if task_id and hasattr(unwrap_model(model), "current_task"):
        prev_task = unwrap_model(model).current_task
        unwrap_model(model).current_task = task_id
    else:
        prev_task = None

    model.to(device)
    model.eval()
    total = 0
    correct = 0.0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)

            correct += (pred == y).float().mean().item()
            total += 1

            all_preds.append(pred.cpu())
            all_targets.append(y.cpu())

    voxel_acc = correct / max(total, 1)

    if all_preds:
        pred_cat = torch.cat(all_preds, dim=0)
        target_cat = torch.cat(all_targets, dim=0)
        seg = segmentation_metrics(pred_cat, target_cat, num_classes=num_classes, include_background=False, compute_hd95=True)
    else:
        seg = {
            "dice_mean": float("nan"),
            "dice_per_class": {},
            "hd95_mean": float("nan"),
            "hd95_per_class": {},
            "warnings": ["No validation batches; segmentation metrics set to NaN."],
        }

    # Aggregate metrics across DDP ranks
    if dist_ctx is not None and dist_ctx.enabled:
        import torch as _torch
        # All-reduce scalar metrics (average across ranks)
        for key in ["dice_mean", "hd95_mean"]:
            val = seg.get(key, float("nan"))
            if not (isinstance(val, float) and val != val):  # skip NaN
                t = _torch.tensor(val, device="cuda")
                t = dist_ctx.reduce_tensor(t)
                seg[key] = float(t.item())
        voxel_acc_t = _torch.tensor(voxel_acc, device="cuda")
        voxel_acc_t = dist_ctx.reduce_tensor(voxel_acc_t)
        voxel_acc = float(voxel_acc_t.item())

    for w in seg.get("warnings", []):
        logger.warning(w)

    logger.info(
        "val_metrics "
        f"voxel_acc={voxel_acc:.4f} "
        f"dice_mean={seg['dice_mean']:.4f} "
        f"hd95_mean={seg['hd95_mean']:.4f}"
    )

    # Restore previous task head
    if prev_task is not None and hasattr(unwrap_model(model), "current_task"):
        unwrap_model(model).current_task = prev_task

    return {
        "voxel_acc": float(voxel_acc),
        "dice_mean": float(seg["dice_mean"]),
        "dice_per_class": seg["dice_per_class"],
        "hd95_mean": float(seg["hd95_mean"]),
        "hd95_per_class": seg["hd95_per_class"],
    }
