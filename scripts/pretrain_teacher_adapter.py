#!/usr/bin/env python3
"""Pre-train an external teacher's output adapter on task data.

Trains only the lightweight _OutputAdapter (~ 600K params) while keeping the
frozen backbone (MedSAM2/MedSAM3) fixed. This gives the teacher meaningful
segmentation predictions instead of random projections.

Usage:
    # Pre-train MedSAM2 adapter on Task A organs
    python scripts/pretrain_teacher_adapter.py \
        --base-config configs/base.yaml \
        --task-config configs/tasks/taskA_organs.yaml \
        --dataset-config configs/datasets/totalseg_train_clean.yaml \
        --teacher-type medsam2 \
        --teacher-ckpt third_party/medsam2/checkpoints/MedSAM2_latest.pt \
        --output checkpoints/medsam2_adapter_taskA.pt \
        --epochs 10

    # Pre-train MedSAM3 adapter on Task A organs
    python scripts/pretrain_teacher_adapter.py \
        --base-config configs/base.yaml \
        --task-config configs/tasks/taskA_organs.yaml \
        --dataset-config configs/datasets/totalseg_train_clean.yaml \
        --teacher-type medsam3 \
        --teacher-ckpt checkpoints/medsam3_teacher.pt \
        --output checkpoints/medsam3_adapter_taskA.pt \
        --epochs 10

Produces:
    - Adapter state dict (.pt file)
    - Training metrics CSV alongside the .pt file
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.utils.config import load_yaml, merge_dicts
from src.utils.logging import setup_logger
from src.utils.reproducibility import set_seed
from src.data.registry import create_loaders


def _dicece_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Dice + CE loss matching the student's training loss."""
    if logits.ndim != 5 or target.ndim != 4:
        raise ValueError(f"_dicece_loss shape error: logits={logits.shape}, target={target.shape}")
    num_classes = logits.shape[1]
    probs = F.softmax(logits, dim=1)
    target_one_hot = F.one_hot(target.long(), num_classes).permute(0, -1, *range(1, target.ndim)).float()

    smooth = 1e-5
    dice_loss = 0.0
    count = 0
    for c in range(1, num_classes):
        pred_c = probs[:, c]
        target_c = target_one_hot[:, c]
        intersection = (pred_c * target_c).sum()
        dice_score = (2.0 * intersection + smooth) / (pred_c.sum() + target_c.sum() + smooth)
        dice_loss += 1.0 - dice_score
        count += 1
    if count > 0:
        dice_loss /= count

    ce_loss = F.cross_entropy(logits, target)
    return dice_loss + ce_loss


def create_teacher_backend(
    teacher_type: str, teacher_ckpt: str, out_channels: int, device: str,
    adapter_type: str = "standard", task_id: str = "task_0",
    deep_adapter: bool = False,
):
    """Create and load a teacher backend."""
    cfg = {
        "type": teacher_type,
        "ckpt_path": teacher_ckpt,
        "output_channels": out_channels,
        "adapter_channels": 256,
        "adapter_type": adapter_type,
        "initial_task_id": task_id,
        "deep_adapter": deep_adapter,
    }

    from src.methods.teacher_backends import create_backend
    backend = create_backend(cfg)
    backend.to(device)
    return backend


def main():
    ap = argparse.ArgumentParser(description="Pre-train teacher output adapter")
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--task-config", default=None, help="Task YAML for class/organ definitions")
    ap.add_argument("--dataset-config", default=None)
    ap.add_argument("--teacher-type", required=True, choices=["medsam2", "medsam3", "sam3"])
    ap.add_argument("--teacher-ckpt", required=True, help="Path to teacher backbone checkpoint")
    ap.add_argument("--output", required=True, help="Path to save adapter weights (.pt)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-steps-per-epoch", type=int, default=100)
    ap.add_argument("--adapter-type", default="standard", choices=["standard", "gated_residual"],
                    help="Adapter architecture (standard or gated_residual)")
    ap.add_argument("--deep-adapter", action="store_true",
                    help="Use deeper adapter (more conv layers) for higher capacity")
    ap.add_argument("--task-id", default="task_0", help="Task ID for gated adapter")
    ap.add_argument("--gate-weight", type=float, default=0.5, help="Weight for gate BCE loss")
    ap.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    logger = setup_logger("adapter_pretrain")

    # Build config
    cfg = load_yaml(args.base_config)
    if args.task_config:
        task_yaml = load_yaml(args.task_config)
        # Merge task-level overrides (data.totalseg.organs, model.out_channels, etc.)
        task_data = task_yaml.get("data", task_yaml.get("task", {}).get("data", {}))
        if task_data:
            cfg = merge_dicts(cfg, {"data": task_data})
        task_model = task_yaml.get("model", {})
        if task_model:
            cfg = merge_dicts(cfg, {"model": task_model})
    if args.dataset_config:
        cfg = merge_dicts(cfg, load_yaml(args.dataset_config))

    out_channels = cfg.get("model", {}).get("out_channels", 6)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.setdefault("runtime", {})["device"] = device

    max_steps = 3 if args.dry_run else args.max_steps_per_epoch
    epochs = 1 if args.dry_run else args.epochs

    logger.info(f"Teacher type: {args.teacher_type}")
    logger.info(f"Teacher checkpoint: {args.teacher_ckpt}")
    logger.info(f"Output channels: {out_channels}")
    logger.info(f"Device: {device}")
    logger.info(f"Epochs: {epochs}, max_steps/epoch: {max_steps}")

    # Create data loaders
    train_loader, val_loader = create_loaders(cfg)
    logger.info(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")

    # Create teacher backend (backbone frozen, adapter random)
    backend = create_teacher_backend(
        args.teacher_type, args.teacher_ckpt, out_channels, device,
        adapter_type=args.adapter_type, task_id=args.task_id,
        deep_adapter=args.deep_adapter,
    )
    is_gated = args.adapter_type == "gated_residual"
    logger.info(f"Teacher backend loaded (backbone frozen, adapter={args.adapter_type})")

    # Only optimize the adapter
    adapter = backend._adapter
    optimizer = torch.optim.Adam(adapter.parameters(), lr=args.lr)
    logger.info(f"Adapter params: {sum(p.numel() for p in adapter.parameters()):,}")

    # Training loop
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = output_path.with_suffix(".csv")

    best_val_dice = -1.0
    start_epoch = 0
    metrics_rows = []

    # Resume from checkpoint if provided
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        adapter.load_state_dict(ckpt["adapter_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        best_val_dice = ckpt.get("val_dice", -1.0)
        logger.info(f"Resumed from {args.resume} (epoch={start_epoch}, dice={best_val_dice:.4f})")

    for epoch in range(start_epoch, start_epoch + epochs):
        adapter.train()
        train_loss_sum = 0.0
        steps = 0

        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{start_epoch + epochs}")
        for i, batch in enumerate(pbar):
            if i >= max_steps:
                break

            x = batch["image"].to(device)
            y = batch["label"].to(device)

            # Forward: backbone (frozen, no_grad) → adapter (trainable)
            with torch.no_grad():
                features = backend._extract_features_3d(x)
                # Force channels-first layout (MedSAM3 may produce channels-last from backbone)
                if features.stride(1) == 1 and features.ndim == 5:
                    features = features.to(memory_format=torch.contiguous_format)

            B, C_in, D, H, W = x.shape

            if is_gated:
                logits, gate = adapter(features, (D, H, W))
                if logits.ndim != 5:
                    raise ValueError(f"GRACE logits wrong dim: {logits.shape}, features={features.shape}, target=({D},{H},{W})")
                seg_loss = _dicece_loss(logits, y)

                # Gate supervision: gate should be high where prediction is correct
                with torch.no_grad():
                    pred_correct = (logits.argmax(1) == y).float().unsqueeze(1)
                gate_loss = F.binary_cross_entropy(gate, pred_correct)

                loss = seg_loss + args.gate_weight * gate_loss

                # Accumulate prototypes
                with torch.no_grad():
                    adapter.update_prototypes(features, y, args.task_id, out_channels)
            else:
                logits = adapter(features, (D, H, W))
                loss = _dicece_loss(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item()
            steps += 1
            pbar.set_postfix(loss=loss.item())

        avg_train_loss = train_loss_sum / max(steps, 1)

        # Validation
        adapter.eval()
        val_dice_sum = 0.0
        val_steps = 0
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= max_steps:
                    break
                x = batch["image"].to(device)
                y = batch["label"].to(device)
                features = backend._extract_features_3d(x)
                B, C_in, D, H, W = x.shape
                if is_gated:
                    logits, _ = adapter(features, (D, H, W))
                else:
                    logits = adapter(features, (D, H, W))
                pred = logits.argmax(dim=1)

                # Quick per-class dice
                dice_scores = []
                for c in range(1, out_channels):
                    pred_c = (pred == c).float()
                    target_c = (y == c).float()
                    intersection = (pred_c * target_c).sum()
                    denom = pred_c.sum() + target_c.sum()
                    if denom > 0:
                        dice_scores.append((2.0 * intersection / denom).item())
                if dice_scores:
                    val_dice_sum += sum(dice_scores) / len(dice_scores)
                val_steps += 1

        avg_val_dice = val_dice_sum / max(val_steps, 1)
        metrics_rows.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_dice_mean": avg_val_dice,
        })

        logger.info(
            f"epoch={epoch+1} train_loss={avg_train_loss:.4f} "
            f"val_dice={avg_val_dice:.4f}"
        )

        if avg_val_dice > best_val_dice:
            best_val_dice = avg_val_dice
            save_dict = {
                "out_channels": out_channels,
                "teacher_type": args.teacher_type,
                "adapter_type": args.adapter_type,
                "epoch": epoch + 1,
                "val_dice": avg_val_dice,
            }
            if is_gated:
                save_dict["adapter_state_dict"] = adapter.state_dict_full()
                save_dict["num_prototypes"] = adapter.num_prototypes
            else:
                save_dict["adapter_state_dict"] = adapter.state_dict()
            torch.save(save_dict, output_path)
            logger.info(f"  saved best adapter (dice={avg_val_dice:.4f})")

    # Save metrics CSV
    if metrics_rows:
        import csv
        with metrics_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=metrics_rows[0].keys())
            w.writeheader()
            w.writerows(metrics_rows)

    logger.info(f"Done. Best val dice: {best_val_dice:.4f}")
    logger.info(f"Adapter saved to: {output_path}")
    logger.info(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
