from __future__ import annotations

from typing import Dict, Any
import torch


def evaluate(model: torch.nn.Module, val_loader, cfg: Dict[str, Any], logger):
    device = cfg.get("runtime", {}).get("device", "cpu")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for batch in val_loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).float().mean().item()
            total += 1

    metric = correct / max(total, 1)
    logger.info(f"val_voxel_acc={metric:.4f}")
    # TODO: replace with Dice/HD95 and class-wise metrics.
    return {"voxel_acc": metric}
