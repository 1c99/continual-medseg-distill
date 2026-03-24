from __future__ import annotations

from typing import Dict, Any
import torch
from tqdm import tqdm


def train(model: torch.nn.Module, method, train_loader, cfg: Dict[str, Any], logger, dry_run: bool = False):
    tcfg = cfg.get("train", {})
    device = cfg.get("runtime", {}).get("device", "cpu")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.train()

    lr = tcfg.get("lr", 1e-3)
    epochs = tcfg.get("epochs", 1)
    max_steps = tcfg.get("max_steps_per_epoch", 3 if dry_run else 100)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0
    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{epochs}")
        for i, batch in enumerate(pbar):
            if i >= max_steps:
                break
            optimizer.zero_grad()
            loss = method.training_loss(model, batch, device)
            loss.backward()
            optimizer.step()
            global_step += 1
            pbar.set_postfix(loss=float(loss.item()), step=global_step)

        logger.info(f"epoch={epoch+1} done")
        if dry_run:
            logger.info("dry-run enabled; stopping after one epoch")
            break

    method.post_task_update(model)
    return model
