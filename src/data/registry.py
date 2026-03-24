from __future__ import annotations

from typing import Dict, Tuple
from torch.utils.data import DataLoader

from .synthetic import Synthetic3DSpec, SyntheticSeg3DDataset


def create_loaders(cfg: Dict) -> Tuple[DataLoader, DataLoader]:
    data_cfg = cfg.get("data", {})
    train_bs = data_cfg.get("batch_size", 2)
    val_bs = data_cfg.get("val_batch_size", train_bs)

    source = data_cfg.get("source", "synthetic")
    if source == "synthetic":
        synth_cfg = data_cfg.get("synthetic", {})
        train_ds = SyntheticSeg3DDataset(
            Synthetic3DSpec(
                samples=synth_cfg.get("train_samples", 16),
                channels=synth_cfg.get("channels", 1),
                num_classes=synth_cfg.get("num_classes", 3),
                shape=tuple(synth_cfg.get("shape", [32, 32, 32])),
            )
        )
        val_ds = SyntheticSeg3DDataset(
            Synthetic3DSpec(
                samples=synth_cfg.get("val_samples", 8),
                channels=synth_cfg.get("channels", 1),
                num_classes=synth_cfg.get("num_classes", 3),
                shape=tuple(synth_cfg.get("shape", [32, 32, 32])),
            )
        )
    else:
        # TODO: implement real dataset registry adapters (MSD/BTCV/KiTS/AMOS/CHAOS/BraTS)
        raise NotImplementedError(
            f"Dataset source '{source}' is not implemented yet. "
            "Use data.source=synthetic for now."
        )

    train_loader = DataLoader(train_ds, batch_size=train_bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=val_bs, shuffle=False)
    return train_loader, val_loader


def build_dataloader(cfg: Dict, split: str = "train") -> DataLoader:
    """Compatibility helper used by scripts."""
    train_loader, val_loader = create_loaders(cfg)
    return train_loader if split == "train" else val_loader
