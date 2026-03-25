from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import yaml
from torch.utils.data import DataLoader

from .acdc import ACDCDataset
from .brats21 import Brats21Dataset
from .synthetic import Synthetic3DSpec, SyntheticSeg3DDataset
from .totalseg import TotalSegmentatorDataset


def _coerce_id_list(values: Iterable) -> list[str]:
    """Normalize split entries to a list of subject IDs.

    Supports either:
    - ["s0001", "s0002", ...]
    - [{"id": "s0001"}, {"subject_id": "s0002"}, ...]
    """
    ids: list[str] = []
    for item in values:
        if isinstance(item, str):
            ids.append(item)
        elif isinstance(item, dict):
            subject_id = item.get("id") or item.get("subject_id")
            if subject_id:
                ids.append(str(subject_id))
            else:
                raise ValueError(
                    "Split manifest entries as objects must include 'id' or 'subject_id' fields."
                )
        else:
            raise ValueError(f"Unsupported split entry type: {type(item)!r}")
    return ids


def _load_ids_from_split_manifest(split_manifest: str | Path) -> tuple[list[str], list[str]]:
    path = Path(split_manifest)
    if not path.exists():
        raise FileNotFoundError(f"Split manifest not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    elif suffix in {".yaml", ".yml"}:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        raise ValueError(
            f"Unsupported split manifest extension '{path.suffix}'. Use .json, .yaml, or .yml."
        )

    if not isinstance(payload, dict):
        raise ValueError("Split manifest root must be a mapping/object.")

    train_values = payload.get("train")
    val_values = payload.get("val")

    # Compatibility with existing key style
    if train_values is None:
        train_values = payload.get("train_ids")
    if val_values is None:
        val_values = payload.get("val_ids")

    if train_values is None or val_values is None:
        raise ValueError(
            "Split manifest must include both train/val (or train_ids/val_ids) entries."
        )

    train_ids = _coerce_id_list(train_values)
    val_ids = _coerce_id_list(val_values)

    if not train_ids or not val_ids:
        raise ValueError("Split manifest train/val lists must both be non-empty.")

    return train_ids, val_ids


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
    elif source == "totalseg":
        tcfg = data_cfg.get("totalseg", {})
        root = tcfg.get("root")
        if not root:
            raise ValueError("data.totalseg.root is required for source=totalseg")

        split_manifest = tcfg.get("split_manifest")
        if split_manifest:
            train_ids, val_ids = _load_ids_from_split_manifest(split_manifest)
        else:
            train_ids = tcfg.get("train_ids", [])
            val_ids = tcfg.get("val_ids", [])

        if not train_ids or not val_ids:
            raise ValueError(
                "Provide either data.totalseg.split_manifest or data.totalseg.train_ids and val_ids in config"
            )

        organ = tcfg.get("organ", "liver")
        shape = tuple(tcfg.get("shape", [128, 128, 128]))

        train_ds = TotalSegmentatorDataset(root=root, split_ids=train_ids, organ=organ, target_shape=shape)
        val_ds = TotalSegmentatorDataset(root=root, split_ids=val_ids, organ=organ, target_shape=shape)
    elif source == "brats21":
        bcfg = data_cfg.get("brats21", {})
        root = bcfg.get("root")
        if not root:
            raise ValueError("data.brats21.root is required for source=brats21")

        split_manifest = bcfg.get("split_manifest")
        if split_manifest:
            train_ids, val_ids = _load_ids_from_split_manifest(split_manifest)
        else:
            train_ids = bcfg.get("train_ids", [])
            val_ids = bcfg.get("val_ids", [])

        if not train_ids or not val_ids:
            raise ValueError(
                "Provide either data.brats21.split_manifest or data.brats21.train_ids and val_ids in config"
            )

        shape = tuple(bcfg.get("shape", [128, 128, 128]))
        normalize_per_channel = bool(bcfg.get("normalize_per_channel", True))

        train_ds = Brats21Dataset(
            root=root,
            split_ids=train_ids,
            target_shape=shape,
            normalize_per_channel=normalize_per_channel,
        )
        val_ds = Brats21Dataset(
            root=root,
            split_ids=val_ids,
            target_shape=shape,
            normalize_per_channel=normalize_per_channel,
        )
    elif source == "acdc":
        acfg = data_cfg.get("acdc", {})
        root = acfg.get("root")
        if not root:
            raise ValueError("data.acdc.root is required for source=acdc")

        split_manifest = acfg.get("split_manifest")
        if split_manifest:
            train_ids, val_ids = _load_ids_from_split_manifest(split_manifest)
        else:
            train_ids = acfg.get("train_ids", [])
            val_ids = acfg.get("val_ids", [])

        if not train_ids or not val_ids:
            raise ValueError(
                "Provide either data.acdc.split_manifest or data.acdc.train_ids and val_ids in config"
            )

        shape = tuple(acfg.get("shape", [16, 160, 160]))
        normalize = bool(acfg.get("normalize", True))

        train_ds = ACDCDataset(root=root, split_ids=train_ids, target_shape=shape, normalize=normalize)
        val_ds = ACDCDataset(root=root, split_ids=val_ids, target_shape=shape, normalize=normalize)
    else:
        raise NotImplementedError(
            f"Dataset source '{source}' is not implemented yet. "
            "Use data.source=synthetic, totalseg, brats21, or acdc."
        )

    train_loader = DataLoader(train_ds, batch_size=train_bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=val_bs, shuffle=False)
    return train_loader, val_loader


def build_dataloader(cfg: Dict, split: str = "train") -> DataLoader:
    """Compatibility helper used by scripts."""
    train_loader, val_loader = create_loaders(cfg)
    return train_loader if split == "train" else val_loader
