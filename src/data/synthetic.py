from __future__ import annotations

from dataclasses import dataclass
import torch
from torch.utils.data import Dataset


@dataclass
class Synthetic3DSpec:
    samples: int = 16
    channels: int = 1
    num_classes: int = 3
    shape: tuple[int, int, int] = (32, 32, 32)


class SyntheticSeg3DDataset(Dataset):
    """Tiny synthetic 3D segmentation dataset for smoke tests.

    TODO: replace with real dataset adapters + deterministic cache.
    """

    def __init__(self, spec: Synthetic3DSpec):
        self.spec = spec

    def __len__(self) -> int:
        return self.spec.samples

    def __getitem__(self, idx: int):
        x = torch.randn(self.spec.channels, *self.spec.shape)
        y = torch.randint(0, self.spec.num_classes, self.spec.shape)
        return {"image": x.float(), "label": y.long(), "id": idx}
