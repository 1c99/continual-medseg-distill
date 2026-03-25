from __future__ import annotations

from pathlib import Path
from typing import Sequence
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


class TotalSegmentatorDataset(Dataset):
    """Minimal TotalSegmentator loader (scaffold).

    Current mode:
    - single-organ binary segmentation
    - center crop or pad to target shape

    TODO:
    - multi-organ multi-class mapping
    - spacing-aware resampling and normalization policy
    """

    def __init__(
        self,
        root: str,
        split_ids: Sequence[str],
        organ: str = "liver",
        target_shape: tuple[int, int, int] = (128, 128, 128),
    ):
        self.root = Path(root)
        self.ids = list(split_ids)
        self.organ = organ
        self.target_shape = target_shape

    def __len__(self):
        return len(self.ids)

    def _load_nii(self, path: Path) -> np.ndarray:
        return np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)

    def _crop_or_pad(self, vol: np.ndarray) -> np.ndarray:
        z, y, x = vol.shape
        tz, ty, tx = self.target_shape

        out = np.zeros((tz, ty, tx), dtype=vol.dtype)
        zs = max((z - tz) // 2, 0)
        ys = max((y - ty) // 2, 0)
        xs = max((x - tx) // 2, 0)

        zc = min(z, tz)
        yc = min(y, ty)
        xc = min(x, tx)

        out_zs = max((tz - zc) // 2, 0)
        out_ys = max((ty - yc) // 2, 0)
        out_xs = max((tx - xc) // 2, 0)

        out[out_zs:out_zs + zc, out_ys:out_ys + yc, out_xs:out_xs + xc] = vol[
            zs:zs + zc, ys:ys + yc, xs:xs + xc
        ]
        return out

    def __getitem__(self, idx: int):
        sid = self.ids[idx]
        sp = self.root / sid
        ct = self._load_nii(sp / "ct.nii.gz")
        organ_mask = self._load_nii(sp / "segmentations" / f"{self.organ}.nii.gz")

        ct = self._crop_or_pad(ct)
        organ_mask = self._crop_or_pad((organ_mask > 0).astype(np.float32))

        # simple z-score normalization
        ct = (ct - ct.mean()) / (ct.std() + 1e-6)

        x = torch.from_numpy(ct[None, ...]).float()  # C,Z,Y,X
        y = torch.from_numpy(organ_mask).long()      # Z,Y,X binary class indices

        return {"image": x, "label": y, "id": sid}
