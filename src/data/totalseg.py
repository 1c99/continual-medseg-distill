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
        if not self.root.exists():
            raise FileNotFoundError(
                f"TotalSegmentator root directory not found: {self.root}. "
                "Set data.totalseg.root to a valid local path."
            )

        self.ids = list(split_ids)
        if not self.ids:
            raise ValueError("TotalSegmentator split_ids is empty. Provide at least one subject id.")

        self.organ = organ
        self.target_shape = target_shape

    @classmethod
    def validate_subject(cls, root: str, sid: str, organ: str = "liver") -> dict:
        """Check that a subject directory has the expected files and return diagnostics."""
        root_path = Path(root)
        sp = root_path / sid
        result = {"id": sid, "valid": True, "errors": []}
        if not sp.exists():
            result["valid"] = False
            result["errors"].append(f"Subject directory not found: {sp}")
            return result
        ct_path = sp / "ct.nii.gz"
        if not ct_path.exists():
            result["valid"] = False
            result["errors"].append(f"Missing: {ct_path}")
        seg_path = sp / "segmentations" / f"{organ}.nii.gz"
        if not seg_path.exists():
            result["valid"] = False
            result["errors"].append(f"Missing: {seg_path}")
        return result

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _load_nii(path: Path) -> np.ndarray:
        if not path.exists():
            raise FileNotFoundError(f"Required NIfTI file is missing: {path}")
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
        if not sp.exists():
            raise FileNotFoundError(
                f"TotalSegmentator subject directory not found: {sp}. "
                "Check split IDs against data.totalseg.root."
            )
        ct = self._load_nii(sp / "ct.nii.gz")
        if ct.ndim != 3:
            raise ValueError(f"Expected 3D CT volume for {sid}, got shape {ct.shape}")
        organ_mask = self._load_nii(sp / "segmentations" / f"{self.organ}.nii.gz")
        if organ_mask.ndim != 3:
            raise ValueError(f"Expected 3D organ mask for {sid}, got shape {organ_mask.shape}")

        ct = self._crop_or_pad(ct)
        organ_mask = self._crop_or_pad((organ_mask > 0).astype(np.float32))

        # simple z-score normalization
        ct = (ct - ct.mean()) / (ct.std() + 1e-6)

        x = torch.from_numpy(ct[None, ...]).float()  # C,Z,Y,X
        y = torch.from_numpy(organ_mask).long()      # Z,Y,X binary class indices

        return {"image": x, "label": y, "id": sid}
