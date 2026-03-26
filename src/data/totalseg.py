from __future__ import annotations

from pathlib import Path
from typing import Sequence
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


class TotalSegmentatorDataset(Dataset):
    """TotalSegmentator loader supporting single-organ binary and multi-organ
    multi-class segmentation.

    Single-organ mode (backward compatible):
        organ="liver"  →  2 classes (0=background, 1=organ)

    Multi-organ mode:
        organs=["liver", "spleen", "pancreas"]
        →  N+1 classes (0=background, 1=liver, 2=spleen, 3=pancreas)

    If both ``organ`` and ``organs`` are provided, ``organs`` takes precedence.
    """

    def __init__(
        self,
        root: str,
        split_ids: Sequence[str],
        organ: str = "liver",
        organs: Sequence[str] | None = None,
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

        # Multi-organ list takes precedence over single organ
        if organs is not None and len(organs) > 0:
            self.organs = list(organs)
        else:
            self.organs = [organ]

        self.num_classes = len(self.organs) + 1  # +1 for background
        self.target_shape = target_shape

        # For backward compat
        self.organ = self.organs[0]

    @classmethod
    def validate_subject(cls, root: str, sid: str, organ: str = "liver",
                         organs: Sequence[str] | None = None) -> dict:
        """Check that a subject directory has the expected files."""
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
        check_organs = list(organs) if organs else [organ]
        for org in check_organs:
            seg_path = sp / "segmentations" / f"{org}.nii.gz"
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

        ct = self._crop_or_pad(ct)

        # Build multi-class label map: 0=background, i=organs[i-1]
        label = np.zeros(self.target_shape, dtype=np.int64)
        for class_idx, org in enumerate(self.organs, start=1):
            mask = self._load_nii(sp / "segmentations" / f"{org}.nii.gz")
            if mask.ndim != 3:
                raise ValueError(f"Expected 3D mask for {sid}/{org}, got shape {mask.shape}")
            mask = self._crop_or_pad((mask > 0).astype(np.float32))
            # Later organs overwrite earlier ones at overlapping voxels
            label[mask > 0.5] = class_idx

        # simple z-score normalization
        ct = (ct - ct.mean()) / (ct.std() + 1e-6)

        x = torch.from_numpy(ct[None, ...]).float()  # C,Z,Y,X
        y = torch.from_numpy(label)                   # Z,Y,X class indices

        return {"image": x, "label": y, "id": sid}
