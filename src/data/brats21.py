from __future__ import annotations

from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


class Brats21Dataset(Dataset):
    """BraTS 2021 dataset adapter (local-path only).

    Supports two on-disk layouts via the ``layout`` parameter:

    ``per_case`` (default):
      ``root/<case_id>/<case_id>_t1.nii.gz``, etc.

    ``flat`` (nnUNet-style, e.g. BraTS21 challenge download):
      - ``root/imagesTr/<case_id>_t1_0000.nii.gz``   (t1)
      - ``root/imagesTr/<case_id>_t1ce_0000.nii.gz``  (t1ce)
      - ``root/etc/images_flair/<case_id>_0000.nii.gz`` (flair)
      - ``root/etc/images_t2/<case_id>_0000.nii.gz``    (t2)
      - ``root/etc/labels_4cls/<case_id>_seg.nii.gz``   (seg)

    Returns dict with keys: image (C,Z,Y,X), label (Z,Y,X), id.
    """

    MODALITIES = ("t1", "t1ce", "t2", "flair")

    def __init__(
        self,
        root: str,
        split_ids: Sequence[str],
        target_shape: tuple[int, int, int] = (128, 128, 128),
        normalize_per_channel: bool = True,
        layout: str = "per_case",
    ):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(
                f"BraTS root directory not found: {self.root}. "
                "Set data.brats21.root to a valid local path."
            )

        if layout not in ("per_case", "flat"):
            raise ValueError(f"Unsupported BraTS layout '{layout}'. Use 'per_case' or 'flat'.")
        self.layout = layout

        self.ids = [str(x) for x in split_ids]
        if not self.ids:
            raise ValueError("BraTS split_ids is empty. Provide at least one case id.")

        self.target_shape = target_shape
        self.normalize_per_channel = normalize_per_channel

    @classmethod
    def validate_subject(
        cls, root: str, sid: str, layout: str = "per_case"
    ) -> dict:
        """Check that a subject has all expected files and return diagnostics."""
        root_path = Path(root)
        result = {"id": sid, "valid": True, "errors": []}
        if layout == "flat":
            files = {
                "t1": root_path / "imagesTr" / f"{sid}_t1_0000.nii.gz",
                "t1ce": root_path / "imagesTr" / f"{sid}_t1ce_0000.nii.gz",
                "flair": root_path / "etc" / "images_flair" / f"{sid}_0000.nii.gz",
                "t2": root_path / "etc" / "images_t2" / f"{sid}_0000.nii.gz",
                "seg": root_path / "etc" / "labels_4cls" / f"{sid}_seg.nii.gz",
            }
        else:
            case_dir = root_path / sid
            files = {m: case_dir / f"{sid}_{m}.nii.gz" for m in cls.MODALITIES}
            files["seg"] = case_dir / f"{sid}_seg.nii.gz"
        for key, path in files.items():
            if not path.exists():
                result["valid"] = False
                result["errors"].append(f"Missing {key}: {path}")
        return result

    def __len__(self) -> int:
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

    @staticmethod
    def _zscore(v: np.ndarray) -> np.ndarray:
        return (v - v.mean()) / (v.std() + 1e-6)

    def _resolve_paths(self, sid: str):
        """Return (modality_paths_dict, seg_path) for the given subject ID."""
        if self.layout == "per_case":
            case_dir = self.root / sid
            mod_paths = {m: case_dir / f"{sid}_{m}.nii.gz" for m in self.MODALITIES}
            seg_path = case_dir / f"{sid}_seg.nii.gz"
        else:  # flat
            mod_paths = {
                "t1": self.root / "imagesTr" / f"{sid}_t1_0000.nii.gz",
                "t1ce": self.root / "imagesTr" / f"{sid}_t1ce_0000.nii.gz",
                "flair": self.root / "etc" / "images_flair" / f"{sid}_0000.nii.gz",
                "t2": self.root / "etc" / "images_t2" / f"{sid}_0000.nii.gz",
            }
            seg_path = self.root / "etc" / "labels_4cls" / f"{sid}_seg.nii.gz"
        return mod_paths, seg_path

    def __getitem__(self, idx: int):
        sid = self.ids[idx]
        mod_paths, seg_path = self._resolve_paths(sid)

        channels: list[np.ndarray] = []
        for m in self.MODALITIES:
            p = mod_paths[m]
            vol = self._load_nii(p)
            if vol.ndim != 3:
                raise ValueError(
                    f"Expected 3D volume for {sid} modality {m}, got shape {vol.shape}"
                )
            vol = self._crop_or_pad(vol)
            if self.normalize_per_channel:
                vol = self._zscore(vol)
            channels.append(vol)

        seg = self._load_nii(seg_path)
        if seg.ndim != 3:
            raise ValueError(
                f"Expected 3D segmentation for {sid}, got shape {seg.shape}"
            )
        seg = self._crop_or_pad(seg)

        # BraTS labels can be {0,1,2,4}; remap 4->3 for contiguous class ids.
        seg = seg.astype(np.int64)
        seg[seg == 4] = 3
        unexpected = set(np.unique(seg).tolist()) - {0, 1, 2, 3}
        if unexpected:
            raise ValueError(
                f"Unexpected label values {unexpected} in {seg_path} after remap. "
                "Expected only {0, 1, 2, 3}."
            )

        x = torch.from_numpy(np.stack(channels, axis=0)).float()  # C,Z,Y,X
        y = torch.from_numpy(seg).long()  # Z,Y,X

        return {"image": x, "label": y, "id": sid}
