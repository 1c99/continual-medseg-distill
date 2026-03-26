from __future__ import annotations

from pathlib import Path
from typing import Sequence

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


class ACDCDataset(Dataset):
    """Scaffold-level ACDC adapter (local-path only).

    Expects image/label pairs like:
      - patientXXX/frameYY.nii.gz
      - patientXXX/frameYY_gt.nii.gz

    ``split_ids`` entries should be sample IDs of the form ``patientXXX_frameYY``.
    """

    def __init__(
        self,
        root: str,
        split_ids: Sequence[str],
        target_shape: tuple[int, int, int] = (16, 160, 160),
        normalize: bool = True,
    ):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(
                f"ACDC root directory not found: {self.root}. "
                "Set data.acdc.root to a valid local path."
            )

        self.target_shape = target_shape
        self.normalize = normalize

        self._pairs = self._index_pairs()
        self.ids = [str(x) for x in split_ids]
        if not self.ids:
            raise ValueError("ACDC split_ids is empty. Provide at least one sample id.")

        missing = [sid for sid in self.ids if sid not in self._pairs]
        if missing:
            preview = ", ".join(missing[:5])
            raise FileNotFoundError(
                "Some ACDC split IDs are not available under the provided root. "
                f"Missing ({len(missing)}): {preview}"
            )

    def _index_pairs(self) -> dict[str, tuple[Path, Path]]:
        pairs: dict[str, tuple[Path, Path]] = {}

        for gt_path in sorted(self.root.rglob("*_gt.nii.gz")):
            image_path = gt_path.with_name(gt_path.name.replace("_gt.nii.gz", ".nii.gz"))
            if not image_path.exists():
                raise FileNotFoundError(
                    f"Found ACDC label but corresponding image is missing: {gt_path}"
                )

            patient = gt_path.parent.name
            frame = image_path.stem.replace(".nii", "")  # handle .nii.gz stems
            sample_id = f"{patient}_{frame}"
            pairs[sample_id] = (image_path, gt_path)

        if not pairs:
            raise FileNotFoundError(
                f"No ACDC *_gt.nii.gz files found under {self.root}. "
                "Expected local extracted dataset structure with frameXX_gt.nii.gz labels."
            )

        return pairs

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

    def __getitem__(self, idx: int):
        sid = self.ids[idx]
        image_path, label_path = self._pairs[sid]

        image = self._load_nii(image_path)
        label = self._load_nii(label_path)

        if image.ndim != 3:
            raise ValueError(
                f"Expected 3D image for {sid}, got shape {image.shape}"
            )
        if label.ndim != 3:
            raise ValueError(
                f"Expected 3D label for {sid}, got shape {label.shape}"
            )

        image = self._crop_or_pad(image)
        label = self._crop_or_pad(label)

        if self.normalize:
            image = self._zscore(image)

        label_int = label.astype(np.int64)
        # ACDC classes: 0=BG, 1=RV, 2=MYO, 3=LV
        unexpected = set(np.unique(label_int).tolist()) - {0, 1, 2, 3}
        if unexpected:
            raise ValueError(
                f"Unexpected label values {unexpected} in {label_path}. "
                "ACDC expects classes {0=BG, 1=RV, 2=MYO, 3=LV}."
            )

        x = torch.from_numpy(image[None, ...]).float()  # C,Z,Y,X
        y = torch.from_numpy(label_int).long()  # Z,Y,X

        return {"image": x, "label": y, "id": sid}
