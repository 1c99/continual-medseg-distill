"""Patch-based 3D volume sampling for memory-efficient training.

Extracts fixed-size patches from large 3D volumes and reconstructs
full volumes from patch predictions with overlap averaging.
"""
from __future__ import annotations

from typing import List, Tuple

import torch


def compute_patch_coords(
    volume_shape: Tuple[int, int, int],
    patch_size: Tuple[int, int, int],
    stride: Tuple[int, int, int] | None = None,
) -> List[Tuple[int, int, int]]:
    """Compute top-left corner coordinates for patches covering the volume.

    Patches at boundaries are shifted inward so no padding is needed.
    """
    if stride is None:
        stride = patch_size
    coords = []
    for d in range(3):
        vs, ps, st = volume_shape[d], patch_size[d], stride[d]
        starts = list(range(0, vs - ps + 1, st))
        # Ensure last patch covers the boundary
        if not starts or starts[-1] + ps < vs:
            starts.append(max(vs - ps, 0))
        if d == 0:
            coords = [(s,) for s in starts]
        else:
            coords = [c + (s,) for c in coords for s in starts]
    return coords


def extract_patch(
    volume: torch.Tensor,
    coord: Tuple[int, int, int],
    patch_size: Tuple[int, int, int],
) -> torch.Tensor:
    """Extract a single patch from a volume.

    Args:
        volume: (C, D, H, W) or (D, H, W) tensor.
        coord: (d, h, w) top-left corner.
        patch_size: (pd, ph, pw) patch dimensions.

    Returns:
        Patch tensor with same leading dims.
    """
    d, h, w = coord
    pd, ph, pw = patch_size
    if volume.ndim == 4:
        return volume[:, d:d+pd, h:h+ph, w:w+pw].clone()
    return volume[d:d+pd, h:h+ph, w:w+pw].clone()


def reconstruct_volume(
    patches: List[torch.Tensor],
    coords: List[Tuple[int, int, int]],
    volume_shape: Tuple[int, int, int],
    num_channels: int = 1,
) -> torch.Tensor:
    """Reconstruct a volume from overlapping patches via averaging.

    Args:
        patches: List of (C, pd, ph, pw) tensors.
        coords: Matching list of (d, h, w) coordinates.
        volume_shape: (D, H, W) output shape.
        num_channels: number of channels in output.

    Returns:
        (C, D, H, W) reconstructed volume.
    """
    D, H, W = volume_shape
    output = torch.zeros(num_channels, D, H, W, dtype=patches[0].dtype)
    counts = torch.zeros(1, D, H, W, dtype=torch.float32)

    for patch, (d, h, w) in zip(patches, coords):
        pd, ph, pw = patch.shape[-3], patch.shape[-2], patch.shape[-1]
        if patch.ndim == 3:
            patch = patch.unsqueeze(0)
        output[:, d:d+pd, h:h+ph, w:w+pw] += patch
        counts[:, d:d+pd, h:h+ph, w:w+pw] += 1.0

    counts = counts.clamp(min=1.0)
    return output / counts
