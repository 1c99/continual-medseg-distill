"""Orthogonality regularizer for LoRA adapters in continual learning.

Penalizes subspace overlap between the current task's LoRA adapters and
those saved from previous tasks, pushing each new task to learn in an
orthogonal subspace and reducing catastrophic forgetting.

The loss for each LoRA layer is:

    L = sum_{prev} ( ||A_curr^T @ A_prev||_F^2 + ||B_curr @ B_prev^T||_F^2 )

where A ∈ R^{rank × c_in} and B ∈ R^{c_out × rank}.
"""
from __future__ import annotations

import logging
from typing import Dict, List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def orthogonality_loss(
    model: nn.Module,
    prev_lora_states: List[Dict[str, torch.Tensor]],
) -> torch.Tensor:
    """Compute orthogonality penalty between current and previous LoRA adapters.

    Args:
        model: Current model with LoRA adapters (contains lora_A/lora_B params).
        prev_lora_states: List of saved LoRA state dicts from previous tasks
            (as returned by ``extract_lora_state``).

    Returns:
        Scalar loss tensor (0.0 if no previous states or no LoRA params).
    """
    if not prev_lora_states:
        # First task or no previous adapters — nothing to be orthogonal to
        device = next(model.parameters()).device
        return torch.tensor(0.0, device=device, requires_grad=False)

    # Collect current LoRA A/B params keyed by layer prefix
    current_params: Dict[str, Dict[str, torch.Tensor]] = {}
    for name, param in model.named_parameters():
        if "lora_A" in name and param.requires_grad:
            prefix = name.rsplit(".lora_A", 1)[0]
            current_params.setdefault(prefix, {})["A"] = param
        elif "lora_B" in name and param.requires_grad:
            prefix = name.rsplit(".lora_B", 1)[0]
            current_params.setdefault(prefix, {})["B"] = param

    if not current_params:
        device = next(model.parameters()).device
        return torch.tensor(0.0, device=device, requires_grad=False)

    device = next(iter(next(iter(current_params.values())).values())).device
    loss = torch.tensor(0.0, device=device)
    count = 0

    for prev_state in prev_lora_states:
        # Group previous state by layer prefix
        prev_by_prefix: Dict[str, Dict[str, torch.Tensor]] = {}
        for key, value in prev_state.items():
            if "lora_A" in key:
                prefix = key.rsplit(".lora_A", 1)[0]
                prev_by_prefix.setdefault(prefix, {})["A"] = value
            elif "lora_B" in key:
                prefix = key.rsplit(".lora_B", 1)[0]
                prev_by_prefix.setdefault(prefix, {})["B"] = value

        for prefix, curr in current_params.items():
            prev = prev_by_prefix.get(prefix)
            if prev is None:
                continue

            # A matrices: squeeze spatial dims → (rank, c_in)
            if "A" in curr and "A" in prev:
                A_curr = curr["A"].flatten(1)  # (rank, c_in)
                A_prev = prev["A"].to(device).flatten(1)  # (rank_prev, c_in)
                # ||A_curr^T @ A_prev||_F^2 — cross-correlation of row spaces
                cross = A_curr @ A_prev.t()  # (rank, rank_prev)
                loss = loss + cross.pow(2).sum()
                count += 1

            # B matrices: squeeze spatial dims → (c_out, rank)
            if "B" in curr and "B" in prev:
                B_curr = curr["B"].flatten(1)  # (c_out, rank)
                B_prev = prev["B"].to(device).flatten(1)  # (c_out, rank_prev)
                # ||B_curr @ B_prev^T||_F^2 — cross-correlation of column spaces
                # Actually: B is (c_out, rank), so B^T is (rank, c_out)
                # We want overlap in rank-subspace: B_curr^T @ B_prev
                cross = B_curr.t() @ B_prev  # (rank, rank_prev)
                loss = loss + cross.pow(2).sum()
                count += 1

    if count > 0:
        loss = loss / count

    return loss
