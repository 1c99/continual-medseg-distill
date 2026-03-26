"""Distributed Data Parallel (DDP) utilities for single-node multi-GPU training.

Usage:
    ctx = setup_ddp(cfg)        # init process group
    model = ctx.wrap_model(model)  # wrap in DDP
    sampler = ctx.make_sampler(dataset)
    ...
    cleanup_ddp()

All operations are no-ops when DDP is disabled, so callers don't need
conditional branches.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DistributedContext:
    """Manages DDP state and provides rank-safe helpers."""

    def __init__(
        self,
        enabled: bool = False,
        backend: str = "nccl",
        world_size: int = 1,
        grad_accum_steps: int = 1,
    ):
        self.enabled = enabled
        self.backend = backend
        self.world_size = world_size
        self.grad_accum_steps = max(grad_accum_steps, 1)
        self._rank: int = 0
        self._local_rank: int = 0
        self._initialized: bool = False

        if self.enabled:
            self._init_process_group()

    def _init_process_group(self) -> None:
        self._rank = int(os.environ.get("RANK", 0))
        self._local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world = int(os.environ.get("WORLD_SIZE", self.world_size))
        self.world_size = world

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend=self.backend,
                world_size=self.world_size,
                rank=self._rank,
            )
        self._initialized = True
        logger.info(
            f"DDP initialized: rank={self._rank}, local_rank={self._local_rank}, "
            f"world_size={self.world_size}, backend={self.backend}"
        )

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def local_rank(self) -> int:
        return self._local_rank

    def is_main_process(self) -> bool:
        """True on rank 0 or when DDP is disabled."""
        return not self.enabled or self._rank == 0

    def barrier(self) -> None:
        """Synchronization point across processes."""
        if self.enabled and self._initialized:
            torch.distributed.barrier()

    def reduce_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """All-reduce (mean) a tensor across ranks."""
        if not self.enabled or not self._initialized:
            return tensor
        rt = tensor.clone()
        torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
        rt /= self.world_size
        return rt

    def wrap_model(self, model: nn.Module) -> nn.Module:
        """Wrap model in DDP if enabled."""
        if not self.enabled:
            return model
        device = torch.device(f"cuda:{self._local_rank}")
        model = model.to(device)
        return torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[self._local_rank],
        )

    def make_sampler(self, dataset, shuffle: bool = True):
        """Create a DistributedSampler if DDP enabled, else None."""
        if not self.enabled:
            return None
        from torch.utils.data.distributed import DistributedSampler
        return DistributedSampler(
            dataset, num_replicas=self.world_size, rank=self._rank, shuffle=shuffle,
        )

    def should_accumulate(self, step: int) -> bool:
        """True if this step should skip optimizer.step() (gradient accumulation)."""
        return self.grad_accum_steps > 1 and (step % self.grad_accum_steps != 0)


def setup_ddp(cfg: Dict[str, Any]) -> DistributedContext:
    """Create DistributedContext from config."""
    dist_cfg = cfg.get("runtime", {}).get("distributed", {})
    return DistributedContext(
        enabled=bool(dist_cfg.get("enabled", False)),
        backend=dist_cfg.get("backend", "nccl"),
        world_size=int(dist_cfg.get("world_size", 1)),
        grad_accum_steps=int(dist_cfg.get("grad_accum_steps", 1)),
    )


def cleanup_ddp() -> None:
    """Destroy process group if initialized."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
