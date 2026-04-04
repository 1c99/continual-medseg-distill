"""Distillation method with multiple KD modes.

Supported ``method.kd.mode`` values (config-toggleable):
    logit   — KL-divergence on softened output logits (default, baseline)
    feature — MSE on intermediate representations + logit KD
    weighted — uncertainty-weighted logit KD (down-weights uncertain teacher voxels)
    boundary — boundary-aware logit KD (up-weights voxels near class boundaries)

All modes are dataset-agnostic and fully config-driven.
"""
from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ContinualMethod
from .teacher import Teacher
from .teacher_cache import TeacherCache

logger = logging.getLogger(__name__)

VALID_KD_MODES = {"logit", "feature", "weighted", "boundary"}


class DistillMethod(ContinualMethod):
    """Multi-mode distillation method.

    Config (under ``method``):
        kd:
            mode: logit | feature | weighted | boundary  (default: logit)
            weight: float  (default: 1.0)
            temperature: float  (default: 2.0)
            feature_weight: float  (feature mode, weight for feature loss, default: 1.0)
            boundary_sigma: float  (boundary mode, Gaussian sigma for edge weighting, default: 1.0)
            teacher:
                type: snapshot | checkpoint
                ckpt_path: str  (required if type=checkpoint)
                use_features: bool  (required for feature mode)
                feature_layers: list[str]  (layer name prefixes to hook)
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        kd_cfg = cfg.get("method", {}).get("kd", {})
        self.kd_weight = float(kd_cfg.get("weight", 1.0))
        self.temperature = float(kd_cfg.get("temperature", 2.0))
        self.kd_mode = kd_cfg.get("mode", "logit")
        self.feature_weight = float(kd_cfg.get("feature_weight", 1.0))
        self.boundary_sigma = float(kd_cfg.get("boundary_sigma", 1.0))

        # Per-step loss component tracking (read by trainer for logging)
        self._last_loss_seg = 0.0
        self._last_loss_kd = 0.0

        teacher_cfg = kd_cfg.get("teacher", {})
        # Auto-enable feature hooks if feature mode and not explicitly set
        if self.kd_mode == "feature" and "use_features" not in teacher_cfg:
            teacher_cfg = {**teacher_cfg, "use_features": True}

        self.teacher = Teacher(teacher_cfg=teacher_cfg, global_cfg=cfg)

        # Teacher cache
        cache_cfg = kd_cfg.get("cache", {})
        self._cache_enabled = bool(cache_cfg.get("enabled", False))
        self._cache: TeacherCache | None = None
        if self._cache_enabled:
            from src.utils.config_validation import compute_config_hash
            cache_dir = cache_cfg.get("dir", "outputs/.teacher_cache")
            cfg_hash = compute_config_hash(kd_cfg)
            self._cache = TeacherCache(cache_dir, cfg_hash)
            logger.info(f"Teacher cache: ON (dir={cache_dir}, hash={cfg_hash})")
        else:
            logger.debug("Teacher cache: OFF")

        # Student feature hooks (only for feature mode)
        self._student_hooks: List[torch.utils.hooks.RemovableHook] = []
        self._student_features: Dict[str, torch.Tensor] = {}

    @property
    def teacher_model(self) -> nn.Module | None:
        """Backward-compatible access to teacher model."""
        return self.teacher.model

    def _validate_config(self) -> None:
        mcfg = self.cfg.get("method", {})
        if "kd" not in mcfg:
            logger.warning(
                "DistillMethod: missing method.kd config section; "
                "using defaults (mode=logit, weight=1.0, temperature=2.0)"
            )
        kd_cfg = mcfg.get("kd", {})
        mode = kd_cfg.get("mode", "logit")
        if mode not in VALID_KD_MODES:
            raise ValueError(
                f"Invalid method.kd.mode='{mode}'. Must be one of {VALID_KD_MODES}"
            )
        if "weight" not in kd_cfg:
            logger.warning("DistillMethod: method.kd.weight not set; defaulting to 1.0")
        if "temperature" not in kd_cfg:
            logger.warning("DistillMethod: method.kd.temperature not set; defaulting to 2.0")

        if mode == "feature":
            teacher_cfg = kd_cfg.get("teacher", {})
            if not teacher_cfg.get("feature_layers"):
                logger.warning(
                    "DistillMethod: kd.mode=feature but teacher.feature_layers is empty. "
                    "Feature KD will fall back to logit-only unless layers are specified."
                )
        if mode == "checkpoint":
            teacher_cfg = kd_cfg.get("teacher", {})
            if not teacher_cfg.get("ckpt_path"):
                raise ValueError(
                    "method.kd.teacher.ckpt_path is required when teacher.type=checkpoint"
                )

    # ---- student feature hooks (for feature KD) ----

    def _register_student_hooks(self, model: nn.Module) -> None:
        self._remove_student_hooks()
        self._student_features = {}
        feature_layers = self.teacher._feature_layers
        for name, module in model.named_modules():
            if any(name.startswith(prefix) for prefix in feature_layers):
                hook = module.register_forward_hook(self._make_student_hook(name))
                self._student_hooks.append(hook)

    def _make_student_hook(self, name: str):
        def hook_fn(module, input, output):
            self._student_features[name] = output
        return hook_fn

    def _remove_student_hooks(self) -> None:
        for h in self._student_hooks:
            h.remove()
        self._student_hooks.clear()
        self._student_features.clear()

    # ---- KD loss computation ----

    def _match_channels(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Align channel dimensions between student and teacher logits.

        When teacher and student have different output channels (e.g., different
        tasks in multi-head setup), uses the shared (minimum) channels for KD.
        """
        s_ch = student_logits.shape[1]
        t_ch = teacher_logits.shape[1]
        if s_ch != t_ch:
            min_ch = min(s_ch, t_ch)
            logger.debug(
                f"KD channel mismatch: student={s_ch}, teacher={t_ch}; "
                f"using first {min_ch} channels"
            )
            student_logits = student_logits[:, :min_ch]
            teacher_logits = teacher_logits[:, :min_ch]
        return student_logits, teacher_logits

    def _logit_kd_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Standard logit KD via KL divergence with temperature scaling.

        If gate is provided (from CGAD), KD is modulated per-voxel.
        """
        student_logits, teacher_logits = self._match_channels(
            student_logits, teacher_logits
        )
        T = self.temperature
        if gate is not None:
            # Gated KD: per-voxel weighting
            kl_per_voxel = F.kl_div(
                F.log_softmax(student_logits / T, dim=1),
                F.softmax(teacher_logits / T, dim=1),
                reduction="none",
            )
            # gate: (B, 1, D, H, W), kl_per_voxel: (B, C, D, H, W)
            return (gate * kl_per_voxel).mean() * (T * T)
        return F.kl_div(
            F.log_softmax(student_logits / T, dim=1),
            F.softmax(teacher_logits / T, dim=1),
            reduction="batchmean",
        ) * (T * T)

    def _feature_kd_loss(self) -> torch.Tensor:
        """MSE between matched student and teacher intermediate features."""
        if not self.teacher.features or not self._student_features:
            return torch.tensor(0.0)
        loss = torch.tensor(0.0)
        count = 0
        for name in self.teacher.features:
            if name in self._student_features:
                t_feat = self.teacher.features[name]
                s_feat = self._student_features[name]
                # Handle shape mismatch via adaptive pooling
                if s_feat.shape != t_feat.shape:
                    target_shape = t_feat.shape[2:]
                    s_feat = F.adaptive_avg_pool3d(s_feat, target_shape)
                loss = loss.to(s_feat.device) + F.mse_loss(s_feat, t_feat.to(s_feat.device))
                count += 1
        return loss / max(count, 1)

    def _uncertainty_weights(self, teacher_logits: torch.Tensor) -> torch.Tensor:
        """Compute per-voxel confidence weights from teacher predictions.

        High-confidence teacher voxels get higher weight; uncertain voxels
        are down-weighted to avoid distilling noise.
        """
        probs = F.softmax(teacher_logits, dim=1)
        # max probability as confidence measure
        confidence, _ = probs.max(dim=1, keepdim=True)  # (B,1,*spatial)
        # Expand to match logit channel dim for broadcasting
        return confidence

    def _boundary_weights(self, teacher_logits: torch.Tensor) -> torch.Tensor:
        """Compute per-voxel boundary-proximity weights from teacher predictions.

        Voxels near class boundaries get higher weight to focus distillation
        on the most informative regions.
        """
        pred = teacher_logits.argmax(dim=1)  # (B, *spatial)
        # Detect boundaries via discrete Laplacian (voxels where neighbours differ)
        boundary = torch.zeros_like(pred, dtype=torch.float32)
        for dim in range(1, pred.ndim):  # skip batch dim
            shifted_fwd = torch.roll(pred, shifts=1, dims=dim)
            shifted_bwd = torch.roll(pred, shifts=-1, dims=dim)
            boundary += (pred != shifted_fwd).float()
            boundary += (pred != shifted_bwd).float()
        boundary = (boundary > 0).float()

        # Gaussian blur to soften boundary mask
        if self.boundary_sigma > 0 and boundary.ndim >= 4:
            # Simple dilation via 3D avg pooling as approximate Gaussian
            k = max(int(self.boundary_sigma * 2) * 2 + 1, 3)
            boundary_5d = boundary.unsqueeze(1)  # (B,1,*spatial)
            boundary_5d = F.avg_pool3d(
                boundary_5d, kernel_size=k, stride=1, padding=k // 2
            )
            boundary = boundary_5d.squeeze(1)

        # Minimum weight of 1.0, boundary voxels get upweighted
        weights = 1.0 + boundary
        return weights.unsqueeze(1)  # (B,1,*spatial)

    def _compute_kd_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Dispatch to the configured KD loss mode."""
        if self.kd_mode == "logit":
            return self._logit_kd_loss(student_logits, teacher_logits, gate=gate)

        elif self.kd_mode == "feature":
            logit_loss = self._logit_kd_loss(student_logits, teacher_logits, gate=gate)
            feat_loss = self._feature_kd_loss()
            return logit_loss + self.feature_weight * feat_loss

        elif self.kd_mode == "weighted":
            student_logits, teacher_logits = self._match_channels(
                student_logits, teacher_logits
            )
            T = self.temperature
            s_log_probs = F.log_softmax(student_logits / T, dim=1)
            t_probs = F.softmax(teacher_logits / T, dim=1)
            weights = self._uncertainty_weights(teacher_logits)
            # Per-voxel KL, weighted by teacher confidence
            kl_per_voxel = F.kl_div(s_log_probs, t_probs, reduction="none")
            # kl_per_voxel is (B,C,*spatial), weights is (B,1,*spatial)
            weighted_kl = (kl_per_voxel * weights).mean()
            return weighted_kl * (T * T)

        elif self.kd_mode == "boundary":
            student_logits, teacher_logits = self._match_channels(
                student_logits, teacher_logits
            )
            T = self.temperature
            s_log_probs = F.log_softmax(student_logits / T, dim=1)
            t_probs = F.softmax(teacher_logits / T, dim=1)
            weights = self._boundary_weights(teacher_logits)
            kl_per_voxel = F.kl_div(s_log_probs, t_probs, reduction="none")
            weighted_kl = (kl_per_voxel * weights).mean()
            return weighted_kl * (T * T)

        else:
            raise ValueError(f"Unknown KD mode: {self.kd_mode}")

    # ---- training ----

    def training_loss(
        self, model: nn.Module, batch: Dict, device: str
    ) -> torch.Tensor:
        x = batch["image"].to(device)
        y = batch["label"].to(device)

        # Register student hooks if needed for feature KD
        if self.kd_mode == "feature" and self.teacher.has_model:
            self._register_student_hooks(model)

        student_logits = model(x)
        ce = self._compute_loss(student_logits, y)

        if not self.teacher.has_model:
            self._remove_student_hooks()
            return ce

        # --- Teacher cache integration ---
        teacher_logits = None
        sample_ids = batch.get("id", [])
        cache_hit = False

        if self._cache is not None and len(sample_ids) == 1:
            sid = sample_ids[0] if isinstance(sample_ids, list) else str(sample_ids)
            cached = self._cache.get(str(sid))
            if cached is not None:
                teacher_logits = cached["logits"].to(device)
                cache_hit = True

        teacher_gate = None
        if teacher_logits is None:
            self.teacher.to(device)
            teacher_logits, teacher_gate = self.teacher.forward_with_gate(x)
            # Store in cache
            if self._cache is not None and len(sample_ids) == 1:
                sid = sample_ids[0] if isinstance(sample_ids, list) else str(sample_ids)
                self._cache.put(str(sid), teacher_logits)

        kd = self._compute_kd_loss(student_logits, teacher_logits, gate=teacher_gate)

        self._last_loss_seg = float(ce.item())
        self._last_loss_kd = float(kd.item())

        self._remove_student_hooks()
        return ce + self.kd_weight * kd

    def set_task_output_channels(self, out_channels: int, task_id: str | None = None) -> None:
        """Reconfigure teacher adapter for the current task's output channels."""
        if self.teacher.is_external:
            self.teacher.reconfigure_adapter(out_channels, task_id=task_id)

    def pretrain_teacher_for_task(
        self,
        train_loader,
        task_id: str,
        out_channels: int,
        cfg: Dict,
        task_logger,
    ) -> None:
        """Briefly train the new adapter residual on this task's data.

        Delegates to the same logic as DistillReplayEWCMethod. See that
        class's docstring for details.
        """
        if not self.teacher.is_external:
            return
        backend = self.teacher._backend
        adapter = getattr(backend, "_adapter", None)
        if adapter is None or not hasattr(adapter, "residuals"):
            return

        pretrain_cfg = self.cfg.get("method", {}).get("kd", {}).get("adapter_pretrain", {})
        max_steps = int(pretrain_cfg.get("steps", 200))
        lr = float(pretrain_cfg.get("lr", 1e-3))
        if max_steps <= 0 or task_id not in adapter.residuals:
            return

        device = cfg.get("runtime", {}).get("device", "cpu")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        trainable = list(adapter.residuals[task_id].parameters())
        if not trainable:
            return

        optimizer = torch.optim.Adam(trainable, lr=lr)
        backend.to(device)
        adapter.train()
        task_logger.info(
            f"  Pretraining adapter residual for '{task_id}' ({max_steps} steps)"
        )

        steps = 0
        for batch in train_loader:
            if steps >= max_steps:
                break
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            with torch.no_grad():
                features = backend._extract_features_3d(x)
            B, C_in, D, H, W = x.shape
            is_gated = getattr(backend, "_gated", False)
            if is_gated:
                logits, _ = adapter(features, (D, H, W))
            else:
                logits = adapter(features, (D, H, W))
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            steps += 1

        adapter.eval()
        task_logger.info(f"  Adapter residual pretrained: {steps} steps")

    # ---- lifecycle ----

    def post_task_update(self, model: nn.Module, **kwargs) -> None:
        # Log cache stats before invalidation
        if self._cache is not None:
            stats = self._cache.stats
            logger.info(
                f"Teacher cache stats: hits={stats['hits']}, misses={stats['misses']}, "
                f"total={stats['total']}"
            )
            self._cache.invalidate()  # Teacher changed, cache invalid
        if not self.teacher.is_external:
            self.teacher.snapshot(model)

    def save_state(self, path: Path, model_template: nn.Module | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = self.teacher.state_dict()
        state["kd_mode"] = self.kd_mode
        torch.save(state, path)

    def load_state(self, path: Path, model_template: nn.Module | None = None) -> None:
        state = torch.load(Path(path), map_location="cpu", weights_only=False)
        self.teacher.load_state_dict(state, model_template=model_template)
