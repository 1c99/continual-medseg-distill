"""Combined method: distillation + replay + Fisher-based EWC regularization.

Inherits replay buffer from ReplayMethod and adds:
- Teacher-based knowledge distillation (via Teacher abstraction)
- Elastic Weight Consolidation with diagonal Fisher estimation
- Replay-aware KD with teacher head switching (FIX 1)
- Fixed adaptive EWC with reference loss anchoring (FIX 2)
- Prototype KD for replay anti-forgetting (FIX 3)
"""
from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.engine.distributed import unwrap_model
from .replay import ReplayMethod
from .teacher import Teacher

logger = logging.getLogger(__name__)


class DistillReplayEWCMethod(ReplayMethod):
    """Combined scaffold: distillation + replay + Fisher-based EWC regularization.

    Optionally supports Orthogonal LoRA regularization when
    ``model.lora.mode == "orthogonal"``, penalizing subspace overlap between
    the current task's LoRA adapters and those from previous tasks.
    """

    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        mcfg = cfg.get("method", {})
        kd_cfg = mcfg.get("kd", {})
        ewc_cfg = mcfg.get("ewc", {})

        self.kd_weight = float(kd_cfg.get("weight", 1.0))
        self.temperature = float(kd_cfg.get("temperature", 2.0))
        self._current_task_kd = bool(kd_cfg.get("current_task_kd", False))
        self.ewc_weight = float(ewc_cfg.get("weight", 0.1))
        self.fisher_samples = int(ewc_cfg.get("fisher_samples", 64))

        # Adaptive loss scaling: auto-scale EWC and KD relative to task loss
        # When enabled, weight values become target ratios (e.g., 0.5 = half of task loss)
        self._adaptive_scaling = bool(mcfg.get("adaptive_scaling", False))
        self._ewc_target_ratio = float(ewc_cfg.get("target_ratio", 0.5))
        self._kd_target_ratio = float(kd_cfg.get("target_ratio", 0.3))

        # EWC protection scheduler: ramps ratio from low → high over training
        # Like LR scheduler but for protection strength
        ewc_sched = ewc_cfg.get("schedule", {})
        self._ewc_schedule_enabled = bool(ewc_sched.get("enabled", False))
        self._ewc_ratio_start = float(ewc_sched.get("ratio_start", 0.05))
        self._ewc_ratio_end = float(ewc_sched.get("ratio_end", 0.3))
        self._ewc_warmup_epochs = int(ewc_sched.get("warmup_epochs", 10))
        self._ewc_schedule_type = ewc_sched.get("type", "linear")  # linear, cosine
        self._current_task_step = 0
        self._steps_per_epoch = int(mcfg.get("steps_per_epoch_hint", 100))

        # FIX 2: Reference loss for stable adaptive EWC scaling.
        # Anchored at the start of each task so EWC protection does not
        # decay as the task loss decreases during training.
        self._reference_loss: float | None = None

        # Prototype KD config (FIX 3)
        proto_cfg = kd_cfg.get("prototype", {})
        self._proto_kd_weight = float(proto_cfg.get("weight", 0.3))
        # Prototype temperature: higher = softer labels (more class relationship info).
        # Cosine similarity ∈ [-1,1], with T=0.5 → logits ∈ [-2,2] → soft distribution.
        # Too low (0.1) → near-hard labels, losing the benefit of soft KD.
        self._proto_temperature = float(proto_cfg.get("temperature", 0.5))

        # Per-step loss component tracking (read by trainer for logging)
        self._last_loss_seg = 0.0
        self._last_loss_kd = 0.0

        teacher_cfg = kd_cfg.get("teacher", {})
        self.teacher = Teacher(teacher_cfg=teacher_cfg, global_cfg=cfg)

        self.prev_params: Dict[str, torch.Tensor] = {}
        self.fisher: Dict[str, torch.Tensor] = {}

        # Orthogonal LoRA support
        lora_cfg = cfg.get("model", {}).get("lora", {})
        self._lora_enabled = bool(lora_cfg.get("enabled", False))
        self._lora_mode = lora_cfg.get("mode", "standard")
        self._ortho_lambda = float(lora_cfg.get("ortho_lambda", 0.1))
        self.prev_lora_states: list = []

    @property
    def teacher_model(self) -> nn.Module | None:
        """Backward-compatible access."""
        return self.teacher.model

    def _validate_config(self) -> None:
        super()._validate_config()
        mcfg = self.cfg.get("method", {})
        if "kd" not in mcfg:
            logger.warning(
                "DistillReplayEWCMethod: missing method.kd config section; "
                "using defaults (weight=1.0, temperature=2.0)"
            )
        if "ewc" not in mcfg:
            logger.warning(
                "DistillReplayEWCMethod: missing method.ewc config section; "
                "using defaults (weight=0.1)"
            )
        kd_cfg = mcfg.get("kd", {})
        if "weight" not in kd_cfg:
            logger.warning("DistillReplayEWCMethod: method.kd.weight not set; defaulting to 1.0")
        if "temperature" not in kd_cfg:
            logger.warning("DistillReplayEWCMethod: method.kd.temperature not set; defaulting to 2.0")
        ewc_cfg = mcfg.get("ewc", {})
        if "weight" not in ewc_cfg:
            logger.warning("DistillReplayEWCMethod: method.ewc.weight not set; defaulting to 0.1")
        if "fisher_samples" not in ewc_cfg:
            logger.warning("DistillReplayEWCMethod: method.ewc.fisher_samples not set; defaulting to 64")

    def _estimate_fisher(
        self,
        model: nn.Module,
        dataloader,
        device: str,
        n_samples: int = 64,
    ) -> Dict[str, torch.Tensor]:
        fisher = {
            n: torch.zeros_like(p)
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        model.eval()
        count = 0
        for batch in dataloader:
            if count >= n_samples:
                break
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            model.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.detach().pow(2)
            count += 1
        for n in fisher:
            fisher[n] /= max(count, 1)
        model.train()
        return fisher

    def _ewc_penalty(self, model: nn.Module, device) -> torch.Tensor:
        if not self.prev_params or not self.fisher:
            return torch.tensor(0.0, device=device)
        penalty = torch.tensor(0.0, device=device)
        for n, p in model.named_parameters():
            if n in self.prev_params and n in self.fisher:
                penalty += (
                    self.fisher[n].to(device)
                    * (p - self.prev_params[n].to(device)).pow(2)
                ).sum()
        return penalty

    # ------------------------------------------------------------------
    # FIX 1: Replay-aware KD with teacher head switching
    # ------------------------------------------------------------------

    def _get_scheduled_ewc_ratio(self) -> float:
        """Compute EWC ratio based on schedule (low→high over training).

        Like a learning rate scheduler but for protection strength:
          Early epochs: low ratio → model learns new task freely
          Late epochs:  high ratio → protect old task knowledge
        """
        if not self._ewc_schedule_enabled:
            return self._ewc_target_ratio

        start = self._ewc_ratio_start
        end = self._ewc_ratio_end
        warmup = self._ewc_warmup_epochs
        current_epoch = self._current_task_step / max(self._steps_per_epoch, 1)
        t = min(current_epoch / max(warmup, 1), 1.0)

        if self._ewc_schedule_type == "cosine":
            import math
            return start + (end - start) * (1 - math.cos(math.pi * t)) / 2
        else:
            return start + (end - start) * t

    def _compute_replay_kd(
        self,
        model: nn.Module,
        replay: Dict,
        device: str,
    ) -> torch.Tensor:
        """Compute KD loss on replay data, switching the teacher adapter
        to each replay task's head.

        For gated adapters (GRACE), sets ``adapter.current_task`` to the
        replay task's ID before computing teacher logits, then restores.
        """
        if not self.teacher.has_model or not self.teacher.is_external:
            return torch.tensor(0.0, device=device)

        xr = replay["image"].to(device)
        task_ids = replay.get("task_ids", [None] * xr.shape[0])
        unique_tasks = set(t for t in task_ids if t is not None)

        if not unique_tasks:
            return torch.tensor(0.0, device=device)

        backend = self.teacher._backend
        has_gated_adapter = hasattr(backend, "_gated") and backend._gated
        adapter = getattr(backend, "_adapter", None)

        total_kd = torch.tensor(0.0, device=device)
        count = 0

        for task_id in unique_tasks:
            mask = [i for i, t in enumerate(task_ids) if t == task_id]
            xr_task = xr[mask]

            # Switch teacher adapter to replay task
            prev_teacher_task = None
            if has_gated_adapter and adapter is not None:
                prev_teacher_task = adapter.current_task
                if task_id in adapter.residuals:
                    adapter.current_task = task_id
                else:
                    # No adapter for this task — skip
                    continue

            # Switch student head for replay task
            has_multi_head = hasattr(unwrap_model(model), "current_task")
            prev_student_task = None
            if has_multi_head:
                prev_student_task = unwrap_model(model).current_task
                unwrap_model(model).current_task = task_id

            # Compute teacher and student logits
            with torch.no_grad():
                self.teacher.to(device)
                teacher_logits, teacher_gate = self.teacher.forward_with_gate(xr_task)
            student_logits = model(xr_task)

            # Match channels
            s_ch = student_logits.shape[1]
            t_ch = teacher_logits.shape[1]
            if s_ch != t_ch:
                min_ch = min(s_ch, t_ch)
                student_logits = student_logits[:, :min_ch]
                teacher_logits = teacher_logits[:, :min_ch]

            T = self.temperature
            if teacher_gate is not None:
                kl_per_voxel = F.kl_div(
                    F.log_softmax(student_logits / T, dim=1),
                    F.softmax(teacher_logits / T, dim=1),
                    reduction="none",
                )
                kd = (teacher_gate * kl_per_voxel).mean() * (T * T)
            else:
                kd = F.kl_div(
                    F.log_softmax(student_logits / T, dim=1),
                    F.softmax(teacher_logits / T, dim=1),
                    reduction="batchmean",
                ) * (T * T)

            total_kd = total_kd + kd
            count += 1

            # Restore teacher adapter
            if has_gated_adapter and adapter is not None and prev_teacher_task is not None:
                adapter.current_task = prev_teacher_task

            # Restore student head
            if has_multi_head and prev_student_task is not None:
                unwrap_model(model).current_task = prev_student_task

        return total_kd / max(count, 1)

    # ------------------------------------------------------------------
    # FIX 3: Prototype KD for replay anti-forgetting
    # ------------------------------------------------------------------

    def _compute_prototype_kd(
        self,
        model: nn.Module,
        replay: Dict,
        device: str,
    ) -> torch.Tensor:
        """Compute prototype-based KD loss on replay data.

        Uses stored CPA prototypes to generate soft labels for replay
        samples, providing an auxiliary anti-forgetting signal that is
        independent of the teacher's current adapter state.
        """
        if not self.teacher.is_external:
            return torch.tensor(0.0, device=device)

        backend = self.teacher._backend
        adapter = getattr(backend, "_adapter", None)
        if adapter is None or not hasattr(adapter, "prototype_logits"):
            return torch.tensor(0.0, device=device)

        if adapter.num_prototypes == 0:
            return torch.tensor(0.0, device=device)

        xr = replay["image"].to(device)
        task_ids = replay.get("task_ids", [None] * xr.shape[0])
        unique_tasks = set(t for t in task_ids if t is not None)

        if not unique_tasks:
            return torch.tensor(0.0, device=device)

        total_proto_kd = torch.tensor(0.0, device=device)
        count = 0

        for task_id in unique_tasks:
            mask = [i for i, t in enumerate(task_ids) if t == task_id]
            xr_task = xr[mask]

            # Get task's output channels from adapter
            task_channels = adapter._task_channels.get(task_id)
            if task_channels is None:
                continue

            # Extract backbone features for replay samples
            with torch.no_grad():
                self.teacher.to(device)
                features = backend._extract_features_3d(xr_task)

            # Get prototype soft labels
            proto_logits = adapter.prototype_logits(
                features, task_id, task_channels, self._proto_temperature
            )
            if proto_logits is None:
                continue

            # Switch student head for this task
            has_multi_head = hasattr(unwrap_model(model), "current_task")
            prev_student_task = None
            if has_multi_head:
                prev_student_task = unwrap_model(model).current_task
                unwrap_model(model).current_task = task_id

            student_logits = model(xr_task)

            # Restore student head
            if has_multi_head and prev_student_task is not None:
                unwrap_model(model).current_task = prev_student_task

            # Interpolate prototype logits to match student spatial resolution
            if proto_logits.shape[2:] != student_logits.shape[2:]:
                proto_logits = F.interpolate(
                    proto_logits, size=student_logits.shape[2:],
                    mode="trilinear", align_corners=False,
                )

            # Match channels
            s_ch = student_logits.shape[1]
            p_ch = proto_logits.shape[1]
            if s_ch != p_ch:
                min_ch = min(s_ch, p_ch)
                student_logits = student_logits[:, :min_ch]
                proto_logits = proto_logits[:, :min_ch]

            # proto_logits already temperature-scaled in prototype_logits()
            # Use standard KD temperature on student logits for softer distributions
            T = self.temperature
            proto_kd = F.kl_div(
                F.log_softmax(student_logits / T, dim=1),
                F.softmax(proto_logits, dim=1),  # proto_logits already at right scale
                reduction="batchmean",
            ) * (T * T)

            total_proto_kd = total_proto_kd + proto_kd
            count += 1

        return total_proto_kd / max(count, 1)

    # ------------------------------------------------------------------
    # Training loss (combines all components)
    # ------------------------------------------------------------------

    def training_loss(
        self, model: nn.Module, batch: Dict, device: str
    ) -> torch.Tensor:
        # base CE + replay from ReplayMethod
        base_loss = super().training_loss(model, batch, device)

        # Track step count for EWC schedule
        self._current_task_step += 1

        # FIX 2: Anchor reference loss at first step of each task
        if self._reference_loss is None:
            self._reference_loss = base_loss.detach().item()

        # KD on CURRENT TASK data: disabled by default for disjoint tasks
        # (organs->muscles) because the teacher predicts OLD task classes on NEW
        # task data, which is noise. Enable via method.kd.current_task_kd: true
        # for shared-label or overlapping-class scenarios.
        kd = torch.tensor(0.0, device=base_loss.device)
        if self._current_task_kd and self.teacher.has_model:
            x = batch["image"].to(base_loss.device)
            student_logits = model(x)
            self.teacher.to(base_loss.device)
            teacher_logits, teacher_gate = self.teacher.forward_with_gate(x)
            s_ch = student_logits.shape[1]
            t_ch = teacher_logits.shape[1]
            if s_ch != t_ch:
                min_ch = min(s_ch, t_ch)
                student_logits = student_logits[:, :min_ch]
                teacher_logits = teacher_logits[:, :min_ch]
            T = self.temperature
            if teacher_gate is not None:
                kl_per_voxel = F.kl_div(
                    F.log_softmax(student_logits / T, dim=1),
                    F.softmax(teacher_logits / T, dim=1),
                    reduction="none",
                )
                kd = (teacher_gate * kl_per_voxel).mean() * (T * T)
            else:
                kd = F.kl_div(
                    F.log_softmax(student_logits / T, dim=1),
                    F.softmax(teacher_logits / T, dim=1),
                    reduction="batchmean",
                ) * (T * T)

        # FIX 1: KD on REPLAY data with teacher head switching
        replay_kd = torch.tensor(0.0, device=base_loss.device)
        proto_kd = torch.tensor(0.0, device=base_loss.device)
        replay = self._sample_memory(k=batch["image"].shape[0])
        if replay is not None and self.teacher.has_model:
            replay_kd = self._compute_replay_kd(model, replay, str(base_loss.device))
            # FIX 3: Prototype KD on replay data
            proto_kd = self._compute_prototype_kd(model, replay, str(base_loss.device))

        self._last_loss_seg = float(base_loss.item())
        self._last_loss_kd = float(kd.item() + replay_kd.item() + proto_kd.item())

        ewc = self._ewc_penalty(model, base_loss.device)

        # Orthogonal LoRA regularization
        ortho = torch.tensor(0.0, device=base_loss.device)
        if (
            self._lora_enabled
            and self._lora_mode == "orthogonal"
            and self.prev_lora_states
        ):
            from src.models.ortho_reg import orthogonality_loss
            ortho = orthogonality_loss(model, self.prev_lora_states)

        if self._adaptive_scaling:
            # FIX 2: Use reference loss (anchored at task start) instead of
            # the current batch's base_loss for stable adaptive scaling.
            ref_loss = max(self._reference_loss, 1e-6)

            if kd.item() > 0:
                kd_scale = ref_loss / kd.detach().clamp(min=1e-8)
                kd_term = self._kd_target_ratio * kd_scale * kd
            else:
                kd_term = kd

            if ewc.item() > 0:
                ewc_ratio = self._get_scheduled_ewc_ratio()
                ewc_scale = ref_loss / ewc.detach().clamp(min=1e-8)
                ewc_term = ewc_ratio * ewc_scale * ewc
            else:
                ewc_term = ewc

            # Replay KD and prototype KD use the same scaling as current-task KD
            if replay_kd.item() > 0:
                rkd_scale = ref_loss / replay_kd.detach().clamp(min=1e-8)
                replay_kd_term = self._kd_target_ratio * rkd_scale * replay_kd
            else:
                replay_kd_term = replay_kd

            if proto_kd.item() > 0:
                pkd_scale = ref_loss / proto_kd.detach().clamp(min=1e-8)
                proto_kd_term = self._proto_kd_weight * pkd_scale * proto_kd
            else:
                proto_kd_term = proto_kd

            return (
                base_loss + kd_term + ewc_term
                + replay_kd_term + proto_kd_term
                + self._ortho_lambda * ortho
            )
        else:
            return (
                base_loss
                + self.kd_weight * kd
                + self.kd_weight * replay_kd
                + self._proto_kd_weight * proto_kd
                + self.ewc_weight * ewc
                + self._ortho_lambda * ortho
            )

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

        When a new task arrives, ``reconfigure_adapter`` adds a randomly
        initialized 1x1 Conv3d residual.  Without training, the teacher
        produces random logits for the new task, making KD useless.

        This method runs a short supervised loop (backbone frozen, core
        frozen, gate frozen, only the new residual trainable) and accumulates
        prototypes.  Controlled by ``method.kd.adapter_pretrain.steps``
        (default 200) and ``method.kd.adapter_pretrain.lr`` (default 1e-3).
        """
        if not self.teacher.is_external:
            return
        backend = self.teacher._backend
        adapter = getattr(backend, "_adapter", None)
        if adapter is None:
            return

        pretrain_cfg = self.cfg.get("method", {}).get("kd", {}).get("adapter_pretrain", {})
        max_steps = int(pretrain_cfg.get("steps", 200))
        lr = float(pretrain_cfg.get("lr", 1e-3))

        if max_steps <= 0:
            return

        device = cfg.get("runtime", {}).get("device", "cpu")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Only train the new residual — gate stays frozen to preserve
        # old-task calibration (trained during initial adapter pretraining).
        trainable_params = []
        is_gated = getattr(backend, "_gated", False)

        if hasattr(adapter, "residuals") and task_id in adapter.residuals:
            for p in adapter.residuals[task_id].parameters():
                p.requires_grad = True
                trainable_params.append(p)
        else:
            return  # no residual to train

        # Explicitly freeze gate and core to prevent accidental updates
        if is_gated and hasattr(adapter, "gate_head"):
            for p in adapter.gate_head.parameters():
                p.requires_grad = False
        if hasattr(adapter, "core"):
            for p in adapter.core.parameters():
                p.requires_grad = False

        if not trainable_params:
            return

        optimizer = torch.optim.Adam(trainable_params, lr=lr)
        backend.to(device)
        adapter.train()

        task_logger.info(
            f"  Pretraining adapter residual for '{task_id}' "
            f"({max_steps} steps, lr={lr}, params={sum(p.numel() for p in trainable_params)})"
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
            target_shape = (D, H, W)

            if is_gated:
                logits, _ = adapter(features, target_shape)
                loss = F.cross_entropy(logits, y)

                # Accumulate prototypes for this task
                with torch.no_grad():
                    adapter.update_prototypes(features, y, task_id, out_channels)
            else:
                logits = adapter(features, target_shape)
                loss = F.cross_entropy(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            steps += 1

        adapter.eval()
        proto_count = adapter.num_prototypes if hasattr(adapter, "num_prototypes") else 0
        task_logger.info(
            f"  Adapter residual pretrained: {steps} steps, "
            f"{proto_count} prototypes stored"
        )

    def post_task_update(self, model: nn.Module, **kwargs) -> None:
        # Reset for next task
        self._reference_loss = None
        self._current_task_step = 0

        # Teacher snapshot (skip for external teachers like SAM3/MedSAM3)
        if not self.teacher.is_external:
            self.teacher.snapshot(model)
        # Fisher estimation
        train_loader = kwargs.get("train_loader")
        device = next(model.parameters()).device
        if train_loader is not None:
            self.fisher = self._estimate_fisher(
                model, train_loader, str(device), self.fisher_samples
            )
        # Param snapshot
        self.prev_params = {
            n: p.detach().cpu().clone() for n, p in model.named_parameters()
        }
        # Save current LoRA state for orthogonality constraint in next task
        if self._lora_enabled:
            from src.models.lora import extract_lora_state
            lora_state = extract_lora_state(model)
            if lora_state:
                self.prev_lora_states.append(lora_state)
                logger.info(
                    f"Saved LoRA state for ortho constraint "
                    f"({len(self.prev_lora_states)} task(s) stored)"
                )

    def save_state(self, path: Path, model_template: nn.Module | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state: Dict[str, Any] = {
            "fisher": self.fisher,
            "prev_params": self.prev_params,
            "memory": [
                {k: v for k, v in m.items()}
                for m in self.memory
            ],
            "prev_lora_states": self.prev_lora_states,
            "_reference_loss": self._reference_loss,
        }
        state.update(self.teacher.state_dict())
        torch.save(state, path)

    def load_state(self, path: Path, model_template: nn.Module | None = None) -> None:
        state = torch.load(Path(path), map_location="cpu", weights_only=False)
        self.fisher = state.get("fisher", {})
        self.prev_params = state.get("prev_params", {})
        self.memory = state.get("memory", [])
        self.prev_lora_states = state.get("prev_lora_states", [])
        self._reference_loss = state.get("_reference_loss", None)
        self.teacher.load_state_dict(state, model_template=model_template)
