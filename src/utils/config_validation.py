"""Strict config schema validation.

Call ``validate_config(cfg)`` before training to fail fast with
actionable error messages for missing or incompatible settings.

Also provides ``compute_config_hash()`` for reproducibility tracking
and ``save_resolved_config()`` for run artifact persistence.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import yaml

logger = logging.getLogger(__name__)


class ConfigError(ValueError):
    """Raised when config validation fails."""
    pass


# ---------------------------------------------------------------------------
# Individual validators
# ---------------------------------------------------------------------------

def _require(cfg: Dict, dotpath: str, context: str = "") -> Any:
    """Walk *cfg* along *dotpath* (e.g. ``"method.kd.weight"``).

    Raises ConfigError if any segment is missing.
    """
    keys = dotpath.split(".")
    node = cfg
    traversed: List[str] = []
    for k in keys:
        traversed.append(k)
        if not isinstance(node, dict) or k not in node:
            msg = f"Missing required config key: {dotpath}"
            if context:
                msg += f" ({context})"
            raise ConfigError(msg)
        node = node[k]
    return node


def _validate_model(cfg: Dict) -> List[str]:
    errors: List[str] = []
    mcfg = cfg.get("model", {})
    if "out_channels" not in mcfg:
        errors.append("model.out_channels is required")
    if "in_channels" not in mcfg:
        errors.append("model.in_channels is required")
    out_ch = mcfg.get("out_channels", 0)
    if isinstance(out_ch, int) and out_ch < 2:
        errors.append(
            f"model.out_channels={out_ch} is too low for segmentation (min 2)"
        )
    return errors


def _validate_data(cfg: Dict) -> List[str]:
    errors: List[str] = []
    data_cfg = cfg.get("data", {})
    source = data_cfg.get("source")
    if not source:
        errors.append("data.source is required (synthetic, totalseg, brats21, acdc)")
        return errors

    if source == "synthetic":
        return errors

    source_cfg = data_cfg.get(source, {})
    if not source_cfg.get("root"):
        errors.append(f"data.{source}.root is required for source={source}")

    has_manifest = bool(source_cfg.get("split_manifest"))
    has_ids = bool(source_cfg.get("train_ids")) and bool(source_cfg.get("val_ids"))
    if not has_manifest and not has_ids:
        errors.append(
            f"data.{source} needs either split_manifest or train_ids+val_ids"
        )

    if has_manifest:
        manifest_path = source_cfg["split_manifest"]
        p = Path(manifest_path)
        if not p.is_absolute():
            # Will be resolved relative to repo root at runtime, just check format
            if not p.suffix in {".json", ".yaml", ".yml"}:
                errors.append(
                    f"data.{source}.split_manifest must be .json/.yaml/.yml, "
                    f"got '{p.suffix}'"
                )

    return errors


def _validate_method(cfg: Dict) -> List[str]:
    errors: List[str] = []
    mcfg = cfg.get("method", {})
    name = mcfg.get("name")
    if not name:
        errors.append("method.name is required")
        return errors

    valid_methods = {"finetune", "replay", "distill", "distill_replay_ewc"}
    if name not in valid_methods:
        errors.append(f"method.name='{name}' is not supported. Valid: {valid_methods}")
        return errors

    if name in {"distill", "distill_replay_ewc"}:
        kd_cfg = mcfg.get("kd", {})
        mode = kd_cfg.get("mode", "logit")
        valid_modes = {"logit", "feature", "weighted", "boundary"}
        if mode not in valid_modes:
            errors.append(
                f"method.kd.mode='{mode}' is invalid. Valid: {valid_modes}"
            )

        teacher_cfg = kd_cfg.get("teacher", {})
        teacher_type = teacher_cfg.get("type", "snapshot")
        if teacher_type not in {"snapshot", "checkpoint", "sam3", "medsam3"}:
            errors.append(
                f"method.kd.teacher.type='{teacher_type}' invalid. "
                "Use 'snapshot', 'checkpoint', 'sam3', or 'medsam3'."
            )
        if teacher_type == "checkpoint" and not teacher_cfg.get("ckpt_path"):
            errors.append(
                "method.kd.teacher.ckpt_path is required when teacher.type=checkpoint"
            )
        if teacher_type in {"sam3", "medsam3"}:
            ckpt_val = teacher_cfg.get("ckpt_path")
            if not ckpt_val and not teacher_cfg.get("load_from_hf", False):
                errors.append(
                    f"method.kd.teacher.ckpt_path is required when teacher.type={teacher_type} "
                    "(set to 'auto' for HuggingFace auto-download)"
                )
            if not teacher_cfg.get("output_channels"):
                errors.append(
                    f"method.kd.teacher.output_channels is required when teacher.type={teacher_type}"
                )

        # PEFT/LoRA validation
        peft_cfg = teacher_cfg.get("peft", {})
        if peft_cfg.get("enabled", False):
            peft_type = peft_cfg.get("type", "lora")
            if peft_type != "lora":
                errors.append(
                    f"method.kd.teacher.peft.type='{peft_type}' is not supported. "
                    "Currently only 'lora' is supported."
                )
            rank = peft_cfg.get("rank", 8)
            if isinstance(rank, int) and rank < 1:
                errors.append(
                    f"method.kd.teacher.peft.rank={rank} must be >= 1"
                )
            alpha = peft_cfg.get("alpha", 16)
            if isinstance(alpha, int) and alpha < 1:
                errors.append(
                    f"method.kd.teacher.peft.alpha={alpha} must be >= 1"
                )

        if mode == "feature":
            if not teacher_cfg.get("feature_layers"):
                errors.append(
                    "method.kd.teacher.feature_layers is required when kd.mode=feature"
                )

    if name in {"replay", "distill_replay_ewc"}:
        replay_cfg = mcfg.get("replay", {})
        buf_size = replay_cfg.get("buffer_size", 64)
        if isinstance(buf_size, int) and buf_size < 1:
            errors.append(
                f"method.replay.buffer_size={buf_size} must be >= 1"
            )

    if name == "distill_replay_ewc":
        ewc_cfg = mcfg.get("ewc", {})
        fisher_samples = ewc_cfg.get("fisher_samples", 64)
        if isinstance(fisher_samples, int) and fisher_samples < 1:
            errors.append(
                f"method.ewc.fisher_samples={fisher_samples} must be >= 1"
            )

    return errors


def _validate_lora(cfg: Dict) -> List[str]:
    errors: List[str] = []
    lora_cfg = cfg.get("model", {}).get("lora", {})
    if not lora_cfg.get("enabled", False):
        return errors

    rank = lora_cfg.get("rank", 8)
    if isinstance(rank, int) and rank < 1:
        errors.append(f"model.lora.rank={rank} must be >= 1")
    alpha = lora_cfg.get("alpha", 16)
    if isinstance(alpha, (int, float)) and alpha < 1:
        errors.append(f"model.lora.alpha={alpha} must be >= 1")
    mode = lora_cfg.get("mode", "standard")
    valid_modes = {"standard", "orthogonal"}
    if mode not in valid_modes:
        errors.append(f"model.lora.mode='{mode}' is invalid. Valid: {valid_modes}")
    ortho_lambda = lora_cfg.get("ortho_lambda", 0.1)
    if isinstance(ortho_lambda, (int, float)) and ortho_lambda < 0:
        errors.append(f"model.lora.ortho_lambda={ortho_lambda} must be >= 0")
    return errors


def _validate_train(cfg: Dict) -> List[str]:
    errors: List[str] = []
    tcfg = cfg.get("train", {})
    epochs = tcfg.get("epochs", 1)
    if isinstance(epochs, int) and epochs < 1:
        errors.append(f"train.epochs={epochs} must be >= 1")
    lr = tcfg.get("lr", 0.001)
    if isinstance(lr, (int, float)) and lr <= 0:
        errors.append(f"train.lr={lr} must be positive")
    loss_type = tcfg.get("loss_type", "dicece")
    if loss_type not in {"dicece", "ce"}:
        errors.append(f"train.loss_type='{loss_type}' invalid. Use 'dicece' or 'ce'.")
    return errors


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_config(cfg: Dict[str, Any], strict: bool = True) -> List[str]:
    """Validate a resolved experiment config.

    Args:
        cfg: Fully-merged config dict.
        strict: If True (default), raise ConfigError on first error batch.
                If False, return list of error strings.

    Returns:
        List of error strings (empty if valid). Only returned when strict=False.

    Raises:
        ConfigError: When strict=True and errors are found.
    """
    errors: List[str] = []
    errors.extend(_validate_model(cfg))
    errors.extend(_validate_data(cfg))
    errors.extend(_validate_method(cfg))
    errors.extend(_validate_lora(cfg))
    errors.extend(_validate_train(cfg))

    if errors and strict:
        msg = "Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ConfigError(msg)

    return errors


# ---------------------------------------------------------------------------
# Config hashing and persistence
# ---------------------------------------------------------------------------

def compute_config_hash(cfg: Dict[str, Any]) -> str:
    """Deterministic SHA256 hash of a config dict for reproducibility tracking."""
    serialized = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def save_resolved_config(cfg: Dict[str, Any], output_dir: str | Path) -> Path:
    """Write resolved config YAML and its hash to *output_dir*.

    Creates:
        - ``resolved_config.yaml``
        - ``config_hash.txt``

    Returns path to the YAML file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    yaml_path = output_dir / "resolved_config.yaml"
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=True)

    hash_val = compute_config_hash(cfg)
    (output_dir / "config_hash.txt").write_text(hash_val + "\n", encoding="utf-8")

    return yaml_path


def validate_paths(cfg: Dict[str, Any]) -> List[str]:
    """Check that configured file/directory paths actually exist on disk."""
    errors: List[str] = []
    data_cfg = cfg.get("data", {})
    source = data_cfg.get("source", "synthetic")
    if source == "synthetic":
        return errors

    source_cfg = data_cfg.get(source, {})
    root = source_cfg.get("root")
    if root and not Path(root).exists():
        errors.append(f"data.{source}.root path does not exist: {root}")

    # Teacher checkpoint
    teacher_cfg = cfg.get("method", {}).get("kd", {}).get("teacher", {})
    ckpt = teacher_cfg.get("ckpt_path")
    if ckpt and ckpt != "auto" and not Path(ckpt).exists():
        errors.append(f"method.kd.teacher.ckpt_path not found: {ckpt}")

    # Teacher LoRA weights
    lora = teacher_cfg.get("lora_path")
    if lora and not Path(lora).exists():
        errors.append(f"method.kd.teacher.lora_path not found: {lora}")

    return errors
