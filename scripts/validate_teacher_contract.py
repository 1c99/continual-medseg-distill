#!/usr/bin/env python3
"""Validate the forward contract of a teacher backend.

Loads a teacher backend, runs a forward pass on synthetic input, and
validates the output shape, dtype, and value sanity (no NaN/Inf).

Usage:
    python scripts/validate_teacher_contract.py \
        --backend medsam3 \
        --lora-path checkpoints/medsam3_lora.pt \
        --output-channels 6

Exit codes:
    0 — contract passed
    1 — contract failed
"""
from __future__ import annotations

import argparse
import datetime
import sys
import traceback
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

from src.methods.teacher_backends import create_backend


def build_teacher_cfg(args: argparse.Namespace) -> dict:
    """Build a teacher config dict from CLI args."""
    cfg = {
        "type": args.backend,
        "output_channels": args.output_channels,
    }
    if args.ckpt_path:
        cfg["ckpt_path"] = args.ckpt_path
    else:
        # Use auto-download for medsam3
        cfg["ckpt_path"] = "auto"
        cfg["load_from_hf"] = True
    if args.lora_path:
        cfg["lora_path"] = args.lora_path
    return cfg


def validate_contract(backend, input_tensor: torch.Tensor, output_channels: int) -> dict:
    """Run forward pass and validate the output contract."""
    result = {
        "input_shape": list(input_tensor.shape),
        "input_dtype": str(input_tensor.dtype),
        "passed": False,
        "errors": [],
        "warnings": [],
    }

    B, C_in, D, H, W = input_tensor.shape
    expected_shape = (B, output_channels, D, H, W)

    try:
        with torch.no_grad():
            output = backend.forward_logits(input_tensor)
    except Exception as e:
        result["errors"].append(f"forward_logits raised {type(e).__name__}: {e}")
        result["traceback"] = traceback.format_exc()
        return result

    result["output_shape"] = list(output.shape)
    result["output_dtype"] = str(output.dtype)

    # Shape check
    if tuple(output.shape) != expected_shape:
        result["errors"].append(
            f"Shape mismatch: expected {expected_shape}, got {tuple(output.shape)}"
        )

    # Dtype check
    if output.dtype != torch.float32:
        result["errors"].append(
            f"Dtype mismatch: expected torch.float32, got {output.dtype}"
        )

    # NaN/Inf check
    if torch.isnan(output).any():
        nan_count = torch.isnan(output).sum().item()
        result["errors"].append(f"Output contains {nan_count} NaN values")

    if torch.isinf(output).any():
        inf_count = torch.isinf(output).sum().item()
        result["errors"].append(f"Output contains {inf_count} Inf values")

    # Value stats
    result["output_stats"] = {
        "min": float(output.min()),
        "max": float(output.max()),
        "mean": float(output.mean()),
        "std": float(output.std()),
    }

    # Test forward_features if available
    try:
        with torch.no_grad():
            features = backend.forward_features(input_tensor)
        result["features_keys"] = list(features.keys())
        for k, v in features.items():
            if torch.isnan(v).any():
                result["warnings"].append(f"Feature '{k}' contains NaN")
    except NotImplementedError:
        result["features_keys"] = "not_supported"
    except Exception as e:
        result["warnings"].append(f"forward_features raised {type(e).__name__}: {e}")

    result["passed"] = len(result["errors"]) == 0
    return result


def write_report(result: dict, backend_name: str, output_dir: Path) -> Path:
    """Write a markdown contract report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{backend_name}_contract_report.md"

    status = "PASSED" if result["passed"] else "FAILED"
    lines = [
        f"# Teacher Contract Validation: {backend_name}",
        "",
        f"**Status:** {status}",
        f"**Timestamp:** {datetime.datetime.now().isoformat()}",
        "",
        "## Input",
        f"- Shape: `{result['input_shape']}`",
        f"- Dtype: `{result['input_dtype']}`",
        "",
        "## Output",
        f"- Shape: `{result.get('output_shape', 'N/A')}`",
        f"- Dtype: `{result.get('output_dtype', 'N/A')}`",
    ]

    if "output_stats" in result:
        stats = result["output_stats"]
        lines.extend([
            "",
            "## Output Statistics",
            f"- Min: {stats['min']:.6f}",
            f"- Max: {stats['max']:.6f}",
            f"- Mean: {stats['mean']:.6f}",
            f"- Std: {stats['std']:.6f}",
        ])

    if result.get("features_keys"):
        lines.extend([
            "",
            "## Features",
            f"- Keys: `{result['features_keys']}`",
        ])

    if result["errors"]:
        lines.extend(["", "## Errors"])
        for e in result["errors"]:
            lines.append(f"- {e}")

    if result.get("warnings"):
        lines.extend(["", "## Warnings"])
        for w in result["warnings"]:
            lines.append(f"- {w}")

    if result.get("traceback"):
        lines.extend(["", "## Traceback", "```", result["traceback"], "```"])

    lines.append("")
    report_path.write_text("\n".join(lines))
    return report_path


def main():
    ap = argparse.ArgumentParser(description="Validate teacher backend forward contract")
    ap.add_argument("--backend", required=True, choices=["sam3", "medsam3"],
                    help="Teacher backend type")
    ap.add_argument("--ckpt-path", default=None, help="Path to base checkpoint")
    ap.add_argument("--lora-path", default=None, help="Path to LoRA weights")
    ap.add_argument("--output-channels", type=int, default=6,
                    help="Expected number of output channels")
    ap.add_argument("--output-dir", default="outputs/teacher_contract",
                    help="Directory for contract report")
    args = ap.parse_args()

    print(f"=== Teacher Contract Validation: {args.backend} ===")
    print(f"ckpt_path: {args.ckpt_path or 'auto (HF)'}")
    print(f"lora_path: {args.lora_path}")
    print(f"output_channels: {args.output_channels}")

    # Build config and load backend
    cfg = build_teacher_cfg(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    try:
        backend = create_backend(cfg)
        backend = backend.to(device)
        backend.eval()
    except Exception as e:
        print(f"FAILED to load backend: {e}")
        traceback.print_exc()
        result = {
            "input_shape": [1, 1, 64, 64, 64],
            "input_dtype": "torch.float32",
            "passed": False,
            "errors": [f"Backend load failed: {type(e).__name__}: {e}"],
            "traceback": traceback.format_exc(),
        }
        report = write_report(result, args.backend, Path(args.output_dir))
        print(f"Report written to: {report}")
        sys.exit(1)

    # Create synthetic input
    x = torch.randn(1, 1, 64, 64, 64, device=device)
    print(f"Input shape: {x.shape}")

    # Validate
    result = validate_contract(backend, x, args.output_channels)

    # Write report
    report = write_report(result, args.backend, Path(args.output_dir))
    print(f"\nReport written to: {report}")

    if result["passed"]:
        print(f"PASSED: output shape {result['output_shape']}, dtype {result['output_dtype']}")
        sys.exit(0)
    else:
        print(f"FAILED:")
        for e in result["errors"]:
            print(f"  - {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
