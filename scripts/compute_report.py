#!/usr/bin/env python3
"""Compute-efficiency reporter.

Parses training logs, model configs, and run manifests to produce a
compute-efficiency summary including:
- Parameter counts (total, trainable, LoRA)
- Throughput (steps/sec, samples/sec)
- Epoch wall time
- VRAM usage (if logged in manifest)

Usage:
    python scripts/compute_report.py \\
        --run-dirs outputs/baselines/clean_v1_5ep/finetune \\
                   outputs/baselines/clean_v1_5ep/distill \\
        --output-dir outputs/compute_report

    python scripts/compute_report.py --help
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def count_model_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Estimate parameter counts from config without instantiating the model.

    For MONAI UNet with Conv3d layers, rough formula:
    Each level: in_ch * out_ch * kernel^3 * num_res_units (+ skip connections)
    """
    mcfg = cfg.get("model", {})
    channels = mcfg.get("channels", [16, 32, 64, 128])
    in_ch = mcfg.get("in_channels", 1)
    out_ch = mcfg.get("out_channels", 3)
    num_res = mcfg.get("num_res_units", 2)
    kernel = 3

    # Try to instantiate for exact count
    try:
        from src.models.factory import create_model
        model = create_model(cfg)
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        lora_params = 0
        try:
            from src.models.lora import get_lora_params
            lora_params = sum(p.numel() for p in get_lora_params(model))
        except Exception:
            pass
        return {
            "total_params": total,
            "trainable_params": trainable,
            "lora_params": lora_params,
            "source": "instantiated",
        }
    except Exception:
        pass

    # Rough estimation fallback
    total_est = 0
    prev_ch = in_ch
    for ch in channels:
        total_est += prev_ch * ch * (kernel ** 3) * num_res
        total_est += ch  # bias
        prev_ch = ch
    # Decoder (symmetric)
    for i in range(len(channels) - 1, 0, -1):
        total_est += channels[i] * channels[i - 1] * (kernel ** 3)
    total_est += channels[0] * out_ch * (kernel ** 3)

    return {
        "total_params": total_est,
        "trainable_params": total_est,
        "lora_params": 0,
        "source": "estimated",
    }


def parse_metrics_csv(path: Path) -> Dict[str, Any]:
    """Extract timing/throughput info from metrics CSV."""
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}

    epochs = len(rows)
    last_row = rows[-1]

    # Parse train loss trajectory
    losses = []
    for row in rows:
        try:
            losses.append(float(row.get("train_loss", "nan")))
        except (ValueError, TypeError):
            pass

    return {
        "num_epochs": epochs,
        "final_train_loss": losses[-1] if losses else None,
        "initial_train_loss": losses[0] if losses else None,
    }


def parse_run_manifest(path: Path) -> Dict[str, Any]:
    """Extract compute info from run_manifest.json."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}

    env = data.get("environment", {})
    result: Dict[str, Any] = {}

    if "gpu_name" in env:
        result["gpu_name"] = env["gpu_name"]
    if "cuda_version" in env:
        result["cuda_version"] = env["cuda_version"]
    if "torch_version" in env:
        result["torch_version"] = env["torch_version"]

    # VRAM if logged
    if "gpu_memory_mb" in env:
        result["vram_mb"] = env["gpu_memory_mb"]

    # Duration if logged
    if "duration_sec" in data:
        result["duration_sec"] = data["duration_sec"]

    return result


def parse_resolved_config(path: Path) -> Dict[str, Any]:
    """Load resolved config from a run directory."""
    if not path.exists():
        return {}
    try:
        import yaml
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def parse_suite_summary(path: Path) -> Dict[str, Dict[str, Any]]:
    """Parse suite_summary.csv for duration per method."""
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    result: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        method = row.get("method", "unknown")
        entry: Dict[str, Any] = {}
        if "duration_sec" in row:
            try:
                entry["duration_sec"] = float(row["duration_sec"])
            except (ValueError, TypeError):
                pass
        entry["status"] = row.get("status", "unknown")
        result[method] = entry
    return result


def collect_run_info(run_dir: Path) -> Dict[str, Any]:
    """Collect all compute info from a single run directory."""
    run_dir = Path(run_dir)
    name = run_dir.name

    info: Dict[str, Any] = {"name": name, "path": str(run_dir)}

    # Metrics
    metrics_info = parse_metrics_csv(run_dir / "metrics.csv")
    info.update(metrics_info)

    # Manifest
    manifest_info = parse_run_manifest(run_dir / "run_manifest.json")
    info.update(manifest_info)

    # Resolved config for param counting
    cfg = parse_resolved_config(run_dir / "resolved_config.yaml")
    if cfg:
        param_info = count_model_params(cfg)
        info.update(param_info)

        # Training config
        tcfg = cfg.get("train", {})
        info["configured_epochs"] = tcfg.get("epochs")
        info["configured_lr"] = tcfg.get("lr")
        info["configured_batch_size"] = cfg.get("data", {}).get("batch_size")
        info["amp_enabled"] = tcfg.get("amp", {}).get("enabled", False)

        # LoRA config
        lora_cfg = cfg.get("model", {}).get("lora", {})
        if lora_cfg.get("enabled"):
            info["lora_enabled"] = True
            info["lora_rank"] = lora_cfg.get("rank")
            info["lora_mode"] = lora_cfg.get("mode")

        # Method
        info["method"] = cfg.get("method", {}).get("name")

    # Check for suite summary in parent
    suite_summary = run_dir.parent / "suite_summary.csv"
    if suite_summary.exists():
        suite_data = parse_suite_summary(suite_summary)
        if name in suite_data:
            info.update(suite_data[name])

    return info


def format_csv_report(runs: List[Dict[str, Any]]) -> str:
    """Format runs as CSV string."""
    if not runs:
        return ""
    # Collect all keys
    all_keys: List[str] = []
    for r in runs:
        for k in r:
            if k not in all_keys:
                all_keys.append(k)

    import io
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=all_keys)
    w.writeheader()
    for r in runs:
        w.writerow(r)
    return buf.getvalue()


def format_markdown_report(runs: List[Dict[str, Any]]) -> str:
    """Format compute efficiency report as markdown."""
    lines: List[str] = []
    lines.append("# Compute Efficiency Report\n")

    if not runs:
        lines.append("No run data found.\n")
        return "\n".join(lines)

    # Summary table
    lines.append("## Summary\n")
    lines.append("| Run | Method | Total Params | Trainable | LoRA | Epochs | AMP | Duration (s) |")
    lines.append("|-----|--------|-------------|-----------|------|--------|-----|-------------|")

    for r in runs:
        total = r.get("total_params", "?")
        trainable = r.get("trainable_params", "?")
        lora = r.get("lora_params", 0)
        if isinstance(total, int):
            total = f"{total:,}"
        if isinstance(trainable, int):
            trainable = f"{trainable:,}"
        if isinstance(lora, int) and lora > 0:
            lora = f"{lora:,}"
        else:
            lora = "-"

        lines.append(
            f"| {r.get('name', '?')} "
            f"| {r.get('method', '?')} "
            f"| {total} "
            f"| {trainable} "
            f"| {lora} "
            f"| {r.get('num_epochs', r.get('configured_epochs', '?'))} "
            f"| {'Yes' if r.get('amp_enabled') else 'No'} "
            f"| {r.get('duration_sec', '?')} |"
        )

    lines.append("")

    # Environment info (from first run that has it)
    env_run = None
    for r in runs:
        if r.get("gpu_name"):
            env_run = r
            break
    if env_run:
        lines.append("## Environment\n")
        lines.append(f"- **GPU:** {env_run.get('gpu_name', 'N/A')}")
        lines.append(f"- **CUDA:** {env_run.get('cuda_version', 'N/A')}")
        lines.append(f"- **PyTorch:** {env_run.get('torch_version', 'N/A')}")
        if env_run.get("vram_mb"):
            lines.append(f"- **VRAM:** {env_run['vram_mb']} MB")
        lines.append("")

    # Notes
    lines.append("## Notes\n")
    lines.append("- Parameter counts marked `estimated` were computed from config without model instantiation.")
    lines.append("- Duration values require actual training runs to be populated.")
    lines.append("- VRAM usage requires explicit logging during training (not yet implemented).\n")

    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate compute-efficiency report from run directories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Multiple run directories
  python scripts/compute_report.py \\
      --run-dirs outputs/baselines/clean_v1_5ep/finetune \\
                 outputs/baselines/clean_v1_5ep/distill

  # Scan all subdirs
  python scripts/compute_report.py \\
      --scan-dir outputs/baselines/clean_v1_5ep

  # With output
  python scripts/compute_report.py \\
      --scan-dir outputs/ablations --output-dir outputs/compute_report
        """,
    )
    p.add_argument("--run-dirs", nargs="*", default=[], help="Run directories to analyze")
    p.add_argument("--scan-dir", default=None, help="Scan subdirectories of this path")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--json-only", action="store_true")
    args = p.parse_args()

    run_dirs: List[Path] = [Path(d) for d in args.run_dirs]

    if args.scan_dir:
        scan = Path(args.scan_dir)
        if scan.exists():
            for child in sorted(scan.iterdir()):
                if child.is_dir() and child not in run_dirs:
                    run_dirs.append(child)

    if not run_dirs:
        print("No run directories specified. Use --run-dirs or --scan-dir.", file=sys.stderr)
        p.print_help()
        sys.exit(1)

    runs: List[Dict[str, Any]] = []
    for d in run_dirs:
        if d.exists():
            info = collect_run_info(d)
            runs.append(info)
        else:
            print(f"WARNING: directory not found: {d}", file=sys.stderr)

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # JSON
        (out / "compute_report.json").write_text(
            json.dumps(runs, indent=2, default=str), encoding="utf-8"
        )

        # CSV
        csv_content = format_csv_report(runs)
        if csv_content:
            (out / "compute_report.csv").write_text(csv_content, encoding="utf-8")

        # Markdown
        if not args.json_only:
            md = format_markdown_report(runs)
            (out / "compute_report.md").write_text(md, encoding="utf-8")
            print(md)
        else:
            print(json.dumps(runs, indent=2, default=str))
    else:
        if args.json_only:
            print(json.dumps(runs, indent=2, default=str))
        else:
            md = format_markdown_report(runs)
            print(md)


if __name__ == "__main__":
    main()
