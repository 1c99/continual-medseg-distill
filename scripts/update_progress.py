#!/usr/bin/env python3
"""Append a standardized checkpoint entry to docs/PROGRESS.md."""

from __future__ import annotations

import argparse
import subprocess
from datetime import date
from pathlib import Path


DEFAULT_PROGRESS_PATH = Path("docs/PROGRESS.md")
PROGRESS_HEADER = "# Project Progress Log\n\nThis file tracks implementation checkpoints in a consistent, low-friction format.\n"


def current_commit_short_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or "N/A"
    except Exception:
        return "N/A"


def ensure_progress_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(PROGRESS_HEADER + "\n", encoding="utf-8")


def render_entry(entry_date: str, commit: str, scope: str, impact: str, next_step: str) -> str:
    return (
        f"\n## {entry_date} — Checkpoint\n\n"
        f"- **date:** {entry_date}\n"
        f"- **commit:** `{commit}`\n"
        f"- **scope:** {scope}\n"
        f"- **impact:** {impact}\n"
        f"- **next:** {next_step}\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Append a standardized progress checkpoint to docs/PROGRESS.md "
            "with date/commit/scope/impact/next fields."
        )
    )
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Checkpoint date in ISO format (default: today).",
    )
    parser.add_argument(
        "--commit",
        default=current_commit_short_sha(),
        help="Commit SHA (default: current git short SHA, or N/A).",
    )
    parser.add_argument("--scope", required=True, help="What changed in this checkpoint.")
    parser.add_argument("--impact", required=True, help="Why this change matters.")
    parser.add_argument("--next", dest="next_step", required=True, help="Next planned step.")
    parser.add_argument(
        "--progress-file",
        default=str(DEFAULT_PROGRESS_PATH),
        help="Path to progress markdown file (default: docs/PROGRESS.md).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    progress_path = Path(args.progress_file)

    ensure_progress_file(progress_path)

    entry = render_entry(
        entry_date=args.date,
        commit=args.commit,
        scope=args.scope.strip(),
        impact=args.impact.strip(),
        next_step=args.next_step.strip(),
    )

    with progress_path.open("a", encoding="utf-8") as f:
        f.write(entry)

    print(f"Appended checkpoint to {progress_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
