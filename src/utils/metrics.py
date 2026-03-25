from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Any


class CSVMetricsLogger:
    """Minimal CSV logger for epoch-wise metrics."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._header_written = self.path.exists() and self.path.stat().st_size > 0

    def log(self, row: Dict[str, Any]) -> None:
        fieldnames = list(row.keys())
        with self.path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
            writer.writerow(row)
