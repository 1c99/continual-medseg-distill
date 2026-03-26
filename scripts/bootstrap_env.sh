#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .

echo "[bootstrap] environment ready"
echo "[bootstrap] run doctor: python scripts/doctor.py --dataset-root /media/user/data2/data2/data"
echo "[bootstrap] dry-run: python scripts/train.py --config configs/base.yaml --dry-run"
