#!/usr/bin/env bash
# Download teacher checkpoints from HuggingFace Hub.
# Usage: bash scripts/download_checkpoints.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CKPT_DIR="${REPO_ROOT}/checkpoints"
mkdir -p "${CKPT_DIR}"

echo "=== Downloading MedSAM3 LoRA weights ==="
LORA_PATH=$(python3 -c "
from huggingface_hub import hf_hub_download
p = hf_hub_download('lal-Joey/MedSAM3_v1', 'best_lora_weights.pt')
print(p)
")

ln -sf "${LORA_PATH}" "${CKPT_DIR}/medsam3_lora.pt"
echo "LoRA weights symlinked to ${CKPT_DIR}/medsam3_lora.pt"

echo ""
echo "=== Verifying checksum ==="
EXPECTED="499e638bb7c51dbe0dcc3bfb9dbfada74fc2d725e953fbb5bdb2dd1b72106f91"
ACTUAL=$(sha256sum "${CKPT_DIR}/medsam3_lora.pt" | awk '{print $1}')
if [ "${ACTUAL}" = "${EXPECTED}" ]; then
    echo "Checksum OK: ${ACTUAL}"
else
    echo "WARNING: checksum mismatch!"
    echo "  expected: ${EXPECTED}"
    echo "  actual:   ${ACTUAL}"
fi

echo ""
echo "=== SAM3 base weights ==="
echo "SAM3 base will be auto-downloaded on first use via build_sam3_image_model(load_from_HF=True)."
echo "No separate download needed."
echo ""
echo "Done."
