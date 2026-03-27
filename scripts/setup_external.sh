#!/usr/bin/env bash
# Clone external model repositories into third_party/ with pinned commits.
# Usage: bash scripts/setup_external.sh [--sam3] [--medsam3] [--all]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
THIRD_PARTY="$REPO_ROOT/third_party"

SAM3_REPO="https://github.com/facebookresearch/sam3.git"
SAM3_COMMIT="86ed77094094e5cabb16b0414ec60c5ba9ce0a0f"

MEDSAM3_REPO="https://github.com/Joey-S-Liu/MedSAM3.git"
MEDSAM3_COMMIT="f79eef3f4ccfc880e63a0a5a758153137e75d34b"

clone_repo() {
    local name="$1"
    local url="$2"
    local commit="$3"
    local dest="$THIRD_PARTY/$name"

    if [ -d "$dest" ]; then
        echo "[skip] $dest already exists"
        return
    fi

    echo "[clone] $url -> $dest"
    mkdir -p "$THIRD_PARTY"
    git clone "$url" "$dest"

    if [ -n "$commit" ]; then
        echo "[pin] $name @ $commit"
        (cd "$dest" && git checkout "$commit")
    fi
}

do_sam3=false
do_medsam3=false

if [ $# -eq 0 ] || [[ " $* " == *" --all "* ]]; then
    do_sam3=true
    do_medsam3=true
else
    for arg in "$@"; do
        case "$arg" in
            --sam3) do_sam3=true ;;
            --medsam3) do_medsam3=true ;;
            *) echo "Unknown option: $arg"; exit 1 ;;
        esac
    done
fi

if $do_sam3; then
    clone_repo "sam3" "$SAM3_REPO" "$SAM3_COMMIT"
fi

if $do_medsam3; then
    clone_repo "medsam3" "$MEDSAM3_REPO" "$MEDSAM3_COMMIT"
fi

echo "[done] External repos ready in $THIRD_PARTY/"
