#!/bin/bash
# GCP T1 MC Data Generation - Startup Script
# For spot/preemptible e2-highcpu-32 instances
#
# Usage:
#   1. Create VM with this as startup script
#   2. Or SSH in and run: bash gcp_t1_mc.sh
#
# Metadata keys (set via --metadata):
#   VM_ID       - unique VM identifier (0-19)
#   N_HANDS     - hands per VM (default: 500)
#   N2          - T2 samples (default: 10)
#   N3          - T3 samples (default: 3)
#   N4          - T4 samples (default: 3)

set -euo pipefail

# ── Config ──
REPO="https://github.com/giovinco0807/pineapple.git"
BRANCH="verify-gcp-phase-one-20260501"
WORK_DIR="$HOME/ofc"
BUCKET="gs://ofc-solver-results"

# Read metadata (GCP) or use defaults
get_meta() {
    curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" \
        -H "Metadata-Flavor: Google" 2>/dev/null || echo "$2"
}

VM_ID=$(get_meta "VM_ID" "0")
N_HANDS=$(get_meta "N_HANDS" "500")
N2=$(get_meta "N2" "10")
N3=$(get_meta "N3" "3")
N4=$(get_meta "N4" "3")
SEED=$((42 + VM_ID * 100000))

echo "============================================"
echo "  T1 MC Generator - VM #${VM_ID}"
echo "  Hands: ${N_HANDS}, N2=${N2}, N3=${N3}, N4=${N4}"
echo "  Seed: ${SEED}"
echo "  Start: $(date)"
echo "============================================"

# ── Setup ──
setup() {
    echo "=== Installing Python deps ==="
    sudo apt-get update -qq
    sudo apt-get install -y -qq python3-pip python3-venv git > /dev/null 2>&1

    echo "=== Cloning repo ==="
    if [ -d "$WORK_DIR" ]; then
        cd "$WORK_DIR" && git pull --ff-only
    else
        git clone --branch "$BRANCH" --depth 1 "$REPO" "$WORK_DIR"
    fi
    cd "$WORK_DIR"

    echo "=== Installing Python packages ==="
    pip3 install --quiet torch numpy --break-system-packages 2>/dev/null || \
    pip3 install --quiet torch numpy

    echo "=== Setup complete ==="
}

# ── Run ──
run() {
    cd "$WORK_DIR"
    OUTPUT="t1_mc_vm${VM_ID}.jsonl"

    echo "=== Starting generation ==="
    # Kill any existing python processes to avoid env corruption
    pkill -9 python3 2>/dev/null || true
    sleep 1

    exec python3 ai/generate_t1_mc.py \
        --n-hands "$N_HANDS" \
        --n2 "$N2" \
        --n3 "$N3" \
        --n4 "$N4" \
        --workers 0 \
        --output "$OUTPUT" \
        --seed "$SEED" \
        --batch-size 50

    echo "=== Generation complete ==="
    echo "=== Uploading results ==="
    gsutil cp "$OUTPUT" "${BUCKET}/t1_mc/${OUTPUT}"
    echo "=== Upload complete: ${BUCKET}/t1_mc/${OUTPUT} ==="
}

# ── Main ──
setup
run
