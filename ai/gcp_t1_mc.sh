#!/bin/bash
# GCP T1 MC Data Generation - Startup Script
# For spot/preemptible e2-highcpu-32 instances
#
# Features:
#   - Periodic GCS upload (every batch) for mid-run data access
#   - Automatic resume on spot preemption
#   - Progress tracking via GCS metadata

set -eo pipefail

# ── Config ──
export HOME="${HOME:-/root}"
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
N_HANDS=$(get_meta "N_HANDS" "400")
N1=$(get_meta "N1" "30")
N2=$(get_meta "N2" "3")
N3=$(get_meta "N3" "3")
N4=$(get_meta "N4" "3")
SEED=$((42 + VM_ID * 100000))

echo "============================================"
echo "  T1 MC Generator - VM #${VM_ID}"
echo "  Hands: ${N_HANDS}, N1=${N1}, N2=${N2}, N3=${N3}, N4=${N4}"
echo "  Records: ~$((N_HANDS * N1)) per VM"
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

    echo "=== Downloading model files ==="
    mkdir -p "$WORK_DIR/ai/models"
    gsutil -q cp gs://ofc-solver-results/models/t0_placement_net_v4.pt "$WORK_DIR/ai/models/"

    echo "=== Setup complete ==="
}

# ── Run with periodic upload ──
run() {
    cd "$WORK_DIR"
    OUTPUT="t1_mc_vm${VM_ID}.jsonl"
    GCS_PATH="${BUCKET}/t1_mc/${OUTPUT}"

    echo "=== Starting generation ==="
    pkill -9 python3 2>/dev/null || true
    sleep 1

    # Run in background, upload periodically
    python3 ai/generate_t1_mc.py \
        --n-hands "$N_HANDS" \
        --n1 "$N1" \
        --n2 "$N2" \
        --n3 "$N3" \
        --n4 "$N4" \
        --workers 0 \
        --output "$OUTPUT" \
        --seed "$SEED" \
        --batch-size 10 &
    PY_PID=$!

    # Periodic upload loop (every 5 min)
    UPLOAD_INTERVAL=300
    while kill -0 $PY_PID 2>/dev/null; do
        sleep $UPLOAD_INTERVAL
        if [ -f "$OUTPUT" ]; then
            LINES=$(wc -l < "$OUTPUT")
            echo "[upload] ${LINES} records -> ${GCS_PATH}"
            gsutil -q cp "$OUTPUT" "$GCS_PATH" 2>/dev/null || true
        fi
    done

    # Final upload
    wait $PY_PID || true
    if [ -f "$OUTPUT" ]; then
        LINES=$(wc -l < "$OUTPUT")
        echo "=== Final upload: ${LINES} records ==="
        gsutil cp "$OUTPUT" "$GCS_PATH"
        # Also upload a "done" marker
        echo "{\"vm_id\":$VM_ID,\"records\":$LINES,\"done\":\"$(date -Iseconds)\"}" \
            | gsutil cp - "${BUCKET}/t1_mc/status/vm${VM_ID}_done.json"
    fi
    echo "=== Complete: $(date) ==="
}

# ── Main ──
setup
run
