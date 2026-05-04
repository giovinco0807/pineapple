#!/bin/bash
# GCP Self-Play Data Generation - Startup Script
# For spot/preemptible e2-highcpu-32 instances

set -eo pipefail

# ── Config ──
export HOME="/root"
REPO="https://github.com/giovinco0807/pineapple.git"
BRANCH="verify-gcp-phase-one-20260501"
WORK_DIR="$HOME/ofc"
BUCKET="gs://ofc-solver-results/selfplay"

get_meta() {
    curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" \
        -H "Metadata-Flavor: Google" 2>/dev/null || echo "$2"
}

VM_ID=$(get_meta "VM_ID" "0")
N_GAMES=$(get_meta "N_GAMES" "1000")
N2=$(get_meta "N2" "3")
N3=$(get_meta "N3" "3")
N4=$(get_meta "N4" "3")
SEED=$((42 + VM_ID * 100000))

echo "============================================"
echo "  Self-Play Generator - VM #${VM_ID}"
echo "  Games: ${N_GAMES}, N2=${N2}, N3=${N3}, N4=${N4}"
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
    # Download T0 model for deterministic pruning
    gsutil -q cp gs://ofc-solver-results/models/t0_placement_net_v4.pt "$WORK_DIR/ai/models/" || echo "Model download failed or missing"

    echo "=== Setup complete ==="
}

# ── Run with periodic upload ──
run() {
    cd "$WORK_DIR"
    OUTPUT="selfplay_vm${VM_ID}.jsonl"
    GCS_PATH="${BUCKET}/${OUTPUT}"

    echo "=== Starting generation ==="
    pkill -9 python3 2>/dev/null || true
    sleep 1

    # Run self-play script in background
    python3 ai/generate_selfplay_data_mp.py \
        --games "$N_GAMES" \
        --n2 "$N2" \
        --n3 "$N3" \
        --n4 "$N4" \
        --output "$OUTPUT" \
        --seed "$SEED" &
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
        echo "{\"vm_id\":$VM_ID,\"games\":$N_GAMES,\"done\":\"$(date -Iseconds)\"}" \
            | gsutil cp - "${BUCKET}/status/vm${VM_ID}_done.json"
    fi
    echo "=== Complete: $(date) ==="
}

# ── Main ──
setup
run
