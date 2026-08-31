#!/bin/bash
# GCP Worker Startup Script V2 - Phase B Data Generation
# Runs as root on VM boot. Self-deletes on completion.
#
# Key differences from V1:
#   - Outputs to gs://ofc-solver-485418/t0_phase_b/ (separate from Phase 1)
#   - Incremental GCS uploads every 50 hands (enables mid-run download)
#   - Better error handling and progress logging
set -e

export HOME="/root"

WORKER_ID=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/worker-id" -H "Metadata-Flavor: Google")
SEED=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/seed" -H "Metadata-Flavor: Google")
HANDS=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/hands" -H "Metadata-Flavor: Google")
SAMPLES=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/samples" -H "Metadata-Flavor: Google")
NESTING=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/nesting" -H "Metadata-Flavor: Google")

GCS_DEST="gs://ofc-solver-485418/t0_phase_b"

LOG="/tmp/worker_${WORKER_ID}.log"
exec > >(tee -a "$LOG") 2>&1

echo "=========================================="
echo "Worker $WORKER_ID started at $(date -u)"
echo "Seed=$SEED Hands=$HANDS Samples=$SAMPLES Nesting=$NESTING"
echo "Cores: $(nproc) | HOME=$HOME"
echo "GCS: $GCS_DEST"
echo "=========================================="

# Install dependencies
apt-get update -qq
apt-get install -y -qq git build-essential curl 2>&1 | tail -1

# Install Rust
export RUSTUP_HOME="/root/.rustup"
export CARGO_HOME="/root/.cargo"
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "/root/.cargo/env"

echo "Rust: $(rustc --version)"
echo "Cargo: $(cargo --version)"

# Clone and build
cd /tmp
git clone --depth 1 https://github.com/giovinco0807/pineapple.git ofc-pineapple 2>&1 | tail -2
cd ofc-pineapple/ai/rust_solver
echo "Building (release)..."
cargo build --release -p cfr_solver 2>&1 | tail -3
EXE="./target/release/cfr_solver"
echo "Build complete: $(ls -lh $EXE | awk '{print $5}')"

# Run batch evaluation
OUTFILE="/tmp/t0_worker_${WORKER_ID}.jsonl"
echo ""
echo "=== Starting evaluation ==="
echo "Command: $EXE t0-batch --hands $HANDS --samples $SAMPLES --nesting $NESTING --output $OUTFILE --seed $SEED"

# Start in background and monitor for incremental uploads
$EXE t0-batch \
    --hands "$HANDS" \
    --samples "$SAMPLES" \
    --nesting "$NESTING" \
    --output "$OUTFILE" \
    --seed "$SEED" &
SOLVER_PID=$!

# Incremental upload loop: upload every 5 minutes
LAST_LINES=0
while kill -0 $SOLVER_PID 2>/dev/null; do
    sleep 300  # 5 minutes
    if [ -f "$OUTFILE" ]; then
        CURRENT_LINES=$(wc -l < "$OUTFILE" 2>/dev/null || echo "0")
        if [ "$CURRENT_LINES" -gt "$LAST_LINES" ]; then
            echo "[$(date -u)] Progress: $CURRENT_LINES/$HANDS hands completed. Uploading snapshot..."
            gcloud storage cp "$OUTFILE" "$GCS_DEST/worker_${WORKER_ID}_s${SEED}.jsonl" 2>/dev/null || true
            gcloud storage cp "$LOG" "$GCS_DEST/logs/worker_${WORKER_ID}.log" 2>/dev/null || true
            LAST_LINES=$CURRENT_LINES
        fi
    fi
done

# Wait for solver to finish
wait $SOLVER_PID
EXIT_CODE=$?

echo ""
echo "=== Evaluation complete (exit=$EXIT_CODE) ==="
if [ -f "$OUTFILE" ]; then
    LINES=$(wc -l < "$OUTFILE")
    echo "Output: $OUTFILE ($LINES lines)"
else
    echo "ERROR: Output file not found!"
    LINES=0
fi

# Final upload to GCS
DEST="$GCS_DEST/worker_${WORKER_ID}_s${SEED}.jsonl"
if [ -f "$OUTFILE" ] && [ "$LINES" -gt 0 ]; then
    gcloud storage cp "$OUTFILE" "$DEST"
    echo "Uploaded to $DEST"
else
    echo "WARNING: No data to upload"
fi
gcloud storage cp "$LOG" "$GCS_DEST/logs/worker_${WORKER_ID}.log"

echo ""
echo "Worker $WORKER_ID DONE at $(date -u). $LINES hands generated. Self-deleting..."

# Self-delete VM
ZONE=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/zone" -H "Metadata-Flavor: Google" | awk -F/ '{print $NF}')
INSTANCE=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/name" -H "Metadata-Flavor: Google")
gcloud compute instances delete "$INSTANCE" --zone="$ZONE" --quiet
