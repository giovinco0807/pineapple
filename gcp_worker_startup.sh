#!/bin/bash
# GCP Worker Startup Script - runs as root on VM boot
set -e

export HOME="/root"

WORKER_ID=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/worker-id" -H "Metadata-Flavor: Google")
SEED=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/seed" -H "Metadata-Flavor: Google")
HANDS=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/hands" -H "Metadata-Flavor: Google")
SAMPLES=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/samples" -H "Metadata-Flavor: Google")
NESTING=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/nesting" -H "Metadata-Flavor: Google")

LOG="/tmp/worker_${WORKER_ID}.log"
exec > >(tee -a "$LOG") 2>&1

echo "=========================================="
echo "Worker $WORKER_ID started at $(date -u)"
echo "Seed=$SEED Hands=$HANDS Samples=$SAMPLES Nesting=$NESTING"
echo "Cores: $(nproc) | HOME=$HOME"
echo "=========================================="

# Install dependencies
apt-get update -qq
apt-get install -y -qq git build-essential curl 2>&1 | tail -1

# Install Rust (as root, explicitly set CARGO_HOME/RUSTUP_HOME)
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

# Run batch
OUTFILE="/tmp/t0_worker_${WORKER_ID}.jsonl"
echo ""
echo "=== Starting evaluation ==="
echo "Command: $EXE t0-batch --hands $HANDS --samples $SAMPLES --nesting $NESTING --output $OUTFILE --seed $SEED"
$EXE t0-batch \
    --hands "$HANDS" \
    --samples "$SAMPLES" \
    --nesting "$NESTING" \
    --output "$OUTFILE" \
    --seed "$SEED"

echo ""
echo "=== Evaluation complete ==="
LINES=$(wc -l < "$OUTFILE")
echo "Output: $OUTFILE ($LINES lines)"

# Upload results to GCS
DEST="gs://ofc-solver-485418/t0_training/worker_${WORKER_ID}_s${SEED}.jsonl"
gcloud storage cp "$OUTFILE" "$DEST"
gcloud storage cp "$LOG" "gs://ofc-solver-485418/t0_training/logs/worker_${WORKER_ID}.log"
echo "Uploaded to $DEST"

echo ""
echo "Worker $WORKER_ID DONE at $(date -u). Self-deleting..."

# Self-delete VM
ZONE=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/zone" -H "Metadata-Flavor: Google" | awk -F/ '{print $NF}')
INSTANCE=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/name" -H "Metadata-Flavor: Google")
gcloud compute instances delete "$INSTANCE" --zone="$ZONE" --quiet
