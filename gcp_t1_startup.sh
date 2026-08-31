#!/bin/bash
# GCP Worker Startup Script - T1 MC Data Generation

export HOME="/root"

# Ensure self-deletion on exit (success or error)
cleanup() {
    echo ""
    echo "Worker $WORKER_ID finishing at $(date -u). Self-deleting..."
    ZONE=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/zone" -H "Metadata-Flavor: Google" | awk -F/ '{print $NF}')
    INSTANCE=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/name" -H "Metadata-Flavor: Google")
    gcloud compute instances delete "$INSTANCE" --zone="$ZONE" --quiet
}
trap cleanup EXIT

WORKER_ID=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/worker-id" -H "Metadata-Flavor: Google")
SEED=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/seed" -H "Metadata-Flavor: Google")
HANDS=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/hands" -H "Metadata-Flavor: Google")
N1=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/n1" -H "Metadata-Flavor: Google")
SAMPLES=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/samples" -H "Metadata-Flavor: Google")
NESTING=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/nesting" -H "Metadata-Flavor: Google")
GCS_DEST=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/gcs-dest" -H "Metadata-Flavor: Google")
GIT_BRANCH=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/git-branch" -H "Metadata-Flavor: Google" 2>/dev/null || echo "verify-gcp-phase-one-20260501")

LOG="/tmp/worker_${WORKER_ID}.log"
exec > >(tee -a "$LOG") 2>&1

echo "=========================================="
echo "T1 Worker $WORKER_ID started at $(date -u)"
echo "Hands=$HANDS N1=$N1 Samples=$SAMPLES Nesting=$NESTING Seed=$SEED"
echo "Cores: $(nproc) | HOME=$HOME"
echo "GCS Output: $GCS_DEST"
echo "Branch: $GIT_BRANCH"
echo "=========================================="

# Install dependencies
apt-get update -qq
apt-get install -y -qq git build-essential curl python3-pip python3-venv 2>&1 | tail -1

# Install Rust
export RUSTUP_HOME="/root/.rustup"
export CARGO_HOME="/root/.cargo"
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "/root/.cargo/env"

echo "Rust: $(rustc --version)"
echo "Cargo: $(cargo --version)"

# Clone repo
cd /tmp
git clone --depth 1 --branch "$GIT_BRANCH" https://github.com/giovinco0807/pineapple.git ofc-pineapple 2>&1 | tail -2
cd ofc-pineapple

# Set up Python
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt 2>&1 | tail -2
pip install torch numpy --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -2

# Download T0 NN model to ai/models/
mkdir -p ai/models
gcloud storage cp "gs://ofc-solver-485418/models/t0_placement_net_v4.pt" ai/models/t0_placement_net_v4.pt || true

# Build Rust binary first so it doesn't build inside the Python script output loop
echo "Building Rust solver..."
cd rust_solver
cargo build --release -p cfr_solver 2>&1 | tail -3
cd ..

# Run generation
OUTFILE="/tmp/t1_worker_${WORKER_ID}.jsonl"
echo ""
echo "=== Starting T1 Generation ==="

# We run python ai/generate_t1_mc_rust.py
python3 ai/generate_t1_mc_rust.py \
    --n-hands "$HANDS" \
    --n1 "$N1" \
    --samples "$SAMPLES" \
    --nesting "$NESTING" \
    --output "$OUTFILE" \
    --seed "$SEED" \
    --tmp-in "/tmp/rust_in_${WORKER_ID}.jsonl" \
    --tmp-out "/tmp/rust_out_${WORKER_ID}.jsonl"

EXIT_CODE=$?

echo ""
echo "=== Generation complete (exit=$EXIT_CODE) ==="
if [ -f "$OUTFILE" ]; then
    LINES=$(wc -l < "$OUTFILE")
    echo "Output: $OUTFILE ($LINES lines)"
else
    echo "ERROR: Output file not found!"
    LINES=0
fi

# Upload to GCS
DEST="$GCS_DEST/t1_worker_${WORKER_ID}_s${SEED}.jsonl"
if [ -f "$OUTFILE" ] && [ "$LINES" -gt 0 ]; then
    gcloud storage cp "$OUTFILE" "$DEST"
    echo "Uploaded to $DEST"
else
    echo "WARNING: No data to upload"
fi
gcloud storage cp "$LOG" "$GCS_DEST/logs/worker_${WORKER_ID}.log"
