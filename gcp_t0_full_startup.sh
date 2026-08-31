#!/bin/bash
# GCP T0 Full Batch Startup Script (no PolicyNet pre-filter)
# Each VM generates a range of hands with all 232 placements evaluated
# nesting=5,3,1, samples=30

set -e

BUCKET="gs://ofc-solver-485418"
RESULT_DIR="t0_full_results_v2"
HANDS_PER_VM=25
SAMPLES=30
NESTING="5,3,1"

# Get VM index from hostname (e.g., t0-full-00 -> 0)
VM_NAME=$(hostname)
VM_IDX_RAW=$(echo "$VM_NAME" | grep -oP '\d+$')
VM_IDX=$((10#$VM_IDX_RAW))  # Force decimal parsing (fix 08/09 octal issue)
SEED=$((VM_IDX * 1000 + 12345))

echo "=== T0 Full Batch Worker ==="
echo "VM: $VM_NAME | Index: $VM_IDX | Hands: $HANDS_PER_VM | Seed: $SEED"
echo "Nesting: $NESTING | Samples: $SAMPLES"

# Install build tools
apt-get update -qq && apt-get install -y -qq build-essential pkg-config git > /dev/null 2>&1

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
export PATH="/root/.cargo/bin:$PATH"

# Clone repo
cd /tmp
git clone https://github.com/giovinco0807/pineapple.git
cd pineapple/ai/rust_solver

# Build
echo "Building solver..."
/root/.cargo/bin/cargo build --release 2>&1

OUTPUT="/tmp/t0_full_shard_$(printf '%02d' $VM_IDX).jsonl"

echo "Running T0 batch evaluation..."
echo "Hands: $HANDS_PER_VM | Samples: $SAMPLES | Nesting: $NESTING"
./target/release/cfr_solver t0-batch \
    --hands $HANDS_PER_VM \
    --samples $SAMPLES \
    --output "$OUTPUT" \
    --seed $SEED \
    --nesting "$NESTING"

echo "Uploading results to GCS..."
gsutil cp "$OUTPUT" "${BUCKET}/${RESULT_DIR}/"

echo "=== Done! Shutting down... ==="
shutdown -h now
