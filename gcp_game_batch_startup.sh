#!/bin/bash
set -e

# ===== GCP Game Batch Worker (T1-T4 Evaluation) =====
# Each VM processes a shard of game scenarios (500 games per VM)
# Evaluates all T1-T4 actions with nested Monte Carlo
# Result: uploaded to gs://ofc-solver-485418/game_batch_results/

# Install dependencies
apt-get update -q && apt-get install -y -q curl gcc git python3

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
export PATH="/root/.cargo/bin:$PATH"

# Get VM index from hostname (e.g., game-batch-00 -> 0)
VM_NAME=$(hostname)
VM_IDX_RAW=$(echo "$VM_NAME" | grep -oP '\d+$')
VM_IDX=$((10#$VM_IDX_RAW))  # Force decimal parsing (fix 08/09 octal issue)
SEED=$((VM_IDX * 1000000 + 12345))

echo "=== Game Batch Worker (T1-T4) ==="
echo "VM_NAME=$VM_NAME VM_IDX=$VM_IDX SEED=$SEED"

# Clone repository
cd /tmp
git clone https://github.com/giovinco0807/pineapple.git ofc
cd ofc/ai/rust_solver

# Build solver
cargo build --release -p cfr_solver 2>&1 | tail -3

# Download game scenarios from GCS (pre-sharded)
cd /tmp/ofc
gsutil cp "gs://ofc-solver-485418/game_batch_scenarios/shard_$(printf '%02d' $VM_IDX).json" /tmp/game_shard.json

N_GAMES=$(python3 -c "import json; print(len(json.load(open('/tmp/game_shard.json'))))")
echo "Shard has $N_GAMES games"

OUTPUT="/tmp/game_batch_shard_$(printf '%02d' $VM_IDX).jsonl"

echo "Starting game batch evaluation (T1-T4)..."
echo "Games: $N_GAMES | Nesting: 6,3,1 | Samples: 30 | Seed: $SEED"
echo "Output: $OUTPUT"

# Run game batch evaluation
./ai/rust_solver/target/release/cfr_solver game-batch \
    --input /tmp/game_shard.json \
    --output "$OUTPUT" \
    --samples 30 \
    --seed $SEED \
    --nesting "6,3,1"

echo "=== Evaluation complete ==="
echo "Uploading results to GCS..."
gsutil cp "$OUTPUT" "gs://ofc-solver-485418/game_batch_results/"
echo "=== Upload complete ==="

# Self-terminate
echo "Shutting down..."
shutdown -h now
