#!/bin/bash
set -e

# ===== GCP T0 Filtered Batch Worker =====
# Each VM processes a 50-hand shard from the filtered_t0_v2.json file (500 hands / 10 VMs)
# Uses nesting=10,6,3 for high-fidelity evaluation (only 50 placements per hand)
# Result: uploaded to gs://ofc-solver-485418/t0_filtered_results_v2/

# Install dependencies
apt-get update -q && apt-get install -y -q curl gcc git python3

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
export PATH="/root/.cargo/bin:$PATH"

# Get VM index from hostname (e.g., t0-filt-00 -> 0)
VM_NAME=$(hostname)
VM_IDX_RAW=$(echo "$VM_NAME" | grep -oP '\d+$')
VM_IDX=$((10#$VM_IDX_RAW))  # Force decimal parsing (fix 08/09 octal issue)
SEED=$((VM_IDX * 1000 + 54321))

echo "=== T0 Filtered Batch Worker ==="
echo "VM_NAME=$VM_NAME VM_IDX=$VM_IDX SEED=$SEED"

# Clone repository
cd /tmp
git clone https://github.com/giovinco0807/pineapple.git ofc
cd ofc/ai/rust_solver

# Build solver
cargo build --release -p cfr_solver 2>&1 | tail -3

# Download filtered placements JSON from GCS
cd /tmp/ofc
gsutil cp gs://ofc-solver-485418/filtered_t0_v2.json /tmp/filtered_t0_v2.json

# Split: extract this VM's 50-hand chunk using python
SHARD_START=$((VM_IDX * 50))
SHARD_END=$((SHARD_START + 50))
python3 -c "
import json
data = json.load(open('/tmp/filtered_t0_v2.json'))
shard = data[$SHARD_START:$SHARD_END]
json.dump(shard, open('/tmp/filtered_shard.json', 'w'))
print(f'Shard: hands {$SHARD_START}-{$SHARD_END}, {len(shard)} hands')
"

OUTPUT="/tmp/t0_filtered_shard_$(printf '%02d' $VM_IDX).jsonl"

echo "Starting filtered batch evaluation..."
echo "Shard: $SHARD_START to $SHARD_END"
echo "Nesting: 10,6,3 | Samples: 30 | Seed: $SEED"
echo "Output: $OUTPUT"

# Run filtered batch evaluation
./ai/rust_solver/target/release/cfr_solver t0-batch-filtered \
    --input /tmp/filtered_shard.json \
    --output "$OUTPUT" \
    --samples 30 \
    --seed $SEED \
    --nesting "10,6,3"

echo "=== Evaluation complete ==="
echo "Uploading results to GCS..."
gsutil cp "$OUTPUT" "gs://ofc-solver-485418/t0_filtered_results_v2/"
echo "=== Upload complete ==="

# Self-terminate
echo "Shutting down..."
shutdown -h now
