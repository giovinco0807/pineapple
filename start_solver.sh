#!/bin/bash
# Run on GCP instance to start Phase F solver
# Usage: bash /tmp/start_solver.sh <SHARD_ID>
set -e

export PATH=/usr/bin:/usr/local/bin:/usr/sbin:/sbin:/root/.cargo/bin
export HOME=/root
source /root/.cargo/env 2>/dev/null || true

SHARD_ID=${1:-00}
GCS_DIR="gs://ofc-solver-485418/t0_phase_f"
SHARD_FILE="shard_${SHARD_ID}.jsonl"
OUTPUT_FILE="phase_f_${SHARD_ID}.jsonl"
LOG_FILE="solver_f_${SHARD_ID}.log"
BRANCH="verify-gcp-phase-one-20260501"
SAMPLES=30
NESTING="10,6,3"
SEED=$((42 + 10#$SHARD_ID))

echo "=== Phase F Solver Start ==="
echo "Shard: $SHARD_ID"
echo "Nesting: $NESTING, Samples: $SAMPLES, Seed: $SEED"

# Kill existing solver
pkill -f cfr_solver 2>/dev/null || true
sleep 1

# Git pull
git config --global --add safe.directory /tmp/ofc-pineapple
cd /tmp/ofc-pineapple
git fetch origin
git checkout $BRANCH
git reset --hard origin/$BRANCH

# Build
cd /tmp/ofc-pineapple/ai/rust_solver
cargo build --release -p cfr_solver 2>&1 | tail -2
EXE="./target/release/cfr_solver"
echo "Binary: $(ls -lh $EXE | awk '{print $5}')"

# Download shard
echo "Downloading shard from GCS..."
gcloud storage cp "${GCS_DIR}/shards/${SHARD_FILE}" "/tmp/${SHARD_FILE}"
N_HANDS=$(wc -l < "/tmp/${SHARD_FILE}")
echo "Shard loaded: $N_HANDS hands"

# Start solver
echo "Starting solver..."
nohup $EXE t0-batch-filtered \
    --input "/tmp/${SHARD_FILE}" \
    --samples $SAMPLES \
    --nesting $NESTING \
    --output "/tmp/${OUTPUT_FILE}" \
    --seed $SEED \
    > "/tmp/${LOG_FILE}" 2>&1 &
SOLVER_PID=$!
echo "Solver PID: $SOLVER_PID"

# Start auto-upload loop
nohup bash -c "
while kill -0 $SOLVER_PID 2>/dev/null; do
    sleep 300
    if [ -f /tmp/${OUTPUT_FILE} ]; then
        gcloud storage cp /tmp/${OUTPUT_FILE} ${GCS_DIR}/results/${OUTPUT_FILE} 2>/dev/null || true
        gcloud storage cp /tmp/${LOG_FILE} ${GCS_DIR}/logs/${LOG_FILE} 2>/dev/null || true
    fi
done
# Final upload
gcloud storage cp /tmp/${OUTPUT_FILE} ${GCS_DIR}/results/${OUTPUT_FILE} 2>/dev/null || true
gcloud storage cp /tmp/${LOG_FILE} ${GCS_DIR}/logs/${LOG_FILE} 2>/dev/null || true
echo 'Upload complete'
" > /dev/null 2>&1 &

echo "Auto-upload loop started"
echo "=== Setup Complete ==="
