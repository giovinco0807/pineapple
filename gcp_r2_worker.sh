#!/bin/bash
# Round 2 restart script - runs ON the VM
# NOTE: chown must be done BEFORE this script runs (from SSH command directly)
# Args: VM_INDEX
set -euo pipefail
VM_IDX=$1
STATES_PER_CHUNK=1000
SEED_OFFSET=52000
BTN_SAMPLES=100
GCS_BASE="gs://ofc-solver-results/t3_rust_fleet/run_20260509"

cd /home/Owner/ofc-pineapple
export HOME=/home/Owner
source "$HOME/.cargo/env" 2>/dev/null || true

RESULT_DIR="/home/Owner/ofc-pineapple/ai/data/t3_fleet/vm${VM_IDX}"

for PART in 3 4 5; do
    CHUNK_DIR="$RESULT_DIR/chunk_${PART}"
    mkdir -p "$CHUNK_DIR"
    JSONL="$CHUNK_DIR/t3_vm${VM_IDX}_chunk${PART}.jsonl"
    SEED=$(( SEED_OFFSET + VM_IDX * 1000 + PART ))

    echo "=== VM${VM_IDX} chunk_${PART} (seed=${SEED}) ==="

    ./ai/rust_solver/target/release/t3_generator \
        --states $STATES_PER_CHUNK \
        --mode bb \
        --btn-samples $BTN_SAMPLES \
        --seed $SEED \
        --onnx-model ai/data/t4_oracle_v2/t4_oracle.onnx \
        --output $JSONL \
        --log-interval 100 2>&1 | tee "$CHUNK_DIR/generate.log"

    python3 ai/training/convert_t3_rust_jsonl.py \
        --input $JSONL \
        --output-dir $CHUNK_DIR \
        --chunk-size $STATES_PER_CHUNK 2>&1 | tee "$CHUNK_DIR/convert.log"

    gsutil -m cp "$CHUNK_DIR"/*.npz "$CHUNK_DIR"/*.log \
        "${GCS_BASE}/vm${VM_IDX}/chunk_${PART}/" 2>/dev/null || true
done

echo "DONE_R2 vm=${VM_IDX} states=$((STATES_PER_CHUNK * 3))" | tee "$RESULT_DIR/DONE_R2"
gsutil cp "$RESULT_DIR/DONE_R2" "${GCS_BASE}/vm${VM_IDX}/DONE_R2"
echo "=== Round 2 complete for VM${VM_IDX} ==="
