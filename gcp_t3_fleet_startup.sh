#!/bin/bash
set -euxo pipefail
exec > >(tee -a /var/log/ofc-t3-startup.log) 2>&1
VM_INDEX=$(curl -fs -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/attributes/vm-index)
BASE=/home/Owner/ofc-pineapple
RESULT_DIR="$BASE/ai/data/t3_fleet/vm${VM_INDEX}"
GCS_BASE="gs://ofc-solver-results/t3_rust_fleet/run_20260509/vm${VM_INDEX}"
STATES_PER_CHUNK=1000
CHUNKS=3
BTN_SAMPLES=100
mkdir -p "$BASE/ai/data/t4_oracle_v2" "$RESULT_DIR"
cd "$BASE"
gsutil cp gs://ofc-solver-results/t3_rust_fleet/ofc_t3_rust_src.tar.gz /home/Owner/ofc_t3_rust_src.tar.gz
gsutil cp gs://ofc-solver-results/t3_rust_fleet/t4_oracle.onnx "$BASE/ai/data/t4_oracle_v2/t4_oracle.onnx"
tar xzf /home/Owner/ofc_t3_rust_src.tar.gz
sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y curl build-essential pkg-config libssl-dev python3-numpy
if ! command -v cargo >/dev/null 2>&1; then
  curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal
fi
cd "$BASE/ai/rust_solver"
python3 - <<'PY'
from pathlib import Path
p = Path('Cargo.toml')
s = p.read_text()
start = s.index('members = [')
end = s.index(']', start) + 1
s = s[:start] + 'members = ["ofc_core", "fl_solver", "t3_generator"]' + s[end:]
p.write_text(s)
PY
/home/Owner/.cargo/bin/cargo build -p t3_generator --release
cd "$BASE"
for PART in $(seq 0 $((CHUNKS - 1))); do
  CHUNK_DIR="$RESULT_DIR/chunk_${PART}"
  mkdir -p "$CHUNK_DIR"
  JSONL="$CHUNK_DIR/t3_vm${VM_INDEX}_chunk${PART}.jsonl"
  SEED=$((42000 + VM_INDEX * 1000 + PART))
  ./ai/rust_solver/target/release/t3_generator \
    --states "$STATES_PER_CHUNK" \
    --mode bb \
    --btn-samples "$BTN_SAMPLES" \
    --seed "$SEED" \
    --onnx-model ai/data/t4_oracle_v2/t4_oracle.onnx \
    --output "$JSONL" \
    --log-interval 100 2>&1 | tee "$CHUNK_DIR/generate.log"
  python3 ai/training/convert_t3_rust_jsonl.py \
    --input "$JSONL" \
    --output-dir "$CHUNK_DIR" \
    --chunk-size "$STATES_PER_CHUNK" 2>&1 | tee "$CHUNK_DIR/convert.log"
  gsutil -m cp "$CHUNK_DIR"/*.npz "$CHUNK_DIR"/*.log "$GCS_BASE/chunk_${PART}/"
done
echo "DONE vm=${VM_INDEX} states=$((STATES_PER_CHUNK * CHUNKS))" | tee "$RESULT_DIR/DONE"
gsutil cp "$RESULT_DIR/DONE" "$GCS_BASE/DONE"
