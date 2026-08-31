#!/bin/bash
set -euo pipefail

LOG=/tmp/t2_hu_oracle_startup.log
exec > >(tee -a "$LOG") 2>&1

PROJECT="ofc-solver-485418"
BUCKET="gs://ofc-solver-485418"
CODE_ARCHIVE="${BUCKET}/t2_hu_oracle/code/ofc_t2_hu_oracle_code.tar.gz"
MODEL_OBJECT="${BUCKET}/t2_oracle/models/t3_policyvalue_v2_best.pt"
RESULT_ROOT="${BUCKET}/t2_hu_oracle/results/run_20260509"
REMOTE_ROOT="/home/Owner/ofc-pineapple"

metadata() {
  curl -sf -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" || true
}

WORKER_ID="$(metadata worker-id)"
MAX_SECONDS="$(metadata max-seconds)"
N_SAMPLES="$(metadata n-samples)"
CHUNK_SIZE="$(metadata chunk-size)"
STATES="$(metadata states)"
POSITION="$(metadata position)"

WORKER_ID="${WORKER_ID:-0}"
MAX_SECONDS="${MAX_SECONDS:-10800}"
N_SAMPLES="${N_SAMPLES:-30}"
CHUNK_SIZE="${CHUNK_SIZE:-1000}"
STATES="${STATES:-100000000}"
POSITION="${POSITION:-bb}"

echo "worker_id=${WORKER_ID} position=${POSITION} max_seconds=${MAX_SECONDS} n_samples=${N_SAMPLES} chunk_size=${CHUNK_SIZE}"

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv python3-pip

mkdir -p "$REMOTE_ROOT"
cd "$REMOTE_ROOT"

gcloud storage cp "$CODE_ARCHIVE" /tmp/ofc_t2_hu_oracle_code.tar.gz
tar xzf /tmp/ofc_t2_hu_oracle_code.tar.gz -C "$REMOTE_ROOT"
rm -f /tmp/ofc_t2_hu_oracle_code.tar.gz

mkdir -p ai/data/t3_oracle_rust_discard_sensitive_ft
gcloud storage cp "$MODEL_OBJECT" ai/data/t3_oracle_rust_discard_sensitive_ft/t3_policyvalue_v2_best.pt

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy torch --index-url https://download.pytorch.org/whl/cpu

OUT_DIR="ai/data/t2_hu_${POSITION}_worker_${WORKER_ID}"
GCS_OUT="${RESULT_ROOT}/${POSITION}/worker_${WORKER_ID}"
mkdir -p "$OUT_DIR"

python -u ai/training/generate_t2_hu_oracle.py \
  --position "$POSITION" \
  --t3-model ai/data/t3_oracle_rust_discard_sensitive_ft/t3_policyvalue_v2_best.pt \
  --save-dir "$OUT_DIR" \
  --states "$STATES" \
  --n-samples "$N_SAMPLES" \
  --chunk-size "$CHUNK_SIZE" \
  --max-seconds "$MAX_SECONDS" \
  --gcs-output "$GCS_OUT" \
  --device cpu \
  --seed "$((20260509 + WORKER_ID))" \
  2>&1 | tee "t2_hu_${POSITION}_worker_${WORKER_ID}.log"

gcloud storage cp "t2_hu_${POSITION}_worker_${WORKER_ID}.log" "${GCS_OUT}/t2_hu_${POSITION}_worker_${WORKER_ID}.log" || true
gcloud storage cp "$LOG" "${GCS_OUT}/startup_${WORKER_ID}.log" || true

INSTANCE_NAME="$(curl -sf -H Metadata-Flavor:Google http://metadata.google.internal/computeMetadata/v1/instance/name || true)"
ZONE_PATH="$(curl -sf -H Metadata-Flavor:Google http://metadata.google.internal/computeMetadata/v1/instance/zone || true)"
ZONE="${ZONE_PATH##*/}"
if [[ -n "$INSTANCE_NAME" && -n "$ZONE" ]]; then
  gcloud compute instances delete "$INSTANCE_NAME" --zone="$ZONE" --project="$PROJECT" --quiet || true
fi
