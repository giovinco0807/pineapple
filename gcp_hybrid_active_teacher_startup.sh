#!/bin/bash
set -euo pipefail

STARTUP_LOG="/tmp/hybrid_active_teacher_startup.log"
exec > >(tee -a "$STARTUP_LOG") 2>&1
export HOME="${HOME:-/root}"

metadata_attr() {
  curl -sf -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" || true
}

metadata_inst() {
  curl -sf -H "Metadata-Flavor: Google" \
    "http://metadata.google.internal/computeMetadata/v1/instance/$1" || true
}

PROJECT="$(curl -sf -H "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/project/project-id" || echo "ofc-solver-485418")"
BUCKET="$(metadata_attr bucket)"
RUN_ID="$(metadata_attr run-id)"
WORKER_ID="$(metadata_attr worker-id)"
CODE_ARCHIVE="$(metadata_attr code-archive)"
TARGET_SHARD="$(metadata_attr target-shard)"
SIMS="$(metadata_attr sims)"
TURNS="$(metadata_attr turns)"
MC_TURNS="$(metadata_attr mc-turns)"
RAYON_THREADS="$(metadata_attr rayon-threads)"
UPLOAD_INTERVAL="$(metadata_attr upload-interval)"
BATCH_TIMEOUT="$(metadata_attr batch-timeout)"
SELF_DELETE="$(metadata_attr self-delete)"

BUCKET="${BUCKET:-gs://pokerhu-ofc-solver-485418-training}"
RUN_ID="${RUN_ID:-hybrid-active-t1t2-$(date +%Y%m%d-%H%M%S)}"
WORKER_ID="${WORKER_ID:-0}"
CODE_ARCHIVE="${CODE_ARCHIVE:-${BUCKET}/runs/${RUN_ID}/code/ofc_hybrid_active_teacher_code.tar.gz}"
TARGET_SHARD="${TARGET_SHARD:-${BUCKET}/runs/${RUN_ID}/inputs/targets_shard_00.jsonl}"
SIMS="${SIMS:-300}"
TURNS="${TURNS:-1,2}"
MC_TURNS="${MC_TURNS:-1,2}"
RAYON_THREADS="${RAYON_THREADS:-16}"
UPLOAD_INTERVAL="${UPLOAD_INTERVAL:-180}"
BATCH_TIMEOUT="${BATCH_TIMEOUT:-86400}"
SELF_DELETE="${SELF_DELETE:-true}"
TURNS="${TURNS//;/,}"
MC_TURNS="${MC_TURNS//;/,}"

REMOTE_ROOT="/opt/ofc-pineapple"
OUT_DIR="${REMOTE_ROOT}/teacher_results"
NAME="worker_${WORKER_ID}"
INPUT_PATH="${OUT_DIR}/${NAME}.targets.jsonl"
OUTPUT_PATH="${OUT_DIR}/${NAME}.teacher.jsonl"
STATUS_FILE="${OUT_DIR}/${NAME}.status.txt"
GCS_OUT="${BUCKET}/runs/${RUN_ID}/results/${NAME}"

upload_once() {
  set +e
  if command -v gcloud >/dev/null 2>&1; then
    [ -f "$INPUT_PATH" ] && gcloud storage cp -q "$INPUT_PATH" "${GCS_OUT}/${NAME}.targets.jsonl"
    [ -f "$OUTPUT_PATH" ] && gcloud storage cp -q "$OUTPUT_PATH" "${GCS_OUT}/${NAME}.teacher.jsonl"
    [ -f "${OUTPUT_PATH%.jsonl}.summary.json" ] && gcloud storage cp -q "${OUTPUT_PATH%.jsonl}.summary.json" "${GCS_OUT}/${NAME}.summary.json"
    [ -f "${OUTPUT_PATH%.jsonl}.batch_requests.jsonl" ] && gcloud storage cp -q "${OUTPUT_PATH%.jsonl}.batch_requests.jsonl" "${GCS_OUT}/${NAME}.batch_requests.jsonl"
    [ -f "${OUTPUT_PATH%.jsonl}.batch_responses.jsonl" ] && gcloud storage cp -q "${OUTPUT_PATH%.jsonl}.batch_responses.jsonl" "${GCS_OUT}/${NAME}.batch_responses.jsonl"
    [ -f "${OUT_DIR}/${NAME}.log" ] && gcloud storage cp -q "${OUT_DIR}/${NAME}.log" "${GCS_OUT}/${NAME}.log"
    [ -f "$STATUS_FILE" ] && gcloud storage cp -q "$STATUS_FILE" "${GCS_OUT}/${NAME}.status.txt"
    [ -f "$STARTUP_LOG" ] && gcloud storage cp -q "$STARTUP_LOG" "${GCS_OUT}/startup.log"
  fi
  set -e
}

trap upload_once EXIT

echo "run_id=${RUN_ID} worker_id=${WORKER_ID} sims=${SIMS} turns=${TURNS} mc_turns=${MC_TURNS}"
echo "bucket=${BUCKET}"
echo "code_archive=${CODE_ARCHIVE}"
echo "target_shard=${TARGET_SHARD}"

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y python3 python3-venv python3-pip curl build-essential pkg-config

if [ ! -x "/root/.cargo/bin/cargo" ]; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
set +u
source /root/.cargo/env
set -u

mkdir -p "$REMOTE_ROOT" "$OUT_DIR"
cd "$REMOTE_ROOT"

gcloud storage cp "$CODE_ARCHIVE" /tmp/ofc_hybrid_active_teacher_code.tar.gz
tar xzf /tmp/ofc_hybrid_active_teacher_code.tar.gz -C "$REMOTE_ROOT"
rm -f /tmp/ofc_hybrid_active_teacher_code.tar.gz

gcloud storage cp "$TARGET_SHARD" "$INPUT_PATH"

python3 -m venv "${REMOTE_ROOT}/.venv"
source "${REMOTE_ROOT}/.venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install numpy

cd "${REMOTE_ROOT}/ai/rust_solver"
cargo build --release -p prob_engine

cd "$REMOTE_ROOT"
export PYTHONUNBUFFERED=1
export RAYON_NUM_THREADS="$RAYON_THREADS"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PROB_ENGINE_TIMEOUT="$BATCH_TIMEOUT"

{
  echo "started_at=$(date -Is)"
  echo "project=${PROJECT}"
  echo "input_records=$(wc -l < "$INPUT_PATH" || echo 0)"
  echo "gcs_out=${GCS_OUT}"
} > "$STATUS_FILE"

set +e
python -u -m ai.training.generate_active_teacher "$INPUT_PATH" \
  --output "$OUTPUT_PATH" \
  --turns "$TURNS" \
  --mc-turns "$MC_TURNS" \
  --sims "$SIMS" \
  --batch-engine \
  --engine-path "${REMOTE_ROOT}/ai/rust_solver/target/release/prob_engine" \
  --batch-timeout "$BATCH_TIMEOUT" \
  --print-errors \
  2>&1 | tee "${OUT_DIR}/${NAME}.log" &
GEN_PID=$!
set -e

while kill -0 "$GEN_PID" 2>/dev/null; do
  sleep "$UPLOAD_INTERVAL"
  {
    echo "heartbeat_at=$(date -Is)"
    [ -f "${OUTPUT_PATH%.jsonl}.batch_responses.jsonl" ] && echo "batch_responses=$(wc -l < "${OUTPUT_PATH%.jsonl}.batch_responses.jsonl")"
    [ -f "$OUTPUT_PATH" ] && echo "teacher_records=$(wc -l < "$OUTPUT_PATH")"
  } >> "$STATUS_FILE"
  upload_once
done

set +e
wait "$GEN_PID"
GEN_STATUS=$?
set -e

{
  echo "finished_at=$(date -Is)"
  echo "generator_exit_status=${GEN_STATUS}"
  [ -f "${OUTPUT_PATH%.jsonl}.summary.json" ] && echo "summary_present=1"
  [ -f "$OUTPUT_PATH" ] && echo "teacher_records=$(wc -l < "$OUTPUT_PATH")"
} >> "$STATUS_FILE"

upload_once

if [ "$SELF_DELETE" = "true" ]; then
  INSTANCE_NAME="$(metadata_inst name)"
  ZONE_PATH="$(metadata_inst zone)"
  ZONE="${ZONE_PATH##*/}"
  if [ -n "$INSTANCE_NAME" ] && [ -n "$ZONE" ]; then
    gcloud compute instances delete "$INSTANCE_NAME" --zone="$ZONE" --project="$PROJECT" --quiet || true
  fi
  shutdown -h now || true
fi

exit "$GEN_STATUS"
