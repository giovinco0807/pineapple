#!/bin/bash
set -euo pipefail

STARTUP_LOG="/tmp/branch_expanded_t3_startup.log"
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
ROOTS="$(metadata_attr roots)"
MAX_OUTPUTS="$(metadata_attr max-outputs)"
T0_TOP_K="$(metadata_attr t0-top-k)"
REGULAR_TOP_K="$(metadata_attr regular-top-k)"
POSITION="$(metadata_attr position)"
SEED_BASE="$(metadata_attr seed-base)"
BATCH_SIZE="$(metadata_attr batch-size)"
DEVICE="$(metadata_attr device)"
RAYON_THREADS="$(metadata_attr rayon-threads)"
UPLOAD_INTERVAL="$(metadata_attr upload-interval)"
SELF_DELETE="$(metadata_attr self-delete)"

BUCKET="${BUCKET:-gs://pokerhu-ofc-solver-485418-training}"
RUN_ID="${RUN_ID:-branch-t0t2-top10-100k-$(date +%Y%m%d-%H%M%S)}"
WORKER_ID="${WORKER_ID:-0}"
CODE_ARCHIVE="${CODE_ARCHIVE:-${BUCKET}/runs/${RUN_ID}/code/ofc_branch_expanded_t3_code.tar.gz}"
ROOTS="${ROOTS:-6}"
MAX_OUTPUTS="${MAX_OUTPUTS:-10000}"
T0_TOP_K="${T0_TOP_K:-10}"
REGULAR_TOP_K="${REGULAR_TOP_K:-10}"
POSITION="${POSITION:-both}"
SEED_BASE="${SEED_BASE:-2026060500}"
BATCH_SIZE="${BATCH_SIZE:-512}"
DEVICE="${DEVICE:-cpu}"
RAYON_THREADS="${RAYON_THREADS:-16}"
UPLOAD_INTERVAL="${UPLOAD_INTERVAL:-180}"
SELF_DELETE="${SELF_DELETE:-true}"

REMOTE_ROOT="/opt/ofc-pineapple"
OUT_ROOT="${REMOTE_ROOT}/branch_expanded_t3"
NAME="worker_${WORKER_ID}"
INPUT_PATH="${OUT_ROOT}/${NAME}.t3_inputs.jsonl"
EXACT_PATH="${OUT_ROOT}/${NAME}.t3_exact.rust.jsonl"
TEACHER_PATH="${OUT_ROOT}/${NAME}.t3_teacher.jsonl"
RERANKER_DIR="${OUT_ROOT}/${NAME}.reranker"
STATUS_FILE="${OUT_ROOT}/${NAME}.status.txt"
RUN_LOG="${OUT_ROOT}/${NAME}.run.log"
GCS_OUT="${BUCKET}/runs/${RUN_ID}/results/${NAME}"
SEED=$((SEED_BASE + WORKER_ID * 1009))

upload_once() {
  set +e
  if command -v gcloud >/dev/null 2>&1; then
    [ -f "$STATUS_FILE" ] && gcloud storage cp -q "$STATUS_FILE" "${GCS_OUT}/${NAME}.status.txt"
    [ -f "$RUN_LOG" ] && gcloud storage cp -q "$RUN_LOG" "${GCS_OUT}/${NAME}.run.log"
    [ -f "$STARTUP_LOG" ] && gcloud storage cp -q "$STARTUP_LOG" "${GCS_OUT}/startup.log"
    [ -f "$INPUT_PATH" ] && gcloud storage cp -q "$INPUT_PATH" "${GCS_OUT}/${NAME}.t3_inputs.jsonl"
    [ -f "${INPUT_PATH%.jsonl}.summary.json" ] && gcloud storage cp -q "${INPUT_PATH%.jsonl}.summary.json" "${GCS_OUT}/${NAME}.t3_inputs.summary.json"
    [ -f "$EXACT_PATH" ] && gcloud storage cp -q "$EXACT_PATH" "${GCS_OUT}/${NAME}.t3_exact.rust.jsonl"
    [ -f "$TEACHER_PATH" ] && gcloud storage cp -q "$TEACHER_PATH" "${GCS_OUT}/${NAME}.t3_teacher.jsonl"
    [ -f "${TEACHER_PATH%.jsonl}.summary.json" ] && gcloud storage cp -q "${TEACHER_PATH%.jsonl}.summary.json" "${GCS_OUT}/${NAME}.t3_teacher.summary.json"
    if [ -d "$RERANKER_DIR" ]; then
      tar czf "/tmp/${NAME}.reranker.tar.gz" -C "$OUT_ROOT" "${NAME}.reranker"
      gcloud storage cp -q "/tmp/${NAME}.reranker.tar.gz" "${GCS_OUT}/${NAME}.reranker.tar.gz"
    fi
  fi
  set -e
}

run_with_heartbeat() {
  local label="$1"
  shift
  set +e
  "$@" 2>&1 | tee -a "$RUN_LOG" &
  local pid=$!
  set -e
  while kill -0 "$pid" 2>/dev/null; do
    sleep "$UPLOAD_INTERVAL"
    {
      echo "heartbeat_at=$(date -Is)"
      echo "current_step=${label}"
      [ -f "$INPUT_PATH" ] && echo "input_records=$(wc -l < "$INPUT_PATH" || echo 0)"
      [ -f "$EXACT_PATH" ] && echo "exact_records=$(wc -l < "$EXACT_PATH" || echo 0)"
      [ -f "$TEACHER_PATH" ] && echo "teacher_records=$(wc -l < "$TEACHER_PATH" || echo 0)"
    } >> "$STATUS_FILE"
    upload_once
  done
  set +e
  wait "$pid"
  local status=$?
  set -e
  return "$status"
}

trap upload_once EXIT

mkdir -p "$REMOTE_ROOT" "$OUT_ROOT"
{
  echo "started_at=$(date -Is)"
  echo "project=${PROJECT}"
  echo "run_id=${RUN_ID}"
  echo "worker_id=${WORKER_ID}"
  echo "gcs_out=${GCS_OUT}"
  echo "roots=${ROOTS}"
  echo "max_outputs=${MAX_OUTPUTS}"
  echo "t0_top_k=${T0_TOP_K}"
  echo "regular_top_k=${REGULAR_TOP_K}"
  echo "position=${POSITION}"
  echo "seed=${SEED}"
} > "$STATUS_FILE"

echo "run_id=${RUN_ID} worker=${WORKER_ID} roots=${ROOTS} max_outputs=${MAX_OUTPUTS}"
echo "code_archive=${CODE_ARCHIVE}"

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y python3 python3-venv python3-pip curl build-essential pkg-config libssl-dev

if [ ! -x "/root/.cargo/bin/cargo" ]; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
fi
set +u
source /root/.cargo/env
set -u

cd "$REMOTE_ROOT"
gcloud storage cp "$CODE_ARCHIVE" /tmp/ofc_branch_expanded_t3_code.tar.gz
tar xzf /tmp/ofc_branch_expanded_t3_code.tar.gz -C "$REMOTE_ROOT"
rm -f /tmp/ofc_branch_expanded_t3_code.tar.gz

python3 -m venv "${REMOTE_ROOT}/.venv"
source "${REMOTE_ROOT}/.venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install numpy
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

cd "${REMOTE_ROOT}/ai/rust_solver"
cargo build --release -p t3_exact_solver

cd "$REMOTE_ROOT"
export PYTHONUNBUFFERED=1
export RAYON_NUM_THREADS="$RAYON_THREADS"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

{
  echo "python=$(command -v python)"
  echo "rust_solver_built_at=$(date -Is)"
} >> "$STATUS_FILE"
upload_once

run_with_heartbeat generate_inputs \
  python -u -m ai.tutor.build_branch_expanded_t3_targets \
    --roots "$ROOTS" \
    --position "$POSITION" \
    --t0-top-k "$T0_TOP_K" \
    --regular-top-k "$REGULAR_TOP_K" \
    --max-outputs "$MAX_OUTPUTS" \
    --seed "$SEED" \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --output "$INPUT_PATH"

{
  echo "generated_at=$(date -Is)"
  echo "input_records=$(wc -l < "$INPUT_PATH" || echo 0)"
} >> "$STATUS_FILE"
upload_once

run_with_heartbeat exact_t3 \
  "${REMOTE_ROOT}/ai/rust_solver/target/release/t3_exact_solver" \
    --input "$INPUT_PATH" \
    --output "$EXACT_PATH" \
    --top-n 27

{
  echo "exact_finished_at=$(date -Is)"
  echo "exact_records=$(wc -l < "$EXACT_PATH" || echo 0)"
} >> "$STATUS_FILE"
upload_once

run_with_heartbeat convert_teacher \
  python -u -m ai.tutor.convert_rust_t3_exact_teacher \
    --input "$INPUT_PATH" \
    --rust-output "$EXACT_PATH" \
    --output "$TEACHER_PATH"

run_with_heartbeat convert_reranker \
  python -u -m ai.training.convert_action_value_teacher "$TEACHER_PATH" \
    --output "$RERANKER_DIR" \
    --turns 3 \
    --state-dim 520 \
    --regular-max-candidates 0

{
  echo "finished_at=$(date -Is)"
  echo "teacher_records=$(wc -l < "$TEACHER_PATH" || echo 0)"
  [ -f "$RERANKER_DIR/metadata.json" ] && echo "reranker_metadata=1"
  echo "generator_exit_status=0"
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
