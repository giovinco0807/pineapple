#!/bin/bash
set -euo pipefail

STARTUP_LOG="/tmp/joint_t0_teacher_startup.log"
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
HANDS="$(metadata_attr hands)"
SEED="$(metadata_attr seed)"
SIMS="$(metadata_attr sims)"
BEAM="$(metadata_attr beam)"
CHILD_SIMS="$(metadata_attr child-sims)"
T0_POOL_SIZE="$(metadata_attr t0-pool-size)"
TRACE_ROLLOUTS="$(metadata_attr trace-rollouts)"
TRACE_SIMS="$(metadata_attr trace-sims)"
TRACE_POOL_SIZES="$(metadata_attr trace-pool-sizes)"
FL_ANY_WEIGHT="$(metadata_attr fl-any-weight)"
FL_QQ_WEIGHT="$(metadata_attr fl-qq-weight)"
FL_KK_WEIGHT="$(metadata_attr fl-kk-weight)"
FL_AA_WEIGHT="$(metadata_attr fl-aa-weight)"
FL_TRIPS_WEIGHT="$(metadata_attr fl-trips-weight)"
EV_WEIGHT="$(metadata_attr ev-weight)"
BUST_WEIGHT="$(metadata_attr bust-weight)"
RAYON_THREADS="$(metadata_attr rayon-threads)"
PROB_ENGINE_TIMEOUT="$(metadata_attr prob-engine-timeout)"
UPLOAD_INTERVAL="$(metadata_attr upload-interval)"
SELF_DELETE="$(metadata_attr self-delete)"

BUCKET="${BUCKET:-gs://pokerhu-ofc-solver-485418-training}"
RUN_ID="${RUN_ID:-joint-t0-aa-focus-$(date +%Y%m%d-%H%M%S)}"
WORKER_ID="${WORKER_ID:-0}"
CODE_ARCHIVE="${CODE_ARCHIVE:-${BUCKET}/runs/${RUN_ID}/code/ofc_joint_t0_teacher_code.tar.gz}"
HANDS="${HANDS:-1}"
SEED="${SEED:-20260521}"
SIMS="${SIMS:-100}"
BEAM="${BEAM:-10}"
CHILD_SIMS="${CHILD_SIMS:-2}"
T0_POOL_SIZE="${T0_POOL_SIZE:-30}"
TRACE_ROLLOUTS="${TRACE_ROLLOUTS:-1}"
TRACE_SIMS="${TRACE_SIMS:-32}"
TRACE_POOL_SIZES="${TRACE_POOL_SIZES:-0;15;10}"
TRACE_POOL_SIZES="${TRACE_POOL_SIZES//;/,}"
FL_ANY_WEIGHT="${FL_ANY_WEIGHT:-0}"
FL_QQ_WEIGHT="${FL_QQ_WEIGHT:-0}"
FL_KK_WEIGHT="${FL_KK_WEIGHT:-20}"
FL_AA_WEIGHT="${FL_AA_WEIGHT:-30}"
FL_TRIPS_WEIGHT="${FL_TRIPS_WEIGHT:-45}"
EV_WEIGHT="${EV_WEIGHT:-0.02}"
BUST_WEIGHT="${BUST_WEIGHT:-0}"
RAYON_THREADS="${RAYON_THREADS:-16}"
PROB_ENGINE_TIMEOUT="${PROB_ENGINE_TIMEOUT:-600}"
UPLOAD_INTERVAL="${UPLOAD_INTERVAL:-180}"
SELF_DELETE="${SELF_DELETE:-true}"

REMOTE_ROOT="/opt/ofc-pineapple"
OUT_DIR="${REMOTE_ROOT}/teacher_results"
NAME="worker_${WORKER_ID}"
GCS_OUT="${BUCKET}/runs/${RUN_ID}/results/${NAME}"
STATUS_FILE="${OUT_DIR}/${NAME}.status.txt"

upload_once() {
  set +e
  if command -v gcloud >/dev/null 2>&1; then
    [ -f "${OUT_DIR}/${NAME}.jsonl" ] && gcloud storage cp -q "${OUT_DIR}/${NAME}.jsonl" "${GCS_OUT}/${NAME}.jsonl"
    [ -f "${OUT_DIR}/${NAME}.partial.jsonl" ] && gcloud storage cp -q "${OUT_DIR}/${NAME}.partial.jsonl" "${GCS_OUT}/${NAME}.partial.jsonl"
    [ -f "${OUT_DIR}/${NAME}.progress.json" ] && gcloud storage cp -q "${OUT_DIR}/${NAME}.progress.json" "${GCS_OUT}/${NAME}.progress.json"
    [ -f "${OUT_DIR}/${NAME}.summary.json" ] && gcloud storage cp -q "${OUT_DIR}/${NAME}.summary.json" "${GCS_OUT}/${NAME}.summary.json"
    [ -f "${OUT_DIR}/${NAME}.log" ] && gcloud storage cp -q "${OUT_DIR}/${NAME}.log" "${GCS_OUT}/${NAME}.log"
    [ -f "$STATUS_FILE" ] && gcloud storage cp -q "$STATUS_FILE" "${GCS_OUT}/${NAME}.status.txt"
    [ -f "$STARTUP_LOG" ] && gcloud storage cp -q "$STARTUP_LOG" "${GCS_OUT}/startup.log"
  fi
  set -e
}

trap upload_once EXIT

echo "run_id=${RUN_ID} worker_id=${WORKER_ID} hands=${HANDS} seed=${SEED}"
echo "sims=${SIMS} beam=${BEAM} child_sims=${CHILD_SIMS} t0_pool_size=${T0_POOL_SIZE}"
echo "trace_rollouts=${TRACE_ROLLOUTS} trace_sims=${TRACE_SIMS} trace_pool_sizes=${TRACE_POOL_SIZES}"
echo "weights fl_any=${FL_ANY_WEIGHT} qq=${FL_QQ_WEIGHT} kk=${FL_KK_WEIGHT} aa=${FL_AA_WEIGHT} trips=${FL_TRIPS_WEIGHT} ev=${EV_WEIGHT} bust=${BUST_WEIGHT}"
echo "prob_engine_timeout=${PROB_ENGINE_TIMEOUT}"
echo "bucket=${BUCKET}"
echo "code_archive=${CODE_ARCHIVE}"

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

gcloud storage cp "$CODE_ARCHIVE" /tmp/ofc_joint_t0_teacher_code.tar.gz
tar xzf /tmp/ofc_joint_t0_teacher_code.tar.gz -C "$REMOTE_ROOT"
rm -f /tmp/ofc_joint_t0_teacher_code.tar.gz

python3 -m venv "${REMOTE_ROOT}/.venv"
source "${REMOTE_ROOT}/.venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install numpy

cd "${REMOTE_ROOT}/ai/rust_solver"
cargo build --release -p prob_engine

cd "$REMOTE_ROOT"
export PYTHONUNBUFFERED=1
export RAYON_NUM_THREADS="$RAYON_THREADS"
export PROB_ENGINE_TIMEOUT="$PROB_ENGINE_TIMEOUT"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

{
  echo "started_at=$(date -Is)"
  echo "project=${PROJECT}"
  echo "gcs_out=${GCS_OUT}"
} > "$STATUS_FILE"

set +e
python -u ai/training/generate_joint_t0_recursive_teacher.py \
  --hands "$HANDS" \
  --seed "$SEED" \
  --sims "$SIMS" \
  --beam "$BEAM" \
  --child-sims "$CHILD_SIMS" \
  --t0-pool-size "$T0_POOL_SIZE" \
  --route-aware-t0-pool \
  --save-turn-traces \
  --trace-rollouts "$TRACE_ROLLOUTS" \
  --trace-sims "$TRACE_SIMS" \
  --trace-pool-sizes "$TRACE_POOL_SIZES" \
  --fl-any-weight "$FL_ANY_WEIGHT" \
  --fl-qq-weight "$FL_QQ_WEIGHT" \
  --fl-kk-weight "$FL_KK_WEIGHT" \
  --fl-aa-weight "$FL_AA_WEIGHT" \
  --fl-trips-weight "$FL_TRIPS_WEIGHT" \
  --ev-weight "$EV_WEIGHT" \
  --bust-weight "$BUST_WEIGHT" \
  --out-dir "$OUT_DIR" \
  --name "$NAME" \
  --resume \
  2>&1 | tee "${OUT_DIR}/${NAME}.log" &
GEN_PID=$!
set -e

while kill -0 "$GEN_PID" 2>/dev/null; do
  sleep "$UPLOAD_INTERVAL"
  upload_once
done

set +e
wait "$GEN_PID"
GEN_STATUS=$?
set -e

{
  echo "finished_at=$(date -Is)"
  echo "generator_exit_status=${GEN_STATUS}"
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
