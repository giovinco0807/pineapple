#!/bin/bash
set -euo pipefail

STARTUP_LOG="/tmp/t3_capacity_startup.log"
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
CODE_ARCHIVE="$(metadata_attr code-archive)"
DATA_ARCHIVE="$(metadata_attr data-archive)"
EPOCHS="$(metadata_attr epochs)"
MAX_SECONDS="$(metadata_attr max-seconds)"
BATCH_SIZE="$(metadata_attr batch-size)"
RANKING_BATCHES="$(metadata_attr ranking-batches)"
UPLOAD_INTERVAL="$(metadata_attr upload-interval)"
SELF_DELETE="$(metadata_attr self-delete)"

BUCKET="${BUCKET:-gs://pokerhu-ofc-solver-485418-training}"
RUN_ID="${RUN_ID:-t3-capacity-$(date +%Y%m%d-%H%M%S)}"
CODE_ARCHIVE="${CODE_ARCHIVE:-${BUCKET}/runs/${RUN_ID}/code/ofc_t3_capacity_code.tar.gz}"
DATA_ARCHIVE="${DATA_ARCHIVE:-${BUCKET}/runs/${RUN_ID}/code/ofc_t3_capacity_data.tar.gz}"
EPOCHS="${EPOCHS:-40}"
MAX_SECONDS="${MAX_SECONDS:-0}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
RANKING_BATCHES="${RANKING_BATCHES:-128}"
UPLOAD_INTERVAL="${UPLOAD_INTERVAL:-300}"
SELF_DELETE="${SELF_DELETE:-true}"

REMOTE_ROOT="/opt/ofc-pineapple"
OUT_DIR="ai/data/hybrid_t1t2_active_20260531/t3_data_capacity_gcp_${RUN_ID}"
GCS_OUT="${BUCKET}/runs/${RUN_ID}/results"
STATUS_FILE="${REMOTE_ROOT}/t3_capacity_status.txt"
RUN_LOG="${REMOTE_ROOT}/t3_capacity_run.log"

upload_once() {
  set +e
  if command -v gcloud >/dev/null 2>&1; then
    [ -f "$STATUS_FILE" ] && gcloud storage cp -q "$STATUS_FILE" "${GCS_OUT}/status.txt"
    [ -f "$RUN_LOG" ] && gcloud storage cp -q "$RUN_LOG" "${GCS_OUT}/run.log"
    [ -f "$STARTUP_LOG" ] && gcloud storage cp -q "$STARTUP_LOG" "${GCS_OUT}/startup.log"
    if [ -d "${REMOTE_ROOT}/${OUT_DIR}" ]; then
      tar czf /tmp/t3_capacity_outputs.tar.gz -C "$REMOTE_ROOT" "$OUT_DIR"
      gcloud storage cp -q /tmp/t3_capacity_outputs.tar.gz "${GCS_OUT}/t3_capacity_outputs.tar.gz"
    fi
  fi
  set -e
}

trap upload_once EXIT

echo "run_id=${RUN_ID}"
echo "bucket=${BUCKET}"
echo "code_archive=${CODE_ARCHIVE}"
echo "data_archive=${DATA_ARCHIVE}"
echo "epochs=${EPOCHS} max_seconds=${MAX_SECONDS} batch_size=${BATCH_SIZE} ranking_batches=${RANKING_BATCHES}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y python3 python3-venv python3-pip curl build-essential

mkdir -p "$REMOTE_ROOT"
cd "$REMOTE_ROOT"

gcloud storage cp "$CODE_ARCHIVE" /tmp/ofc_t3_capacity_code.tar.gz
tar xzf /tmp/ofc_t3_capacity_code.tar.gz -C "$REMOTE_ROOT"
rm -f /tmp/ofc_t3_capacity_code.tar.gz

gcloud storage cp "$DATA_ARCHIVE" /tmp/ofc_t3_capacity_data.tar.gz
tar xzf /tmp/ofc_t3_capacity_data.tar.gz -C "$REMOTE_ROOT"
rm -f /tmp/ofc_t3_capacity_data.tar.gz

if [ -x /opt/conda/bin/python ]; then
  PYTHON_BIN="/opt/conda/bin/python"
else
  PYTHON_BIN="python3"
fi

"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install numpy
if ! "$PYTHON_BIN" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
then
  echo "Torch with CUDA is not available from the base image; installing CUDA wheel fallback"
  "$PYTHON_BIN" -m pip install torch --index-url https://download.pytorch.org/whl/cu121
  "$PYTHON_BIN" - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
PY
fi

{
  echo "started_at=$(date -Is)"
  echo "project=${PROJECT}"
  echo "gcs_out=${GCS_OUT}"
  echo "python=${PYTHON_BIN}"
} > "$STATUS_FILE"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2

set +e
"$PYTHON_BIN" -u -m ai.tutor.run_t3_data_capacity_experiment \
  --python "$PYTHON_BIN" \
  --out-dir "$OUT_DIR" \
  --epochs "$EPOCHS" \
  --max-seconds "$MAX_SECONDS" \
  --batch-size "$BATCH_SIZE" \
  --ranking-batches-per-epoch "$RANKING_BATCHES" \
  --device auto \
  2>&1 | tee "$RUN_LOG"
RUN_STATUS=${PIPESTATUS[0]}
set -e
upload_once

if [ "$RUN_STATUS" -eq 0 ]; then
  {
    echo "commands_started_at=$(date -Is)"
    echo "commands_sh=${REMOTE_ROOT}/${OUT_DIR}/commands.sh"
  } >> "$STATUS_FILE"
  chmod +x "${REMOTE_ROOT}/${OUT_DIR}/commands.sh"
  set +e
  bash "${REMOTE_ROOT}/${OUT_DIR}/commands.sh" >> "$RUN_LOG" 2>&1 &
  COMMANDS_PID=$!
  set -e

  while ps -p "$COMMANDS_PID" -o stat= 2>/dev/null | grep -qv Z; do
    sleep "$UPLOAD_INTERVAL"
    {
      echo "commands_heartbeat_at=$(date -Is)"
      find "${REMOTE_ROOT}/${OUT_DIR}" -maxdepth 5 -type f \( -name "summary.md" -o -name "metrics.json" -o -name "eval.json" \) -print 2>/dev/null || true
    } >> "$STATUS_FILE"
    upload_once
  done

  set +e
  wait "$COMMANDS_PID"
  CMD_STATUS=$?
  set -e
else
  CMD_STATUS="$RUN_STATUS"
fi

{
  echo "finished_at=$(date -Is)"
  echo "runner_exit_status=${RUN_STATUS}"
  echo "commands_exit_status=${CMD_STATUS}"
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

exit "$CMD_STATUS"
