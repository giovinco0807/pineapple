#!/bin/bash
set -euo pipefail

LOG=/tmp/t0_hu_oracle_startup.log
exec > >(tee -a "$LOG") 2>&1

PROJECT="ofc-solver-485418"
BUCKET="gs://ofc-solver-485418"
CODE_ARCHIVE="${BUCKET}/t0_hu_oracle/code/ofc_t0_hu_oracle_code.tar.gz"
T1_BB_OBJECT="${BUCKET}/t0_hu_oracle/models/t1_bb_policyvalue_model.pt"
T1_BTN_OBJECT="${BUCKET}/t0_hu_oracle/models/t1_btn_policyvalue_model.pt"
T0_BB_OBJECT="${BUCKET}/t0_hu_oracle/models/t0_bb_policyvalue_model.pt"
T0_BTN_OBJECT="${BUCKET}/t0_hu_oracle/models/t0_btn_policyvalue_model.pt"
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
RUN_ID="$(metadata run-id)"
T0_BB_POLICY="$(metadata t0-bb-policy)"
T0_OPP_POLICY="$(metadata t0-opp-policy)"

WORKER_ID="${WORKER_ID:-0}"
MAX_SECONDS="${MAX_SECONDS:-10800}"
N_SAMPLES="${N_SAMPLES:-1}"
CHUNK_SIZE="${CHUNK_SIZE:-100}"
STATES="${STATES:-100000000}"
POSITION="${POSITION:-bb}"
RUN_ID="${RUN_ID:-run_20260510}"
T0_BB_POLICY="${T0_BB_POLICY:-heuristic}"
T0_OPP_POLICY="${T0_OPP_POLICY:-heuristic}"
RESULT_ROOT="${BUCKET}/t0_hu_oracle/results/${RUN_ID}"

echo "worker_id=${WORKER_ID} position=${POSITION} run_id=${RUN_ID} max_seconds=${MAX_SECONDS} n_samples=${N_SAMPLES} chunk_size=${CHUNK_SIZE} t0_bb_policy=${T0_BB_POLICY} t0_opp_policy=${T0_OPP_POLICY}"

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y python3-venv python3-pip

mkdir -p "$REMOTE_ROOT"
cd "$REMOTE_ROOT"

gcloud storage cp "$CODE_ARCHIVE" /tmp/ofc_t0_hu_oracle_code.tar.gz
tar xzf /tmp/ofc_t0_hu_oracle_code.tar.gz -C "$REMOTE_ROOT"
rm -f /tmp/ofc_t0_hu_oracle_code.tar.gz

mkdir -p ai/data/t1_hu_bb_model ai/data/t1_hu_btn_model
gcloud storage cp "$T1_BB_OBJECT" ai/data/t1_hu_bb_model/t2_policyvalue_model.pt
gcloud storage cp "$T1_BTN_OBJECT" ai/data/t1_hu_btn_model/t2_policyvalue_model.pt
mkdir -p ai/data/t0_hu_bb_model ai/data/t0_hu_btn_model
gcloud storage cp "$T0_BB_OBJECT" ai/data/t0_hu_bb_model/t2_policyvalue_model.pt || true
gcloud storage cp "$T0_BTN_OBJECT" ai/data/t0_hu_btn_model/t2_policyvalue_model.pt || true

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy torch --index-url https://download.pytorch.org/whl/cpu

OUT_DIR="ai/data/t0_hu_${POSITION}_worker_${WORKER_ID}"
GCS_OUT="${RESULT_ROOT}/${POSITION}/worker_${WORKER_ID}"
mkdir -p "$OUT_DIR"

set +e
timeout --kill-after=300s "${MAX_SECONDS}s" python -u ai/training/generate_t0_hu_oracle.py \
  --position "$POSITION" \
  --t1-bb-model ai/data/t1_hu_bb_model/t2_policyvalue_model.pt \
  --t1-btn-model ai/data/t1_hu_btn_model/t2_policyvalue_model.pt \
  --save-dir "$OUT_DIR" \
  --states "$STATES" \
  --n-samples "$N_SAMPLES" \
  --chunk-size "$CHUNK_SIZE" \
  --max-seconds "$MAX_SECONDS" \
  --gcs-output "$GCS_OUT" \
  --t0-bb-policy "$T0_BB_POLICY" \
  --t0-opp-policy "$T0_OPP_POLICY" \
  --t0-bb-model ai/data/t0_hu_bb_model/t2_policyvalue_model.pt \
  --t0-opp-model ai/data/t0_hu_btn_model/t2_policyvalue_model.pt \
  --device cpu \
  --seed "$((20260510 + WORKER_ID))" \
  2>&1 | tee "t0_hu_${POSITION}_worker_${WORKER_ID}.log"
GEN_STATUS=${PIPESTATUS[0]}
set -e
echo "generator_exit_status=${GEN_STATUS}"

gcloud storage cp "t0_hu_${POSITION}_worker_${WORKER_ID}.log" "${GCS_OUT}/t0_hu_${POSITION}_worker_${WORKER_ID}.log" || true
gcloud storage cp "$LOG" "${GCS_OUT}/startup_${WORKER_ID}.log" || true

INSTANCE_NAME="$(curl -sf -H Metadata-Flavor:Google http://metadata.google.internal/computeMetadata/v1/instance/name || true)"
ZONE_PATH="$(curl -sf -H Metadata-Flavor:Google http://metadata.google.internal/computeMetadata/v1/instance/zone || true)"
ZONE="${ZONE_PATH##*/}"
if [[ -n "$INSTANCE_NAME" && -n "$ZONE" ]]; then
  gcloud compute instances delete "$INSTANCE_NAME" --zone="$ZONE" --project="$PROJECT" --quiet || true
fi
shutdown -h now || true
