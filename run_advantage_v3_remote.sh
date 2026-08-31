#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/Owner/ofc
source .venv/bin/activate

RUN_ID="advantage_t0t1_pairguard_foulrisk_v3_x40_20260516"
RUN_DIR="/home/Owner/ofc/ai/models/candidate_runs/${RUN_ID}"
GCS_URI="gs://ofc-solver-485418/ofc_all_turn_runs/${RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"

sync_to_gcs() {
  if command -v gsutil >/dev/null 2>&1 && [[ -d "${RUN_DIR}" ]]; then
    gsutil -m rsync -r "${RUN_DIR}" "${GCS_URI}" || true
  fi
}

PROGRESS_PID=""
cleanup() {
  local status=$?
  set +e
  if [[ -n "${PROGRESS_PID}" ]]; then
    kill "${PROGRESS_PID}" >/dev/null 2>&1 || true
  fi
  mkdir -p "${RUN_DIR}"
  {
    echo "run_id=${RUN_ID}"
    echo "exit_status=${status}"
    date -u +"finished_utc=%Y-%m-%dT%H:%M:%SZ"
  } > "${RUN_DIR}/status.txt"
  sync_to_gcs
  sudo shutdown -h now || true
  exit "${status}"
}
trap cleanup EXIT INT TERM

(
  while true; do
    sleep 900
    sync_to_gcs
  done
) &
PROGRESS_PID=$!

python ai/run_advantage_distill_cycle.py \
  --run-id "${RUN_ID}" \
  --overwrite \
  --models t0_bb,t0_btn,t1_bb,t1_btn \
  --teacher-shards 40 \
  --teacher-hands-per-shard 250 \
  --teacher-seed-start 90260516 \
  --max-samples-per-shard 3200 \
  --max-parallel-teacher 4 \
  --teacher-sims 20 \
  --teacher-topk 12 \
  --teacher-paired-rollouts \
  --actor-lookahead-sims 10 \
  --actor-lookahead-topk 5 \
  --include-advantage \
  --advantage-temperature 6.0 \
  --advantage-ratio 0.30 \
  --broad-ratio 0.55 \
  --teacher-ratio 0.10 \
  --hard-ratio 0.05 \
  --variants lr5e6_e1:5e-6:1:0.01,lr2e6_e2:2e-6:2:0.01,lr1e5_e1:1e-5:1:0.01 \
  --session-hands 1000 \
  --session-seeds 21260511,23260511,24260511 \
  --eval-lookahead-sims 10 \
  --eval-lookahead-topk 5 \
  --max-parallel-eval 2 \
  --min-avg 0 \
  --max-foul-rate-diff 0 \
  --require-session-win \
  2>&1 | tee "${LOG_DIR}/run.log"
