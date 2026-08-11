#!/usr/bin/env bash
set -euo pipefail

META='http://metadata.google.internal/computeMetadata/v1'
meta() { curl -fsS -H 'Metadata-Flavor: Google' "$META/instance/attributes/$1"; }
imeta() { curl -fsS -H 'Metadata-Flavor: Google' "$META/instance/$1"; }
sha() { sha256sum "$1" | awk '{print $1}'; }

PROJECT_ID="$(meta PROJECT_ID)"
BUCKET="$(meta BUCKET)"
RUN_NAME="$(meta RUN_NAME)"
SHARD="$(meta SHARD)"
SOURCE_URI="$(meta SOURCE_URI)"
SOURCE_SHA256="$(meta SOURCE_SHA256)"
MANIFEST_SHA256="$(meta MANIFEST_SHA256)"
SCHEDULE_SHA256="$(meta SCHEDULE_SHA256)"
AUTHORIZATION_SHA256="$(meta AUTHORIZATION_SHA256)"
SELF_DELETE="$(meta SELF_DELETE)"
INSTANCE_NAME="$(imeta name)"
ZONE="$(imeta zone)"; ZONE="${ZONE##*/}"
PREFIX="gs://$BUCKET/runs/$RUN_NAME"
WORK=/opt/m31-step6a
RESULT=/var/lib/m31-step6a-result
VENV=/opt/m31-step6a-venv
LOG=/var/log/m31-step6a-worker.log
PUMP_PID=''

mkdir -p "$RESULT"
exec > >(tee -a "$LOG") 2>&1

sync_progress() {
  if [[ -d "$RESULT" ]]; then
    gcloud storage rsync --recursive "$RESULT" "$PREFIX/progress/shard-000" \
      --project "$PROJECT_ID" >/dev/null 2>&1 || true
  fi
}

stop_pump() {
  if [[ -n "$PUMP_PID" ]]; then
    kill "$PUMP_PID" >/dev/null 2>&1 || true
    wait "$PUMP_PID" >/dev/null 2>&1 || true
    PUMP_PID=''
  fi
}

cleanup() {
  code=$?
  stop_pump
  sync_progress
  gcloud storage cp "$LOG" "$PREFIX/logs/shard-000-startup.log" \
    --project "$PROJECT_ID" >/dev/null 2>&1 || true
  if [[ "$SELF_DELETE" == 1 ]]; then
    gcloud compute instances delete "$INSTANCE_NAME" --zone "$ZONE" \
      --project "$PROJECT_ID" --quiet >/dev/null 2>&1 || true
  fi
  exit "$code"
}
trap cleanup EXIT

if [[ "$SHARD" != 0 ]]; then
  echo 'Step 6a startup authorizes shard 0 only' >&2
  exit 64
fi

sudo rm -f /etc/apt/sources.list.d/debian.sources
printf '%s\n' \
  'deb [check-valid-until=no] http://snapshot.debian.org/archive/debian/20260609T000000Z bookworm main' \
  'deb [check-valid-until=no] http://snapshot.debian.org/archive/debian-security/20260609T000000Z bookworm-security main' \
  | sudo tee /etc/apt/sources.list >/dev/null
sudo apt-get -o Acquire::Check-Valid-Until=false update -y
sudo apt-get -o Acquire::Check-Valid-Until=false install -y \
  python3 python3-venv unzip time libgomp1 ca-certificates

gcloud storage cp "$SOURCE_URI" /tmp/source.zip --project "$PROJECT_ID" >/dev/null
gcloud storage cp "$PREFIX/manifest.json" /tmp/manifest.json \
  --project "$PROJECT_ID" >/dev/null
gcloud storage cp "$PREFIX/source/shards_manifest.jsonl" /tmp/shards.jsonl \
  --project "$PROJECT_ID" >/dev/null
gcloud storage cp "$PREFIX/source/launch_authorization.json" /tmp/authorization.json \
  --project "$PROJECT_ID" >/dev/null
[[ "$(sha /tmp/source.zip)" == "$SOURCE_SHA256" ]]
[[ "$(sha /tmp/manifest.json)" == "$MANIFEST_SHA256" ]]
[[ "$(sha /tmp/shards.jsonl)" == "$SCHEDULE_SHA256" ]]
[[ "$(sha /tmp/authorization.json)" == "$AUTHORIZATION_SHA256" ]]

rm -rf "$WORK" "$VENV"
mkdir -p "$WORK"
unzip -q /tmp/source.zip -d "$WORK"
cd "$WORK"
python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install -r configs/hu_m43_attempt08_runtime_requirements.txt
export PYTHONPATH="$WORK/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

gcloud storage rsync --recursive "$PREFIX/progress/shard-000" "$RESULT" \
  --project "$PROJECT_ID" >/dev/null 2>&1 || true

(
  while true; do
    sync_progress
    sleep 60
  done
) &
PUMP_PID=$!

COMMON_ARGS=(
  --manifest /tmp/manifest.json
  --schedule /tmp/shards.jsonl
  --source-package-sha256 "$SOURCE_SHA256"
  --repository-root "$WORK"
  --shard 0
  --output-dir "$RESULT"
  --parity-golden "$WORK/artifacts/step6a/parity_golden.json"
)

python -m ofc_regular.run_hu_m31_t3_step6a_shard \
  "${COMMON_ARGS[@]}" --parity-only

TASK_COUNT=0
if [[ -d "$RESULT/tasks" ]]; then
  TASK_COUNT="$(find "$RESULT/tasks" -maxdepth 1 -type f -name 'hand_*.json' | wc -l)"
fi
if [[ "$TASK_COUNT" -eq 0 ]]; then
  set +e
  python -m ofc_regular.run_hu_m31_t3_step6a_shard \
    "${COMMON_ARGS[@]}" --stop-after-tasks 1 >"$RESULT/drill_stdout.json"
  drill_code=$?
  set -e
  if [[ "$drill_code" -ne 75 ]]; then
    echo "resume drill expected exit 75, got $drill_code" >&2
    exit 65
  fi
  sync_progress
  rm -rf "$RESULT"
  mkdir -p "$RESULT"
  gcloud storage rsync --recursive "$PREFIX/progress/shard-000" "$RESULT" \
    --project "$PROJECT_ID" >/dev/null
  RECOVERED="$(find "$RESULT/tasks" -maxdepth 1 -type f -name 'hand_*.json' | wc -l)"
  if [[ "$RECOVERED" -lt 1 ]]; then
    echo 'resume drill failed to recover a completed task from GCS' >&2
    exit 66
  fi
fi

/usr/bin/time -v -o "$RESULT/time.txt" \
  python -m ofc_regular.run_hu_m31_t3_step6a_shard "${COMMON_ARGS[@]}" \
  >"$RESULT/runner_stdout.json"
cp "$LOG" "$RESULT/run.log"
stop_pump
sync_progress

python - "$RESULT" /tmp/manifest.json /tmp/authorization.json <<'PY'
import hashlib,json,pathlib,sys,time
r=pathlib.Path(sys.argv[1])
m=json.load(open(sys.argv[2],encoding='utf-8'))
a=json.load(open(sys.argv[3],encoding='utf-8'))
summary=json.load(open(r/'summary.json',encoding='utf-8'))
assert summary['all_gates_passed'] is True
assert summary['resumed_task_count'] >= 1
assert summary['training_eligible'] is False
assert summary['production_fanout_authorized'] is False
files={}
for path in sorted(r.rglob('*')):
    if path.is_file() and path.name != 'DONE.json':
        rel=path.relative_to(r).as_posix()
        data=path.read_bytes()
        files[rel]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
done={
  'schema':'hu_m31_t3_step6a_done_v1','status':'complete',
  'run_name':m['run_name'],'shard':0,
  'source_sha256':m['source_sha256'],
  'manifest_sha256':hashlib.sha256(pathlib.Path(sys.argv[2]).read_bytes()).hexdigest(),
  'schedule_sha256':m['schedule_sha256'],
  'authorization_sha256':hashlib.sha256(pathlib.Path(sys.argv[3]).read_bytes()).hexdigest(),
  'native_library_sha256':m['native_library']['sha256'],
  'files':files,'resume_drill_passed':True,
  'training_eligible':False,'remaining_canary_shards_authorized':False,
  'production_fanout_authorized':False,'current_profile_changed':False,
  'completed_unix_seconds':time.time(),
}
(r/'DONE.json').write_text(json.dumps(done,sort_keys=True,separators=(',',':'))+'\n',encoding='utf-8')
PY

upload_once() {
  local source="$1" target="$2"
  if gcloud storage cp "$source" "$target" --project "$PROJECT_ID" \
      --if-generation-match=0 >/dev/null 2>&1; then return 0; fi
  local existing; existing="$(mktemp)"
  gcloud storage cp "$target" "$existing" --project "$PROJECT_ID" >/dev/null
  [[ "$(sha "$source")" == "$(sha "$existing")" ]]
  rm -f "$existing"
}

RESULT_URI="$PREFIX/results/shard-000"
while IFS= read -r -d '' path; do
  relative="${path#"$RESULT/"}"
  upload_once "$path" "$RESULT_URI/$relative"
done < <(find "$RESULT" -type f ! -name DONE.json -print0)
upload_once "$RESULT/DONE.json" "$RESULT_URI/DONE.json"
