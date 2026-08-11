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
WORK=/opt/attempt11
RESULT=/var/lib/attempt11-result
VENV=/opt/attempt11-venv
LOG=/var/log/attempt11-worker.log

mkdir -p "$RESULT"
exec > >(tee -a "$LOG") 2>&1

cleanup() {
  code=$?
  gcloud storage cp "$LOG" "$PREFIX/logs/shard-${SHARD}.log" \
    --project "$PROJECT_ID" >/dev/null 2>&1 || true
  if [[ "$SELF_DELETE" == 1 ]]; then
    gcloud compute instances delete "$INSTANCE_NAME" --zone "$ZONE" \
      --project "$PROJECT_ID" --quiet >/dev/null 2>&1 || true
  fi
  exit "$code"
}
trap cleanup EXIT

sudo rm -f /etc/apt/sources.list.d/debian.sources
printf '%s\n' \
  'deb [check-valid-until=no] http://snapshot.debian.org/archive/debian/20260609T000000Z bookworm main' \
  'deb [check-valid-until=no] http://snapshot.debian.org/archive/debian-security/20260609T000000Z bookworm-security main' \
  | sudo tee /etc/apt/sources.list >/dev/null
sudo apt-get -o Acquire::Check-Valid-Until=false update -y
sudo apt-get -o Acquire::Check-Valid-Until=false install -y \
  python3 python3-venv unzip time libgomp1 ca-certificates

gcloud storage cp "$SOURCE_URI" /tmp/source.zip --project "$PROJECT_ID" >/dev/null
gcloud storage cp "$PREFIX/manifest.json" /tmp/manifest.json --project "$PROJECT_ID" >/dev/null
gcloud storage cp "$PREFIX/source/shards_manifest.jsonl" /tmp/shards.jsonl --project "$PROJECT_ID" >/dev/null
[[ "$(sha /tmp/source.zip)" == "$SOURCE_SHA256" ]]
[[ "$(sha /tmp/manifest.json)" == "$MANIFEST_SHA256" ]]
[[ "$(sha /tmp/shards.jsonl)" == "$SCHEDULE_SHA256" ]]

AUTH_ARGS=()
if [[ "$AUTHORIZATION_SHA256" != "none" ]]; then
  gcloud storage cp "$PREFIX/source/execution_authorization.json" \
    /tmp/execution_authorization.json --project "$PROJECT_ID" >/dev/null
  [[ "$(sha /tmp/execution_authorization.json)" == "$AUTHORIZATION_SHA256" ]]
  AUTH_ARGS=(--authorization /tmp/execution_authorization.json \
    --source-package-sha256 "$SOURCE_SHA256")
fi

rm -rf "$WORK" "$VENV"
mkdir -p "$WORK"
unzip -q /tmp/source.zip -d "$WORK"
cd "$WORK"
python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu \
  'pip==26.1.2' 'setuptools==83.0.0' 'wheel==0.46.1'
python -m pip install -r configs/hu_m43_attempt08_runtime_requirements.txt
export PYTHONPATH="$WORK/src"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export OFC_HU_M3_BATCH_THREADS=4

python - /tmp/manifest.json /tmp/shards.jsonl "$SHARD" \
  "$SOURCE_SHA256" <<'PY' >/tmp/spec.env
import json,re,sys
m=json.load(open(sys.argv[1],encoding='utf-8'))
lines=open(sys.argv[2],encoding='utf-8').read().splitlines()
i=int(sys.argv[3]); assert m['source_sha256']==sys.argv[4]
assert len(lines)==m['total_shards']; s=json.loads(lines[i]); assert s['shard']==i
for key in ('mode','output_prefix','run_id'):
 value=str(s[key]); assert re.fullmatch(r'[A-Za-z0-9_.:=/-]+',value)
 print(key.upper()+'='+value)
print('ROOT_INDEX='+str(int(s['root_index'])))
print('BATCH='+('1' if s['batch_child_selectors'] else '0'))
PY
source /tmp/spec.env
if [[ "$BATCH" != 1 && "$MODE" != preflight ]]; then
  echo 'scalar child selectors are allowed only in Attempt11 preflight' >&2
  exit 64
fi

RUN_ARGS=(
  --mode "$MODE" --root-index "$ROOT_INDEX"
  --output "$RESULT/teacher.jsonl"
  --checkpoint "$RESULT/checkpoint.json"
  --heartbeat "$RESULT/heartbeat.json"
  --model outputs/hu_joint_policy/m43_attempt05_old_dev900_architecture_once/lambda_rank_candidate.pkl
  --model-sha256 e7fa728308c8973e2e202baf79309e30b9b6f58c37431d8ea55850390abc28c3
  --run-id "$RUN_ID" --plan configs/hu_joint_policy_m43_attempt11.json
  --native-batch-threads 4
)
if [[ "$BATCH" == 1 ]]; then RUN_ARGS+=(--batch-child-selectors); fi
RUN_ARGS+=("${AUTH_ARGS[@]}")

set +e
/usr/bin/time -v -o "$RESULT/time.txt" \
  python -m ofc_regular.run_hu_m43_attempt11 "${RUN_ARGS[@]}" \
  >"$RESULT/generator_summary.json" 2>"$RESULT/run.log"
run_code=$?
set -e
if [[ $run_code -ne 0 ]]; then
  gcloud storage cp "$RESULT/run.log" "$PREFIX/failures/$OUTPUT_PREFIX/run.log" \
    --project "$PROJECT_ID" >/dev/null 2>&1 || true
  exit "$run_code"
fi

python - "$RESULT/generator_summary.json" <<'PY'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1]); value=json.loads(p.read_text(encoding='utf-8'))
p.write_text(json.dumps(value,sort_keys=True,separators=(',',':'))+'\n',encoding='utf-8')
PY

python - "$RESULT" /tmp/manifest.json /tmp/shards.jsonl "$SHARD" \
  "$SOURCE_SHA256" "$MANIFEST_SHA256" "$SCHEDULE_SHA256" \
  "$AUTHORIZATION_SHA256" <<'PY'
import hashlib,json,pathlib,sys,time
r=pathlib.Path(sys.argv[1]); m=json.load(open(sys.argv[2],encoding='utf-8'))
s=json.loads(open(sys.argv[3],encoding='utf-8').read().splitlines()[int(sys.argv[4])])
names=('teacher.jsonl','checkpoint.json','heartbeat.json','generator_summary.json','run.log','time.txt')
files={n:{'sha256':hashlib.sha256((r/n).read_bytes()).hexdigest(),'bytes':(r/n).stat().st_size} for n in names}
d={'schema':'hu_m43_attempt11_done_v1','status':'complete','run_name':m['run_name'],
   'mode':m['mode'],'shard':s['shard'],'root_index':s['root_index'],
   'output_prefix':s['output_prefix'],'source_sha256':sys.argv[5],
   'manifest_sha256':sys.argv[6],'schedule_sha256':sys.argv[7],
   'authorization_sha256':sys.argv[8],'files':files,
   'current_profile_mutated':False,'runtime_policy_activated':False,
   'completed_unix_seconds':time.time()}
(r/'DONE.json').write_text(json.dumps(d,sort_keys=True,separators=(',',':'))+'\n',encoding='utf-8')
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

RESULT_URI="$PREFIX/results/$OUTPUT_PREFIX"
for name in teacher.jsonl checkpoint.json heartbeat.json generator_summary.json run.log time.txt; do
  upload_once "$RESULT/$name" "$RESULT_URI/$name"
done
upload_once "$RESULT/DONE.json" "$RESULT_URI/DONE.json"
