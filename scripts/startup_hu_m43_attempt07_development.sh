#!/usr/bin/env bash
set -euo pipefail

# One immutable Attempt07 development root per Spot VM.  DONE is checked before
# any claim.  Both global and per-root claims are create-only and byte-verified
# before the runner is allowed to materialize the fresh root.

sha(){ sha256sum "$1" | awk '{print $1}'; }
meta(){ curl -fsS -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }
imeta(){ curl -fsS -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/$1"; }

PROJECT_ID="$(meta PROJECT_ID)"
BUCKET="$(meta BUCKET)"
RUN_NAME="$(meta RUN_NAME)"
SHARD_INDEX="$(meta SHARD_INDEX)"
SOURCE_URI="$(meta SOURCE_URI)"
SOURCE_SHA256="$(meta SOURCE_SHA256)"
MANIFEST_URI="$(meta MANIFEST_URI)"
MANIFEST_SHA256="$(meta MANIFEST_SHA256)"
AUTHORIZATION_URI="$(meta AUTHORIZATION_URI)"
AUTHORIZATION_SHA256="$(meta AUTHORIZATION_SHA256)"
STARTUP_SHA256="$(meta STARTUP_SHA256)"
SCHEDULE_SHA256="$(meta SCHEDULE_SHA256)"
PLAN_SHA256="$(meta PLAN_SHA256)"
STATUS_SHA256="$(meta STATUS_SHA256)"
MODEL_SHA256="$(meta MODEL_SHA256)"
AI_PROFILES_SHA256="$(meta AI_PROFILES_SHA256)"
PREFLIGHT_PLAN_SHA256="$(meta PREFLIGHT_PLAN_SHA256)"
PREFLIGHT_AGGREGATE_SHA256="$(meta PREFLIGHT_AGGREGATE_SHA256)"
CLOSURE_SHA256="$(meta CLOSURE_SHA256)"
SOURCE_MODEL_MANIFEST_SHA256="$(meta SOURCE_MODEL_MANIFEST_SHA256)"
SOURCE_NATIVE_MANIFEST_SHA256="$(meta SOURCE_NATIVE_MANIFEST_SHA256)"
NATIVE_BATCH_THREADS="$(meta NATIVE_BATCH_THREADS)"
SYNC_INTERVAL_SECONDS="$(meta SYNC_INTERVAL_SECONDS)"
SELF_DELETE="$(meta SELF_DELETE)"
INSTANCE_NAME="$(imeta name)"
ZONE_PATH="$(imeta zone)"
ZONE="${ZONE_PATH##*/}"

curl -fsS -H 'Metadata-Flavor: Google' \
  'http://metadata.google.internal/computeMetadata/v1/instance/attributes/startup-script' \
  > /tmp/attempt07_executing_startup.sh
[[ "$(sha /tmp/attempt07_executing_startup.sh)" == "$STARTUP_SHA256" ]]

PREFIX="gs://${BUCKET}/runs/${RUN_NAME}"
WORK=/opt/ofc-hu-m43-attempt07
RESULT=/tmp/ofc-hu-m43-attempt07-result
STARTUP_LOG=/tmp/ofc-hu-m43-attempt07-startup.log
PID=''
OUTPUT_PREFIX=''
mkdir -p "$RESULT"
: > "$STARTUP_LOG"
exec > >(tee -a "$STARTUP_LOG") 2>&1

gcs_state(){
  local uri="$1" stdout_file stderr_file message
  stdout_file="$(mktemp)"; stderr_file="$(mktemp)"
  if gcloud storage objects describe "$uri" --project "$PROJECT_ID" --format='value(name)' >"$stdout_file" 2>"$stderr_file"; then
    rm -f "$stdout_file" "$stderr_file"; echo present; return 0
  fi
  message="$(cat "$stdout_file" "$stderr_file")"
  rm -f "$stdout_file" "$stderr_file"
  if grep -Eiq 'not found|does not exist|No URLs matched|404' <<<"$message"; then
    echo absent; return 0
  fi
  echo "unable to prove exact GCS state for $uri: $message" >&2
  return 1
}

upload_once_or_verify(){
  local source="$1" uri="$2" existing
  if gcloud storage cp "$source" "$uri" --project "$PROJECT_ID" --if-generation-match=0 >/dev/null 2>&1; then
    return 0
  fi
  existing="$(mktemp)"
  gcloud storage cp "$uri" "$existing" --project "$PROJECT_ID" >/dev/null
  cmp -s "$source" "$existing" || { rm -f "$existing"; echo "immutable remote object differs: $uri" >&2; return 1; }
  rm -f "$existing"
}

claim_once(){
  local marker="$1" uri="$2" existing
  if gcloud storage cp "$marker" "$uri" --project "$PROJECT_ID" --if-generation-match=0 >/dev/null 2>&1; then
    echo owner; return 0
  fi
  existing="$(mktemp)"
  gcloud storage cp "$uri" "$existing" --project "$PROJECT_ID" >/dev/null
  cmp -s "$marker" "$existing" || { rm -f "$existing"; echo "immutable claim differs: $uri" >&2; return 1; }
  rm -f "$existing"; echo existing
}

status(){
  local state="$1" completed=0 alive=false
  [[ -s "$RESULT/checkpoint.json" ]] && completed="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("completed_roots",0))' "$RESULT/checkpoint.json" 2>/dev/null || echo 0)"
  [[ -n "$PID" ]] && kill -0 "$PID" >/dev/null 2>&1 && alive=true || true
  python3 - "$RUN_NAME" "$SHARD_INDEX" "$state" "$completed" "$alive" > /tmp/attempt07_status.json <<'PY'
import json,sys,time
run,shard,state,completed,alive=sys.argv[1:]
print(json.dumps({
  'schema':'hu_m43_attempt07_development_spot_status_v1',
  'run_name':run,'shard':int(shard),'status':state,
  'completed_roots':int(completed),'target_roots':1,
  'teacher_process_alive':alive=='true','updated_unix_seconds':time.time(),
  'teacher_values_exposed':False,'selector_executed':False,
  'current_profile_mutated':False,'runtime_policy_activated':False,
},sort_keys=True,separators=(',',':')))
PY
  gcloud storage cp /tmp/attempt07_status.json "$PREFIX/status/shard_${SHARD_INDEX}.json" --project "$PROJECT_ID" >/dev/null || true
}

cleanup(){
  local code=$?
  set +e
  if [[ $code -ne 0 ]]; then
    status failed
    gcloud storage cp "$STARTUP_LOG" "$PREFIX/logs/startup_shard_${SHARD_INDEX}.log" --project "$PROJECT_ID" >/dev/null 2>&1 || true
  fi
  if [[ "$SELF_DELETE" == 1 ]]; then
    gcloud compute instances delete "$INSTANCE_NAME" --zone "$ZONE" --project "$PROJECT_ID" --quiet >/dev/null 2>&1 || sudo shutdown -h now
  fi
  exit "$code"
}
trap cleanup EXIT

export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -y
sudo apt-get install -y unzip curl ca-certificates python3 python3-venv python3-pip time
if ! command -v gcloud >/dev/null 2>&1; then
  sudo apt-get install -y apt-transport-https gnupg
  curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg
  echo 'deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main' | sudo tee /etc/apt/sources.list.d/google-cloud-sdk.list >/dev/null
  sudo apt-get update -y
  sudo apt-get install -y google-cloud-cli
fi

sudo rm -rf "$WORK"
sudo mkdir -p "$WORK"
sudo chown "$(id -u):$(id -g)" "$WORK"
gcloud storage cp "$SOURCE_URI" /tmp/attempt07_source.zip --project "$PROJECT_ID" >/dev/null
gcloud storage cp "$MANIFEST_URI" /tmp/attempt07_manifest.json --project "$PROJECT_ID" >/dev/null
gcloud storage cp "$AUTHORIZATION_URI" /tmp/attempt07_authorization.json --project "$PROJECT_ID" >/dev/null
[[ "$(sha /tmp/attempt07_source.zip)" == "$SOURCE_SHA256" ]]
[[ "$(sha /tmp/attempt07_manifest.json)" == "$MANIFEST_SHA256" ]]
[[ "$(sha /tmp/attempt07_authorization.json)" == "$AUTHORIZATION_SHA256" ]]
unzip -q /tmp/attempt07_source.zip -d "$WORK"
cd "$WORK"
[[ "$(sha shards_manifest.jsonl)" == "$SCHEDULE_SHA256" ]]
[[ "$(sha configs/hu_joint_policy_m43_attempt07.json)" == "$PLAN_SHA256" ]]
[[ "$(sha configs/hu_joint_policy_m43_attempt07_status.json)" == "$STATUS_SHA256" ]]
[[ "$(sha configs/hu_joint_policy_m43_attempt07_preflight.json)" == "$PREFLIGHT_PLAN_SHA256" ]]
[[ "$(sha frozen/attempt07_preflight_aggregate.json)" == "$PREFLIGHT_AGGREGATE_SHA256" ]]
[[ "$(sha artifacts/lambda_rank_candidate.pkl)" == "$MODEL_SHA256" ]]
[[ "$(sha src/ofc_regular/ai_profiles.py)" == "$AI_PROFILES_SHA256" ]]
[[ "$(sha source_closure_manifest.json)" == "$CLOSURE_SHA256" ]]
[[ "$(sha source_model_manifest.json)" == "$SOURCE_MODEL_MANIFEST_SHA256" ]]
[[ "$(sha source_native_manifest.json)" == "$SOURCE_NATIVE_MANIFEST_SHA256" ]]
python3 - <<'PY'
import hashlib,json,pathlib
root=pathlib.Path('.')
closure=json.loads((root/'source_closure_manifest.json').read_text(encoding='utf-8'))
assert set(closure)=={'schema','status','run_name','plan_sha256','status_sha256','model_sha256','ai_profiles_sha256','preflight_aggregate_sha256','preflight_plan_sha256','preflight_manifest_sha256','preflight_schedule_sha256','preflight_launch_authorization_sha256','preflight_done_sha256','files','fresh_root_opened','teacher_executed','current_profile_mutated'}
assert closure['schema']=='hu_m43_attempt07_development_source_closure_v1'
assert closure['status']=='closed_before_any_development_root'
assert closure['fresh_root_opened'] is False
assert closure['teacher_executed'] is False
assert closure['current_profile_mutated'] is False
assert isinstance(closure['files'],list) and closure['files']
listed=set()
for row in closure['files']:
    assert set(row)=={'path','bytes','sha256'}
    assert isinstance(row['path'],str) and row['path']
    relative=pathlib.PurePosixPath(row['path'])
    assert not relative.is_absolute() and '..' not in relative.parts
    assert row['path']==relative.as_posix() and row['path'] not in listed
    assert type(row['bytes']) is int and row['bytes']>=0
    assert isinstance(row['sha256'],str) and len(row['sha256'])==64
    int(row['sha256'],16)
    listed.add(row['path'])
    path=root/row['path']
    assert path.is_file() and not path.is_symlink() and path.stat().st_size==row['bytes']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
entries=list(root.rglob('*'))
assert all(not path.is_symlink() for path in entries)
actual={path.relative_to(root).as_posix() for path in entries if path.is_file()}
assert actual==listed|{'source_closure_manifest.json'}
PY
cp /tmp/attempt07_manifest.json "$WORK/manifest.json"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'numpy==2.2.6' 'scikit-learn==1.8.0' 'lightgbm==4.6.0'
python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.6.0'
export PYTHONPATH="$WORK/src"
export OMP_NUM_THREADS="$NATIVE_BATCH_THREADS"
export MKL_NUM_THREADS="$NATIVE_BATCH_THREADS"
export OPENBLAS_NUM_THREADS="$NATIVE_BATCH_THREADS"
export OFC_HU_M3_BATCH_THREADS="$NATIVE_BATCH_THREADS"
[[ "$NATIVE_BATCH_THREADS" == 4 ]]

python -B -m ofc_regular.hu_m43_attempt07_spot validate-authorization \
  --authorization /tmp/attempt07_authorization.json \
  --manifest manifest.json \
  --preflight-aggregate frozen/attempt07_preflight_aggregate.json >/tmp/authorization_audit.json
python - <<'PY'
from ofc_regular.hu_m3_rust import engine_version,load_native_engine
from ofc_regular.hu_turn3_stage3_feature_rust import rust_direct_available
print('hu_m3_engine='+engine_version(library=load_native_engine()))
assert rust_direct_available()
PY

SPEC="$(sed -n "$((SHARD_INDEX+1))p" shards_manifest.jsonl)"
[[ -n "$SPEC" ]]
field(){ python3 -c 'import json,sys; print(json.loads(sys.argv[1])[sys.argv[2]])' "$SPEC" "$1"; }
[[ "$(field schema)" == hu_m43_attempt07_development_spot_shard_v1 ]]
[[ "$(field shard)" == "$SHARD_INDEX" ]]
[[ "$(field root_index)" == "$SHARD_INDEX" ]]
[[ "$(field roots)" == 1 ]]
[[ "$(field native_batch_threads)" == 4 ]]
[[ "$(field batch_child_selectors)" == True ]]
[[ "$(field baseline_profile)" == stage18_p1 ]]
[[ "$(field continuation_profile)" == stage9f_p2 ]]
ROOT_INDEX="$(field root_index)"
ROOT_PROFILE="$(field root_profile)"
OUTPUT_PREFIX="$(field output_prefix)"
RUN_ID="${RUN_NAME}:shard=${SHARD_INDEX}"
RESULT_URI="$PREFIX/results/$OUTPUT_PREFIX"
DONE_URI="$RESULT_URI/DONE.json"
GLOBAL_CLAIM_URI="$PREFIX/development_boundary/CLAIMED.json"
ROOT_CLAIM_URI="$PREFIX/development_boundary/$OUTPUT_PREFIX/CLAIMED.json"
RESUME_URI="$PREFIX/resume/$OUTPUT_PREFIX"

# DONE is operational metadata only.  If it already exists, validate the exact
# immutable identity and exit without touching a claim, root, or teacher object.
if [[ "$(gcs_state "$DONE_URI")" == present ]]; then
  gcloud storage cp "$DONE_URI" /tmp/existing_DONE.json --project "$PROJECT_ID" >/dev/null
  python3 - /tmp/existing_DONE.json "$RUN_NAME" "$SHARD_INDEX" "$MANIFEST_SHA256" "$AUTHORIZATION_SHA256" "$SOURCE_SHA256" "$SCHEDULE_SHA256" "$PLAN_SHA256" "$MODEL_SHA256" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding='utf-8'))
expected={'schema':'hu_m43_attempt07_development_spot_done_v1','status':'complete',
 'run_name':sys.argv[2],'shard':int(sys.argv[3]),'root_index':int(sys.argv[3]),
 'manifest_sha256':sys.argv[4],'authorization_sha256':sys.argv[5],
 'source_sha256':sys.argv[6],'schedule_sha256':sys.argv[7],
 'plan_sha256':sys.argv[8],'model_sha256':sys.argv[9],
 'current_profile_mutated':False,'runtime_policy_activated':False}
assert all(p.get(k)==v for k,v in expected.items())
PY
  status complete
  exit 0
fi

COMMON_ARGS=("$RUN_NAME" "$MANIFEST_SHA256" "$SOURCE_SHA256" "$STARTUP_SHA256" "$SCHEDULE_SHA256" "$PLAN_SHA256" "$STATUS_SHA256" "$MODEL_SHA256" "$AI_PROFILES_SHA256" "$PREFLIGHT_PLAN_SHA256" "$PREFLIGHT_AGGREGATE_SHA256" "$CLOSURE_SHA256" "$SOURCE_MODEL_MANIFEST_SHA256" "$SOURCE_NATIVE_MANIFEST_SHA256" "$AUTHORIZATION_SHA256")
python3 - "${COMMON_ARGS[@]}" > /tmp/global_claim.json <<'PY'
import json,sys
keys=('run_name','manifest_sha256','source_sha256','startup_sha256','schedule_sha256','plan_sha256','status_sha256','model_sha256','ai_profiles_sha256','preflight_plan_sha256','preflight_aggregate_sha256','source_closure_sha256','source_model_manifest_sha256','source_native_manifest_sha256','authorization_sha256')
payload=dict(zip(keys,sys.argv[1:],strict=True))
payload.update({'schema':'hu_m43_attempt07_development_global_claim_v1','status':'claimed_before_any_development_root','roots':100,'shards':100,'roots_per_shard':1,'native_batch_threads':4,'development_started_when_claimed':False,'current_profile_mutated':False,'runtime_policy_activated':False})
print(json.dumps(payload,sort_keys=True,separators=(',',':')))
PY
claim_once /tmp/global_claim.json "$GLOBAL_CLAIM_URI" >/tmp/global_claim_state.txt

python3 - "$SPEC" "$RUN_NAME" "$MANIFEST_SHA256" "$SOURCE_SHA256" "$STARTUP_SHA256" "$SCHEDULE_SHA256" "$PLAN_SHA256" "$STATUS_SHA256" "$MODEL_SHA256" "$AI_PROFILES_SHA256" "$PREFLIGHT_PLAN_SHA256" "$PREFLIGHT_AGGREGATE_SHA256" "$CLOSURE_SHA256" "$SOURCE_MODEL_MANIFEST_SHA256" "$SOURCE_NATIVE_MANIFEST_SHA256" "$AUTHORIZATION_SHA256" > /tmp/root_claim.json <<'PY'
import json,sys
spec=json.loads(sys.argv[1])
keys=('run_name','manifest_sha256','source_sha256','startup_sha256','schedule_sha256','plan_sha256','status_sha256','model_sha256','ai_profiles_sha256','preflight_plan_sha256','preflight_aggregate_sha256','source_closure_sha256','source_model_manifest_sha256','source_native_manifest_sha256','authorization_sha256')
p=dict(zip(keys,sys.argv[2:],strict=True))
p.update({'schema':'hu_m43_attempt07_development_root_claim_v1','status':'claimed_before_root_generation','run_id':f"{p['run_name']}:shard={spec['shard']}",'root_index':spec['root_index'],'root_profile':spec['root_profile'],'seeds':spec['seeds'],'output_prefix':spec['output_prefix'],'native_batch_threads':4,'alternate_seed_or_result_retry_allowed':False,'deterministic_recompute_allowed':True,'deterministic_recompute_mode':'same_root_index_same_six_seeds_same_frozen_closure_after_matching_claim'})
print(json.dumps(p,sort_keys=True,separators=(',',':')))
PY
claim_once /tmp/root_claim.json "$ROOT_CLAIM_URI" >/tmp/root_claim_state.txt
status claimed

# Only a COMMIT written after its content-addressed output/checkpoint pair is
# accepted.  Orphaned resume objects are ignored.  With no valid commit the
# exact already-claimed root is recomputed from the same six frozen seeds.
COMMIT_URI="$RESUME_URI/COMMIT.json"
if [[ "$(gcs_state "$COMMIT_URI")" == present ]]; then
  gcloud storage cp "$COMMIT_URI" /tmp/resume_COMMIT.json --project "$PROJECT_ID" >/dev/null
  read -r RESUME_OUTPUT_SHA RESUME_CHECKPOINT_SHA RESUME_ELAPSED RESUME_RSS < <(python3 - /tmp/resume_COMMIT.json "$RUN_NAME" "$SHARD_INDEX" "$MANIFEST_SHA256" "$AUTHORIZATION_SHA256" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding='utf-8'))
assert p['schema']=='hu_m43_attempt07_development_resume_commit_v1'
assert p['status']=='committed_completed_root_pair'
assert p['run_name']==sys.argv[2] and p['shard']==int(sys.argv[3])
assert p['manifest_sha256']==sys.argv[4] and p['authorization_sha256']==sys.argv[5]
print(p['output_sha256'],p['checkpoint_sha256'],p['elapsed_seconds'],p['peak_rss_bytes'])
PY
  )
  gcloud storage cp "$RESUME_URI/objects/$RESUME_OUTPUT_SHA/teacher.jsonl" "$RESULT/teacher.jsonl.partial" --project "$PROJECT_ID" >/dev/null
  gcloud storage cp "$RESUME_URI/objects/$RESUME_CHECKPOINT_SHA/checkpoint.json" "$RESULT/checkpoint.json" --project "$PROJECT_ID" >/dev/null
  [[ "$(sha "$RESULT/teacher.jsonl.partial")" == "$RESUME_OUTPUT_SHA" ]]
  [[ "$(sha "$RESULT/checkpoint.json")" == "$RESUME_CHECKPOINT_SHA" ]]
fi

status running
set +e
/usr/bin/time -v -o "$RESULT/time.txt" \
  python -B -m ofc_regular.run_hu_m43_attempt07_development \
    --root-index "$ROOT_INDEX" \
    --output "$RESULT/teacher.jsonl" \
    --checkpoint "$RESULT/checkpoint.json" \
    --heartbeat "$RESULT/heartbeat.json" \
    --model artifacts/lambda_rank_candidate.pkl \
    --model-sha256 "$MODEL_SHA256" \
    --run-id "$RUN_ID" \
    --plan configs/hu_joint_policy_m43_attempt07.json \
    --ai-profiles src/ofc_regular/ai_profiles.py \
    --batch-child-selectors --native-batch-threads 4 \
    >"$RESULT/generator_summary.json" 2>"$RESULT/run.log" &
PID=$!
while kill -0 "$PID" >/dev/null 2>&1; do
  status running
  [[ -s "$RESULT/heartbeat.json" ]] && \
    gcloud storage cp "$RESULT/heartbeat.json" "$PREFIX/heartbeat/$OUTPUT_PREFIX.json" --project "$PROJECT_ID" >/dev/null 2>&1 || true
  # This progress checkpoint is operational only.  It is never used for resume
  # without the content-addressed output pair and COMMIT published below.
  [[ -s "$RESULT/checkpoint.json" ]] && \
    gcloud storage cp "$RESULT/checkpoint.json" "$PREFIX/progress/$OUTPUT_PREFIX.json" --project "$PROJECT_ID" >/dev/null 2>&1 || true
  sleep "$SYNC_INTERVAL_SECONDS"
done
wait "$PID"
RUN_EXIT=$?
PID=''
set -e
[[ $RUN_EXIT -eq 0 ]]

read -r ELAPSED_SECONDS PEAK_RSS_BYTES < <(python3 - "$RESULT/time.txt" <<'PY'
import re,sys
text=open(sys.argv[1],encoding='utf-8').read()
elapsed=float(re.search(r'Elapsed \(wall clock\) time.*:\s*([0-9:.]+)',text).group(1).split(':')[-1])
# GNU time also provides a precise user+system lower bound; wall format can be
# h:mm:ss, so parse it again robustly.
token=re.search(r'Elapsed \(wall clock\) time.*:\s*([0-9:.]+)',text).group(1)
parts=[float(v) for v in token.split(':')]
elapsed=sum(value*(60**index) for index,value in enumerate(reversed(parts)))
rss=int(re.search(r'Maximum resident set size \(kbytes\):\s*(\d+)',text).group(1))*1024
assert elapsed>=0 and rss>=0
print(elapsed,rss)
PY
)
if [[ -n "${RESUME_ELAPSED:-}" ]]; then
  ELAPSED_SECONDS="$RESUME_ELAPSED"
  PEAK_RSS_BYTES="$RESUME_RSS"
fi

OUTPUT_SHA256="$(sha "$RESULT/teacher.jsonl")"
CHECKPOINT_SHA256="$(sha "$RESULT/checkpoint.json")"
HEARTBEAT_SHA256="$(sha "$RESULT/heartbeat.json")"
SUMMARY_SHA256="$(sha "$RESULT/generator_summary.json")"
RUN_LOG_SHA256="$(sha "$RESULT/run.log")"
CONFIG_SHA256="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["config_sha256"])' "$RESULT/generator_summary.json")"

upload_once_or_verify "$RESULT/teacher.jsonl" "$RESUME_URI/objects/$OUTPUT_SHA256/teacher.jsonl"
upload_once_or_verify "$RESULT/checkpoint.json" "$RESUME_URI/objects/$CHECKPOINT_SHA256/checkpoint.json"
python3 - "$RUN_NAME" "$SHARD_INDEX" "$MANIFEST_SHA256" "$AUTHORIZATION_SHA256" "$OUTPUT_SHA256" "$CHECKPOINT_SHA256" "$ELAPSED_SECONDS" "$PEAK_RSS_BYTES" > /tmp/resume_COMMIT_new.json <<'PY'
import json,sys
run,shard,manifest,auth,output,checkpoint,elapsed,rss=sys.argv[1:]
p={'schema':'hu_m43_attempt07_development_resume_commit_v1','status':'committed_completed_root_pair','run_name':run,'shard':int(shard),'manifest_sha256':manifest,'authorization_sha256':auth,'output_sha256':output,'checkpoint_sha256':checkpoint,'elapsed_seconds':float(elapsed),'peak_rss_bytes':int(rss)}
print(json.dumps(p,sort_keys=True,separators=(',',':')))
PY
upload_once_or_verify /tmp/resume_COMMIT_new.json "$COMMIT_URI"

GLOBAL_CLAIM_SHA256="$(sha /tmp/global_claim.json)"
ROOT_CLAIM_SHA256="$(sha /tmp/root_claim.json)"
cp /tmp/attempt07_authorization.json "$RESULT/authorization.json"
cp /tmp/global_claim.json "$RESULT/global_claim.json"
cp /tmp/root_claim.json "$RESULT/root_claim.json"

python3 - "$SPEC" "$RUN_NAME" "$MANIFEST_SHA256" "$SOURCE_SHA256" "$STARTUP_SHA256" "$SCHEDULE_SHA256" "$PLAN_SHA256" "$STATUS_SHA256" "$MODEL_SHA256" "$AI_PROFILES_SHA256" "$PREFLIGHT_PLAN_SHA256" "$PREFLIGHT_AGGREGATE_SHA256" "$CLOSURE_SHA256" "$SOURCE_MODEL_MANIFEST_SHA256" "$SOURCE_NATIVE_MANIFEST_SHA256" "$AUTHORIZATION_SHA256" "$GLOBAL_CLAIM_SHA256" "$ROOT_CLAIM_SHA256" "$OUTPUT_SHA256" "$CHECKPOINT_SHA256" "$HEARTBEAT_SHA256" "$SUMMARY_SHA256" "$RUN_LOG_SHA256" "$CONFIG_SHA256" "$ELAPSED_SECONDS" "$PEAK_RSS_BYTES" > "$RESULT/DONE.json" <<'PY'
import json,sys
spec=json.loads(sys.argv[1]); v=sys.argv[2:]
keys=('run_name','manifest_sha256','source_sha256','startup_sha256','schedule_sha256','plan_sha256','status_sha256','model_sha256','ai_profiles_sha256','preflight_plan_sha256','preflight_aggregate_sha256','source_closure_sha256','source_model_manifest_sha256','source_native_manifest_sha256','authorization_sha256','global_claim_sha256','root_claim_sha256','output_sha256','checkpoint_sha256','heartbeat_sha256','generator_summary_sha256','run_log_sha256','config_sha256','elapsed_seconds','peak_rss_bytes')
p=dict(zip(keys,v,strict=True))
p.update({'schema':'hu_m43_attempt07_development_spot_done_v1','status':'complete','run_id':f"{p['run_name']}:shard={spec['shard']}",'shard':spec['shard'],'root_index':spec['root_index'],'root_profile':spec['root_profile'],'seeds':spec['seeds'],'output_prefix':spec['output_prefix'],'native_batch_threads':4,'teacher_values_are_realized_match_ev':False,'current_profile_mutated':False,'runtime_policy_activated':False})
p['elapsed_seconds']=float(p['elapsed_seconds']); p['peak_rss_bytes']=int(p['peak_rss_bytes'])
print(json.dumps(p,sort_keys=True,separators=(',',':')))
PY

upload_once_or_verify "$RESULT/teacher.jsonl" "$RESULT_URI/teacher.jsonl"
upload_once_or_verify "$RESULT/checkpoint.json" "$RESULT_URI/checkpoint.json"
upload_once_or_verify "$RESULT/heartbeat.json" "$RESULT_URI/heartbeat.json"
upload_once_or_verify "$RESULT/generator_summary.json" "$RESULT_URI/generator_summary.json"
upload_once_or_verify "$RESULT/run.log" "$RESULT_URI/run.log"
upload_once_or_verify "$RESULT/authorization.json" "$RESULT_URI/authorization.json"
upload_once_or_verify "$RESULT/global_claim.json" "$RESULT_URI/global_claim.json"
upload_once_or_verify "$RESULT/root_claim.json" "$RESULT_URI/root_claim.json"
gcloud storage cp "$STARTUP_LOG" "$PREFIX/logs/startup_shard_${SHARD_INDEX}.log" --project "$PROJECT_ID" >/dev/null || true
# DONE is the final create-only publication.  Readers never address teacher
# content until every one of these 100 exact markers has validated.
upload_once_or_verify "$RESULT/DONE.json" "$DONE_URI"
status complete
