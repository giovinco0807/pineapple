#!/usr/bin/env bash
set -euo pipefail

# This worker is intentionally one root per VM.  Package verification and both
# irreversible claims happen before build-root-input is allowed to deal cards.
export HOME="${HOME:-/root}"
log(){ echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
meta(){ curl -fsS -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }
imeta(){ curl -fsS -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/$1"; }
sha(){ sha256sum "$1" | awk '{print $1}'; }

RUN_NAME="$(meta RUN_NAME)"
PROJECT_ID="$(meta PROJECT_ID)"
BUCKET="$(meta BUCKET)"
SHARD_INDEX="$(meta SHARD_INDEX)"
SOURCE_URI="$(meta SOURCE_URI)"
SOURCE_SHA256="$(meta SOURCE_SHA256)"
STARTUP_SHA256="$(meta STARTUP_SHA256)"
MANIFEST_SHA256="$(meta MANIFEST_SHA256)"
SCHEDULE_SHA256="$(meta SCHEDULE_SHA256)"
PLAN_SHA256="$(meta PLAN_SHA256)"
STATUS_SHA256="$(meta STATUS_SHA256)"
MODEL_SHA256="$(meta MODEL_SHA256)"
CLOSURE_SHA256="$(meta CLOSURE_SHA256)"
SOURCE_MODEL_MANIFEST_SHA256="$(meta SOURCE_MODEL_MANIFEST_SHA256)"
SOURCE_NATIVE_MANIFEST_SHA256="$(meta SOURCE_NATIVE_MANIFEST_SHA256)"
SYNC_INTERVAL_SECONDS="$(meta SYNC_INTERVAL_SECONDS)"
NATIVE_BATCH_THREADS="$(meta NATIVE_BATCH_THREADS)"
SELF_DELETE="$(meta SELF_DELETE)"
INSTANCE_NAME="$(imeta name)"
ZONE_PATH="$(imeta zone)"
ZONE="${ZONE_PATH##*/}"

PREFIX="gs://${BUCKET}/runs/${RUN_NAME}"
WORK=/opt/ofc-hu-m43-attempt06
RESULT=/tmp/ofc-hu-m43-attempt06-result
SYNC=/tmp/ofc-hu-m43-attempt06-sync
STARTUP_LOG=/tmp/ofc-hu-m43-attempt06-startup.log
: > "$STARTUP_LOG"
exec > >(tee -a "$STARTUP_LOG") 2>&1
OUTPUT_PREFIX=""
PID=""

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
  echo "unable to prove exact GCS object state for $uri: $message" >&2
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
    echo owner
    return 0
  fi
  existing="$(mktemp)"
  gcloud storage cp "$uri" "$existing" --project "$PROJECT_ID" >/dev/null
  cmp -s "$marker" "$existing" || { rm -f "$existing"; echo "fresh-boundary marker identity differs: $uri" >&2; return 1; }
  rm -f "$existing"
  echo existing
}

completed(){
  [[ -s "$RESULT/checkpoint.json" ]] || { echo 0; return; }
  python3 -c 'import json,sys; print(int(json.load(open(sys.argv[1])).get("completed_roots",0)))' "$RESULT/checkpoint.json" 2>/dev/null || echo 0
}

status(){
  python3 - "$RUN_NAME" "$SHARD_INDEX" "$1" "$(completed)" "$OUTPUT_PREFIX" > /tmp/status.json <<'PY'
import json,sys,time
run,shard,state,done,prefix=sys.argv[1:]
print(json.dumps({
    "schema":"hu_m43_attempt06_spot_status_v1",
    "run_name":run,
    "shard":int(shard),
    "status":state,
    "completed_roots":int(done),
    "target_roots":1,
    "output_prefix":prefix,
    "updated_unix_seconds":time.time(),
    "current_profile_mutated":False,
    "runtime_policy_activated":False,
},sort_keys=True,separators=(",",":")))
PY
  gcloud storage cp /tmp/status.json "$PREFIX/status/shard_${SHARD_INDEX}.json" --project "$PROJECT_ID" >/dev/null || true
}

worker_heartbeat(){
  local state="$1" alive=false
  [[ -n "$PID" ]] && kill -0 "$PID" >/dev/null 2>&1 && alive=true || true
  python3 - "$RUN_NAME" "$SHARD_INDEX" "$state" "$(completed)" "$OUTPUT_PREFIX" "$alive" > /tmp/worker_heartbeat.json <<'PY'
import json,sys,time
run,shard,state,done,prefix,alive=sys.argv[1:]
print(json.dumps({
  'schema':'hu_m43_attempt06_worker_heartbeat_v1','run_name':run,
  'shard':int(shard),'status':state,'completed_roots':int(done),'target_roots':1,
  'output_prefix':prefix,'teacher_process_alive':alive=='true',
  'root_internal_checkpoint_supported':False,
  'mid_root_preemption_resume_mode':'deterministic_full_root_recompute_same_frozen_closure',
  'updated_unix_seconds':time.time(),
},sort_keys=True,separators=(',',':')))
PY
  gcloud storage cp /tmp/worker_heartbeat.json "$PREFIX/heartbeat/shard_${SHARD_INDEX}.json" --project "$PROJECT_ID" >/dev/null || true
}

sync_resume(){
  [[ -n "$OUTPUT_PREFIX" ]] || return 0
  [[ -s "$RESULT/checkpoint.json" ]] || return 0
  rm -rf "$SYNC"
  mkdir -p "$SYNC"
  cp "$RESULT/checkpoint.json" "$SYNC/checkpoint.json"
  [[ -s "$RESULT/heartbeat.json" ]] && cp "$RESULT/heartbeat.json" "$SYNC/heartbeat.json" || true
  local checkpoint_state
  checkpoint_state="$(python3 - "$RESULT/teacher.jsonl.partial" "$SYNC/checkpoint.json" <<'PY'
import hashlib,json,pathlib,sys
partial,checkpoint=map(pathlib.Path,sys.argv[1:])
payload=json.loads(checkpoint.read_text(encoding="utf-8"))
assert payload["schema"]=="hu_m43_attempt06_t1_second_checkpoint_v1"
completed=int(payload["completed_roots"])
assert completed in (0,1)
if completed==0:
    assert payload["partial_sha256"]==hashlib.sha256(b"").hexdigest()
    assert not partial.exists() or partial.stat().st_size==0
    print("checkpoint0")
else:
    assert partial.is_file()
    assert hashlib.sha256(partial.read_bytes()).hexdigest()==payload["partial_sha256"]
    print("checkpoint1")
PY
  )"
  local resume_uri="$PREFIX/resume/$OUTPUT_PREFIX"
  gcloud storage cp "$SYNC/checkpoint.json" "$resume_uri/checkpoint.json" --project "$PROJECT_ID" >/dev/null
  if [[ "$checkpoint_state" == checkpoint1 ]]; then
    cp "$RESULT/teacher.jsonl.partial" "$SYNC/teacher.jsonl.partial"
    gcloud storage cp "$SYNC/teacher.jsonl.partial" "$resume_uri/teacher.jsonl.partial" --project "$PROJECT_ID" >/dev/null
  fi
  if [[ -s "$SYNC/heartbeat.json" ]]; then
    gcloud storage cp "$SYNC/heartbeat.json" "$resume_uri/heartbeat.json" --project "$PROJECT_ID" >/dev/null
  fi
  status running
}

cleanup(){
  local code=$?
  set +e
  if [[ $code -ne 0 ]]; then
    sync_resume
    gcloud storage cp "$STARTUP_LOG" "$PREFIX/logs/startup_shard_${SHARD_INDEX}.log" --project "$PROJECT_ID" >/dev/null 2>&1 || true
    status failed
    worker_heartbeat failed
  fi
  if [[ "$SELF_DELETE" == 1 ]]; then
    gcloud compute instances delete "$INSTANCE_NAME" --zone "$ZONE" --project "$PROJECT_ID" --quiet >/dev/null 2>&1 || sudo shutdown -h now
  fi
  exit "$code"
}
trap cleanup EXIT

export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -y
sudo apt-get install -y unzip curl ca-certificates python3 python3-venv python3-pip
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
gcloud storage cp "$SOURCE_URI" /tmp/ofc_attempt06_source.zip --project "$PROJECT_ID" >/dev/null
[[ "$(sha /tmp/ofc_attempt06_source.zip)" == "$SOURCE_SHA256" ]]
unzip -q /tmp/ofc_attempt06_source.zip -d "$WORK"
cd "$WORK"
[[ "$(sha shards_manifest.jsonl)" == "$SCHEDULE_SHA256" ]]
[[ "$(sha configs/hu_joint_policy_m43_attempt06.json)" == "$PLAN_SHA256" ]]
[[ "$(sha configs/hu_joint_policy_m43_attempt06_status.json)" == "$STATUS_SHA256" ]]
[[ "$(sha artifacts/lambda_rank_candidate.pkl)" == "$MODEL_SHA256" ]]
[[ "$(sha source_closure_manifest.json)" == "$CLOSURE_SHA256" ]]
[[ "$(sha source_model_manifest.json)" == "$SOURCE_MODEL_MANIFEST_SHA256" ]]
[[ "$(sha source_native_manifest.json)" == "$SOURCE_NATIVE_MANIFEST_SHA256" ]]
python3 - <<'PY'
import hashlib,json,pathlib
root=pathlib.Path('.')
closure=json.loads((root/'source_closure_manifest.json').read_text(encoding='utf-8'))
assert closure['schema']=='hu_m43_attempt06_spot_source_closure_v1'
assert closure['status']=='closed_no_fresh_seed_materialized'
assert closure['fresh_seed_content_opened'] is False
assert closure['teacher_executed'] is False
for row in closure['files']:
    path=root/row['path']
    assert path.is_file() and path.stat().st_size==row['bytes']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
PY

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

python - <<'PY'
from ofc_regular.hu_m3_rust import engine_version,load_native_engine
from ofc_regular.hu_turn3_stage3_feature_rust import rust_direct_available
print('hu_m3_engine='+engine_version(library=load_native_engine()))
assert rust_direct_available()
PY

SPEC="$(sed -n "$((SHARD_INDEX+1))p" shards_manifest.jsonl)"
[[ -n "$SPEC" ]]
field(){ python3 -c 'import json,sys; print(json.loads(sys.argv[1])[sys.argv[2]])' "$SPEC" "$1"; }
[[ "$(field schema)" == hu_m43_attempt06_spot_shard_v1 ]]
[[ "$(field shard)" == "$SHARD_INDEX" ]]
[[ "$(field root_index)" == "$SHARD_INDEX" ]]
[[ "$(field roots)" == 1 ]]
[[ "$(field candidate_samples)" == 8 ]]
[[ "$(field evaluation_samples)" == 128 ]]
[[ "$(field native_batch_threads)" == "$NATIVE_BATCH_THREADS" ]]
[[ "$(field learned_nonbaseline_top_k)" == 8 ]]
[[ "$(field baseline_profile)" == stage18_p1 ]]
[[ "$(field t2_profile)" == stage9f_p2 ]]
ROOT_INDEX="$(field root_index)"
HAND_SEED="$(field hand_seed)"
ROOT_PROFILE="$(field root_profile)"
OUTPUT_PREFIX="$(field output_prefix)"
RESULT_URI="$PREFIX/results/$OUTPUT_PREFIX"
DONE_URI="$RESULT_URI/DONE.json"
RESUME_URI="$PREFIX/resume/$OUTPUT_PREFIX"
ROOT_URI="$PREFIX/fresh_inputs/$OUTPUT_PREFIX/root.jsonl"
GLOBAL_MARKER_URI="$PREFIX/fresh_boundary/CONSUMED.json"
ROOT_CLAIM_URI="$PREFIX/fresh_boundary/$OUTPUT_PREFIX/CLAIMED.json"

mkdir -p "$RESULT" "$SYNC"
python3 - "$RUN_NAME" "$MANIFEST_SHA256" "$SOURCE_SHA256" "$STARTUP_SHA256" "$SCHEDULE_SHA256" "$PLAN_SHA256" "$STATUS_SHA256" "$MODEL_SHA256" "$CLOSURE_SHA256" "$SOURCE_MODEL_MANIFEST_SHA256" "$SOURCE_NATIVE_MANIFEST_SHA256" "$NATIVE_BATCH_THREADS" > "$RESULT/global_consumption_marker.json" <<'PY'
import json,sys
(run,manifest,source,startup,schedule,plan,status,model,closure,
 source_model_manifest,source_native_manifest,native_batch_threads)=sys.argv[1:]
print(json.dumps({
  'schema':'hu_m43_attempt06_global_consumption_marker_v1',
  'status':'consumed_before_any_fresh_root_content_read',
  'run_name':run,'manifest_sha256':manifest,'source_sha256':source,
  'startup_sha256':startup,'schedule_sha256':schedule,'plan_sha256':plan,
  'status_sha256':status,'model_sha256':model,'source_closure_sha256':closure,
  'source_model_manifest_sha256':source_model_manifest,
  'source_native_manifest_sha256':source_native_manifest,
  'native_batch_threads':int(native_batch_threads),
  'roots':50,'shards':50,'roots_per_shard':1,
  'fresh_200_fit_authorized':False,'runtime_policy_activated':False,
},sort_keys=True,separators=(',',':')))
PY
claim_once "$RESULT/global_consumption_marker.json" "$GLOBAL_MARKER_URI" >/dev/null

if [[ "$(gcs_state "$DONE_URI")" == present ]]; then
  gcloud storage cp "$DONE_URI" "$RESULT/existing_DONE.json" --project "$PROJECT_ID" >/dev/null
  python3 - "$RESULT/existing_DONE.json" "$RUN_NAME" "$SHARD_INDEX" "$ROOT_INDEX" "$HAND_SEED" "$ROOT_PROFILE" "$OUTPUT_PREFIX" "$SOURCE_SHA256" "$STARTUP_SHA256" "$MANIFEST_SHA256" "$SCHEDULE_SHA256" "$PLAN_SHA256" "$STATUS_SHA256" "$MODEL_SHA256" "$CLOSURE_SHA256" "$SOURCE_MODEL_MANIFEST_SHA256" "$SOURCE_NATIVE_MANIFEST_SHA256" "$NATIVE_BATCH_THREADS" <<'PY'
import json,sys
(path,run,shard,root,seed,profile,prefix,source,startup,manifest,schedule,plan,
 status,model,closure,source_model_manifest,source_native_manifest,native_batch_threads)=sys.argv[1:]
done=json.load(open(path,encoding='utf-8'))
expected={
 'schema':'hu_m43_attempt06_spot_done_v1','status':'complete','run_name':run,
 'run_id':f'{run}:shard={shard}',
 'shard':int(shard),'root_index':int(root),'hand_seed':int(seed),
 'root_profile':profile,'roots':1,'output_prefix':prefix,'source_sha256':source,
 'startup_sha256':startup,'manifest_sha256':manifest,'schedule_sha256':schedule,
 'plan_sha256':plan,'status_sha256':status,'model_sha256':model,
 'source_closure_sha256':closure,
 'source_model_manifest_sha256':source_model_manifest,
 'source_native_manifest_sha256':source_native_manifest,
 'native_batch_threads':int(native_batch_threads),
 'teacher_values_are_realized_match_ev':False,'current_profile_mutated':False,
 'runtime_policy_activated':False,
}
for key,value in expected.items(): assert done.get(key)==value,(key,done.get(key),value)
for key in ('input_sha256','output_sha256','checkpoint_sha256','heartbeat_sha256','config_sha256'):
 value=done.get(key); assert isinstance(value,str) and len(value)==64
PY
  status complete
  log "verified DONE exists shard=$SHARD_INDEX; no root reopened"
  exit 0
fi

python3 - "$RUN_NAME" "$ROOT_INDEX" "$HAND_SEED" "$ROOT_PROFILE" "$MANIFEST_SHA256" "$SOURCE_SHA256" "$STARTUP_SHA256" "$SCHEDULE_SHA256" "$PLAN_SHA256" "$STATUS_SHA256" "$MODEL_SHA256" "$CLOSURE_SHA256" "$SOURCE_MODEL_MANIFEST_SHA256" "$SOURCE_NATIVE_MANIFEST_SHA256" "$NATIVE_BATCH_THREADS" > "$RESULT/root_claim.json" <<'PY'
import json,sys
(run,index,seed,profile,manifest,source,startup,schedule,plan,status,model,closure,
 source_model_manifest,source_native_manifest,native_batch_threads)=sys.argv[1:]
print(json.dumps({
  'schema':'hu_m43_attempt06_root_consumption_claim_v1',
  'status':'claimed_before_materializing_policy_observation',
  'run_name':run,'run_id':f'{run}:shard={index}',
  'root_index':int(index),'hand_seed':int(seed),'root_profile':profile,
  'manifest_sha256':manifest,'source_sha256':source,'startup_sha256':startup,
  'schedule_sha256':schedule,'plan_sha256':plan,'status_sha256':status,
  'model_sha256':model,'source_closure_sha256':closure,
  'retry_same_seed_after_open_allowed':False,
  'fresh_audit_retry_or_alternate_sample_allowed':False,
  'deterministic_claim_recovery_allowed':True,
  'deterministic_claim_recovery_mode':
    'same_seed_same_frozen_closure_after_matching_claim_and_absent_root_only',
  'source_model_manifest_sha256':source_model_manifest,
  'source_native_manifest_sha256':source_native_manifest,
  'native_batch_threads':int(native_batch_threads),
},sort_keys=True,separators=(',',':')))
PY
ROOT_CLAIM_STATE="$(claim_once "$RESULT/root_claim.json" "$ROOT_CLAIM_URI")"

if [[ "$(gcs_state "$ROOT_URI")" == present ]]; then
  gcloud storage cp "$ROOT_URI" "$RESULT/root.jsonl" --project "$PROJECT_ID" >/dev/null
elif [[ "$ROOT_CLAIM_STATE" == owner || "$ROOT_CLAIM_STATE" == existing ]]; then
  # FIRST CARD-BEARING OPERATION for an owner.  For an existing byte-identical
  # claim with no immutable root object, this is deterministic crash recovery:
  # exactly the same hand seed and frozen source/startup/config closure are
  # rematerialized.  It is not a fresh-audit retry or an alternate sample.
  python -B -m ofc_regular.hu_m43_attempt06_spot build-root-input \
    --output "$RESULT/root.jsonl" \
    --root-index "$ROOT_INDEX" \
    --hand-seed "$HAND_SEED" \
    --root-profile "$ROOT_PROFILE" \
    --plan-sha256 "$PLAN_SHA256" \
    --schedule-sha256 "$SCHEDULE_SHA256" \
    --model-sha256 "$MODEL_SHA256" \
    --manifest-sha256 "$MANIFEST_SHA256" \
    --source-sha256 "$SOURCE_SHA256" \
    --startup-sha256 "$STARTUP_SHA256" \
    --status-sha256 "$STATUS_SHA256" \
    --source-closure-sha256 "$CLOSURE_SHA256" \
    --global-marker "$RESULT/global_consumption_marker.json" \
    --root-claim "$RESULT/root_claim.json" \
    --run-name "$RUN_NAME" \
    --run-id "$RUN_NAME:shard=$SHARD_INDEX" > "$RESULT/root_build_summary.json"
  upload_once_or_verify "$RESULT/root.jsonl" "$ROOT_URI"
else
  echo "invalid root claim state; refusing root materialization" >&2
  exit 73
fi

python - "$RESULT/root.jsonl" "$ROOT_INDEX" <<'PY'
import sys
from ofc_regular.hu_m43_attempt06_teacher import load_attempt06_roots
roots=load_attempt06_roots(sys.argv[1])
assert len(roots)==1 and roots[0].root_index==int(sys.argv[2])
PY

if [[ "$(gcs_state "$RESUME_URI/checkpoint.json")" == present ]]; then
  gcloud storage cp "$RESUME_URI/checkpoint.json" "$RESULT/checkpoint.json" --project "$PROJECT_ID" >/dev/null
  RESUME_COMPLETED="$(python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["schema"]=="hu_m43_attempt06_t1_second_checkpoint_v1"; print(int(p["completed_roots"]))' "$RESULT/checkpoint.json")"
  PARTIAL_STATE="$(gcs_state "$RESUME_URI/teacher.jsonl.partial")"
  if [[ "$RESUME_COMPLETED" == 0 ]]; then
    [[ "$PARTIAL_STATE" == absent ]] || { echo 'checkpoint0 unexpectedly has a remote partial; refusing ambiguous resume' >&2; exit 74; }
  elif [[ "$RESUME_COMPLETED" == 1 ]]; then
    [[ "$PARTIAL_STATE" == present ]] || { echo 'checkpoint1 exists without partial output' >&2; exit 74; }
    gcloud storage cp "$RESUME_URI/teacher.jsonl.partial" "$RESULT/teacher.jsonl.partial" --project "$PROJECT_ID" >/dev/null
  else
    echo 'resume checkpoint completed_roots is outside 0..1' >&2; exit 74
  fi
  if [[ "$(gcs_state "$RESUME_URI/heartbeat.json")" == present ]]; then
    gcloud storage cp "$RESUME_URI/heartbeat.json" "$RESULT/heartbeat.json" --project "$PROJECT_ID" >/dev/null
  fi
fi

status running
worker_heartbeat running
set +e
python -B -m ofc_regular.hu_m43_attempt06_teacher shard \
  --input-roots "$RESULT/root.jsonl" \
  --output "$RESULT/teacher.jsonl" \
  --checkpoint "$RESULT/checkpoint.json" \
  --heartbeat "$RESULT/heartbeat.json" \
  --model artifacts/lambda_rank_candidate.pkl \
  --model-sha256 "$MODEL_SHA256" \
  --source-model-manifest-sha256 "$SOURCE_MODEL_MANIFEST_SHA256" \
  --source-native-manifest-sha256 "$SOURCE_NATIVE_MANIFEST_SHA256" \
  --run-id "$RUN_NAME:shard=$SHARD_INDEX" \
  --batch-child-selectors \
  --native-batch-threads "$NATIVE_BATCH_THREADS" \
  > "$RESULT/generator_summary.json" 2> "$RESULT/run.log" &
PID=$!
set -e
while kill -0 "$PID" >/dev/null 2>&1; do
  for ((i=0;i<SYNC_INTERVAL_SECONDS;i++)); do
    kill -0 "$PID" >/dev/null 2>&1 || break
    sleep 1
  done
  if kill -0 "$PID" >/dev/null 2>&1; then
    sync_resume
    status running
    worker_heartbeat running
  fi
done
set +e
wait "$PID"
code=$?
set -e
PID=""
[[ $code -eq 0 ]] || exit "$code"
[[ -s "$RESULT/teacher.jsonl" && -s "$RESULT/checkpoint.json" && -s "$RESULT/heartbeat.json" ]]
[[ "$(wc -l < "$RESULT/teacher.jsonl" | tr -d ' ')" == 1 ]]

INPUT_SHA="$(sha "$RESULT/root.jsonl")"
OUTPUT_SHA="$(sha "$RESULT/teacher.jsonl")"
CHECKPOINT_SHA="$(sha "$RESULT/checkpoint.json")"
HEARTBEAT_SHA="$(sha "$RESULT/heartbeat.json")"
SUMMARY_SHA="$(sha "$RESULT/generator_summary.json")"
RUN_LOG_SHA="$(sha "$RESULT/run.log")"
GLOBAL_MARKER_SHA="$(sha "$RESULT/global_consumption_marker.json")"
ROOT_CLAIM_SHA="$(sha "$RESULT/root_claim.json")"
CONFIG_SHA="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8"))["config_sha256"])' "$RESULT/generator_summary.json")"
python3 - "$RESULT/DONE.json" "$RUN_NAME" "$SHARD_INDEX" "$ROOT_INDEX" "$HAND_SEED" "$ROOT_PROFILE" "$OUTPUT_PREFIX" "$INPUT_SHA" "$OUTPUT_SHA" "$CHECKPOINT_SHA" "$HEARTBEAT_SHA" "$SUMMARY_SHA" "$RUN_LOG_SHA" "$GLOBAL_MARKER_SHA" "$ROOT_CLAIM_SHA" "$SOURCE_SHA256" "$STARTUP_SHA256" "$MANIFEST_SHA256" "$SCHEDULE_SHA256" "$PLAN_SHA256" "$STATUS_SHA256" "$MODEL_SHA256" "$CLOSURE_SHA256" "$SOURCE_MODEL_MANIFEST_SHA256" "$SOURCE_NATIVE_MANIFEST_SHA256" "$NATIVE_BATCH_THREADS" "$CONFIG_SHA" <<'PY'
import json,sys,time
(path,run,shard,root,seed,profile,prefix,input_sha,output_sha,checkpoint_sha,
 heartbeat_sha,summary_sha,run_log_sha,global_marker_sha,root_claim_sha,source_sha,
 startup_sha,manifest_sha,schedule_sha,plan_sha,status_sha,model_sha,closure_sha,
 source_model_manifest_sha,source_native_manifest_sha,native_batch_threads,config_sha)=sys.argv[1:]
payload={
 'schema':'hu_m43_attempt06_spot_done_v1','status':'complete',
 'run_name':run,'run_id':f'{run}:shard={shard}',
 'shard':int(shard),'root_index':int(root),'hand_seed':int(seed),
 'root_profile':profile,'roots':1,'output_prefix':prefix,
 'input_sha256':input_sha,'output_sha256':output_sha,
 'checkpoint_sha256':checkpoint_sha,'heartbeat_sha256':heartbeat_sha,
 'generator_summary_sha256':summary_sha,'run_log_sha256':run_log_sha,
 'global_consumption_marker_sha256':global_marker_sha,
 'root_consumption_claim_sha256':root_claim_sha,'source_sha256':source_sha,
 'startup_sha256':startup_sha,'manifest_sha256':manifest_sha,
 'schedule_sha256':schedule_sha,'plan_sha256':plan_sha,'status_sha256':status_sha,
 'model_sha256':model_sha,'source_closure_sha256':closure_sha,
 'source_model_manifest_sha256':source_model_manifest_sha,
 'source_native_manifest_sha256':source_native_manifest_sha,
 'native_batch_threads':int(native_batch_threads),'config_sha256':config_sha,
 'teacher_values_are_realized_match_ev':False,'current_profile_mutated':False,
 'runtime_policy_activated':False,'completed_unix_seconds':time.time(),
}
with open(path,'w',encoding='utf-8',newline='\n') as handle:
 json.dump(payload,handle,sort_keys=True,separators=(',',':')); handle.write('\n')
PY

upload_once_or_verify "$RESULT/root.jsonl" "$RESULT_URI/root.jsonl"
upload_once_or_verify "$RESULT/teacher.jsonl" "$RESULT_URI/teacher.jsonl"
upload_once_or_verify "$RESULT/checkpoint.json" "$RESULT_URI/checkpoint.json"
upload_once_or_verify "$RESULT/heartbeat.json" "$RESULT_URI/heartbeat.json"
upload_once_or_verify "$RESULT/generator_summary.json" "$RESULT_URI/generator_summary.json"
upload_once_or_verify "$RESULT/run.log" "$RESULT_URI/run.log"
upload_once_or_verify "$RESULT/global_consumption_marker.json" "$RESULT_URI/global_consumption_marker.json"
upload_once_or_verify "$RESULT/root_claim.json" "$RESULT_URI/root_claim.json"
gcloud storage cp "$RESULT/DONE.json" "$DONE_URI" --project "$PROJECT_ID" --if-generation-match=0 >/dev/null
status complete
worker_heartbeat complete
log "complete shard=$SHARD_INDEX root=$ROOT_INDEX profile=$ROOT_PROFILE"
