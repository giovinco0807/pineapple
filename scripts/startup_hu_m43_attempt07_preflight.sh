#!/usr/bin/env bash
set -euo pipefail

# One bounded, already-opened Attempt06 root per VM.  This worker never calls
# a root generator and the exported proof contains no per-arm values.
export HOME="${HOME:-/root}"
log(){ echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
meta(){ curl -fsS -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }
imeta(){ curl -fsS -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/$1"; }
sha(){ sha256sum "$1" | awk '{print $1}'; }

RUN_NAME="$(meta RUN_NAME)"
PROJECT_ID="$(meta PROJECT_ID)"
BUCKET="$(meta BUCKET)"
JOB_INDEX="$(meta JOB_INDEX)"
SOURCE_URI="$(meta SOURCE_URI)"
SOURCE_SHA256="$(meta SOURCE_SHA256)"
STARTUP_SHA256="$(meta STARTUP_SHA256)"
MANIFEST_SHA256="$(meta MANIFEST_SHA256)"
SCHEDULE_SHA256="$(meta SCHEDULE_SHA256)"
PREFLIGHT_PLAN_SHA256="$(meta PREFLIGHT_PLAN_SHA256)"
ATTEMPT07_PLAN_SHA256="$(meta ATTEMPT07_PLAN_SHA256)"
SOURCE_MERGED_SHA256="$(meta SOURCE_MERGED_SHA256)"
MODEL_SHA256="$(meta MODEL_SHA256)"
AI_PROFILES_SHA256="$(meta AI_PROFILES_SHA256)"
AUTHORIZATION_SHA256="$(meta AUTHORIZATION_SHA256)"
OVERLAY_CLOSURE_SHA256="$(meta OVERLAY_CLOSURE_SHA256)"
PACKAGE_TREE_SHA256="$(meta PACKAGE_TREE_SHA256)"
SYNC_INTERVAL_SECONDS="$(meta SYNC_INTERVAL_SECONDS)"
NATIVE_BATCH_THREADS="$(meta NATIVE_BATCH_THREADS)"
SELF_DELETE="$(meta SELF_DELETE)"
INSTANCE_NAME="$(imeta name)"
ZONE_PATH="$(imeta zone)"
ZONE="${ZONE_PATH##*/}"

curl -fsS -H 'Metadata-Flavor: Google' \
  'http://metadata.google.internal/computeMetadata/v1/instance/attributes/startup-script' \
  > /tmp/attempt07_preflight_executing_startup.sh
[[ "$(sha /tmp/attempt07_preflight_executing_startup.sh)" == "$STARTUP_SHA256" ]]

[[ "$NATIVE_BATCH_THREADS" == 4 ]]
PREFIX="gs://${BUCKET}/runs/${RUN_NAME}"
WORK=/opt/ofc-hu-m43-attempt07-preflight
RESULT=/tmp/ofc-hu-m43-attempt07-preflight-result
STARTUP_LOG=/tmp/ofc-hu-m43-attempt07-preflight-startup.log
PID=""
OUTPUT_PREFIX=""
START_MONOTONIC=""
: > "$STARTUP_LOG"
exec > >(tee -a "$STARTUP_LOG") 2>&1

gcs_state(){
  local uri="$1" out err message
  out="$(mktemp)"; err="$(mktemp)"
  if gcloud storage objects describe "$uri" --project "$PROJECT_ID" --format='value(name)' >"$out" 2>"$err"; then
    rm -f "$out" "$err"; echo present; return 0
  fi
  message="$(cat "$out" "$err")"; rm -f "$out" "$err"
  if grep -Eiq 'not found|does not exist|No URLs matched|404' <<<"$message"; then
    echo absent; return 0
  fi
  echo "unable to prove exact GCS object state: $uri: $message" >&2
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

write_checkpoint(){
  local state="$1" output_sha="${2:-}" attempts="${3:-1}"
  python3 - "$RESULT/checkpoint.json" "$RUN_NAME" "$JOB_INDEX" "$JOB_ID" "$SOURCE_ROOT_INDEX" "$BATCH_FLAG" "$OUTPUT_PREFIX" "$state" "$output_sha" "$attempts" "$MANIFEST_SHA256" "$AUTHORIZATION_SHA256" "$ATTEMPT07_PLAN_SHA256" "$SOURCE_MERGED_SHA256" "$MODEL_SHA256" "$AI_PROFILES_SHA256" <<'PY'
import json,sys,time
(path,run,index,job,root,batch,prefix,state,output_sha,attempts,manifest,authorization,plan,
 source,model,ai)=sys.argv[1:]
payload={
 'schema':'hu_m43_attempt07_preflight_checkpoint_v1','run_name':run,
 'job_index':int(index),'job_id':job,'source_root_index':int(root),
 'batch_child_selectors':batch=='true','output_prefix':prefix,'status':state,
 'output_sha256':output_sha or None,
 'manifest_sha256':manifest,'authorization_sha256':authorization,'attempt07_plan_sha256':plan,
 'source_merged_sha256':source,'model_sha256':model,'ai_profiles_sha256':ai,
 'deterministic_recompute_allowed':True,'new_root_generation_allowed':False,
 'current_profile_resolved':False,
}
if state=='running':
 payload['attempts']=int(attempts); payload['updated_unix_seconds']=time.time()
with open(path,'w',encoding='utf-8',newline='\n') as handle:
 json.dump(payload,handle,sort_keys=True,separators=(',',':')); handle.write('\n')
PY
  gcloud storage cp "$RESULT/checkpoint.json" "$PREFIX/resume/$OUTPUT_PREFIX/checkpoint.json" --project "$PROJECT_ID" >/dev/null
}

write_heartbeat(){
  local state="$1" alive=false
  [[ -n "$PID" ]] && kill -0 "$PID" >/dev/null 2>&1 && alive=true || true
  python3 - "$RESULT/live_heartbeat.json" "$RUN_NAME" "$JOB_INDEX" "$JOB_ID" "$SOURCE_ROOT_INDEX" "$BATCH_FLAG" "$OUTPUT_PREFIX" "$state" "$alive" <<'PY'
import json,sys,time
path,run,index,job,root,batch,prefix,state,alive=sys.argv[1:]
payload={
 'schema':'hu_m43_attempt07_preflight_worker_heartbeat_v1','run_name':run,
 'job_index':int(index),'job_id':job,'source_root_index':int(root),
 'batch_child_selectors':batch=='true','output_prefix':prefix,'status':state,
 'process_alive':alive=='true' if state=='running' else False,
 'mid_job_resume_mode':'same_input_same_seed_deterministic_recompute',
 'new_root_generation_allowed':False,'current_profile_resolved':False,
}
if state=='running': payload['updated_unix_seconds']=time.time()
with open(path,'w',encoding='utf-8',newline='\n') as handle:
 json.dump(payload,handle,sort_keys=True,separators=(',',':')); handle.write('\n')
PY
  gcloud storage cp "$RESULT/live_heartbeat.json" "$PREFIX/heartbeat/$OUTPUT_PREFIX.json" --project "$PROJECT_ID" >/dev/null || true
}

cleanup(){
  local code=$?
  set +e
  if [[ $code -ne 0 && -n "$OUTPUT_PREFIX" ]]; then
    [[ -s "$RESULT/checkpoint.json" ]] && gcloud storage cp "$RESULT/checkpoint.json" "$PREFIX/resume/$OUTPUT_PREFIX/checkpoint.json" --project "$PROJECT_ID" >/dev/null 2>&1 || true
    gcloud storage cp "$STARTUP_LOG" "$PREFIX/logs/$OUTPUT_PREFIX.log" --project "$PROJECT_ID" >/dev/null 2>&1 || true
    write_heartbeat failed || true
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

sudo rm -rf "$WORK" "$RESULT"
sudo mkdir -p "$WORK" "$RESULT"
sudo chown -R "$(id -u):$(id -g)" "$WORK" "$RESULT"
gcloud storage cp "$SOURCE_URI" /tmp/ofc_attempt07_preflight_source.zip --project "$PROJECT_ID" >/dev/null
[[ "$(sha /tmp/ofc_attempt07_preflight_source.zip)" == "$SOURCE_SHA256" ]]
gcloud storage cp "$PREFIX/source/shards_manifest.jsonl" /tmp/shards_manifest.jsonl --project "$PROJECT_ID" >/dev/null
gcloud storage cp "$PREFIX/manifest.json" /tmp/manifest.json --project "$PROJECT_ID" >/dev/null
gcloud storage cp "$PREFIX/source/startup_hu_m43_attempt07_preflight.sh" /tmp/published_startup.sh --project "$PROJECT_ID" >/dev/null
gcloud storage cp "$PREFIX/source/spot_authorization.json" /tmp/spot_authorization.json --project "$PROJECT_ID" >/dev/null
[[ "$(sha /tmp/shards_manifest.jsonl)" == "$SCHEDULE_SHA256" ]]
[[ "$(sha /tmp/manifest.json)" == "$MANIFEST_SHA256" ]]
[[ "$(sha /tmp/published_startup.sh)" == "$STARTUP_SHA256" ]]
[[ "$(sha /tmp/spot_authorization.json)" == "$AUTHORIZATION_SHA256" ]]
unzip -q /tmp/ofc_attempt07_preflight_source.zip -d "$WORK"
cd "$WORK"

[[ "$(sha shards_manifest.jsonl)" == "$SCHEDULE_SHA256" ]]
[[ "$(sha configs/hu_joint_policy_m43_attempt07_preflight.json)" == "$PREFLIGHT_PLAN_SHA256" ]]
[[ "$(sha configs/hu_joint_policy_m43_attempt07.json)" == "$ATTEMPT07_PLAN_SHA256" ]]
[[ "$(sha preflight_source/teacher.jsonl)" == "$SOURCE_MERGED_SHA256" ]]
[[ "$(sha artifacts/lambda_rank_candidate.pkl)" == "$MODEL_SHA256" ]]
[[ "$(sha src/ofc_regular/ai_profiles.py)" == "$AI_PROFILES_SHA256" ]]
[[ "$(sha preflight_overlay_manifest.json)" == "$OVERLAY_CLOSURE_SHA256" ]]
python3 - "$PACKAGE_TREE_SHA256" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path('.')
closure=json.loads((root/'preflight_overlay_manifest.json').read_text(encoding='utf-8'))
assert closure['schema']=='hu_m43_attempt07_preflight_overlay_closure_v1'
assert closure['new_root_generated'] is False and closure['teacher_executed'] is False
expected={row['path'] for row in closure['files']}|{'preflight_overlay_manifest.json'}
actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
assert actual==expected
for row in closure['files']:
 p=root/row['path']; data=p.read_bytes()
 assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['sha256']
rows=[]
for relative in sorted(actual):
 p=root/relative; rows.append({'path':relative,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
encoded=(json.dumps(rows,sort_keys=True,separators=(',',':'))+'\n').encode('ascii')
assert hashlib.sha256(encoded).hexdigest()==sys.argv[1]
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

SPEC="$(sed -n "$((JOB_INDEX+1))p" shards_manifest.jsonl)"
[[ -n "$SPEC" ]]
field(){ python3 -c 'import json,sys; print(json.loads(sys.argv[1])[sys.argv[2]])' "$SPEC" "$1"; }
[[ "$(field schema)" == hu_m43_attempt07_preflight_spot_job_v1 ]]
[[ "$(field job_index)" == "$JOB_INDEX" ]]
[[ "$(field machine_type)" == c4-standard-4 ]]
[[ "$(field native_batch_threads)" == 4 ]]
[[ "$(field new_root_generation_allowed)" == False ]]
JOB_ID="$(field job_id)"
SOURCE_ROOT_INDEX="$(field source_root_index)"
BATCH_FLAG="$(field batch_child_selectors | tr '[:upper:]' '[:lower:]')"
OUTPUT_PREFIX="$(field output_prefix)"
RESULT_URI="$PREFIX/results/$OUTPUT_PREFIX"
DONE_URI="$RESULT_URI/DONE.json"
PROOF_URI="$RESULT_URI/preflight.json"

[[ "$SOURCE_ROOT_INDEX" == 0 || "$SOURCE_ROOT_INDEX" == 1 || "$SOURCE_ROOT_INDEX" == 2 ]]
[[ "$BATCH_FLAG" == true || "$BATCH_FLAG" == false ]]
mkdir -p "$RESULT"

if [[ "$(gcs_state "$DONE_URI")" == present ]]; then
  gcloud storage cp "$DONE_URI" "$RESULT/existing_DONE.json" --project "$PROJECT_ID" >/dev/null
  python3 - "$RESULT/existing_DONE.json" "$RUN_NAME" "$JOB_INDEX" "$JOB_ID" "$SOURCE_ROOT_INDEX" "$BATCH_FLAG" "$OUTPUT_PREFIX" "$MANIFEST_SHA256" "$AUTHORIZATION_SHA256" "$SCHEDULE_SHA256" "$ATTEMPT07_PLAN_SHA256" "$SOURCE_MERGED_SHA256" "$MODEL_SHA256" "$AI_PROFILES_SHA256" <<'PY'
import json,math,re,sys
(path,run,index,job,root,batch,prefix,manifest,authorization,schedule,plan,source,model,ai)=sys.argv[1:]
d=json.load(open(path,encoding='utf-8'))
expected={'schema':'hu_m43_attempt07_preflight_spot_done_v1','status':'complete','run_name':run,
 'job_index':int(index),'job_id':job,'source_root_index':int(root),
 'batch_child_selectors':batch=='true','native_batch_threads':4,'output_prefix':prefix,
 'manifest_sha256':manifest,'authorization_sha256':authorization,'schedule_sha256':schedule,'attempt07_plan_sha256':plan,
 'source_merged_sha256':source,'model_sha256':model,'ai_profiles_sha256':ai,
 'teacher_values_exported':False,'arm_selection_performed':False,
 'new_root_generated':False,'current_profile_resolved':False,
 'current_profile_mutated':False,'runtime_policy_activated':False}
for key,value in expected.items(): assert type(d.get(key)) is type(value) and d[key]==value,(key,d.get(key),value)
assert isinstance(d.get('elapsed_seconds'),(int,float)) and not isinstance(d['elapsed_seconds'],bool) and math.isfinite(float(d['elapsed_seconds'])) and d['elapsed_seconds']>=0
assert type(d.get('peak_rss_bytes')) is int and d['peak_rss_bytes']>=0
for key in ('output_sha256','checkpoint_sha256','heartbeat_sha256','summary_sha256','run_log_sha256'):
 assert isinstance(d.get(key),str) and re.fullmatch(r'[0-9a-f]{64}',d[key])
PY
  log "verified existing DONE job=$JOB_ID; proof not opened"
  exit 0
fi

ATTEMPTS=1
if [[ "$(gcs_state "$PREFIX/resume/$OUTPUT_PREFIX/checkpoint.json")" == present ]]; then
  gcloud storage cp "$PREFIX/resume/$OUTPUT_PREFIX/checkpoint.json" "$RESULT/previous_checkpoint.json" --project "$PROJECT_ID" >/dev/null
  ATTEMPTS="$(python3 - "$RESULT/previous_checkpoint.json" "$RUN_NAME" "$JOB_INDEX" "$JOB_ID" "$SOURCE_ROOT_INDEX" "$BATCH_FLAG" "$MANIFEST_SHA256" "$AUTHORIZATION_SHA256" <<'PY'
import json,sys
path,run,index,job,root,batch,manifest,authorization=sys.argv[1:]
p=json.load(open(path,encoding='utf-8'))
assert p['schema']=='hu_m43_attempt07_preflight_checkpoint_v1'
expected={'run_name':run,'job_index':int(index),'job_id':job,'source_root_index':int(root),
          'batch_child_selectors':batch=='true','manifest_sha256':manifest,
          'authorization_sha256':authorization,
          'deterministic_recompute_allowed':True,'new_root_generation_allowed':False,
          'current_profile_resolved':False}
for key,value in expected.items(): assert type(p.get(key)) is type(value) and p[key]==value
print(int(p.get('attempts',0))+1)
PY
)"
fi

validate_proof(){
  python3 - "$1" "$SOURCE_ROOT_INDEX" "$BATCH_FLAG" "$ATTEMPT07_PLAN_SHA256" <<'PY'
import json,sys
path,root,batch,plan=sys.argv[1:]
r=json.load(open(path,encoding='utf-8'))
assert r['schema']=='hu_m43_attempt07_preflight_row_v1'
assert r['status']=='pass_preflight_only_no_arm_selection'
assert r['source']['source_root_index']==int(root)
assert r['execution']['batch_child_selectors'] is (batch=='true')
assert r['contract']['plan_sha256']==plan
assert r['science_boundary']['arm_selection_allowed'] is False
assert r['science_boundary']['current_profile_resolved'] is False
forbidden={'arms','selected_action_key','override_fired','mean','p01','p05','min'}
def visit(v,path='row'):
 if isinstance(v,dict):
  for k,c in v.items():
   assert k not in forbidden,(path,k)
   if k in {'screen','rerank','veto','assessment'} and isinstance(c,(dict,list)): raise AssertionError((path,k))
   visit(c,path+'.'+k)
 elif isinstance(v,list):
  for i,c in enumerate(v): visit(c,f'{path}[{i}]')
visit(r)
PY
}

RECOVER_COMPLETED=0
SUMMARY_URI="$RESULT_URI/summary.json"
if [[ "$(gcs_state "$PROOF_URI")" == present ]]; then
  gcloud storage cp "$PROOF_URI" "$RESULT/previous_preflight.json" --project "$PROJECT_ID" >/dev/null
  validate_proof "$RESULT/previous_preflight.json"
  if [[ "$(gcs_state "$SUMMARY_URI")" == present ]]; then
    gcloud storage cp "$SUMMARY_URI" "$RESULT/summary.json" --project "$PROJECT_ID" >/dev/null
    read -r ELAPSED_SECONDS PEAK_RSS_BYTES < <(python3 - "$RESULT/summary.json" "$RUN_NAME" "$JOB_INDEX" "$JOB_ID" "$SOURCE_ROOT_INDEX" "$BATCH_FLAG" "$OUTPUT_PREFIX" "$(sha "$RESULT/previous_preflight.json")" "$MANIFEST_SHA256" "$AUTHORIZATION_SHA256" <<'PY'
import json,math,sys
path,run,index,job,root,batch,prefix,output_sha,manifest,authorization=sys.argv[1:]
s=json.load(open(path,encoding='utf-8'))
expected={'schema':'hu_m43_attempt07_preflight_summary_v1','status':'complete','run_name':run,
 'job_index':int(index),'job_id':job,'source_root_index':int(root),
 'batch_child_selectors':batch=='true','native_batch_threads':4,'output_prefix':prefix,
 'output_sha256':output_sha,'manifest_sha256':manifest,'authorization_sha256':authorization,
 'teacher_values_exported':False,'arm_selection_performed':False,
 'new_root_generated':False,'current_profile_resolved':False}
for key,value in expected.items(): assert type(s.get(key)) is type(value) and s[key]==value
elapsed=s.get('elapsed_seconds'); rss=s.get('peak_rss_bytes')
assert isinstance(elapsed,(int,float)) and not isinstance(elapsed,bool) and math.isfinite(float(elapsed)) and elapsed>=0
assert type(rss) is int and rss>=0
print(float(elapsed),rss)
PY
)
    cp "$RESULT/previous_preflight.json" "$RESULT/preflight.json"
    if [[ "$(gcs_state "$RESULT_URI/run.log")" == present ]]; then
      gcloud storage cp "$RESULT_URI/run.log" "$RESULT/run.log" --project "$PROJECT_ID" >/dev/null
    else
      printf 'recovered completed proof and resource summary after preemption\n' > "$RESULT/run.log"
    fi
    OUTPUT_SHA="$(sha "$RESULT/preflight.json")"
    RECOVER_COMPLETED=1
    log "recovered immutable proof and truthful resource summary job=$JOB_ID"
  else
    log "prior proof lacks resource summary; deterministic recompute required job=$JOB_ID"
  fi
fi

if [[ "$RECOVER_COMPLETED" == 0 ]]; then
  write_checkpoint running "" "$ATTEMPTS"
  write_heartbeat running
  MODE_FLAG=--no-batch-child-selectors
  [[ "$BATCH_FLAG" == true ]] && MODE_FLAG=--batch-child-selectors
  START_MONOTONIC="$(python3 -c 'import time; print(time.monotonic())')"
  set +e
  /usr/bin/time -f '%M' -o "$RESULT/peak_rss_kb.txt" \
    python -B -m ofc_regular.run_hu_m43_attempt07_preflight \
      --source-root-index "$SOURCE_ROOT_INDEX" \
      --output "$RESULT/preflight.json" \
      --source preflight_source/teacher.jsonl \
      --preflight-plan configs/hu_joint_policy_m43_attempt07_preflight.json \
      --attempt07-plan configs/hu_joint_policy_m43_attempt07.json \
      --model artifacts/lambda_rank_candidate.pkl \
      --ai-profiles src/ofc_regular/ai_profiles.py \
      "$MODE_FLAG" --native-batch-threads 4 \
      > "$RESULT/run.log" 2>&1 &
  PID=$!
  set -e
  while kill -0 "$PID" >/dev/null 2>&1; do
    for ((i=0;i<SYNC_INTERVAL_SECONDS;i++)); do
      kill -0 "$PID" >/dev/null 2>&1 || break
      sleep 1
    done
    if kill -0 "$PID" >/dev/null 2>&1; then
      write_checkpoint running "" "$ATTEMPTS"
      write_heartbeat running
      gcloud storage cp "$RESULT/run.log" "$PREFIX/logs/$OUTPUT_PREFIX.log" --project "$PROJECT_ID" >/dev/null || true
    fi
  done
  set +e; wait "$PID"; code=$?; set -e; PID=""
  [[ $code -eq 0 ]] || exit "$code"
  validate_proof "$RESULT/preflight.json"
  upload_once_or_verify "$RESULT/preflight.json" "$PROOF_URI"
  OUTPUT_SHA="$(sha "$RESULT/preflight.json")"
  ELAPSED_SECONDS="$(python3 - "$START_MONOTONIC" <<'PY'
import sys,time
print(max(0.0,time.monotonic()-float(sys.argv[1])))
PY
)"
  if [[ -s "$RESULT/peak_rss_kb.txt" ]]; then
    PEAK_RSS_BYTES="$(( $(tr -d '[:space:]' < "$RESULT/peak_rss_kb.txt") * 1024 ))"
  else
    PEAK_RSS_BYTES=0
  fi
  python3 - "$RESULT/summary.json" "$RUN_NAME" "$JOB_INDEX" "$JOB_ID" "$SOURCE_ROOT_INDEX" "$BATCH_FLAG" "$OUTPUT_PREFIX" "$OUTPUT_SHA" "$ELAPSED_SECONDS" "$PEAK_RSS_BYTES" "$MANIFEST_SHA256" "$AUTHORIZATION_SHA256" <<'PY'
import json,sys
(path,run,index,job,root,batch,prefix,output_sha,elapsed,rss,manifest,
 authorization)=sys.argv[1:]
payload={'schema':'hu_m43_attempt07_preflight_summary_v1','status':'complete','run_name':run,
 'job_index':int(index),'job_id':job,'source_root_index':int(root),
 'batch_child_selectors':batch=='true','native_batch_threads':4,'output_prefix':prefix,
 'output_sha256':output_sha,'manifest_sha256':manifest,'authorization_sha256':authorization,
 'elapsed_seconds':float(elapsed),'peak_rss_bytes':int(rss),
 'teacher_values_exported':False,'arm_selection_performed':False,
 'new_root_generated':False,'current_profile_resolved':False}
with open(path,'w',encoding='utf-8',newline='\n') as handle:
 json.dump(payload,handle,sort_keys=True,separators=(',',':')); handle.write('\n')
PY
fi

write_checkpoint complete "$OUTPUT_SHA" "$ATTEMPTS"
write_heartbeat complete
cp "$RESULT/live_heartbeat.json" "$RESULT/heartbeat.json"

[[ -s "$RESULT/run.log" ]] || printf 'completed without runner stdout\n' > "$RESULT/run.log"
CHECKPOINT_SHA="$(sha "$RESULT/checkpoint.json")"
HEARTBEAT_SHA="$(sha "$RESULT/heartbeat.json")"
SUMMARY_SHA="$(sha "$RESULT/summary.json")"
RUN_LOG_SHA="$(sha "$RESULT/run.log")"
python3 - "$RESULT/DONE.json" "$RUN_NAME" "$JOB_INDEX" "$JOB_ID" "$SOURCE_ROOT_INDEX" "$BATCH_FLAG" "$OUTPUT_PREFIX" "$OUTPUT_SHA" "$CHECKPOINT_SHA" "$HEARTBEAT_SHA" "$SUMMARY_SHA" "$RUN_LOG_SHA" "$MANIFEST_SHA256" "$AUTHORIZATION_SHA256" "$SCHEDULE_SHA256" "$ATTEMPT07_PLAN_SHA256" "$SOURCE_MERGED_SHA256" "$MODEL_SHA256" "$AI_PROFILES_SHA256" "$ELAPSED_SECONDS" "$PEAK_RSS_BYTES" <<'PY'
import json,sys
(path,run,index,job,root,batch,prefix,output_sha,checkpoint_sha,heartbeat_sha,
 summary_sha,run_log_sha,manifest,authorization,schedule,plan,source,model,ai,elapsed,rss)=sys.argv[1:]
payload={'schema':'hu_m43_attempt07_preflight_spot_done_v1','status':'complete',
 'run_name':run,'job_index':int(index),'job_id':job,'source_root_index':int(root),
 'batch_child_selectors':batch=='true','native_batch_threads':4,'output_prefix':prefix,
 'output_sha256':output_sha,'checkpoint_sha256':checkpoint_sha,
 'heartbeat_sha256':heartbeat_sha,'summary_sha256':summary_sha,'run_log_sha256':run_log_sha,
 'manifest_sha256':manifest,'authorization_sha256':authorization,'schedule_sha256':schedule,'attempt07_plan_sha256':plan,
 'source_merged_sha256':source,'model_sha256':model,'ai_profiles_sha256':ai,
 'elapsed_seconds':float(elapsed),'peak_rss_bytes':int(rss),
 'teacher_values_exported':False,'arm_selection_performed':False,
 'new_root_generated':False,'current_profile_resolved':False,
 'current_profile_mutated':False,'runtime_policy_activated':False}
with open(path,'w',encoding='utf-8',newline='\n') as handle:
 json.dump(payload,handle,sort_keys=True,separators=(',',':')); handle.write('\n')
PY

upload_once_or_verify "$RESULT/preflight.json" "$RESULT_URI/preflight.json"
upload_once_or_verify "$RESULT/checkpoint.json" "$RESULT_URI/checkpoint.json"
upload_once_or_verify "$RESULT/heartbeat.json" "$RESULT_URI/heartbeat.json"
upload_once_or_verify "$RESULT/summary.json" "$RESULT_URI/summary.json"
upload_once_or_verify "$RESULT/run.log" "$RESULT_URI/run.log"
upload_once_or_verify "$RESULT/DONE.json" "$DONE_URI"
gcloud storage cp "$STARTUP_LOG" "$PREFIX/logs/$OUTPUT_PREFIX.log" --project "$PROJECT_ID" >/dev/null || true
log "complete job=$JOB_ID source_root=$SOURCE_ROOT_INDEX batch=$BATCH_FLAG"
