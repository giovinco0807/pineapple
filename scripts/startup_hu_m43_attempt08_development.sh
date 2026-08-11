#!/usr/bin/env bash
set -euo pipefail

export HOME="${HOME:-/root}"
export PYTHONDONTWRITEBYTECODE=1
meta(){ curl -fsS -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }
imeta(){ curl -fsS -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/$1"; }
sha(){ sha256sum "$1" | awk '{print $1}'; }
remote_exists(){
  local error_file
  error_file="$(mktemp)"
  if gcloud storage objects describe "$1" --project "$PROJECT_ID" \
      >/dev/null 2>"$error_file"; then
    rm -f "$error_file"; return 0
  fi
  if grep -Eiq 'not found|does not exist|No URLs matched|404' "$error_file"; then
    rm -f "$error_file"; return 1
  fi
  cat "$error_file" >&2; rm -f "$error_file"
  echo "unable to prove remote object state: $1" >&2
  exit 70
}

RUN_NAME="$(meta RUN_NAME)"; PROJECT_ID="$(meta PROJECT_ID)"; BUCKET="$(meta BUCKET)"
SHARD="$(meta SHARD)"; SOURCE_URI="$(meta SOURCE_URI)"; SOURCE_SHA256="$(meta SOURCE_SHA256)"
MANIFEST_SHA256="$(meta MANIFEST_SHA256)"; AUTHORIZATION_SHA256="$(meta LAUNCH_AUTHORIZATION_SHA256)"
SCHEDULE_SHA256="$(meta SCHEDULE_SHA256)"; STARTUP_SHA256="$(meta STARTUP_SHA256)"
SELF_DELETE="$(meta SELF_DELETE)"; INSTANCE_NAME="$(imeta name)"; ZONE="$(imeta zone)"; ZONE="${ZONE##*/}"
PREFIX="gs://${BUCKET}/runs/${RUN_NAME}"
RUNMETA=/opt/ofc-attempt08-development
WORK="$RUNMETA/package_src"
VENV="$RUNMETA/.venv"
RESULT=/tmp/ofc-attempt08-development-result
LIVE_LOG=/tmp/ofc-attempt08-development-live.log
mkdir -p "$RESULT"; : >"$LIVE_LOG"; exec > >(tee -a "$LIVE_LOG") 2>&1

cleanup(){
  code=$?; set +e
  gcloud storage cp "$LIVE_LOG" "$PREFIX/logs/shard-${SHARD}.log" --project "$PROJECT_ID" >/dev/null 2>&1 || true
  if [[ "$SELF_DELETE" == 1 ]]; then
    gcloud compute instances delete "$INSTANCE_NAME" --zone "$ZONE" --project "$PROJECT_ID" --quiet >/dev/null 2>&1 || sudo shutdown -h now
  fi
  exit "$code"
}
trap cleanup EXIT

curl -fsS -H 'Metadata-Flavor: Google' \
  'http://metadata.google.internal/computeMetadata/v1/instance/attributes/startup-script' \
  >/tmp/executing-startup.sh
[[ "$(sha /tmp/executing-startup.sh)" == "$STARTUP_SHA256" ]]
[[ "$SHARD" =~ ^([0-9]|[1-9][0-9]|1[0-9][0-9])$ ]] && (( SHARD >= 0 && SHARD < 200 ))
command -v gcloud >/dev/null; command -v curl >/dev/null

# Record and later bind the actual boot disk image, not merely requested metadata.
DISK="$(gcloud compute instances describe "$INSTANCE_NAME" --zone "$ZONE" --project "$PROJECT_ID" --format='value(disks[0].source.basename())')"
DISK_JSON="$(gcloud compute disks describe "$DISK" --zone "$ZONE" --project "$PROJECT_ID" --format=json)"
python3 - "$DISK_JSON" "$RESULT/boot_image_evidence.json" "$RUN_NAME" "$SHARD" "$INSTANCE_NAME" "$DISK" <<'PY'
import json,sys
d=json.loads(sys.argv[1])
assert str(d.get('sourceImage','')).endswith('projects/debian-cloud/global/images/debian-12-bookworm-v20260609')
assert str(d.get('sourceImageId'))=='1449487925682397051'
p={'schema':'hu_m43_attempt08_development_boot_image_evidence_v1','run_name':sys.argv[3],
   'shard':int(sys.argv[4]),'instance_name':sys.argv[5],'disk_name':sys.argv[6],
   'source_image':d['sourceImage'],'source_image_id':str(d['sourceImageId'])}
open(sys.argv[2],'w').write(json.dumps(p,sort_keys=True,separators=(',',':'))+'\n')
PY

export DEBIAN_FRONTEND=noninteractive
sudo rm -f /etc/apt/sources.list.d/debian.sources
printf '%s\n' \
  'deb [check-valid-until=no] http://snapshot.debian.org/archive/debian/20260609T000000Z bookworm main' \
  'deb [check-valid-until=no] http://snapshot.debian.org/archive/debian-security/20260609T000000Z bookworm-security main' \
  | sudo tee /etc/apt/sources.list >/dev/null
sudo apt-get -o Acquire::Check-Valid-Until=false update -y
sudo apt-get -o Acquire::Check-Valid-Until=false install -y python3 python3-venv unzip time libgomp1 ca-certificates

sudo rm -rf "$RUNMETA"; sudo mkdir -p "$WORK"; sudo chown -R "$(id -u):$(id -g)" "$RUNMETA"
gcloud storage cp "$SOURCE_URI" /tmp/source.zip --project "$PROJECT_ID" >/dev/null
gcloud storage cp "$PREFIX/manifest.json" /tmp/manifest.json --project "$PROJECT_ID" >/dev/null
gcloud storage cp "$PREFIX/source/launch_authorization.json" /tmp/launch_authorization.json --project "$PROJECT_ID" >/dev/null
gcloud storage cp "$PREFIX/source/shards_manifest.jsonl" /tmp/shards_manifest.jsonl --project "$PROJECT_ID" >/dev/null
[[ "$(sha /tmp/source.zip)" == "$SOURCE_SHA256" ]]
[[ "$(sha /tmp/manifest.json)" == "$MANIFEST_SHA256" ]]
[[ "$(sha /tmp/launch_authorization.json)" == "$AUTHORIZATION_SHA256" ]]
[[ "$(sha /tmp/shards_manifest.jsonl)" == "$SCHEDULE_SHA256" ]]
unzip -q /tmp/source.zip -d "$WORK"

# Reconstruct the outer package so the worker re-runs the exact same validator.
cp /tmp/source.zip "$RUNMETA/ofc_regular_hu_m43_attempt08_development_source.zip"
cp /tmp/manifest.json "$RUNMETA/manifest.json"
cp /tmp/launch_authorization.json "$RUNMETA/launch_authorization.json"
cp /tmp/shards_manifest.jsonl "$RUNMETA/shards_manifest.jsonl"
cp /tmp/executing-startup.sh "$RUNMETA/startup_hu_m43_attempt08_development.sh"
cp "$WORK/source_closure_manifest.json" "$RUNMETA/source_closure_manifest.json"
cp "$WORK/configs/hu_joint_policy_m43_attempt08.json" "$RUNMETA/hu_joint_policy_m43_attempt08.json"
cp "$WORK/configs/hu_joint_policy_m43_attempt08_preflight.json" "$RUNMETA/hu_joint_policy_m43_attempt08_preflight.json"
cp "$WORK/frozen/development_open_authorization.json" "$RUNMETA/development_open_authorization.json"
cp "$WORK/frozen/preflight_aggregate.json" "$RUNMETA/preflight_aggregate.json"
cp "$WORK/frozen/preflight_execution_evidence.json" "$RUNMETA/preflight_execution_evidence.json"
cp "$WORK/frozen/preflight_finalization.json" "$RUNMETA/preflight_finalization.json"
cp "$WORK/frozen/preflight_source.json" "$RUNMETA/preflight_source.json"
mkdir -p "$RUNMETA/preflight_proofs"
cp "$WORK/frozen/preflight_proofs/"*.json "$RUNMETA/preflight_proofs/"
cp "$WORK/source_model_manifest.json" "$RUNMETA/source_model_manifest.json"
cp "$WORK/source_native_manifest.json" "$RUNMETA/source_native_manifest.json"

cd "$WORK"
python3 -m venv "$VENV"; source "$VENV/bin/activate"
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu 'pip==26.1.2' 'setuptools==83.0.0' 'wheel==0.46.1'
python -m pip install -r configs/hu_m43_attempt08_runtime_requirements.txt
export PYTHONPATH="$WORK/src" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 OFC_HU_M3_BATCH_THREADS=4
python -m ofc_regular.hu_m43_attempt08_spot validate-launch \
  --run-dir "$RUNMETA" --authorization "$RUNMETA/launch_authorization.json" >/dev/null
python - <<'PY'
from pathlib import Path
from ofc_regular.hu_m43_attempt08_runtime_anchor import validate_runtime_semantic_anchor
from ofc_regular.hu_m43_attempt08_runtime_anchor_contract import (
 ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SHA256,ATTEMPT08_RUNTIME_SOURCE_CLOSURE_SHA256,
 ATTEMPT08_RUNTIME_SOURCE_FILE_COUNT)
from ofc_regular.hu_m43_attempt08_runtime_identity import validate_expected_runtime_fingerprint
validate_runtime_semantic_anchor(repository_root=Path('.'),
 expected_source_closure_sha256=ATTEMPT08_RUNTIME_SOURCE_CLOSURE_SHA256,
 expected_anchor_sha256=ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SHA256,
 expected_source_file_count=ATTEMPT08_RUNTIME_SOURCE_FILE_COUNT,
 runtime_artifact_root=Path('.'),model_manifest_path=Path('source_model_manifest.json'),
 native_manifest_path=Path('source_native_manifest.json'),
 requirements_path=Path('configs/hu_m43_attempt08_runtime_requirements.txt'))
validate_expected_runtime_fingerprint()
PY

python - /tmp/shards_manifest.jsonl "$SHARD" <<'PY' >/tmp/spec.env
import json,sys
lines=open(sys.argv[1],encoding='utf-8').read().splitlines(); i=int(sys.argv[2]); assert len(lines)==200
s=json.loads(lines[i]); assert s['shard']==i and s['root_index']==i and s['roots']==1
assert s['native_batch_threads']==4 and s['batch_child_selectors'] is True
print('OUTPUT_PREFIX='+s['output_prefix']); print('ROOT_PROFILE='+s['root_profile'])
PY
source /tmp/spec.env
RESULT_URI="$PREFIX/results/$OUTPUT_PREFIX"; RESUME_URI="$PREFIX/resume/$OUTPUT_PREFIX"

upload_once_or_verify(){
  source="$1"; uri="$2"
  if gcloud storage cp "$source" "$uri" --project "$PROJECT_ID" --if-generation-match=0 >/dev/null 2>&1; then return 0; fi
  existing="$(mktemp)"; gcloud storage cp "$uri" "$existing" --project "$PROJECT_ID" >/dev/null
  cmp -s "$source" "$existing" || { rm -f "$existing"; echo "immutable object differs: $uri" >&2; return 1; }
  rm -f "$existing"
}
download_if_exists(){
  uri="$1"; path="$2"
  if remote_exists "$uri"; then gcloud storage cp "$uri" "$path" --project "$PROJECT_ID" >/dev/null; return 0; fi
  return 1
}

# Claims are made before the root is opened. Existing identical claims are a
# deterministic retry of the same logical shard; a different claim is fatal.
python -m ofc_regular.hu_m43_attempt08_spot global-claim \
  --run-dir "$RUNMETA" --authorization "$RUNMETA/launch_authorization.json" \
  --output "$RESULT/global_claim.json" >/dev/null
upload_once_or_verify "$RESULT/global_claim.json" "$PREFIX/claims/global_claim.json"
python -m ofc_regular.hu_m43_attempt08_spot root-claim \
  --run-dir "$RUNMETA" --authorization "$RUNMETA/launch_authorization.json" \
  --shard "$SHARD" --output "$RESULT/root_claim.json" >/dev/null
upload_once_or_verify "$RESULT/root_claim.json" "$PREFIX/claims/root-${SHARD}.json"

# A valid remote DONE is terminal, but it is never trusted by existence alone.
if remote_exists "$RESULT_URI/DONE.json"; then
  for name in DONE.json teacher.jsonl checkpoint.json heartbeat.json generator_summary.json run.log time.txt resume_commit.json global_claim.json root_claim.json boot_image_evidence.json; do
    gcloud storage cp "$RESULT_URI/$name" "$RESULT/$name" --project "$PROJECT_ID" >/dev/null
  done
  python -m ofc_regular.hu_m43_attempt08_spot validate-completed-bundle \
    --directory "$RESULT" --shard "$SHARD" --run-dir "$RUNMETA" \
    --authorization "$RUNMETA/launch_authorization.json" >/dev/null
  exit 0
fi

# Recover only content-addressed state. A complete commit preserves original
# timing and RSS; it is never re-timed or replaced.
COMMITTED=0
if download_if_exists "$RESUME_URI/resume_commit.json" "$RESULT/resume_commit.json"; then
  while IFS=$'\t' read -r name digest; do
    [[ "$digest" =~ ^[0-9a-f]{64}$ ]]
    gcloud storage cp "$RESUME_URI/objects/$digest/$name" "$RESULT/$name" \
      --project "$PROJECT_ID" >/dev/null
  done < <(python - "$RESULT/resume_commit.json" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]))
for name,field in (
 ('teacher.jsonl','output_sha256'),('checkpoint.json','checkpoint_sha256'),
 ('heartbeat.json','heartbeat_sha256'),('generator_summary.json','generator_summary_sha256'),
 ('run.log','run_log_sha256'),('boot_image_evidence.json','boot_image_evidence_sha256'),
 ('time.txt','time_report_sha256'),('global_claim.json','global_claim_sha256'),
 ('root_claim.json','root_claim_sha256')):
 print(name+'\t'+c[field])
PY
  )
  COMMITTED=1
else
  for name in teacher.jsonl teacher.jsonl.partial checkpoint.json heartbeat.json generator_summary.json time.txt; do
    download_if_exists "$RESUME_URI/$name" "$RESULT/$name" || true
  done
fi

RUN_ARGS=(
  --root-index "$SHARD" --output "$RESULT/teacher.jsonl"
  --checkpoint "$RESULT/checkpoint.json" --heartbeat "$RESULT/heartbeat.json"
  --model "$WORK/outputs/hu_joint_policy/m43_attempt05_old_dev900_architecture_once/lambda_rank_candidate.pkl"
  --model-sha256 e7fa728308c8973e2e202baf79309e30b9b6f58c37431d8ea55850390abc28c3
  --run-id "${RUN_NAME}:shard=${SHARD}"
  --development-open-authorization "$RUNMETA/development_open_authorization.json"
  --preflight-plan "$RUNMETA/hu_joint_policy_m43_attempt08_preflight.json"
  --plan "$RUNMETA/hu_joint_policy_m43_attempt08.json"
  --ai-profiles "$WORK/src/ofc_regular/ai_profiles.py"
  --batch-child-selectors --native-batch-threads 4
)

UNCOMMITTED_COMPLETE=0
if [[ "$COMMITTED" == 0 && -f "$RESULT/teacher.jsonl" ]]; then
  required=(checkpoint.json heartbeat.json generator_summary.json time.txt)
  complete_set=1
  for name in "${required[@]}"; do [[ -f "$RESULT/$name" ]] || complete_set=0; done
  if [[ "$complete_set" == 1 ]]; then
    set +e
    python -m ofc_regular.run_hu_m43_attempt08_development "${RUN_ARGS[@]}" \
      >/tmp/uncommitted-summary.raw 2>/tmp/uncommitted-summary.err
    validation_exit=$?
    set -e
    if [[ "$validation_exit" == 0 ]] && python - /tmp/uncommitted-summary.raw "$RESULT/generator_summary.json" <<'PY'
import json,sys
assert json.load(open(sys.argv[1]))==json.load(open(sys.argv[2]))
PY
    then
      UNCOMMITTED_COMPLETE=1
    fi
  fi
  if [[ "$UNCOMMITTED_COMPLETE" == 0 ]]; then
    # No immutable COMMIT exists, so a cut between mutable checkpoint uploads
    # is not authoritative.  Remove the incomplete completion set and reopen
    # the same already-claimed deterministic root from scratch.
    rm -f "$RESULT/teacher.jsonl" "$RESULT/teacher.jsonl.partial" \
      "$RESULT/checkpoint.json" "$RESULT/heartbeat.json" \
      "$RESULT/generator_summary.json" "$RESULT/time.txt"
  fi
fi
if [[ "$COMMITTED" == 0 && ! -f "$RESULT/teacher.jsonl" ]]; then
  partial_pair_valid=0
  if [[ -f "$RESULT/teacher.jsonl.partial" && -f "$RESULT/checkpoint.json" ]]; then
    if python - "$RESULT/teacher.jsonl.partial" "$RESULT/checkpoint.json" "$SHARD" <<'PY'
import hashlib,json,sys
raw=open(sys.argv[1],'rb').read(); checkpoint_raw=open(sys.argv[2],'rb').read()
c=json.loads(checkpoint_raw)
canonical=(json.dumps(c,sort_keys=True,separators=(',',':'))+'\n').encode()
assert checkpoint_raw==canonical
assert c['schema']=='hu_m43_attempt08_development_checkpoint_v1'
assert c['completed_roots']==1 and c['target_roots']==1
assert c['root_index']==int(sys.argv[3])
assert c['partial_sha256']==hashlib.sha256(raw).hexdigest()
assert raw.count(b'\n')==1 and raw.endswith(b'\n')
PY
    then
      partial_pair_valid=1
    fi
  fi
  if [[ "$partial_pair_valid" == 0 ]]; then
    rm -f "$RESULT/teacher.jsonl.partial" "$RESULT/checkpoint.json" \
      "$RESULT/heartbeat.json" "$RESULT/generator_summary.json" "$RESULT/time.txt"
  fi
fi

if [[ "$COMMITTED" == 1 ]]; then
  python -m ofc_regular.run_hu_m43_attempt08_development "${RUN_ARGS[@]}" >/tmp/resume-summary.raw
  python - /tmp/resume-summary.raw "$RESULT/generator_summary.json" <<'PY'
import json,sys
a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2])); assert a==b
PY
elif [[ "$UNCOMMITTED_COMPLETE" == 1 ]]; then
  cp /tmp/uncommitted-summary.raw /tmp/resume-summary.raw
  python - /tmp/resume-summary.raw "$RESULT/generator_summary.json" <<'PY'
import json,sys
a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2])); assert a==b
PY
else
  (
    while true; do
      sleep 60
      for name in teacher.jsonl teacher.jsonl.partial checkpoint.json heartbeat.json; do
        [[ -f "$RESULT/$name" ]] && gcloud storage cp "$RESULT/$name" "$RESUME_URI/$name" --project "$PROJECT_ID" >/dev/null 2>&1 || true
      done
    done
  ) & SYNC_PID=$!
  set +e
  timeout --signal=TERM --kill-after=60s 3600s /usr/bin/time -v -o "$RESULT/time.txt" \
    python -m ofc_regular.run_hu_m43_attempt08_development "${RUN_ARGS[@]}" >/tmp/generator-summary.raw
  RUN_EXIT=$?
  set -e
  kill "$SYNC_PID" >/dev/null 2>&1 || true; wait "$SYNC_PID" 2>/dev/null || true
  if [[ "$RUN_EXIT" != 0 ]]; then
    [[ -f "$RESULT/heartbeat.json" ]] && gcloud storage cp "$RESULT/heartbeat.json" "$PREFIX/heartbeat/$OUTPUT_PREFIX.json" --project "$PROJECT_ID" >/dev/null 2>&1 || true
    exit "$RUN_EXIT"
  fi
  python - /tmp/generator-summary.raw "$RESULT/generator_summary.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1])); open(sys.argv[2],'w').write(json.dumps(p,sort_keys=True,separators=(',',':'))+'\n')
PY
  for name in teacher.jsonl checkpoint.json heartbeat.json generator_summary.json time.txt; do
    gcloud storage cp "$RESULT/$name" "$RESUME_URI/$name" --project "$PROJECT_ID" >/dev/null
  done
fi

printf 'attempt08-development-complete run=%s shard=%s output_sha256=%s\n' \
  "$RUN_NAME" "$SHARD" "$(sha "$RESULT/teacher.jsonl")" >"$RESULT/run.log"

if [[ "$COMMITTED" == 0 ]]; then
  python - "$RESULT/resume_commit.json" "$RESULT/teacher.jsonl" "$RESULT/checkpoint.json" \
    "$RESULT/heartbeat.json" "$RESULT/generator_summary.json" "$RESULT/run.log" \
    "$RESULT/boot_image_evidence.json" "$RESULT/time.txt" "$RESULT/global_claim.json" \
    "$RESULT/root_claim.json" "$RUN_NAME" "$SHARD" "$MANIFEST_SHA256" \
    "$AUTHORIZATION_SHA256" <<'PY'
import json,sys
from pathlib import Path
from ofc_regular.hu_m43_attempt08_spot import _parse_gnu_time_report,sha256_file
out,teacher,checkpoint,heartbeat,summaryp,runlog,boot,timep,globalc,rootc,run,shard,manifest,auth=sys.argv[1:]
s=json.load(open(summaryp)); elapsed,rss=_parse_gnu_time_report(Path(timep))
p={'schema':'hu_m43_attempt08_development_resume_commit_v1','status':'committed_completed_root_pair',
 'run_name':run,'shard':int(shard),'manifest_sha256':manifest,'launch_authorization_sha256':auth,
 'output_sha256':sha256_file(teacher),'checkpoint_sha256':sha256_file(checkpoint),
 'heartbeat_sha256':sha256_file(heartbeat),'generator_summary_sha256':sha256_file(summaryp),
 'run_log_sha256':sha256_file(runlog),'boot_image_evidence_sha256':sha256_file(boot),
 'time_report_sha256':sha256_file(timep),'generator_elapsed_seconds':s['elapsed_seconds'],
 'global_claim_sha256':sha256_file(globalc),'root_claim_sha256':sha256_file(rootc),
 'process_elapsed_seconds':elapsed,'peak_rss_bytes':rss}
open(out,'w').write(json.dumps(p,sort_keys=True,separators=(',',':'))+'\n')
PY
  # A completed root becomes durable in the resume namespace first.  The
  # immutable commit is published last, so every later retry can replay the
  # exact same bytes across any crash cut in final publication.
  for name in teacher.jsonl checkpoint.json heartbeat.json generator_summary.json run.log boot_image_evidence.json time.txt global_claim.json root_claim.json; do
    digest="$(sha "$RESULT/$name")"
    upload_once_or_verify "$RESULT/$name" "$RESUME_URI/objects/$digest/$name"
  done
  upload_once_or_verify "$RESULT/resume_commit.json" "$RESUME_URI/resume_commit.json"
fi

python - "$RESULT/resume_commit.json" "$RESULT/teacher.jsonl" "$RESULT/checkpoint.json" \
  "$RESULT/heartbeat.json" "$RESULT/generator_summary.json" "$RESULT/run.log" \
  "$RESULT/boot_image_evidence.json" "$RESULT/time.txt" "$RESULT/global_claim.json" \
  "$RESULT/root_claim.json" <<'PY'
import json,sys,hashlib
h=lambda p:hashlib.sha256(open(p,'rb').read()).hexdigest()
c=json.load(open(sys.argv[1])); s=json.load(open(sys.argv[5]))
fields=('output_sha256','checkpoint_sha256','heartbeat_sha256','generator_summary_sha256',
        'run_log_sha256','boot_image_evidence_sha256','time_report_sha256',
        'global_claim_sha256','root_claim_sha256')
for field,path in zip(fields,sys.argv[2:]): assert c[field]==h(path),(field,path)
assert c['generator_elapsed_seconds']==s['elapsed_seconds']
PY

python - "$RESULT/DONE.json" "$RUNMETA/manifest.json" "$RUNMETA/launch_authorization.json" \
  /tmp/shards_manifest.jsonl "$RESULT" "$SHARD" <<'PY'
import json,sys
from pathlib import Path
from ofc_regular.hu_m43_attempt08_spot import DONE_SCHEMA,_parse_gnu_time_report,sha256_file
done,mp,ap,sp,result,shard=sys.argv[1:]; result=Path(result); shard=int(shard)
m=json.load(open(mp)); s=json.loads(open(sp).read().splitlines()[shard]); summary=json.load(open(result/'generator_summary.json'))
commit=json.load(open(result/'resume_commit.json')); elapsed,time_rss=_parse_gnu_time_report(result/'time.txt')
fields=('development_open_authorization_sha256','source_zip_sha256','startup_sha256','schedule_sha256',
 'plan_sha256','model_sha256','ai_profiles_sha256','source_closure_sha256','source_model_manifest_sha256',
 'source_native_manifest_sha256','preflight_aggregate_sha256','preflight_finalization_sha256',
 'preflight_proof_sha256','preflight_execution_evidence_sha256','runtime_semantic_anchor_sha256',
 'runtime_source_closure_sha256','runtime_fingerprint_sha256','runtime_requirements_sha256',
 'gcp_image_name','gcp_image_id','gcp_image_self_link')
d={'schema':DONE_SCHEMA,'status':'complete','run_name':m['run_name'],'run_id':f"{m['run_name']}:shard={shard}",
 'shard':shard,'root_index':shard,'root_profile':s['root_profile'],'seeds':s['seeds'],'output_prefix':s['output_prefix'],
 'manifest_sha256':sha256_file(mp),'launch_authorization_sha256':sha256_file(ap),
 'development_open_authorization_sha256':m['development_open_authorization_sha256'],
 'source_sha256':m['source_zip_sha256'],'startup_sha256':m['startup_sha256'],'schedule_sha256':m['schedule_sha256'],
 'plan_sha256':m['plan_sha256'],'model_sha256':m['model_sha256'],'ai_profiles_sha256':m['ai_profiles_sha256'],
 'source_closure_sha256':m['source_closure_sha256'],'source_model_manifest_sha256':m['source_model_manifest_sha256'],
 'source_native_manifest_sha256':m['source_native_manifest_sha256'],'preflight_aggregate_sha256':m['preflight_aggregate_sha256'],
 'preflight_finalization_sha256':m['preflight_finalization_sha256'],'preflight_proof_sha256':m['preflight_proof_sha256'],
 'preflight_execution_evidence_sha256':m['preflight_execution_evidence_sha256'],
 'runtime_semantic_anchor_sha256':m['runtime_semantic_anchor_sha256'],'runtime_source_closure_sha256':m['runtime_source_closure_sha256'],
 'runtime_fingerprint_sha256':m['runtime_fingerprint_sha256'],'runtime_requirements_sha256':m['runtime_requirements_sha256'],
 'gcp_image_name':m['gcp_image_name'],'gcp_image_id':m['gcp_image_id'],'gcp_image_self_link':m['gcp_image_self_link'],
 'global_claim_sha256':sha256_file(result/'global_claim.json'),'root_claim_sha256':sha256_file(result/'root_claim.json'),
 'output_sha256':sha256_file(result/'teacher.jsonl'),'checkpoint_sha256':sha256_file(result/'checkpoint.json'),
 'heartbeat_sha256':sha256_file(result/'heartbeat.json'),'generator_summary_sha256':sha256_file(result/'generator_summary.json'),
 'run_log_sha256':sha256_file(result/'run.log'),'boot_image_evidence_sha256':sha256_file(result/'boot_image_evidence.json'),
 'time_report_sha256':sha256_file(result/'time.txt'),'resume_commit_sha256':sha256_file(result/'resume_commit.json'),
 'config_sha256':summary['config_sha256'],'teacher_generator_elapsed_seconds':summary['elapsed_seconds'],
 'process_elapsed_seconds':elapsed,'peak_rss_bytes':max(summary['generator_peak_rss_bytes'],time_rss,commit['peak_rss_bytes']),
 'native_batch_threads':4,'teacher_values_are_realized_match_ev':False,'selector_executed':False,
 'future_audit_authorized':False,'fit_performed':False,'threshold_selected':False,
 'current_profile_mutated':False,'runtime_policy_activated':False}
open(done,'w').write(json.dumps(d,sort_keys=True,separators=(',',':'))+'\n')
PY
python -m ofc_regular.hu_m43_attempt08_spot validate-done --input "$RESULT/DONE.json" \
  --shard "$SHARD" --run-dir "$RUNMETA" --authorization "$RUNMETA/launch_authorization.json" >/dev/null
python -m ofc_regular.hu_m43_attempt08_spot validate-completed-bundle \
  --directory "$RESULT" --shard "$SHARD" --run-dir "$RUNMETA" \
  --authorization "$RUNMETA/launch_authorization.json" >/dev/null

# Publish all content first and DONE last. Every object is immutable.
for name in teacher.jsonl checkpoint.json heartbeat.json generator_summary.json run.log boot_image_evidence.json time.txt resume_commit.json global_claim.json root_claim.json; do
  upload_once_or_verify "$RESULT/$name" "$RESULT_URI/$name"
done
upload_once_or_verify "$RESULT/DONE.json" "$RESULT_URI/DONE.json"
