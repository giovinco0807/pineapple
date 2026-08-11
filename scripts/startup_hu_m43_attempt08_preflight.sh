#!/usr/bin/env bash
set -euo pipefail

export HOME="${HOME:-/root}"
meta(){ curl -fsS -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }
imeta(){ curl -fsS -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/$1"; }
sha(){ sha256sum "$1" | awk '{print $1}'; }

RUN_NAME="$(meta RUN_NAME)"; PROJECT_ID="$(meta PROJECT_ID)"; BUCKET="$(meta BUCKET)"
JOB_INDEX="$(meta JOB_INDEX)"; SOURCE_URI="$(meta SOURCE_URI)"; SOURCE_SHA256="$(meta SOURCE_SHA256)"
MANIFEST_SHA256="$(meta MANIFEST_SHA256)"; AUTHORIZATION_SHA256="$(meta AUTHORIZATION_SHA256)"
STARTUP_SHA256="$(meta STARTUP_SHA256)"; SELF_DELETE="$(meta SELF_DELETE)"
INSTANCE_NAME="$(imeta name)"; ZONE="$(imeta zone)"; ZONE="${ZONE##*/}"
PREFIX="gs://${BUCKET}/runs/${RUN_NAME}"; RUNMETA=/opt/ofc-attempt08-preflight
WORK="$RUNMETA/package_src"
RESULT=/tmp/ofc-attempt08-preflight-result; LOG=/tmp/ofc-attempt08-preflight.log
mkdir -p "$RESULT"; : >"$LOG"; exec > >(tee -a "$LOG") 2>&1

cleanup(){
  code=$?; set +e
  [[ -n "${OUTPUT_PREFIX:-}" ]] && gcloud storage cp "$LOG" "$PREFIX/logs/$OUTPUT_PREFIX.log" --project "$PROJECT_ID" >/dev/null 2>&1 || true
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
command -v gcloud >/dev/null; command -v curl >/dev/null

# Prove the actual boot disk came from the authorized immutable image.
DISK="$(gcloud compute instances describe "$INSTANCE_NAME" --zone "$ZONE" --project "$PROJECT_ID" --format='value(disks[0].source.basename())')"
DISK_JSON="$(gcloud compute disks describe "$DISK" --zone "$ZONE" --project "$PROJECT_ID" --format=json)"
python3 - "$DISK_JSON" "$RESULT/boot_image_evidence.json" "$RUN_NAME" "$JOB_INDEX" "$INSTANCE_NAME" "$DISK" <<'PY'
import json,sys
d=json.loads(sys.argv[1])
assert str(d.get('sourceImage','')).endswith('/projects/debian-cloud/global/images/debian-12-bookworm-v20260609')
assert str(d.get('sourceImageId'))=='1449487925682397051'
p={'schema':'hu_m43_attempt08_boot_image_evidence_v1','run_name':sys.argv[3],
   'job_index':int(sys.argv[4]),'instance_name':sys.argv[5],'disk_name':sys.argv[6],
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
gcloud storage cp "$PREFIX/source/spot_authorization.json" /tmp/authorization.json --project "$PROJECT_ID" >/dev/null
gcloud storage cp "$PREFIX/source/local_evidence.json" /tmp/local_evidence.json --project "$PROJECT_ID" >/dev/null
gcloud storage cp "$PREFIX/source/shards_manifest.jsonl" /tmp/shards.jsonl --project "$PROJECT_ID" >/dev/null
[[ "$(sha /tmp/source.zip)" == "$SOURCE_SHA256" ]]
[[ "$(sha /tmp/manifest.json)" == "$MANIFEST_SHA256" ]]
[[ "$(sha /tmp/authorization.json)" == "$AUTHORIZATION_SHA256" ]]
unzip -q /tmp/source.zip -d "$WORK"
cp /tmp/source.zip "$RUNMETA/source.zip"
cp /tmp/manifest.json "$RUNMETA/manifest.json"
cp /tmp/authorization.json "$RUNMETA/spot_authorization.json"
cp /tmp/local_evidence.json "$RUNMETA/local_evidence.json"
cp /tmp/shards.jsonl "$RUNMETA/shards_manifest.jsonl"
cp /tmp/executing-startup.sh "$RUNMETA/startup_hu_m43_attempt08_preflight.sh"
cp "$WORK/configs/hu_joint_policy_m43_attempt08.json" "$RUNMETA/hu_joint_policy_m43_attempt08.json"
cp "$WORK/configs/hu_joint_policy_m43_attempt08_preflight.json" "$RUNMETA/hu_joint_policy_m43_attempt08_preflight.json"
cd "$WORK"

python3 - /tmp/manifest.json /tmp/authorization.json /tmp/shards.jsonl "$JOB_INDEX" <<'PY' > /tmp/spec.env
import json,sys
m=json.load(open(sys.argv[1])); a=json.load(open(sys.argv[2])); lines=open(sys.argv[3]).read().splitlines()
assert m['schema']=='hu_m43_attempt08_preflight_spot_package_manifest_v1'
assert a['schema']=='hu_m43_attempt08_preflight_spot_launch_authorization_v1'
assert a['spot_authorized'] is True and a['manifest_sha256']==__import__('hashlib').sha256(open(sys.argv[1],'rb').read()).hexdigest()
assert m['gcp_image_name']=='debian-12-bookworm-v20260609' and m['gcp_image_id']=='1449487925682397051'
assert m['schedule_sha256']==a['schedule_sha256']==__import__('hashlib').sha256(open(sys.argv[3],'rb').read()).hexdigest()
s=json.loads(lines[int(sys.argv[4])]); assert s['schema']=='hu_m43_attempt08_preflight_spot_job_v1'
for k,v in {'JOB_ID':s['job_id'],'SLOT':s['slot'],'SOURCE_ROOT_INDEX':s['source_root_index'],
            'BATCH_FLAG':str(s['batch_child_selectors']).lower(),'OUTPUT_PREFIX':s['output_prefix']}.items():
 print(f'{k}={v}')
PY
source /tmp/spec.env
RESULT_URI="$PREFIX/results/$OUTPUT_PREFIX"
REMOTE_DONE=0
if gcloud storage objects describe "$RESULT_URI/DONE.json" --project "$PROJECT_ID" >/dev/null 2>&1; then REMOTE_DONE=1; fi

python3 -m venv "$RUNMETA/.venv"; source "$RUNMETA/.venv/bin/activate"
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu 'pip==26.1.2' 'setuptools==83.0.0' 'wheel==0.46.1'
python -m pip install -r configs/hu_m43_attempt08_runtime_requirements.txt
export PYTHONPATH="$WORK/src" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 OFC_HU_M3_BATCH_THREADS=4
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
python - "$RUNMETA" <<'PY'
import sys
from pathlib import Path
from ofc_regular.hu_m43_attempt08_preflight_spot import (
 sha256_file,validate_launch_authorization,validate_package_artifacts)
root=Path(sys.argv[1])
manifest=validate_package_artifacts(root)
validate_launch_authorization(root/'spot_authorization.json',manifest=manifest,
 manifest_sha256=sha256_file(root/'manifest.json'),
 local_evidence_path=root/'local_evidence.json',revalidate_local_commands=False)
PY

if [[ "$REMOTE_DONE" == 1 ]]; then
  echo "Remote DONE exists; reopening the complete immutable job bundle before exit."
  REMOTE_FETCH_OK=1
  for name in DONE.json proof.json checkpoint.json heartbeat.json summary.json run.log boot_image_evidence.json; do
    if ! gcloud storage cp "$RESULT_URI/$name" "$RESULT/$name" --project "$PROJECT_ID" >/dev/null; then
      REMOTE_FETCH_OK=0
    fi
  done
  if [[ "$REMOTE_FETCH_OK" != 1 ]]; then
    echo "REMOTE_DONE_INVALID: completion marker is missing bound support; use a new run name." >&2
    exit 42
  fi
  if python - "$RUNMETA" "$RESULT" "$JOB_INDEX" "$SLOT" <<'PY'
import sys
from pathlib import Path
from ofc_regular.aggregate_hu_m43_attempt08_preflight import _load_canonical_json,_validate_proof
from ofc_regular.hu_m43_attempt08_preflight_spot import validate_done_metadata,validate_received_job_artifacts
runmeta,result,index,slot=Path(sys.argv[1]),Path(sys.argv[2]),int(sys.argv[3]),sys.argv[4]
done=validate_done_metadata(done_path=result/'DONE.json',run_dir=runmeta,
 authorization_path=runmeta/'spot_authorization.json',
 local_evidence_path=runmeta/'local_evidence.json',job_index=index,
 revalidate_local_commands=False)
validate_received_job_artifacts(directory=result,done=done)
proof=_load_canonical_json(result/'proof.json',label='existing remote proof')
_validate_proof(proof,slot=slot,source=runmeta/'package_src/preflight_source/teacher.jsonl')
PY
  then
    echo "Remote DONE and all bound artifacts are valid; no recomputation required."
    exit 0
  else
    echo "REMOTE_DONE_INVALID: canonical/full validation failed; use a new run name." >&2
    exit 42
  fi
fi

heartbeat(){
  state="$1"
  proof_sha=""; [[ -s "$RESULT/proof.json" ]] && proof_sha="$(sha "$RESULT/proof.json")"
  python - "$RESULT/heartbeat.json" "$RUN_NAME" "$JOB_INDEX" "$SLOT" "$state" "$proof_sha" <<'PY'
import json,sys,time
p={'schema':'hu_m43_attempt08_preflight_spot_heartbeat_v1','run_name':sys.argv[2],
   'job_index':int(sys.argv[3]),'slot':sys.argv[4],'status':sys.argv[5],
   'updated_unix_seconds':(0.0 if sys.argv[5]=='complete' else time.time()),'deterministic_recompute_on_preemption':True,
   'proof_sha256':sys.argv[6] or None,'process_alive':sys.argv[5]=='running',
   'new_root_generated':False,'current_profile_resolved':False}
open(sys.argv[1],'w').write(json.dumps(p,sort_keys=True,separators=(',',':'))+'\n')
PY
  gcloud storage cp "$RESULT/heartbeat.json" "$PREFIX/heartbeat/$OUTPUT_PREFIX.json" --project "$PROJECT_ID" >/dev/null || true
}

REUSED_PROOF=0
if gcloud storage objects describe "$RESULT_URI/proof.json" --project "$PROJECT_ID" >/dev/null 2>&1; then
  gcloud storage cp "$RESULT_URI/proof.json" "$RESULT/proof.json" --project "$PROJECT_ID" >/dev/null
  python - "$RESULT/proof.json" "$SLOT" <<'PY'
import sys
from ofc_regular.aggregate_hu_m43_attempt08_preflight import _load_canonical_json,_validate_proof
from ofc_regular.run_hu_m43_attempt08_preflight import DEFAULT_SOURCE_PATH
p=_load_canonical_json(sys.argv[1],label='reused Spot proof')
_validate_proof(p,slot=sys.argv[2],source=DEFAULT_SOURCE_PATH)
PY
  if gcloud storage objects describe "$RESULT_URI/boot_image_evidence.json" --project "$PROJECT_ID" >/dev/null 2>&1; then
    gcloud storage cp "$RESULT_URI/boot_image_evidence.json" "$RESULT/boot_image_evidence.remote.json" --project "$PROJECT_ID" >/dev/null
    python - "$RESULT/boot_image_evidence.remote.json" "$RUN_NAME" "$JOB_INDEX" <<'PY'
import json,sys
p=json.load(open(sys.argv[1]))
assert set(p)=={'schema','run_name','job_index','instance_name','disk_name','source_image','source_image_id'}
assert p['schema']=='hu_m43_attempt08_boot_image_evidence_v1'
assert p['run_name']==sys.argv[2] and p['job_index']==int(sys.argv[3])
assert str(p['source_image']).endswith('/projects/debian-cloud/global/images/debian-12-bookworm-v20260609')
assert str(p['source_image_id'])=='1449487925682397051'
PY
    mv "$RESULT/boot_image_evidence.remote.json" "$RESULT/boot_image_evidence.json"
  fi
  REUSED_PROOF=1
fi

if [[ "$REUSED_PROOF" == 0 ]]; then
  python - "$RESULT/checkpoint.json" "$RUN_NAME" "$JOB_INDEX" "$SLOT" <<'PY'
import json,sys
p={'schema':'hu_m43_attempt08_preflight_spot_checkpoint_v1','run_name':sys.argv[2],
   'job_index':int(sys.argv[3]),'slot':sys.argv[4],'status':'running',
   'resume_mode':'same_input_same_seed_deterministic_recompute','new_root_generated':False}
open(sys.argv[1],'w').write(json.dumps(p,sort_keys=True,separators=(',',':'))+'\n')
PY
  gcloud storage cp "$RESULT/checkpoint.json" "$PREFIX/resume/$OUTPUT_PREFIX/checkpoint.json" --project "$PROJECT_ID" >/dev/null
  heartbeat running
  (
    while true; do sleep 60; heartbeat running; done
  ) & HEARTBEAT_PID=$!

  set +e
  timeout --signal=TERM --kill-after=60s 2500s python -m ofc_regular.run_hu_m43_attempt08_preflight \
    --slot "$SLOT" --output "$RESULT/proof.json" \
    --source preflight_source/teacher.jsonl \
    --preflight-plan configs/hu_joint_policy_m43_attempt08_preflight.json \
    --attempt08-plan configs/hu_joint_policy_m43_attempt08.json \
    --model outputs/hu_joint_policy/m43_attempt05_old_dev900_architecture_once/lambda_rank_candidate.pkl \
    --ai-profiles src/ofc_regular/ai_profiles.py \
    --runtime-model-manifest source_model_manifest.json \
    --runtime-native-manifest source_native_manifest.json \
    --runtime-artifact-root . \
    --runtime-requirements configs/hu_m43_attempt08_runtime_requirements.txt
  TEACHER_EXIT=$?
  set -e
  kill "$HEARTBEAT_PID" >/dev/null 2>&1 || true; wait "$HEARTBEAT_PID" 2>/dev/null || true
  [[ "$TEACHER_EXIT" == 0 ]]
fi

python - "$RESULT/checkpoint.json" "$RESULT/summary.json" \
  "$RESULT/proof.json" /tmp/shards.jsonl \
  "$RUN_NAME" "$JOB_INDEX" <<'PY'
import hashlib,json,sys
cp,summary,proofp,sp,run,index=sys.argv[1:]
h=lambda p:hashlib.sha256(open(p,'rb').read()).hexdigest()
proof=json.load(open(proofp)); s=json.loads(open(sp).read().splitlines()[int(index)])
checkpoint={'schema':'hu_m43_attempt08_preflight_spot_checkpoint_v1','run_name':run,'job_index':int(index),
 'slot':s['slot'],'status':'complete','proof_sha256':h(proofp),'resume_mode':'complete_no_recompute_needed','new_root_generated':False}
open(cp,'w').write(json.dumps(checkpoint,sort_keys=True,separators=(',',':'))+'\n')
su={'schema':'hu_m43_attempt08_preflight_spot_summary_v1','run_name':run,'job_index':int(index),'slot':s['slot'],
 'proof_sha256':h(proofp),'teacher_elapsed_seconds':proof['execution']['teacher_elapsed_seconds'],
 'process_peak_rss_bytes':proof['execution']['process_peak_rss_bytes'],'teacher_action_or_value_details_exported':False}
open(summary,'w').write(json.dumps(su,sort_keys=True,separators=(',',':'))+'\n')
PY
heartbeat complete
python - "$RESULT/run.log" "$RUN_NAME" "$JOB_INDEX" "$SLOT" "$RESULT/proof.json" <<'PY'
import hashlib,sys
h=hashlib.sha256(open(sys.argv[5],'rb').read()).hexdigest()
open(sys.argv[1],'w').write(
 f'attempt08-preflight-complete run={sys.argv[2]} job={sys.argv[3]} slot={sys.argv[4]} proof_sha256={h}\n')
PY
python - "$RESULT/DONE.json" "$RESULT/proof.json" "$RESULT/checkpoint.json" \
  "$RESULT/heartbeat.json" "$RESULT/summary.json" "$RESULT/run.log" "$RESULT/boot_image_evidence.json" \
  /tmp/manifest.json /tmp/authorization.json /tmp/shards.jsonl "$RUN_NAME" "$JOB_INDEX" <<'PY'
import hashlib,json,sys
done,proofp,checkpoint,heartbeat,summary,runlog,boot,mp,ap,sp,run,index=sys.argv[1:]
h=lambda p:hashlib.sha256(open(p,'rb').read()).hexdigest()
m=json.load(open(mp)); a=json.load(open(ap)); proof=json.load(open(proofp)); s=json.loads(open(sp).read().splitlines()[int(index)])
keys=('schedule_sha256','attempt08_plan_sha256','preflight_plan_sha256','source_merged_sha256','model_sha256',
 'ai_profiles_sha256','attempt07_closeout_sha256','attempt07_selection_sha256','runtime_semantic_anchor_sha256',
 'runtime_source_closure_sha256','runtime_requirements_sha256','runtime_fingerprint_sha256',
 'source_model_manifest_sha256','source_native_manifest_sha256','gcp_image_name','gcp_image_id')
d={'schema':'hu_m43_attempt08_preflight_spot_done_v1','status':'complete','run_name':run,'job_index':int(index),
 'job_id':s['job_id'],'slot':s['slot'],'source_root_index':s['source_root_index'],
 'batch_child_selectors':s['batch_child_selectors'],'output_prefix':s['output_prefix'],'machine_type':'c4-highmem-4',
 'native_batch_threads':4,'proof_sha256':h(proofp),'teacher_elapsed_seconds':proof['execution']['teacher_elapsed_seconds'],
 'process_peak_rss_bytes':proof['execution']['process_peak_rss_bytes'],'checkpoint_sha256':h(checkpoint),
 'heartbeat_sha256':h(heartbeat),'summary_sha256':h(summary),'run_log_sha256':h(runlog),
 'boot_image_evidence_sha256':h(boot),'manifest_sha256':h(mp),
 'authorization_sha256':h(ap),**{k:a[k] for k in keys},'teacher_action_or_value_details_exported':False,
 'new_root_generated':False,'policy_science_performed':False,'current_profile_resolved':False,
 'current_profile_mutated':False,'runtime_policy_activated':False}
open(done,'w').write(json.dumps(d,sort_keys=True,separators=(',',':'))+'\n')
PY
python - "$RUNMETA" "$RESULT" "$JOB_INDEX" "$SLOT" <<'PY'
import sys
from pathlib import Path
from ofc_regular.aggregate_hu_m43_attempt08_preflight import _load_canonical_json,_validate_proof
from ofc_regular.hu_m43_attempt08_preflight_spot import validate_done_metadata,validate_received_job_artifacts
runmeta,result,index,slot=Path(sys.argv[1]),Path(sys.argv[2]),int(sys.argv[3]),sys.argv[4]
done=validate_done_metadata(done_path=result/'DONE.json',run_dir=runmeta,
 authorization_path=runmeta/'spot_authorization.json',
 local_evidence_path=runmeta/'local_evidence.json',job_index=index,
 revalidate_local_commands=False)
validate_received_job_artifacts(directory=result,done=done)
proof=_load_canonical_json(result/'proof.json',label='new local proof')
_validate_proof(proof,slot=slot,source=runmeta/'package_src/preflight_source/teacher.jsonl')
PY
upload_once_or_verify(){
  source="$1"; uri="$2"
  if gcloud storage cp "$source" "$uri" --project "$PROJECT_ID" --if-generation-match=0 >/dev/null 2>&1; then return 0; fi
  existing="$(mktemp)"; gcloud storage cp "$uri" "$existing" --project "$PROJECT_ID" >/dev/null
  cmp -s "$source" "$existing" || { rm -f "$existing"; echo "immutable object differs: $uri" >&2; return 1; }
  rm -f "$existing"
}
# Proof and support are immutable.  A retry first reuses and validates an
# existing proof, then deterministically reconstructs the same support bytes.
upload_once_or_verify "$RESULT/proof.json" "$RESULT_URI/proof.json"
for name in checkpoint.json heartbeat.json summary.json run.log boot_image_evidence.json; do
  upload_once_or_verify "$RESULT/$name" "$RESULT_URI/$name"
done
# DONE is the only completion marker and is always uploaded last.
upload_once_or_verify "$RESULT/DONE.json" "$RESULT_URI/DONE.json"
