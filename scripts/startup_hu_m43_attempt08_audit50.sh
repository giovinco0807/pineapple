#!/usr/bin/env bash
set -euo pipefail

export HOME="${HOME:-/root}" PYTHONDONTWRITEBYTECODE=1
meta(){ curl -fsS -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }
imeta(){ curl -fsS -H 'Metadata-Flavor: Google' "http://metadata.google.internal/computeMetadata/v1/instance/$1"; }
sha(){ sha256sum "$1" | awk '{print $1}'; }
remote_exists(){
  local e; e="$(mktemp)"
  if gcloud storage objects describe "$1" --project "$PROJECT_ID" >/dev/null 2>"$e"; then rm -f "$e"; return 0; fi
  if grep -Eiq 'not found|does not exist|No URLs matched|404' "$e"; then rm -f "$e"; return 1; fi
  cat "$e" >&2; rm -f "$e"; echo "unable to prove object state: $1" >&2; exit 70
}
upload_once_or_verify(){
  local source="$1" uri="$2" existing
  if gcloud storage cp "$source" "$uri" --project "$PROJECT_ID" --if-generation-match=0 >/dev/null 2>&1; then return 0; fi
  existing="$(mktemp)"; gcloud storage cp "$uri" "$existing" --project "$PROJECT_ID" >/dev/null
  cmp -s "$source" "$existing" || { rm -f "$existing"; echo "immutable object differs: $uri" >&2; return 1; }
  rm -f "$existing"
}

RUN_NAME="$(meta RUN_NAME)"; DEV_RUN_NAME="$(meta DEV_RUN_NAME)"; PROJECT_ID="$(meta PROJECT_ID)"; BUCKET="$(meta BUCKET)"; SHARD="$(meta SHARD)"
AUDIT_SOURCE_SHA256="$(meta AUDIT_SOURCE_SHA256)"; AUDIT_MANIFEST_SHA256="$(meta AUDIT_MANIFEST_SHA256)"; AUDIT_LAUNCH_SHA256="$(meta AUDIT_LAUNCH_SHA256)"
AUDIT_OPEN_SHA256="$(meta AUDIT_OPEN_SHA256)"; CORE_AUTH_SHA256="$(meta CORE_AUTH_SHA256)"; FREEZE_SHA256="$(meta FREEZE_SHA256)"
DECISION_SHA256="$(meta DECISION_SHA256)"; RECEIPT_SHA256="$(meta RECEIPT_SHA256)"; AUDIT_SCHEDULE_SHA256="$(meta AUDIT_SCHEDULE_SHA256)"
STARTUP_SHA256="$(meta STARTUP_SHA256)"; DEV_MANIFEST_SHA256="$(meta DEV_MANIFEST_SHA256)"; DEV_LAUNCH_SHA256="$(meta DEV_LAUNCH_SHA256)"
DEV_SOURCE_SHA256="$(meta DEV_SOURCE_SHA256)"; DEV_SCHEDULE_SHA256="$(meta DEV_SCHEDULE_SHA256)"; DEV_STARTUP_SHA256="$(meta DEV_STARTUP_SHA256)"
SELF_DELETE="$(meta SELF_DELETE)"; INSTANCE_NAME="$(imeta name)"; ZONE="$(imeta zone)"; ZONE="${ZONE##*/}"
[[ "$SHARD" =~ ^([0-9]|[1-4][0-9])$ ]] && (( SHARD >= 0 && SHARD < 50 ))

PREFIX="gs://${BUCKET}/runs/${RUN_NAME}"; DEV_PREFIX="gs://${BUCKET}/runs/${DEV_RUN_NAME}"
BASE=/opt/ofc-attempt08-audit50; AUDIT="$BASE/audit_run"; DEV="$BASE/development_run"; OVERLAY="$AUDIT/overlay_src"; VENV="$BASE/.venv"
RESULT=/tmp/ofc-attempt08-audit50-result; LIVE_LOG=/tmp/ofc-attempt08-audit50-live.log
HEARTBEAT_PID=""
mkdir -p "$RESULT"; : >"$LIVE_LOG"; exec > >(tee -a "$LIVE_LOG") 2>&1
cleanup(){ code=$?; set +e; if [[ -n "$HEARTBEAT_PID" ]]; then kill "$HEARTBEAT_PID" >/dev/null 2>&1 || true; fi; gcloud storage cp "$LIVE_LOG" "$PREFIX/logs/shard-${SHARD}.log" --project "$PROJECT_ID" >/dev/null 2>&1 || true; if [[ "$SELF_DELETE" == 1 ]]; then gcloud compute instances delete "$INSTANCE_NAME" --zone "$ZONE" --project "$PROJECT_ID" --quiet >/dev/null 2>&1 || sudo shutdown -h now; fi; exit "$code"; }
trap cleanup EXIT

curl -fsS -H 'Metadata-Flavor: Google' 'http://metadata.google.internal/computeMetadata/v1/instance/attributes/startup-script' >/tmp/executing-startup.sh
[[ "$(sha /tmp/executing-startup.sh)" == "$STARTUP_SHA256" ]]
DISK="$(gcloud compute instances describe "$INSTANCE_NAME" --zone "$ZONE" --project "$PROJECT_ID" --format='value(disks[0].source.basename())')"
DISK_JSON="$(gcloud compute disks describe "$DISK" --zone "$ZONE" --project "$PROJECT_ID" --format=json)"
python3 - "$DISK_JSON" "$RESULT/boot_image_evidence.json" "$RUN_NAME" "$SHARD" "$INSTANCE_NAME" "$DISK" <<'PY'
import json,sys
d=json.loads(sys.argv[1]); assert str(d.get('sourceImage','')).endswith('projects/debian-cloud/global/images/debian-12-bookworm-v20260609'); assert str(d.get('sourceImageId'))=='1449487925682397051'
p={'schema':'hu_m43_attempt08_audit50_boot_image_evidence_v1','run_name':sys.argv[3],'shard':int(sys.argv[4]),'instance_name':sys.argv[5],'disk_name':sys.argv[6],'source_image':d['sourceImage'],'source_image_id':str(d['sourceImageId'])}
open(sys.argv[2],'w').write(json.dumps(p,sort_keys=True,separators=(',',':'))+'\n')
PY

export DEBIAN_FRONTEND=noninteractive
sudo rm -f /etc/apt/sources.list.d/debian.sources
printf '%s\n' 'deb [check-valid-until=no] http://snapshot.debian.org/archive/debian/20260609T000000Z bookworm main' 'deb [check-valid-until=no] http://snapshot.debian.org/archive/debian-security/20260609T000000Z bookworm-security main' | sudo tee /etc/apt/sources.list >/dev/null
sudo apt-get -o Acquire::Check-Valid-Until=false update -y
sudo apt-get -o Acquire::Check-Valid-Until=false install -y python3 python3-venv unzip time libgomp1 ca-certificates
sudo rm -rf "$BASE"; sudo mkdir -p "$AUDIT" "$DEV/package_src"; sudo chown -R "$(id -u):$(id -g)" "$BASE"

# Download and reconstruct the separately hashed audit overlay.
for pair in \
  "manifest.json manifest.json $AUDIT_MANIFEST_SHA256" \
  "source/launch_authorization.json launch_authorization.json $AUDIT_LAUNCH_SHA256" \
  "source/audit_open_authorization.json audit_open_authorization.json $AUDIT_OPEN_SHA256" \
  "source/core_audit_authorization.json core_audit_authorization.json $CORE_AUTH_SHA256" \
  "source/development_pass_freeze.json development_pass_freeze.json $FREEZE_SHA256" \
  "source/development_decision.json development_decision.json $DECISION_SHA256" \
  "source/development_selector_receipt.json development_selector_receipt.json $RECEIPT_SHA256" \
  "source/shards_manifest.jsonl shards_manifest.jsonl $AUDIT_SCHEDULE_SHA256" \
  "source/ofc_regular_hu_m43_attempt08_audit50_overlay.zip overlay.zip $AUDIT_SOURCE_SHA256"; do
  read -r remote local digest <<<"$pair"; gcloud storage cp "$PREFIX/$remote" "$AUDIT/$local" --project "$PROJECT_ID" >/dev/null; [[ "$(sha "$AUDIT/$local")" == "$digest" ]]
done
gcloud storage cp "$PREFIX/source/source_closure_manifest.json" "$AUDIT/source_closure_manifest.json" --project "$PROJECT_ID" >/dev/null
unzip -q "$AUDIT/overlay.zip" -d "$OVERLAY"
cp "$OVERLAY/configs/hu_joint_policy_m43_attempt08_audit50.json" "$AUDIT/hu_joint_policy_m43_attempt08_audit50.json"
cp /tmp/executing-startup.sh "$AUDIT/startup_hu_m43_attempt08_audit50.sh"
cp "$AUDIT/overlay.zip" "$AUDIT/ofc_regular_hu_m43_attempt08_audit50_overlay.zip"

# Reconstruct the byte-exact frozen development package in a separate tree.
gcloud storage cp "$DEV_PREFIX/manifest.json" "$DEV/manifest.json" --project "$PROJECT_ID" >/dev/null
gcloud storage cp "$DEV_PREFIX/source/launch_authorization.json" "$DEV/launch_authorization.json" --project "$PROJECT_ID" >/dev/null
gcloud storage cp "$DEV_PREFIX/source/shards_manifest.jsonl" "$DEV/shards_manifest.jsonl" --project "$PROJECT_ID" >/dev/null
gcloud storage cp "$DEV_PREFIX/source/ofc_regular_hu_m43_attempt08_development_source.zip" /tmp/dev-source.zip --project "$PROJECT_ID" >/dev/null
[[ "$(sha "$DEV/manifest.json")" == "$DEV_MANIFEST_SHA256" ]]; [[ "$(sha "$DEV/launch_authorization.json")" == "$DEV_LAUNCH_SHA256" ]]; [[ "$(sha "$DEV/shards_manifest.jsonl")" == "$DEV_SCHEDULE_SHA256" ]]; [[ "$(sha /tmp/dev-source.zip)" == "$DEV_SOURCE_SHA256" ]]
unzip -q /tmp/dev-source.zip -d "$DEV/package_src"
cp /tmp/dev-source.zip "$DEV/ofc_regular_hu_m43_attempt08_development_source.zip"
cp "$DEV/package_src/source_closure_manifest.json" "$DEV/source_closure_manifest.json"
cp "$DEV/package_src/scripts/startup_hu_m43_attempt08_development.sh" "$DEV/startup_hu_m43_attempt08_development.sh"; [[ "$(sha "$DEV/startup_hu_m43_attempt08_development.sh")" == "$DEV_STARTUP_SHA256" ]]
cp "$DEV/package_src/configs/hu_joint_policy_m43_attempt08.json" "$DEV/hu_joint_policy_m43_attempt08.json"
cp "$DEV/package_src/configs/hu_joint_policy_m43_attempt08_preflight.json" "$DEV/hu_joint_policy_m43_attempt08_preflight.json"
for name in development_open_authorization preflight_aggregate preflight_execution_evidence preflight_finalization preflight_source; do cp "$DEV/package_src/frozen/${name}.json" "$DEV/${name}.json"; done
mkdir -p "$DEV/preflight_proofs"; cp "$DEV/package_src/frozen/preflight_proofs/"*.json "$DEV/preflight_proofs/"
cp "$DEV/package_src/source_model_manifest.json" "$DEV/source_model_manifest.json"; cp "$DEV/package_src/source_native_manifest.json" "$DEV/source_native_manifest.json"

python3 -m venv "$VENV"; source "$VENV/bin/activate"
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu 'pip==26.1.2' 'setuptools==83.0.0' 'wheel==0.46.1'
python -m pip install -r "$DEV/package_src/configs/hu_m43_attempt08_runtime_requirements.txt"
export PYTHONPATH="$DEV/package_src/src" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 OFC_HU_M3_BATCH_THREADS=4
AUDIT_MODULE="$OVERLAY/src/ofc_regular/hu_m43_attempt08_audit50_spot.py"
python -B "$AUDIT_MODULE" validate-launch --run-dir "$AUDIT" --development-run-dir "$DEV" --authorization "$AUDIT/launch_authorization.json" >/dev/null

python -B "$AUDIT_MODULE" global-claim --run-dir "$AUDIT" --development-run-dir "$DEV" --output "$RESULT/global_claim.json" >/dev/null
upload_once_or_verify "$RESULT/global_claim.json" "$PREFIX/claims/global_claim.json"
python -B "$AUDIT_MODULE" root-claim --run-dir "$AUDIT" --development-run-dir "$DEV" --shard "$SHARD" --global-claim "$RESULT/global_claim.json" --output "$RESULT/root_claim.json" >/dev/null
upload_once_or_verify "$RESULT/root_claim.json" "$PREFIX/claims/root-${SHARD}.json"
OUTPUT_PREFIX="$(printf 'shard_%03d' "$SHARD")"; RESULT_URI="$PREFIX/results/$OUTPUT_PREFIX"; RESUME_URI="$PREFIX/resume/$OUTPUT_PREFIX"
FILES=(teacher.jsonl checkpoint.json heartbeat.json generator_summary.json run.log time.txt global_claim.json root_claim.json boot_image_evidence.json resume_commit.json)
if remote_exists "$RESULT_URI/DONE.json"; then
  for name in "${FILES[@]}" DONE.json; do gcloud storage cp "$RESULT_URI/$name" "$RESULT/$name" --project "$PROJECT_ID" >/dev/null; done
  python -B "$AUDIT_MODULE" validate-completed --run-dir "$AUDIT" --development-run-dir "$DEV" --shard "$SHARD" --directory "$RESULT" >/dev/null
  exit 0
fi
RESTORED_RESUME=0
if remote_exists "$RESUME_URI/resume_commit.json"; then
  RESTORED_RESUME=1
  gcloud storage cp "$RESUME_URI/resume_commit.json" "$RESULT/resume_commit.json" --project "$PROJECT_ID" >/dev/null
  python -B "$AUDIT_MODULE" validate-resume-commit --run-dir "$AUDIT" --development-run-dir "$DEV" --shard "$SHARD" --commit "$RESULT/resume_commit.json" >/dev/null
  while IFS=$'\t' read -r name digest; do
    [[ "$digest" =~ ^[0-9a-f]{64}$ ]]
    gcloud storage cp "$RESUME_URI/objects/$digest/$name" "$RESULT/$name" --project "$PROJECT_ID" >/dev/null
    [[ "$(sha "$RESULT/$name")" == "$digest" ]]
  done < <(python - "$RESULT/resume_commit.json" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]))
assert c['schema']=='hu_m43_attempt08_audit50_resume_commit_v1'
assert set(c['files'])=={'teacher.jsonl','checkpoint.json','heartbeat.json','generator_summary.json','run.log','time.txt','global_claim.json','root_claim.json','boot_image_evidence.json'}
for name,digest in sorted(c['files'].items()): print(name+'\t'+digest)
PY
  )
else
  heartbeat_loop(){ while true; do printf '{"epoch":%s,"run_name":"%s","schema":"hu_m43_attempt08_audit50_live_heartbeat_v1","shard":%s,"status":"running"}\n' "$(date +%s)" "$RUN_NAME" "$SHARD" >/tmp/audit50-live-heartbeat.json; gcloud storage cp /tmp/audit50-live-heartbeat.json "$PREFIX/heartbeat/shard-${SHARD}.json" --project "$PROJECT_ID" >/dev/null 2>&1 || true; sleep 60; done; }
  heartbeat_loop & HEARTBEAT_PID=$!
  python -B "$AUDIT_MODULE" run-shard --run-dir "$AUDIT" --development-run-dir "$DEV" --shard "$SHARD" --directory "$RESULT" >/dev/null
  kill "$HEARTBEAT_PID" >/dev/null 2>&1 || true; wait "$HEARTBEAT_PID" 2>/dev/null || true; HEARTBEAT_PID=""
fi
python -B "$AUDIT_MODULE" complete-shard --run-dir "$AUDIT" --development-run-dir "$DEV" --shard "$SHARD" --directory "$RESULT" >/dev/null
python -B "$AUDIT_MODULE" validate-completed --run-dir "$AUDIT" --development-run-dir "$DEV" --shard "$SHARD" --directory "$RESULT" >/dev/null
if [[ "$RESTORED_RESUME" == 0 ]]; then
  while IFS=$'\t' read -r name digest; do
    upload_once_or_verify "$RESULT/$name" "$RESUME_URI/objects/$digest/$name"
  done < <(python - "$RESULT/resume_commit.json" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]))
assert c['schema']=='hu_m43_attempt08_audit50_resume_commit_v1'
for name,digest in sorted(c['files'].items()): print(name+'\t'+digest)
PY
  )
  upload_once_or_verify "$RESULT/resume_commit.json" "$RESUME_URI/resume_commit.json"
fi
for name in "${FILES[@]}"; do upload_once_or_verify "$RESULT/$name" "$RESULT_URI/$name"; done
upload_once_or_verify "$RESULT/DONE.json" "$RESULT_URI/DONE.json"
python -B "$AUDIT_MODULE" validate-completed --run-dir "$AUDIT" --development-run-dir "$DEV" --shard "$SHARD" --directory "$RESULT" >/dev/null
