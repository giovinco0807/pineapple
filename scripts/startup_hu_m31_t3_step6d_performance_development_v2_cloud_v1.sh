#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

# Fresh startup for the performance-development-v2 two-VM tail probe.  It is
# deliberately independent of every rearm/performance-lock startup.  The
# controller pins these bytes and every metadata value before VM creation.

META_ROOT='http://metadata.google.internal/computeMetadata/v1/instance/attributes'
META_HEADER='Metadata-Flavor: Google'
WORK='/opt/ofc-perfdev-v2'
BASE="$WORK/base"
VENV='/opt/ofc-perfdev-v2-venv'
OUTPUT='/var/lib/ofc-perfdev-v2/output'
LOG='/var/log/ofc-perfdev-v2-startup.log'
MAIN_PID=$$
PUMP_PID=''
WATCHDOG_PID=''
VALIDATED_DONE=0

meta() {
  curl -fsS -H "$META_HEADER" "$META_ROOT/$1"
}

sha() {
  sha256sum "$1" | awk '{print $1}'
}

token() {
  curl -fsS -H "$META_HEADER" \
    'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token' \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])'
}

encoded_object() {
  python3 - "$1" <<'PY'
import sys
import urllib.parse
print(urllib.parse.quote(sys.argv[1], safe=""))
PY
}

download_object() {
  local object=$1
  local destination=$2
  local encoded
  encoded="$(encoded_object "$object")"
  curl -fsS -H "Authorization: Bearer $(token)" \
    "https://storage.googleapis.com/download/storage/v1/b/${BUCKET}/o/${encoded}?alt=media" \
    -o "$destination"
}

upload_once() {
  local source=$1
  local object=$2
  local encoded
  encoded="$(encoded_object "$object")"
  # The diagnostic worker intentionally has create-only access outside control/.
  # Therefore a 412 collision cannot be GET-compared and is never treated as a
  # successful resume.  Attempt-1/restart is not implemented: required result,
  # checkpoint, and heartbeat callers fail this attempt closed; the shutdown log
  # remains explicitly best-effort and never proves success.
  if curl -fsS -X POST -H "Authorization: Bearer $(token)" \
      -H 'Content-Type: application/octet-stream' \
      --data-binary "@$source" \
      "https://storage.googleapis.com/upload/storage/v1/b/${BUCKET}/o?uploadType=media&ifGenerationMatch=0&name=${encoded}" \
      >/dev/null; then
    return 0
  fi
  return 1
}

upload_progress_checkpoint_once() {
  local source=$1
  local object=$2
  local object_key marker current_sha snapshot snapshot_sha after_snapshot_sha
  local after_upload_sha marker_tmp
  local -a marker_lines
  if [[ ! -f "$source" || "$object" == *$'\n'* || "$object" == *$'\r'* ]]; then
    return 1
  fi
  object_key="$(printf '%s' "$object" | sha256sum | awk '{print $1}')"
  marker="${PROGRESS_UPLOAD_REGISTRY}/${object_key}.sent"
  current_sha="$(sha "$source")"
  if [[ ! "$object_key" =~ ^[0-9a-f]{64}$ || ! "$current_sha" =~ ^[0-9a-f]{64}$ ]]; then
    return 1
  fi
  if [[ -f "$marker" ]]; then
    if ! mapfile -t marker_lines < "$marker"; then
      return 1
    fi
    if [[ "${#marker_lines[@]}" -ne 2 \
        || "${marker_lines[0]}" != "$object" \
        || "${marker_lines[1]}" != "$current_sha" ]]; then
      # A published immutable object path changed locally.  Never issue a
      # second create-only request and never treat the path as resumable.
      return 1
    fi
    return 0
  fi

  if ! snapshot="$(mktemp "${PROGRESS_UPLOAD_REGISTRY}/.snapshot.XXXXXX")"; then
    return 1
  fi
  if ! cp --no-preserve=mode,ownership,timestamps "$source" "$snapshot"; then
    rm -f "$snapshot"
    return 1
  fi
  snapshot_sha="$(sha "$snapshot")"
  after_snapshot_sha="$(sha "$source")"
  if [[ "$current_sha" != "$snapshot_sha" || "$current_sha" != "$after_snapshot_sha" ]]; then
    rm -f "$snapshot"
    return 1
  fi
  if ! upload_once "$snapshot" "$object"; then
    rm -f "$snapshot"
    return 1
  fi
  after_upload_sha="$(sha "$source")"
  rm -f "$snapshot"
  if [[ ! "$after_upload_sha" =~ ^[0-9a-f]{64}$ \
      || "$current_sha" != "$after_upload_sha" ]]; then
    return 1
  fi
  if ! marker_tmp="$(mktemp "${PROGRESS_UPLOAD_REGISTRY}/.marker.XXXXXX")"; then
    return 1
  fi
  if ! printf '%s\n%s\n' "$object" "$current_sha" > "$marker_tmp" \
      || ! chmod 0600 "$marker_tmp" \
      || ! mv "$marker_tmp" "$marker"; then
    rm -f "$marker_tmp"
    return 1
  fi
  return 0
}

progress_upload_pass() {
  local checkpoint relative
  while IFS= read -r -d '' checkpoint; do
    relative="${checkpoint#"$OUTPUT"/}"
    upload_progress_checkpoint_once \
      "$checkpoint" "${RESULT_PREFIX}progress/${SOURCE_ROLE}/${relative}" || return 1
  done < <(find "$OUTPUT" -type f -name '*.json' -print0 2>/dev/null)
}

require_progress_pump_alive() {
  local pump_status=0
  if [[ -z "$PUMP_PID" ]]; then
    return 1
  fi
  if kill -0 "$PUMP_PID" >/dev/null 2>&1; then
    return 0
  fi
  wait "$PUMP_PID" >/dev/null 2>&1 || pump_status=$?
  PUMP_PID=''
  echo "heartbeat/progress pump exited before validated completion (status=${pump_status})" >&2
  return 1
}

stop_background() {
  if [[ -n "$PUMP_PID" ]]; then
    kill "$PUMP_PID" >/dev/null 2>&1 || true
    wait "$PUMP_PID" >/dev/null 2>&1 || true
    PUMP_PID=''
  fi
  if [[ -n "$WATCHDOG_PID" ]]; then
    kill "$WATCHDOG_PID" >/dev/null 2>&1 || true
    wait "$WATCHDOG_PID" >/dev/null 2>&1 || true
    WATCHDOG_PID=''
  fi
}

cleanup() {
  local code=$?
  trap - EXIT TERM INT
  set +e
  stop_background
  if [[ "$code" -ne 0 || "$VALIDATED_DONE" -ne 1 ]]; then
    [[ "$code" -ne 0 ]] || code=1
  fi
  if [[ -f "$LOG" ]]; then
    upload_once "$LOG" "${RESULT_PREFIX}logs/${SOURCE_ROLE}/attempt-${ATTEMPT_INDEX}/startup.log" || true
  fi
  # Never issue a guest shutdown/poweroff.  A guest stop can leave the VM and
  # boot disk TERMINATED indefinitely.  The controller owns exact deletion;
  # if it disappears, GCE maxRunDuration + instanceTerminationAction=DELETE
  # deletes the still-running instance at the provider boundary.
  exit "$code"
}
trap cleanup EXIT
trap 'exit 124' TERM INT
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

PROJECT_ID="$(meta PROJECT_ID)"
BUCKET="$(meta BUCKET)"
RUN_NAME="$(meta RUN_NAME)"
IDENTITY_NAMESPACE="$(meta IDENTITY_NAMESPACE)"
SOURCE_ROLE="$(meta SOURCE_ROLE)"
RESULT_PREFIX="$(meta RESULT_PREFIX)"
SOURCE_OBJECT="$(meta SOURCE_OBJECT)"
SOURCE_SHA256="$(meta SOURCE_SHA256)"
WHEELHOUSE_OBJECT="$(meta WHEELHOUSE_OBJECT)"
WHEELHOUSE_SHA256="$(meta WHEELHOUSE_SHA256)"
WHEELHOUSE_MANIFEST_SHA256="$(meta WHEELHOUSE_MANIFEST_SHA256)"
PLAN_OBJECT="$(meta PLAN_OBJECT)"
PLAN_SHA256="$(meta PLAN_SHA256)"
AUTHORIZATION_OBJECT="$(meta AUTHORIZATION_OBJECT)"
AUTHORIZATION_SHA256="$(meta AUTHORIZATION_SHA256)"
AUTHORIZATION_NONCE_SHA256="$(meta AUTHORIZATION_NONCE_SHA256)"
STARTUP_SHA256="$(meta STARTUP_SHA256)"
OWNERSHIP_LABEL="$(meta OWNERSHIP_LABEL)"
INSTANCE_NAME_EXPECTED="$(meta INSTANCE_NAME)"
ATTEMPT_INDEX="$(meta ATTEMPT_INDEX)"
RESUME_OBJECT="$(meta RESUME_OBJECT)"
RESUME_SHA256="$(meta RESUME_SHA256)"
MAX_RUNTIME_SECONDS="$(meta MAX_RUNTIME_SECONDS)"
HEARTBEAT_SECONDS="$(meta HEARTBEAT_SECONDS)"

INSTANCE_NAME="$(curl -fsS -H "$META_HEADER" \
  'http://metadata.google.internal/computeMetadata/v1/instance/name')"

[[ "$RUN_NAME" =~ ^regular-hu-m31-c02-perfdev-v2-[a-z0-9][a-z0-9-]{7,47}$ ]]
[[ "$IDENTITY_NAMESPACE" =~ ^perfdev-v2-[a-z0-9][a-z0-9-]{7,47}$ ]]
[[ "$SOURCE_ROLE" == candidate || "$SOURCE_ROLE" == reference ]]
[[ "$RESULT_PREFIX" == "hu-m31-t3/perfdev-v2/${RUN_NAME}/" ]]
[[ "$SOURCE_SHA256" =~ ^[0-9a-f]{64}$ ]]
[[ "$WHEELHOUSE_SHA256" =~ ^[0-9a-f]{64}$ ]]
[[ "$WHEELHOUSE_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]]
[[ "$PLAN_SHA256" =~ ^[0-9a-f]{64}$ ]]
[[ "$AUTHORIZATION_SHA256" =~ ^[0-9a-f]{64}$ ]]
[[ "$AUTHORIZATION_NONCE_SHA256" =~ ^[0-9a-f]{64}$ ]]
[[ "$STARTUP_SHA256" =~ ^[0-9a-f]{64}$ ]]
[[ "$OWNERSHIP_LABEL" =~ ^pdv2-[0-9a-f]{20}$ ]]
[[ "$INSTANCE_NAME" == "$INSTANCE_NAME_EXPECTED" ]]
[[ "$ATTEMPT_INDEX" == 0 ]]
[[ "$MAX_RUNTIME_SECONDS" == 4200 ]]
[[ "$HEARTBEAT_SECONDS" == 30 ]]
[[ "$(sha "$0")" == "$STARTUP_SHA256" ]]
[[ "$RESUME_OBJECT" == none && "$RESUME_SHA256" == none ]]

(
  sleep "$MAX_RUNTIME_SECONDS"
  kill -TERM "$MAIN_PID" >/dev/null 2>&1 || true
) &
WATCHDOG_PID=$!

printf '%s\n' \
  'deb [check-valid-until=no] https://snapshot.debian.org/archive/debian/20260722T000000Z bookworm main' \
  'deb [check-valid-until=no] https://snapshot.debian.org/archive/debian-security/20260722T000000Z bookworm-security main' \
  > /etc/apt/sources.list
apt-get -o Acquire::Check-Valid-Until=false update -y
apt-get -o Acquire::Check-Valid-Until=false install -y \
  python3 python3-venv unzip libgomp1 ca-certificates

rm -rf "$WORK" "$VENV" "$OUTPUT"
mkdir -p "$WORK" "$OUTPUT"
download_object "$SOURCE_OBJECT" "$WORK/source.zip"
download_object "$WHEELHOUSE_OBJECT" "$WORK/wheelhouse.zip"
download_object "$PLAN_OBJECT" "$WORK/execution_plan.json"
download_object "$AUTHORIZATION_OBJECT" "$WORK/authorization.json"
[[ "$(sha "$WORK/source.zip")" == "$SOURCE_SHA256" ]]
[[ "$(sha "$WORK/wheelhouse.zip")" == "$WHEELHOUSE_SHA256" ]]
[[ "$(sha "$WORK/execution_plan.json")" == "$PLAN_SHA256" ]]
[[ "$(sha "$WORK/authorization.json")" == "$AUTHORIZATION_SHA256" ]]
python3 -m zipfile -e "$WORK/source.zip" "$BASE"
mkdir -p "$WORK/wheelhouse"
python3 -m zipfile -e "$WORK/wheelhouse.zip" "$WORK/wheelhouse"

python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --disable-pip-version-check --no-input --no-index \
  --find-links "$WORK/wheelhouse" \
  -r "$BASE/payload/tooling/configs/hu_m43_attempt08_runtime_requirements.txt"
python -m pip check
python - "$BASE/payload/tooling/configs/hu_m43_attempt08_runtime_requirements.txt" <<'PY'
import importlib.metadata
import pathlib
import sys

for raw in pathlib.Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("--"):
        continue
    name, expected = line.split("==", 1)
    actual = importlib.metadata.version(name)
    if actual != expected:
        raise SystemExit(f"offline pinned dependency mismatch: {name} {actual} != {expected}")
PY

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$BASE/payload/tooling/src"

# Validate the extracted package while it is still byte-for-byte identical to
# the immutable manifest.  The feature encoder runtime copy below is required
# by the worker, but it is intentionally not a package entry and therefore
# must only be created after this closed-world validation succeeds.
python3 - "$BASE" "$WORK/execution_plan.json" "$WORK/authorization.json" \
  "$PROJECT_ID" "$BUCKET" "$RUN_NAME" "$IDENTITY_NAMESPACE" "$RESULT_PREFIX" \
  "$SOURCE_ROLE" "$INSTANCE_NAME" "$SOURCE_OBJECT" "$SOURCE_SHA256" \
  "$WHEELHOUSE_OBJECT" "$WHEELHOUSE_SHA256" "$WHEELHOUSE_MANIFEST_SHA256" \
  "$PLAN_OBJECT" "$PLAN_SHA256" "$AUTHORIZATION_OBJECT" \
  "$AUTHORIZATION_SHA256" "$AUTHORIZATION_NONCE_SHA256" "$STARTUP_SHA256" \
  "$OWNERSHIP_LABEL" "$ATTEMPT_INDEX" "$RESUME_OBJECT" "$RESUME_SHA256" \
  "$MAX_RUNTIME_SECONDS" "$HEARTBEAT_SECONDS" <<'PY'
import hashlib
import json
import pathlib
import sys

from ofc_regular import hu_m31_t3_step6d_performance_development_v2_cloud_execution_v1 as cloud
from ofc_regular import hu_m31_t3_step6d_performance_development_v2_local_package as local

base = pathlib.Path(sys.argv[1])
plan = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
authorization = json.loads(pathlib.Path(sys.argv[3]).read_text(encoding="utf-8"))
local.validate_local_package(base, _expected_run_name=plan["run_name"])
cloud.validate_execution_plan(plan, embedded_local_package=base, require_fresh_receipt=False)
expected = {
    "project_id": sys.argv[4],
    "bucket": sys.argv[5],
    "run_name": sys.argv[6],
    "identity_namespace": sys.argv[7],
    "result_prefix": sys.argv[8],
    "source_role": sys.argv[9],
    "instance_name": sys.argv[10],
    "source_object": sys.argv[11],
    "source_sha256": sys.argv[12],
    "wheelhouse_object": sys.argv[13],
    "wheelhouse_sha256": sys.argv[14],
    "wheelhouse_manifest_sha256": sys.argv[15],
    "plan_object": sys.argv[16],
    "plan_sha256": sys.argv[17],
    "authorization_object": sys.argv[18],
    "authorization_sha256": sys.argv[19],
    "authorization_nonce_sha256": sys.argv[20],
    "startup_sha256": sys.argv[21],
    "ownership_label": sys.argv[22],
    "attempt_index": int(sys.argv[23]),
    "resume_object": sys.argv[24],
    "resume_sha256": sys.argv[25],
    "max_runtime_seconds": int(sys.argv[26]),
    "heartbeat_seconds": int(sys.argv[27]),
}
cloud.validate_runtime_binding(plan, authorization, expected)
PY

FEATURE_SOURCE="$BASE/payload/source/target/release/libofc_stage3_feature_encoder.so"
FEATURE_RUNTIME="$BASE/payload/tooling/target/release/libofc_stage3_feature_encoder.so"
[[ "$(sha "$FEATURE_SOURCE")" == 82510e563ee290cd536e7aebe6bcf0efefac061af5488851f0a582bb4ef83411 ]]
mkdir -p "$(dirname "$FEATURE_RUNTIME")"
cp --no-preserve=mode,ownership,timestamps "$FEATURE_SOURCE" "$FEATURE_RUNTIME"
[[ ! -L "$FEATURE_RUNTIME" ]]
[[ "$(sha "$FEATURE_RUNTIME")" == 82510e563ee290cd536e7aebe6bcf0efefac061af5488851f0a582bb4ef83411 ]]

export RAYON_NUM_THREADS=16
export OFC_HU_M3_BATCH_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python - <<'PY'
import lightgbm
import numpy
import scipy
import sklearn
import torch
from ofc_regular.hu_turn3_stage3_feature_rust import rust_direct_available

if not rust_direct_available():
    raise SystemExit("hash-pinned Rust feature encoder failed its load probe")
PY

HEARTBEAT_DIR="$WORK/heartbeats"
PROGRESS_UPLOAD_REGISTRY="$WORK/progress-upload-registry"
mkdir -p "$HEARTBEAT_DIR" "$PROGRESS_UPLOAD_REGISTRY"
chmod 0700 "$PROGRESS_UPLOAD_REGISTRY"
(
  sequence=0
  while true; do
    sequence=$((sequence + 1))
    stamp="$(date -u +%s)"
    heartbeat="$HEARTBEAT_DIR/${stamp}-${sequence}.json"
    python3 - "$heartbeat" "$RUN_NAME" "$SOURCE_ROLE" "$INSTANCE_NAME" \
      "$ATTEMPT_INDEX" "$stamp" "$sequence" "$PLAN_SHA256" <<'PY'
import json, pathlib, sys
value = {
    "schema": "hu_m31_t3_step6d_perfdev_v2_heartbeat_v1",
    "run_name": sys.argv[2], "source_role": sys.argv[3],
    "instance_name": sys.argv[4], "attempt_index": int(sys.argv[5]),
    "unix_seconds": int(sys.argv[6]), "sequence": int(sys.argv[7]),
    "execution_plan_sha256": sys.argv[8],
}
pathlib.Path(sys.argv[1]).write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="ascii")
PY
    upload_once "$heartbeat" "${RESULT_PREFIX}heartbeats/${SOURCE_ROLE}/attempt-${ATTEMPT_INDEX}/${stamp}-${sequence}.json" || {
      kill -TERM "$MAIN_PID" >/dev/null 2>&1 || true
      exit 1
    }
    progress_upload_pass || {
      kill -TERM "$MAIN_PID" >/dev/null 2>&1 || true
      exit 1
    }
    sleep "$HEARTBEAT_SECONDS"
  done
) &
PUMP_PID=$!

ROLE_MANIFEST="$BASE/payload/runtime/roles/${SOURCE_ROLE}.json"
if [[ "$SOURCE_ROLE" == candidate ]]; then
  LIBRARY="$BASE/payload/source/native/candidate/release/libofc_hu_m3_engine.so"
else
  LIBRARY="$BASE/payload/source/native/reference/release/libofc_hu_m3_engine.so"
fi

python3 -m ofc_regular.run_hu_m31_t3_step6d_performance_v2 \
  --repository-root "$BASE/payload/tooling" \
  --output-dir "$OUTPUT" \
  --shard-manifest "$ROLE_MANIFEST" \
  --library "$LIBRARY"

# A background failure must never be hidden by a successful runner exit or by
# locally generated (but not remotely published) stale heartbeat files.
require_progress_pump_alive

python3 - "$OUTPUT" "$WORK/RESULT_MANIFEST.json" "$RUN_NAME" "$SOURCE_ROLE" \
  "$INSTANCE_NAME" "$ATTEMPT_INDEX" "$SOURCE_SHA256" "$PLAN_SHA256" \
  "$AUTHORIZATION_SHA256" "$AUTHORIZATION_NONCE_SHA256" "$STARTUP_SHA256" \
  "$OWNERSHIP_LABEL" "$HEARTBEAT_DIR" <<'PY'
import hashlib
import json
import pathlib
import sys

from ofc_regular import hu_m31_t3_step6d_performance_development_v2_cloud_execution_v1 as cloud
from ofc_regular import run_hu_m31_t3_step6d_performance_v2 as runner

root = pathlib.Path(sys.argv[1])
runner.validate_completed_output(root)
_contract, shard_manifest = cloud._validate_tail_v2_runtime_artifacts(
    {
        "run_contract.json": (root / "run_contract.json").read_bytes(),
        "shard_manifest.json": (root / "shard_manifest.json").read_bytes(),
    },
    source_role=sys.argv[4],
)
records = []
for path in sorted(root.rglob("*")):
    if path.is_file():
        raw = path.read_bytes()
        records.append({
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        })
value = {
    "schema": cloud.ROLE_RESULT_SCHEMA,
    "status": "complete_validated_role_result",
    "run_name": sys.argv[3], "source_role": sys.argv[4],
    "instance_name": sys.argv[5], "attempt_index": int(sys.argv[6]),
    "source_sha256": sys.argv[7], "execution_plan_sha256": sys.argv[8],
    "authorization_sha256": sys.argv[9],
    "authorization_nonce_sha256": sys.argv[10], "startup_sha256": sys.argv[11],
    "ownership_label": sys.argv[12],
    "work_hand_indices": shard_manifest["work_hand_indices"],
    "artifact_count": len(records), "artifacts": records,
    "artifact_manifest_sha256": hashlib.sha256(
        (json.dumps(records, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")
    ).hexdigest(),
    "heartbeat_count": len(list(pathlib.Path(sys.argv[13]).glob("*.json"))),
    "runner_validation_passed": True,
    "partial_result": False,
}
pathlib.Path(sys.argv[2]).write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="ascii")
PY

while IFS= read -r -d '' artifact; do
  relative="${artifact#"$OUTPUT"/}"
  upload_once "$artifact" "${RESULT_PREFIX}results/${SOURCE_ROLE}/${relative}"
done < <(find "$OUTPUT" -type f -print0)
upload_once "$WORK/RESULT_MANIFEST.json" \
  "${RESULT_PREFIX}results/${SOURCE_ROLE}/RESULT_MANIFEST.json"

# The pump must remain healthy through the final immutable publish, not merely
# through runner exit.  Otherwise locally stale heartbeat files could be used
# to mark a run successful after remote heartbeat/checkpoint publication died.
require_progress_pump_alive
VALIDATED_DONE=1
exit 0
