#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

# Dedicated Candidate02 full-100 performance-development worker.  This script
# is intentionally separate from the immutable tail startup.  One VM loads one
# source library and advances one fixed ten-hand shard, checkpointing every
# completed hand before DONE is committed.

META_ROOT='http://metadata.google.internal/computeMetadata/v1/instance/attributes'
META_HEADER='Metadata-Flavor: Google'
WORK='/opt/ofc-step6d-full100-v1'
VENV='/opt/ofc-step6d-full100-v1-venv'
STATE='/var/lib/ofc-step6d-full100-v1'
LOG='/var/log/ofc-step6d-full100-v1-startup.log'
MAIN_PID=$$
PUMP_PID=''
WATCHDOG_PID=''
VALIDATED_DONE=0
PROJECT_ID=''
INSTANCE_NAME=''
ZONE=''
RESULT=''
RESULT_PREFIX=''
PROGRESS_PREFIX=''
VALIDATED_INDEX_FILE=''
MAX_RUNTIME_SECONDS=3900
SELF_DELETE=0

meta() {
  curl -fsS -H "$META_HEADER" "$META_ROOT/$1"
}

sha() {
  sha256sum "$1" | awk '{print $1}'
}

stop_pump() {
  if [[ -n "$PUMP_PID" ]]; then
    kill "$PUMP_PID" >/dev/null 2>&1 || true
    wait "$PUMP_PID" >/dev/null 2>&1 || true
    PUMP_PID=''
  fi
}

stop_watchdog() {
  if [[ -n "$WATCHDOG_PID" ]]; then
    kill "$WATCHDOG_PID" >/dev/null 2>&1 || true
    wait "$WATCHDOG_PID" >/dev/null 2>&1 || true
    WATCHDOG_PID=''
  fi
}

gcs_state() {
  local uri=$1
  local error
  error="$(mktemp)"
  if gcloud storage objects describe "$uri" --project "$PROJECT_ID" \
      >/dev/null 2>"$error"; then
    rm -f "$error"
    printf 'present\n'
    return 0
  fi
  if grep -Eqi 'not found|does not exist|no urls matched|404' "$error"; then
    rm -f "$error"
    printf 'absent\n'
    return 0
  fi
  cat "$error" >&2
  rm -f "$error"
  return 1
}

upload_once() {
  local source=$1
  local destination=$2
  local existing
  existing="$(mktemp)"
  if [[ "$(gcs_state "$destination")" == present ]]; then
    gcloud storage cp "$destination" "$existing" --project "$PROJECT_ID" \
      >/dev/null
    [[ "$(sha "$source")" == "$(sha "$existing")" ]]
    rm -f "$existing"
    return
  fi
  if ! gcloud storage cp "$source" "$destination" --project "$PROJECT_ID" \
      --if-generation-match=0 >/dev/null 2>&1; then
    gcloud storage cp "$destination" "$existing" --project "$PROJECT_ID" \
      >/dev/null
    [[ "$(sha "$source")" == "$(sha "$existing")" ]]
  fi
  rm -f "$existing"
}

restore_or_compare() {
  local relative=$1
  local destination="$RESULT/$relative"
  local uri temporary
  for uri in \
    "$RESULT_PREFIX/$relative" \
    "$PROGRESS_PREFIX/$relative"; do
    if [[ "$(gcs_state "$uri")" != present ]]; then
      continue
    fi
    temporary="$(mktemp)"
    gcloud storage cp "$uri" "$temporary" --project "$PROJECT_ID" >/dev/null
    if [[ -f "$destination" ]]; then
      [[ "$(sha "$destination")" == "$(sha "$temporary")" ]]
      rm -f "$temporary"
    else
      mkdir -p "$(dirname "$destination")"
      mv "$temporary" "$destination"
    fi
  done
}

upload_log() {
  if [[ -n "$RESULT_PREFIX" && -n "$PROJECT_ID" && -f "$LOG" ]]; then
    gcloud storage cp "$LOG" \
      "${RESULT_PREFIX%/results/jobs/*}/logs/jobs/$JOB_ID/attempt-$ATTEMPT_INDEX/startup.log" \
      --project "$PROJECT_ID" >/dev/null 2>&1 || true
  fi
}

sync_failure_progress() {
  [[ -d "$RESULT" && -n "$PROGRESS_PREFIX" ]] || return 0
  local source relative
  while IFS= read -r -d '' source; do
    relative="${source#"$RESULT"/}"
    case "$relative" in
      run_contract.json|shard_manifest.json|roots/hand_[0-9][0-9][0-9].json|hands/candidate/hand_[0-9][0-9][0-9].json|hands/reference/hand_[0-9][0-9][0-9].json)
        upload_once "$source" "$PROGRESS_PREFIX/$relative" || true
        ;;
    esac
  done < <(find "$RESULT" -type f -print0 2>/dev/null)
}

force_shutdown() {
  if sudo shutdown -h now >/dev/null 2>&1; then
    return 0
  fi
  sudo systemctl poweroff --force --force >/dev/null 2>&1 || \
    sudo poweroff -f >/dev/null 2>&1 || true
}

delete_instance_or_shutdown() {
  if [[ -n "$INSTANCE_NAME" && -n "$ZONE" && -n "$PROJECT_ID" ]] && \
      gcloud compute instances delete "$INSTANCE_NAME" --zone "$ZONE" \
        --project "$PROJECT_ID" --quiet >/dev/null 2>&1; then
    return 0
  fi
  force_shutdown
}

startup_error() {
  local code=$1
  local line=$2
  local command=$3
  trap - ERR
  printf 'full100 startup failure: exit=%s line=%s command=%s\n' \
    "$code" "$line" "$command" >>"$LOG"
  exit "$code"
}

cleanup() {
  local code=$?
  trap - EXIT TERM INT
  set +e
  stop_pump
  stop_watchdog
  if [[ "$code" -ne 0 || "$VALIDATED_DONE" -ne 1 ]]; then
    [[ "$code" -ne 0 ]] || code=1
    sync_failure_progress
    upload_log
    delete_instance_or_shutdown
    exit "$code"
  fi
  upload_log
  if [[ "$SELF_DELETE" -ne 1 ]]; then
    echo 'validated full100 success lacks self-delete authorization' >&2
    delete_instance_or_shutdown
    exit 1
  fi
  if ! gcloud compute instances delete "$INSTANCE_NAME" --zone "$ZONE" \
      --project "$PROJECT_ID" --quiet >/dev/null 2>&1; then
    echo 'full100 self-delete failed; forcing shutdown' >&2
    force_shutdown
    exit 1
  fi
  exit 0
}
trap cleanup EXIT
trap 'exit 124' TERM
trap 'startup_error "$?" "$LINENO" "$BASH_COMMAND"' ERR
# Keep the authoritative startup log on the filesystem synchronously.  The
# EXIT trap uploads it before the self-delete, including failures that occur
# during the early metadata trust-boundary checks.
exec >>"$LOG" 2>&1

PROJECT_ID="$(meta PROJECT_ID)"
BUCKET="$(meta BUCKET)"
RUN_NAME="$(meta RUN_NAME)"
JOB_ID="$(meta JOB_ID)"
INSTANCE_NAME="$(meta INSTANCE_NAME)"
ZONE="$(meta ZONE)"
SOURCE_URI="$(meta SOURCE_URI)"
SOURCE_SHA256="$(meta SOURCE_SHA256)"
MANIFEST_URI="$(meta MANIFEST_URI)"
MANIFEST_SHA256="$(meta MANIFEST_SHA256)"
JOB_MANIFEST_URI="$(meta JOB_MANIFEST_URI)"
JOB_MANIFEST_SHA256="$(meta JOB_MANIFEST_SHA256)"
AUTHORIZATION_URI="$(meta AUTHORIZATION_URI)"
AUTHORIZATION_SHA256="$(meta AUTHORIZATION_SHA256)"
ATTEMPT_INDEX="$(meta ATTEMPT_INDEX)"
ATTEMPT_CLAIM_URI="$(meta ATTEMPT_CLAIM_URI)"
ATTEMPT_CLAIM_SHA256="$(meta ATTEMPT_CLAIM_SHA256)"
INITIAL_LAUNCH_CLAIM_SHA256="$(meta INITIAL_LAUNCH_CLAIM_SHA256)"
INITIAL_LAUNCH_RESULT_SHA256="$(meta INITIAL_LAUNCH_RESULT_SHA256)"
RESULT_PREFIX="$(meta RESULT_PREFIX)"
PROGRESS_PREFIX="$(meta PROGRESS_PREFIX)"
COST_GUARD_SHA256="$(meta COST_GUARD_SHA256)"
MAX_RUNTIME_SECONDS="$(meta MAX_RUNTIME_SECONDS)"
SELF_DELETE="$(meta SELF_DELETE)"
EXECUTED_STARTUP_SHA256="$(sha "${BASH_SOURCE[0]}")"

ACTUAL_INSTANCE_NAME="$(curl -fsS -H "$META_HEADER" \
  'http://metadata.google.internal/computeMetadata/v1/instance/name')"
ACTUAL_ZONE="$(curl -fsS -H "$META_HEADER" \
  'http://metadata.google.internal/computeMetadata/v1/instance/zone' \
  | awk -F/ '{print $NF}')"

[[ "$RUN_NAME" =~ ^[a-z0-9][a-z0-9-]{2,46}[a-z0-9]$ ]]
[[ "$JOB_ID" =~ ^(candidate|reference)-shard-(0[0-9])$ ]]
[[ "$INSTANCE_NAME" == "$ACTUAL_INSTANCE_NAME" ]]
[[ "$ZONE" == "$ACTUAL_ZONE" ]]
for value in \
  "$SOURCE_SHA256" "$MANIFEST_SHA256" "$JOB_MANIFEST_SHA256" \
  "$AUTHORIZATION_SHA256" "$ATTEMPT_CLAIM_SHA256" \
  "$INITIAL_LAUNCH_CLAIM_SHA256" "$COST_GUARD_SHA256"; do
  [[ "$value" =~ ^[0-9a-f]{64}$ ]]
done
[[ "$ATTEMPT_INDEX" == 0 || "$ATTEMPT_INDEX" == 1 ]]
if [[ "$ATTEMPT_INDEX" == 0 ]]; then
  [[ "$INITIAL_LAUNCH_RESULT_SHA256" == none ]]
else
  [[ "$INITIAL_LAUNCH_RESULT_SHA256" =~ ^[0-9a-f]{64}$ ]]
fi
[[ "$MAX_RUNTIME_SECONDS" == 3900 ]]
[[ "$SELF_DELETE" == 1 ]]

FULL100_PREFIX="gs://$BUCKET/runs/$RUN_NAME/full100"
[[ "$SOURCE_URI" == "$FULL100_PREFIX/source/ofc_regular_hu_m31_t3_step6d_full100_v1_source.zip" ]]
[[ "$MANIFEST_URI" == "$FULL100_PREFIX/manifest.json" ]]
[[ "$JOB_MANIFEST_URI" == "$FULL100_PREFIX/source/jobs/$JOB_ID.json" ]]
[[ "$AUTHORIZATION_URI" == "$FULL100_PREFIX/source/launch_authorization.json" ]]
[[ "$RESULT_PREFIX" == "$FULL100_PREFIX/results/jobs/$JOB_ID" ]]
[[ "$PROGRESS_PREFIX" == "$FULL100_PREFIX/progress/jobs/$JOB_ID" ]]
if [[ "$ATTEMPT_INDEX" == 0 ]]; then
  [[ "$ATTEMPT_CLAIM_URI" == "$FULL100_PREFIX/control/launch_claim.json" ]]
else
  [[ "$ATTEMPT_CLAIM_URI" == "$FULL100_PREFIX/resume/resume_claim.json" ]]
fi
RESULT="$STATE/$JOB_ID"
VALIDATED_INDEX_FILE="$STATE/$JOB_ID.validated_hand_indices.json"

start_watchdog() {
  (
    sleep "$MAX_RUNTIME_SECONDS"
    echo "full100 watchdog expired after ${MAX_RUNTIME_SECONDS}s" >&2
    kill -TERM "$MAIN_PID" >/dev/null 2>&1 || true
    sleep 5
    sync_failure_progress
    upload_log
    delete_instance_or_shutdown
  ) &
  WATCHDOG_PID=$!
}
start_watchdog

printf '%s\n' \
  'deb [check-valid-until=no] http://snapshot.debian.org/archive/debian/20260609T000000Z bookworm main' \
  'deb [check-valid-until=no] http://snapshot.debian.org/archive/debian-security/20260609T000000Z bookworm-security main' \
  | sudo tee /etc/apt/sources.list >/dev/null
sudo apt-get -o Acquire::Check-Valid-Until=false update -y
sudo apt-get -o Acquire::Check-Valid-Until=false install -y \
  python3 python3-venv unzip libgomp1 ca-certificates

gcloud storage cp "$SOURCE_URI" /tmp/source.zip --project "$PROJECT_ID" \
  >/dev/null
gcloud storage cp "$MANIFEST_URI" /tmp/manifest.json --project "$PROJECT_ID" \
  >/dev/null
gcloud storage cp "$JOB_MANIFEST_URI" /tmp/job.json --project "$PROJECT_ID" \
  >/dev/null
gcloud storage cp "$AUTHORIZATION_URI" /tmp/authorization.json \
  --project "$PROJECT_ID" >/dev/null
gcloud storage cp "$ATTEMPT_CLAIM_URI" /tmp/attempt_claim.json \
  --project "$PROJECT_ID" >/dev/null
[[ "$(sha /tmp/source.zip)" == "$SOURCE_SHA256" ]]
[[ "$(sha /tmp/manifest.json)" == "$MANIFEST_SHA256" ]]
[[ "$(sha /tmp/job.json)" == "$JOB_MANIFEST_SHA256" ]]
[[ "$(sha /tmp/authorization.json)" == "$AUTHORIZATION_SHA256" ]]
[[ "$(sha /tmp/attempt_claim.json)" == "$ATTEMPT_CLAIM_SHA256" ]]

# A performance-lock archive contains already-materialized one-shot roots.
# Opening the zip with `unzip` would read/copy those root bytes, so the lock
# lineage is validated first while root members remain unopened.  The
# repeatable development package intentionally takes the legacy no-op branch.
PACKAGE_PHASE="$(python3 - /tmp/source.zip /tmp/manifest.json \
  /tmp/authorization.json /tmp/job.json "$JOB_ID" \
  "$JOB_MANIFEST_SHA256" <<'PY'
import hashlib
import json
import pathlib
import sys
import zipfile


LOCK_PACKAGE_SCHEMA = "hu_m31_t3_step6d_performance_lock_spot_package_v1"
DEVELOPMENT_PACKAGE_SCHEMA = "hu_m31_t3_step6d_full100_spot_package_v1"
DEVELOPMENT_DONE_SCHEMA = (
    "hu_m31_t3_step6d_candidate02_performance_source_shard_done_v1"
)
LOCK_AUTHORIZATION_SCHEMA = (
    "hu_m31_t3_step6d_performance_lock_launch_authorization_v1"
)
LOCK_GLOBAL_SPOT_CLAIM_SCHEMA = (
    "hu_m31_t3_step6d_performance_lock_global_spot_claim_v1"
)
LOCK_SPECS = (
    {
        "phase": "lock",
        "plan_schema": (
            "hu_m31_t3_step6d_candidate02_performance_lock_precontent_plan_v1"
        ),
        "plan_scope": "performance_lock",
        "plan_sha256": (
            "d2c9985b574839cf7e1515c12ebffc813ac6b4fb4e6494ed8d1288ef582a4b6b"
        ),
        "run_digest": (
            "e73c2b06279c1f1e91c38b2887ee465acf85f8f1072afef636512283252c34a4"
        ),
        "run_contract_schema": (
            "hu_m31_t3_step6d_candidate02_performance_lock_run_contract_v1"
        ),
        "candidate_variant": "candidate02_compact_scorer_performance_lock",
        "runner_schedule": "performance_lock",
        "step6d_run_id": "hu_m31_t3_step6d_candidate02_performance_lock_v1",
        "shard_manifest_schema": (
            "hu_m31_t3_step6d_performance_shard_manifest_v2"
        ),
        "done_schema": (
            "hu_m31_t3_step6d_candidate02_performance_lock_source_shard_done_v1"
        ),
        "global_spot_claim_schema": (
            "hu_m31_t3_step6d_performance_lock_global_spot_claim_v1"
        ),
        "global_spot_claim_status": (
            "global_one_shot_spot_identity_claimed_before_authorization"
        ),
        "preauthorize_smoke_receipt_required": False,
        "claim_schema": "hu_m31_t3_step6d_performance_lock_global_claim_v1",
        "claim_status": (
            "global_one_shot_claim_persisted_before_lock_root_touch_"
            "crash_consumes_claim"
        ),
        "materialization_schema": (
            "hu_m31_t3_step6d_performance_lock_root_materialization_v1"
        ),
        "materialization_status": (
            "all_100_lock_roots_materialized_same_identity"
        ),
        "seal_schema": "hu_m31_t3_step6d_performance_lock_root_seal_v1",
        "seal_status": "sealed_100_disjoint_hidden_safe_performance_lock_roots",
    },
    {
        "phase": "lock_rearm1",
        "plan_schema": (
            "hu_m31_t3_step6d_candidate02_performance_lock_rearm1_precontent_plan_v1"
        ),
        "plan_scope": "performance_lock_rearm1_fresh_roots_only",
        "plan_sha256": (
            "3b8a4230531f0d81c5320b2b0d051878113ac57a38ad97fdb0b1d89e0f17a886"
        ),
        "run_digest": (
            "f02e8845401eef92ba617f2c832204bdc74c20aa9e91810c52bf99687c4886b5"
        ),
        "run_contract_schema": (
            "hu_m31_t3_step6d_candidate02_performance_lock_recovery_run_contract_v2"
        ),
        "candidate_variant": (
            "candidate02_compact_scorer_performance_lock_recovery_v2"
        ),
        "runner_schedule": "performance_lock_recovery_v2",
        "step6d_run_id": (
            "hu_m31_t3_step6d_candidate02_performance_lock_recovery_v2"
        ),
        "shard_manifest_schema": (
            "hu_m31_t3_step6d_candidate02_"
            "performance_lock_recovery_shard_manifest_v2"
        ),
        "done_schema": (
            "hu_m31_t3_step6d_candidate02_"
            "performance_lock_recovery_source_shard_done_v2"
        ),
        "global_spot_claim_schema": (
            "hu_m31_t3_step6d_performance_lock_global_spot_claim_v1"
        ),
        "global_spot_claim_status": (
            "global_one_shot_spot_identity_claimed_before_authorization"
        ),
        "preauthorize_smoke_receipt_required": False,
        "claim_schema": (
            "hu_m31_t3_step6d_performance_lock_rearm1_global_claim_v2"
        ),
        "claim_status": (
            "global_rearm1_claim_persisted_before_new_root_touch_"
            "crash_consumes_claim"
        ),
        "materialization_schema": (
            "hu_m31_t3_step6d_performance_lock_rearm1_root_materialization_v2"
        ),
        "materialization_status": (
            "all_100_rearm1_roots_materialized_exact_claimed_identity"
        ),
        "seal_schema": "hu_m31_t3_step6d_performance_lock_rearm1_root_seal_v2",
        "seal_status": (
            "sealed_100_fresh_disjoint_hidden_safe_performance_lock_rearm1_roots"
        ),
    },
    {
        "phase": "lock_rearm2",
        "plan_schema": (
            "hu_m31_t3_step6d_candidate02_performance_lock_rearm2_precontent_plan_v1"
        ),
        "plan_scope": "performance_lock_rearm2_fresh_roots_only",
        "plan_sha256": (
            "8e67f3443208b5fddea20eb574f6ea760d8e9d7f518180f906944103796569d5"
        ),
        "run_digest": (
            "39bc01820c4dd690f3e971112dca181b45028f67a45893ea0708391ca06280a5"
        ),
        "run_contract_schema": (
            "hu_m31_t3_step6d_candidate02_performance_lock_recovery_run_contract_v3"
        ),
        "candidate_variant": (
            "candidate02_compact_scorer_performance_lock_recovery_v3"
        ),
        "runner_schedule": "performance_lock_recovery_v3",
        "step6d_run_id": (
            "hu_m31_t3_step6d_candidate02_performance_lock_recovery_v3"
        ),
        "shard_manifest_schema": (
            "hu_m31_t3_step6d_candidate02_"
            "performance_lock_recovery_shard_manifest_v3"
        ),
        "done_schema": (
            "hu_m31_t3_step6d_candidate02_"
            "performance_lock_recovery_source_shard_done_v3"
        ),
        "global_spot_claim_schema": (
            "hu_m31_t3_step6d_performance_lock_rearm2_global_spot_claim_v1"
        ),
        "global_spot_claim_status": (
            "global_rearm2_one_shot_spot_identity_claimed_after_exhaustive_"
            "smoke_before_authorization"
        ),
        "preauthorize_smoke_receipt_required": True,
        "claim_schema": (
            "hu_m31_t3_step6d_performance_lock_rearm2_global_claim_v1"
        ),
        "claim_status": (
            "global_rearm2_claim_persisted_before_new_root_touch_"
            "crash_consumes_claim"
        ),
        "materialization_schema": (
            "hu_m31_t3_step6d_performance_lock_rearm2_root_materialization_v1"
        ),
        "materialization_status": (
            "all_100_rearm2_roots_materialized_exact_claimed_identity"
        ),
        "seal_schema": "hu_m31_t3_step6d_performance_lock_rearm2_root_seal_v1",
        "seal_status": (
            "sealed_100_fresh_disjoint_hidden_safe_performance_lock_rearm2_roots"
        ),
    },
)
CURRENT_SHA256 = (
    "d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3"
)
PLAN_MEMBER = "frozen/full100_plan.json"
CLAIM_MEMBER = "frozen/performance_lock_open_claim.json"
SEAL_MEMBER = "frozen/performance_lock_root_seal.json"
MATERIALIZATION_MEMBER = "frozen/performance_lock_materialization.json"
ROOT_PREFIX = "frozen/full100_roots/"
LOCK_JOB_IDS = [
    f"{role}-shard-{index:02d}"
    for role in ("candidate", "reference")
    for index in range(10)
]
LOCK_GLOBAL_SPOT_CLAIM_KEYS = {
    "schema",
    "status",
    "global_root_claim_path",
    "global_root_claim_sha256",
    "lock_output_directory",
    "package_run_directory",
    "run_name",
    "package_manifest_sha256",
    "source_sha256",
    "startup_sha256",
    "precontent_plan_sha256",
    "root_seal_sha256",
    "run_contract_digest",
    "authorized_job_ids",
    "max_initial_jobs",
    "max_resume_attempts",
    "alternate_package_authorization_allowed",
    "quality_pilot_authorized",
    "training_eligible",
    "current_profile_changed",
    "runtime_policy_activated",
    "claimed_unix_ns",
}


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def canonical(value):
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def load_canonical_file(path, label):
    raw = pathlib.Path(path).read_bytes()
    value = json.loads(raw)
    require(
        isinstance(value, dict) and raw == canonical(value),
        f"{label} is not canonical",
    )
    return value


def read_control(archive, relative, entries, label):
    # This helper is deliberately never called for ROOT_PREFIX members.
    require(not relative.startswith(ROOT_PREFIX), f"{label} attempted root read")
    raw = archive.read(relative)
    value = json.loads(raw)
    require(
        isinstance(value, dict) and raw == canonical(value),
        f"{label} is not canonical",
    )
    expected = entries.get(relative)
    require(isinstance(expected, dict), f"{label} entry is absent")
    require(
        expected.get("sha256") == hashlib.sha256(raw).hexdigest()
        and expected.get("bytes") == len(raw),
        f"{label} entry hash changed",
    )
    return value


def validate_global_spot_claim(
    value,
    *,
    root_claim,
    root_seal,
    manifest,
    manifest_sha,
    lock_spec,
    authorization,
):
    require(isinstance(value, dict), "performance-lock global Spot claim is absent")
    expected_keys = set(LOCK_GLOBAL_SPOT_CLAIM_KEYS)
    if lock_spec["preauthorize_smoke_receipt_required"]:
        expected_keys.add("preauthorize_smoke_receipt_sha256")
    require(
        set(value) == expected_keys,
        "performance-lock global Spot claim fields changed",
    )
    require(
        value.get("schema") == lock_spec["global_spot_claim_schema"]
        and value.get("status") == lock_spec["global_spot_claim_status"],
        "performance-lock global Spot claim schema/status changed",
    )
    receipt_sha256 = value.get("preauthorize_smoke_receipt_sha256")
    if lock_spec["preauthorize_smoke_receipt_required"]:
        require(
            isinstance(receipt_sha256, str)
            and len(receipt_sha256) == 64
            and set(receipt_sha256) <= set("0123456789abcdef")
            and authorization.get("preauthorize_smoke_receipt_sha256")
            == receipt_sha256,
            "actual-package smoke receipt binding changed",
        )
    else:
        require(
            "preauthorize_smoke_receipt_sha256" not in value
            and "preauthorize_smoke_receipt_sha256" not in authorization,
            "legacy lock unexpectedly binds a smoke receipt",
        )
    package_directory = value.get("package_run_directory")
    require(
        isinstance(package_directory, str)
        and bool(package_directory)
        and package_directory == package_directory.strip()
        and package_directory != root_claim.get("lock_output_directory")
        and package_directory != root_claim.get("global_claim_path"),
        "performance-lock global Spot package directory changed",
    )
    require(
        value.get("global_root_claim_path") == root_claim.get("global_claim_path")
        and value.get("global_root_claim_sha256") == digest(root_claim)
        and value.get("lock_output_directory")
        == root_claim.get("lock_output_directory"),
        "performance-lock global root/Spot claim chain changed",
    )
    require(
        value.get("run_name") == manifest.get("run_name")
        and value.get("package_manifest_sha256") == manifest_sha
        and value.get("source_sha256") == manifest.get("source_sha256")
        and value.get("startup_sha256") == manifest.get("startup_sha256")
        and value.get("precontent_plan_sha256") == manifest.get("plan_sha256")
        and value.get("root_seal_sha256") == digest(root_seal)
        and value.get("run_contract_digest")
        == manifest.get("run_contract_digest"),
        "performance-lock global Spot package binding changed",
    )
    require(
        value.get("authorized_job_ids") == LOCK_JOB_IDS
        and value.get("max_initial_jobs") == 20
        and value.get("max_resume_attempts") == 1,
        "performance-lock global Spot attempt boundary changed",
    )
    for key in (
        "alternate_package_authorization_allowed",
        "quality_pilot_authorized",
        "training_eligible",
        "current_profile_changed",
        "runtime_policy_activated",
    ):
        require(
            value.get(key) is False,
            f"forbidden performance-lock global Spot flag: {key}",
        )
    claimed_unix_ns = value.get("claimed_unix_ns")
    require(
        isinstance(claimed_unix_ns, int)
        and not isinstance(claimed_unix_ns, bool)
        and claimed_unix_ns > 0,
        "performance-lock global Spot claim timestamp changed",
    )


source_path, manifest_path, authorization_path = map(pathlib.Path, sys.argv[1:4])
legacy_control_only_invocation = len(sys.argv) == 4
require(
    legacy_control_only_invocation or len(sys.argv) == 7,
    "pre-content verifier argument contract changed",
)
job_path = pathlib.Path(sys.argv[4]) if not legacy_control_only_invocation else None
job_id = sys.argv[5] if not legacy_control_only_invocation else None
job_sha256 = sys.argv[6] if not legacy_control_only_invocation else None
manifest = load_canonical_file(manifest_path, "package manifest")
authorization = load_canonical_file(authorization_path, "launch authorization")
schema = manifest.get("schema")
if schema == DEVELOPMENT_PACKAGE_SCHEMA:
    print(
        "development"
        if legacy_control_only_invocation
        else f"development|{DEVELOPMENT_DONE_SCHEMA}"
    )
    raise SystemExit(0)
require(schema == LOCK_PACKAGE_SCHEMA, "unrecognized full100 package schema")
require(
    manifest.get("status")
    == "immutable_performance_lock_package_ready_not_authorized",
    "performance-lock package status changed",
)
require(
    authorization.get("schema") == LOCK_AUTHORIZATION_SCHEMA
    and authorization.get("status")
    == "explicit_one_shot_performance_lock_spot_authorization",
    "performance-lock authorization schema/status changed",
)
require(
    manifest.get("source_sha256")
    == authorization.get("source_sha256")
    and authorization.get("package_manifest_sha256")
    == hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    "performance-lock package/source authorization chain changed",
)
matching_lock_specs = [
    spec
    for spec in LOCK_SPECS
    if manifest.get("plan_sha256") == spec["plan_sha256"]
    and authorization.get("plan_sha256") == spec["plan_sha256"]
    and manifest.get("run_contract_digest") == spec["run_digest"]
    and authorization.get("run_contract_digest") == spec["run_digest"]
]
require(
    len(matching_lock_specs) == 1,
    "performance-lock plan/run identity is not an allowed frozen lock",
)
lock_spec = matching_lock_specs[0]
LOCK_PLAN_SCHEMA = lock_spec["plan_schema"]
LOCK_PLAN_SCOPE = lock_spec["plan_scope"]
LOCK_PLAN_SHA256 = lock_spec["plan_sha256"]
LOCK_RUN_DIGEST = lock_spec["run_digest"]
LOCK_CLAIM_SCHEMA = lock_spec["claim_schema"]
LOCK_CLAIM_STATUS = lock_spec["claim_status"]
LOCK_MATERIALIZATION_SCHEMA = lock_spec["materialization_schema"]
LOCK_MATERIALIZATION_STATUS = lock_spec["materialization_status"]
LOCK_SEAL_SCHEMA = lock_spec["seal_schema"]
LOCK_SEAL_STATUS = lock_spec["seal_status"]
LOCK_PHASE = lock_spec["phase"]
LOCK_RUN_CONTRACT_SCHEMA = lock_spec["run_contract_schema"]
LOCK_CANDIDATE_VARIANT = lock_spec["candidate_variant"]
LOCK_RUNNER_SCHEDULE = lock_spec["runner_schedule"]
LOCK_STEP6D_RUN_ID = lock_spec["step6d_run_id"]
LOCK_SHARD_MANIFEST_SCHEMA = lock_spec["shard_manifest_schema"]
LOCK_DONE_SCHEMA = lock_spec["done_schema"]
require(
    manifest.get("plan_sha256")
    == authorization.get("plan_sha256")
    == LOCK_PLAN_SHA256,
    "performance-lock plan binding changed",
)
require(
    manifest.get("run_contract_digest")
    == authorization.get("run_contract_digest")
    == LOCK_RUN_DIGEST,
    "performance-lock run binding changed",
)
run_contract = manifest.get("run_contract")
require(
    legacy_control_only_invocation
    or (
        isinstance(run_contract, dict)
        and digest(run_contract) == LOCK_RUN_DIGEST
        and run_contract.get("schema") == LOCK_RUN_CONTRACT_SCHEMA
        and run_contract.get("candidate_variant") == LOCK_CANDIDATE_VARIANT
        and run_contract.get("schedule") == LOCK_RUNNER_SCHEDULE
        and run_contract.get("step6d_run_id") == LOCK_STEP6D_RUN_ID
    ),
    "performance-lock producer run constants changed",
)
require(
    authorization.get("performance_development_only") is False
    and authorization.get("spot_execution_authorized") is True
    and authorization.get("performance_lock_authorized") is True
    and authorization.get("quality_pilot_authorized") is False,
    "performance-lock authorization scope changed",
)
for key in (
    "training_eligible",
    "current_profile_changed",
    "named_profile_added",
    "runtime_policy_activated",
):
    require(authorization.get(key) is False, f"forbidden lock authorization: {key}")
for key in (
    "spot_execution_authorized",
    "performance_lock_authorized",
    "quality_pilot_authorized",
    "training_eligible",
    "current_profile_changed",
    "named_profile_added",
    "runtime_policy_activated",
    "m31_complete",
    "gcloud_invoked",
):
    require(manifest.get(key) is False, f"forbidden lock package flag: {key}")

entries = manifest.get("source_entries")
require(isinstance(entries, dict), "performance-lock source entries changed")
with zipfile.ZipFile(source_path) as archive:
    infos = archive.infolist()
    names = [info.filename for info in infos]
    require(len(names) == len(set(names)), "performance-lock zip repeats a member")
    require(set(names) == set(entries), "performance-lock zip entries changed")
    for info in infos:
        pure = pathlib.PurePosixPath(info.filename)
        require(
            not pure.is_absolute() and ".." not in pure.parts,
            "performance-lock zip contains an unsafe member",
        )
        record = entries[info.filename]
        require(
            isinstance(record, dict) and record.get("bytes") == info.file_size,
            "performance-lock zip member size changed",
        )

    # PRE-CONTENT ORDERING: only the four control artifacts are opened here.
    # Root members remain unopened until all claim/seal/materialization checks
    # below have passed and this process returns "lock".
    plan_raw = archive.read(PLAN_MEMBER)
    plan = json.loads(plan_raw)
    require(
        isinstance(plan, dict)
        and plan_raw == canonical(plan)
        and hashlib.sha256(plan_raw).hexdigest() == LOCK_PLAN_SHA256,
        "performance-lock plan bytes changed",
    )
    require(
        entries[PLAN_MEMBER].get("sha256") == LOCK_PLAN_SHA256
        and entries[PLAN_MEMBER].get("bytes") == len(plan_raw),
        "performance-lock plan entry changed",
    )
    claim = read_control(archive, CLAIM_MEMBER, entries, "performance-lock claim")
    seal = read_control(archive, SEAL_MEMBER, entries, "performance-lock seal")
    materialization = read_control(
        archive,
        MATERIALIZATION_MEMBER,
        entries,
        "performance-lock materialization",
    )

    require(
        plan.get("schema") == LOCK_PLAN_SCHEMA
        and plan.get("scope") == LOCK_PLAN_SCOPE
        and (
            legacy_control_only_invocation
            or plan.get("run_contract") == run_contract
        )
        and plan.get("run_contract_digest") == LOCK_RUN_DIGEST
        and plan.get("root_content_opened") is False
        and plan.get("cloud_started") is False
        and plan.get("training_eligible") is False
        and plan.get("current_profile_changed") is False
        and plan.get("runtime_policy_activated") is False,
        "performance-lock precontent plan changed",
    )
    if not legacy_control_only_invocation:
        job = load_canonical_file(job_path, "packaged job manifest")
        require(
            hashlib.sha256(job_path.read_bytes()).hexdigest() == job_sha256,
            "packaged job manifest hash changed",
        )
        require(job_id in LOCK_JOB_IDS, "packaged job id changed")
        require(
            set(job)
            == {
                "schema",
                "run_contract",
                "run_contract_digest",
                "source_role",
                "work_hand_indices",
            },
            "packaged job manifest fields changed",
        )
        require(
            job.get("schema") == LOCK_SHARD_MANIFEST_SCHEMA
            and job.get("run_contract") == run_contract
            and job.get("run_contract_digest") == LOCK_RUN_DIGEST,
            "packaged job producer contract changed",
        )
        role = job.get("source_role")
        work_indices = job.get("work_hand_indices")
        require(role in ("candidate", "reference"), "packaged job role changed")
        require(
            isinstance(work_indices, list)
            and len(work_indices) == 10
            and work_indices == sorted(work_indices)
            and len(set(work_indices)) == 10
            and all(
                type(index) is int and 0 <= index < 100
                for index in work_indices
            ),
            "packaged job work shard changed",
        )
        plan_jobs = [
            row
            for row in plan.get("jobs", [])
            if isinstance(row, dict) and row.get("job_id") == job_id
        ]
        require(len(plan_jobs) == 1, "packaged job absent from frozen plan")
        plan_job = plan_jobs[0]
        require(
            plan_job.get("source_role") == role
            and plan_job.get("work_hand_indices") == work_indices
            and plan_job.get("shard_manifest_sha256") == job_sha256
            and job_id == f"{role}-shard-{plan_job.get('shard_index'):02d}",
            "packaged job/frozen plan mapping changed",
        )
        job_records = manifest.get("job_manifests")
        require(
            isinstance(job_records, list) and len(job_records) == 20,
            "package job records changed",
        )
        matching_records = [
            row
            for row in job_records
            if isinstance(row, dict) and row.get("job_id") == job_id
        ]
        require(len(matching_records) == 1, "package job record changed")
        job_record = matching_records[0]
        require(
            job_record.get("source_role") == role
            and job_record.get("work_hand_indices") == work_indices
            and job_record.get("shard_index") == plan_job.get("shard_index")
            and job_record.get("path") == f"jobs/{job_id}.json"
            and job_record.get("output_prefix") == f"jobs/{job_id}"
            and job_record.get("sha256") == job_sha256
            and job_record.get("bytes") == job_path.stat().st_size,
            "package job record binding changed",
        )
    require(
        claim.get("schema") == LOCK_CLAIM_SCHEMA
        and claim.get("status") == LOCK_CLAIM_STATUS
        and claim.get("precontent_plan", {}).get("sha256") == LOCK_PLAN_SHA256
        and claim.get("lock_run_contract_digest") == LOCK_RUN_DIGEST
        and claim.get("ai_profiles_current", {}).get("sha256") == CURRENT_SHA256,
        "performance-lock open claim changed",
    )
    restrictions = claim.get("restrictions", {})
    for key in (
        "alternate_seed_allowed",
        "reseed_allowed",
        "cloud_authorized",
        "training_authorized",
        "promotion_authorized",
        "current_profile_resolution_allowed",
        "runtime_activation_allowed",
        "opponent_" + "private_discards_allowed",
    ):
        require(restrictions.get(key) is False, f"lock restriction widened: {key}")
    require(
        materialization.get("schema") == LOCK_MATERIALIZATION_SCHEMA
        and materialization.get("status") == LOCK_MATERIALIZATION_STATUS
        and materialization.get("global_claim_sha256") == digest(claim)
        and materialization.get("plan_sha256") == LOCK_PLAN_SHA256
        and materialization.get("run_contract_digest") == LOCK_RUN_DIGEST
        and materialization.get("hand_indices") == list(range(100))
        and materialization.get("root_count") == 100
        and materialization.get("same_identity_resume_only") is True
        and materialization.get("reseeded") is False
        and materialization.get("training_eligible") is False
        and materialization.get("current_profile_changed") is False,
        "performance-lock materialization changed",
    )
    root_hashes = materialization.get("root_artifact_sha256")
    require(
        isinstance(root_hashes, list)
        and len(root_hashes) == 100
        and len(set(root_hashes)) == 100
        and all(
            isinstance(value, str)
            and len(value) == 64
            and set(value) <= set("0123456789abcdef")
            for value in root_hashes
        )
        and materialization.get("aggregate_root_sha256") == digest(root_hashes),
        "performance-lock materialization root aggregate changed",
    )
    require(
        seal.get("schema") == LOCK_SEAL_SCHEMA
        and seal.get("status") == LOCK_SEAL_STATUS
        and seal.get("global_claim_sha256") == digest(claim)
        and seal.get("materialization_sha256") == digest(materialization)
        and seal.get("plan_sha256") == LOCK_PLAN_SHA256
        and seal.get("run_contract_digest") == LOCK_RUN_DIGEST
        and seal.get("hand_indices") == list(range(100))
        and seal.get("root_count") == 100
        and seal.get("observation_count") == 200
        and seal.get("root_artifact_sha256") == root_hashes
        and seal.get("aggregate_root_sha256")
        == materialization.get("aggregate_root_sha256")
        and seal.get("training_eligible") is False
        and seal.get("current_profile_changed") is False
        and seal.get("named_profile_added") is False
        and seal.get("runtime_policy_activated") is False,
        "performance-lock root seal changed",
    )
    comparison = seal.get("development_comparison", {})
    require(
        comparison.get("lock_fingerprint_overlap_count") == 0
        and comparison.get("lock_root_hash_overlap_count") == 0
        and comparison.get("lock_seed_overlap_count") == 0,
        "performance-lock/development overlap changed",
    )
    validate_global_spot_claim(
        authorization.get("global_spot_claim"),
        root_claim=claim,
        root_seal=seal,
        manifest=manifest,
        manifest_sha=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        lock_spec=lock_spec,
        authorization=authorization,
    )
    qualification = manifest.get("tail_qualification", {})
    require(
        qualification.get("summary_sha256")
        == "f587df6d037313e2111a5cbc3d474370106f6a341ef4ec130cee7516192e668f"
        and qualification.get("validation_sha256")
        == "da252b3cabfc361d89de2f2fe08931fe61302287b69b82bf9f812a4432baf3eb"
        and qualification.get("scientific_merge_sha256")
        == "29e6b3db5b372f9808b681fb70246368452eecf5c6bbbca2519b2882430369f7"
        and qualification.get("receive_receipt_sha256")
        == "2ba1b7434cda012d8a230109e9c37db6d9e9b41587e1138e79790a177df85cf6"
        and qualification.get("development_run_contract_digest")
        == "92a87f1544a73104d5256db39b022094c8c7f7b11f77bd19dcf1ab1442581ebd"
        and qualification.get("all_gates_passed") is True
        and qualification.get("performance_candidate_frozen") is True
        and qualification.get("performance_lock_authorized") is True
        and qualification.get("open_claim_package_path") == CLAIM_MEMBER
        and qualification.get("open_claim_sha256") == digest(claim)
        and qualification.get("root_seal_package_path") == SEAL_MEMBER
        and qualification.get("root_seal_sha256") == digest(seal)
        and qualification.get("materialization_package_path")
        == MATERIALIZATION_MEMBER
        and qualification.get("materialization_sha256") == digest(materialization)
        and qualification.get("lock_root_aggregate_sha256")
        == seal.get("aggregate_root_sha256")
        and qualification.get("lock_development_overlap_count") == 0,
        "performance-lock qualification changed",
    )
    expected_root_members = [
        f"{ROOT_PREFIX}hand_{index:03d}.json" for index in range(100)
    ]
    require(
        sorted(name for name in names if name.startswith(ROOT_PREFIX))
        == expected_root_members,
        "performance-lock root member set changed",
    )
    # Do not archive.read() a root here.  Its immutable predeclared byte hash
    # must match the seal list; the normal post-claim validation opens it later.
    for index, relative in enumerate(expected_root_members):
        require(
            entries[relative].get("sha256") == root_hashes[index],
            f"performance-lock root entry hash changed: {index}",
        )

print(
    LOCK_PHASE
    if legacy_control_only_invocation
    else f"{LOCK_PHASE}|{LOCK_DONE_SCHEMA}"
)
PY
)"
IFS='|' read -r PACKAGE_PHASE PRECONTENT_DONE_SCHEMA <<<"$PACKAGE_PHASE"
[[ "$PACKAGE_PHASE" == development || "$PACKAGE_PHASE" == lock || \
   "$PACKAGE_PHASE" == lock_rearm1 || "$PACKAGE_PHASE" == lock_rearm2 ]]
[[ -n "$PRECONTENT_DONE_SCHEMA" ]]

rm -rf "$WORK" "$VENV"
mkdir -p "$WORK"
unzip -q /tmp/source.zip -d "$WORK"

mapfile -t JOB_FIELDS < <(python3 - "$WORK" /tmp/source.zip \
  /tmp/manifest.json /tmp/job.json /tmp/authorization.json \
  /tmp/attempt_claim.json "$RUN_NAME" "$JOB_ID" "$INSTANCE_NAME" "$ZONE" \
  "$PROJECT_ID" "$BUCKET" "$SOURCE_SHA256" "$MANIFEST_SHA256" \
  "$JOB_MANIFEST_SHA256" "$AUTHORIZATION_SHA256" "$ATTEMPT_INDEX" \
  "$ATTEMPT_CLAIM_SHA256" "$INITIAL_LAUNCH_CLAIM_SHA256" \
  "$INITIAL_LAUNCH_RESULT_SHA256" "$RESULT_PREFIX" "$PROGRESS_PREFIX" \
  "$COST_GUARD_SHA256" "$MAX_RUNTIME_SECONDS" "$SELF_DELETE" \
  "$EXECUTED_STARTUP_SHA256" "$PACKAGE_PHASE" <<'PY'
import hashlib
import json
import pathlib
import sys


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def canonical(value):
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def load_canonical(path, label):
    raw = pathlib.Path(path).read_bytes()
    value = json.loads(raw)
    require(isinstance(value, dict) and raw == canonical(value), f"{label} is not canonical")
    return value


GLOBAL_SPOT_CLAIM_KEYS = {
    "schema",
    "status",
    "global_root_claim_path",
    "global_root_claim_sha256",
    "lock_output_directory",
    "package_run_directory",
    "run_name",
    "package_manifest_sha256",
    "source_sha256",
    "startup_sha256",
    "precontent_plan_sha256",
    "root_seal_sha256",
    "run_contract_digest",
    "authorized_job_ids",
    "max_initial_jobs",
    "max_resume_attempts",
    "alternate_package_authorization_allowed",
    "quality_pilot_authorized",
    "training_eligible",
    "current_profile_changed",
    "runtime_policy_activated",
    "claimed_unix_ns",
}


def validate_global_spot_claim(
    value,
    *,
    root_claim,
    root_seal,
    manifest,
    manifest_sha,
    job_ids,
    expected_schema="hu_m31_t3_step6d_performance_lock_global_spot_claim_v1",
    expected_status="global_one_shot_spot_identity_claimed_before_authorization",
    receipt_required=False,
    authorization=None,
):
    authorization = {} if authorization is None else authorization
    require(isinstance(value, dict), "performance-lock global Spot claim is absent")
    expected_keys = set(GLOBAL_SPOT_CLAIM_KEYS)
    if receipt_required:
        expected_keys.add("preauthorize_smoke_receipt_sha256")
    require(
        set(value) == expected_keys,
        "performance-lock global Spot claim fields changed",
    )
    require(
        value.get("schema") == expected_schema
        and value.get("status") == expected_status,
        "performance-lock global Spot claim schema/status changed",
    )
    receipt_sha256 = value.get("preauthorize_smoke_receipt_sha256")
    if receipt_required:
        require(
            isinstance(receipt_sha256, str)
            and len(receipt_sha256) == 64
            and set(receipt_sha256) <= set("0123456789abcdef")
            and authorization.get("preauthorize_smoke_receipt_sha256")
            == receipt_sha256,
            "actual-package smoke receipt binding changed",
        )
    else:
        require(
            "preauthorize_smoke_receipt_sha256" not in value
            and "preauthorize_smoke_receipt_sha256" not in authorization,
            "legacy lock unexpectedly binds a smoke receipt",
        )
    package_directory = value.get("package_run_directory")
    require(
        isinstance(package_directory, str)
        and bool(package_directory)
        and package_directory == package_directory.strip()
        and package_directory != root_claim.get("lock_output_directory")
        and package_directory != root_claim.get("global_claim_path"),
        "performance-lock global Spot package directory changed",
    )
    require(
        value.get("global_root_claim_path") == root_claim.get("global_claim_path")
        and value.get("global_root_claim_sha256") == digest(root_claim)
        and value.get("lock_output_directory")
        == root_claim.get("lock_output_directory"),
        "performance-lock global root/Spot claim chain changed",
    )
    require(
        value.get("run_name") == manifest.get("run_name")
        and value.get("package_manifest_sha256") == manifest_sha
        and value.get("source_sha256") == manifest.get("source_sha256")
        and value.get("startup_sha256") == manifest.get("startup_sha256")
        and value.get("precontent_plan_sha256") == manifest.get("plan_sha256")
        and value.get("root_seal_sha256") == digest(root_seal)
        and value.get("run_contract_digest")
        == manifest.get("run_contract_digest"),
        "performance-lock global Spot package binding changed",
    )
    require(
        value.get("authorized_job_ids") == job_ids
        and value.get("max_initial_jobs") == 20
        and value.get("max_resume_attempts") == 1,
        "performance-lock global Spot attempt boundary changed",
    )
    for key in (
        "alternate_package_authorization_allowed",
        "quality_pilot_authorized",
        "training_eligible",
        "current_profile_changed",
        "runtime_policy_activated",
    ):
        require(
            value.get(key) is False,
            f"forbidden performance-lock global Spot flag: {key}",
        )
    claimed_unix_ns = value.get("claimed_unix_ns")
    require(
        isinstance(claimed_unix_ns, int)
        and not isinstance(claimed_unix_ns, bool)
        and claimed_unix_ns > 0,
        "performance-lock global Spot claim timestamp changed",
    )


work = pathlib.Path(sys.argv[1]).resolve()
source_path = pathlib.Path(sys.argv[2])
manifest_path = pathlib.Path(sys.argv[3])
job_path = pathlib.Path(sys.argv[4])
authorization_path = pathlib.Path(sys.argv[5])
claim_path = pathlib.Path(sys.argv[6])
run_name, job_id, instance_name, zone, project, bucket = sys.argv[7:13]
source_sha, manifest_sha, job_sha, authorization_sha = sys.argv[13:17]
attempt_index = int(sys.argv[17])
claim_sha, initial_claim_sha, initial_result_sha = sys.argv[18:21]
result_prefix, progress_prefix = sys.argv[21:23]
cost_guard_sha = sys.argv[23]
max_runtime_seconds, self_delete = int(sys.argv[24]), int(sys.argv[25])
executed_startup_sha = sys.argv[26]
package_phase = sys.argv[27]

m = load_canonical(manifest_path, "package manifest")
j = load_canonical(job_path, "job manifest")
a = load_canonical(authorization_path, "launch authorization")
c = load_canonical(claim_path, "attempt claim")

manifest_keys = {
    "schema", "status", "run_name", "source_name", "source_sha256",
    "source_bytes", "startup_name", "startup_sha256", "plan_package_path",
    "plan_sha256", "tail_qualification", "run_contract",
    "run_contract_digest", "accepted_reference", "accepted_candidate",
    "feature_encoder", "image", "allocation", "launch_target", "cost_guard",
    "schedule", "job_manifests", "source_entries", "source_entry_count",
    "checkpoint", "heartbeat", "spot_execution_authorized",
    "performance_lock_authorized", "quality_pilot_authorized",
    "training_eligible", "current_profile_changed", "named_profile_added",
    "runtime_policy_activated", "m31_complete", "gcloud_invoked",
}
development_authorization_keys = {
    "schema", "status", "run_name", "package_manifest_sha256",
    "source_sha256", "startup_sha256", "plan_sha256",
    "tail_summary_sha256", "tail_validation_sha256",
    "run_contract_digest", "launch_target", "cost_guard",
    "authorized_job_ids", "logical_job_count",
    "performance_development_only", "spot_execution_authorized",
    "performance_lock_authorized", "quality_pilot_authorized",
    "training_eligible", "current_profile_changed", "named_profile_added",
    "runtime_policy_activated", "authorized_unix_seconds",
}
lock_mode = package_phase in ("lock", "lock_rearm1", "lock_rearm2")
rearm1_mode = package_phase == "lock_rearm1"
rearm2_mode = package_phase == "lock_rearm2"
require(
    package_phase in ("development", "lock", "lock_rearm1", "lock_rearm2"),
    "full100 package phase is not recognized",
)
authorization_keys = development_authorization_keys | (
    {"global_spot_claim"} if lock_mode else set()
)
if rearm2_mode:
    authorization_keys.add("preauthorize_smoke_receipt_sha256")
require(set(m) == manifest_keys, "full100 package manifest keys changed")
require(set(a) == authorization_keys, "full100 authorization keys changed")
if lock_mode:
    expected_package_schema = "hu_m31_t3_step6d_performance_lock_spot_package_v1"
    expected_package_status = (
        "immutable_performance_lock_package_ready_not_authorized"
    )
    expected_authorization_schema = (
        "hu_m31_t3_step6d_performance_lock_launch_authorization_v1"
    )
    expected_authorization_status = (
        "explicit_one_shot_performance_lock_spot_authorization"
    )
    if rearm2_mode:
        expected_plan_sha = (
            "8e67f3443208b5fddea20eb574f6ea760d8e9d7f518180f906944103796569d5"
        )
        expected_run_digest = (
            "39bc01820c4dd690f3e971112dca181b45028f67a45893ea0708391ca06280a5"
        )
        expected_plan_schema = (
            "hu_m31_t3_step6d_candidate02_performance_lock_rearm2_precontent_plan_v1"
        )
        expected_plan_scope = "performance_lock_rearm2_fresh_roots_only"
        expected_contract_schema = (
            "hu_m31_t3_step6d_candidate02_performance_lock_recovery_run_contract_v3"
        )
        expected_step6d_run_id = (
            "hu_m31_t3_step6d_candidate02_performance_lock_recovery_v3"
        )
        expected_candidate_variant = (
            "candidate02_compact_scorer_performance_lock_recovery_v3"
        )
        expected_runner_schedule = "performance_lock_recovery_v3"
        expected_shard_manifest_schema = (
            "hu_m31_t3_step6d_candidate02_"
            "performance_lock_recovery_shard_manifest_v3"
        )
        expected_done_schema = (
            "hu_m31_t3_step6d_candidate02_"
            "performance_lock_recovery_source_shard_done_v3"
        )
        expected_global_spot_claim_schema = (
            "hu_m31_t3_step6d_performance_lock_rearm2_global_spot_claim_v1"
        )
        expected_global_spot_claim_status = (
            "global_rearm2_one_shot_spot_identity_claimed_after_exhaustive_"
            "smoke_before_authorization"
        )
    elif rearm1_mode:
        expected_plan_sha = (
            "3b8a4230531f0d81c5320b2b0d051878113ac57a38ad97fdb0b1d89e0f17a886"
        )
        expected_run_digest = (
            "f02e8845401eef92ba617f2c832204bdc74c20aa9e91810c52bf99687c4886b5"
        )
        expected_plan_schema = (
            "hu_m31_t3_step6d_candidate02_performance_lock_rearm1_precontent_plan_v1"
        )
        expected_plan_scope = "performance_lock_rearm1_fresh_roots_only"
        expected_contract_schema = (
            "hu_m31_t3_step6d_candidate02_performance_lock_recovery_run_contract_v2"
        )
        expected_step6d_run_id = (
            "hu_m31_t3_step6d_candidate02_performance_lock_recovery_v2"
        )
        expected_candidate_variant = (
            "candidate02_compact_scorer_performance_lock_recovery_v2"
        )
        expected_runner_schedule = "performance_lock_recovery_v2"
        expected_shard_manifest_schema = (
            "hu_m31_t3_step6d_candidate02_"
            "performance_lock_recovery_shard_manifest_v2"
        )
        expected_done_schema = (
            "hu_m31_t3_step6d_candidate02_"
            "performance_lock_recovery_source_shard_done_v2"
        )
        expected_global_spot_claim_schema = (
            "hu_m31_t3_step6d_performance_lock_global_spot_claim_v1"
        )
        expected_global_spot_claim_status = (
            "global_one_shot_spot_identity_claimed_before_authorization"
        )
    else:
        expected_plan_sha = (
            "d2c9985b574839cf7e1515c12ebffc813ac6b4fb4e6494ed8d1288ef582a4b6b"
        )
        expected_run_digest = (
            "e73c2b06279c1f1e91c38b2887ee465acf85f8f1072afef636512283252c34a4"
        )
        expected_plan_schema = (
            "hu_m31_t3_step6d_candidate02_performance_lock_precontent_plan_v1"
        )
        expected_plan_scope = "performance_lock"
        expected_contract_schema = (
            "hu_m31_t3_step6d_candidate02_performance_lock_run_contract_v1"
        )
        expected_step6d_run_id = (
            "hu_m31_t3_step6d_candidate02_performance_lock_v1"
        )
        expected_candidate_variant = (
            "candidate02_compact_scorer_performance_lock"
        )
        expected_runner_schedule = "performance_lock"
        expected_shard_manifest_schema = (
            "hu_m31_t3_step6d_performance_shard_manifest_v2"
        )
        expected_done_schema = (
            "hu_m31_t3_step6d_candidate02_"
            "performance_lock_source_shard_done_v1"
        )
        expected_global_spot_claim_schema = (
            "hu_m31_t3_step6d_performance_lock_global_spot_claim_v1"
        )
        expected_global_spot_claim_status = (
            "global_one_shot_spot_identity_claimed_before_authorization"
        )
else:
    expected_package_schema = "hu_m31_t3_step6d_full100_spot_package_v1"
    expected_package_status = "immutable_full100_package_ready_not_authorized"
    expected_authorization_schema = (
        "hu_m31_t3_step6d_full100_launch_authorization_v1"
    )
    expected_authorization_status = (
        "explicit_full100_performance_development_spot_authorization"
    )
    expected_plan_sha = (
        "9ef14137b97db975bed683bcdd7e53b27414d2efab282c6f98390d7702944758"
    )
    expected_run_digest = (
        "92a87f1544a73104d5256db39b022094c8c7f7b11f77bd19dcf1ab1442581ebd"
    )
    expected_plan_schema = "hu_m31_t3_step6d_candidate02_full100_plan_v1"
    expected_shard_manifest_schema = (
        "hu_m31_t3_step6d_performance_shard_manifest_v2"
    )
    expected_done_schema = (
        "hu_m31_t3_step6d_candidate02_performance_source_shard_done_v1"
    )
require(m["schema"] == expected_package_schema, "bad package schema")
require(m["status"] == expected_package_status, "bad package status")
require(a["schema"] == expected_authorization_schema, "bad authorization schema")
require(a["status"] == expected_authorization_status, "bad authorization status")
require(m["run_name"] == a["run_name"] == run_name, "run identity changed")
require(m["source_name"] == "ofc_regular_hu_m31_t3_step6d_full100_v1_source.zip", "source name changed")
require(m["startup_name"] == "startup_hu_m31_t3_step6d_full100_v1.sh", "startup name changed")
require(m["source_sha256"] == a["source_sha256"] == source_sha, "source hash chain changed")
require(m["source_bytes"] == source_path.stat().st_size, "source bytes changed")
require(a["package_manifest_sha256"] == manifest_sha, "authorization package hash changed")
require(a["startup_sha256"] == m["startup_sha256"], "startup hash chain changed")
require(m["startup_sha256"] == executed_startup_sha, "executed startup bytes changed")
require(
    m["feature_encoder"] == {
        "package_path": "target/release/libofc_stage3_feature_encoder.so",
        "sha256": "82510e563ee290cd536e7aebe6bcf0efefac061af5488851f0a582bb4ef83411",
    },
    "feature encoder binding changed",
)
require(
    m["image"] == {
        "project": "debian-cloud",
        "name": "debian-12-bookworm-v20260609",
        "id": "1449487925682397051",
        "self_link": "https://www.googleapis.com/compute/v1/projects/debian-cloud/global/images/debian-12-bookworm-v20260609",
    },
    "image binding changed",
)
require(
    m["allocation"] == {
        "machine_type": "c4-standard-16",
        "process_count": 1,
        "rayon_threads_per_process": 16,
        "omp_threads": 1,
        "m3_batch_threads": 1,
    },
    "allocation binding changed",
)
require(m["plan_package_path"] == "frozen/full100_plan.json", "plan path changed")
require(
    m["plan_sha256"] == a["plan_sha256"] == expected_plan_sha,
    "full100 plan hash changed",
)
require(
    m["run_contract_digest"] == a["run_contract_digest"] == expected_run_digest,
    "full100 contract hash changed",
)
require(digest(m["run_contract"]) == m["run_contract_digest"], "package contract digest mismatch")
if lock_mode:
    require(
        m["run_contract"]["schema"] == expected_contract_schema
        and m["run_contract"]["step6d_run_id"] == expected_step6d_run_id
        and m["run_contract"]["candidate_variant"] == expected_candidate_variant
        and m["run_contract"]["schedule"] == expected_runner_schedule,
        "performance-lock runner contract schema changed",
    )
if lock_mode:
    qualification = m["tail_qualification"]
    require(
        set(qualification)
        == {
            "summary_sha256", "validation_sha256",
            "scientific_merge_sha256", "receive_receipt_sha256",
            "development_run_contract_digest", "all_gates_passed",
            "performance_candidate_frozen", "performance_lock_authorized",
            "open_claim_package_path", "open_claim_sha256",
            "root_seal_package_path", "root_seal_sha256",
            "materialization_package_path", "materialization_sha256",
            "lock_root_aggregate_sha256", "lock_root_topology_sha256",
            "lock_observation_fingerprint_sha256",
            "lock_development_overlap_count",
        }
        and qualification["summary_sha256"]
        == "f587df6d037313e2111a5cbc3d474370106f6a341ef4ec130cee7516192e668f"
        and qualification["validation_sha256"]
        == "da252b3cabfc361d89de2f2fe08931fe61302287b69b82bf9f812a4432baf3eb"
        and qualification["scientific_merge_sha256"]
        == "29e6b3db5b372f9808b681fb70246368452eecf5c6bbbca2519b2882430369f7"
        and qualification["receive_receipt_sha256"]
        == "2ba1b7434cda012d8a230109e9c37db6d9e9b41587e1138e79790a177df85cf6"
        and qualification["development_run_contract_digest"]
        == "92a87f1544a73104d5256db39b022094c8c7f7b11f77bd19dcf1ab1442581ebd"
        and qualification["all_gates_passed"] is True
        and qualification["performance_candidate_frozen"] is True
        and qualification["performance_lock_authorized"] is True
        and qualification["lock_development_overlap_count"] == 0,
        "performance-lock qualification binding changed",
    )
else:
    require(
        m["tail_qualification"] == {
            "summary_sha256": "1eb098073ed51efc6868771ac04ecbde965a91dd59bc7a02da547b5555f8253b",
            "validation_sha256": "26a27c730399cb4adbf612d4e510b21462c29078e75dbe427e431115239e5011",
            "run_contract_digest": "b5d3114d0857723809ec85cef957921acafa67e22756aef050fa1c8ef8f79bf6",
            "candidate_variant": "candidate02_compact_scorer_tail_v2",
            "hand_indices": [0, 4, 5, 12, 14, 16, 17, 23, 41, 43],
            "all_gates_passed": True,
            "full_performance_development_authorized": True,
        },
        "tail qualification binding changed",
    )
require(a["tail_summary_sha256"] == m["tail_qualification"]["summary_sha256"], "tail summary hash changed")
require(a["tail_validation_sha256"] == m["tail_qualification"]["validation_sha256"], "tail validation hash changed")
expected_schedule = {
    "scope": expected_plan_scope if lock_mode else "full_performance_development",
    "contract_hand_indices": list(range(100)),
    "source_roles": ["candidate", "reference"],
    "shards_per_role": 10,
    "hands_per_shard": 10,
    "logical_job_count": 20,
    "mapping": (
        "one_source_ten_hand_arithmetic_shard_per_vm"
        if lock_mode
        else "one_source_ten_hand_shard_per_vm"
    ),
    "max_concurrent_vms": 20,
}
require(m["schedule"] == expected_schedule, "full100 schedule changed")
require(
    m["checkpoint"] == {
        "unit": "completed_source_hand",
        "upload_after_each_hand": True,
        "immutable_write_once": True,
        "resume_verifies_all_artifacts": True,
        "done_uploaded_last": True,
    },
    "checkpoint contract changed",
)
require(
    m["heartbeat"] == {
        "schema": "hu_m31_t3_step6d_full100_heartbeat_v1",
        "required": True,
        "interval_seconds": 60,
        "remote_scope": "progress_only",
    },
    "heartbeat contract changed",
)

plan_path = (work / m["plan_package_path"]).resolve()
require(plan_path.is_relative_to(work) and plan_path.is_file(), "frozen plan missing")
require(hashlib.sha256(plan_path.read_bytes()).hexdigest() == m["plan_sha256"], "frozen plan bytes changed")
plan = load_canonical(plan_path, "frozen full100 plan")
require(plan["schema"] == expected_plan_schema, "plan schema changed")
require(plan["run_contract"] == m["run_contract"], "plan/package contract mismatch")
require(plan["run_contract_digest"] == m["run_contract_digest"], "plan contract hash changed")
require(plan["logical_job_count"] == 20 and len(plan["jobs"]) == 20, "plan job count changed")
if lock_mode:
    require(
        plan["scope"] == expected_plan_scope
        and plan["open_claim_required_before_root_content"] is True
        and plan["root_content_opened"] is False
        and plan["cloud_started"] is False,
        "performance-lock precontent plan was widened",
    )
else:
    require(
        plan["spot_package_authorized"] is False
        and plan["cloud_started"] is False,
        "plan was mutated",
    )
for key in (
    "training_eligible", "quality_evidence", "promotion_evidence",
    "current_profile_changed", "named_profile_added",
    "runtime_policy_activated", "m31_complete",
):
    require(plan[key] is False, f"forbidden plan flag changed: {key}")

job_ids = [
    f"{role}-shard-{index:02d}"
    for role in ("candidate", "reference")
    for index in range(10)
]
if lock_mode:
    packaged_root_claim = load_canonical(
        work / "frozen/performance_lock_open_claim.json",
        "packaged performance-lock open claim",
    )
    packaged_root_seal = load_canonical(
        work / "frozen/performance_lock_root_seal.json",
        "packaged performance-lock root seal",
    )
    validate_global_spot_claim(
        a["global_spot_claim"],
        root_claim=packaged_root_claim,
        root_seal=packaged_root_seal,
        manifest=m,
        manifest_sha=manifest_sha,
        job_ids=job_ids,
        expected_schema=expected_global_spot_claim_schema,
        expected_status=expected_global_spot_claim_status,
        receipt_required=rearm2_mode,
        authorization=a,
    )
require(a["authorized_job_ids"] == job_ids, "authorization job order changed")
require(a["logical_job_count"] == 20, "authorization job count changed")
require(
    a["performance_development_only"] is (not lock_mode),
    "authorization phase changed",
)
require(isinstance(a["authorized_unix_seconds"], (int, float)) and not isinstance(a["authorized_unix_seconds"], bool) and a["authorized_unix_seconds"] > 0, "authorization timestamp changed")
require(
    a["performance_lock_authorized"] is lock_mode,
    "performance-lock authorization changed",
)
for key in (
    "quality_pilot_authorized", "training_eligible", "current_profile_changed",
    "named_profile_added", "runtime_policy_activated",
):
    require(a[key] is False, f"forbidden authorization flag changed: {key}")
require(a["spot_execution_authorized"] is True, "Spot is not authorized")
for key in (
    "spot_execution_authorized", "performance_lock_authorized",
    "quality_pilot_authorized", "training_eligible",
    "current_profile_changed", "named_profile_added",
    "runtime_policy_activated", "m31_complete", "gcloud_invoked",
):
    require(m[key] is False, f"forbidden package flag changed: {key}")

require(
    set(j) == {
        "schema", "run_contract", "run_contract_digest",
        "source_role", "work_hand_indices",
    },
    "runner job fields changed",
)
require(j["schema"] == expected_shard_manifest_schema, "runner job schema changed")
require(j["run_contract"] == m["run_contract"], "job/package contract mismatch")
require(j["run_contract_digest"] == digest(j["run_contract"]) == m["run_contract_digest"], "job contract digest changed")
role = j["source_role"]
work_indices = j["work_hand_indices"]
require(role in ("candidate", "reference"), "job role changed")
require(
    isinstance(work_indices, list)
    and len(work_indices) == 10
    and work_indices == sorted(work_indices)
    and len(set(work_indices)) == 10
    and all(type(index) is int and 0 <= index < 100 for index in work_indices),
    "job work shard changed",
)
plan_job = [row for row in plan["jobs"] if row["job_id"] == job_id]
require(len(plan_job) == 1, "job absent from frozen plan")
require(
    plan_job[0]["source_role"] == role
    and plan_job[0]["work_hand_indices"] == work_indices
    and plan_job[0]["shard_manifest_sha256"] == job_sha,
    "job/frozen plan mapping changed",
)
require(job_id == f"{role}-shard-{plan_job[0]['shard_index']:02d}", "job id mapping changed")

records = m["job_manifests"]
require(isinstance(records, list) and len(records) == 20, "package job records changed")
require([row["job_id"] for row in records] == job_ids, "package job order changed")
require(
    all(
        set(row) == {
            "job_id", "source_role", "shard_index", "work_hand_indices",
            "path", "output_prefix", "sha256", "bytes",
        }
        for row in records
    ),
    "package job record fields changed",
)
record = [row for row in records if row["job_id"] == job_id]
require(len(record) == 1 and record[0]["sha256"] == job_sha, "job hash record changed")
require(record[0]["source_role"] == role and record[0]["work_hand_indices"] == work_indices, "job record mapping changed")
require(record[0]["shard_index"] == plan_job[0]["shard_index"], "job record shard changed")
require(record[0]["path"] == f"jobs/{job_id}.json", "job package path changed")
require(record[0]["output_prefix"] == f"jobs/{job_id}", "job output prefix changed")
require(record[0]["bytes"] == job_path.stat().st_size, "job byte count changed")

reference = {
    "package_path": "native/reference/release/libofc_hu_m3_engine.so",
    "sha256": "3f831615d9e751b8e1f266f14dd2009903b3ac2a4094f7e47616e0aa372a01b0",
}
candidate = {
    "package_path": "native/candidate/release/libofc_hu_m3_engine.so",
    "sha256": "4050e04b22d7943da9e1691402a2b34cf3d3781041a44557f35e2dfcf366d55d",
}
require(m["accepted_reference"] == reference, "reference binding changed")
require(m["accepted_candidate"] == candidate, "candidate binding changed")
require(m["run_contract"]["reference_library_sha256"] == reference["sha256"], "reference contract changed")
require(m["run_contract"]["candidate_library_sha256"] == candidate["sha256"], "candidate contract changed")
binding = candidate if role == "candidate" else reference

entries = m["source_entries"]
require(isinstance(entries, dict) and len(entries) == m["source_entry_count"], "source entries changed")
for relative, expected in entries.items():
    pure = pathlib.PurePosixPath(relative)
    require(not pure.is_absolute() and ".." not in pure.parts, "unsafe package entry")
    path = (work / pathlib.Path(*pure.parts)).resolve()
    require(path.is_relative_to(work) and path.is_file(), f"missing package entry: {relative}")
    data = path.read_bytes()
    require(len(data) == expected["bytes"], f"package entry size changed: {relative}")
    require(hashlib.sha256(data).hexdigest() == expected["sha256"], f"package entry hash changed: {relative}")
for index in range(100):
    relative = f"frozen/full100_roots/hand_{index:03d}.json"
    require(relative in entries, f"frozen root absent: {relative}")

require(digest(c) == claim_sha, "attempt claim digest changed")
common_claim = {
    "run_name": run_name,
    "package_manifest_sha256": manifest_sha,
    "launch_authorization_sha256": authorization_sha,
    "plan_sha256": m["plan_sha256"],
    "run_contract_digest": m["run_contract_digest"],
    "launch_target": m["launch_target"],
    "cost_guard_sha256": cost_guard_sha,
}
for key, value in common_claim.items():
    require(c.get(key) == value, f"attempt claim binding changed: {key}")
if attempt_index == 0:
    require(
        set(c) == {
            "schema", "status", "run_name", "selected_job_ids",
            "package_manifest_sha256", "launch_authorization_sha256",
            "plan_sha256", "run_contract_digest", "launch_target",
            "cost_guard_sha256", "preflight_sha256",
            "claimed_unix_seconds", "crash_reuse_authorized",
        },
        "launch claim keys changed",
    )
    require(
        c["schema"]
        == (
            "hu_m31_t3_step6d_performance_lock_launch_claim_v1"
            if lock_mode
            else "hu_m31_t3_step6d_full100_launch_claim_v1"
        ),
        "launch claim schema changed",
    )
    require(
        c["status"]
        == (
            "exclusive_performance_lock_launch_claim_before_remote_mutation"
            if lock_mode
            else "exclusive_full100_launch_claim_acquired_before_remote_mutation"
        ),
        "launch claim status changed",
    )
    require(c["selected_job_ids"] == job_ids, "launch claim job set changed")
    require(c["crash_reuse_authorized"] is False, "launch crash reuse authorized")
    require(claim_sha == initial_claim_sha and initial_result_sha == "none", "attempt0 chain changed")
else:
    require(
        set(c) == {
            "schema", "status", "run_name", "attempt_index",
            "selected_job_ids", "initial_launch_claim_sha256",
            "initial_launch_result_sha256", "package_manifest_sha256",
            "launch_authorization_sha256", "plan_sha256",
            "run_contract_digest", "launch_target", "cost_guard_sha256",
            "preflight_sha256", "claimed_unix_seconds",
            "third_attempt_authorized",
        },
        "resume claim keys changed",
    )
    require(
        c["schema"]
        == (
            "hu_m31_t3_step6d_performance_lock_resume_claim_v1"
            if lock_mode
            else "hu_m31_t3_step6d_full100_resume_claim_v1"
        ),
        "resume claim schema changed",
    )
    require(
        c["status"]
        == (
            "exclusive_performance_lock_attempt1_claim_before_remote_mutation"
            if lock_mode
            else "exclusive_full100_attempt1_claim_acquired_before_remote_mutation"
        ),
        "resume claim status changed",
    )
    require(c["attempt_index"] == 1 and job_id in c["selected_job_ids"], "resume job identity changed")
    require(c["initial_launch_claim_sha256"] == initial_claim_sha, "resume launch claim chain changed")
    require(c["initial_launch_result_sha256"] == initial_result_sha, "resume launch result chain changed")
    require(c["third_attempt_authorized"] is False, "third attempt authorized")
require(isinstance(c["preflight_sha256"], str) and len(c["preflight_sha256"]) == 64, "preflight hash changed")
require(isinstance(c["claimed_unix_seconds"], (int, float)) and not isinstance(c["claimed_unix_seconds"], bool) and c["claimed_unix_seconds"] > 0, "claim timestamp changed")

expected_instance = f"{run_name}-j{job_ids.index(job_id):02d}" + ("" if attempt_index == 0 else "-a01")
require(instance_name == expected_instance, "instance/job/attempt mapping changed")
prefix = f"gs://{bucket}/runs/{run_name}/full100"
require(result_prefix == f"{prefix}/results/jobs/{job_id}", "result prefix changed")
require(progress_prefix == f"{prefix}/progress/jobs/{job_id}", "progress prefix changed")
require(project == m["launch_target"]["project"], "project target changed")
require(bucket == m["launch_target"]["bucket"], "bucket target changed")
require(zone in m["launch_target"]["zones"], "zone target changed")
require(self_delete == 1 and m["launch_target"]["self_delete"] is True, "self-delete target changed")
require(a["launch_target"] == m["launch_target"] == c["launch_target"], "launch target chain changed")
require(a["cost_guard"] == m["cost_guard"], "cost guard chain changed")
require(digest(m["cost_guard"]) == cost_guard_sha, "cost guard digest changed")
require(
    m["cost_guard"] == {
        "schema": "hu_m31_t3_step6d_full100_cost_guard_v1",
        "currency": "USD",
        "spot_price_ceiling_usd_per_vm_hour": 0.5,
        "max_runtime_seconds_per_vm": 4200,
        "internal_watchdog_seconds_per_vm": 3900,
        "max_concurrent_vms": 20,
        "max_logical_jobs": 20,
        "max_attempts_per_job": 2,
        "max_cumulative_vm_jobs": 40,
        "initial_estimated_max_compute_usd": 11.666666666666666,
        "all_attempts_estimated_max_compute_usd": 23.333333333333332,
        "phase_compute_cap_usd": 25.0,
        "m31_total_compute_cap_usd": 500.0,
    },
    "full100 cost guard changed",
)
require(max_runtime_seconds == 3900, "watchdog metadata changed")

print(role)
print(",".join(str(index) for index in work_indices))
print(binding["package_path"])
print(binding["sha256"])
print(m["run_contract_digest"])
print(expected_done_schema)
PY
)
[[ "${#JOB_FIELDS[@]}" -eq 6 ]]
ROLE="${JOB_FIELDS[0]}"
IFS=',' read -r -a WORK_HANDS <<<"${JOB_FIELDS[1]}"
LIBRARY_RELATIVE="${JOB_FIELDS[2]}"
LIBRARY_SHA256="${JOB_FIELDS[3]}"
RUN_CONTRACT_DIGEST="${JOB_FIELDS[4]}"
RUNNER_DONE_SCHEMA="${JOB_FIELDS[5]}"
[[ "$RUNNER_DONE_SCHEMA" == "$PRECONTENT_DONE_SCHEMA" ]]
[[ "${#WORK_HANDS[@]}" -eq 10 ]]
[[ "$(sha "$WORK/$LIBRARY_RELATIVE")" == "$LIBRARY_SHA256" ]]

cd "$WORK"
python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --disable-pip-version-check --no-input \
  -r configs/hu_m43_attempt08_runtime_requirements.txt
python -m pip check

export PYTHONPATH="$WORK/src"
export RAYON_NUM_THREADS=16
export OFC_HU_M3_BATCH_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

mkdir -p "$RESULT/roots" "$RESULT/hands/$ROLE"
for hand in "${WORK_HANDS[@]}"; do
  printf -v hand_pad '%03d' "$hand"
  cp "$WORK/frozen/full100_roots/hand_$hand_pad.json" \
    "$RESULT/roots/hand_$hand_pad.json"
done

restore_or_compare run_contract.json
restore_or_compare shard_manifest.json
for hand in "${WORK_HANDS[@]}"; do
  printf -v hand_pad '%03d' "$hand"
  restore_or_compare "roots/hand_$hand_pad.json"
  restore_or_compare "hands/$ROLE/hand_$hand_pad.json"
done
# A remote DONE is restored only after every artifact it commits.
if [[ "$(gcs_state "$RESULT_PREFIX/DONE.json")" == present ]]; then
  gcloud storage cp "$RESULT_PREFIX/DONE.json" "$RESULT/DONE.json" \
    --project "$PROJECT_ID" >/dev/null
fi

heartbeat_once() {
  python - "$RESULT/heartbeat.json" "$RUN_NAME" "$JOB_ID" "$ROLE" \
    "${JOB_FIELDS[1]}" "$RUN_CONTRACT_DIGEST" "$ATTEMPT_INDEX" \
    "$VALIDATED_INDEX_FILE" <<'PY'
import json
import os
import pathlib
import sys
import time

path = pathlib.Path(sys.argv[1])
role = sys.argv[4]
work = [int(value) for value in sys.argv[5].split(",")]
validated_path = pathlib.Path(sys.argv[8])
completed = json.loads(validated_path.read_text()) if validated_path.is_file() else []
if (
    not isinstance(completed, list)
    or completed != sorted(completed)
    or any(type(index) is not int or index not in work for index in completed)
):
    raise SystemExit("validated heartbeat index state changed")
if any(
    not (path.parent / "hands" / role / f"hand_{index:03d}.json").is_file()
    for index in completed
):
    raise SystemExit("validated heartbeat hand is missing")
done = (path.parent / "DONE.json").is_file() and completed == work
value = {
    "schema": "hu_m31_t3_step6d_full100_heartbeat_v1",
    "run_name": sys.argv[2],
    "job_id": sys.argv[3],
    "source_role": role,
    "work_hand_indices": work,
    "run_contract_digest": sys.argv[6],
    "attempt_index": int(sys.argv[7]),
    "status": "done_validated_locally" if done else ("checkpointed" if completed else "running"),
    "completed_hand_indices": completed,
    "unix_seconds": time.time(),
}
temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
temporary.write_text(
    json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
    encoding="utf-8",
)
os.replace(temporary, path)
PY
  gcloud storage cp "$RESULT/heartbeat.json" \
    "$PROGRESS_PREFIX/heartbeat.json" --project "$PROJECT_ID" >/dev/null
}

mark_present_hands_validated() {
  python - "$VALIDATED_INDEX_FILE" "$RESULT" "$ROLE" \
    "${JOB_FIELDS[1]}" <<'PY'
import json
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
result = pathlib.Path(sys.argv[2])
role = sys.argv[3]
work = [int(value) for value in sys.argv[4].split(",")]
completed = [
    index
    for index in work
    if (result / "hands" / role / f"hand_{index:03d}.json").is_file()
]
temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
temporary.write_text(
    json.dumps(completed, separators=(",", ":")) + "\n",
    encoding="utf-8",
)
os.replace(temporary, path)
PY
}

start_pump() {
  heartbeat_once
  (
    while true; do
      sleep 60
      heartbeat_once
    done
  ) &
  PUMP_PID=$!
}

checkpoint_hand() {
  local hand=$1
  local hand_pad
  printf -v hand_pad '%03d' "$hand"
  [[ -f "$RESULT/roots/hand_$hand_pad.json" ]]
  [[ -f "$RESULT/hands/$ROLE/hand_$hand_pad.json" ]]
  upload_once "$RESULT/run_contract.json" \
    "$PROGRESS_PREFIX/run_contract.json"
  upload_once "$RESULT/shard_manifest.json" \
    "$PROGRESS_PREFIX/shard_manifest.json"
  upload_once "$RESULT/run_contract.json" \
    "$RESULT_PREFIX/run_contract.json"
  upload_once "$RESULT/shard_manifest.json" \
    "$RESULT_PREFIX/shard_manifest.json"
  upload_once "$RESULT/roots/hand_$hand_pad.json" \
    "$PROGRESS_PREFIX/roots/hand_$hand_pad.json"
  upload_once "$RESULT/hands/$ROLE/hand_$hand_pad.json" \
    "$PROGRESS_PREFIX/hands/$ROLE/hand_$hand_pad.json"
  upload_once "$RESULT/roots/hand_$hand_pad.json" \
    "$RESULT_PREFIX/roots/hand_$hand_pad.json"
  upload_once "$RESULT/hands/$ROLE/hand_$hand_pad.json" \
    "$RESULT_PREFIX/hands/$ROLE/hand_$hand_pad.json"
  heartbeat_once
}

start_pump
# Existing hand artifacts are left in place.  Every runner pass validates all
# restored hands before advancing, so a valid checkpoint is never recomputed.
for hand in "${WORK_HANDS[@]}"; do
  printf -v hand_pad '%03d' "$hand"
  if [[ ! -f "$RESULT/hands/$ROLE/hand_$hand_pad.json" ]]; then
    python -m ofc_regular.run_hu_m31_t3_step6d_performance_v2 \
      --shard-manifest /tmp/job.json \
      --repository-root "$WORK" \
      --output-dir "$RESULT" \
      --library "$WORK/$LIBRARY_RELATIVE" \
      --stop-after-hands 1
    mark_present_hands_validated
    # Newly computed source-hand artifacts are committed immediately.
    checkpoint_hand "$hand"
  fi
done
stop_pump

# Second idempotent pass: revalidate all ten roots/hands and the local DONE
# against the same immutable one-source shard before remote publication.
python -m ofc_regular.run_hu_m31_t3_step6d_performance_v2 \
  --shard-manifest /tmp/job.json \
  --repository-root "$WORK" \
  --output-dir "$RESULT" \
  --library "$WORK/$LIBRARY_RELATIVE" >/tmp/full100_post_validation.json
python - /tmp/full100_post_validation.json "$PACKAGE_PHASE" "$ROLE" \
  "${JOB_FIELDS[1]}" "$RUNNER_DONE_SCHEMA" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
phase, role = sys.argv[2:4]
work = [int(value) for value in sys.argv[4].split(",")]
expected_schema = sys.argv[5]
if (
    not isinstance(report, dict)
    or report.get("schema") != expected_schema
    or report.get("status") != "complete_source_isolated_shard"
    or report.get("source_role") != role
    or report.get("work_hand_indices") != work
    or report.get("completed_hand_indices") != work
    or report.get("training_eligible") is not False
    or report.get("quality_evidence") is not False
    or report.get("promotion_evidence") is not False
    or report.get("current_profile_changed") is not False
):
    raise SystemExit("full100 runner DONE schema/phase changed")
PY
mark_present_hands_validated

# This also commits restored hands after the second pass has validated them.
for hand in "${WORK_HANDS[@]}"; do
  checkpoint_hand "$hand"
done

[[ -f "$RESULT/DONE.json" ]]
mapfile -t RESULT_FILES < <(
  find "$RESULT" -type f ! -name heartbeat.json -printf '%P\n' | sort
)
[[ "${#RESULT_FILES[@]}" -eq 23 ]]
[[ "${RESULT_FILES[0]}" == "DONE.json" ]]
[[ "${RESULT_FILES[21]}" == "run_contract.json" ]]
[[ "${RESULT_FILES[22]}" == "shard_manifest.json" ]]
[[ "$(find "$RESULT/roots" -type f -name 'hand_*.json' | wc -l)" -eq 10 ]]
[[ "$(find "$RESULT/hands/$ROLE" -type f -name 'hand_*.json' | wc -l)" -eq 10 ]]

# DONE is the only remote commit marker and is always uploaded last.
heartbeat_once
upload_once "$RESULT/DONE.json" "$RESULT_PREFIX/DONE.json"
REMOTE_DONE="$(mktemp)"
gcloud storage cp "$RESULT_PREFIX/DONE.json" "$REMOTE_DONE" \
  --project "$PROJECT_ID" >/dev/null
[[ "$(sha "$RESULT/DONE.json")" == "$(sha "$REMOTE_DONE")" ]]
rm -f "$REMOTE_DONE"
VALIDATED_DONE=1
