#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

META_ROOT='http://metadata.google.internal/computeMetadata/v1/instance/attributes'
META_HEADER='Metadata-Flavor: Google'
WORK='/opt/ofc-step6d-v2'
VENV='/opt/ofc-step6d-v2-venv'
STATE='/var/lib/ofc-step6d-v2'
LOG='/var/log/ofc-step6d-v2-startup.log'
VALIDATED_DONE=0
PUMP_PID=''
WATCHDOG_PID=''
MAIN_PID=$$
INSTANCE_NAME=''
ZONE=''
PROJECT_ID=''
BUCKET=''
RUN_NAME=''
JOB_ID=''
SELF_DELETE=0
PROGRESS_URI=''
RESULT=''
LOG_URI=''
MAX_RUNTIME_SECONDS=3000
ATTEMPT_INDEX=0

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

upload_once() {
  local source=$1
  local destination=$2
  local existing
  existing="$(mktemp)"
  if gcloud storage objects describe "$destination" --project "$PROJECT_ID" \
      >/dev/null 2>&1; then
    gcloud storage cp "$destination" "$existing" --project "$PROJECT_ID" >/dev/null
    [[ "$(sha "$source")" == "$(sha "$existing")" ]]
    rm -f "$existing"
    return
  fi
  if ! gcloud storage cp "$source" "$destination" --project "$PROJECT_ID" \
      --if-generation-match=0 >/dev/null 2>&1; then
    gcloud storage cp "$destination" "$existing" --project "$PROJECT_ID" >/dev/null
    [[ "$(sha "$source")" == "$(sha "$existing")" ]]
  fi
  rm -f "$existing"
}

sync_failure_state() {
  [[ -n "$PROGRESS_URI" && -n "$PROJECT_ID" && -d "$RESULT" ]] || return 0
  local source relative
  while IFS= read -r -d '' source; do
    relative="${source#"$RESULT"/}"
    case "$relative" in
      run_contract.json|shard_manifest.json|roots/hand_[0-9][0-9][0-9].json|hands/candidate/hand_[0-9][0-9][0-9].json|hands/reference/hand_[0-9][0-9][0-9].json)
        upload_once "$source" "$PROGRESS_URI/$relative" || true
        ;;
    esac
  done < <(find "$RESULT" -type f -print0 2>/dev/null)
}

upload_startup_log() {
  if [[ -n "$LOG_URI" && -f "$LOG" && -n "$PROJECT_ID" ]]; then
    gcloud storage cp "$LOG" "$LOG_URI" --project "$PROJECT_ID" \
      >/dev/null 2>&1 || true
  fi
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

cleanup() {
  local code=$?
  trap - EXIT TERM INT
  set +e
  stop_pump
  stop_watchdog
  if [[ "$code" -ne 0 || "$VALIDATED_DONE" -ne 1 ]]; then
    [[ "$code" -ne 0 ]] || code=1
    sync_failure_state
    upload_startup_log
    delete_instance_or_shutdown
    exit "$code"
  fi
  upload_startup_log
  if [[ "$SELF_DELETE" -ne 1 ]]; then
    echo 'validated success lacked mandatory self-delete authorization' >&2
    delete_instance_or_shutdown
    exit 1
  fi
  if ! gcloud compute instances delete "$INSTANCE_NAME" \
      --zone "$ZONE" --project "$PROJECT_ID" --quiet >/dev/null 2>&1; then
    echo 'self-delete failed; forcing shutdown' >&2
    force_shutdown
    exit 1
  fi
  exit 0
}
trap cleanup EXIT
trap 'exit 124' TERM
exec > >(tee -a "$LOG") 2>&1

PROJECT_ID="$(meta PROJECT_ID)"
BUCKET="$(meta BUCKET)"
RUN_NAME="$(meta RUN_NAME)"
JOB_ID="$(meta JOB_ID)"
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
COST_GUARD_SHA256="$(meta COST_GUARD_SHA256)"
MAX_RUNTIME_SECONDS="$(meta MAX_RUNTIME_SECONDS)"
SELF_DELETE="$(meta SELF_DELETE)"
INSTANCE_NAME="$(curl -fsS -H "$META_HEADER" \
  'http://metadata.google.internal/computeMetadata/v1/instance/name')"
ZONE="$(curl -fsS -H "$META_HEADER" \
  'http://metadata.google.internal/computeMetadata/v1/instance/zone' | awk -F/ '{print $NF}')"

[[ "$RUN_NAME" =~ ^[a-z0-9][a-z0-9-]{2,46}[a-z0-9]$ ]]
[[ "$JOB_ID" =~ ^(candidate|reference)-hand-[0-9]{3}$ ]]
[[ "$SOURCE_SHA256" =~ ^[0-9a-f]{64}$ ]]
[[ "$MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]]
[[ "$JOB_MANIFEST_SHA256" =~ ^[0-9a-f]{64}$ ]]
[[ "$AUTHORIZATION_SHA256" =~ ^[0-9a-f]{64}$ ]]
[[ "$ATTEMPT_INDEX" == 0 || "$ATTEMPT_INDEX" == 1 ]]
[[ "$ATTEMPT_CLAIM_SHA256" =~ ^[0-9a-f]{64}$ ]]
if [[ "$ATTEMPT_INDEX" == 0 ]]; then
  [[ "$INITIAL_LAUNCH_CLAIM_SHA256" =~ ^[0-9a-f]{64}$ ]]
  [[ "$INITIAL_LAUNCH_RESULT_SHA256" == none ]]
else
  [[ "$INITIAL_LAUNCH_CLAIM_SHA256" =~ ^[0-9a-f]{64}$ ]]
  [[ "$INITIAL_LAUNCH_RESULT_SHA256" =~ ^[0-9a-f]{64}$ ]]
fi
[[ "$COST_GUARD_SHA256" =~ ^[0-9a-f]{64}$ ]]
[[ "$MAX_RUNTIME_SECONDS" == 3000 ]]
[[ "$SELF_DELETE" == 1 ]]

PREFIX="gs://$BUCKET/runs/$RUN_NAME"
if [[ "$ATTEMPT_INDEX" == 0 ]]; then
  [[ "$ATTEMPT_CLAIM_URI" == "$PREFIX/control/launch_claim.json" ]]
else
  [[ "$ATTEMPT_CLAIM_URI" == "$PREFIX/resume/resume_claim.json" ]]
fi
PROGRESS_URI="$PREFIX/progress/jobs/$JOB_ID"
RESULT_URI="$PREFIX/results/jobs/$JOB_ID"
RESULT="$STATE/$JOB_ID"
LOG_URI="$PREFIX/logs/jobs/$JOB_ID/attempt-$ATTEMPT_INDEX/startup.log"

start_watchdog() {
  (
    sleep "$MAX_RUNTIME_SECONDS"
    echo "whole-VM runtime watchdog expired after ${MAX_RUNTIME_SECONDS}s" >&2
    kill -TERM "$MAIN_PID" >/dev/null 2>&1 || true
    sleep 5
    sync_failure_state
    upload_startup_log
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

gcloud storage cp "$SOURCE_URI" /tmp/source.zip --project "$PROJECT_ID" >/dev/null
gcloud storage cp "$MANIFEST_URI" /tmp/manifest.json --project "$PROJECT_ID" >/dev/null
gcloud storage cp "$JOB_MANIFEST_URI" /tmp/job.json --project "$PROJECT_ID" >/dev/null
gcloud storage cp "$AUTHORIZATION_URI" /tmp/authorization.json \
  --project "$PROJECT_ID" >/dev/null
gcloud storage cp "$ATTEMPT_CLAIM_URI" /tmp/attempt_claim.json \
  --project "$PROJECT_ID" >/dev/null
[[ "$(sha /tmp/source.zip)" == "$SOURCE_SHA256" ]]
[[ "$(sha /tmp/manifest.json)" == "$MANIFEST_SHA256" ]]
[[ "$(sha /tmp/job.json)" == "$JOB_MANIFEST_SHA256" ]]
[[ "$(sha /tmp/authorization.json)" == "$AUTHORIZATION_SHA256" ]]
[[ "$(sha /tmp/attempt_claim.json)" == "$ATTEMPT_CLAIM_SHA256" ]]

rm -rf "$WORK" "$VENV"
mkdir -p "$WORK"
unzip -q /tmp/source.zip -d "$WORK"

mapfile -t JOB_FIELDS < <(python3 - "$WORK" /tmp/manifest.json /tmp/job.json \
  /tmp/authorization.json /tmp/attempt_claim.json "$RUN_NAME" "$JOB_ID" "$SOURCE_SHA256" \
  "$MANIFEST_SHA256" "$JOB_MANIFEST_SHA256" "$COST_GUARD_SHA256" \
  "$MAX_RUNTIME_SECONDS" "$PROJECT_ID" "$BUCKET" "$ZONE" "$INSTANCE_NAME" \
  "$SELF_DELETE" "$ATTEMPT_INDEX" "$ATTEMPT_CLAIM_SHA256" \
  "$INITIAL_LAUNCH_CLAIM_SHA256" "$INITIAL_LAUNCH_RESULT_SHA256" <<'PY'
import hashlib
import json
import pathlib
import sys


def require(condition, message):
    if not condition:
        raise SystemExit(message)


work = pathlib.Path(sys.argv[1]).resolve()
manifest_path = pathlib.Path(sys.argv[2])
job_path = pathlib.Path(sys.argv[3])
authorization_path = pathlib.Path(sys.argv[4])
attempt_claim_path = pathlib.Path(sys.argv[5])
run_name, job_id = sys.argv[6], sys.argv[7]
source_sha, manifest_sha, job_sha = sys.argv[8:11]
cost_guard_sha = sys.argv[11]
max_runtime_seconds = int(sys.argv[12])
project, bucket, zone, instance_name = sys.argv[13:17]
self_delete = int(sys.argv[17])
attempt_index = int(sys.argv[18])
attempt_claim_sha = sys.argv[19]
initial_claim_sha, initial_result_sha = sys.argv[20:22]
m = json.loads(manifest_path.read_text(encoding="utf-8"))
j = json.loads(job_path.read_text(encoding="utf-8"))
a = json.loads(authorization_path.read_text(encoding="utf-8"))
attempt_claim = json.loads(attempt_claim_path.read_text(encoding="utf-8"))
canonical = lambda value: (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode()
digest = lambda value: hashlib.sha256(canonical(value)).hexdigest()
contract = j.get("run_contract")
require(isinstance(contract, dict), "job run contract is absent")
contract_indices = contract.get("contract_hand_indices")
tail = contract.get("tail_hand_indices")
require(contract_indices == list(range(100)), "contract coverage changed")
require(
    isinstance(tail, list)
    and len(tail) == 10
    and len(set(tail)) == 10
    and all(
        isinstance(index, int)
        and not isinstance(index, bool)
        and index in contract_indices
        for index in tail
    ),
    "contract tail changed or is invalid",
)
roles = ["candidate", "reference"]
job_ids = [f"{role}-hand-{index:03d}" for role in roles for index in tail]
manifest_keys = {
    "schema", "status", "run_name", "source_name", "source_sha256",
    "source_bytes", "startup_name", "startup_sha256", "run_contract",
    "run_contract_digest", "accepted_reference", "accepted_candidate",
    "feature_encoder", "image", "allocation", "tail_schedule",
    "launch_target", "cost_guard",
    "job_manifests", "source_entries", "source_entry_count", "checkpoint",
    "heartbeat", "spot_execution_authorized", "production_fanout_authorized",
    "training_eligible", "current_profile_changed", "named_profile_added",
    "runtime_policy_activated", "m31_complete", "gcloud_invoked",
}
authorization_keys = {
    "schema", "status", "run_name", "package_manifest_sha256",
    "source_sha256", "startup_sha256", "run_contract_digest",
    "launch_target", "cost_guard",
    "authorized_job_ids", "logical_job_count", "tail_only",
    "spot_execution_authorized", "production_fanout_authorized",
    "training_eligible", "current_profile_changed", "named_profile_added",
    "runtime_policy_activated", "authorized_unix_seconds",
}

require(set(m) == manifest_keys, "package manifest keys changed")
require(set(a) == authorization_keys, "authorization keys changed")
require(m.get("schema") == "hu_m31_t3_step6d_spot_package_v2", "bad package schema")
require(m.get("status") == "immutable_package_ready_not_authorized", "bad package status")
require(m.get("run_name") == a.get("run_name") == run_name, "run name mismatch")
require(m.get("source_sha256") == a.get("source_sha256") == source_sha, "source hash mismatch")
require(a.get("package_manifest_sha256") == manifest_sha, "manifest hash mismatch")
require(a.get("startup_sha256") == m.get("startup_sha256"), "startup hash binding changed")
require(m.get("source_bytes") == pathlib.Path("/tmp/source.zip").stat().st_size, "source byte count changed")
require(a.get("schema") == "hu_m31_t3_step6d_spot_launch_authorization_v2", "bad authorization schema")
require(a.get("status") == "explicit_tail_spot_authorization", "bad authorization status")
require(a.get("authorized_job_ids") == job_ids, "authorization job set changed")
require(a.get("logical_job_count") == 20 and a.get("tail_only") is True, "authorization fanout changed")
require(a.get("spot_execution_authorized") is True, "Spot is not authorized")
require(isinstance(a.get("authorized_unix_seconds"), (int, float)) and not isinstance(a.get("authorized_unix_seconds"), bool) and a["authorized_unix_seconds"] > 0, "authorization timestamp changed")
require(digest(attempt_claim) == attempt_claim_sha, "attempt claim digest changed")
for key in ("production_fanout_authorized", "training_eligible", "current_profile_changed", "named_profile_added", "runtime_policy_activated"):
    require(a.get(key) is False, f"forbidden authorization flag changed: {key}")
for key in ("spot_execution_authorized", "production_fanout_authorized", "training_eligible", "current_profile_changed", "named_profile_added", "runtime_policy_activated", "m31_complete", "gcloud_invoked"):
    require(m.get(key) is False, f"forbidden package flag changed: {key}")
launch_target = {
    "project": "ofc-solver-485418",
    "bucket": "pokerhu-ofc-solver-485418-training",
    "region": "asia-northeast1",
    "zones": ["asia-northeast1-b", "asia-northeast1-c"],
    "self_delete": True,
}
cost_guard = {
    "schema": "hu_m31_t3_step6d_spot_cost_guard_v1",
    "currency": "USD",
    "spot_price_ceiling_usd_per_vm_hour": 0.5,
    "max_runtime_seconds_per_vm": 3300,
    "internal_watchdog_seconds_per_vm": 3000,
    "max_logical_jobs": 20,
    "max_attempts_per_job": 2,
    "max_cumulative_vm_jobs": 40,
    "all_20_estimated_max_compute_usd": 9.166666666666666,
    "all_attempts_estimated_max_compute_usd": 18.333333333333332,
    "hard_tail_compute_cap_usd": 20.0,
    "m31_total_compute_cap_usd": 500.0,
}
require(m.get("launch_target") == a.get("launch_target") == launch_target, "launch target changed")
require(project == launch_target["project"] and bucket == launch_target["bucket"], "metadata cloud target changed")
require(zone in launch_target["zones"] and self_delete == 1, "metadata deletion target changed")
require(m.get("cost_guard") == a.get("cost_guard") == cost_guard, "cost guard changed")
require(digest(cost_guard) == cost_guard_sha, "metadata cost guard digest changed")
require(max_runtime_seconds == cost_guard["internal_watchdog_seconds_per_vm"], "metadata watchdog cap changed")
require(20 * 0.5 * cost_guard["max_runtime_seconds_per_vm"] / 3600 == cost_guard["all_20_estimated_max_compute_usd"], "all-20 cost math changed")
require(40 * 0.5 * cost_guard["max_runtime_seconds_per_vm"] / 3600 == cost_guard["all_attempts_estimated_max_compute_usd"], "all-attempt cost math changed")
require(max_runtime_seconds < cost_guard["max_runtime_seconds_per_vm"], "watchdog lacks cleanup margin")
require(cost_guard["all_20_estimated_max_compute_usd"] <= cost_guard["all_attempts_estimated_max_compute_usd"] <= cost_guard["hard_tail_compute_cap_usd"] <= cost_guard["m31_total_compute_cap_usd"], "cost cap nesting changed")
if attempt_index == 0:
    launch_claim_keys = {
        "schema", "status", "run_name", "selected_job_ids",
        "run_contract_digest", "launch_target", "cost_guard_sha256",
        "preflight_sha256", "claimed_unix_seconds", "crash_reuse_authorized",
    }
    require(set(attempt_claim) == launch_claim_keys, "launch claim keys changed")
    require(attempt_claim.get("schema") == "hu_m31_t3_step6d_spot_launch_claim_v1", "bad launch claim schema")
    require(attempt_claim.get("status") == "exclusive_claim_acquired_before_remote_mutation", "bad launch claim status")
    require(attempt_claim.get("run_name") == run_name, "launch claim run changed")
    require(attempt_claim.get("selected_job_ids") == job_ids, "launch claim job set changed")
    require(attempt_claim.get("run_contract_digest") == m.get("run_contract_digest"), "launch claim contract changed")
    require(attempt_claim.get("launch_target") == launch_target, "launch claim target changed")
    require(attempt_claim.get("cost_guard_sha256") == digest(cost_guard), "launch claim cost guard changed")
    require(isinstance(attempt_claim.get("preflight_sha256"), str) and len(attempt_claim["preflight_sha256"]) == 64 and all(ch in "0123456789abcdef" for ch in attempt_claim["preflight_sha256"]), "launch preflight digest changed")
    require(isinstance(attempt_claim.get("claimed_unix_seconds"), (int, float)) and not isinstance(attempt_claim.get("claimed_unix_seconds"), bool) and attempt_claim["claimed_unix_seconds"] > 0, "launch claim timestamp changed")
    require(attempt_claim.get("crash_reuse_authorized") is False, "launch crash reuse became authorized")
    require(attempt_claim_sha == initial_claim_sha and initial_result_sha == "none", "attempt0 claim chain changed")
else:
    resume_keys = {
        "schema", "status", "run_name", "attempt_index", "selected_job_ids",
        "initial_launch_claim_sha256", "initial_launch_result_sha256",
        "run_contract_digest", "launch_target", "cost_guard_sha256",
        "preflight_sha256", "claimed_unix_seconds", "third_attempt_authorized",
    }
    require(set(attempt_claim) == resume_keys, "resume claim keys changed")
    require(attempt_claim.get("schema") == "hu_m31_t3_step6d_spot_resume_claim_v1", "bad resume claim schema")
    require(attempt_claim.get("status") == "exclusive_attempt1_claim_acquired_before_remote_mutation", "bad resume claim status")
    require(attempt_claim.get("run_name") == run_name and attempt_claim.get("attempt_index") == 1, "resume identity changed")
    require(job_id in attempt_claim.get("selected_job_ids", []), "job absent from resume claim")
    require(attempt_claim.get("initial_launch_claim_sha256") == initial_claim_sha, "initial claim chain changed")
    require(attempt_claim.get("initial_launch_result_sha256") == initial_result_sha, "initial result chain changed")
    require(attempt_claim.get("run_contract_digest") == m.get("run_contract_digest"), "resume contract changed")
    require(attempt_claim.get("launch_target") == launch_target, "resume target changed")
    require(attempt_claim.get("cost_guard_sha256") == digest(cost_guard), "resume cost guard changed")
    require(isinstance(attempt_claim.get("preflight_sha256"), str) and len(attempt_claim["preflight_sha256"]) == 64, "resume preflight digest changed")
    require(isinstance(attempt_claim.get("claimed_unix_seconds"), (int, float)) and not isinstance(attempt_claim.get("claimed_unix_seconds"), bool) and attempt_claim["claimed_unix_seconds"] > 0, "resume timestamp changed")
    require(attempt_claim.get("third_attempt_authorized") is False, "third attempt became authorized")
require(m.get("source_name") == "ofc_regular_hu_m31_t3_step6d_v2_source.zip", "source name changed")
require(m.get("startup_name") == "startup_hu_m31_t3_step6d_v2.sh", "startup name changed")
require(m.get("image") == {
    "project": "debian-cloud",
    "name": "debian-12-bookworm-v20260609",
    "id": "1449487925682397051",
    "self_link": "https://www.googleapis.com/compute/v1/projects/debian-cloud/global/images/debian-12-bookworm-v20260609",
}, "image binding changed")
require(m.get("allocation") == {
    "machine_type": "c4-standard-16",
    "process_count": 1,
    "rayon_threads_per_process": 16,
    "omp_threads": 1,
    "m3_batch_threads": 1,
}, "allocation binding changed")
require(m.get("feature_encoder") == {
    "package_path": "target/release/libofc_stage3_feature_encoder.so",
    "sha256": "82510e563ee290cd536e7aebe6bcf0efefac061af5488851f0a582bb4ef83411",
}, "feature encoder binding changed")
require(m.get("tail_schedule") == {
    "contract_hand_indices": list(range(100)),
    "work_hand_indices": tail,
    "source_roles": roles,
    "mapping": "one_source_hand_per_vm",
    "logical_job_count": 20,
    "max_logical_jobs": 20,
}, "tail scheduling changed")
require(m.get("checkpoint") == {
    "unit": "completed_source_hand",
    "upload_after_each_hand": True,
    "immutable_write_once": True,
    "resume_verifies_all_artifacts": True,
    "done_uploaded_last": True,
}, "checkpoint contract changed")
require(m.get("heartbeat") == {
    "schema": "hu_m31_t3_step6d_spot_heartbeat_v2",
    "required": True,
    "interval_seconds": 60,
    "remote_scope": "progress_only",
}, "heartbeat contract changed")
require(set(j) == {"schema", "run_contract", "run_contract_digest", "source_role", "work_hand_indices"}, "job keys changed")
require(j.get("schema") == "hu_m31_t3_step6d_performance_shard_manifest_v2", "bad job schema")
require(j.get("run_contract_digest") == digest(j.get("run_contract")), "job contract digest mismatch")
require(j.get("run_contract") == m.get("run_contract"), "job contract differs from package")
require(j.get("run_contract_digest") == m.get("run_contract_digest") == a.get("run_contract_digest"), "shared contract mismatch")
require(j.get("source_role") in roles and j.get("work_hand_indices") in [[index] for index in tail], "job role/work changed")
role = j["source_role"]
hand = j["work_hand_indices"][0]
require(job_id == f"{role}-hand-{hand:03d}" and job_id in job_ids, "job id mapping changed")
expected_instance = f"{run_name}-j{job_ids.index(job_id):02d}" + ("" if attempt_index == 0 else "-a01")
require(instance_name == expected_instance, "instance metadata mapping changed")
require(contract.get("allocation") == {"workers": 1, "rayon_threads_per_worker": 16}, "runner allocation changed")
reference = m.get("accepted_reference")
candidate = m.get("accepted_candidate")
require(reference == {"package_path": "native/reference/release/libofc_hu_m3_engine.so", "sha256": "3f831615d9e751b8e1f266f14dd2009903b3ac2a4094f7e47616e0aa372a01b0"}, "reference binding changed")
require(isinstance(candidate, dict) and candidate.get("package_path") == "native/candidate/release/libofc_hu_m3_engine.so", "candidate path changed")
require(isinstance(candidate.get("sha256"), str) and len(candidate["sha256"]) == 64 and candidate["sha256"] == candidate["sha256"].lower() and all(ch in "0123456789abcdef" for ch in candidate["sha256"]), "candidate hash changed")
require(contract.get("reference_library_sha256") == reference["sha256"], "reference contract mismatch")
require(contract.get("candidate_library_sha256") == candidate.get("sha256"), "candidate contract mismatch")

records = m.get("job_manifests")
require(isinstance(records, list) and len(records) == 20, "job record count changed")
require([row.get("job_id") for row in records] == job_ids, "job record order changed")
require(all(set(row) == {"job_id", "source_role", "work_hand_indices", "path", "output_prefix", "sha256", "bytes"} for row in records), "job record keys changed")
record = [row for row in records if row.get("job_id") == job_id]
require(len(record) == 1 and record[0].get("sha256") == job_sha, "job record hash mismatch")
require(record[0].get("source_role") == role and record[0].get("work_hand_indices") == [hand], "job record mapping mismatch")
require(record[0].get("path") == f"jobs/{job_id}.json" and record[0].get("output_prefix") == f"jobs/{job_id}", "job record path changed")
require(record[0].get("bytes") == job_path.stat().st_size, "job record byte count changed")

entries = m.get("source_entries")
require(isinstance(entries, dict) and len(entries) == m.get("source_entry_count"), "source entries absent")
selection_relative = "configs/hu_joint_policy_m31_t3_candidate02_tail_v2_selection.json"
selection_sha = contract.get("selection_manifest_sha256")
if selection_sha is None:
    require(selection_relative not in entries, "unexpected tail selection manifest")
else:
    selection_entry = entries.get(selection_relative)
    require(
        selection_sha == "62cfbe2d95477ed7686a1b583997ee2e35ab23291fd338cfeac597e9a216fed1"
        and selection_entry == {"sha256": selection_sha, "bytes": 1523},
        "tail selection manifest binding changed",
    )
for relative, expected in entries.items():
    pure = pathlib.PurePosixPath(relative)
    require(not pure.is_absolute() and ".." not in pure.parts, "unsafe source entry")
    path = (work / pathlib.Path(*pure.parts)).resolve()
    require(path.is_relative_to(work) and path.is_file(), f"missing source entry: {relative}")
    data = path.read_bytes()
    require(len(data) == expected.get("bytes"), f"source size mismatch: {relative}")
    require(hashlib.sha256(data).hexdigest() == expected.get("sha256"), f"source hash mismatch: {relative}")

binding = candidate if role == "candidate" else reference
print(role)
print(hand)
print(binding["package_path"])
print(binding["sha256"])
print(j["run_contract_digest"])
PY
)
[[ "${#JOB_FIELDS[@]}" -eq 5 ]]
ROLE="${JOB_FIELDS[0]}"
HAND_INDEX="${JOB_FIELDS[1]}"
LIBRARY_RELATIVE="${JOB_FIELDS[2]}"
LIBRARY_SHA256="${JOB_FIELDS[3]}"
RUN_CONTRACT_DIGEST="${JOB_FIELDS[4]}"
printf -v HAND_PAD '%03d' "$HAND_INDEX"
[[ "$(sha "$WORK/$LIBRARY_RELATIVE")" == "$LIBRARY_SHA256" ]]

cd "$WORK"
python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --disable-pip-version-check --no-input \
  -r configs/hu_m43_attempt08_runtime_requirements.txt
python -m pip check
python - configs/hu_m43_attempt08_runtime_requirements.txt <<'PY'
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
        raise SystemExit(f"pinned dependency mismatch: {name} {actual} != {expected}")
PY

export PYTHONPATH="$WORK/src"
export RAYON_NUM_THREADS=16
export OFC_HU_M3_BATCH_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

restore_checkpoint() {
  local relative=$1
  local source="$PROGRESS_URI/$relative"
  local destination="$RESULT/$relative"
  local error
  error="$(mktemp)"
  if gcloud storage objects describe "$source" --project "$PROJECT_ID" \
      >/dev/null 2>"$error"; then
    mkdir -p "$(dirname "$destination")"
    gcloud storage cp "$source" "$destination" --project "$PROJECT_ID" \
      >/dev/null
  elif ! grep -Eqi 'not found|does not exist|no urls matched|404' "$error"; then
    cat "$error" >&2
    rm -f "$error"
    return 1
  fi
  rm -f "$error"
}

mkdir -p "$RESULT"
for relative in \
  run_contract.json \
  shard_manifest.json \
  "roots/hand_$HAND_PAD.json" \
  "hands/$ROLE/hand_$HAND_PAD.json"; do
  restore_checkpoint "$relative"
done
gcloud storage rsync --recursive "$RESULT_URI" "$RESULT" \
  --project "$PROJECT_ID" >/dev/null 2>&1 || true

heartbeat_once() {
  python - "$RESULT/heartbeat.json" "$RUN_NAME" "$JOB_ID" "$ROLE" \
    "$HAND_INDEX" "$RUN_CONTRACT_DIGEST" "$ATTEMPT_INDEX" <<'PY'
import json
import os
import pathlib
import sys
import time

path = pathlib.Path(sys.argv[1])
role = sys.argv[4]
hand_index = int(sys.argv[5])
hand_path = path.parent / "hands" / role / f"hand_{hand_index:03d}.json"
done_path = path.parent / "DONE.json"
completed = [hand_index] if hand_path.is_file() else []
value = {
    "schema": "hu_m31_t3_step6d_spot_heartbeat_v2",
    "run_name": sys.argv[2],
    "job_id": sys.argv[3],
    "source_role": role,
    "work_hand_indices": [hand_index],
    "run_contract_digest": sys.argv[6],
    "attempt_index": int(sys.argv[7]),
    "status": "done_validated_locally" if done_path.is_file() else ("hand_complete" if completed else "running"),
    "completed_hand_indices": completed,
    "unix_seconds": time.time(),
}
temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
os.replace(temporary, path)
PY
  gcloud storage cp "$RESULT/heartbeat.json" "$PROGRESS_URI/heartbeat.json" \
    --project "$PROJECT_ID" >/dev/null
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

start_pump
python -m ofc_regular.run_hu_m31_t3_step6d_performance_v2 \
  --shard-manifest /tmp/job.json \
  --repository-root "$WORK" \
  --output-dir "$RESULT" \
  --library "$WORK/$LIBRARY_RELATIVE"
stop_pump

# A second idempotent runner pass validates every restored/new immutable
# artifact and DONE against the same source-role contract before publication.
python -m ofc_regular.run_hu_m31_t3_step6d_performance_v2 \
  --shard-manifest /tmp/job.json \
  --repository-root "$WORK" \
  --output-dir "$RESULT" \
  --library "$WORK/$LIBRARY_RELATIVE" >/tmp/post_validation.json

ROOT_ARTIFACT="$RESULT/roots/hand_$HAND_PAD.json"
HAND_ARTIFACT="$RESULT/hands/$ROLE/hand_$HAND_PAD.json"
[[ -f "$RESULT/run_contract.json" && -f "$RESULT/shard_manifest.json" ]]
[[ -f "$ROOT_ARTIFACT" && -f "$HAND_ARTIFACT" && -f "$RESULT/DONE.json" ]]
mapfile -t RESULT_FILES < <(find "$RESULT" -type f ! -name heartbeat.json -printf '%P\n' | sort)
[[ "${#RESULT_FILES[@]}" -eq 5 ]]
[[ "${RESULT_FILES[0]}" == "DONE.json" ]]
[[ "${RESULT_FILES[1]}" == "hands/$ROLE/hand_$HAND_PAD.json" ]]
[[ "${RESULT_FILES[2]}" == "roots/hand_$HAND_PAD.json" ]]
[[ "${RESULT_FILES[3]}" == "run_contract.json" ]]
[[ "${RESULT_FILES[4]}" == "shard_manifest.json" ]]

# One logical job contains one hand, so this is the required per-hand
# checkpoint.  The scientific result copy is also write-once and byte-equal.
for relative in \
  run_contract.json \
  shard_manifest.json \
  "roots/hand_$HAND_PAD.json" \
  "hands/$ROLE/hand_$HAND_PAD.json"; do
  upload_once "$RESULT/$relative" "$PROGRESS_URI/$relative"
  upload_once "$RESULT/$relative" "$RESULT_URI/$relative"
done

# DONE is the commit marker and is always uploaded last.
heartbeat_once
upload_once "$RESULT/DONE.json" "$RESULT_URI/DONE.json"
REMOTE_DONE="$(mktemp)"
gcloud storage cp "$RESULT_URI/DONE.json" "$REMOTE_DONE" \
  --project "$PROJECT_ID" >/dev/null
[[ "$(sha "$RESULT/DONE.json")" == "$(sha "$REMOTE_DONE")" ]]
rm -f "$REMOTE_DONE"
VALIDATED_DONE=1
