#!/usr/bin/env bash
set -euo pipefail
umask 077

# One VM executes exactly one performance-lock-v4 wave-v2 job.  Scientific
# inputs are content-addressed and replayed locally before the runner starts.

META_HEADER='Metadata-Flavor: Google'
META_ATTR='http://metadata.google.internal/computeMetadata/v1/instance/attributes'
META_INSTANCE='http://metadata.google.internal/computeMetadata/v1/instance'
OWNED_ROOT="${OFC_FULL100_WAVE_V2_TEST_ROOT:-/var/lib/ofc-full100-wave-v2}"
LOG_ROOT="${OFC_FULL100_WAVE_V2_TEST_ROOT:-/var/log/ofc-full100-wave-v2}"
MAIN_PID=$$
WATCHDOG_PID=''
VALIDATED_DONE=0

meta_attr() {
  if [[ -n "${OFC_FULL100_WAVE_V2_TEST_METADATA:-}" ]]; then
    cat "$OFC_FULL100_WAVE_V2_TEST_METADATA/attributes/$1"
  else
    curl -fsS -H "$META_HEADER" "$META_ATTR/$1"
  fi
}

instance_meta() {
  if [[ -n "${OFC_FULL100_WAVE_V2_TEST_METADATA:-}" ]]; then
    cat "$OFC_FULL100_WAVE_V2_TEST_METADATA/instance/$1"
  else
    curl -fsS -H "$META_HEADER" "$META_INSTANCE/$1"
  fi
}

project_meta() {
  if [[ -n "${OFC_FULL100_WAVE_V2_TEST_METADATA:-}" ]]; then
    cat "$OFC_FULL100_WAVE_V2_TEST_METADATA/project/$1"
  else
    curl -fsS -H "$META_HEADER" \
      "http://metadata.google.internal/computeMetadata/v1/project/$1"
  fi
}

sha() {
  sha256sum "$1" | awk '{print $1}'
}

cleanup() {
  local code=$?
  trap - EXIT TERM INT
  set +e
  if [[ -n "$WATCHDOG_PID" ]]; then
    kill "$WATCHDOG_PID" >/dev/null 2>&1 || true
    wait "$WATCHDOG_PID" >/dev/null 2>&1 || true
  fi
  if [[ "$code" -eq 0 && "$VALIDATED_DONE" -ne 1 ]]; then
    code=1
  fi
  exit "$code"
}
trap cleanup EXIT
trap 'exit 124' TERM INT

mkdir -p "$OWNED_ROOT" "$LOG_ROOT"
chmod 0700 "$OWNED_ROOT" "$LOG_ROOT"
[[ ! -L "$OWNED_ROOT" ]]
OWNED_ROOT_REAL="$(realpath -e "$OWNED_ROOT")"
[[ "$OWNED_ROOT_REAL" == "$OWNED_ROOT" ]]
exec >>"$LOG_ROOT/startup.log" 2>&1

atomic_commit_file() {
  python3 - "$1" "$2" "$OWNED_ROOT_REAL" <<'PY'
import hashlib, os, pathlib, sys
temporary, destination = map(pathlib.Path, sys.argv[1:3])
owned = pathlib.Path(sys.argv[3]).resolve(strict=True)
destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
parent = destination.parent.resolve(strict=True)
if owned.is_symlink() or os.path.commonpath((str(owned), str(parent))) != str(owned):
    raise SystemExit("atomic destination escaped owned root")
if temporary.is_symlink() or not temporary.is_file() or destination.is_symlink():
    raise SystemExit("unsafe atomic path")
with temporary.open("rb") as stream:
    os.fsync(stream.fileno())
if destination.exists():
    if not destination.is_file():
        raise SystemExit("immutable destination is not a file")
    if hashlib.sha256(temporary.read_bytes()).digest() != hashlib.sha256(destination.read_bytes()).digest():
        raise SystemExit("immutable local content changed")
    temporary.unlink()
else:
    os.replace(temporary, destination)
    os.chmod(destination, 0o600)
fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
try:
    os.fsync(fd)
finally:
    os.close(fd)
PY
}

fsync_file_parent() {
  python3 - "$1" "$OWNED_ROOT_REAL" <<'PY'
import os, pathlib, sys
path = pathlib.Path(sys.argv[1])
owned = pathlib.Path(sys.argv[2]).resolve(strict=True)
resolved = path.resolve(strict=True)
if path.is_symlink() or not path.is_file():
    raise SystemExit("unsafe publish source")
if os.path.commonpath((str(owned), str(resolved))) != str(owned):
    raise SystemExit("publish source escaped owned root")
with path.open("rb") as stream:
    os.fsync(stream.fileno())
fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
try:
    os.fsync(fd)
finally:
    os.close(fd)
PY
}

token() {
  curl -fsS -H "$META_HEADER" \
    'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token' \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["access_token"])'
}

encoded_object() {
  python3 -c 'import sys,urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=""))' "$1"
}

gcs_state() {
  local object=$1 encoded response status
  encoded="$(encoded_object "$object")"
  response="$(mktemp "$OWNED_ROOT_REAL/.gcs-state.XXXXXX")"
  if ! status="$(curl -sS -o "$response" -w '%{http_code}' \
      -H "Authorization: Bearer $(token)" \
      "https://storage.googleapis.com/storage/v1/b/${BUCKET}/o/${encoded}")"; then
    cat "$response" >&2
    rm -f "$response"
    return 1
  fi
  case "$status" in
    200) rm -f "$response"; printf 'present\n' ;;
    404) rm -f "$response"; printf 'absent\n' ;;
    *) cat "$response" >&2; rm -f "$response"; return 1 ;;
  esac
}

download_bound() {
  local object=$1 expected_sha=$2 expected_bytes=$3 destination=$4 temporary encoded
  if [[ -f "$destination" ]]; then
    [[ "$(sha "$destination")" == "$expected_sha" ]]
    [[ "$(stat -c '%s' "$destination")" == "$expected_bytes" ]]
    return
  fi
  temporary="$(mktemp "$OWNED_ROOT_REAL/.download.XXXXXX")"
  encoded="$(encoded_object "$object")"
  curl -fsS -H "Authorization: Bearer $(token)" \
    "https://storage.googleapis.com/download/storage/v1/b/${BUCKET}/o/${encoded}?alt=media" \
    -o "$temporary"
  [[ "$(sha "$temporary")" == "$expected_sha" ]]
  [[ "$(stat -c '%s' "$temporary")" == "$expected_bytes" ]]
  atomic_commit_file "$temporary" "$destination"
}

restore_object() {
  local object=$1 destination=$2 temporary encoded
  [[ "$(gcs_state "$object")" == present ]] || return 1
  temporary="$(mktemp "$OWNED_ROOT_REAL/.restore.XXXXXX")"
  encoded="$(encoded_object "$object")"
  curl -fsS -H "Authorization: Bearer $(token)" \
    "https://storage.googleapis.com/download/storage/v1/b/${BUCKET}/o/${encoded}?alt=media" \
    -o "$temporary"
  atomic_commit_file "$temporary" "$destination"
}

upload_create_only() {
  local encoded
  fsync_file_parent "$1"
  encoded="$(encoded_object "$2")"
  curl -fsS -X POST -H "Authorization: Bearer $(token)" \
    -H 'Content-Type: application/octet-stream' \
    --data-binary "@$1" \
    "https://storage.googleapis.com/upload/storage/v1/b/${BUCKET}/o?uploadType=media&ifGenerationMatch=0&name=${encoded}" \
    >/dev/null
}

verify_remote_equal() {
  local source=$1 object=$2 temporary encoded
  temporary="$(mktemp "$OWNED_ROOT_REAL/.verify.XXXXXX")"
  encoded="$(encoded_object "$object")"
  curl -fsS -H "Authorization: Bearer $(token)" \
    "https://storage.googleapis.com/download/storage/v1/b/${BUCKET}/o/${encoded}?alt=media" \
    -o "$temporary"
  [[ "$(sha "$source")" == "$(sha "$temporary")" ]]
  [[ "$(stat -c '%s' "$source")" == "$(stat -c '%s' "$temporary")" ]]
  rm -f "$temporary"
}

JOB_BOOTSTRAP_B64="$(meta_attr job-bootstrap-b64)"
[[ -n "$JOB_BOOTSTRAP_B64" ]]
[[ "$JOB_BOOTSTRAP_B64" != *$'\n'* && "$JOB_BOOTSTRAP_B64" != *$'\r'* ]]
[[ "$JOB_BOOTSTRAP_B64" =~ ^[A-Za-z0-9+/]*={0,2}$ ]]
tmp="$(mktemp "$OWNED_ROOT_REAL/.bootstrap.XXXXXX")"
printf '%s' "$JOB_BOOTSTRAP_B64" | base64 --decode >"$tmp"
atomic_commit_file "$tmp" "$OWNED_ROOT_REAL/job-bootstrap.json"
BOOTSTRAP="$OWNED_ROOT_REAL/job-bootstrap.json"

EXECUTED_STARTUP_SHA256="$(sha "${BASH_SOURCE[0]}")"
ACTUAL_INSTANCE_NAME="$(instance_meta name)"
ACTUAL_WORKER_PRINCIPAL="$(instance_meta service-accounts/default/email)"

mapfile -t BOOT < <(
python3 - "$BOOTSTRAP" "$JOB_BOOTSTRAP_B64" "$EXECUTED_STARTUP_SHA256" \
  "$ACTUAL_INSTANCE_NAME" "$ACTUAL_WORKER_PRINCIPAL" <<'PY'
import base64, hashlib, json, pathlib, re, sys
raw = pathlib.Path(sys.argv[1]).read_bytes()
encoded, startup_sha, actual_instance, principal = sys.argv[2:6]
value = json.loads(raw.decode("ascii"))
canonical = json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False).encode("ascii")
if raw != canonical or base64.b64encode(raw).decode("ascii") != encoded:
    raise SystemExit("bootstrap is not canonical standard-base64 JSON")
keys = {
 "schema","status","run_name","execution_identity_sha256","wave_plan_sha256",
 "attempt_ledger_sha256","resume_sha256","observed_transition_digest",
 "wave_index","job_id","source_role","attempt_id","instance_name",
 "artifact_prefix","bucket","content_prefix","outer_manifest_sha256",
 "content_payload_sha256","scientific_source","scientific_manifest",
 "wheelhouse","wheelhouse_manifest","startup","wave_plan","job_manifest",
 "prelaunch_authorization_sha256","worker_principal",
 "one_vm_one_job_one_role","additional_create_authorized",
 "hidden_truth_exposed","bootstrap_sha256",
}
if set(value) != keys:
    raise SystemExit("bootstrap metadata fields changed")
digest = value.pop("bootstrap_sha256", None)
expected = hashlib.sha256(json.dumps(
 value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
 allow_nan=False).encode("ascii")).hexdigest()
value["bootstrap_sha256"] = digest
sha = re.compile(r"^[0-9a-f]{64}$")
if (
 digest != expected
 or value["schema"] != "hu_m31_t3_step6d_full100_wave_job_bootstrap_v2"
 or value["status"] != "single_job_bootstrap_bound_to_prelaunch_authorization"
 or not re.fullmatch(r"regular-hu-m31-c02-f100wv2-[a-z0-9-]{8,48}",
                     value["run_name"])
 or any(not sha.fullmatch(str(value[field])) for field in (
   "execution_identity_sha256","wave_plan_sha256","attempt_ledger_sha256",
   "resume_sha256","observed_transition_digest","outer_manifest_sha256",
   "content_payload_sha256","prelaunch_authorization_sha256"))
 or type(value["wave_index"]) is not int or value["wave_index"] not in (0,1,2)
 or not re.fullmatch(r"(candidate|reference)-shard-0[0-9]", value["job_id"])
 or value["source_role"] not in ("candidate","reference")
 or not value["job_id"].startswith(value["source_role"] + "-")
 or value["attempt_id"] not in ("a00","a01")
 or value["instance_name"] != actual_instance
 or value["worker_principal"] != principal
 or value["one_vm_one_job_one_role"] is not True
 or value["additional_create_authorized"] is not False
 or value["hidden_truth_exposed"] is not False
):
    raise SystemExit("bootstrap trust boundary changed")
suffix = f"/results/jobs/{value['job_id']}/attempts/{value['attempt_id']}"
for field in ("artifact_prefix","content_prefix"):
    item = value[field]
    if (not isinstance(item, str) or not item or item.startswith("/")
        or "\\" in item or any(part in ("", ".", "..") for part in item.split("/"))):
        raise SystemExit("unsafe bootstrap prefix")
if not value["artifact_prefix"].endswith(suffix):
    raise SystemExit("attempt-specific artifact prefix changed")
if not re.fullmatch(r"[a-z0-9][a-z0-9._-]{1,221}[a-z0-9]", value["bucket"]):
    raise SystemExit("unsafe bootstrap bucket")
bindings = ("scientific_source","scientific_manifest","wheelhouse",
            "wheelhouse_manifest","startup","wave_plan","job_manifest")
for field in bindings:
    binding = value[field]
    if (not isinstance(binding, dict)
        or set(binding) != {"object_name","sha256","bytes"}
        or not isinstance(binding["object_name"], str)
        or not binding["object_name"].startswith(value["content_prefix"] + "/")
        or "\\" in binding["object_name"]
        or any(part in ("", ".", "..") for part in binding["object_name"].split("/"))
        or not sha.fullmatch(str(binding["sha256"]))
        or type(binding["bytes"]) is not int or binding["bytes"] <= 0):
        raise SystemExit(f"invalid {field} object binding")
if value["startup"]["sha256"] != startup_sha:
    raise SystemExit("executed startup bytes changed")
if not value["job_manifest"]["object_name"].endswith(
        f"/content/jobs/{value['job_id']}.json"):
    raise SystemExit("extra/different job selected")
suffixes = {
    "scientific_source": "/content/science/source.zip",
    "scientific_manifest": "/content/science/manifest.json",
    "wheelhouse": "/content/wheelhouse/wheelhouse.zip",
    "wheelhouse_manifest": "/content/wheelhouse/wheelhouse_manifest.json",
    "startup": (
      "/content/startup/"
      "startup_hu_m31_t3_step6d_performance_lock_v4_wave_v2.sh"
    ),
    "wave_plan": "/control/wave_plan.json",
}
if any(not value[field]["object_name"].endswith(suffix)
       for field, suffix in suffixes.items()):
    raise SystemExit("bootstrap object topology changed")
for field in (
 "run_name","execution_identity_sha256","wave_plan_sha256",
 "attempt_ledger_sha256","resume_sha256","observed_transition_digest",
 "wave_index","job_id","source_role","attempt_id","instance_name",
 "artifact_prefix","bucket","content_prefix","outer_manifest_sha256",
 "content_payload_sha256","prelaunch_authorization_sha256","worker_principal"):
    print(value[field])
for field in bindings:
    print(value[field]["object_name"])
    print(value[field]["sha256"])
    print(value[field]["bytes"])
PY
)
[[ "${#BOOT[@]}" -eq 39 ]]
RUN_NAME="${BOOT[0]}"; EXECUTION_IDENTITY_SHA256="${BOOT[1]}"
WAVE_PLAN_SHA256="${BOOT[2]}"; ATTEMPT_LEDGER_SHA256="${BOOT[3]}"
RESUME_SHA256="${BOOT[4]}"; OBSERVED_TRANSITION_DIGEST="${BOOT[5]}"
WAVE_INDEX="${BOOT[6]}"; JOB_ID="${BOOT[7]}"; SOURCE_ROLE="${BOOT[8]}"
ATTEMPT_ID="${BOOT[9]}"; INSTANCE_NAME="${BOOT[10]}"
ARTIFACT_PREFIX="${BOOT[11]}"; BUCKET="${BOOT[12]}"
CONTENT_PREFIX="${BOOT[13]}"; OUTER_MANIFEST_SHA256="${BOOT[14]}"
CONTENT_PAYLOAD_SHA256="${BOOT[15]}"
PRELAUNCH_AUTHORIZATION_SHA256="${BOOT[16]}"; WORKER_PRINCIPAL="${BOOT[17]}"
SCIENTIFIC_SOURCE_OBJECT="${BOOT[18]}"; SCIENTIFIC_SOURCE_SHA256="${BOOT[19]}"; SCIENTIFIC_SOURCE_BYTES="${BOOT[20]}"
SCIENTIFIC_MANIFEST_OBJECT="${BOOT[21]}"; SCIENTIFIC_MANIFEST_SHA256="${BOOT[22]}"; SCIENTIFIC_MANIFEST_BYTES="${BOOT[23]}"
WHEELHOUSE_OBJECT="${BOOT[24]}"; WHEELHOUSE_SHA256="${BOOT[25]}"; WHEELHOUSE_BYTES="${BOOT[26]}"
WHEELHOUSE_MANIFEST_OBJECT="${BOOT[27]}"; WHEELHOUSE_MANIFEST_SHA256="${BOOT[28]}"; WHEELHOUSE_MANIFEST_BYTES="${BOOT[29]}"
STARTUP_OBJECT="${BOOT[30]}"; STARTUP_SHA256="${BOOT[31]}"; STARTUP_BYTES="${BOOT[32]}"
WAVE_PLAN_OBJECT="${BOOT[33]}"; WAVE_PLAN_OBJECT_SHA256="${BOOT[34]}"; WAVE_PLAN_BYTES="${BOOT[35]}"
JOB_MANIFEST_OBJECT="${BOOT[36]}"; JOB_MANIFEST_SHA256="${BOOT[37]}"; JOB_MANIFEST_BYTES="${BOOT[38]}"
[[ "$STARTUP_SHA256" == "$EXECUTED_STARTUP_SHA256" ]]
[[ "$(stat -c '%s' "${BASH_SOURCE[0]}")" == "$STARTUP_BYTES" ]]

JOB_ROOT="$OWNED_ROOT_REAL/runs/$RUN_NAME/$JOB_ID/$ATTEMPT_ID"
mkdir -p "$JOB_ROOT"/{content,science,wheelhouse,result,state}
chmod 0700 "$JOB_ROOT" "$JOB_ROOT"/*
JOB_ROOT_REAL="$(realpath -e "$JOB_ROOT")"
SCIENCE="$JOB_ROOT_REAL/science"; WHEELS="$JOB_ROOT_REAL/wheelhouse"
RESULT="$JOB_ROOT_REAL/result"; STATE="$JOB_ROOT_REAL/state"

( sleep 4200; kill -TERM "$MAIN_PID" >/dev/null 2>&1 || true ) &
WATCHDOG_PID=$!

printf '%s\n' \
  'deb [check-valid-until=no] https://snapshot.debian.org/archive/debian/20260722T000000Z bookworm main' \
  'deb [check-valid-until=no] https://snapshot.debian.org/archive/debian-security/20260722T000000Z bookworm-security main' \
  > /etc/apt/sources.list
apt-get -o Acquire::Check-Valid-Until=false update -y
apt-get -o Acquire::Check-Valid-Until=false install -y \
  python3 python3-venv unzip libgomp1 ca-certificates

download_bound "$SCIENTIFIC_SOURCE_OBJECT" "$SCIENTIFIC_SOURCE_SHA256" \
  "$SCIENTIFIC_SOURCE_BYTES" "$JOB_ROOT_REAL/content/source.zip"
download_bound "$SCIENTIFIC_MANIFEST_OBJECT" "$SCIENTIFIC_MANIFEST_SHA256" \
  "$SCIENTIFIC_MANIFEST_BYTES" "$JOB_ROOT_REAL/content/scientific_manifest.json"
download_bound "$WHEELHOUSE_OBJECT" "$WHEELHOUSE_SHA256" \
  "$WHEELHOUSE_BYTES" "$JOB_ROOT_REAL/content/wheelhouse.zip"
download_bound "$WHEELHOUSE_MANIFEST_OBJECT" "$WHEELHOUSE_MANIFEST_SHA256" \
  "$WHEELHOUSE_MANIFEST_BYTES" "$JOB_ROOT_REAL/content/wheelhouse_manifest.json"
download_bound "$WAVE_PLAN_OBJECT" "$WAVE_PLAN_OBJECT_SHA256" \
  "$WAVE_PLAN_BYTES" "$JOB_ROOT_REAL/content/wave_plan.json"
download_bound "$JOB_MANIFEST_OBJECT" "$JOB_MANIFEST_SHA256" \
  "$JOB_MANIFEST_BYTES" "$JOB_ROOT_REAL/content/job_manifest.json"

# Phase 1: verify the closed-world archives before importing any archive code.
python3 - "$JOB_ROOT_REAL" "$SCIENTIFIC_SOURCE_SHA256" \
  "$SCIENTIFIC_SOURCE_BYTES" "$SCIENTIFIC_MANIFEST_SHA256" \
  "$WAVE_PLAN_OBJECT_SHA256" "$JOB_MANIFEST_SHA256" "$WHEELHOUSE_SHA256" \
  "$WHEELHOUSE_MANIFEST_SHA256" <<'PY'
import hashlib, json, os, pathlib, re, stat, sys, tempfile, zipfile
root = pathlib.Path(sys.argv[1]).resolve(strict=True)
(source_sha, source_bytes, manifest_sha, wave_sha, job_sha,
 wheel_sha, wheel_manifest_sha) = sys.argv[2:9]
content, science, wheels = root / "content", root / "science", root / "wheelhouse"
sha_re = re.compile(r"^[0-9a-f]{64}$")

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii") + b"\n"
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path, label):
    raw = path.read_bytes()
    value = json.loads(raw.decode("ascii"))
    if not isinstance(value, dict) or raw != canonical(value):
        raise SystemExit(f"{label} is not canonical JSON")
    return value
def safe_relative(name):
    if (not isinstance(name, str) or not name or "\\" in name
        or name.startswith("/") or "\x00" in name
        or any(part in ("", ".", "..") for part in name.split("/"))):
        raise SystemExit("unsafe archive member")
    return pathlib.PurePosixPath(name)
def atomic_write(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink():
        raise SystemExit("archive destination is a symlink")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        if path.exists():
            if not path.is_file() or path.read_bytes() != raw:
                raise SystemExit("immutable extracted content changed")
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

source = content / "source.zip"
manifest_path = content / "scientific_manifest.json"
wave_path = content / "wave_plan.json"
job_path = content / "job_manifest.json"
wheel_archive = content / "wheelhouse.zip"
wheel_manifest_path = content / "wheelhouse_manifest.json"
if (digest(source) != source_sha or source.stat().st_size != int(source_bytes)
    or digest(manifest_path) != manifest_sha or digest(wave_path) != wave_sha
    or digest(job_path) != job_sha or digest(wheel_archive) != wheel_sha
    or digest(wheel_manifest_path) != wheel_manifest_sha):
    raise SystemExit("downloaded object hash/size binding changed")
manifest = load(manifest_path, "scientific manifest")
wave = load(wave_path, "wave plan")
load(job_path, "selected job manifest")
wheel_manifest = load(wheel_manifest_path, "wheelhouse manifest")
if (manifest.get("schema")
      != "hu_m31_t3_step6d_performance_lock_v4_spot_package_v1"
    or manifest.get("source_sha256") != source_sha
    or manifest.get("source_bytes") != int(source_bytes)
    or manifest.get("plan_sha256")
      != "2ad08116835a58f5b5927e4de986f2717915d0dd128e7a5f3fa288e0cac6e5be"
    or manifest.get("run_contract_digest")
      != "669c1efa1afeebe41fcc531c6458c9d72fffdd5df2cca751c99988a872f3e2b6"
    or manifest.get("job_count") != 20
    or len(manifest.get("job_manifests", [])) != 20
    or wave.get("schema") != "hu_m31_t3_step6d_full100_wave_plan_v2"
    or wave.get("full100_plan", {}).get("schema")
      != "hu_m31_t3_step6d_candidate02_performance_lock_plan_v4"
    or wave.get("full100_plan_sha256") != manifest.get("plan_sha256")
    or wave.get("run_contract_digest") != manifest.get("run_contract_digest")
    or wave.get("current_profile_changed") is not False
    or wave.get("cloud_launch_authorized") is not False):
    raise SystemExit("v4 scientific/wave schema binding changed")
entries = manifest.get("source_entries")
if not isinstance(entries, dict) or manifest.get("source_entry_count") != len(entries):
    raise SystemExit("v4 scientific inventory changed")
with zipfile.ZipFile(source) as archive:
    infos = archive.infolist()
    names = [item.filename for item in infos]
    if set(names) != set(entries) or len(names) != len(set(names)):
        raise SystemExit("scientific archive missing/extra member")
    for info in infos:
        relative = safe_relative(info.filename)
        mode = info.external_attr >> 16
        record = entries[info.filename]
        raw = archive.read(info)
        if (info.is_dir() or stat.S_ISLNK(mode) or not isinstance(record, dict)
            or set(record) != {"sha256", "bytes"}
            or not sha_re.fullmatch(str(record["sha256"]))
            or record["bytes"] != len(raw)
            or hashlib.sha256(raw).hexdigest() != record["sha256"]):
            raise SystemExit("scientific archive member contract changed")
        atomic_write(science.joinpath(*relative.parts), raw)
wheel_entries = wheel_manifest.get("entries")
if (wheel_manifest.get("schema") != "hu_m31_t3_step6d_perfdev_v2_wheelhouse_v1"
    or wheel_manifest.get("status") != "complete_hash_pinned_offline_wheelhouse"
    or wheel_manifest.get("python_abi") != "cp311"
    or wheel_manifest.get("target_os") != "linux"
    or wheel_manifest.get("target_architecture") != "x86_64"
    or wheel_manifest.get("network_install_allowed") is not False
    or not isinstance(wheel_entries, list)
    or wheel_manifest.get("entry_count") != len(wheel_entries)
    or wheel_manifest.get("entries_sha256")
       != hashlib.sha256(canonical(wheel_entries)).hexdigest()):
    raise SystemExit("offline wheelhouse manifest changed")
expected_wheels = {row.get("filename"): row for row in wheel_entries
                   if isinstance(row, dict)}
with zipfile.ZipFile(wheel_archive) as archive:
    names = archive.namelist()
    if set(names) != set(expected_wheels) or len(names) != len(set(names)):
        raise SystemExit("wheelhouse archive missing/extra member")
    for name in names:
        raw = archive.read(name); record = expected_wheels[name]
        if (not re.fullmatch(r"[A-Za-z0-9_.+!-]+\.whl", name)
            or len(raw) != record.get("bytes")
            or hashlib.sha256(raw).hexdigest() != record.get("sha256")):
            raise SystemExit("wheelhouse archive member changed")
        atomic_write(wheels.joinpath(*safe_relative(name).parts), raw)
PY

python3 -m venv "$JOB_ROOT_REAL/venv"
source "$JOB_ROOT_REAL/venv/bin/activate"
mapfile -t WHEEL_FILES < <(
python3 - "$JOB_ROOT_REAL/content/wheelhouse_manifest.json" "$WHEELS" <<'PY'
import json, pathlib, sys
manifest = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="ascii"))
root = pathlib.Path(sys.argv[2]).resolve(strict=True)
for record in manifest["entries"]:
    path = (root / record["filename"]).resolve(strict=True)
    if path.parent != root or not path.is_file() or path.is_symlink():
        raise SystemExit("offline wheel path changed")
    print(path)
PY
)
[[ "${#WHEEL_FILES[@]}" -gt 0 ]]
python -m pip install --disable-pip-version-check --no-input --no-index \
  --find-links "$WHEELS" "${WHEEL_FILES[@]}"
python -m pip check
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$SCIENCE/src"
export RAYON_NUM_THREADS=16
export OFC_HU_M3_BATCH_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Phase 2: replay the exact v4 validators and all 20 reconstructed job
# manifests from the now hash-verified source tree.
mapfile -t JOB < <(
python - "$JOB_ROOT_REAL" "$BOOTSTRAP" "$JOB_MANIFEST_SHA256" <<'PY'
import hashlib, json, pathlib, sys
from ofc_regular import hu_m31_t3_step6d_performance_lock_v4_spot_package as package

root = pathlib.Path(sys.argv[1]).resolve(strict=True)
bootstrap = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="ascii"))
selected_sha = sys.argv[3]
content, science = root / "content", root / "science"
manifest = package._validate_manifest_value(
    package._read_canonical_file(
        content / "scientific_manifest.json", "worker scientific manifest"
    )
)
payloads = {
    relative: science.joinpath(*relative.split("/")).read_bytes()
    for relative in manifest["source_entries"]
}
plan = package._validate_archived_science(payloads, manifest)
expected_payloads, expected_records = package._job_payloads(plan)
if manifest["job_manifests"] != expected_records:
    raise SystemExit("all 20 v4 job manifest records changed")
job_id = bootstrap["job_id"]
relative = f"jobs/{job_id}.json"
selected_raw = (content / "job_manifest.json").read_bytes()
selected = json.loads(selected_raw.decode("ascii"))
frozen = {row["job_id"]: row for row in plan["jobs"]}.get(job_id)
record = {row["job_id"]: row for row in expected_records}.get(job_id)
wave = package._read_canonical_file(content / "wave_plan.json", "worker wave plan")
if (frozen is None or record is None
    or selected_raw != expected_payloads.get(relative)
    or hashlib.sha256(selected_raw).hexdigest() != selected_sha
    or selected_sha != frozen["shard_manifest_sha256"]
    or selected.get("source_role") != bootstrap["source_role"]
    or selected.get("work_hand_indices") != frozen["work_hand_indices"]
    or selected.get("run_contract_digest") != package.v4.RUN_CONTRACT_DIGEST
    or selected.get("run_contract") != plan["run_contract"]
    or wave.get("full100_plan") != plan
    or wave.get("full100_plan_sha256") != package.v4.PLAN_SHA256
    or wave.get("run_contract_digest") != package.v4.RUN_CONTRACT_DIGEST):
    raise SystemExit("selected v4 job/wave binding changed")
role = bootstrap["source_role"]
binding = manifest[f"accepted_{role}"]
library = science.joinpath(*binding["path"].split("/"))
if (hashlib.sha256(library.read_bytes()).hexdigest() != binding["sha256"]
    or wave["runtime_binding"]["binary_sha256_by_role"][role] != binding["sha256"]
    or wave["runtime_binding"]["package_sha256"] != manifest["source_sha256"]
    or plan["allocation"] != {"workers": 1, "rayon_threads_per_worker": 16}):
    raise SystemExit("v4 binary/allocation binding changed")
print(role)
print(",".join(str(i) for i in frozen["work_hand_indices"]))
print(binding["path"])
print(binding["sha256"])
print(package.v4.RUN_CONTRACT_DIGEST)
print(manifest["source_sha256"])
print(wave["runtime_binding"]["image_digest"])
print(wave["runtime_binding"]["allocation_digest"])
PY
)
[[ "${#JOB[@]}" -eq 8 ]]
ROLE="${JOB[0]}"; IFS=',' read -r -a WORK_HANDS <<<"${JOB[1]}"
LIBRARY_RELATIVE="${JOB[2]}"; LIBRARY_SHA256="${JOB[3]}"
RUN_CONTRACT_DIGEST="${JOB[4]}"; PACKAGE_SHA256="${JOB[5]}"
IMAGE_DIGEST="${JOB[6]}"; ALLOCATION_DIGEST="${JOB[7]}"
[[ "$ROLE" == "$SOURCE_ROLE" && "${#WORK_HANDS[@]}" -eq 10 ]]

mkdir -p "$RESULT/roots" "$RESULT/hands/$ROLE"
declare -A ROOT_PUBLISHED HAND_PUBLISHED
for hand in "${WORK_HANDS[@]}"; do
  printf -v pad '%03d' "$hand"
  tmp="$(mktemp "$OWNED_ROOT_REAL/.root.XXXXXX")"
  cp --no-preserve=mode,ownership,timestamps \
    "$SCIENCE/frozen/roots/hand_$pad.json" "$tmp"
  atomic_commit_file "$tmp" "$RESULT/roots/hand_$pad.json"
  if restore_object "$ARTIFACT_PREFIX/roots/hand_$pad.json" \
      "$RESULT/roots/hand_$pad.json"; then ROOT_PUBLISHED[$hand]=1
  else ROOT_PUBLISHED[$hand]=0; fi
  if restore_object "$ARTIFACT_PREFIX/hands/$ROLE/hand_$pad.json" \
      "$RESULT/hands/$ROLE/hand_$pad.json"; then
    HAND_PUBLISHED[$hand]=1
    [[ "${ROOT_PUBLISHED[$hand]}" -eq 1 ]]
  else HAND_PUBLISHED[$hand]=0; fi
done

REMOTE_DONE="$STATE/remote_DONE.json"
REMOTE_DONE_PRESENT=0
if restore_object "$ARTIFACT_PREFIX/DONE.json" "$REMOTE_DONE"; then
  REMOTE_DONE_PRESENT=1
  for hand in "${WORK_HANDS[@]}"; do
    [[ "${ROOT_PUBLISHED[$hand]}" -eq 1 && "${HAND_PUBLISHED[$hand]}" -eq 1 ]]
  done
fi

checkpoint_hand() {
  local hand=$1 pad
  printf -v pad '%03d' "$hand"
  [[ -f "$RESULT/roots/hand_$pad.json" ]]
  [[ -f "$RESULT/hands/$ROLE/hand_$pad.json" ]]
  if [[ "${ROOT_PUBLISHED[$hand]}" -eq 0 ]]; then
    upload_create_only "$RESULT/roots/hand_$pad.json" \
      "$ARTIFACT_PREFIX/roots/hand_$pad.json"
    ROOT_PUBLISHED[$hand]=1
  fi
  if [[ "${HAND_PUBLISHED[$hand]}" -eq 0 ]]; then
    upload_create_only "$RESULT/hands/$ROLE/hand_$pad.json" \
      "$ARTIFACT_PREFIX/hands/$ROLE/hand_$pad.json"
    HAND_PUBLISHED[$hand]=1
  fi
}

for hand in "${WORK_HANDS[@]}"; do
  printf -v pad '%03d' "$hand"
  if [[ ! -f "$RESULT/hands/$ROLE/hand_$pad.json" ]]; then
    python -m ofc_regular.run_hu_m31_t3_step6d_performance_v2 \
      --shard-manifest "$JOB_ROOT_REAL/content/job_manifest.json" \
      --repository-root "$SCIENCE" --output-dir "$RESULT" \
      --library "$SCIENCE/$LIBRARY_RELATIVE" --stop-after-hands 1
    [[ -f "$RESULT/hands/$ROLE/hand_$pad.json" ]]
  fi
  checkpoint_hand "$hand"
done

python -m ofc_regular.run_hu_m31_t3_step6d_performance_v2 \
  --shard-manifest "$JOB_ROOT_REAL/content/job_manifest.json" \
  --repository-root "$SCIENCE" --output-dir "$RESULT" \
  --library "$SCIENCE/$LIBRARY_RELATIVE" >"$STATE/runner_validation.json"

python3 - "$RESULT" "$STATE/runner_validation.json" \
  "$STATE/transport_DONE.json" "$RUN_NAME" "$EXECUTION_IDENTITY_SHA256" \
  "$JOB_ID" "$ROLE" "$ATTEMPT_ID" "$PACKAGE_SHA256" "$IMAGE_DIGEST" \
  "$LIBRARY_SHA256" "$ALLOCATION_DIGEST" "$RUN_CONTRACT_DIGEST" \
  "$WAVE_PLAN_SHA256" "$ATTEMPT_LEDGER_SHA256" "$RESUME_SHA256" \
  "$OBSERVED_TRANSITION_DIGEST" "$WAVE_INDEX" "$CONTENT_PAYLOAD_SHA256" \
  "$OUTER_MANIFEST_SHA256" "$PRELAUNCH_AUTHORIZATION_SHA256" \
  "$WORKER_PRINCIPAL" "${JOB[1]}" "$OWNED_ROOT_REAL" <<'PY'
import hashlib, json, os, pathlib, sys, tempfile
result = pathlib.Path(sys.argv[1])
report_path, output = map(pathlib.Path, sys.argv[2:4])
(run_name, identity, job_id, role, attempt_id, package_sha, image_digest,
 binary_sha, allocation_digest, run_digest, wave_sha, ledger_sha, resume_sha,
 transition_digest, wave_index, content_sha, outer_sha, auth_sha, principal,
 hand_csv, owned_root) = sys.argv[4:25]
work = [int(value) for value in hand_csv.split(",")]
report = json.loads(report_path.read_text(encoding="utf-8"))
runner_done_path = result / "DONE.json"
runner_done = json.loads(runner_done_path.read_text(encoding="utf-8"))
def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")
def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
def exposes_hidden(value):
    forbidden = {"opponent_private_discards", "opponent_hidden_discards",
                 "realized_deck_tail", "hidden_truth", "unknown_deck_order"}
    if isinstance(value, dict):
        return any(key in forbidden or exposes_hidden(item)
                   for key, item in value.items())
    if isinstance(value, list):
        return any(exposes_hidden(item) for item in value)
    return False
if (not isinstance(report, dict)
    or report.get("status") != "complete_source_isolated_shard"
    or report.get("source_role") != role
    or report.get("work_hand_indices") != work
    or report.get("completed_hand_indices") != work
    or report.get("training_eligible") is not False
    or report.get("quality_evidence") is not False
    or report.get("promotion_evidence") is not False
    or report.get("current_profile_changed") is not False
    or exposes_hidden(report) or exposes_hidden(runner_done)):
    raise SystemExit("runner completion/hidden-truth boundary changed")
roots, artifacts = [], []
for hand in work:
    root = result / "roots" / f"hand_{hand:03d}.json"
    source_hand = result / "hands" / role / f"hand_{hand:03d}.json"
    if not root.is_file() or not source_hand.is_file():
        raise SystemExit("complete hand artifact is missing")
    if exposes_hidden(json.loads(source_hand.read_text(encoding="utf-8"))):
        raise SystemExit("source result exposed hidden truth")
    roots.append({"hand_index": hand, "sha256": file_sha(root)})
    artifacts.extend((
        {"path": f"roots/hand_{hand:03d}.json",
         "sha256": file_sha(root), "bytes": root.stat().st_size},
        {"path": f"hands/{role}/hand_{hand:03d}.json",
         "sha256": file_sha(source_hand), "bytes": source_hand.stat().st_size},
    ))
root_digest = hashlib.sha256(canonical(roots) + b"\n").hexdigest()
identity_value = {
    "schema": "hu_m31_t3_step6d_full100_done_identity_v1",
    "run_name": run_name, "execution_identity_sha256": identity,
    "job_id": job_id, "source_role": role, "attempt_id": attempt_id,
    "package_sha256": package_sha, "image_digest": image_digest,
    "binary_sha256": binary_sha, "allocation_digest": allocation_digest,
    "root_digest": root_digest,
}
done_identity = hashlib.sha256(canonical(identity_value) + b"\n").hexdigest()
value = {
    "schema": "hu_m31_t3_step6d_full100_wave_attempt_done_v2",
    "status": "complete_validated_single_job_attempt",
    "run_name": run_name, "execution_identity_sha256": identity,
    "wave_plan_sha256": wave_sha, "attempt_ledger_sha256": ledger_sha,
    "resume_sha256": resume_sha, "observed_transition_digest": transition_digest,
    "wave_index": int(wave_index), "job_id": job_id,
    "source_role": role, "attempt_id": attempt_id,
    "package_sha256": package_sha, "image_digest": image_digest,
    "binary_sha256": binary_sha, "allocation_digest": allocation_digest,
    "run_contract_digest": run_digest, "root_digest": root_digest,
    "done_identity_sha256": done_identity,
    "content_payload_sha256": content_sha, "outer_manifest_sha256": outer_sha,
    "prelaunch_authorization_sha256": auth_sha, "worker_principal": principal,
    "work_hand_indices": work, "artifact_count": len(artifacts),
    "artifacts": artifacts, "runner_done_sha256": file_sha(runner_done_path),
    "metadata_hidden_truth_exposed": False,
    "opponent_private_discards_used": False,
    "training_eligible": False, "quality_evidence": False,
    "promotion_evidence": False, "current_profile_changed": False,
}
raw = canonical(value)
owned = pathlib.Path(owned_root).resolve(strict=True)
if os.path.commonpath((str(owned), str(output.parent.resolve(strict=True)))) != str(owned):
    raise SystemExit("DONE destination escaped owned root")
descriptor, temporary_name = tempfile.mkstemp(
    prefix=".DONE.", suffix=".tmp", dir=output.parent)
temporary = pathlib.Path(temporary_name)
try:
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, output)
    fd = os.open(output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
finally:
    temporary.unlink(missing_ok=True)
PY

for hand in "${WORK_HANDS[@]}"; do
  printf -v pad '%03d' "$hand"
  verify_remote_equal "$RESULT/roots/hand_$pad.json" \
    "$ARTIFACT_PREFIX/roots/hand_$pad.json"
  verify_remote_equal "$RESULT/hands/$ROLE/hand_$pad.json" \
    "$ARTIFACT_PREFIX/hands/$ROLE/hand_$pad.json"
done

TRANSPORT_DONE="$STATE/transport_DONE.json"
if [[ "$REMOTE_DONE_PRESENT" -eq 1 ]]; then
  [[ "$(sha "$TRANSPORT_DONE")" == "$(sha "$REMOTE_DONE")" ]]
else
  upload_create_only "$TRANSPORT_DONE" "$ARTIFACT_PREFIX/DONE.json"
fi
verify_remote_equal "$TRANSPORT_DONE" "$ARTIFACT_PREFIX/DONE.json"
VALIDATED_DONE=1
exit 0
