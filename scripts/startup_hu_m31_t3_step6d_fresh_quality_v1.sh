#!/usr/bin/env bash
set -euo pipefail
umask 077

# Local/cloud-neutral startup boundary for exactly one fresh-quality job.
# A later GCE adapter may download the immutable staging tree, but this script
# itself has no cloud client and performs no network operation.
#
# Usage:
#   startup...sh STAGING_ROOT LAUNCH_MANIFEST JOB_ID WORK_ROOT \
#     [validate-only|run]

if [[ "$#" -lt 4 || "$#" -gt 5 ]]; then
  echo "usage: $0 STAGING_ROOT LAUNCH_MANIFEST JOB_ID WORK_ROOT [validate-only|run]" >&2
  exit 64
fi

STAGING_ROOT="$(realpath -e "$1")"
LAUNCH_MANIFEST="$(realpath -e "$2")"
JOB_ID="$3"
WORK_ROOT="$4"
MODE="${5:-run}"
[[ "$MODE" == "validate-only" || "$MODE" == "run" ]]
[[ -d "$STAGING_ROOT" && ! -L "$STAGING_ROOT" ]]
[[ -f "$LAUNCH_MANIFEST" && ! -L "$LAUNCH_MANIFEST" ]]

mkdir -p "$WORK_ROOT"
WORK_ROOT="$(realpath -e "$WORK_ROOT")"
[[ "$WORK_ROOT" != "/" && -d "$WORK_ROOT" && ! -L "$WORK_ROOT" ]]
case "$WORK_ROOT/" in
  "$STAGING_ROOT/"*) echo "work root must not be inside immutable staging" >&2; exit 65 ;;
esac

SCIENCE="$WORK_ROOT/science"
WHEELS="$WORK_ROOT/wheelhouse"
mkdir -p "$SCIENCE" "$WHEELS"

mapfile -t BINDING < <(
python3 - "$STAGING_ROOT" "$LAUNCH_MANIFEST" "$JOB_ID" "$SCIENCE" "$WHEELS" <<'PY'
import hashlib
import json
import os
import pathlib
import re
import sys
import zipfile

staging = pathlib.Path(sys.argv[1]).resolve(strict=True)
launch_path = pathlib.Path(sys.argv[2]).resolve(strict=True)
job_id = sys.argv[3]
science = pathlib.Path(sys.argv[4]).resolve(strict=True)
wheels = pathlib.Path(sys.argv[5]).resolve(strict=True)

def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def load_canonical(path, label):
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except Exception as exc:
        raise SystemExit(f"{label} is not JSON: {exc}")
    if not isinstance(value, dict) or raw != canonical(value):
        raise SystemExit(f"{label} is not canonical")
    return value

def safe_relative(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise SystemExit("unsafe relative path")
    path = pathlib.PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != value:
        raise SystemExit("unsafe relative path")
    return path

def staged_file(record, label):
    if not isinstance(record, dict) or set(record) != {"path", "sha256", "bytes"}:
        raise SystemExit(f"{label} record changed")
    relative = safe_relative(record["path"])
    path = staging.joinpath(*relative.parts).resolve(strict=True)
    if (path.is_symlink() or not path.is_file()
            or os.path.commonpath((str(staging), str(path))) != str(staging)
            or digest(path.read_bytes()) != record["sha256"]
            or path.stat().st_size != record["bytes"]):
        raise SystemExit(f"{label} artifact changed")
    return path

def zip_payloads(path, *, deterministic):
    payloads = {}
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if names != sorted(names) or len(names) != len(set(names)):
            raise SystemExit("archive ordering/uniqueness changed")
        for info in infos:
            relative = safe_relative(info.filename)
            mode = (info.external_attr >> 16) & 0xFFFF
            if info.is_dir() or (mode & 0o170000) not in (0, 0o100000):
                raise SystemExit("archive contains unsafe member")
            if deterministic and info.date_time != (1980, 1, 1, 0, 0, 0):
                raise SystemExit("archive timestamp changed")
            payloads[relative.as_posix()] = archive.read(info)
    return payloads

def extract_exact(payloads, root):
    if any(path.is_symlink() for path in root.rglob("*")):
        raise SystemExit("resume extraction contains a symlink")
    expected = set(payloads)
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if actual:
        if actual != expected:
            raise SystemExit("resume extraction inventory changed")
        for name, raw in payloads.items():
            path = root.joinpath(*pathlib.PurePosixPath(name).parts)
            if path.is_symlink() or path.read_bytes() != raw:
                raise SystemExit("resume extraction bytes changed")
        return
    for name, raw in sorted(payloads.items()):
        path = root.joinpath(*pathlib.PurePosixPath(name).parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(raw)

launch = load_canonical(launch_path, "launch manifest")
required = {
    "schema","status","run_name","quality_package","runtime_source",
    "runtime_source_manifest","wheelhouse","wheelhouse_manifest","startup",
    "candidate_library","feature_encoder","allocation","wave_job_counts",
    "waves","jobs","job_count","package_plan_sha256",
    "package_root_seal_sha256","profile_registry_sha256","transport_ready",
    "cloud_launch_authorized","cloud_execution_started","training_eligible",
    "promotion_evidence","current_profile_changed",
}
jobs = launch.get("jobs")
if (set(launch) != required
    or launch.get("schema") != "hu_m31_t3_step6d_fresh_quality_local_launch_v1"
    or launch.get("status") != "local_transport_ready_cloud_not_authorized"
    or launch.get("allocation") != {"processes": 1, "rayon_threads": 16}
    or launch.get("wave_job_counts") != [8, 7]
    or launch.get("job_count") != 15
    or not isinstance(jobs, list) or len(jobs) != 15
    or launch.get("transport_ready") is not True
    or any(launch.get(field) is not False for field in (
        "cloud_launch_authorized","cloud_execution_started","training_eligible",
        "promotion_evidence","current_profile_changed",
    ))):
    raise SystemExit("fresh-quality launch contract changed")
selected = [row for row in jobs if isinstance(row, dict) and row.get("job_id") == job_id]
if len(selected) != 1:
    raise SystemExit("selected fresh-quality job is not unique")
selected = selected[0]
job_keys = {
    "job_id","phase","wave_index","package_job_path","job_manifest_sha256",
    "result_path","output_prefix","rayon_threads","processes","create_only",
}
if (set(selected) != job_keys
    or selected["phase"] not in ("primary", "confirmation")
    or selected["package_job_path"] != f"jobs/{job_id}.json"
    or selected["result_path"] != f"results/{job_id}.json"
    or selected["rayon_threads"] != 16 or selected["processes"] != 1
    or selected["create_only"] is not True):
    raise SystemExit("selected fresh-quality launch job changed")

package_path = staged_file(launch["quality_package"], "quality package")
source_path = staged_file(launch["runtime_source"], "runtime source")
source_manifest_path = staged_file(
    launch["runtime_source_manifest"], "runtime source manifest"
)
wheel_path = staged_file(launch["wheelhouse"], "wheelhouse")
wheel_manifest_path = staged_file(
    launch["wheelhouse_manifest"], "wheelhouse manifest"
)
staged_file(launch["startup"], "startup")

package_payloads = zip_payloads(package_path, deterministic=True)
job_raw = package_payloads.get(selected["package_job_path"])
if (job_raw is None or digest(job_raw) != selected["job_manifest_sha256"]):
    raise SystemExit("selected packaged job digest changed")
try:
    packaged_job = json.loads(job_raw.decode("ascii"))
except Exception as exc:
    raise SystemExit(f"selected packaged job is invalid: {exc}")
if (job_raw != canonical(packaged_job)
    or packaged_job.get("job_id") != job_id
    or packaged_job.get("phase") != selected["phase"]
    or packaged_job.get("result_path") != selected["result_path"]
    or packaged_job.get("candidate_library_sha256")
       != launch["candidate_library"]["sha256"]):
    raise SystemExit("selected packaged job binding changed")

source_manifest = load_canonical(source_manifest_path, "runtime source manifest")
source_payloads = zip_payloads(source_path, deterministic=True)
source_entries = {
    name: {"sha256": digest(raw), "bytes": len(raw)}
    for name, raw in sorted(source_payloads.items())
}
source_archive_record = source_manifest.get("archive")
if (source_manifest.get("schema")
      != "hu_m31_t3_step6d_fresh_quality_runtime_source_v1"
    or source_manifest.get("status")
      != "complete_hash_pinned_runtime_source"
    or not isinstance(source_archive_record, dict)
    or source_archive_record.get("path") != source_path.name
    or source_archive_record.get("sha256") != launch["runtime_source"]["sha256"]
    or source_archive_record.get("bytes") != launch["runtime_source"]["bytes"]
    or source_manifest.get("entries") != source_entries
    or source_manifest.get("entry_count") != len(source_entries)
    or source_manifest.get("entry_aggregate_sha256")
       != digest(canonical(source_entries))
    or source_manifest.get("candidate_library")
       != launch["candidate_library"]
    or source_manifest.get("feature_encoder") != launch["feature_encoder"]
    or source_manifest.get("profile_registry_sha256")
       != launch["profile_registry_sha256"]
    or source_manifest.get("content_addressed") is not True
    or any(source_manifest.get(field) is not False for field in (
        "cloud_execution_started","training_eligible","current_profile_changed",
    ))):
    raise SystemExit("runtime source manifest changed")
extract_exact(source_payloads, science)

wheel_manifest_raw = wheel_manifest_path.read_bytes()
try:
    wheel_manifest = json.loads(wheel_manifest_raw.decode("ascii"))
except Exception as exc:
    raise SystemExit(f"wheelhouse manifest is not JSON: {exc}")
if (not isinstance(wheel_manifest, dict)
    or wheel_manifest_raw not in (
        canonical(wheel_manifest), canonical(wheel_manifest) + b"\n"
    )):
    raise SystemExit("wheelhouse manifest is not canonical")
wheel_entries = wheel_manifest.get("entries")
if (wheel_manifest.get("schema")
      != "hu_m31_t3_step6d_perfdev_v2_wheelhouse_v1"
    or wheel_manifest.get("status")
      != "complete_hash_pinned_offline_wheelhouse"
    or wheel_manifest.get("python_abi") != "cp311"
    or wheel_manifest.get("target_os") != "linux"
    or wheel_manifest.get("target_architecture") != "x86_64"
    or wheel_manifest.get("network_install_allowed") is not False
    or not isinstance(wheel_entries, list) or not wheel_entries
    or wheel_manifest.get("entry_count") != len(wheel_entries)
    or wheel_manifest.get("entries_sha256")
       != digest(canonical(wheel_entries) + b"\n")):
    raise SystemExit("offline wheelhouse manifest changed")
wheel_payloads = zip_payloads(wheel_path, deterministic=False)
expected_wheels = {str(row.get("filename")): row for row in wheel_entries}
if set(wheel_payloads) != set(expected_wheels):
    raise SystemExit("offline wheelhouse inventory changed")
for name, raw in wheel_payloads.items():
    row = expected_wheels[name]
    if (not re.fullmatch(r"[A-Za-z0-9_.+!-]+\.whl", name)
        or digest(raw) != row.get("sha256") or len(raw) != row.get("bytes")):
        raise SystemExit("offline wheelhouse member changed")
extract_exact(wheel_payloads, wheels)

candidate = science.joinpath(
    *safe_relative(launch["candidate_library"]["path"]).parts
).resolve(strict=True)
feature = science.joinpath(
    *safe_relative(launch["feature_encoder"]["path"]).parts
).resolve(strict=True)
for path, record, label in (
    (candidate, launch["candidate_library"], "candidate"),
    (feature, launch["feature_encoder"], "feature encoder"),
):
    if (path.is_symlink() or not path.is_file()
        or digest(path.read_bytes()) != record["sha256"]
        or path.stat().st_size != record["bytes"]):
        raise SystemExit(f"{label} native binding changed")

print(package_path)
print(launch["quality_package"]["sha256"])
print(selected["job_manifest_sha256"])
print(candidate)
print(selected["phase"])
print(selected["output_prefix"])
PY
)

[[ "${#BINDING[@]}" -eq 6 ]]
PACKAGE="${BINDING[0]}"
PACKAGE_SHA256="${BINDING[1]}"
JOB_MANIFEST_SHA256="${BINDING[2]}"
CANDIDATE="${BINDING[3]}"
[[ "${BINDING[4]}" == "primary" || "${BINDING[4]}" == "confirmation" ]]
[[ -n "${BINDING[5]}" ]]

export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$SCIENCE/src"
export RAYON_NUM_THREADS=16
export OFC_HU_M3_BATCH_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

if [[ "$MODE" == "validate-only" ]]; then
  # The embedded preflight above already rehashed every staged artifact,
  # replayed the selected immutable job bytes, extracted source/wheels
  # byte-identically, and checked both native binaries.  The Python transport
  # module performs the deeper 55-root replay before an actual run.
  python3 - "$JOB_ID" "$PACKAGE_SHA256" "$JOB_MANIFEST_SHA256" <<'PY'
import json, sys
print(json.dumps({
    "schema": "hu_m31_t3_step6d_fresh_quality_startup_smoke_v1",
    "status": "selected_job_transport_validated_not_executed",
    "job_id": sys.argv[1],
    "package_archive_sha256": sys.argv[2],
    "job_manifest_sha256": sys.argv[3],
    "cloud_execution_started": False,
    "current_profile_changed": False,
}, sort_keys=True, separators=(",", ":")))
PY
  exit 0
fi

[[ "$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" == "3.11" ]]
VENV="$WORK_ROOT/venv"
VENV_PYTHON="$VENV/bin/python"
# The pinned Debian image does not guarantee ensurepip/python3-venv.  Build
# the environment without pip, then bootstrap pip only from the already
# hash-validated offline wheelhouse.  No apt or network install is permitted.
if [[ ! -x "$VENV_PYTHON" ]]; then
  python3 -m venv --without-pip "$VENV"
fi
source "$VENV/bin/activate"
INSTALL_MARKER="$VENV/.fresh-quality-wheelhouse-installed"
# Installation is local-only and hash-pinned.  The marker contains the
# wheelhouse file digest and is compared on every resume.
WHEELHOUSE_SHA256="$(python3 - "$LAUNCH_MANIFEST" <<'PY'
import json, pathlib, sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text("ascii"))["wheelhouse"]["sha256"])
PY
)"
mapfile -t WHEEL_FILES < <(
  find "$WHEELS" -maxdepth 1 -type f -name '*.whl' -print | sort
)
mapfile -t PIP_BOOTSTRAP_WHEELS < <(
  find "$WHEELS" -maxdepth 1 -type f -name 'pip-*.whl' -print | sort
)
[[ "${#WHEEL_FILES[@]}" -gt 0 ]]
[[ "${#PIP_BOOTSTRAP_WHEELS[@]}" -eq 1 ]]
PIP_BOOTSTRAP_WHEEL="${PIP_BOOTSTRAP_WHEELS[0]}"
if [[ -f "$INSTALL_MARKER" ]]; then
  [[ ! -L "$INSTALL_MARKER" ]]
  [[ "$(cat "$INSTALL_MARKER")" == "$WHEELHOUSE_SHA256" ]]
else
  PYTHONPATH="$PIP_BOOTSTRAP_WHEEL" "$VENV_PYTHON" -m pip install \
    --disable-pip-version-check --no-input --no-index \
    --find-links "$WHEELS" "${WHEEL_FILES[@]}"
  "$VENV_PYTHON" -m pip check
  printf '%s' "$WHEELHOUSE_SHA256" >"$INSTALL_MARKER"
fi
"$VENV_PYTHON" -m pip check

python -m ofc_regular.hu_m31_t3_step6d_fresh_quality_transport_v1 \
  run-job \
  --package-archive "$PACKAGE" \
  --package-sha256 "$PACKAGE_SHA256" \
  --job-id "$JOB_ID" \
  --job-manifest-sha256 "$JOB_MANIFEST_SHA256" \
  --library "$CANDIDATE" \
  --output-dir "$WORK_ROOT/output"

[[ -f "$WORK_ROOT/output/DONE.json" && ! -L "$WORK_ROOT/output/DONE.json" ]]
