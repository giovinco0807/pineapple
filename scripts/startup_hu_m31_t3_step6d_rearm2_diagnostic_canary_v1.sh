#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

# This is a diagnostic worker entrypoint, not a launcher.  It accepts only
# locally materialized immutable package paths.  No metadata lookup, gcloud
# call, claim write, authorization write, or remote upload exists here.

if [[ "$#" -ne 5 ]]; then
  echo "usage: startup <verifier.py> <source.zip> <manifest.json> <job.json> <output-dir>" >&2
  exit 64
fi

VERIFIER=$1
SOURCE=$2
MANIFEST=$3
JOB=$4
OUTPUT=$5

: "${OFC_DIAGNOSTIC_EXPECTED_SOURCE_SHA256:?missing source identity}"
: "${OFC_DIAGNOSTIC_EXPECTED_MANIFEST_SHA256:?missing manifest identity}"
: "${OFC_DIAGNOSTIC_EXPECTED_JOB_SHA256:?missing job identity}"
: "${OFC_DIAGNOSTIC_EXPECTED_JOB_ID:?missing job id}"
: "${OFC_DIAGNOSTIC_EXPECTED_STAGE_ID:?missing stage id}"

python3 "$VERIFIER" \
  --source "$SOURCE" \
  --manifest "$MANIFEST" \
  --job "$JOB" \
  --expected-source-sha256 "$OFC_DIAGNOSTIC_EXPECTED_SOURCE_SHA256" \
  --expected-manifest-sha256 "$OFC_DIAGNOSTIC_EXPECTED_MANIFEST_SHA256" \
  --expected-job-sha256 "$OFC_DIAGNOSTIC_EXPECTED_JOB_SHA256" \
  --expected-job-id "$OFC_DIAGNOSTIC_EXPECTED_JOB_ID" \
  --expected-stage-id "$OFC_DIAGNOSTIC_EXPECTED_STAGE_ID" \
  ${OFC_DIAGNOSTIC_POISON_ROOT_READS:+--poison-root-reads}

if [[ "${OFC_DIAGNOSTIC_PRECONTENT_ONLY:-0}" == "1" ]]; then
  exit 0
fi
if [[ "${OFC_DIAGNOSTIC_OFFLINE_WORKER_AUTHORIZED:-0}" != "1" ]]; then
  echo "diagnostic worker execution is not authorized" >&2
  exit 78
fi

WORK=$(mktemp -d)
cleanup() {
  rm -rf "$WORK"
}
trap cleanup EXIT

python3 - "$SOURCE" "$WORK" <<'PY'
import pathlib
import sys
import zipfile

source = pathlib.Path(sys.argv[1])
target = pathlib.Path(sys.argv[2])
with zipfile.ZipFile(source) as archive:
    archive.extractall(target)
PY

# The v2 runner looks for preseeded roots under OUTPUT/roots.  Populate exactly
# this job's ten development roots only after the pre-content verifier has
# accepted the package.  Existing byte-identical roots are resumable; any
# mismatch is fatal.
python3 - "$WORK" "$MANIFEST" "$JOB" "$OUTPUT" <<'PY'
import hashlib
import json
import os
import pathlib
import sys

work = pathlib.Path(sys.argv[1])
manifest_path = pathlib.Path(sys.argv[2])
job_path = pathlib.Path(sys.argv[3])
output = pathlib.Path(sys.argv[4])

def canonical(value):
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")

manifest_raw = manifest_path.read_bytes()
manifest = json.loads(manifest_raw)
job_raw = job_path.read_bytes()
job = json.loads(job_raw)
if canonical(manifest) != manifest_raw or canonical(job) != job_raw:
    raise ValueError("diagnostic root seeding controls are not canonical")
indices = job.get("work_hand_indices")
if (
    not isinstance(indices, list)
    or len(indices) != 10
    or len(set(indices)) != 10
    or any(isinstance(index, bool) or not isinstance(index, int) for index in indices)
):
    raise ValueError("diagnostic root seeding escaped the exact ten-hand job")
root_dir = output / "roots"
root_dir.mkdir(parents=True, exist_ok=True)
for index in indices:
    relative = (
        "outputs/hu_joint_policy/m31_t3_step6d/candidate02_development/"
        f"tail_reselection_v2/roots/hand_{index:03d}.json"
    )
    entry = manifest.get("source_entries", {}).get(relative)
    source = work / relative
    raw = source.read_bytes()
    value = json.loads(raw)
    if (
        not isinstance(entry, dict)
        or entry.get("kind") != "root"
        or entry.get("bytes") != len(raw)
        or entry.get("sha256") != hashlib.sha256(raw).hexdigest()
        or canonical(value) != raw
        or value.get("hand_index") != index
        or value.get("schema")
        != "hu_m31_t3_step6d_candidate02_performance_root_v1"
        or value.get("training_eligible") is not False
        or value.get("current_profile_resolved") is not False
        or value.get("opponent_private_discards_used") is not False
    ):
        raise ValueError("diagnostic preseed root changed")
    destination = root_dir / f"hand_{index:03d}.json"
    if destination.exists():
        if destination.is_symlink() or destination.read_bytes() != raw:
            raise ValueError("diagnostic preseed root resume collision")
    else:
        with destination.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
PY

ROLE=$(python3 - "$JOB" <<'PY'
import json
import pathlib
import sys

print(json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))["source_role"])
PY
)
case "$ROLE" in
  candidate)
    LIBRARY="$WORK/native/candidate/release/libofc_hu_m3_engine.so"
    ;;
  reference)
    LIBRARY="$WORK/native/reference/release/libofc_hu_m3_engine.so"
    ;;
  *)
    echo "diagnostic source role escaped candidate/reference" >&2
    exit 65
    ;;
esac

if [[ "${OFC_DIAGNOSTIC_ROOT_SEED_ONLY:-0}" == "1" ]]; then
  exit 0
fi

python3 -m venv "$WORK/venv"
"$WORK/venv/bin/python" -m pip install \
  --disable-pip-version-check \
  --requirement \
  "$WORK/configs/hu_m31_t3_step6d_rearm2_diagnostic_runtime_requirements_v1.txt"

export PYTHONPATH="$WORK/src"
export RAYON_NUM_THREADS=16
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OFC_HU_M3_BATCH_THREADS=1
"$WORK/venv/bin/python" -m ofc_regular.run_hu_m31_t3_step6d_performance_v2 \
  --repository-root "$WORK" \
  --output-dir "$OUTPUT" \
  --shard-manifest "$JOB" \
  --library "$LIBRARY"
