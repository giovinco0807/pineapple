#!/usr/bin/env bash
set -euo pipefail

META='http://metadata.google.internal/computeMetadata/v1'
meta() { curl -fsS -H 'Metadata-Flavor: Google' "$META/instance/attributes/$1"; }
imeta() { curl -fsS -H 'Metadata-Flavor: Google' "$META/instance/$1"; }
sha() { sha256sum "$1" | awk '{print $1}'; }

PROJECT_ID="$(meta PROJECT_ID)"
BUCKET="$(meta BUCKET)"
RUN_NAME="$(meta RUN_NAME)"
SHARD="$(meta SHARD)"
SOURCE_URI="$(meta SOURCE_URI)"
SOURCE_SHA256="$(meta SOURCE_SHA256)"
MANIFEST_SHA256="$(meta MANIFEST_SHA256)"
SCHEDULE_SHA256="$(meta SCHEDULE_SHA256)"
AUTHORIZATION_SHA256="$(meta AUTHORIZATION_SHA256)"
SELF_DELETE="$(meta SELF_DELETE)"
INSTANCE_NAME="$(imeta name)"
ZONE="$(imeta zone)"; ZONE="${ZONE##*/}"
PREFIX="gs://$BUCKET/runs/$RUN_NAME"
WORK=/opt/m31-step6b
RESULT=/var/lib/m31-step6b-result
VENV=/opt/m31-step6b-venv
LOG=/var/log/m31-step6b-worker.log
PUMP_PID=''
SHARD_PAD=''
PROGRESS_URI=''
RESULT_URI=''

mkdir -p "$RESULT"
exec > >(tee -a "$LOG") 2>&1

sync_progress() {
  if [[ -n "$PROGRESS_URI" && -d "$RESULT" ]]; then
    gcloud storage rsync --recursive "$RESULT" "$PROGRESS_URI" \
      --project "$PROJECT_ID" >/dev/null 2>&1 || true
  fi
}

start_pump() {
  (
    while true; do
      sync_progress
      sleep 60
    done
  ) &
  PUMP_PID=$!
}

stop_pump() {
  if [[ -n "$PUMP_PID" ]]; then
    kill "$PUMP_PID" >/dev/null 2>&1 || true
    wait "$PUMP_PID" >/dev/null 2>&1 || true
    PUMP_PID=''
  fi
}

cleanup() {
  code=$?
  stop_pump
  sync_progress
  if [[ -n "$SHARD_PAD" ]]; then
    gcloud storage cp "$LOG" "$PREFIX/logs/shard-$SHARD_PAD-startup.log" \
      --project "$PROJECT_ID" >/dev/null 2>&1 || true
  fi
  if [[ "$SELF_DELETE" == 1 ]]; then
    gcloud compute instances delete "$INSTANCE_NAME" --zone "$ZONE" \
      --project "$PROJECT_ID" --quiet >/dev/null 2>&1 || true
  fi
  exit "$code"
}
trap cleanup EXIT

if [[ ! "$SHARD" =~ ^[1-9]$ ]]; then
  echo 'Step 6b startup authorizes integer shards 1 through 9 only' >&2
  exit 64
fi
SHARD_NUM=$((10#$SHARD))
printf -v SHARD_PAD '%03d' "$SHARD_NUM"
PROGRESS_URI="$PREFIX/progress/shard-$SHARD_PAD"
RESULT_URI="$PREFIX/results/shard-$SHARD_PAD"

if [[ ! "$RUN_NAME" =~ ^[a-z0-9][a-z0-9-]*$ ]]; then
  echo 'unsafe Step 6b run name' >&2
  exit 64
fi

if gcloud storage objects describe "$RESULT_URI/DONE.json" \
    --project "$PROJECT_ID" >/dev/null 2>&1; then
  echo "immutable Step 6b result already complete for shard $SHARD_PAD" >&2
  exit 67
fi

sudo rm -f /etc/apt/sources.list.d/debian.sources
printf '%s\n' \
  'deb [check-valid-until=no] http://snapshot.debian.org/archive/debian/20260609T000000Z bookworm main' \
  'deb [check-valid-until=no] http://snapshot.debian.org/archive/debian-security/20260609T000000Z bookworm-security main' \
  | sudo tee /etc/apt/sources.list >/dev/null
sudo apt-get -o Acquire::Check-Valid-Until=false update -y
sudo apt-get -o Acquire::Check-Valid-Until=false install -y \
  python3 python3-venv unzip time libgomp1 ca-certificates

gcloud storage cp "$SOURCE_URI" /tmp/source.zip --project "$PROJECT_ID" >/dev/null
gcloud storage cp "$PREFIX/manifest.json" /tmp/manifest.json \
  --project "$PROJECT_ID" >/dev/null
gcloud storage cp "$PREFIX/source/shards_manifest.jsonl" /tmp/shards.jsonl \
  --project "$PROJECT_ID" >/dev/null
gcloud storage cp "$PREFIX/source/launch_authorization.json" /tmp/authorization.json \
  --project "$PROJECT_ID" >/dev/null
[[ "$(sha /tmp/source.zip)" == "$SOURCE_SHA256" ]]
[[ "$(sha /tmp/manifest.json)" == "$MANIFEST_SHA256" ]]
[[ "$(sha /tmp/shards.jsonl)" == "$SCHEDULE_SHA256" ]]
[[ "$(sha /tmp/authorization.json)" == "$AUTHORIZATION_SHA256" ]]

rm -rf "$WORK" "$VENV"
mkdir -p "$WORK"
unzip -q /tmp/source.zip -d "$WORK"

python3 - "$WORK" /tmp/manifest.json /tmp/shards.jsonl \
  /tmp/authorization.json "$RUN_NAME" "$SHARD_NUM" \
  "$SOURCE_SHA256" "$MANIFEST_SHA256" "$SCHEDULE_SHA256" <<'PY'
import hashlib
import json
import pathlib
import sys


def require(condition, message):
    if not condition:
        raise SystemExit(message)


work = pathlib.Path(sys.argv[1]).resolve()
manifest_path = pathlib.Path(sys.argv[2])
schedule_path = pathlib.Path(sys.argv[3])
authorization_path = pathlib.Path(sys.argv[4])
run_name = sys.argv[5]
shard = int(sys.argv[6])
source_sha256 = sys.argv[7]
manifest_sha256 = sys.argv[8]
schedule_sha256 = sys.argv[9]
m = json.loads(manifest_path.read_text(encoding="utf-8"))
a = json.loads(authorization_path.read_text(encoding="utf-8"))
rows = [
    json.loads(line)
    for line in schedule_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
authorized = list(range(1, 10))

require(m.get("schema") == "hu_m31_t3_step6b_spot_package_v1", "bad manifest schema")
require(
    a.get("schema") == "hu_m31_t3_step6b_launch_authorization_v1",
    "bad authorization schema",
)
require(m.get("run_name") == a.get("run_name") == run_name, "run name mismatch")
require(m.get("authorized_shards") == authorized, "manifest must authorize exactly 1..9")
require(a.get("authorized_shards") == authorized, "authorization must contain exactly 1..9")
require(shard in authorized, "selected shard is not authorized")
require(a.get("spot_authorized") is True, "Spot execution is not authorized")
require(
    a.get("remaining_canary_shards_authorized") is True,
    "remaining canary authorization is absent",
)
require(a.get("production_fanout_authorized") is False, "production fanout is forbidden")
require(a.get("current_profile_changed") is False, "current profile mutation is forbidden")
require(m.get("production_fanout_authorized") is False, "bad manifest fanout flag")
require(m.get("current_profile_changed") is False, "bad manifest current flag")
require(m.get("source_sha256") == a.get("source_sha256") == source_sha256, "source hash mismatch")
require(m.get("schedule_sha256") == a.get("schedule_sha256") == schedule_sha256, "schedule hash mismatch")
require(a.get("manifest_sha256") == manifest_sha256, "authorization manifest hash mismatch")
require(m.get("paired_hands_per_shard") == 25, "unexpected paired-hand count")
require(m.get("roots_per_shard") == 50, "unexpected root count")

schedule_shards = [row.get("shard") for row in rows]
require(len(schedule_shards) == len(set(schedule_shards)), "duplicate schedule shard")
require(all(schedule_shards.count(value) == 1 for value in authorized), "authorized schedule membership mismatch")
selected = [row for row in rows if row.get("shard") == shard]
require(len(selected) == 1, "selected shard must have exactly one schedule row")
row = selected[0]
require(row.get("run_name") == run_name, "schedule run name mismatch")
require(row.get("global_hand_start") == shard * 25, "schedule hand offset mismatch")
require(row.get("hand_count") == 25 and row.get("root_count") == 50, "bad schedule size")

entries = m.get("source_entries")
require(isinstance(entries, dict), "source entry manifest is absent")
require(len(entries) == m.get("source_entry_count"), "source entry count mismatch")
for relative, expected in entries.items():
    pure = pathlib.PurePosixPath(relative)
    require(not pure.is_absolute() and ".." not in pure.parts, "unsafe source entry")
    path = (work / pathlib.Path(*pure.parts)).resolve()
    require(path.is_relative_to(work) and path.is_file(), f"missing source entry: {relative}")
    data = path.read_bytes()
    require(len(data) == expected.get("bytes"), f"source size mismatch: {relative}")
    require(
        hashlib.sha256(data).hexdigest() == expected.get("sha256"),
        f"source hash mismatch: {relative}",
    )
PY

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
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

gcloud storage rsync --recursive "$PROGRESS_URI" "$RESULT" \
  --project "$PROJECT_ID" >/dev/null 2>&1 || true
start_pump

COMMON_ARGS=(
  --manifest /tmp/manifest.json
  --schedule /tmp/shards.jsonl
  --source-package-sha256 "$SOURCE_SHA256"
  --repository-root "$WORK"
  --shard "$SHARD_NUM"
  --output-dir "$RESULT"
  --parity-golden "$WORK/artifacts/step6b/parity_golden.json"
)

python -m ofc_regular.run_hu_m31_t3_step6b_shard \
  "${COMMON_ARGS[@]}" --parity-only
python - "$RESULT/parity.json" <<'PY'
import json
import pathlib
import sys

parity = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
if parity.get("schema") != "hu_m31_t3_step6b_linux_parity_v1":
    raise SystemExit("bad Step 6b parity schema")
if parity.get("all_gates_passed") is not True:
    raise SystemExit("Step 6b Linux parity failed")
PY

TASK_COUNT=0
if [[ -d "$RESULT/tasks" ]]; then
  TASK_COUNT="$(find "$RESULT/tasks" -maxdepth 1 -type f -name 'hand_*.json' | wc -l)"
fi
if [[ "$TASK_COUNT" -eq 0 ]]; then
  set +e
  python -m ofc_regular.run_hu_m31_t3_step6b_shard \
    "${COMMON_ARGS[@]}" --stop-after-tasks 1 >"$RESULT/drill_stdout.json"
  drill_code=$?
  set -e
  if [[ "$drill_code" -ne 75 ]]; then
    echo "resume drill expected exit 75, got $drill_code" >&2
    exit 65
  fi
  stop_pump
  sync_progress
  rm -rf "$RESULT"
  mkdir -p "$RESULT"
  gcloud storage rsync --recursive "$PROGRESS_URI" "$RESULT" \
    --project "$PROJECT_ID" >/dev/null
  RECOVERED="$(find "$RESULT/tasks" -maxdepth 1 -type f -name 'hand_*.json' | wc -l)"
  if [[ "$RECOVERED" -lt 1 ]]; then
    echo 'resume drill failed to recover a completed task from GCS' >&2
    exit 66
  fi
  start_pump
fi

/usr/bin/time -v -o "$RESULT/time.txt" \
  python -m ofc_regular.run_hu_m31_t3_step6b_shard "${COMMON_ARGS[@]}" \
  >"$RESULT/runner_stdout.json"
cp "$LOG" "$RESULT/run.log"
stop_pump
sync_progress

python - "$RESULT" /tmp/manifest.json /tmp/authorization.json "$SHARD_NUM" <<'PY'
import hashlib
import json
import pathlib
import sys
import time


def require(condition, message):
    if not condition:
        raise SystemExit(message)


r = pathlib.Path(sys.argv[1])
manifest_path = pathlib.Path(sys.argv[2])
authorization_path = pathlib.Path(sys.argv[3])
shard = int(sys.argv[4])
m = json.loads(manifest_path.read_text(encoding="utf-8"))
a = json.loads(authorization_path.read_text(encoding="utf-8"))
summary = json.loads((r / "summary.json").read_text(encoding="utf-8"))
parity = json.loads((r / "parity.json").read_text(encoding="utf-8"))
require(summary.get("schema") == "hu_m31_t3_step6b_summary_v1", "bad summary schema")
require(parity.get("schema") == "hu_m31_t3_step6b_linux_parity_v1", "bad parity schema")
require(summary.get("run_name") == m.get("run_name"), "summary run mismatch")
require(summary.get("shard") == shard, "summary shard mismatch")
require(summary.get("all_gates_passed") is True, "summary gates failed")
require(summary.get("resumed_task_count", 0) >= 1, "resume drill was not observed")
require(summary.get("training_eligible") is False, "canary rows cannot train")
require(summary.get("production_fanout_authorized") is False, "production fanout is forbidden")
require(summary.get("current_profile_changed") is False, "current profile mutation is forbidden")
files = {}
for path in sorted(r.rglob("*")):
    if path.is_file() and path.name != "DONE.json":
        relative = path.relative_to(r).as_posix()
        data = path.read_bytes()
        files[relative] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
done = {
    "schema": "hu_m31_t3_step6b_done_v1",
    "status": "complete",
    "run_name": m["run_name"],
    "shard": shard,
    "source_sha256": m["source_sha256"],
    "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    "schedule_sha256": m["schedule_sha256"],
    "authorization_sha256": hashlib.sha256(authorization_path.read_bytes()).hexdigest(),
    "native_library_sha256": m["native_library"]["sha256"],
    "files": files,
    "resume_drill_passed": True,
    "training_eligible": False,
    "authorized_shards": a["authorized_shards"],
    "production_fanout_authorized": False,
    "current_profile_changed": False,
    "completed_unix_seconds": time.time(),
}
(r / "DONE.json").write_text(
    json.dumps(done, sort_keys=True, separators=(",", ":")) + "\n",
    encoding="utf-8",
)
PY

upload_once() {
  local source="$1" target="$2"
  if gcloud storage cp "$source" "$target" --project "$PROJECT_ID" \
      --if-generation-match=0 >/dev/null 2>&1; then return 0; fi
  local existing; existing="$(mktemp)"
  gcloud storage cp "$target" "$existing" --project "$PROJECT_ID" >/dev/null
  [[ "$(sha "$source")" == "$(sha "$existing")" ]]
  rm -f "$existing"
}

while IFS= read -r -d '' path; do
  relative="${path#"$RESULT/"}"
  upload_once "$path" "$RESULT_URI/$relative"
done < <(find "$RESULT" -type f ! -name DONE.json -print0)
upload_once "$RESULT/DONE.json" "$RESULT_URI/DONE.json"
