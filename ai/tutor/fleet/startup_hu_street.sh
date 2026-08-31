#!/usr/bin/env bash
# HU street-teacher worker: label a slice of one street's requests.
#
# Sibling of `startup_t1_vs_fl.sh`, and everything operational carries over
# unchanged because each item was learned the expensive way:
#
#   * the log ships on ANY exit -- a worker that dies with its explanation on
#     a disk that stops being readable the moment it stops costs a cycle;
#   * the binary is verified by sha256 rather than trusted, because a
#     silently substituted labeler corrupts every label it writes;
#   * a shard's already-published work is read from the object store, not
#     from local disk, because a recreated Spot instance boots with an empty one;
#   * labels are published as they are produced;
#   * the instance DELETES itself.  A Spot VM that merely stops keeps billing
#     for its disk: 84 stopped-but-undeleted instances cost ~$10 here once.
#
# What is new: the boundary net travels as an exported `.npz` rather than a
# torch checkpoint, so the image stays `python3-numpy` and a worker boots in
# seconds instead of installing 800 MB to do one matrix multiply per batch.
# The solver binary is placed where `solver_paths` looks for it rather than
# passed, because the teacher builds that path itself.
set -euo pipefail
umask 022
export HOME="${HOME:-/root}"
ROOT=/var/lib/hu-street
LOG=/var/log/hu-street
META='http://metadata.google.internal/computeMetadata/v1/instance/attributes'
IMETA='http://metadata.google.internal/computeMetadata/v1/instance'
HEADER='Metadata-Flavor: Google'
mkdir -p "$ROOT" "$LOG"
exec > >(tee -a "$LOG/startup.log" >/dev/console) 2>&1
meta() { curl -fsS -H "$HEADER" "$META/$1"; }
NAME="$(curl -fsS -H "$HEADER" "$IMETA/name")"
ZONE="$(curl -fsS -H "$HEADER" "$IMETA/zone" | rev | cut -d/ -f1 | rev)"
BUCKET="$(meta hu-bucket)"
PREFIX="$(meta hu-prefix)"
RUN_ID="$(meta hu-run-id)"
JOB="$(meta hu-job)"
SEAT="$(meta hu-seat)"
START="$(meta hu-start)"
COUNT="$(meta hu-count)"
SRC="$(meta hu-src)"
BINSTAMP="$(meta hu-binstamp)"
REQ_OBJ="$(meta hu-requests-object)"
VAL_OBJ="$(meta hu-value-object)"
JOINT="$(meta hu-joint-samples)"
BATCH="$(meta hu-batch-roots)"
WATCHDOG="$(meta hu-watchdog-seconds)"
EXTRA="$(meta hu-extra-args)"
ENCODE="$(meta hu-encode)"
LABELS_OBJ="$(meta hu-labels-object)"

ship() {
  gcloud storage cp "$LOG/startup.log" \
    "gs://$BUCKET/$PREFIX/runs/$RUN_ID/logs/${JOB}_${START}_$(date -u +%s).log" \
    2>/dev/null || true
}
trap 'ship; gcloud compute instances delete "$NAME" --zone "$ZONE" --quiet || true' EXIT
trap 'echo "TERM received"; exit 143' TERM
( sleep "$WATCHDOG"; echo "WATCHDOG FIRED after ${WATCHDOG}s"; kill -TERM $$ ) &

mark() { echo "PHASE:$1"; echo "$1 $(date -u +%FT%TZ)" \
  | gcloud storage cp - "gs://$BUCKET/$PREFIX/runs/$RUN_ID/progress/${JOB}_${START}_$1.txt" \
  2>/dev/null || true; }

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
# numpy only: the boundary net rides as an exported .npz, so nothing here
# reaches torch.
apt-get install -y -qq python3-numpy
mark deps_ok

cd "$ROOT"
gcloud storage cp "gs://$BUCKET/$PREFIX/src/$SRC" src.tar.gz
mkdir -p src artifacts
tar -xzf src.tar.gz -C src
# `solver_paths` builds this path itself; the teacher never takes it as a flag.
mkdir -p src/ai/rust_solver/target/release
gcloud storage cp "gs://$BUCKET/$PREFIX/bin/$BINSTAMP/t4_first_exact" \
  src/ai/rust_solver/target/release/t4_first_exact
gcloud storage cp "gs://$BUCKET/$PREFIX/bin/$BINSTAMP/BINARIES.sha256" artifacts/BINARIES.sha256
chmod +x src/ai/rust_solver/target/release/t4_first_exact
EXPECT="$(awk '/t4_first_exact/ {print $1}' artifacts/BINARIES.sha256)"
ACTUAL="$(sha256sum src/ai/rust_solver/target/release/t4_first_exact | awk '{print $1}')"
[ -n "$EXPECT" ] && [ "$EXPECT" = "$ACTUAL" ] || { echo "FATAL: binary sha mismatch"; exit 1; }
gcloud storage cp "gs://$BUCKET/$PREFIX/artifacts/$REQ_OBJ" artifacts/requests.jsonl
gcloud storage cp "gs://$BUCKET/$PREFIX/artifacts/$VAL_OBJ" artifacts/value.npz
LABELS_ARG=""
if [ -n "$LABELS_OBJ" ]; then
  gcloud storage cp "gs://$BUCKET/$PREFIX/artifacts/$LABELS_OBJ" artifacts/labels.jsonl
  [ -s artifacts/labels.jsonl ] || { echo "FATAL: empty $LABELS_OBJ"; exit 1; }
  LABELS_ARG="--labels $ROOT/artifacts/labels.jsonl"
fi
for path in artifacts/requests.jsonl artifacts/value.npz src/ai/config/fl_ev.json; do
  [ -s "$path" ] || { echo "FATAL: missing artefact $path"; exit 1; }
done
echo "artifacts: $(du -sh artifacts | cut -f1), requests $(wc -l < artifacts/requests.jsonl) rows"
mark artifacts_ok

export PYTHONPATH="$ROOT/src"
cd "$ROOT/src"
# shellcheck disable=SC2086
python3 -m ai.tutor.fleet.run_hu_street_shard \
  --bucket "$BUCKET" --prefix "$PREFIX" --run-id "$RUN_ID" \
  --job "$JOB" --seat "$SEAT" --start "$START" --count "$COUNT" \
  --requests "$ROOT/artifacts/requests.jsonl" \
  --value-model "$ROOT/artifacts/value.npz" \
  --work-dir "$ROOT/work" \
  --workspace-root "$ROOT/src" \
  --joint-samples "$JOINT" --batch-roots "$BATCH" \
  --extra-args "$EXTRA" $ENCODE $LABELS_ARG
mark generate_ok
echo "SHARD_DONE $JOB $START"
