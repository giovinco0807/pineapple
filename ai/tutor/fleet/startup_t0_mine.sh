#!/usr/bin/env bash
# T0-BB mining worker: referee the serving pick on a slice of roots.
#
# Sibling of `startup_hu_street.sh`; the operational discipline carries over
# (log ships on any exit, binary sha-verified, resume from the object store,
# partials published every two minutes, instance deletes itself).  What is
# new: the worker stages the full serving bundle (models tar) because the
# referee replays whole deals under the serving stack, and the work unit is
# a root range of the shuffled requests file.
set -euo pipefail
umask 022
export HOME="${HOME:-/root}"
ROOT=/var/lib/t0-mine
LOG=/var/log/t0-mine
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
START="$(meta hu-start)"
COUNT="$(meta hu-count)"
SRC="$(meta hu-src)"
BINSTAMP="$(meta hu-binstamp)"
REQ_OBJ="$(meta hu-requests-object)"
MODELS_OBJ="$(meta hu-models-object)"
WATCHDOG="$(meta hu-watchdog-seconds)"
# bb (default) or btn; absent on launches made before the seat existed.
SEAT="$(meta hu-seat 2>/dev/null || echo bb)"

ship() {
  gcloud storage cp "$LOG/startup.log" \
    "gs://$BUCKET/$PREFIX/runs/$RUN_ID/logs/mine_${START}_$(date -u +%s).log" \
    2>/dev/null || true
}
trap 'ship; gcloud compute instances delete "$NAME" --zone "$ZONE" --quiet || true' EXIT
trap 'echo "TERM received"; exit 143' TERM
( sleep "$WATCHDOG"; echo "WATCHDOG FIRED after ${WATCHDOG}s"; kill -TERM $$ ) &

OUT_OBJ="gs://$BUCKET/$PREFIX/runs/$RUN_ID/mine/$(printf '%06d' "$START").jsonl"
if gcloud storage ls "$OUT_OBJ" >/dev/null 2>&1; then
  DONE=$(gcloud storage cat "$OUT_OBJ" | wc -l)
  if [ "$DONE" -ge "$COUNT" ]; then
    echo "already published: $OUT_OBJ ($DONE rows)"
    exit 0
  fi
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
echo "PHASE:deps_ok"

cd "$ROOT"
gcloud storage cp "gs://$BUCKET/$PREFIX/src/$SRC" src.tar.gz
mkdir -p src work
tar -xzf src.tar.gz -C src
mkdir -p src/ai/rust_solver/target/release
gcloud storage cp "gs://$BUCKET/$PREFIX/bin/$BINSTAMP/t4_first_exact" \
  src/ai/rust_solver/target/release/t4_first_exact
gcloud storage cp "gs://$BUCKET/$PREFIX/bin/$BINSTAMP/BINARIES.sha256" BINARIES.sha256
chmod +x src/ai/rust_solver/target/release/t4_first_exact
EXPECT="$(awk '/t4_first_exact/ {print $1}' BINARIES.sha256)"
ACTUAL="$(sha256sum src/ai/rust_solver/target/release/t4_first_exact | awk '{print $1}')"
[ -n "$EXPECT" ] && [ "$EXPECT" = "$ACTUAL" ] || { echo "FATAL: binary sha mismatch"; exit 1; }
gcloud storage cp "gs://$BUCKET/$PREFIX/artifacts/$MODELS_OBJ" models.tar.gz
tar -xzf models.tar.gz
# The directory the tar actually carries, not one guessed from the object
# name: a bundle whose top directory is `models/` (the match workers' layout)
# sent a 32-VM fleet to "no hu/" in under a minute.  `sed` reads the whole
# listing -- `| head -1` would SIGPIPE tar under `set -o pipefail`.
MODELS_DIR="$ROOT/$(tar -tzf models.tar.gz | sed -n '1{s|/.*||;p;}')"
[ -d "$MODELS_DIR/hu" ] || { echo "FATAL: models dir $MODELS_DIR has no hu/"; exit 1; }
gcloud storage cp "gs://$BUCKET/$PREFIX/artifacts/$REQ_OBJ" requests.jsonl
# Resume: pull whatever this shard already published so the driver skips it.
gcloud storage cp "$OUT_OBJ" results.jsonl 2>/dev/null || true
echo "PHASE:artifacts_ok"

( while sleep 120; do
    [ -s "$ROOT/results.jsonl" ] || continue
    gcloud storage cp "$ROOT/results.jsonl" "$OUT_OBJ" >/dev/null 2>&1 || true
  done ) &
PARTIAL_PID=$!

python3 src/ai/tutor/t0_mine.py \
  --requests "$ROOT/requests.jsonl" \
  --models-dir "$MODELS_DIR" \
  --binary "$ROOT/src/ai/rust_solver/target/release/t4_first_exact" \
  --fl-ev-config "$ROOT/src/ai/config/fl_ev.json" \
  --work "$ROOT/work" --out "$ROOT/results.jsonl" \
  --start "$START" --count "$COUNT" --seat "$SEAT"
kill "$PARTIAL_PID" 2>/dev/null || true
[ -s "$ROOT/results.jsonl" ] || { echo "FATAL: mine wrote nothing"; exit 1; }
echo "rows: $(wc -l < "$ROOT/results.jsonl")"
gcloud storage cp "$ROOT/results.jsonl" "$OUT_OBJ"
tar -czf ranks.tar.gz -C "$ROOT/work" . 2>/dev/null || true
gcloud storage cp ranks.tar.gz "gs://$BUCKET/$PREFIX/runs/$RUN_ID/ranks/$(printf '%06d' "$START").tar.gz" || true
echo "PHASE:generate_ok"
echo "SHARD_DONE mine $START"
