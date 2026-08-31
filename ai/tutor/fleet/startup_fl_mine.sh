#!/usr/bin/env bash
# vs-FL T0 mining worker: referee the served opening on a slice of roots.
#
# Sibling of `startup_t0_mine.sh`; the operational discipline carries over
# (log ships on any exit, binary sha-verified, resume from the object store,
# partials published every two minutes, instance deletes itself).  Two things
# differ, and both are failure modes this fleet has already paid for once:
#
#   * the driver derives the multiplicity file from the requests path with
#     `with_suffix`, so the two objects must land side by side under their
#     own names -- a missing multiplicity file does not error, it silently
#     re-orders the pool and every shard mines different roots than intended;
#   * the referee plays the own-hand chain only, so it wants
#     models_ship_*/own_lap4 and not the HU bundle root.
set -euo pipefail
umask 022
export HOME="${HOME:-/root}"
ROOT=/var/lib/fl-mine
LOG=/var/log/fl-mine
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
MULT_OBJ="$(meta hu-multiplicity-object)"
MODELS_OBJ="$(meta hu-models-object)"
MODELS_SUBDIR="$(meta hu-models-subdir)"
WATCHDOG="$(meta hu-watchdog-seconds)"

ship() {
  gcloud storage cp "$LOG/startup.log"     "gs://$BUCKET/$PREFIX/runs/$RUN_ID/logs/mine_${START}_$(date -u +%s).log"     2>/dev/null || true
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
gcloud storage cp "gs://$BUCKET/$PREFIX/bin/$BINSTAMP/t4_first_exact"   src/ai/rust_solver/target/release/t4_first_exact
gcloud storage cp "gs://$BUCKET/$PREFIX/bin/$BINSTAMP/BINARIES.sha256" BINARIES.sha256
chmod +x src/ai/rust_solver/target/release/t4_first_exact
EXPECT="$(awk '/t4_first_exact/ {print $1}' BINARIES.sha256)"
ACTUAL="$(sha256sum src/ai/rust_solver/target/release/t4_first_exact | awk '{print $1}')"
[ -n "$EXPECT" ] && [ "$EXPECT" = "$ACTUAL" ] || { echo "FATAL: binary sha mismatch"; exit 1; }
# The mode is new in this binary; a stale one would run the old CLI and the
# shard would burn its whole budget before anyone read the log.
src/ai/rust_solver/target/release/t4_first_exact --help 2>&1 | grep -q -- '--fl-t0-deep'   || { echo "FATAL: binary $BINSTAMP has no --fl-t0-deep"; exit 1; }

gcloud storage cp "gs://$BUCKET/$PREFIX/artifacts/$MODELS_OBJ" models.tar.gz
tar -xzf models.tar.gz
MODELS_DIR="$ROOT/$(basename "$MODELS_OBJ" .tar.gz)/$MODELS_SUBDIR"
for m in t0.bin t1.bin t2.bin; do
  [ -f "$MODELS_DIR/$m" ] || { echo "FATAL: $MODELS_DIR/$m missing"; exit 1; }
done

# Side by side and under their own names: the driver reads the requests path
# and then that same path with .multiplicity.json substituted for .jsonl.
gcloud storage cp "gs://$BUCKET/$PREFIX/artifacts/$REQ_OBJ" "$ROOT/$REQ_OBJ"
gcloud storage cp "gs://$BUCKET/$PREFIX/artifacts/$MULT_OBJ" "$ROOT/$MULT_OBJ"
DERIVED="$(python3 -c "import sys,pathlib;print(pathlib.Path(sys.argv[1]).with_suffix('.multiplicity.json').name)" "$REQ_OBJ")"
[ "$DERIVED" = "$MULT_OBJ" ]   || { echo "FATAL: driver would look for $DERIVED, not $MULT_OBJ"; exit 1; }

# Resume: pull whatever this shard already published so the driver skips it.
gcloud storage cp "$OUT_OBJ" results.jsonl 2>/dev/null || true
echo "PHASE:artifacts_ok"

( while sleep 120; do
    [ -s "$ROOT/results.jsonl" ] || continue
    gcloud storage cp "$ROOT/results.jsonl" "$OUT_OBJ" >/dev/null 2>&1 || true
  done ) &
PARTIAL_PID=$!

python3 src/ai/tutor/fl_mine_local.py   --requests "$ROOT/$REQ_OBJ"   --models-dir "$MODELS_DIR"   --binary "$ROOT/src/ai/rust_solver/target/release/t4_first_exact"   --fl-ev-config "$ROOT/src/ai/config/fl_ev.json"   --work "$ROOT/work" --out "$ROOT/results.jsonl"   --start "$START" --count "$COUNT"
kill "$PARTIAL_PID" 2>/dev/null || true
[ -s "$ROOT/results.jsonl" ] || { echo "FATAL: mine wrote nothing"; exit 1; }
echo "rows: $(wc -l < "$ROOT/results.jsonl")"
gcloud storage cp "$ROOT/results.jsonl" "$OUT_OBJ"
tar -czf ranks.tar.gz -C "$ROOT/work" . 2>/dev/null || true
gcloud storage cp ranks.tar.gz "gs://$BUCKET/$PREFIX/runs/$RUN_ID/ranks/$(printf '%06d' "$START").tar.gz" || true
echo "PHASE:generate_ok"
echo "SHARD_DONE mine $START"
