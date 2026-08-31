#!/usr/bin/env bash
# HU match worker: play a shard of hands and publish the settlement log.
#
# Sibling of `startup_hu_street.sh`; the operational parts are identical and
# deliberately so, because each was learned the expensive way: the log ships
# on any exit, the binary is verified by sha256 rather than trusted, and the
# instance is deleted rather than stopped.
#
# What differs is the unit of work.  A label shard is a slice of a requests
# file and can be resumed root by root; a match shard is a stream of hands
# whose sessions carry stacks and Fantasyland state forward, so it cannot be
# resumed halfway -- a preempted shard is replayed from its seed, which is
# why each shard gets its own seed and its own file.  Nothing is shared, so
# nothing has to agree.
set -euo pipefail
umask 022
export HOME="${HOME:-/root}"
ROOT=/var/lib/hu-match
LOG=/var/log/hu-match
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
SHARD="$(meta hu-shard)"
HANDS="$(meta hu-hands)"
SEED="$(meta hu-seed)"
SRC="$(meta hu-src)"
BINSTAMP="$(meta hu-binstamp)"
MODELS_OBJ="$(meta hu-models-object)"
MATCH_ARGS="$(meta hu-match-args)"
WORKERS="$(meta hu-workers)"
WATCHDOG="$(meta hu-watchdog-seconds)"

ship() {
  gcloud storage cp "$LOG/startup.log" \
    "gs://$BUCKET/$PREFIX/runs/$RUN_ID/logs/${JOB}_${SHARD}_$(date -u +%s).log" \
    2>/dev/null || true
}
trap 'ship; gcloud compute instances delete "$NAME" --zone "$ZONE" --quiet || true' EXIT
trap 'echo "TERM received"; exit 143' TERM
( sleep "$WATCHDOG"; echo "WATCHDOG FIRED after ${WATCHDOG}s"; kill -TERM $$ ) &

mark() { echo "PHASE:$1"; echo "$1 $(date -u +%FT%TZ)" \
  | gcloud storage cp - "gs://$BUCKET/$PREFIX/runs/$RUN_ID/progress/${JOB}_${SHARD}_$1.txt" \
  2>/dev/null || true; }

# A shard already published is a shard already paid for.
OUT="gs://$BUCKET/$PREFIX/runs/$RUN_ID/matches/$JOB/$(printf '%04d' "$SHARD").jsonl"
if gcloud storage ls "$OUT" >/dev/null 2>&1; then
  echo "already published: $OUT"
  mark generate_ok
  exit 0
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
mark deps_ok

cd "$ROOT"
gcloud storage cp "gs://$BUCKET/$PREFIX/src/$SRC" src.tar.gz
mkdir -p src models
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
echo "models: $(find models -name '*.bin' | wc -l) evaluators"
[ -s src/ai/config/fl_ev.json ] || { echo "FATAL: no fl_ev.json"; exit 1; }
mark artifacts_ok

# Ship what has been played every two minutes, to `partial/`.  A trace shard
# used to publish only on its last hand, so Spot taking a worker at hour two
# threw away every hand of it (the tr2 corpus lost thirteen shards that way).
# The binary appends and flushes per chunk, so a partial is always whole
# lines; the collector prefers the complete object and falls back to these.
PARTIAL="gs://$BUCKET/$PREFIX/runs/$RUN_ID/partial/$JOB/$(printf '%04d' "$SHARD").jsonl"
(
  while sleep 120; do
    [ -s "$ROOT/match.jsonl" ] || continue
    gcloud storage cp "$ROOT/match.jsonl" "$PARTIAL" >/dev/null 2>&1 || true
  done
) &
PARTIAL_PID=$!

# shellcheck disable=SC2086
src/ai/rust_solver/target/release/t4_first_exact \
  --hu-match --hands "$HANDS" --workers "$WORKERS" --self-play-seed "$SEED" \
  --fl-ev-config src/ai/config/fl_ev.json \
  --output "$ROOT/match.jsonl" $MATCH_ARGS
kill "$PARTIAL_PID" 2>/dev/null || true
[ -s "$ROOT/match.jsonl" ] || { echo "FATAL: match wrote nothing"; exit 1; }
echo "hands played: $(wc -l < "$ROOT/match.jsonl")"
gcloud storage cp --if-generation-match=0 "$ROOT/match.jsonl" "$OUT" || \
  gcloud storage ls "$OUT" >/dev/null
mark generate_ok
echo "SHARD_DONE $JOB $SHARD"
