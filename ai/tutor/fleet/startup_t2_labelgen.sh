#!/usr/bin/env bash
# T2 own-label mass production worker: deal roots [start, start+count) from the
# canonical seed and price every action at t3-draws with the pilot shortcut.
#
# Sibling of `startup_ship_gate.sh`; the operational discipline carries over
# (log ships on any exit, binary sha-verified, resume from the object store,
# instance deletes itself).  What is different:
#
#   * the labeler is `fl_solver teach-t2`, which DEALS its own roots from
#     (seed, root ordinal) -- no requests file.  The window [start, count)
#     continues the exact deal stream of `t2_labels_own_10k.jsonl`
#     (seed 3238398113, stream-offset 0), so ids never collide with the
#     original ten thousand and the corpus stays one stream.
#   * the fl_ev table is written HERE, as the prev table pinned by
#     fl14_v1.jfl1 -- the labels must carry the same value semantics as the
#     corpus they join.  The tarball's ai/config/fl_ev.json is the CURRENT
#     table and would refuse the pool.
#   * work is chunked (250 roots per object) so a Spot preemption loses
#     minutes, not the shard: a chunk whose object already exists is skipped.
set -euo pipefail
umask 022
export HOME="${HOME:-/root}"
ROOT=/var/lib/t2-labelgen
LOG=/var/log/t2-labelgen
META='http://metadata.google.internal/computeMetadata/v1/instance/attributes'
IMETA='http://metadata.google.internal/computeMetadata/v1/instance'
HEADER='Metadata-Flavor: Google'
mkdir -p "$ROOT" "$LOG"
exec > >(tee -a "$LOG/startup.log" >/dev/console) 2>&1
meta() { curl -fsS -H "$HEADER" "$META/$1"; }
NAME="$(curl -fsS -H "$HEADER" "$IMETA/name")"
ZONE="$(curl -fsS -H "$HEADER" "$IMETA/zone" | rev | cut -d/ -f1 | rev)"
BUCKET="$(meta t2l-bucket)"
PREFIX="$(meta t2l-prefix)"
RUN_ID="$(meta t2l-run-id)"
START="$(meta t2l-start)"
COUNT="$(meta t2l-count)"
BINSTAMP="$(meta t2l-binstamp)"
POOL_OBJ="$(meta t2l-pool-object)"
# When set, positions come from this artifacts object (a played-roots jsonl,
# one root per line, global ordinal = line number) instead of teach-t2's
# dealt (2,3,2) boards -- which are NOT play-reachable shapes and produced
# the wrong value distribution on 2026-09-01.  Dealing stays only as the
# legacy fallback for an empty value.
ROOTS_OBJ="$(meta t2l-roots-object || true)"
SEED="$(meta t2l-seed)"
DRAWS="$(meta t2l-draws)"
PILOT="$(meta t2l-pilot)"
PILOT_KEEP="$(meta t2l-pilot-keep)"
CHUNK="$(meta t2l-chunk)"
WATCHDOG="$(meta t2l-watchdog-seconds)"

ship() {
  gcloud storage cp "$LOG/startup.log" \
    "gs://$BUCKET/$PREFIX/runs/$RUN_ID/logs/t2l_${START}_$(date -u +%s).log" \
    2>/dev/null || true
}
trap 'ship; gcloud compute instances delete "$NAME" --zone "$ZONE" --quiet || true' EXIT
trap 'echo "TERM received"; exit 143' TERM
( sleep "$WATCHDOG"; echo "WATCHDOG FIRED after ${WATCHDOG}s"; kill -TERM $$ ) &

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
echo "PHASE:deps_ok"

cd "$ROOT"
gcloud storage cp "gs://$BUCKET/$PREFIX/bin/$BINSTAMP/fl_solver" fl_solver
gcloud storage cp "gs://$BUCKET/$PREFIX/bin/$BINSTAMP/BINARIES.sha256" BINARIES.sha256
chmod +x fl_solver
EXPECT="$(awk '/fl_solver/ {print $1}' BINARIES.sha256)"
ACTUAL="$(sha256sum fl_solver | awk '{print $1}')"
[ -n "$EXPECT" ] && [ "$EXPECT" = "$ACTUAL" ] || { echo "FATAL: binary sha mismatch"; exit 1; }
gcloud storage cp "gs://$BUCKET/$PREFIX/artifacts/$POOL_OBJ" pool.jfl1
if [ -n "$ROOTS_OBJ" ]; then
  gcloud storage cp "gs://$BUCKET/$PREFIX/artifacts/$ROOTS_OBJ" roots_all.jsonl
  wc -l < roots_all.jsonl
fi
echo "PHASE:artifacts_ok"

# The prev fl_ev table, matching the pool header and the 10k corpus pricing.
# teach-t2 reads ai/config/fl_ev.json relative to its CWD, so a private
# workspace pins the table without touching the shipped tarball's config.
mkdir -p ws/ai/config
cat > ws/ai/config/fl_ev.json <<'JSON'
{
  "fl_ev": {"14": 0.0, "15": 10.7, "16": 29.9, "17": 63.5},
  "note": "fl_ev_prev, pinned by fl14_v1.jfl1; the t2_labels_own corpus pricing"
}
JSON

for ((base = START; base < START + COUNT; base += CHUNK)); do
  n=$((START + COUNT - base)); [ "$n" -gt "$CHUNK" ] && n=$CHUNK
  OBJ="gs://$BUCKET/$PREFIX/runs/$RUN_ID/labels/$(printf '%08d' "$base").jsonl"
  if gcloud storage ls "$OBJ" >/dev/null 2>&1; then
    HAVE=$(gcloud storage cat "$OBJ" | wc -l)
    if [ "$HAVE" -ge "$n" ]; then echo "chunk $base already published"; continue; fi
  fi
  rm -rf out; mkdir -p out
  if [ -n "$ROOTS_OBJ" ]; then
    sed -n "$((base + 1)),$((base + n))p" roots_all.jsonl > chunk_roots.jsonl
    [ "$(wc -l < chunk_roots.jsonl)" -eq "$n" ] || { echo "FATAL: roots slice short"; exit 1; }
    ( cd ws && "$ROOT/fl_solver" teach-t2 \
        --pool "$ROOT/pool.jfl1" \
        --roots-file "$ROOT/chunk_roots.jsonl" --root-offset "$base" --stream-offset 0 \
        --opponents 60 --own-only \
        --t3-draws "$DRAWS" --t4-draws 0 \
        --t3-pilot-draws "$PILOT" --t3-keep "$PILOT_KEEP" \
        --out-dir "$ROOT/out" )
  else
    ( cd ws && "$ROOT/fl_solver" teach-t2 \
        --pool "$ROOT/pool.jfl1" \
        --roots "$n" --root-offset "$base" --seed "$SEED" --stream-offset 0 \
        --opponents 60 --own-only \
        --t3-draws "$DRAWS" --t4-draws 0 \
        --t3-pilot-draws "$PILOT" --t3-keep "$PILOT_KEEP" \
        --out-dir "$ROOT/out" )
  fi
  LINES=$(wc -l < out/t2_labels.jsonl)
  [ "$LINES" -ge "$n" ] || { echo "FATAL: chunk $base wrote $LINES of $n"; exit 1; }
  gcloud storage cp out/t2_labels.jsonl "$OBJ"
  echo "chunk $base done ($LINES roots)"
done
echo "PHASE:generate_ok"
echo "SHARD_DONE t2l $START"
