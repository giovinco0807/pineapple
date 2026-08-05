#!/usr/bin/env bash
# Joker-track label-generation worker.
#
# Structure follows the regular track's proven fleet worker -- artifacts are
# fetched by generation and verified by sha256, uploads are create-only, the
# shard's already-published work is read from the object store rather than
# from local disk (a recreated Spot instance boots with an empty disk), and a
# watchdog guarantees the instance cannot outlive its budget.
#
# The joker generators already checkpoint per chunk, keyed by absolute root
# seed, so a shard is just a root range and resume is "download my chunks
# first".  That replaces the regular track's position-file protocol entirely.
set -euo pipefail
umask 077
ROOT=/var/lib/joker-labelgen
LOG=/var/log/joker-labelgen
META='http://metadata.google.internal/computeMetadata/v1/instance/attributes'
HEADER='Metadata-Flavor: Google'
mkdir -p "$ROOT" "$LOG"
# Mirror to the serial console: an instance that dies with its explanation on
# an unreachable disk costs a whole debugging cycle (it cost us two).
exec > >(tee -a "$LOG/startup.log" >/dev/console) 2>&1
meta() { curl -fsS -H "$HEADER" "$META/$1"; }
BUCKET="$(meta jk-bucket)"
PREFIX="$(meta jk-prefix)"
STREET="$(meta jk-street)"
SEED="$(meta jk-seed)"
ROOTS="$(meta jk-roots)"
BATCH="$(meta jk-batch)"
EXTRA="$(meta jk-extra-args)"
WATCHDOG="$(meta jk-watchdog-seconds)"
( sleep "$WATCHDOG"; echo "WATCHDOG FIRED"; shutdown -h now ) &
mark() { echo "PHASE:$1"; echo "$1 $(date -u +%FT%TZ)" \
  | gcloud storage cp - "gs://$BUCKET/$PREFIX/progress/${SEED}_$1.txt" || true; }
# Ship the log on ANY exit.  Three rehearsals died with their explanation on
# a disk that stops being readable the moment the instance stops, and each
# one cost a full cycle to re-derive.
trap 'gcloud storage cp "$LOG/startup.log" \
  "gs://$BUCKET/$PREFIX/logs/${SEED}.log" 2>/dev/null || true' EXIT

apt-get update -qq && apt-get install -y -qq python3-numpy
mark deps_ok

cd "$ROOT"
gcloud storage cp "gs://$BUCKET/joker-fleet/src/joker_src_20260805.tar.gz" src.tar.gz
tar -xzf src.tar.gz
mkdir -p ai/rust_solver/target/release
gcloud storage cp "gs://$BUCKET/joker-fleet/bin2/t4_first_exact" \
  ai/rust_solver/target/release/t4_first_exact
gcloud storage cp "gs://$BUCKET/joker-fleet/bin2/BINARIES.sha256" BINARIES.sha256
chmod +x ai/rust_solver/target/release/t4_first_exact
# The binary is the one thing whose silent substitution would corrupt every
# label, so it is verified rather than trusted.
EXPECT="$(awk '/t4_first_exact/ {print $1}' BINARIES.sha256)"
ACTUAL="$(sha256sum ai/rust_solver/target/release/t4_first_exact | awk '{print $1}')"
[ "$EXPECT" = "$ACTUAL" ] || { echo "FATAL: binary sha mismatch"; exit 1; }
# cp -r refuses a destination that does not already exist, and the trailing
# /* form is what puts the objects directly under it rather than nested.
mkdir -p models library fl_library_15_v3 fl_library_16_v3 fl_library_17_v3
gcloud storage cp "gs://$BUCKET/joker-fleet/models/*" models/
gcloud storage cp "gs://$BUCKET/joker-fleet/library/*" library/
gcloud storage cp "gs://$BUCKET/joker-fleet/library15/*" fl_library_15_v3/
gcloud storage cp "gs://$BUCKET/joker-fleet/library16/*" fl_library_16_v3/
gcloud storage cp "gs://$BUCKET/joker-fleet/library17/*" fl_library_17_v3/
for count in 15 16 17; do
  [ "$(find "fl_library_${count}_v3" -name 'shard_*.jsonl' | wc -l)" -gt 0 ]     || { echo "FATAL: count library $count empty"; exit 1; }
done
[ -f models/t1_evaluator.bin ] && [ -f models/t2_evaluator.bin ] && [ -f models/t3_evaluator.bin ] \
  || { echo "FATAL: models missing"; exit 1; }
LIB_SHARDS="$(find library -name 'shard_*.jsonl' | wc -l)"
[ "$LIB_SHARDS" -gt 0 ] || { echo "FATAL: FL library empty"; exit 1; }
echo "artifacts: $LIB_SHARDS library shard(s)"
mark artifacts_ok

OUT="$ROOT/out"
mkdir -p "$OUT/chunks"
# What this shard already published.  A recreated instance has an empty disk,
# so without this it would regenerate work that is already in the bucket and
# then collide with it on upload.  A listing that cannot be read is fatal:
# proceeding would mean redoing a shard whose state is unknown.
if ! gcloud storage cp "gs://$BUCKET/$PREFIX/chunks/chunk_*.npz" "$OUT/chunks/" 2>/dev/null; then
  echo "no prior chunks for this run (or none match this shard)"
fi
EXISTING="$(find "$OUT/chunks" -name 'chunk_*.npz' | wc -l)"
echo "resume: $EXISTING chunk(s) already present"
mark resume_ok

export PYTHONPATH="$ROOT"
# Per-count referees: labels score 15/16/17-card opponents against their own
# board distributions instead of the 14-card stand-in.
export JOKER_COUNT_LIBS=1
export JOKER_LIBS_ROOT="$ROOT"
cd "$ROOT"

# Create-only publication of whatever is finished.  A chunk that already
# exists was written by an earlier attempt of this shard and is identical by
# construction: chunks are named by absolute root seed and the generator is
# deterministic, so the collision is benign rather than a conflict.
publish() {
  local published=0
  for path in "$OUT"/chunks/chunk_*.npz; do
    [ -e "$path" ] || continue
    local name; name="$(basename "$path")"
    [ -e "$OUT/.published_$name" ] && continue
    if gcloud storage cp --if-generation-match=0 "$path" \
        "gs://$BUCKET/$PREFIX/chunks/$name" 2>/dev/null; then
      published=$(( published + 1 ))
    fi
    : >"$OUT/.published_$name"
  done
  [ "$published" -gt 0 ] && echo "published $published new chunk(s)"
  return 0
}

# Generation runs in the background and its finished chunks are published as
# they appear.  The first rehearsal published nothing because it uploaded
# only after the whole shard finished, and the watchdog cut it off mid-run --
# on Spot, where preemption is routine, that loses everything.
# shellcheck disable=SC2086
python3 -m "ai.tutor.generate_${STREET}_vs_fl_teacher" \
  --roots "$ROOTS" --seed "$SEED" --batch "$BATCH" \
  --out-dir "$OUT" \
  --library "$ROOT/library" \
  --t2-model "$ROOT/models/t2_evaluator.bin" \
  --t3-model "$ROOT/models/t3_evaluator.bin" \
  --workspace-root "$ROOT" \
  --no-merge $EXTRA &
GEN_PID=$!
while kill -0 "$GEN_PID" 2>/dev/null; do
  sleep 60
  publish
done
wait "$GEN_PID"; GEN_STATUS=$?
publish
if [ "$GEN_STATUS" -ne 0 ]; then
  echo "FATAL: generator exited $GEN_STATUS (published chunks are still valid)"
  exit "$GEN_STATUS"
fi
mark generate_ok
mark publish_ok
echo "SHARD_DONE $SEED"
echo "done $(date -u +%FT%TZ)" | gcloud storage cp - "gs://$BUCKET/$PREFIX/done/${SEED}.txt"
shutdown -h now
