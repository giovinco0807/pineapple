#!/usr/bin/env bash
# HU trace worker: play champion-vs-champion hands and publish the traces.
#
# The traces are teaching material for the fast continuation nets: one hand
# carries seven (state, chosen move) pairs, and the champion's MOVE is the
# label -- never its value.  Choosing wrongly costs a candidate comparison
# something that mostly cancels; returning a wrong VALUE reorders candidates,
# which is why the distillation targets moves.
#
# Sibling of `startup_t1_labelgen.sh`.  Same discipline: sha-verified binary,
# chunked resumable output, log shipped on any exit, instance deletes itself.
# The work unit is a seed window -- hand h of this run is `--self-play-seed`
# SEED with `--hu-trace` covering [start, start+count), and hu-trace numbers
# hands from zero, so each shard runs its own seed and the shards cannot
# collide.
set -euo pipefail
umask 022
export HOME="${HOME:-/root}"
ROOT=/var/lib/hu-trace
LOG=/var/log/hu-trace
META='http://metadata.google.internal/computeMetadata/v1/instance/attributes'
IMETA='http://metadata.google.internal/computeMetadata/v1/instance'
HEADER='Metadata-Flavor: Google'
mkdir -p "$ROOT" "$LOG"
exec > >(tee -a "$LOG/startup.log" >/dev/console) 2>&1
meta() { curl -fsS -H "$HEADER" "$META/$1"; }
NAME="$(curl -fsS -H "$HEADER" "$IMETA/name")"
ZONE="$(curl -fsS -H "$HEADER" "$IMETA/zone" | rev | cut -d/ -f1 | rev)"
BUCKET="$(meta ht-bucket)"
PREFIX="$(meta ht-prefix)"
RUN_ID="$(meta ht-run-id)"
SHARD="$(meta ht-shard)"
HANDS="$(meta ht-hands)"
SEED="$(meta ht-seed)"
BINSTAMP="$(meta ht-binstamp)"
MODELS_OBJ="$(meta ht-models-object)"
CHUNK="$(meta ht-chunk)"
WATCHDOG="$(meta ht-watchdog-seconds)"

ship() {
  gcloud storage cp "$LOG/startup.log" \
    "gs://$BUCKET/$PREFIX/runs/$RUN_ID/logs/ht_${SHARD}_$(date -u +%s).log" \
    2>/dev/null || true
}
trap 'ship; gcloud compute instances delete "$NAME" --zone "$ZONE" --quiet || true' EXIT
trap 'echo "TERM received"; exit 143' TERM
( sleep "$WATCHDOG"; echo "WATCHDOG FIRED after ${WATCHDOG}s"; kill -TERM $$ ) &

OUT_OBJ="gs://$BUCKET/$PREFIX/runs/$RUN_ID/traces/$(printf '%04d' "$SHARD").jsonl"
if gcloud storage ls "$OUT_OBJ" >/dev/null 2>&1; then
  HAVE=$(gcloud storage cat "$OUT_OBJ" | wc -l)
  if [ "$HAVE" -ge "$HANDS" ]; then echo "shard $SHARD already published"; exit 0; fi
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
echo "PHASE:deps_ok"

cd "$ROOT"
gcloud storage cp "gs://$BUCKET/$PREFIX/bin/$BINSTAMP/t4_first_exact" t4_first_exact
gcloud storage cp "gs://$BUCKET/$PREFIX/bin/$BINSTAMP/BINARIES.sha256" BINARIES.sha256
chmod +x t4_first_exact
EXPECT="$(awk '/t4_first_exact/ {print $1}' BINARIES.sha256)"
ACTUAL="$(sha256sum t4_first_exact | awk '{print $1}')"
[ -n "$EXPECT" ] && [ "$EXPECT" = "$ACTUAL" ] || { echo "FATAL: binary sha mismatch"; exit 1; }
gcloud storage cp "gs://$BUCKET/$PREFIX/artifacts/$MODELS_OBJ" models.tar.gz
tar -xzf models.tar.gz
# The directory the tar actually carries, not one guessed from the object
# name: a bundle republished under a suffixed name (…_tr.tar.gz) still
# unpacks to its original directory, and guessing sent a whole fleet to
# "t0_bb.bin missing" in under a minute.
# `| head -1` would close the pipe early, SIGPIPE the tar, and under
# `set -o pipefail` kill this script with no message at all -- which is how
# the first two waves died.  sed reads the whole listing instead.
MODELS_DIR="$ROOT/$(tar -tzf models.tar.gz | sed -n '1{s|/.*||;p;}')"
for f in hu/t0_bb.bin rankers/t0_bb.bin policy.bin own_lap4/t0.bin; do
  [ -f "$MODELS_DIR/$f" ] || { echo "FATAL: $MODELS_DIR/$f missing"; exit 1; }
done
echo "PHASE:artifacts_ok"

mkdir -p ws/ai/config
cp "$MODELS_DIR/fl_ev.json" ws/ai/config/fl_ev.json 2>/dev/null || \
cat > ws/ai/config/fl_ev.json <<'JSON'
{"fl_ev": {"14": 6.57, "15": 16.61, "16": 38.76, "17": 70.07},
 "note": "current HU table; traces must be played under the serving table"}
JSON

N="t0_bb.bin t0_btn.bin t1_bb.bin t1_btn.bin t2_bb.bin t2_btn.bin t3_bb.bin t3_btn.bin"
HU=""; RK=""
for n in $N; do HU="$HU,$MODELS_DIR/hu/$n"; RK="$RK,$MODELS_DIR/rankers/$n"; done
HU=${HU#,}; RK=${RK#,}
OWN="$MODELS_DIR/own_lap4/t0.bin,$MODELS_DIR/own_lap4/t1.bin,$MODELS_DIR/own_lap4/t2.bin"

( cd ws && "$ROOT/t4_first_exact" --hu-match --hu-trace "$HANDS" \
    --trace-chunk "$CHUNK" --self-play-seed "$SEED" \
    --hu-a-models "$HU" --hu-b-models "$HU" \
    --hu-a-rankers "$RK" --hu-b-rankers "$RK" --hu-topk 4 \
    --hu-a-t0-policy "$MODELS_DIR/policy.bin" \
    --hu-b-t0-policy "$MODELS_DIR/policy.bin" --hu-t0-policy-topk 8 \
    --serve-joint-samples 200 --serve-joint-samples-b 200 \
    --arm-a-own "$OWN" --arm-b-own "$OWN" \
    --fl-ev-config ai/config/fl_ev.json \
    --output "$ROOT/trace.jsonl" ) &
WORKER=$!
# The binary flushes every chunk, so a partial file is always publishable:
# a preemption costs the current chunk, not the shard.
( while sleep 180; do
    [ -s "$ROOT/trace.jsonl" ] && gcloud storage cp "$ROOT/trace.jsonl" "$OUT_OBJ" >/dev/null 2>&1 || true
  done ) &
PARTIAL=$!
wait "$WORKER"
kill "$PARTIAL" 2>/dev/null || true

LINES=$(wc -l < "$ROOT/trace.jsonl")
[ "$LINES" -ge "$HANDS" ] || { echo "FATAL: wrote $LINES of $HANDS"; exit 1; }
gcloud storage cp "$ROOT/trace.jsonl" "$OUT_OBJ"
echo "PHASE:generate_ok"
echo "SHARD_DONE ht $SHARD $LINES"
