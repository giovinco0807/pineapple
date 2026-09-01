#!/usr/bin/env bash
# T1 own-label worker: price a line-window of a T1 request file with
# `t4_first_exact --t1-vs-fl-library`.
#
# Sibling of `startup_t2_labelgen.sh` with three differences: the labeler is
# t4_first_exact (not fl_solver), positions ALWAYS come from a requests
# object (T1 has no dealt fallback -- the request lines carry the knobs:
# t2_samples/t3_samples/t4_draw_sample/own_only), and the T2/T3 movers ship
# as a small artifacts tar.  The fl_ev config is written here as the prev
# table pinned by fl14_v1.jfl1, matching every own-chain label since 8/14.
set -euo pipefail
umask 022
export HOME="${HOME:-/root}"
ROOT=/var/lib/t1-labelgen
LOG=/var/log/t1-labelgen
META='http://metadata.google.internal/computeMetadata/v1/instance/attributes'
IMETA='http://metadata.google.internal/computeMetadata/v1/instance'
HEADER='Metadata-Flavor: Google'
mkdir -p "$ROOT" "$LOG"
exec > >(tee -a "$LOG/startup.log" >/dev/console) 2>&1
meta() { curl -fsS -H "$HEADER" "$META/$1"; }
NAME="$(curl -fsS -H "$HEADER" "$IMETA/name")"
ZONE="$(curl -fsS -H "$HEADER" "$IMETA/zone" | rev | cut -d/ -f1 | rev)"
BUCKET="$(meta t1l-bucket)"
PREFIX="$(meta t1l-prefix)"
RUN_ID="$(meta t1l-run-id)"
START="$(meta t1l-start)"
COUNT="$(meta t1l-count)"
BINSTAMP="$(meta t1l-binstamp)"
POOL_OBJ="$(meta t1l-pool-object)"
REQS_OBJ="$(meta t1l-requests-object)"
MOVERS_OBJ="$(meta t1l-movers-object)"
CHUNK="$(meta t1l-chunk)"
WATCHDOG="$(meta t1l-watchdog-seconds)"

ship() {
  gcloud storage cp "$LOG/startup.log" \
    "gs://$BUCKET/$PREFIX/runs/$RUN_ID/logs/t1l_${START}_$(date -u +%s).log" \
    2>/dev/null || true
}
trap 'ship; gcloud compute instances delete "$NAME" --zone "$ZONE" --quiet || true' EXIT
trap 'echo "TERM received"; exit 143' TERM
( sleep "$WATCHDOG"; echo "WATCHDOG FIRED after ${WATCHDOG}s"; kill -TERM $$ ) &

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
gcloud storage cp "gs://$BUCKET/$PREFIX/artifacts/$POOL_OBJ" pool.jfl1
gcloud storage cp "gs://$BUCKET/$PREFIX/artifacts/$REQS_OBJ" requests_all.jsonl
gcloud storage cp "gs://$BUCKET/$PREFIX/artifacts/$MOVERS_OBJ" movers.tar.gz
tar -xzf movers.tar.gz
[ -f movers/t2_mover.bin ] && [ -f movers/t3_mover.bin ] || { echo "FATAL: movers missing"; exit 1; }
wc -l < requests_all.jsonl
echo "PHASE:artifacts_ok"

mkdir -p ws/ai/config
cat > ws/ai/config/fl_ev.json <<'JSON'
{
  "fl_ev": {"14": 0.0, "15": 10.7, "16": 29.9, "17": 63.5},
  "note": "fl_ev_prev, pinned by fl14_v1.jfl1; own-chain label semantics since 2026-08-14"
}
JSON

for ((base = START; base < START + COUNT; base += CHUNK)); do
  n=$((START + COUNT - base)); [ "$n" -gt "$CHUNK" ] && n=$CHUNK
  OBJ="gs://$BUCKET/$PREFIX/runs/$RUN_ID/labels/$(printf '%08d' "$base").jsonl"
  if gcloud storage ls "$OBJ" >/dev/null 2>&1; then
    HAVE=$(gcloud storage cat "$OBJ" | wc -l)
    if [ "$HAVE" -ge "$n" ]; then echo "chunk $base already published"; continue; fi
  fi
  sed -n "$((base + 1)),$((base + n))p" requests_all.jsonl > chunk_in.jsonl
  [ "$(wc -l < chunk_in.jsonl)" -eq "$n" ] || { echo "FATAL: request slice short"; exit 1; }
  ./t4_first_exact --t1-vs-fl-library \
    --fl-pool pool.jfl1 \
    --t1-t2-model movers/t2_mover.bin \
    --t2-t3-model movers/t3_mover.bin \
    --fl-ev-config ws/ai/config/fl_ev.json \
    --input chunk_in.jsonl --output chunk_out.jsonl --chunk-size 16
  LINES=$(wc -l < chunk_out.jsonl)
  [ "$LINES" -ge "$n" ] || { echo "FATAL: chunk $base wrote $LINES of $n"; exit 1; }
  gcloud storage cp chunk_out.jsonl "$OBJ"
  echo "chunk $base done ($LINES roots)"
done
echo "PHASE:generate_ok"
echo "SHARD_DONE t1l $START"
