#!/usr/bin/env bash
# Rust-teacher worker: run one t4_first_exact labelling mode over a slice of
# a requests file and publish the labels as a single shard object.
#
# Third sibling of startup_hu_street / startup_hu_match, for the teachers
# that are Rust modes rather than Python drivers -- the T3 teachers read the
# exact V4 directly and need no boundary net, no torch, no numpy, nothing
# but the verified binary and a chooser model.  The operational spine is
# unchanged: log ships on any exit, binary verified by sha256, instance
# deletes itself (the reaper does it in practice; the service account cannot,
# and max-run-duration backstops both).
#
# A shard's output is one object, published create-only when the mode
# finishes, and the shard skips itself if that object already exists -- so
# preemption recovery is relaunching with --skip-done and nothing needs to
# agree with anything.
set -euo pipefail
umask 022
export HOME="${HOME:-/root}"
ROOT=/var/lib/hu-teach
LOG=/var/log/hu-teach
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
MODE="$(meta hu-mode)"
START="$(meta hu-start)"
COUNT="$(meta hu-count)"
SRC="$(meta hu-src)"
BINSTAMP="$(meta hu-binstamp)"
REQ_OBJ="$(meta hu-requests-object)"
CHOOSER_OBJ="$(meta hu-chooser-object)"
WATCHDOG="$(meta hu-watchdog-seconds)"

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

OUT="gs://$BUCKET/$PREFIX/runs/$RUN_ID/labels/$JOB/$(printf '%07d' "$START").jsonl"
if gcloud storage ls "$OUT" >/dev/null 2>&1; then
  echo "already published: $OUT"
  mark generate_ok
  exit 0
fi
mark deps_ok

cd "$ROOT"
gcloud storage cp "gs://$BUCKET/$PREFIX/src/$SRC" src.tar.gz
mkdir -p src
tar -xzf src.tar.gz -C src
gcloud storage cp "gs://$BUCKET/$PREFIX/bin/$BINSTAMP/t4_first_exact" t4_first_exact
gcloud storage cp "gs://$BUCKET/$PREFIX/bin/$BINSTAMP/BINARIES.sha256" BINARIES.sha256
chmod +x t4_first_exact
EXPECT="$(awk '/t4_first_exact/ {print $1}' BINARIES.sha256)"
ACTUAL="$(sha256sum t4_first_exact | awk '{print $1}')"
[ -n "$EXPECT" ] && [ "$EXPECT" = "$ACTUAL" ] || { echo "FATAL: binary sha mismatch"; exit 1; }
gcloud storage cp "gs://$BUCKET/$PREFIX/artifacts/$REQ_OBJ" requests.jsonl
gcloud storage cp "gs://$BUCKET/$PREFIX/artifacts/$CHOOSER_OBJ" chooser.bin
for path in requests.jsonl chooser.bin src/ai/config/fl_ev.json; do
  [ -s "$path" ] || { echo "FATAL: missing artefact $path"; exit 1; }
done
sed -n "$((START + 1)),$((START + COUNT))p" requests.jsonl > slice.jsonl
GOT="$(wc -l < slice.jsonl)"
[ "$GOT" -eq "$COUNT" ] || { echo "FATAL: slice has $GOT of $COUNT rows"; exit 1; }
mark artifacts_ok

./t4_first_exact "--$MODE" \
  --t2-t3-model chooser.bin \
  --input slice.jsonl \
  --output labels.jsonl \
  --fl-ev-config src/ai/config/fl_ev.json
LINES="$(wc -l < labels.jsonl)"
# A teacher that quietly dropped roots poisons the merge downstream, where a
# short file reads as a smaller corpus, not as an error.
[ "$LINES" -eq "$COUNT" ] || { echo "FATAL: labeler wrote $LINES of $COUNT"; exit 1; }
gcloud storage cp --if-generation-match=0 labels.jsonl "$OUT" || \
  gcloud storage ls "$OUT" >/dev/null
mark generate_ok
echo "SHARD_DONE $JOB $START"
