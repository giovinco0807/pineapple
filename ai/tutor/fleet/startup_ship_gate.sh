#!/usr/bin/env bash
# Ship-gate worker: price the 120 challenger against the shipped 96 on a slice
# of fresh canonical classes.
#
# Sibling of `startup_sharpen.sh`; the operational discipline carries over
# (log ships on any exit, binary sha-verified, resume from the object store,
# partials published every two minutes, instance deletes itself).  What is
# different:
#
#   * the work unit is a slice of the fresh-class requests file; the seed is
#     derived from each class's ABSOLUTE canonical index, so slicing never
#     renumbers and a shard is byte-compatible with a local run.
#   * it stages a second T0 image (the challenger).  Only T0 differs between
#     the two arms; T1/T2 are the shipped ones in both, because the gate
#     prices one decision and not a different chain.
#   * the binary must carry the 120-dim encoder, which was added after
#     binstamp 20260830b -- so the width is checked functionally below rather
#     than assumed from the stamp.
#   * no python packages are needed: the driver is stdlib only.
set -euo pipefail
umask 022
export HOME="${HOME:-/root}"
ROOT=/var/lib/ship-gate
LOG=/var/log/ship-gate
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
CHAL_OBJ="$(meta hu-challenger-object)"
MODELS_OBJ="$(meta hu-models-object)"
MODELS_SUBDIR="$(meta hu-models-subdir)"
BATCHES="$(meta hu-batches)"
ROLLOUTS="$(meta hu-rollouts)"
WATCHDOG="$(meta hu-watchdog-seconds)"

ship() {
  gcloud storage cp "$LOG/startup.log" \
    "gs://$BUCKET/$PREFIX/runs/$RUN_ID/logs/shipgate_${START}_$(date -u +%s).log" \
    2>/dev/null || true
}
trap 'ship; gcloud compute instances delete "$NAME" --zone "$ZONE" --quiet || true' EXIT
trap 'echo "TERM received"; exit 143' TERM
( sleep "$WATCHDOG"; echo "WATCHDOG FIRED after ${WATCHDOG}s"; kill -TERM $$ ) &

OUT_OBJ="gs://$BUCKET/$PREFIX/runs/$RUN_ID/mine/$(printf '%06d' "$START").jsonl"
if gcloud storage ls "$OUT_OBJ" >/dev/null 2>&1; then
  DONE=$(gcloud storage cat "$OUT_OBJ" | wc -l)
  if [ "$DONE" -ge "$COUNT" ]; then
    echo "already published: $OUT_OBJ ($DONE classes)"
    exit 0
  fi
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
echo "PHASE:deps_ok"

cd "$ROOT"
gcloud storage cp "gs://$BUCKET/$PREFIX/src/$SRC" src.tar.gz
mkdir -p src work _gate
tar -xzf src.tar.gz -C src
mkdir -p src/ai/rust_solver/target/release
gcloud storage cp "gs://$BUCKET/$PREFIX/bin/$BINSTAMP/t4_first_exact" \
  src/ai/rust_solver/target/release/t4_first_exact
gcloud storage cp "gs://$BUCKET/$PREFIX/bin/$BINSTAMP/BINARIES.sha256" BINARIES.sha256
chmod +x src/ai/rust_solver/target/release/t4_first_exact
EXPECT="$(awk '/t4_first_exact/ {print $1}' BINARIES.sha256)"
ACTUAL="$(sha256sum src/ai/rust_solver/target/release/t4_first_exact | awk '{print $1}')"
[ -n "$EXPECT" ] && [ "$EXPECT" = "$ACTUAL" ] || { echo "FATAL: binary sha mismatch"; exit 1; }
src/ai/rust_solver/target/release/t4_first_exact --help 2>&1 | grep -q -- '--fl-t0-deep' \
  || { echo "FATAL: binary $BINSTAMP has no --fl-t0-deep"; exit 1; }

gcloud storage cp "gs://$BUCKET/$PREFIX/artifacts/$MODELS_OBJ" models.tar.gz
tar -xzf models.tar.gz
MODELS_DIR="$ROOT/$(basename "$MODELS_OBJ" .tar.gz)/$MODELS_SUBDIR"
for m in t0.bin t1.bin t2.bin; do
  [ -f "$MODELS_DIR/$m" ] || { echo "FATAL: $MODELS_DIR/$m missing"; exit 1; }
done
gcloud storage cp "gs://$BUCKET/$PREFIX/artifacts/$REQ_OBJ" fresh.jsonl
gcloud storage cp "gs://$BUCKET/$PREFIX/artifacts/$CHAL_OBJ" challenger.bin
wc -l < fresh.jsonl
# Functional width check: the challenger is 120-dim and `playout.rs` selects
# its encoder by width, so a binary built before that branch existed refuses
# the model.  Asking the stamp would not catch it; asking the binary does.
src/ai/rust_solver/target/release/t4_first_exact --fl-t0-deep   --t0-cards "2h,Kh,8d,Kd,8c" --rollouts 0   --arm-a-own "$ROOT/challenger.bin,$MODELS_DIR/t1.bin,$MODELS_DIR/t2.bin"   --fl-ev-config "$ROOT/src/ai/config/fl_ev.json" --output /tmp/widthprobe.jsonl   || { echo "FATAL: binary $BINSTAMP cannot serve the 120-dim challenger"; exit 1; }
echo "PHASE:width_ok"

# Resume: the published shard is one JSON object per finished root; exploding
# it back into per-root files is what the driver checks to skip work.
if gcloud storage cp "$OUT_OBJ" resume.jsonl 2>/dev/null; then
  python3 - <<'PY'
import json, pathlib
out = pathlib.Path("/var/lib/ship-gate/_gate")
out.mkdir(parents=True, exist_ok=True)
n = 0
for line in open("/var/lib/ship-gate/resume.jsonl", encoding="utf-8"):
    if line.strip():
        rec = json.loads(line)
        (out / f"{rec['id']}.json").write_text(json.dumps(rec), encoding="utf-8")
        n += 1
print("resumed", n, "classes")
PY
fi
echo "PHASE:artifacts_ok"

SHARD="$ROOT/shard.jsonl"
( while sleep 120; do
    [ -s "$SHARD" ] || continue
    gcloud storage cp "$SHARD" "$OUT_OBJ" >/dev/null 2>&1 || true
  done ) &
PARTIAL_PID=$!

# --out-dir is the parent of the driver's per-class shard directory
# ($ROOT/_gate), which is also where the resume step put its files.
python3 src/ai/tutor/ship_gate_fresh.py \
  --requests "$ROOT/fresh.jsonl" \
  --challenger "$ROOT/challenger.bin" \
  --models-dir "$MODELS_DIR" \
  --binary "$ROOT/src/ai/rust_solver/target/release/t4_first_exact" \
  --fl-ev-config "$ROOT/src/ai/config/fl_ev.json" \
  --out-dir "$ROOT" --work "$ROOT/work" \
  --batches "$BATCHES" --rollouts "$ROLLOUTS" \
  --start "$START" --count "$COUNT" --shard-jsonl "$SHARD"
kill "$PARTIAL_PID" 2>/dev/null || true
[ -s "$SHARD" ] || { echo "FATAL: ship gate wrote nothing"; exit 1; }
echo "classes: $(wc -l < "$SHARD")"
gcloud storage cp "$SHARD" "$OUT_OBJ"
echo "PHASE:generate_ok"
echo "SHARD_DONE shipgate $START"
