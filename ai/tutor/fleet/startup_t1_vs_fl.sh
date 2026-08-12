#!/usr/bin/env bash
# T1-vs-FL label worker (best-responding Fantasyland opponent, pool leaf).
#
# Successor to `startup_joker_labelgen.sh`, which drove the superseded static
# FL-library pipeline.  What carries over is everything operational that was
# learned the expensive way:
#
#   * the log ships on ANY exit -- an instance that dies with its explanation
#     on a disk that stops being readable the moment it stops costs a cycle
#     to re-derive, three times over;
#   * the binary is verified by sha256 rather than trusted, because a silently
#     substituted labeler corrupts every label it writes;
#   * a shard's already-published work is read from the object store, not from
#     local disk, because a recreated Spot instance boots with an empty one;
#   * labels are published as they are produced (see `run_t1_shard.py`);
#   * the instance DELETES itself.  A Spot VM that merely stops keeps billing
#     for its disk: 84 stopped-but-undeleted instances cost ~$10 here once.
set -euo pipefail
umask 022
# The metadata script runner starts this with no HOME, and `set -u` makes any
# tool that reads it (gcloud's config dir, pip, cargo) a fatal unbound
# variable instead of a default.
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
BUCKET="$(meta t1-bucket)"
PREFIX="$(meta t1-prefix)"
RUN_ID="$(meta t1-run-id)"
SEED="$(meta t1-seed)"
ROOTS="$(meta t1-roots)"
SRC="$(meta t1-src)"
BINSTAMP="$(meta t1-binstamp)"
POOL_OBJ="$(meta t1-pool-object)"
T2_OBJ="$(meta t1-t2-object)"
T3_OBJ="$(meta t1-t3-object)"
PUBLISH="$(meta t1-publish-seconds)"
WATCHDOG="$(meta t1-watchdog-seconds)"
EXTRA="$(meta t1-extra-args)"

ship() {
  gcloud storage cp "$LOG/startup.log" \
    "gs://$BUCKET/$PREFIX/runs/$RUN_ID/logs/${SEED}_$(date -u +%s).log" \
    2>/dev/null || true
}
# Every exit path -- success, labeler failure, watchdog -- ships the log and
# then deletes the instance.  Nothing this script can do leaves a VM running.
trap 'ship; gcloud compute instances delete "$NAME" --zone "$ZONE" --quiet || true' EXIT
trap 'echo "TERM received"; exit 143' TERM
( sleep "$WATCHDOG"; echo "WATCHDOG FIRED after ${WATCHDOG}s"; kill -TERM $$ ) &

mark() { echo "PHASE:$1"; echo "$1 $(date -u +%FT%TZ)" \
  | gcloud storage cp - "gs://$BUCKET/$PREFIX/runs/$RUN_ID/progress/${SEED}_$1.txt" \
  2>/dev/null || true; }

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
# numpy only: the request generator reaches the T1 teacher's dealing logic,
# whose import chain stops well short of torch.
apt-get install -y -qq python3-numpy
mark deps_ok

cd "$ROOT"
gcloud storage cp "gs://$BUCKET/$PREFIX/src/$SRC" src.tar.gz
tar -xzf src.tar.gz
mkdir -p bin artifacts
gcloud storage cp "gs://$BUCKET/$PREFIX/bin/$BINSTAMP/t4_first_exact" bin/t4_first_exact
gcloud storage cp "gs://$BUCKET/$PREFIX/bin/$BINSTAMP/BINARIES.sha256" bin/BINARIES.sha256
chmod +x bin/t4_first_exact
EXPECT="$(awk '/t4_first_exact/ {print $1}' bin/BINARIES.sha256)"
ACTUAL="$(sha256sum bin/t4_first_exact | awk '{print $1}')"
[ -n "$EXPECT" ] && [ "$EXPECT" = "$ACTUAL" ] || { echo "FATAL: binary sha mismatch"; exit 1; }
gcloud storage cp "gs://$BUCKET/$PREFIX/artifacts/$POOL_OBJ" artifacts/pool.jfl1
gcloud storage cp "gs://$BUCKET/$PREFIX/artifacts/$T2_OBJ" artifacts/t2_evaluator.bin
gcloud storage cp "gs://$BUCKET/$PREFIX/artifacts/$T3_OBJ" artifacts/t3_evaluator.bin
for path in artifacts/pool.jfl1 artifacts/t2_evaluator.bin artifacts/t3_evaluator.bin \
            ai/config/fl_ev.json; do
  [ -s "$path" ] || { echo "FATAL: missing artefact $path"; exit 1; }
done
echo "artifacts: $(du -sh artifacts | cut -f1), pool $(stat -c%s artifacts/pool.jfl1) bytes"
mark artifacts_ok

export PYTHONPATH="$ROOT"
cd "$ROOT"
# shellcheck disable=SC2086
python3 -m ai.tutor.fleet.run_t1_shard \
  --bucket "$BUCKET" --prefix "$PREFIX" --run-id "$RUN_ID" \
  --seed "$SEED" --roots "$ROOTS" \
  --work-dir "$ROOT/work" \
  --solver "$ROOT/bin/t4_first_exact" \
  --pool "$ROOT/artifacts/pool.jfl1" \
  --t2-model "$ROOT/artifacts/t2_evaluator.bin" \
  --t3-model "$ROOT/artifacts/t3_evaluator.bin" \
  --fl-ev-config "$ROOT/ai/config/fl_ev.json" \
  --publish-seconds "$PUBLISH" $EXTRA
mark generate_ok
echo "SHARD_DONE $SEED"
