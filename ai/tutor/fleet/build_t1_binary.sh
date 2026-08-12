#!/usr/bin/env bash
# Build the Linux `t4_first_exact` the T1-vs-FL fleet runs, and stash it in GCS.
#
# Cross-compiling from the Windows workstation is not worth the toolchain, and
# the crate is a workspace member with two path dependencies (`fl_solver`,
# `ofc_core`) plus eight sibling members the workspace manifest insists exist,
# so the unit that ships is the whole `ai/rust_solver` tree.  This runs as a
# one-shot Spot instance's startup script: it builds, publishes, and deletes
# itself.
#
# The binary is published under a stamp rather than a fixed name so an
# in-flight run can never have its binary swapped underneath it, and its
# sha256 goes up beside it because a silently substituted labeler corrupts
# every label it touches.
set -euo pipefail
umask 022
# The metadata script runner starts this with no HOME, and `set -u` turns
# rustup's `$HOME` into a fatal unbound variable.
export HOME="${HOME:-/root}"
export RUSTUP_HOME="$HOME/.rustup"
export CARGO_HOME="$HOME/.cargo"
export PATH="$CARGO_HOME/bin:$PATH"
ROOT=/var/lib/t1-build
LOG=/var/log/t1-build
META='http://metadata.google.internal/computeMetadata/v1/instance/attributes'
IMETA='http://metadata.google.internal/computeMetadata/v1/instance'
HEADER='Metadata-Flavor: Google'
mkdir -p "$ROOT" "$LOG"
exec > >(tee -a "$LOG/build.log" >/dev/console) 2>&1
meta() { curl -fsS -H "$HEADER" "$META/$1"; }
BUCKET="$(meta t1-bucket)"
PREFIX="$(meta t1-prefix)"
SRC="$(meta t1-src)"
STAMP="$(meta t1-binstamp)"
NAME="$(curl -fsS -H "$HEADER" "$IMETA/name")"
ZONE="$(curl -fsS -H "$HEADER" "$IMETA/zone" | rev | cut -d/ -f1 | rev)"

# Ship the log on ANY exit: a build that dies with its explanation on a disk
# that vanishes with the instance costs a whole cycle to re-derive.
trap 'gcloud storage cp "$LOG/build.log" \
  "gs://$BUCKET/$PREFIX/bin/$STAMP/build.log" 2>/dev/null || true' EXIT

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq build-essential curl pkg-config
echo "PHASE:deps_ok"

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
  | sh -s -- -y --profile minimal --default-toolchain stable --no-modify-path
rustc --version
echo "PHASE:toolchain_ok"

cd "$ROOT"
gcloud storage cp "gs://$BUCKET/$PREFIX/src/$SRC" src.tar.gz
tar -xzf src.tar.gz
cd ai/rust_solver
# No `-C target-cpu`: the fleet falls back across machine families when Spot is
# dry (c4 / n2 / e2 are Emerald Rapids / Cascade Lake / Broadwell-or-EPYC), and
# a binary tuned for the builder would fault on the fallback.
cargo build --release -p t4_first_exact
echo "PHASE:build_ok"

BIN=target/release/t4_first_exact
./"$BIN" --help >/dev/null 2>&1 || true
sha256sum "$BIN" | awk '{print $1"  t4_first_exact"}' > BINARIES.sha256
cat BINARIES.sha256
gcloud storage cp "$BIN" "gs://$BUCKET/$PREFIX/bin/$STAMP/t4_first_exact"
gcloud storage cp BINARIES.sha256 "gs://$BUCKET/$PREFIX/bin/$STAMP/BINARIES.sha256"
echo "built $(date -u +%FT%TZ)" \
  | gcloud storage cp - "gs://$BUCKET/$PREFIX/bin/$STAMP/DONE.txt"
echo "PHASE:published"

# Delete, do not stop.  A Spot VM that stops itself keeps billing for its
# disk; 84 stopped-but-undeleted instances cost this project ~$10 once.
gcloud compute instances delete "$NAME" --zone "$ZONE" --quiet
