#!/usr/bin/env bash
# Build the Linux `fl_solver` the T2 label fleet runs, and stash it in GCS.
# Sibling of `build_t1_binary.sh`; same stamping and self-deletion discipline.
set -euo pipefail
umask 022
export HOME="${HOME:-/root}"
export RUSTUP_HOME="$HOME/.rustup"
export CARGO_HOME="$HOME/.cargo"
export PATH="$CARGO_HOME/bin:$PATH"
ROOT=/var/lib/fls-build
LOG=/var/log/fls-build
META='http://metadata.google.internal/computeMetadata/v1/instance/attributes'
IMETA='http://metadata.google.internal/computeMetadata/v1/instance'
HEADER='Metadata-Flavor: Google'
mkdir -p "$ROOT" "$LOG"
exec > >(tee -a "$LOG/build.log" >/dev/console) 2>&1
meta() { curl -fsS -H "$HEADER" "$META/$1"; }
BUCKET="$(meta t2l-bucket)"
PREFIX="$(meta t2l-prefix)"
SRC="$(meta t2l-src)"
STAMP="$(meta t2l-binstamp)"
NAME="$(curl -fsS -H "$HEADER" "$IMETA/name")"
ZONE="$(curl -fsS -H "$HEADER" "$IMETA/zone" | rev | cut -d/ -f1 | rev)"

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
# No `-C target-cpu`: Spot falls back across machine families.
cargo build --release -p fl_solver
echo "PHASE:build_ok"

BIN=target/release/fl_solver
sha256sum "$BIN" | awk '{print $1"  fl_solver"}' > BINARIES.sha256
cat BINARIES.sha256
gcloud storage cp "$BIN" "gs://$BUCKET/$PREFIX/bin/$STAMP/fl_solver"
gcloud storage cp BINARIES.sha256 "gs://$BUCKET/$PREFIX/bin/$STAMP/BINARIES.sha256"
echo "built $(date -u +%FT%TZ)" \
  | gcloud storage cp - "gs://$BUCKET/$PREFIX/bin/$STAMP/DONE.txt"
echo "PHASE:published"

# Delete, do not stop: a stopped Spot VM keeps billing for its disk.
gcloud compute instances delete "$NAME" --zone "$ZONE" --quiet
