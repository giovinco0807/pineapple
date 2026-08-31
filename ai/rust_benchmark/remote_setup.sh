#!/bin/bash
# Remote setup script - runs on each GCP VM
# Usage: bash remote_setup.sh <seed_base>
set -e

SEED_BASE=${1:-100000}

# Install Rust
if ! command -v cargo &>/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
source "$HOME/.cargo/env"

# Install build deps
sudo apt-get update -qq
sudo apt-get install -y -qq build-essential pkg-config libssl-dev

# Extract and build
sudo mkdir -p /home/ofc && sudo chown $(whoami) /home/ofc
cd /home/ofc
tar xzf ofc_selfplay.tar.gz
cd rust_benchmark
cargo build --release 2>&1 | tail -3
echo "Build OK: $(ls -lh target/release/ofc_benchmark)"

# Launch 12 parallel selfplay processes
mkdir -p /home/ofc/results
for i in $(seq 0 11); do
    SEED=$((SEED_BASE + i * 250))
    echo "Starting proc $i: seed=$SEED games=250"
    nohup ./target/release/ofc_benchmark \
        --games 250 --seed $SEED --mode selfplay \
        --mcts-sims 400 --rollouts 250 \
        --output /home/ofc/results/selfplay_${i}.jsonl \
        > /home/ofc/results/selfplay_${i}.log 2>&1 &
done

echo ""
echo "12 processes started (seed_base=$SEED_BASE)"
echo "PIDs: $(pgrep -f ofc_benchmark | tr '\n' ' ')"
