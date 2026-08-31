#!/bin/bash
# GCP T0 Batch Evaluation Script
# Run on the VM after SSH connection

set -e

echo "=== Setting up T0 Batch Evaluation on GCP ==="

# 1. Install Rust (if not installed)
if ! command -v cargo &> /dev/null; then
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi

# 2. Build the solver
echo "Building cfr_solver (release)..."
cd ~/ofc-pineapple/ai/rust_solver
cargo build --release -p cfr_solver

echo "Build complete!"

# 3. Run batch evaluation
echo ""
echo "=== Starting T0 Batch Evaluation ==="
echo "200 hands × 1000 samples"
echo "Output: ~/t0_batch.jsonl"
echo ""

./target/release/cfr_solver t0-batch \
    --hands 200 \
    --samples 1000 \
    --output ~/t0_batch.jsonl \
    --seed 42

echo ""
echo "=== Batch complete! ==="
echo "Results saved to: ~/t0_batch.jsonl"
