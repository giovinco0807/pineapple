#!/bin/bash
# Build cfr_solver on GCP and upload to GCS
# Run on a build VM to create the Linux binary

set -e
cd /tmp

echo "=== Building cfr_solver for T0 batch ==="

# Install Rust
if ! command -v cargo &> /dev/null; then
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi
source "$HOME/.cargo/env"

# Clone repo
echo "Cloning repository..."
git clone https://github.com/giovinco0807/pineapple.git ofc-pineapple

# Build
echo "Building cfr_solver (release)..."
cd ofc-pineapple/ai/rust_solver
cargo build --release -p cfr_solver

# Upload binary to GCS
echo "Uploading binary to GCS..."
gcloud storage cp target/release/cfr_solver gs://ofc-solver-485418/cfr_solver

# Verify
echo "Binary uploaded. Size:"
ls -lh target/release/cfr_solver

echo "=== Build complete! ==="
echo "You can now launch the batch VMs."
echo "Shutting down..."
shutdown -h now
