#!/bin/bash
# GCP Spot VM setup and T1 MC data generation (Rust)
# Run this on the VM after uploading files

set -e

echo "=== Installing dependencies ==="
sudo apt-get update -qq
sudo apt-get install -y -qq build-essential curl

echo "=== Installing Rust ==="
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env

cd /tmp/t1_gen
echo "=== Extracting Rust solver ==="
tar -xzf rust_solver.tar.gz

echo "=== Building Rust T1 Generator ==="
cd rust_solver
cargo build --release --bin t1_gen

echo "=== Starting T1 Data Generation ==="
# With 32 vCPUs (e2-highcpu-32), rayon will spawn 32 threads automatically
# 50,000 hands, n1=5, n2=3, n3=3, n4=30
./target/release/t1_gen \
    --n-hands 50000 \
    --n1 5 \
    --n2 3 \
    --n3 3 \
    --n4 30 \
    --seed 42 \
    -o t1_data_50k.jsonl

echo "=== Generation complete ==="
wc -l t1_data_50k.jsonl

# Copy to GCS bucket for easy download
gsutil cp t1_data_50k.jsonl gs://ofc-solver-data/t1_data_50k.jsonl 2>/dev/null || echo "GCS upload skipped (bucket may not exist)"

echo "All done"
