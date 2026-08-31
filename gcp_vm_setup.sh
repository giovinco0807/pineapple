#!/bin/bash
# Remote VM setup script - uploaded and executed on each VM
set -e

echo "--- Installing system packages ---"
sudo apt-get update -qq
sudo apt-get install -y -qq python3.11 python3.11-venv python3.11-dev curl build-essential

echo "--- Creating Python venv ---"
python3.11 -m venv ~/venv
source ~/venv/bin/activate
pip install --upgrade pip -q
pip install torch --index-url https://download.pytorch.org/whl/cpu -q
pip install numpy -q

echo "--- Verifying Python ---"
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import numpy; print(f'NumPy {numpy.__version__}')"

echo "--- Installing Rust ---"
if ! command -v cargo &>/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
source "$HOME/.cargo/env"
rustc --version

echo "--- Building FL solver + prob_engine ---"
cd ~/ofc-pineapple/ai/rust_solver
cargo build --release -p fl_solver -p prob_engine
ls -la target/release/fl_solver target/release/prob_engine

echo "--- Patching paths for Linux ---"
cd ~/ofc-pineapple
sed -i 's/fl_solver\.exe/fl_solver/' ai/rust_solver_wrapper.py 2>/dev/null || true
sed -i 's/prob_engine\.exe/prob_engine/' ai/prob_engine_wrapper.py 2>/dev/null || true

echo "=== Setup complete ==="
