#!/bin/bash
exec > >(tee /var/log/startup-script.log) 2>&1
echo "Starting GCP RL continuous pipeline..."

sudo apt-get update
sudo apt-get install -y git wget curl python3-pip python3-venv build-essential pkg-config libssl-dev cmake python-is-python3 tmux

# Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
export PATH="$HOME/.cargo/bin:$PATH"

# Setup repo
git clone https://github.com/giovinco0807/pineapple.git /ofc-pineapple
cd /ofc-pineapple

# Get branch from metadata
BRANCH=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/git-branch" -H "Metadata-Flavor: Google")
if [ -n "$BRANCH" ]; then
    git checkout $BRANCH
fi

# Initial weights / sync
gsutil -m rsync -r gs://ofc-solver-485418/rl_output/models ai/models/
gsutil cp gs://ofc-solver-485418/rl_output/training_metrics.csv ai/ || true

# Python
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install matplotlib pandas numpy

# Build rust
cd rust_solver
cargo build --release --bin self_play_worker
cd ..

# Run continuous RL in a detached tmux session so we can attach if needed, but the startup script will exit and let it run
chmod +x run_continuous_rl.py
tmux new-session -d -s rl 'source venv/bin/activate && python run_continuous_rl.py 2>&1 | tee rl_output.log'
