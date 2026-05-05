#!/bin/bash
PROJECT_ID="ofc-solver-485418"
ZONE="asia-northeast1-b"
MACHINE_TYPE="c2d-highcpu-112"
IMAGE_FAMILY="ubuntu-2204-lts"
IMAGE_PROJECT="ubuntu-os-cloud"
INSTANCE_NAME="ofc-rl-master"
BUCKET="gs://ofc-solver-485418/ofc_rl_output"

echo "Creating Cloud Storage Bucket if it doesn't exist..."
gcloud storage buckets create $BUCKET --project=$PROJECT_ID --location=asia-northeast1 2>/dev/null || true

echo "Launching $INSTANCE_NAME..."

gcloud compute instances create $INSTANCE_NAME \
  --project=$PROJECT_ID \
  --zone=$ZONE \
  --machine-type=$MACHINE_TYPE \
  --provisioning-model=SPOT \
  --instance-termination-action=DELETE \
  --image-family=$IMAGE_FAMILY \
  --image-project=$IMAGE_PROJECT \
  --boot-disk-size=100GB \
  --scopes=https://www.googleapis.com/auth/cloud-platform \
  --metadata=startup-script="#!/bin/bash
exec > >(tee /var/log/startup-script.log) 2>&1
echo 'Starting OFC Pineapple Continuous RL Pipeline...'

sudo apt-get update
sudo apt-get install -y git python3-pip python3-venv tmux curl build-essential

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source \"\$HOME/.cargo/env\"
export PATH=\"\$HOME/.cargo/bin:\$PATH\"

# Clone Repository
git clone https://github.com/giovinco0807/pineapple.git /ofc-pineapple
cd /ofc-pineapple
# Checkout the current branch used for verification
git checkout verify-gcp-phase-one-20260501

# Setup Python
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas matplotlib

# Create directories
mkdir -p ai/models/checkpoints
mkdir -p ai/data/selfplay_replays

# Download BC checkpoints from GCS
gsutil -m rsync -r gs://ofc-solver-485418/ofc_rl_output/models/checkpoints ai/models/checkpoints/ || true

# Pre-build rust binary
cd ai/rust_solver/mcts_gen
cargo build --release --bin self_play
cd ../../../

# Run RL loop in tmux
cat << 'EOF' > run_rl.sh
#!/bin/bash
source venv/bin/activate
source \"\$HOME/.cargo/env\"
export PATH=\"\$HOME/.cargo/bin:\$PATH\"

python ai/training/train_selfplay.py --threads 110 --iterations 100 > rl_pipeline.log 2>&1

echo \"Pipeline finished. Shutting down...\"
sudo poweroff
EOF

chmod +x run_rl.sh
# Run in detached tmux session so we can attach if needed
tmux new-session -d -s rl_pipeline './run_rl.sh'
"

echo "Instance launched! SSH into it and run 'tmux attach -t rl_pipeline' to view progress."
