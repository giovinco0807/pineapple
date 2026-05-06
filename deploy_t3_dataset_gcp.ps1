$PROJECT_ID="ofc-solver-485418"
$ZONE="asia-northeast1-b"
$MACHINE_TYPE="c2d-highcpu-56"
$IMAGE_FAMILY="ubuntu-2204-lts"
$IMAGE_PROJECT="ubuntu-os-cloud"
$INSTANCE_NAME="ofc-rl-t3-dataset"
$BUCKET="gs://ofc-solver-485418/ofc_rl_output/t3_dataset"

Write-Host "Creating Cloud Storage Bucket if it doesn't exist..."
gcloud storage buckets create gs://$PROJECT_ID --project=$PROJECT_ID --location=asia-northeast1

Write-Host "Launching $INSTANCE_NAME..."

$STARTUP_SCRIPT=@"
#!/bin/bash
exec > >(tee /var/log/startup-script.log) 2>&1
echo 'Starting OFC Pineapple T3 Dataset Generation Pipeline...'

sudo apt-get update
sudo apt-get install -y git python3-pip python3-venv tmux curl build-essential

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "`$HOME/.cargo/env"
export PATH="`$HOME/.cargo/bin:`$PATH"

# Clone Repository
git clone https://github.com/giovinco0807/pineapple.git /ofc-pineapple
cd /ofc-pineapple
# Checkout the current branch used for verification
git checkout verify-gcp-phase-one-20260501

# Setup Python
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy pandas matplotlib gymnasium tqdm

# Create directories
mkdir -p data/t3_dataset

# Run T3 dataset generation in tmux
cat << 'EOF' > run_generation.sh
#!/bin/bash
cd /ofc-pineapple
source venv/bin/activate
source /root/.cargo/env
export PATH="/root/.cargo/bin:`$PATH"

# Pre-build rust binary for T3
cd ai/rust_solver
cargo build --release --bin t3_exact
cd ../../

# Generate 50,000 states (Takes ~30 mins on 55 cores)
python3 ai/rust_solver/generate_t3_dataset.py --states 50000 > t3_generation.log 2>&1

echo "Uploading dataset to GCS..."
gsutil -m rsync -r data/t3_dataset gs://ofc-solver-485418/ofc_rl_output/t3_dataset/

echo "Pipeline finished. Shutting down..."
sudo poweroff
EOF

chmod +x run_generation.sh
# Run in detached tmux session so we can attach if needed
tmux new-session -d -s t3_dataset './run_generation.sh'
"@

[System.IO.File]::WriteAllText("$PWD\startup_t3.sh", $STARTUP_SCRIPT)

gcloud compute instances create $INSTANCE_NAME `
  --project=$PROJECT_ID `
  --zone=$ZONE `
  --machine-type=$MACHINE_TYPE `
  --provisioning-model=SPOT `
  --instance-termination-action=DELETE `
  --image-family=$IMAGE_FAMILY `
  --image-project=$IMAGE_PROJECT `
  --boot-disk-size=100GB `
  --scopes=https://www.googleapis.com/auth/cloud-platform `
  --metadata-from-file=startup-script=startup_t3.sh

Write-Host "Instance launched! Data will be uploaded to $BUCKET upon completion."
