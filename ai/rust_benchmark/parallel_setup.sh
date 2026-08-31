#!/bin/bash
# Parallel setup of new VMs - each VM set up in background
# Usage: bash parallel_setup.sh

PROJECT="ofc-solver-485418"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

NEW_VMS=(
    "ofc-selfplay-5:us-central1-c"
    "ofc-selfplay-6:us-central1-f"
    "ofc-selfplay-7:us-east1-c"
    "ofc-selfplay-8:us-east1-d"
    "ofc-selfplay-9:us-west1-a"
    "ofc-selfplay-10:us-west1-c"
    "ofc-selfplay-11:europe-west1-c"
    "ofc-selfplay-12:europe-west1-d"
    "ofc-selfplay-13:us-east4-a"
    "ofc-selfplay-14:us-east4-c"
)

setup_vm() {
    local VM="${1%%:*}"
    local ZONE="${1##*:}"
    local LOG="/tmp/setup_${VM}.log"

    echo "[$(date +%H:%M:%S)] Starting $VM" | tee -a "$LOG"

    # Wait for SSH
    for retry in $(seq 1 10); do
        gcloud compute ssh "$VM" --zone="$ZONE" --command="echo SSH_OK" 2>/dev/null && break
        sleep 10
    done

    # Step 1: Install Rust + deps
    echo "[$(date +%H:%M:%S)] $VM: Installing Rust..." | tee -a "$LOG"
    gcloud compute ssh "$VM" --zone="$ZONE" --command='
        curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y 2>&1 | tail -1
        export PATH="$HOME/.cargo/bin:$PATH"
        sudo apt-get update -qq 2>&1 | tail -1
        sudo apt-get install -y -qq build-essential pkg-config libssl-dev g++ 2>&1 | tail -1
        echo "DEPS_DONE"
    ' >> "$LOG" 2>&1

    # Step 2: Upload source
    echo "[$(date +%H:%M:%S)] $VM: Uploading source..." | tee -a "$LOG"
    gcloud compute scp /tmp/ofc_selfplay.tar.gz "${VM}:/home/Owner/" --zone="$ZONE" >> "$LOG" 2>&1

    # Step 3: Build
    echo "[$(date +%H:%M:%S)] $VM: Building..." | tee -a "$LOG"
    gcloud compute ssh "$VM" --zone="$ZONE" --command='
        export PATH="$HOME/.cargo/bin:$PATH"
        cd /home/Owner
        tar xzf ofc_selfplay.tar.gz
        cd rust_benchmark
        cargo build --release 2>&1 | tail -3
        ls -lh target/release/ofc_benchmark
    ' >> "$LOG" 2>&1

    # Step 4: Upload models
    echo "[$(date +%H:%M:%S)] $VM: Uploading models..." | tee -a "$LOG"
    gcloud compute scp /tmp/ofc_models.tar.gz "${VM}:/home/Owner/" --zone="$ZONE" >> "$LOG" 2>&1
    gcloud compute ssh "$VM" --zone="$ZONE" --command='
        cd /home/Owner/rust_benchmark
        tar xzf /home/Owner/ofc_models.tar.gz
        ls -lh models/
    ' >> "$LOG" 2>&1

    # Step 5: Upload remote_run.sh
    gcloud compute scp "$LOCAL_DIR/remote_run.sh" "${VM}:/home/Owner/rust_benchmark/" --zone="$ZONE" >> "$LOG" 2>&1

    echo "[$(date +%H:%M:%S)] $VM: DONE!" | tee -a "$LOG"
}

echo "=== Parallel setup of ${#NEW_VMS[@]} VMs ==="
echo "=== Logs in /tmp/setup_ofc-selfplay-*.log ==="
echo ""

for entry in "${NEW_VMS[@]}"; do
    setup_vm "$entry" &
done

echo "All setup jobs launched. Waiting..."
wait
echo ""
echo "=== All VMs setup complete! ==="

# Verify
echo ""
echo "=== Verification ==="
for entry in "${NEW_VMS[@]}"; do
    VM="${entry%%:*}"
    ZONE="${entry##*:}"
    RESULT=$(gcloud compute ssh "$VM" --zone="$ZONE" --command='
        ls /home/Owner/rust_benchmark/target/release/ofc_benchmark 2>/dev/null && echo "BINARY_OK" || echo "BINARY_MISSING"
        ls /home/Owner/rust_benchmark/models/bc.onnx 2>/dev/null && echo "MODEL_OK" || echo "MODEL_MISSING"
    ' 2>/dev/null) || RESULT="OFFLINE"
    echo "  $VM: $RESULT"
done
