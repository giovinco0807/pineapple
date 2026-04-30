#!/bin/bash
# Deploy Expectimax to GCP Spot VM
#
# Usage:
#   bash gcp_deploy.sh create   # Create Spot VM
#   bash gcp_deploy.sh upload   # Upload code to VM
#   bash gcp_deploy.sh ssh      # SSH into VM
#   bash gcp_deploy.sh start    # Start expectimax run
#   bash gcp_deploy.sh download # Download results
#   bash gcp_deploy.sh delete   # Delete VM

set -e

PROJECT="ofc-solver-485418"
ZONE="us-central1-a"
INSTANCE="expectimax-solver"
MACHINE_TYPE="c3-highcpu-176"   # 176 vCPU, 352GB RAM, Spot
LOCAL_RUST_DIR="$(cd "$(dirname "$0")/.." && pwd)"  # ai/rust_solver
LOCAL_PATTERNS="$LOCAL_RUST_DIR/canonical_patterns.txt"

create_vm() {
    echo "=== Creating Spot VM: $INSTANCE ($MACHINE_TYPE) ==="
    gcloud compute instances create "$INSTANCE" \
        --project="$PROJECT" \
        --zone="$ZONE" \
        --machine-type="$MACHINE_TYPE" \
        --provisioning-model=SPOT \
        --instance-termination-action=STOP \
        --boot-disk-size=50GB \
        --boot-disk-type=pd-ssd \
        --image-family=debian-12 \
        --image-project=debian-cloud \
        --metadata=startup-script='#!/bin/bash
apt-get update -qq
apt-get install -y -qq build-essential pkg-config libssl-dev python3
'
    echo "=== VM created. Wait ~30s for startup script ==="
    sleep 30
}

upload_code() {
    echo "=== Uploading Rust source ==="
    # Create directory structure on VM
    gcloud compute ssh "$INSTANCE" --zone="$ZONE" --command="mkdir -p ~/expectimax/rust_solver"

    # Upload workspace files
    gcloud compute scp "$LOCAL_RUST_DIR/Cargo.toml" "$INSTANCE:~/expectimax/rust_solver/Cargo.toml" --zone="$ZONE"

    # Upload crates
    for crate in ofc_core backward; do
        gcloud compute ssh "$INSTANCE" --zone="$ZONE" --command="mkdir -p ~/expectimax/rust_solver/$crate/src"
        gcloud compute scp --recurse "$LOCAL_RUST_DIR/$crate/src/" "$INSTANCE:~/expectimax/rust_solver/$crate/src/" --zone="$ZONE"
        gcloud compute scp "$LOCAL_RUST_DIR/$crate/Cargo.toml" "$INSTANCE:~/expectimax/rust_solver/$crate/Cargo.toml" --zone="$ZONE"
    done

    # Upload fl_solver Cargo.toml (workspace member, needed even if not built)
    gcloud compute ssh "$INSTANCE" --zone="$ZONE" --command="mkdir -p ~/expectimax/rust_solver/fl_solver/src"
    gcloud compute scp "$LOCAL_RUST_DIR/fl_solver/Cargo.toml" "$INSTANCE:~/expectimax/rust_solver/fl_solver/Cargo.toml" --zone="$ZONE"
    # Create dummy lib.rs for fl_solver
    gcloud compute ssh "$INSTANCE" --zone="$ZONE" --command="echo '' > ~/expectimax/rust_solver/fl_solver/src/lib.rs"

    # Upload patterns and setup script
    gcloud compute scp "$LOCAL_PATTERNS" "$INSTANCE:~/expectimax/canonical_patterns.txt" --zone="$ZONE"
    gcloud compute scp "$LOCAL_RUST_DIR/backward/gcp_setup.sh" "$INSTANCE:~/expectimax/gcp_setup.sh" --zone="$ZONE"
    gcloud compute scp "$LOCAL_RUST_DIR/backward/gcp_t2_deep.sh" "$INSTANCE:~/expectimax/gcp_t2_deep.sh" --zone="$ZONE"

    echo "=== Upload complete ==="
}

setup_and_build() {
    echo "=== Setting up and building on VM ==="
    gcloud compute ssh "$INSTANCE" --zone="$ZONE" --command="cd ~/expectimax && chmod +x gcp_setup.sh && ./gcp_setup.sh setup"
}

smoke_test() {
    echo "=== Running smoke test ==="
    gcloud compute ssh "$INSTANCE" --zone="$ZONE" --command="cd ~/expectimax && ./gcp_setup.sh test"
}

start_run() {
    echo "=== Starting expectimax run in background ==="
    gcloud compute ssh "$INSTANCE" --zone="$ZONE" --command="cd ~/expectimax && nohup ./gcp_setup.sh run > run.log 2>&1 &"
    echo "=== Run started. Check progress with: ==="
    echo "  bash gcp_deploy.sh ssh"
    echo "  tail -f ~/expectimax/run.log"
}

download_results() {
    echo "=== Downloading results ==="
    mkdir -p results_gcp
    gcloud compute scp --recurse "$INSTANCE:~/expectimax/results/" ./results_gcp/ --zone="$ZONE"
    echo "=== Downloaded to ./results_gcp/ ==="
    ls results_gcp/ | wc -l
    echo "files downloaded"
}

ssh_vm() {
    gcloud compute ssh "$INSTANCE" --zone="$ZONE"
}

delete_vm() {
    echo "=== Deleting VM: $INSTANCE ==="
    gcloud compute instances delete "$INSTANCE" --zone="$ZONE" --quiet
}

status_vm() {
    gcloud compute instances describe "$INSTANCE" --zone="$ZONE" --format="value(status)" 2>/dev/null || echo "NOT FOUND"
}

case "${1:-}" in
    create)   create_vm ;;
    upload)   upload_code ;;
    build)    setup_and_build ;;
    test)     smoke_test ;;
    ssh)      ssh_vm ;;
    start)    start_run ;;
    download) download_results ;;
    delete)   delete_vm ;;
    status)   status_vm ;;
    *)
        echo "Usage: $0 {create|upload|build|test|ssh|start|download|delete|status}"
        echo ""
        echo "Typical workflow:"
        echo "  $0 create    # Create Spot VM"
        echo "  $0 upload    # Upload Rust source + patterns"
        echo "  $0 build     # Install Rust & compile on VM"
        echo "  $0 test      # Smoke test (1 pattern)"
        echo "  $0 start     # Start full run (background)"
        echo "  $0 ssh       # SSH to check progress"
        echo "  $0 download  # Download results"
        echo "  $0 delete    # Delete VM when done"
        exit 1
        ;;
esac
