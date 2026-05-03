#!/bin/bash
# Deploy T2 Deep Eval to 3 GCP Spot VMs
#
# Usage:
#   bash gcp_t2_fleet.sh create     # Create 3 Spot VMs
#   bash gcp_t2_fleet.sh upload     # Upload code to all VMs
#   bash gcp_t2_fleet.sh build      # Build on all VMs
#   bash gcp_t2_fleet.sh test       # Smoke test on VM-1
#   bash gcp_t2_fleet.sh start      # Start runs (each VM gets 1/3 of patterns)
#   bash gcp_t2_fleet.sh status     # Check all VM statuses
#   bash gcp_t2_fleet.sh download   # Download results from all VMs
#   bash gcp_t2_fleet.sh delete     # Delete all VMs

set -e

PROJECT="ofc-solver-485418"
ZONE="us-central1-a"
MACHINE_TYPE="c3-highcpu-176"
NUM_VMS=3
TOTAL_PATTERNS=152646

LOCAL_RUST_DIR="$(cd "$(dirname "$0")/.." && pwd)"  # ai/rust_solver
LOCAL_PATTERNS="$LOCAL_RUST_DIR/canonical_patterns.txt"

vm_name() { echo "t2deep-vm${1}"; }

create_all() {
    for i in $(seq 1 $NUM_VMS); do
        NAME=$(vm_name $i)
        echo "=== Creating $NAME ($MACHINE_TYPE Spot) ==="
        gcloud compute instances create "$NAME" \
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
    done
    echo "=== All $NUM_VMS VMs created. Wait ~30s for startup scripts ==="
    sleep 30
}

upload_one() {
    local IDX=$1
    local NAME=$(vm_name $IDX)
    echo "=== Uploading to $NAME ==="

    gcloud compute ssh "$NAME" --zone="$ZONE" --command="mkdir -p ~/expectimax/rust_solver"
    gcloud compute scp "$LOCAL_RUST_DIR/Cargo.toml" "$NAME:~/expectimax/rust_solver/Cargo.toml" --zone="$ZONE"

    for crate in ofc_core backward; do
        gcloud compute ssh "$NAME" --zone="$ZONE" --command="mkdir -p ~/expectimax/rust_solver/$crate/src"
        gcloud compute scp --recurse "$LOCAL_RUST_DIR/$crate/src/" "$NAME:~/expectimax/rust_solver/$crate/src/" --zone="$ZONE"
        gcloud compute scp "$LOCAL_RUST_DIR/$crate/Cargo.toml" "$NAME:~/expectimax/rust_solver/$crate/Cargo.toml" --zone="$ZONE"
    done

    # fl_solver stub (workspace member)
    gcloud compute ssh "$NAME" --zone="$ZONE" --command="mkdir -p ~/expectimax/rust_solver/fl_solver/src"
    gcloud compute scp "$LOCAL_RUST_DIR/fl_solver/Cargo.toml" "$NAME:~/expectimax/rust_solver/fl_solver/Cargo.toml" --zone="$ZONE"
    gcloud compute ssh "$NAME" --zone="$ZONE" --command="echo '' > ~/expectimax/rust_solver/fl_solver/src/lib.rs"

    gcloud compute scp "$LOCAL_PATTERNS" "$NAME:~/expectimax/canonical_patterns.txt" --zone="$ZONE"
    gcloud compute scp "$LOCAL_RUST_DIR/backward/gcp_setup.sh" "$NAME:~/expectimax/gcp_setup.sh" --zone="$ZONE"
    gcloud compute scp "$LOCAL_RUST_DIR/backward/gcp_t2_deep.sh" "$NAME:~/expectimax/gcp_t2_deep.sh" --zone="$ZONE"
}

upload_all() {
    for i in $(seq 1 $NUM_VMS); do
        upload_one $i &
    done
    wait
    echo "=== Upload complete to all $NUM_VMS VMs ==="
}

build_all() {
    for i in $(seq 1 $NUM_VMS); do
        NAME=$(vm_name $i)
        echo "=== Building on $NAME ==="
        gcloud compute ssh "$NAME" --zone="$ZONE" --command="cd ~/expectimax && chmod +x gcp_setup.sh gcp_t2_deep.sh && ./gcp_t2_deep.sh setup" &
    done
    wait
    echo "=== Build complete on all VMs ==="
}

test_run() {
    NAME=$(vm_name 1)
    echo "=== Smoke test on $NAME ==="
    gcloud compute ssh "$NAME" --zone="$ZONE" --command="cd ~/expectimax && ./gcp_t2_deep.sh test"
}

start_all() {
    # Split patterns evenly across VMs
    PER_VM=$(( (TOTAL_PATTERNS + NUM_VMS - 1) / NUM_VMS ))

    for i in $(seq 1 $NUM_VMS); do
        NAME=$(vm_name $i)
        START=$(( (i - 1) * PER_VM + 1 ))
        END=$(( i * PER_VM ))
        if [ $END -gt $TOTAL_PATTERNS ]; then END=$TOTAL_PATTERNS; fi

        echo "=== Starting $NAME: patterns $START-$END ==="
        gcloud compute ssh "$NAME" --zone="$ZONE" --command="cd ~/expectimax && nohup ./gcp_t2_deep.sh run $START $END > run.log 2>&1 &"
    done
    echo ""
    echo "=== All $NUM_VMS VMs started ==="
    echo "  VM1: patterns 1-$PER_VM"
    echo "  VM2: patterns $((PER_VM+1))-$((PER_VM*2))"
    echo "  VM3: patterns $((PER_VM*2+1))-$TOTAL_PATTERNS"
    echo ""
    echo "Monitor: bash gcp_t2_fleet.sh status"
    echo "SSH:     gcloud compute ssh t2deep-vm1 --zone=$ZONE"
}

status_all() {
    for i in $(seq 1 $NUM_VMS); do
        NAME=$(vm_name $i)
        STATUS=$(gcloud compute instances describe "$NAME" --zone="$ZONE" --format="value(status)" 2>/dev/null || echo "NOT FOUND")
        echo "=== $NAME: $STATUS ==="
        if [ "$STATUS" = "RUNNING" ]; then
            # Show last 3 lines from progress log (has timing/ETA)
            gcloud compute ssh "$NAME" --zone="$ZONE" --command="tail -3 ~/expectimax/t2_results/progress_*.log 2>/dev/null || tail -1 ~/expectimax/run.log 2>/dev/null || echo '(no log yet)'" 2>/dev/null || true
        fi
        echo ""
    done
}

download_all() {
    mkdir -p t2_results_gcp
    for i in $(seq 1 $NUM_VMS); do
        NAME=$(vm_name $i)
        echo "=== Downloading from $NAME ==="
        gcloud compute scp --recurse "$NAME:~/expectimax/t2_results/" ./t2_results_gcp/ --zone="$ZONE" 2>/dev/null || echo "  (no results yet)"
    done
    echo "=== Downloaded to ./t2_results_gcp/ ==="
    ls t2_results_gcp/*.jsonl 2>/dev/null | wc -l
    echo "result files"
}

delete_all() {
    for i in $(seq 1 $NUM_VMS); do
        NAME=$(vm_name $i)
        echo "=== Deleting $NAME ==="
        gcloud compute instances delete "$NAME" --zone="$ZONE" --quiet 2>/dev/null || true
    done
}

ssh_vm() {
    local IDX=${1:-1}
    gcloud compute ssh "$(vm_name $IDX)" --zone="$ZONE"
}

case "${1:-}" in
    create)   create_all ;;
    upload)   upload_all ;;
    build)    build_all ;;
    test)     test_run ;;
    start)    start_all ;;
    status)   status_all ;;
    download) download_all ;;
    delete)   delete_all ;;
    ssh)      ssh_vm "${2:-1}" ;;
    *)
        echo "Usage: $0 {create|upload|build|test|start|status|download|delete|ssh [N]}"
        echo ""
        echo "  create    Create $NUM_VMS Spot VMs"
        echo "  upload    Upload Rust source + patterns to all VMs"
        echo "  build     Build on all VMs (parallel)"
        echo "  test      Smoke test on VM-1"
        echo "  start     Start t2eval on all VMs (pattern range split)"
        echo "  status    Check VM status + last log line"
        echo "  download  Download results from all VMs"
        echo "  delete    Delete all VMs"
        echo "  ssh [N]   SSH to VM N (default: 1)"
        exit 1
        ;;
esac
