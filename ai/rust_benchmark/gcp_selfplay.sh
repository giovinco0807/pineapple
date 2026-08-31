#!/bin/bash
# GCP Spot VM selfplay - 5 VMs × 1000 games = 5000 total
# Usage:
#   bash gcp_selfplay.sh create    # Create 5 VMs + build
#   bash gcp_selfplay.sh run       # Start selfplay on all VMs
#   bash gcp_selfplay.sh status    # Check progress
#   bash gcp_selfplay.sh download  # Download results
#   bash gcp_selfplay.sh cleanup   # Stop all VMs
#   bash gcp_selfplay.sh delete    # Delete all VMs

set -e

PROJECT="ofc-solver-485418"
ZONES=("us-central1-a" "us-central1-b" "us-east1-b" "us-west1-b" "europe-west1-b")
MACHINE_TYPE="n2-highcpu-32"
IMAGE_FAMILY="ubuntu-2404-lts-amd64"
IMAGE_PROJECT="ubuntu-os-cloud"
NUM_VMS=5
VM_PREFIX="ofc-selfplay"

GAMES_PER_VM=1008          # 84 × 12 procs
NUM_PROCS=12
GAMES_PER_PROC=84
BASE_SEEDS=(50000 53000 56000 59000 62000)

LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"
REMOTE_DIR="/home/Owner"

cmd=${1:-help}

vm_name() { echo "${VM_PREFIX}-${1}"; }
vm_zone() { echo "${ZONES[$1]}"; }

case "$cmd" in

create)
    echo "=== Packaging source ==="
    cd "$LOCAL_DIR/.."
    tar czf /tmp/ofc_selfplay.tar.gz \
        rust_benchmark/src \
        rust_benchmark/Cargo.toml \
        rust_benchmark/Cargo.lock \
        rust_benchmark/models \
        rust_solver/Cargo.toml \
        rust_solver/ofc_core
    echo "  Package: $(du -h /tmp/ofc_selfplay.tar.gz | cut -f1)"

    for i in $(seq 0 $((NUM_VMS - 1))); do
        VM=$(vm_name $i)
        ZONE=$(vm_zone $i)
        echo ""
        echo "=== Creating VM $i: $VM in $ZONE ==="

        gcloud compute instances create "$VM" \
            --project="$PROJECT" \
            --zone="$ZONE" \
            --machine-type="$MACHINE_TYPE" \
            --provisioning-model=SPOT \
            --instance-termination-action=STOP \
            --image-family="$IMAGE_FAMILY" \
            --image-project="$IMAGE_PROJECT" \
            --boot-disk-size=30GB \
            --boot-disk-type=pd-ssd \
            --no-restart-on-failure \
            2>/dev/null &
    done
    wait
    echo ""
    echo "=== Waiting for SSH (30s) ==="
    sleep 30

    for i in $(seq 0 $((NUM_VMS - 1))); do
        VM=$(vm_name $i)
        ZONE=$(vm_zone $i)
        echo ""
        echo "=== Setting up $VM ==="

        # Wait for SSH
        for retry in $(seq 1 10); do
            gcloud compute ssh "$VM" --zone="$ZONE" --command="echo 'SSH ready'" 2>/dev/null && break
            echo "  Retry $retry..."
            sleep 10
        done

        # Install Rust + upload + build
        gcloud compute ssh "$VM" --zone="$ZONE" --command="
            curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
            source \"\$HOME/.cargo/env\"
            sudo apt-get update -qq
            sudo apt-get install -y -qq build-essential pkg-config
        " 2>&1 | tail -3

        gcloud compute scp /tmp/ofc_selfplay.tar.gz "${VM}:${REMOTE_DIR}/" --zone="$ZONE"

        gcloud compute ssh "$VM" --zone="$ZONE" --command="
            source \"\$HOME/.cargo/env\"
            cd ${REMOTE_DIR}
            tar xzf ofc_selfplay.tar.gz
            cd rust_benchmark
            cargo build --release 2>&1 | tail -5
            echo 'Build complete!'
            ls -lh target/release/ofc_benchmark
        "
        echo "  $VM ready!"
    done

    echo ""
    echo "=== All $NUM_VMS VMs ready. Run: bash gcp_selfplay.sh run ==="
    ;;

run)
    echo "=== Starting selfplay: ${NUM_VMS} VMs × ${GAMES_PER_VM} games ==="

    for i in $(seq 0 $((NUM_VMS - 1))); do
        VM=$(vm_name $i)
        ZONE=$(vm_zone $i)
        BSEED=${BASE_SEEDS[$i]}
        echo ""
        echo "--- $VM (seed base=$BSEED) ---"

        gcloud compute ssh "$VM" --zone="$ZONE" --command="
            # Kill any existing processes
            pkill -f ofc_benchmark 2>/dev/null || true
            sleep 1
            mkdir -p /tmp/selfplay_results

            for j in \$(seq 0 $((NUM_PROCS - 1))); do
                SEED=\$((${BSEED} + j * ${GAMES_PER_PROC}))
                echo \"  Proc \$j: seed=\$SEED games=${GAMES_PER_PROC}\"
                cd ${REMOTE_DIR}/rust_benchmark
                nohup ./target/release/ofc_benchmark \
                    --games ${GAMES_PER_PROC} \
                    --seed \$SEED \
                    --mode selfplay \
                    --mcts-sims 400 \
                    --rollouts 250 \
                    --output /tmp/selfplay_\${j}.jsonl \
                    > /tmp/selfplay_\${j}.log 2>&1 &
            done

            echo ''
            echo \"PIDs: \$(pgrep -f ofc_benchmark | tr '\n' ' ')\"
            echo \"Total procs: \$(pgrep -f ofc_benchmark | wc -l)\"
        "
    done

    echo ""
    echo "=== All launched. Monitor: bash gcp_selfplay.sh status ==="
    ;;

status)
    echo "=== Selfplay Status ==="
    GRAND_TOTAL=0
    for i in $(seq 0 $((NUM_VMS - 1))); do
        VM=$(vm_name $i)
        ZONE=$(vm_zone $i)
        echo ""
        echo "--- $VM ($ZONE) ---"
        RESULT=$(gcloud compute ssh "$VM" --zone="$ZONE" --command="
            RUNNING=\$(pgrep -f ofc_benchmark | wc -l)
            TOTAL=0
            for f in /tmp/selfplay_*.jsonl; do
                [ -f \"\$f\" ] || continue
                N=\$(wc -l < \"\$f\")
                TOTAL=\$((TOTAL + N))
            done
            echo \"  Running: \$RUNNING/${NUM_PROCS}  Games: \$TOTAL/${GAMES_PER_VM}\"
            echo \$TOTAL
        " 2>/dev/null) || { echo "  OFFLINE"; continue; }
        echo "$RESULT" | head -1
        COUNT=$(echo "$RESULT" | tail -1)
        GRAND_TOTAL=$((GRAND_TOTAL + COUNT))
    done
    echo ""
    TOTAL_TARGET=$((NUM_VMS * GAMES_PER_VM))
    echo "=== Grand Total: $GRAND_TOTAL / $TOTAL_TARGET ==="
    ;;

download)
    echo "=== Downloading results ==="
    mkdir -p "$LOCAL_DIR/results"

    for i in $(seq 0 $((NUM_VMS - 1))); do
        VM=$(vm_name $i)
        ZONE=$(vm_zone $i)
        echo ""
        echo "--- $VM ---"

        # Merge on remote
        gcloud compute ssh "$VM" --zone="$ZONE" --command="
            cat /tmp/selfplay_*.jsonl > /tmp/selfplay_merged_vm${i}.jsonl
            wc -l /tmp/selfplay_merged_vm${i}.jsonl
        " 2>/dev/null || { echo "  OFFLINE, skipping"; continue; }

        gcloud compute scp "${VM}:/tmp/selfplay_merged_vm${i}.jsonl" \
            "$LOCAL_DIR/results/vm${i}.jsonl" --zone="$ZONE" || echo "  SCP failed"
    done

    # Merge all VM results
    cd "$LOCAL_DIR/results"
    cat vm*.jsonl > all_selfplay.jsonl 2>/dev/null
    echo ""
    echo "=== Total: $(wc -l < all_selfplay.jsonl) games ==="
    ls -lh "$LOCAL_DIR/results/"
    ;;

cleanup)
    echo "=== Stopping all VMs ==="
    for i in $(seq 0 $((NUM_VMS - 1))); do
        VM=$(vm_name $i)
        ZONE=$(vm_zone $i)
        gcloud compute instances stop "$VM" --zone="$ZONE" --quiet 2>/dev/null &
    done
    wait
    echo "All VMs stopped."
    ;;

delete)
    echo "=== Deleting all VMs ==="
    for i in $(seq 0 $((NUM_VMS - 1))); do
        VM=$(vm_name $i)
        ZONE=$(vm_zone $i)
        gcloud compute instances delete "$VM" --zone="$ZONE" --quiet 2>/dev/null &
    done
    wait
    echo "All VMs deleted."
    ;;

*)
    echo "Usage: bash gcp_selfplay.sh {create|run|status|download|cleanup|delete}"
    echo ""
    echo "  create   - Create ${NUM_VMS} Spot VMs, install Rust, build binary"
    echo "  run      - Start selfplay (${NUM_VMS} × ${GAMES_PER_VM} = $((NUM_VMS * GAMES_PER_VM)) games)"
    echo "  status   - Check progress across all VMs"
    echo "  download - Download and merge JSONL results"
    echo "  cleanup  - Stop all VMs (preserves data)"
    echo "  delete   - Delete all VMs permanently"
    ;;
esac
