#!/bin/bash
# Iter2 deployment: 15 VMs × 1,008 games = 15,120 games
# Usage:
#   bash iter2_deploy.sh create     # Create 10 new VMs
#   bash iter2_deploy.sh setup      # Setup new VMs (Rust, build, models)
#   bash iter2_deploy.sh run        # Launch selfplay on ALL 15 VMs
#   bash iter2_deploy.sh status     # Check progress
#   bash iter2_deploy.sh download   # Download results
#   bash iter2_deploy.sh cleanup    # Stop all VMs
#   bash iter2_deploy.sh delete     # Delete all VMs

set -e

PROJECT="ofc-solver-485418"
MACHINE_TYPE="n2-highcpu-32"
IMAGE_FAMILY="ubuntu-2404-lts-amd64"
IMAGE_PROJECT="ubuntu-os-cloud"

# All 15 VMs: 5 existing + 10 new
ALL_VMS=(
    "ofc-selfplay-0:us-central1-a"
    "ofc-selfplay-1:us-central1-b"
    "ofc-selfplay-2:us-east1-b"
    "ofc-selfplay-3:us-west1-b"
    "ofc-selfplay-4:europe-west1-b"
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

NEW_VMS=("${ALL_VMS[@]:5}")  # VMs 5-14 (new)

NUM_PROCS=12
GAMES_PER_PROC=84
GAMES_PER_VM=$((NUM_PROCS * GAMES_PER_PROC))  # 1008

# Iter2 seeds: 70000+ (iter1 used 50000-62999)
ITER2_BASE=70000

LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

cmd=${1:-help}

get_vm_name() { echo "${1%%:*}"; }
get_vm_zone() { echo "${1##*:}"; }

case "$cmd" in

create)
    echo "=== Creating 10 new VMs ==="
    for entry in "${NEW_VMS[@]}"; do
        VM=$(get_vm_name "$entry")
        ZONE=$(get_vm_zone "$entry")
        echo "  Creating $VM in $ZONE..."
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
    echo "=== All 10 VMs created. Wait 30s for SSH... ==="
    sleep 30
    echo "=== Run: bash iter2_deploy.sh setup ==="
    ;;

setup)
    echo "=== Setting up new VMs (Rust + build + models) ==="
    FAIL_COUNT=0

    for entry in "${NEW_VMS[@]}"; do
        VM=$(get_vm_name "$entry")
        ZONE=$(get_vm_zone "$entry")
        echo ""
        echo "--- Setting up $VM ($ZONE) ---"

        # Wait for SSH
        SSH_OK=0
        for retry in $(seq 1 10); do
            if gcloud compute ssh "$VM" --zone="$ZONE" --command="echo 'SSH ready'" 2>/dev/null; then
                SSH_OK=1
                break
            fi
            echo "  Retry $retry..."
            sleep 10
        done

        if [ "$SSH_OK" -eq 0 ]; then
            echo "  FAILED: Cannot SSH to $VM"
            FAIL_COUNT=$((FAIL_COUNT + 1))
            continue
        fi

        # Install Rust + dependencies
        gcloud compute ssh "$VM" --zone="$ZONE" --command='
            curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
            export PATH="$HOME/.cargo/bin:$PATH"
            sudo apt-get update -qq
            sudo apt-get install -y -qq build-essential pkg-config libssl-dev g++
        ' 2>&1 | tail -3

        # Upload source
        gcloud compute scp /tmp/ofc_selfplay.tar.gz "${VM}:/home/Owner/" --zone="$ZONE"

        # Build
        gcloud compute ssh "$VM" --zone="$ZONE" --command='
            export PATH="$HOME/.cargo/bin:$PATH"
            cd /home/Owner
            tar xzf ofc_selfplay.tar.gz
            cd rust_benchmark
            cargo build --release 2>&1 | tail -5
            echo "Build complete!"
            ls -lh target/release/ofc_benchmark
        '

        # Upload models
        gcloud compute scp /tmp/ofc_models.tar.gz "${VM}:/home/Owner/" --zone="$ZONE"
        gcloud compute ssh "$VM" --zone="$ZONE" --command='
            cd /home/Owner/rust_benchmark
            tar xzf /home/Owner/ofc_models.tar.gz
            ls -lh models/
        '

        # Upload remote_run.sh
        gcloud compute scp "$LOCAL_DIR/remote_run.sh" "${VM}:/home/Owner/rust_benchmark/" --zone="$ZONE"

        echo "  $VM ready!"
    done

    echo ""
    if [ "$FAIL_COUNT" -gt 0 ]; then
        echo "=== WARNING: $FAIL_COUNT VMs failed setup ==="
    fi
    echo "=== Setup complete. Run: bash iter2_deploy.sh run ==="
    ;;

run)
    echo "=== Starting iter2 selfplay: 15 VMs × ${GAMES_PER_VM} games = $((15 * GAMES_PER_VM)) games ==="
    echo ""

    for i in "${!ALL_VMS[@]}"; do
        entry="${ALL_VMS[$i]}"
        VM=$(get_vm_name "$entry")
        ZONE=$(get_vm_zone "$entry")
        SEED=$((ITER2_BASE + i * GAMES_PER_VM))

        echo "--- $VM (seed=$SEED) ---"

        # Clean old data and launch
        gcloud compute ssh "$VM" --zone="$ZONE" --command="
            pkill -f ofc_benchmark 2>/dev/null || true
            sleep 1
            rm -f /tmp/selfplay_*.jsonl /tmp/selfplay_*.log
            cd /home/Owner/rust_benchmark
            bash remote_run.sh $SEED $GAMES_PER_PROC $NUM_PROCS
        " 2>/dev/null &
    done
    wait

    echo ""
    echo "=== All launched. Monitor: bash iter2_deploy.sh status ==="
    ;;

status)
    echo "=== Iter2 Selfplay Status ==="
    GRAND_TOTAL=0
    RUNNING_VMS=0
    DONE_VMS=0

    for i in "${!ALL_VMS[@]}"; do
        entry="${ALL_VMS[$i]}"
        VM=$(get_vm_name "$entry")
        ZONE=$(get_vm_zone "$entry")

        RESULT=$(gcloud compute ssh "$VM" --zone="$ZONE" --command='
            RUNNING=$(pgrep -f ofc_benchmark | wc -l)
            TOTAL=0
            for f in /tmp/selfplay_*.jsonl; do
                [ -f "$f" ] || continue
                N=$(wc -l < "$f")
                TOTAL=$((TOTAL + N))
            done
            echo "$RUNNING $TOTAL"
        ' 2>/dev/null) || { echo "  $VM: OFFLINE"; continue; }

        PROCS=$(echo "$RESULT" | awk '{print $1}')
        GAMES=$(echo "$RESULT" | awk '{print $2}')
        GRAND_TOTAL=$((GRAND_TOTAL + GAMES))

        if [ "$PROCS" -gt 0 ]; then
            RUNNING_VMS=$((RUNNING_VMS + 1))
            echo "  $VM: ${PROCS}/${NUM_PROCS} procs, ${GAMES}/${GAMES_PER_VM} games"
        else
            DONE_VMS=$((DONE_VMS + 1))
            echo "  $VM: DONE (${GAMES} games)"
        fi
    done

    TOTAL_TARGET=$((15 * GAMES_PER_VM))
    echo ""
    echo "=== Grand Total: $GRAND_TOTAL / $TOTAL_TARGET ($RUNNING_VMS running, $DONE_VMS done) ==="
    ;;

download)
    echo "=== Downloading iter2 results ==="
    RESULTS_DIR="$LOCAL_DIR/results_iter2"
    mkdir -p "$RESULTS_DIR"

    for i in "${!ALL_VMS[@]}"; do
        entry="${ALL_VMS[$i]}"
        VM=$(get_vm_name "$entry")
        ZONE=$(get_vm_zone "$entry")
        echo ""
        echo "--- $VM ---"

        # Merge on remote
        gcloud compute ssh "$VM" --zone="$ZONE" --command="
            cat /tmp/selfplay_*.jsonl > /tmp/selfplay_merged_vm${i}.jsonl
            wc -l /tmp/selfplay_merged_vm${i}.jsonl
        " 2>/dev/null || { echo "  OFFLINE, skipping"; continue; }

        gcloud compute scp "${VM}:/tmp/selfplay_merged_vm${i}.jsonl" \
            "$RESULTS_DIR/vm${i}.jsonl" --zone="$ZONE" || echo "  SCP failed"
    done

    # Merge all VM results
    cd "$RESULTS_DIR"
    cat vm*.jsonl > all_selfplay_iter2.jsonl 2>/dev/null
    TOTAL=$(wc -l < all_selfplay_iter2.jsonl)
    echo ""
    echo "=== Total: $TOTAL games ==="
    ls -lh "$RESULTS_DIR/"
    ;;

cleanup)
    echo "=== Stopping all 15 VMs ==="
    for entry in "${ALL_VMS[@]}"; do
        VM=$(get_vm_name "$entry")
        ZONE=$(get_vm_zone "$entry")
        gcloud compute instances stop "$VM" --zone="$ZONE" --quiet 2>/dev/null &
    done
    wait
    echo "All VMs stopped."
    ;;

delete)
    echo "=== Deleting all 15 VMs ==="
    for entry in "${ALL_VMS[@]}"; do
        VM=$(get_vm_name "$entry")
        ZONE=$(get_vm_zone "$entry")
        gcloud compute instances delete "$VM" --zone="$ZONE" --quiet 2>/dev/null &
    done
    wait
    echo "All VMs deleted."
    ;;

*)
    echo "Usage: bash iter2_deploy.sh {create|setup|run|status|download|cleanup|delete}"
    echo ""
    echo "  create   - Create 10 new Spot VMs"
    echo "  setup    - Install Rust, build, upload models on new VMs"
    echo "  run      - Start selfplay on ALL 15 VMs (15,120 games)"
    echo "  status   - Check progress across all VMs"
    echo "  download - Download and merge JSONL results"
    echo "  cleanup  - Stop all VMs"
    echo "  delete   - Delete all VMs"
    ;;

esac
