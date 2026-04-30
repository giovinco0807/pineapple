#!/bin/bash
# Deploy and start Expectimax on a fleet of VMs
# Usage: bash deploy_fleet.sh

set -e

TAR_FILE="/tmp/expectimax_src.tar.gz"
PATTERNS_TOTAL=500
ZONE_MAP="expectimax-solver:asia-northeast1-b
expectimax-solver-2:asia-northeast1-b
expectimax-s3:asia-northeast2-a
expectimax-s4:asia-northeast2-a
expectimax-s5:asia-northeast1-b
expectimax-s6:asia-northeast1-b
expectimax-s7:asia-northeast2-a
expectimax-s8:asia-northeast1-b
expectimax-s9:us-central1-a
expectimax-s10:us-central1-a
expectimax-s11:us-central1-a
expectimax-s12:asia-northeast1-b
expectimax-s13:europe-west1-b
expectimax-s14:europe-west1-b
expectimax-s15:asia-northeast1-b
expectimax-s16:europe-west1-b"

NUM_VMS=$(echo "$ZONE_MAP" | wc -l)
CHUNK=$((PATTERNS_TOTAL / NUM_VMS))
echo "=== Deploying to $NUM_VMS VMs, $CHUNK patterns each ==="

IDX=0
while IFS=: read -r VM ZONE; do
    START=$((IDX * CHUNK + 1))
    END=$(((IDX + 1) * CHUNK))
    if [ $IDX -eq $((NUM_VMS - 1)) ]; then
        END=$PATTERNS_TOTAL
    fi
    IDX=$((IDX + 1))

    echo "[$IDX/$NUM_VMS] $VM ($ZONE) patterns $START-$END"

    # Skip VM1 and VM2 (already set up and running)
    if [ "$VM" = "expectimax-solver" ] || [ "$VM" = "expectimax-solver-2" ]; then
        echo "  Reconfiguring existing VM..."
        gcloud compute ssh "$VM" --zone="$ZONE" --command "
            cd /home/\$USER/expectimax
            # Kill existing process
            pkill -f 'backward stdin' 2>/dev/null || true
            sleep 2
            # Extract pattern chunk
            sed -n '${START},${END}p' canonical_patterns.txt > my_patterns.txt
            sed -i 's|PATTERNS_FILE=.*|PATTERNS_FILE=\"\$WORK_DIR/my_patterns.txt\"|' gcp_setup.sh
            sed -i 's|MAX_PATTERNS=.*|MAX_PATTERNS=999|' gcp_setup.sh
            mkdir -p results
            nohup bash gcp_setup.sh run > run.log 2>&1 &
            echo \"Started \$(wc -l < my_patterns.txt) patterns\"
        " &
        continue
    fi

    # New VMs: upload, setup, run
    (
        # Upload source
        gcloud compute scp "$TAR_FILE" "$VM:/tmp/expectimax_src.tar.gz" --zone="$ZONE" 2>/dev/null

        # Setup and run
        gcloud compute ssh "$VM" --zone="$ZONE" --command "
            cd /home/\$USER
            tar xzf /tmp/expectimax_src.tar.gz
            cd expectimax

            # Install Rust + build
            if ! command -v cargo &>/dev/null; then
                curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y 2>/dev/null
                source \"\$HOME/.cargo/env\"
            fi
            source \"\$HOME/.cargo/env\" 2>/dev/null || true
            cd rust_solver && cargo build --release -p backward 2>/dev/null && cd ..

            # Extract pattern chunk
            sed -n '${START},${END}p' canonical_patterns.txt > my_patterns.txt
            sed -i 's|PATTERNS_FILE=.*|PATTERNS_FILE=\"\$WORK_DIR/my_patterns.txt\"|' gcp_setup.sh
            sed -i 's|MAX_PATTERNS=.*|MAX_PATTERNS=999|' gcp_setup.sh
            mkdir -p results
            nohup bash gcp_setup.sh run > run.log 2>&1 &
            echo \"Started \$(wc -l < my_patterns.txt) patterns\"
        " 2>/dev/null
        echo "  $VM done"
    ) &

done <<< "$ZONE_MAP"

echo "Waiting for all deployments..."
wait
echo "=== All VMs deployed ==="
