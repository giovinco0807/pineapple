#!/bin/bash
# ============================================================================
# Restart data generation on completed VMs (new chunks 3-5, new seeds)
# ============================================================================
# Runs on VMs that have already finished chunks 0-2 and are still RUNNING.
# Uses different seed offset (52000 vs 42000) to avoid duplicates.
#
# Usage:
#   ./gcp_restart_done_vms.sh run       # Start new chunks on completed VMs
#   ./gcp_restart_done_vms.sh status    # Check progress of new chunks
# ============================================================================

set -e
export CLOUDSDK_CORE_ACCOUNT="giovinco.080807@gmail.com"

PROJECT="ofc-solver-485418"
GCS_BASE="gs://ofc-solver-results/t3_rust_fleet/run_20260509"

# VM zone mapping (same as fleet launch)
# vm0-23: us-east1-c
# vm24-47: us-central1-a
# vm48-71: us-east4-a
# vm72-95: us-west1-a
# vm96-121: asia-northeast1-b
# vm122-127: asia-east1-a

get_zone() {
    local idx=$1
    if [ $idx -lt 24 ]; then echo "us-east1-c"
    elif [ $idx -lt 48 ]; then echo "us-central1-a"
    elif [ $idx -lt 72 ]; then echo "us-east4-a"
    elif [ $idx -lt 96 ]; then echo "us-west1-a"
    elif [ $idx -lt 122 ]; then echo "asia-northeast1-b"
    else echo "asia-east1-a"
    fi
}

# Generation config (same as original, new seed offset)
STATES_PER_CHUNK=1000
NEW_CHUNKS=3           # chunks 3,4,5
SEED_OFFSET=52000      # original was 42000
BTN_SAMPLES=100

# Completed + RUNNING VMs (excludes TERMINATED: 24,47,75,76,82,84,86,89,93)
# From DONE list minus TERMINATED
DONE_RUNNING_VMS=(
    8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
    25 26 27 28 29 30 31 34 37 38 39 40 41 46
    48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71
    72 73 78 80 81 83 85 90 95
    96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121
    122 123 124 125 126 127
)

run_restart() {
    TOTAL=${#DONE_RUNNING_VMS[@]}
    echo "=== Restarting data generation on ${TOTAL} completed VMs ==="
    echo "    New chunks: 3-5 (${STATES_PER_CHUNK} states each)"
    echo "    Seed offset: ${SEED_OFFSET}"
    echo "    Expected: ${TOTAL} x 3 x ${STATES_PER_CHUNK} = $((TOTAL * 3 * STATES_PER_CHUNK)) new states"
    echo ""

    BATCH_SIZE=20
    for ((batch_start=0; batch_start < TOTAL; batch_start += BATCH_SIZE)); do
        batch_end=$((batch_start + BATCH_SIZE))
        if [ $batch_end -gt $TOTAL ]; then batch_end=$TOTAL; fi
        echo "--- Batch $((batch_start/BATCH_SIZE + 1)): VMs $batch_start-$((batch_end-1)) ---"

        for ((j=batch_start; j < batch_end; j++)); do
            VM_IDX=${DONE_RUNNING_VMS[$j]}
            ZONE=$(get_zone $VM_IDX)
            INSTANCE="ofc-t3fleet-${VM_IDX}"
            (
                gcloud compute ssh "$INSTANCE" --zone="$ZONE" --project="$PROJECT" --command="
                    cd ~/ofc-pineapple
                    source \"\\\$HOME/.cargo/env\" 2>/dev/null || true
                    RESULT_DIR=\"\\\$HOME/ofc-pineapple/ai/data/t3_fleet/vm${VM_IDX}\"
                    for PART in 3 4 5; do
                        CHUNK_DIR=\"\\\$RESULT_DIR/chunk_\\\${PART}\"
                        mkdir -p \"\\\$CHUNK_DIR\"
                        JSONL=\"\\\$CHUNK_DIR/t3_vm${VM_IDX}_chunk\\\${PART}.jsonl\"
                        SEED=\\\$(( ${SEED_OFFSET} + ${VM_IDX} * 1000 + \\\${PART} ))
                        ./ai/rust_solver/target/release/t3_generator \\
                            --states ${STATES_PER_CHUNK} \\
                            --mode bb \\
                            --btn-samples ${BTN_SAMPLES} \\
                            --seed \\\$SEED \\
                            --onnx-model ai/data/t4_oracle_v2/t4_oracle.onnx \\
                            --output \\\$JSONL \\
                            --log-interval 100 2>&1 | tee \"\\\$CHUNK_DIR/generate.log\"
                        python3 ai/training/convert_t3_rust_jsonl.py \\
                            --input \\\$JSONL \\
                            --output-dir \\\$CHUNK_DIR \\
                            --chunk-size ${STATES_PER_CHUNK} 2>&1 | tee \"\\\$CHUNK_DIR/convert.log\"
                        gsutil -m cp \"\\\$CHUNK_DIR\"/*.npz \"\\\$CHUNK_DIR\"/*.log \\
                            ${GCS_BASE}/vm${VM_IDX}/chunk_\\\${PART}/ 2>/dev/null
                    done
                    echo \"DONE_R2 vm=${VM_IDX} states=\\\$((${STATES_PER_CHUNK} * ${NEW_CHUNKS}))\" | tee \"\\\$RESULT_DIR/DONE_R2\"
                    gsutil cp \"\\\$RESULT_DIR/DONE_R2\" ${GCS_BASE}/vm${VM_IDX}/DONE_R2
                " > /dev/null 2>&1 &
                echo "  ${INSTANCE} (${ZONE}): restart dispatched"
            ) &
        done
        wait
        echo "--- Batch dispatched ---"
    done
    echo ""
    echo "=== All ${TOTAL} VMs restarted ==="
    echo "  Expected additional: $((TOTAL * 3 * STATES_PER_CHUNK)) states"
    echo "  Check progress: $0 status"
}

check_status() {
    echo "=== Checking restart progress ==="
    DONE_R2=$(gsutil ls "${GCS_BASE}/*/DONE_R2" 2>/dev/null | wc -l)
    echo "  Completed round 2: ${DONE_R2} / ${#DONE_RUNNING_VMS[@]} VMs"

    # Sample a few VMs for progress
    for VM_IDX in 10 50 100; do
        ZONE=$(get_zone $VM_IDX)
        INSTANCE="ofc-t3fleet-${VM_IDX}"
        echo ""
        echo "  --- VM${VM_IDX} (${INSTANCE}) ---"
        gcloud compute ssh "$INSTANCE" --zone="$ZONE" --project="$PROJECT" --command="
            for p in 3 4 5; do
                d=~/ofc-pineapple/ai/data/t3_fleet/vm${VM_IDX}/chunk_\${p}
                if [ -d \"\$d\" ]; then
                    npz=\$(ls \$d/*.npz 2>/dev/null | wc -l)
                    lines=\$(wc -l < \$d/t3_vm${VM_IDX}_chunk\${p}.jsonl 2>/dev/null || echo 0)
                    echo \"    chunk_\${p}: \${lines} lines, \${npz} npz\"
                else
                    echo \"    chunk_\${p}: not started\"
                fi
            done
            pgrep -af t3_generator 2>/dev/null | head -1 || echo '    no t3_generator running'
        " 2>/dev/null || echo "  (unreachable)"
    done
}

case "${1:-help}" in
    run)    run_restart ;;
    status) check_status ;;
    *)
        echo "Usage: $0 {run|status}"
        echo ""
        echo "  VMs to restart: ${#DONE_RUNNING_VMS[@]}"
        echo "  New states: $((${#DONE_RUNNING_VMS[@]} * 3 * STATES_PER_CHUNK))"
        ;;
esac
