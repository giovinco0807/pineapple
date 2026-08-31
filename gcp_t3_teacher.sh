#!/bin/bash
# ============================================================================
# GCP T3 Teacher Data Generation (GPU VMs - Backward Induction)
# ============================================================================
# Uses the Rust T3 generator plus T4 Oracle ONNX as leaf evaluator.
# 4 GPU VMs, each generating a portion of the dataset.
#
# Usage:
#   ./gcp_t3_teacher.sh create     # Create 4 GPU VMs
#   ./gcp_t3_teacher.sh upload     # Upload code + T4 model
#   ./gcp_t3_teacher.sh setup      # Install deps (Rust, NumPy)
#   ./gcp_t3_teacher.sh run        # Start generation
#   ./gcp_t3_teacher.sh status     # Check progress
#   ./gcp_t3_teacher.sh download   # Download + merge results
#   ./gcp_t3_teacher.sh delete     # Delete all VMs
#   ./gcp_t3_teacher.sh all        # create + upload + setup + run
# ============================================================================

set -e

# Ensure gcloud uses the correct account
export CLOUDSDK_CORE_ACCOUNT="giovinco.080807@gmail.com"

PROJECT="ofc-solver-485418"
ZONE="us-central1-a"
PREFIX="ofc-t3gen"
NUM_VMS=1
MACHINE_TYPE="n1-standard-4"
GPU_TYPE="nvidia-tesla-t4"
GPU_COUNT=1
IMAGE_FAMILY="pytorch-2-9-cu129-ubuntu-2204-nvidia-580"
IMAGE_PROJECT="deeplearning-platform-release"
DISK_SIZE="50GB"

# Generation config
TOTAL_STATES=100000
BTN_SAMPLES=500
STATES_PER_VM=$((TOTAL_STATES / NUM_VMS))  # 25000
NPZ_CHUNK_SIZE=10000

# GCS bucket for model + results
GCS_BUCKET="gs://ofc-solver-results"
GCS_T4_ONNX="t4_oracle_v2/t4_oracle.onnx"
GCS_RESULTS="t3_teacher_results"

LOCAL_ROOT="$(cd "$(dirname "$0")" && pwd)"

# -- Create VMs -------------------------------------------------------------

create_vms() {
    echo "=== Creating ${NUM_VMS} GPU VMs (${MACHINE_TYPE} + ${GPU_TYPE}) ==="
    for i in $(seq 0 $((NUM_VMS - 1))); do
        local instance="${PREFIX}-${i}"
        echo "  Creating ${instance}..."
        gcloud compute instances create "$instance" \
            --project="$PROJECT" \
            --zone="$ZONE" \
            --machine-type="$MACHINE_TYPE" \
            --image-family="$IMAGE_FAMILY" \
            --image-project="$IMAGE_PROJECT" \
            --boot-disk-size="$DISK_SIZE" \
            --boot-disk-type="pd-balanced" \
            --accelerator="type=${GPU_TYPE},count=${GPU_COUNT}" \
            --maintenance-policy=TERMINATE \
            --provisioning-model=SPOT \
            --instance-termination-action=STOP \
            --scopes="default,storage-rw" &
    done
    wait
    echo "=== All VMs created. Waiting 60s for boot + GPU driver init... ==="
    sleep 60
}

# -- Upload code + model -----------------------------------------------------

upload_code() {
    echo "=== Packing code archive ==="
    cd "$LOCAL_ROOT"

    TAR_PATH="/tmp/ofc_t3_code.tar.gz"
    tar czf "$TAR_PATH" \
        ai/rust_solver \
        ai/training/convert_t3_rust_jsonl.py

    echo "=== Uploading code to ${NUM_VMS} VMs ==="
    for i in $(seq 0 $((NUM_VMS - 1))); do
        local instance="${PREFIX}-${i}"
        (
            gcloud compute ssh "$instance" --zone="$ZONE" --command="mkdir -p ~/ofc-pineapple/ai/data/t4_oracle_v2" 2>/dev/null
            gcloud compute scp "$TAR_PATH" "$instance:/tmp/ofc_t3_code.tar.gz" --zone="$ZONE" 2>/dev/null
            gcloud compute ssh "$instance" --zone="$ZONE" --command="
                cd ~/ofc-pineapple && tar xzf /tmp/ofc_t3_code.tar.gz && rm /tmp/ofc_t3_code.tar.gz
            " 2>/dev/null
            echo "  ${instance}: code uploaded"
        ) &
    done
    wait

    # Upload latest T4 ONNX to GCS first (if not already there)
    echo "=== Uploading T4 ONNX to GCS ==="
    gsutil -q cp "${LOCAL_ROOT}/ai/data/t4_oracle_v2/t4_oracle.onnx" \
        "${GCS_BUCKET}/${GCS_T4_ONNX}" 2>/dev/null || true

    # Download model from GCS to each VM
    echo "=== Downloading T4 ONNX to VMs from GCS ==="
    for i in $(seq 0 $((NUM_VMS - 1))); do
        local instance="${PREFIX}-${i}"
        (
            gcloud compute ssh "$instance" --zone="$ZONE" --command="
                gsutil -q cp ${GCS_BUCKET}/${GCS_T4_ONNX} ~/ofc-pineapple/ai/data/t4_oracle_v2/t4_oracle.onnx
            " 2>/dev/null
            echo "  ${instance}: model downloaded"
        ) &
    done
    wait
    echo "=== Upload complete ==="
}

# -- Setup VMs ---------------------------------------------------------------

setup_vms() {
    echo "=== Setting up ${NUM_VMS} VMs ==="

    SETUP_SCRIPT="/tmp/gcp_t3_setup.sh"
    cat > "$SETUP_SCRIPT" << 'SETUP_EOF'
#!/bin/bash
set -e

echo "--- Checking GPU ---"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "No GPU detected"

echo "--- Installing Rust + Python packages ---"
if ! command -v cargo >/dev/null 2>&1; then
    curl https://sh.rustup.rs -sSf | sh -s -- -y --profile minimal
    source "$HOME/.cargo/env"
fi
pip install -q numpy 2>/dev/null || true

# Create __init__.py if missing
touch ~/ofc-pineapple/ai/__init__.py
touch ~/ofc-pineapple/ai/training/__init__.py 2>/dev/null || true

echo "--- Verifying setup ---"
cd ~/ofc-pineapple
source "$HOME/.cargo/env" 2>/dev/null || true
cargo build -p t3_generator --release --manifest-path ai/rust_solver/Cargo.toml
python3 -c "
import numpy
print(f'NumPy: {numpy.__version__}')
"
echo "=== Setup complete ==="
SETUP_EOF

    for i in $(seq 0 $((NUM_VMS - 1))); do
        local instance="${PREFIX}-${i}"
        (
            gcloud compute scp "$SETUP_SCRIPT" "$instance:/tmp/gcp_t3_setup.sh" --zone="$ZONE" 2>/dev/null
            gcloud compute ssh "$instance" --zone="$ZONE" --command="
                sed -i 's/\r$//' /tmp/gcp_t3_setup.sh && bash /tmp/gcp_t3_setup.sh
            " 2>&1 | tail -8
            echo "  ${instance}: setup done"
        ) &
    done
    wait
    echo "=== All VMs set up ==="
}

# -- Run generation ----------------------------------------------------------

run_generation() {
    echo "=== Starting T3 teacher data generation ==="
    echo "    ${TOTAL_STATES} states / ${NUM_VMS} VMs = ${STATES_PER_VM} states/VM"
    echo "    BTN samples: ${BTN_SAMPLES}"
    echo ""

    for i in $(seq 0 $((NUM_VMS - 1))); do
        local instance="${PREFIX}-${i}"
        (
            gcloud compute ssh "$instance" --zone="$ZONE" --command="
                cd ~/ofc-pineapple
                mkdir -p ai/data/t3_results_vm${i}
                source \"\$HOME/.cargo/env\" 2>/dev/null || true

                # BB T3 (main workload)
                nohup bash -lc '
                    ./ai/rust_solver/target/release/t3_generator \
                        --states ${STATES_PER_VM} \
                        --mode bb \
                        --btn-samples ${BTN_SAMPLES} \
                        --onnx-model ai/data/t4_oracle_v2/t4_oracle.onnx \
                        --output ai/data/t3_results_vm${i}/bb/t3_teacher.jsonl \
                        --log-interval 100
                    python3 ai/training/convert_t3_rust_jsonl.py \
                        --input ai/data/t3_results_vm${i}/bb/t3_teacher.jsonl \
                        --output-dir ai/data/t3_results_vm${i}/bb \
                        --chunk-size ${NPZ_CHUNK_SIZE}
                ' > ai/data/t3_results_vm${i}/bb_gen.log 2>&1 &
                echo \"BB T3 PID: \$!\"

                # BTN T3 (fast, run simultaneously)
                nohup bash -lc '
                    ./ai/rust_solver/target/release/t3_generator \
                        --states ${STATES_PER_VM} \
                        --mode btn \
                        --btn-samples ${BTN_SAMPLES} \
                        --onnx-model ai/data/t4_oracle_v2/t4_oracle.onnx \
                        --output ai/data/t3_results_vm${i}/btn/t3_teacher.jsonl \
                        --log-interval 100
                    python3 ai/training/convert_t3_rust_jsonl.py \
                        --input ai/data/t3_results_vm${i}/btn/t3_teacher.jsonl \
                        --output-dir ai/data/t3_results_vm${i}/btn \
                        --chunk-size ${NPZ_CHUNK_SIZE}
                ' > ai/data/t3_results_vm${i}/btn_gen.log 2>&1 &
                echo \"BTN T3 PID: \$!\"

                echo \"Generation started on VM ${i}\"
            " 2>/dev/null
            echo "  ${instance}: generation started"
        ) &
    done
    wait
    echo "=== All VMs running ==="
}

# -- Check status ------------------------------------------------------------

check_status() {
    echo "=== Checking progress (${NUM_VMS} VMs) ==="
    for i in $(seq 0 $((NUM_VMS - 1))); do
        local instance="${PREFIX}-${i}"
        (
            result=$(gcloud compute ssh "$instance" --zone="$ZONE" --command="
                # BB status
                bb_files=\$(ls ~/ofc-pineapple/ai/data/t3_results_vm${i}/bb/*.npz 2>/dev/null | wc -l)
                bb_jsonl=\$(wc -l < ~/ofc-pineapple/ai/data/t3_results_vm${i}/bb/t3_teacher.jsonl 2>/dev/null || echo 0)
                bb_running=\$(pgrep -f 't3_generator.*mode bb' 2>/dev/null | wc -l)
                bb_log=\$(tail -1 ~/ofc-pineapple/ai/data/t3_results_vm${i}/bb_gen.log 2>/dev/null || echo 'no log')

                # BTN status
                btn_files=\$(ls ~/ofc-pineapple/ai/data/t3_results_vm${i}/btn/*.npz 2>/dev/null | wc -l)
                btn_jsonl=\$(wc -l < ~/ofc-pineapple/ai/data/t3_results_vm${i}/btn/t3_teacher.jsonl 2>/dev/null || echo 0)
                btn_running=\$(pgrep -f 't3_generator.*mode btn' 2>/dev/null | wc -l)
                btn_log=\$(tail -1 ~/ofc-pineapple/ai/data/t3_results_vm${i}/btn_gen.log 2>/dev/null || echo 'no log')

                echo \"BB: ${bb_jsonl} rows, ${bb_files} chunks, running=${bb_running} | BTN: ${btn_jsonl} rows, ${btn_files} chunks, running=${btn_running}\"
                echo \"  BB log: ${bb_log}\"
                echo \"  BTN log: ${btn_log}\"
            " 2>/dev/null)
            echo "  VM${i}: ${result}"
        ) &
    done
    wait
}

# -- Download + merge results ------------------------------------------------

download_results() {
    echo "=== Downloading results from ${NUM_VMS} VMs ==="
    LOCAL_DEST="${LOCAL_ROOT}/ai/data/t3_dataset_bi"
    mkdir -p "$LOCAL_DEST"

    for i in $(seq 0 $((NUM_VMS - 1))); do
        local instance="${PREFIX}-${i}"
        (
            # Upload to GCS from VM
            gcloud compute ssh "$instance" --zone="$ZONE" --command="
                gsutil -m cp ~/ofc-pineapple/ai/data/t3_results_vm${i}/bb/*.npz \
                    ${GCS_BUCKET}/${GCS_RESULTS}/bb/ 2>/dev/null || true
                gsutil -m cp ~/ofc-pineapple/ai/data/t3_results_vm${i}/btn/*.npz \
                    ${GCS_BUCKET}/${GCS_RESULTS}/btn/ 2>/dev/null || true
            " 2>/dev/null
            echo "  ${instance}: uploaded to GCS"
        ) &
    done
    wait

    # Download from GCS locally
    echo "=== Downloading from GCS ==="
    mkdir -p "$LOCAL_DEST/bb" "$LOCAL_DEST/btn"
    gsutil -m cp "${GCS_BUCKET}/${GCS_RESULTS}/bb/*.npz" "$LOCAL_DEST/bb/" 2>/dev/null || true
    gsutil -m cp "${GCS_BUCKET}/${GCS_RESULTS}/btn/*.npz" "$LOCAL_DEST/btn/" 2>/dev/null || true

    # Merge
    echo "=== Merging NPZ files ==="
    python3 -c "
import numpy as np
from pathlib import Path
import glob

dest = Path('${LOCAL_DEST}')
for mode in ['bb', 'btn']:
    npz_files = sorted(glob.glob(str(dest / mode / '*.npz')))
    if not npz_files:
        print(f'  {mode}: no files found')
        continue
    all_s, all_e, all_m = [], [], []
    for f in npz_files:
        d = np.load(f)
        all_s.append(d['states'])
        all_e.append(d['action_evs'])
        all_m.append(d['valid_masks'] if 'valid_masks' in d else d['action_masks'])
    states = np.concatenate(all_s)
    evs = np.concatenate(all_e)
    masks = np.concatenate(all_m)
    np.save(dest / f'{mode}_states.npy', states)
    np.save(dest / f'{mode}_action_evs.npy', evs)
    np.save(dest / f'{mode}_valid_masks.npy', masks)
    valid_evs = evs[masks]
    print(f'  {mode}: {states.shape[0]:,} samples, EV range [{valid_evs.min():.2f}, {valid_evs.max():.2f}]')

# Also merge both into single files
all_s, all_e, all_m = [], [], []
for mode in ['bb', 'btn']:
    s_file = dest / f'{mode}_states.npy'
    if s_file.exists():
        all_s.append(np.load(s_file))
        all_e.append(np.load(dest / f'{mode}_action_evs.npy'))
        all_m.append(np.load(dest / f'{mode}_valid_masks.npy'))
if all_s:
    states = np.concatenate(all_s)
    evs = np.concatenate(all_e)
    masks = np.concatenate(all_m)
    np.save(dest / 'states.npy', states)
    np.save(dest / 'action_evs.npy', evs)
    np.save(dest / 'valid_masks.npy', masks)
    print(f'  TOTAL: {states.shape[0]:,} samples')
"
    echo "=== Download complete ==="
}

# -- Delete VMs --------------------------------------------------------------

delete_vms() {
    echo "=== Deleting ${NUM_VMS} VMs ==="
    for i in $(seq 0 $((NUM_VMS - 1))); do
        local instance="${PREFIX}-${i}"
        gcloud compute instances delete "$instance" \
            --project="$PROJECT" --zone="$ZONE" --quiet &
    done
    wait
    echo "=== All VMs deleted ==="
}

# -- Main --------------------------------------------------------------------

case "${1:-help}" in
    create)  create_vms ;;
    upload)  upload_code ;;
    setup)   setup_vms ;;
    run)     run_generation ;;
    status)  check_status ;;
    download) download_results ;;
    delete)  delete_vms ;;
    all)
        create_vms
        upload_code
        setup_vms
        run_generation
        echo ""
        echo "============================================"
        echo "  ${NUM_VMS} GPU VMs running."
        echo "  Check progress: ./gcp_t3_teacher.sh status"
        echo "  Download:       ./gcp_t3_teacher.sh download"
        echo "  Cleanup:        ./gcp_t3_teacher.sh delete"
        echo "============================================"
        ;;
    *)
        echo "Usage: $0 {create|upload|setup|run|status|download|delete|all}"
        echo ""
        echo "  Config:"
        echo "    VMs:          ${NUM_VMS} x ${MACHINE_TYPE} + ${GPU_TYPE}"
        echo "    States:       ${TOTAL_STATES} total (${STATES_PER_VM}/VM)"
        echo "    BTN samples:  ${BTN_SAMPLES}"
        echo ""
        echo "  Est. time: ~3-5h for BB T3, BTN T3 finishes in minutes"
        echo "  Est. cost: ~\$5-10 (Spot pricing)"
        echo ""
        echo "  Workflow:"
        echo "    $0 all        # Create + upload + setup + run"
        echo "    $0 status     # Check progress"
        echo "    $0 download   # Download + merge results"
        echo "    $0 delete     # Cleanup"
        ;;
esac
