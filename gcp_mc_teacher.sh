#!/bin/bash
# ============================================================================
# GCP MC Teacher Data Generation (10 VMs x 8 workers x 125 hands x 50 sims)
# ============================================================================
# Generates prob_engine MC teacher data for VN/BC training.
# Each VM runs 8 parallel workers, each handling a sub-shard.
#
# Usage:
#   ./gcp_mc_teacher.sh create     # Create 10 VMs
#   ./gcp_mc_teacher.sh upload     # Upload code to all VMs
#   ./gcp_mc_teacher.sh setup      # Install deps + build Rust on all VMs
#   ./gcp_mc_teacher.sh run        # Start generation on all VMs
#   ./gcp_mc_teacher.sh status     # Check progress on all VMs
#   ./gcp_mc_teacher.sh download   # Download results from all VMs
#   ./gcp_mc_teacher.sh delete     # Delete all VMs
#   ./gcp_mc_teacher.sh all        # create + upload + setup + run
# ============================================================================

set -e

PROJECT="ofc-solver-485418"
ZONES=("us-central1-a" "us-central1-b" "us-central1-c" "us-central1-f" "us-east1-b" "us-east1-c" "us-east1-d" "us-east4-a" "us-east4-b" "us-east4-c")
PREFIX="ofc-teacher"
NUM_VMS=30
MACHINE_TYPE="n2-highcpu-32"
IMAGE_FAMILY="ubuntu-2204-lts"
IMAGE_PROJECT="ubuntu-os-cloud"
DISK_SIZE="30GB"

TOTAL_HANDS=4800
SIMS=200
SEED=42
WORKERS_PER_VM=8
HANDS_PER_VM=$((TOTAL_HANDS / NUM_VMS))  # 1000
HANDS_PER_WORKER=$((HANDS_PER_VM / WORKERS_PER_VM))  # 125

LOCAL_ROOT="$(cd "$(dirname "$0")" && pwd)"

# -- Helper: get zone for VM index -----------------------------------------

vm_zone() {
    local i=$1
    local num_zones=${#ZONES[@]}
    echo "${ZONES[$((i % num_zones))]}"
}

for_each_vm() {
    local cmd="$1"
    for i in $(seq 0 $((NUM_VMS - 1))); do
        local instance="${PREFIX}-${i}"
        local z=$(vm_zone $i)
        echo "--- VM ${i}: ${instance} (${z}) ---"
        eval "$cmd" &
    done
    wait
    echo "--- All VMs done ---"
}

# -- Create VMs -------------------------------------------------------------

create_vms() {
    echo "=== Creating ${NUM_VMS} Spot VMs distributed across ${#ZONES[@]} zones ==="
    for i in $(seq 0 $((NUM_VMS - 1))); do
        local instance="${PREFIX}-${i}"
        local z=$(vm_zone $i)
        echo "  Creating ${instance} in ${z}..."
        gcloud compute instances create "$instance" \
            --project="$PROJECT" \
            --zone="$z" \
            --machine-type="$MACHINE_TYPE" \
            --image-family="$IMAGE_FAMILY" \
            --image-project="$IMAGE_PROJECT" \
            --boot-disk-size="$DISK_SIZE" \
            --boot-disk-type="pd-standard" \
            --provisioning-model=SPOT \
            --instance-termination-action=STOP \
            --scopes="default" &
    done
    wait
    echo "=== All VMs created. Waiting 30s for SSH... ==="
    sleep 30
}

# -- Upload code -------------------------------------------------------------

upload_code() {
    echo "=== Packing code archive ==="
    cd "$LOCAL_ROOT"

    TAR_PATH="/tmp/ofc_teacher_code.tar.gz"
    tar czf "$TAR_PATH" \
        generate_mc_teacher.py \
        ai/__init__.py \
        ai/rust_solver_wrapper.py \
        ai/prob_engine_wrapper.py \
        ai/engine/__init__.py \
        ai/engine/encoding.py \
        ai/engine/action_space.py \
        ai/engine/game_engine.py \
        ai/engine/scoring.py \
        ai/config/fl_ev.json \
        ai/rust_solver/Cargo.toml \
        ai/rust_solver/Cargo.lock \
        ai/rust_solver/fl_solver/Cargo.toml \
        ai/rust_solver/fl_solver/src/main.rs \
        ai/rust_solver/ofc_core/Cargo.toml \
        ai/rust_solver/ofc_core/src/lib.rs \
        ai/rust_solver/backward/Cargo.toml \
        ai/rust_solver/backward/src/main.rs \
        ai/rust_solver/prob_engine/Cargo.toml \
        ai/rust_solver/prob_engine/src/main.rs

    echo "=== Uploading to ${NUM_VMS} VMs ==="
    for i in $(seq 0 $((NUM_VMS - 1))); do
        local instance="${PREFIX}-${i}"
        local z=$(vm_zone $i)
        (
            gcloud compute ssh "$instance" --zone="$z" --command="mkdir -p ~/ofc-pineapple" 2>/dev/null
            gcloud compute scp "$TAR_PATH" "$instance:/tmp/ofc_teacher_code.tar.gz" --zone="$z" 2>/dev/null
            gcloud compute ssh "$instance" --zone="$z" --command="
                cd ~/ofc-pineapple && tar xzf /tmp/ofc_teacher_code.tar.gz && rm /tmp/ofc_teacher_code.tar.gz
            " 2>/dev/null
            echo "  ${instance}: uploaded"
        ) &
    done
    wait
    echo "=== Upload complete ==="
}

# -- Setup VMs ---------------------------------------------------------------

setup_vms() {
    echo "=== Setting up ${NUM_VMS} VMs ==="

    # Upload setup script
    SETUP_SCRIPT="/tmp/gcp_vm_setup.sh"
    cat > "$SETUP_SCRIPT" << 'SETUP_EOF'
#!/bin/bash
set -e

echo "--- Installing system packages ---"
sudo apt-get update -qq
sudo apt-get install -y -qq python3.11 python3.11-venv python3.11-dev curl build-essential

echo "--- Creating Python venv ---"
python3.11 -m venv ~/venv
source ~/venv/bin/activate
pip install --upgrade pip -q
pip install numpy -q

echo "--- Installing Rust ---"
if ! command -v cargo &>/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
source "$HOME/.cargo/env"
rustc --version

echo "--- Building prob_engine ---"
cd ~/ofc-pineapple/ai/rust_solver
cargo build --release -p prob_engine

echo "--- Patching paths for Linux ---"
cd ~/ofc-pineapple
sed -i 's/fl_solver\.exe/fl_solver/' ai/rust_solver_wrapper.py 2>/dev/null || true
sed -i 's/prob_engine\.exe/prob_engine/' ai/prob_engine_wrapper.py 2>/dev/null || true

echo "=== Setup complete ==="
SETUP_EOF

    for i in $(seq 0 $((NUM_VMS - 1))); do
        local instance="${PREFIX}-${i}"
        local z=$(vm_zone $i)
        (
            gcloud compute scp "$SETUP_SCRIPT" "$instance:/tmp/gcp_vm_setup.sh" --zone="$z" 2>/dev/null
            gcloud compute ssh "$instance" --zone="$z" --command="
                sed -i 's/\r$//' /tmp/gcp_vm_setup.sh && bash /tmp/gcp_vm_setup.sh
            " 2>&1 | tail -5
            echo "  ${instance}: setup done"
        ) &
    done
    wait
    echo "=== All VMs set up ==="
}

# -- Run teacher data generation ---------------------------------------------

run_teacher() {
    echo "=== Starting teacher data generation: ${TOTAL_HANDS} hands, ${SIMS} sims, ${NUM_VMS} VMs x ${WORKERS_PER_VM} workers ==="
    for i in $(seq 0 $((NUM_VMS - 1))); do
        local instance="${PREFIX}-${i}"
        local z=$(vm_zone $i)
        local vm_hand_start=$((i * HANDS_PER_VM))
        (
            gcloud compute ssh "$instance" --zone="$z" --command="
                source ~/venv/bin/activate
                source \"\\\$HOME/.cargo/env\" 2>/dev/null || true
                cd ~/ofc-pineapple
                mkdir -p teacher_results
                export PYTHONUNBUFFERED=1
                for w in \$(seq 0 $((WORKERS_PER_VM - 1))); do
                    hand_start=\$((${vm_hand_start} + w * ${HANDS_PER_WORKER}))
                    hand_end=\$((hand_start + ${HANDS_PER_WORKER}))
                    shard_id=\$((${i} * ${WORKERS_PER_VM} + w))
                    outfile=\"teacher_results/mc_s${SIMS}_shard\${shard_id}.jsonl\"
                    nohup python -u generate_mc_teacher.py \
                        --n-hands ${TOTAL_HANDS} --seed ${SEED} --sims ${SIMS} \
                        --hand-start \${hand_start} --hand-end \${hand_end} \
                        --output \${outfile} \
                        > teacher_results/shard\${shard_id}.log 2>&1 &
                    echo \"  Worker \${w}: shard \${shard_id}, hands [\${hand_start}, \${hand_end})\"
                done
                echo \"Started ${WORKERS_PER_VM} workers on VM ${i}\"
            " 2>/dev/null
            echo "  ${instance}: ${WORKERS_PER_VM} workers started"
        ) &
    done
    wait
    echo "=== All shards started (${NUM_VMS} VMs x ${WORKERS_PER_VM} workers = $((NUM_VMS * WORKERS_PER_VM)) shards) ==="
}

# -- Check status ------------------------------------------------------------

check_status() {
    echo "=== Checking progress (${NUM_VMS} VMs x ${WORKERS_PER_VM} workers) ==="
    for i in $(seq 0 $((NUM_VMS - 1))); do
        local instance="${PREFIX}-${i}"
        local z=$(vm_zone $i)
        (
            result=$(gcloud compute ssh "$instance" --zone="$z" --command="
                total_records=0
                total_hands=0
                running=0
                for w in \$(seq 0 $((WORKERS_PER_VM - 1))); do
                    sid=\$((${i} * ${WORKERS_PER_VM} + w))
                    logf=~/ofc-pineapple/teacher_results/shard\${sid}.log
                    dataf=~/ofc-pineapple/teacher_results/mc_s${SIMS}_shard\${sid}.jsonl
                    if [ -f \${dataf} ]; then
                        lines=\$(wc -l < \${dataf} 2>/dev/null || echo 0)
                        total_records=\$((total_records + lines))
                        hands=\$((lines / 6))
                        total_hands=\$((total_hands + hands))
                    fi
                    if pgrep -f \"shard\${sid}.jsonl\" >/dev/null 2>&1; then
                        running=\$((running + 1))
                    fi
                done
                echo \"records=\${total_records} hands=\${total_hands}/${HANDS_PER_VM} running=\${running}/${WORKERS_PER_VM}\"
                # Show latest log line from first active worker
                for w in \$(seq 0 $((WORKERS_PER_VM - 1))); do
                    sid=\$((${i} * ${WORKERS_PER_VM} + w))
                    logf=~/ofc-pineapple/teacher_results/shard\${sid}.log
                    if [ -f \${logf} ]; then
                        tail -1 \${logf} 2>/dev/null
                        break
                    fi
                done
            " 2>/dev/null)
            echo "  VM${i}: ${result}"
        ) &
    done
    wait
}

# -- Download results --------------------------------------------------------

download_results() {
    TOTAL_SHARDS=$((NUM_VMS * WORKERS_PER_VM))
    echo "=== Downloading results (${TOTAL_SHARDS} shards) ==="
    LOCAL_DEST="/d/ofc_data/mc_teacher"
    mkdir -p "$LOCAL_DEST"
    for i in $(seq 0 $((NUM_VMS - 1))); do
        local instance="${PREFIX}-${i}"
        local z=$(vm_zone $i)
        (
            for w in $(seq 0 $((WORKERS_PER_VM - 1))); do
                sid=$((i * WORKERS_PER_VM + w))
                # Use SSH cat to avoid pscp issues on Windows
                gcloud compute ssh "$instance" --zone="$z" --command="
                    cat ~/ofc-pineapple/teacher_results/mc_s${SIMS}_shard${sid}.jsonl 2>/dev/null
                " 2>/dev/null > "$LOCAL_DEST/mc_s${SIMS}_shard${sid}.jsonl"
                gcloud compute ssh "$instance" --zone="$z" --command="
                    cat ~/ofc-pineapple/teacher_results/shard${sid}.log 2>/dev/null
                " 2>/dev/null > "$LOCAL_DEST/shard${sid}.log"
            done
            echo "  ${instance}: downloaded ${WORKERS_PER_VM} shards"
        ) &
    done
    wait

    # Merge results
    echo "=== Merging results ==="
    cat "$LOCAL_DEST"/mc_s${SIMS}_shard*.jsonl \
        > "$LOCAL_DEST/mc_s${SIMS}_merged.jsonl"
    total=$(wc -l < "$LOCAL_DEST/mc_s${SIMS}_merged.jsonl")
    echo "  Merged: ${total} records (saved to $LOCAL_DEST)"

    # Compute aggregate stats
    python3 -c "
import json, numpy as np
results = []
with open('/d/ofc_data/mc_teacher/mc_s${SIMS}_merged.jsonl') as f:
    for line in f:
        d = json.loads(line)
        if d.get('turn') == -1:
            results.append(d)
n = len(results)
busts = sum(1 for r in results if r['busted'])
fls = sum(1 for r in results if r['fl_entry'])
scores = [r['score'] for r in results]
s = np.array(scores)
se = s.std() / np.sqrt(n)
print(f'\n  Results ({n} hands):')
print(f'  Score:  {s.mean():+.2f} +/- {s.std():.2f}  (95%CI: [{s.mean()-1.96*se:+.2f}, {s.mean()+1.96*se:+.2f}])')
print(f'  Bust:   {busts/n*100:.1f}%')
print(f'  FL:     {fls/n*100:.1f}%')
# Count turn records
all_records = []
with open('/d/ofc_data/mc_teacher/mc_s${SIMS}_merged.jsonl') as f:
    for line in f:
        all_records.append(json.loads(line))
turn_records = [r for r in all_records if r.get('turn', -1) >= 0]
print(f'  Turn records: {len(turn_records)}')
for t in range(5):
    tr = [r for r in turn_records if r['turn'] == t]
    if tr:
        modes = set(r.get('eval_mode', '?') for r in tr)
        print(f'    T{t}: {len(tr)} records, modes={modes}')
"
    echo "=== Done ==="
}

# -- Delete VMs --------------------------------------------------------------

delete_vms() {
    echo "=== Deleting ${NUM_VMS} VMs ==="
    for i in $(seq 0 $((NUM_VMS - 1))); do
        local instance="${PREFIX}-${i}"
        local z=$(vm_zone $i)
        gcloud compute instances delete "$instance" \
            --project="$PROJECT" --zone="$z" --quiet &
    done
    wait
    echo "=== All VMs deleted ==="
}

# -- Main --------------------------------------------------------------------

case "${1:-help}" in
    create)  create_vms ;;
    upload)  upload_code ;;
    setup)   setup_vms ;;
    run)     run_teacher ;;
    status)  check_status ;;
    download) download_results ;;
    delete)  delete_vms ;;
    all)
        create_vms
        upload_code
        setup_vms
        run_teacher
        echo ""
        echo "============================================"
        echo "  ${NUM_VMS} VMs running. Check progress with:"
        echo "    ./gcp_mc_teacher.sh status"
        echo "  Download results when done:"
        echo "    ./gcp_mc_teacher.sh download"
        echo "  Delete VMs:"
        echo "    ./gcp_mc_teacher.sh delete"
        echo "============================================"
        ;;
    *)
        echo "Usage: $0 {create|upload|setup|run|status|download|delete|all}"
        echo ""
        echo "  Config: ${TOTAL_HANDS} hands, ${SIMS} sims, ${NUM_VMS} VMs x ${WORKERS_PER_VM} workers (${HANDS_PER_WORKER}/worker)"
        echo "  Est. time: ~2.5h (${HANDS_PER_WORKER} hands/worker @ ~70s/hand)"
        echo ""
        echo "  Workflow:"
        echo "    $0 all        # Create + upload + setup + run"
        echo "    $0 status     # Check progress"
        echo "    $0 download   # Download + merge results"
        echo "    $0 delete     # Cleanup"
        ;;
esac
