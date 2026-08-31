#!/bin/bash
# ============================================================================
# GCP prob_engine MC Benchmark (10 VMs × 30 games × 50 sims)
# ============================================================================
# Uses Spot VMs for cost savings. Each VM runs a shard of games.
#
# Usage:
#   ./gcp_pe_mc_bench.sh create     # Create 10 VMs
#   ./gcp_pe_mc_bench.sh upload     # Upload code to all VMs
#   ./gcp_pe_mc_bench.sh setup      # Install deps + build Rust on all VMs
#   ./gcp_pe_mc_bench.sh run        # Start benchmark on all VMs
#   ./gcp_pe_mc_bench.sh status     # Check progress on all VMs
#   ./gcp_pe_mc_bench.sh download   # Download results from all VMs
#   ./gcp_pe_mc_bench.sh delete     # Delete all VMs
#   ./gcp_pe_mc_bench.sh all        # create + upload + setup + run
# ============================================================================

set -e

PROJECT="ofc-solver-485418"
ZONE="us-central1-a"
PREFIX="ofc-pemc"
NUM_VMS=10
MACHINE_TYPE="n2-highcpu-8"
IMAGE_FAMILY="ubuntu-2204-lts"
IMAGE_PROJECT="ubuntu-os-cloud"
DISK_SIZE="30GB"

TOTAL_GAMES=300
SIMS=50
SEED=42
GAMES_PER_VM=$((TOTAL_GAMES / NUM_VMS))  # 30

LOCAL_ROOT="$(cd "$(dirname "$0")" && pwd)"

# ── Helper: run on all VMs ────────────────────────────────────────────────

for_each_vm() {
    local cmd="$1"
    for i in $(seq 0 $((NUM_VMS - 1))); do
        local instance="${PREFIX}-${i}"
        echo "--- VM ${i}: ${instance} ---"
        eval "$cmd" &
    done
    wait
    echo "--- All VMs done ---"
}

# ── Create VMs ────────────────────────────────────────────────────────────

create_vms() {
    echo "=== Creating ${NUM_VMS} Spot VMs ==="
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
            --boot-disk-type="pd-standard" \
            --provisioning-model=SPOT \
            --instance-termination-action=STOP \
            --scopes="default" &
    done
    wait
    echo "=== All VMs created. Waiting 30s for SSH... ==="
    sleep 30
}

# ── Upload code ───────────────────────────────────────────────────────────

upload_code() {
    echo "=== Packing code archive ==="
    cd "$LOCAL_ROOT"

    TAR_PATH="/tmp/ofc_pemc_code.tar.gz"
    tar czf "$TAR_PATH" \
        eval_mcts.py \
        ai/__init__.py \
        ai/rust_solver_wrapper.py \
        ai/prob_engine_wrapper.py \
        ai/engine/__init__.py \
        ai/engine/encoding.py \
        ai/engine/action_space.py \
        ai/engine/game_engine.py \
        ai/engine/scoring.py \
        ai/mcts/__init__.py \
        ai/mcts/mcts.py \
        ai/mcts/rollout_evaluator.py \
        ai/mcts/multi_turn_mcts.py \
        ai/mcts/ofc_mcts.py \
        ai/models/__init__.py \
        ai/models/networks.py \
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
        (
            gcloud compute ssh "$instance" --zone="$ZONE" --command="mkdir -p ~/ofc-pineapple" 2>/dev/null
            gcloud compute scp "$TAR_PATH" "$instance:/tmp/ofc_pemc_code.tar.gz" --zone="$ZONE" 2>/dev/null
            gcloud compute ssh "$instance" --zone="$ZONE" --command="
                cd ~/ofc-pineapple && tar xzf /tmp/ofc_pemc_code.tar.gz && rm /tmp/ofc_pemc_code.tar.gz
            " 2>/dev/null
            echo "  ${instance}: uploaded"
        ) &
    done
    wait

    # Upload BC model
    echo "=== Uploading BC model ==="
    BC_MODEL="ai/models/expectimax_bc_v3/bc_policy_best.pt"
    if [ -f "$LOCAL_ROOT/$BC_MODEL" ]; then
        for i in $(seq 0 $((NUM_VMS - 1))); do
            local instance="${PREFIX}-${i}"
            (
                gcloud compute ssh "$instance" --zone="$ZONE" --command="mkdir -p ~/ofc-pineapple/ai/models/expectimax_bc_v3" 2>/dev/null
                gcloud compute scp "$LOCAL_ROOT/$BC_MODEL" "$instance:~/ofc-pineapple/$BC_MODEL" --zone="$ZONE" 2>/dev/null
                echo "  ${instance}: model uploaded"
            ) &
        done
        wait
    else
        echo "  WARNING: $BC_MODEL not found!"
    fi

    echo "=== Upload complete ==="
}

# ── Setup VMs ─────────────────────────────────────────────────────────────

setup_vms() {
    echo "=== Setting up ${NUM_VMS} VMs ==="
    for i in $(seq 0 $((NUM_VMS - 1))); do
        local instance="${PREFIX}-${i}"
        (
            gcloud compute ssh "$instance" --zone="$ZONE" --command="
                set -e

                # Install system packages
                sudo apt-get update -qq
                sudo apt-get install -y -qq python3.11 python3.11-venv python3.11-dev curl build-essential

                # Python venv
                python3.11 -m venv ~/venv
                source ~/venv/bin/activate
                pip install --upgrade pip -q
                pip install torch --index-url https://download.pytorch.org/whl/cpu -q
                pip install numpy -q

                # Rust
                if ! command -v cargo &>/dev/null; then
                    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
                fi
                source \"\\\$HOME/.cargo/env\"

                # Build FL solver + prob_engine
                cd ~/ofc-pineapple/ai/rust_solver
                cargo build --release -p fl_solver -p prob_engine

                # Patch paths for Linux
                cd ~/ofc-pineapple
                sed -i 's/fl_solver\\.exe/fl_solver/' ai/rust_solver_wrapper.py
                sed -i 's/prob_engine\\.exe/prob_engine/' ai/prob_engine_wrapper.py

                echo '=== VM ${i} setup complete ==='
            " 2>&1 | tail -5
            echo "  ${instance}: setup done"
        ) &
    done
    wait
    echo "=== All VMs set up ==="
}

# ── Run benchmark ─────────────────────────────────────────────────────────

run_bench() {
    echo "=== Starting benchmark: ${TOTAL_GAMES} games, ${SIMS} sims, ${NUM_VMS} VMs ==="
    for i in $(seq 0 $((NUM_VMS - 1))); do
        local instance="${PREFIX}-${i}"
        local game_start=$((i * GAMES_PER_VM))
        local game_end=$(((i + 1) * GAMES_PER_VM))
        local outfile="bench_results/pemc_s${SIMS}_shard${i}.jsonl"
        (
            gcloud compute ssh "$instance" --zone="$ZONE" --command="
                source ~/venv/bin/activate
                source \"\\\$HOME/.cargo/env\" 2>/dev/null || true
                cd ~/ofc-pineapple
                mkdir -p bench_results
                export PYTHONUNBUFFERED=1
                nohup python -u eval_mcts.py \
                    --games ${TOTAL_GAMES} --seed ${SEED} \
                    --pe-mc-sims ${SIMS} \
                    --game-start ${game_start} --game-end ${game_end} \
                    --output ${outfile} \
                    > bench_results/pemc_shard${i}.log 2>&1 &
                echo \"Started shard ${i}: games [${game_start}, ${game_end})\"
            " 2>/dev/null
            echo "  ${instance}: shard ${i} started [${game_start}, ${game_end})"
        ) &
    done
    wait
    echo "=== All shards started ==="
}

# ── Check status ──────────────────────────────────────────────────────────

check_status() {
    echo "=== Checking progress ==="
    for i in $(seq 0 $((NUM_VMS - 1))); do
        local instance="${PREFIX}-${i}"
        (
            result=$(gcloud compute ssh "$instance" --zone="$ZONE" --command="
                if [ -f ~/ofc-pineapple/bench_results/pemc_shard${i}.log ]; then
                    lines=\$(wc -l < ~/ofc-pineapple/bench_results/pemc_shard${i}.jsonl 2>/dev/null || echo 0)
                    tail -1 ~/ofc-pineapple/bench_results/pemc_shard${i}.log 2>/dev/null
                    echo \"  -> \${lines}/${GAMES_PER_VM} games done\"
                else
                    echo 'Not started yet'
                fi
            " 2>/dev/null)
            echo "  VM${i}: ${result}"
        ) &
    done
    wait
}

# ── Download results ──────────────────────────────────────────────────────

download_results() {
    echo "=== Downloading results ==="
    mkdir -p "$LOCAL_ROOT/data/test_results"
    for i in $(seq 0 $((NUM_VMS - 1))); do
        local instance="${PREFIX}-${i}"
        (
            gcloud compute scp \
                "$instance:~/ofc-pineapple/bench_results/pemc_s${SIMS}_shard${i}.jsonl" \
                "$LOCAL_ROOT/data/test_results/" --zone="$ZONE" 2>/dev/null
            gcloud compute scp \
                "$instance:~/ofc-pineapple/bench_results/pemc_shard${i}.log" \
                "$LOCAL_ROOT/data/test_results/" --zone="$ZONE" 2>/dev/null
            echo "  ${instance}: downloaded"
        ) &
    done
    wait

    # Merge results
    echo "=== Merging results ==="
    cat "$LOCAL_ROOT/data/test_results"/pemc_s${SIMS}_shard*.jsonl \
        > "$LOCAL_ROOT/data/test_results/pemc_s${SIMS}_merged.jsonl"
    total=$(wc -l < "$LOCAL_ROOT/data/test_results/pemc_s${SIMS}_merged.jsonl")
    echo "  Merged: ${total} games"

    # Compute aggregate stats
    python3 -c "
import json, numpy as np
results = []
with open('$LOCAL_ROOT/data/test_results/pemc_s${SIMS}_merged.jsonl') as f:
    for line in f:
        results.append(json.loads(line))
n = len(results)
busts = sum(1 for r in results if r['busted'])
fls = sum(1 for r in results if r['fl_entry'])
scores = [r['total_score'] for r in results]
s = np.array(scores)
se = s.std() / np.sqrt(n)
print(f'\\n  Results ({n} games):')
print(f'  Score:  {s.mean():+.2f} +/- {s.std():.2f}  (95%CI: [{s.mean()-1.96*se:+.2f}, {s.mean()+1.96*se:+.2f}])')
print(f'  Bust:   {busts/n*100:.1f}%')
print(f'  FL:     {fls/n*100:.1f}%')
print(f'  Win:    {(s>0).sum()/n*100:.1f}%')
"
    echo "=== Done ==="
}

# ── Delete VMs ────────────────────────────────────────────────────────────

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

# ── Main ──────────────────────────────────────────────────────────────────

case "${1:-help}" in
    create)  create_vms ;;
    upload)  upload_code ;;
    setup)   setup_vms ;;
    run)     run_bench ;;
    status)  check_status ;;
    download) download_results ;;
    delete)  delete_vms ;;
    all)
        create_vms
        upload_code
        setup_vms
        run_bench
        echo ""
        echo "============================================"
        echo "  10 VMs running. Check progress with:"
        echo "    ./gcp_pe_mc_bench.sh status"
        echo "  Download results when done:"
        echo "    ./gcp_pe_mc_bench.sh download"
        echo "============================================"
        ;;
    *)
        echo "Usage: $0 {create|upload|setup|run|status|download|delete|all}"
        echo ""
        echo "  Config: ${TOTAL_GAMES} games, ${SIMS} sims, ${NUM_VMS} VMs (${GAMES_PER_VM}/VM)"
        echo "  Est. time: ~32 min per VM"
        echo ""
        echo "  Workflow:"
        echo "    $0 all        # Create + upload + setup + run"
        echo "    $0 status     # Check progress"
        echo "    $0 download   # Download + merge results"
        echo "    $0 delete     # Cleanup"
        ;;
esac
