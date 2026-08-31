#!/bin/bash
# GCP benchmark deployment script for per-turn BC evaluation
# Usage:
#   ./gcp_benchmark.sh setup       - Install dependencies and build FL solver
#   ./gcp_benchmark.sh run SEED N  - Run N games with seed SEED
#   ./gcp_benchmark.sh parallel N_JOBS GAMES_PER_JOB BASE_SEED - Run parallel jobs
#
# Example for 4 VMs, 3000 total games (16 parallel jobs each):
#   VM1: ./gcp_benchmark.sh parallel 16 47 0
#   VM2: ./gcp_benchmark.sh parallel 16 47 10000
#   VM3: ./gcp_benchmark.sh parallel 16 47 20000
#   VM4: ./gcp_benchmark.sh parallel 16 47 30000

set -e

WORKDIR=~/benchmark
MCTS_SIMS=500
VN_TRUNCATE_N=500

setup() {
    echo "=== Setting up benchmark environment ==="
    mkdir -p $WORKDIR
    cd $WORKDIR

    # Install Python, pip, Rust
    sudo apt-get update -qq
    sudo apt-get install -y -qq python3 python3-pip python3-venv curl build-essential > /dev/null 2>&1

    # Install Rust
    if ! command -v rustc &> /dev/null; then
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source "$HOME/.cargo/env"
    fi

    # Create venv and install deps
    python3 -m venv venv
    source venv/bin/activate
    pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
    pip install --quiet numpy

    # Extract code
    tar xzf ~/benchmark_code.tar.gz

    # Build FL solver
    echo "Building FL solver..."
    cd $WORKDIR/ai/rust_solver/fl_solver
    source "$HOME/.cargo/env"
    cargo build --release 2>&1 | tail -3
    cd $WORKDIR

    echo "=== Setup complete ==="
    echo "FL solver: $(ls -la ai/rust_solver/fl_solver/target/release/fl_solver 2>/dev/null || echo 'NOT FOUND')"
    echo "Models:"
    ls -la ai/models/expectimax_bc_v3/bc_policy_best.pt
    ls -la ai/models/value_v3/value_best.pt
    ls -la ai/models/bottomup_t*/bc_policy_best.pt
}

run() {
    SEED=$1
    N_GAMES=$2
    cd $WORKDIR
    source venv/bin/activate

    echo "=== Running benchmark: seed=$SEED, games=$N_GAMES ==="

    PYTHONUNBUFFERED=1 python3 -u eval_mcts.py \
        --games $N_GAMES \
        --seed $SEED \
        --full-mcts \
        --full-mcts-sims $MCTS_SIMS \
        --vn-truncate-depth 1 \
        --vn-truncate-n $VN_TRUNCATE_N \
        --bc-t1 ai/models/bottomup_t1/bc_policy_best.pt \
        --bc-t2 ai/models/bottomup_t2/bc_policy_best.pt \
        --bc-t3 ai/models/bottomup_t3/bc_policy_best.pt \
        --output results/result_${SEED}.jsonl
}

parallel_run() {
    N_JOBS=$1
    GAMES_PER_JOB=$2
    BASE_SEED=$3
    cd $WORKDIR
    source venv/bin/activate
    mkdir -p results

    echo "=== Launching $N_JOBS parallel jobs, $GAMES_PER_JOB games each ==="
    echo "Base seed: $BASE_SEED"

    for i in $(seq 0 $((N_JOBS - 1))); do
        SEED=$((BASE_SEED + i * GAMES_PER_JOB))
        echo "  Job $i: seed=$SEED, games=$GAMES_PER_JOB"

        PYTHONUNBUFFERED=1 python3 -u eval_mcts.py \
            --games $GAMES_PER_JOB \
            --seed $SEED \
            --full-mcts \
            --full-mcts-sims $MCTS_SIMS \
            --vn-truncate-depth 1 \
            --vn-truncate-n $VN_TRUNCATE_N \
            --bc-t1 ai/models/bottomup_t1/bc_policy_best.pt \
            --bc-t2 ai/models/bottomup_t2/bc_policy_best.pt \
            --bc-t3 ai/models/bottomup_t3/bc_policy_best.pt \
            --output results/result_${SEED}.jsonl \
            > results/log_${SEED}.txt 2>&1 &
    done

    echo "=== All jobs launched. Waiting... ==="
    wait
    echo "=== All jobs complete ==="

    # Quick summary
    total_games=0
    total_busts=0
    total_fl=0
    total_score=0
    for f in results/result_*.jsonl; do
        n=$(wc -l < "$f")
        busts=$(grep -c '"busted": true' "$f" || true)
        fl=$(grep -c '"fl_entry": true' "$f" || true)
        total_games=$((total_games + n))
        total_busts=$((total_busts + busts))
        total_fl=$((total_fl + fl))
    done
    echo "Total games: $total_games"
    echo "Total busts: $total_busts ($(echo "scale=1; $total_busts * 100 / $total_games" | bc)%)"
    echo "Total FL:    $total_fl ($(echo "scale=1; $total_fl * 100 / $total_games" | bc)%)"
    echo "Results in: $WORKDIR/results/"
}

case "$1" in
    setup)
        setup
        ;;
    run)
        run $2 $3
        ;;
    parallel)
        parallel_run $2 $3 $4
        ;;
    *)
        echo "Usage: $0 {setup|run SEED N|parallel N_JOBS GAMES_PER_JOB BASE_SEED}"
        exit 1
        ;;
esac
