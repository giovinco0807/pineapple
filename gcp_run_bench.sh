#!/bin/bash
# ============================================================================
# GCP Benchmark Runner
# ============================================================================
# Run eval_mcts.py benchmarks on a GCP VM.
# Designed for n2-standard-8 (8 vCPU, 32GB RAM), CPU-only PyTorch.
#
# Usage (on the VM):
#   ./gcp_run_bench.sh                    # Run default benchmark suite
#   ./gcp_run_bench.sh single             # Run a single quick test (50 games)
#   ./gcp_run_bench.sh custom "ARGS"      # Run with custom arguments
#
# Results are saved to ~/ofc-pineapple/bench_results/
# ============================================================================

set -e

cd ~/ofc-pineapple
source ~/venv/bin/activate
source "$HOME/.cargo/env" 2>/dev/null || true

export PYTHONUNBUFFERED=1

RESULTS_DIR="bench_results"
mkdir -p "$RESULTS_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ── Default model paths ──────────────────────────────────────────────────
BC_MODEL="ai/models/expectimax_bc_v4/bc_policy_best.pt"
VN_MODEL_V5="ai/models/value_v5/value_best.pt"
VN_MODEL_V6="ai/models/value_v6/value_best.pt"

# ── Verify setup ─────────────────────────────────────────────────────────

verify() {
    echo "============================================================"
    echo "  OFC Pineapple - GCP Benchmark Runner"
    echo "============================================================"
    echo "  Host:     $(hostname)"
    echo "  CPUs:     $(nproc)"
    echo "  RAM:      $(free -h | awk '/Mem:/ {print $2}')"
    echo "  Python:   $(python --version 2>&1)"
    echo "  PyTorch:  $(python -c 'import torch; print(torch.__version__)' 2>&1)"
    echo "  NumPy:    $(python -c 'import numpy; print(numpy.__version__)' 2>&1)"
    echo "  Rust:     $(rustc --version 2>&1 || echo 'not found')"
    echo ""

    # Check FL solver binary
    FL_BIN="ai/rust_solver/target/release/fl_solver"
    if [ -f "$FL_BIN" ]; then
        echo "  FL solver: OK ($FL_BIN)"
    else
        echo "  FL solver: MISSING! Run: cd ai/rust_solver && cargo build --release -p fl_solver"
        exit 1
    fi

    # Check model files
    for model in "$BC_MODEL" "$VN_MODEL_V5" "$VN_MODEL_V6"; do
        if [ -f "$model" ]; then
            size=$(du -h "$model" | cut -f1)
            echo "  Model:    OK $model ($size)"
        else
            echo "  Model:    MISSING $model"
        fi
    done

    # Quick import test
    echo ""
    echo "  Testing imports..."
    python -c "
from ai.engine.encoding import Board, Observation, ALL_CARDS, encode_state
from ai.engine.action_space import get_initial_actions, get_turn_actions, create_action_mask
from ai.engine.game_engine import GameEngine, Hand, evaluate_hand
from ai.models.networks import PolicyNetwork, ValueNetwork
from ai.mcts.rollout_evaluator import RolloutEvaluator
from ai.mcts.multi_turn_mcts import MultiTurnMCTS
from ai.mcts.ofc_mcts import OFC_MCTS
from ai.rust_solver_wrapper import RustFLSolver
from ai.engine.scoring import check_fl_stay_from_cards
print('  All imports OK')
"
    echo "============================================================"
    echo ""
}

# ── Run a benchmark ──────────────────────────────────────────────────────

run_bench() {
    local label="$1"
    shift
    local logfile="${RESULTS_DIR}/${TIMESTAMP}_${label}.log"

    echo "──────────────────────────────────────────────────────────────"
    echo "  Benchmark: ${label}"
    echo "  Log: ${logfile}"
    echo "  Args: $@"
    echo "  Started: $(date)"
    echo "──────────────────────────────────────────────────────────────"

    # Run with tee to both console and log file
    python -u eval_mcts.py "$@" 2>&1 | tee "$logfile"

    echo ""
    echo "  Finished: $(date)"
    echo "  Log saved: ${logfile}"
    echo ""
}

# ── Benchmark suite ──────────────────────────────────────────────────────

run_suite() {
    verify

    echo "============================================================"
    echo "  Running benchmark suite ($(date))"
    echo "============================================================"
    echo ""

    # ── 1. OFC_MCTS full (500 sims) + VN v5 + Rollout T1+ (r=50, top-k=3) ──
    run_bench "fullmcts_500s_vnv5_r50_k3" \
        --games 500 --seed 42 \
        --full-mcts --full-mcts-sims 500 --full-mcts-depth 3 \
        --bc-model "$BC_MODEL" \
        --vn-model "$VN_MODEL_V5" \
        --rollout-vn-top-k 3 --t1-rollouts 50

    # ── 2. OFC_MCTS full (500 sims) + VN v6 + Rollout T1+ (r=50, top-k=3) ──
    if [ -f "$VN_MODEL_V6" ]; then
        run_bench "fullmcts_500s_vnv6_r50_k3" \
            --games 500 --seed 42 \
            --full-mcts --full-mcts-sims 500 --full-mcts-depth 3 \
            --bc-model "$BC_MODEL" \
            --vn-model "$VN_MODEL_V6" \
            --rollout-vn-top-k 3 --t1-rollouts 50
    fi

    # ── 3. OFC_MCTS full (1000 sims) + VN v5 ──
    run_bench "fullmcts_1000s_vnv5_r50_k3" \
        --games 500 --seed 42 \
        --full-mcts --full-mcts-sims 1000 --full-mcts-depth 3 \
        --bc-model "$BC_MODEL" \
        --vn-model "$VN_MODEL_V5" \
        --rollout-vn-top-k 3 --t1-rollouts 50

    # ── 4. MultiTurnMCTS T0 (500 sims) + Rollout T1+ (r=50, top-k=5) ──
    run_bench "mcts_500s_vnv5_r50_k5" \
        --games 500 --seed 42 \
        --mcts-sims 500 \
        --bc-model "$BC_MODEL" \
        --vn-model "$VN_MODEL_V5" \
        --rollout-vn-top-k 5 --t1-rollouts 50

    # ── 5. Rollout only (r=250, no MCTS) ──
    run_bench "rollout_r250_k10" \
        --games 500 --seed 42 \
        --mcts-sims 0 \
        --bc-model "$BC_MODEL" \
        --vn-model "$VN_MODEL_V5" \
        --rollout-vn-top-k 10 --t1-rollouts 250

    echo ""
    echo "============================================================"
    echo "  All benchmarks complete!"
    echo "  Results in: ${RESULTS_DIR}/"
    echo "============================================================"
    ls -la "${RESULTS_DIR}/"
}

# ── Single quick test ────────────────────────────────────────────────────

run_single() {
    verify

    run_bench "quick_test" \
        --games 50 --seed 42 \
        --full-mcts --full-mcts-sims 500 --full-mcts-depth 3 \
        --bc-model "$BC_MODEL" \
        --vn-model "$VN_MODEL_V5" \
        --rollout-vn-top-k 3 --t1-rollouts 50
}

# ── Custom run ───────────────────────────────────────────────────────────

run_custom() {
    verify

    local label="custom_$(date +%H%M%S)"
    run_bench "$label" $@
}

# ── Main ─────────────────────────────────────────────────────────────────

case "${1:-suite}" in
    suite)
        run_suite
        ;;
    single)
        run_single
        ;;
    custom)
        shift
        run_custom "$@"
        ;;
    verify)
        verify
        ;;
    *)
        echo "Usage: $0 {suite|single|custom|verify}"
        echo ""
        echo "  suite    - Run full benchmark suite (5 configs x 500 games each)"
        echo "  single   - Quick smoke test (50 games)"
        echo "  custom   - Custom args, e.g.: $0 custom --games 100 --full-mcts --mcts-sims 800"
        echo "  verify   - Check setup without running benchmarks"
        echo ""
        echo "Default model paths:"
        echo "  BC:  $BC_MODEL"
        echo "  VN5: $VN_MODEL_V5"
        echo "  VN6: $VN_MODEL_V6"
        ;;
esac
