#!/bin/bash
# Remote selfplay launcher - run on GCP VM
# Usage: bash remote_run.sh <base_seed> <games_per_proc> <num_procs>

BASE_SEED=${1:-50000}
GAMES_PER_PROC=${2:-84}
NUM_PROCS=${3:-12}

BINARY="$HOME/rust_benchmark/target/release/ofc_benchmark"

echo "Killing existing processes..."
pkill -f ofc_benchmark 2>/dev/null || true
sleep 1

cd "$HOME/rust_benchmark"
echo "Working dir: $(pwd)"
echo "Launching $NUM_PROCS processes (seed base=$BASE_SEED, $GAMES_PER_PROC games each)"

for j in $(seq 0 $((NUM_PROCS - 1))); do
    SEED=$((BASE_SEED + j * GAMES_PER_PROC))
    OUTPUT="/tmp/selfplay_${j}.jsonl"
    LOG="/tmp/selfplay_${j}.log"

    # Clear old files
    rm -f "$OUTPUT" "$LOG"

    nohup "$BINARY" \
        --games "$GAMES_PER_PROC" \
        --seed "$SEED" \
        --mode selfplay \
        --mcts-sims 400 \
        --rollouts 250 \
        --output "$OUTPUT" \
        > "$LOG" 2>&1 &

    echo "  Proc $j: seed=$SEED PID=$!"
done

sleep 2
RUNNING=$(pgrep -f ofc_benchmark | wc -l)
echo ""
echo "Running processes: $RUNNING / $NUM_PROCS"
echo "Total games: $((GAMES_PER_PROC * NUM_PROCS))"
