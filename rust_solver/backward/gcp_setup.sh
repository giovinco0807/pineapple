#!/bin/bash
# GCP Expectimax Setup & Run Script
# Upload this + Rust source to a GCP Spot VM, then run.
#
# Usage on VM:
#   chmod +x gcp_setup.sh
#   ./gcp_setup.sh setup    # Install Rust, compile
#   ./gcp_setup.sh run      # Run all patterns
#   ./gcp_setup.sh test     # Run 1 pattern (smoke test)

set -e

WORK_DIR="$HOME/expectimax"
PATTERNS_FILE="$WORK_DIR/canonical_patterns.txt"
RESULTS_DIR="$WORK_DIR/results"
BINARY="$WORK_DIR/rust_solver/target/release/backward"

# Sampling params (set after local Mini vs Practical comparison)
N1=30
N2=10
N3=3
N4=2
SEED=42

# FL EV values (from 1000-trial calc_fl_ev.py, chain EV with opp_R=5.0)
# These are hardcoded in the binary, but also used for logging

# Number of patterns to process (0 = all)
MAX_PATTERNS=1500

setup() {
    echo "=== Installing Rust ==="
    if ! command -v cargo &>/dev/null; then
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source "$HOME/.cargo/env"
    fi
    rustc --version
    cargo --version

    echo "=== Building Expectimax engine ==="
    cd "$WORK_DIR/rust_solver"
    cargo build --release -p backward
    echo "=== Build complete ==="
    ls -la target/release/backward
}

test_run() {
    echo "=== Smoke test: 1 pattern ==="
    RAYON_NUM_THREADS=$(nproc) "$BINARY" expectimax \
        --t0 Ah,Kh,Qd,7c,3s \
        --n1 $N1 --n2 $N2 --n3 $N3 --n4 $N4 \
        --seed $SEED --top 20
}

run_all() {
    mkdir -p "$RESULTS_DIR"

    if [ ! -f "$PATTERNS_FILE" ]; then
        echo "ERROR: $PATTERNS_FILE not found"
        exit 1
    fi

    TOTAL=$(wc -l < "$PATTERNS_FILE")
    if [ "$MAX_PATTERNS" -gt 0 ] && [ "$MAX_PATTERNS" -lt "$TOTAL" ]; then
        TOTAL=$MAX_PATTERNS
    fi

    CORES=$(nproc)
    echo "=== Running Expectimax on $TOTAL patterns ==="
    echo "Params: N1=$N1, N2=$N2, N3=$N3, N4=$N4"
    echo "Cores: $CORES"
    echo "Start: $(date)"

    COUNT=0
    while IFS= read -r pattern; do
        COUNT=$((COUNT + 1))
        if [ "$MAX_PATTERNS" -gt 0 ] && [ "$COUNT" -gt "$MAX_PATTERNS" ]; then
            break
        fi

        # Output file: pattern hash as filename
        HASH=$(echo "$pattern" | md5sum | cut -c1-12)
        OUTFILE="$RESULTS_DIR/${HASH}.json"

        if [ -f "$OUTFILE" ]; then
            echo "[$COUNT/$TOTAL] SKIP $pattern (already done)"
            continue
        fi

        START_TIME=$(date +%s)

        # Run expectimax via stdin JSON protocol for full output
        echo "{\"t0_hand\":[$(echo "$pattern" | sed 's/,/","/g' | sed 's/^/"/;s/$/"/')],\"n1\":$N1,\"n2\":$N2,\"n3\":$N3,\"n4\":$N4,\"seed\":$((SEED + COUNT)),\"fl_ev\":{\"14\":15.8,\"15\":22.7,\"16\":28.6,\"17\":35.1},\"bust_penalty\":-6.0,\"top_k\":0}" \
            | RAYON_NUM_THREADS=$CORES "$BINARY" stdin \
            > "$OUTFILE"

        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))

        # Extract best EV from JSON
        BEST_EV=$(python3 -c "import json; d=json.load(open('$OUTFILE')); print(f'{d[\"best_ev\"]:.2f}')" 2>/dev/null || echo "?")

        echo "[$COUNT/$TOTAL] ${ELAPSED}s  EV=$BEST_EV  $pattern"

    done < "$PATTERNS_FILE"

    echo "=== Done: $(date) ==="
    echo "Results in $RESULTS_DIR/ ($COUNT files)"
}

case "${1:-}" in
    setup)  setup ;;
    test)   test_run ;;
    run)    run_all ;;
    *)
        echo "Usage: $0 {setup|test|run}"
        exit 1
        ;;
esac
