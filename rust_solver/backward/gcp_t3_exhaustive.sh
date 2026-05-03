#!/bin/bash
# GCP T3 Exhaustive Evaluation Script
# Runs t3eval --exhaustive on assigned pattern range.
#
# Usage:
#   ./gcp_t3_exhaustive.sh setup       # Install Rust, compile
#   ./gcp_t3_exhaustive.sh run START END  # Run patterns [START, END)
#   ./gcp_t3_exhaustive.sh test        # Smoke test (2 patterns)

set -e

WORK_DIR="$HOME/expectimax"
PATTERNS_FILE="$WORK_DIR/canonical_patterns.txt"
BINARY="$WORK_DIR/rust_solver/target/release/backward"
RESULTS_DIR="$WORK_DIR/t3_results"

# Bottom-up params
N1=50
N2=10
N3=3
SEED=42

# FL EV (chain mode)
BUST_PENALTY=-6.0
FL_EV_14=16.8
FL_EV_15=27.9
FL_EV_16=52.4
FL_EV_17=104.5

setup() {
    echo "=== Installing Rust ==="
    if ! command -v cargo &>/dev/null; then
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source "$HOME/.cargo/env"
    fi
    rustc --version
    cargo --version

    echo "=== Building backward engine ==="
    cd "$WORK_DIR/rust_solver"
    cargo build --release -p backward
    echo "=== Build complete ==="
    ls -la target/release/backward
}

test_run() {
    echo "=== Smoke test: 2 patterns, exhaustive T4 ==="
    head -2 "$PATTERNS_FILE" | RAYON_NUM_THREADS=$(nproc) "$BINARY" t3eval \
        --exhaustive --n1 2 --n2 2 --n3 2 \
        --bust-penalty $BUST_PENALTY \
        --fl-ev-14 $FL_EV_14 --fl-ev-15 $FL_EV_15 \
        --fl-ev-16 $FL_EV_16 --fl-ev-17 $FL_EV_17 \
        > /dev/null
    echo "=== Test passed ==="
}

run_range() {
    START=${1:-1}
    END=${2:-0}

    mkdir -p "$RESULTS_DIR"

    TOTAL=$(wc -l < "$PATTERNS_FILE")
    if [ "$END" -eq 0 ] || [ "$END" -gt "$TOTAL" ]; then
        END=$TOTAL
    fi

    # Extract pattern range
    RANGE_FILE="$RESULTS_DIR/patterns_${START}_${END}.txt"
    sed -n "${START},${END}p" "$PATTERNS_FILE" > "$RANGE_FILE"
    RANGE_COUNT=$(wc -l < "$RANGE_FILE")

    OUTFILE="$RESULTS_DIR/t3_exhaustive_${START}_${END}.jsonl"

    echo "=== Running T3 Exhaustive Evaluation ==="
    echo "  Patterns: $RANGE_COUNT (lines $START-$END)"
    echo "  N1=$N1, N2=$N2, N3=$N3, T4=exhaustive"
    echo "  Output: $OUTFILE"
    echo "  Cores: $(nproc)"
    echo "  Start: $(date)"

    RAYON_NUM_THREADS=$(nproc) "$BINARY" t3eval \
        --exhaustive --n1 $N1 --n2 $N2 --n3 $N3 \
        --seed $SEED --patterns "$RANGE_FILE" \
        --bust-penalty $BUST_PENALTY \
        --fl-ev-14 $FL_EV_14 --fl-ev-15 $FL_EV_15 \
        --fl-ev-16 $FL_EV_16 --fl-ev-17 $FL_EV_17 \
        > "$OUTFILE"

    echo "=== Done: $(date) ==="
    RECORDS=$(wc -l < "$OUTFILE")
    echo "  Output: $OUTFILE ($RECORDS lines)"
    SIZE=$(du -h "$OUTFILE" | cut -f1)
    echo "  Size: $SIZE"
}

case "${1:-}" in
    setup)  setup ;;
    test)   test_run ;;
    run)    run_range "${2:-1}" "${3:-0}" ;;
    *)
        echo "Usage: $0 {setup|test|run START END}"
        echo "  setup: Install Rust and compile"
        echo "  test:  Smoke test (2 patterns)"
        echo "  run 1 1000: Process patterns 1-1000"
        exit 1
        ;;
esac
