#!/bin/bash
# GCP T2 Deep Evaluation Script (with Domain Pruning)
# Runs t2eval with deep T3+T4 exhaustive evaluation on assigned pattern range.
#
# Usage:
#   ./gcp_t2_deep.sh setup          # Install Rust, compile
#   ./gcp_t2_deep.sh run START END  # Run patterns [START, END)
#   ./gcp_t2_deep.sh test           # Smoke test (1 pattern)

set -e

WORK_DIR="$HOME/expectimax"
PATTERNS_FILE="$WORK_DIR/canonical_patterns.txt"
BINARY="$WORK_DIR/rust_solver/target/release/backward"
RESULTS_DIR="$WORK_DIR/t2_results"

# T2 Deep Eval params
N1=30        # T1 sampling (for quick T0 best pick)
N2=50        # T2 deal samples
N3_DEEP=20   # T3 deep samples per T2 action
TOP_K=4      # Top-K pre-filter
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
    echo "=== Smoke test: 1 pattern, t2eval ==="
    echo "Ah,Kh,Qd,7c,3s" | RAYON_NUM_THREADS=$(nproc) "$BINARY" t2eval \
        --exhaustive --n1 3 --n2 3 --n3-deep 3 --top-k 4 \
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

    OUTFILE="$RESULTS_DIR/t2_deep_${START}_${END}.jsonl"
    LOGFILE="$RESULTS_DIR/progress_${START}_${END}.log"

    echo "=== Running T2 Deep Evaluation ===" | tee "$LOGFILE"
    echo "  Patterns: $RANGE_COUNT (lines $START-$END)" | tee -a "$LOGFILE"
    echo "  N1=$N1, N2=$N2, N3_deep=$N3_DEEP, top_K=$TOP_K, T4=exhaustive" | tee -a "$LOGFILE"
    echo "  Output: $OUTFILE" | tee -a "$LOGFILE"
    echo "  Cores: $(nproc)" | tee -a "$LOGFILE"
    echo "  Start: $(date)" | tee -a "$LOGFILE"
    echo "" | tee -a "$LOGFILE"

    RUN_START=$(date +%s)
    COUNT=0
    TOTAL_ELAPSED=0

    # Process patterns one by one for detailed logging
    while IFS= read -r pattern; do
        COUNT=$((COUNT + 1))

        # Skip if already done (resumable)
        HASH=$(echo "$pattern" | md5sum | cut -c1-12)
        DONE_MARKER="$RESULTS_DIR/.done_${HASH}"
        if [ -f "$DONE_MARKER" ]; then
            echo "[$COUNT/$RANGE_COUNT] SKIP $pattern (already done)" | tee -a "$LOGFILE"
            continue
        fi

        PAT_START=$(date +%s)

        # Run t2eval for this single pattern
        echo "$pattern" | RAYON_NUM_THREADS=$(nproc) "$BINARY" t2eval \
            --exhaustive --n1 $N1 --n2 $N2 --n3-deep $N3_DEEP --top-k $TOP_K \
            --seed $((SEED + COUNT)) \
            --bust-penalty $BUST_PENALTY \
            --fl-ev-14 $FL_EV_14 --fl-ev-15 $FL_EV_15 \
            --fl-ev-16 $FL_EV_16 --fl-ev-17 $FL_EV_17 \
            >> "$OUTFILE" 2>/dev/null

        PAT_END=$(date +%s)
        PAT_ELAPSED=$((PAT_END - PAT_START))
        TOTAL_ELAPSED=$((PAT_END - RUN_START))

        # Compute ETA
        AVG_SECS=$((TOTAL_ELAPSED / COUNT))
        REMAINING=$(( (RANGE_COUNT - COUNT) * AVG_SECS ))
        ETA_HOURS=$((REMAINING / 3600))
        ETA_MINS=$(( (REMAINING % 3600) / 60 ))

        touch "$DONE_MARKER"

        echo "[$COUNT/$RANGE_COUNT] ${PAT_ELAPSED}s  avg=${AVG_SECS}s  ETA=${ETA_HOURS}h${ETA_MINS}m  $pattern" | tee -a "$LOGFILE"

    done < "$RANGE_FILE"

    RUN_END=$(date +%s)
    TOTAL_TIME=$((RUN_END - RUN_START))
    TOTAL_HOURS=$((TOTAL_TIME / 3600))
    TOTAL_MINS=$(( (TOTAL_TIME % 3600) / 60 ))

    echo "" | tee -a "$LOGFILE"
    echo "=== Done: $(date) ===" | tee -a "$LOGFILE"
    echo "  Total time: ${TOTAL_HOURS}h${TOTAL_MINS}m (${TOTAL_TIME}s)" | tee -a "$LOGFILE"
    echo "  Patterns processed: $COUNT" | tee -a "$LOGFILE"
    echo "  Avg per pattern: ${AVG_SECS}s" | tee -a "$LOGFILE"
    SIZE=$(du -h "$OUTFILE" | cut -f1)
    echo "  Output: $OUTFILE ($SIZE)" | tee -a "$LOGFILE"
}

case "${1:-}" in
    setup)  setup ;;
    test)   test_run ;;
    run)    run_range "${2:-1}" "${3:-0}" ;;
    *)
        echo "Usage: $0 {setup|test|run START END}"
        echo "  setup: Install Rust and compile"
        echo "  test:  Smoke test (1 pattern)"
        echo "  run 1 1000: Process patterns 1-1000"
        exit 1
        ;;
esac
