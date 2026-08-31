#!/bin/bash
# Full setup + benchmark on fresh GCP VM
# Installs Rust, clones, builds, runs matrix, uploads results

OUTDIR="/tmp/bench_matrix"
mkdir -p "$OUTDIR"

HANDS=10
SEED=42

echo "=== [1/3] Build cfr_solver ==="
cd /tmp

# Install Rust
if ! command -v cargo &> /dev/null; then
    echo "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
source "$HOME/.cargo/env"

# Clone and build
if [ ! -d "ofc-pineapple" ]; then
    echo "Cloning repository..."
    git clone https://github.com/giovinco0807/pineapple.git ofc-pineapple
fi
cd ofc-pineapple/ai/rust_solver
echo "Building (release)..."
cargo build --release -p cfr_solver 2>&1 | tail -3
EXE="./target/release/cfr_solver"
echo "Binary ready: $(ls -lh $EXE | awk '{print $5}')"

echo ""
echo "=== [2/3] Running Benchmark Matrix ==="
echo "Hands: $HANDS | Seed: $SEED | Cores: $(nproc)"
echo ""

# Reduced matrix: skip s1000/s2000 (too slow), focus on practical configs
CONFIGS=(
    "3,2,1 30 n321_s30"
    "3,2,1 50 n321_s50"
    "3,2,1 100 n321_s100"
    "3,2,1 200 n321_s200"
    "3,2,1 500 n321_s500"
    "5,3,2 30 n532_s30"
    "5,3,2 50 n532_s50"
    "5,3,2 100 n532_s100"
    "5,3,2 200 n532_s200"
    "8,5,3 30 n853_s30"
    "8,5,3 50 n853_s50"
    "3,2,1 1000 n321_s1000"
)

TIMING_FILE="$OUTDIR/timing.csv"
echo "config,nesting,samples,hands,total_seconds,sec_per_hand" > "$TIMING_FILE"

for cfg in "${CONFIGS[@]}"; do
    read -r nesting samples label <<< "$cfg"
    outfile="$OUTDIR/${label}.jsonl"

    # Skip if already done
    if [ -f "$outfile" ]; then
        lines=$(wc -l < "$outfile")
        if [ "$lines" -ge 7 ]; then
            echo "--- SKIP $label (already done: $lines lines) ---"
            continue
        else
            echo "--- REDO $label (incomplete: $lines lines) ---"
            rm -f "$outfile"
        fi
    fi

    echo "============================================================"
    echo "Config: $label  |  nesting=[$nesting]  samples=$samples"
    echo "------------------------------------------------------------"

    START_SEC=$(date +%s)
    $EXE t0-batch --hands $HANDS --samples $samples --nesting $nesting --output "$outfile" --seed $SEED 2>&1
    END_SEC=$(date +%s)

    ELAPSED=$((END_SEC - START_SEC))
    PER_HAND=$((ELAPSED / HANDS))

    echo "  Total: ${ELAPSED}s  |  Per hand: ${PER_HAND}s"
    echo "${label},${nesting},${samples},${HANDS},${ELAPSED},${PER_HAND}" >> "$TIMING_FILE"

    echo ""
done

echo ""
echo "=== [3/3] Upload results ==="
gcloud storage cp -r "$OUTDIR" gs://ofc-solver-485418/bench_matrix/ 2>&1
echo ""
echo "=== COMPLETE ==="
echo ""
cat "$TIMING_FILE"
