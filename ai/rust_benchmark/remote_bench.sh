#!/bin/bash
# Remote benchmark: single process using all CPU cores via rayon
# Usage: bash remote_bench.sh <bust_penalty> <fl_ev_scale> <total_games> <seed>
#
# Output format (last line):
#   RESULT score=+12.34 bust=5.6 fl=12.3 win=78.9

BP=${1:?bust_penalty required}
FL_S=${2:?fl_ev_scale required}
TOTAL_GAMES=${3:?total_games required}
SEED=${4:?seed required}

cd /home/Owner/rust_benchmark

# Kill any leftover benchmark processes
pkill -9 ofc_benchmark 2>/dev/null
sleep 1

echo "Running $TOTAL_GAMES games (single process, all cores)..."

./target/release/ofc_benchmark \
    --games "$TOTAL_GAMES" --seed "$SEED" \
    --mcts-sims 400 --rollouts 250 \
    --bust-penalty "$BP" --fl-ev-scale "$FL_S" \
    --mode benchmark > /tmp/bench_single.txt 2>&1

echo "Done."

# Parse and output RESULT line
python3 -c "
import re

text = open('/tmp/bench_single.txt').read()

score = bust = fl = win = -1

m = re.search(r'Total score:\s+([+-]?\d+\.?\d*)/hand', text)
if m:
    score = float(m.group(1))

m = re.search(r'Bust.*?\[(\d+\.?\d*)%', text)
if m:
    bust = float(m.group(1))
else:
    m = re.search(r'Bust rate:\s*(\d+\.?\d*)%', text)
    if m:
        bust = float(m.group(1))

m = re.search(r'FL entry.*?(\d+\.?\d*)%', text)
if m:
    fl = float(m.group(1))

m = re.search(r'Win rate:\s*(\d+\.?\d*)%', text)
if m:
    win = float(m.group(1))

if score != -1:
    print(f'RESULT score={score:+.2f} bust={bust:.1f} fl={fl:.1f} win={win:.1f}')
else:
    print('RESULT FAILED')
"
