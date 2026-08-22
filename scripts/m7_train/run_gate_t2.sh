#!/usr/bin/env bash
# The M7 T2 promotion gates: one mirrored duplicate match per street-seat.
#
#   first  seat  swaps t2_first   seed block 948,000,000
#   second seat  swaps t2_second  seed block 949,000,000
#
# 6 shards x 3,334 deals = 20,004 deals per seat, matching the T3 v7 gates so
# the two generations' numbers are read on the same scale. Both seats run at
# once: 12 processes on 16 cores, and the engine call is single-threaded.
#
# Fresh seed blocks, verified unspent. T3's gates spent 945/946 million.
set -u
REPO=/mnt/c/Users/Owner/.gemini/antigravity/scratch/ofc-pineapple/regular-ofc-pineapple
export PYTHONPATH=/home/wner/ofc-m7/t2probe/pylib:$REPO/src
export OFC_FL_EV_CONFIG=configs/fl_ev_regular_v4_selfplay.json
PY=/home/wner/.local/bin/python3.11
OUT=/home/wner/ofc-m7/gate_t2_2048
SHIP=/home/wner/ofc-m7/t2_2048
DEALS=3334
SHARDS=6
cd "$REPO" || exit 1
mkdir -p "$OUT/first" "$OUT/second"

launch() {  # seat stem model seed_base dir
  local seat=$1 stem=$2 model=$3 base=$4 dir=$5
  for i in $(seq 0 $((SHARDS - 1))); do
    local start=$((i * DEALS))
    "$PY" /tmp/gate_match.py \
      --deals "$DEALS" --start "$start" --seed-base "$base" \
      --swap-stem "$stem" --swap-path "$model" \
      --out "$dir/shard_0$i.json" \
      > "$dir/shard_0$i.log" 2>&1 &
  done
  echo "$seat: launched $SHARDS shards x $DEALS deals from seed $base"
}

launch first t2_first \
  "$SHIP/first/ship/weights/t2first_model_v2.bin" 948000000 "$OUT/first"
launch second t2_second \
  "$SHIP/second/ship/weights/t2_model_v2.bin" 949000000 "$OUT/second"

wait
echo "GATE RUNS FINISHED $(date -u +%H:%M:%SZ)"
