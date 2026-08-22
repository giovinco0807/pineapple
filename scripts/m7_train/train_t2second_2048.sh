#!/usr/bin/env bash
# T2 second seat on the 2048-particle corpus: the shipping retrain, then the
# scaling curve that says whether 25,000 positions is now enough.
#
# The ship model warm-starts from the incumbent (t2_model_v1) with first-layer
# renormalisation onto the new corpus's standardization -- the same procedure
# the T3 v7 models shipped under, reproduced and verified this morning.
#
# The curve is COLD on nested prefixes, seeds 997,990,031-033 reused from this
# morning's 128-particle curve on purpose: same seeds across the two corpora
# makes the two curves paired rather than independent draws, which is the whole
# point of measuring it again.
set -u
PY=/home/wner/ofc-t0-cuda-venv/bin/python3
CORPUS=/home/wner/ofc-m7/t2_2048/second/features_rs
INCUMBENT=/home/wner/ofc-t2/model_v1/model.pt
SHIP=/home/wner/ofc-m7/t2_2048/second/ship
CURVE=/home/wner/ofc-m7/t2_2048/second/curve
mkdir -p "$SHIP" "$CURVE"
cd /tmp || exit 1

PROV="M7 T2 relabel, second seat: 25,000 positions, samples 2048, fl_ev 9.6, \
worker plan 8800ab83, T4 v6 763b77a0, T3 first v2 2d94b3e3, T3 second v3 75119ebe, \
engine e17aa39f, hand seed base 947,000,000; warm start from t2_model_v1"

echo "=== ship retrain (warm start) $(date -u +%H:%M:%SZ)"
"$PY" /tmp/train_ship.py \
  --corpus "$CORPUS" --width 27 --street T2 --seat second \
  --warm-start "$INCUMBENT" --seed 997990041 \
  --epochs 120 --batch 512 \
  --label-provenance "$PROV" \
  --out "$SHIP/t2second_2048p" 2>&1 | tail -6

echo "=== scaling curve (cold, nested prefixes) $(date -u +%H:%M:%SZ)"
for n in 4000 8000 12000 18000 25000; do
  for seed in 997990031 997990032 997990033; do
    f="$CURVE/t2second2048_${n}_${seed}"
    [ -f "$f.json" ] && { echo "SKIP $n $seed"; continue; }
    "$PY" /tmp/train_curve_2048.py --street t2second2048 --arm curve_cold --seed "$seed" \
      --positions "$n" --epochs 120 --batch 512 --out "$f" 2>&1 | tail -1
  done
done
echo "=== done $(date -u +%H:%M:%SZ)"
