#!/bin/bash
set -e
cd /home/Owner/ofc-pineapple
source ~/venv/bin/activate

echo "Starting self-play loop at $(date)"
echo "CWD: $(pwd)"
echo "Python: $(which python)"

export PYTHONUNBUFFERED=1

python ai/run_selfplay_loop.py \
  --iterations 5 \
  --start-iter 7 \
  --games 500 \
  --rollouts 50 \
  --top-k 30 \
  --workers 20 \
  --epochs 100 \
  --eval-games 50 \
  --eval-rollouts 50 \
  --base-model ai/models/selfplay_iter6/bc_policy_best.pt \
  --include \
    data/ryo_aug_t18.jsonl \
    data/selfplay_v3_500.jsonl \
    data/selfplay_iter1.jsonl \
    data/selfplay_iter2.jsonl \
    data/selfplay_iter3.jsonl \
    data/selfplay_iter4.jsonl \
    data/selfplay_iter5.jsonl \
    data/selfplay_iter6.jsonl

echo "Done at $(date)"
