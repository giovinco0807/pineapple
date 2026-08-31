#!/usr/bin/env bash
# One mirror match between two arms of models_verdict.tar.gz.
#
#   usage: launch_verdict.sh <run-id> <arm-a> <arm-b> <deals> <seed> [shards]
#   e.g.   launch_verdict.sh v1 union gen2 12000 86000001 20
#
# --hands counts DEALS, and a mirror deal is played twice with the seats
# traded, so a 12,000-deal set is 24,000 hands of engine time.  Common random
# numbers make identical arms settle to exactly zero -- that is the acceptance
# test, and it is worth running once whenever the bundle changes.
#
# Rankers and own-hand fallbacks are read from arm A's directory for both
# sides on purpose: every arm carries the same copies, so pointing at one
# keeps the match a comparison of evaluators alone.
set -uo pipefail
cd "C:/Users/Owner/.gemini/antigravity/scratch/ofc-pineapple"
RUN=${1:?run-id}; A=${2:?arm-a}; B=${3:?arm-b}
DEALS=${4:-12000}; SEED=${5:-86000001}; SHARDS=${6:-20}
M="models_verdict"
STREETS="t0_bb t0_btn t1_bb t1_btn t2_bb t2_btn t3_bb t3_btn"
bins() { local arm=$1 kind=$2; local out=""; for s in $STREETS; do
  out="$out,$M/$arm/$kind/$s.bin"; done; echo "${out:1}"; }
HA=$(bins "$A" hu); HB=$(bins "$B" hu); RK=$(bins "$A" rankers)
OW="$M/$A/own_lap4/t0.bin,$M/$A/own_lap4/t1.bin,$M/$A/own_lap4/t2.bin"
PER=$(( (DEALS + SHARDS - 1) / SHARDS ))

ARGS="--hu-mirror --hu-a-models $HA --hu-b-models $HB --hu-a-rankers $RK --hu-b-rankers $RK --hu-topk 4 --arm-a-own $OW --arm-b-own $OW --serve-joint-samples 200 --serve-joint-samples-b 200"
echo "$RUN: $A vs $B, $DEALS deals / $SHARDS shards = $PER each, seed $SEED"
python -m ai.tutor.fleet.launch_hu_match_fleet --run-id "$RUN" --job mirror \
  --hands "$DEALS" --shards "$SHARDS" --models-object models_verdict.tar.gz \
  --src hu_src_20260820b.tar.gz --binstamp 20260820b --seed-base "$SEED" \
  --match-args "$ARGS" | tail -1
# A match fleet has no reaper of its own; finished workers hold region quota
# and starve the shards still waiting.  This cost ninety minutes on tr3.
nohup python -m ai.tutor.fleet.reap_hu_fleet --run-id "$RUN" --jobs mirror \
  --roots "$DEALS" --shards "$SHARDS" --poll-seconds 120 --stall-minutes 60 \
  > /d/ofc_data/hu/${RUN}_reaper.log 2>&1 &
echo "reaper armed for $RUN"
