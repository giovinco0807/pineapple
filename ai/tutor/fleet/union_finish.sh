#!/usr/bin/env bash
# Build the three arms the verdict needs, once lap 2's T0 lands.
#
# Held-out regret cannot pick between lap2-only and the union, because the two
# disagree about which ruler counts.  Scored on gen-2's own states -- the ones
# the engine actually visits -- lap2-only wins four streets of six; scored on
# a corpus half made of gen-1's states, the union wins all six.  Both readings
# are arithmetically correct and they point opposite ways.
#
# So all three go in one bundle and the mirror decides:
#   gen2  = lap1-only            (the reigning champion)
#   lap2  = lap2-only            (specialist on the current distribution)
#   union = lap1 fit + lap2 fit  (blunter, but at home in both)
#
# Rankers and own-hand fallbacks are identical in every arm, so the match
# measures the evaluators and nothing else.
set -uo pipefail
cd "C:/Users/Owner/.gemini/antigravity/scratch/ofc-pineapple"
D=/d/ofc_data/hu
SP="/c/Users/Owner/AppData/Local/Temp/claude/C--Users-Owner--gemini-antigravity-scratch-ofc-pineapple/aaa9fafe-db52-41b7-91a1-9392858aee59/scratchpad"
U=$D/bundle_union
STREETS="t0_bb t0_btn t1_bb t1_btn t2_bb t2_btn t3_bb t3_btn"
stamp() { echo "=== $* :: $(date +%T) ==="; }

stamp "wait for the lap-2 chain to finish T0"
until grep -q "LAP 2 DONE" $D/lap2_resume.log; do sleep 120; done
stamp "chain done"

stamp "union fit for T0"
python "$SP/build_both.py" t0_bb t0_btn
for s in t0_bb t0_btn; do
  stamp "train $s both"
  python -m ai.tutor.train_t4_first_evaluator --data-dir $D/${s}_enc_both \
    --out-dir $D/${s}_both_s20260815 --select-on regret --skip-test \
    --seed 20260815 --hidden "512,256,128" > /dev/null 2>&1
done

stamp "verdict: pooled ruler (all eight streets)"
python "$SP/street_verdict.py" $STREETS
stamp "verdict: per-distribution (which ruler you believe changes the answer)"
python "$SP/split_verdict.py" $STREETS

stamp "export bins"
mkdir -p $U/union_bins $U/lap2_bins
for s in $STREETS; do
  python -m ai.tutor.export_t4_first_evaluator \
    --model $D/${s}_both_s20260815/evaluator_best.pt --out $U/union_bins/$s.bin > /dev/null
  python -m ai.tutor.export_t4_first_evaluator \
    --model $D/${s}_lap2_s20260815/evaluator_best.pt --out $U/lap2_bins/$s.bin > /dev/null
done
echo "exported $(ls $U/union_bins/*.bin | wc -l) union + $(ls $U/lap2_bins/*.bin | wc -l) lap2"

stamp "assemble models_verdict.tar.gz"
rm -rf $U/models_verdict
mkdir -p $U/models_verdict
for arm in gen2 lap2 union; do cp -r $U/models_t3ab/a $U/models_verdict/$arm; done
for s in $STREETS; do
  cp $U/union_bins/$s.bin $U/models_verdict/union/hu/$s.bin
  cp $U/lap2_bins/$s.bin  $U/models_verdict/lap2/hu/$s.bin
done
# gen2 keeps models_t3ab/a untouched -- it IS the reigning bundle.
( cd $U && tar -czf models_verdict.tar.gz models_verdict )
gcloud storage cp $U/models_verdict.tar.gz \
  gs://pokerhu-ofc-solver-485418-training/hu-street/artifacts/ 2>&1 | tail -1
echo "arms: $(ls $U/models_verdict)"
sha256sum $U/models_verdict/gen2/hu/t1_bb.bin $U/models_verdict/lap2/hu/t1_bb.bin \
          $U/models_verdict/union/hu/t1_bb.bin | cut -c1-20
stamp "BUNDLE READY -- three arms, identical rankers"
