# HU T2 Stage8b Safe Override 20k MC512 Training

This is a broad teacher-cache training artifact, not a production candidate.

- model: `models\hu_turn2_stage8b_safe_lcb196_20k_reference_override_cached_rank_wide.pt`
- cache: `D:\ofc-pineapple-storage\regular-ofc-pineapple\feature_cache\hu_t2_stage8_20k_mc512`
- device: `cuda`
- gate label source: `stage8b_labels_csv`
- epochs ran: `30`
- best epoch: `29`
- train/val/test states: `13994` / `3003` / `3003`

## Holdout Metrics

- val EV MAE / avg_regret / top3: `1.2146` / `0.3548` / `0.8538`
- test EV MAE / avg_regret / top3: `1.2434` / `0.3748` / `0.8398`
- test delta baseline MAE: `1.5070`
- test pairwise ranking accuracy: `0.8386`
- test gate accuracy pos/neg: `0.7704`

## Threshold Sweep

- configs with positive teacher gain and at least one override: `0`
- This sweep is teacher-holdout only; do not use it as production evidence.
- `reference_margin_raw` is a T2 baseline/reference score margin and is not comparable to the T3 Stage7 `r10` reference gate.

## Next Step

GO to a larger 20k-50k MC512 broad pass only after the same pipeline is repeated with a larger holdout and then seat-swap validation. This pilot is enough to validate the cache/training pipeline, not production adoption.
