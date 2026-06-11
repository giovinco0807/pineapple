# HU Turn2 Stage8b Training Plan

Stage8b should not be a wider threshold search over the current proxy. It should add a deployable confidence head trained to predict safe override labels derived from teacher LCB and selected high-MC replay.

## Model

- Keep existing EV, delta, and ranking/listwise heads.
- Add `safe_override_head` for `safe_lcb196_label` or `safe_lcb164_label`.
- Optionally add a separate `hard_negative_head` or use a high-weight negative term in the safe head.

## Loss

- Huber losses for EV and delta stay in place.
- Pairwise/listwise ranking loss stays in place.
- Add BCE or focal loss for `safe_override_head`.
- Weight hard negatives higher than ordinary negatives.
- Give gray rows low or zero gate-loss weight.

## Runtime Gate

Candidate gate for validation only:

```text
override if
  predicted_delta >= m
  and safe_override_probability >= p
  and candidate_rank <= k
  and optional predicted_EV_margin >= pm
```

Recommended grid:

- `m`: 2.0, 2.25, 2.5, 2.75
- `p`: 0.80, 0.85, 0.90, 0.95
- `k`: 1, 2, 3, 5
- `pm`: none, 0.25, 0.50

## Evaluation Gates

1. Holdout teacher cache: low FP, positive avg gain, p95/p99 loss controlled.
2. Proxy-vs-oracle overlap: safe head improves recall without losing precision.
3. C3-style seat-swap: non-overlapping seeds, `--seed-stride`, T3 Stage7 m5_r10 fixed.
4. Only after a positive larger seat-swap should selected MC4096/8192 and C4 be used for promotion evidence.

50k teacher, T1, and production remain No-Go until Stage8b passes these gates.
