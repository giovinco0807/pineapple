# HU T0 Stage19 P0 Rollout Note

Status: fixed first-seat P0 selective override. Explicit profile selection is
required; `current` is unchanged.

## Enable

Use profile `stage19_p0`. It locks:

```text
topk=60 / first_margin=0.5 / first_safe_selector=0.6 / seat=first
```

Downstream chain:

```text
T0 stage19_p0 -> T1 stage18_p1 -> T2 stage9f_p2 -> T3 stage7_m5_r10
```

## Safety

The Stage19 action is only a candidate. The Stage18 opening action remains the
default and fallback. Model-load, feature, prediction, non-finite output,
selector, threshold, seat, or legality failure keeps the fallback action.

Runtime T0 JSONL logs include the boards, dealt/dead cards, action indices,
candidate and fallback actions, predicted margin, selector score and threshold,
legality result, final action, no-override reason, latency, runtime profile, and
fixed continuation identifiers.

## Rollback

Select `stage18_p1`, or preset `stage19_p0_off` from
`configs/hu_turn0_stage19_p0_presets.json`.

## Evidence

The preregistered 100,000-paired-seed holdout produced `+0.017303/hand` with
95% CI `[+0.008292, +0.026314]`, and `+0.9236` per fire with 95% CI
`[+0.4434, +1.4037]`. Non-fired counterfactual differences were zero. The
independent top-30 MC512 tail audit had mean delta `+0.8255` and `13.33%`
negative labels.
