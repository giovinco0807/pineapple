# HU T1 Stage18 P1 Rollout Note

Status: fixed first-seat T1 P1 for downstream experiments. Explicit profile
selection is required; `current` is unchanged.

## Enable

Use profile `stage18_p1`. It locks the following runtime:

```text
k5/mc8/d3/confirm32/cse1.5/pd1.5/seat=first/safe0.7
```

The profile loads:

- `models/hu_turn1_stage18_first1801_mc32_hgb_abs_aug3.pkl`
- `models/hu_turn1_stage18_highmc_safe_selector_a_meta_only.pkl`

T2 remains `stage9f_p2`; T3 remains `stage7_m5_r10`.

## Safety

The Stage18 action is only a candidate. The Stage9f P2 T1 baseline action is
kept unless every gate passes. The policy falls back when the seat is second,
either T1 model cannot load, feature/prediction output is invalid, the candidate
is illegal, or any runtime threshold is unmet.

This is not full replacement. Confirm-MC delta is a gate diagnostic and must
not be reported as realized performance.

## Rollback

Select profile `stage9f_p2`, or preset `stage18_p1_off` from
`configs/hu_turn1_stage18_p1_presets.json`.

## Validation

C6 evaluated 500,000 T1 decisions over 250,000 fresh paired seeds. It produced
340 valid fires, realized per-fire delta `+3.5103` with 95% CI
`[+1.9683, +5.0523]`, EV/decision `+0.002389` with positive CI lower bound,
and zero non-fired cancellation errors.
