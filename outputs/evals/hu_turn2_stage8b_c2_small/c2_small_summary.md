# HU T2 Stage8b C2-small Validation

This is validation-only. It is not production approval, not P2 fixed, and does not authorize T1 or 50k teacher generation.

## Inputs

- T3 continuation: `Stage7_candidate_A m5_r10` fixed (`hu_turn3_min_margin=5.0`, `hu_turn3_reference_min_margin=10.0`)
- T2 model: `models/hu_turn2_stage8b_safe_lcb196_20k_reference_override_cached_rank_wide.pt`
- paired seat-swap: `3 seeds x 300 paired seeds`
- seed stride: `1000000`

## Seat-swap Results

- primary `m2.75_p0.95_k1`: EV/hand `-0.0201`, CI `[-0.0914, 0.0512]`, overrides `36/1800`
- looser margin `m2.50_p0.95_k1`: EV/hand `0.0017`, CI `[-0.1361, 0.1394]`, overrides `44/1800`
- looser rank `m2.75_p0.95_k3`: EV/hand `0.0040`, CI `[-0.0643, 0.0723]`, overrides `46/1800`
- looser probability `m2.75_p0.90_k1`: EV/hand `-0.0201`, CI `[-0.0914, 0.0512]`, overrides `37/1800`
- old Stage8 control `m2.5_g0.9`: EV/hand `0.0173`, CI `[-0.0035, 0.0382]`, overrides `6/1800`
- position probe `m2.75_p0.95_seatfirst_k3`: EV/hand `0.0574`, CI `[0.0044, 0.1105]`, overrides `17/1800`

## Interpretation

- Unrestricted primary `k1` did not reproduce a positive trend.
- `k3` improved the aggregate to roughly breakeven but still had a negative second-position split.
- First-only `m2.75_p0.95_k3` was the only clear positive C2-small candidate in this run.
- Old Stage8 control remained positive but fired very rarely, so it is not a sufficient production baseline by itself.

## Go / No-Go

- Stage8b C2-small execution: `Pass`
- C3 larger seat-swap: `Go only for m2.75_p0.95_seatfirst_k3`
- unrestricted `m2.75_p0.95_k1`: `No-Go for C3 as configured`
- 50k teacher: `No-Go`
- T1: `No-Go`
- production / P2 fixed: `No-Go`
