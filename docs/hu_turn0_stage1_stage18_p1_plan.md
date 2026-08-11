# HU T0 Stage1 Plan With Fixed Stage18 P1

Date: 2026-07-11 JST

T0 is not yet complete. The current opening policy is
`models/opening_stage7_torch_wide.pt`, trained from 20k self-board rows by
bootstrapping into the old T1 Stage6 model. It does not encode the visible
opponent opening board and was not trained against the accepted HU continuation
chain.

## Fixed Downstream Chain

- T1: `stage18_p1`
- T2: `stage9f_p2`
- T3: `stage7_m5_r10`
- FL EV: `10.227020614683454`

These remain fixed while T0 is built.

## T0 Teacher

`ofc_regular.hu_turn0_teacher_pilot` evaluates all 232 legal opening
placements. Every action in one state uses the same sampled future card
sequences. Continuation decision seeds exclude the candidate action index, so
the paired action comparison preserves common-random-number coupling.

The first-seat state samples the unknown opponent opening five cards inside
each future. The second-seat state includes the already visible first-seat
opening board. Both paths retain hidden-discard visibility and save replay
metadata.

## Measured Speed

- first seat, all 232, MC1: `202.06s/state`
- second seat, all 232, MC1: `66.12s/state`
- dominant cost: fixed T2 continuation
- estimated both-seat average, all 232, MC4: `536.37s/state`

The first broad pilot is therefore distributed across Spot VMs rather than run
locally.

## Initial Pilot

- states: `100`
- first/second: `50 / 50`
- actions: all `232`
- outer MC: `4`
- shards: `50` of two states
- workers: `20` Spot `c4-highmem-2`
- run: `regular-hu-t0-stage1-all232-mc4-100-20260711-001`

The original `e2-highcpu-4` attempt produced no completed shard after more
than one hour. The equivalent C4 benchmark completed two records in `913.37`
seconds (`456.68` seconds/state), with all legality, finite-value, common-future,
and replay-readiness checks passing. No incomplete e2 output is used.

This pilot is candidate-generation data, not P0 acceptance evidence. After it
passes completeness and replay checks, train an initial HU-aware T0 candidate,
audit TopK coverage on heldout all-action states, then move higher MC to the
covered candidate subset. Final acceptance still requires fresh paired
seat-swap using realized fired counterfactual deltas.
