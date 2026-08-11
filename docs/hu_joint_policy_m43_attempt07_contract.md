# M4.3 Attempt07 development contract

Date: 2026-07-14

## Decision boundary

Attempt07 is frozen as a development search-architecture comparison, not as a
policy promotion. Attempt06 remains closed as a search-quality No-Go; its
opened result may be used only as postmortem evidence that MC8 was too noisy to
be the final selector. Attempt07 does not retry an Attempt06 threshold or seed.

The only authorized experiment in this contract is 100 new balanced T1-second
development roots: 20 roots against each of `stage19_p0`, `stage9f_p2`,
`stage7_m5_r10`, `stage3_baseline`, and `random_exact_final`. The fixed T1
baseline is `stage18_p1`; first-seat behavior is unchanged. Fit, threshold
selection, runtime activation, full replacement, and any change to `current`
remain prohibited.

Development execution is also part of the frozen experiment: one root per
restartable shard, batched child selection with four native threads, and zero
opening-lookahead samples. Root and baseline policies are seeded from the
per-root hand seed through the named profile-seed function; continuation
policies use the per-root child seed plus seat offset. Preemption may only
recompute the identical root under the identical contract and may never
substitute another root or seed after a result exists.

## Frozen four-stage search

Every root starts with eight unique legal non-baseline LambdaRank proposals and
the explicit baseline exactly once. Complete legal-action enumeration,
ActionKey mapping, illegal-action masking, and infoset-safe inputs are
mandatory. Opponent private discards and opponent identity are forbidden model
or search inputs.

The search stages are:

1. `S8`: score rank8 plus baseline using eight common-random futures in the
   screen RNG domain. S8 cannot fire directly.
2. `K3`: retain the three highest-S8 unique legal non-baseline actions, tied by
   ActionKey, then add the baseline exactly once.
3. `R32` or `R64`: rerank K3 plus baseline in an independent RNG domain. R32 is
   the exact prefix of R64, so the two budgets do not create two independent
   searches. An exact tie selects the explicit baseline; non-baseline ties use
   ActionKey order.
4. `V64` or `V128`: independently confirm the locked rerank winner against the
   baseline. V64 is the exact prefix of V128. A non-baseline action is eligible
   only when its paired delta has mean strictly above zero, p05 at least -25,
   p01 at least -40, and minimum at least -50. Otherwise the arm falls back to
   the exact baseline.

After every arm has locked its output, disjoint `A128` common-random futures
score all rank8 actions plus baseline. A128 is retained for development arm
comparison and diagnostics only. It cannot rerank, veto, or otherwise change
an arm's output, and its values are not realized match EV.

Every hypothetical continuation is also frozen: T2 uses the explicit
`stage9f_p2` policy, T3 uses the M3 Rust evaluator with candidate/evaluation/
downstream budgets all equal to one, and hypothetical T4 uses the M3 Rust
evaluator with candidate/evaluation budgets equal to one. The real live T4
selector remains the unchanged exact solver.

The finite arm set is exactly:

| Arm | Rerank | Veto | Total action-futures/root |
|---|---:|---:|---:|
| `r32_v64` | 32 | 64 | 1,480 |
| `r64_v64` | 64 | 64 | 1,608 |
| `r32_v128` | 32 | 128 | 1,608 |
| `r64_v128` | 64 | 128 | 1,736 |

The totals include the common S8 cost over nine actions, rerank over K3 plus
baseline, veto over the locked winner plus baseline, and A128 over all nine
actions. Shared nested prefixes must be computed once and reused.

## Development Go/No-Go and winner

An arm is eligible only if every A128 gate passes:

- at least 20 fires overall and at least two fires for every profile;
- mean paired delta per state and per fire both strictly above zero;
- false-positive rate at most 0.50, where a fired action whose A128 paired mean
  is at most zero is a false positive;
- for each fired root, derive loss from A128 p05, p01, and minimum using NumPy
  linear quantiles, then require the maximum across fired roots to be at most
  25, 40, and 50 respectively;
- zero action-mapping, RNG-domain, and hidden-information violations; and
- exact fallback to the baseline ActionKey for every non-fire.

This offline teacher comparison cannot honestly prove runtime trajectory
cancellation because no distilled runtime policy exists yet. Exact full-
trajectory counterfactual cancellation is therefore deferred, not waived: it
is a mandatory gate in the later fresh runtime acceptance evaluation.

At most one winner is selected from eligible arms. The frozen lexicographic
order is higher mean delta per state, lower false-positive rate, lower
`(p95, p99, max)` tail tuple, lower total action-futures per root, then fixed
arm name. No eligible arm closes Attempt07 development as No-Go. A winner only
permits writing a separate winner freeze; it does not authorize the audit.

## Seed separation

All namespaces use stride `1000003` and
`seed = namespace base + stride * root_index`:

| Namespace | Base |
|---|---:|
| hand | 60,106,071,901 |
| screen | 61,106,071,901 |
| rerank | 62,106,071,901 |
| veto | 63,106,071,901 |
| assessment | 64,106,071,901 |
| child policy | 65,106,071,901 |

Development uses indices 0 through 99. A possible future audit is declared at
indices 100 through 149. The validator derives all twelve population/namespace
slices, proves them pairwise disjoint, and proves they do not overlap the four
known Attempt06 hand/candidate/evaluation/child schedules at indices 0 through
49.

## Future audit remains closed

The 50-root future audit is declared only to reserve a disjoint schedule. It is
not authorized, started, or opened by this contract. Authorization requires a
separate immutable winner freeze after the 100-root development comparison.
That later freeze must still pass correctness, determinism,
scalar/batch/exact-reference parity, and latency checks before any Spot run.

The policy registry remains bound to SHA-256
`d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3`.
The baseline chain and `current` are unchanged.
