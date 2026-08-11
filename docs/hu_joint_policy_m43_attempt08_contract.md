# M4.3 Attempt08 pre-development contract

Date: 2026-07-14

## Decision boundary

Attempt08 freezes one new T1-second search arm for a fresh development run. It
does not claim a best response, a Nash policy, or mathematical optimality, and
it does not promote a runtime policy. The fixed T1 baseline remains
`stage18_p1`; first-seat behavior is unchanged. `current`, the existing P0/P1/
P2/T3 baselines, and the live exact T4 solver are not changed.

Attempt07 is already closed as `complete_no_go_development`. Its closeout is
bound to SHA-256
`cf9da0b0da80f66e97eab70bb99d8658a01c3052e068b128f2d7318d6ad323fb`,
and its arm-selection evidence is bound to
`53b797756747cbb71f89251020e7fdc86dabce75cd298507dc59c0764ed6373a`.
Those opened results may inform this new architecture as development
postmortem evidence, but Attempt08 cannot retry an Attempt07 seed, reopen its
reserved audit, exclude a profile, or reselect a threshold after seeing new
results.

The only declared development population is 200 new roots, indices 0 through
199: 40 roots against each of `stage19_p0`, `stage9f_p2`,
`stage7_m5_r10`, `stage3_baseline`, and `random_exact_final`. A possible
50-root audit is reserved at indices 200 through 249, ten roots per profile.
That audit is closed and unopened. Development generation itself also remains
unauthorized until the separate correctness, determinism, scalar/batch parity,
exact-reference parity, and latency preflight succeeds.

## Frozen single search arm

Complete legal-action enumeration, illegal-action masking, and stable
`ActionKey` mappings are mandatory. The policy/search observation may contain
the hero board, opponent public board, hero private discards, cards dealt to the
hero, action order, and the remaining-deck belief. It must not contain the
opponent's private discard or opponent identity as a runtime feature.

The candidate artifact is the frozen Lambda model at
`outputs/hu_joint_policy/m43_attempt05_old_dev900_architecture_once/lambda_rank_candidate.pkl`,
bound to SHA-256
`e7fa728308c8973e2e202baf79309e30b9b6f58c37431d8ea55850390abc28c3`.
It proposes exactly eight unique legal non-baseline actions by strict action
score, tied by ActionKey. The explicit baseline is added exactly once.

The single arm is:

1. `R128` evaluates all eight proposals plus the explicit baseline with 128
   common-random futures in the rerank namespace. It ranks the eight
   non-baseline actions by paired-delta mean, tied by ActionKey; the baseline is
   the reference and does not consume a non-baseline rank.
2. `K4` takes R ranks 1 through 3 and one reserve from R ranks 4 through 8.
   For each reserve candidate, the frozen Lambda raw downside heads define
   `max(p95/22, p99/36, max/45)`. The smallest score wins, tied only by
   ActionKey. Conformal upper bounds and the gain head are not gates. After the
   reserve is chosen, all four actions return to their original R order.
3. `V256` evaluates K4 plus the baseline using an independent 256-future
   common-random batch. Traversing actions in original R order, it selects the
   first action whose paired delta has mean strictly above zero, p05 at least
   -22, p01 at least -36, and minimum at least -45. It must not maximize the
   V256 mean. If no action passes, the exact baseline is selected.
4. When V256 provisionally selects a non-baseline action, `X512` evaluates only
   that locked action and the baseline in the independent stress namespace.
   The locked action survives only if its paired minimum is at least -45.
   X512 is cancel-only: it cannot promote, rerank, or switch to another action.
5. When the final output after X512 is still non-baseline, disjoint `A256`
   evaluates only the locked output and baseline. It retains raw paired deltas
   for development diagnostics and gates. It cannot alter the root action, and
   its teacher values must not be reported as realized match EV.

Candidate-selection randomness (`R`, `V`, and `X`) is independent of A256.
Candidate MC is therefore not reused as evaluation MC.

## Fixed compute accounting

The action-future counts are derived from the fixed action scopes:

| Phase/path | Action-futures |
|---|---:|
| R128, nine actions | 1,152 |
| V256, five actions | 1,280 |
| Prefire / immediate V non-fire | 2,432 |
| X512, two actions on a provisional fire | 1,024 |
| X-cancelled non-fire total | 3,456 |
| A256, two actions on a final fire | 512 |
| Full final-fire path | 3,968 |

X512 is skipped when V256 returns the baseline. A256 is skipped whenever the
final output is the baseline. One root per restartable shard, checkpoints,
heartbeats, batched child selectors, and four native threads are frozen.
Preemption may recompute the identical root under the identical contract; it
may not replace a completed root or seed based on its result.

## Development Go/No-Go

The single arm passes only if every gate succeeds on all 200 development
roots:

- at least 40 final fires and at least three final fires for every profile;
- A256 mean delta per state strictly above zero, treating every non-fire as
  zero, and A256 mean delta per fire strictly above zero;
- false-positive rate at most 0.40, where a final fired root with A256 paired
  mean at or below zero is a false positive;
- A256 tail limits 25/40/50 under strict maximum-per-fired-root semantics; and
- zero action-mapping, RNG-domain, hidden-information, risk-reserve-contract,
  and locked-action-change violations, with exact baseline ActionKey fallback
  on every non-fire.

For the tail gate, each final fired root independently yields
`loss_p95=max(0,-p05)`, `loss_p99=max(0,-p01)`, and
`loss_max=max(0,-minimum)` from its raw A256 deltas using NumPy linear
quantiles. The observed metric is the maximum across fired roots for each
component. It is not a quantile pooled across roots and is not an average.

If every gate passes, the only permitted next action is a separate immutable
development-pass freeze. That freeze still does not authorize the 50-root
audit. If any gate fails, Attempt08 closes as development No-Go. There is no
arm comparison, winner selection, threshold reselection, seed retry, fit, or
runtime activation in this contract.

The development teacher can verify exact baseline ActionKey fallback, but no
distilled runtime policy exists yet. Full non-fire trajectory cancellation is
therefore deferred, not waived; it remains mandatory in a later fresh runtime
acceptance evaluation.

## Seed proof

Every schedule uses stride `1000003` and
`seed = namespace base + stride * root_index`:

| Namespace | Base |
|---|---:|
| hand | 80,108,071,901 |
| rerank | 81,108,071,901 |
| veto | 82,108,071,901 |
| stress | 83,108,071,901 |
| assessment | 84,108,071,901 |
| child policy | 85,108,071,901 |

The validator materializes six development and six future-audit schedules,
proves all twelve pairwise disjoint, and proves each is disjoint from all known
Attempt06 schedules at indices 0 through 49, all known Attempt07 development/
reserved-audit schedules at indices 0 through 149, and the actually opened
Attempt07 preflight screen/rerank/veto/assessment/child schedules at source
indices 0 through 2 (bases 71,106,071,901 through 75,106,071,901).

## Frozen status

Development generation authorization/start, audit authorization/start, fit,
threshold selection, runtime activation, `current` mutation, full replacement,
and large-scale execution are all false. The policy registry remains bound to
SHA-256
`d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3`.

The contract can be checked locally with:

```powershell
python -m pytest tests/test_hu_m43_attempt08_contract.py -q
```
