# M4.3 Attempt11 frozen pre-preflight contract

Date: 2026-07-15

## Decision boundary

Attempt11 freezes one profile-blind, development-only T1-second search
architecture. It is not a best-response, Nash, low-exploitability, or
mathematically optimality claim. The fixed T1 baseline remains `stage18_p1`,
first-seat behavior remains unchanged, and `current`, P0/P1/P2/T3, and the
live exact T4 solver are not changed.

Attempt10 stopped before labels at development roots 4, 24, and 79 because
those `random_exact_final` roots had fewer than twelve legal nonbaseline
actions. Every Attempt10 source file and artifact remains preserved and its
defaults are unchanged. Attempt11 is a new opt-in contract; Attempt10 seeds,
partial results, and artifacts may not be reused as Attempt11 results.
The frozen Attempt11 plan is
`configs/hu_joint_policy_m43_attempt11.json`, SHA-256
`1896f311eec31a48eb19de13ffebdbdabda9629015f48e25291b8dcb2f96c26c`.

## Information and action boundary

Complete legal-action enumeration, illegal-action masking, stable
`regular_ofc_action_key_v1` mappings, unique non-baseline candidates, and the
explicit baseline added exactly once are mandatory. Search may use only the
actor's information set: hero board, opponent public board, hero private
discards, hero dealt cards, seat/action order, turn, scoring context, and the
remaining-deck belief. Opponent private discards and opponent profile or
identity are forbidden runtime inputs.

The candidate generator remains the frozen Lambda artifact at
`outputs/hu_joint_policy/m43_attempt05_old_dev900_architecture_once/lambda_rank_candidate.pkl`,
SHA-256
`e7fa728308c8973e2e202baf79309e30b9b6f58c37431d8ea55850390abc28c3`.
Its downside heads may diversify candidates but may not directly fire an
override.

## Single frozen search architecture

1. **Variable candidates plus baseline.** Let `n=min(12, legal nonbaseline
   action count)`. The Lambda scorer proposes exactly `n` unique legal
   nonbaseline actions, ordered by model score descending and then ActionKey.
   No padding or duplication is allowed. The exact baseline is added once and
   is always last in each open phase mapping.
2. **R128.** The `n+1` mapped actions use 128 common-random futures.
   Nonbaselines are ordered by paired mean descending, then ActionKey.
3. **K=min(8,n).** Let `h=min(4,n)` and `k=min(8,n)`. Keep the first `h`
   actions in R order, then fill exactly `k-h` positions from the remaining R
   positions by minimum frozen Lambda risk and ActionKey, without replacement.
   Restore the selected `k` actions to original R order.
4. **V256 coarse filter.** Evaluate the `k` actions plus baseline using a new 256-future
   common-random batch. Retain every action with paired mean greater than zero,
   p05 at least -25, p01 at least -40, and minimum at least -50. V cannot
   select or rerank the final action.
5. **X1024 strict filter.** If V is nonempty, evaluate all V survivors plus
   baseline using an independent 1,024-future batch. Retain every action with
   mean greater than zero, p05 at least -22, p01 at least -36, and minimum at
   least -45.
6. **C512 strict confirmation and selection.** If X is nonempty, evaluate all
   X survivors plus baseline using an independent 512-future batch and apply
   the same strict thresholds. Select the retained action by this frozen key:
   normalized tail risk ascending, paired mean descending, original R position
   ascending, then ActionKey ascending. Normalized tail risk is
   `max(max(0,-p05)/22, max(0,-p01)/36, max(0,-minimum)/45)`.
7. **Exact fallback.** An empty retained set after V, X, or C returns the exact
   baseline ActionKey. When `n=0`, R and V each contain only the baseline; X,
   C, and E remain closed and the exact baseline is returned.
8. **E256 diagnostic.** A final non-baseline output is compared with baseline
   on a wholly independent 256-future common-random batch. E cannot rerank,
   veto, confirm, promote, or change the locked output.

R, V, X, C, and E have pairwise-disjoint RNG namespaces. Candidate-selection
MC is independent from evaluation MC. Teacher values are diagnostics, not
realized match EV and not a runtime LCB gate.

## Fixed compute ceiling

R128 costs `(n+1)*128`, from 128 to 1,664 action-futures. V256 costs
`(min(8,n)+1)*256`, from 256 to 2,304. The fixed prefilter therefore ranges
from 384 to 3,968 action-futures. X1024 costs
`(V survivors + 1) * 1024`, from 2,048 to 9,216 when opened. C512 costs
`(X survivors + 1) * 512`, from 1,024 to 4,608 when opened. E256 costs 512 on
a fire. The maximum full-fire path is 18,304 action-futures per root.

## Population and one-shot decision

This plan itself authorizes no run. A separate correctness preflight must first
pass deterministic recomputation, scalar/batch parity, exact-reference parity,
action mapping, information-set redaction, and latency checks. It has seven
shards: root0 batch twice and scalar once, then roots1, 2, 3, and 4 in batch.
This covers all five opponent profiles, and root4 explicitly validates the
variable candidate/action-count mapping for `random_exact_final`.

The only development population is 200 fresh roots, indices 0 through 199,
balanced 40 each across `stage19_p0`, `stage9f_p2`, `stage7_m5_r10`,
`stage3_baseline`, and `random_exact_final` by root index modulo five. A fresh
50-root audit, indices 200 through 249, is declared but remains closed until a
separate immutable development-pass freeze authorizes it.

Development passes only if every fixed E256 gate passes:

- at least 40 fires and at least three fires per profile;
- mean delta per state and per fire strictly above zero;
- false-positive rate at most 0.40;
- strict maximum-per-fired-root p95/p99/max losses at most 25/40/50;
- zero action-mapping, RNG-domain, hidden-information, risk-reserve,
  retained-order, phase-filter, and locked-action-change violations; and
- exact baseline ActionKey fallback on every non-fire.

Any failure closes Attempt11 No-Go. No threshold reselection, profile removal,
alternative seed, result retry, or reserved-audit opening is allowed.
Non-fire complete-trajectory cancellation remains mandatory for later runtime
acceptance and is not claimed by this search-only contract.

## Fresh seed proof

All schedules use stride `1,000,003` and
`seed = namespace base + stride * root_index`.

| Namespace | Development/audit base | Preflight base |
|---|---:|---:|
| hand | 140,108,071,901 | 150,108,071,901 |
| rerank | 141,108,071,901 | 151,108,071,901 |
| veto | 142,108,071,901 | 152,108,071,901 |
| stress | 143,108,071,901 | 153,108,071,901 |
| confirmation | 144,108,071,901 | 154,108,071,901 |
| evaluation | 145,108,071,901 | 155,108,071,901 |
| child policy | 146,108,071,901 | 156,108,071,901 |

The validator materializes all 14 development/audit schedules and seven
preflight schedules, proves the 21 schedules pairwise disjoint, and proves
them disjoint from every known numeric schedule for Attempts06 through 10,
including prior preflights and reserved audits.

## Frozen status

Preflight generation/start, development generation/start, audit
authorization/start, fitting, threshold selection, runtime activation,
`current` mutation, full replacement, and large-scale execution are false.
The policy registry remains bound to SHA-256
`d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3`.

Validate locally with:

```powershell
python -m pytest tests/test_hu_m43_attempt11.py -q
```
