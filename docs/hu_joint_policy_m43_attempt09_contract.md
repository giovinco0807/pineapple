# M4.3 Attempt09 frozen pre-preflight contract

Date: 2026-07-14

## Decision boundary

Attempt09 freezes one development-only T1-second search architecture. It is
not a best-response claim, a Nash or low-exploitability claim, or a
mathematically optimal solution. The fixed baseline remains `stage18_p1`, the
first-seat behavior remains unchanged, and `current`, the existing P0/P1/P2/T3
profiles, and the live exact T4 solver are not changed.

Attempt08 is closed No-Go. Attempt09 binds the exact No-Go decision and receipt:

- decision SHA-256:
  `4bdc90be7049c201e8b7dbd788603eb087a7df591ab69df7002a52773774217f`;
- receipt SHA-256:
  `f486ae8c02cffcea66997d491c3bedc593bbbf9f12dfc69e3d80a72c16cbc8af`.

Those opened results may inform the architecture postmortem only. Attempt09
cannot retry an Attempt08 seed, open its reserved audit, reselect a threshold,
or promote Attempt08. The frozen Attempt09 plan is
`configs/hu_joint_policy_m43_attempt09.json`, SHA-256
`8c8d5b2a4ce67d555dcea2b499f9c72fea169ead11837bbd6cf131f14fdeabda`.

## Information and action boundary

Complete legal-action enumeration, illegal-action masking, stable
`regular_ofc_action_key_v1` mappings, and the explicit baseline added exactly
once are mandatory. Search may use only the actor's public information set:
hero board, opponent public board, hero private discards, hero dealt cards,
seat/action order, turn, scoring context, and the remaining-deck belief. The
opponent's private discard and opponent profile/identity are forbidden runtime
features.

The candidate generator is the unchanged frozen Lambda artifact at
`outputs/hu_joint_policy/m43_attempt05_old_dev900_architecture_once/lambda_rank_candidate.pkl`,
SHA-256
`e7fa728308c8973e2e202baf79309e30b9b6f58c37431d8ea55850390abc28c3`.
Its raw downside heads are candidate-generation diagnostics only and may not
directly fire an override.

## Single frozen search architecture

Attempt09 preserves Attempt08 Top8, R128, K4, and V256 eligibility. The only
arm is:

1. **Top8 + baseline.** The frozen Lambda scorer proposes eight unique legal
   non-baseline actions, tied by ActionKey. The explicit baseline is added once.
2. **R128.** All nine actions are evaluated with 128 common-random futures.
   Non-baselines are ordered by paired mean descending, then ActionKey.
3. **K4.** Take R ranks 1 through 3 and the minimum raw Lambda-risk reserve
   from R ranks 4 through 8. Restore the four actions to original R order.
4. **V256.** Evaluate K4 plus baseline with an independent 256-future
   common-random batch. Retain *every* non-baseline satisfying mean greater
   than zero, p05 at least -22, p01 at least -36, and minimum at least -45.
   Retained actions remain in original R order. V256 cannot select the final
   action or rerank the set.
5. **X512.** If V retained at least one action, evaluate all retained actions
   plus baseline with an independent 512-future common-random batch. Retain
   every action whose paired minimum is at least -45, without reranking.
6. **C256.** If X retained at least one action, evaluate all retained actions
   plus baseline with an independent 256-future common-random batch. Again
   retain every action whose paired minimum is at least -45. The first
   remaining action in the original R order fires.
7. **Exact baseline fallback.** An empty retained set after V, X, or C returns
   the exact baseline ActionKey. No later candidate is promoted after C.
8. **E256.** A final non-baseline action is evaluated against the explicit
   baseline using a wholly disjoint 256-future common-random batch. E256 is
   diagnostics-only and cannot rerank, veto, confirm, promote, or change the
   already frozen root action.

R, V, X, C, and E use pairwise-disjoint RNG namespaces. Candidate-selection MC
is therefore independent of the E256 acceptance MC. Teacher outputs remain
diagnostics and must not be reported as realized match EV or used as a runtime
LCB gate.

## Fixed compute ceiling

R128 costs 1,152 action-futures and V256 costs 1,280 per root. X costs
`(V-retained + 1) * 512`, from 1,024 to 2,560 when opened. C costs
`(X-retained + 1) * 256`, from 512 to 1,280 when opened. E256 costs 512 on a
final fire. With four candidates surviving both filters, the maximum complete
fire path is 6,784 action-futures per root.

## Preflight and population freeze

No run is authorized by this contract. A later, separate authorization must
first run the fixed correctness preflight covering deterministic recomputation,
scalar/batch parity, exact-reference parity, action mapping, information-set
redaction, and latency. Preflight generation and content opening are currently
false.

The only declared development population is 200 new roots, indices 0 through
199: 40 roots for each of `stage19_p0`, `stage9f_p2`, `stage7_m5_r10`,
`stage3_baseline`, and `random_exact_final`, assigned by root index modulo five.
A disjoint 50-root audit at indices 200 through 249 is reserved but closed,
unauthorized, unstarted, and unopened.

## Development Go/No-Go

The single arm passes only if every gate succeeds on the fresh 200-root
development set, using E256 only:

- at least 40 fires and at least three fires for every profile;
- mean delta per state and mean delta per fire strictly above zero;
- false-positive rate at most 0.40;
- strict maximum-per-fired-root E256 loss p95/p99/max at most 25/40/50;
- zero action-mapping, RNG-domain, hidden-information, risk-reserve,
  retained-order, phase-filter, and locked-action-change violations; and
- exact baseline ActionKey fallback for every non-fire.

Non-fire complete-trajectory cancellation remains mandatory for later runtime
acceptance and is not claimed by this search-only contract. Passing development
would authorize only a separate immutable development-pass freeze; it does not
directly authorize the reserved audit, fitting, or runtime activation. Any gate
failure closes Attempt09 No-Go without threshold reselection or seed retry.

## Fresh seed proof

All schedules use stride `1,000,003` and
`seed = namespace base + stride * root_index`.

| Namespace | Development/audit base | Preflight base |
|---|---:|---:|
| hand | 100,108,071,901 | 110,108,071,901 |
| rerank | 101,108,071,901 | 111,108,071,901 |
| veto | 102,108,071,901 | 112,108,071,901 |
| stress | 103,108,071,901 | 113,108,071,901 |
| confirmation | 104,108,071,901 | 114,108,071,901 |
| evaluation | 105,108,071,901 | 115,108,071,901 |
| child policy | 106,108,071,901 | 116,108,071,901 |

The validator materializes 14 development/audit schedules plus seven preflight
schedules, proves all 21 pairwise disjoint, and proves them disjoint from known
Attempt06, Attempt07 (including opened preflight), and Attempt08 (including its
reserved audit and opened preflight) numeric seed schedules.

## Frozen status

Preflight generation/start, development generation/start, audit
authorization/start, fit, threshold selection, runtime activation, `current`
mutation, full replacement, and large-scale execution are all false. The policy
registry remains bound to SHA-256
`d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3`.

Validate the frozen contract locally with:

```powershell
python -m pytest tests/test_hu_m43_attempt09_contract.py -q
```
