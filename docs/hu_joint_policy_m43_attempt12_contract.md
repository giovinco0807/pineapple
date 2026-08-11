# M4.3 Attempt12 frozen pre-preflight contract

Date: 2026-07-15

## Decision boundary

Attempt12 is one new, profile-blind, development-only T1-second search
architecture. It is not a best response, equilibrium, low-exploitability, or
mathematical optimality claim. The fixed T1 baseline remains `stage18_p1`,
first-seat behavior remains unchanged, and `current`, P0/P1/P2/T3, and the
live exact T4 solver are not changed.

Attempt11 is closed No-Go. Its fresh Development200 produced 18 fires:
`stage19_p0=0`, `stage9f_p2=2`, `stage7_m5_r10=1`,
`stage3_baseline=1`, and `random_exact_final=14`. Every non-coverage gate
passed: mean delta per state and per fire were positive, false positives were
2/18, the maximum fired-root E256 p95/p99/max losses were
22.0/34.92702061468345/38.227020614683454, and every integrity count was
zero. The observed funnel was V=79 roots, X=27, C=18. Attempt12 changes the
search structure to remove repeated independent X/C filters. It does not
retune a threshold on the Attempt11 Development result.

All Attempt11 files, artifacts, seeds, and defaults remain preserved. No
Attempt11 label is reused as an Attempt12 result. The frozen Attempt12 plan is
`configs/hu_joint_policy_m43_attempt12.json`, SHA-256
`6a862e355d6136f488b6190fd74d1ec87b896d3c07ba6c48d97442cea2ac48c9`.

## Information and action boundary

Only the actor's information set may be used: hero board, opponent public
board, hero private discards, hero dealt cards, seat/action order, turn,
fantasyland state, scoring context, and the remaining-deck belief. Opponent
private discards, opponent profile, and opponent identity are forbidden
runtime inputs.

Every complete legal T1 action is enumerated before search. Illegal-action
masking, stable `regular_ofc_action_key_v1` mappings, unique actions, and the
explicit baseline added exactly once are mandatory. At T1 there are at most
27 legal actions, so the number `n` of legal non-baseline alternatives is
variable from 0 through 26. No padding, duplicate action, or fixed-top-12
assumption is allowed.

The frozen Lambda artifact remains
`outputs/hu_joint_policy/m43_attempt05_old_dev900_architecture_once/lambda_rank_candidate.pkl`,
SHA-256
`e7fa728308c8973e2e202baf79309e30b9b6f58c37431d8ea55850390abc28c3`.
It scores every complete legal non-baseline action. Its raw downside heads are
used only for the risk-reserve part of K8; Lambda may not cut the legal set,
directly fire an override, or act as a runtime LCB gate.

## Single frozen search architecture

1. **All-legal R128.** Evaluate all `n` unique legal non-baseline actions and
   the explicit baseline on 128 common-random futures. Order non-baselines by
   paired mean descending, then ActionKey.
2. **K=min(8,n).** Keep the first `min(4,n)` R actions. Fill the remaining
   `min(8,n)-min(4,n)` positions from every remaining R action by minimum
   frozen Lambda raw risk, then ActionKey, without replacement. Restore the
   selected actions to original R order and append baseline exactly once.
3. **V256 screen.** Evaluate K plus baseline on a new 256-future
   common-random batch. Retain every non-baseline with paired mean greater
   than zero, p05 at least -25, and p01 at least -40. Raw minimum is retained
   only as a diagnostic; it cannot filter, select, or gate.
4. **Parallel X1024 and C1024.** If V is nonempty, evaluate the exact same full
   V-survivor action set plus baseline in each of two independent 1,024-future
   namespaces. Neither X nor C independently filters or drops an action.
5. **One pooled P2048 decision.** Join X and C one-to-one by stable ActionKey
   and concatenate their raw paired deltas. An action is eligible exactly once
   if its pooled mean is greater than zero, pooled p05 is at least -22, and
   pooled p01 is at least -36. Pooled raw minimum remains diagnostic only.
   Select eligible actions by normalized p05/p01 tail risk ascending, pooled
   mean descending, original R position ascending, then ActionKey ascending.
   The normalized risk is
   `max(max(0,-p05)/22, max(0,-p01)/36)`.
6. **Exact fallback.** Empty V or empty pooled eligibility returns the exact
   baseline ActionKey. When `n=0`, R and V contain only baseline, X/C/E remain
   closed, and the exact baseline is returned.
7. **E512 diagnostic.** A locked non-baseline output is compared with the
   explicit baseline using a wholly independent 512-future common-random
   batch. E cannot rerank, veto, confirm, promote, or change the output.

R, V, X, C, and E use pairwise-disjoint RNG namespaces. Candidate selection
MC is independent of E. Teacher values are diagnostics, not realized match EV.
The stricter -22/-36 thresholds are carried forward unchanged from Attempt11;
they were not selected by a sweep of the failed Development200 result.

## Fixed compute ceiling

R costs `(n+1)*128`, from 128 to 3,456 action-futures. V costs
`(min(8,n)+1)*256`, from 256 to 2,304. Each of X and C costs
`(V survivors+1)*1024`, from 2,048 to 9,216 when opened. E costs 1,024
action-futures on a fire. The maximum full-fire path is 25,216
action-futures per root. Each root is an independent resumable Spot shard with
checkpoint and heartbeat artifacts.

## Correctness preflight

The seven frozen slots are root0 batch twice, root0 scalar once, and roots1,
2, 3, and 4 once in batch. This covers all five opponent policies. Root4 must
explicitly exercise complete-legal variable cardinality, including the old
fewer-than-twelve case. Preflight passes only with:

- byte-stable repeat determinism;
- scalar/batch value and decision parity;
- exact-reference parity on the required anchors;
- complete legal-set, ActionKey, baseline-once, K8, and X/C ActionKey-join
  integrity;
- zero hidden-information and RNG-domain violations;
- verified raw-min diagnostic-only semantics; and
- acceptable recorded latency.

Preflight generation is not authorized merely by this plan.

## Development and one-shot gates

The one Development population has 200 fresh roots, 40 per profile, assigned
by root index modulo five in the fixed order `stage19_p0`, `stage9f_p2`,
`stage7_m5_r10`, `stage3_baseline`, `random_exact_final`. It is a single arm;
there is no architecture winner selection, threshold sweep, profile removal,
alternate seed retry, fitting, or runtime activation. A 50-root Audit remains
closed until an immutable Development-Go freeze authorizes it.

Development passes only if all frozen E512 gates pass:

- at least 40 fires and at least three fires for each profile;
- mean delta per state and per fire strictly above zero;
- false-positive rate per fire at most 0.40;
- strict maximum-per-fired-root p95/p99/max losses at most 25/40/50;
- zero action-mapping, RNG, hidden-information, risk-reserve, retained-order,
  phase, pooled-join/filter, and locked-action-change violations; and
- exact baseline ActionKey fallback on every non-fire.

Any failed gate closes Attempt12 No-Go. Development teacher values may not be
reported as match EV. A Go only opens the separately frozen Audit50; it still
does not authorize fitting, runtime use, or population acceptance.

## Fresh seed proof

All schedules use stride `1,000,003` and
`seed = namespace base + stride * root_index`.

| Namespace | Development/audit base | Preflight base |
|---|---:|---:|
| hand | 200,108,071,901 | 210,108,071,901 |
| rerank | 201,108,071,901 | 211,108,071,901 |
| veto | 202,108,071,901 | 212,108,071,901 |
| stress/X | 203,108,071,901 | 213,108,071,901 |
| confirmation/C | 204,108,071,901 | 214,108,071,901 |
| evaluation/E | 205,108,071,901 | 215,108,071,901 |
| child policy | 206,108,071,901 | 216,108,071,901 |

Against the live frozen registry snapshot of 35 sources and 85,275 enumerated
seeds (SHA-256
`0fafc65e2baa41f6a2389488805864e3fea3722f3f3732391dfa8930b4c27fbf`),
all 14 Development/audit schedules and all seven preflight schedules have zero
overlap and are pairwise disjoint. They also have zero overlap with the frozen
Attempt11 population schedule at 190,108,071,901. The separately reserved
future population base 220,108,071,901 also had zero overlap at freeze, but it
is not authorized by this search contract. The live validator must repeat the
proof before every authorization.

## Frozen status

Preflight generation/start, Development generation/start, Audit
authorization/start, fitting, threshold selection, runtime activation,
`current` mutation, full replacement, and large-scale execution are false.
The policy registry remains bound to SHA-256
`d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3`.
