# M4.3 Attempt13 frozen pre-preflight contract

Date: 2026-07-15

## Boundary

Attempt13 is one profile-blind, development-only T1-second search teacher. Its
plan is `configs/hu_joint_policy_m43_attempt13.json`, SHA-256
`9b860be1de05570840e0fd7b1e4a3e3c2f52e5785be4a882d530b5f9baf5c6c2`.
The T1 baseline remains `stage18_p1`; T2 remains `stage9f_p2`, T3 remains
`stage7_m5_r10`, and live T4 remains exact. No profile is activated and
`current` is unchanged.

Only the actor information set is permitted. Opponent private discards,
opponent identity, and opponent profile are forbidden runtime inputs. Every
complete legal action is enumerated with stable ActionKey mapping, illegal
actions are masked, and the explicit baseline is added once.

## Frozen teacher

1. R128 evaluates all legal non-baseline actions plus baseline on common
   random futures and orders non-baselines by paired mean then ActionKey.
2. `K=min(8,n)` keeps the first `min(4,n)` R actions and fills the remaining
   slots without replacement by frozen Lambda raw-risk order. Original R order
   is restored.
3. V256 retains actions with paired mean above zero and normalized p95/p99
   risk `max(loss95/25, loss99/40) <= 1.05`. This is equivalent to p05 at
   least -26.25 and p01 at least -42.
4. Independent X1024 and C1024 receive the identical full V survivor set plus
   baseline. Neither phase filters, reranks, or selects.
5. X and C are joined one-to-one by ActionKey as P2048. Eligibility requires
   mean above zero, normalized p95/p99 risk at most 1.0, linearly interpolated
   q0.1% at least -50, and ES1% at least -40. Eligible actions minimize the
   normalized p95/p99/q0.1%/ES1% risk, then maximize mean, then use frozen R
   position and ActionKey.
6. Raw minimum is diagnostic only in both V and P; it cannot filter, select,
   or gate. Empty V or P eligibility returns the exact baseline.
7. E512 opens only after the final ActionKey is locked. It is independent and
   diagnostic-only and cannot change the action.

## One-shot gates

Development has 200 fresh roots, 40 for each of five frozen opponent
policies. It requires at least 40 fires and at least three per profile,
positive mean delta per state and fire, false-positive rate at most 0.40,
maximum fired-root p95/p99/max losses at most 25/40/50, every integrity count
equal to zero, and exact baseline fallback on every non-fire. No threshold,
profile, arm, or alternate seed may be chosen after seeing the result.

Only an immutable Development-Go freeze may open the disjoint 50-root Audit.
Audit rows cannot fit or calibrate. Realized paired seat-swap population
acceptance remains required after Audit-Go.

## Fresh seeds

All schedules use stride 1,000,003. Development/audit namespaces use bases
230,108,071,901 through 236,108,071,901 and roots 0..199 / 200..249.
Correctness-preflight namespaces use bases 240,108,071,901 through
246,108,071,901 and source roots 0..4. Seven population namespaces at bases
250,108,071,901 through 256,108,071,901 reserve 1,000 paired seeds each but
are not authorized. The contract validator proves all 28 schedules are
pairwise disjoint and have no overlap with the frozen prior registry,
Attempt12 development/audit/preflight schedules, or its reserved population
schedule.
