# M4.3 Attempt06 corrected pre-fresh contract

Date: 2026-07-14

## Frozen scope

Attempt06 is frozen at `frozen_preflight`. It may run one fresh 50-root
T1-second search-quality audit only after local correctness, determinism,
scalar/batch/exact-reference parity, and latency checks pass. Spot, fresh data,
fit, distillation, threshold selection, acceptance evaluation, population
evaluation, runtime activation, and changes to `current` are not authorized at
this freeze.

Attempt05 remains a No-Go. Reusing its LambdaRank artifact for candidate
generation does not promote its model or reinterpret its teacher metrics as
realized match EV.

## Coverage metric correction

The old candidate-set diagnostic required absolute positive coverage of at
least `0.70` over all 900 states. Only 565 of those states contain any legal
non-baseline action with positive independent e64 paired delta. Even a set
containing every legal action therefore has a ceiling of `565 / 900 =
0.6277777777777778`; the `0.70` gate is mathematically unreachable. The old
per-profile `0.60` gate is also unreachable for `stage7_m5_r10`, whose
opportunity ceiling is `107 / 180 = 0.5944444444444444`.

Attempt06 corrects the denominator before opening fresh data:

`rank8 conditional recall = opportunity states covered by rank8 / all legal opportunity states`

The frozen design gate is overall recall `>= 0.95` and every-profile recall
`>= 0.93`. On the multiple-use dev900 design set, rank8 covered 541 of 565
opportunities, `0.9575221238938053`. The minimum profile was
`stage3_baseline`, 108 of 115, `0.9391304347826087`. All five profiles pass.

This correction only qualifies rank8 as the candidate set for Attempt06. It
is not a fresh-audit Go gate, threshold fit, generalization claim, acceptance
result, or runtime authorization. The source diagnostic remains immutable
with its original status; the corrected contract does not rewrite it.

## Frozen search and continuation

- Candidate generator: LambdaRank artifact SHA-256
  `e7fa728308c8973e2e202baf79309e30b9b6f58c37431d8ea55850390abc28c3`.
- Candidate set: eight unique legal non-baseline ActionKeys, followed by the
  exact baseline once.
- Candidate search: c8 with common random futures.
- Independent locked evaluation: e128 on a disjoint RNG namespace scores only
  the c8-locked action and explicit baseline. It cannot rerank the c8 winner;
  the other seven candidates have null e128 fields. This avoids opening
  unnecessary fresh counterfactuals and cuts the dominant search cost without
  changing the selected-action-vs-baseline gate.
- Teacher schema v3 retains all 128 selected-action-vs-baseline paired future
  deltas. The paired summary and p05/p01/min downside inputs are recomputed
  from those raw values with NumPy linear quantiles; self-reported summaries
  alone are not trusted.
- T2 continuation: the explicitly loaded real `stage9f_p2` policy, never
  `current`.
- T3 continuation: the existing M3 Rust selector.
- T4 inside the hypothetical T1 counterfactual: direct and nested MC1.
- T4 in actual live play: the existing exact solver, unchanged.

The model and search may use only the canonical T1-second actor observation,
complete legal action set, exact ActionKey mapping, and public-information
belief. Opponent private discards, opponent profile identity, teacher EV, and
teacher LCB are forbidden runtime inputs or gates.

## Transferred one-shot schedule

Attempt05 never opened its `pilot_audit` role. Attempt06 exclusively takes
ownership of that unused numeric schedule:

- 50 roots, five profiles, 10 roots per profile;
- profile assignment by `root_index mod 5` in frozen profile order;
- hand seed start `17306071901`;
- candidate seed start `23306071901`;
- evaluation seed start `24306071901`;
- child-policy seed start `25306071901`;
- stride `1000003` in every namespace.

Every root index 0 through 49 receives one hand/candidate/evaluation/child seed
tuple. The four enumerated 50-seed namespaces must each be unique and pairwise
disjoint. Attempt05 may not reuse them after transfer.

The operational layout is 50 shards of one root each. The root is therefore
the checkpoint, heartbeat, completion, and preemption-resume unit; a Spot
preemption cannot invalidate a larger multi-root shard.

## One-shot Go/No-Go

All gates must pass on the independent e128 result:

- at least 10 fires overall and at least one fire for every profile;
- mean paired delta per state strictly greater than zero;
- mean paired delta per fire strictly greater than zero;
- false-positive rate per fire at most `0.50`, where an e128 paired delta
  `<= 0` is a false positive;
- tail gate B: for every fired root, compute downside from its 128 independent
  paired-future deltas as `p95_loss = max(0, -p05)`, `p99_loss = max(0,
  -p01)`, and `max_loss = max(0, -min)` using NumPy linear quantiles; across
  all fires, gate the maximum per-root p95 at 25, maximum per-root p99 at 40,
  and maximum per-root max loss at 50;
- zero ActionKey/mapping, RNG-domain, or hidden-information violations.

This is the same selected-action paired-p05/p01/min maximum semantics used by
the Attempt04 locked evaluator. A second statistic may take
`max(0, -paired_delta_mean)` for each fired root and compute NumPy quantiles
across fires, but that statistic is diagnostic only and is not any Attempt06
tail gate. The clarification is frozen before fresh content is opened; it is
not a post-result gate change.

The audit must claim its one-shot marker before reading content. It cannot fit
a model, distill, select or retune a threshold, alter a gate, or retry the same
seeds after opening them.

A Spot preemption after an immutable per-root claim but before publication of
`fresh_inputs/.../root.jsonl` is crash recovery, not another audit sample.  It
may deterministically rematerialize that root only when the existing claim is
byte-identical, the immutable root object is still absent, and the run, hand
seed, profile, source ZIP, startup script, source-closure manifest, plan,
status, model, schedule, and package hashes all match.  Any mismatch fails
closed.  The recovery must use the same seed and frozen code path; alternate
sampling, seed reuse for a new observation, and post-result retries remain
forbidden.

A Go does not authorize the next data run. Its only allowed outcome is to
create a separate freeze that may authorize a fresh 200-root fit/distillation
stage. A No-Go closes Attempt06 without fit, threshold rescue, or Spot
expansion.

## Pre-Spot boundary

The required local order is:

1. focused correctness and fail-closed tests;
2. deterministic repeated digests under fixed inputs and seeds;
3. scalar/batch parity at the same budgets plus bounded exact-reference
   mapping parity;
4. latency profiling for top8 c8/e128 and the bounded exact reference.

The frozen native path uses four batch threads and requires batched child
selectors. A canonical fixed-contract SHA binds the teacher schema, plan,
root/input/model/source manifests, run identity, native/batch settings,
top8/c8/e128 scope, and continuation settings. The same SHA must appear in the
teacher-row provenance, checkpoint, heartbeat, generator summary, and DONE
record.

MC1 values are not required or expected to equal exact values. The parity
check covers execution and mapping semantics, not equality between an MC1
estimate and exact enumeration. Spot remains false until all four preflight
roles have explicit passing evidence.
