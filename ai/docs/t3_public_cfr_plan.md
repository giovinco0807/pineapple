# T3 public-range CFR plan

Status: M2 reduced reference complete; M3 solver foundation in progress. Strict physical-particle validation,
conditioned BB T4 action vectors, explicit public transitions, recursive
reduced-tree CFR+, exhaustive infoset-aware best responses, all six canonical
actor/Joker real-card fixtures, and the content-bound promotion artifact are
implemented and verified. Full-card T3 solver wiring is implemented but no
full-card T3 policy is strength-gated or promoted; M3 remains in progress. The
54-card history-weighted range builder, the explicit
physical path through BTN T4 and 13/13 terminal state, the fail-closed M3 range
gate, and an exact-oracle-tested external-sampling MCCFR implementation for the
finite reduced tree are implemented. The lazy full-card generative adapter and
encountered-only dynamic external-sampling MCCFR+ loop are also implemented and
covered by deterministic integration tests. A real-card six-stratum smoke
artifact has also passed and read back successfully while retaining explicit
non-promotion status. An independently calibrated all-seat behavior checkpoint,
full-card root-disjoint strength evaluation, and promotion evidence remain
unfinished.

## Why PIMC stopped

The sampled PIMC evaluator solves future actions separately inside each
hidden-card determinization. Increasing BB from outer=4/inner=4 to
outer=8/inner=4 left independent-seed agreement at 15/20 on the selected hard
set; outer=16/inner=4 left only 2/5 on the remaining hard set. More samples
would converge to the PIMC objective, not to a legal public-information
strategy.

## Information-set contract

A T3/T4 policy key contains:

- `position_contract_version=bb_first_v1`;
- actor, turn, and phase;
- both public boards;
- ordered public placement history;
- the actor's ordered private draw/discard recall;
- the actor's current private draw;
- Fantasyland/session state when applicable.

It never contains the opponent's discard/draw, the physical remaining deck,
a particle/determinization ID, a future draw, or an RNG seed.

At a fixed information set, candidate values are averaged across compatible
particles before action selection. `E[max]` and `E[min]` per world are retained
only as illegal PIMC diagnostics.

## Why CFR is required

Public placements signal private card information. After observing action
`a`, a hidden world has posterior weight

```text
P(world | history, a) proportional to
P(world | history) * strategy(infoset(world), a)
```

The strategy determines the posterior that later best responses use, so a
single backward `max(E)`/`min(E)` pass is circular. The target solver is a
public-tree external-sampling MCCFR/Deep-CFR path with exact T4 response
leaves.

The first promoted T3/T4 candidate is necessarily conditional on a pinned
upstream T1/T2 behavior policy. After T2 and T1 are improved, that behavior
hash changes and the dependent T3/T4 artifact becomes stale. Final all-turn
completion therefore requires a fixed-point loop: update upstream policies,
rebuild the history-weighted range, re-solve T3/T4, and repeat until policy and
posterior drift pass their independent-seed thresholds.

## Causal fixed-point artifact contract

The content-addressed production chain must be acyclic. For state round `r`
the only valid dependency order is:

```text
candidate-query bundle Q_r
  -> exact candidate table C_r
  -> history-weighted evaluation ranges R_r(C_r)
  -> six-stratum evaluation bundle B_r(R_r)
```

`C_r` is rebuilt from the physical checkpoint and strategy bytes in `Q_r`.
Every evaluation range is independently replayed and must bind `C_r`; every
entry in `B_r` must bind that range's content and build hashes. Candidate-query
and evaluation bundles are distinct artifacts with disjoint coverage. The
invalid cycle `C_r -> B_r -> R_r(C_r) -> C_r` cannot be produced from real
content hashes and must be rejected even when all copied summaries are
self-hashed.

A state transition compares `(C_{r-1}, R_{r-1}, B_{r-1})` with
`(C_r, R_r, B_r)`. Independent-seed policy and BTN-posterior drift are
stability evidence only. In particular, identical deterministic cold starts
can have zero drift without being strategically strong. The fixed-point gate
therefore does not claim exact exploitability or approximate-equilibrium
strength; a separate root-disjoint strength gate remains mandatory.

There is one further production requirement beyond artifact wiring. Solving
each T3-BB private query in an isolated game does not by itself prove a
signaling equilibrium, because BTN information sets reached from different BB
private types must share one regret/strategy table. Promotion-grade solving
must either train those compatible query roots jointly with shared responder
information sets, or demonstrate equivalent shared-infoset behavior in an
independent strength test. Per-query jobs remain useful for wiring and
likelihood diagnostics but are not, alone, proof of globally optimal T3 play.

## Implemented references

- `ai/tutor/t3_hu_public_cfr.py`
  - information-set and private-recall identities;
  - full particle/observation compatibility checks before private data is
    removed from the policy key;
  - exact epsilon-smoothed Bayes update;
  - grouped backup/PIMC counterexamples;
  - deterministic reduced signaling CFR+, average strategy, best responses,
    NashConv, and exploitability.
- `ai/tutor/exact_late.py`
  - strict BB T4 uniform-range PlayerView adapter;
  - all legal candidate metrics returned in stable action-key order;
  - no BB selection performed in the leaf adapter;
  - natural and X1/X2 Python/Rust parity;
  - explicitly `hu_exact=false`: the uniform hidden-discard marginal is not a
    substitute for a history-weighted particle range.
- `ai/tutor/t3_hu_particle_cfr.py`
  - one policy/regret node per shared `InfoSetKey`;
  - chance-weighted complete action vectors are aggregated before regret
    updates for both BB/max and BTN/min;
  - no particle ID or hidden cards enter the policy identity;
  - explicitly a fixed-leaf correctness prototype, not a full public tree.
- `ai/tutor/exact_late.py` physical-particle API
  - accepts either both exact hidden-discard sets or an explicit complete
    reduced remaining-card range;
  - returns every BB T4 action vector in stable key order without selection;
  - labels the result `public_policy_safe=false` and requires infoset
    aggregation before any policy choice.
- `ai/tutor/t3_hu_public_tree.py`
  - strict `t3_first -> t3_second -> t4_first -> t4_second -> terminal`
    physical transitions;
  - exact supplied chance mass, X1/X2-preserving deck removal, both players'
    T4 recall, and validated 13/13 terminal state;
  - verifies that a fixed BTN T4 information set has the same terminal action
    values when only BB's hidden T4 discard changes.
- `ai/tutor/t3_hu_full_card_range.py`
  - enumerates or deterministically samples opponent hidden-discard histories
    from the canonical 54-card deck and proves the phase-specific full-deck
    partition (`29/26/23/20` undealt cards);
  - sequentially weights histories with a content-addressed frozen behavior
    policy that returns an exact complete legal-action distribution;
  - keeps the observed action outside the behavior-policy query, memoizes by
    query digest, records raw distribution/source/fallback audit entries, and
    computes exact normalized weights and ESS;
  - separates range content and build hashes and independently re-verifies the
    physical particles, manifests, hashes, ESS, and raw behavior audit;
  - supports X1/X2 as distinct physical hypotheses and remains explicitly
    `hu_exact=false` until the full MCCFR policy is promoted.
- `ai/tutor/frozen_behavior_torch.py`
  - infers the real `520 -> 250` legacy PolicyNetwork dimensions from a frozen
    state dict instead of silently constructing the incompatible 522-input
    default;
  - maps every legal regular-turn action through the stable semantic index,
    runs CPU masked inference, and uses deterministic largest-remainder Q32
    quantization so the returned Fraction distribution sums exactly to one;
  - binds checkpoint bytes, adapter source, architecture, state/action/rules
    contracts, scope, temperature, and quantization into the model manifest;
  - can dispatch turn/actor specialists without a fallback route. Existing
    bottom-up T1/T2 checkpoints load and drive a verified sampled range, but
    remain explicitly non-promoted legacy assets because their training used
    an empty opponent board and BTN-only state;
  - also loads the four 522-input, position-specific HU T1/T2 PolicyValueNet
    checkpoints with their exact byte hashes. Those models include the visible
    opponent board and BB/BTN flag and drive a verified full-card range with no
    fallback, but their ranking logits are explicitly marked as an uncalibrated
    Boltzmann prior and remain `promotion_eligible=false`.
- `ai/tutor/promotion_gate_m3_range.py`
  - pins the behavior-model content hash and accepted source types;
  - re-derives raw query counts, unique queries, fallback/model-hit rates, and
    content/build/model hashes rather than trusting published aggregates;
  - requires BB/BTN x visible Joker 0/1/2 exactly once and independently binds
    each stratum's Joker count into the range content manifest;
  - rejects uniform, fallback, unapproved, legacy, or otherwise non-promotable
    behavior models.
- `ai/tutor/t3_hu_public_mccfr.py`
  - tabular external-sampling MCCFR+ over the same explicit reduced-tree API as
    the deterministic recursive CFR+ oracle;
  - samples chance and the opponent, expands every traverser action, caches one
    opponent action per `InfoSetKey` per traversal, and samples the root
    posterior exactly once without multiplying `JointParticle.weight`;
  - exposes a raw fixed-profile regret estimator for unbiasedness tests and a
    direct sampled-vs-exact oracle comparison API;
  - remains explicitly `full_card=false` and serves as the exact-oracle-tested
    finite reference for the dynamic physical sampler;
  - stores content-addressed canonical checkpoints at completed iteration
    boundaries and reproduces one-shot results byte-for-byte after resume.
- `ai/tutor/t3_hu_full_card_mccfr.py`
  - samples one posterior root particle per traversal with exact `Fraction`
    mass, then resets the physical particle weight to one;
  - samples later unordered three-card draws directly by combinadic rank at
    the exact `29 -> 26 -> 23 -> 20` deck boundaries, without multiplying
    sampled chance probabilities into regret;
  - runs lazy encountered-only alternating external-sampling MCCFR+, expands
    every traverser action, caches one opponent action per shared `InfoSetKey`
    per traversal, and clips aggregated CFR+ regret once per infoset;
  - serializes strategies in stable infoset/action order without particle
    commitment, hidden remainder, world ID, or particle weight;
  - stores content-addressed, atomic iteration-boundary checkpoints containing
    RNG state, encountered infosets, cumulative regret/strategy sums, range
    and behavior commitments, and scoring/FL/source hashes; split resume is
    byte- and SHA-identical to a one-shot run;
  - remains explicitly `full_card_policy_promoted=false`, `hu_exact=false`,
    and `runtime_integrated=false`; implementation correctness is not a
    root-disjoint strength gate.
- `ai/tutor/promotion_gate_m3_full_card_smoke.py`
  - runs and validates BB/BTN x visible Joker 0/1/2 exactly once each;
  - re-derives role/Joker/range/model/solver bindings and raw behavior-query
    counters, rejects fallback or uniform sources, and writes canonical atomic
    evidence/results;
  - reports `execution_smoke_passed` independently while always retaining
    `m3_promotion_passed=false` and `full_card_policy_promoted=false`.
- `ai/tutor/run_m3_full_card_smoke.py`
  - loads exact-hash T1/T2 BB/BTN ranking priors plus the exact-hash T3 BB
    discard-sensitive ranking prior needed to condition a BTN-root range on
    the observed preceding BB T3 action;
  - compiles six real-card fixtures, builds four-particle physical ranges, and
    runs one canonical-terminal-scoring dynamic iteration per stratum;
  - reproduces `ai/reports/m3_full_card_smoke_20260713/`, whose evidence
    artifact SHA-256 is
    `5f4375a67fa21d7bd55107c888ff589895b17f9309bef61cb01cc780a8b1e22f`.
- `ai/tutor/behavior_calibration_contract.py`
  - defines content-addressed full-private T1/T2 decision logs without placing
    hidden trace or the observed label in the behavior-policy input;
  - independently reconstructs `BehaviorInfoSet`, legal actions, semantic
    index, discard, mask, Joker cell, and root-hash fit/dev/test split;
  - remains `promotion_eligible=false` until real logs, fitting, and the locked
    calibration gate exist.
- `ai/tutor/collect_hu_behavior_trace_shards.py`,
  `ai/tutor/evaluate_hu_behavior_trace_shards.py`, and
  `ai/tutor/behavior_temperature_calibration_shards.py`
  - publish immutable root shards and one direct-logit evaluation shard per
    input shard without a monolithic JSONL merge;
  - bind collection/evaluation layout, ordered entries, source files, four
    checkpoint routes, and every row join;
  - rebuild the existing v2 fit/dev/locked-test temperatures, metrics,
    2,000-replicate root bootstrap, and gate result exactly while retaining
    only one shard plus an ephemeral SQLite metric store in memory;
  - reject incomplete collections, tamper, gaps, orphan shards, evaluator
    drift, and manual promotion claims.
- `ai/tutor/calibrated_behavior_sharded_bootstrap.py`
  - fresh-verifies all four immutable collection/evaluation trees and the
    saved sharded calibration artifact before loading any fixed-point route;
  - binds the exact four checkpoint files, temperatures, collection/evaluation
    content/layout/manifest/ordered-chain hashes, and gate result into the
    existing no-fallback T1/T2 fixed-point bootstrap dispatch;
  - remains `promotion_eligible=false`, `fixed_point_bootstrap_only=true`, and
    `strategic_strength_evaluated=false` even when the source calibration
    passes. Production callers may additionally require a promoted source and
    fail closed before constructing the runtime.
- `ai/tutor/t3_bb_fixed_point_gate_v2.py` and
  `ai/tutor/run_t3_bb_fixed_point_smoke.py`
  - implement and replay the acyclic `Q_r -> C_r -> R_r(C_r) -> B_r`
    contract with separate candidate-query and six-stratum evaluation bundles;
  - reconstruct query/root information sets, exact Q32 candidate tables,
    checkpoint strategies, restricted physical ranges, BTN posterior weights,
    and every same-root cross-seed TV comparison from physical artifacts;
  - hard-code at least three seeds, 100 roots in each BB/BTN x Joker stratum,
    three trailing converged transitions, and all policy/posterior TV caps at
    `1/100`; weak diagnostic settings cannot emit the T3-BB likelihood binding;
  - reproduce `ai/reports/t3_bb_fixed_point_smoke_20260713/` from 72 real
    MCCFR jobs. Fresh readback passes with report SHA-256
    `23e92e8e965981aeaa5f8c2f017f339778f4443d2c2e48f3e9bf66b8a5d06986`;
    the v2 evidence SHA-256 is
    `f51d64b9371f424b70b85f7feeb386342a9c80f17d4646d43ba7f0742ee4766a`.
    The smoke correctly remains non-promoting because it has two seeds, one
    root per stratum, two transitions, and cross-seed TV `1/1`.
- `ai/tutor/t3_hu_multi_root_mccfr.py`
  - adds an exact-prior chance super-root above multiple compatible private T3
    roots while all roots update one regret/strategy table keyed only by
    `InfoSetKey`;
  - samples one super-root and one conditional posterior per traversal,
    alternates BB/BTN traversers, and caches one opponent action per shared
    information set without putting root IDs, private-type IDs, or particle
    commitments into policy identity or strategy serialization;
  - converges toward the existing exact signaling-game CFR reference across
    three seeds, exposes the strategy-fusion error of independent per-root
    solves, and merges two real full-card private roots at one reached BTN
    `InfoSetKey`;
  - writes complete-iteration atomic checkpoints containing RNG, shared
    regret/strategy tables, exact prior, per-infoset root support/visits, and
    cumulative statistics; resume requires an externally retained SHA-256 and
    is byte-identical to one-shot execution;
  - binds reachable live solver/action/scoring Python semantics in addition to
    disk source hashes, rejects arbitrary generic adapters for checkpointing,
    and allows only exact-type FullCard or repository-owned canonical reduced
    adapters;
  - remains a supplied-root-set online tabular solve: sampled, non-promoting,
    not reusable as an unseen-root profile, and without an exact exploitability
    claim.
- `ai/tutor/t3_shared_multi_root_strength_gate.py`
  - converts a shared multi-root result into a content-verified strategy
    artifact without trusting producer claims, reconstructing every
    `InfoSetKey`, legal action set, exact root prior, and strategy SHA;
  - locks exactly BB/BTN x visible Joker 0/1/2, requires the production T3-BB
    fixed-point likelihood binding, and binds the candidate to its supplied
    root/range/source scope;
  - rejects candidate-training/holdout/calibration/smoke observation, range,
    build, and seed reuse, so relabeling the same observation with a new opaque
    root ID cannot create fake holdout evidence;
  - independently re-derives candidate and reference raw moments, standard
    errors, paired deltas, payoff, regret, continuation coverage, runtime, and
    seat-swap metrics;
  - is permanently algorithm-validation-only: `passed`, `promotion_eligible`,
    and `full_card_policy_promoted` remain false even when every diagnostic
    threshold passes.
- `ai/tutor/t3_production_holdout_evaluator.py`
  - evaluates every legal root action with exact candidate continuation lookup
    at every later `InfoSetKey`; a missing continuation fails closed and no
    uniform fallback is permitted;
  - evaluates a separate locked uniform-reference policy rather than reusing
    candidate-continuation values, shares physical chance entropy across all
    actions/reference/seat-swap pairs, and separates policy entropy;
  - publishes resumable root x evaluation-seed shards with candidate/reference
    `count/sum/sum_squares`, standard errors, paired deltas, required-continuation
    coverage, canonical source/range bindings, and manifest-last verification;
  - cannot promote a global policy. Independent chance draws can reach unseen
    tabular information sets, which must fail instead of being hidden by a
    fallback.
- `ai/tutor/t3_hu_public_tree_cfr.py`
  - synchronous counterfactual-reach CFR+ over a recursive finite public tree;
  - own-reach-weighted average strategy and action-order invariance;
  - exhaustive pure infoset-policy best responses, NashConv, exploitability,
    and an explicit strategy-fusion diagnostic;
  - commitment-checked physical Rust T4 vectors compile into all-action leaves
    without preselection.
- `ai/config/promotion_gate_v1.json`
  - retained as the first locked threshold draft;
  - superseded for real evidence by v2 because v1 accepted self-asserted metrics
    and did not bind hashes to embedded content.
- `ai/config/promotion_gate_v2.json`, `ai/tutor/promotion_gate_v2.py`, and
  `ai/tutor/m2_promotion_evidence.py`
  - derive every thresholded metric from embedded raw run records;
  - bind source, action, infoset, fixture, chance, range, solver, and raw
    measurement manifests to the artifact hashes;
  - pass all BB/BTN x Joker 0/1/2 reduced strata while explicitly retaining
    `full_card_policy_promoted=false` and `hu_exact=false`.
- `ai/reports/m2_promotion_gate_v2_20260713/`
  - saved evidence and gate result; save/readback validation passes;
  - maximum exploitability `0.000152017689331152`;
  - all-leaf Python/Rust difference, order-replay policy TV, and independent
    best-response re-evaluation residual are zero.
- `ai/tutor/t3_t4_public_root_mixture.py`
  - compiles only public context plus an explicitly declared finite set of
    counterfactual actor-private types into real FullCard multi-root entries;
  - never accepts the actor's actual private hand as a compiler argument;
  - binds exact priors, ranges, adapters, rules, sources, and live runtime, but
    intentionally reports `production_sampling_ready=false` and cannot claim
    exhaustive full-deck support.
- `ai/tutor/t3_t4_public_online_resolve.py`
  - solves the whole compiled mixture before an actual `InfoSetKey` is accepted;
  - allows actual-hand policy selection only after an exact compiled-member
    match, and binds the solver result back to the current prior/range/adapter;
  - freshly validates every strategy/regret action support, checkpoint trust,
    fixed nonpromotion flags, and a transitive live runtime semantic graph.
- `ai/tutor/t3_t4_infoset_encoder.py`
  - losslessly round-trips the full public history, acting-player recall,
    current draw, boards, phase, actor, and turn in a fixed 3313-dimensional
    binary vector;
  - leaves the legacy 522-dimensional model path and serving defaults unchanged.
- `ai/tutor/t3_t4_distillation_teacher.py`
  - assigns train/dev/test from an immutable pre-label full-deal commitment and
    keeps all descendants, private types, seeds, seat swaps, and suit variants
    in that same split group;
  - writes bounded content-addressed v2 shards and verifies manifest-last,
    lineage ownership, hashes, gaps, duplicates, orphans, and hidden omission;
  - separates sampled MCCFR provenance from seedless/sampleless T4 BTN exact
    provenance. It is a teacher contract, not a trained or promoted policy.
- `ai/tutor/t4_btn_exact_resolver.py`
  - bypasses MCCFR only for BTN `t4_second`, where BB is complete and every
    legal BTN action can be terminal-scored directly;
  - returns the exact BTN-perspective argmax and binds Joker scoring, live FL EV,
    source files, and transitive runtime semantics. T3 and T4 BB still require
    public-belief solving.

## Next implementation phases

1. Finish the pre-registered natural full-private trace collection. The
   24,000-root targeted Joker challenge collection is complete; the
   challenge direct-logit evaluation is independently fresh-verified at 24
   shards and 96,000 rows. The 134,057-root natural collection remains in
   progress. Then evaluate the natural immutable shard tree and fit the locked
   observed-action calibration.
2. Calibrate or retrain the all-seat T1/T2 behavior policy and pass the M3 range
   gate. Uniform, legacy, and uncalibrated ranking-logit priors remain wiring or
   sensitivity baselines and cannot pass promotion.
3. Replace the now-validated finite explicit public-root compiler input with a
   production, calibrated, full-deck-compatible private-type generator. The
   actual private hand must remain absent from mixture construction.
4. Use the validated online wrapper and T4 BTN exact bypass to generate large,
   content-addressed teacher shards on root-family-disjoint splits.
5. Distill a lossless-`InfoSetKey` policy/value/Q model with root-family-disjoint
   train/dev/test partitions. Promote T3/T4 only after its BB/BTN x visible-Joker
   holdout, paired cross-play, OOD fallback, and runtime gates pass.
6. Only then generate T2 continuation targets. A root-scoped tabular profile is
   never a substitute for the generalizing model promotion gate.

Required output metadata includes method/information model, strategy-fusion
flag, range model/hash/ESS, iterations, seed, exploitability estimate,
position/rules/action contracts, FL-config hash, and runtime.
