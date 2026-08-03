# Joker-Variant Transfer Assessment (2026-08-03)

Question answered here: does the regular-track methodology (backward
curriculum, particle teachers, referee verification, distillation, fleet ops)
transfer to the 54-card Joker variant — and what exactly remains to build,
given what already exists on both tracks?

Grounding documents: the accepted decision record
`ai/docs/engine_unification_decision_20260728.md` (B-phased, owner-approved
07-28) and its inventory; commit `b3f5c8f` (Joker port phase 1); the joker
track's live commit stream through 08-03.

## 1. What already exists (verified in-repo, not assumed)

### In this repository (unification phase 1, landed 07-28)

- `src/ofc_regular/joker_rules.py` — 54-card `joker_chain` RuleSet: physical
  X1/X2 jokers, chain FL schedule **QQ=14 / KK=15 / AA=16 / trips=17** (unlike
  regular's all-14). The `RuleSet` dataclass already parameterizes
  `include_jokers` through to fl_ev/solver config.
- `src/ofc_regular/joker_evaluator.py` — exhaustive distinct-substitution
  joker evaluation with bottom-up row constraint.
- `tests/test_joker_evaluator_parity.py` — 11/11 parity vs `ai/engine`
  (320 boards across 0/1/2-joker strata, pairwise line results, wheel,
  double-joker constraint, FL chain counts). REGULAR_RULES untouched.

### On the joker track (`ai/`), moving fast in a parallel workstream

Commits through 08-03 show a substantial **vs-FL solver program** already
achieving referee-grade numbers on the joker variant:

- `fl_solver` v3: table-driven exact FL search at production latency.
- T4-vs-FL v2: corr 0.959, regret 0.006 on a 120k library.
- T3-vs-FL v2: gap 0.0047 (later 0.0036 re-verified) against a 0.099 referee
  floor — the same charge-by-disjoint-seed methodology.
- FL-vs-FL convolution (14v14 measured, symmetry-checked); chain libraries for
  15/16/17; a resumable library builder and an "M-C fixed-point" FL-value
  solver — i.e. the joker track is solving the **FL-value fixed point and the
  vs-FL streets first**, the exact area the regular track has deferred to M11.
- Older but native: `ofc_core` (joker Card type), prob_engine, exact
  T3/T4 solvers, CFR/backward crates, the M3 MCCFR+ solver line.

### Regular-track assets the joker variant would inherit via unification

Everything built in the last week is rules-parameterized or joker-agnostic in
structure: two-stage prefilter + adaptive beam + standing audit, fingerprinted
locked deciders, distillation recipe (three gated clones so far), the referee
harnesses, fleet lifecycle (manifest resume, babysitter, 58-shard/464-core
shapes), GPU training path, fl_ev estimation harness, and 2,379+ regression
tests worth of engine discipline.

## 2. Does the method transfer? Yes — with one structural advantage over 3-way

Joker HU remains **two-player zero-sum**, so the gap metric keeps its
theoretical grounding (best-response distance bounds exploitation). That makes
joker a *closer* port than 3-way (M10), where the metric itself has to be
replaced by population evaluation.

## 3. What actually has to be built (delta list)

| Layer | Work | Notes |
|---|---|---|
| Engine cards | 52-fixed arrays (`CARD_TOKENS: [&str; 52]`, `ALL_CARDS`) → 54; joker-aware `partial_value` | The decision record's phase 2; `hu_m3_engine` is 52-locked at the index level today |
| Evaluator | Port the exhaustive-substitution evaluator to the Rust hot path | Python reference exists with parity tests — the port has its golden target ready |
| Outlooks/features | Unknown set gains X1/X2; row histograms and joint blocks must price "a joker may complete this row" | Same combinatorial machinery; ~2-3x per-completion cost expected (substitution resolution); the coarse/fast variants absorb most of it in rollouts |
| FL machinery | fl_ev becomes a **4-entry chain table** {14,15,16,17} + stay; the joker track's M-C fixed-point + FL-vs-FL convolution work is exactly this and should be adopted rather than rebuilt | Regular's single-entry lesson applies doubly: measure with the strongest available opponent, seat-aware, converged |
| Teachers | Same evaluate_t4→t0 ladder over the joker engine | No new design; re-run the construction recipe |
| Labels/models | Full per-street relabel + retrain (weights do not transfer — feature semantics change) | Fleet + GPU paths as-is |

## 4. The weak-production problem, and the bootstrap that defuses it

The joker track's serving baseline is the legacy MCTS+VN system — much weaker
than regular's production chain. Two implications:

1. **Round-1 label distribution risk**: behavior roots sampled from weak play
   are less representative. Mitigation: plan the cascade (M7-equivalent) as
   two rounds from the start, and
2. **Bootstrap the behavior policy from regular models — partially**: the
   joker-free rate is P = C(52,17)/C(54,17) = (37×36)/(54×53) = **46.5%** of
   own 17-card hands (so **53.5% contain a joker**, and **86.7% of games**
   see one across the 34 dealt cards). *(This paragraph originally claimed
   ~7.2% joker exposure — an arithmetic error caught in cross-track review on
   08-03; the formula was right, the evaluation was not.)* A hybrid root
   policy — regular models on joker-free hands, legacy/heuristic on joker
   hands — therefore covers roughly half of round-1 hands well, not "almost
   all", and the quality of the joker-hand side matters correspondingly more.
   Two further honesty notes: even joker-free positions are not strictly
   regular-equivalent (the unknown set still contains two jokers, which
   shifts every outlook), which is acceptable for ROOT GENERATION (roots need
   realism, not optimality) but disqualifies naive regular-model *serving* on
   joker tables; and the joker track's own T3-vs-FL data confirms all three
   joker strata (0/1/2) carry substantial mass.

Upside of the weak baseline: expected whole-game gains are larger than
regular's +1.73/hand, and the commercial story (one API covering both
variants) strengthens.

## 5. Sequencing options

The unification decision already binds: joker M4+ teacher generation waits for
unified-engine acceptance gates 1-4. Beyond that, two orderings:

- **Option J-first**: after regular M5-M7, do joker HU before 3-way. Pros:
  zero-sum metric intact; unification gates already in motion; the joker
  track's FL solvers slot into regular's M11 gap (both variants gain vs-FL at
  once). Cons: 3-way (platform requirement) waits ~2-3 more weeks.
- **Option 3way-first** (current roadmap): platform completeness first; joker
  inherits an even more mature toolchain later. Cons: population-evaluation
  design is new risk taken earlier.

Estimate for joker HU either way, with mature machinery: **engine phase 2-4
~1 week (agent work + parity gates), construction ~1.5-2 weeks,
compute ~$500-1,500** — same envelope as 3-way, lower methodological risk.
Cross-track review note (08-03): the vs-FL half of that construction estimate
is likely conservative — the joker track's T2-vs-FL playout labeler landed
overnight (commit `e228adc`: T3-model moves + exact-library scoring), so the
downward vs-FL construction is already underway on their side.

A pragmatic note: the two tracks are converging on the same problems from
opposite ends (regular built the streets and deferred FL; joker built FL and
lacks the street ladder). The unification decision's B-phased path is what
lets each side adopt the other's finished half instead of rebuilding it.

## 6. Open questions for the owner

1. J-first vs 3way-first after M7 (this document deliberately does not decide).
2. Whether the joker FL chain values should be re-measured with the regular
   fl_ev-v2 harness once unified, or the M-C fixed-point numbers adopted as-is
   with a cross-check (recommended: cross-check first — the June lesson says
   the opponent's strength inside the estimator is the dominant term).
   Shared precondition flagged in cross-track review: the FL libraries'
   placement objective currently uses a fixed +100 stay term; substituting the
   measured FL_EV requires regenerating the libraries (~1 hour under
   fl_solver v3) — this gates BOTH tracks' adoption of measured chain values.
3. Whether joker-variant models share the API surface at launch (schema
   already rules-parameterized; the decision is product, not technical).
