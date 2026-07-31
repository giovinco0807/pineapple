# Backward Curriculum Construction: T2 Both Seats, T1 Both Seats, T0 Engine (2026-07-29 – 07-31)

Working notes for the three-day construction sprint that took the learned-evaluator
ladder from "T3 both seats + T4" to "every street except the T0 models
themselves". All implementation lives in the intentionally untracked worktree
(`src/ofc_regular`, `rust/hu_m3_engine`, scratchpads); this document is the
committed record of what was built, how it was tested, and what the referees
measured. Governance held throughout: named baselines untouched, `current`
untouched, `ai_profiles.py` pinned at `d2eb0266…b868d3` and verified at this
commit.

## 1. The model ladder as it now stands

One row per learned evaluator. "Gap" is referee-judged: two disjoint-seed
teacher runs per position, chooser charged by the seed that had no part in the
choice, gap = model regret − teacher self-disagreement floor. Floors differ by
street and by referee strength, so gaps are comparable only with their floors
alongside; absolute charged regret is listed for that reason.

| Street/seat | Labels | Referee | Floor | Model abs. | Gap | Weights (sha256 prefix) |
|---|---|---|---|---|---|---|
| T4 (leaf) | exhaustive | — | ~0 | — | +0.00013 | t4_model_v5 `1aef64fc` |
| T3 second | 996K exact | 512p×2 | +0.0152 | +0.0376 | +0.0224 | t3_model_v2 `e9cd7416` |
| T3 first | 25K @512p | 512p×2 | — | — | +0.0310 | t3first_model_v1 `36334b4b` |
| T2 second | 25K @128p | 512p×2 | +0.0593 | +0.1121 | +0.0528 | t2_model_v1 `996c430e` |
| T2 first | 25K @128p | 512p×2 | +0.0863 | +0.1218 | +0.0356 | t2first_model_v1 `0434a40d` |
| T1 second | 25K @256p | 1024p×2 | +0.0572 | +0.0993 | +0.0421 | t1_model_v1 `9a9cc514` |
| T1 first | 25K @256p (labels complete 07-31) | 1024p×2 planned | — | training pending | — | — |
| T0 both | engine complete, labels not yet run | — | — | — | — | — |

Lesson recorded the hard way: a small gap over a high floor is not the same
result as a small gap over a low floor. The T2-first gap (+0.0356) initially
read as "better than T2-second"; the seat-difficulty comparison showed the
first-seat teacher is simply noisier (same top-two spread, 1.45x floor), and in
absolute regret T2-first is slightly worse. Cross-street claims should quote
absolute regret; within-street improvement claims should quote the gap.

### Versus the production policy (paired, same positions, same referee)

| Street/seat | Production charged | Model advantage |
|---|---|---|
| T2 second | +0.567 | **+0.445** [+0.371, +0.519] |
| T1 second | +0.363 | **+0.264** [+0.211, +0.317] |

### Whole-game (paired seat-swapped decks, challenger vs production chain)

| Assembly | Hands | Mean/hand | 95% CI |
|---|---|---|---|
| + T2 second, T3 both, T4 (seed base 980M) | 3,008 | +0.7114 | [+0.4757, +0.9471] |
| + T2 both, T3 both, T4 (seed base 985M) | 3,008 | **+1.0685** | [+0.7920, +1.3450] |

Independent samples (different seed bases); the improvement is directional
(z≈1.9), both runs decisively positive.

### Distribution shift (the cascade question)

1,200 T3-second positions captured from the challenger's own play (learned
T2-second upstream), judged by 512p×2 referees against the production-reached
reference: gap +0.0102 vs +0.0224 — ratio 0.45x, far under the 2x alarm
threshold. The floor doubled while the disagreement *rate* stayed equal: the
challenger reaches sharper positions (top-two EV spread +34%), not more
confused ones. Conclusion adopted: cascade retraining stays scheduled as M7,
not pulled forward.

## 2. Engine work (rust/hu_m3_engine, all untracked; test ladder 68 → 86)

- **T2 both seats** (`evaluate_t2`, `rollout_t2_second/first`, 12/15 particle
  cards): second seat requires three learned models, first seat four; nested
  replies via fingerprint-cached `locked_*_action` deciders; no sampled
  fallback by design.
- **decide** (in-play path): per-street arms now cover T1 second, T2 both, T3
  both, T4 exact. Shared decision body `learned_t2_action` serves T1-second,
  T2-second, T2-first, T1-first (geometry guards differ; behavior of existing
  seats pinned byte-identical when each new seat landed).
- **T1 both seats** (`evaluate_t1`, 18/21 particle cards, 6/7 nested
  decisions): second seat five models, first seat six. Geometry chain validated
  end-to-end (hero/opponent discard ladders 0→3).
- **T0 both seats** (`evaluate_t0`, `rollout_t0_second/first`, 24/29 particle
  cards, 8/9 nested decisions): second seat seven models, first seat eight.
  232 initial actions confirmed hand-independent (243 − 11 top-overflow).
  **Two-stage prefilter** added for the 232-action fan-out: stage 1 scores all
  actions at `prefilter_samples`, stage 2 rescores the top `prefilter_keep`
  with the full particle set on an independently derived seed; every action
  keeps a row tagged `stage: 1|2`; zeroed fields = single-stage (the exact
  path the pruning validation will compare against).
- **Belief widening for T0** — the one deliberate boundary change, signed off
  explicitly: `supports_hidden_belief`, particle shape validation and
  `belief_geometry` admit exactly the two T0 geometries (0/0 and 0/5, five
  dealt, zero discards). Street-aware only: T1–T4 validation byte-identical,
  malformed T0 shapes refused, dealt-5-at-T1 refused — all pinned by boundary
  tests from both sides.
- **Free-outlook generalization 4→6→8 open slots** with Python reference
  parity at every step: fixtures 681 rows (4-slot, worst 1.94e-7), 216 rows
  (6-slot, 1.27e-7), 201 rows (8-slot, 1.46e-7). The 8-slot step forced two
  affordability fixes proven to be no-ops (rank-unranked strided combinations
  instead of materializing C(39,8)=61.5M; completion-view caching), plus a
  part-set bound raise (20→70=C(8,4)) caught before it could overrun.
- **Performance, all output-preserving with parity gates:**
  - RowMemo table optimization: free outlook 9.4ms → 0.59ms first-run
    (10.7x), joint block ~17x; T2-first teacher 17min → 76s/position.
  - Decision-scoped `FreeOutlookCache` sharing across candidate boards:
    nested T2 decisions 2.3–3.5x faster; `evaluate_t1` 197s → ~80s/position
    at 128p. Cross-particle reuse measured at exactly 0.0% and abandoned
    (kept as an ignored test, not a claim).
- Suite at time of writing: **86 passed / 0 failed** across 12 binaries
  (timing-assertion test `the_learned_leaf_is_the_cheaper_of_the_two` is
  load-sensitive; passes in isolation at 1.7x).

## 3. Teacher noise and the particle policy

2-seed probes before each fleet, per the sequential-sizing policy:

| Teacher | Particles | Disagree | Floor |
|---|---|---|---|
| T1 second | 128 | 37.5% | +0.335 |
| T1 second (referee) | 1024 | 20.9% | +0.0572 |

First-seat teachers are systematically noisier (longer rollouts, one extra
nested street): T2-first floor 1.45x T2-second at equal particles with equal
decision sharpness. Consequence, adopted at user direction: labels 128p→**256p**
from T1 on, referees 512p→**1024p**. The 1024p referee floor landing below the
512p T2 floors is what made the T1 numbers readable at all.

## 4. Fleet infrastructure (six production runs, ~2,000 VM-hours, ≈$100–130 total)

Runs: `t2first-128p-r1`, `t2firstref-512p-r1`, `t1-256p-r1` (3 cycles),
`t1ref-1024p-r1`, `t1first-256p-r1` (3 cycles, finishing), `t1firstref-1024p-r1`
(planned). All SPOT, create-only GCS, content-bindings with generation pinning,
plan/receipt/stage/execute/poll/receive/cleanup lifecycle.

Defects found in production and fixed, each with a rehearsal or live proof:

1. **Async insert failure via wrong service-account name** — 32 inserts
   accepted then failed server-side; zero cost; relaunched with
   `ofc-labelgen-worker@ofc-solver-485418.iam.gserviceaccount.com`.
2. **Recreated-instance resume defect** — workers resumed from *local* disk
   only; a cleanup+recreate after mass preemption regenerated whole shards and
   died colliding with prior uploads. Fix: startup lists the shard's GCS
   objects once into a manifest, workers take `--existing-manifest` (missing
   path is fatal, entries feed skip/count/SHARD_DONE through the single
   `already_done` predicate); per-file 412s benign. Proven live: cycle 2 of
   `t1first` resumed "54/98 to generate" per stride.
3. **Heartbeat-name 412 killed the parent loop** — heartbeats were tick-named;
   a recreated instance restarts ticks and collides with the prior attempt's
   objects. Third 412 site (checkpoints and files were already fixed). Fix:
   attempt-id in the object name plus benign-412 catch. The empty-listing
   `set -euo pipefail` abort (grep exits 1 on no match) was caught by the
   parser tests before it could kill any boot.
4. **Throughput was quota-limited by stale knowledge** — the remembered
   128-vCPU cap was March-era; actual `PREEMPTIBLE_CPUS` limit is 468.
   `--worker-count` exposed on the planner CLI (was internal-only), fleets
   moved to c4-standard-8 × 8 workers = 256 cores mid-run under the same run
   name (receipt regenerated, stage receipt carried, resume proved).
   Quota preference `pokerhu-spot-cpus-asia-ne1-256-20260514` updated to
   preferred 1024 (reconciling at time of writing). Instance quota (96/region)
   makes 96 × c4-standard-8 = 768 cores the practical ceiling if granted.

Known-stale note: the seven pre-T1-first worker plans in
`~/ofc-labelgen/package/` pin older engine digests and will be refused by the
current runtime — deliberate (provenance of already-generated labels); any
rerun takes a `-r2` job id, never an edit.

## 5. Training pipeline notes

- Rust batch extractor (`labelgen_feature_dump`, 314x Python, exact A/B
  parity) extended per street by widening one match arm; every extraction
  sanity-checks width 168, one row per legal action against Python
  `generate_turn_actions`, zero NaN/inf.
- Packing-width lesson, twice: the trainer's default width 24 silently
  truncates 27-action streets. T2-first: w24 vs w27 both trained, w24 won on
  the untruncated validation rescore. T1-second: w24 truncated 54,378 rows and
  **lost** to w27; w27 shipped. The width check is now a standing step.
- A silent-fallback bug in `load_model_bundle` (unrequested profile → default
  turn-1 policy instead of an error) was found during the T1 vs-production
  comparison and fixed with an assertion; numbers unchanged, validity restored.

## 6. Remaining work

1. T1-first: extract → train (width check) → 1024p referee → decide arm is
   already live; integrate and re-benchmark.
2. T0 second then first: cost probe, two-stage prune validation against the
   single-stage exact path (50 positions), 25K labels each at 256p+prefilter,
   1024p referees, train, judge. User decision recorded: full-accuracy T0
   build, no shortcut on coverage (232-action space; ~43K suit-isomorphism
   classes in C(52,5)).
3. M6: formal G3 whole-game benchmark with the complete assembly.
4. M7: cascade Expert Iteration (relabel on challenger distribution; fold the
   256p particle upgrade into the same fleets).
5. Deferred: receive-phase token_provider (bulk `gcloud storage cp` workaround
   stands), T2-second decide latency (~10ms post-cache vs G4 <1ms), option C
   (versioning the rust crate) still open.
