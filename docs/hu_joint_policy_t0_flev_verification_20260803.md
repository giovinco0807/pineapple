# T0 Both Seats, the FL-EV Correction, and the Verification Campaign (2026-08-01 – 08-03)

Continuation of `hu_joint_policy_backward_curriculum_t2_t1_t0_construction_20260731.md`.
Covers: the T1-first and T0-second models, the whole-game ladder reaching
+1.73/hand, the distillation program, the pruning-safety mechanisms, and the
discovery and correction of a systematic error in the Fantasy Land constant —
the single most consequential finding of the project so far. Governance held:
`ai_profiles.py` pinned at `d2eb0266…b868d3`, docs-only commits, implementation
untracked.

## 1. The ladder as of 08-03

Referee-judged per street (floor | model absolute | gap; floors are
instrument-dependent — cross-street comparisons need both columns):

| Street/seat | Floor | Model | Gap | Referee | Weights |
|---|---|---|---|---|---|
| T4 leaf | ~0 | — | +0.00013 | exact | `1aef64fc` |
| T3 second | +0.0152 | +0.0376 | +0.0224 | 512p | `e9cd7416` |
| T3 first | — | — | +0.0310 | 512p | `36334b4b` |
| T2 second | +0.0593 | +0.1121 | +0.0528 | 512p | `996c430e` |
| T2 first | +0.0863 | +0.1218 | +0.0356 | 512p | `0434a40d` |
| T1 second | +0.0572 | +0.0993 | +0.0421 | 1024p | `9a9cc514` |
| T1 first | +0.0715 | +0.0949 | +0.0233 | 1024p | `ce9cfc01` |
| T0 second | +0.1417 | +0.1573 | **+0.0141 [−0.0114, +0.0397]** | 1024p 64/48 | `e0ba6519` |

T0-second's gap spans zero — statistically indistinguishable from its teacher
— but its floor is the highest measured, so the honest reading is "as
well-fitted as any street" rather than "closest to optimal". The
floor-independent number is decisive regardless: production charged +0.454 per
opening vs the model's +0.157, paired recovery **+0.299 [+0.253, +0.346]**.

### Whole-game ladder (3,008 paired seat-swapped hands per rung, independent seed bases)

| Assembly | Mean/hand | 95% CI |
|---|---|---|
| T2-second only [980M] | +0.7114 | [+0.48, +0.95] |
| T2 both [985M] | +1.0685 | [+0.79, +1.35] |
| + T1 both [987M] | +0.9137 | [+0.57, +1.25] |
| **+ T0-second [983M]** | **+1.7258** | **[+1.32, +2.13]** |

The T1 rung's non-transfer (distribution shift eating per-decision gains) did
not repeat at T0: the uplift over the prior rung is z=+3.01. Production now
plays only T0-first.

## 2. The FL-EV arc — a shared-constant error invisible to every internal gate

The chronology matters because the discovery was driven from outside the
verification system:

1. A reviewer question ("how is FL entry valued?") exposed that `fl_ev` is a
   constant added at terminal scoring — measured once (June 12) and shared by
   teacher and referee alike. **A shared constant's error cannot appear in any
   gap or gate this project computes** — teacher and judge err identically.
2. External anchor: Oleg's solver publishes 9.35/9.15 (button/UTG) for the
   comparable variant, ~1.0 below our 10.227 — 3-4σ beyond our CI.
3. Sensitivity probe (32 referee positions, same seed, fl_ev ∈ {9.25, 10.227,
   11.2}): **47-59% of T0 teacher argmaxes flip** across the range. The
   constant is first-order for opening decisions.
4. Re-estimation with the current assembly (10,000 hands, std err 0.119):
   **fl_ev(14) = 9.109 [8.886, 9.346]** — inside Oleg's band (UTG 9.15).
5. Root cause of June's 10.227, proven by a control arm that rebuilt June's
   exact chain inside the new harness (reproduced 10.251 on shared decks):
   **the June estimator's non-FL opponent silently played a two-generations-old
   solo chain** (every heads-up branch was guarded on an argument the harness
   passed as None), fouling 38.9% of hands vs the current assembly's 24.8%.
   The entire gap is opponent strength; conditional-on-surviving royalties
   match within a few percent.
6. Also found: June's fixed-point iteration never converged (alternating
   iterates; map slope ≈ stay − entry ≈ −0.14); the new estimate reads the
   crossing of a measured-linear V(fl_ev) instead. A frozen-weights loop test
   (arm D, engine told 8.75) moved the fixed point +0.006 — the one loop that
   cannot be closed (weights trained under 10.227) is worth <0.02.
7. Caveats recorded: both June's and the new arms play the non-FL side
   *unaware* it faces FL (geometry rejects concealed-FL observations), which
   biases the value upward — 9.109 is a ceiling. Seat-conditionality is real
   (arm C, FL on the button: 9.776 [9.40, 10.17], but with a weaker opponent
   in that orientation — an upper bound). Seat alternation means a
   single-constant ideal is the two-seat average; resolution deferred to the
   definitive measurement (§6).

**Adoption** (08-03): `configs/fl_ev_regular_v3_direct2.json` = 9.109 with a
full provenance block; `hu_infoset` now reads the config (the old carrier was
a hardcoded literal — there is now one source of truth); 91 occurrences of the
old literal swept and classified; propagation proven end-to-end (37/37 checks,
engine terminal delta float-exact); datasets labelled under the old constant
are now **rejected by the training loaders by design**. June's config file and
run records are preserved untouched. Known debris: six Rust golden tests
invalidated by the constant change (fix queued; proven unrelated to concurrent
work by a revert-only control build), and a frozen Windows audit binary
(`hu_rl_scalar_trace.exe`, built 07-22) that needs a re-pin.

Consequences: T0-first labels (not yet burned — deliberately held) will use
9.109 from the start. Everything already trained encodes 10.227 and is
scheduled for relabeling in M7. The whole-game ladder is unaffected as a
comparison (both sides scored under the same constant).

## 3. The distillation program (fast rollout policies)

Teacher rollouts spend ~92% of their time encoding features for nested model
decisions (profiled). Distilled clones — coarse features (histogram caps
4000→400, joint draws 512→64), tiny nets trained to imitate the full model's
choices, used *only inside rollouts* (decide/root paths stay full precision) —
recover most of it. Every clone passes an argmax/EV gate against the full
teacher before adoption.

| Clone | Imitation top-1/top-3 | Gate | Effect |
|---|---|---|---|
| fast_t1_first + fast_t1_second | 70.4/93.7, 72.0/94.4 | excess regret −0.019 vs floor +1.218 → PASS | evaluate_t1 2.8x |
| fast_t0_second | 70.5/93.7 | running (36 evals, setsid-protected) | T0-first eval 120.8s → 22.8s (**5.31x**) measured |

Also landed: Pascal-table combination unranking (~1.06-1.10x, output-identical),
LTO confirmed already on, dead-cache profiling (fingerprints 0.3-0.6% — left
alone; the 0% cross-particle reuse finding pinned by test). The honest failure:
the Finish-representation rewrite was measured to a ceiling of 1.17x and
correctly abandoned under its 1.3x contract.

## 4. Pruning safety (the 232-action fan)

The two-stage prefilter (32p over all → 256p over top-48) was validated on
behavior-root distributions: exact-best survival 19/20, and the two-stage-vs-
single-stage disagreement (+0.232) sits *below* the single-stage teacher's own
2-seed floor (+0.352) — pruning loss is statistically zero there. Because that
guarantee is distribution-local, two standing mechanisms now exist (both
opt-in, byte-identical when off, pinned by a pre-change fixture):

- **Adaptive beam** (`prefilter_margin`): the keep boundary extends while the
  coarse-score gap is inside the noise margin (re-anchoring, capped at 2×keep).
- **Standing audit** (`audit_full_every`): a fingerprint-hash cadence runs the
  full single-stage evaluation beside the staged one and stores the comparison
  verbatim in the position file. Fired on its first rehearsal position (toy
  settings) — the instrument works. Fleet plans carry margin 2.4 / audit 1/500
  (labels) and 1/200 (referee). Target, after ~1,000 accumulated audits:
  a resolved "prune loss < 0.02/hand" claim (the bar Oleg's FAQ implies for
  their own 8-20-candidate heuristic beam).

A cross-stage label finding worth keeping: the 32p prefilter scores are biased
**+1.21 high** relative to 256p rescores (1.35M-pair measurement). The
T0-second trainer de-biases cross-stage ranking pairs accordingly (raw /
fine-only ablations trained as controls; end-to-end indistinguishable at
n=2.5k, adopted on the pair-level evidence).

## 5. Verification campaign (what was checked beyond the referees)

- **T4 leaf under distribution shift**: 2,000 challenger-reached vs 2,000
  production-reached T4-first positions, exact ground truth. Unique-best
  agreement 98.6% vs 98.2%; EV given up +0.00105 vs +0.00056 (difference
  p=0.266). The leaf's acceptance number (+0.00013) was mildly optimistic for
  play in general (both arms ~4x above it) but equally for both — no shift
  failure. The tower's foundation holds.
- **T3-second under challenger distribution** (earlier): gap ratio 0.45x — no
  cascade urgency.
- **External cross-validation (Oleg)**: T3 divergence case — model = teacher #1
  = Oleg vs production 5th (−2.65); T0-second demo — model = Oleg exactly,
  teacher #1 within noise of it; FL values within each other's bands after our
  correction. Also learned from their FAQ: their 5-card list is itself an
  8-20-candidate heuristic beam with claimed <0.02 suboptimality — tail-rank
  cross-solver comparisons are beam-vs-beam and should be read top-5 only.
- **GPU training** (RTX 2060 SUPER via WSL CUDA): 1.5 s/epoch on 5.8M rows —
  17x the uncontended CPU number; whole-tensor-resident + manual batching.
  The agent's own audit noted its switch decision was right on outcome but
  made on a contended (invalid) CPU measurement — recorded as a process error.

## 6. Roadmap changes

- **M6 redefined (user design)**: one 100k-hand full-format run **with FL
  actually played** (FL side = `regular_fl_solver`, non-FL side = current
  policies) producing three outputs at once: G3 verdict at ±0.16 resolution,
  the definitive fl_ev (both seats, chains, both-FL organically), and vs-FL
  position data seeding M11. Requires the FL-play harness (shared with M11 and
  the webapp).
- **M7 additions**: label-generation reuse across rounds where shift is small;
  fl_ev re-measured every round; architecture ablation (current MLP vs larger
  vs raw-input) after 1024p labels; the raw-input net reframed as a *ceiling
  diagnostic* (one training run distinguishing feature-ceiling from
  label-noise-ceiling), triggered by data-scaling flatline.
- **M8 book**: two-pass (all 134,459 canonical classes at standard precision,
  contested ~30-40% re-raced deep), ~2-3 fleet days post-M7, verified by a
  1,000-class 2-seed sample; placed after cascade to avoid double-solving.
- **M10 3-way**: reframed as a different game — zero-sum grounding for the gap
  metric is lost; evaluation moves to population round-robins; rollout
  opponent diversity required to avoid collusion-shaped fixed points
  (Pluribus precedent: practical strength without equilibrium guarantees).
- **M11 vs-FL models**: opponent-FL streets are unmodeled today (the vs-FL
  play in estimates is approximate); planned post-M7 with the FL solver
  distilled for rollouts.
- **M9 API**: own-platform real-time solver; stateless observation-in/EV-out
  protocol (already the engine's shape), precision as a request parameter,
  N-opponent observation schema from day one; RTA use excluded by ToS.

## 7. Fleet operations record

Runs this period: `t1first-256p-r1` (3 cycles, incl. one full-fleet
preemption; manifest resume proven live: "54/98 to generate"), `t1firstref`,
`t0-256p-r1` (58×c4-standard-8 = 464 cores; babysitter did 2 auto-resumes and
auto-download unattended), `t0ref-1024p-r1`, `t0miniref`. The heartbeat-412
parent-killer was fixed (attempt-scoped names) and has not recurred. Straggler
policy: cut the tail and backfill locally (~100-position tails cost an hour of
fleet otherwise). Quota reality: PREEMPTIBLE_CPUS limit was 468 all along
(March's "128" was stale memory); c4-standard-8 × 58 shards saturates it;
a 1024 preference is filed and pending; instance quota (96/region) caps the
granted future at ~768 cores.

Cumulative compute through 08-03: ~20,000 core-hours ≈ $400-1,000 at spot-price
uncertainty (console reconciliation still pending).

## 8. Incident log (this period)

- int64-in-JSON crash killed a chained pipeline post-extraction (1h31m lost);
  fixed at source + shared serializer + metadata-cannot-kill-weights guard.
- Two agents slept at "waiting" checkpoints (the resume-on-notification trap);
  both woken by status probes. A gate run lost 70 min to harness process-group
  reaping; setsid + idempotent workers now standard.
- Three agent rule-violation self-reports (one heredoc, one self-matching
  pkill, one premature GPU-switch justification) — all disclosed unprompted,
  none affecting deliverables.
- The T0-first fleet was deliberately held by the user pending the fl_ev
  question — in hindsight the single best call of the period: the labels
  would have been burned at the wrong constant.

## 9. State at close (08-03)

Running: fast_t0_second gate (last prerequisite). Queued behind it: package
rebuild (picks up fl_ev 9.109 + new engine), T0-first plan generation
(margin 2.4, audit 1/500, eight models + three fast clones), fast-mode
rehearsal, fleet ignition (~17h labels + ~5h referee), then extract/train/
judge → M5 complete. Open: six Rust goldens (fl_ev re-pin), Windows audit
binary rebuild, arm-C re-run for the button-seat FL value once T0-first lands.
