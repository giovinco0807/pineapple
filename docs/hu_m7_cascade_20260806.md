# M7: the bottom-up cascade relabel at fl_ev 9.6

Status: operations record, opened 2026-08-06, written as the milestone runs.
This is the companion deliverable to the models themselves — the models say
what was chosen, this says what was measured and why the choice followed.

M7 relabels the joint-exact teacher corpora street by street, from the bottom
(T4) upward, at the v4 Fantasyland constant `fl_ev = 9.6`. It exists because
the constant changed: a corpus labelled at one constant and consumed by a
teacher searching at another is not a teacher, it is a slow-acting bug, and the
whole point of a cascade is that each street's labels are resolved by the
already-relabelled street beneath it.

## What was inherited, and what of it was re-verified

The transition that made the cascade possible was completed before this
record opens. Its claims and their re-verification on 2026-08-06:

| inherited claim | re-verified | value |
| --- | --- | --- |
| v4 config live in the tree | yes, byte digest | `configs/fl_ev_regular_v4_selfplay.json` sha `0cfe78f7…3fea`, `fl_ev = {14: 9.6}` |
| engine compiled at the new constant | yes, source read | `DEFAULT_FL_EV = 9.6` (infoset.rs:23), `DEFAULT_FL_EV_14 = 9.6` (scoring.rs:20) |
| pinned engine `.so` | yes, byte digest | `e17aa39f797e01e4…3670f` |
| T4 v6 model | yes, byte digest | `t4_model_v6.bin` sha `763b77a025bd3ce3…668c`, EV-given-up 9.95e-5 |
| package ledger agrees with package bytes | yes, recomputed from `runtime.tar.gz` | 4/4 digests agree |
| `ai_profiles.py` untouched | yes, before and after | sha `d2eb0266…868d3` |

Also inherited and taken on the predecessor's evidence rather than re-run:
26 binaries / 208 tests passed / 0 failed across both suites; golden re-pins
with byte-level attribution; the corpus-mixing guard exiting 1 on a constant
mismatch; the worker's 27-plan regression including the `(T3, second)` kind.

## The protocol this milestone runs at every street

Each street gets the same treatment, and the table below is the deliverable
that accumulates from it:

1. **particle probe** — how much evaluation noise is in a label at each
   sample rung, measured as regret against a 4x-heavier reference, paired on
   roots, with an independent particle stream (the self-comparison control is
   run on purpose, because sharing the reference's own stream flatters the
   light rung by roughly 0.6x and that bug has been burned before);
2. **feature / capacity ablation** on the street's existing corpora, baseline
   reproduced first;
3. **verdict** — which of the two axes, if either, actually binds;
4. **plans** written against the v4 package, digests recomputed from the bytes
   the fleet opens;
5. **READY markers** validated through the worker's own `load_plan`;
6. **retrain**, export, and a mirrored match gate per street-seat against the
   current chain, on the standing criterion: lower CI > −0.05 and mean ≥ 0.

## The running diagnosis table

| street / seat | particle verdict | feature verdict | chosen setting | evidence |
| --- | --- | --- | --- | --- |
| T4 | n/a — terminal, exhaustive | n/a | exhaustive | EV-given-up 9.95e-5 |
| T3 / first | **UNDERPOWERED at the full 150 roots** (+0.0012, CI [−0.0075,+0.0109]) — not flat, unresolved | not run | **samples = 1024**, 18,000 labels delivered + verified; retrained to v7, **gate NO_GAIN** (−0.0027, CI [−0.0201,+0.0147]) → **not promoted** | held-out regret −13.3 % but play indistinguishable; ~700 roots needed for a flatness verdict |
| T3 / second | **REAL_GAP** (+0.0184, CI [+0.0055,+0.0334]); rule says SHIP 512, but 512-vs-1024 also excludes zero → **1024 possibly not converged** | not run | **samples = 1024**, 18,000 labels delivered + verified; retrained to v7, **gate ADOPT** (+0.0043, CI [−0.0088,+0.0173]) | criterion met as written; the interval still straddles zero, so this is "no regression", not a measured gain |
| T2 / first | **NO VERDICT YET.** First run executed the LABEL worker (1,000 labels/seat at 512p, 1.4 h) — relaunch READY as `probe1000b` with a `samples: 0` structural guard and a pre-flight check | **capacity NULL** (+1.11 %), **features NULL** (+0.95 % / +0.43 %); `tier1_only` 0.9988 vs 0.2308 ⇒ **redundant, not irrelevant**; width-truncation hypothesis **refuted** | rung awaits the probe | 3 seeds, frozen protocol; shipped labels are 128-particle |
| T2 / second | as above — no verdict, relaunch READY as `probe1000b` | **capacity NULL** (+1.61 %), **features NULL at 6 seeds** (+0.0040 gain vs 0.0097 spread; the 3-seed read of −3.73 % was a favourable draw); row order rebuilt and proved bit-identical | rung awaits the probe | 6 seeds on the borderline arms; shipped labels are 128-particle |
| T1 / first | pending | pending (T1f) | pending | |
| T1 / second | pending | ablation already run = **null** | pending | |
| T0 / first | deferred (diagnosis only) | pending (T0f, local 18.8k corpus) | relabel DEFERRED per owner | |
| T0 / second | deferred (diagnosis only) | pending | relabel DEFERRED per owner | |

## T4 — done, at v6

The terminal street is exhaustive, so it has no sample-rung question: the only
thing to get right is the constant. `t4_model_v6.bin` is the relabel-and-
retrain at 9.6, EV-given-up 9.95e-5 against its own exhaustive reference, with
sidecar and predictions verified. It is the leaf every street above it resolves
through, which is why v5 — fitted on labels carrying the June constant, and two
re-measurements stale — is pinned nowhere in this generation.

## T3 — the top rung purchased rather than the ladder resolved

### What the probe measured

150 roots requested, both seats, rungs 256/512/1024 against a 4096-sample
reference, independent particle streams, shared seed control. At the 75 roots
scored when the T3 plans were written:

| seat | rung | mean regret | 95% CI | agree % | per-label SE | core-s/root |
| --- | --- | --- | --- | --- | --- | --- |
| first | 256 | 0.0145 | [0.0056, 0.0265] | 69.3 | 0.3963 | 15.2 |
| first | 512 | 0.0053 | [0.0028, 0.0082] | 72.0 | 0.3093 | 31.4 |
| first | 1024 | 0.0081 | [0.0037, 0.0133] | 73.3 | 0.2108 | 66.4 |
| second | 256 | 0.0126 | [0.0031, 0.0262] | 86.7 | 0.2997 | 1.3 |
| second | 512 | 0.0091 | [0.0019, 0.0198] | 89.3 | 0.2197 | 2.5 |
| second | 1024 | 0.0038 | [0.0008, 0.0078] | 92.0 | 0.1570 | 5.1 |

Decision statistic (256 vs 1024, paired on roots): first `+0.0065`
CI95 `[−0.0024, +0.0189]`, second `+0.0088` CI95 `[−0.0012, +0.0221]`.

**Both seats return UNDERPOWERED, and the report says so rather than forcing a
branch.** The CI straddles zero, so no real gap is demonstrated; and its upper
end still admits a gap above the 0.0100 meaningful threshold, so flatness is
not demonstrated either. The reason is visible in the raw statistic: median
regret is 0.0000 at every rung, and the mean is carried entirely by the
quarter of roots that disagree with the reference at all. The first seat's
regret means do not even *order* the rungs (512 beats 1024 on the point
estimate) — which is the signature of a statistic dominated by its own noise,
not of a flat ladder.

### Why it was bought instead of measured

Supervisor ruling, executed as given: stop probing, buy the top rung. The
arithmetic behind it is that T3 is cheap. At 1024 samples a first-seat label
costs 66.4 core-s and a second-seat label 5.1, so 18,000 positions per seat is
about 360 core-hours in total — under an hour of wall time on the 58-shard,
464-CPU fleet. Resolving the ladder would cost more roots than the labels are
worth, and the owner directive is that precision decisions err heavy: cost
pressure never picks a config, and the fleet absorbs.

The purchase is **not** justified on the regret ladder, because as shown that
ladder is underpowered and non-monotone. It is justified on the two quantities
that *are* monotone in samples at both seats:

* per-label standard error: first 0.3963 → 0.3093 → 0.2108; second
  0.2997 → 0.2197 → 0.1570;
* agreement with the 4096-sample reference: first 69.3 → 72.0 → 73.3 %;
  second 86.7 → 89.3 → 92.0 %.

Both say the heaviest rung is the least noisy label, which is what a teacher
corpus is bought for.

### Two caveats, recorded rather than hidden

1. The probe table embedded in the plans is **partial** — 75 of 150 roots, the
   run still in flight when the plans were written. It cannot change the
   purchased rung; it can only change confidence in a verdict nothing is
   relying on.
2. The probe ran against the **generation-1** runtime `.so` (sha `4e1284dc…`),
   whose compiled `DEFAULT_FL_EV` predates the v4 switch, with 9.6 supplied
   from the v4 config through the Python scoring context. Rung comparisons are
   paired through the same binary, so the noise-scaling read survives it, but
   this probe is not a v4-binary measurement. **The relabel itself pins the v4
   engine `e17aa39f…`.** The T2 and T1 probes must be run against the v4
   package runtime, not the generation-1 build, and this is the reason.

A third, on power budgeting for the streets above: the probe OOMed at 15
workers × 3.76 GB and the surviving 13 needed roughly 250–930 core-s per root
per stride. T2 and T1 probes get 6–8 workers, and their root counts must be
argued from the power actually needed rather than from what fits.

### What resolving the ladder would actually have cost — the arithmetic

This is the number that makes the ruling defensible, and it is recorded so the
streets above can budget from it instead of guessing. The first seat's paired
`256 vs 1024` CI half-width is 0.01065 at n = 75. CI width falls as
`1/sqrt(n)`, so:

| goal | required half-width | roots needed |
| --- | --- | --- |
| CI excludes zero (a real gap is demonstrated), at the observed mean +0.0065 | < 0.0065 | **~200** |
| CI upper end below the 0.0100 threshold (flatness demonstrated) | < 0.0035 | **~700** |

The run was sized at 150. **It was never going to resolve anything**: 150 roots
is below the ~200 needed for the easier of the two verdicts, and a factor of
4.6 below the ~700 needed to confirm flatness. That is the real lesson, and it
is a sizing error rather than an execution failure — the probe behaved exactly
as designed and reported UNDERPOWERED honestly.

**Standing rule for T2 and T1, from this:** compute the required root count
from the target half-width *before* launching, and either fund it or state on
the record that the street is being bought rather than measured. A probe sized
by what fits in memory answers nothing.

### The probe was terminated, then resumed — and both are on the record

**Attempt 1** was killed on owner directive at 86 roots, correctly: the decision
it fed was closed, T3 ships at 1024 by ruling, and it was holding 13 of the
box's 16 cores against the rehearsal that was gating the fleet launch.

**Attempt 2** was launched at 22:40:03 by `resume_v6.sh` — created by the
coordinator's sweep, **not** by the M7 executor, who killed attempt 1 and did
not restart it. Recorded plainly because provenance that misattributes its own
author is not provenance. It runs 10 strides instead of 15, which is the right
response to the oversubscription that killed the first attempt.

It is allowed to continue under the coordinator's three conditions, all now
satisfied:

**(1) The seed reuse is a deliberate continuation, not a collision.** Written to
`seed_provenance.json` beside the data, and *verified* rather than asserted:

* `t3noise_probe.py:312` builds its work list as
  `[o for o in offsets if not (out / f"root_{o:05d}.json").exists()]` — a scored
  root is never recomputed or overwritten;
* `rung_seed(offset, rung_index) = rung_seed_base + offset*7 + rung_index*3_000_017`
  is a function of the **root offset alone**. Neither stride index nor stride
  count appears, so re-striding 15 → 10 cannot change any root's seeds.
  `reference_seed(offset)` likewise.

Every root offset is therefore scored exactly once, with the seeds it would have
received had attempt 1 run to completion. The two attempts concatenate into one
clean sample; they do not overlap and do not replay each other. Re-spending
61,000,000 / 340,000,000 here is the one case where re-spending a block is
correct — it is the same measurement continuing.

**(2) It yields to the T3 retraining.** Standing priority for this box:
T3 retraining (when labels land) > T2 ablation > this probe > T2 probe pilot.
Kill it with `pkill -f '[t]3noise_probe'` the moment it contends.

**(3) Its verdict is measurement, not decision.** It enters the diagnosis table
in its own row, held separate from the fleet decision. **T3 ships at 1024
regardless of what it eventually says** — the labels were purchased and are
being generated now. If the finished probe disagrees with the purchase, that is
information for T2/T1 sizing and for a future T3 generation, not a retraction.

Data at `/home/wner/ofc-m7/t3noise/out/decision_v6_150/`; `t3noise_report.py
--out <dir>` re-merges whatever is on disk at any moment.

### The probe finished: 150/150 roots, and it does NOT say what a summary of it said

It reached the full 150 roots before being killed when the labels landed.
`report_final_150.json`. **The verdicts are not the reassuring ones, and the
difference matters, so they are recorded verbatim:**

| seat | decision statistic (256 vs 1024, paired) | branch | the rule's own words |
| --- | --- | --- | --- |
| first | **+0.0012** CI95 [−0.0075, +0.0109] | **UNDERPOWERED** | "The CI straddles zero … AND its upper end (+0.0109) still admits a gap above the meaningful threshold (0.0100), so flatness is NOT confirmed with power either." |
| second | **+0.0184** CI95 [+0.0055, +0.0334] | **REAL_GAP** | "SHIP 512 (converged-plus-one) … CAVEAT: 512 vs 1024 ALSO excludes zero (+0.0004), so 512 is not itself converged and the ladder should be extended before 512 is treated as sufficient." |

Two corrections against the summary that reached this desk:

* **The first seat is UNDERPOWERED, not FLAT.** Doubling the roots from 75 to 150
  shrank the point estimate (+0.0065 → +0.0012) but the CI still admits a gap
  above threshold. Calling the 1024 purchase "harmless overkill" at this seat is
  not supported — the data cannot distinguish the rungs, which is a different
  statement from "the rungs are the same". The half-width arithmetic said ~700
  roots for a flatness verdict; 150 was never going to deliver one.
* **The second seat's REAL_GAP does not validate 1024 as sufficient — it points
  the other way.** The rule's output is *SHIP 512*, and the caveat is the real
  finding: `512 vs 1024` also excludes zero (+0.0072, CI [+0.0004, +0.0171]).
  Every step up the ladder is still buying accuracy at the second seat, so
  **1024 may itself not be converged**. The purchase is defensible as the
  heaviest measured rung — but the honest reading is under-bought, not overkill.

Net effect on the shipped corpus: **none, and correctly none.** 1024 is the
heaviest rung anyone measured, the labels are generated, and no verdict here
argues for a lighter one. What it changes is the *next* generation: a T3 v8
should extend the ladder above 1024 at the second seat rather than assume
convergence, and the first seat's rung question remains genuinely open.

### The plans

Written by `m7v4_make_t3_plans.py`, which recomputes every digest from the
members of the package's own `runtime.tar.gz` — never quoting a hash — and
cross-checks them against the ledger before writing anything.

| | first seat | second seat |
| --- | --- | --- |
| plan | `worker_plan_m7v4_t3first_1024p.json` | `worker_plan_m7v4_t3second_1024p.json` |
| plan sha256 | `ed5c07aa9072a761…4963` | `0fb2afbc45ad129d…737c` |
| samples | 1024 | 1024 |
| positions | 18,000 | 18,000 |
| shards | 58 (38 × 310, 20 × 311) | 58 (38 × 310, 20 × 311) |
| `t4_model` | `weights/t4_model_v6.bin` `763b77a0…` | same |
| `t3_second_model` | `weights/t3_model_v2.bin` `e9cd7416…` | **deliberately absent** |
| fl_ev | 9.6 (14 cards) | 9.6 (14 cards) |

The absent field on the second seat is load-bearing, not an omission. Acting
second at T3 the opponent's T3 turn is already behind the teacher, so the
rollout reaches the terminal through two T4 decisions and never consults a T3
second-seat model. The worker **refuses** a `(T3, second)` plan that pins one,
because the engine reports an evaluator as `learned` when its weights were
loaded rather than reached — a pinned-but-unreachable model would have the
provenance gate confirming a dependency the label does not have, and would make
a later change to those weights read as invalidating a corpus it cannot affect.

### Seeds

Hand block **942,000,000 .. 942,018,000**, taken WHOLE by both seats because
they are the matched pair from one `generate_behavior_t3_roots` call; eval
seed base **9,000,000**. Both fresh, established by running the seed audit over
every plan on disk before allocating, not by assumption:

| block | n | spent by |
| --- | --- | --- |
| 940,000,000 .. 940,050,000 | 50,000 | `worker_plan_t3first_512p.json` |
| 960,000,000 .. 960,025,000 | 25,000 | the generation-1 fleet, all eight street-seats |
| 970,000,000 .. 970,000,904 | 904 | the reference plans |
| 971,000,000 .. 971,000,100 | 100 | `worker_plan_t0miniref_1024p.json` |
| eval bases | | 5,000,000 / 6,000,000 / 8,000,000 |

The 18,000 positions are also the position-count decision: the data-scaling
curve measured on the generation-1 corpora flattens well below the 25,000 that
generation spent, so this generation is sized at the knee rather than at the
previous count, and 18,000 is the heavy end of the sized range.

### The rehearsal that gated the READY markers

`load_plan` accepting a plan proves the gate accepts it. It does not prove the
engine can score the roots the plan selects. So before either marker was
written, **the exact plan bytes the fleet receives** were run through the real
worker against the unpacked v4 runtime, one position per seat:

| | first seat | second seat |
| --- | --- | --- |
| worker exit | 0 | 0 |
| positions written | 1 | 1 |
| street / seat / to_act | T3 / first / first | T3 / second / second |
| geometry (hero, opponent public) | 9, 9 | 9, 11 |
| `scoring.fl_ev` in the label | `{14: 9.6}` | `{14: 9.6}` |
| candidates scored | 21 | 21 |
| score spread | +0.9842 … +11.6232 | −13.7277 … −12.5441 |
| `samples` recorded in the label | 1024 | 1024 |
| `plan_sha256` in the label | `ed5c07aa…` | `0fb2afbc…` |
| throughput | 0.01 pos/s | 0.22 pos/s |

The 9 / 11 asymmetry at the second seat is the discriminator between the two
T3 roots, not decoration: acting second, the opponent has already placed its T3
cards.

Two harness facts, recorded because they cost time. The worker must run with
the **runtime root as its working directory** — `ModelPaths()` defaults are
relative, and a first attempt from the wrong cwd died on
`models/opening_stage7_torch_wide.pt`. And piping the worker into `tail` makes
`$?` the status of `tail`, which reported `WORKER_EXIT=0` over a stack trace.

On where T4 provenance lives: a position record carries `plan_sha256`, not
copies of the weight digests, and does not need to. `pinned()`
(`hu_m31_label_gen_worker_v1.py:690`) re-hashes the engine, the feature encoder
and the T4 model out of the runtime root and refuses to generate on any
mismatch with the plan. **A run that exits 0 having written a label is itself
the proof that the pinned bytes were the loaded bytes.** A first version of the
verification script looked for a `continuation_policy` field this schema does
not carry and failed two passing seats; the corrected version asserts on
`plan_sha256`, `samples`, and the pins in the bound plan.

### READY

Both markers written by `m7v4_ready.py`, which validates through the worker's
own `load_plan` before writing anything, so an unvalidated plan looks missing
rather than ready.

| marker | job_id | validated |
| --- | --- | --- |
| `/home/wner/ofc-m7/ready/m7v4-t3first-1024p.ready.json` | `m7v4-t3first-1024p` | 2026-08-06T22:40:33 |
| `/home/wner/ofc-m7/ready/m7v4-t3second-1024p.ready.json` | `m7v4-t3second-1024p` | 2026-08-06T22:40:33 |

Package runtime archive sha `2835d72a1c738db8…494c`.

## T3 labels delivered and verified — 2026-08-07

18,000 positions per seat at `~/ofc-m7/labels_t3{first,second}/`, fleet layout
`<shard>/files/position_*.json` plus a per-shard `SHARD_DONE.json`.

| check | first | second |
| --- | --- | --- |
| positions | 18,000 / 18,000 | 18,000 / 18,000 |
| shard spread matches plan (38×310 + 20×311) | yes | yes |
| `SHARD_DONE` present, all 58 shards | yes | yes |
| wrong `plan_sha256` | 0 | 0 |
| wrong `samples` (≠1024) | 0 | 0 |
| wrong `fl_ev` (≠{14: 9.6}) | 0 | 0 |
| wrong street/seat/geometry | 0 | 0 |
| duplicate / missing offsets | 0 / 0 | 0 / 0 |
| candidate score span (median) | 6.3371 | 6.2490 |

**The corpus is clean on every provenance dimension.**

### All-tied labels: legitimate, but they bias the metric

40 first-seat (0.22 %) and 487 second-seat (2.71 %) positions have every
candidate scoring identically. Classified rather than assumed, and they are
**real dead positions, not broken labels**: they occur where only one row still
has space (candidate counts 3, 9, 12, 21 — the constrained-board action counts),
and the second seat's 487 carry **321 distinct values** spanning −31.87…+6.04,
so nothing is returning a constant.

The consequence is worth stating because it will be misread later: a dead
position is a **free win on the regret metric** — every pick is optimal, so
regret is 0 by construction. The second seat has 12× more of them than the
first, which mechanically deflates second-seat regret relative to first-seat.
Do not compare the two seats' regret numbers as if they were the same exam, and
check the degenerate fraction before comparing against any other corpus.

## T3 v7 — the retrain, and what had to be built to do it honestly

### The extractor the street did not have

The corpus is per-position JSON in the fleet's layout; the trainer wants rows.
`reshape_ref.py` concatenates in offset order (so row *i* is root *i*, which is
what makes the stable-hash holdout reproducible) and the native
`labelgen_feature_dump` encodes them — for the **first** seat. It refuses
`(T3, second)` **by name**, and that refusal is correct rather than an oversight:
its arms cover the T2-style composition and the T3 first seat, and an
unvalidated composition is supposed to fail loudly instead of quietly producing
a row.

So the second seat needed the missing arm. Written as its own crate at
`~/ofc-m7/t3second_dump` — a path dependency on the engine crate, its own target
dir, the repo's binary untouched — composing the row through
`t3_features::encode` fed by `t3_features::side_outlook`, which is the *same
call* `learned_t3_second_action` makes inside the engine (`search.rs:4925`).

**Which binary encodes matters, and the obvious one was wrong.** The repo's
`target/release/labelgen_feature_dump` was built 2026-08-04, while
`t3first_features.rs` was edited 2026-08-05. The build that *does* match this
generation is `~/ofc-m7prep/target/release/labelgen_feature_dump`, whose sibling
`libofc_hu_m3_engine.so` in the same target dir hashes `e17aa39f…` — the pinned
v4 engine. Checked by digest, not by date.

| seat | extractor | positions | rows | dim | NaN | seconds |
| --- | --- | --- | --- | --- | --- | --- |
| first | `labelgen_feature_dump` `d52f72cc…` | 18,000 | 292,422 | 168 | 0 | 5.1 |
| second | `t3second_dump` `c954de64…` | 18,000 | 292,212 | 168 | 0 | 3.1 |

### Feature-space continuity, established rather than assumed

The warm-start question is a feature-space question first: fine-tuning gen-1
weights on rows from a different encoder is worse than not warm-starting at all.
Two checks, both empirical.

**Which checkpoint is actually pinned.** The metadata beside the gen-1 T3
checkpoints is demonstrably copy-pasted — `~/ofc-t3first/model_v1` records seat
`"second"` — so it cannot be trusted to identify anything. Re-exporting every
candidate checkpoint and comparing the weight images against the pinned bytes
settles it without believing any of it:

| pinned image | is | arch |
| --- | --- | --- |
| `t3first_model_v1.bin` `36334b4b…` | `~/ofc-t3first/model_v1/model.pt` | [168, 256, 128, 64, 1] |
| `t3_model_v2.bin` `e9cd7416…` | `~/ofc-t3/model_v2_512/model.pt` | [168, 256, 128, 64, 1] |

Note the second seat's pinned model is the **512-particle** one from
`model_v2_512`, not the `model_v2` its name suggests — that directory holds a
167-dimension checkpoint and is pinned nowhere.

**Rows against the reference the gen-1 models were fitted on.** Both native
extractors were compared row by row, matched by action key, against the Python
modules whose digests the gen-1 metadata records (`t3_features.py` `095c7f6e…`,
`t3_outlook.py` `37fdedf0…`, `t3first_outlook.py` `d28c7037…`):

| seat | positions | rows | worst absolute difference | verdict |
| --- | --- | --- | --- | --- |
| first | 30 | 540 | 1.192093e-07 | float32-equal |
| second | 30 | 519 | 1.192093e-07 | float32-equal |

1.19e-07 is 2⁻²³ — the float32 unit of rounding. The new rows are the old rows,
so the warm start is honest and the gen-1 models can be graded on the new exam.

### Corpus facts

| | first | second |
| --- | --- | --- |
| positions / rows | 18,000 / 292,422 | 18,000 / 292,212 |
| candidates min / mean / max | 3 / 16.25 / **21** | 3 / 16.23 / **21** |
| candidate-count histogram | {3: 493, 9: 3218, 12: 4232, 21: 10057} | {3: 557, 9: 3177, 12: 4182, 21: 10084} |
| train (code < 230) / held out | 16,213 / 1,787 | 16,168 / 1,832 |
| all-tied positions (held out) | 40 (2) | 487 (46) |

The degenerate counts reproduce the corpus verification exactly. Width is 21 at
both seats, so nothing is truncated at any packing.

### The clamp was wrong, and the warm start is what caught it

The retrain was to standardise with a ±8 clamp. It does not ship, and the reason
is not taste:

* The engine's weight image stores a mean and a std and **nothing else**;
  `Model::predict_with` computes `(x - mean) * inverse_std` and feeds the first
  layer directly (`t4_model.rs:176`). A model fitted on clamped inputs is run
  **unclamped** inside every rollout — asked about rows it was never shown.
* It is not the no-op it looks like. Under the incumbent's own statistics
  **0.045 %** of feature entries fall outside ±8 (one in ~2,200), one dimension
  reaching |z| = 833.
* It was found because the warm start refused itself. Re-expressing the gen-1
  first layer in the new statistics is exact algebra, so the drift on real rows
  must be zero; it came out at **5.6e-01** with the clamp on and **5.7e-06**
  with it off. The clamp was truncating the two normalisations differently.

The gen-1 checkpoints use an additive `std + 1e-6` rather than a floor, so 21
constant dimensions carry std ≈ 1e-6 and appear as a 1e6 std ratio in the
transfer. They are genuinely constant in both corpora, so they contribute
nothing; the corpora are otherwise closely aligned (worst mean shift 1.15 z,
almost all below 0.03 z).

### The incumbent on the new exam — the honest "before"

| seat | regret | live-only | top-1 | top-3 | SE |
| --- | --- | --- | --- | --- | --- |
| first (`t3first_model_v1`) | 0.052784 | 0.052843 | 0.6782 | 0.9183 | 0.003660 |
| second (`t3_model_v2`) | 0.031601 | 0.032415 | 0.8275 | 0.9651 | 0.002817 |

Seats are **not** comparable to each other: the second seat's exam carries 46
dead positions against the first seat's 2, and a dead position is regret 0 by
construction.

### Scaling curve on nested subsets — still descending at 18k

Prefixes 4k ⊂ 8k ⊂ 12k ⊂ 18k of the training split, the exam held fixed at the
full corpus's `code >= 230`. Two recipes, because they answer different
questions.

| seat | recipe | 4,000 | 8,000 | 12,000 | 18,000 |
| --- | --- | --- | --- | --- | --- |
| first | warm (ships) | 0.051415 | 0.049366 | 0.046966 | **0.044502** |
| first | cold | 0.175139 | 0.098727 | 0.075090 | 0.063391 |
| second | warm (ships) | 0.032873 | 0.036654 | 0.030506 | **0.029540** |
| second | cold | 0.180307 | 0.179480 | 0.108233 | 0.063325 |

(12k and 18k warm figures are the 3-seed means from the rate sweep below; 4k
and 8k are 2 seeds. Cold is 1 seed and is context, not a decision input.)

**Cold is steeply climbing at 18k at both seats** — unsurprising, since gen-1
was fitted on 50,000 positions. **Warm is still descending, but slowly**: the
12k → 18k step buys 0.0025 (first) and 0.0010 (second) against a seed spread of
0.0005–0.0022 and an exam SE of 0.0037 / 0.0028. So the corpus is at the knee
rather than past it, and the honest statement is that another 18,000 positions
would be expected to buy on the order of 0.002 more at the first seat and less
at the second. Reported for a ruling rather than acted on.

**Ruling: the extension is DECLINED**, and the reason is this milestone's own
gate result rather than cost. The first seat's retrain improved held-out regret
by 13.3 % and produced **no measurable play gain at all** over 20,004 mirrored
deal-pairs. Buying another 18,000 positions to move the same label-metric a
further ~0.002 — a fifth of the improvement that already failed to reach the
scoreboard — has no expected payoff at this street. The curve is recorded so a
future generation with a *different* consumer of these labels can revisit it;
it is not a reason to spend fleet time now.

### The fine-tune rate: measured, and null

The warm runs kept epochs 3, 4, 6, 11 in several cells, which is the signature
of a first optimiser step damaging a model that is already close. Swept rather
than assumed — three seeds, two corpus sizes, both seats:

| seat | roots | lr 1e-3 | lr 3e-4 | lr 1e-4 |
| --- | --- | --- | --- | --- |
| first | 12,000 | 0.046966 | 0.046610 | 0.047973 |
| first | 18,000 | 0.044502 | 0.044244 | 0.043946 |
| second | 12,000 | 0.030506 | 0.028627 | 0.028787 |
| second | 18,000 | 0.029540 | 0.029410 | 0.029678 |

Every difference is inside the seed spread of the cells it is drawn from
(0.0003–0.0022). **Rate is NULL**, so the frozen protocol keeps its 1e-3 and no
deviation is bought with a null. Lower rates do give tighter spreads, which is
worth knowing for reproducibility but is not an outcome difference.

### The shipped models

Protocol: 120 epochs, Adam 1e-3 cosined to zero, batch 512 positions,
`reg_weight` 0.1, masked padding, stable-hash holdout at `split_code` 230, the
epoch with the lowest held-out regret kept, **no clamp**, warm start from the
gen-1 checkpoint re-expressed in the new statistics. Ship seed **997,990,027**,
declared before the run and shipped wherever it landed in the spread — the five
other 18k runs at this configuration are the spread, not a menu.

| | first | second |
| --- | --- | --- |
| checkpoint | `~/ofc-m7/t3v7/ship/t3first_v7.pt` | `~/ofc-m7/t3v7/ship/t3second_v7.pt` |
| image | `t3first_model_v2.bin` `2d94b3e3…` | `t3_model_v3.bin` `75119ebe…` |
| kept epoch | 48 | 56 |
| held-out regret | **0.045776** (was 0.052784) | **0.030501** (was 0.031601) |
| live-only regret | 0.045827 | 0.031286 (was 0.032415) |
| top-1 | 0.6827 (was 0.6782) | 0.8242 (was 0.8275) |
| top-3 | 0.9222 (was 0.9183) | 0.9602 (was 0.9651) |
| warm-start drift | 5.7e-06 | 5.7e-06 |

The first seat improves by 13.3 % of its regret; the second by 3.5 %. The second
seat's top-1 and top-3 move the *other* way by less than a standard error
(top-1 SE ≈ 0.009), which is worth stating plainly: at that seat the retrain
buys a smaller loss on the actions it gets wrong rather than more actions right.

Naming follows the ledger rather than inventing one: the first seat's gen-1
image is `t3first_model_v1.bin`, so the relabel is v2; the second seat's is
`t3_model_v2.bin`, so the relabel is v3.

### The engine loads what was trained — checked before any gate ran

Training metrics are computed in Python on rows this session extracted; the
model is consumed in Rust on rows the engine composes, from an image a third
program wrote. All three were made to line up in one test: the engine's `decide`
against the checkpoint's own argmax, over held-out positions.

| seat | positions | engine == checkpoint | tie-break only | **mismatches** | v7 vs gen-1 divergence |
| --- | --- | --- | --- | --- | --- |
| first | 200 | 194 (97.0 %) | 6 | **0** | 23.5 % |
| second | 200 | 198 (99.0 %) | 2 | **0** | 25.5 % |

The divergence column is the gate's expected divergence rate, bought for a
minute instead of found after 20,000 deals: two models that agreed everywhere
could not produce a gate verdict at all.

### The gates

Mirrored duplicate match, one per seat, on the standing criterion **lower CI >
−0.05 and mean ≥ 0**. Adapted from the T0-first harness with one structural
change: both chains are **pure engine at all ten slots** — `decide` now covers
every decision a hand has, so the production behaviour policy, its bundle and
the stage-3 encoder are out of the loop, and the chains are two sets of pinned
weights and nothing else. T4 is exact enumeration for both.

Self-test first, and it is the reason to trust the rest: NEW against NEW must
give **exactly** 0.0 on every pair. Both seats **PASS** (30 deals, 0 nonzero
pairs, 0 divergence). Throughput 6,111 games/h per process — about 2.9× the
T0-first harness, from taking Python out of the decision loop.

Deal seeds from fresh blocks, audited against every plan on disk plus the
non-plan ledger: **945,000,000** (first seat) and **946,000,000** (second),
20,004 deals each over 6 shards.

**The verdicts, 20,004 deal-pairs per seat, 40,008 games each:**

| | first seat | second seat |
| --- | --- | --- |
| deals / games | 20,004 / 40,008 | 20,004 / 40,008 |
| divergence rate | 22.49 % (4,499 deals) | 19.69 % (3,938 deals) |
| paired mean per hand | **−0.0027** | **+0.0043** |
| standard error | 0.0089 | 0.0067 |
| 95 % CI | [−0.0201, +0.0147] | [−0.0088, +0.0173] |
| mean on divergent deals | −0.0119 | +0.0217 |
| mean on identical decisions | 0.0 | 0.0 |
| max abs pair on identical | **0.0** | **0.0** |
| mirror clean | yes | yes |
| **verdict** | **NO_GAIN** | **ADOPT** |

Throughput 19,318 / 19,432 games per hour pooled; longest shard 2.07 h.

**Read these together, because separately they mislead.** Both intervals
straddle zero and both clear the −0.05 floor by a wide margin. What separates
the verdicts is the *sign of a point estimate that is a fifth of its own
half-width* — the first seat's −0.0027 against a half-width of 0.0174, the
second's +0.0043 against 0.0131. The honest summary is:

* **Neither seat is a meaningful regression.** That is the strong, well-powered
  statement here: at 20,004 deals a regression worse than −0.02 per hand would
  have shown, and neither seat has one.
* **Neither seat demonstrates a gain either.** The second seat's ADOPT is the
  standing rule applied exactly as written (`lower CI > −0.05 AND mean ≥ 0`),
  not a measured improvement.
* **Resolving these effects would cost far more deals.** To bring the first
  seat's half-width down to ±0.005 takes about **320,000 deal-pairs** at the
  observed spread — sixteen times this run, roughly 33 hours of this box.

**And the finding that matters more than either verdict:** the first seat's
held-out regret improved by **13.3 %** while its play did not measurably improve
at all, and the second seat improved by **3.5 %** on the same metric while its
play point estimate went the other way, positive. That is this harness's own
premise arriving as data — a held-out number is a statement about the labels,
not about play — and it says the T3 street's leverage on head-to-head EV is
small enough that label-metric gains at this scale do not survive into the
scoreboard. Worth carrying into the T2 and T1 decisions, which are cheaper to
justify on provenance than on expected EV.

**The mirror check passed at both seats over 15,505 and 16,066 identical-decision
deals**: every one of them contributed a hard 0.0. Combined with the self-play
pre-check, the duplication is sound and the nonzero pairs are all real policy
differences.

### What the verdicts do NOT settle, and is the supervisor's call

The first seat's NO_GAIN leaves a genuine tension, stated rather than resolved
here:

* the standing rule says do not adopt;
* but the whole premise of M7 is that a corpus labelled at one constant and
  consumed by a teacher searching at another is a slow-acting bug, and the
  incumbent `t3first_model_v1` is fitted on labels carrying the **June**
  constant. Declining the relabel leaves the T2 street pinning a first-seat T3
  continuation from the wrong constant — while the gate says the two play
  indistinguishably.

Adoption on **provenance** grounds where play is measurably equivalent is a
defensible ruling, and so is holding the rule. It was escalated rather than
decided here.

### The ruling: ADOPT BOTH SEATS — supervisor override, recorded verbatim

> **T3-first NO_GAIN: ADOPT BOTH v7 SEATS** — supervisor override of the
> mechanical sign, recorded as follows: the gate's decline rule exists to
> prevent regression; first seat shows none (mean −0.0027 against half-width
> 0.0174, floor cleared by 3x). Declining would pin a continuation fitted at the
> June constant (10.227) inside every T2 label — reintroducing the exact defect
> M7 exists to remove. Constant-consistency through the cascade dominates the
> sign of a noise-sized point estimate.

Executed: the package at `/home/wner/ofc-labelgen-m7v5/package` carries both
images beside every generation-1 image (17 weights, nothing dropped), with
ledger keys `t3_first_model_v2_sha256` `2d94b3e3…` and
`t3_second_model_v3_sha256` `75119ebe…` recomputed from the bytes that shipped.
Both T2 plan generators carry the override's reasoning in their provenance
rather than a bare version number, so a later reader finds the argument attached
to the pin.

## DEBT — the input clamp and the T4M1 / VFL1 divergence

**Filed here so it is not rediscovered a third time.** Status: open. Owner: the
next executor that trains any model destined for a `T4M1` image.

**The defect.** `T4M1` stores a mean, a standard deviation and the layer
weights, and nothing else. `Model::predict_with` computes
`(x - mean) * inverse_std` and feeds the first layer directly
(`t4_model.rs:176`). **There is no clamp in the image and none at inference.**
A model fitted on clamped standardized inputs and exported as `T4M1` is
therefore *run outside the regime it was fitted in*, silently — the export
succeeds, the digests match, the provenance gate passes, and the served function
is not the trained one.

**Why it is not hypothetical.** On the T3 v7 corpora, **0.045 %** of feature
entries lie outside ±8 under the incumbent's own statistics (one entry in
~2,200), with one dimension reaching |z| = 833. Those are exactly the rows a
clamp exists to tame, and exactly the rows that would be served untamed.

**M11 already solved this for its own models and the fix did not travel.**
`hu_m11_vsfl_opponent_model_20260805.md` records that the vs-Fantasyland arms
need both a zero inverse std and a ±8 clamp, that `T4M1` "can store neither …
and would silently ship a different model", and that `VFL1` was introduced as
the superset that carries the clamp explicitly. Every vs-FL image is `VFL1`. The
regular cascade's images are all `T4M1`, and nothing in the export path
*prevents* a clamped checkpoint from being written into one.

**What was done here.** The T3 v7 models are trained and exported **unclamped**
(`--clamp 0`), which is the correct pairing for a `T4M1` consumer, and the
warm-start renormalisation drift of **5.7e-06** is the proof that no clamp is in
the composed function. `m7v4_t3v7_train.py` defaults `--clamp` to 0 and carries
the reason in the flag's own help text.

**What is still owed** — any one of these closes it:

1. `export_weights_only.py` refuses a checkpoint whose `clamp` field is nonzero,
   pointing the caller at `VFL1`; or
2. `T4M1` gains a version that stores a clamp, and the engine applies it; or
3. the regular cascade's images move to `VFL1`, which already carries it.

Until one lands, the rule is: **a clamped model may not be exported as `T4M1`**,
and every trainer in this line defaults the clamp off.

## T2 — groundwork done, street not yet run

The labels the T2 street will replace were generated at **128 particles**, 25,000
positions per seat, against a teacher of learned T4 + T3-second + T3-first. That
is an eighth of the sample rung just purchased at T3, which is the first thing
the T2 particle probe should be pointed at.

### Baseline reproduced, exactly

The protocol requires reproducing the baseline before ablating against it. Done,
and it is an exact match rather than an approximate one — the shipped metadata's
held-out `ev_given_up` is recovered to six decimals by re-grading the shipped
checkpoint on the same stable-hash split (`pos_code >= 230`):

| seat | shipped metadata | re-graded, exam truncated to 24 | re-graded, **all 27 candidates** |
| --- | --- | --- | --- |
| T2 first (`model_v1` == `model_w24`) | 0.23288019 | **0.232880** | **0.233859** |
| T2 second (`model_v1`) | 0.20727602 | **0.207276** | **0.209076** |

**The right-hand column is the baseline the T2 ablation arms must beat**, because
the current `ofcdata.pack` refuses to pack below the true maximum width.

### A hypothesis raised and killed

The T2 corpora hold up to **27** candidates per position, but both shipped models
record `width: 24`. That looked like a real defect: a truncating model is graded
on a truncated exam — never asked about actions it cannot see — and `model_w27`
reporting *worse* numbers is exactly what that artefact would produce. A first
pass appeared to confirm it: the true best action sits in a dropped slot in 3.14 %
of positions, worth a mean 1.87 EV, which is 0.047–0.062 of `ev_given_up` on the
held-out split — a quarter of the shipped model's total measured loss.

**It is not a defect.** Re-grading the same checkpoint on the complete candidate
set moves `ev_given_up` by only `+0.000978` (first seat) and `+0.001800`
(second). The two large effects cancel: widening the exam raises the reference
best by ~0.05, and it raises what the model actually achieves by very nearly the
same amount, because candidate order is uncorrelated with quality (the best-slot
histogram over 27-candidate positions is flat, and the best action lands in the
last three slots at 11.3 % against a chance level of 11.1 %). Regret is a
difference of two quantities that both move together.

So the shipped width-24 choice stands on its merits: on the identical complete
exam, w24 scores 0.233859 and w27 scores 0.238831. w27 really is worse. What the
width-24 packing does cost is **training rows** — roughly 3 % of all rows never
enter the loss — which is a legitimate ablation arm, not a metric artefact.

Recorded because the first number was alarming and wrong, and the sequence that
caught it (measure, disbelieve, re-measure through the code that ships) is the
protocol working.

### Capacity arm: NULL at both seats

Run on the frozen M7 Task-0 protocol, unchanged from T1 and T3 — 120 epochs,
Adam lr 1e-3, batch 512 positions, cosine to zero, `reg_weight` 0.1,
`split_code` 230, keep the lowest held-out mean regret. Fresh seeds
997,990,011–013 (the 997,990,001–003 block is spent by the T1/T3 arms).
Capacity = `768,384,192,96` against the shipped `256,128,64`: ~6x the parameters
(517,633 vs 84,481).

| seat | arm | n | mean regret | [min, max] | seed spread | top1 | vs baseline |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T2 first | baseline | 3 | **0.230801** | [0.223286, 0.238784] | 0.015498 | 0.6120 | — |
| T2 first | capacity | 3 | 0.233365 | [0.226650, 0.237243] | 0.010592 | 0.6138 | **+1.11 % (worse)** |
| T2 second | baseline | 3 | **0.205341** | [0.200438, 0.208896] | 0.008458 | 0.6262 | — |
| T2 second | capacity | 3 | 0.206615 | [0.199941, 0.211384] | 0.011442 | 0.6293 | **+0.62 % (worse)** |

**Verdict: capacity is NULL at both T2 seats.** Six times the parameters buys
nothing; both arms land inside the baseline's own seed spread, on the wrong side
of it. The baseline arms also re-derive the shipped models to −1.31 % (first) and
−1.79 % (second), which is a clean Gate-B style reproduction and confirms the
harness is measuring the right thing.

**The honest caveat, and it is the same lesson as the probe.** The baseline's
seed spread is 0.0155 (first) and 0.0085 (second) — larger than any arm
difference measured. At three seeds this ablation can only resolve effects of
roughly 1.5 % of the metric or bigger. It is powered enough to say "6x capacity
is not a large win", which is a real and useful negative, and **not** powered to
distinguish a 0.5 % effect from zero. Do not read the capacity arm's small
positive numbers as evidence it actively hurts.

Arms (b) enriched and (d) both are **queued, not skipped**: they need a T2 tier-1
feature extraction (the T1/T3 equivalents are
`/home/wner/ofc-m7/ablation/feat/t{1,3}_tier1.npz`, 13 dims), which does not
exist yet. Harness is ready for them — `train_ablation.py` now carries `t2first`
and `t2second` entries (original backed up to `train_ablation.py.pre_t2`), and
takes `--extra <npz>` plus `--drop-base` for the redundancy diagnostic.

### Arms (b) and (d) at the first seat: also NULL, and the diagnostic says why

The tier-1 joint-event extractor now has a T2 arm, generated from
`extract_joint.py` by the same asserted-substitution discipline as the probe
(four edits, each matched once). The row order is the corpus's own `keys.txt` —
reconstructing it from the score map would silently transpose rows within a
position, because the native dump walks the action generator rather than the map
— and the extractor's alignment gate passed at **max |dy| = 3.6e-15** over
571,053 rows, which is float64 round-trip noise. 150 seconds for the seat.

| arm | dims | params | n | mean regret | [min, max] | spread | top-1 | vs baseline |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 168 | 84,481 | 3 | **0.230801** | [0.223286, 0.238784] | 0.015498 | 0.6120 | — |
| capacity | 168 | 517,633 | 3 | 0.233365 | [0.226650, 0.237243] | 0.010592 | 0.6138 | +1.11 % |
| **enriched** | **181** | 87,809 | 3 | 0.232999 | [0.230108, 0.234991] | 0.004883 | 0.6125 | **+0.95 %** |
| **enriched+capacity** | **181** | 527,617 | 3 | 0.231782 | [0.225091, 0.236942] | 0.011851 | 0.6167 | **+0.43 %** |
| tier1_only | 13 | 44,801 | 3 | 0.998774 | [0.992273, 1.010095] | 0.017823 | 0.3752 | +332.7 % |

**Every arm is NULL at the T2 first seat**: all three land inside the baseline's
own 0.0155 seed spread, and on the wrong side of it. Adding the 13 joint-event
dimensions buys nothing, and adding six times the parameters alongside them
buys nothing either.

The `tier1_only` arm is the diagnostic that makes the null interpretable rather
than merely disappointing. Trained on the 13 dims **alone** the model reaches
0.9988 against the 168-dim block's 0.2308 — so the joint-event features are
genuinely informative (a random ranker would be far worse) but carry a small
fraction of what the shipped block carries. Combined with the null on the full
arm, the reading is **redundancy, not irrelevance**: the foul and Fantasy-Land
joint structure is already implied by the per-row marginals plus the
head-to-head block, at least at the resolution 25,000 positions of 128-particle
labels can resolve.

**So neither axis binds at T2.** Capacity is null at both seats; features are
null at the first seat. What is left is the labels themselves, which is exactly
what the particle probe is for — and it makes the probe the load-bearing
measurement of this street rather than one of three.

### The second seat's row order, rebuilt — and what the rebuild found

`~/ofc-t2/features_rs` predates `keys.txt`, so nothing on disk said which action
each row belonged to, and the order is **not** the score map's (the native dump
walks the action generator). Reconstructing it by guessing would have transposed
rows inside positions and fitted the arms against shuffled labels.

The rebuild turned up something the corpus had not recorded. The 27,499 label
lines are **two families**, and they are not duplicates:

| family | files | offsets | |
| --- | --- | --- | --- |
| `vm_shard_*.jsonl` | 8 | 25,000 | complete |
| `shard_*.jsonl` | 8 | 2,499 | **semantically different labels for the same roots** |

All 2,499 shared offsets disagree — different scores for the same position, not
formatting noise. So "dedupe by offset" was never a safe operation, and which
family is authoritative is a question the files cannot answer.

**The shipped labels answered it.** Both candidate assemblies were re-dumped
through the same native extractor and compared against the corpus's own
`y.bin`: the `vm_shard`-preferred assembly reproduces it **bit-identically**
(sha `b471cdc5…`, 4,560,552 bytes), so the vm family is the corpus and the small
local shards are superseded work that was never part of it. 570,069 keys
recovered, and the tier-1 extraction's alignment gate then passed at
**3.6e-15**.

### Arms (b) and (d) at the second seat: null too, and the extra seeds are why

| arm | n | mean regret | [min, max] | spread | top-1 | vs baseline |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 6 | **0.203335** | [0.199213, 0.208896] | 0.009683 | 0.6268 | — |
| capacity | 3 | 0.206615 | [0.199941, 0.211384] | 0.011442 | 0.6293 | +1.61 % |
| enriched | 3 | 0.200179 | [0.195539, 0.205895] | 0.010356 | 0.6304 | −1.55 % |
| enriched+capacity | 6 | 0.199295 | [0.195571, 0.204136] | 0.008566 | 0.6349 | −1.99 % |
| tier1_only | 3 | 1.039699 | [1.029527, 1.057011] | 0.027484 | 0.3744 | +411 % |

**Recorded because the first three seeds looked like a result and were not.** At
three seeds `enriched+capacity` read −3.73 % with a seed spread of 0.0017 — five
times tighter than the baseline's — and sat just under the rule's threshold.
Three more seeds each (997,990,014–016) moved it to −1.99 % with a spread of
0.0086, and the gain of +0.0040 is now comfortably inside the baseline's own
0.0097. **NULL**, and the first three seeds were a favourable draw. Doubling the
seeds on a borderline arm is the cheapest honest test there is, and it should be
the default whenever an arm lands within a spread of the threshold.

So the picture is the same at both seats: **capacity null, features null,
`tier1_only` far worse than the shipped block** — informative dimensions that
the 168-dim block already implies. Nothing about the T2 model's shape binds; the
labels are what is left, which is what the funded probe is for.

### T2 corpus facts, verified

| | T2 first | T2 second |
| --- | --- | --- |
| corpus | `~/ofc-t2first/features_rs` | `~/ofc-t2/features_rs` |
| positions / rows | 25,000 / 571,053 | 25,000 / 570,069 |
| candidates min / mean / max | 9 / 22.84 / **27** | 9 / 22.80 / **27** |
| feature dim | 168 | 168 |
| held-out positions (code ≥ 230) | 2,524 | 2,622 |
| label particles | 128 | 128 |

Note: `~/ofc-t2/model_v1/metadata.json` records `"street": "T3"` while its own
`label_provenance` says "T2 second seat". The provenance string is right and the
field is wrong; worth fixing when that model is superseded.

### The T2 particle probe: what adapting the harness actually needs

Assessed rather than estimated. `t3noise_probe.py` is T3-specific in exactly two
places — it calls `generate_behavior_t3_roots` and `hu_m3_rust.evaluate_t3` —
and **both already have T2 equivalents that ship in the runtime**:

* `generate_behavior_t2_roots` exists alongside the T3/T1/T0 generators;
* `hu_m3_rust.evaluate_t2` exists (`hu_m3_rust.py:360`);
* the label-gen worker's own `make_root` (`worker:1276`) already dispatches all
  four streets through them, so there is a working reference implementation.

So the adaptation is a bounded change, not a build:

1. add `--street`, dispatching root generation and evaluation through the
   worker's existing mapping;
2. add the T2 continuation pins `load_plan` requires — `t3_first_model` on both
   seats, plus `t2_second_model` on the first seat;
3. everything else (seed formulas, rungs, reference, paired statistics, the
   self-comparison control) is already street-agnostic.

**A sequencing point that argues for queueing it anyway.** A T2 probe run today
would pin the *current* T3 models as its continuations, but the T2 labels it is
sizing will be generated against **T3 v7**. The probe measures variance scaling,
which should transfer across a continuation swap — but that is an assumption,
and it is cheaper to run the probe after the v7 models exist than to defend it.
Queue the T2 probe behind the T3 retrain, which is also where the priority order
puts it.

**Cost it honestly before committing.** T2 rollouts are strictly deeper than T3's
(the teacher must play both seats' T3 replies before reaching T4), so the T3
reference cost of 289 core-s/root at the first seat is a *floor*, not an
estimate. Measure it on a 2–4 worker pilot and size the root count from the
half-width arithmetic above — do not size it by what fits in memory.

### The T2 probe exists now — built by substitution, and costed

`m7v4_t2noise_probe.py` is generated from `t3noise_probe.py` by
`m7v4_t2noise_make.py`, which applies **18 literal substitutions and refuses to
write unless every one of them matched exactly once**. Retyping a 484-line
harness is how a probe quietly stops being the probe that produced the earlier
verdicts; this way the seed formulas, the paired statistics, the shared-seed
control and the resume-by-file logic arrive untouched by construction, and the
T3 harness is left byte-identical because it is the evidence behind the T3 row
of the diagnosis table.

What changed: `generate_behavior_t2_roots`, `hu_m3_rust.evaluate_t2`, the pin
set (both T3 evaluators at both seats; `t2_second_model` at the first seat
only), and a ladder of **128/256/512 against a 2,048 reference** — bracketing
the 128 particles the shipped T2 corpora were labelled at rather than starting
above them.

**One capability is genuinely lost at T2, and it is not papered over.** The T3
result carries `evaluation_rng_key_digests`, one per particle, and the probe
*measures* rung/reference independence as the size of that intersection.
`hu_m3_t2_result_v1` does not carry it — it reports only the static claim
`sample_independence: disjoint_particle_rng_keys`, which is about
candidate-versus-evaluation batches inside a single call, not about two calls.
So at T2 the check degrades from **measured** to **argued from the key
construction** (`belief.rs` keys a particle by base_seed, run_id, street, root
fingerprint and sample index, and the probe differs in both base_seed and
run_id), backed empirically by the shared-seed control. Each root record states
which basis it used in an `independence_basis` field, and the run refuses if the
rung and the reference share either seed or run_id.

### Cost, measured on a pilot rather than extrapolated from T3

Two roots, both seats, a 64-rung against a 256-reference, against the **v4
package runtime** (engine `e17aa39f…`, T4 v6, fl_ev 9.6 confirmed in the
manifest) with the T3 **v7** images pinned as continuations — which also smoke-
tested those images inside a real T2 rollout.

| seat | core-s per sample | fixed | one root at 128/256/512 + 2,048 |
| --- | --- | --- | --- |
| first | 0.4441 | +0.62 s | 1,310 core-s |
| second | 0.2652 | +1.25 s | 786 core-s |

Both seats, one root: **2,096 core-s = 0.58 core-h.**

(The T3 rehearsal trap repeated itself here and is worth restating: the probe
must run with the **runtime root as its working directory**, because the
behaviour bundle that generates the roots names its checkpoints relatively. The
first pilot died on `models/opening_stage7_torch_wide.pt` before any engine
call.)

### Sizing, before launching — the rule T3 wrote in blood

The dispersion that sets the root count is measured on the finished T3 probe's
own 150 roots, not assumed:

| seat | paired mean (256 − 1024) | SD | 95 % half-width at n = 150 |
| --- | --- | --- | --- |
| first | +0.00122 | 0.05718 | 0.00915 |
| second | +0.01839 | 0.08574 | 0.01372 |

At the meaningful threshold **δ = 0.010**, with T2's dispersion taken as T3's
inflated by a factor (T2's rollout is strictly deeper, so it should scatter at
least as much):

| inflation | seat | roots for a GAP verdict | roots for a FLAT verdict | core-h (flat, both seats) |
| --- | --- | --- | --- | --- |
| 1.0x | first | 126 | 197 | 115 |
| 1.0x | second | 283 | 442 | 257 |
| **1.5x** | first | 283 | 442 | 257 |
| **1.5x** | second | **636** | **993** | **578** |
| 2.0x | first | 503 | 785 | 457 |
| 2.0x | second | 1,130 | 1,766 | 1,028 |

**The verdict on affordability is the whole point, and it is unambiguous: a
powered T2 probe is a fleet job, not a local one.** Sizing at the worst seat
with a 1.5x inflation — 993 roots for a flatness verdict — costs 578 core-h,
which is **72 hours on eight local workers and 1.25 hours on the 464-core
fleet**. Even the cheaper gap verdict (636 roots at the second seat) is 2.4 days
locally and under an hour on the fleet.

So the recommendation is **fund it on the fleet**: 1,000 roots, both seats,
128/256/512 against 2,048, which resolves both verdicts at both seats under the
1.5x assumption and still costs under two hours of fleet time. Running it
locally would repeat exactly the T3 mistake — a probe sized by what fits, which
answers nothing — and this time the arithmetic is on the record beforehand.

### FUNDED and READY: the probe as a fleet run

> **T2 probe: FUNDED at 1,000 roots on the fleet.**

It rides the label-generation lifecycle rather than inventing one, and it can
because **a probe plan is a legal label plan**. `load_plan` requires fields; it
does not forbid extras. Every field it requires the probe genuinely has — same
street, same seat, same hand block, same pins — `samples` is the top rung, and
the ladder travels in an extra `probe` block the label worker would never read.
Both plans were put through `load_plan` itself and **accepted**.

That buys the whole proven lifecycle unchanged — staging by content binding with
observed generations, create-only publishing, heartbeats, numbered checkpoints,
preemption resume, bare-digit shards, `SHARD_DONE.json` for the chain watcher —
with **exactly one substitution**: the startup script.

`scripts/startup_hu_m7v4_t2probe_v1.sh` `3c55705a…` is derived from the label
startup by seven asserted substitutions, so the fetch half, the create-only
publishing and four fleets' worth of preemption handling arrive untouched. What
differs is only what the probe genuinely does differently:

1. the unit of work is a **root**, so the resume listing and completion count
   look for `root_NNNNN.json`;
2. the process spawned is the probe module with this shard's root range;
3. the probe has no `--existing-manifest` — it resumes by refusing to overwrite
   a root that exists — so already-published roots are **downloaded back onto
   the disk** before the workers start. Without it a recreated Spot instance
   redoes durable work;
4. the probe writes no completion marker, so the parent writes `SHARD_DONE.json`
   — and **only** when the roots on disk equal the count this shard's plan entry
   asks for, so a shard that ends early is never marked finished;
5. **six workers per instance**, not one per vCPU: the T3 probe OOMed at 15 ×
   3.76 GB and the standing note from that failure is 6–8 for probe-class loads.

The probe module ships **inside the package** as
`src/ofc_regular/hu_m7v4_t2_noise_probe_v1.py` `e7717b1a…`, written by the same
generator that writes the scratchpad copy so the two cannot drift, and its
digest is **pinned in the plan beside the weights** — it is the thing doing the
measuring, and pinning the models it calls while leaving it unpinned would be
provenance with a hole in the middle.

| | first seat | second seat |
| --- | --- | --- |
| plan | `worker_plan_m7v4_t2first_probe1000.json` | `worker_plan_m7v4_t2second_probe1000.json` |
| plan sha256 | `1868f8e9…` | `1cabfdbf…` |
| roots / shards | 1,000 / 58 | 1,000 / 58 |
| ladder | 128/256/512 vs 2,048 | same |
| machine / workers | c4-standard-8 / 6 | same |
| READY | `~/ofc-m7/ready/m7v4-t2first-probe1000.ready.json` | `…t2second-probe1000.ready.json` |
| run plan | `~/ofc-m7/t2probe_runs/run_plan_m7v4-t2first-probe1000.json` | `…t2second…` |

### The rehearsal, and the cost it corrected

One root per seat, through the real probe module from the unpacked m7v5 runtime,
with the package's **own cp311 wheelhouse** on `PYTHONPATH` — the wheelhouse is
part of what is being rehearsed, and this box's system `python3` is 3.12, which
loads the pure-python wheels and then fails on numpy's C extensions. The
rehearsal names the 3.11 interpreter for that reason, and asserts every flag it
passes appears in the startup script before it runs anything, so the two cannot
drift into running different commands.

| | first seat | second seat |
| --- | --- | --- |
| exit | 0 | 0 |
| geometry (hero, opponent public) | **7, 7** | **7, 9** |
| candidates scored | 24 | 27 |
| reference (2,048) | 529 core-s | 401 core-s |
| rungs 128 / 256 / 512 | 30 / 59 / 119 core-s | 19 / 37 / 79 core-s |
| `independence_basis` | argued_from_key_construction | same |
| pins in the manifest | t4 `763b77a0…`, t3first `2d94b3e3…`, t3second `75119ebe…`, fl_ev 9.6 | same |

The 7/7 against 7/9 asymmetry is the seat discriminator, as 9/9 against 9/11 was
at T3: acting second at T2 the opponent has already placed its T2 cards.

**And the rehearsal corrected the budget downward.** Uncontended, a root costs
**737 core-s** (first) and **536** (second) against the 1,310 / 786 the pilot
predicted — the pilot ran while twelve match-gate shards held the box, and
core-seconds inflate under contention. The funded run is therefore **354 core-h
for both seats at 1,000 roots**, about **0.76 h on the 464-core fleet**, against
the 582 core-h the sizing note quotes. The plans keep the conservative number:
being under budget is not a reason to re-cut a plan that is already validated.

### The first probe run generated LABELS, not probe roots

**There is no T2 probe verdict.** Both runs completed, all 58 shards, fleets
cleaned, ~1.4 h of wall time — and produced the wrong artifact. Checked before
anything was computed from them, which is the only reason this is a paragraph
and not a table of fabricated verdicts.

| what a probe root looks like | what arrived |
| --- | --- |
| `root_NNNNN.json`, schema `ofc_m7_t2_noise_probe_root_v1` | `position_NNNNNNNN.json`, schema `hu_m31_label_gen_position_v1` |
| several rungs plus a deeper reference per root | **one** run at **one** sample count |
| `SHARD_DONE.json` schema `hu_m7v4_t2probe_shard_done_v1` | schema `hu_m31_label_gen_shard_done_v1` |
| `manifest_stride*.json` per shard | none |

1,000 position files per seat, `samples: 512`, zero root files, zero probe
manifests. The **label worker** ran. The pre-registered statistic — a paired
delta between rungs on the same root — cannot be computed from data that
contains one rung and no reference, so no amount of care in the analysis could
have rescued it.

**Root cause, and it is a design decision of mine.** `phase_execute` renders the
startup script named by the *run-plan receipt*; the receipt used was not the one
at `~/ofc-m7/t2probe_runs/` (which names the probe script and whose sha check
would have failed against the label one), so the stock label startup went into
instance metadata. The plan bytes were mine — every position file carries plan
sha `1868f8e9…` / `1cabfdbf…` — and the label worker accepted them **because I
made a probe plan a legal label plan on purpose**. That property is what bought
the entire proven lifecycle; it is also what let the wrong binary consume the
plan silently and succeed at the wrong job for 1.4 hours.

### Two guards, one of them structural

**`samples: 0`.** The probe never reads that field — its ladder lives in
`probe.rungs` — so zero costs the probe nothing. A label worker handed this plan
builds `JointExactConfig(evaluation_samples=0)` and dies in
`__post_init__`: *"evaluation_samples must be a positive integer"*. **Proven
rather than asserted**: the real worker was run against the new plan locally and
exited 1 having written zero files, before any engine call. The failure mode
moves from "1.4 hours of the wrong output" to "first position, seconds, loud".

**`m7v4_t2probe_preflight.py`.** Run on the receipt you are about to execute, it
refuses anything whose `startup_script_relative` is not
`scripts/startup_hu_m7v4_t2probe_v1.sh` at sha `3c55705a…`, whose worker plan
carries no `probe` block, or whose `samples` is nonzero. Tested both ways: it
passes the new receipt and refuses the old one by name.

The plan now also carries `probe.expected_output`, so a watcher can tell within
**one shard** rather than after the fleet: correct output is `root_*.json` plus a
`hu_m7v4_t2probe_shard_done_v1` marker, and `position_*.json` is the
wrong-binary signature.

### What the 1.4 hours did buy

Not nothing. 1,000 T2 positions per seat, **512 particles** — four times the
shipped corpus's 128 — with the v7 T3 continuations, T4 v6, engine `e17aa39f…`
and `fl_ev {14: 9.6}` verified on every record, over hand block
942,000,000 + 0…1,000, which is the prefix of the block the T2 corpus will use.
If the probe returns 512, these are directly reusable as the first 1,000
positions of that corpus through the existing carry-manifest path; if it returns
a different rung they are a held-out reference set at a heavier rung than the
corpus. Kept at `~/ofc-m7/probe_t2{first,second}/`.

### Relaunch: READY as `probe1000b`

Runs are write-once, so the corrected run takes fresh names. Re-rehearsed on the
**new** plan bytes — one root per seat, exit 0, geometry 7/7 and 7/9, 24 and 27
candidates — because a rehearsal that vouches for different bytes vouches for
nothing.

| | first | second |
| --- | --- | --- |
| plan | `worker_plan_m7v4_t2first_probe1000b.json` `418ddf06…` | `…second…` `2ec530f2…` |
| READY | `m7v4-t2first-probe1000b.ready.json` | `m7v4-t2second-probe1000b.ready.json` |
| run plan | `~/ofc-m7/t2probe_runs/run_plan_m7v4-t2first-probe1000b.json` | `…second…` |

The superseded markers are renamed `*.SUPERSEDED.json` rather than deleted —
they name consumed run prefixes, and a stale READY marker is exactly the kind of
thing that gets launched by accident.

### The T2 plans: exact pin set, ready to generate

T2 plans wait on the T3 v7 models (they are the continuations), but nothing else
about them is unknown. The required pins, **read off `load_plan` rather than
guessed**, so `m7v4_make_t2_plans.py` is a copy of the T3 generator with three
lines added:

| pin | T2 first | T2 second | why |
| --- | --- | --- | --- |
| `engine_library`, `feature_encoder_library` | ✓ | ✓ | base requirement |
| `t4_model` | ✓ v6 | ✓ v6 | the leaf |
| `t3_second_model` | ✓ **v7** | ✓ **v7** | base requirement — only `(T3, second)` is exempt |
| `t3_first_model` | ✓ **v7** | ✓ **v7** | `worker:288` — "the T2 teacher plays the opponent's T3 first-seat reply through that model" |
| `t2_second_model` | ✓ (gen-1) | — | `worker:295` — first seat only; the opponent's T2 reply |

Note the asymmetry: `t2_second_model` is the *opponent's* T2 policy and stays at
the current generation, because T2 is the street being relabelled — it is not
yet v7 when its own labels are generated. Both T3 pins move to v7 together.

Seed block: take the next fresh block whole across both seats, same convention
as T3 (`942,000,000` was T3's; audit before allocating). Positions and rung come
from the T2 probe verdict; if the probe is queued behind, state on the record
that the street was bought rather than measured, as T3 was.

## T1 — the probe sized before the street is opened

T1 will be the most expensive probe of the cascade: a T1 teacher's rollout plays
both seats' T2 replies, then both seats' T3 replies, then reaches T4. So the
cost was **measured**, not extrapolated from T2 with a per-street multiplier —
two sample counts per seat on real behaviour roots through the pinned v4 engine,
with the full T1 pin set (the engine refuses a T1 evaluation without every
learned evaluator pinned, which is the T1 plan's own requirement showing up as a
runtime error).

| street / seat | core-s per sample | ×  the street below |
| --- | --- | --- |
| T3 first (from the T3 probe) | 0.0648 | — |
| T2 first (rehearsal, uncontended) | ~0.232 | 3.6× |
| **T1 first (measured)** | **0.8399** | **3.6×** |
| T2 second (rehearsal, uncontended) | ~0.154 | — |
| **T1 second (measured)** | **0.4657** | 3.0× |

The per-street factor is ~3.5×, and it held from T3→T2→T1, which is worth
recording for T0 rather than re-measuring blind.

**Cost at the same ladder (128/256/512 against a 2,048 reference — the shipped
T1 corpora are 256-particle, so it brackets them):** 2,498 core-s per root at
the first seat, 1,383 at the second, **3,881 core-s = 1.08 core-h per root for
the pair**.

**Root counts** come from the same half-width rule at δ = 0.010, with the paired
dispersion again taken from the finished T3 probe and inflated (T1's regret
should scatter at least as much — the doc's standing expectation is that noise
binds harder there):

| inflation | verdict | roots | core-h | on the 464-core fleet |
| --- | --- | --- | --- | --- |
| 1.5× | gap | 636 | 686 | **1.5 h** |
| 1.5× | flat | 993 | 1,070 | **2.3 h** |
| 2.0× | gap | 1,130 | 1,218 | 2.6 h |
| 2.0× | flat | 1,766 | 1,904 | **4.1 h** |

**Presented at full size rather than shrunk.** A 1,000-root T1 probe costs about
**2.3 hours of fleet time**; the pessimistic 2× case is 4.1 hours. That is the
honest number, and it is affordable — the reason to state it plainly is that the
T3 probe was sized by what fitted in memory, answered nothing, and cost a
generation's worth of confidence instead of an afternoon of CPU.

Two things carry over to the T1 probe before it is written: it is the same
asserted-substitution job as the T2 one (`generate_behavior_t1_roots`,
`hu_m3_rust.evaluate_t1`, the T1 pin set, both T1 models included), and it
inherits the `samples: 0` guard and the pre-flight check — a T1 probe plan is a
legal T1 label plan for exactly the same reason, and would be consumed by the
label worker for exactly the same reason.

## Exact next actions — as of the T2 probe being READY

1. **Relaunch the two probe runs as `probe1000b`** — and run
   `m7v4_t2probe_preflight.py --run-plan <receipt> --worker-plan <plan>` on the
   receipt you are about to execute, first. Then check the first shard's
   `SHARD_DONE.json` schema before letting the fleet run out: `position_*.json`
   means the label worker is running.
   **The T2 label plans cannot be emitted until this returns a verdict** — there
   is no rung to emit them at. If the street is to be bought rather than
   measured, as T3 was, that is a ruling, and the honest default rung would be
   512 (the heaviest the shipped 128-particle corpus has been compared against,
   and the rung the 2,000 salvaged positions already carry).
2. **When the verdicts land**, emit the T2 plans: `m7v4_make_t2_plans.py
   --samples <rung> --t3-first v7 --t3-second v7 --probe-note …
   --probe-table <report>`. It recomputes every digest from the m7v5 archive and
   cross-checks the ledger before writing. Then rehearse through the real worker
   and write the READY markers.
3. **Report the probe with `t3noise_report.py --out <dir>`** — the report merges
   whatever is on disk at any moment, so a partially-returned fleet still reads.
   Remember the T3 lesson when reading it: check the required half-width against
   what actually arrived before believing a branch.
4. **A future T3 v8**, if the street is revisited: extend the particle ladder
   above 1024 at the second seat (512-vs-1024 still excluded zero). The position
   count is settled — the scaling extension was declined on the gate's evidence.
5. **Close the clamp debt** (see the DEBT section) before any model in this line
   is trained with a nonzero clamp again.

## Earlier next actions, kept for the record

1. **When the T3 labels return** (path announced by the supervisor): retrain
   both seats, warm-started from the current T3 models, on house trainer
   discipline — stable-hash holdout, early stopping on held-out regret, masked
   padding, seeds documented. Export T4M1 + sidecars. Then a mirrored match
   gate per street-seat against the current chain with only that model swapped,
   on the standing criterion **lower CI > −0.05 and mean ≥ 0** (harness pattern:
   the `m7prep` match-gate scripts in the session scratchpad).
2. **T2 street**: particle probe — this time *powered*, budgeted by the
   arithmetic above and run at 6–8 workers **against the v4 package runtime,
   not the generation-1 build** — plus a feature/capacity ablation on the
   existing T2 corpora, baseline reproduced first. Then verdict → plans (fresh
   T3-v7 continuations) → READY → retrain → gate.
3. **T1 street**: same protocol. Noise is expected to bind harder. The T1
   second-seat ablation is already run and returned **null**; T1 first-seat
   (T1f) is outstanding.
4. **T0 both seats**: diagnosis only. T0f ablation on the local 18.8k corpus
   when the GPU is idle. **Relabel is DEFERRED per owner** — do not plan it.

### Where things are

| what | where |
| --- | --- |
| plans + ledger + package | `/home/wner/ofc-labelgen-m7v4/package` |
| unpacked v4 runtime (v4 engine, v6 T4) | `/home/wner/ofc-labelgen-m7v4/build/runtime` |
| READY markers | `/home/wner/ofc-m7/ready/` |
| T3 probe partial data + report | `/home/wner/ofc-m7/t3noise/out/decision_v6_150/` |
| rehearsal positions and logs | `/home/wner/ofc-m7/t3_plan_smoke/`, `/home/wner/ofc-m7/logs/t3_plan_smoke.log` |
| T2 corpora | `~/ofc-t2` (second seat), `~/ofc-t2first` (first seat, 25,000 labels) |
| all `m7v4_*` drivers | session scratchpad |
| T3 v7 work root | `~/ofc-m7/t3v7/` — `{first,second}/` corpora, `curve/`, `lrsweep/`, `ship/`, `gate_{first,second}/`, `t2probe/` |
| T3 v7 shipped images | `~/ofc-m7/t3v7/ship/weights/t3first_model_v2.bin`, `t3_model_v3.bin` (+ sidecars) |
| the (T3, second) feature dump crate | `~/ofc-m7/t3second_dump`, built into `~/ofc-m7/t3dump-target` |
| T2 tier-1 features | `~/ofc-m7/ablation/feat/t2first_tier1.npz`, `t2second_tier1.npz` |
| T2 second-seat row order | `~/ofc-m7/t2second_rows/features_rs_vm_shard/keys.txt` (+ `labels_vm_shard.jsonl`) |
| **the m7v5 package** (T3 v7 promoted, 17 weights) | `/home/wner/ofc-labelgen-m7v5/package` |
| T2 probe plans, run plans, rehearsals | package above; `~/ofc-m7/t2probe_runs/`; `~/ofc-m7/t2probe/rehearse_{first,second}/` |
| the probe module the fleet runs | `src/ofc_regular/hu_m7v4_t2_noise_probe_v1.py` `e7717b1a…` (inside the package) |
| the probe startup script | `scripts/startup_hu_m7v4_t2probe_v1.sh` `3c55705a…` |

### Seed blocks spent, cumulative

**The T2 probe plans take a PREFIX of the T3 hand block, and that is deliberate
— flagged here because the seed audit will show it as a partial overlap.** The
house rule is that a plan's hand range is shared *whole* or not at all, because
a partial overlap between two **label** plans produces duplicate positions that
are invisible in the output. Neither hazard exists here: the probe writes
`root_*.json` measurements under its own run prefix, not labels into a corpus,
so nothing can be double-counted; and measuring label noise on the first 1,000
hands of the very block the corpus will use is what makes the measurement apply
to that corpus. The precedent is the T3 probe, which drew its roots from
`960,000,000` — the generation-1 fleet's block — for the same reason. The rung
this probe returns is therefore chosen on 1,000 hands that the T2 corpus will
also contain; that is a variance measurement, not a selection on labels.

**Added by the T2 probe:** rung seed base `63,000,000` and reference seed base
`350,000,000` (both fresh; the T3 probe spent `61,000,000` / `340,000,000`),
plus trainer seeds `997,990,014`–`016` for the second-seat arms' extra seeds.

**Added by the T3 v7 retrain:** match-gate deal blocks `945,000,000`+20,004
(first seat) and `946,000,000`+20,004 (second); T2 probe pilot rung base
`63,000,000` and reference base `350,000,000`; trainer seeds
`997,990,021`–`997,990,027` (021/022 curve, 023 cold curve, 024–026 rate sweep,
**027 the declared ship seed**), with `997,990,011`–`013` re-used for the T2
arms (b)/(d) because they are the seeds the arms those compare against ran on.

`940,000,000` / `960,000,000`+25,000 / `970,000,000`+904 / `971,000,000`+100
(prior plans); **`942,000,000`+18,000 (M7 v4 T3, both seats)**; eval bases
`5,000,000` `6,000,000` `8,000,000` spent, **`9,000,000` now spent by M7 v4
T3**; probe `61,000,000` and `340,000,000`; T4 v6 `930,000,000` and seed
`20260726`; smokes `998,600,000` / `998,700,000`; `998,000,000`–`998,507,499`
various. **Run the seed audit before allocating anything new** — it reads every
plan on disk rather than trusting this list.

### Environment traps that cost time here

* The outer Git Bash mangles `$VARS` and `$?` through `wsl.exe` — put logic in
  `.sh` files and invoke them with `MSYS_NO_PATHCONV=1`, or the script path
  itself is rewritten to `C:/Program Files/Git/mnt/...`.
* Detached jobs need `PATH="$HOME/.cargo/bin:$PATH"`.
* Verify every detached launch with an indirect `pgrep` (`'[t]3noise_probe'`)
  so the check does not match itself.
* The box has 16 cores. A local probe at 13 workers starves anything that
  gates a fleet launch; check what is running before starting long local work.
