# T0 vs Fantasyland: measured against a real yardstick, and stopped

2026-08-10 (JST). Companion to [`vsfl_streets_20260809.md`](vsfl_streets_20260809.md),
which left three questions open about T0. This closes them and records why the
next corpus was **not** built.

The short version: **the T0-vs-FL model puts the best opening in its top three
98.1% of the time and its top ten always, and costs 0.078 points a hand when it
picks wrong.** That is inside the precision of the best measurement available,
so the ~$200 of relabelling that was costed out was not bought. What was bought
instead — a 34.6x cheaper labelling path and a silent out-of-memory kill — is
banked whether or not the corpus is ever rebuilt.

## 1. Every T0 number before today was scored with a broken ruler

The corpus was labelled at 16 continuation draws either side. Two seeds of that
teacher disagree by 0.43 on which opening is best, while the model's own error
was reported as 0.23. A ruler coarser than the thing it measures cannot say
whether 0.23 is good, and inside that gap three extrapolations and one
"saturated" reading had already been wrong.

So 200 of the held-out positions were relabelled on forty machines at **64
draws over the full 232-opening fan** — `t0first-vsfl-ref200-d64-pool-r1`, 40
shards, asia-northeast1-b, 10:35Z to 11:33Z, ~$11. All 200 sit inside the
training holdout (encoder rows 2780-3377 against a holdout of 2780-3479), so
nothing scored against them was ever trained on them.

### What the corpus labels are worth

| judged on the reference | agrees | mean regret | p90 | worst |
| --- | --- | --- | --- | --- |
| corpus label, 16 draws | 71.5% | 0.1572 | 0.6527 | 2.0768 |
| the model | 82.5% | 0.0538 | 0.0987 | 1.4364 |

**The model is three times more accurate than the labels it was trained on.**
That is the case for cheap labels in one line: the fit averages the noise out
rather than memorising it, so 30,000 noisy positions beat a few thousand careful
ones. It also explains why every earlier T0 reading was pessimistic — the model
was being marked wrong wherever the label was wrong.

## 2. The target, stated exactly

Goal: the best opening inside the model's top three, essentially always.

| k | best opening is in the top k | regret of best-of-k | worst |
| --- | --- | --- | --- |
| 1 | 82.5% | 0.0538 | 1.4364 |
| 2 | 96.0% | 0.0059 | 0.3720 |
| **3** | **98.5%** | **0.0022** | 0.3720 |
| 5 | 99.0% | 0.0021 | 0.3720 |
| 10 | **100.0%** | 0.0000 | 0.0000 |

Across four training seeds: top-3 **98.1%, spread 1.0 point**; top-1 80.1%,
spread 4.0; regret 0.078. The shipped image is the best of the four (0.0538) —
its number is real for that artefact, but the recipe's expectation is 0.078 and
that is what a retrain should be held to. **Top-3 is the stable statistic**;
top-1 is not.

### Why 0.078 is small enough

* The reference's own standard error on its best candidate is **0.041**. The
  model's error is under two of those.
* In **64.5%** of positions the reference cannot separate its own #1 from its
  own #2, and in 37% not from its #3. Most of what "the model missed" names is
  two openings worth the same.
* The reference's median margin between its #1 and #2 is 0.634, so 0.078 is
  about an eighth of the gap the decision is even about.
* An OFC hand swings by several points. This does not show up in play.

What 200 positions cannot do is separate 98.1% from 100%: the interval around
98% is ±2 points. Settling that needs a 2,000-position reference, which at
16/64/2 and K=32 is 63 VM-hours — **$10-18**, an order of magnitude under a
corpus rebuild. It was left unbought too, but it is the cheap thing to buy first
if the question is ever reopened.

## 3. The axes, now that a rung costs seconds

Narrowing plus the pre-solved library took a T0 root from 130.5 seconds to 3.77,
which made the ladder that had been unaffordable since the first T0 run simply
measurable. Six roots, two seeds, K=20, on twelve threads:

| samples | t1/t2 draws | t3/t4 | s/root | teacher's own noise | bias vs the top rung |
| --- | --- | --- | --- | --- | --- |
| 16 | 16 | 2 | 3.7 | 0.4263 | 0.0599 |
| 16 | 32 | 2 | 14.6 | 0.1750 | 0.0143 |
| 16 | 64 | 2 | 55.9 | 0.0954 | 0.0000 |
| 16 | 64 | 4 | 88.2 | 0.0656 | 0.0000 |
| 32 | 64 | 4 | 94.1 | 0.0654 | 0.0000 |

* **`samples` does nothing.** 16 to 32 moves the noise from 0.0656 to 0.0654.
  The Fantasyland side is solved exactly by the frontier, so it contributes no
  variance to remove; the variance that is left is hero's own draw. The axis the
  library made cheap is the axis not worth buying.
* **`t3/t4` draws do nothing.** 2 and 4 pick the same opening on 6 of 6 with a
  0.0000 difference, for 58% more time.
* **`t1/t2` draws are the only lever**, and they are also where the corpus's
  systematic error lives: 0.060 at 16 draws, 0.014 at 32, 0.000 at 64.

The 0.060 matters more than it looks. Noise averages away over positions;
systematic error does not. At 16 draws it is the same size as the model's whole
error, so **covering more of the space at 16 draws would have hit a floor around
0.06** — which is why the costed plan raised t1/t2 to 32 rather than buying more
cheap positions.

## 4. Narrowing is free at K=20, and the cost is not on the clock

Cut by the shipped `vsfl_t0_v1.vfl1` on 700 held-out positions:

| K | keeps the label's argmax | mean regret | worst | s/root |
| --- | --- | --- | --- | --- |
| 10 | 99.1% | 0.0024 | 0.7575 | — |
| **20** | **99.7%** | **0.0000** | 0.0184 | **3.77** |
| 40 | 100.0% | 0.0000 | 0.0000 | 9.40 |
| 232 | — | — | — | 42.0 |

Against the reference the true best sits inside the model's top ten in 200 of
200 positions, so K=20 carries a factor of two in hand.

The real cost of narrowing is not the dropped answer, it is what the corpus can
teach: **a position labelled at K=20 carries values for twenty openings, so the
next model cannot learn that its own 50th choice was best.** Each generation
inherits the last one's blind spot. Auditing 2% of positions at K=64 detects
that for +6%, and any bulk narrowed run should carry that slice.

Predictions before the measurement were wrong in both directions, again:
narrowing was called at 11.6x and came in at 1.42x against the solve-per-leaf
shape; the library was called at 3.1x and came in at 2.3x. Together they gave
34.6x, because the library removes the per-root fixed cost that was masking the
fan, and only then does cutting the fan scale.

## 5. Two defects found by measuring rather than by failing

**The frontiers were being copied out of the pool.** `t0_teacher` materialises
the whole draw tree before scoring any opening, and the library path cloned each
leaf's frontiers. At 64 draws either side and four below that is 65,536 leaves
of copies: 32 GB, and the kernel killed the process with no message at all — the
rung simply produced an empty output file. `Cow::Borrowed` from the pool fixes
it. Verified: the same rung under both builds is **byte-identical**, peak RSS
714 MB against ~16 GB, and 6% faster. The rung that was killed now runs at 1.0 GB.

A c4-standard-8 has 30 GB, so this was a live way for production workers to die
silently — the same shape as the solver-chunk stall that cost 4.25 hours.

**Training longer makes T0 worse.** Same corpus, same seeds, scored on the
reference:

| stopping rule | regret | agrees |
| --- | --- | --- |
| fixed 70 epochs | 0.0695 | 81.2% |
| fixed 120 epochs | 0.1330 | 74.0% |
| best of 12 checkpoints, 700-position holdout | 0.0715 | 80.2% |
| best of 12 checkpoints, 500 clean positions | 0.0826 | 79.7% |

120 epochs is nearly twice the error of 70. The exporter's default of 150 with
best-epoch selection was hiding that behind the selection. The 700-position
holdout overlaps the reference by 200, and choosing an epoch by it is worth
about 0.011 — small, but it is the answer sheet, and the honest holdout is the
500.

## 6. What the rebuild would have cost, if it is ever wanted

Anchor, measured rather than assumed: the reference run put **one c4-standard-8
at 1.08x this machine's twelve threads** (697 VM-seconds a position against 648
local, at full fan and 64 draws).

Spot prices pulled from the Cloud Billing catalogue, c4-standard-8 = 8 vCPU +
30 GB: europe-west4 **$0.165/h**, us-west1/us-east1/us-central1 **$0.237/h**,
asia-northeast1 **$0.293/h**.

| work | positions | VM-hours |
| --- | --- | --- |
| remaining classes at 16/32/2, K=20 | 104,022 | 456 |
| relabel the existing corpus to match | 30,437 | 134 |
| 2% audit at K=64 | 2,689 | 26 |
| 2,000-position reference at 16/64/2, K=32 | 2,000 | 63 |
| **total** | | **679** |

$199 in Tokyo, $161 in us-west1, $112 in europe-west4, before a preemption
contingency. The whole T0 space is 134,459 suit-isomorphic classes, so this is
a bounded job: it ends, and after it there is nothing left to buy but draws.

**Not bought.** The model is already inside the reference's resolution, and the
0.014 of systematic error that 64 draws would remove over 32 is smaller than the
0.041 the reference can see.

## 7. Normal-table and vs-Fantasyland T0: the same decision, almost

The question this teacher was written to test, finally asked with a real ruler.
The shipped normal-table T0 model — which knows nothing about Fantasyland — was
run over the same 200 positions through the m3 engine's `model_scores`, and both
models' picks were scored by the reference's values.

Only one direction is answerable. The vs-FL side has a reference; the normal
side does not, and cannot cheaply get one — its teacher gave five different best
openings across five seeds with per-action scores moving 26 points. So "what
does the vs-FL opening cost on a normal table" stays unmeasured. What is
measured is the reverse, paired over the same positions:

| depth | normal model costs | vs-FL model costs | paired difference (95%) | better / worse / same |
| --- | --- | --- | --- | --- |
| top-1 | 0.1260 | 0.0538 | **+0.0722 [+0.0311, +0.1142]** | 31 / 11 / 158 |
| top-3 | 0.0072 | 0.0022 | +0.0050 [−0.0037, +0.0174] | 4 / 1 / 195 |

**At top-3 the two are the same decision.** The interval covers zero and the
outcome is identical on 195 of 200 positions. **At top-1 the difference is real**
— the interval excludes zero — but it is 0.072 against a median #1-to-#2 margin
of 0.634, and the two models pick differently on only 42 of 200.

The orderings agree in both directions, which is the closest thing to a
symmetric statement available without a normal-table reference:

| | median | p90 | inside the other's top 3 | top 5 |
| --- | --- | --- | --- | --- |
| vs-FL pick, in the normal order | 1 | 2 | 97.0% | 100.0% |
| normal pick, in the vs-FL order | 1 | 3 | 95.5% | 99.0% |

Neither column is a cost — the normal model is not truth on its own street
either — but an opening the other model also ranks first or second is not a
departure from that street's play.

### Where the difference comes from

At T0 first seat both boards are empty and nothing has been discarded, so hero
faces **identical information** in both games. Any real difference has to be the
objective, not the knowledge: a Fantasyland opponent is far stronger on average,
so contesting rows is worth less and hero's own Fantasyland entry is worth more.

The disagreements point that way:

| | cards up top | in the bottom | reaching for Fantasyland |
| --- | --- | --- | --- |
| the 42 disagreements, vs-FL pick | 0.90 | 2.69 | 6/42 |
| the 42 disagreements, normal pick | 0.69 | 2.71 | 1/42 |
| all 200, vs-FL pick | 0.76 | 2.75 | 18/200 |
| all 200, normal pick | 0.71 | 2.76 | 13/200 |

The vs-FL model puts more cards up top and reaches for Fantasyland more often,
which is the predicted direction. **6 against 1 is six against one**, though —
consistent with the theory, nowhere near enough to establish it. It is the thing
to check on a larger reference, not a finding.

## 8. Can the vs-FL opening replace the normal-table one? Measured: yes

2026-08-11. §7 scored both policies on the vs-FL reference, which answers what
the *normal* opening costs against a Fantasyland opponent. The replacement
question is the other direction, and it needs a normal-table judge.

### The judge, and why the argmax problem does not block it

The engine has one — `kind: "t0"` scores all 232 openings — and it ships at
`evaluation_samples = 1`. Every opening is scored on **one** sampled world.
That is the entire explanation for "five seeds gave five different best
openings"; it was never a deep property of the street. It also means the normal
T0 model was distilled from single-particle labels, which is the likeliest
reason it recovers its own teacher's pick barely half the time.

A one-particle argmax over 232 is useless. A one-particle *difference between
two named openings* is not, because the engine scores every action against the
same particle batch, so the gap is paired and most of the variance cancels. And
the selection and evaluation batches are disjoint (`sample_independence:
disjoint_particle_rng_keys`), so the reported score of an action is not the one
that chose it.

Two further economies made the run affordable. Only the 42 of 200 positions
where the policies actually differ were evaluated — on the rest both play the
same opening and the gap is identically zero. And width was bought by
**repeating** a one-particle evaluation rather than raising
`evaluation_samples`, which is superlinear: 1 particle 36.4 s, 2 particles
106.9 s, 4 particles 210.3 s. Repeating is 1.45x cheaper per particle and
parallel across cores, since the engine runs at one core per process.

`decide` — what the trainer actually plays — was confirmed to return the
`model_scores` top on 42/42 of the positions the comparison rests on, so the
normal arm is the opening the product plays.

### The result

42 positions, 128 repeats each — 5,376 playouts. The gap is the vs-FL pick minus
the normal pick, scored by the engine on a normal table.

| | mean gap | 95% |
| --- | --- | --- |
| over the 42 disagreements | **+0.445** | **[−0.088, +0.953]** |
| per position over all 200 | +0.094 | [−0.018, +0.200] |

**No measurable difference.** One position clearly favours each side and the rest
sit inside their own error. The direction is positive but the interval covers
zero, so "the vs-FL policy is better" is not supported; "it is not worse" is.

That is the answer the replacement question needed. It also leaves the reason to
switch where it started: the two policies play equally well, and the vs-FL
teacher costs 3.77 s a position against the normal one's 26 s, with label noise
smaller by more than an order of magnitude.

**A caution worth recording, because it was ignored here twice.** At 36 and again
at 37 finished positions this read `+0.659 [+0.205, +1.130]` and
`+0.575 [+0.105, +1.054]` — both excluding zero, both reported as if they meant
something. The last five positions pulled the mean to +0.445 and the interval
back over zero. Partial results in this measurement are not weak evidence of the
final answer; they are a different answer.

Precision itself behaved as designed: per-position resolution went from ±2.87 at
32 repeats to ±1.40 at 128, the √4 the sample size predicts. The decomposition
showed measurement error dominating between-position variance, which is why
repeats were bought rather than positions — though with hindsight the 42
positions were the binding constraint on the interval, not the repeats.

## 9. Wiring the model into the engine: built, and blocked one step short

The engine loads `T4M1` images; the model is a `VFL1` image. They hold the same
network — 168 inputs, four Linear/ReLU layers with a linear head, weights
row-major `[output][input]`, little-endian f32 — and differ only in the header.
Three things had to be established before a conversion could be trusted, and all
three were measured rather than assumed:

* **The clamp is real.** Over 7,061,384 candidate rows the largest standardised
  feature reaches **1141.8** against a clamp of 8, and the clamp binds on
  **2.89%** of rows. Dropping it would ship a model nobody measured.
* **Version 1 cannot carry the weights.** It stores `std` and rejects zeros;
  the image stores `inverse_std`, where zero means "no spread, pin to the mean".
  **95 of the 168 features are constant** at T0 first seat, so a `std` image
  would need 95 infinities.
* **Existing images must not move.** Every other model the engine pins was
  fitted without a clamp.

So `t4_model.rs` now accepts a **version 2**: same magic, a clamp after
`input_dim`, and `inverse_std` stored directly. Version 1 reads exactly as
before — same bytes, same arithmetic, no clamp. A converter re-wraps the `VFL1`
header and copies the body through unchanged (1,004,920 bytes in and out).

### Where it stops

End-to-end, the engine loading the converted image agrees with the PyTorch model
on **152 of 200** openings. Not 200. The image is being read — random weights
would agree on about one in 232 — so the conversion and the loader are doing
their job and **the inputs differ**.

*(§9 records how the blocker was found. §9a records how it was removed — the
model now serves identically from both ends and the replacement is ready to
pin.)*

Three things were then ruled out, which narrows it to one place.

**The override reaches the evaluator.** The swapped engine agrees with PyTorch
on 152 and with the *unswapped* engine on only 140. A no-op override would have
scored the other way round, since the two policies already agreed on 158 before
any of this. The image is loaded and used.

**The opponent tail is innocent.** The vs-FL encoder's docstring says those
rankers were trained on *"the engine's 168-wide row: `structural | hero outlook`
real in `[0, HERO_BLOCK_END)`, **opponent tail zeroed**"*, and the engine fills
that tail in when scoring a normal T0. But of the 168 features exactly **73 are
live** (non-zero `inverse_std`) and **all 73 sit in `[0, 122)`** — none in the
tail. A feature with `inverse_std = 0` contributes `(x − mean) × 0 = 0` whatever
the engine puts there, so the tail cannot move a pick.

**It is not a layout mismatch.** Both sides call the same encoder: the engine
runs `fast_encode_hidden_opponent`, and the corpus records its encoding as
`engine_fast_features_hidden_opponent_v1`.

So the disagreement is in the **hero and structural block**, produced by the
same function on both sides, which leaves the arguments. The engine calls it as
`fast_encode_hidden_opponent(observation, board_after_action, unknown, cache)`,
and `unknown` is the strongest suspect: the vs-FL teacher's opponent holds
fourteen cards at once while the normal opponent draws progressively, so the two
can derive different unseen sets from the same observation.

**Next step, and it is one measurement:** add a request that returns the
168-wide row, and diff one position's row against the corpus row across the 73
live features. Whichever disagrees names the cause exactly, and the fix is then
either an argument correction (cheap) or re-extracting the corpus features
through the engine's own T0 entry point and refitting (certain, and the fit
itself is four minutes).

Until then the replacement is not safe to pin, and the A/B result above stands
on the PyTorch model rather than on anything the engine currently serves.

## 9a. Removed: the model now serves identically from both ends

The diff was taken, and it named the place exactly.

`t0_features` was added to the engine — a request returning the 168-wide row the
T0 first-seat model is handed, from the same `fast_encode_hidden_opponent` call
the scoring path makes, so it cannot report a row the model was not given.
Comparing 696 candidate rows against the corpus, in standardised units so a
feature the model barely weights is not mistaken for one it leans on:

| block | range | verdict |
| --- | --- | --- |
| structural | `[0, 86)` | **identical to the bit** |
| hero outlook — histograms | `[86, 113)` | 14 features differ, 0.7-2.9 apart |
| hero outlook — room | `[113, 116)` | identical |
| hero outlook — forced foul | `[116, 118)` | **feature 117 differs by 51.1** |
| hero outlook — slack | `[118, 120)` | 1.8 apart |
| hero outlook — FL, royalty | `[120, 122)` | **9.6 and 26.8 apart** |
| opponent outlook, head-to-head | `[122, 168)` | inert — no live feature |

27 of the 73 live features, every one of them in hero's own outlook, and every
one a forward-looking quantity: what the unseen cards are assumed to do. The
structural block agreeing exactly rules out a layout mismatch and leaves the
arguments, exactly as suspected.

**The fix was not to decide which side was right.** The engine is what serves
the model in production, so its rows are the definition; the corpus features
were re-extracted through `t0_features` and the model refitted on those. Twenty
minutes end to end: two to re-extract 30,437 positions across twelve processes,
two and a half to refit, one to convert and check.

| | regret | top-1 |
| --- | --- | --- |
| `vsfl_t0_v1`, offline features | 0.2331 | 64.1% |
| **`vsfl_t0_v2`, engine features** | **0.2189** | **65.7%** |

The refit is *better*, so the engine's outlook is the more informative of the
two encodings, not merely the authoritative one.

**Engine and PyTorch now pick the same opening on 200/200.** The distinction
that mattered is worth stating plainly, because three things were live at once
and only one of them was ever measured:

| | what it is | measured |
| --- | --- | --- |
| v1 + offline features | what the A/B scored | +0.095 |
| v1 + engine features | what the engine actually ran | never — a model fed inputs it had not seen |
| v2 + engine features | consistent end to end | +0.116 |

### v2 re-measured, and the answer holds

v2 is a different policy, not a re-export of v1: they pick the same opening on
170 of 200, and v2 disagrees with the normal model on 50 positions where v1
disagreed on 42, overlapping on 34. So the match A/B was rerun against v2's own
openings — 50 positions, 200 trials each, 10,000 hand-pairs.

| | mean gap | 95% |
| --- | --- | --- |
| v2, over the 50 disagreements | **+0.116** | [−0.228, +0.448] |
| v1, over its 42 | +0.095 | [−0.275, +0.453] |
| per position played (v2) | +0.029 | [−0.057, +0.112] |

Two independently fitted models, differing on 30 of 200 openings, measured two
different ways, all returning no difference larger than about 0.1 a hand. The
replacement is supported.

### Ready to pin, deliberately not pinned

`vsfl_t0_v2.bin` (sha256 `f6f9631f…`) is a version-2 image the engine reads and
reproduces exactly. The runtime is untouched: `t0first_model_v1.bin`
(`07301034…`) is still what production loads, and no vs-FL file has been copied
into the runtime weights directory. Pinning is three edits — the image into
`runtime/weights`, the path and digest in the config, and the trainer's own
`WEIGHTS` table — and it changes what the product plays, so it waits for a
decision rather than following from the measurement.

## 10. The same 30x is available on the normal table, mostly already built

Found while looking for the judge above, and worth recording because it is the
cheapest unclaimed speedup in the repository.

The vs-FL side's 34.6x came from two independent things: a pre-solved
Fantasyland pool (3.1x) and cutting the 232-opening fan with a model before any
particle is spent (11.1x). The pool does not port — it works because the
Fantasyland side can be solved exactly, and a normal opponent responds to hero's
board so there is no board-independent object to precompute. **The fan cut does
port, and it is the larger half.**

The engine already implements it. `narrow_by_learned_model` keeps the top `k`
candidates by the street's learned evaluator before any particle is drawn, and
its own comment records the measurement: *"the rollout's own best action
survives the model's top ten 91-96% of the time … At top five, survival falls to
84-86% for a saving that is 5.4x rather than 2.6x"*. It is off by default, with
the same reasoning this project reached independently: *"Off for label
generation unless a plan sets it: narrowing changes what a label means."*

It matches T3, T2 and T1 — and falls through to `_ => return Ok(actions)` for
T0, and is not called from `evaluate_t0`. The config already carries
`learned_t0_first_model_path`.

Measured on one position, `prefilter_keep` (T0's *sampled* two-stage cut, which
is a different mechanism) gives only 1.74x, and the cost model says why exactly:

| shape | opening-evaluations | predicted | measured |
| --- | --- | --- | --- |
| no cut | 232 x 2 batches = 464 | — | 36.4 s |
| `prefilter_keep = 40` | 232 + 80 = 312 | 24.5 s | 22.3 s |
| `prefilter_keep = 20` | 232 + 40 = 272 | 21.3 s | 21.0 s |
| **model cut to 20** | **20 x 2 = 40** | **~3.1 s** | not built |

The floor is the stage-one pass, which by construction touches all 232. A
deterministic model cut does not — 464 to 40 is **11.6x**, which is the vs-FL
side's 11.1x arriving by the same arithmetic. Composed with the 2.6x already
measured for T1-T3 (different work: fan size versus rollout depth), that is
**about 30x**.

One caveat carried over from the vs-FL side: the 91-96% survival was measured
against *the rollout's own best action* at one particle, a quantity with a
per-run spread near 18.7. Most of the missing 4-9% is that noise, not narrowing
loss, so the true safety is better than the number suggests — and it can now be
measured properly.

## 11. Played hands, not evaluated ones

§8's A/B differenced the engine's *evaluation* of two openings. The owner
specified the sharper design and it is the one to trust: deal a deck, play the
whole hand out with the engine deciding every turn for both seats, then replay
the identical deck changing only hero's T0 first-seat placement, and difference
the scores.

The distinction is not pedantic. An evaluation holds hero's opening fixed and
rolls forward; a real opponent *sees* that opening and answers it, so the two
arms diverge in the opponent's play as well as hero's. Only the match captures
that.

It is also far cheaper. `decide` — what the trainer actually plays — is 0.09 s,
against 26 s for a full 232-opening evaluation:

| | per unit | total for the comparison |
| --- | --- | --- |
| evaluator A/B | 26 s a position | 5,376 evaluations ≈ 39 CPU-hours |
| **played-out A/B** | **2.15 s a hand-pair** | 10,000 pairs ≈ **5 CPU-hours** |

and its interval came out tighter (±0.34 against ±0.43).

Fantasyland entry is scored, at `FL_EV_14`. Leaving it out would mark the vs-FL
policy down for the thing it is buying — it reaches for a qualifying top row
more often than the normal model does.

Two structural notes for anyone extending it. Hero's opening five are pinned to
reference positions because that is where the vs-FL model's pick is known; going
beyond 200 positions needs features for arbitrary hands, which the `t0_features`
endpoint now makes possible. And positions where the two policies already agree
are skipped — both arms are the same hand played twice, contributing exactly
zero — so the aggregate is scaled by the disagreement rate rather than padded
with guaranteed draws.

Roughly a third of hands score identically even when the openings differ (10% to
36% by position): the difference washes out by the river. The positions with the
largest gaps are the ones where it does not.

## 12. What is still not known

* **Whether top-3 is 98% or 100%.** 200 positions cannot tell. $10-18 of
  reference would.
* ~~**What the vs-FL opening costs on a normal table.**~~ Answered in §8 and
  §11: nothing measurable, by two methods and two models.
* **Whether the advantage is exactly zero.** It is not shown to be. The point
  estimate is +0.116 over the disagreements and the interval reaches +0.45; a
  real effect of that size would be worth having. Settling it needs roughly four
  times the positions, which now needs features for hands outside the reference
  set — possible since `t0_features` exists, and about 13 hours on twelve cores
  at no cloud cost.
* **Whether the Fantasyland-seeking tendency in §7 is real.** Six positions
  against one.
* **T1 was never rebuilt** and its ladder was never run. Unchanged from
  2026-08-09.
