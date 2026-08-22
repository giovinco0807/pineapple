# T3: the penalties the search cannot find

2026-08-15.  A hand in the trainer's history looked wrong at T3.  It was.  This
measures how often, separates the part that costs EV from the part that cannot,
and rules on what to do about each.

## Where it started

Hand log #4, hero acting second at T3.  The trainer showed **rank 2, EV loss
0.186** for `Th->top, 4c->middle, discard 7s`, preferring `7s->top ... discard
Th`.

The two are **provably worth the same**:

* hero's top is `Kc Ah` plus one of `{Th, 7s}` — ace-high either way, no
  royalty either way;
* **all four kings are accounted for** (Kc, Kd on hero's board, Kh on the
  opponent's, Ks in hero's discards), so the opponent's top — `Ac Td` plus one
  — can never be A-K-x and a kicker comparison never happens.  Hero wins the
  top row, or loses it to a pair, identically in both lines;
* the card not placed is discarded, so the deck is identical either way;
* the foul comparison is ace-high against the middle in both lines.

The engine's own numbers agree: the two rows score `-12.087500` selection /
`-3.615625` ev, **identical to six decimals**, and the search returns the same
pair ordering on 12 independent seeds with a paired gap of exactly 0.000.

## It is not one checkpoint's slip

| checkpoint | played (Th->T) | model's pick (7s->T) | gap (truth: 0) |
| --- | ---: | ---: | ---: |
| `t3_model_v3` (shipped, v7) | −5.7096 | −5.5233 | **+0.1863** |
| `t3_model_v2` (generation-1) | −4.3005 | −4.2914 | +0.0091 |
| `t3_model_v1` (oldest) | −3.4558 | −3.1785 | +0.2773 |

Same sign, thirty-fold spread in size, and **the newest image is worse here
than the one it replaced**.  Approximation error is redistributed by a retrain,
not monotonically removed — which is the same lesson the v7 gate recorded from
the other direction (held-out regret −13.3 % at the first seat, play unchanged).

## The pilot was read on the wrong field, twice over — corrected

Everything in the two sections below was first computed on `selection_score`,
the field the engine sorts T3 by.  **That field does not converge.**  Held at
one root, moving only the evaluation batch:

| | selection gap | ev gap |
| --- | ---: | ---: |
| evaluation_samples 1,024 | +2.875 | +4.782 |
| 8,192 | +7.325 | +4.989 |
| 32,768 | +6.950 | +4.924 |
| candidate_samples 8 → 512 (eval fixed) | +7.325 → +4.438 | +4.989 → +4.927 |

`score` (the evaluation batch's ev) settles inside 0.2; `selection_score` swings
by more than four points and moves with *either* knob.  The pilot recorded both,
so the numbers below are the same roots re-read on ev.  Three conclusions
reversed, all of them in the favourable direction — most importantly, **the
referee turned out to be good enough after all**.

This also indicts something in production: `trainer/README.md`'s rank-field
table sends T3 and T4 to `selection_score`, so **the trainer's teacher grading
ranks T3 on a statistic that does not converge**.  That is the mechanism behind
hand #3's four distinct best moves over eight seeds.  Ranking T3 by `ev`
satisfies the same rank-and-score-must-agree rule the table was written to
enforce, and is stable.

## How common, measured

`t3_tie_pilot.py`: behaviour roots from a fresh block (hand base 950,000,000),
the model's top two re-scored by the search at **8,192 particles on two
independent seeds**.  Two seeds because "the search finds no difference" only
means something against what the search does when nothing changed.

Seats reported separately — they are not one exam (the doc's own rule), and
they cost eight times apart.

A tie on a continuous scale is called at **≤ 0.01 on both seeds** — two orders
below the model's own held-out regret.

| | first (n=58) | second (n=108) |
| --- | ---: | ---: |
| search vs itself, mean \|gap_a−gap_b\| (the floor) | 0.5261 | 0.3103 |
| …median | 0.0492 | 0.0187 |
| model vs search, mean \|error\| | 0.7995 | 0.7071 |
| …median | 0.2031 | 0.2018 |
| **model error / floor** | **1.52×** | **2.28×** |
| corr(model gap, search gap) | +0.863 | +0.851 |
| search put the pair within 0.01, both seeds | 6.9 % | **34.3 %** |
| …and the model still charged more than that | 75.0 % of those | 62.2 % of those |
| **spurious penalties, share of all roots** | **5.2 %** | **21.3 %** |
| what it charged there (median / p90 / max) | 0.071 / 0.077 / 0.118 | 0.028 / 0.122 / 0.272 |

**One T3 spot in five at the second seat is shown a penalty the teacher puts at
zero**, and a third of that seat's roots are genuine ties between the top two —
the street is often forced.

## The part that costs EV, and the referee that can see it

**The model's error is 1.5–2.3× the referee's own floor.**  That settles the
question the `selection_score` version got backwards: at 8,192 particles on the
narrowed pair, the search resolves better than the model errs, so it *can*
grade a retrain.  No sharper referee has to be bought first.

Real errors — both seeds agree the pair is separated and agree which way, they
disagree by less than the effect, and the model ordered it the other way:

| | first (n=58) | second (n=108) |
| --- | ---: | ---: |
| roots with a real separation | 86.2 % | 62.0 % |
| **model ordered against the search** | **8.6 %** | **10.2 %** |
| what those cost (median / max) | 0.057 / 1.178 | 0.146 / 0.857 |
| EV cost averaged over all roots | **0.0257** | **0.0280** |

So the T3 model misorders about **one root in ten**, and a perfect T3 model is
worth on the order of **0.026–0.028 a hand** over this one.  That is a real
number resting on 16 roots rather than four, and it is measurable — which the
earlier reading said it was not.

## Is the value function distorted, or only blind at ties?

*(The slope and floor figures in this section were computed on
`selection_score` and are superseded by the ev reading above; the retraction of
the shrinkage claim stands, since selection bias was its cause and that is
field-independent.)*

Asked because "a missed tie costs no EV" answers the *play* question and not
the *model* question: a value function that prices equal boards unequally is
wrong, whatever it costs to act on.

**A first cut said the model was shrunk** — gaps at 0.73–0.76× the search's
across every band at the first seat.  **That was selection bias and it is
retracted.**  Bucketing roots by the search's gap and then comparing the
model's gap to that same number selects on noise and measures it in the same
breath, which drives the ratio under one by regression to the mean.

Selecting on seed A and measuring on seed B:

| | fitted slope (truth = a × model) | \|error\| raw | rescaled | search's own floor |
| --- | ---: | ---: | ---: | ---: |
| first (36 live roots) | **1.114** | 2.238 ± 0.685 | 2.334 ± 0.715 | 2.356 ± 0.878 |
| second (46 live roots) | **0.961** | 1.961 ± 0.554 | 1.938 ± 0.544 | 2.408 ± 0.952 |

Slope ≈ 1 at both seats, and rescaling does not reduce the error.  **There is
no scale error to correct.**

What survives is per-position noise, read off the roots where the truth is
exactly zero — the only place the answer is known:

| | ε = what the model says when the truth is 0 |
| --- | --- |
| first | mean **0.147** ± 0.077, median 0.095, max 0.401 |
| second | mean **0.100** ± 0.044, median 0.034, max 0.792 |

ε is the part that matters twice over: no rescaling can remove it (nothing
times 0.10 is 0), and it is the *only* part that can flip a ranking, since a
monotone rescaling cannot change an argmax.  So the owner's objection — that a
difference in expected value is itself the defect — lands on exactly the right
quantity.

## Rulings

**1. Rank T3 by `ev`, in the trainer and anywhere else that reads
`sorted_index`.**  The engine's T3 ordering comes from a statistic that does not
converge in either sample count; ev does.  This is the cheapest fix here and it
is the one that explains the seed instability a user can see.

### Ruling 1, done and verified — 2026-08-15

`RANK_FIELD_BY_TURN[3]` is now `("ev",)` and `_normalize_rows` sorts by the
number it prices the rank with, so the ordering follows the ranker instead of
the engine's `sorted_index`.  T0 keeps the candidate batch (its staged search
prices pruned rows on fewer particles, so its `ev` is not comparable across
rows); T4 is left alone as unmeasured.  Six tests in
`tests/test_trainer_t3_rank_field.py`, including one that drives the real
engine across three sample counts and requires the winner not to move.

The stored hands, re-graded through the API before and after:

| hand / seat | before | after |
| --- | --- | --- |
| #2 hero first | disagree, teacher ranked the played move 5th | **agree** |
| #2 opp second | agree | disagree by 0.015 — a tie in all but name |
| #3 hero first | disagree, **teacher charged 1.550** | **agree, 0.000** |
| #3 opp second | agree | agree |
| #4 hero second | disagree | **agree** |
| #4 opp first | disagree | **agree** |

Disagreements 4/6 → 1/6.  The 1.550 that looked like a model error was the
ranking field, not the model.

And on the hand that started this, the teacher now returns **rank 2, loss
0.000** — a tie, priced correctly, exactly as the card argument says it should
be.  The model still says 0.186.  What is left is the real ε.

**2. Then fix the display.**  A 5 % (first) / 21 % (second) rate of penalties
the teacher puts at zero is a defect in what the trainer tells a user.  When the
model charges a loss smaller than it can resolve, re-grade that one decision
with the search before showing a rank.

### Ruling 2, done and verified — 2026-08-15

`handlog.CONFIRM_STREETS = {3}`, `CONFIRM_BELOW = 0.30`, `CONFIRM_PRECISION =
"deep"`.  The threshold is measured, not chosen: every spurious claim in the
pilot was under 0.30 (max 0.118 first, 0.272 second), so it catches 100 % of
them at both seats and fires on 41–46 % of T3 decisions at about two seconds
each.

| threshold | spurious caught (first / second) | fires on |
| ---: | ---: | ---: |
| 0.05 | 0 % / 52 % | 16 % / 25 % |
| 0.15 | 100 % / 87 % | 29 % / 39 % |
| **0.30** | **100 % / 100 %** | 41 % / 46 % |

T0 is excluded (minutes a decision), T1/T2 are excluded until their rate is
measured, T4 is exact and has nothing to confirm.  Seven tests in
`tests/test_trainer_confirm_small_losses.py` pin the routing with a stub, so
they run without the engine.

Through the running server, on the hand that started this:

```
hand #4  hero (second)  rank 2   ev_loss 0.0000
         <- model said 0.1863, confirmed by teacher, OVERTURNED, tied=True
```

and the UI shows `2位 / 21 · 同率 · 探索で確認: 差なし`, with the model's
original number kept in the tooltip.

**It corrects in both directions**, which was not the design intent but is the
better property: on hand #2's first seat the model charged 0.0443 and the
search returned **0.5250**.  A confirmation that only ever exonerated would be
a whitewash; this one is a re-measurement.

**3. A T3 retrain is now measurable, and worth about 0.026–0.028 a hand.**  The
referee resolves 1.5–2.3× better than the model errs, so a candidate can be
graded on the ev field against this same protocol.  That is the gate the v7
generation lacked — its held-out metric moved 13.3 % and its play did not, and
neither number could see the one root in ten that is actually misordered.
Whether better labels close that gap is untested; the exam to test it now
exists.

### Ruling 3, answered — the labels are not the lever, and here is the proof

Ruling 3 asked whether better labels would close the model's gap.  Measured
2026-08-15/16, four ways, all agreeing:

**(a) The corpus already ranks better than the model.**  120 second-seat and 30
first-seat corpus positions, re-scored by the 8,192-particle referee:

| | agrees with the referee | regret vs referee |
| --- | ---: | ---: |
| corpus label (1,024p, what v7 learned) | **79.2 %** | **0.0064** |
| model v7 | 65.8 % | 0.1002 |

paired difference +0.0939 [+0.0129, +0.1748], excluding zero; on the
disagreements the corpus is right 22 times to the model's 6 (exact p = 0.004).
The first seat agrees in direction (+0.4085 [+0.0382, +0.7788]).

**(b) The features are not the cap.**  292,212 rows: 6.85 % share a feature
vector with another row, but only **3 collision groups out of 9,953** carry
disagreeing labels (max spread 0.0078), and in **0 of 18,000 positions** does
the best candidate share its vector with a worse one.  Nothing is being asked
to separate the inseparable.

**(c) Capacity helps and then stops.**  Trained with no holdout, so this is
memorisation, not generalisation:

| arch | params | train regret | train top-1 |
| --- | ---: | ---: | ---: |
| 256,128,64 (v7's) | 84k | 0.0468 | 80.5 % |
| 512,256,128 | 232k | 0.0345 | 82.1 % |
| 1024,512,256,128 | 862k | 0.0264 | 83.5 % |

An 862k-parameter network cannot memorise **1,000** positions (0.0266 / 83.3 %).

**(d) The residual is all in the fine gaps, and the labels are right there.**
Bucketed by each position's own top-two label gap:

| gap | n | train regret | train top-1 | share of total regret |
| --- | ---: | ---: | ---: | ---: |
| ≥ 2 | 3,254 | **0.0000** | **100 %** | 0.0 % |
| [0.5, 2) | 2,954 | 0.0228 | 97.0 % | 14.3 % |
| [0.2, 0.5) | 1,675 | 0.0628 | 81.5 % | 22.3 % |
| [0.05, 0.2) | 1,937 | 0.0561 | 60.8 % | 23.0 % |
| [0.01, 0.05) | 1,103 | 0.0333 | 49.1 % | 7.8 % |
| [0, 0.01) | 7,077 | 0.0217 | 82.3 % | 32.6 % |

Clear decisions are fit perfectly.  Everything the model gets wrong is a fine
one — and asking the referee who is right in that band settles it:

| gap (second seat) | n | label agrees with referee | model agrees |
| --- | ---: | ---: | ---: |
| [0.05, 0.5) | 17 | **17/17 = 100 %** | 14/17 = 82 % |
| [0, 0.05) | 57 | 32/57 = 56 % | 22/57 = 39 % |

**In the [0.05, 0.5) band — where 45 % of the training regret lives — the
1,024-particle label is right every single time and the model is not.**  Only
below 0.05 is the label itself unreliable, and there the whole decision is
worth less than 0.05.

So a higher-precision relabel would sharpen labels that are already correct
where it matters.  **The lever is the model side: features rich enough to make
the 0.05–0.5 band learnable, and capacity/data to learn it.**  Note (b) rules
out only *exact* feature collisions; it does not rule out features that are
merely too close for a smooth function to separate, which remains the leading
suspect for why (c) saturates.

### Where the lever turned out to be: the pair weighting

Two hypotheses for (c)'s saturation were tested and **both failed**:

* *the features are too close to separate* — refuted.  On the [0.05, 0.5) band,
  the pairs the model gets WRONG sit **1.29× further apart** in standardised
  feature space than the pairs it gets right (same pair definition on both
  sides: label-best against label-runner-up).  A first pass read 1.43× and
  compared (best, model's pick) on the failures, which measures how far down
  the model reached rather than how separable the decision was.
* *the encoder is thin* — 82 of 168 dimensions ever differ between two
  candidates of the same position; the other 86 describe the opponent and the
  context and are constant within a position.  Worth knowing, but it is 82
  dimensions that are demonstrably not too close.

What does explain it is the loss.  The shipped protocol weights each ordered
pair by the label gap, so a pair separated by 0.1 carries a twentieth of the
gradient of one separated by 2.0 — and the ≥2 band is already fitted **100 %**,
so that weight buys nothing.  Same architecture, same epochs, weighting only:

| pair weight | train regret | train top-1 |
| --- | ---: | ---: |
| `gap` (shipped) | 0.0358 | 81.8 % |
| `sqrt(gap)` | 0.0306 | 83.5 % |
| `uniform` | 0.0278 | **85.8 %** |
| **`gap` clipped at 0.5** | **0.0266 (−26 %)** | 84.2 % |

512-wide with a clipped weight matches the **1024-wide** net's 0.0264 under the
shipped weight: the loss change is worth as much as four times the capacity.

**All of these are training-set numbers** — the probe deliberately holds nothing
out, because it was asking what is representable.

### It survives to held-out, on six paired seeds

The shipped harness with a `--pair-weight` option added (default `gap`, which
reproduces v7 exactly), cold start, 120 epochs, split_code 230, arms sharing
every seed:

| arm | n | held regret | top-1 | top-3 |
| --- | ---: | ---: | ---: | ---: |
| `gap` (shipped) | 6 | 0.06177 | 0.7739 | 0.9413 |
| **`uniform`** | 6 | **0.05608** | **0.7909** | **0.9511** |

| paired difference (uniform − gap) | | |
| --- | ---: | --- |
| held regret | **−0.00570** [−0.01022, −0.00117] | excludes 0 (−9.2 %) |
| top-1 | **+0.0170** [+0.0101, +0.0240] | excludes 0, **better on 6/6 seeds** |

**Pairing is what made this readable.**  On the first two seeds the arms
swapped places — one had `uniform` ahead by 0.006 and the next behind by 0.002
— and an unpaired reading of those two says "no effect".  It took six seeds and
a paired comparison to resolve a difference the seed spread was hiding.

**Which metric matters depends on the consumer**, and this is the same split
that runs through the whole document.  A 9.2 % held-out regret gain deserves
the caution v7 earned: that street improved its held-out metric 13.3 % and
moved head-to-head EV by nothing measurable.  But **top-1 is what the trainer
renders as a rank**, and that is the number the owner's original complaint was
about.

### …and does NOT survive warm start, which is the shipping protocol

Same harness, same corpus, warm-started from the gen-1 checkpoint v7 itself
used (`/home/wner/ofc-t3/model_v2_512/model.pt`), three shared seeds:

| arm | held regret | top-1 |
| --- | ---: | ---: |
| **`gap` (shipped)** | **0.02964** | 0.8237 |
| `uniform` | 0.03423 (**15 % worse**) | 0.8255 |

`gap` wins on 3/3 seeds and the top-1 difference vanishes.  Warm `gap`
reproduces the shipped v7's 0.029540 almost exactly, so the harness is sound
and the reversal is real.

The reading: `uniform` helped a **randomly initialised** net find the fine
band; a warm start already carries that from generation 1, and the heavier
gradient on small-gap pairs then disturbs a function that is already close.
Its kept epochs — 26, 39, 29 — are the same "first optimiser step damages a
model that is already there" signature the rate sweep found.

### The ceiling of this corpus, measured

Every strengthening route tried on the second seat, held-out:

| approach | held regret | vs v7 |
| --- | ---: | --- |
| warm + `gap` (v7's own protocol) | 0.0296 | reproduces it |
| warm + `uniform` | 0.0342 | worse |
| warm from **v7 itself** + `gap` | 0.0303 | worse; kept epochs 10 and 36 |
| cold + `gap`, 256-128-64 | 0.0618 | far worse |
| cold + `uniform`, 256-128-64 | 0.0561 | far worse |
| cold + `uniform`, **1024-512-256-128, 300 epochs** | 0.0560 | far worse |

**v7 is at the ceiling of this corpus and this feature set.**  Capacity, epochs
and loss shape were all tried; none reaches what a warm start already gives,
and nothing beats the warm start.

### It is a fitting limit, not a generalisation one — the distinction matters

An earlier draft called this "generalisation, not optimisation".  That is
wrong, and the numbers say so.  For the shipped architecture:

| | regret |
| --- | ---: |
| training data, no holdout (memorisation allowed) | 0.0468 |
| held out, cold start | 0.0618 |
| **held out, warm start (the shipping protocol)** | **0.0296** |
| the training-data floor of a 10× bigger net | **0.0264** |

**The warm-started model's HELD-OUT error is essentially the best TRAINING
error any size of this model reaches.**  Cold start does have a generalisation
gap (0.047 train against 0.062 test); warm start closes it, because generation
1 brings knowledge from 50,000 positions with it.  What binds the shipped model
is therefore the *fitting* ceiling: with no holdout at all and 862k parameters,
it still cannot get training regret below ~0.026 or training top-1 above 83.5 %.

The implications differ, which is why the sloppy word was worth correcting:

| lever | if generalisation were the limit | since fitting is |
| --- | --- | --- |
| more positions | **would help** | does not — measured at ~0.002 per 18,000 |
| more capacity | would not help | should help, but **saturated** |
| different features | — | the only one left |

The measured scaling curve agrees with the corrected reading rather than the
first one.

It also still does not contradict the label-ceiling result (the model agrees
with the referee 65.8 % where its labels agree 79.2 %): the labels contain the
answer, and the model cannot fit it even when shown it.

**What remains genuinely unexplained** is why.  There are no exact feature
collisions, and the failures are not the pairs sitting closest together in
feature space — they are 1.29× further apart than the ones it gets right.  So
the information is present and not crowded, and an MLP still will not take it.

**So T3 model work is finished on the training side.**  What is left is the
encoder, which means Rust-side feature work and a full re-extraction: a
different size of job, and one nothing here has shown would pay.

Untested and low prior: distilling v7 into a wider net and fine-tuning that,
which would carry the warm start into more capacity.  Recorded because it is
the only route the measurements above do not close, not because they suggest
it will work.

**4. If T3 labels are rebuilt anyway, here is what they cost.**
Measured label costs at this street, per root, both seats (the second is an
eighth of the first):

| arm | core-s/root | 18k a seat | fleet-h | cost |
| --- | ---: | ---: | ---: | ---: |
| full fan, 1024p (how v7 was made) | 102.7 | 514 core-h | 1.1 | $15 |
| top-2, 1024p | 15.4 | 77 | 0.2 | $2 |
| top-2, 8192p | 123.5 | 618 | 1.3 | $18 |
| top-3, 8192p | 181.0 | 905 | 2.0 | $27 |
| top-2, 16384p | 254.2 | 1,271 | 2.7 | $38 |

`learned_prefilter_keep` works at T3 (rows scored drop to exactly 2 or 3), so
eight times the particles on the pair that matters costs about what the current
full-fan label costs.  The standing decline of a T3 corpus extension
(`hu_m7_cascade_20260806.md`) was ruled for the play-strength consumer and says
so; a grading consumer is the different consumer that ruling anticipated.

## Traps hit while measuring this, recorded

* **Thread oversubscription, twice.**  Workers took torch's default pool (49
  threads each, 771 on 16 cores) and then, after pinning OMP/MKL, still took
  rayon's (the m3 engine is Rust; `RAYON_NUM_THREADS` is the one that mattered).
  The run went at roughly a fifth of its rate and the first ETA was wrong by
  hours.  The project's fleet launchers already set `RAYON_NUM_THREADS=1`.
* **A resume that ate its own predecessor.**  The pilot opened its output `"w"`,
  so a second resume truncated the first resume's file — and because the
  launcher reads those same files to decide what is still owing, 15 roots ended
  up in neither the data nor anyone's work list.  Now opens `"a"`.
* **Cost model off by the seed count.**  The pilot scores each pair twice; the
  first estimate priced one pass.

## Precision was then ruled out directly — a null, on the metric that matters

Before buying higher-precision labels, the payoff was estimated for nothing by
**injecting** noise instead of removing it.  The corpus labels carry some
unknown sigma; adding a known one makes the total sqrt(sigma0^2 + sigma^2), so
a flat response means the model cannot see label noise at that scale at all.
Warm start, three shared seeds, held-out:

| noise added | held regret | top-1 |
| ---: | ---: | ---: |
| 0.0 | 0.02938 | 0.8210 |
| 0.1 | 0.02982 | 0.8255 |
| 0.2 | 0.02987 | 0.8226 |
| **0.4** | **0.02955** | 0.8250 |

**Adding 0.4 changes nothing** (paired 0.4 − 0.0 = +0.00017).  If that much
noise does not hurt, removing the far smaller noise already present cannot
help.  It fits the rest: the model cannot fit finer than ~0.115 a candidate, so
detail below that is invisible to it either way.

**A higher-precision T3 relabel was therefore not run**, though it was costed
(47.8 core-s a position at 8,192 particles measured, ~$7 for 18,000 on the
fleet) and authorised.  The measurement arrived first and removed the reason.

### One caveat on top-1 as a target

Raising precision would **lower** measured top-1, not raise it.  36.6 % of
second-seat positions have their top-two labels bitwise equal — common random
numbers preserve exact ties exactly, so the corpus records them correctly and
the ranking loss (which pairs only on `gap > 0`) already ignores them.  Those
are free wins on top-1.  The referee finds only 29.6 % still exactly tied at
8,192 particles, so about 7 % of positions would convert from free win to real
decision.  Better labels, lower score.

**Top-1 is the wrong target here.**  Its ceiling at this seat is
`(1 − 0.296 × 0.5) = 0.852`, against a measured 0.830 — the model is already at
97 % of a ceiling built out of coin flips.  Regret against a high-precision
referee is the metric that prices a tie at zero, which is what a tie is worth.

## T2, measured the same way — a different street

Same protocol, 24 first-seat and 64 second-seat roots (hand block 951,000,000):

| | T3 second | **T2 second** | **T2 first** |
| --- | ---: | ---: | ---: |
| ties (both seeds within 0.01) | 34.3 % | **3.1 %** | **4.2 %** |
| model error / referee floor | 2.28× | **4.12×** | **4.02×** |
| corr(model gap, search gap) | +0.851 | **+0.966** | **+0.952** |
| **model ordered against the search** | 10.2 % | **0.0 %** | **0.0 %** |

**Not one misordering in 88 roots.**  T2's model gets the order right and
misses on magnitude (mean error 0.33–0.39).  Two consequences:

* **Do not extend `CONFIRM_STREETS` to T2.**  The confirmation exists for
  penalties on ties; at 3–4 % ties and zero misorderings there is nothing for
  it to catch, and it would cost 2–22 s a decision.
* T2 is in better shape than T3, which is consistent with its provenance —
  25,000 roots at 2,048 particles, both seats ADOPT.

## Artifacts

* pilot `t3_tie_pilot.py`, report `t3_tie_report.py`, data
  `D:/ofc_data/t3_tie_pilot/` (hand block 950,000,000, eval seeds 13,000,000 /
  13,500,017)
* the seed-instability probe behind ruling 3: `t3_spot_probe.py` — hand #3's
  T3 gave **4 distinct best moves over 8 seeds**, scores spanning −3.81…−12.96
* checkpoint comparison: `t3_two_checkpoints.py`
