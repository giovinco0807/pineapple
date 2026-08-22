# How good does an interior policy have to be? (3-max, 2026-08-13)

## Why this measurement exists

The 3-max cascade is exact exactly where the tree has no opponent decision left
in it. That is T4 at every seat, and T3 for the BTN because the BTN closes the
street — both opponents have already finished their own T3, so everything below
the hero's move is one T4 round, and `exact.evaluate_t3` solves it outright at
about a second a root.

Every rung above that has an interior opponent node, and the plan for all of
them is the same: fill the node with a distilled model instead of a solver.
Nobody has measured what that costs. The previous session's finding —
[`three-max-t3-residual-settled`](../../) — closed four hypotheses for the
T3-BTN model's residual (capacity, missing features, label noise, three-handedness)
and ended on the recommendation to stop polishing T3-BTN until this number
exists, because the answer decides whether the polishing is needed at all.

(T3, BB) is the cheapest place to ask. It has exactly one interior node — the
BTN's T3 — and an exact T4 round underneath it.

## What is being compared

For one (T3, BB) root, the hero's ~17 actions are each valued by:

1. the BTN's T3 decision, taken by a swappable **interior policy**, then
2. a T4 round (SB, hero, BTN) resolved exactly by backward induction, each
   actor maximising its own 3-max total (contract R7).

Only step 1 is an approximation. Swap it, and see whether the ranking in step 2
moves.

### Draws are stratified, not i.i.d.

The BB hero sees 34 cards, so 18 are unseen: the BTN's 2 private discards, the
SB's 3, the 12 still to be dealt, and the 1 the hand never reaches. A draw
assigns all of them.

The BTN's discards are sampled rather than ignored on purpose: they are dead
cards *it* knows about and the hero does not, so they are part of the interior
information set. Handing the interior policy a deck that still contains them
would be a silent error — every board stays legal, every number stays
plausible, and the label is simply wrong.

The interior decision depends on the BTN's own cards, never on the T4 cards
nobody has seen yet. So a **stratum** fixes (BTN discards, SB discards, BTN's
T3 deal) and varies only the T4 round beneath it. With `J` strata and `L`
sub-draws that is `J × |hero actions|` interior solves instead of
`J × L × |hero actions|` — an 8× saving at J=L=8 — and every arm reads the same
strata, so the whole comparison runs under common random numbers.

### Arms

| arm | interior node played by |
|---|---|
| `random` | uniform over legal actions — the no-information control |
| `mc4`, `mc32` | the Monte-Carlo referee at 4 and 32 sims |
| `model` | the distilled T3-BTN ranker, `t3_btn_v2.pt` |
| `hu` | the frozen heads-up m7v5 models via `hu_bridge` |
| `exact_floor` | the reference's own solver on a **disjoint seed stream**, same strata |
| `exact_other_draws` | the reference's own solver, same seeds, **different strata** |

The reference is `exact_interior(samples=32)`.

The two floors measure different things and both are needed. `exact_floor` is
how much the interior node's *sampling* alone moves the pick — the bar an arm
has to reach to be called indistinguishable from solving the node.
`exact_other_draws` is the yardstick's own noise: the reference labels come
from 64 sampled continuations, so they are not truth, and no arm can be
meaningfully judged below this.

The control is listed first because this track has twice shipped a measurement
that could not tell a policy that knew nothing from one that knew everything
(a mask applied after the tensors were built; ties resolving toward a corpus
sorted by the answer). `test_a_worse_interior_policy_moves_the_label` pins the
wiring directly: a random interior must move the label values on every root
tested.

## Result 1 — how many samples the reference itself needs

`evaluate_t3`'s cost is linear in how many T4 draws the BTN averages over, and
its *choice* is far more stable than its values, so the reference arm can be
cheap. Measured on 48 hu-played BTN roots against a 512-sample reference:

| samples | 4 | 8 | 16 | **32** | 64 |
|---|---|---|---|---|---|
| normalised selection regret | 0.126 | 0.068 | 0.035 | **0.0104** | 0.0115 |
| agrees with the 512-sample pick | 44% | 58% | 71% | 81% | 83% |

The curve falls tenfold to 32 and then flattens — 64 is no better than 32, which
is the yardstick's own noise showing through. The reference arm uses 32.

This corrects a note carried in memory that 64 samples had *zero* selection
regret against 512. That was measured on the old Monte-Carlo-played root
distribution, whose within-root EV spread is 2.4× narrower and whose decisions
are correspondingly easier; it does not hold on the hu-played roots the corpus
is actually built from.

Report: `D:/ofc_data/three_max_interior/samples_ladder.json`.

## Result 2 — the interior reply is reactive to the hero's move

Before the main run, a cheap check on whether the interior solve could be
hoisted out of the hero loop — the 17× saving that would make an exact interior
affordable at corpus scale.

It cannot. Over 18 strata, the BTN's exact reply is the same for every hero
action in only 33% of them (mean 1.83 distinct replies). The BTN maximises its
own total and the hero's finished board is half of what it is maximising
against, so the reply is a reply.

The distilled model is markedly **less** reactive: 61% constant, mean 1.44
distinct replies. That is a concrete, mechanical difference between the two
interior policies, and the first place to look if the arms diverge.

## Result 3 — the interior policy does not matter. The draw count does.

210 hu-played roots, 8×8 = 64 draws, mean fan width 17.7, mean within-root EV
spread 22.82. Regret is what the arm's pick gives up under the reference's own
labels; `vs floor` is the paired per-root difference against `exact_floor`.

| arm | regret | normalised | vs floor (paired) | σ | agrees | label MAE |
|---|---|---|---|---|---|---|
| `random` | 0.1071 | 0.00469 | +0.0007 ± 0.0238 | +0.03 | 83.8% | 5.483 |
| `mc4` | 0.1066 | 0.00467 | +0.0003 ± 0.0261 | +0.01 | 82.9% | 1.534 |
| `mc32` | 0.1309 | 0.00574 | +0.0246 ± 0.0279 | +0.88 | 82.4% | 0.864 |
| `model` | 0.0815 | 0.00357 | −0.0248 ± 0.0200 | −1.24 | 84.8% | 0.438 |
| `hu` | 0.1186 | 0.00520 | +0.0122 ± 0.0276 | +0.44 | 82.9% | 0.469 |
| **`exact_floor`** | **0.1064** | **0.00466** | — | — | **82.9%** | 0.375 |
| **`exact_other_draws`** | **0.6393** | **0.02802** | **+0.5329 ± 0.1246** | **+4.3** | **65.2%** | 3.347 |

**Every interior policy is indistinguishable from the solver, including the one
that plays at random.** The measurement resolves paired differences of 0.048
raw / 0.0021 normalised at 2σ, and no policy arm is within a mile of that.

**Changing which 64 continuations are sampled — with the interior node solved
identically — costs six times the floor, at 4.3σ.** That is the same
measurement, the same code, the same roots, the same reference; only the draw
set moved.

### The mechanism, and why this is not a broken harness

The `label MAE` column orders the arms exactly as one would expect and spans a
factor of fifteen: random 5.48, mc4 1.53, mc32 0.86, hu 0.47, model 0.44, the
solver's own reseed 0.38. On a within-root spread of 22.82, a random interior
opponent moves the hero's action values by an average of 5.5 points. The
interior policy is emphatically wired in and emphatically changes the numbers.

It just moves them *together*. The BTN picks its reply to maximise its own
total, and how much that reply hurts the hero is largely common across the
hero's own candidate placements — so it shifts the whole column and leaves the
ranking where it was. Result 2 rules out the trivial version of this (the reply
is not constant; it changes with the hero's move on two thirds of strata), so
the cancellation is in the values, not in the decision.

`test_a_worse_interior_policy_moves_the_label` pins the wiring independently: a
random interior must move the label values on every root it is given.

### What this settles

The previous session's recommendation was to build the T2 teacher on an exact
T3 and see how much interior accuracy is required. Measured one rung lower, the
requirement is: **none that the current model does not already exceed by a
wide margin.** The T3-BTN model's normalised regret of 0.019 against a floor of
0.008 — the residual four hypotheses failed to explain — does not propagate into
the street above it. Neither would a residual ten times larger.

Two consequences:

1. **Stop improving T3-BTN.** It was already the recommendation on the grounds
   that the exact solver is cheap; it is now also the recommendation on the
   grounds that the improvement would buy nothing measurable upstream.
2. **A T3-BB corpus should spend its budget on roots and draws, not on the
   interior node.** `model` is the sensible default anyway — it costs
   microseconds and is the policy the shipped agent will actually play, which
   is what the self-consistent-chain solution concept asks for — but the choice
   is now known not to be load-bearing.

### The Fantasyland split — the null is not two effects cancelling

An aggregate null can hide a subgroup, and the previous generation flagged one:
FL-live roots were where the T3-BTN model lost, on a control bucket of 28 roots
too small to carry weight. A root is FL-live here when at least one legal move
finishes the hero's top row at QQ+. Re-run on the same 210 roots (the aggregate
numbers reproduce to the digit, which is the CRN working):

| bucket | roots | spread | `exact_floor` | `model` vs floor | `random` vs floor | `exact_other_draws` vs floor |
|---|---|---|---|---|---|---|
| FL live | 101 | 27.27 | 0.0357 | +0.0148 ± 0.0212 (+0.7σ) | +0.0564 ± 0.0338 (+1.7σ) | **+0.9463 ± 0.2417 (+3.9σ)** |
| no FL | 109 | 18.69 | 0.1718 | −0.0616 ± 0.0328 (−1.9σ) | −0.0508 ± 0.0329 (−1.5σ) | **+0.1499 ± 0.0700 (+2.1σ)** |

The FL-live bucket orders the policy arms the way a real effect would — random
worse than model, both worse than the solver — but at 1.7σ and 0.7σ. The no-FL
bucket produces an effect of the same size with the **opposite** sign: a random
interior beating the exact solver, which has no mechanism and is therefore the
noise scale, ~0.05 raw per bucket. Across six policy-arm comparisons the
largest is 1.9σ, and it is in the favourable direction. There is no subgroup
effect here; the aggregate null is a null.

What the split *does* find is where the draw-count lever lives. On FL-live
roots, changing which continuations are sampled costs **27× the floor**
(0.982 against 0.036); on no-FL roots only 1.9× (0.322 against 0.172).

That has a clean mechanism, and it is the same one that explains the whole
probe. Fantasyland entry is a large discrete jump in value that depends on
whether the hero's own top row completes at QQ+ — decided by the cards that
come, and not by anything an opponent places. So on exactly the roots where the
most value is at stake, the interior opponent is the one thing that *cannot*
move it, and the draw is the only thing that can.

Practical consequence: draws should not be spent uniformly. An FL-live root is
worth several times the continuations of a quiet one.

### Result 4 — the distilled model is not the cheap option

The plan assumed the interior node would be filled by a model *because* a model
is cheap and a solver is not. Measured, at 8×8 draws, on the same roots:

| interior policy | s/root label | 20k-root corpus, core-h (label + holdout + hu root) |
|---|---|---|
| `random` | 0.88 | 41 |
| `mc4` | 1.23 | 45 |
| `mc32` | 3.41 | 69 |
| `model` (`t3_btn_v2`) | 20.20 | **256** |
| `exact` (samples=32) | 30.65 | 372 |

**The distilled model costs two thirds of what the exact solver costs.** Its
forward pass is nothing; `features.encode` is everything — 4.6 ms per candidate
board times the BTN's ~17 candidates is 78 ms per interior decision, against the
solver's 157 ms. Distilling a rung does not make the rung above it cheap to
label. It makes it 6× dearer than the Monte-Carlo referee, for labels the probe
cannot tell apart.

With a cheap interior the corpus cost is dominated by something else entirely:
hu root generation at 5.6 s/root is 31 of those 41 core-hours.

### The larger consequence: the cascade may not need to be a cascade

The bottom-up order — T4 exact, then T3-BTN exact, then **distil T3-BTN**, then
T3-BB, then distil, then T2 — is a strict sequence for one reason: each rung's
teacher needs an interior policy, and the plan was for the rung below to supply
it. If any interior policy will do, that dependency is gone. `mc4` matched the
solver to +0.0003 ± 0.0261 and costs nothing; it needs no model, no corpus, and
no training run.

So the rungs could be labelled independently and in parallel, each with an
exact continuation below its interior nodes and something cheap standing in
them. The distilled models would then be what they are actually for — the
shipped policy for a (street, seat) — rather than load-bearing infrastructure
for the rung above.

This is a strong suggestion, not a result. It rests on one interior node
generalising to three nested ones, which is exactly the compounding question
below. It should be tested before the roadmap is rewritten around it.

### What it does not settle

- **Compounding.** (T3, BB) has one interior node. T2 has three, nested. This
  says one substitution is free; it does not say three are. The same
  measurement at (T3, SB) — two interior nodes, one inside the other — is the
  natural next check, but its reference needs an exact BB inside an exact SB,
  which is roughly an hour a root and so is affordable only at n in the tens.
- **The label floor.** `exact_other_draws` says a 64-draw BB label is worth
  about 0.028 normalised — three and a half times the T3-BTN corpus's label
  floor of 0.008. Raising the draw count is the lever this probe actually
  found, and it has not been costed.

### Files

- Probe: `scripts/probe_three_max_interior.py` (`samples` and `arms` modes)
- Evaluator: `src/ofc_regular/three_max/interior.py`
- Reports: `D:/ofc_data/three_max_interior/{samples_ladder,arms_hu_210}.json`
