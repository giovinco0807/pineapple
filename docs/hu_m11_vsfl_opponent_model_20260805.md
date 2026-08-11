# M11 vs-Fantasyland: the opponent model, and what it changed

Status: design note, 2026-08-05. Written alongside the `fl_solver_regular`
crate (`rust/fl_solver_regular`), which implements everything described here.

## The two Fantasyland situations, and why they need different machinery

There are two distinct games that both involve Fantasyland, and they were
briefly conflated. They are not the same and they do not share an opponent
model.

### Normal hero vs a Fantasyland opponent — the opponent is ADAPTIVE

The Fantasyland player receives fourteen cards at once and sets thirteen of
them face down, but they do not have to commit before the hand develops. They
watch the normal player's progressive placement and choose their 3/5/5 only
after that board is finished. The Fantasyland side is therefore a **best
response** to a concrete hero board, not a draw from a fixed distribution.

This is the rule the game runs, and it is what `opponent_mode: adaptive_v1`
implements.

### Both players in Fantasyland — the opponent is SIMULTANEOUS-BLIND

Owner ruling, 2026-08-05: when both seats are in Fantasyland, both set their
boards simultaneously and neither sees the other. There is nothing to react to.

**Consequently the static `pure_v1` solve IS the correct FL-vs-FL machinery.**
`FlSolver::solve` with `ObjectiveKind::PureV1` maximises
`royalty + fl_ev * stay`, which is exactly the right objective for a player who
cannot condition on the opponent. No new work, no adaptive frontier, no
separate mode. The existing solver is the FL-vs-FL solver.

## What the adaptive correction actually cost

Measured on 300 T3-vs-FL roots, matched pairs — same roots, same opponent
hands, same exhaustive hero-deal enumeration, only the opponent's ability to
react differs:

| quantity | value |
| --- | --- |
| mean label EV shift, all candidates | **-0.420** (never positive) |
| best-candidate EV shift | mean -0.457, median -0.362 |
| argmax changed | **8.33 %** (25/300) |
| candidate rank moves | 19.1 % move, max 14 places |

Split by whether a QQ+ Fantasyland entry is still reachable for the hero:

| | roots | argmax changed | mean best-EV shift |
| --- | --- | --- | --- |
| FL entry live (top row open) | 276 | **9.06 %** | -0.481 |
| FL entry dead (top row full) | 24 | **0.00 %** | -0.184 |

Every single argmax change sits in a root where the hero's top row is still
open. Static labelling was optimistic by roughly 0.42-0.46 points per label,
uniformly downward, and concentrated exactly where the theory said it would be.

## The frontier: why best-responding is affordable and exact

Rescanning all 1,009,008 arrangements per hero board is far too slow, and
unnecessary. Write the Fantasyland side's score against a fixed hero board `H`:

```
f(A, H) = g(s_top, s_mid, s_bot) + static(A)
  s_row     = sign(row_key(A) - row_key(H))  in {-1, 0, +1}
  g(s)      = sum(s), plus 3*sign(sum(s)) when |sum(s)| == 3     <- scoop bonus
  static(A) = royalty(A) + fl_ev * stay(A)
```

`g` is monotone non-decreasing in each argument — verified over all 27 sign
triples by `frontier::tests::scoop_aware_line_is_monotone`, not asserted — and
each `s_row` is monotone in `row_key(A)`. So `f(., H)` is monotone in all four
of `(top_key, mid_key, bot_key, static)` **simultaneously for every `H`**.

Therefore a dominated arrangement can never be the unique best response, and
the maximum is always attained on the non-dominated frontier. Scanning the
frontier is **exact, not approximate**. `tests/frontier_exactness.rs` pins this
against brute force on 320 random (deal, hero board) pairs plus 24 fouled-hero
pairs, requiring bit-identical scores.

Note the scoop bonus is *inside* `g`. That is not incidental: the monotonicity
of the scoop term is precisely what licenses the reduction, so the maximisation
is scoop-aware by construction.

### Measured behaviour (2,000 deals x 8 hero boards)

| | mean | p50 | p95 | max |
| --- | --- | --- | --- | --- |
| frontier size | 59.2 | 39 | 184 | 655 |
| build time | 46.6 ms | 45.5 | 57.7 | 80.9 |
| scan per terminal | 539 ns | | | |

**Static-value deficit of the best response** — how much royalty+stay the
line-adaptive winner gives up:

| mean | p95 | max | share > 5 | share = 0 |
| --- | --- | --- | --- | --- |
| 0.141 | 1.0 | **4.0** | **0.0 %** | 94.2 % |

The working heuristic that the answer never sits more than ~5 points below the
static optimum holds: empirical max 4.0 against a theory bound of 12, and 94.2 %
of the time the best response *is* the static optimum. A serving implementation
that keeps the frontier only a few points deep is safe.

### For the serving flow

Precompute-at-deal is sound, but budget **46.6 ms mean (p95 58 ms, max 81 ms)**
per opponent hand for the frontier build, not the ~11 ms once assumed. That is
still comfortably hidden inside the opponent's placement time. The answer at
showdown is a **539 ns** scan, i.e. instant.

## Consequence for `fl_ev`

`estimate_hu_fl_ev_direct.run_iteration` solves the Fantasyland board at line
215 with `solve_fantasyland(hero_cards, stay_bonus=current_ev)` — **before**
line 220 creates the opponent board — and `solve_fantasyland` takes no opponent
argument at all. The estimator's Fantasyland side is therefore **static**.

An adaptive Fantasyland player scores at least as well as a static one against
the same board, and the measurement above puts that edge at **+0.42 to +0.46
points per hand** from the Fantasyland side. So `fl_ev = 9.109` is a **floor**,
not a point estimate, and independent higher anchors are consistent with it
rather than in conflict. No config was changed on the strength of this note;
the remeasure is its own piece of work.

## Labels are not interchangeable across modes

`fl_solver_regular/0.1.x` produced static-opponent labels; `0.2.0` added the
adaptive opponent. Plans pin `opponent_mode` (`static_v1` or `adaptive_v1`) and
the worker's provenance gate refuses to write when the solver reports a
different mode than the plan pinned. A label built one way is a different
quantity from a label built the other way, and mixing them silently is exactly
the failure the gate exists to prevent.

## The fallback encoding cannot carry a vs-Fantasyland model

Both distillation arms were trained on `fallback_flat_v1`, a flat multi-hot
card encoding, because the engine-family fast features refuse every vs-FL
geometry at two independent gates: `t3first_features.rs:674` rejects a 2-open
-slot outlook, and the `ActorObservation` validator rejects a hidden opponent
(`expected (5,5,3,0), got (5,0,3,0)` at T1, and the same shape at T2 and T3).
These arms therefore measure the encoding, not the design.

Acceptance for vs-FL models is teacher-regret based, not match-play based. The
bar is the learned chain's own regret against the same exact teacher on the
same roots: a model is worth serving only if it leaves less EV on the table
than the chain already does.

| Street | Chain regret (bar) | Fallback model | Ratio | Chain argmax | Model top-1 |
|--------|-------------------:|---------------:|------:|-------------:|------------:|
| T3-vsFL (20k adaptive labels) | 0.485 | 1.595 | 3.3x | 60.0% | - |
| T2-vsFL (50k adaptive labels) | 0.335 | 0.662 | 2.0x | 62.8% | 58.2% |

The T2 comparison is also available contamination-free: the 600 roots the bar
was measured on were pinned out of training and scored with the model the
holdout selected. Paired on those identical unseen roots, the model's regret is
**0.774 against the chain's 0.335** — 2.3x worse, and it picks the teacher's
argmax less often than the chain does (57.0% vs 62.8%).

The scaling curve settles whether this is data starvation. Over train sizes
3,125 / 6,250 / 12,500 / 25,000 / 45,000 the fit is

    regret = 1.634 * n^-0.0823   (R^2 = 0.92)

which reaches the bar at **237 million roots**, about 5,270x the 50,000-root
batch a full fleet run produced. The T3 arm's exponent was -0.148; T2's is
roughly half that, so the fallback encoding scales *worse* at the street with
more data and more open geometry. Two arms, two streets, both far above their
bar, with a projection that is not a scheduling question but an impossibility.
The conclusion is about the feature space, not the sample count.

Neither fallback model is deployable, and neither should be served. The exact
solve remains the fallback for both streets. The real-feature reruns wait on an
observation variant that admits a hidden opponent and an outlook widened to two
open slots; when those land, the standalone `vsfl_encoder` consumes them and
both distillations rerun with fast-family warm starts.

## With the engine features, both vs-Fantasyland models clear the bar

The gates are open: hidden-opponent observations, a two-open-slot outlook, and
vs-FL fast-feature arms for (T1,T2,T3)/First. Re-verified here from a rebuild
against current source -- 3 of 3 streets accepted, observation valid, JSON round
trip identical, every candidate encoding, hero block carrying >=51 of 122
nonzero values, opponent tail exactly 0.

The standalone `vsfl_encoder` emits the 168-wide engine row per candidate,
reconstructing each candidate board from the label rather than re-deriving it
from the action fan, so a feature row cannot silently pair with another
candidate's expected value. Extraction enforces the contract over the whole
corpus, not one hand: hero-block nonzero minimum 43, tail max |v| exactly 0
across 1,200,699 T2 rows and 480,000 T3 rows.

Everything about the training protocol is held fixed against the fallback arms
-- split seed, holdout fraction, arm seeds, tie tolerance, loss, early stopping,
paired-root pinning -- so the encoding is the only thing that moved.

| Street | Chain regret (bar) | Fallback | Real features | vs bar | vs fallback |
|--------|-------------------:|---------:|--------------:|-------:|------------:|
| T3-vsFL (20k) | 0.485 | 1.595 | **0.096** | 5.0x better | 16.6x better |
| T2-vsFL (50k) | 0.335 | 0.662 | **0.155** | 2.2x better | 4.3x better |

Paired on the identical unseen roots the bars were measured on, with those
roots pinned out of training: T2 scores **0.156 against 0.335** and picks the
teacher's argmax 73.3% of the time against the chain's 62.8%; T3 scores
**0.104 against 0.485** at 78.3% argmax. Both are deployable, and the ordering
the fallback arms implied is reversed -- T3, the street the fallback did worst
on, is now the better model, which is what a feature space that carries row
interaction should do at the street where the board is nearly complete.

The arms are stable rather than lucky: all four T2 arms land between 0.155 and
0.162, all four T3 arms between 0.096 and 0.107. Warm-starting from
`fast_t2_first_v1.bin` wins narrowly at both streets (T2 0.155 vs 0.157 scratch;
T3 0.096 vs 0.098) -- worth taking, not worth depending on.

Two mechanical findings from the port, both of which produced a
confidently-wrong result before being caught:

  * The extractor pads unused candidate slots with 0.0, and every vs-FL expected
    value is negative, so an unmasked `max` returns the padding and concludes no
    real candidate is optimal. The soft target summed to zero, the loss was NaN
    on the first step, and the two arms agreed to sixteen digits because both had
    become NaN. The fallback trainer had padded with -1e9 and was never affected,
    so the earlier fallback numbers stand. Values are now masked before every
    comparison, which is correct for any padding convention.
  * The pretrained standardization comes from normal-vs-normal data where some
    dims are near-constant. Feature 81 is a constant 6 there and a constant 13
    in vs-FL, which standardizes to 7000 and saturates the network before a
    single step; dims 99-112 are one-hots that are near-absent normally and
    common vs FL. Standardized inputs are clamped to +/-8 -- the range the
    pretrained function was actually fitted on. This touches 0.9% of hero cells,
    0.098% outside three dims. The zeroed tail is separately neutralized to its
    pretrained mean, which is ordinary mean imputation for features this
    situation does not have.

Two debts carried forward from the gate verification, neither blocking here:
`belief.rs`'s `opponent_discard_count` returns 0 for hidden opponents, which
bites only if vs-FL ever routes through engine search (these teachers do not);
and the hero outlook's unknown set still includes the opponent's 14 unavailable
cards, uniform across the fan and so second-order for ranking -- distinct from
the zeroed-tail debt.

## The T1 teacher: continuation models, and what they cost

T1 is continued rather than searched. Its candidates are played out by the
trained T2-vs-FL and T3-vs-FL models (the warm winners), T4 is exhaustive, and
the terminal is the adaptive Fantasyland frontier under common random numbers.
This is the standard bottom-up compromise, and what makes it defensible here is
that the continuation quality is measured rather than assumed: 0.155 and 0.096
mean regret against the exact teacher, both well inside the chain they replace.

### The models had to be re-exported, not reused

The trained rankers are PyTorch checkpoints, and the engine's `T4M1` image
cannot hold them: it stores a standard deviation and refuses zeros, and it has
no field for an input clamp. The winning arms need both -- the opponent tail's
inverse std is forced to exactly zero (imputing the tail to its pretrained
mean), and standardized inputs are clamped to +/-8. Writing them as `T4M1`
would silently drop both and ship a model that is not the model that was
measured.

`VFL1` is that superset: inverse std stored directly, clamp recorded in the
header. `fl_solver_regular::vfl_model` reads it and runs the forward pass with a
single fixed-order accumulator -- no split lanes, no rayon inside a prediction
-- so a score is bit-reproducible from a pinned digest. Ties in `argmax` go to
the lower index, so a label never depends on iteration order upstream.

The reader is held to the Python model rather than trusted: the exporter writes
a fixture of real feature rows with the scores the training-time module
produced, and `fl_solver_regular vfl-parity` reloads the pinned image and
compares. Max absolute delta **4.77e-6** for T2 and **1.91e-6** for T3 over 256
rows each, which is float32 rounding.

| Model | sha256 (first 16) | Source arm | Regret |
|-------|-------------------|------------|-------:|
| `vsfl_t2_v1.vfl1` | `f9c59ad4027d8380` | warm, seed 997980001 | 0.15499 |
| `vsfl_t3_v1.vfl1` | `a1a2828affd16867` | warm, seed 997980001 | 0.09620 |

### Cost, from measured components

Per T1 root, with ~27 candidates:

    encodes = 27 * t2_draws * (24 + t3_draws * 21)

at a measured **~22 us per encode-and-score per core** (480,000 T3 rows in 2.6s
on four cores). The Fantasyland frontier costs 46.6ms per opponent sample and is
built once per root and shared by everything below it, so at N=200 it is
**9.3 core-seconds per root regardless of the draw grid**.

| t2_draws | t3_draws | encode core-s | + frontier | total core-s/root |
|---------:|---------:|--------------:|-----------:|------------------:|
| 8 | 4 | 0.51 | 9.3 | 9.8 |
| 16 | 8 | 1.82 | 9.3 | 11.1 |
| 32 | 16 | 6.84 | 9.3 | 16.1 |

The shape of that table is the whole point: the shared frontier dominates, so
model-guided T1 costs only 5-70% more per root than T2 did, not the three
orders of magnitude an exhaustive T1 continuation would have. At 20,000 roots
the middle row is ~62 core-hours, about 1.1 hours per shard across 58 shards.

The noise axis cannot be filled in from components -- it needs the teacher to
exist and be run against a high-draw reference, exactly as the T2 grid was
done.

### One architectural decision to make first

`fl_solver_regular` keeps `ofc_hu_m3_engine` as a **dev-dependency only**, and
its manifest says why: so that "nothing in this crate can perturb the engine's
build plan", and so the two never contend over a target directory. Computing
engine fast features inside the T1 labeller makes the engine a runtime
dependency and links it into the binary the fleet ships. That is a real change
to a pinned artifact and to a deliberate design choice, so it is named here
rather than made quietly.

## A yardstick must not share the ruler's draws

The first T1 grid said the `32x16x8` rung reproduced a `32x16x16` reference to
within 0.0005 mean regret. That number was an artifact, and a large one.

Deals are addressed by `(seed_base, root, t2_index, t3_index, t4_index)`. A rung
that shares the reference's hero-deal seed therefore replays a **prefix** of the
reference's own draws: `32x16x8` against `32x16x16` is the same estimator, on
the same 32 T2 deals and the same 16 T3 deals, differing only in taking the
first 8 of 16 T4 deals. It was being compared against itself.

Common random numbers belong *between candidates*, where shared deals cancel
the noise that would otherwise swamp the difference the label is trying to
measure. Between an estimator and its yardstick they manufacture agreement. The
probe now takes `--reference-hero-deal-seed` and reports
`independent_of_rungs`, and re-running with an independent reference moved that
rung from 0.0005 to **0.0304 -- sixty times larger**. Every rung's regret in the
first table was a lower bound.

| grid | argmax agreement | mean regret vs reference | label SE | core-s/root |
|------|-----------------:|-------------------------:|---------:|------------:|
| 8x4x4 | 0.650 | 0.3924 | 0.992 | 7.5 |
| 8x8x8 | 0.750 | 0.1033 | 0.816 | 8.2 |
| 16x8x8 | 0.825 | 0.0624 | 0.615 | 10.2 |
| 16x16x8 | 0.775 | 0.0664 | 0.561 | 13.0 |
| 32x16x8 | 0.900 | 0.0304 | 0.400 | 19.5 |

(40 roots, N=200, reference 32x16x16 at N=400 on its own seed.)

The measured cost also came in above the component estimate -- 10.2 core-s/root
at 16x8x8 against a predicted 11.1 for the whole config, with the heavier rungs
running 1.5-2x the projection. The shared-frontier term was right; the
per-encode term was optimistic, because a rollout builds a fresh
`FastOutlookCache` at every continuation node rather than amortizing one across
a position's whole fan.

### The grid does not converge, so it cannot choose

Under the selection rule -- take the config whose noise metrics have plateaued,
then one rung heavier -- this table selects nothing. The label standard error is
still falling steeply at the top rung (0.561 to 0.400) and argmax agreement is
still climbing (0.775 to 0.900). The plateau is above the grid, so the shipped
config has to be heavier than anything measured here, and the earlier
cost-based recommendation of 16x8x8 is withdrawn. The grid is being re-run over
the rungs above, at N=400, against a 64x32x32 reference on its own seed.

Convergence in the opponent-sample axis `N` is a separate question from
convergence in the draw axes, and the draw-axis probe does not answer it.

## The label SE has no plateau to find, so regret has to choose

Extending the grid upward, at N=400 against a `64x32x32` reference on its own
seed (32 roots):

| grid | argmax agreement | mean regret | max regret | label SE | core-s/root |
|------|-----------------:|------------:|-----------:|---------:|------------:|
| 16x8x8 | 0.656 | 0.0646 | 0.439 | 0.578 | 28.1 |
| 32x16x8 | 0.812 | 0.0226 | 0.439 | 0.387 | 43.0 |
| 32x16x16 | 0.781 | 0.0256 | 0.439 | 0.378 | 59.5 |
| 64x32x16 | 0.688 | 0.0144 | 0.094 | 0.257 | 144.7 |

### Why "wait for the SE plateau" cannot terminate here

The label's standard error is the Monte Carlo error of the OUTER expectation
over T2 draws, so it decays as `1/sqrt(t2_draws)` and never flattens. The data
say exactly that, three times over: doubling T2 from 16 to 32 moves SE by
0.578/0.387 = 1.49, from 32 to 64 by 0.387/0.257 = 1.51, against a predicted
sqrt(2) = 1.414. Doubling only T4 (32x16x8 to 32x16x16) moves it by 2%, because
T4 sits inside two averages and barely reaches the outer variance.

An SE plateau is therefore not a reachable criterion on the axis that dominates
it. What looks like a plateau in a narrow grid is just a rung where only the
inner draws moved. The usable criterion is the other one: regret against a
deeper reference, falling into that reference's own noise.

### Argmax agreement is the wrong metric, and inverts here

The heaviest rung has the LOWEST argmax agreement (0.688) and the LOWEST regret
(0.0144). That is not a contradiction. Agreement counts disagreements; regret
prices them. At 64x32x16 the maximum regret over 32 roots collapses from 0.439
to 0.094 -- it still disagrees with the reference, but only ever on decisions
that are near-ties, where picking either costs the student almost nothing. The
lighter rungs agree more often and are occasionally expensively wrong, which is
the failure mode that actually damages a distilled policy.

Read on regret, the sequence is 0.065, 0.023, 0.026, 0.014, with the expensive
tail gone by the last rung.

### Shipped config

Selection rule: the last rung whose regret has settled, then one rung heavier.
Regret settles at **64x32x16**, so the shipped T1 config is **64x32x32 at
N=400** -- the reference config itself, at 111.3 core-s/root measured. For
20,000 roots that is about 618 core-hours, which the fleet absorbs; per the
standing directive, precision is not traded for core-seconds here.

Because the shipped config IS the old reference, QC can no longer measure it
against that: a deeper still reference is required, and it must be deeper in the
DRAW axes rather than only in N, since N is not what dominates the SE. The QC
reference is therefore `128x64x32` at N=800 on a small root sample, not
`64x32x32` at N=800.

## Opponent samples are not the sample size: collision survival by street

The opponent's fourteen Fantasyland cards and the hero's future draws come out
of the same deck, so a sampled opponent is only usable at a terminal if its
fourteen miss every card the hero drew on the way there. That is the correct
conditional and the estimator is unbiased, but it means the sample behind a
terminal is not `N`, it is `N x P(disjoint)` -- and `P` collapses as the street
gets earlier, because an earlier street has more future draws to avoid.

With the hero holding `5 + 3*(street index)` cards and drawing three per
remaining street, `P = C(unseen - 3k, 14) / C(unseen, 14)`:

| street | unseen | hero future draw cards | survival | effective N at 400 | at 6,400 | at 25,600 |
|--------|-------:|-----------------------:|---------:|-------------------:|---------:|----------:|
| T3 | 38 | 3 | 23.99% | 96.0 | 1,535 | 6,142 |
| T2 | 41 | 6 | 6.58% | 26.3 | 421 | 1,685 |
| T1 | 44 | 9 | **2.02%** | **8.1** | 129 | 517 |
| T0 | 47 | 12 | **0.68%** | **2.7** | 43 | 174 |

Measured, not just derived: the T1 labeller reports
`mean_usable_opponents = 8.07` against `samples = 400`, which is 2.02%.

Three consequences.

**The draw grid could not see this.** Every rung of the T1 grid held N at 200 or
400, so the whole grid sat at an effective 4 to 8 opponents per terminal. The
`1/sqrt(t2_draws)` law describes the OUTER expectation over T2 deals; terminal
noise is governed by effective N, which is a separate axis and was never varied.

**A yardstick must clear the rungs on the axis being probed.** The first QC
reference took N=800 -- an effective 16, below two of the three N rungs it was
meant to judge. Deeper draws do not fix that; only deeper N does. The N-axis
reference therefore runs N=25,600 (effective ~517, four times the largest rung)
and deliberately keeps the rungs' draw shape, since the draw axis is already
settled and holding it fixed isolates N. Independence comes from the seed.

**T0 must budget N from this table, not rediscover it.** At T0 survival is
0.68%, so the N that buys T1 an effective 129 buys T0 only 43. N is
frontier-linear and the frontier is shared per root, so this is the cheapest
axis in the teacher -- but it has to be chosen up front rather than inherited
from T1's config.

## The N axis is flat, so N is chosen for margin rather than convergence

The N-axis probe ran three rungs at N=400 / 1,600 / 6,400 on the shipped
`64x32x32` grid against a reference at N=25,600 -- an effective ~517, four times
the largest rung -- on its own seed pair, over the same 20 roots.

| rung | N | effective | mean regret | median | max | argmax agreement | mean label SE |
|------|--:|----------:|------------:|-------:|----:|-----------------:|--------------:|
| rung400 | 400 | 8.1 | 0.0350 | 0.0000 | 0.6337 | 0.800 | 0.2455 |
| rung1600 | 1,600 | 32.3 | 0.0350 | 0.0000 | 0.6337 | 0.800 | 0.2353 |
| rung6400 | 6,400 | 129.2 | 0.0350 | 0.0000 | 0.6337 | 0.800 | 0.2305 |
| reference | 25,600 | 516.5 | - | - | - | - | 0.2339 |

Every rung produces the **same decision on all 20 roots**, so the three regret
columns are identical rather than merely close.

That agreement is not an artifact of the rungs sharing a seed base. They do
share one, so their opponent samples are nested and correlated -- but nesting
would make the estimates similar, not the decisions identical, and the estimates
are in fact far apart: mean `|dEV|` between rung400 and rung1600 is **0.3255**,
max **0.8139**. The values move substantially and the argmax never does.

**The label SE is essentially N-independent here**: 0.2455 to 0.2305 across a 16x
increase, a 6% move where `1/sqrt(N)` would predict 4x. The reference at N=25,600
even carries a *higher* mean SE (0.2339) than rung6400, because on this grid the
SE is dominated by the outer expectation over T2 draws and not by N at all. This
is the same result the draw-axis section reached from the other side, and it
means the N axis has no plateau to find for the opposite reason: it is already
flat at the first rung.

So the selection rule that chose the draw grid cannot choose here, and neither
can regret: regret does not fall with N because nothing about the decision
changes with N. **N is therefore chosen for margin, not for convergence** -- an
honest statement of what the data support, rather than a convergence claim the
axis cannot supply.

Where the rungs and the reference disagree, three of the four disagreements are
exact near-ties in which the rung took the reference's second choice, so the
regret equals the reference's own margin:

| root | regret | reference margin | reference SE | reading |
|-----:|-------:|-----------------:|-------------:|---------|
| 1 | 0.0027 | 0.0027 | 0.2326 | coin flip, 100x inside the noise |
| 5 | 0.0261 | 0.0261 | 0.3013 | near-tie |
| 8 | 0.0382 | 0.0382 | 0.7044 | near-tie, reference very noisy here |
| 17 | 0.6337 | 0.1790 | 0.5628 | a real gap, but the reference's own top two are 0.179 apart against an SE of 0.563 |

Root 17 alone contributes 0.0317 of the 0.0350 mean. This is the failure mode
the draw-axis section named -- lighter configurations are occasionally
expensively wrong -- except that here it does not improve with N either, which
places it in the draw axes and the reference's own resolution rather than in the
opponent-sample count.

### Shipped T1 config

**`64x32x32` at N=1,600**, four times the smallest rung tested and comfortably
inside the flat region, at a measured **147.4 core-s/root** (from `/usr/bin/time`
over the 20-root rung, against 76.3 at N=400). For 20,000 roots that is about
819 core-hours, roughly 26 minutes per shard across 58 shards. The margin over
N=400 costs about 395 core-hours -- some thirteen extra minutes of fleet time --
which is the cheap side of the standing directive that precision is not traded
for core-seconds.

A caution for anyone reading a cost from a rehearsal rather than from the grid:
a one-shard rehearsal at `--solver-chunk 1` pays the solver's start-up per root
and reads about ten times the production cost. The fleet runs `--solver-chunk 32`
and amortises it.

### The shipped T1 run

| item | value |
|------|-------|
| package | `~/ofc-labelgen-vsfl-t1/package` (runtime `c4d97f3acbb46ed9...`, 70.1 MB) |
| plan | `worker_plan_t1vsfl_r1.json`, sha256 `00ea585941c96f3c...` |
| solver | `fl_solver_regular/0.3.0`, binary `45c6113cc51313a1...` |
| continuation | `vsfl_t2_v1.vfl1` `f9c59ad4027d8380...`, `vsfl_t3_v1.vfl1` `a1a2828affd16867...` |
| engine features rev | `4e1284dc74f7d98f...` |
| roots | `t1roots20k.jsonl` `cea10523b8b2cdff...`, 20,000 |
| shards | 58 (`00`..`57`), 344 x 10 + 345 x 48 |
| seed bases | opponent 997,320,000 / hero deal 997,370,000 |

The seed bases are new. They must not be the rungs' (997,620,000 / 997,720,000)
or the reference's (997,690,000 / 997,760,000): a production run that replayed
the draws of the probe that sized it would inherit the probe's noise instead of
averaging independently over its own.

### The T1 model clears its bar

20,000 adaptive labels, extracted through the standalone `vsfl_encoder` T1 arm
rebuilt against current engine source. The corpus contract holds over every row,
not a sample: hero-block nonzero minimum **41** of 122, opponent tail max |v|
**exactly 0** across all 20,000 positions x up to 27 candidates.

The bar is the learned chain's own T1-vs-FL regret against this same teacher, on
600 roots that are pinned OUT of training, so the comparison is paired on
identical unseen roots rather than merely comparable in aggregate.

| quantity | value |
| --- | --- |
| chain bar (600 roots) | **0.1750**, chain picks the teacher's argmax 67.0% |
| model, 2,551-root holdout | **0.0944** regret, top-1 73.5%, top-3 96.5% |
| model, paired on the 600 bar roots | **0.0926** against the chain's 0.1750 -- **1.89x better**, top-1 73.3% vs 67.0% |
| teacher's own noise floor | **0.0350** (the N=1,600 rung's regret against the N=25,600 reference) |

The model sits 2.6x above the teacher it is distilling, which is the honest
ceiling statement: the remaining gap is partly the model and partly that the
teacher itself is 0.0350 from a sixteen-times deeper reference.

The arms are stable rather than lucky -- all four land between **0.0944 and
0.0988** -- and warm-starting from `fast_t1_first_v1.bin` wins narrowly over
scratch (0.0944 vs 0.0975), the same margin and the same conclusion as at T2 and
T3: worth taking, not worth depending on.

| Model | sha256 (first 16) | Source arm | Regret | parity max abs delta |
|-------|-------------------|------------|-------:|---------------------:|
| `vsfl_t1_v1.vfl1` | `2d02958d27f289d4` | warm, seed 997980001 | 0.09442 | 3.81e-6 |

### The bar falls as the street gets earlier, and so does the headroom

| Street | chain bar | model regret | model / bar |
|--------|----------:|-------------:|------------:|
| T3-vsFL | 0.485 | 0.096 | 5.0x better |
| T2-vsFL | 0.335 | 0.155 | 2.2x better |
| T1-vsFL | 0.175 | 0.094 | 1.9x better |

The chain leaves less on the table the earlier the street, which is what a
street with more hand remaining should do: a single early placement is more
forgiving because later streets can still repair it. The model's *advantage*
shrinks with it. That is a sizing signal for T0 rather than a disappointment --
the earlier the street, the less a distilled ranker can win, and T0 is one
street earlier still.

Two things were re-established rather than inherited before the plan was cut.
`main.rs` was rebuilt at 06:03, between rung1600 and rung6400, which would have
spanned the comparison across two binaries; re-running rung400's root 0 under the
current binary reproduces the label **bit-identically** in all 24 candidate EVs
and standard errors, so the rebuild was label-neutral. That check also shows the
labeller is deterministic across thread count (3 vs 12) and roots-file
composition (a 1-root file vs a 10-root chunk). Separately, the 997,400,000
sub-block is **not** free -- it is the root-generation block -- which a seed audit
across this note, the shipped plans and every driver script caught before it was
spent twice.

## T0-vs-Fantasyland is not the same plumbing, and not the same price

The T1 stage was plan generation against machinery that already existed. T0 is
not. The crate carries `teacher.rs` (T4), `t2_teacher.rs` and `t1_teacher.rs`;
there is **no `t0_teacher.rs`**, `label-t0` is not a subcommand, and the worker's
vs-FL kinds are exactly `t3_vs_fl`, `t2_vs_fl`, `t1_vs_fl`. Nothing rejects a T0
plan with a good error message because nothing accepts one.

What a T0 stage needs before any plan can be written:

  * a `t0_teacher.rs` with a FOURTH nested draw level, and a `Continuation`
    carrying THREE model digests rather than two -- `t1_teacher.rs` hardcodes
    both the three-level nesting and the two-model struct;
  * a `label-t0` subcommand wiring three VFL1 images and four draw counts;
  * a `t0_vs_fl` kind in the worker, with the all-or-nothing field set and the
    provenance gate extended to a third digest;
  * `SOLVER_VERSION` to 0.4.0, because the set of label kinds changed;
  * and a sizing study for `t1_draws`, which is a genuinely new axis. The T1
    grid study is the precedent for what sizing one costs.

### The cost model was a floor, and a loose one

The earlier sizing note put T0 near 1,100 core-s/root by treating its tree as
T1's tree plus a little. That is wrong in kind: adding a street **multiplies**
the tree, because every T1 draw spawns a full T1-shaped subtree beneath it. And
the T0 fan is not 27 -- five distinct cards into rows of capacity (3,5,5) is
`3^5 - 10 - 1 = 232` placements, 8.6x the T1 fan.

    encodes(T0) = 232 * t1_draws * (27 + t2_draws * (24 + t3_draws * 21))

Calibrated against the measured T1 rung (the modelled encode term underestimates
the measured tree by **2.75x**, consistent with a rollout rebuilding a fresh
`FastOutlookCache` at every continuation node), at N=22,100 over 18,000 roots:

| t1_draws | t2 x t3 | core-s/root | core-hours |
|---------:|--------:|------------:|-----------:|
| 4 | 16x8 | 1,204 | 6,020 |
| 8 | 32x16 | 2,328 | 11,639 |
| 16 | 32x16 | 3,626 | 18,129 |
| 32 | 64x32 | 21,070 | 105,352 |
| 64 | 64x32 | 41,111 | 205,554 |

The entire T1 fleet run was **819 core-hours**. Matching T1's draw depth at T0 is
**129x** that. Even the shallowest row is 7x the T1 run, and its draw depth is
below the rung the T1 grid study rejected as too noisy to ship.

These are projections, not measurements -- there is no T0 teacher to time -- so
the first thing a T0 stage should do after the teacher exists is a small probe
that replaces this table with measured numbers.

### The sizing question this raises

Chain regret falls as the street gets earlier (0.485, 0.335, 0.175 at T3, T2,
T1) and the model's advantage falls with it (5.0x, 2.2x, 1.9x). Extrapolating,
a T0 chain bar is likely near 0.10-0.15 with a model advantage under 2x -- the
smallest prize of the four streets, at by far the largest teacher cost. That is
an argument for deciding what T0 is worth *before* building the teacher, rather
than discovering it after.

## M11 summary

### The models

All three are 168-wide engine fast-feature rankers, body 168-128-64-32-1,
standardized inputs clamped to +/-8, opponent tail's inverse std forced to zero
(mean imputation for features this situation does not have). Ties in `argmax` go
to the lower index. Every one is `VFL1`, not `T4M1`, because `T4M1` can store
neither a zero inverse std nor a clamp and would silently ship a different model
from the one measured.

| Street | Warm base | Chain bar | Model regret (holdout) | Paired vs chain, unseen roots | sha256 (first 16) | Parity |
|--------|-----------|----------:|-----------------------:|------------------------------:|-------------------|-------:|
| T1-vsFL | `fast_t1_first_v1` | 0.1750 | **0.0944** | 0.0926 vs 0.1750 (1.89x) | `2d02958d27f289d4` | 3.81e-6 |
| T2-vsFL | `fast_t2_first_v1` | 0.335 | **0.155** | 0.156 vs 0.335 (2.2x) | `f9c59ad4027d8380` | 4.77e-6 |
| T3-vsFL | `fast_t2_first_v1` | 0.485 | **0.096** | 0.104 vs 0.485 (4.7x) | `a1a2828affd16867` | 1.91e-6 |

**Encoding contract**, enforced over every row of every corpus rather than a
sample: 168 features per candidate, hero block `[0,122)` with a nonzero minimum
of 41 (T1), 43 (T2/T3), opponent tail `[122,168)` exactly zero. A feature row is
built by reconstructing the candidate board from the label rather than
re-deriving it from the action fan, so a row cannot silently pair with another
candidate's expected value.

### Solver and frontier

| quantity | value |
| --- | --- |
| Fantasyland frontier build | 46.6 ms mean, p95 57.7, max 80.9 per opponent hand |
| frontier size | 59.2 mean, p50 39, p95 184, max 655 |
| best-response scan at showdown | 539 ns |
| static-value deficit of the best response | 0.141 mean, max 4.0, zero 94.2% of the time |
| adaptive correction vs static labelling | -0.420 mean EV, 8.33% of argmaxes move |
| T1 teacher, shipped config | 64x32x32 at N=1,600, 147.4 core-s/root |
| T1 fleet run | 20,000 roots, 58 shards, 819 core-hours |

### Serving mode, per street

| Situation | Serve | Why |
|-----------|-------|-----|
| T1-vsFL | **model** | no affordable exact alternative; 1.89x inside the chain |
| T2-vsFL | **model** | same, 2.2x inside the chain |
| T3-vsFL | **dual-mode** | the model is 0.096 and instant; the exhaustive solve is exact at ~1 s. Serve the model interactively, the exact solve when a second is affordable |
| T4-vsFL | **exhaustive** | the fan is small enough to enumerate outright |
| FL placement (hero in FL vs normal) | **frontier** | exact, not approximate: the maximum always lies on the non-dominated frontier, pinned against brute force on 344 pairs |
| FL vs FL | **static `pure_v1`** | both seats set blind, so there is nothing to react to and the existing solver already maximises the right objective |

### Integration manifest

Everything the M6 and self-play workstreams consume, by path and sha256. Digests
are of the artifacts as they stand at M11 close; a consumer that pins them will
notice a silent replacement.

**Rankers.** All four are 168-wide, body `[128, 64, 32, 1]`, hero block
`[0,122)`, opponent tail `[122,168)` exactly zero.

| Street | Path | sha256 |
|--------|------|--------|
| T1-vsFL | `~/ofc-vsfl/models/vsfl_t1_v1.vfl1` | `2d02958d27f289d48ed0059c8aafec8afd98720993f20758a4f9ab962eb35588` |
| T2-vsFL | `~/ofc-vsfl/models/vsfl_t2_v1.vfl1` | `f9c59ad4027d8380dbd6c4dc35e127109aea3f166ac669303b0cea490bfdd58b` |
| T3-vsFL | `~/ofc-vsfl/models/vsfl_t3_v1.vfl1` | `a1a2828affd1686757af4666f58fdba87ae651707477e502d4065db4c21abd1d` |
| T0-vsFL (**stand-in**, T4M1) | `~/ofc-t0first/train_early/curve/t0first_model_v1.bin` | `07301034502de626d4deabe95d576dfd72207bd57550ebbae7934ebf0f191e84` |

**Parity fixtures**, and the gate that must stay green before serving:

| Fixture | sha256 |
|---------|--------|
| `vsfl_t1_v1.parity.json` | `a99a7ba7cc149dfc5b7f0847ccf7e28161f3a22de64741f54103cf03673822c8` |
| `vsfl_t2_v1.parity.json` | `1d047415e394a0b123612d5673657bd776d477c4614f225021d015db348752fe` |
| `vsfl_t3_v1.parity.json` | `298c11f73bf3adf1e32db2dac33d8021cf5e08f2ea7c4bf6bf995c78b418fa84` |

    fl_solver_regular vfl-parity --model M --fixture F --tolerance 0.00001

**Serving contract for the three VFL1 models.** Scores are *ranking* scores,
comparable only within one position's candidate fan -- never across positions,
and never as calibrated EVs. The reader is `fl_solver_regular::vfl_model`: a
single fixed-order accumulator, no rayon inside a prediction, ties in `argmax`
to the lower index, so a score is bit-reproducible from a pinned digest.
Standardized inputs are clamped to +/-8 and the tail's inverse std is exactly
zero; both are recorded in the VFL1 header, which is why these are not `T4M1`.

The T0 stand-in is the exception to read carefully: same geometry, but fitted
for normal-vs-normal opening play, so it is situation-blind to a Fantasyland
opponent. See the ruling in the debt ledger.

**Exact entry points**, for the situations where no model is served:

| Situation | Entry point | Cost |
|-----------|-------------|------|
| T4-vsFL | `fl_solver_regular label-t4` / exhaustive fan enumeration | small fan |
| T3-vsFL exact (dual-mode) | `fl_solver_regular label-t3 --hero-deals 0` | ~1 s |
| Hero in FL vs normal | `frontier` best-response scan | 46.6 ms build, 539 ns scan |
| FL vs FL | `FlSolver::solve` with `ObjectiveKind::PureV1` | static, no opponent argument |

| Artifact | Path | sha256 |
|----------|------|--------|
| solver binary (0.3.0) | `~/ofc-vsfl/target/release/fl_solver_regular` | `45c6113cc51313a1ca7605e8a56b258454ddddfff14222f1ea9754a56a2d5a3b` |
| FL EV config | `configs/fl_ev_regular_v3_direct2.json` | `c7279f3fa2e22490374f999774f112dac806ec8d64face5275838d9b832053dc` |

`fl_ev[14] = 9.109` is a **floor**, not a point estimate: the estimator's
Fantasyland side is static, and an adaptive one scores 0.42-0.46 higher per
hand. Any sensitivity analysis should push it upward, not symmetrically.

**Encoder arms**, for regenerating feature rows or adding a street:

| Binary | Path | sha256 |
|--------|------|--------|
| feature dump | `~/ofc-vsfl/target-encoder/release/vsfl_feature_dump` | `c3554036680c4f3ca70b74edb9afdb305e1073c028d468868ec4fd0c581e7763` |
| geometry probe | `~/ofc-vsfl/target-encoder/release/vsfl_geometry_probe` | `9fdea627875a017f3678b9c2236c1c40f51b8a8a219fd725c8856aced86be7d2` |

    vsfl_feature_dump --labels L.jsonl --street t1|t2|t3 --out PREFIX --threads N

Source is `vsfl_encoder/` (standalone crate, engine as a read-only path dep, its
own `CARGO_TARGET_DIR`). Extraction refuses to write if any candidate encodes an
all-zero hero block or if the opponent tail is anywhere nonzero, so the contract
is enforced corpus-wide rather than spot-checked.

**Label corpora and provenance:**

| Corpus | Path | Positions |
|--------|------|----------:|
| T1-vsFL | `~/ofc-vsfl/labels_raw_t1` | 20,000 |
| T2-vsFL | `~/ofc-vsfl/labels_raw_t2` | 50,000 |
| T3-vsFL | `~/ofc-vsfl/labels_raw_adaptive` | 20,000 |

T1's package and plan are `~/ofc-labelgen-vsfl-t1/package`, plan sha256
`00ea585941c96f3cc8dcf4a850d166c89746344817f66cb38e278f2e06f31288`, engine
features rev `4e1284dc74f7d98fd5b4821766a4c859fa7db877e464355db14276d65c963910`.
Labels of different `opponent_mode`, continuation policy, or engine features rev
are **different quantities** and must not be pooled; every label carries all
three so a consumer can check rather than assume.

### Debt ledger

| item | severity | note |
|------|----------|------|
| `belief.rs` `opponent_discard_count` returns 0 for hidden opponents | low | bites only if vs-FL routes through engine search; these teachers do not |
| hero outlook's unknown set includes the opponent's 14 unavailable cards | low | uniform across the fan, so second-order for ranking |
| `fl_ev = 9.109` measured against a static FL side | medium | a remeasure with an adaptive side is its own piece of work |
| T0-vsFL served by the T0-first stand-in rather than a vs-FL model | **accepted, owner ruling 2026-08-06** | see below |
| T1 label SE is draw-dominated, not N-dominated | informational | a deeper T1 teacher needs more T2 draws, not more N |
| `model_files` count in the package ledger excludes the two `.vfl1` images | cosmetic | the `t1_vs_fl` section names them with digests |

### Owner ruling, 2026-08-06: the T0-first stand-in is sufficient

M11 closes at three vs-Fantasyland models. T0-vsFL is served by
`t0first_model_v1.bin`, the ordinary T0 first-seat root policy, with **no T0
teacher built and no cost probe run**. The ruling was made against the corrected
cost table above rather than the earlier floor.

What makes the stand-in defensible, and what it does not fix:

  * **The interface matches exactly.** `t0first_model_v1` is 168 wide with body
    widths `[128, 64, 32, 1]` -- the same shape as all three VFL1 images -- and
    its last 46 columns are zero *by construction*, because acting first the
    opponent's board is empty and the free-slot outlook refuses it. That is
    precisely the geometry a hidden Fantasyland opponent produces, so the
    stand-in consumes a vs-FL row without reshaping, reheading, or imputation.
  * **The training signal does not.** It was fitted for normal-vs-normal opening
    play. It cannot know the opponent is in Fantasyland, so it cannot shade the
    opening toward the higher variance that situation rewards. It is
    situation-blind, not merely less accurate.
  * **The prize was the smallest of the four streets anyway.** Chain regret falls
    as the street gets earlier (0.485, 0.335, 0.175) and model advantage with it
    (5.0x, 2.2x, 1.9x); a T0 bar extrapolates near 0.10-0.15 at under 2x. The
    ruling trades the smallest expected gain against the largest teacher cost in
    the milestone.

**Revisit triggers.** Reopen T0-vsFL if either fires:

  1. **An M6 `fl_ev` sensitivity check flags T0-vsFL play quality.** `fl_ev` is a
     floor, not a point estimate, and the opening is where a mis-valued
     Fantasyland most changes the correct line. If that check shows the opening
     is where the sensitivity concentrates, the stand-in is the reason.
  2. **The M7 cascade.** If a generation's positions are collected by a chain
     whose T0 is situation-blind against a Fantasyland opponent, the error is not
     one street's regret -- it selects which positions every later street ever
     sees. A vs-FL T0 becomes a data-quality question rather than a play-quality
     one at that point.

Anything reopening this should start from the corrected cost table, not from the
1,123 core-s/root floor that preceded it, and should replace the projection with
a measured probe before a fleet is sized.
