# Trainer ranking quality at T0 and T1 — what was measured, and the plan

Status: investigation record, 2026-08-08. Every number below was measured on
this box against the pinned m7v5 runtime (engine `e17aa39f…`, fl_ev 9.6); none
is extrapolated. Where something is unmeasured it says so.

The question this opened with was narrow — "T0 takes 20 seconds, can we
precompute it?" — and the measurements turned it into a different question,
which is why this record starts from the beginning rather than from the plan.

## 0. Two things run at T0, and they must not be confused

| | `decide` | `evaluate_t0` |
| --- | --- | --- |
| returns | one action | all 232 candidates, scored |
| method | the street's learned model, once | rolls every candidate out |
| seed-dependent | **no** | **yes** |
| cost, T0 first seat | < 1 s | ~21 s |
| who uses it | the webapp, the match gate | **the trainer's grading** |

`decide` was checked directly: three hands, five seeds each, identical
placements every time, `evaluator=learned`. The model's play is a function of
the hand alone.

So nothing here is a statement about the T0 model, and nothing here affects the
webapp or the gates. It is a statement about the ranking the trainer builds in
order to grade a user's opening.

## 1. The T0 ranking is seed-dominated

One hand, five seeds, the trainer's own rung (1,1,1):

| seed | best-action score | opening chosen |
| --- | --- | --- |
| 11 | +7.000 | A |
| 222 | +26.600 | B |
| 3333 | +12.000 | C |
| 44444 | +13.000 | D |
| 555555 | +13.000 | E |

Five seeds, five different openings. Per-action score spread across the seeds:
mean 25.87, median 25.60, max 51.20. Each seed's pick is scored between −25.6
and +9.0 by the others.

An OFC hand settles for roughly ±40 points, so a 26-point spread is not a noisy
signal — it is close to no signal. The trainer grades a user's opening against
"the best move" and the EV they gave up; both are drawn from this.

The trainer runs T0 at (1,1,1) on **every** precision setting, including
`high`. The code says why: T0 is the deepest tree and `high` would cost minutes.
That is a defensible speed decision. What was not recorded anywhere is what it
costs in grading quality, which is the gap this document closes.

## 2. More worlds did not help — but the measurement could not have shown it if they had

**Read §2a before drawing a conclusion from this section.** The ladder below is
kept as recorded, and the section that follows explains why its headline claim
was withdrawn.

`evaluation_samples` is how many sampled worlds the score averages over — one
world being one draw of the opponent's five cards, both players' twelve future
cards, and every opponent decision.

A ladder over 20 hands, prefilter fixed at keep 30, against a 128-world
reference on an independent seed base:

| worlds | mean regret | 95% CI | agreed with reference | s/hand |
| --- | --- | --- | --- | --- |
| 2 | 8.28 | [+5.72, +10.84] | 0.0 % | 77 |
| 8 | 8.89 | [+6.05, +11.74] | 5.0 % | 115 |
| 32 | 8.90 | [+6.13, +11.67] | 0.0 % | 222 |

Every paired difference includes zero (2 vs 32: −0.63, CI [−4.59, +3.34]).
Sixteen times the work bought nothing. Two of 60 rung evaluations matched the
reference exactly.

**And the reference does not agree with itself.** Two independent 128-world
runs of one hand chose different openings, each scoring the other's pick 8.98
points worse. So the regret column above is partly the reference's own noise,
and the honest reading of the ladder is not "32 worlds is as good as 2" but
"this comparison cannot resolve anything, including its own baseline".

## 2a. Correction: "worlds do not fix T0" was not supported

An earlier draft of this document concluded from §2 that worlds do not help at
T0. That conclusion is withdrawn. §2 states that the reference failed its own
self-check and then reasons as though it had not: **a ladder cannot detect an
improvement smaller than its baseline's own noise**, and the baseline's was
~9 points. Owner's correction, and it is right.

Two mechanisms were raised for why the ladder might have been blind rather than
the improvement absent, and both are real:

* **the rungs were all too low.** The hero-only evaluation (§5) improves with
  worlds over the same 4→64 range. That is the same game with less variance
  per world, so the head-to-head form should improve too — just needing far
  more worlds than 32.
* **the fan was too wide.** With 232 candidates each carrying tens of points of
  noise, the argmax mostly picks whichever candidate got lucky. That winner's
  curse grows with the fan and averaging does not remove it while the fan
  stays wide.

So the ladder was re-run in a form that needs no reference at all: the **same**
configuration twice under independent seeds, asking whether the two agree. The
fan was cut to the top 10 to suppress the curse. Three hands:

| hand | 16 worlds | 64 worlds | 256 worlds |
| --- | --- | --- | --- |
| 2c2d2h2s3c | agree, cross 0.00 | cross 8.70 | cross 11.83 |
| 2c3c5dAdAh | cross 12.00 | cross 18.80 | cross 19.26 |
| 2c3d7hKcQd | cross 16.10 | cross 26.30 | cross 6.50 |

`cross` is what one run gives up by taking the other run's opening, scored by
its own numbers. Mean per-action spread stayed at 7–12 points across the whole
sweep; one of nine cells agreed. At 256 worlds a single ranking costs 357 s.

**What this does and does not establish.** It does not show worlds can never
fix T0 — three hands is a small sample and the third hand's cross fell at 256.
It does show that **through 256 worlds with the fan cut to 10, no convergence
appeared**, at a cost already far past anything interactive. The earlier
claim is replaced by this one, which is what the measurement supports.

## 2b. The original ladder, as recorded

## 3. Cost, for the record

One hand, T0 first seat, no prefilter:

| (candidate, evaluation, downstream) | seconds | vs the trainer's rung |
| --- | --- | --- |
| (1, 1, 1) | 20.8 | 1.00x |
| (2, 4, 2) | 131.7 | 6.32x |
| (4, 8, 4) | 261.9 | 12.57x |
| (8, 32, 4) | 932.9 | 44.78x |

Across 20 hands spread through the canonical index the cheapest rung costs
21.06 s ± 7 % (min 19.0, max 22.1), so these are not one-hand accidents.

## 4. Narrowing the candidate fan does work

The budget is `candidates × worlds × depth`. T0 is the only street that spends
it on 232 candidates; T1–T3 have 27. Cutting candidates buys worlds at the same
price.

Measured with the engine's built-in prefilter, one hand, four seeds:

| arm | s/hand | distinct openings | mean spread |
| --- | --- | --- | --- |
| (1,1,1), no prefilter | 19.2 | 3 / 4 | 26.20 |
| 1 particle over all → 8 worlds over top 30 | 30.9 | 4 / 4 | 19.58 |
| 2 particles over all → 16 worlds over top 30 | 84.9 | 2 / 4 | 11.82 |

Reaching a spread of ~12 costs 261.9 s without a prefilter and 84.9 s with one:
about **3x more efficient**. The direction is right; the level reached is still
not usable.

A **model-based** cut should be strictly better than the sampling cut measured
here — it is deterministic, so the cut itself carries no seed dependence, and
the T0-first model was trained on 256-particle joint-exact labels rather than
the 1–2 particles the coarse stage can afford. `fast_t0_first_action`
(`search.rs:2170`) already computes a score for all 232 candidates and returns
only the argmax; exposing that vector is a small change, not new logic.

Note the prefilter is **T0-only** in the engine today: `prefilter_samples` /
`prefilter_keep` are read by `evaluate_t0` and by nothing else. T1–T3 are three
independent implementations with no shared scoring helper, so narrowing them
means three separate changes.

## 5. Removing the opponent looks better than averaging over it

Owner's observation, and it is the one that reframed the problem: at T0 first
seat the opponent is unseen and exchangeable, so **no opening can be preferred
because of what the opponent holds**. Sampling them adds variance and no
information. Hero's own draw distribution is unchanged by integrating them out
— drawing from the 47 unseen cards is exactly the marginal.

A hero-only evaluation was built to test that: hero plays their own hand with
the engine's real policy, the opponent board is filled only to satisfy the
observation geometry and is never scored, and the finished board is scored
standalone (royalties + Fantasyland − foul) via `terminal_score(board, None)`.

Twenty candidates, three seeds:

| worlds | s | distinct top-1 across 3 seeds |
| --- | --- | --- |
| 4 | 4.6 | 3 |
| 16 | 17.2 | 2 |
| 64 | 67.9 | 2 |

Two things separate this from §2: it **improves with worlds** where the
head-to-head evaluation did not, and it is about **4.5x cheaper** per world
because no opponent decisions are played. Scores still moved 0.6 points between
seeds at 64 worlds, so it is not converged either — but it is moving in the
right direction, which nothing else here was.

What it gives up is the head-to-head term. Against an unknown symmetric
opponent the expected row score should be a function of hero's own board
strength, which the standalone score already carries — **that is an argument,
not a measurement**, and §7 lists it as the open question it is.

## 6. T1 has the same defect, and there the ordinary fix works

The trainer runs T1 at one world too, on every precision. Two hands, both
seats, five seeds:

| worlds | s | mean per-action spread |
| --- | --- | --- |
| 1 (current) | 0.3–0.4 | **16–40** |
| 4 | 1.0–1.5 | 12–16 |
| 16 | 3.3–5.3 | 5–9 |
| 64 | 13–21 | **3.7–4.4** |

The spread falls monotonically, 40 points to 4. T1 differs from T0 in two ways
that matter: 27 candidates instead of 232, so a world costs about sixty times
less; and the opponent's five cards are already visible, so there is far less
hidden state for a world to guess wrong about.

T3 already runs 32 worlds and T2 runs 4. Why T1 was left at 1 is not recorded;
the "cheaper the deeper" shape of the table does not explain it, because T1,
T2 and T3 all have 27 candidates.

## 7. What is not known

* Whether the hero-only ranking (§5) picks the **same** openings a trustworthy
  head-to-head evaluation would. It cannot be checked against the §2 reference,
  because that reference does not agree with itself.
* Whether the true best action survives a top-K cut, at any K. The project has
  one prior measurement — the T0 label pipeline's two-stage prefilter kept 48
  and reported exact-best survival 19/20 — but nothing at K = 5 or 10.
* Whether T2 and T3 are noise-dominated at their current world counts. Not
  measured. T3 at 32 worlds may well be fine.
* Whether a T0 ranking against a **Fantasyland** opponent would serve as the
  reference the normal case lacks. It is attractive because the FL side is
  solved exactly (`fl_solver_regular`'s non-dominated frontier best response,
  pinned against brute force on 344 pairs), so the opponent contributes no
  variance at all — and hero's information at T0 first seat is the same in both
  cases: nothing. Two caveats: the EV landscape differs (a Fantasyland opponent
  is far stronger on average, which may shift hero's optimum toward royalties
  and Fantasyland entry over row wins), and **the argument is specific to T0** —
  owner's correction, and it is right: from T1 on, the normal game shows hero
  the opponent's board while the Fantasyland game shows nothing, so the two
  stop being the same decision. `fl_solver_regular` has `label-t3`, `label-t2`
  and `label-t1` but no T0 command; porting one from `t1_teacher.rs` is
  200–300 lines, since the exact scoring is reusable as is.

## 8. Plan

**Step 1 — measure T2 and T3 (today, minutes).** The same five-seed check §6
used. Until this is done, "T2 and T3 need narrowing" is an assumption. T3 at 32
worlds may already be adequate, in which case it needs nothing.

**Step 2 — set the world counts from the measurements (today, config only).**
T1 from 1 world to 16 is already justified by §6: 3–5 s for a spread of 5–9
against 0.35 s for 16–40. T2 and T3 follow from step 1. No code changes; this
is the trainer's `SAMPLES` table.

**Step 3 — narrow the candidate fan (engine work).** In cost order per street:

| street | candidates | now | proposed | projected |
| --- | --- | --- | --- | --- |
| T1 | 27 | 1 world, 0.35 s | top 10 × 64 worlds | 8.3 s |
| T2 | 27 | 4 worlds, 1.1 s | top 5 × 32 worlds | 1.6 s |
| T3 | 27 | 32 worlds, 4.2 s | top 5 × 64 worlds | 1.6 s |
| T0 | 232 | 1 world, 21 s | top 10, and §5 rather than more worlds | see below |

T2 and T3 get *more* worlds for *less* time. T1 gets 64x the worlds for 8 s.
Each street needs its own change; there is no shared helper to modify once.

Before any of it: **measure top-K survival** — how often the reference's best
action falls outside the model's top K, for K = 5, 10, 20, 30. A cut that drops
the true best does not become safe by spending the savings on worlds, and at
K = 5 nothing has ever been measured.

**T0 is the exception and does not follow this plan.** Narrowing to 10 and
raising worlds to 256 was tried directly (§2a) and did not converge, at 357 s a
ranking. The candidate route is §5's hero-only evaluation, gated on the top-K
survival question and on finding a reference it can be checked against, for
which §7's T0-vs-Fantasyland is the most promising idea on the table.

## 8a. Where step 3 stands

Steps 1 and 2 are done. The world counts in `trainer/engine_eval.py` were
raised on 2026-08-08 and re-timed on fresh roots: T1 16 worlds (5.8 s worst
seat), T2 16 (2.5 s), T3 128 (3.3 s), T0 unchanged. Worst single ranking 5.8 s.

Step 3 was scoped against the engine. Three findings shape it:

* **T1, T2 and T3 share one scoring helper.** `learned_action_over`
  (`search.rs:1942`) computes a value for every candidate and is called from
  exactly three places — the T1, T2 and T3 learned evaluators. It already
  builds the full `values` vector and already ranks it with
  `canonical_descending_indices`; only the first element is returned. Exposing
  the vector is one change, not three.
* **T0 has its own path**, `fast_t0_first_action` (`search.rs:2170`), which
  does the same thing over the 232-candidate fan.
* **Label generation already narrows at T0 and nowhere else.** The T0 plans
  carry `prefilter_samples: 32` / `prefilter_keep: 48`; the T1, T2 and T3 plans
  carry no prefilter and score all 27. The continuation streets inside every
  label are played by the learned models one action at a time — the same path
  `decide` uses — so a label's noise comes from its particles, not from its
  continuations. That is why T2 needed 2,048 particles.

So narrowing is worth as much to **label generation** as to the trainer: at
25,000 positions a cut to the top 10 of 27 is 2.7x the particles for the same
spend, and the T1 relabel is currently budgeted at 10–20 fleet hours.

### Top-K survival, measured

`model_scores` was added to the engine for this (§8b) and the question is now
answered. For each root: the model's ranking, and the rollout's best action at
the world count the trainer now runs; the statistic is where that action sits
in the model's order.

Twelve hands per street and seat:

| street / seat | K=3 | K=5 | K=10 | K=20 |
| --- | --- | --- | --- | --- |
| T1 first | 75.0 % | 91.7 % | 91.7 % | 100 % |
| T1 second | 75.0 % | 91.7 % | 91.7 % | 100 % |
| T2 first | 83.3 % | 100 % | 100 % | 100 % |
| T2 second | 75.0 % | 83.3 % | 100 % | 100 % |
| T3 first | 83.3 % | 100 % | 100 % | 100 % |
| T3 second | 83.3 % | 91.7 % | 91.7 % | 100 % |

T2 was then re-run at **50 hands**, because a cut of 5 was being considered on
the strength of its two 100 % cells, and twelve hands cannot support that:

| street / seat | K=3 | K=5 | K=10 | K=20 |
| --- | --- | --- | --- | --- |
| T2 first | 72.0 % | **86.0 %** | **94.0 %** | 98.0 % |
| T2 second | 74.0 % | **84.0 %** | **96.0 %** | 100 % |

The 100 % at K=5 did not survive the larger sample. Recorded because it is the
same shape of error this document already corrected once: a small sample read
as a result.

What a miss costs matters as much as how often it happens, since a cut that
drops a near-tie has dropped nothing worth having:

| K | T2 first: misses, mean / max EV lost | T2 second |
| --- | --- | --- |
| 5 | 7/50, 1.34 / 4.49 | 8/50, 1.54 / 2.83 |
| 10 | 3/50, 0.89 / 1.48 | 2/50, 2.18 / 2.83 |

**Ruling: K = 10 at every street.** At K=5 the cut misses once in six hands for
about 1.4 points, which is inside the rollout's own spread (4.8–6.1 points at
T2's 16 worlds) but with no margin; at K=10 it misses once in twenty for less.
The saving barely changes — 27 candidates down to 10 is 2.6x, down to 5 is
5.4x, and both are spent on worlds — so the safer cut costs little.

Note the "truth" here is the rollout's argmax, which carries its own 1–6 points
of spread. Some misses are therefore the rollout preferring a near-tie rather
than the model erring, which makes these survival rates a **lower bound**.

### 8b. `model_scores`, added 2026-08-08

The engine gained one request kind, `model_scores`, which returns every legal
action's learned score in canonical descending order. It is what `decide`
already computes and discards.

The change is small because `learned_action_over` (`search.rs`) was already the
one place T1, T2 and T3 score their fans: it was split into
`learned_values_over`, which returns the vector, and a two-line
`learned_action_over` that calls it and takes the first canonical element as
before. Existing behaviour is unchanged by construction — same values, same
order, same tie-break. T0 first seat has its own arm mirroring
`fast_t0_first_action`. T4 is refused with a message: it decides by exact
enumeration and has no learned ranking to publish.

Verified rather than assumed: at every street and both seats, plus T0 first,
the top row of `model_scores` is byte-identical to the action `decide` returns,
and the scores descend. If the two had disagreed, the published ranking would
not have been the one in service and any cut built on it would have been
cutting against the wrong order.

Engine rebuild safety, checked rather than assumed: the pinned engine
`e17aa39f…` lives in `/home/wner/ofc-labelgen-m7v5/build/runtime/native/`, not
in the repository's `target/`, so an ordinary `cargo build` cannot touch it.
Model weights are separate files that no build reads or writes. The rule is
simply that a fresh build must never be copied into the package runtime; the
label worker's digest check is fail-closed if one ever is.

## 9. Precompute — deferred, and why

The opening question was whether the 134,459 canonical T0 openings (measured:
2,598,960 hands, 19.33x suit-isomorphism reduction, index written to
`t0first_precompute/index.json`) could be solved once and looked up.

They can, and the arithmetic is not the obstacle: 21 s a hand is 778 core-hours,
about 40 minutes on the 146-instance fleet. The obstacle is that at 21 s a hand
the answer is the noise measured in §1, and freezing one draw of it forever is
worse than computing a fresh one each time. A precompute is worth doing at a
rung that is *stable*, and no such rung has been found yet — that is what makes
this a §8 problem rather than a scheduling one.

Two further points for whoever picks it up: the suit-isomorphism claim the
19.33x rests on has **not** been verified key-by-key (an attempt could not
separate a symmetry failure from the §1 noise, and the test needs an evaluator
whose answer does not move); and the current T0-first model's labels carry
fl_ev **9.109**, not the 9.6 the rest of the chain now runs, so anything frozen
against it is frozen against a superseded constant.
