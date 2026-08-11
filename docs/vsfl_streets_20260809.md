# The vs-Fantasyland streets: what limits each one, and what was bought

2026-08-09 (JST; the fleet receipts are stamped 2026-08-08 UTC).

Three distilled rankers serve the streets where hero plays a normal board
against a Fantasyland opponent — `vsfl_t1_v1.vfl1`, `vsfl_t2_v1.vfl1`,
`vsfl_t3_v1.vfl1`, all built 2026-08-06 at 168 features and 128-64-32. They are
used by every vs-FL rollout and by the T0-vs-FL teacher written on 2026-08-08,
so their error is a floor under everything downstream.

This records what was measured about them, what was wrong with each, and what
was rebuilt. The short version: **all three were limited, and by three
different things.** Nothing in the diagnosis generalised from one street to the
next, and on three separate occasions the axis that looked obvious was the wrong
one to buy.

## 0. Where they stood

Holdout regret, from the training reports the models shipped with. Regret is the
teacher's value of its own best placement minus its value of the model's pick;
top-1 agreement is reported alongside but is not the criterion, because naming a
different placement of equal value costs nothing.

| street | train | holdout | **regret** | top-1 | chain bar (no model) |
| --- | --- | --- | --- | --- | --- |
| T1 | 18,000 | 2,551 | **0.0944** | 73.3% | 0.1750 |
| T2 | 45,000 | 5,543 | **0.1550** | 73.3% | 0.3346 |
| T3 | 18,000 | 2,273 | **0.0962** | 78.3% | 0.4847 |

All three beat the heuristic chain they replaced, T3 by 4.7x. The question was
why T2 was 1.6x worse than its neighbours on the largest corpus of the three.

**A correction worth recording.** The 0.4847 in the last column was read earlier
in the day as the vs-FL chain's *systematic error*, and the conclusion drawn was
that no amount of work on the T0 teacher could beat it. That was wrong:
`chain_bar` is the baseline the model is scored against, not the model's own
error. The models are at 0.09–0.16, not 0.48.

## 1. What limits each street

The labels carry a standard error per candidate, so the noise floor is
computable without relabelling anything: resample every candidate's value from
`N(ev, se)` twice and ask what one draw gives up by taking the other draw's
argmax, scored on the recorded means. A model fitted to one draw and scored on
one draw cannot do better than that floor.

| street | samples | mean SE | **noise floor** | holdout regret | floor share |
| --- | --- | --- | --- | --- | --- |
| T1 | 1,600 | 0.238 | 0.0885 | 0.0944 | **94%** |
| T2 | 200 | 0.499 | 0.2393 | 0.1550 | **154%** |
| T3 | 400 | 0.000 (unfilled) | 0.0004 (measured at training time) | 0.0962 | **0.4%** |

Three different verdicts:

* **T1 is at its labels' ceiling.** 94% of its error is the labels'.
* **T2 is past it.** Its floor *exceeds* the error it achieves, which means the
  model is averaging label noise out and is already better than a single draw of
  its own teacher. More positions or more capacity cannot move it.
* **T3 is model-limited**, and uniquely so. Its labels enumerate hero's draw
  exhaustively (`deal_mode: exhaustive`, `hero_deal_samples: 8436`) and 99.6% of
  its error is the model's own.

T2's flat positions curve is the same finding seen from the other side:
24,000 → 48,000 positions bought 4.8%, against 17% for the same doubling at T3.

## 2. Most of the samples paid for are discarded

Both vs-FL teachers draw the opponent's fourteen cards from the deck **as it
stands at the root**, before hero's own future cards are dealt, and then reject
whichever samples collide. The rejection is statistically correct — the retained
samples are distributed as the conditional hero faces — but the effective sample
count is a fraction of the nominal one:

| street | nominal | mean usable | share |
| --- | --- | --- | --- |
| T1 | 1,600 | 32.3 | **2.0%** |
| T2 | 200 | 13.2 | **6.6%** |
| T3 | 400 | 96.0 | 24.0% |

T2's 0.499 standard error is the error of thirteen samples, not two hundred.

The T0-vs-FL teacher written on 2026-08-08 avoids this by drawing the opponent
at the leaf, once hero's seventeen cards are fixed. That is **not** free to
retrofit: the root-drawn pool is *solved* once and reused by every leaf, and
drawing per leaf multiplies the solving cost by the number of leaves. Enlarging
the pool is the cheap version of the same fix — and, as §3 shows, at T2 it is
not worth doing at all, because the opponent is not where the variance is.

## 3. Which axis to buy — asked three times, wrong twice

Each teacher has several axes. On three separate streets the plausible axis was
measured and found nearly worthless.

**T0-vs-FL** (6 roots, self-agreement between two independent seeds):

| rung | self-disagreement |
| --- | --- |
| 16 samples / 8 draws | 0.9651 |
| **32** samples / 8 draws | 0.9672 — 1.65x the cost, **zero** gain |
| 16 samples / **16** draws | 0.4277 — halved |

**T2-vs-FL** (32 roots, mean label standard error, fl_ev 9.6):

| change from 200/32/16 | cost | SE | cost per 1% of error removed |
| --- | --- | --- | --- |
| samples ×4 | 4.2x | −7.7% | 42.1 |
| samples ×16 | 17.1x | −10.7% | 58.9 |
| t4_draws ×2 | 1.26x | −4.3% | 6.1 |
| **t3_draws ×2** | **1.60x** | **−27.9%** | **2.2** |

Re-asked on top of `t3_draws=128`, the other two axes stayed bad: samples ×4
bought 7.5% for 4.4x, t4_draws ×2 bought 5.1% for 1.4x.

`t3_draws` scales as a textbook Monte Carlo axis and does not flatten:

| t3_draws | 32 | 256 | 512 | 1024 |
| --- | --- | --- | --- | --- |
| label SE | 0.4498 | 0.1620 | 0.1147 | 0.0812 |
| ratio per doubling | — | 1/√2 | 1/√2 | 1/√2 |
| core-s per root | 9.95 | 22.50 | 31.13 | 60.24 |

It is cheap because ~8.4 of the 9.95 core-seconds is fixed Fantasyland work that
every extra draw reuses.

**T3-vs-FL**: samples are worth *less* than nothing to raise. Cost is exactly
linear in them (6.15 / 12.87 / 25.87 core-s at 100 / 200 / 400), and labelling
at 100 instead of 400 costs **0.0009** of regret against the richer label — 1.2%
of the model error it feeds — while buying 4.2x the positions.

## 4. What was rebuilt

Both runs use the Rust `label-t2` / `label-t3` teachers directly. The Python
fleet worker is a wrapper around the same binary, so the fleet path was reused
unchanged; what it is *not* is a route for local experiments, since it pins the
roots file inside the runtime root and the fl_ev constant by digest.

Both move fl_ev **9.109 → 9.6**, closing the loop the v4 config's own provenance
names as the point of the cascade relabel.

### T3 — `t3vsfl-100k-s100-flev96-r1`

100,000 positions at 100 samples, 58 × c4-standard-8 in asia-northeast1-b.
**28 minutes**, 100,000 files, no gaps.

| positions | 128-64-32 | 256-128-64 | 512-256-128 |
| --- | --- | --- | --- |
| 11,875 | 0.0960 | | |
| 23,750 | 0.0693 | | |
| 47,500 | 0.0611 | | |
| **95,000** | **0.0508** | **0.0455** | **0.0420** |

**0.0962 → 0.0420**, a 2.3x improvement, and now the best of the three streets.

The projection made before the run was 0.030, and it was too optimistic: the old
corpus's n^-0.53 slope was extrapolated straight out, but the new corpus flattens
to about n^-0.27 by 95,000. The direction was right and the size was not.

Top-1 agreement fell, 78.3% → 69.9%, while regret more than halved. That is what
the two metrics measure: the new model misidentifies the best placement more
often and loses far less when it does.

### T2 — `t2vsfl-50k-t3d512-flev96-r1a` / `-r1b`

50,000 positions at `t3_draws=512`, samples and `t4_draws` unchanged because
neither pays. Cut into 59 shards across us-west1-a (31) and us-east1-b (28),
because asia-northeast1's PREEMPTIBLE_CPUS was spent on T3 and 59 is what the
other two approved zones hold between them.

The corpus stays at 50,000 deliberately. Its scaling curve is flat *at the
current label noise*; whether it stays flat once the labels are fixed is a
question the new corpus answers, not one to pre-buy roots for.

## 5. Supervisor contract

`validate_m7_t2_2048_run_contract` hard-coded the M7 T2 relabel's identity: 2048
samples, a 174-shard partition, three-digit shard ids. Those are contract fields
now (`partition`, `samples`), with the partitions in `CANONICAL_PARTITIONS`. The
same assertions run, on the values the named run is allowed to carry.

Regression evidence: all eight run plans the supervisor launched from during the
M7 T2 relabel are still accepted, the two superseded un-suffixed names are still
refused, and a plan with a drifted sample count or a shard boundary moved by one
is still refused. 13/13.

## 6. What is not known

* **Whether T2's positions curve steepens once its labels are fixed.** It should
  — the flatness was diagnosed as the label ceiling — but it has not been
  measured, and the roots file holds exactly 50,000, so more would need
  generating.
* **T1 was not rebuilt.** Its labels are 2.0% efficient and its floor is 94% of
  its error, so it is the same problem as T2 and probably the same fix; the axis
  ladder has not been run there.
* **Whether the T0-vs-FL convergence ladder means anything now.** It was
  measured with these three rankers as continuations and was stopped at the
  (16,32) rung on the grounds that measuring convergence against continuations
  about to be replaced is measuring the wrong thing. It has to be redone.
* **Whether vs-FL T0 rankings differ from normal-table T0 rankings** — the
  hypothesis the T0-vs-FL teacher was written to test — is still open. The one
  measurement made, on six roots, found the vs-FL best inside the normal model's
  top 20 every time and inside its top 10 five times out of six.
