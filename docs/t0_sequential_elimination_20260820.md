# Sequential elimination — spending particles where the answer is still open

2026-08-20. The measuring apparatus behind the T0 openings work, written up as a
method rather than as a run.

## The problem a fixed schedule cannot solve

Twenty openings were solved on a fixed schedule: twelve 1,024-particle batches
of each hand's top ten candidates. What the measurements then showed is that no
constant would have been right.

| opening | particles to decide the ranking |
|---|---|
| `Ah Kh Ts 6s 4h` | 3k — one batch of thirty-two leaves one survivor |
| `Ah Kc Qs 8h 7c` | 81k |
| `As Kd Kh Ts 7s` | 2,529k — two candidates 0.038 apart |

A factor of eight hundred between the cheapest and the dearest. Twelve batches
overpays the first by eleven and underpays the last by two hundred.

The requirement is quadratic in the reciprocal of the gap — halve the separation
and the particles quadruple — so the spread is a property of the game, not of
this hand set. Any fixed schedule is wrong at both ends of it.

## The rule

Measure a batch. Drop every candidate more than four standard errors behind the
leader. Measure the survivors. Stop when one remains, or when the budget runs
out and the honest answer is "closer than this instrument resolves".

The bar is four sigma throughout; what shrinks is how many points that buys,
because the standard error falls as the square root of particles spent:

```
1 batch   1.90 points        8 batches   0.67
2 batches 1.34              12 batches   0.55
4 batches 0.95              22 batches   0.40
```

Sequential testing normally needs a threshold that grows with the number of
looks, so repeated chances at a false rejection do not accumulate. Not at this
bar: a one-sided four-sigma test errs about 3e-5 of the time, so twelve looks
over thirty candidates expects 0.01 false eliminations.

## Three details that are the whole method

**Candidates are compared over the batches they share, not pooled separately.**
Every batch scores its candidates against the same sampled worlds, so the luck
of the draw cancels in the difference and does not cancel in a difference of
separately-pooled means. Measured on one opening, the scatter of an action's
level is 0.24 while the scatter of the gap between two is 0.13.

**A late entrant is judged at its own precision.** A candidate that survives to
round five has five batches; the leader may have twelve. Comparing all twelve to
all five would test a newcomer against a bar it has no evidence for. Only
batches measuring both count.

**Batches are weighted by particle count.** A 4,096-particle batch is worth four
1,024-particle ones in the estimate and in its variance alike, which is what
lets a hand carrying deep 4,096-particle work be pooled with shallow ones
without either being mis-weighted. Getting this wrong is not academic: the first
version of the runner could not see the 4,096-particle directory at all and was
about to spend forty more batches on the only opening already settled.

**One batch per open hand per pass.** Not one hand to completion — that stalls.
The first version ran each hand to its end, and a worker that reached an opening
needing two and a half million particles stayed there while every hand queued
behind it waited. Two of those were already past four sigma and needed nothing
but a convergence check; a third worker had finished its list and sat idle, five
of sixteen cores doing nothing.

## What it closed

The top-K it replaces was chosen by a single 1,024-particle batch — the
measurement this project concluded cannot rank anything (see
`docs/t0_particle_noise_20260820.md`). A true best sitting at K+1 was therefore
invisible by construction, and this was not hypothetical: on `Ad Kd Qs 9c 7h`
eighteen candidates were still within four sigma after one batch, and a top-ten
cut had already discarded eight of them.

Restated as a rate: of eleven openings the fixed schedule called "settled", only
six were actually converged once all thirty-two candidates were in the
comparison. The other five were settled *conditional on the top-ten assumption*,
which is a weaker claim than it appeared to be.

## What it costs

Simulated over the twenty openings, 40-55% of twelve batches of the top ten.
Three of the twenty finish on the first batch. Projected over all 134,459
canonical T0 openings: roughly $31,000 of GCP Spot against $67,000 for the fixed
schedule.

Both figures rest on a per-particle cost that is a lower bound — the fleet run of
2026-08-19 established only that GCP is at least 2.6x the local measurement
before it died — so treat them as ratios that are solid and absolutes that are
not. `docs/gpu_playout_investigation_20260820.md` covers what else could move
the absolute.

## Where it lives

`src/ofc_regular/hu_t0_sequential_elimination_v1.py` holds the statistics as
pure functions — `survivors`, `paired_gap`, `pooled_means`, `particles_needed`,
`verdict` — with no engine dependency, and `tests/test_hu_t0_sequential_elimination.py`
checks each of the properties above, including the two the first implementation
got wrong.

`particles_needed(gap)` is worth calling before committing to finish a hand: it
is what showed three of the twenty needing 147k, 1,642k and 2,529k particles to
separate actions worth the same within a twentieth of a point, which is weeks of
local compute or about $1,000 of GCP each. Those stay unresolved, and "the
difference is below what this instrument can resolve" is the correct result to
record rather than a failure to reach one.
