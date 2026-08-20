# Solved T0 first-seat openings — the lookup table

2026-08-20. Twenty-four openings measured, folded into one entry apiece, keyed
so a repeat of any of them — or any of their suit-permutations — is answered
without computing anything.

```
python -m ofc_regular.hu_t0_solved_openings_v1 --hand "Ah Kc Qs 8h 7c"
```
```
Ah Kc Qs 8h 7c
  (solved as Ac Kd Qh 8c 7d, suit-isomorphic)
  best      top: Qs  mid: Kc 7c  bot: Ah 8h
  runner_up top: Qs  mid: Ah Kc  bot: 8h 7c
  EV +0.718   gap to second +0.185  95% CI [+0.090, +0.281]
  status settled  (92,160 particles behind the gap)
```

`--list` prints the whole table, `--build` regenerates it from the measurement
files, and `lookup()` is importable.

## One entry answers up to twenty-four deals

At T0 acting first both boards are empty and nothing is discarded, so the
position is the five cards and nothing else, and permuting suits maps it to an
equally-valued position. Entries are stored under the lexicographically smallest
of the twenty-four relabellings; a query is canonicalised the same way and the
stored placement handed back through the *inverse* relabelling, so the answer
names the querent's own cards. `Ah Kc Qs 8h 7c` and `Ad Ks Qh 8d 7s` are one
entry, and the second is told about diamonds and spades.

The same symmetry is what collapses 2,598,960 raw openings to 134,459.

## Only eleven of the twenty-four are decided

Every row carries a `status`, because most of them do not support the ranking
they appear to state:

| | |
|---|---|
| `settled` | first-to-second gap excludes zero at 95% |
| `unresolved` | it does not — best known, not an established ranking |

Thirteen are `unresolved`. That is not a defect in the table, it is the state of
the measurements: a single 1,024-particle batch has a top-two gap scatter of
0.46, so it separates two actions only when they are about 0.9 apart, and most
entries rest on one or two batches. See `docs/t0_particle_noise_20260820.md`.

The interval uses that measured 0.46 rather than a per-hand standard deviation,
because a standard deviation from two samples is worth less than a constant
measured over twelve. Batches are pooled by particle count — a 4,096-particle
batch counts four times a 1,024-particle one, in the mean and in the variance
alike.

## The one entry measured properly

`Ac Kd Qh 8c 7d` (canonical form of `Ah Kc Qs 8h 7c`) carries 92,160 particles:
two 1,024-particle batches and twenty-two at 4,096. It is the hand the particle-
noise study was run on, and the only row whose gap is known to better than a
tenth of a point.

```
1 vs 2   scattered seeds only  n=10  +0.222  95% CI [+0.079, +0.365]
         all batches           n=22  +0.185  95% CI [+0.090, +0.280]
         original arms as units n=3  +0.154  95% CI [-0.348, +0.656]
```

The three readings exist because the first twelve batches were run as three arms
of four sharing a seed neighbourhood each, and pooling them as three units left
the question open. Twelve further seeds scattered individually across 300M–1.6B
have no neighbourhood structure to argue about, and ten of them alone exclude
zero. Ten rather than twelve: the run was stopped once overturning it would have
required the last two batches to come in at −0.94 apiece, 4.9 standard
deviations below anything observed.

**The best action is `top Qs / mid Kc 7c / bot Ah 8h`**, and the single
1,024-particle batch that first measured this hand claimed a gap of +0.623 —
about three times the truth.

## What it says about the policy

On this one settled opening the shipped T0 first-seat model:

| model rank (of 232) | model score | measured EV | measured rank |
|---|---|---|---|
| 1 | 4.88 | +0.530 | 2 |
| 2 | 4.40 | +0.083 | 3 |
| 3 | 3.31 | **+0.715** | **1** |

It picks the measured second, ranks the true best third, and orders the three
contenders backwards. Choosing its top action costs **0.185 points**. Its scores
are not on the measured scale either — 4.88 against +0.530.

Both halves of that are worth keeping in view. As a *narrowing* stage it works:
the true best is inside its top three of two hundred and thirty-two. As a
*policy* it does not: asked to choose, it loses a fifth of a point here. And its
EV output is not usable as a value at all. The current arrangement — let the
model narrow, let particles measure — is the one that matches what it can do.

## Files

* `src/ofc_regular/hu_t0_solved_openings_v1.py` — build, lookup, CLI
* `D:/ofc_data/t0_rung/t0_solved_openings.json` — the table
* `D:/ofc_data/t0_rung/local4096/hand0_verdict.json` — the settled hand's three readings
* `D:/ofc_data/t0_rung/akq20_candidates.csv` — the twenty akq openings as a
  sheet, with `batches`, `spread` and `hand_resolved` columns carrying the same
  caution as `status` here
