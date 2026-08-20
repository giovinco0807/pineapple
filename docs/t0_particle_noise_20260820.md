# What a 1,024-particle T0 batch can resolve

2026-08-20. Twelve 4,096-particle batches of one opening, measured to find out
whether the single-batch rankings this project has been quoting are decidable at
the resolution they were measured at. They are not.

## The claim under test

Twenty openings were solved on 2026-08-19 — a 64-particle sieve keeping sixteen
with margin 2.4, then every survivor re-scored uniformly at 1,024 particles, no
race — one batch each. Four more followed on hand-picked special hands. Those
twenty-four batches produced two numbers that were then quoted as facts:

* the shipped T0 first-seat policy picks the measured best on 20 of 24
* its mean regret is 0.107 points a hand

Both read a ranking off one draw of the particle stream. Nine of the
twenty-four separated first from second by less than 0.5 and two by less than
0.2, the narrowest 0.037 on `Ad Ts 9c 7d 6d`, and nothing established that 1,024
particles could see a gap that size.

## The measurement

`Ah Kc Qs 8h 7c`, the first of the twenty. Its measured top three at 1,024:

| | 1,024 | placement |
|---|---|---|
| #1 | +1.331 | T:Qs / M:Kc 7c / B:Ah 8h |
| #2 | +0.708 | T:Qs / M:Ah Kc / B:8h 7c |
| #3 | +0.086 | T:Kc Qs / M:Ah / B:8h 7c |

Those three actions alone, re-scored at 4,096 particles, twelve times, using
`restrict_action_keys` so nothing is spent on the twenty-nine candidates already
far behind. Twelve batches in three arms of four, the arms differing only in how
far apart their evaluation seeds sit — a question that arose mid-run and is
answered in §2.

## 1. The gap was overstated four times over

```
gap #1 vs #2, twelve batches at 4,096:  +0.154   95% CI [+0.023, +0.285]
the single 1,024 batch said:            +0.623
```

Four times too large. The mechanism is winner's curse and it is not subtle: the
action that wins an argmax over thirty noisy estimates is disproportionately one
that drew well, so its lead over everything else is inflated by however much of
that draw was luck. Any regret computed by subtracting from a single-batch
maximum inherits the same factor. **The 0.107 figure is an upper bound roughly
four times its true value.**

The ranking itself survived — the interval excludes zero, so #1 is still the
better action — but by 0.15, not 0.62.

Per-batch scatter of the gap:

```
at 4,096:  SD 0.231
at 1,024:  SD ~0.46      (SD doubles as particles quarter)
```

A single 1,024-particle batch therefore separates two actions only when their
true gap exceeds about 0.9. Against the twenty-four solved hands:

| | |
|---|---|
| gap ≥ 0.9 — decided by one batch | **9 / 24** |
| gap < 0.46 — inside one SD, a coin flip | 7 / 24 |

**Fifteen of the twenty-four rankings were never established.** This does not
say the policy is worse than 20-of-24; it says the instrument could not tell,
and "matches the measured best" was counting undecided hands as correct.

## 2. Seed spacing is not the problem

The first four batches came back with #1 rising monotonically (+0.517, +0.804,
+0.962, +1.079) on seeds spaced a thousand apart. Four independent draws land in
order about four times in a hundred, and the GCS ladder plan addresses its roots
at `base + offset * 7` — seven apart — so the remaining eight batches were split
into two arms to test whether nearby seeds share structure.

| arm | seed spacing | gap mean | gap SD |
|---|---|---|---|
| stride7 | 7 | −0.067 | 0.176 |
| spacing1000 | 1,000 | +0.330 | 0.128 |
| far | ~1.3 × 10⁸ | +0.200 | 0.200 |

One-way ANOVA over the three arms gives F(2,9) = 5.61 against a 5% critical
value of 4.26, which is marginally significant — but **the means do not order by
spacing**. The tightest arm sits lowest, the widest in the middle. No
correlation story produces that shape. With four batches an arm, a post-hoc test
run because the pattern was noticed first, and three arms whose SDs (0.128,
0.176, 0.200) are all the same size, the parsimonious reading is that per-batch
scatter is simply large and the opening run's tight, ordered four was a
coincidence.

Two things follow. The `stride 7` addressing in the label plans is not
implicated. And an arm of four batches is not enough to pin a gap — the first
arm's own interval, [+0.204, +0.455], excluded the twelve-batch estimate
entirely.

## 3. What this settles about the ladder run

`hu_t0_ladder20_explicit_plan_v1` buys twelve 1,024-particle batches of each of
the twenty openings, on the arithmetic that k batches carry the precision of one
batch of 1,024k. At the scatter measured here that gives:

```
SEM of the gap over twelve batches at 1,024:  0.134
resolves a gap of about:                      0.27
```

| | hands decided |
|---|---|
| one batch at 1,024 | 9 / 24 |
| twelve batches at 1,024 | **20 / 24** |

The four it will not settle are the ones separated by 0.248, 0.219, 0.180 and
0.037 — and because those gaps *are* the cost of choosing wrongly, picking
either action there loses at most a quarter of a point. For building a table
that is not a defect.

The twelve-repeat design was chosen before any of this was measured, on the
grounds that repeats also expose the 1,024 scatter directly. It happens to be
about the right size, which is luck rather than judgement; four repeats would
have been too few by the margin §2 demonstrates.

## Files

* `D:/ofc_data/t0_rung/local4096/hand_000_top3_4096.json` — the first arm
* `D:/ofc_data/t0_rung/local4096/seed_independence.json` — all three arms
* `src/ofc_regular/analyze_hu_t0_ladder20_particle_noise.py` — the analysis the
  GCS corpus will go through, including the naive/split-sample bracket that
  exists because of §1
