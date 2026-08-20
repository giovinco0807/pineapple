# T0 ladder for the twenty akq openings — state at 2026-08-20

Stopped by request part-way through the first batch. This is what exists, what
it cost, and what starting again requires.

## Where it stopped

9 of 240 runs, all of batch 0:

| hand | opening | batches |
|---|---|---|
| 0 | `Ah Kc Qs 8h 7c` | 1 |
| 1 | `Ah Kh Ts 6s 4h` | 1 |
| 2 | `As Jd 6d 2d 2h` | 1 |
| 3 | `As Jc Th 6h 6s` | 1 |
| 4 | `Qd Js 8s 6c 4c` | 1 |
| 5 | `Ac Tc 7h 7s 5s` | 1 |
| 6 | `As Kd Kh Ts 7s` | 1 |
| 7 | `Ad Kd Qs 9c 7h` | 1 |
| 8 | `Kh Jd 9s 8h 7c` | 1 |

Hands 9–19 untouched. 32 minutes of local compute spent, 210 s a run.

**Nine hands with one batch each is not a measurement.** One batch is the
resolution the whole exercise exists to distrust; these files are a resumption
point, not a result.

## Restarting

`C:/TMP/ladder20_local.py`, unchanged. It skips whatever is already in
`D:/ofc_data/t0_rung/ladder20_local/` by filename, so a restart continues from
run ten. Configuration as launched:

```
20 hands x 12 batches x measured top-10 x 1,024 particles = 240 runs
seeds 150,000,000 + hand*100,000 + batch*1,000
batch-major: every hand gains a batch before any hand gains a second
1 run ~210 s  ->  one batch across 20 hands ~70 min, twelve batches ~14 h
```

Output is written in the shape
`src/ofc_regular/analyze_hu_t0_ladder20_particle_noise.py` reads, so the
analysis needs nothing new.

## How many batches are worth running

Projected from the 0.46 gap scatter measured on 2026-08-20, against the twenty
openings' single-batch gaps:

| batches | hours | resolves a gap of | hands decided |
|---|---|---|---|
| 1 | 1.2 | 0.92 | 8/20 |
| 3 | 3.5 | 0.53 | 12/20 |
| 4 | 4.7 | 0.46 | 16/20 |
| 6 | 7.0 | 0.38 | 16/20 |
| 12 | 14.0 | 0.27 | 17/20 |

The curve flattens after four, and on the face of it four batches buys almost
everything twelve does. Two things argue against reading it that way.

**The gaps in that table are inflated.** They come from single 1,024-particle
batches, and the one hand measured properly had its gap fall from +0.623 to
+0.154 — four times over. The true gaps are smaller, so the true "hands decided"
column is worse than shown, by an unknown amount that only more batches can
establish.

**Four batches has already been caught lying.** In the seed-spacing experiment
of 2026-08-20 the first arm of four produced a 95% interval of [+0.204, +0.455]
for a quantity whose twelve-batch estimate is +0.154 — the interval excluded the
answer. With four samples the standard deviation estimate is itself unreliable,
so the interval built on it cannot be trusted. Twelve gives a usable variance,
not merely a smaller one.

The batch-major ordering exists so this does not have to be decided in advance:
stop after any completed batch and every hand has the same number.

## Context

* `docs/t0_particle_noise_20260820.md` — why the ladder is needed at all: a
  single 1,024 batch resolves a gap only above ~0.9, so 15 of 24 hands solved on
  2026-08-19 were never actually ranked, and the 0.107 regret figure is roughly
  four times its true value.
* The GCS attempt at the same measurement produced 0 of 240 positions for about
  $22 — per-root wall time exceeded the shard watchdog because the cost model
  was taken from a local timing on a differently-built engine and never checked
  on the fleet machine. Its m7v7 staging and worker plan remain valid and
  reusable; what a fleet run needs first is a ~$1 single-root calibration.
