# T0 ladder for the twenty akq openings — closed 2026-08-20

Two schedules were run against the same twenty openings. The second replaced the
first, and both are on disk. This is where they left off and what would restart
them.

## What is decided

The lookup table (`hu_t0_solved_openings_v1`) now reads every batch from every
run and reports **19 of 24 openings settled**, up from 11 when it was first
built. `settled` there means the first-to-second gap excludes zero at 95%.

```
python -m ofc_regular.hu_t0_solved_openings_v1 --list
```

The deepest entry, `Ac Kd Qh 8c 7d` (canonical form of `Ah Kc Qs 8h 7c`), carries
132,096 particles. The five still unresolved are separated by 0.11 to 0.43
points and rest on 1,024 to 57,344 particles apiece.

Note the table's bar and the elimination run's bar are different on purpose:
the table asks whether a ranking may be *claimed* (95%, ~2 sigma), the
elimination asks whether measuring may *stop* (4 sigma). A hand can be settled
in the table and still open to the eliminator.

## The two schedules

**Fixed** (`C:/TMP/ladder20_worker.py`, 155 batches on disk). Twelve
1,024-particle batches of each hand's measured top ten, with a 4-sigma cutoff
added mid-run that dropped a hand once its top two separated. Seven hands were
cut this way.

Its flaw was the top ten itself: that set was chosen by a single 1,024-particle
batch, the measurement this whole exercise concluded cannot be trusted. Four
openings had candidates surviving a 4-sigma test that the cut had already
discarded — on `Ad Kd Qs 9c 7h`, eighteen candidates were still live after one
batch. A true best sitting eleventh was invisible by construction.

**Elimination** (`C:/TMP/ladder_elim_worker.py`, 144 batches on disk). Start
from all thirty-two the sieve kept, measure, drop whatever is more than four
standard errors behind the leader, repeat with the survivors; a hand is finished
when one remains. Batches are pooled by particle count and each candidate is
compared to the leader only over batches where both were measured, so a
candidate that entered late is judged at its own precision rather than the
leader's.

Eleven hands converged to a single candidate: 0, 1, 2, 3, 4, 5, 7, 10, 13, 16,
19. Simulated cost was 40-55% of the fixed schedule, and the first three hands
finish on one batch where the fixed schedule paid twelve.

## Where it stopped, and why

Seventeen of the twenty openings converged to a single candidate. Three were
left open by choice:

| opening | particles | live | gap | sigma | needed for 4 sigma |
|---|---|---|---|---|---|
| As Kd Kh Ts 7s | 40k | 2 | +0.184 | 2.4 | 107k |
| Kh Jd 9s 8h 7c | 63k | 2 | +0.109 | 1.8 | 303k |
| Kc Ks 7h 5d 4s | 17k | 5 | +0.184 | 1.6 | 107k |

They are separated by 0.11 to 0.18 points, which is less than the model's own
error on the two openings it gets wrong (0.218 and 0.283). Deciding them would
cost hours to establish which of two actions worth the same within a fifth of a
point is better, and the table records them as `unresolved`, which says exactly
that. **"Below what this instrument resolves" is a result, not a failure to
reach one.**

An estimate to distrust, from this run: `As Kd Kh Ts 7s` reported a gap of
+0.038 at 26k particles, +0.240 at 36k, and +0.184 at 40k. `particles_needed`
moved with it -- 2,529k, then 63k, then 107k. The requirement goes as the
inverse square of the gap, so a gap estimated from a few standard errors carries
a requirement that can move by a factor of forty. Projections made at 2 sigma
are not worth quoting, and one made here was retracted twice.

## A scheduling flaw, found and fixed mid-run

The first version of the eliminator ran each hand to completion before starting
the next. Two workers reached openings needing millions of particles and stalled
there permanently while every hand queued behind them waited -- two of those
already past four sigma and needing nothing but a convergence check -- and the
third worker, having finished its list, sat idle. Five of sixteen cores doing
nothing.

The fix is one batch per open hand per pass, which the fixed schedule had and
the adaptive one failed to inherit. See `docs/t0_sequential_elimination_20260820.md`.

## What the model looks like against the decided hands

Of the seventeen converged openings the shipped T0 first-seat policy picks the
measured best on fifteen. Mean cost 0.029 points a hand.

**The two misses are the two that took the most particles to decide.**

| opening | particles to converge | gap | model rank of the best | cost |
|---|---|---|---|---|
| `Ah Kc Qs 8h 7c` | 81k | 0.218 | 3 of 232 | 0.218 |
| `Ac Qc 7h 3d 3s` | 45k | 0.283 | 2 of 232 | 0.283 |

Everything it got right converged in 3k to 35k. That is the shape the earlier
readings kept half-showing and this one states: **the policy is reliable exactly
where the answer is obvious and unreliable exactly where it is not.** An earlier
version of this file said the correlation was not supported; it was, at 45k
particles and above, and the sample then was too small and too easy to see it.

The number remains biased toward easy hands -- the three left open are the
tightest of the twenty -- but the direction is now measured rather than
suspected.

## Restarting

Both workers skip whatever is already on disk. `ladder_elim_worker.py` reads
`akq_local` (32 candidates, one batch), `ladder20_local`, `ladder20_elim` and —
for hand 0 only — `local4096/hand0_settled`, and pools them by particle count.
Fix the scheduling first; the rest is unchanged.

Related: `docs/t0_particle_noise_20260820.md` for why the particle counts are
what they are, `docs/t0_solved_openings_20260820.md` for the table's contract.
