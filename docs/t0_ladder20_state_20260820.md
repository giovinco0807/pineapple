# T0 ladder for the twenty akq openings — state at 2026-08-20 (stopped)

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

Nine hands remained, in three clear tiers:

| opening | particles | live | gap | sigma | needed for 4 sigma |
|---|---|---|---|---|---|
| Qd Jc 3c 3h 2h | 3k | 1 | +1.187 | 4.3 | 3k — **already there** |
| Kc Qs 7s 3c 3d | 10k | 1 | +0.631 | 4.2 | 9k — **already there** |
| Qc 9d 7s 3h 2s | 9k | 2 | +0.617 | 3.9 | 9k |
| Qs 6s 5d 3c 3h | 13k | 5 | +0.438 | 3.3 | 19k |
| Qh Js 9h 3h 2c | 13k | 2 | +0.438 | 3.3 | 19k |
| Ac Qc 7h 3d 3s | 13k | 2 | +0.393 | 3.0 | 23k |
| Kc Ks 7h 5d 4s | 13k | 16 | +0.156 | 1.2 | 147k |
| Kh Jd 9s 8h 7c | 47k | 2 | +0.047 | 0.7 | 1,642k |
| As Kd Kh Ts 7s | 26k | 2 | +0.038 | 0.4 | 2,529k |

**A scheduling flaw stopped the top two from being recorded.** Each worker
finishes one hand before starting the next, so the two workers that reached
`As Kd Kh Ts 7s` and `Kh Jd 9s 8h 7c` — the two that need millions of particles
— stalled there forever, and every hand queued behind them waited. The third
worker had finished its whole list and sat idle: five of sixteen cores doing
nothing while two hands that needed only a convergence check went unvisited.

**Fix before restarting: advance every open hand by one batch per pass**, the
same batch-major ordering the fixed schedule used and the adaptive one failed to
inherit. Then a hand that cannot converge costs one batch a pass instead of
blocking a queue.

## The bottom three are not worth finishing

`Kc Ks 7h 5d 4s`, `Kh Jd 9s 8h 7c` and `As Kd Kh Ts 7s` need 147k, 1,642k and
2,529k particles for a 4-sigma verdict. At 170 ms of core time per particle-
evaluation the last two are weeks of local compute or about $1,000 of GCP each.

They are separated by 0.038 to 0.156 points. Paying that to establish which of
two actions worth the same within a twentieth of a point is better is not a
trade anyone should take. **"The difference is below what this instrument can
resolve" is itself a correct result**, and the table already records them as
`unresolved`, which says exactly that.

## What the model looks like against the decided hands

Of the eleven converged openings the shipped T0 first-seat policy picks the
measured best on ten. It misses on `Ah Kc Qs 8h 7c` — the most deeply measured
of them — where it plays the measured second and ranks the true best third of
232, costing 0.218 points.

**91% is not a number to quote yet.** Hands converge in order of how separated
they are, so the easy ones finished first and this sample is biased toward them.
The nine left are the tight ones. Ten zeros and one 0.218 also averages to
0.0198 points a hand, which will move a lot if the remaining hands go the other
way.

## Restarting

Both workers skip whatever is already on disk. `ladder_elim_worker.py` reads
`akq_local` (32 candidates, one batch), `ladder20_local`, `ladder20_elim` and —
for hand 0 only — `local4096/hand0_settled`, and pools them by particle count.
Fix the scheduling first; the rest is unchanged.

Related: `docs/t0_particle_noise_20260820.md` for why the particle counts are
what they are, `docs/t0_solved_openings_20260820.md` for the table's contract.
