# The Fantasyland opponent: exact best response, and the pool that makes it cheap

2026-08-10 (JST). How `fl_solver_regular` models the Fantasyland side of a
hand — the rule it follows, why the answer is exact rather than sampled, and how
a pre-solved pool cut a T0 label from 130.5 seconds to 3.77 without changing a
single decision.

Two independent pieces, and the second only works because of the first:

1. **The frontier** (`src/frontier.rs`) — the Fantasyland player's best reply to
   any hero board, computed exactly, by scanning tens of rows instead of a
   million arrangements.
2. **The library** (`src/fl_library.rs`) — a pool of pre-solved Fantasyland
   hands, drawn from at each leaf instead of solved there, exact in distribution
   by conditioning.

---

## 1. The rule everything follows from

The Fantasyland player does not commit blind. They see the normal player's
progressive placement and set their 13-from-14 **after** hero's board is
complete.

So the Fantasyland side is a **best response to a concrete hero board**, not a
draw from a fixed distribution. Scoring a hero candidate against a
statically-solved Fantasyland board is optimistic for hero — it lets hero beat
an opponent who was not allowed to react.

Everything below exists to make that best response affordable, because the
naive version is a million arrangements per hero board, and a T0 position asks
for it at 16,384 leaves × 232 openings.

### The stay rule as implemented

`stays` is **trips in the top row, or quads-or-better in the bottom row**
(`frontier.rs:210`, `frontier.rs:295`). A full house in the middle is *not* a
stay condition in this codebase. If you compare numbers against a ruleset that
allows it, that is where the difference comes from.

The value of staying is the `fl_ev` constant from the objective config —
`configs/fl_ev_regular_v4_selfplay.json` carries **9.6** for a 14-card entry,
measured as a self-consistent fixed point from self-play run B (V0 = 9.669,
n = 52,303). It is priced into every frontier row, which is why a pool built
under one constant cannot be read by a run asking for another.

---

## 2. Why the frontier is exact, not an approximation

Write the Fantasyland player's score against a fixed hero board `H`:

```text
f(A, H) = g(s_top, s_mid, s_bot) + static(A)
  where s_row  = sign(row_key(A) - row_key(H))   in {-1, 0, +1}
        g(s)   = sum(s), except +/-6 when all three agree (the scoop)
        static(A) = royalty(A) + fl_ev * stays(A)
```

Three steps:

* `g` is **monotone non-decreasing in each of its three arguments**. This is
  checked over all 27 sign triples by `scoop_aware_line_is_monotone` rather than
  asserted — the scoop bonus is a discontinuity and it would be easy to be wrong
  about by hand.
* Each `s_row` is monotone non-decreasing in `row_key(A)` for fixed `H`.
* Therefore `f(·, H)` is monotone non-decreasing in all four of
  `(top_key, middle_key, bottom_key, static_value)` — **simultaneously for every
  hero board `H`**.

The consequence is the whole method: if `A'` dominates `A` in all four
components, then `f(A', H) ≥ f(A, H)` for *every* hero board, so a dominated
arrangement can never be the unique best response. The maximum is always
attained on the non-dominated frontier, and scanning that frontier is exact.

Crucially, the frontier does not depend on `H`. It is built **once per
Fantasyland hand** and answers every hero board that hand will ever face.

### What it is pinned against

`tests/frontier_exactness.rs` compares the frontier's argmax score against the
argmax over all **1,009,008** arrangements (C(14,5)·C(9,5)·C(4,3)) and requires
bit-identical values — 320 (deal, hero board) pairs in one test and 24 in
another, with hero boards chosen to actually contest the rows rather than be
trivially beaten.

### Size and cost

A 14-card hand's frontier averages **57.2 rows** (measured over the 200,000-hand
pool). So the best response is a scan of about sixty rows instead of a million
arrangements — roughly a 17,000x reduction, with no loss.

Building one costs **38.3 ms**, and the build is where the remaining cost sits:
it ranks each 3- and 5-card subset once (2,366 evaluations, indexed by position
mask) rather than three hand rankings per arrangement, which is what dominated
before.

---

## 3. The library: the same distribution, drawn instead of solved

### The problem it solves

Two thirds of a T0 position is one thing. At each of its leaves the teacher
draws `samples` Fantasyland hands from the deck hero left, solves each, and
builds each one's frontier: 5.18 ms + 38.3 ms, times 16,384 leaves — about **700
core-seconds before a single one of the 232 openings has been scored.**

None of that work depends on the opening. All 232 use the same five cards, so
the deck is the same, so the opponent's hands are the same. **This is why
narrowing the fan could not touch it**, and why cutting 232 openings to 20 was
measured at only 1.42x.

### Why drawing from a pool is exact

A uniform 14-card hand from the whole deck, *conditioned on sharing no card with
hero's seventeen*, is distributed exactly as a uniform 14-card hand from the
thirty-five hero left. **That is the definition of conditioning, not a modelling
assumption.**

So a pool drawn once from the full 52-card deck, filtered at each leaf to the
hands that fit, samples the right distribution. The filter is a bitwise AND
against a 52-bit mask.

A hand fits a given seventeen with probability C(35,14)/C(52,14) = **0.13%**,
one in 762, so a pool of 200,000 leaves about **262 candidates at each leaf** —
enough for 16 samples with a wide margin, and the reason 200,000 was the size
chosen.

### What it actually costs

**Reuse.** Independent draws give every leaf its own opponents; a pool gives
overlapping ones, so the same hand is scored at many leaves. The marginal
distribution is untouched, but the estimate carries less independent information
than its sample count suggests.

That is a variance claim, so it was measured rather than argued:

| | picks the same opening | seed-to-seed noise | worst |
| --- | --- | --- | --- |
| solve at every leaf | — | 0.4277 | 1.4868 |
| pool | 6/6 vs solving | 0.4263 | 1.4991 |

**The pool is not noisier.** Its own seed-to-seed disagreement is
indistinguishable from the solve-per-leaf shape's. Judged on the solve-per-leaf
run's own values, the pool picked the same opening on all six roots with mean
regret 0.0000. The values shifted by −0.0096 uniformly, which changes no
decision.

### Two details that are not incidental

**Random probe, not first-fit.** The pool's order is arbitrary but fixed, so
reading it from the front would let early entries serve far more leaves than
late ones. `draw` probes at random and marks entries taken.

**Short draws are an error, not a shrug.** If the pool cannot supply `want`
hands, `draw` returns fewer and the caller **fails the position**. A leaf that
quietly averages over three opponents when the plan said sixteen is the same
silent failure mode as the collision-rejection shape it replaced.

**The stream index is the pool position.** `build_range` seeds each entry by its
index in the pool, not its index in the call, so splitting the build across
threads produces the same pool as building it in one. A pool that changed with
the thread count would make a label depend on the machine that produced it.

---

## 4. On disk: `FLL1`

The pool is built once and read many times, on machines that did not build it.
Rebuilding 200,000 frontiers costs about eighteen minutes on a worker's eight
vCPU, and a worker that rebuilds is a worker not labelling — so it travels as
bytes, not as a seed to replay.

```text
magic "FLL1" | version u32 | count u64 | fl_ev f64 | seed u64
then per entry:
  mask u64 | deal [u8;14] | rows u32
  then per row:
    top_key u32 | middle_key u32 | bottom_key u32
    static_value f64 | total_royalty i32 | stays u8
```

Flat, little-endian. The reader checks the magic, the version, **and the
Fantasyland constant** — a pool built at a different `fl_ev` prices its stay term
differently and is a different pool, so loading it under the wrong constant is
refused rather than silently accepted. Trailing bytes are an error too.

The shipped pool: 200,000 entries, **291,268,107 bytes**, `fl_ev` 9.6, seed
999300000, built in **970.9 s** on 12 threads.

---

## 5. Using it

Build once:

```bash
fl_solver_regular build-library --count 200000 --seed 999300000 --out pool_200k.bin
```

Then pass it to a teacher, pinned by digest so a run cannot silently use a
different pool:

```bash
fl_solver_regular label-t0 ... \
  --fl-library pool_200k.bin \
  --fl-library-sha256 85cc2db90f6d26e85c221b14f319c190cdda07c91c874b2e7ee72b63824e41d1
```

Without `--fl-library` the teacher falls back to solving at every leaf. Both
paths are live and produce the same decisions; the pool is an optimisation, not
a different model.

### What it bought

Measured on six roots, twelve threads, samples 16 / t1,t2 draws 16 / t3,t4 draws 2:

| shape | s/root | vs baseline |
| --- | --- | --- |
| solve at every leaf, all 232 openings | 130.5 | 1x |
| pool, all 232 openings | 42.0 | 3.1x |
| pool + `--narrow-keep 40` | 9.40 | 13.9x |
| pool + `--narrow-keep 20` | **3.77** | **34.6x** |

The two savings multiply because they remove different things. The pool removes
the per-root fixed cost that all 232 openings share; only once that is gone does
cutting the fan scale with the fan. Predicting either one in isolation got it
wrong in both directions — narrowing was called at 11.6x and came in at 1.42x
against the old shape; the pool was called at 3.1x and came in at 2.3x under
load.

---

## 6. The bug this shape introduced, and its fix

`t0_teacher` materialises the whole draw tree before scoring any opening. The
first library implementation **cloned each leaf's frontiers out of the pool**.

At 64 draws either side and four below, that is 65,536 leaves × 64 samples × ~57
rows of copies: **32 GB**, and the kernel killed the process with no message at
all. The rung simply produced an empty output file — indistinguishable from a
hang, and the same silent shape as the solver-chunk stall that cost 4.25 hours.

The fix is `Cow<'a, [FrontierEntry]>`: borrowed from the pool when there is one,
owned only when the leaf solved for itself. The pool outlives the computation,
so there was never a reason to copy.

| | peak RSS | s/root |
| --- | --- | --- |
| copying | ~16 GB (32 GB at 64 samples → killed) | 94.06 |
| borrowing | **714 MB** (1.0 GB at 64 samples) | **88.66** |

Verified byte-identical output on the same rung and seeds. A c4-standard-8 has
30 GB, so this was a live way for production workers to die silently.

---

## 7. What this method does not do

* **It does not model a Fantasyland opponent who plays badly.** The frontier is
  a best response, so every number here assumes the Fantasyland side is optimal
  given hero's finished board. Against a weaker opponent hero's true values are
  higher than these.
* **It contributes no variance, and that has a consequence.** Because the
  Fantasyland side is solved exactly, raising `opponent_samples` does almost
  nothing: 16 to 32 moved a T0 teacher's seed-to-seed noise from 0.0656 to
  0.0654. The remaining variance is hero's own draw. The axis the pool made
  cheap is the axis not worth buying.
* **It says nothing about a Fantasyland opponent's *entry* decision.** `fl_ev`
  is a constant fitted elsewhere; nothing here re-derives it.
* **The pool is deck-wide on purpose.** Restricting it to a root's unseen set
  would make it a different pool for every position and give up the amortisation
  that is the whole point.

## Related

* [`t0_vsfl_stopped_20260810.md`](t0_vsfl_stopped_20260810.md) — what the T0
  teacher built on this measured, and why it was stopped.
* [`vsfl_streets_20260809.md`](vsfl_streets_20260809.md) — the three distilled
  rankers that continue T1/T2/T3 inside these teachers.
