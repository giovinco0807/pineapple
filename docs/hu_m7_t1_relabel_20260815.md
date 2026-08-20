# M7 T1: the relabel at fl_ev 9.6, planned and sized

2026-08-15.  The owner directed the cascade's remaining work forward: retrain
the T1 pair (and then T0) on labels priced at the current constant.  T1's
shipped labels are 256-particle at the June constant 10.227; every street below
it is now the 9.6 generation — T4 v6, T3 v7, and since 2026-08-08 the T2 v2
pair.  This document is the live record of that street, in the same standing as
`hu_m7_cascade_20260806.md` was for the streets before it.

## Owner rulings, 2026-08-15

* **Particles: 1024, bought.**  The 60-root probe (2026-08-08,
  `/home/wner/ofc-m7/t1probe/`, reference 4096, K=10 narrowing) has paired
  256-vs-1024 including zero at both seats at n=30 — first seat −0.0000
  [−0.065, +0.065], second +0.049 [−0.009, +0.106] — with the second seat's
  mean leaning 1024.  The probe resolves ~0.05, not the house δ=0.010, so the
  rung is a purchase, exactly as T3's was, and is recorded as such in the plan
  provenance.
* **Scoring: all 27 actions.**  The K=10 narrowing (2.7× cheaper, recall
  99.3–99.7% measured) was declined: the corpus must not condition on the
  incumbent model's preferences.  Matches the T2 v2 corpus discipline.
* **Corpus: 25,000 paired roots a seat**, the T2 v2 size, directly comparable
  to the existing `t1curve` scaling points (12k/18k/25k).

## The package: m7v6

The m7v5 image carries the generation-1 T2 pair, which would make a T1 label's
continuation the street that was just replaced.  m7v6 = m7v5 + the T2 v2 pair
promoted, generation-1 images retained, same engine (`e17aa39f…`), same fl_ev
v4 config, same wheelhouse.

| | |
| --- | --- |
| package | `/home/wner/ofc-labelgen-m7v6/package` |
| ledger sha256 | `66389e97e2bec5169db1225ea988ba3730055767cfbbf0c589a21fe8d7b0fabf` |
| runtime.tar.gz | `4ce40384c7fcdb64a507c45b36273584d1c57f1930a0ffd0ed0fc1569aea2466` (70.1 MB) |
| wheelhouse.zip | `338ca072…` (unchanged from m7v5) |
| new ledger keys | `t2_first_model_v2_sha256` = `bed915308d1d…`, `t2_second_model_v2_sha256` = `3ebd614bb0ed…` |
| weights | 19 (17 + the v2 pair) |
| builder | session scratchpad `m7v6_build_package.py` over `labelgen_build_package.py` |

The T2 v2 exports were shipped from `/home/wner/ofc-m7/t2_2048/{first,second}/ship/weights/`
with `.sha256.json` sidecars written from their `.metadata.json` (digests
verified equal to the repo fixtures `t2first_model_v2.bin` / `t2_model_v2.bin`).

## The plans: `hu_m7_t1_1024_plan_v1`

`src/ofc_regular/hu_m7_t1_1024_plan_v1.py` + 12 tests, the same immutable
narrow-generator shape as the T2 street's `hu_m7_t2_2048_plan_v1`.  Audit mode
against the real package: **all 24 declared hashes recomputed, PASS**.

* street T1, both seats, 1024 particles, all 27 actions, seeds_per_position 1
* both plans pin the **T2 v2 pair** (both T2 replies are ahead of either seat)
* first seat additionally pins `t1_model_v1.bin` — the incumbent this relabel
  replaces answers the opponent's T1 reply, the same bootstrap every street
  below ran under
* **no fast pins** — full-precision replies, matching the generation-1 T1
  plans (subject to the A/B below; the validator currently refuses fast pins)
* 300 shards (83–84 roots each): T1-first ≈ 860 core-s/root at 1024p makes a
  174-shard layout run ~5.7 h/shard at 6 workers, against the 6-hour watchdog;
  84 roots ≈ 3.3 h
* job ids `m7v6-t1first-25k-1024p` / `m7v6-t1second-25k-1024p`

### Seed allocation, audited

Scanned every plan on disk (labelgen*/package, plans/, m7 run dirs) rather than
trusting the cascade doc's list.  Bases in use: 940/942/947/960/970/971M
(plans), 930/945/946/956M (doc-recorded gates, probes, T4v6), eval bases
5/6/8/9/10M, probe bases 61/63/72/73(chain A/B)/340/350/370M.

**Allocated fresh: hand block [948,000,000, 948,025,000), behavior offset
+500,000, eval base 11,000,000.**  Whole block shared by both seats to pair
their roots; no partial reuse.

## The chain A/B experiment — NULL, and full precision stands anyway

The rollouts inside a T1 label play both T2 replies.  The engine can answer
them through the full pinned pair (what generation-1 T1 plans did) or through
the distilled `fast_t2_*` clones — which imitate the T2 **v1** generation and
would halve the label cost (measured 2.05× on the smoke root).  Before the
fleet is sized, 40 roots × both seats × three arms, paired on roots and seeds
(A: full T2 v2 pins + fast_t2 v1 clones answering; B: full T2 v2 answering;
C: arm A reseeded = the noise floor), 256 particles, all 27 actions:

* scripts: session scratchpad `t1_chain_ab.py`, `run_t1_chain_ab.sh`,
  `t1_chain_ab_report.py`; output `/home/wner/ofc-m7/t1_chain_ab/`
* measurement seeds 73,000,000 base (fresh; probe spent 72M/370M)
* known limitation: both arms keep `fast_t1` pinned, which a real T1 label
  does not; constant across arms, so the A-vs-B contrast is unaffected, but
  absolute first-seat timings undershoot the label-real cost
* verdict rule: paired judge-regret (chain minus reseed-noise) — if it
  includes 0, the clones are free and the plans may adopt them for ~2× cost
  reduction; if not, full precision stands

### Result: 80/80 roots, no failures

| seat | A vs B (chain) | A vs C (reseed noise) | paired, chain − noise |
| --- | --- | --- | --- |
| first (n=40) | flips 45%, judge-regret +0.1916 ± 0.1185, MAE 0.7918 | flips 42%, +0.2366 ± 0.1220, MAE 0.7926 | **−0.0450 [−0.1336, +0.0435]** includes 0 |
| second (n=40) | flips 35%, +0.2075 ± 0.1291, MAE 0.7216 | flips 35%, +0.2437 ± 0.1237, MAE 0.7970 | **−0.0362 [−0.1439, +0.0715]** includes 0 |

Substituting the clones is, at this resolution, exactly as disruptive to a T1
label as reseeding the same chain: identical flip rates, identical MAE, and a
paired difference that leans very slightly *toward* the clones.  Cost ratio
measured under 10-way contention: full replies are 1.53× (first) and 2.04×
(second).

**The plans keep full precision regardless, and the reason is in the numbers
above.**  This ran at 256 particles for affordability; the labels will be made
at 1024.  Raising particles lowers the reseed floor but not a systematic chain
effect, so a null at 256 does not transfer to 1024 — the test's power here is
±0.09, and a systematic effect hiding under that is the same size as the
particle noise (0.04–0.05, from the rung probe) that buying 1024 exists to
remove.  Trading the purchased precision back for ~$140 is not a trade worth
making, and full precision is also what the generation-1 T1 corpus used, which
keeps the retrain's before/after comparison on one footing.

## Cost — the rehearsal broke the sizing, which is what a rehearsal is for

The plans were sized from the 2026-08-08 cost probe's slope, fitted between 32
and 128 samples and extrapolated 8× to 1024: first 0.8399, second 0.4657
core-s/sample, so 860 and 477 core-s a root.

**The real worker, same config, disagrees by 2–3×:**

| seat | probe line at 1024p | rehearsal, measured | ratio |
| --- | ---: | ---: | ---: |
| second | 479 core-s | **1,054 core-s** (position 1, startup excluded) | 2.2× |
| first | 860 core-s | **2,650 core-s** (44 m 17 s wall, 44 m 11 s user) | 3.0× |

Not a configuration difference: the worker builds `candidate_samples=8`,
`downstream_t3_samples=4`, which is exactly what the cost probe timed.  The
line is simply wrong somewhere between 128 and 1024 — the probe's own points
are 32 and 128 only, and every number above that is extrapolation.

The 12-root-a-seat confirmation (6 shards a seat, 2 positions each, 12
processes on 16 cores = the fleet's own 0.75 occupancy) settles it:

| condition | first | second | n |
| --- | ---: | ---: | ---: |
| single process, uncontended | 2,650 | 1,054 | 1 |
| **12 parallel (fleet-equivalent occupancy)** | **3,300** | **1,660** | 12 |
| probe line, for reference | 860 | 477 | — |

Per-process spread is tight (first 6,267–7,320 core-s for two roots; second
3,058–3,539), so this is a rate, not a lucky root.

**The superlinearity is confirmed by independent data.**  The chain A/B run
measured the same full-precision continuation at 256 particles, all 27
actions: 383 (first) and 309 (second) core-s a root under 10-way contention.
Against the 1024p rates that is **8.6× and 5.4× the cost for 4× the
particles.**  Whatever the mechanism — the plausible one is that the engine's
per-observation caches stop paying as the particle batch widens the set of
distinct child observations — the price of a particle at T1 rises with the
batch, and every estimate in this project that extrapolated a low-rung slope
is wrong in the same direction.

### The bill, measured

| rung | first | second | pair × 25k | fleet-h @464 | cost @ $0.0296/core-h |
| --- | ---: | ---: | ---: | ---: | ---: |
| 256p (measured) | 383 | 309 | 4,810 core-h | 10 h | **$142** |
| 512p (interpolated, NOT measured) | ~1,100 | ~700 | ~12,500 core-h | 27 h | ~$370 |
| **1024p (measured, contended)** | 3,300 | 1,660 | **34,450 core-h** | **74 h** | **$1,020** |
| 1024p (uncontended lower bound) | 2,650 | 1,054 | 25,700 core-h | 55 h | $760 |

Against the **$275** the rung was chosen at.  And T0 is next: the cascade's
own per-street factor is ~3.5× per sample, so a T0 relabel at this rung
projects to **$2,700–3,600**.  The superlinearity compounds down the ladder,
which is why the rung is worth re-deciding once rather than absorbing twice.

### Consequences already visible

* **The plan provenance's `sizing_reason` is superseded by measurement.**  The
  plans are write-once evidence and stay as written; the run plan carries the
  real sizing.  A first-seat shard of 84 roots at 6 workers is ~10 h at the
  measured rate, past the 6-hour watchdog, so the fleet must split shards with
  the worker's `--stride-count` (which is what that flag is for, and why the
  plan says `concurrency_is_not_authorized_by_this_plan`).
* **The rung purchase deserves re-deciding at the true price.**  1024 was
  bought over 256 on a probe whose paired verdict *included zero at both
  seats*; the case for buying it was that precision is cheap insurance at
  ~$275.  At ~$750 that is the owner's call again, not an implementation
  detail.

## Retrain harness — reusable, checked

`m7v4_t3v7_train.py` (session 55448e7b scratchpad) is the frozen M7 Task-0
protocol with the two things this street needs: `--init` warm start that
**re-expresses the first layer in the new standardisation** so the composed
function is unchanged at load (verified numerically on real rows before any
step), and nested-subset scaling curves.  Street-agnostic apart from a
hardcoded `"street": "T3"` metadata string.  The existing `t1curve` models
(gen-1 corpus, 12k/18k/25k, arch [168,256,128,64,1], clamp 0.0, split_code
230) are the same protocol family and give a same-size before/after baseline.

## Protocol after the labels land

1. Warm-start retrain both seats from `t1first_model_v1` / `t1_model_v1`
   (house discipline: stable-hash holdout, early stop on held-out regret,
   seeds recorded; clamp **must stay 0** — the DEBT section's rule).
2. Gates per seat: held-out regret against floor, then the mirrored duplicate
   match with only that model swapped (`gate_match.py` harness), standing
   criterion lower CI > −0.05 and mean ≥ 0 on 20,004 mirrored deals.
3. Export T4M1 + sidecars → `t1first_model_v2.bin` / `t1_model_v2.bin` →
   repo fixtures + trainer `WEIGHT_FILES` pins.
4. Then T0, same protocol (probe first; its 2026-08-06 deferral was lifted by
   the owner's 2026-08-15 direction).

## Status

| step | state |
| --- | --- |
| m7v6 package | **DONE**, audited |
| plan module + tests | **DONE** (12 tests pass; audit vs real package PASS) |
| chain A/B | RUNNING (40 roots × 2 seats × 3 arms, 10 workers) |
| plans written | not yet — waits on the A/B verdict |
| rehearsal shard | not yet |
| fleet | not yet — cost approval at launch |
| retrain + gates | not yet |
