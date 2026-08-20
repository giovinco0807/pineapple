# Moving the T0 playout to the GPU — brief for whoever implements it

2026-08-20. A handoff, not a design. **The first deliverable is a profile, not a
port**, because the one measurement that exists points two ways at once and the
port is only worth doing if the profile says so.

## What the work is

`evaluate_t0` scores a set of candidate openings by playing each one out against
sampled worlds. One *particle-evaluation* is one candidate action against one
sampled world: place the opening, then run T1 through T4 for both seats under
the pinned policies, then score the final boards.

The entry point is `evaluate_t0` in `rust/hu_m3_engine/src/search.rs`; the
playout itself is `rollout_t0_first` (around line 3119 in the worktree copy),
which walks the streets in order calling `locked_t0_second_action`,
`locked_t1_first_action`, `locked_t1_second_action`, `locked_t2_*`, `locked_t3_*`
and finally scoring. Every one of those calls generates the street's legal
actions, builds a feature vector per action, runs the policy net, and takes the
argmax.

Measured cost, on a 16-thread desktop:

```
one particle-evaluation           ~170 ms of core time
a full hand solve (32 candidates x 1,024 particles + 232x64 sieve)  ~8,100 core-s
```

That is the number the whole cost model rests on, and it is why 134,459
openings projects to roughly $31,000 on GCP Spot.

## The measurement that motivates this, and why it is not enough

`kind: "model_scores"` runs the T0 first-seat policy over all 232 legal openings
without any playout: generate actions, build features, forward pass, sort.

```
232 actions        113.6 ms
per action         490 us
```

Extrapolating naively — a playout touches maybe eight streets of maybe sixty
actions each — gives 480 actions x 490 us = 235 ms against a measured 170 ms,
i.e. **138% of the budget**. The estimate exceeds the thing it is estimating, so
it is wrong somewhere. What it does establish is that the per-action policy work
is the same order as the whole playout, not a rounding error.

**But `t4_model.rs` says the forward pass is not where that 490 us goes.** The
comment above `dot()` records that splitting the accumulator to let it vectorise
took the network "from 51 us to single digits per action". A model of 32,349
float32 parameters is about 32k multiply-accumulates; microseconds is the right
order. So of the 490 us per action, the matrix arithmetic is single-digit
microseconds and **roughly 480 us is feature generation, action generation and
board bookkeeping**.

That inverts the case for a GPU. Dense small matmuls batched across particles
are exactly what a GPU is for; per-action card-logic that fills a feature array
with lookups and branches is exactly what it is not.

**So: profile first.** Nobody has separated these costs, and the two readings
above disagree about which one matters.

## Task 1 — the profile (half a day, decides everything after it)

Instrument one full `evaluate_t0` call and report wall-clock share for:

1. legal action generation per street (`generate_turn_actions_trusted`,
   `generate_initial_actions`)
2. feature construction (`t3_features`, `t3first_features`, `t4_features`,
   `fast_features` — note `FreeOutlookCache`, which suggests some of this is
   already memoised)
3. the forward pass itself (`Model::evaluate` in `t4_model.rs`)
4. terminal scoring (`compact_scoring`, `scoring`)
5. everything else

A sampling profiler on the existing binary is enough; there is no need to change
the code to get this. `D:/ofc-build-cache/par-experiment/release/ofc_hu_m3_engine.dll`
is a debuggable release build already in use.

**Decision rule.** If (3) is under 20% of the total, a GPU port of the forward
pass cannot beat 1.25x by Amdahl and should not be attempted; the lever is
elsewhere (see "cheaper levers"). If (2)+(3) together are over 60% and (2) is
mostly arithmetic over fixed-size arrays rather than branchy lookups, the port
is worth designing.

## Task 2 — the port, if the profile justifies it

The shape is **lockstep particles**. Today one particle is played to the end
before the next begins, so each policy call scores ~60 actions — far too small a
batch for a GPU to be worth the transfer. Restructured:

```
for each street:
    for all P particles simultaneously:
        generate legal actions        (CPU, per particle)
        build features                (CPU or GPU)
        one batched forward pass      (GPU: P x ~60 rows)
        argmax per particle           (GPU)
    apply the chosen action per particle   (CPU)
```

With P = 1,024 a street's batch is ~61,000 rows of ~500 features. That is a real
GPU batch. Notes for whoever builds it:

* **The playout is sequential across streets and cannot be parallelised within
  one particle.** All the width comes from particles. This is why the current
  code, which parallelises over *candidates* (`par_chunks` in `search.rs` around
  line 3364), tops out at the candidate count — three candidates use three
  threads no matter how many are offered. That limitation is documented and was
  measured this session: a 3-candidate restricted run used 2.1 of 16 cores.
* **Determinism is a hard requirement, not a nicety.** Every label in this
  project is reproducible from its seed, and the corpus provenance depends on
  it. A GPU port must reproduce the CPU result exactly, or the change is a new
  engine version with all the re-pinning that implies. Note `dot()` already
  documents that its summation order differs from PyTorch's and that the parity
  fixture compares against a bound rather than bit equality — read that fixture
  before assuming what "exactly" means here.
* **The models are tiny and there are many of them**: t0_second, t1_first,
  t1_second, t2_first, t2_second, t3_first, t3_second, t4, plus five `fast_*`
  variants. All are `Model` from `t4_model.rs` — dense layers, ReLU, a final
  linear output, with per-feature standardisation and a clamp applied first.
  They would all need to live on the device.
* Available hardware is an RTX 2060 SUPER (8 GB). Weights are ~130 KB each, so
  residency is not the constraint; transfer latency per street is.

## Cheaper levers, for comparison

The port competes with these, not with doing nothing:

* **Candidate elimination** — drop candidates more than four standard errors
  behind the leader, re-measuring only survivors. Implemented and running this
  session; simulated at 40-55% of the fixed twelve-batch schedule. This is
  already banked.
* **More CPU** — the workload scales linearly with cores and Spot CPU is
  cheap. If the goal is throughput rather than latency, more VMs is a solved
  problem and a GPU port is not.
* **Fewer labels** — the open question is how many of the 134,459 openings a
  model actually needs; the project's own T1 evidence had the learning curve
  flat by 18,000 roots. If the answer is 10,000, the $31,000 figure is moot and
  so is the port.

## Ground rules

* **Do not rebuild or replace `rust/hu_m3_engine/target/release/ofc_hu_m3_engine.dll`.**
  The production trainer loads it. Experiments build to
  `D:/ofc-build-cache/par-experiment/` and load via `load_native_engine(path=...)`.
* Work in a git worktree, not the main checkout — `C:/TMP/par-exp-wt` is the one
  in use for the current engine experiments and holds unmerged changes
  (`restrict_action_keys`, a T0 parallel patch). Coordinate before touching it.
* The pinned model digests are part of every plan's identity. Changing what the
  engine computes changes the labels; changing how fast it computes them must
  not.

## Reference numbers

| | |
|---|---|
| particle-evaluation | ~170 ms core |
| full hand solve (32 cand x 1,024p + sieve) | ~8,100 core-s |
| `model_scores`, 232 actions | 113.6 ms (490 us/action) |
| forward pass alone, per `t4_model.rs` | single-digit us |
| t0_first model | 129,396 bytes = 32,349 float32 |
| GCP cost multiplier over local | >=2.6x, unmeasured beyond that lower bound |
| candidate parallelism ceiling | = candidate count (`par_chunks`) |
| full table at 134,459 openings | ~$31,000 GCP Spot, elimination method |

Context for the cost figures: `docs/t0_particle_noise_20260820.md` (why the
measurement needs this many particles at all) and
`docs/t0_solved_openings_20260820.md` (what the table is for).
