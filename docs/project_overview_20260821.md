# What this project is, and how its pieces fit

2026-08-21. An orientation document: the two tracks, the engine, the thirteen
models, how labels are made, and what the measurement apparatus built this month
is for. Written for someone arriving with no context.

Depth varies by section, honestly marked. The regular track is described from
having worked in it; the joker track from reading it.

## The game, and why it is hard to solve

Open-Face Chinese Pineapple, heads-up. Each seat builds three rows — top (3
cards), middle (5), bottom (5) — and the rows must end ordered bottom ≥ middle ≥
top or the hand fouls and scores nothing. Five cards arrive at T0 and three at
each of T1–T4, of which two are placed and one discarded.

Two properties make it awkward:

**Decisions are irreversible and made under uncertainty.** A card placed cannot
move, and the cards that decide whether the placement was right are still in the
deck. So the value of an action is an expectation over futures, and the only way
to get it is to sample them.

**The branching is in the chance nodes, not the actions.** At T0 there are 232
legal placements — small. But each is evaluated against a deck of 47 unknown
cards from which 1.5 million opponent hands can be dealt, and the full future
space is about 7.4 x 10^28 worlds. No tree search reaches the bottom; the engine
samples.

## Two tracks

| | joker | regular |
|---|---|---|
| directory | `ai/` | `regular-ofc-pineapple/` |
| deck | 53+ cards, jokers | 52, no jokers |
| repo | outer git | its own git |

They are separate codebases with separate engines and separate model families,
sharing only the game's shape. Anything in memory or documentation from before
about March 2026 describes the joker track and does not transfer.

**This document is mostly about the regular track.**

## The regular track

### Layout

```
rust/hu_m3_engine/           the engine: rollouts, search, model inference
rust/ofc_stage3_feature_encoder/
rust/hu_rl_engine/           scalar full-hand environment, narrow contract
rust/fl_solver_regular/      Fantasyland machinery, opponent-in-FL family
rust/regular_fl_solver/
src/ofc_regular/             557 Python modules: label generation, training,
                             analysis, plan generators, GCP fleet control
trainer/                     FastAPI practice app, port 8093
tests/                       515 test modules
docs/                        99 canonical write-ups, dated
```

The Python is where experiments and pipelines live; the Rust is where anything
measured in the inner loop lives. The boundary is a C ABI: Python builds a
request dict, the engine returns a result dict, and every call carries the
digests of the model files it used.

### The engine

`rust/hu_m3_engine` — "deterministic heads-up regular OFC rollout and search
engine". Deterministic is the load-bearing word: every result is reproducible
from its seed, and the label corpora's provenance depends on that.

Its main entry points:

* `evaluate_t0` / `evaluate_request` — score a set of candidate actions by
  playing each out against sampled worlds. This is the teacher.
* `decide` — ask a model what it plays. One forward pass over the action fan,
  0.09 s at T0 first seat against the teacher's 20 s.
* `model_scores` — rank every legal action by model score, no playout.

A **particle** is one sampled world: the opponent's five T0 cards plus every
subsequent draw for both seats, run to the end. Candidates within a batch share
the same particles (common random numbers), so the draw's luck cancels when
comparing them — the scatter of an action's level is 0.24 where the scatter of a
gap between two is 0.13.

The playout is not a search. At each street both seats' actions are chosen by
the pinned policy for that (street, seat), the hand is played to T4, and the
final boards are scored. The teacher's strength comes from averaging many
worlds, not from looking ahead.

### The thirteen models

All are the same shape — dense layers, ReLU, linear output, with per-feature
standardisation and a clamp — loaded from `.bin` images whose sha256 is pinned
into every plan and every result.

```
t0_first    129 KB    t0_second   339 KB
t1_first    339       t1_second   339
t2_first    339       t2_second   339
t3_first    339       t3_second   339
t4          283
fast_t0_second, fast_t1_first, fast_t1_second, fast_t2_first, fast_t2_second
                    129 KB each — cheap stand-ins for inner-loop use
```

**One model per (street, seat).** There is no joint model in the runtime: the
`hu_m4_joint_model` and `hu_m43_joint_model_v*` modules exist but nothing in the
current engine or trainer references them.

The *state* is joint even though the models are not — every observation carries
the opponent's public board, so each seat's model sees what the other has built.
What is per-seat is the parameters, not the information.

**The models are policies, not evaluators.** They emit a score per action on a
points-like scale, but that scale is not trustworthy: measured over 768
candidates, correlation 0.841 with the true EV, MAE 4.11 points, and about 5
points high in the band that actually gets played. They are good at ranking and
bad at valuing. The pipeline uses them accordingly — to narrow, never to price.

### How labels are made

The models are trained on the teacher's output. The chain runs bottom-up,
because a street's teacher needs the streets below it to be settled:

```
T4  exhaustive where the fan is small enough
T3  teacher over T4
T2  teacher over T3
T1  teacher over T2
T0  teacher over T1
```

Each rung: generate positions, run the teacher over them on a GCP Spot fleet,
collect, retrain the (street, seat) model, gate it against the incumbent, and
pin the winner. The plan for a run is written once, hashed, and never edited;
`src/ofc_regular/hu_m31_label_gen_*` is the fleet machinery and
`hu_m7_*_plan_v1.py` the plan generators.

Current pinning is generation m7 at `fl_ev 9.6` — the value a 14-card
Fantasyland entry is worth, which enters every terminal score.

### The trainer

`trainer/` — a FastAPI app on port 8093, local and single-user. Deals hands,
takes the user's placement, grades it against the engine, and keeps mistakes in
SQLite. `engine_eval.py` is its bridge to the engine and holds the choice
between *decide* (the AI opponent's move) and *evaluate* (the grading teacher).

## What was built this month, and why

The T0 first-seat work of 2026-08-19 to 08-21 exists because of one discovery:
**a single 1,024-particle batch cannot rank T0 actions.** The top-two gap
scatters with SD 0.475 at that particle count, so one batch separates two
actions only when their true gap exceeds about 0.9 points. Fifteen of the
twenty-four openings then in hand had gaps below that and had never actually
been ranked, though they were quoted as if they had.

Three things came out of it.

**A rule for spending particles.** `hu_t0_sequential_elimination_v1` — measure a
batch, drop every candidate more than four standard errors behind the leader,
repeat with the survivors, stop when one remains or the gap is below what the
instrument resolves. The requirement is quadratic in the reciprocal of the gap,
so across twenty openings it ranged from 3,000 particles to 2,529,000; no fixed
schedule fits a factor of eight hundred. `hu_t0_elimination_runner_v1` runs it.

**A table of answered openings.** `hu_t0_solved_openings_v1` — keyed by canonical
suiting, so one entry answers up to twenty-four raw deals and returns the
placement in the querent's own suits. Every row carries whether its ranking is
established or merely the best known. 19 of 24 settled, deepest at 132,096
particles.

**A measurement of the policy.** On the seventeen openings settled so far the
shipped T0 first-seat model picks the measured best on fifteen, mean cost 0.029
points a hand — and the two it misses are the two that took the most particles
to decide. It is reliable where the answer is obvious and unreliable where it is
not.

## The scale problem, unresolved

T0 first seat has **134,459 canonical openings** (2,598,960 raw, collapsed by
the 24 suit permutations — Burnside). That symmetry exists only at this street:
once the first seat places, it is gone, and T0 second seat is 1,533,939 deals
per first-seat position.

Solving all 134,459 by elimination projects to roughly $31,000 of GCP Spot, on a
per-particle cost known only as a lower bound. Whether that is worth paying
depends on a number nobody has measured: **how many solved openings the model
actually needs before its learning curve flattens.** The model holds 32,349
float32 parameters against a table whose raw information content is 129 KB, so
memorising it is arithmetically impossible — it must generalise or grow. The
project's own T1 evidence had the curve flat by 18,000 roots.

About $700 of labels would buy a learning curve at 750 / 1,500 / 3,000 and
decide it. That is the open question worth answering first; see
`docs/t0_handoff_20260821.md`.

## The joker track, from reading it

`ai/` carries its own Rust solvers — `t4_first_exact`, `t3_exact_solver`,
`prob_engine`, `fl_solver`, `cfr_solver`, `backward`, `t3_generator` — and a
different model arrangement:

```
--hu-a-models    8 slots: T0bb, T0btn, T1bb, T1btn, T2bb, T2btn, T3bb, T3btn
                 T3btn is ignored; that seat is solved exactly
--hu-a-rankers   8 more, for shortlisting
--arm-a-own      3, the generation-3 own-hand chain
```

"Joint" there means feature dimensions, not the search: rankers are **207-dim
no-joint retrains served on a joint-zeroed encoding**, with 623-dim hybrids as
the alternative. The 207-dim path pays 400 sampled arrangements per feature
vector to build its joint block; the hybrid samples nothing. That trade is
probably where the self-play cost lives, but it has not been profiled.

I have not worked in this track and the above is read from its source, not from
having run it.

## Operating rules that cost something to learn

**Never rebuild `rust/hu_m3_engine/target/release/ofc_hu_m3_engine.dll`.** The
trainer loads it. Experiments build to `D:/ofc-build-cache/` and load with
`load_native_engine(path=...)`.

**Before any GCP fleet: one VM, one worker, one root, to completion.** Two
fleets and about $90 were spent on configurations that could never finish a
shard, because the per-root cost came from a local timing never checked on the
fleet machine. The existing smoke test proves a worker *starts*, which is a
different question. `docs/gcp_calibration_gate_20260820.md`.

**Of the five approved zones, `europe-west4-a` has neither Private Google Access
nor Cloud NAT**, so its VMs cannot reach storage.googleapis.com and sit with a
live downloader and an empty staging directory. The first symptom is the guest
agent logging an ACS i/o timeout, which reads like telemetry noise and is not.

**Adding the repository's `src` to `sys.path` ahead of the worktree's** puts an
unpatched `ofc_regular` first and silently strips `restrict_action_keys` from
`JointExactConfig`.

**Three sigma before quoting a number.** Seven figures in the August work were
stated confidently and later retracted; every one was a two-point extrapolation
or a statistic computed before enough samples existed. The list is in
`docs/t0_handoff_20260821.md` and is worth reading as a class of mistake rather
than as seven separate ones.

## Where to start reading

```
docs/t0_handoff_20260821.md              current state, retractions, next steps
docs/t0_particle_noise_20260820.md       why the measurements need this many particles
docs/t0_sequential_elimination_20260820.md   how particles are spent
docs/t0_solved_openings_20260820.md      the table's contract
docs/gcp_calibration_gate_20260820.md    what fleet runs cost when unmeasured
docs/gpu_playout_investigation_20260820.md   where the engine's time actually goes
```
