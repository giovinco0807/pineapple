# T1: what the relabel is for, and what it needs to cost

2026-08-16.  The T1 plans were written on 2026-08-15 at 1,024 particles over
25,000 roots a seat, all 27 actions — **$1,020** at measured rates.  This
records four measurements taken before spending it, and what they do to the
number.

Nothing has been run on the fleet.  The plans on disk are unchanged.

## 1. The premise holds: the fl_ev constant really does reorder T1

The relabel's first justification is that T1's labels bake in the June constant
10.227 where production scores at 9.6.  There was reason to doubt it mattered:
T4 was measured **completely robust** to fl_ev ±20 %, 0 of 36 decisions moving.

Tested on the shipped corpus's own roots — same candidates, **same evaluation
seed**, so both arms draw the same particles and only the constant differs.
24 roots, 26 actions each, at 1,024 particles (up from the corpus's 256, so a
real difference is not buried in label noise):

| | |
| --- | ---: |
| same move ranked first | **19/24 (79 %)** |
| same top-3 set | 17/24 (71 %) |
| same top-5 set | 13/24 (54 %) |
| **Kendall tau** | **0.8919 ± 0.0178** |
| pairs reordered | 438 of 7,914 (5.53 %) |
| what the old pick gives up under the new constant | mean 0.0356 ± 0.0402, **median 0.0000**, max 0.4596 |

**One root in five gets a different best move.**  The premise is real.

The mechanism explains why T1 differs from T4.  The constant shifts a
candidate's value by `0.627 × P(entering Fantasyland)`, so a shift common to
every candidate cannot reorder anything — only the **spread** can.  Here the
mean within-position spread is **1.1253**, max 1.6880.  T1 is the street where
FL entry is least settled, so candidates differ most in it; by T4 entry is
nearly determined and every candidate moves together.

Worth carrying: the loss distribution is skewed.  The **median cost is exactly
zero** — most roots are unaffected — and the mean is made by a few large ones.

## 2. Particles: 1,024 was bought, and the evidence says 256 is enough

The T1 probe's own paired verdict (30 roots a seat, reference 4,096) put
**256-vs-1024 across zero at both seats**; 1,024 was a purchase, recorded as
such.  Since then, T3 measured what label precision is worth to a model at all:
**adding σ = 0.4 of noise to the labels moved held-out regret by +0.00017** —
nothing.  A model that cannot see 0.4 of noise cannot see the difference
between 256 and 1,024 particles.

Cost is superlinear in particles here (measured: 256p costs 383/309 core-s a
root, 1,024p costs 3,300/1,660), so dropping the rung saves **7.2×**, not 4×.

## 3. Roots: 25,000 is above the knee

The cascade's own note, written when T3 was sized: the generation-1 scaling
curve "flattens well below the 25,000 that generation spent", which is why T3
was built at 18,000.  The same argument applies here — another 1.4×.

## 4. Narrowing: top-5 to top-10 costs nothing measurable, top-3 breaks

Measured on the T1 corpus itself by masking training to the top K actions a
position while the exam stays the full fan:

| kept | held regret | top-1 | training rows |
| --- | ---: | ---: | ---: |
| all 27 | 0.16632 | 0.6904 | 100 % |
| top-10 | 0.16565 | 0.6899 | 39 % |
| top-8 | 0.17196 | 0.6899 | 31 % |
| top-5 | 0.17352 | 0.6898 | 19 % |
| **top-3** | **0.21074 (+26 %)** | 0.6735 | 12 % |

Ranked by the **label**, which is the optimistic case: a real narrowing ranks
by the incumbent model and drops the true best sometimes (the trainer doc
measured the search's best surviving the model's top-5 in 91.7 % of hands).
So the numbers above are a lower bound on what narrowing costs.

**top-5, top-8 and top-10 cannot be told apart at the seed counts run so far**
— the paired intervals are wider than the gaps between them, and one arm at
n=2 produced a spuriously tight interval that should not be read.

## The bill

Two axes, priced separately, because an earlier version of this table quoted
$59 for "all 27" while silently assuming the clone continuations as well:

| configuration | continuations | cost |
| --- | --- | ---: |
| the plans as written (1,024p, 25k, all 27) | full | **$1,020** |
| 256p, 25k, all 27 | full | $142 |
| 256p, 18k, all 27 | full | $102 |
| **256p, 18k, all 27** | **fast_t2 clones** | **$59** |
| 256p, 18k, top-10 | full | $39 |
| 256p, 18k, top-10 | clones | $23 |

The clone continuations are priced in because the chain A/B that cleared them
was run **at 256 particles** — the rung this configuration uses — so the caveat
that held them back on 2026-08-15 ("the null was measured at 256 and the plans
run at 1,024") no longer applies.

### Narrowing is not available at T1 anyway, and the mechanism to want is a
### different one

The worker gates `prefilter_samples` / `prefilter_keep` to **T0 only**, and
never reads `learned_prefilter_keep` at all — that field exists in the engine
and was what the ablation above simulated, but no plan can reach it.  Narrowing
T1 therefore needs a change to the label worker, which every corpus in this
project depends on.

And if that change is made, the thing to wire is **not** the model-ranked
narrowing measured above.  `prefilter_keep` narrows with a cheap pass of the
**search itself**, so it barely conditions the corpus on the incumbent model,
and it is already validated: exact-best survival 19/20, and the two-stage
versus single-stage disagreement (+0.232) sits below the single-stage teacher's
own two-seed floor (+0.352).

**Chosen: $59 — 256p, 18,000 roots a seat, all 27 actions, clone
continuations** (owner, 2026-08-16).  Everything cut is something measured not
to matter; nothing cut trades away a best move.

## 5. Truncating the rollout at a model's value: measured, and it is not free

The owner asked why the teacher plays every street out when a model could score
the position instead — "T3先攻の配置をT3後攻モデルが評価する" — which would
finish a node in a few dozen forward passes.

**Most of that is already the design.** Inside a T1 rollout nothing below T1 is
searched: T2 both seats, T3 both seats and the first seat's T4 are each one
forward pass of a learned model. Only the terminal scoring is real, and that is
where `fl_ev` enters — which is the whole point of this relabel.

The one place the engine still offers a choice is the T4 leaf, and only at T3.
At T1 it refuses outright:

    T1 evaluation requires learned_t4_model; this street's teacher exists only
    with every learned evaluator pinned

So the substitution was measured one street shallower, on 12 T3 second-seat
roots at 1,024 particles, `learned_t4_model` pinned versus unpinned, sharing
the root, the candidate set and the evaluation seed:

| | model leaf | exact leaf |
| --- | ---: | ---: |
| same best move | **9 / 12** | — |
| Kendall tau | **0.9461** (min 0.8265) | — |
| discordant pairs | **34 / 1,339** (2.5 %) | — |
| regret of the model leaf's pick | **0.0271** mean, 0.2166 max | — |
| core-seconds | 82 | 5,417 (**66×**) |

On the three roots it misses, the model leaf's pick is **third** under the exact
leaf, not second.

**A single root said the opposite and was wrong.** The smoke root returned tau
1.0000, 0/148 discordant, regret 0.0000, and a candidate spread of 0.0996 —
from which the reading "every candidate moves by the same amount, so nothing can
reorder" is very tempting. Over 12 roots the spread is **0.6157** (max 1.1824)
and the order does move. One root is not a measurement of a rank statistic.

Set against the two numbers already on record for a T3 position, all three rungs
of the ladder are now known:

| leaf | regret | cost |
| --- | ---: | --- |
| played out and scored (what the T3 corpus does) | 0.0064 | baseline |
| **T4 chosen by a model** (what the T1 run does) | **0.0271** | 66× cheaper |
| T3 replaced by a model's value (the proposal) | 0.1002 | cheaper still |

The bases differ — 0.0064 and 0.1002 are against the 8,192-particle referee,
0.0271 is against the exact leaf at 1,024 — so the ladder is ordinal, not a
ratio scale. The ordering is not in doubt.

Two readings follow, and they point the same way. Substituting a model to
**choose a move** deep in the tree is worth its price. Substituting a model to
**return a value** is roughly four times worse again, and it is worse for a
reason the numbers show directly: a chooser's error is close to common across
candidates and mostly cancels in the comparison, while a value's error varies
per candidate, which is precisely what reorders them.

The 0.0271 is also the pessimistic end for the run in flight. At T3 the model
leaf is the very next decision; at T1 it is three streets down and averaged over
256 particles.

## Status

**Second seat running** (2026-08-16): plan `365d23d8…`, 256 particles, 18,000
roots, 116 shards, all 27 actions, clone continuations. First-seat run plan
generated at `run_plan_m7v6-t1first-18k-256p.json`, **not staged and not
launched** — that is a separate charge and needs the owner's word.

A shard is **26 minutes** wall including boot (insert 21:46 UTC → receipt 22:12
UTC), and the region caps at 31 concurrent c4-standard-8 (248 vCPU), so 116
shards is four waves and about 1 h 45 min of fleet time.

It took far longer than that, for two reasons worth keeping:

* **`--only-missing` skips shards whose INSTANCE EXISTS, not shards whose WORK
  IS DONE.** The chain script deleted terminated instances before relaunching —
  to avoid the disk billing recorded after the 133-instance cleanup — and every
  relaunch then re-issued the shards that had just finished. The workers found
  `0 positions to generate`, collided on the create-only `complete.json`, exited
  1, and so never reached their own shutdown line: 31 VMs idled 90 minutes. A
  finished instance must be **stopped, never deleted**, until the run completes.
* **The watch called gcloud with a space inside an argument.** gcloud lives
  under a path containing a space and git-bash hands a `.cmd` its arguments
  through cmd.exe, which re-parses the whole line; `--filter="name~X AND
  status=RUNNING"` breaks the quoting around the interpreter path, fails to
  stderr, and returns **empty stdout**. The watch read that as "0 running" and
  topped up on every pass. Filter locally; keep every gcloud argument
  space-free.

Both failures returned a plausible wrong number rather than an error, which is
why counting shards and instances did not catch either. The replacement watch
(`drive_t1_fleet.sh`) defines a stall in **time since the last receipt** and
stops any instance still running for an already-published shard.
