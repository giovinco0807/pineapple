# FL14 lap three: a T3 teacher on the boards the chain reaches

2026-08-14 (JST).  The T3 street had no valid teacher.  This builds one from
positions the lap-two chain actually plays into, labels it with the exact
solver, and measures what a net trained on it is worth against the solver it
would replace.

Nothing served was changed.  `lap2` remains the control.

---

## Why the street had nothing

Two artifacts looked like a T3 teacher and neither was one.

* `D:/ofc_data/fl14_t3_onpolicy_model` was trained at 21:11 on 2026-08-12.
  The action-enumeration fix landed at 22:49 the same day (`d3cca6c`), so every
  label under it is a maximum over 57% of the legal moves.
* `D:/ofc_data/lap2_t3_teacher` was the rebuild.  Its log stops at
  `[2000/10000]` and its output directory is empty: `teach` collected
  everything and wrote once, so a process that died took the run with it.  The
  chunked writer (`ecf8d9a`) landed four hours later.

## What was built

| | |
| --- | --- |
| deals | seed `0xC0FFEE21`, offset 200,000-229,999, 30,000 hands |
| played by | `lap2_t0/t1/t2` = 96 / 96 / 104 dims, confirmed at load |
| labelled by | `fl_solver teach --opponents 240` |
| corpus | `D:/ofc_data/lap3_t3_teacher/t3_labels_30k.jsonl` (91 MB) |
| harvest | 59,925 T4 decisions from the same pass |

Sharded at 5,000 in both stages: `--play-roots` accumulates memory across
chunks, and `teach` opens its output with `File::create` and cannot resume.

### The corpus is what it claims to be

* 30,000 records, **30,000 distinct deal ids** — no ordinal collisions across
  the six shards;
* **all 531,865 actions carry `t4_draws = 9880`**, so hero's last draw is
  enumerated, not sampled, at every action of every root;
* 13 board shapes.  `1-4-4` 24.9%, `2-3-4` 23.6%, `2-4-3` 18.8% — the shape
  the old dealt teacher gave 100% of its roots is under a fifth here;
* mean 17.7 actions a root, max 21 — the corrected enumeration;
* mean best-action value -5.239, against -5.195 for lap two's T3 expected
  value in the paired 10,000-hand run.  Different measurement, same street.

**26.05% of legal T3 actions foul on all 9,880 draws.**  A quarter of what the
model must rank apart is already dead.

## The floor

A second labelling pass over the 2,977 test roots at `--stream-offset
1300000` — chosen because the pool has 200,000 entries and `draw` walks it
modulo that, so an offset that is a multiple of the pool size reproduces the
first pass bit for bit, which is how lap one measured a floor of exactly zero.

**floor 0.000385**, 95.2% of roots identical between the two opponent streams,
p99 0.0096, max 0.104.

## The model

110-dimensional v2 features.  Six seeds, two widths, selected on dev regret
with test unread:

| hidden | 20260814 | 20260815 | 20260816 | mean |
| --- | ---: | ---: | ---: | ---: |
| 256,128,64 | 0.01379 | 0.01471 | 0.01280 | 0.01377 |
| **512,256,128** | 0.01362 | 0.01250 | **0.01213** | **0.01275** |

The wider model won 3/3 seeds.  Seed `20260816` was frozen and then measured
once on test:

**regret 0.01312, gap over floor 0.01274, top-1 86.7%**, p99 0.362, max 1.66.

By jokers on the board: 0 jokers 0.01076 (1,857 roots), 1 joker 0.01583
(1,014), 2 jokers 0.02871 (106).  The two-joker layer is 2.7x the joker-free
one — the same signature earlier models showed at 4.4x, and here it is worth
0.029 of a point.

## What it is worth against what it would replace

All three judged by the same 240-opponent teacher on the same 2,977 test roots:

| server | seconds a root | regret | top-1 |
| --- | ---: | ---: | ---: |
| exact, 240 opponents (second stream) | 0.10 | 0.00046 | 96.2% |
| exact, 60 opponents | **0.02** | 0.00116 | 92.3% |
| **the net** | ~0 | 0.01312 | 86.7% |

The cheapest exact server is still **11x more accurate** than the net, and it
costs 20 ms.  This does not make the net useless — a search that asks the
street thousands of times cannot pay 20 ms — but it settles what the artifact
is for.  It is not a replacement for solving T3.

## Against the model it replaces, on one measurement

The old model was scored on a corpus whose action sets were incomplete, so its
published 0.0905 says nothing here.  Encoding this corpus at 104 dims as well
puts all three on the same 2,977 test roots against the same teacher, and
separates what the corpus bought from what the features did:

| model | trained on | dims | regret | top-1 | p99 | max |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `fl14_t3_onpolicy_model` | pre-fix 10,000 roots | 104 | 0.5924 | 63.2% | 7.14 | 12.37 |
| lap3 | this corpus | 104 | 0.01745 | 83.8% | 0.379 | 2.53 |
| **lap3** | this corpus | **110** | **0.01312** | **86.7%** | 0.362 | 1.66 |

**34x of the 45x is the corpus** — corrected enumeration, played roots, three
times as many.  The v2 allocation-rank features are the remaining 1.33x, and
they won 3/3 seeds on dev at 104 dims too (mean 0.01513 against 0.01275).

The old model's tail is the part that matters: it throws away 7.1 points at the
99th percentile on positions the chain actually reaches.

## Measurement fixes made here

`eval_fl14_regret --floor` joined its two passes on `record["root"]`, the
labeler's worker-local ordinal.  Six concatenated shards reuse each ordinal six
times, so the merge fused unrelated hands and the floor pass compared **1 root
of 2,977**.  It now uses `stable_root_id`, the same played-deal `id` the
encoder splits and groups on.  The first floor it produced under the old key
was 0.00134 from a single root; the real one is 0.000385 over all of them.

## Timings, measured not estimated

| stage | rate | 30,000 roots |
| --- | ---: | ---: |
| play T0-T2 | 0.043 s/deal | 21 min |
| teach T3, 240 opponents | 0.097 s/root | 48 min |
| encode, 110 dims | 1,685 rows/s | 5 min |
| train, one run | — | 1 min 45 s |

The published figures this work was planned against — 0.44-0.82 s a deal and
0.61-0.77 s a root — predate the row memo and the `JOKER_EVAL_CACHE` insert
(both 2026-08-13).  Play is 10-17x faster and labelling 6-7x.

## Artifacts

* corpus `D:/ofc_data/lap3_t3_teacher/t3_labels_30k.jsonl`, roots
  `roots_30k.jsonl`, deals `deals_30k.jsonl`
* floor pass `teach_floor_s1300000/`, 60-opponent pass `teach_exact60/`
* encoded `encoded_110/` (fit 425,811 / dev 53,204 / test 52,850 rows)
* ablation `ablation/h{25612864,512256128}_s2026081{4,5,6}/`, 104-dim control
  `encoded_104/` and `ablation104/h512256128_s2026081{4,5,6}/`
* runtime `lap3_t3_model_110_wide/evaluator.bin`,
  sha256 `8564461b66d89af483851e857d537a3ba7fe13e475e6d4b528b26cb055653f2b`

## What this does not establish

* Nothing was promoted.  The chain that plays hands is unchanged.
* No whole-hand number moved.  Every figure here is regret against a teacher;
  the chain's score against a Fantasyland opponent is still lap two's -4.997.
* A T3 net has no consumer in the chain as it stands.  T1 and T0 truncate at
  depth 0 into the T2 net and never reach a T3 chooser; putting one at the T2
  leaf was measured slower than enumerating.  The open question this artifact
  could answer is whether T1 labels improve when their playout can price a T3
  position cheaply, which is the truncation experiment lap three's contract
  lists as its second stop/go.
