# Three engine-version-preserving CPU wins in the T0 feature path

2026-08-20. Follow-up to `gpu_playout_investigation_20260820.md`, which asked
for a profile before any port. The profile was taken (below), it killed the
GPU case by the brief's own decision rule, and the three cheap levers it
pointed at instead are now implemented and measured. **None of the three
changes any byte the engine emits** — that is the design constraint, checked
per change, so the pinned models, the corpus provenance and every label
already generated stay exactly what they were.

## The profile the brief asked for

Task 1 ran in a Linux container on the repo's fixture stand-in weights
(every evaluator slot pinned from `tests/fixtures/`, fast_* slots included),
callgrind over one full `evaluate_t0`. Instruction share by the brief's five
buckets:

| bucket | share of instructions |
|---|---|
| feature construction (`fast_encode` inclusive) | 77.1% — of which `FreeOutlookCache::outlook` 72.7% |
| forward pass (`Model::predict_with`) | 9.1% |
| allocator (malloc/free family, self) | ~14% |
| legal action generation | 0.2% |
| terminal scoring | ~0.01% |

The brief's rule was: under 20% forward pass, a GPU port of the forward pass
cannot beat 1.25x by Amdahl and should not be attempted. 9.1% is under 20%,
and the dominant 77% is the sampled outlook simulation — card lookups and
branches, not the fixed-size arithmetic the port design needed. So the port
is dead on the brief's own terms, and the levers below are what was left.

Wall-clock corroboration on the same container: the forward pass measures
3.8–10.1 us/row against 320 us/action for `model_scores`, i.e. ~1% of the
per-action cost — the same split the `dot()` comment always claimed.

## The three changes

**1. Skip finish construction the hidden-opponent encoder drops**
(`573803d`). `fast_encode_hidden_opponent` zeroes its head-to-head columns,
so the finishes the outlook built for it — three cloned `HandValue`s per
surviving draw, up to ~190 heap allocations per action — were dropped unread
on every action of the 232-wide T0 first-seat fan. The outlook now takes a
collect flag; the survivor count the block reads is kept by its own counter,
so the block is the same bytes either way.

**2. Key the rollout action caches by card masks** (`d491406`). Every locked
T0–T3 selector computed the public observation fingerprint — canonical JSON
plus SHA-256 — once per call, only to key its action cache. `locked_t4_action`
already showed the cheaper identity; the T4 mask key became `ObservationKey`,
gained the street, and now keys all thirteen caches and the census set. The
public fingerprint stays what it was everywhere it is visible: root
fingerprints in results, RNG phase strings, the sampled T3 fallback.

**3. Carry finishes as packed ordering keys** (`3da433b`). The head-to-head
block only ever *orders* a finish's rows, and `compare_key` orders identically
to `HandValue` — the equivalence the t3first parity test pins. `Finish` now
carries three u64 keys; the 24×24 head-to-head sample compares integers
instead of walking `Vec` tie-breakers; the joint loop hands the winner's keys
straight from the arrangement views it was already reading — no lookup back
into the tables, no clone; the row memo stops retaining a `HandValue` per
distinct completion.

## Measurements

Environment: 4-core Xeon 2.10 GHz container, single-thread
(`RAYON_NUM_THREADS=1`), fixture stand-in weights. Wall numbers are medians
of 5 (`t0bench`) or 20 in-process repeats (`scores`); the learned rows are
single runs. Callgrind instruction counts are deterministic and
layout-independent — they are the per-change attribution; wall medians on
this shared container carry a demonstrated ±3–5% build-layout jitter (change
2 moved `model_scores` wall time 6% while its instruction count moved 18
instructions).

`model_scores`, T0 first seat, 232 openings (the root-fan / screening path):

| | baseline | +1 | +1+2 | +1+2+3 |
|---|---|---|---|---|
| ms/call (wall) | 74.2 | 70.4 | 66.4 | 63.1 |
| instructions | 11.709G | 11.343G | 11.343G | 10.995G |
| vs baseline (Ir) | — | −3.1% | −3.1% | **−6.1%** |

Full `evaluate_t0`, fast configuration (`ps=1 keep=4 n=8`, the in-rollout
path):

| | baseline | +1 | +1+2 | +1+2+3 |
|---|---|---|---|---|
| median s | 13.68 | 13.29 | 13.38 | **11.29** |
| vs baseline | — | −2.9% | −2.2% | **−17.5%** |

Full `evaluate_t0`, learned (full-precision) configuration
(`ps=1 keep=2 n=2`, single runs):

| | baseline | +1 | +1+2 | +1+2+3 |
|---|---|---|---|---|
| s | 128.5 | 124.5 | 124.0 | **108.2** |
| vs baseline | — | — | — | **−15.8%** |

Full `evaluate_t0`, fast configuration (`ps=1 keep=1 n=1`), instructions:

| | baseline | +1 | +1+2 | +1+2+3 |
|---|---|---|---|---|
| instructions | 72.750G | 73.146G | 72.822G | **66.888G** |
| vs baseline | — | +0.5%* | +0.1%* | **−8.1%** |

\* Cross-build codegen differences run to about ±0.5% on this workload;
change 1's target path is absent from a full solve (its attribution is the
scores table above), so these two columns are that noise, not added work.

Where change 3's instructions went, by function: the malloc/free family
collapses from ~9.7G instructions (13.3% of the program) to ~1.6G, and
`head_to_head` drops 3.35G → 2.50G. `Model::predict_with` counts an
identical 6,647,073,621 instructions before and after — the forward pass
is untouched to the instruction, which is what "same engine" should look
like in a profile.

Change 2 is the honest disappointment: the fingerprint was ~9 hashes per
playout and its absolute cost was already small, so its effect on a full
solve is inside the noise band here. It stands on removing a per-decision
JSON+SHA-256 whose cost *scales with nothing it should* — and on the byte-pin
suite proving the mask identity safe — not on a measured full-solve win.

## What proves the bytes did not move

Per change, all of:

- `evaluate_t0` output sha256 unchanged on three configurations — fast
  `ps=1 keep=4 n=8` (`1f1eae62…`), fast `ps=1 keep=1 n=1` (`d152e974…`),
  learned `ps=1 keep=2 n=2` (`29233cbf…`) — via the harness's digest mode,
  which also asserts repeats within a run stay byte-identical.
- `model_scores` output sha256 unchanged (`68c7dea8…`).
- `t0_pruning_safety` — the byte-for-byte regression pin — green, along with
  the t3/t3first/t4 parity fixtures, the fast rollout-policy suites, and
  `candidate01_semantics`.

The `examples/t0_profile.rs` harness (`e9bf822`) is committed so the same
proof can be re-run against the production weights on the desktop, which is
the one measurement this container cannot make: stand-in weights change which
actions argmax picks, so cache hit rates and fan sizes differ from a
production plan's. The structure — 8 turn decisions per playout, the 232-wide
opening fan, the bucket shares — does not.

## What was deliberately not done

- No batching, no GPU, no change to what any evaluator computes. The brief's
  Task 2 stays dead unless a production-weight profile contradicts the bucket
  shares above.
- `partial_value` still allocates per distinct completion inside the row
  memo; the memo amortizes it and the remaining share did not justify
  touching hand evaluation itself.
- The biggest lever in the investigation remains unpulled and is not an
  engine change at all: the label-count question (T1's learning curve was
  flat by 18,000 roots; if T0 needs 10–18k of its 134,459 openings, the
  $31k figure collapses by 7–13x). These wins compose with that decision,
  they do not compete with it.
