# Oracle-First Strength Plan

## Decision

The training direction should be oracle-first:

1. Build the slowest credible labels first.
2. Audit the current fast model against those labels.
3. Distill the labels into candidate generation and final selection models.
4. Only then optimize the runtime path back under 5 seconds per turn.

The 5-second target is a product/runtime constraint.  It should not define the
teacher quality.

## Exact Scope

T3/T4 can be exact.

- `ai/rust_solver/t3_exact_solver` evaluates T3 and T4 positions by enumerating
  legal actions and future draws.
- It includes `opponent_board`, `exclude`, and `known_discards` in the remaining
  deck calculation.

T2 can be exact for the acting player's remaining T3/T4 tree when the visible
opponent board and known dead cards are fixed.

- The Rust solver has an `exact_t2` path via `turn=2`.
- `--t2-draw-limit 0` means all T3 draws are enumerated.
- Each T3 draw is solved by exact T3/T4 evaluation.

T1/T0 cannot be treated as full exact cheaply in heads-up play because future
opponent actions alter visible boards and dead cards.  The next best oracle is a
slow recursive teacher:

- enumerate all legal T1 actions;
- sample or enumerate enough T2 deals for the acting player;
- evaluate each resulting T2 state with Rust exact T2 where possible;
- include the current visible opponent board and known discards in every leaf;
- store every candidate, not just the chosen action.

## Current Runtime Finding

The latest hard-5s gate experiments are useful diagnostics but not the main
strength path.

- Wide logistic challenger gate improved hard-miss regret but damaged normal
  source/runtime guards.
- A safer rule, `logistic_gate_refined_override_delta_ge_0677`, improves hard
  misses and runtime112 without 5-second violations, but still slightly damages
  source136:
  - local80 hard misses: Top1 `0.00% -> 5.00%`, avg regret `1.4500 -> 1.2201`
  - runtime112: Top1 `52.68% -> 53.57%`, avg regret `0.6197 -> 0.6174`
  - source136: Top1 `70.59% -> 69.85%`, avg regret `0.2353 -> 0.2376`
- This is not enough to call the model strong.  It is a runtime patch, not an
  oracle.

## Next Work

1. Generate a small local T2 exact benchmark from existing T2/T3 positions using
   `ai/rust_solver/t3_exact_solver` with `turn=2` and `--t2-draw-limit 0`.
2. Build a T1 slow-oracle generator that uses T2 exact leaves.
3. Evaluate the current fast T1/T2 stack against that oracle:
   - Top1/Top3/Top5/Top10/Top15 recall
   - average regret
   - max regret
   - FL/bust metric error
4. Train or fine-tune only after the oracle miss set is available.
5. Reintroduce 5-second runtime gates after oracle recall improves.

## Local Smoke

Command smoke-tested:

```powershell
cargo run --release --manifest-path ai\rust_solver\t3_exact_solver\Cargo.toml -- --input ai\data\tutor_t2_top20_weak_20260524\teacher_labels_t2_top20_suit24_mc1000.jsonl --output ai\data\hybrid_t1t2_active_20260531\t2_exact_smoke_draw1_20260602.jsonl --limit 1 --top-n 24 --t2-draw-limit 1
```

Result:

- The existing MC teacher JSONL can be passed directly to the Rust exact solver.
- The output contained `turn=2`, `legal_actions=24`, and candidate metrics with
  `source=exact_t2_capped`.
- Solver-reported elapsed time for one capped T2 position was about `26.2s`.
- Full `--t2-draw-limit 0` is likely too slow for broad local generation before
  improving the exact generator path.

Immediate implication:

- Do not spend more time tuning 5-second gates yet.
- First optimize or batch the Rust T2 exact generator enough that a small exact
  oracle set is practical.

## T2 Oracle Harness

Added:

- `ai/tutor/run_t2_exact_oracle.py`

Purpose:

- run `ai/rust_solver/t3_exact_solver` on an existing T2 teacher JSONL;
- write exact/capped-exact candidate labels;
- compare the original MC Top1 against the exact/capped-exact Top1;
- report Top1 agreement, exact regret of the source Top1, FL/bust differences,
  solver elapsed time, and estimated full exact cost.

Smoke command:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.tutor.run_t2_exact_oracle --input ai\data\tutor_t2_top20_weak_20260524\teacher_labels_t2_top20_suit24_mc1000.jsonl --out-dir ai\data\hybrid_t1t2_active_20260531\t2_exact_oracle_smoke_20260602 --limit 1 --top-n 24 --t2-draw-limit 1 --binary ai\rust_solver\target\release\t3_exact_solver.exe
```

Smoke result:

- output: `ai/data/hybrid_t1t2_active_20260531/t2_exact_oracle_smoke_20260602/t2_oracle_cap1_limit1.jsonl`
- summary: `ai/data/hybrid_t1t2_active_20260531/t2_exact_oracle_smoke_20260602/t2_oracle_cap1_limit1.summary.md`
- records: `1`
- MC1000 Top1 vs capped-exact Top1: `1/1` matched
- capped exact elapsed: `26919.1ms`
- estimated naive full T2 exact cost from this cap: `161084023.7ms`, about `44.7h` for one position

This confirms the current exact semantics are usable, but the naive full T2
exact path is not yet a practical local teacher generator.

Next implementation target:

- keep exact semantics;
- replace naive T2 full enumeration with a faster exact dynamic-programming or
  memoized generator before attempting a broad T2 oracle set.

## Exact-Preserving Speedup Smoke

Added a Rust shortcut in `ai/rust_solver/t3_exact_solver/src/main.rs`:

- detect irreparable busts when completed adjacent rows already violate
  ordering:
  - complete top and complete middle with `top > middle`
  - complete middle and complete bottom with `middle > bottom`
- return forced-bust metrics immediately instead of enumerating future T4 draws.

This preserves exactness because no future placement can change completed rows.

Rebuilt with:

```powershell
cargo build --release --manifest-path ai\rust_solver\t3_exact_solver\Cargo.toml
```

Repeated the same capped T2 smoke:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.tutor.run_t2_exact_oracle --input ai\data\tutor_t2_top20_weak_20260524\teacher_labels_t2_top20_suit24_mc1000.jsonl --out-dir ai\data\hybrid_t1t2_active_20260531\t2_exact_oracle_smoke_forcedbust_20260602 --limit 1 --top-n 24 --t2-draw-limit 1 --binary ai\rust_solver\target\release\t3_exact_solver.exe
```

Result:

- MC1000 Top1 vs capped-exact Top1: `1/1` matched
- capped exact elapsed improved from `26919.1ms` to `23170.6ms`
- estimated naive full T2 exact cost improved from about `44.7h` to about
  `38.5h` for the same position

Interpretation:

- The shortcut is correct and useful, but not sufficient.
- The next speedup must be structural: dynamic programming or memoization of
  late-turn values, not another small rule.

## T3 Exact-State Memoization

Added a shared per-position cache in `ai/rust_solver/t3_exact_solver/src/main.rs`.

Cached value:

- the exact T4 expectation for a fixed post-T3-action state:
  - hero board after the T3 action
  - visible opponent board
  - known/excluded cards including the T2/T3 discards

Why this preserves exactness:

- the cached key contains the entire state that determines the remaining T4
  deck and terminal scoring;
- repeated T2/T3 paths that reach the same post-T3 state now reuse the same
  exact T4 value instead of recomputing it.

Build:

```powershell
cargo build --release --manifest-path ai\rust_solver\t3_exact_solver\Cargo.toml
```

Smoke results on the same T2 position:

| mode | Top1 vs MC1000 | solver elapsed | naive full estimate |
| --- | ---: | ---: | ---: |
| cap1 before forced-bust/cache | 1/1 | 26919.1ms | 44.7h |
| cap1 after forced-bust | 1/1 | 23170.6ms | 38.5h |
| cap1 after cache | 1/1 | 9564.1ms | 15.9h |
| cap2 after cache | 1/1 | 18510.8ms | 15.4h |
| cap1 after cache, limit2 | 2/2 | 10391.1ms avg | 17.3h avg |

Interpretation:

- The first smoke looked like a cache speedup, but the follow-up cache audit
  below showed zero cache hits on the same shape of T2 capped oracle.
- Treat these numbers as historical smoke results, not proof that cache reuse is
  the right path.
- Full exact is still too slow for broad local generation.
- The next likely speedup is deeper dynamic programming over row-fill states or
  a compact numeric evaluator, so the T3/T4 leaf is not rebuilt from strings for
  every state.

## Cache Audit And Exclude Normalization

Retested the shared T3/T4 cache idea with explicit hit/miss statistics.

Result:

- Direct T4 completion cache was counterproductive on the smoke T2 position:
  - cap1 elapsed increased to `12212.0ms`
  - T4 cache stats showed `1,051,830` misses and `0` hits
- T3 post-action cache also showed no reuse on the same cap tests:
  - cap1: `342` misses, `0` hits
  - cap2: `684` misses, `0` hits

Decision:

- Remove the T3/T4 cache path rather than keeping a misleading optimization.
- Keep only exact-preserving `exclude` normalization:
  - deduplicate `exclude`/`known_discards`
  - remove cards already represented by hero board, opponent board, or current
    dealt cards
  - continue adding action discards exactly once

Why this is exact:

- `remaining_deck` already treats used cards as a set.
- Duplicate dead cards and board/opponent cards inside `exclude` do not change
  the legal remaining deck.

Smoke after removing the cache path and normalizing `exclude`:

| mode | Top1 vs MC1000 | solver elapsed | naive full estimate |
| --- | ---: | ---: | ---: |
| cap1, limit1 | 1/1 | 8447.4ms | 14.0h |
| cap2, limit1 | 1/1 | 18258.9ms | 15.2h |
| cap1, limit2 | 2/2 | 8910.2ms avg | 14.8h avg |
| cap1, limit1 final recheck | 1/1 | 10966.0ms | 18.2h |

Interpretation:

- This is the current best local T2 capped-oracle path.
- Single-position timings vary, so the table should be read as a small smoke
  range, not a stable benchmark.
- It is still far from broad full exact generation.
- The next useful implementation step is not another cache; it is a compact
  numeric evaluator / DP for T4 completion and row-fill states.

Also tested streaming 3-card combination iteration instead of materializing the
small draw vector.  It preserved Top1 but was slower on cap1/cap2, so it was
reverted.

## Terminal Board Evaluation Reuse

Added `BoardEval` in `ai/rust_solver/t3_exact_solver/src/main.rs`.

Purpose:

- evaluate the fixed opponent board once per position;
- evaluate each terminal hero board once and reuse:
  - row hand values
  - bust state
  - royalty
  - FL key/type
- remove the older terminal score path that re-evaluated the same rows multiple
  times.

Correctness check:

- Compared the new output against the previous exact-preserving output.
- cap1/limit1: all 24 candidate actions matched exactly for:
  - `score`
  - `raw_score`
  - `royalty`
  - `bust_rate`
  - `fl_rate`
  - `fl_type_rates`
  - `samples`
  - `forced_bust`
- cap2/limit1: all 24 candidate actions matched exactly on the same fields.

Smoke after `BoardEval`:

| mode | Top1 vs MC1000 | solver elapsed | naive full estimate |
| --- | ---: | ---: | ---: |
| cap1, limit1 | 1/1 | 9802.1ms | 16.3h |
| cap2, limit1 | 1/1 | 18956.6ms | 15.8h |

Interpretation:

- The result is exact-equivalent to the previous path on the tested candidates.
- The speed gain is modest and noisy, not enough to make T2 full exact practical.
- The remaining bottleneck is still the huge number of T4 terminal hand
  evaluations.  The next real speed step should be a compact numeric card/row
  representation and/or row-fill DP, not more high-level Rust plumbing.

## Row-Level Hand Eval Cache

Added a local `RowEvalCache` in `ai/rust_solver/t3_exact_solver/src/main.rs`.

What changed:

- cache `evaluate_hand(row, expected_count)` by a compact sorted numeric card
  key;
- scope the cache to one candidate evaluation, so memory does not grow across
  the whole run;
- use the same cache for:
  - irreparable-bust checks;
  - terminal `BoardEval`;
  - all T3/T4 leaf evaluations inside one T2 candidate;
- derive top royalty and FL type from the cached top-hand value instead of
  scanning top cards separately.

Why this is exact:

- every cached value is produced by the existing `evaluate_hand` function;
- the cache key is order-insensitive and includes expected row size;
- no candidate, draw, or terminal state is skipped.

Correctness check:

- cap1/limit1: compared all 24 candidate actions against the previous
  `BoardEval` output; all fields matched exactly.
- cap2/limit1: compared all 24 candidate actions; all fields matched exactly.
- cap1/limit2: compared both records; all candidate fields matched exactly.

Smoke after row-eval cache:

| mode | Top1 vs MC1000 | solver elapsed | naive full estimate |
| --- | ---: | ---: | ---: |
| cap1, limit1 | 1/1 | 3670.6ms | 6.1h |
| cap2, limit1 | 1/1 | 6656.1ms | 5.5h |
| cap1, limit2 | 2/2 | 3623.0ms avg | 6.0h avg |

Interpretation:

- This is the first meaningful local oracle speedup after forced-bust pruning.
- Full T2 exact is still too slow for large local teacher generation, but the
  estimate moved from roughly `15h+` per position to roughly `5.5h-6.1h` on the
  smoke shape.
- The next step should be a row-fill DP / precomputed row evaluator, because the
  remaining cost is now repeated terminal enumeration rather than obvious
  duplicate hand evaluation.

## Row Cache Stats And T4 Partial Evaluation

Added per-candidate `row_eval_cache` stats to exact solver output.

cap1/limit1 stats:

- candidates: `24`
- total entries: `7,971`
- total hits: `17,468,805`
- total misses: `7,971`
- total hit rate: `99.95%`
- entries per candidate: min `128`, avg `332.1`, max `658`

Interpretation:

- `evaluate_hand` calls are now mostly eliminated.
- The remaining cost includes millions of cache lookups and T4 terminal action
  construction/evaluation.

Added T4 partial evaluation:

- precompute row values that are already complete before T4 placement;
- evaluate only rows touched by the T4 action;
- avoid cloning/applying the full board for every terminal action;
- keep the same row-eval cache for changed rows.

Correctness check:

- cap1/limit1: all 24 candidate fields matched the previous row-cache output.
- cap2/limit1: all 24 candidate fields matched the previous row-cache output.
- cap1/limit2: both records matched the previous row-cache output.

Smoke after T4 partial evaluation:

| mode | Top1 vs MC1000 | solver elapsed | naive full estimate |
| --- | ---: | ---: | ---: |
| cap1, limit1 | 1/1 | 2973.7ms | 4.9h |
| cap2, limit1 | 1/1 | 5900.3ms | 4.9h |
| cap1, limit2 | 2/2 | 3385.7ms avg | 5.6h avg |

Next implication:

- This is still exact-equivalent on the smoke checks.
- The next meaningful speed step is a compact terminal-action evaluator or
  row-fill DP that avoids building temporary row `Vec<String>` values for
  changed rows.

## Key-Only Row Cache Hits

Added key-only row lookup for T4 changed-row evaluation.

What changed:

- compute the row cache key from fixed-size numeric card codes;
- check the cache before constructing a temporary `Vec<String>`;
- build the temporary row card vector only on cache miss;
- keep the previous cache key semantics:
  - order-insensitive;
  - includes expected row size;
  - still computes misses with the existing `evaluate_hand` function.

Correctness check:

- cap1/limit1: all 24 candidate fields matched the previous T4 partial output.
- cap2/limit1: all 24 candidate fields matched the previous T4 partial output.
- cap1/limit2: both records matched the previous T4 partial output.

Smoke after key-only row cache hits:

| mode | Top1 vs MC1000 | solver elapsed | naive full estimate |
| --- | ---: | ---: | ---: |
| cap1, limit1 | 1/1 | 1879.9ms | 3.1h |
| cap2, limit1 | 1/1 | 3838.7ms | 3.2h |
| cap1, limit2 | 2/2 | 1970.9ms avg | 3.3h avg |

Interpretation:

- The row cache hit/miss counts are unchanged, but hit handling is much cheaper.
- The smoke estimate is now roughly `3.1h-3.3h` per full T2 position.
- The next useful speed step is likely eliminating temporary `Vec<String>` in
  T3 action application and T2/T3 draw construction, or moving to fixed-card
  arrays for row-fill DP.

## T4 Direct Action Enumerator And Draw Refs

Added a T4-specific terminal action evaluator.

What changed:

- avoid `get_turn_actions` inside every T4 leaf;
- enumerate T4 discard/placement combinations directly in the same order as the
  old action list;
- preserve duplicate-action suppression with a compact canonical key;
- preserve old tie behavior by replacing the best terminal on equal score;
- pass T4 draws as sorted references instead of cloning a 3-card `Vec<String>`.

Correctness check:

- cap1/limit1: all 24 candidate fields matched the previous key-only output.
- cap2/limit1: all 24 candidate fields matched the previous key-only output.
- cap1/limit2: both records matched the previous key-only output.

Smoke after direct T4 action enumeration:

| mode | Top1 vs MC1000 | solver elapsed | naive full estimate |
| --- | ---: | ---: | ---: |
| cap1, limit1 | 1/1 | 179.6ms | 17.9m |
| cap2, limit1 | 1/1 | 356.7ms | 17.8m |
| cap1, limit2 | 2/2 | 200.6ms avg | 20.0m avg |
| cap10, limit1 | 0/1 | 1854.4ms | 18.5m |
| cap100, limit1 | 0/1 | 18555.5ms | 18.5m |

cap100 detail on the first smoke position:

- MC1000 source Top1:
  - `6d->top; 8s->top; discard 5c`
  - capped exact score: `0.3481`
- cap100 Top1:
  - `6d->middle; 8s->top; discard 5c`
  - capped exact score: `0.4669`
- exact regret of the MC1000 Top1 under cap100: `0.1189`

Interpretation:

- This is the largest exact-preserving speedup so far.
- A stronger local capped oracle is now practical for T2 auditing: cap100 on one
  T2 position took under 20 seconds.
- Full T2 exact is still not cheap, but the estimate moved from hours to about
  18-20 minutes on this smoke shape.
- The cap100 disagreement with MC1000 confirms that existing MC teacher labels
  can be materially wrong even when top candidates look plausible.

## Static Deck And T2 cap100 Audit

Tried two follow-up changes:

- direct T3 action enumeration;
- static full-deck representation for `remaining_deck`.

Result:

- Direct T3 action enumeration was exact-equivalent but slightly slower on the
  smoke checks, so it was reverted.
- Static full-deck representation preserved exact output and draw order.  It
  reduces repeated full-deck string construction, but timing impact was small:
  - cap1/limit1: `180.8ms`
  - cap2/limit1: `355.1ms`
  - cap100/limit1: `19625.8ms`

Because cap100 is now feasible locally, generated a small T2 cap100 audit set:

- output:
  `ai/data/hybrid_t1t2_active_20260531/t2_cap100_audit_limit5_20260602/t2_oracle_cap100_limit5.jsonl`
- records: `5`
- same MC1000 Top1 vs cap100 oracle: `0/5`
- avg exact elapsed: `18643.2ms`
- avg full T2 estimate from cap100: `1115611.4ms` (`18.6m`)
- avg exact regret of source Top1: `0.2581`
- max exact regret of source Top1: `0.4669`

Pattern in the first five audited T2 positions:

- cap100 consistently preferred placing the `8` to top and the `6` to middle,
  discarding the low `5` variant;
- source MC1000 Top1 was often only rank `3`, `13`, or `14` under cap100;
- the observed disagreement is not a small tie artifact.

Implication:

- The existing `mc1000` T2 labels are not reliable enough as "correct" labels.
- Next training data should prioritize a capped-exact T2 oracle sweep, starting
  with cap100/cap200 local batches and logging all MC-vs-oracle misses.
- Full T2 exact is still expensive, but cap100 is now practical enough to build
  a useful audit and distillation set locally before using GCP.

## T2 Oracle Miss Logging And cap100/cap200 Check

Extended `ai/tutor/run_t2_exact_oracle.py` so every MC-vs-oracle Top1 mismatch
is written to a dedicated miss file:

- output pattern: `t2_oracle_<cap>_limit<N>.misses.jsonl`
- each miss contains:
  - source hand context: `board`, `opponent_board`, `dealt`, `known_discards`,
    `exclude`, `source`, `source_line`;
  - source Top1 and oracle Top1;
  - source Top1 rank under oracle;
  - oracle Top1 rank under source labels;
  - exact regret of the source Top1;
  - top source candidates and top oracle candidates for review/training.

Validation:

```powershell
C:\Users\Owner\anaconda3\python.exe -m py_compile ai\tutor\run_t2_exact_oracle.py
cargo build --release --manifest-path ai\rust_solver\t3_exact_solver\Cargo.toml
```

Both checks passed.

cap100 / 10-record audit:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.tutor.run_t2_exact_oracle --input ai\data\tutor_t2_top20_weak_20260524\teacher_labels_t2_top20_suit24_mc1000.jsonl --out-dir ai\data\hybrid_t1t2_active_20260531\t2_cap100_audit_limit10_20260602 --limit 10 --top-n 24 --t2-draw-limit 100 --binary ai\rust_solver\target\release\t3_exact_solver.exe
```

Artifacts:

- oracle labels:
  `ai/data/hybrid_t1t2_active_20260531/t2_cap100_audit_limit10_20260602/t2_oracle_cap100_limit10.jsonl`
- summary:
  `ai/data/hybrid_t1t2_active_20260531/t2_cap100_audit_limit10_20260602/t2_oracle_cap100_limit10.summary.md`
- miss set:
  `ai/data/hybrid_t1t2_active_20260531/t2_cap100_audit_limit10_20260602/t2_oracle_cap100_limit10.misses.jsonl`

Results:

- records: `10`
- MC1000 Top1 vs cap100 oracle Top1: `0/10`
- misses written: `10`
- avg exact elapsed: `27612.5ms`
- avg full T2 estimate from cap100: `1652331.1ms` (`27.5m`)
- avg exact regret of source Top1: `0.2658`
- max exact regret of source Top1: `0.4669`

cap200 / 3-record convergence check:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.tutor.run_t2_exact_oracle --input ai\data\tutor_t2_top20_weak_20260524\teacher_labels_t2_top20_suit24_mc1000.jsonl --out-dir ai\data\hybrid_t1t2_active_20260531\t2_cap200_audit_limit3_20260602 --limit 3 --top-n 24 --t2-draw-limit 200 --binary ai\rust_solver\target\release\t3_exact_solver.exe
```

Artifacts:

- oracle labels:
  `ai/data/hybrid_t1t2_active_20260531/t2_cap200_audit_limit3_20260602/t2_oracle_cap200_limit3.jsonl`
- summary:
  `ai/data/hybrid_t1t2_active_20260531/t2_cap200_audit_limit3_20260602/t2_oracle_cap200_limit3.summary.md`
- miss set:
  `ai/data/hybrid_t1t2_active_20260531/t2_cap200_audit_limit3_20260602/t2_oracle_cap200_limit3.misses.jsonl`

Results:

- records: `3`
- MC1000 Top1 vs cap200 oracle Top1: `0/3`
- misses written: `3`
- avg exact elapsed: `35748.2ms`
- avg full T2 estimate from cap200: `1069585.8ms` (`17.8m`)
- avg exact regret of source Top1: `0.1849`
- max exact regret of source Top1: `0.1849`
- cap100 and cap200 agreed on Top1 for all first 3 records:
  - `6d->middle; 8s->top; discard 5c`
  - `6d->middle; 8c->top; discard 5s`
  - `6c->middle; 8s->top; discard 5d`

Interpretation:

- cap100 is not final exact; score and bust estimates still change at cap200.
- However, the Top1 disagreement with MC1000 is stable enough to treat these as
  high-value active-learning misses.
- The next data step should be a larger capped-exact sweep plus a cap100 vs
  cap200 stability filter before using the rows as hard labels.
- The next speed step should target full T2 exact or higher caps, not 5-second
  runtime tuning.

## T2 Stable Capped-Oracle Labels

Added:

- `ai/tutor/build_t2_stable_oracle_labels.py`

Purpose:

- compare two capped-exact T2 oracle outputs by `record_index`;
- emit a hard teacher JSONL row only when both caps choose the same Top1 action;
- write missing or unstable rows separately for higher-cap/full-exact follow-up;
- preserve the original source context plus all strong-cap candidate metrics.

Validation:

```powershell
C:\Users\Owner\anaconda3\python.exe -m py_compile ai\tutor\build_t2_stable_oracle_labels.py
```

The check passed.

First stable extraction from cap100/10 and cap200/3:

- output:
  `ai/data/hybrid_t1t2_active_20260531/stable_t2_oracle_cap100_cap200_first3_20260602.jsonl`
- common records: `3`
- stable Top1: `3/3`
- missing strong cap rows: `7`

Then generated cap200 for all 10 audited rows:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.tutor.run_t2_exact_oracle --input ai\data\tutor_t2_top20_weak_20260524\teacher_labels_t2_top20_suit24_mc1000.jsonl --out-dir ai\data\hybrid_t1t2_active_20260531\t2_cap200_audit_limit10_20260602 --limit 10 --top-n 24 --t2-draw-limit 200 --binary ai\rust_solver\target\release\t3_exact_solver.exe
```

cap200 / 10-record result:

- oracle labels:
  `ai/data/hybrid_t1t2_active_20260531/t2_cap200_audit_limit10_20260602/t2_oracle_cap200_limit10.jsonl`
- miss set:
  `ai/data/hybrid_t1t2_active_20260531/t2_cap200_audit_limit10_20260602/t2_oracle_cap200_limit10.misses.jsonl`
- records: `10`
- MC1000 Top1 vs cap200 oracle Top1: `0/10`
- avg exact elapsed: `36861.2ms`
- avg full T2 estimate from cap200: `1102887.7ms` (`18.4m`)
- avg exact regret of source Top1: `0.3397`
- max exact regret of source Top1: `0.6598`

Stable extraction from cap100/10 and cap200/10:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.tutor.build_t2_stable_oracle_labels --source ai\data\tutor_t2_top20_weak_20260524\teacher_labels_t2_top20_suit24_mc1000.jsonl --weak ai\data\hybrid_t1t2_active_20260531\t2_cap100_audit_limit10_20260602\t2_oracle_cap100_limit10.jsonl --strong ai\data\hybrid_t1t2_active_20260531\t2_cap200_audit_limit10_20260602\t2_oracle_cap200_limit10.jsonl --output ai\data\hybrid_t1t2_active_20260531\stable_t2_oracle_cap100_cap200_limit10_20260602.jsonl
```

Artifacts:

- stable labels:
  `ai/data/hybrid_t1t2_active_20260531/stable_t2_oracle_cap100_cap200_limit10_20260602.jsonl`
- summary:
  `ai/data/hybrid_t1t2_active_20260531/stable_t2_oracle_cap100_cap200_limit10_20260602.summary.json`
- unstable rows:
  `ai/data/hybrid_t1t2_active_20260531/stable_t2_oracle_cap100_cap200_limit10_20260602.unstable.jsonl`

Results:

- common records: `10`
- stable Top1: `10/10`
- unstable rows: `0`
- avg cap100-vs-cap200 Top1 score absolute difference: `0.1460`
- max cap100-vs-cap200 Top1 score absolute difference: `0.1929`
- avg cap100-vs-cap200 Top1 bust absolute difference: `0.0650`
- max cap100-vs-cap200 Top1 bust absolute difference: `0.0995`

Interpretation:

- These 10 rows are acceptable as Top1/ranking active-learning labels because
  cap100 and cap200 agree on the best action.
- They are not yet high-confidence EV/bust regression labels because metric
  values still move materially between caps.
- The next local data step is to scale this to a broader cap100 sweep, then run
  cap200 only on the rows intended for hard Top1 training.
- The next oracle-quality step is a cap500/full-exact spot check on the stable
  rows to verify that cap200 has not locked onto a still-wrong Top1.

## T2 cap500 Spot Check

Ran a higher-cap spot check on the first 3 T2 audit rows:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.tutor.run_t2_exact_oracle --input ai\data\tutor_t2_top20_weak_20260524\teacher_labels_t2_top20_suit24_mc1000.jsonl --out-dir ai\data\hybrid_t1t2_active_20260531\t2_cap500_audit_limit3_20260602 --limit 3 --top-n 24 --t2-draw-limit 500 --binary ai\rust_solver\target\release\t3_exact_solver.exe
```

Artifacts:

- oracle labels:
  `ai/data/hybrid_t1t2_active_20260531/t2_cap500_audit_limit3_20260602/t2_oracle_cap500_limit3.jsonl`
- summary:
  `ai/data/hybrid_t1t2_active_20260531/t2_cap500_audit_limit3_20260602/t2_oracle_cap500_limit3.summary.md`
- miss set:
  `ai/data/hybrid_t1t2_active_20260531/t2_cap500_audit_limit3_20260602/t2_oracle_cap500_limit3.misses.jsonl`

cap500 result:

- records: `3`
- MC1000 Top1 vs cap500 oracle Top1: `0/3`
- misses written: `3`
- avg exact elapsed: `87783.0ms`
- avg full T2 estimate from cap500: `1050586.7ms` (`17.5m`)
- avg exact regret of source Top1: `0.2152`
- max exact regret of source Top1: `0.2170`

Stable extraction from cap200/10 and cap500/3:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.tutor.build_t2_stable_oracle_labels --source ai\data\tutor_t2_top20_weak_20260524\teacher_labels_t2_top20_suit24_mc1000.jsonl --weak ai\data\hybrid_t1t2_active_20260531\t2_cap200_audit_limit10_20260602\t2_oracle_cap200_limit10.jsonl --strong ai\data\hybrid_t1t2_active_20260531\t2_cap500_audit_limit3_20260602\t2_oracle_cap500_limit3.jsonl --output ai\data\hybrid_t1t2_active_20260531\stable_t2_oracle_cap200_cap500_first3_20260602.jsonl
```

Artifacts:

- stable labels:
  `ai/data/hybrid_t1t2_active_20260531/stable_t2_oracle_cap200_cap500_first3_20260602.jsonl`
- summary:
  `ai/data/hybrid_t1t2_active_20260531/stable_t2_oracle_cap200_cap500_first3_20260602.summary.json`
- unstable rows:
  `ai/data/hybrid_t1t2_active_20260531/stable_t2_oracle_cap200_cap500_first3_20260602.unstable.jsonl`

Results:

- common records: `3`
- stable Top1: `3/3`
- unstable common rows: `0`
- missing cap500 rows from cap200/10: `7`
- avg cap200-vs-cap500 Top1 score absolute difference: `0.0237`
- max cap200-vs-cap500 Top1 score absolute difference: `0.0322`
- avg cap200-vs-cap500 Top1 bust absolute difference: `0.0135`
- max cap200-vs-cap500 Top1 bust absolute difference: `0.0154`

Interpretation:

- For the checked rows, Top1 is stable across cap100, cap200, and cap500.
- cap200-to-cap500 metric movement is much smaller than cap100-to-cap200, so
  cap500 spot checks are a useful quality gate.
- The current best local teacher policy is:
  - use cap100 to find candidate hard rows cheaply;
  - confirm intended training rows with cap200;
  - cap500 spot-check representative slices before treating them as strong
    ranking labels;
  - reserve metric regression labels for cap500/full-exact or rows with small
    cap200-vs-cap500 metric movement.

## Diverse T2 Oracle Inputs

The original `teacher_labels_t2_top20_suit24_mc1000.jsonl` was not diverse:

- total records: `72`
- unique `(source, source_line)`: `3`
- each unique base position had `24` suit-expanded copies

Added:

- `ai/tutor/select_diverse_oracle_inputs.py`

Purpose:

- select rows for expensive oracle generation while limiting repeated
  suit-expanded copies;
- default to at most one row per `(source, source_line, source_decision_index)`;
- preserve `oracle_input_source`, `oracle_input_line`, and group keys in the
  selected records.

Validation:

```powershell
C:\Users\Owner\anaconda3\python.exe -m py_compile ai\tutor\select_diverse_oracle_inputs.py
```

The check passed.

Selected 10 diverse T2 residual-neighbor rows:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.tutor.select_diverse_oracle_inputs --input ai\data\hybrid_t1t2_active_20260531\teacher_t2_residual_neighbors_w120_mc1000_20260601.jsonl --output ai\data\hybrid_t1t2_active_20260531\oracle_inputs_t2_residual_diverse10_20260602.jsonl --turns 2 --limit 10 --max-per-source 1
```

Selection result:

- selected: `10`
- unique source groups: `10`
- unique rank groups: `10`

cap100 on diverse10:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.tutor.run_t2_exact_oracle --input ai\data\hybrid_t1t2_active_20260531\oracle_inputs_t2_residual_diverse10_20260602.jsonl --out-dir ai\data\hybrid_t1t2_active_20260531\t2_cap100_residual_diverse10_20260602 --limit 10 --top-n 64 --t2-draw-limit 100 --binary ai\rust_solver\target\release\t3_exact_solver.exe
```

Result:

- records: `10`
- MC1000 Top1 vs cap100 Top1: `6/10`
- changed Top1: `4`
- avg elapsed: `13826.8ms`
- avg full T2 estimate from cap100: `877800.5ms` (`14.6m`)
- avg exact regret of source Top1: `0.6471`
- max exact regret of source Top1: `5.1029`

cap200 on diverse10:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.tutor.run_t2_exact_oracle --input ai\data\hybrid_t1t2_active_20260531\oracle_inputs_t2_residual_diverse10_20260602.jsonl --out-dir ai\data\hybrid_t1t2_active_20260531\t2_cap200_residual_diverse10_20260602 --limit 10 --top-n 64 --t2-draw-limit 200 --binary ai\rust_solver\target\release\t3_exact_solver.exe
```

Result:

- records: `10`
- MC1000 Top1 vs cap200 Top1: `6/10`
- changed Top1: `4`
- avg elapsed: `28746.9ms`
- avg full T2 estimate from cap200: `913127.5ms` (`15.2m`)
- avg exact regret of source Top1: `0.3035`
- max exact regret of source Top1: `2.6348`

Stable extraction from cap100/cap200 diverse10:

- output:
  `ai/data/hybrid_t1t2_active_20260531/stable_t2_oracle_residual_diverse10_cap100_cap200_20260602.jsonl`
- common records: `10`
- stable Top1: `10/10`
- unstable rows: `0`
- avg cap100-vs-cap200 score absolute difference: `0.6172`
- max cap100-vs-cap200 score absolute difference: `1.8384`
- avg cap100-vs-cap200 bust absolute difference: `0.0277`
- max cap100-vs-cap200 bust absolute difference: `0.0574`

This says cap100/cap200 Top1 can be stable while score is still not stable.

cap500 on cap200-miss rows:

Selected the 4 cap200 Top1-miss rows from diverse10:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.tutor.select_diverse_oracle_inputs --input ai\data\hybrid_t1t2_active_20260531\oracle_inputs_t2_residual_diverse10_20260602.jsonl --output ai\data\hybrid_t1t2_active_20260531\oracle_inputs_t2_residual_diverse10_cap200_misses4_20260602.jsonl --turns 2 --limit 4 --max-per-source 1 --indices 1,5,6,8
```

Then ran cap500:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.tutor.run_t2_exact_oracle --input ai\data\hybrid_t1t2_active_20260531\oracle_inputs_t2_residual_diverse10_cap200_misses4_20260602.jsonl --out-dir ai\data\hybrid_t1t2_active_20260531\t2_cap500_residual_diverse_misses4_20260602 --limit 4 --top-n 64 --t2-draw-limit 500 --binary ai\rust_solver\target\release\t3_exact_solver.exe
```

cap500 result on those 4 rows:

- MC1000 Top1 vs cap500 Top1: `1/4`
- changed Top1: `3`
- avg elapsed: `43209.1ms`
- avg full T2 estimate from cap500: `547914.9ms` (`9.1m`)
- avg exact regret of source Top1: `0.4728`
- max exact regret of source Top1: `1.4821`

Re-ran cap200 on the same 4-row subset and compared cap200/cap500:

- stable labels:
  `ai/data/hybrid_t1t2_active_20260531/stable_t2_oracle_residual_diverse_misses4_cap200_cap500_20260602.jsonl`
- unstable rows:
  `ai/data/hybrid_t1t2_active_20260531/stable_t2_oracle_residual_diverse_misses4_cap200_cap500_20260602.unstable.jsonl`
- common records: `4`
- stable Top1: `3/4`
- cap200-vs-cap500 Top1 changed: `1/4`
- stable rows avg score absolute difference: `0.1905`
- stable rows max score absolute difference: `0.2896`
- stable rows avg bust absolute difference: `0.0131`
- stable rows max bust absolute difference: `0.0200`

The unstable row was source line `946`: cap200 chose
`5h->middle;6c->middle; discard 2h`, while cap500 chose
`2h->middle;5h->middle; discard 6c`.  cap500 also made the original MC1000 Top1
a hit for that row, so cap200 would have produced a false hard miss.

Interpretation:

- Diverse rows give a more realistic audit than suit-expanded rows.
- cap100/cap200 Top1 agreement is not enough for all high-regret hard labels.
- For rows where source and cap200 disagree, cap500 should be run before using
  the row as a hard Top1 training target.
- The immediate active-learning set from this diverse10 probe is the `3/4`
  cap200/cap500-stable miss rows, not all cap200 misses.

## Diverse20 T2 Active-Label Probe

Extended `ai/tutor/select_diverse_oracle_inputs.py` with:

- `--indices-file`
- `--index-field`

This lets later stages read `*.misses.jsonl` and automatically extract the
matching original input rows by `record_index`, instead of manually copying
indices.

Validation:

```powershell
C:\Users\Owner\anaconda3\python.exe -m py_compile ai\tutor\select_diverse_oracle_inputs.py
```

The check passed.

Selected 20 diverse T2 residual-neighbor rows:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.tutor.select_diverse_oracle_inputs --input ai\data\hybrid_t1t2_active_20260531\teacher_t2_residual_neighbors_w120_mc1000_20260601.jsonl --output ai\data\hybrid_t1t2_active_20260531\oracle_inputs_t2_residual_diverse20_20260602.jsonl --turns 2 --limit 20 --max-per-source 1
```

Selection result:

- selected: `20`
- unique source groups: `20`
- unique rank groups: `20`

cap100 on diverse20:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.tutor.run_t2_exact_oracle --input ai\data\hybrid_t1t2_active_20260531\oracle_inputs_t2_residual_diverse20_20260602.jsonl --out-dir ai\data\hybrid_t1t2_active_20260531\t2_cap100_residual_diverse20_20260602 --limit 20 --top-n 64 --t2-draw-limit 100 --binary ai\rust_solver\target\release\t3_exact_solver.exe
```

Result:

- records: `20`
- MC1000 Top1 vs cap100 Top1: `14/20`
- changed Top1: `6`
- avg elapsed: `19840.0ms`
- avg full T2 estimate from cap100: `1310877.8ms` (`21.8m`)
- avg exact regret of source Top1: `0.3811`
- max exact regret of source Top1: `5.1029`

Extracted the 6 cap100 miss rows automatically:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.tutor.select_diverse_oracle_inputs --input ai\data\hybrid_t1t2_active_20260531\oracle_inputs_t2_residual_diverse20_20260602.jsonl --output ai\data\hybrid_t1t2_active_20260531\oracle_inputs_t2_residual_diverse20_cap100_misses6_20260602.jsonl --turns 2 --limit 6 --max-per-source 1 --indices-file ai\data\hybrid_t1t2_active_20260531\t2_cap100_residual_diverse20_20260602\t2_oracle_cap100_limit20.misses.jsonl
```

The selected indices were `1,5,6,8,15,18`.

cap200 on the 6 cap100 misses:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.tutor.run_t2_exact_oracle --input ai\data\hybrid_t1t2_active_20260531\oracle_inputs_t2_residual_diverse20_cap100_misses6_20260602.jsonl --out-dir ai\data\hybrid_t1t2_active_20260531\t2_cap200_residual_diverse20_misses6_20260602 --limit 6 --top-n 64 --t2-draw-limit 200 --binary ai\rust_solver\target\release\t3_exact_solver.exe
```

Result:

- records: `6`
- MC1000 Top1 vs cap200 Top1: `0/6`
- changed Top1: `6`
- avg elapsed: `16889.0ms`
- avg full T2 estimate from cap200: `540879.0ms` (`9.0m`)
- avg exact regret of source Top1: `0.7000`
- max exact regret of source Top1: `2.6348`

cap100/cap200 stable extraction on the 6 rows:

- output:
  `ai/data/hybrid_t1t2_active_20260531/stable_t2_oracle_residual_diverse20_misses6_cap100_cap200_20260602.jsonl`
- common records: `6`
- stable Top1: `6/6`
- unstable rows: `0`

cap500 on the same 6 rows:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.tutor.run_t2_exact_oracle --input ai\data\hybrid_t1t2_active_20260531\oracle_inputs_t2_residual_diverse20_cap100_misses6_20260602.jsonl --out-dir ai\data\hybrid_t1t2_active_20260531\t2_cap500_residual_diverse20_misses6_20260602 --limit 6 --top-n 64 --t2-draw-limit 500 --binary ai\rust_solver\target\release\t3_exact_solver.exe
```

Result:

- records: `6`
- MC1000 Top1 vs cap500 Top1: `1/6`
- changed Top1: `5`
- avg elapsed: `46011.1ms`
- avg full T2 estimate from cap500: `596156.5ms` (`9.9m`)
- avg exact regret of source Top1: `0.4855`
- max exact regret of source Top1: `1.4821`

cap200/cap500 stable extraction:

- stable labels:
  `ai/data/hybrid_t1t2_active_20260531/stable_t2_oracle_residual_diverse20_misses6_cap200_cap500_20260602.jsonl`
- unstable rows:
  `ai/data/hybrid_t1t2_active_20260531/stable_t2_oracle_residual_diverse20_misses6_cap200_cap500_20260602.unstable.jsonl`
- common records: `6`
- stable Top1: `5/6`
- cap200-vs-cap500 Top1 changed: `1/6`
- stable rows avg score absolute difference: `0.3392`
- stable rows max score absolute difference: `1.0911`
- stable rows avg bust absolute difference: `0.0215`
- stable rows max bust absolute difference: `0.0370`

The unstable row was again source line `946`; cap200 chose
`5h->middle;6c->middle; discard 2h`, while cap500 chose
`2h->middle;5h->middle; discard 6c`.

Interpretation:

- In diverse20, cap100 found `6` candidate misses.
- All 6 survived to cap200 as source-vs-oracle misses.
- Only 5 survived cap500 as source-vs-oracle misses and cap200/cap500-stable
  Top1 labels.
- The active-learning set should therefore use the 5 cap200/cap500-stable rows.
- The source-line-946 case should be excluded from hard Top1 training unless a
  higher cap or full exact confirms it.

## Diverse50 T2 Active-Label Probe

Scaled the local diverse T2 probe to 50 base positions.

Selected 50 diverse T2 residual-neighbor rows:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.tutor.select_diverse_oracle_inputs --input ai\data\hybrid_t1t2_active_20260531\teacher_t2_residual_neighbors_w120_mc1000_20260601.jsonl --output ai\data\hybrid_t1t2_active_20260531\oracle_inputs_t2_residual_diverse50_20260602.jsonl --turns 2 --limit 50 --max-per-source 1
```

Selection result:

- selected: `50`
- unique source groups: `50`
- unique rank groups: `50`

cap100 on diverse50:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.tutor.run_t2_exact_oracle --input ai\data\hybrid_t1t2_active_20260531\oracle_inputs_t2_residual_diverse50_20260602.jsonl --out-dir ai\data\hybrid_t1t2_active_20260531\t2_cap100_residual_diverse50_20260602 --limit 50 --top-n 64 --t2-draw-limit 100 --binary ai\rust_solver\target\release\t3_exact_solver.exe
```

Result:

- records: `50`
- MC1000 Top1 vs cap100 Top1: `25/50`
- changed Top1: `25`
- avg elapsed: `20097.4ms`
- avg full T2 estimate from cap100: `1353448.3ms` (`22.6m`)
- avg exact regret of source Top1: `1.0218`
- max exact regret of source Top1: `9.6056`

Extracted the 25 cap100 miss rows automatically:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.tutor.select_diverse_oracle_inputs --input ai\data\hybrid_t1t2_active_20260531\oracle_inputs_t2_residual_diverse50_20260602.jsonl --output ai\data\hybrid_t1t2_active_20260531\oracle_inputs_t2_residual_diverse50_cap100_misses25_20260602.jsonl --turns 2 --limit 25 --max-per-source 1 --indices-file ai\data\hybrid_t1t2_active_20260531\t2_cap100_residual_diverse50_20260602\t2_oracle_cap100_limit50.misses.jsonl
```

Selected indices:

- `1,5,6,8,15,18,21,22,26,27,28,29,31,32,34,35,36,39,40,41,42,43,45,47,49`

cap200 on the 25 cap100 misses:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.tutor.run_t2_exact_oracle --input ai\data\hybrid_t1t2_active_20260531\oracle_inputs_t2_residual_diverse50_cap100_misses25_20260602.jsonl --out-dir ai\data\hybrid_t1t2_active_20260531\t2_cap200_residual_diverse50_misses25_20260602 --limit 25 --top-n 64 --t2-draw-limit 200 --binary ai\rust_solver\target\release\t3_exact_solver.exe
```

Result:

- records: `25`
- MC1000 Top1 vs cap200 Top1: `1/25`
- changed Top1: `24`
- avg elapsed: `33004.3ms`
- avg full T2 estimate from cap200: `1099977.0ms` (`18.3m`)
- avg exact regret of source Top1: `1.7459`
- max exact regret of source Top1: `9.2129`

Regenerated cap100 on the same 25-row subset and compared cap100/cap200:

- stable labels:
  `ai/data/hybrid_t1t2_active_20260531/stable_t2_oracle_residual_diverse50_misses25_cap100_cap200_20260602.jsonl`
- unstable rows:
  `ai/data/hybrid_t1t2_active_20260531/stable_t2_oracle_residual_diverse50_misses25_cap100_cap200_20260602.unstable.jsonl`
- common records: `25`
- stable Top1: `20/25`
- cap100-vs-cap200 Top1 changed: `5/25`

Extracted the 24 cap200 miss rows:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.tutor.select_diverse_oracle_inputs --input ai\data\hybrid_t1t2_active_20260531\oracle_inputs_t2_residual_diverse50_cap100_misses25_20260602.jsonl --output ai\data\hybrid_t1t2_active_20260531\oracle_inputs_t2_residual_diverse50_cap200_misses24_20260602.jsonl --turns 2 --limit 24 --max-per-source 1 --indices-file ai\data\hybrid_t1t2_active_20260531\t2_cap200_residual_diverse50_misses25_20260602\t2_oracle_cap200_limit25.misses.jsonl
```

cap500 on the 24 cap200 misses:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.tutor.run_t2_exact_oracle --input ai\data\hybrid_t1t2_active_20260531\oracle_inputs_t2_residual_diverse50_cap200_misses24_20260602.jsonl --out-dir ai\data\hybrid_t1t2_active_20260531\t2_cap500_residual_diverse50_misses24_20260602 --limit 24 --top-n 64 --t2-draw-limit 500 --binary ai\rust_solver\target\release\t3_exact_solver.exe
```

Result:

- records: `24`
- MC1000 Top1 vs cap500 Top1: `2/24`
- changed Top1: `22`
- avg elapsed: `87468.2ms`
- avg full T2 estimate from cap500: `1163912.5ms` (`19.4m`)
- avg exact regret of source Top1: `1.5060`
- max exact regret of source Top1: `8.1629`

Re-ran cap200 on the same 24-row cap500 input and compared cap200/cap500:

- stable labels:
  `ai/data/hybrid_t1t2_active_20260531/stable_t2_oracle_residual_diverse50_misses24_cap200_cap500_20260602.jsonl`
- unstable rows:
  `ai/data/hybrid_t1t2_active_20260531/stable_t2_oracle_residual_diverse50_misses24_cap200_cap500_20260602.unstable.jsonl`
- common records: `24`
- stable Top1: `21/24`
- cap200-vs-cap500 Top1 changed: `3/24`
- stable rows avg score absolute difference: `0.9204`
- stable rows max score absolute difference: `6.6168`
- stable rows avg bust absolute difference: `0.0294`
- stable rows max bust absolute difference: `0.0824`

The cap200/cap500-unstable rows were source lines:

- `946`: cap200 `5h->middle;6c->middle; discard 2h`; cap500
  `2h->middle;5h->middle; discard 6c`
- `545`: cap200 `3c->middle;Qc->bottom; discard 4h`; cap500
  `3c->middle;Qc->middle; discard 4h`
- `15`: cap200 `9d->middle;Ac->top; discard Jc`; cap500
  `Ac->top;Jc->top; discard 9d`

Interpretation:

- In diverse50, cap100 found `25` candidate misses.
- cap200 preserved `24` of those as source-vs-oracle misses.
- cap500 preserved `22` of the cap200 misses.
- cap200/cap500 agreed on Top1 for `21` rows; these are the current best hard
  Top1 active-learning labels from this local sweep.
- The 3 cap200/cap500-unstable rows should not be used as hard labels.
- Even for stable Top1 rows, score movement is large, so this batch is suitable
  for ranking/Top1 training, not EV/bust regression training.

## 2026-06-03 T2 Exact-Stable Label Training Probe

The cap200/cap500-stable T2 labels were converted into candidate-level
reranker data:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.training.convert_action_value_teacher ai\data\hybrid_t1t2_active_20260531\stable_t2_oracle_residual_diverse50_misses24_cap200_cap500_20260602.jsonl --output ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle_diverse50_cap200_cap500_20260603 --turns 2 --state-dim 520 --regular-max-candidates 0
```

Conversion result:

- records: `21`
- candidate samples: `384`
- skipped: `0`
- score mean/std: `+6.444 / 10.678`
- bust mean: `59.7%`
- FL mean: `10.1%`

A reusable checkpoint evaluator was added:

- script: `ai/training/evaluate_action_value_reranker.py`
- input: candidate-level reranker data directory plus a checkpoint
- output: `summary.json`, `summary.md`, and readable `misses.jsonl`
- metrics: TopK recall, TopK rerank regret, score/bust/FL MAE, avoidable
  100% bust selections

Current model strength on the exact-stable 21 T2 rows:

| checkpoint | Top1 | Top3 | Top5 | Top10 | Top15 | Top20 | Top1 regret | avoidable 100% bust |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `active61_t2resid200_mc1000` | 14.3% | 61.9% | 71.4% | 90.5% | 95.2% | 100.0% | 3.983 | 2 |
| `active61_t2resid300_mc1000` | 14.3% | 61.9% | 71.4% | 90.5% | 95.2% | 100.0% | 3.983 | 2 |
| `active61_joint_mc1000` | 14.3% | 57.1% | 71.4% | 90.5% | 95.2% | 100.0% | 3.983 | 2 |
| `t2_current_modelmiss73_mc1000` | 14.3% | 57.1% | 71.4% | 90.5% | 95.2% | 100.0% | 3.983 | 2 |
| `t1t2_broad500_mc50` | 28.6% | 57.1% | 81.0% | 81.0% | 90.5% | 100.0% | 2.518 | 1 |

Interpretation:

- The current T2 models are not strong enough by exact labels.
- Top20 contains the exact-stable best action for this small probe, but Top15
  still misses at least one row.
- MC-trained residual models did not materially improve this exact probe.
- The broad MC50 model improves Top1 on this probe, but worsens Top10/Top15
  safety, so it is not a clean promotion candidate.

A short local fine-tune was run from the broad MC50 checkpoint using only these
21 exact-stable rows:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.training.train_action_value_reranker --data ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle_diverse50_cap200_cap500_20260603 --save-dir ai\models\candidate_runs\t2-stable-oracle21-cap500-ft-from-broad500-20260603\model --init-checkpoint ai\models\candidate_runs\t1t2-broad500-mc50-ft-20260602\model\action_value_best.pt --normalization-source checkpoint --train-turns 2 --epochs 80 --max-seconds 60 --batch-size 128 --lr 0.000005 --score-weight 1.0 --bust-weight 0.15 --fl-weight 0.15 --fl-type-weight 0.10 --ranking-weight 1.0 --ranking-batches-per-epoch 32 --group-batch-size 16 --topk-rank-weight 0.5 --target-topk 3 --selection-metric regret --val-frac 0.05 --device cpu --seed 20260603
```

Exact-probe evaluation after fine-tune:

| checkpoint | Top1 | Top3 | Top5 | Top10 | Top15 | Top20 | Top1 regret | avoidable 100% bust |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `exact21_ft_best` | 95.2% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.001 | 0 |
| `exact21_ft_final` | 95.2% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.001 | 0 |

The only remaining Top1 miss is almost a tie:

- source line: `336`
- teacher: `9h->middle; X2->bottom; discard 7h`
- model: `7h->middle; X2->bottom; discard 9h`
- teacher EV gap: `0.0152`

Conclusion:

- Exact labels do move the model strongly on the corrected T2 failure set.
- This is not enough to claim general improvement because the evaluation set is
  the same 21 rows used for training.
- The next correct step is not speed work; it is to generate more cap500-stable
  T2 labels and split them into train/holdout exact probes.
- Use cap200/cap500-stable rows as hard Top1/ranking labels. Do not use the
  current capped labels as EV/bust regression ground truth until higher-cap or
  full-exact agreement is established.

## 2026-06-03 Next T2 Stable Label Probe

`select_diverse_oracle_inputs.py` now supports excluding already-selected
source groups:

- `--exclude-files`: skip source groups present in prior JSONL files
- `--exclude-rank-groups`: optionally skip rank-abstracted groups too

This was used to create the next diverse T2 input batch without reusing the
previous diverse50 source groups:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.tutor.select_diverse_oracle_inputs --input ai\data\hybrid_t1t2_active_20260531\teacher_t2_residual_neighbors_w120_mc1000_20260601.jsonl --output ai\data\hybrid_t1t2_active_20260531\oracle_inputs_t2_residual_diverse50_next_20260603.jsonl --turns 2 --limit 50 --max-per-source 1 --exclude-files ai\data\hybrid_t1t2_active_20260531\oracle_inputs_t2_residual_diverse50_20260602.jsonl
```

Selection result:

- selected: `50`
- excluded previous source groups: `50`
- unique source groups: `50`
- unique rank groups: `50`

The first five rows were probed at cap100:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.tutor.run_t2_exact_oracle --input ai\data\hybrid_t1t2_active_20260531\oracle_inputs_t2_residual_diverse50_next_20260603.jsonl --out-dir ai\data\hybrid_t1t2_active_20260531\t2_cap100_residual_diverse50_next_probe5_20260603 --limit 5 --top-n 64 --t2-draw-limit 100 --binary ai\rust_solver\target\release\t3_exact_solver.exe
```

Result:

- records: `5`
- MC1000 Top1 vs cap100 Top1: `3/5`
- changed Top1: `2/5`
- avg cap100 elapsed: `19359.1ms`
- avg full T2 estimate from cap100: `1382240.5ms` (`23.0m`)
- avg source Top1 regret: `0.4517`
- max source Top1 regret: `2.2587`

The two cap100 misses were promoted through cap200 and cap500:

| cap | rows | changed vs source | avg elapsed | avg source Top1 regret |
|---:|---:|---:|---:|---:|
| 100 | 2 | 2 | 25422.5ms | 1.1294 |
| 200 | 2 | 2 | 50082.2ms | 0.8192 |
| 500 | 2 | 2 | 123149.3ms | 0.8141 |

cap100/cap200 comparison:

- stable Top1: `2/2`
- unstable: `0`
- output:
  `ai/data/hybrid_t1t2_active_20260531/stable_t2_oracle_residual_diverse50_next_probe5_misses2_cap100_cap200_20260603.jsonl`

cap200/cap500 comparison:

- stable Top1: `2/2`
- unstable: `0`
- output:
  `ai/data/hybrid_t1t2_active_20260531/stable_t2_oracle_residual_diverse50_next_probe5_misses2_cap200_cap500_20260603.jsonl`

The two new cap200/cap500-stable rows were converted to a holdout reranker set:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.training.convert_action_value_teacher ai\data\hybrid_t1t2_active_20260531\stable_t2_oracle_residual_diverse50_next_probe5_misses2_cap200_cap500_20260603.jsonl --output ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle_next_probe2_cap200_cap500_20260603 --turns 2 --state-dim 520 --regular-max-candidates 0
```

Holdout conversion:

- records: `2`
- candidate samples: `48`
- skipped: `0`

Model evaluation on this new holdout2:

| checkpoint | Top1 | Top3 | Top10 | Top20 | score MAE | bust MAE | FL MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| `holdout2_active61_t2resid200` | 100.0% | 100.0% | 100.0% | 100.0% | 4.787 | 0.129 | 0.043 |
| `holdout2_broad500_mc50` | 100.0% | 100.0% | 100.0% | 100.0% | 6.197 | 0.102 | 0.068 |
| `holdout2_exact21_ft_final` | 100.0% | 100.0% | 100.0% | 100.0% | 3.449 | 0.094 | 0.047 |

Interpretation:

- The new holdout is too small to prove general improvement.
- It does show that exact21 fine-tuning did not break these two new stable rows.
- The exact21 fine-tuned model improved score MAE on holdout2 versus the broad
  MC50 starting checkpoint.
- Continue by running cap100 on more of the `next50` batch, then confirm misses
  with cap200/cap500 before adding them to the hard-label pool.

## 2026-06-03 Next50 Rows 5-14 Stable Label Probe

The next ten rows from `oracle_inputs_t2_residual_diverse50_next_20260603.jsonl`
were isolated:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.tutor.select_diverse_oracle_inputs --input ai\data\hybrid_t1t2_active_20260531\oracle_inputs_t2_residual_diverse50_next_20260603.jsonl --output ai\data\hybrid_t1t2_active_20260531\oracle_inputs_t2_residual_diverse50_next_rows5_14_20260603.jsonl --turns 2 --limit 10 --max-per-source 1 --indices 5,6,7,8,9,10,11,12,13,14
```

cap100 on rows 5-14:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.tutor.run_t2_exact_oracle --input ai\data\hybrid_t1t2_active_20260531\oracle_inputs_t2_residual_diverse50_next_rows5_14_20260603.jsonl --out-dir ai\data\hybrid_t1t2_active_20260531\t2_cap100_residual_diverse50_next_rows5_14_20260603 --limit 10 --top-n 64 --t2-draw-limit 100 --binary ai\rust_solver\target\release\t3_exact_solver.exe
```

Result:

- records: `10`
- MC1000 Top1 vs cap100 Top1: `7/10`
- changed Top1: `3/10`
- avg cap100 elapsed: `24634.7ms`
- avg full T2 estimate from cap100: `1629793.9ms` (`27.2m`)
- avg source Top1 regret: `0.9939`
- max source Top1 regret: `9.9389`

The three cap100 misses were isolated and promoted through cap200/cap500:

| cap | rows | changed vs source | avg elapsed | avg source Top1 regret | max source Top1 regret |
|---:|---:|---:|---:|---:|---:|
| 200 | 3 | 3 | 58250.7ms | 2.6150 | 7.8450 |
| 500 | 3 | 3 | 137615.8ms | 2.7884 | 8.3653 |

cap200/cap500 comparison:

- stable Top1: `3/3`
- unstable: `0`
- output:
  `ai/data/hybrid_t1t2_active_20260531/stable_t2_oracle_residual_diverse50_next_rows5_14_misses3_cap200_cap500_20260603.jsonl`

The three new stable rows were converted:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.training.convert_action_value_teacher ai\data\hybrid_t1t2_active_20260531\stable_t2_oracle_residual_diverse50_next_rows5_14_misses3_cap200_cap500_20260603.jsonl --output ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle_next_rows5_14_misses3_cap200_cap500_20260603 --turns 2 --state-dim 520 --regular-max-candidates 0
```

Conversion:

- records: `3`
- candidate samples: `75`
- skipped: `0`

Model evaluation on this new holdout3:

| checkpoint | Top1 | Top3 | Top10 | Top20 | score MAE | bust MAE | FL MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| `holdout3_active61_t2resid200` | 100.0% | 100.0% | 100.0% | 100.0% | 4.977 | 0.303 | 0.033 |
| `holdout3_broad500_mc50` | 100.0% | 100.0% | 100.0% | 100.0% | 5.799 | 0.226 | 0.037 |
| `holdout3_exact21_ft_final` | 100.0% | 100.0% | 100.0% | 100.0% | 2.868 | 0.206 | 0.041 |

Interpretation:

- This adds `3` more cap200/cap500-stable T2 hard labels.
- The current hard-label pool is now `26` T2 rows (`21 + 2 + 3`).
- The exact21 fine-tuned model still does not regress Top1 on this new small
  holdout and has the lowest score MAE among the compared checkpoints.
- Continue expanding rows 15-24 next; do not spend time on speed optimization
  until the exact-stable holdout is meaningfully larger.

## 2026-06-03 Next50 Rows 15-24 Stable Label Probe

Rows 15-24 from `oracle_inputs_t2_residual_diverse50_next_20260603.jsonl` were
isolated and evaluated at cap100:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.tutor.select_diverse_oracle_inputs --input ai\data\hybrid_t1t2_active_20260531\oracle_inputs_t2_residual_diverse50_next_20260603.jsonl --output ai\data\hybrid_t1t2_active_20260531\oracle_inputs_t2_residual_diverse50_next_rows15_24_20260603.jsonl --turns 2 --limit 10 --max-per-source 1 --indices 15,16,17,18,19,20,21,22,23,24
```

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.tutor.run_t2_exact_oracle --input ai\data\hybrid_t1t2_active_20260531\oracle_inputs_t2_residual_diverse50_next_rows15_24_20260603.jsonl --out-dir ai\data\hybrid_t1t2_active_20260531\t2_cap100_residual_diverse50_next_rows15_24_20260603 --limit 10 --top-n 64 --t2-draw-limit 100 --binary ai\rust_solver\target\release\t3_exact_solver.exe
```

cap100 result:

- records: `10`
- MC1000 Top1 vs cap100 Top1: `4/10`
- changed Top1: `6/10`
- avg cap100 elapsed: `21283.4ms`
- avg full T2 estimate from cap100: `1361070.8ms` (`22.7m`)
- avg source Top1 regret: `1.5104`
- max source Top1 regret: `6.9166`

The six cap100 misses were promoted through cap200/cap500:

| cap | rows | changed vs source | avg elapsed | avg source Top1 regret | max source Top1 regret |
|---:|---:|---:|---:|---:|---:|
| 200 | 6 | 6 | 36631.9ms | 1.5898 | 5.9936 |
| 500 | 6 | 6 | 86356.3ms | 0.8682 | 2.1026 |

cap200/cap500 comparison:

- stable Top1: `5/6`
- unstable: `1/6`
- output:
  `ai/data/hybrid_t1t2_active_20260531/stable_t2_oracle_residual_diverse50_next_rows15_24_misses6_cap200_cap500_20260603.jsonl`

The five new stable rows were converted:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.training.convert_action_value_teacher ai\data\hybrid_t1t2_active_20260531\stable_t2_oracle_residual_diverse50_next_rows15_24_misses6_cap200_cap500_20260603.jsonl --output ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle_next_rows15_24_misses6_cap200_cap500_20260603 --turns 2 --state-dim 520 --regular-max-candidates 0
```

Conversion:

- records: `5`
- candidate samples: `96`
- skipped: `0`

Model evaluation on this holdout5:

| checkpoint | Top1 | Top3 | Top5 | Top10 | Top20 | Top1 regret | score MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| `holdout5_active61_t2resid200` | 60.0% | 60.0% | 80.0% | 100.0% | 100.0% | 0.603 | 5.052 |
| `holdout5_broad500_mc50` | 40.0% | 60.0% | 60.0% | 100.0% | 100.0% | 1.297 | 6.418 |
| `holdout5_exact21_ft_final` | 40.0% | 40.0% | 80.0% | 100.0% | 100.0% | 0.872 | 4.019 |

Interpretation:

- This holdout exposes that exact21-only fine-tuning is not enough.
- active61_t2resid200 still has better Top1 on these five rows.
- exact21 fine-tuning improves score MAE but not ranking enough.
- The next model should train on the accumulated exact-stable pool, not only
  the original 21 rows.

## 2026-06-03 Exact31 Merge And Training Probe

`merge_action_value_reranker_data.py` was added to merge candidate-level
reranker directories while offsetting `group_ids`.

Merged exact-stable T2 data:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.training.merge_action_value_reranker_data --dirs ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle_diverse50_cap200_cap500_20260603 ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle_next_probe2_cap200_cap500_20260603 ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle_next_rows5_14_misses3_cap200_cap500_20260603 ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle_next_rows15_24_misses6_cap200_cap500_20260603 --output ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle31_cap200_cap500_20260603
```

Merged data:

- records/groups: `31`
- candidate samples: `603`
- turns: T2 only
- score mean/std: `+6.764 / 9.818`
- bust mean: `53.8%`
- FL mean: `10.9%`

Fine-tune from `active61_t2resid200`:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.training.train_action_value_reranker --data ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle31_cap200_cap500_20260603 --save-dir ai\models\candidate_runs\t2-stable-oracle31-cap500-ft-from-t2resid200-20260603\model --init-checkpoint ai\models\candidate_runs\tutor-route10-active61-t2resid200-mc1000-suit24-ft-t2only-lr1e7-20260601\model\action_value_best.pt --normalization-source checkpoint --train-turns 2 --epochs 80 --max-seconds 90 --batch-size 128 --lr 0.000003 --score-weight 1.0 --bust-weight 0.15 --fl-weight 0.15 --fl-type-weight 0.10 --ranking-weight 1.0 --ranking-batches-per-epoch 32 --group-batch-size 16 --topk-rank-weight 0.5 --target-topk 3 --selection-metric regret --val-frac 0.2 --device cpu --seed 20260603
```

Evaluation on exact31 itself:

| checkpoint | Top1 | Top3 | Top5 | Top10 | Top15 | Top20 | Top1 regret | score MAE | avoidable 100% bust |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `active61_t2resid200` | 35.5% | 67.7% | 77.4% | 93.5% | 96.8% | 100.0% | 2.796 | 5.756 | 2 |
| `exact31_ft_best` | 64.5% | 96.8% | 96.8% | 100.0% | 100.0% | 100.0% | 1.649 | 4.540 | 1 |
| `exact31_ft_final` | 80.6% | 96.8% | 96.8% | 100.0% | 100.0% | 100.0% | 1.211 | 2.714 | 0 |

Interpretation:

- Exact31 training strongly improves the corrected exact-stable set.
- This is still not promotion evidence because exact31 was used for training.
- The next decisive check is to generate rows25-34 cap500-stable labels and use
  them as a fresh holdout against `exact31_ft_final`.

## 2026-06-03 Next50 Rows 25-34 Stable Label Probe

Rows 25-34 from `oracle_inputs_t2_residual_diverse50_next_20260603.jsonl` were
isolated and evaluated at cap100:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.tutor.select_diverse_oracle_inputs --input ai\data\hybrid_t1t2_active_20260531\oracle_inputs_t2_residual_diverse50_next_20260603.jsonl --output ai\data\hybrid_t1t2_active_20260531\oracle_inputs_t2_residual_diverse50_next_rows25_34_20260603.jsonl --turns 2 --limit 10 --max-per-source 1 --indices 25,26,27,28,29,30,31,32,33,34
```

cap100 result:

- records: `10`
- MC1000 Top1 vs cap100 Top1: `7/10`
- changed Top1: `3/10`
- avg cap100 elapsed: `18127.5ms`
- avg source Top1 regret: `1.4560`
- max source Top1 regret: `12.9191`

The three cap100 misses were promoted through cap200/cap500:

| cap | rows | changed vs source | avg elapsed | avg source Top1 regret | max source Top1 regret |
|---:|---:|---:|---:|---:|---:|
| 200 | 3 | 3 | 33448.8ms | 1.9241 | 4.7203 |
| 500 | 3 | 2 | 80986.4ms | 0.1191 | 0.3574 |

cap200/cap500 comparison:

- stable Top1: `1/3`
- unstable: `2/3`
- output:
  `ai/data/hybrid_t1t2_active_20260531/stable_t2_oracle_residual_diverse50_next_rows25_34_misses3_cap200_cap500_20260603.jsonl`

The one new stable row was converted:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.training.convert_action_value_teacher ai\data\hybrid_t1t2_active_20260531\stable_t2_oracle_residual_diverse50_next_rows25_34_misses3_cap200_cap500_20260603.jsonl --output ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle_next_rows25_34_misses3_cap200_cap500_20260603 --turns 2 --state-dim 520 --regular-max-candidates 0
```

Conversion:

- records: `1`
- candidate samples: `24`
- skipped: `0`

Model evaluation on this fresh holdout1:

| checkpoint | Top1 | Top3 | Top5 | Top10 | Top20 | Top1 regret | score MAE | avoidable 100% bust |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `holdout1_rows25_34_active61_t2resid200` | 0.0% | 0.0% | 100.0% | 100.0% | 100.0% | 0.357 | 2.638 | 0 |
| `holdout1_rows25_34_broad500_mc50` | 0.0% | 0.0% | 0.0% | 100.0% | 100.0% | 0.357 | 3.533 | 0 |
| `holdout1_rows25_34_exact31_ft_best` | 0.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.357 | 2.787 | 0 |
| `holdout1_rows25_34_exact31_ft_final` | 0.0% | 0.0% | 100.0% | 100.0% | 100.0% | 0.357 | 1.029 | 0 |

The stable holdout decision:

- board: top `Th`, middle `8c 2c`, bottom `3s 5c Jc Jh`
- opponent board: top `5h Ac Ad`, middle `6c 8d`, bottom `2d 7s`
- dealt: `Ah Ts Kc`
- oracle best: `Kc->top; Ts->middle; discard Ah`
- exact31 final model best: `Ah->middle; Kc->top; discard Ts`
- oracle EV gap: `0.357`

Interpretation:

- This is a very small fresh holdout, but it is useful because it was not used
  for exact31 training.
- exact31 final has the best score MAE on this row, but still misses Top1.
- exact31 best gets the oracle action into Top3, while exact31 final only gets
  it into Top5.
- cap200-only labels are not reliable enough here: only `1/3` cap200/cap500
  promoted misses kept the same Top1.
- The correct next step is still oracle-first: expand cap500-stable labels
  before optimizing for speed or promoting a model checkpoint.

## 2026-06-03 Next50 Rows 35-44 Stable Label Probe

Rows 35-44 from `oracle_inputs_t2_residual_diverse50_next_20260603.jsonl` were
isolated and evaluated at cap100:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.tutor.select_diverse_oracle_inputs --input ai\data\hybrid_t1t2_active_20260531\oracle_inputs_t2_residual_diverse50_next_20260603.jsonl --output ai\data\hybrid_t1t2_active_20260531\oracle_inputs_t2_residual_diverse50_next_rows35_44_20260603.jsonl --turns 2 --limit 10 --max-per-source 1 --indices 35,36,37,38,39,40,41,42,43,44
```

cap100 result:

- records: `10`
- MC1000 Top1 vs cap100 Top1: `5/10`
- changed Top1: `5/10`
- avg cap100 elapsed: `18028.1ms`
- avg source Top1 regret: `0.1115`
- max source Top1 regret: `0.5850`

The five cap100 misses were promoted through cap200/cap500:

| cap | rows | changed vs source | avg elapsed | avg source Top1 regret | max source Top1 regret |
|---:|---:|---:|---:|---:|---:|
| 200 | 5 | 4 | 42409.2ms | 0.1684 | 0.6885 |
| 500 | 5 | 4 | 103983.0ms | 0.2250 | 0.8850 |

cap200/cap500 comparison:

- stable Top1: `5/5`
- unstable: `0/5`
- output:
  `ai/data/hybrid_t1t2_active_20260531/stable_t2_oracle_residual_diverse50_next_rows35_44_misses5_cap200_cap500_20260603.jsonl`

The five new stable rows were converted:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.training.convert_action_value_teacher ai\data\hybrid_t1t2_active_20260531\stable_t2_oracle_residual_diverse50_next_rows35_44_misses5_cap200_cap500_20260603.jsonl --output ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle_next_rows35_44_misses5_cap200_cap500_20260603 --turns 2 --state-dim 520 --regular-max-candidates 0
```

Conversion:

- records: `5`
- candidate samples: `111`
- skipped: `0`

Model evaluation on this fresh holdout5:

| checkpoint | Top1 | Top3 | Top5 | Top10 | Top20 | Top1 regret | score MAE | avoidable 100% bust |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `holdout5_rows35_44_active61_t2resid200` | 40.0% | 40.0% | 40.0% | 100.0% | 100.0% | 0.685 | 2.550 | 1 |
| `holdout5_rows35_44_broad500_mc50` | 20.0% | 40.0% | 80.0% | 100.0% | 100.0% | 3.684 | 4.000 | 0 |
| `holdout5_rows35_44_exact31_ft_best` | 40.0% | 40.0% | 60.0% | 100.0% | 100.0% | 0.683 | 2.572 | 0 |
| `holdout5_rows35_44_exact31_ft_final` | 40.0% | 60.0% | 60.0% | 100.0% | 100.0% | 0.226 | 1.796 | 1 |

Interpretation:

- These five rows are useful fresh holdout evidence because they were not used
  for exact31 training.
- exact31 final improves score MAE and Top1 regret, but still only has `40%`
  Top1 and `60%` Top3 on this holdout.
- All compared checkpoints keep the oracle action within Top10 here, but this
  is not enough for the user's target of playing in real time with the optimal
  answer.
- The next model should merge these five rows into the exact-stable pool and
  then evaluate on a new fresh holdout, rather than treating this as a solved
  speed problem.

## 2026-06-03 Exact36 Merge And Rows45-49 Holdout

The rows35-44 five stable rows were merged back into the exact-stable T2 pool:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.training.merge_action_value_reranker_data --dirs ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle31_cap200_cap500_20260603 ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle_next_rows35_44_misses5_cap200_cap500_20260603 --output ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle36_cap200_cap500_20260603
```

Merged data:

- records/groups: `36`
- candidate samples: `714`
- turns: T2 only
- score mean/std: `+5.970 / 9.295`
- bust mean: `56.5%`
- FL mean: `9.7%`

Fine-tune from exact31 final:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.training.train_action_value_reranker --data ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle36_cap200_cap500_20260603 --save-dir ai\models\candidate_runs\t2-stable-oracle36-cap500-ft-from-exact31-20260603\model --init-checkpoint ai\models\candidate_runs\t2-stable-oracle31-cap500-ft-from-t2resid200-20260603\model\action_value_final.pt --normalization-source checkpoint --train-turns 2 --epochs 80 --max-seconds 90 --batch-size 128 --lr 0.000002 --score-weight 1.0 --bust-weight 0.15 --fl-weight 0.15 --fl-type-weight 0.10 --ranking-weight 1.0 --ranking-batches-per-epoch 32 --group-batch-size 16 --topk-rank-weight 0.6 --target-topk 3 --selection-metric regret --val-frac 0.2 --device cpu --seed 20260603
```

Rows 45-49 from `oracle_inputs_t2_residual_diverse50_next_20260603.jsonl` were
then used as the next fresh holdout.

cap100 result:

- records: `5`
- MC1000 Top1 vs cap100 Top1: `3/5`
- changed Top1: `2/5`
- avg cap100 elapsed: `19889.4ms`
- avg source Top1 regret: `0.1884`
- max source Top1 regret: `0.9387`

The two cap100 misses were promoted through cap200/cap500:

| cap | rows | changed vs source | avg elapsed | avg source Top1 regret | max source Top1 regret |
|---:|---:|---:|---:|---:|---:|
| 200 | 2 | 2 | 30248.7ms | 0.2575 | 0.5024 |
| 500 | 2 | 2 | 70543.3ms | 0.0071 | 0.0142 |

cap200/cap500 comparison:

- stable Top1: `1/2`
- unstable: `1/2`
- output:
  `ai/data/hybrid_t1t2_active_20260531/stable_t2_oracle_residual_diverse50_next_rows45_49_misses2_cap200_cap500_20260603.jsonl`

The one new stable row was converted:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.training.convert_action_value_teacher ai\data\hybrid_t1t2_active_20260531\stable_t2_oracle_residual_diverse50_next_rows45_49_misses2_cap200_cap500_20260603.jsonl --output ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle_next_rows45_49_misses2_cap200_cap500_20260603 --turns 2 --state-dim 520 --regular-max-candidates 0
```

Conversion:

- records: `1`
- candidate samples: `12`
- skipped: `0`

Model evaluation on this fresh holdout1:

| checkpoint | Top1 | Top3 | Top5 | Top10 | Top20 | Top1 regret | score MAE | avoidable 100% bust |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `holdout1_rows45_49_active61_t2resid200` | 0.0% | 0.0% | 0.0% | 100.0% | 100.0% | 0.418 | 3.057 | 0 |
| `holdout1_rows45_49_exact31_ft_final` | 0.0% | 0.0% | 0.0% | 100.0% | 100.0% | 0.418 | 1.722 | 0 |
| `holdout1_rows45_49_exact36_ft_best` | 0.0% | 0.0% | 0.0% | 100.0% | 100.0% | 0.407 | 2.338 | 0 |
| `holdout1_rows45_49_exact36_ft_final` | 0.0% | 0.0% | 0.0% | 100.0% | 100.0% | 0.418 | 1.214 | 0 |

Interpretation:

- exact36 improves score MAE on this one row, but still misses Top1/Top3/Top5.
- The oracle action remains in Top10 for all compared checkpoints, which is
  useful only as a pruning signal.
- This is another indication that the current model is not an optimal-answer
  model yet; it is a candidate-pruning model.
- The next data-generation pass should target exactly these Top10-only misses
  and the cap-instability rows, because they are the cases blocking Top3/Top1.

## 2026-06-03 Exact38 Candidate Checkpoint

After the rows25-34 and rows45-49 holdouts were evaluated, their stable rows
were added back into the exact-stable training pool:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.training.merge_action_value_reranker_data --dirs ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle31_cap200_cap500_20260603 ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle_next_rows25_34_misses3_cap200_cap500_20260603 ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle_next_rows35_44_misses5_cap200_cap500_20260603 ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle_next_rows45_49_misses2_cap200_cap500_20260603 --output ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle38_cap200_cap500_20260603
```

Merged data:

- records/groups: `38`
- candidate samples: `750`
- turns: T2 only
- score mean/std: `+5.726 / 9.139`
- bust mean: `56.1%`
- FL mean: `9.3%`

Fine-tune from exact36 final:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.training.train_action_value_reranker --data ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle38_cap200_cap500_20260603 --save-dir ai\models\candidate_runs\t2-stable-oracle38-cap500-ft-from-exact36-20260603\model --init-checkpoint ai\models\candidate_runs\t2-stable-oracle36-cap500-ft-from-exact31-20260603\model\action_value_final.pt --normalization-source checkpoint --train-turns 2 --epochs 80 --max-seconds 90 --batch-size 128 --lr 0.000002 --score-weight 1.0 --bust-weight 0.15 --fl-weight 0.15 --fl-type-weight 0.10 --ranking-weight 1.1 --ranking-batches-per-epoch 32 --group-batch-size 16 --topk-rank-weight 0.7 --target-topk 3 --selection-metric regret --val-frac 0.2 --device cpu --seed 20260603
```

Saved checkpoint:

- `ai/models/candidate_runs/t2-stable-oracle38-cap500-ft-from-exact36-20260603/model/action_value_best.pt`
- `ai/models/candidate_runs/t2-stable-oracle38-cap500-ft-from-exact36-20260603/model/action_value_final.pt`

Interpretation:

- exact38 is a candidate checkpoint for the next fresh holdout, not promotion
  evidence.
- The next holdout should not come from the rows already merged into exact38.
- Prior fresh holdouts show the current model family is still mainly useful for
  Top10 pruning. Top1/Top3 requires more exact-stable labels, especially from
  Top10-only misses and cap-instability cases.

## 2026-06-03 Next2 Rows 0-9 And Exact41 Probe

A third diverse T2 batch was selected from the residual-neighbor teacher pool,
excluding the first two 50-row batches by source and rank abstraction:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.tutor.select_diverse_oracle_inputs --input ai\data\hybrid_t1t2_active_20260531\teacher_t2_residual_neighbors_w120_mc1000_20260601.jsonl --output ai\data\hybrid_t1t2_active_20260531\oracle_inputs_t2_residual_diverse50_next2_20260603.jsonl --turns 2 --limit 50 --max-per-source 1 --exclude-files ai\data\hybrid_t1t2_active_20260531\oracle_inputs_t2_residual_diverse50_20260602.jsonl ai\data\hybrid_t1t2_active_20260531\oracle_inputs_t2_residual_diverse50_next_20260603.jsonl --exclude-rank-groups
```

Selection:

- selected: `50`
- excluded source groups: `100`
- excluded rank groups: `100`
- unique source/rank groups: `50 / 50`

Rows 0-9 were evaluated at cap100:

- records: `10`
- MC1000 Top1 vs cap100 Top1: `6/10`
- changed Top1: `4/10`
- avg cap100 elapsed: `18387.7ms`
- avg source Top1 regret: `0.7048`
- max source Top1 regret: `3.4686`

The four cap100 misses were promoted through cap200/cap500:

| cap | rows | changed vs source | avg elapsed | avg source Top1 regret | max source Top1 regret |
|---:|---:|---:|---:|---:|---:|
| 200 | 4 | 2 | 27525.0ms | 1.3186 | 4.1551 |
| 500 | 4 | 3 | 69299.1ms | 2.1322 | 6.0828 |

cap200/cap500 comparison:

- stable Top1: `3/4`
- unstable: `1/4`
- output:
  `ai/data/hybrid_t1t2_active_20260531/stable_t2_oracle_residual_diverse50_next2_rows0_9_misses4_cap200_cap500_20260603.jsonl`

Model evaluation on this fresh holdout3:

| checkpoint | Top1 | Top3 | Top5 | Top10 | Top15 | Top1 regret | score MAE | note |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `exact31_ft_final` | 33.3% | 33.3% | 33.3% | 66.7% | 100.0% | 0.122 | 6.068 | old exact-stable model |
| `exact36_ft_final` | 33.3% | 33.3% | 33.3% | 66.7% | 100.0% | 0.122 | 5.610 | rows35-44 merged |
| `exact38_ft_best` | 33.3% | 33.3% | 33.3% | 66.7% | 100.0% | 2.268 | 5.555 | rows25-49 merged |
| `exact38_ft_final` | 33.3% | 33.3% | 33.3% | 66.7% | 100.0% | 2.268 | 5.531 | rows25-49 merged |

The exact38 final misses included:

- board top `2s As`, middle `Ks`, bottom `4d 5c 6s 4c`
- dealt `3h 7d Qd`
- oracle best `3h->middle; Qd->middle; discard 7d`
- exact38 final best `7d->middle; Qd->top; discard 3h`
- regret `6.382`
- teacher rank by model: `9`

Interpretation:

- exact38 did not improve this fresh holdout; it still missed Top10 on one high
  regret row and only reached Top15 recall.
- This is a strong signal that the model is not yet safe as a Top10-only
  pruning model on new exact-stable T2 rows.

After this evaluation, the three stable rows were merged into exact41:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.training.merge_action_value_reranker_data --dirs ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle38_cap200_cap500_20260603 ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle_next2_rows0_9_misses4_cap200_cap500_20260603 --output ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle41_cap200_cap500_20260603
```

Merged data:

- records/groups: `41`
- candidate samples: `807`
- turns: T2 only
- score mean/std: `+5.900 / 9.081`
- bust mean: `55.0%`
- FL mean: `9.2%`

Fine-tune from exact38 final saved:

- `ai/models/candidate_runs/t2-stable-oracle41-cap500-ft-from-exact38-20260603/model/action_value_best.pt`
- `ai/models/candidate_runs/t2-stable-oracle41-cap500-ft-from-exact38-20260603/model/action_value_final.pt`

Rows 10-19 from `oracle_inputs_t2_residual_diverse50_next2_20260603.jsonl`
were then used as the next fresh holdout.

cap100 result:

- records: `10`
- MC1000 Top1 vs cap100 Top1: `7/10`
- changed Top1: `3/10`
- avg cap100 elapsed: `18496.2ms`
- avg source Top1 regret: `0.2310`
- max source Top1 regret: `1.3755`

The three cap100 misses were promoted through cap200/cap500:

| cap | rows | changed vs source | avg elapsed | avg source Top1 regret | max source Top1 regret |
|---:|---:|---:|---:|---:|---:|
| 200 | 3 | 3 | 31399.1ms | 0.3274 | 0.4490 |
| 500 | 3 | 3 | 74077.7ms | 0.3291 | 0.4251 |

cap200/cap500 comparison:

- stable Top1: `3/3`
- unstable: `0/3`
- output:
  `ai/data/hybrid_t1t2_active_20260531/stable_t2_oracle_residual_diverse50_next2_rows10_19_misses3_cap200_cap500_20260603.jsonl`

Model evaluation on this fresh holdout3:

| checkpoint | Top1 | Top3 | Top5 | Top10 | Top20 | Top1 regret | score MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| `exact31_ft_final` | 66.7% | 100.0% | 100.0% | 100.0% | 100.0% | 0.131 | 1.057 |
| `exact38_ft_final` | 33.3% | 100.0% | 100.0% | 100.0% | 100.0% | 0.187 | 1.672 |
| `exact41_ft_best` | 33.3% | 100.0% | 100.0% | 100.0% | 100.0% | 0.187 | 1.416 |
| `exact41_ft_final` | 33.3% | 100.0% | 100.0% | 100.0% | 100.0% | 0.187 | 1.520 |

Interpretation:

- exact41 preserved Top3/Top10 on this fresh holdout, but did not improve Top1.
- The older exact31 model had better Top1 on this small holdout.
- More labels are still needed, but simple incremental fine-tuning is not
  monotonically improving Top1.
- The practical near-term target should be reliable Top3/Top5 recall plus
  exact refinement inside that shortlist. Top1-only model promotion is not
  justified by current evidence.

## 2026-06-03 Ensemble And Candidate-Pool Probe

`evaluate_action_value_reranker.py` now supports checkpoint ensembling:

- `--checkpoint`: first checkpoint
- `--checkpoints`: additional checkpoints
- `--weights`: optional comma-separated weights; defaults to uniform

A new pool evaluator was added:

- `ai/training/evaluate_action_value_candidate_pool.py`
- It evaluates the union of each model's TopK candidates.
- This matches the intended production pattern better than score averaging:
  keep a safe candidate pool, then exact-refine inside it.

Fresh next2 rows0-19 stable holdout was merged:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.training.merge_action_value_reranker_data --dirs ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle_next2_rows0_9_misses4_cap200_cap500_20260603 ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle_next2_rows10_19_misses3_cap200_cap500_20260603 --output ai\data\hybrid_t1t2_active_20260531\reranker_t2_fresh_next2_rows0_19_stable6_cap200_cap500_20260603
```

Merged fresh holdout:

- records/groups: `6`
- candidate samples: `117`
- turns: T2 only
- score mean/std: `+4.799 / 6.627`
- bust mean: `43.7%`
- FL mean: `4.3%`

Single-model and score-average evaluation on this fresh6:

| checkpoint | Top1 | Top3 | Top5 | Top10 | Top15 | Top1 regret | score MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| `exact31_ft_final` | 50.0% | 66.7% | 66.7% | 83.3% | 100.0% | 0.126 | 3.499 |
| `exact38_ft_final` | 33.3% | 66.7% | 66.7% | 83.3% | 100.0% | 1.228 | 3.552 |
| `exact31+exact38 uniform score average` | 33.3% | 66.7% | 66.7% | 83.3% | 100.0% | 0.155 | 3.463 |

Candidate-pool union evaluation with `exact31_ft_final` and `exact38_ft_final`:

| per-model K | recall | rerank regret | avg pool | max pool |
|---:|---:|---:|---:|---:|
| 1 | 50.0% | 0.126 | 1.5 | 2 |
| 3 | 66.7% | 0.061 | 3.5 | 4 |
| 5 | 66.7% | 0.061 | 5.5 | 7 |
| 10 | 83.3% | 0.000 | 11.0 | 13 |
| 15 | 100.0% | 0.000 | 15.0 | 17 |

Additional split-holdout checks:

- rows0-9: `exact31+exact38` union Top15 reached `100%`, while Top10 was
  only `66.7%`.
- rows10-19: `exact31+exact38`, `exact31+exact41`, and
  `exact31+exact38+exact41` all reached Top3/Top10 `100%`.

Interpretation:

- Simple score averaging does not fix the high-regret Top10 miss.
- Multi-model TopK union helps expose candidate diversity but still needs
  per-model Top15 on the current fresh6 to guarantee oracle retention.
- The current evidence does not support Top10 pruning as safe for T2.
- Near-term runtime design should use model Top15/union-pool into exact
  refinement, while continuing to collect exact-stable labels from Top10-miss
  and cap-instability rows.

## 2026-06-03 Next2 Rows 20-29 TopK Safety Probe

Rows 20-29 from `oracle_inputs_t2_residual_diverse50_next2_20260603.jsonl` were
used as the next unused T2 holdout slice.

cap100 result:

- records: `10`
- MC1000 Top1 vs cap100 Top1: `5/10`
- changed Top1: `5/10`
- avg cap100 elapsed: `17920.0ms`
- avg source Top1 regret: `0.1405`
- max source Top1 regret: `0.6142`

The five cap100 misses were promoted through cap200/cap500:

| cap | rows | changed vs source | avg elapsed | avg source Top1 regret | max source Top1 regret |
|---:|---:|---:|---:|---:|---:|
| 200 | 5 | 4 | 28517.7ms | 0.1852 | 0.6041 |
| 500 | 5 | 4 | 70056.0ms | 0.1814 | 0.5942 |

cap200/cap500 comparison:

- stable Top1: `5/5`
- unstable: `0/5`
- output:
  `ai/data/hybrid_t1t2_active_20260531/stable_t2_oracle_residual_diverse50_next2_rows20_29_misses5_cap200_cap500_20260603.jsonl`

The five stable rows were converted:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.training.convert_action_value_teacher ai\data\hybrid_t1t2_active_20260531\stable_t2_oracle_residual_diverse50_next2_rows20_29_misses5_cap200_cap500_20260603.jsonl --output ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle_next2_rows20_29_misses5_cap200_cap500_20260603 --turns 2 --state-dim 520 --regular-max-candidates 0
```

Model evaluation on this fresh holdout5:

| checkpoint | Top1 | Top3 | Top5 | Top10 | Top15 | Top1 regret | score MAE | avoidable 100% bust |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `exact31_ft_final` | 20.0% | 60.0% | 60.0% | 100.0% | 100.0% | 0.573 | 3.625 | 1 |
| `exact41_ft_final` | 20.0% | 60.0% | 60.0% | 100.0% | 100.0% | 0.572 | 3.265 | 0 |
| `exact31+exact41 uniform score average` | 20.0% | 40.0% | 40.0% | 100.0% | 100.0% | 0.572 | 3.440 | 0 |

Candidate-pool union evaluation with `exact31_ft_final` and `exact41_ft_final`:

| per-model K | recall | rerank regret | avg pool | max pool |
|---:|---:|---:|---:|---:|
| 1 | 20.0% | 0.572 | 1.2 | 2 |
| 3 | 80.0% | 0.036 | 3.8 | 5 |
| 5 | 80.0% | 0.009 | 6.2 | 7 |
| 10 | 100.0% | 0.000 | 10.4 | 11 |
| 15 | 100.0% | 0.000 | 14.4 | 17 |

Rows0-29 aggregate fresh holdout was also merged:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.training.merge_action_value_reranker_data --dirs ai\data\hybrid_t1t2_active_20260531\reranker_t2_fresh_next2_rows0_19_stable6_cap200_cap500_20260603 ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle_next2_rows20_29_misses5_cap200_cap500_20260603 --output ai\data\hybrid_t1t2_active_20260531\reranker_t2_fresh_next2_rows0_29_stable11_cap200_cap500_20260603
```

Aggregate fresh11:

- records/groups: `11`
- candidate samples: `213`
- turns: T2 only
- score mean/std: `+4.607 / 7.390`
- bust mean: `54.1%`
- FL mean: `6.3%`

Aggregate model evaluation:

| checkpoint | Top1 | Top3 | Top5 | Top10 | Top15 | Top1 regret | score MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| `exact31_ft_final` | 36.4% | 63.6% | 63.6% | 90.9% | 100.0% | 0.329 | 3.556 |
| `exact41_ft_final` | 45.5% | 81.8% | 81.8% | 100.0% | 100.0% | 0.311 | 2.518 |
| `exact31+exact41 uniform score average` | 54.5% | 72.7% | 72.7% | 100.0% | 100.0% | 0.296 | 2.972 |

Aggregate candidate-pool union with `exact31_ft_final` and `exact41_ft_final`:

| per-model K | recall | rerank regret | avg pool | max pool |
|---:|---:|---:|---:|---:|
| 1 | 54.5% | 0.296 | 1.4 | 2 |
| 3 | 90.9% | 0.016 | 3.9 | 5 |
| 5 | 90.9% | 0.004 | 5.9 | 7 |
| 10 | 100.0% | 0.000 | 11.1 | 15 |
| 15 | 100.0% | 0.000 | 15.4 | 22 |

Interpretation:

- The additional rows20-29 slice strengthens exact41 as the better current
  T2 pruning candidate on aggregate fresh rows0-29.
- The evidence is still not enough to declare Top10 universally safe, because
  rows0-9 had a Top10 miss and only Top15 caught every stable row there.
- Top15/union pool remains the safer handoff to exact refinement.
- Top1 remains weak. The current best observed aggregate Top1 is `54.5%`
  from score averaging, which is not sufficient for direct play advice.

## 2026-06-03 Oracle-First Batch Entry Point

The next implementation direction is oracle-first: make slow/high-confidence
labels before optimizing the 5 second runtime path.

A new wrapper was added:

- `ai/tutor/run_oracle_first_batch.py`

It runs the existing accurate label builders in one flow:

1. T3/T4: `generate_exact_late_teacher.py`
   - full exact late-turn evaluation
   - T3 enumerates all T4 draws and exact best T4 placement
   - T4 scores every terminal legal action exactly
2. T2: `run_t2_exact_oracle.py`
   - Rust capped-exact evaluator
   - configurable cap schedule, default `100,200,500`
3. T2 stable labels: `build_t2_stable_oracle_labels.py`
   - only emits rows where the selected cap pair agrees on Top1
   - default stable pair is the last two finite caps, so cap200/cap500

Useful command pattern for the next T2 slice:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.tutor.run_oracle_first_batch --input ai\data\hybrid_t1t2_active_20260531\oracle_inputs_t2_residual_diverse50_next2_20260603.jsonl --out-dir ai\data\hybrid_t1t2_active_20260531\oracle_first_next2_rows30_39_cap100_cap200_cap500_20260603 --skip-late --record-offset 30 --record-limit 10 --t2-limit 10 --t2-caps 100,200,500
```

Smoke validation:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.tutor.run_oracle_first_batch --input ai\data\hybrid_t1t2_active_20260531\oracle_inputs_t2_residual_diverse50_next2_20260603.jsonl --out-dir ai\data\hybrid_t1t2_active_20260531\oracle_first_smoke_cap1_cap2_20260603 --skip-late --record-limit 1 --t2-limit 1 --t2-caps 1,2
```

Smoke result:

- Rust solver ran for cap1 and cap2.
- records: `1`
- stable Top1: `1/1`
- stable output:
  `ai/data/hybrid_t1t2_active_20260531/oracle_first_smoke_cap1_cap2_20260603/t2_stable/stable_t2_oracle_cap1_cap2.jsonl`
- elapsed: `3.659s`

This smoke result is not a training-quality label because cap1/cap2 are too
small.  It only proves the oracle-first pipeline wiring.  Training-quality T2
labels should use cap200/cap500 stability, or full cap0 on small rows when the
runtime is acceptable.

Validation commands:

```powershell
C:\Users\Owner\anaconda3\python.exe -m py_compile ai\tutor\run_oracle_first_batch.py ai\tutor\generate_exact_late_teacher.py ai\tutor\run_t2_exact_oracle.py ai\tutor\build_t2_stable_oracle_labels.py
C:\Users\Owner\anaconda3\python.exe -m pytest tests\test_generate_exact_late_teacher.py
git -c safe.directory=C:/Users/Owner/.gemini/antigravity/scratch/ofc-pineapple diff --check -- ai/tutor/run_oracle_first_batch.py
```

Validation status:

- py_compile: passed
- exact late teacher tests: `4 passed`
- diff check: passed

## 2026-06-03 Next2 Rows 30-39 Oracle-First Run

The new wrapper was used to process rows30-39 from the next2 T2 input.

cap100 command:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.tutor.run_oracle_first_batch --input ai\data\hybrid_t1t2_active_20260531\oracle_inputs_t2_residual_diverse50_next2_20260603.jsonl --out-dir ai\data\hybrid_t1t2_active_20260531\oracle_first_next2_rows30_39_cap100_20260603 --skip-late --record-offset 30 --record-limit 10 --t2-limit 10 --t2-caps 100
```

cap100 result:

- records: `10`
- same Top1 vs source MC1000: `5/10`
- changed Top1: `5/10`
- avg cap100 elapsed: `17168.9ms`
- avg estimated full T2 elapsed from cap: `1045065.9ms`
- max source Top1 regret: `0.2557`
- miss indices inside the filtered rows30-39 input: `[2,4,6,7,8]`

The cap100 miss rows were promoted to cap200/cap500 with:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.tutor.run_oracle_first_batch --input ai\data\hybrid_t1t2_active_20260531\oracle_first_next2_rows30_39_cap100_20260603\inputs\t2_input.jsonl --out-dir ai\data\hybrid_t1t2_active_20260531\oracle_first_next2_rows30_39_misses5_cap200_cap500_20260603 --skip-late --record-indices 2,4,6,7,8 --t2-limit 5 --t2-caps 200,500
```

That combined cap200/cap500 run timed out at 6 minutes during cap500.  No
solver process remained afterward.  cap200 completed:

- records: `5`
- same Top1 vs source MC1000: `0/5`
- changed Top1: `5/5`
- avg cap200 elapsed: `35315.7ms`
- avg estimated full T2 elapsed from cap: `1076411.7ms`
- max source Top1 regret: `1.2852`

The timed-out cap500 file was incomplete JSON and should not be used.  To avoid
losing work, cap500 should be run one record at a time.  Record 0 was rerun
separately:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.tutor.run_oracle_first_batch --input ai\data\hybrid_t1t2_active_20260531\oracle_first_next2_rows30_39_misses5_cap200_cap500_20260603\inputs\t2_input.jsonl --out-dir ai\data\hybrid_t1t2_active_20260531\oracle_first_next2_rows30_39_miss0_cap500_20260603 --skip-late --record-indices 0 --t2-limit 1 --t2-caps 500
```

cap500 record0 result:

- records: `1`
- changed Top1 vs source MC1000: `1/1`
- elapsed: `97893.7ms`
- estimated full T2 elapsed from cap: `1171591.4ms`
- source Top1 regret: `0.0`

cap200/cap500 stable check for record0:

- stable Top1: `1/1`
- output:
  `ai/data/hybrid_t1t2_active_20260531/oracle_first_next2_rows30_39_miss0_cap500_20260603/t2_stable/stable_cap200_cap500_record0.jsonl`

Interpretation:

- The oracle-first path is working, but local cap500 is slow enough that
  batched cap500 runs can time out.
- Continue cap500 for remaining miss indices `1,2,3,4` one at a time, or move
  this exact-stability phase back to GCP once the local behavior is confirmed.
- Do not train on the incomplete cap500 combined output.

## 2026-06-03 Next2 Rows 30-39 Cap500 Completion

The remaining cap100 miss rows were rerun one at a time with cap200/cap500 to
avoid the 6 minute timeout seen in the combined run.

Per-row stable result:

| miss index | cap200 elapsed | cap500 elapsed | stable Top1 | source Top1 regret at cap500 |
|---:|---:|---:|---:|---:|
| 0 | previously completed | 97893.7ms | yes | 0.0000 |
| 1 | 39070.9ms | 101870.4ms | yes | 0.0000 |
| 2 | 41666.1ms | 97912.8ms | yes | 0.0000 |
| 3 | 39686.9ms | 98028.7ms | yes | 1.1495 |
| 4 | 18038.8ms | 44942.9ms | no | 0.1294 |

Aggregate stable output:

- `ai/data/hybrid_t1t2_active_20260531/stable_t2_oracle_next2_rows30_39_misses5_cap200_cap500_20260603.jsonl`
- stable rows: `4`
- unstable rows: `1`
- converted reranker data:
  `ai/data/hybrid_t1t2_active_20260531/reranker_t2_stable_oracle_next2_rows30_39_misses5_cap200_cap500_20260603`
- converted samples: `96`
- score mean/std: `+10.740 / 11.036`
- bust mean: `65.7%`
- FL mean: `21.1%`

Rows30-39 stable4 model evaluation:

| checkpoint | Top1 | Top3 | Top5 | Top10 | Top15 | Top1 regret | score MAE | avoidable 100% bust |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `exact31_ft_final` | 50.0% | 75.0% | 100.0% | 100.0% | 100.0% | 0.417 | 4.576 | 1 |
| `exact41_ft_final` | 50.0% | 75.0% | 100.0% | 100.0% | 100.0% | 0.963 | 4.374 | 1 |
| `exact31+exact41 average` | 50.0% | 75.0% | 100.0% | 100.0% | 100.0% | 0.417 | 4.465 | 1 |

Rows30-39 stable4 candidate-pool union:

| per-model K | recall | rerank regret | avg pool | max pool |
|---:|---:|---:|---:|---:|
| 1 | 50.0% | 0.417 | 1.2 | 2 |
| 3 | 75.0% | 0.287 | 3.2 | 4 |
| 5 | 100.0% | 0.000 | 5.2 | 6 |
| 10 | 100.0% | 0.000 | 10.8 | 11 |
| 15 | 100.0% | 0.000 | 15.8 | 17 |

Rows0-39 aggregate fresh15 was merged:

- `ai/data/hybrid_t1t2_active_20260531/reranker_t2_fresh_next2_rows0_39_stable15_cap200_cap500_20260603`
- records/groups: `15`
- candidate samples: `309`
- score mean/std: `+6.512 / 9.140`
- bust mean: `57.7%`
- FL mean: `10.9%`

Fresh15 model evaluation:

| checkpoint | Top1 | Top3 | Top5 | Top10 | Top15 | Top1 regret | score MAE | avoidable 100% bust |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `exact31_ft_final` | 40.0% | 66.7% | 73.3% | 93.3% | 100.0% | 0.353 | 3.873 | 2 |
| `exact41_ft_final` | 46.7% | 80.0% | 86.7% | 100.0% | 100.0% | 0.485 | 3.095 | 1 |
| `exact31+exact41 average` | 53.3% | 73.3% | 80.0% | 100.0% | 100.0% | 0.328 | 3.436 | 1 |

Fresh15 candidate-pool union:

| per-model K | recall | rerank regret | avg pool | max pool |
|---:|---:|---:|---:|---:|
| 1 | 53.3% | 0.328 | 1.3 | 2 |
| 3 | 86.7% | 0.089 | 3.7 | 5 |
| 5 | 93.3% | 0.003 | 5.7 | 7 |
| 10 | 100.0% | 0.000 | 11.0 | 15 |
| 15 | 100.0% | 0.000 | 15.5 | 22 |

Interpretation:

- Oracle-first labels continue to show that MC1000 source labels are not safe
  enough to treat as ground truth.
- exact41 remains the best single current T2 model on aggregate Top3/Top5/Top10
  and score MAE.
- Top1 is still far from direct-play quality. The best fresh15 Top1 is `53.3%`
  from score averaging, not a deployable direct answer.
- Top10/Top15 remains the practical safe candidate pool for exact refinement.

## 2026-06-03 Exact53 Fine-Tune And Rows40-49 Holdout

Training data was expanded from exact41 to exact53:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.training.merge_action_value_reranker_data --dirs ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle41_cap200_cap500_20260603 ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle_next2_rows10_19_misses3_cap200_cap500_20260603 ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle_next2_rows20_29_misses5_cap200_cap500_20260603 ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle_next2_rows30_39_misses5_cap200_cap500_20260603 --output ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle53_cap200_cap500_20260603
```

Merged exact53 data:

- records/groups: `53`
- candidate samples: `1,059`
- score mean/std: `+5.956 / 9.146`
- bust mean: `56.5%`
- FL mean: `9.8%`

Fine-tune command:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.training.train_action_value_reranker --data ai\data\hybrid_t1t2_active_20260531\reranker_t2_stable_oracle53_cap200_cap500_20260603 --save-dir ai\models\candidate_runs\t2-stable-oracle53-cap500-ft-from-exact41-20260603\model --init-checkpoint ai\models\candidate_runs\t2-stable-oracle41-cap500-ft-from-exact38-20260603\model\action_value_final.pt --normalization-source checkpoint --train-turns 2 --epochs 80 --max-seconds 90 --batch-size 128 --lr 0.000002 --score-weight 1.0 --bust-weight 0.15 --fl-weight 0.15 --fl-type-weight 0.10 --ranking-weight 1.1 --ranking-batches-per-epoch 32 --group-batch-size 16 --topk-rank-weight 0.7 --target-topk 3 --selection-metric regret --val-frac 0.2 --device cpu --seed 20260603
```

Training stopped at epoch 55 due to `--max-seconds` and saved:

- `ai/models/candidate_runs/t2-stable-oracle53-cap500-ft-from-exact41-20260603/model/action_value_best.pt`
- `ai/models/candidate_runs/t2-stable-oracle53-cap500-ft-from-exact41-20260603/model/action_value_final.pt`

On the in-sample fresh15 rows0-39 set, exact53 final absorbed the new labels:

| checkpoint | Top1 | Top3 | Top5 | Top10 | Top15 | Top1 regret | score MAE | avoidable 100% bust |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `exact41_ft_final` | 46.7% | 80.0% | 86.7% | 100.0% | 100.0% | 0.485 | 3.095 | 1 |
| `exact53_ft_final` | 93.3% | 93.3% | 93.3% | 100.0% | 100.0% | 0.033 | 2.436 | 0 |

This is not a generalization claim because rows0-39 stable labels were included
in the exact53 training set.

Rows40-49 were then used as the next unused holdout slice.  cap100 result:

- records: `10`
- same Top1 vs source MC1000: `5/10`
- changed Top1: `5/10`
- avg cap100 elapsed: `20477.0ms`
- avg estimated full T2 elapsed from cap: `1368039.5ms`
- avg source Top1 regret: `0.4543`
- max source Top1 regret: `4.1732`
- miss indices inside the filtered rows40-49 input: `[0,2,4,7,9]`

cap200/cap500 stability for the miss rows:

| miss index | cap200 elapsed | cap500 elapsed | stable Top1 | source Top1 regret at cap500 |
|---:|---:|---:|---:|---:|
| 0 | 71950.8ms | 164323.2ms | yes | 0.4992 |
| 2 | 44780.5ms | 116069.0ms | no | 0.0000 |
| 4 | 58896.7ms | 136910.5ms | no | 0.0000 |
| 7 | 13319.8ms | 37337.3ms | yes | 0.0000 |
| 9 | 60042.3ms | 141690.4ms | no | 0.3812 |

Stable holdout output:

- `ai/data/hybrid_t1t2_active_20260531/stable_t2_oracle_next2_rows40_49_misses5_cap200_cap500_20260603.jsonl`
- stable rows: `2`
- unstable rows: `3`
- converted reranker data:
  `ai/data/hybrid_t1t2_active_20260531/reranker_t2_stable_oracle_next2_rows40_49_misses5_cap200_cap500_20260603`
- candidate samples: `36`
- score mean/std: `+9.572 / 5.709`
- bust mean: `45.2%`
- FL mean: `15.3%`

Rows40-49 stable2 holdout evaluation:

| checkpoint | Top1 | Top3 | Top5 | Top10 | Top15 | Top1 regret | score MAE | avoidable 100% bust |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `exact41_ft_final` | 0.0% | 50.0% | 50.0% | 100.0% | 100.0% | 4.349 | 3.902 | 0 |
| `exact53_ft_final` | 0.0% | 0.0% | 50.0% | 100.0% | 100.0% | 4.349 | 4.610 | 0 |
| `exact53_ft_best` | 0.0% | 50.0% | 50.0% | 100.0% | 100.0% | 4.349 | 4.181 | 0 |

Rows40-49 stable2 candidate-pool union with exact41 final + exact53 final:

| per-model K | recall | rerank regret | avg pool | max pool |
|---:|---:|---:|---:|---:|
| 1 | 0.0% | 4.349 | 1.0 | 1 |
| 3 | 50.0% | 4.099 | 3.5 | 4 |
| 5 | 50.0% | 0.000 | 6.0 | 7 |
| 10 | 100.0% | 0.000 | 10.0 | 10 |
| 15 | 100.0% | 0.000 | 13.5 | 15 |

Interpretation:

- exact53 substantially improves on the labels it trained on, but rows40-49
  show that Top1 generalization is not solved.
- The strongest stable claim remains candidate-pool safety: Top10/Top15 caught
  all rows40-49 stable holdout labels.
- Top1 direct-answer deployment is still not justified.
- The next active-loop target should be rows40-49 stable2 misses plus unstable
  rows2/4/9 at higher cap or full exact if feasible.

## 2026-06-03 Rows40-49 Cap1000 Follow-Up

One unstable rows40-49 miss was escalated from cap500 to cap1000:

- input slice: rows40-49 miss index `9`
- output:
  `ai/data/hybrid_t1t2_active_20260531/oracle_first_next2_rows40_49_miss9_cap1000_20260603`
- cap1000 elapsed: `296082.2ms`
- estimated full T2 elapsed from this cap: `2114027.1ms`
- source MC1000 Top1 regret at cap1000: `0.4128`
- cap500 vs cap1000 Top1 stability: stable `1/1`

The cap1000-stabilized row was added to the previous stable2 file:

- stable output:
  `ai/data/hybrid_t1t2_active_20260531/stable_t2_oracle_next2_rows40_49_misses5_cap200_cap500_plus_cap1000_20260603.jsonl`
- converted reranker data:
  `ai/data/hybrid_t1t2_active_20260531/reranker_t2_stable_oracle_next2_rows40_49_misses5_cap200_cap500_plus_cap1000_20260603`
- stable records/groups: `3`
- candidate samples: `60`
- score mean/std: `+7.493 / 6.234`
- bust mean: `54.3%`
- FL mean: `13.8%`

Rows40-49 stable3 holdout evaluation:

| checkpoint | Top1 | Top3 | Top5 | Top10 | Top15 | Top1 regret | score MAE | avoidable 100% bust |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `exact41_ft_final` | 0.0% | 33.3% | 33.3% | 100.0% | 100.0% | 3.579 | 3.724 | 0 |
| `exact53_ft_final` | 0.0% | 0.0% | 33.3% | 100.0% | 100.0% | 3.579 | 4.178 | 0 |
| `exact53_ft_best` | 0.0% | 33.3% | 33.3% | 100.0% | 100.0% | 3.579 | 3.926 | 0 |

Rows40-49 stable3 candidate-pool union with exact41 final + exact53 final:

| per-model K | recall | rerank regret | avg pool | max pool |
|---:|---:|---:|---:|---:|
| 1 | 0.0% | 3.579 | 1.0 | 1 |
| 3 | 33.3% | 3.084 | 3.7 | 4 |
| 5 | 33.3% | 0.351 | 5.7 | 7 |
| 10 | 100.0% | 0.000 | 10.3 | 11 |
| 15 | 100.0% | 0.000 | 14.3 | 16 |

Miss-rank details for exact53 final on stable3:

| group | teacher rank by model | Top1 regret |
|---:|---:|---:|
| 0 | 5 | 0.499 |
| 1 | 9 | 8.198 |
| 2 | 9 | 2.038 |

Interpretation:

- T2 capped labels become more trustworthy when cap500 and cap1000 agree, but
  they are still not the same as full exact.
- exact53 absorbed the rows0-39 labels in-sample, yet did not generalize Top1
  on rows40-49 stable3.
- Direct Top1 T2 deployment is not justified by the current evidence.
- The current reliable runtime path is still model candidate pooling followed by
  exact/refinement: Top10 caught all rows40-49 stable3 labels while Top1/Top3
  were too weak.
- Next data work should either escalate the remaining unstable rows40-49 indices
  `2` and `4`, or train on stable3 and test against a fresh unseen slice.

## 2026-06-03 T3 Value Model Route For Fast T2

Observation:

- A perfect or near-perfect T3 value model would make T2 much faster.
- T2 evaluation is structurally:
  `apply T2 action -> enumerate/sample T3 deals -> solve T3 -> average`.
- The expensive part is repeatedly solving T3 exactly.  If T3 best-action EV
  can be predicted in a large batch, T2 can evaluate thousands of future T3
  deals quickly and reserve exact computation only for close cases.

Current T3 model evidence from the existing T3 exact holdout:

| model/eval | Top1 | Top3 | Top5 | Top10 | Top20 | Top24 | avg regret | score MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `current_t3_exact_500` | 25.0% | 53.4% | 66.0% | 87.2% | 99.0% | 100.0% | 2.929 | 3.958 |
| `t3_exact10000_adapter64` | 25.6% | 54.8% | 70.2% | 91.4% | 99.4% | 100.0% | 2.857 | 6.947 |

Interpretation:

- Existing T3 models are not direct Top1 solvers.
- They are useful as a candidate pool: Top20/Top24 is nearly or fully safe on
  the 500-decision exact holdout.
- For T2 runtime, using this model as a fast value oracle is viable only if the
  T3 EV/value calibration is improved, because T2 averages many T3 predictions
  and small systematic T3 errors can move the T2 Top1.

Preferred runtime architecture:

1. Generate more exact T3 labels from diverse states, with emphasis on high
   regret, high FL, joker, and bust-risk positions.
2. Train a T3 value model that predicts the exact best-action EV, FL, bust, and
   optionally the top action distribution.
3. For T2, enumerate or heavily sample T3 deals and batch-evaluate them with the
   T3 value model.
4. Use the T2 reranker only for shortlist/candidate-pool safety.
5. Exact-refine the top T2 candidates, especially when the model-estimated EV
   gap is small.

This is stronger than asking one T2 model to directly guess Top1 from the T2
state.  It preserves the game structure and moves the expensive exact work to
the smallest set of close candidates.

Local prototype:

- script:
  `ai/tutor/benchmark_t2_t3_value_model.py`
- smoke input:
  `oracle_first_next2_rows40_49_cap100_20260603/inputs/t2_input.jsonl`
- T3 value model:
  `tutor-route10-t3-exact10000-adapter64-20260529/model/action_value_best.pt`
- exact reference:
  `oracle_first_next2_rows40_49_cap100_20260603/t2_caps/t2_oracle_cap100_limit10.jsonl`
- run shape: `1` record, source Top5 T2 candidates, `10` T3 draws per candidate
- elapsed: `287.3ms`
- T3 states scored: `960`
- throughput: `3342` T3 post-action states/sec on CPU
- exact Top1 rank under this T3-model T2 estimate: `4`
- model Top1 exact regret: `2.572`
- Top5 recall: `100%`

This confirms the runtime shape is promising, but current T3 value accuracy is
not enough for direct T2 Top1.

Broader local benchmark after fixing shared joker normalization and loose
joker action-key matching:

| shape | elapsed | T3 states | states/sec | exact Top1 rank | model Top1 exact regret | Top1 | Top3 | Top5 | Top10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 records, source Top10 T2 candidates, 30 T3 draws | 3605.6ms | 13,590 | 3,769 | mean 6.67 | 1.799 | 0.0% | 0.0% | 33.3% | 100.0% |
| 1 record, all 27 T2 candidates, 30 T3 draws | 3728.0ms | 14,580 | 3,911 | 5 | 2.572 | 0.0% | 0.0% | 100.0% | 100.0% |

The full-candidate result is important: even without T2 shortlist pruning, a
30-draw T3-model pass for one T2 decision is already under the 5s target on
CPU.  More careful batching across all candidates in one forward path should
improve this further.  The blocking issue is therefore not raw runtime at this
sample size; it is T3 value-model accuracy/calibration.

T3 bottleneck hypotheses to test:

1. Data shortage:
   exact T3 data is cheap enough to generate compared with T2, so we should
   scale diverse exact T3 labels substantially.
2. Representation shortage:
   the current action-value reranker scores post-action boards independently.
   It may lack enough relational capacity to compare close placements, opponent
   blockers, and joker/FL/bust interactions.
3. Loss mismatch:
   Top1 classification is not the right sole target.  T2 needs calibrated best
   EV, bust, FL, and uncertainty/gap estimates after averaging over many T3
   futures.
4. Candidate-level ambiguity:
   Many T3 actions have close EVs; the model should be judged by regret and
   value calibration, not only exact action match.

Next T3 experiments:

- Train a larger T3 value model variant or transformer/set-style row/card model
  against exact T3 data.
- Compare it against the current reranker on:
  Top1/Top3/Top10/Top20 recall, regret, score MAE, bust MAE, FL MAE, and T2
  downstream rank/regret through `benchmark_t2_t3_value_model.py`.
- If larger capacity improves T3 value calibration, use it as the default T2
  future evaluator and keep exact refinement for close TopK candidates.

## 2026-06-03 T3 Data vs Capacity Follow-Up

Short unfreeze experiment:

- model:
  `ai/models/candidate_runs/t3-exact10000-unfrozen-ft-20260603/model/action_value_best.pt`
- starting point:
  `tutor-route10-t3-exact10000-adapter64-20260529/model/action_value_best.pt`
- data:
  `ai/data/t0_proxy_holdout_20260529/exact_late_reranker_t3t4_10000_20260529`
- local runtime: CPU, stopped after a short `--max-seconds` run

Fixed T3 exact holdout:

| model | Top1 | Top3 | Top5 | Top10 | Top20 | avg regret | score MAE | FL MAE | bust MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `t3_exact10000_adapter64` | 25.6% | 54.8% | 70.2% | 91.4% | 99.4% | 2.857 | 6.947 | 6.9% | 25.0% |
| `t3_exact10000_unfrozen_ft` | 25.0% | 55.8% | 72.4% | 90.4% | 99.0% | 2.573 | 6.596 | 6.9% | 25.0% |

T2 downstream via T3 value model, rows40-49, 3 records, source Top10 T2
candidates, 30 T3 draws:

| model | elapsed | T3 states/sec | exact Top1 rank | model Top1 exact regret | Top1 | Top3 | Top5 | Top10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `t3_exact10000_adapter64` | 3605.6ms | 3,769 | mean 6.67 | 1.799 | 0.0% | 0.0% | 33.3% | 100.0% |
| `t3_exact10000_unfrozen_ft` | 3138.1ms | 4,331 | mean 7.00 | 1.799 | 0.0% | 0.0% | 33.3% | 100.0% |

Interpretation:

- Briefly unfreezing the shared trunk improved T3 regret and Top5 on the fixed
  holdout, but did not improve T3 Top1 or T2 downstream ranking on the small T2
  sample.
- This is evidence that representation/loss can matter, but it is not enough
  to conclude that the current architecture will scale to direct Top1.
- Because exact T3 labels are available at scale, the next experiment should
  separate data volume from model capacity under the same holdout and T2
  downstream benchmarks.

Experiment harness:

- script: `ai/tutor/run_t3_data_capacity_experiment.py`
- default behavior: write `manifest.json` and `commands.ps1`; do not start long
  exact/training jobs unless `--execute` is passed.
- default comparison:
  - current architecture, unfrozen, initialized from the existing T3 checkpoint
  - larger capacity reranker, `hidden=768`, `n_blocks=5`, turn-specific heads
    and adapters, trained on the same T3 action-value data
- evaluation:
  - fixed `teacher_labels_t3_exact_500.jsonl` holdout
  - T2 downstream route through `benchmark_t2_t3_value_model.py`

Local command-plan generation:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.tutor.run_t3_data_capacity_experiment `
  --python C:\Users\Owner\anaconda3\python.exe `
  --out-dir ai\data\hybrid_t1t2_active_20260531\t3_data_capacity_20260603
```

Fresh-data variant:

```powershell
C:\Users\Owner\anaconda3\python.exe -m ai.tutor.run_t3_data_capacity_experiment `
  --python C:\Users\Owner\anaconda3\python.exe `
  --generate-labels `
  --exact-records 5000 `
  --exact-workers 4 `
  --out-dir ai\data\hybrid_t1t2_active_20260531\t3_data_capacity_fresh5k_20260603
```

Decision gate:

- If more data with the same architecture improves Top1/regret/T2 downstream,
  prioritize exact T3 data generation.
- If the larger model improves the same metrics on the same data, prioritize
  model capacity or a row/card relational architecture.
- If neither moves T2 downstream regret, the bottleneck is likely target/loss
  calibration rather than just data count or model width.

Short CPU probe:

- summary:
  `ai/data/hybrid_t1t2_active_20260531/t3_data_capacity_probe_20260603/summary.md`
- environment: `torch 2.7.1+cpu`, CUDA unavailable
- current architecture probe:
  existing adapter64 checkpoint, 2 epochs, stopped at 161s
- large-capacity probe:
  scratch `hidden=768`, `n_blocks=5`, turn heads/adapters, 1 epoch, stopped at
  121s

Fixed T3 holdout:

| run | Top1 | Top3 | Top5 | Top10 | Top20 | avg regret | score MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| `current_arch_unfrozen_probe` | 26.0% | 51.8% | 68.8% | 89.0% | 98.6% | 2.539 | 6.980 |
| `large_capacity_probe` | 25.2% | 48.8% | 64.2% | 87.6% | 99.0% | 3.031 | 19.488 |

T2 downstream via T3 value model:

| run | exact Top1 rank | Top1 | Top3 | Top5 | Top10 | model Top1 regret |
|---|---:|---:|---:|---:|---:|---:|
| `current_arch_unfrozen_probe` | mean 7.33 | 0.0% | 0.0% | 33.3% | 100.0% | 1.799 |
| `large_capacity_probe` | mean 3.67 | 0.0% | 66.7% | 100.0% | 100.0% | 1.877 |

This probe is not deployable evidence, because the large model only completed
one CPU epoch.  It is still useful: the larger model is worse on fixed T3
holdout Top1/regret, but better at downstream T2 candidate ordering on the
small rows40-49 benchmark.  That suggests the next serious experiment should
track T2 downstream rank/regret as a first-class metric, not only T3 exact
Top1.  The command harness now separates learning rates: current-architecture
fine-tune uses `2e-6`, while scratch large-capacity uses `3e-4`.
