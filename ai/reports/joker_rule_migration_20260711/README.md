# Canonical Joker Rule Migration — 2026-07-11

## Rule contract

The canonical complete-board evaluator resolves rows bottom-up:

1. Bottom takes its strongest Joker substitution.
2. Middle takes its strongest substitution not stronger than Bottom.
3. Top takes its strongest substitution not stronger than Middle.
4. Foul, royalties, Fantasyland entry/stay, and score use those constrained
   values.

Joker hand values use exhaustive, distinct natural-card substitutions and the
same base-15 tie-break encoding in Python and Rust.

## T3 label migration

Source decision contexts were reused from:

`ai/data/t3_relabel_jokerfix_20k_20260605/inputs/t3_input_20000.jsonl`

Fresh canonical artifacts:

- exact labels: `ai/data/joker_rule_migration_20260711/t3_full/t3_exact_new_20000.jsonl`
- teacher: `ai/data/joker_rule_migration_20260711/t3_full/t3_teacher_new_20000.jsonl`
- reranker arrays: `ai/data/joker_rule_migration_20260711/t3_full/reranker_t3_canonical`
- migration summary: `ai/data/joker_rule_migration_20260711/t3_full/comparison_summary.json`

All 20,000 records and 269,745 candidates converted with `skipped=0`.

Old-vs-canonical exact-label comparison:

| Metric | Result |
|---|---:|
| Records with any metric change | 18,218 / 20,000 (91.09%) |
| Raw Top1 identity changes | 3,241 (16.21%) |
| Old Top1 genuinely suboptimal under new labels | 677 (3.385%) |
| Mean old-policy regret | 0.0290 |
| Max old-policy regret | 8.2891 |
| Mean absolute best-score delta | 2.6396 |
| Mean bust-rate delta | -0.0842 |

The 20,000-position exact run completed locally in 702.4 seconds. No GCP was
used.

## Retrained T3 models

- BB: `ai/models/candidate_runs/t3-canonical-joker-20k-bb-20260711/action_value_best.pt`
- BTN: `ai/models/candidate_runs/t3-canonical-joker-20k-btn-20260711/action_value_best.pt`

Independent holdout generation used a new seed and branch-expanded routes:

- inputs: `ai/data/joker_rule_migration_20260711/holdout/branch_t3_input_400.jsonl`
- exact teacher: `ai/data/joker_rule_migration_20260711/holdout/branch_t3_teacher_400.jsonl`
- converted data: `ai/data/joker_rule_migration_20260711/holdout/reranker_t3_holdout`

Holdout results (200 groups per position):

| Position | Model | Top1 | Top3 | Top1 regret | Top3 regret | Score MAE |
|---|---|---:|---:|---:|---:|---:|
| BB | old | 30.0% | 57.0% | 1.478 | 0.667 | 2.354 |
| BB | canonical | 33.5% | 60.0% | 1.154 | 0.504 | 1.207 |
| BTN | old | 30.0% | 54.0% | 1.237 | 0.529 | 1.976 |
| BTN | canonical | 28.5% | 61.5% | 1.196 | 0.158 | 1.427 |

The canonical BTN model loses 1.5 points of raw Top1 recall, but improves
Top1 regret, Top3 recall, Top3 regret, score calibration, and bust calibration.
It is suitable as a candidate source followed by exact reranking, not as a
standalone Top1 policy.

## Runtime pool gate

The current union8 and the experimental union9 (union8 plus the canonical
action-value model) were exact-reranked on all 400 independent positions.

| Pool | Exact-best agreement | Mean pool size | Mean latency | P95 latency |
|---|---:|---:|---:|---:|
| existing union8 | 400 / 400 | 15.5025 | 96.69 ms | 219.63 ms |
| canonical union9 | 400 / 400 | 15.6375 | 97.54 ms | 229.32 ms |

The existing runtime remains safe on this gate because final selection uses
the canonical Rust exact solver. The candidate config is retained at
`ai/config/t3_canonical_joker_pool_candidate_20260711.json`, but the default
runtime config was not changed because union9 produced no additional exact-best
hits on this holdout.

## T2 compatibility pilot

The seed52 balanced cap100 labels from 2026-07-10 were sampled at records
0–4 (visible Bottom Joker) and 50–54 (no self-visible Joker, future Joker still
reachable). Every legal action metric and best action matched the current
canonical solver exactly in all 10 records.

Fresh pilot outputs are under:

`ai/data/joker_rule_migration_20260711/t2_pilot`

This is strong evidence that the July-10 T2 exact path already used the same
late-turn Joker semantics for these strata. It is not a bit-for-bit audit of all
100 records. A full rerun was deliberately avoided because cap100 costs roughly
48–161 seconds per position and the pilot found zero migration delta.

Older T2 learned models remain unverified because their training labels predate
the July-10 exact artifacts.

## Additional fixes

- Python CFR no longer removes legal Ace-to-Bottom, Joker-to-Middle, or
  two-Jokers-in-one-row T0 actions.
- `run_oracle_first_batch.py` now passes the required `skip=0` and
  `source_candidate_top_k=0` fields to the T2 exact runner.
- Exact-label migration comparison now streams large JSONL files and reports
  unseen-future-Joker exposure.
- Action-value evaluation now supports BB/BTN group filtering and computes
  sample metrics only from the selected position.

## Verification

- Python: 166 passed, 2 pre-existing FastAPI deprecation warnings.
- Rust workspace: 29 passed, 0 failed.
- Python compile checks and candidate-config JSON validation passed.
- Scoped `git diff --check` passed.

## Next dependency order

1. Rebuild or certify the active T2 learned model against canonical labels.
2. Regenerate/retrain T1 after T2 is stable.
3. Regenerate/retrain T0 after T1 is stable.
4. Retire or retrain the provenance-unknown backend checkpoint if that backend
   remains part of the target AI.

Do not retrain T0 first: its rollout targets depend on all later-turn models.
