# T3 HU sampled pilot (2026-07-12)

## Decision

This pilot does **not** promote the sampled T3 evaluator as the final T3
teacher or serving policy.

- BTN (button, second actor) is seed-stable at the tested budget.
- BB (non-button, first actor) is not seed-stable at the tested budget.
- The evaluator is still a PIMC determinization bootstrap
  (`hu_exact=false`), even where its sampled result is stable.

The next T3 milestone is therefore a higher-sample, role-aware BB evaluator
with a public-information policy gate. T2/T1/T0 labels must not be generated
from this pilot until that gate passes.

## Configuration

- Input: `ai/data/t3_precision_gate_5k_20260711/t3_gate_input_5000.jsonl`
- Positions: 100, balanced across BB/BTN and visible Joker count
- Independent evaluation seeds: `20260712`, `20260713`
- BTN budget: 8 outer samples
- BB budget: 2 outer samples, 1 response-inner sample
- T4 leaf: canonical exact Rust evaluator
- Information model: `pimc_determinization_v1`

The final roots reserved by the dataset split were not used. Selection used
roots 0-99, with 74 unique roots and at most 3 rows from one root.

## Results

| Role | Positions | Strict Top-1 agreement | CI-compatible | Mean cross-regret | Max cross-regret | Runtime p95 | Runs over 5 s |
|---|---:|---:|---:|---:|---:|---:|---:|
| BB / first | 50 | 52% | 94% | 3.708 | 23.957 | 7.014 s | 21/100 |
| BTN / second | 50 | 100% | 100% | 0.000 | 0.000 | 2.893 s | 0/100 |
| Combined | 100 | 76% | 97% | 1.854 | 23.957 | 6.243 s | 21/200 |

BB agreement by visible Joker count was 52.0% (zero), 61.5% (one), and
41.7% (two). The two-sample confidence intervals are broad, so the 94%
CI-compatible figure does not establish decision precision.

Total wall time was 408.66 seconds for 200 position-seed evaluations.

## Interpretation

The exact T4 leaf is working, but BB T3 has a much larger continuation tree:
every root action must compare all legal BTN T3 responses before solving the
remaining T4 sequence. At two outer samples, hidden-card/chance variation can
change the selected action between seeds. Increasing Python-side batching
alone will not solve this; most time is spent in the Rust T4 leaves.

Before promotion:

1. Reuse repeated terminal response work inside the Rust T4 batch path.
2. Add adaptive/common-random-number sampling for BB and rerun the same
   held-out positions at larger outer budgets.
3. Require stable Top-1 and low cross-seed regret separately for BB and BTN,
   including Joker strata.
4. Replace or validate PIMC decisions with information-set/public-belief
   grouping so no future action can condition on an opponent's hidden discard.
5. Only then propagate the T3 continuation target backward to T2, T1, and T0.

Machine-readable evidence is in `summary.json`, `comparisons.jsonl`, and the
two files under `runs/`.

## BB refinement and stop decision

The Rust T4 leaf path was then changed to evaluate all BB candidates jointly.
Natural-card and X1/X2 reference tests kept every candidate metric and the
selected action unchanged. On a representative pilot position, BB T3
outer=2/inner=1 fell from 1101.08 ms to 388.15 ms (2.84x faster).

The 20 largest original BB disagreements were rerun with larger budgets:

| Budget | Seed Top-1 | CI-compatible | Mean cross-regret | Max cross-regret | Runtime p95 |
|---|---:|---:|---:|---:|---:|
| outer=4, inner=4 | 75% | 100% | 0.761 | 10.032 | 12.013 s |
| outer=8, inner=4 | 75% | 100% | 0.828 | 7.120 | 23.751 s |

For each seed separately, outer=4 and outer=8 agreed on 18/20 decisions
(90%). However, the two independent outer=8 seeds still agreed on only 15/20.

The remaining five outer=8 disagreements were raised to outer=16/inner=4.
Only 2/5 decisions agreed between seeds, while runtime p95 reached 36.163 s.
This is a deliberately adversarial subset rather than an estimate of the
population success rate, but it shows that uniform sample inflation is not an
efficient promotion route.

Further brute-force PIMC refinement is stopped here. The next implementation
must group decisions by the information visible to the acting player, so that
one policy is shared across hidden-discard worlds. This is required for the
final objective independently of the sampling variance result.

At outer=4 to outer=8, the median best-action standard error changed from
3.337 to 2.349, almost exactly the expected `1/sqrt(2)` scaling. This confirms
that the remaining uncertainty is dominated by outer hidden-world variation,
not a broken T4 leaf. On the outer=8 runs, 14/40 position-seed results had an
exact zero Top-2 gap and 25/40 had a gap below one expected-score point. Future
gates should therefore report both exact action-key agreement and an
epsilon-optimal set. The provisional teacher criterion is epsilon=0.5 score
points with decision-level family-wise alpha=0.01; unresolved ties remain an
ambiguity set instead of being forced into a single label.

Refinement evidence:

- `../t3_hu_sampling_refine_bb_top20_o4_i4_20260712/summary.json`
- `../t3_hu_sampling_refine_bb_top20_o8_i4_20260712/summary.json`
- `../t3_hu_sampling_refine_bb_top5_o16_i4_20260712/summary.json`
