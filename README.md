# OFC Pineapple Regular

Separate implementation for regular OFC Pineapple rules:

- 52-card deck, no jokers.
- Fantasyland entry condition is unchanged: QQ+, KK, AA, or trips on top.
- Every Fantasyland entry deals 14 cards.
- Fantasyland stay also deals 14 cards.
- Royalty tables match the existing pineapple implementation.

The original repository is only a reference. This repository is intended to evolve independently.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e .[dev]
.\.venv\Scripts\python -m pytest
```

Run a small legacy Fantasyland royalty/stay-rate smoke calculation:

```powershell
python -m ofc_regular.fl_ev --trials 10 --seed 42
```

The current default FL EV is the direct HU fixed-point value in
`configs/fl_ev_regular_2k.json`. The Python `fl_ev` command is a legacy chain
estimator and will not write JSON output unless
`--allow-legacy-chain-output` is explicitly provided.

Use the direct HU estimator for current calibration runs. The Rust solver is
still useful for standalone FL hand solving and legacy chain diagnostics:

```powershell
cargo run --release -- --trials 1000 --iterations 8 --seed 42 --line-scoop-advantage 4
```

Solve one explicit 14-card Fantasyland hand:

```powershell
cargo run --release -- --solve "Ah,Kh,Qh,Jh,Th,9h,8h,7h,6h,5h,4h,3h,2h,As" --stay-bonus 100
```

Write a legacy regular-mode FL EV diagnostic config:

```powershell
cargo run --release -- --trials 10000 --iterations 8 --seed 42 --line-scoop-advantage 4 --output outputs/fl_ev_regular.json
```

## Exact Late-Turn Teacher

The Python package includes exact terminal action evaluation for completed
boards. This is the first teacher-data path for training an AI policy:

```python
from ofc_regular import Board, evaluate_turn_actions, load_fl_ev

board = Board.from_rows(
    top=["Qh", "2d"],
    middle=["Kh", "Kd", "6c", "8s", "Th"],
    bottom=["9c", "9d", "9s", "Kc"],
)
ranked = evaluate_turn_actions(board, ["Qs", "Ah", "7d"], fl_ev=load_fl_ev())
best = ranked[0]
```

Opening action generation places all five cards. Normal-turn action generation
places up to two cards and records all discards in `Action.discards`.

For a 9-card board, use two-turn expectimax. It places the current 3-card deal,
samples or enumerates the next 3-card deal, and solves the final turn exactly:

```python
from ofc_regular import Board, evaluate_two_turn_actions, load_fl_ev

board = Board.from_rows(
    top=["Qh"],
    middle=["Kh", "Kd", "6c", "8s"],
    bottom=["9c", "9d", "9s", "Kc"],
)
ranked = evaluate_two_turn_actions(
    board,
    ["Qs", "Ah", "7d"],
    fl_ev=load_fl_ev(),
    max_future_deals=64,
)
best = ranked[0]
```

Generate final-turn teacher data:

```powershell
python -m ofc_regular.teacher_data --samples 1000 --seed 42 --min-score-gap 1 --fl-ev-config configs/fl_ev_regular_2k.json --output outputs/final_turn_teacher.jsonl
```

Generate Turn3 9-card teacher data with Rust:

```powershell
cargo run --release -- --teacher-output outputs/turn3_teacher.jsonl --teacher-samples 10000 --future-samples 128 --seed 42 --teacher-fl-ev 10.227020614683454 --teacher-min-score-gap 1
```

Keep `--teacher-fl-ev` aligned with `configs/fl_ev_regular_2k.json` unless the
run is intentionally labeled as a legacy/scoring-sensitivity experiment.

Use `--future-samples 0` only for exact enumeration of all final-turn deals;
it is much heavier than sampled teacher generation. Teacher rows where every
evaluated Turn3 action can only reach busted final boards are skipped.

For larger Turn3 runs, measure speed first and then generate in resumable
chunks so a shell timeout does not lose the whole run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\Measure-T3TeacherSpeed.ps1 -Samples 20,100 -FutureSamples 64,128
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\Run-T3TeacherChunks.ps1 -TotalSamples 10000 -ChunkSize 500 -FutureSamples 64 -MergedOutput outputs\turn3_teacher_stage1_10k_f64.jsonl
```

Train the first Turn3 action-value model:

```powershell
python -m ofc_regular.train_turn3 --input outputs/turn3_teacher.jsonl --model-output models/turn3_ridge.npz --metrics-output outputs/turn3_ridge_metrics.json --holdout 0.2 --l2 10
```

This first model is trained on random 9-card states. Earlier turns should later
move from random state generation to policy/self-play state generation once the
Turn3 model is usable.

Smoke-play the baseline AI:

```powershell
python -m ofc_regular.play_ai --games 10 --seed 42 --turn3-model models/turn3_ridge.npz --output outputs/ai_smoke.json
```

The current playable policy uses the Turn3 model at the 9-card decision, exact
search on the final 11-card decision, and random legal actions for opening and
5-card decisions unless earlier-turn models are supplied.

Generate Turn2 7-card teacher data from the Turn3 model:

```powershell
python -m ofc_regular.t2_teacher_data --turn3-model models/turn3_ridge.npz --samples 5000 --future-samples 64 --seed 42 --output outputs/turn2_teacher.jsonl
```

For longer Turn2 / Turn1 / opening teacher-data runs, use the resumable Python
teacher chunk runner:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\Run-PythonTeacherChunks.ps1 -Phase t2 -DownstreamModel models\turn3_stage3.npz -TotalSamples 500 -ChunkSize 50 -FutureSamples 64 -OutputDir outputs\t2_stage3_chunks -MergedOutput outputs\t2_teacher_stage3.jsonl
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\Run-PythonTeacherChunks.ps1 -Phase turn1 -DownstreamModel models\turn2_stage3.npz -TotalSamples 300 -ChunkSize 50 -FutureSamples 64 -OutputDir outputs\turn1_stage3_chunks -MergedOutput outputs\turn1_teacher_stage3.jsonl
```

Use `scripts\Run-PythonTeacherChunksParallel.ps1` for independent chunks that
can safely run at the same time. `-MaxParallel` controls the number of Python
workers, and `-PredictionThreads 1` keeps scikit-learn prediction from
oversubscribing CPU threads inside each worker.

Train and use the Turn2 model:

```powershell
python -m ofc_regular.train_action_value --input outputs/turn2_teacher.jsonl --model-output models/turn2_ridge.npz --metrics-output outputs/turn2_ridge_metrics.json --holdout 0.2 --l2 10
python -m ofc_regular.play_ai --games 10 --seed 42 --turn2-model models/turn2_ridge.npz --turn3-model models/turn3_ridge.npz --output outputs/ai_t2_t3_smoke.json
```

Bootstrap Turn1 and use exact T0 placement search:

```powershell
python -m ofc_regular.early_teacher_data --phase turn1 --downstream-model models/turn2_ridge.npz --samples 5000 --future-samples 64 --seed 42 --output outputs/turn1_teacher.jsonl
python -m ofc_regular.train_action_value --input outputs/turn1_teacher.jsonl --model-output models/turn1_ridge.npz --metrics-output outputs/turn1_ridge_metrics.json --holdout 0.2 --l2 10

python -m ofc_regular.play_ai --games 10 --seed 42 --opening-model models/opening_stage7_torch_wide.pt --turn1-model models/turn1_stage6_torch_wide.pt --turn2-model models/turn2_stage8.pkl --turn3-model models/turn3_stage6.pkl --opening-lookahead-samples 64 --prediction-threads 1 --output outputs/ai_full_smoke.json
```

Current compact model set:

- `models/opening_stage7_torch_wide.pt`
- `models/turn1_stage6_torch_wide.pt`
- `models/turn2_stage8.pkl`
- `models/turn3_stage6.pkl`

The current opening model is a CUDA-trained Torch MLP from 20k Spot teacher
samples using `models/turn1_stage6_torch_wide.pt` as the downstream value
model with 16 sampled future deals. Its 2k holdout metrics are
`top1 = 0.678500`, `top3 = 0.945000`, and `avg_regret = 0.066201`.
On the same holdout, the previous `models/opening_stage6.pkl` reached
`top1 = 0.134500`, `top3 = 0.255500`, and `avg_regret = 1.214851`.
The current T3 model is trained from the 100k Spot teacher run and reached
`top1 = 0.945675` on the unused 80k-sample validation slice.
The current T2 model is trained from 61k teacher samples using
`models/turn3_stage6.pkl` as the downstream value model. Its 6.1k holdout
metrics are `top1 = 0.718852`, `top3 = 0.949016`, and
`avg_regret = 0.017642`.
The current T1 model is a CUDA-trained Torch MLP from 50k Spot teacher samples
using `models/turn2_stage8.pkl` as the downstream value model. Its 5k holdout
metrics are `top1 = 0.761600`, `top3 = 0.975600`, and
`avg_regret = 0.020282`. On the same holdout, the previous
`models/turn1_stage5.pkl` reached `top1 = 0.363200`, `top3 = 0.621000`, and
`avg_regret = 0.449313`.

At T0, the policy enumerates every legal placement of the dealt five cards
exactly, for both players in deal order. In HU play, already visible opponent
board cards are removed from the next-card candidates. If `--opening-model` is
supplied, it is used directly for the opening placement. Otherwise, if
`--turn1-model` is supplied, each T0 placement is scored by sampling the next
3-card turn and taking the best Turn1-model action value.
`--opening-lookahead-samples 0` enumerates every next 3-card deal, which is
exact for this one-ply lookahead but much slower. Use `--prediction-threads 1`
for HGB models unless a dedicated benchmark shows the default thread pool is
faster on the target machine.

Evaluate named profiles with seat swapping and optional hand traces:

```powershell
python -m ofc_regular.evaluate_matchups --profile-a current --profile-b old_opening --games 1000 --seed 42 --prediction-threads 1 --trace-output outputs/current_vs_old_opening_traces.jsonl --trace-limit 10 --output outputs/current_vs_old_opening_eval.json
```

Supported profiles are `current`, `old_opening`, `random_exact_final`,
`late_t2t3`, and `hu_t3_candidate`. The `hu_t3_candidate` profile uses the
current T0-T3 self-board models plus the Cycle10 HU T3 candidate with margin 8.
The evaluator reports score EV, 95% confidence interval, and
FL-aware board classifications: `FL成功`, `FL狙いバースト`, `通常完成`,
and `通常バースト`.

Collect self-play-distribution teacher samples for the next self-board
iteration. T3 samples are labeled directly by T4 exact search, while T0/T1
samples bootstrap through the downstream model:

```powershell
python -m ofc_regular.self_play_teacher_data --phase turn3 --samples 100000 --future-samples 64 --seed 41 --output outputs/turn3_teacher_stage7_selfplay_100k_f64.jsonl
python -m ofc_regular.self_play_teacher_data --phase opening --samples 100000 --future-samples 16 --seed 42 --output outputs/opening_teacher_stage8_selfplay_100k_f16.jsonl
python -m ofc_regular.self_play_teacher_data --phase turn1 --samples 100000 --future-samples 64 --seed 43 --output outputs/turn1_teacher_stage7_selfplay_100k_f64.jsonl
```

These JSONL files keep the existing self-board action-value schema. Train them
as `models/turn3_stage7_torch_wide.pt`,
`models/opening_stage8_torch_wide.pt`, and
`models/turn1_stage7_torch_wide.pt`. Accept a model only if holdout
`avg_regret` improves and seat-swapped matchup EV does not regress. When
rebuilding the full chain, train in order from T3 to T2 to T1 to T0.

For a larger T3 cycle, keep random and self-play-distribution shards separate
until training time:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\Run-T3TeacherChunks.ps1 -TotalSamples 100000 -ChunkSize 1000 -FutureSamples 64 -Seed 41 -MinScoreGap 0 -OutputDir outputs\t3_stage7_random_chunks -MergedOutput outputs\t3_stage7_random_100k_f64.jsonl
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\Run-SelfPlayTeacherChunks.ps1 -Phase turn3 -TotalSamples 50000 -ChunkSize 100 -FutureSamples 64 -Seed 42 -OutputDir outputs\t3_stage7_selfplay_chunks -MergedOutput outputs\t3_stage7_selfplay_50k_f64.jsonl
```

Evaluate candidates on the deterministic holdout split and then by explicit
model-set seat swapping:

```powershell
python -m ofc_regular.evaluate_action_value_split --validation outputs\t3_stage7_mixed.jsonl --seed 44 --model models\turn3_stage6.pkl --model models\turn3_stage7_torch_wide.pt --output outputs\turn3_stage6_vs_stage7_holdout.json
python -m ofc_regular.evaluate_model_set_matchup --name-a t3_stage7 --name-b baseline --turn3-a models\turn3_stage7_torch_wide.pt --turn3-b models\turn3_stage6.pkl --games 1000 --seed 45 --prediction-threads 1 --output outputs\t3_stage7_vs_baseline_matchup.json
```

Start the separate HU-aware data line from Turn3 without changing the existing
self-board models:

```powershell
python -m ofc_regular.hu_teacher_data --samples 1000 --future-samples 64 --seed 42 --output outputs/hu_turn3_stage1_teacher_smoke.jsonl
```

HU-aware samples use `schema = "hu_stage1"` and include `seat`,
`to_act_order`, `opponent_board`, and `dead_cards`; they are intentionally not
fed into the existing self-board trainers. Self-play rollout teachers and
matchup evaluation track prior discards as dead cards, so rollout futures do
not reuse cards that were already discarded earlier in the hand.

For actual HU play, prefer the self-play rollout distribution over random
complete opponent boards. Final-turn exact search uses HU terminal scoring when
the opponent board is already complete; otherwise it falls back to standalone
final-turn scoring because the opponent's unknown final deal is not exacted in
the policy path yet.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\Run-HuSelfPlayTeacherChunksParallel.ps1 -TotalSamples 10000 -ChunkSize 50 -FutureSamples 8 -MaxParallel 4 -Seed 46 -OutputDir outputs\hu_turn3_stage1_selfplay_rollout_chunks -MergedOutput outputs\hu_turn3_stage1_selfplay_rollout_10k_f8.jsonl
python -m ofc_regular.train_hu_turn3 --input outputs/hu_turn3_stage1_selfplay_rollout_10k_f8.jsonl --model-output models/hu_turn3_stage1_hgb.pkl --metrics-output outputs/hu_turn3_stage1_metrics.json --model-type hgb
python -m ofc_regular.play_ai --games 10 --opening-model models/opening_stage7_torch_wide.pt --turn1-model models/turn1_stage6_torch_wide.pt --turn2-model models/turn2_stage8.pkl --turn3-model models/turn3_stage6.pkl --hu-turn3-model models/hu_turn3_stage1_hgb.pkl
python -m ofc_regular.evaluate_hu_policy_advantage --validation outputs/hu_turn3_stage1_selfplay_rollout_10k_f8.jsonl --hu-model models/hu_turn3_stage1_hgb.pkl --baseline-turn3-model models/turn3_stage6.pkl --hu-min-margin 2.0 --hu-max-self-regret 0.25 --output outputs/hu_t3_stage1_teacher_advantage.json
python -m ofc_regular.evaluate_model_set_matchup --name-a hu_t3_stage1 --name-b baseline --hu-turn3-a models/hu_turn3_stage1_hgb.pkl --hu-turn3-min-margin-a 2.0 --hu-turn3-max-self-regret-a 0.25 --games 1000 --prediction-threads 1 --output outputs/hu_t3_stage1_vs_baseline.json
```

Local timing so far: HU self-play rollout `future-samples=8` writes about 50
samples per 85-95 seconds with `MaxParallel=2` on this machine. The 100-sample
cycle was `-1.968 EV/hand` over 100 paired seeds, while the 600-sample cycle was
`-1.214 EV/hand` over 200 paired seeds. This is still below baseline, but the
direction improved with more data. Continue with at least 2k-5k rollout samples
before treating the model quality as meaningful. After adding prior-discard
dead-card tracking, a 100-sample pilot still lost about `-2.05 EV/hand`
ungated, so current HU T3 models should be treated as experimental. Use
`--hu-turn3-min-margin-*` to fall back to the self-board T3 baseline unless the
HU model's predicted improvement over the baseline action is large enough.

The first expanded-feature HU run added opponent FL-threat features and prior
discard tracking. A 500-sample `future-samples=8` model improved ungated EV to
about `-1.47 EV/hand` over 200 paired seeds, but still lost. With margin `3.0`
it looked neutral over 200 paired seeds, then failed the stronger 1,000 paired
seed check at about `-0.32 EV/hand`. A 100-sample `future-samples=32` pilot had
excellent teacher holdout, but still lost in seat-swap (`-1.23 EV/hand` at
margin `2.0`, `-0.50 EV/hand` at margin `4.0` over 500 paired seeds). Do not
promote these HU T3 models; use `evaluate_hu_policy_advantage` plus seat-swap
EV because teacher advantage alone is not predictive enough yet.

Override tracing showed the core failure mode: even at margin `4.0`, the f32
pilot overrode 104 of 1,000 candidate T3 decisions and those overrides lost
about `-4.81 EV` each against the baseline action in same-future
counterfactual rollouts. Add `--hu-turn3-max-self-regret-*` to require the
self-board T3 model to be nearly indifferent before HU can override. With
margin `4.0` and max self-regret `0.25`, the f32 pilot was `+0.0145 EV/hand`
over 1,000 paired seeds, but it only changed 4 paired seeds; this is a safe
guardrail, not a strong adopted HU T3 model.

`trace_hu_turn3_overrides` writes the dead-card list for each override, and
`build_hu_hard_negative_data` converts losing overrides into HU teacher JSONL
samples. A first hard-negative pass used the 39 losing overrides from the f32
margin-4 trace, repeated them 5x, and retrained an HGB model. It did not improve
adoption quality: ungated margin-4 EV was about `-0.56 EV/hand` over 500 paired
seeds because the model created new bad overrides; with self-regret `0.25` it
was only baseline-equivalent (`+0.0006 EV/hand` over 1,000 paired seeds). For
the next cycle, prefer changing the teacher objective or collecting larger
high-quality self-play data over simply appending a small hard-negative set.

`apply_hu_self_regret_penalty` applies that idea directly to HU teacher JSONL by
subtracting a self-board regret penalty from each action score before training.
This also did not produce an adoptable model on the current small datasets. On
the f32 100-sample teacher, penalty weight `1.0` changed 32% of best actions and
still lost at margin `4.0` (`-0.76 EV/hand` over 200 paired seeds); penalty
weight `0.25` still lost (`-0.55 EV/hand` over 200 paired seeds). On the 600 f8
dataset, weight `0.25` changed 41% of best actions and still lost (`-0.53
EV/hand` over 200 paired seeds). The useful safety mechanism remains runtime
`--hu-turn3-max-self-regret-*`; score-penalty relabeling needs more data or a
better calibrated objective before it can replace the gate.

The next HU T3 line should use the direct self-regret penalty during teacher
rollout, then train a higher-capacity Torch MLP instead of another HGB-only
model. A 100-sample `future-samples=32` direct-penalty pilot with weight `0.25`
reduced the margin-4 override loss from roughly `-4.81 EV` to `-1.45 EV` per
override, but it still failed seat-swap adoption: margin `4.0` was about
`-0.26 EV/hand` over 1,000 paired seeds, and margin `8.0` was about `-0.13
EV/hand` over 1,000 paired seeds. Treat this as evidence that the objective is
closer but the dataset/model are still too weak.

Use the streaming HU Torch trainer for the next large run so the local GPU can
consume much larger shard-merged JSONL files without materializing all action
features at once:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\Run-HuSelfPlayTeacherChunksParallel.ps1 -TotalSamples 10000 -ChunkSize 25 -FutureSamples 32 -MaxParallel 4 -Seed 2026062201 -SelfRegretPenaltyWeight 0.25 -SelfRegretFree 0.0 -OutputDir outputs\hu_turn3_stage1_cycle9_direct_penalty_w025_10k_f32_chunks -MergedOutput outputs\hu_turn3_stage1_cycle9_direct_penalty_w025_10k_f32.jsonl
python -m ofc_regular.train_torch_hu_turn3_streaming --input outputs\hu_turn3_stage1_cycle9_direct_penalty_w025_10k_f32.jsonl --model-output models\hu_turn3_stage1_cycle9_direct_penalty_w025_torch_wide.pt --metrics-output outputs\hu_turn3_stage1_cycle9_direct_penalty_w025_torch_wide_metrics.json --device auto --epochs 80 --batch-size 32768 --hidden-layer-sizes 2048,1024,512,256 --best-action-weight 2.0
python -m ofc_regular.evaluate_hu_policy_advantage --validation outputs\hu_turn3_stage1_cycle9_direct_penalty_w025_10k_f32.jsonl --hu-model models\hu_turn3_stage1_cycle9_direct_penalty_w025_torch_wide.pt --baseline-turn3-model models\turn3_stage6.pkl --hu-min-margin 4.0 --output outputs\hu_turn3_stage1_cycle9_teacher_advantage_margin4.json
python -m ofc_regular.evaluate_model_set_matchup --name-a hu_t3_cycle9_margin4 --name-b baseline --hu-turn3-a models\hu_turn3_stage1_cycle9_direct_penalty_w025_torch_wide.pt --hu-turn3-min-margin-a 4.0 --games 1000 --seed 2026062202 --prediction-threads 1 --output outputs\hu_turn3_stage1_cycle9_margin4_vs_baseline_1000.json
```

A local 200-sample direct-penalty extension used the same `future-samples=32`
teacher and took about 560 seconds with `MaxParallel=4` and 10-sample chunks.
Combined with the previous 100-sample pilot, the 300-sample dataset is still
not enough for adoption. A Torch MLP (`1024,512,256`) trained successfully on
CUDA, but its holdout action regret was poor (`1.33`) and margin-4 seat-swap
lost badly over 200 paired seeds (`-1.21 EV/hand` for the 100-sample Torch
pilot). On the 300-sample combined set, ExtraTrees had the best holdout action
regret (`0.24`) but still did not beat baseline in seat-swap: margin `4.0` was
about `-0.16 EV/hand` over 200 paired seeds, margin `8.0` was about `-0.07
EV/hand`, and margin `4.0` plus self-regret `0.25` was baseline-like at about
`-0.015 EV/hand`. The next useful jump is therefore not another tiny model
tweak; collect at least 5k-10k direct-penalty self-play samples and rerun the
same model comparison.

The HU Torch trainer now also supports a pairwise ranking objective:
`--ranking-loss-weight`, `--ranking-max-margin`, and
`--ranking-gap-tolerance`. This directly trains best-vs-other action gaps in
addition to weighted MSE. On the same 300-sample direct-penalty dataset,
ranking loss (`--ranking-loss-weight 1.0`) fixed the Torch action-selection
failure: holdout avg regret improved from `1.33` to `0.25`, close to
ExtraTrees (`0.24`). However, the learned margins became too aggressive.
Margin `4.0` without self-regret gate lost about `-0.93 EV/hand` over 200
paired seeds. Margin `8.0` was safer but still slightly negative. Margin `4.0`
plus self-regret `0.25` was exactly baseline-equivalent over 1,000 paired
seeds (`0.000 EV/hand`, 95% CI about `[-0.039, +0.039]`, 4 wins / 5 losses /
991 ties). Use ranking loss in the next large-data cycle, but keep the
self-regret gate until seat-swap proves the ungated overrides are profitable.

Cycle10 added another 200 direct-penalty `future-samples=32` self-play samples
for a 500-sample dataset:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\Run-HuSelfPlayTeacherChunksParallel.ps1 -TotalSamples 200 -ChunkSize 10 -FutureSamples 32 -MaxParallel 4 -Seed 2026062501 -SelfRegretPenaltyWeight 0.25 -SelfRegretFree 0.0 -OutputDir outputs\hu_turn3_stage1_cycle10_direct_penalty_w025_200_f32_chunks -MergedOutput outputs\hu_turn3_stage1_cycle10_direct_penalty_w025_200_f32.jsonl
python -m ofc_regular.train_hu_turn3 --input outputs\hu_turn3_stage1_cycle10_direct_penalty_w025_500_f32.jsonl --model-output models\hu_turn3_stage1_cycle10_direct_penalty_w025_500_f32_extra_trees.pkl --metrics-output outputs\hu_turn3_stage1_cycle10_direct_penalty_w025_500_f32_extra_trees_metrics.json --model-type extra_trees --n-estimators 500 --min-samples-leaf 2 --seed 2026062502
```

The 500-sample ExtraTrees model is the first HU T3 candidate to clear a
1,000-paired-seat-swap check, but only with a high confidence gate:
`models/hu_turn3_stage1_cycle10_direct_penalty_w025_500_f32_extra_trees.pkl`
at `--hu-turn3-min-margin-a 8.0` scored `+0.0496 EV/hand` over 1,000 paired
seeds, 95% CI `[+0.0017, +0.0975]`, with 7 wins / 1 loss / 992 ties. Override
trace at the same seeds wrote only 12 overrides out of 2,000 T3 decisions
(`0.6%`), but same-future counterfactual delta averaged `+8.27 EV`, with 7
winning and 1 losing override. A second 1,000-paired seed range also cleared:
`+0.0696 EV/hand`, 95% CI `[+0.0254, +0.1138]`, with 12 wins / 0 losses / 988
ties. Combined across the two 1,000-paired checks, margin-8 is about `+0.0596
EV/hand` with 19 wins / 1 loss / 1,980 ties. Margin `4.0` remained
negative-ish over 200 paired seeds (`-0.081 EV/hand`), while margin `4.0` plus
self-regret `0.25` made no overrides in that 200-paired run. For now, treat
Cycle10 margin-8 as the first conservative HU T3 candidate worth carrying
forward; it improves EV only by rare, high-confidence overrides.

Cycle11 added another 500 direct-penalty `future-samples=32` samples and
combined them with Cycle10 into a 1,000-sample dataset. This did not improve
the adopted candidate. The default ExtraTrees 1,000-sample model had holdout
avg regret `0.290` and margin-8 seat-swap over the original 1,000-paired seed
range was `-0.027 EV/hand` with 7 wins / 10 losses / 983 ties. More regularized
ExtraTrees variants with `min_samples_leaf=5` and `10` were also negative-ish
over 200 paired seeds. Keep
`configs/hu_turn3_stage1_cycle10_margin8_candidate.json` as the current HU T3
candidate instead of promoting the Cycle11 retrains. The next data cycle should
improve sample quality or calibration, not just append more of the same
distribution.

The same candidate can also be evaluated through the named-profile matchup
CLI:

```powershell
python -m ofc_regular.evaluate_matchups --profile-a hu_t3_candidate --profile-b current --games 1000 --seed 2026062507 --prediction-threads 1 --output outputs\hu_t3_candidate_vs_current_profile_1000.json
```

For the next data cycle, prefer targeted HU T3 teacher data instead of
appending untargeted self-play data. Mine high-margin candidate/baseline
disagreement states first, then label those states with rollout teacher scores.
This keeps the cheap rarity search separate from the expensive rollout step:

```powershell
python -m ofc_regular.mine_hu_turn3_states --states 100 --seed 2026063701 --selection-hu-turn3-model models\hu_turn3_stage1_cycle10_direct_penalty_w025_500_f32_extra_trees.pkl --selection-min-margin 6 --selection-max-margin 12 --selection-require-disagreement --prediction-threads 1 --output outputs\hu_turn3_stage1_cycle12_mined_margin6_12_states_100.jsonl
python -m ofc_regular.label_hu_turn3_states --input outputs\hu_turn3_stage1_cycle12_mined_margin6_12_states_100.jsonl --output outputs\hu_turn3_stage1_cycle12_mined_margin6_12_teacher_100_f32.jsonl --seed 2026063702 --future-samples 32 --self-regret-penalty-weight 0.25 --self-regret-free 0.0
```

For Spot VM runs, use the chunked runners so partial work is saved and can be
resumed:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\Run-HuTurn3StateMineChunksParallel.ps1 -TotalStates 100 -ChunkSize 10 -MaxParallel 4 -Seed 2026063801 -SelectionHuTurn3Model models\hu_turn3_stage1_cycle10_direct_penalty_w025_500_f32_extra_trees.pkl -SelectionMinMargin 6 -SelectionMaxMargin 12 -SelectionRequireDisagreement -PredictionThreads 1 -OutputDir outputs\hu_turn3_stage1_cycle12_mined_margin6_12_state_chunks -MergedOutput outputs\hu_turn3_stage1_cycle12_mined_margin6_12_states_100.jsonl
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\Run-HuTurn3StateLabelChunksParallel.ps1 -InputDir outputs\hu_turn3_stage1_cycle12_mined_margin6_12_state_chunks -OutputDir outputs\hu_turn3_stage1_cycle12_mined_margin6_12_teacher_chunks -MergedOutput outputs\hu_turn3_stage1_cycle12_mined_margin6_12_teacher_100_f32.jsonl -FutureSamples 32 -MaxParallel 4 -Seed 2026063802 -SelfRegretPenaltyWeight 0.25 -SelfRegretFree 0.0
```

HU T3 feature encoding now includes additional row-matchup features: hero vs
opponent row fill, completed hand category, row royalty, premium draw
potential, and completed-row order deltas. Old HU models remain loadable because
prediction aligns expanded feature matrices back to the estimator's stored
feature count. A quick retrain on the existing Cycle10 500-sample data improved
holdout regret (`0.261 -> 0.172`) but did not improve the same 200-paired
seat-swap slice at margin 8; it tied the Cycle10 candidate at `+0.035 EV/hand`.
So the next real model should combine these features with targeted
candidate/baseline disagreement data rather than promoting a feature-only
retrain.

A 50-sample margin 0-12 disagreement pilot was collected with
`future-samples=32` and direct self-regret penalty `0.25`. It was easy to
collect but mostly low-margin data: average candidate margin `1.08`, max
`4.23`, and no samples at margin 6+. Adding it to Cycle10 and retraining with
the new matchup features improved holdout regret to `0.078`, but seat-swap
still tied the Cycle10 candidate at margin 8 (`+0.035 EV/hand` over 200 paired
seeds) and got worse at margin 4 (`-0.136 EV/hand`). Do not scale this
low-margin filter unchanged.

Using the 12 existing Cycle10 margin-8 override counterfactual rows as a light
high-margin correction also failed: the retrain scored `-0.015 EV/hand` at
margin 8 over the same 200 paired seeds. Keep the Cycle10 margin-8 candidate as
the active candidate. The next data attempt should mine true high-margin
candidate/baseline disagreements, roughly margin 6-12 or 8+, instead of mostly
low-margin disagreements or naive trace replay.

The two-step high-margin miner/labeler was added for that purpose. A 10-state
margin 6-12 mining pilot found 10 states in 352 hands / 703 T3 attempts
(`~1.4%` hit rate) in 126 seconds; the mined margins averaged `8.29`. Labeling
those 10 states with `future-samples=32` took 41 seconds. Mixing just these 10
samples into Cycle10 was not enough: holdout regret worsened to `0.495`, and a
100-paired margin-8 seat-swap slice was only `+0.04 EV/hand`, essentially the
same rare-override behavior as the active Cycle10 candidate. Scale the
high-margin mined set substantially before judging a new model.

Cycle13 scaled this to 100 mined margin 6-12 states using the chunk runners.
Mining took 10 chunks of 10 states; the merged state file has 100 rows. The
mined margins were still concentrated below the adoption gate: min `6.66`, avg
`7.74`, max `9.57`, with 30 states at margin 8+ and none at margin 10+.
Labeling all 100 states with `future-samples=32` produced 100 teacher samples;
teacher score gap averaged `1.67`. Mixing those 100 samples into Cycle10 and
retraining ExtraTrees did not improve the candidate: holdout regret was
`0.274`, and margin-8 seat-swap over the same 200 paired seed slice scored
`+0.005 EV/hand` with 2 wins / 1 loss / 197 ties. As support for the active
Cycle10 candidate, the model was also weaker; support threshold 10 removed the
single observed losing override in the Cycle10 trace but reduced approximate
EV to `+0.0215 EV/hand`, below Cycle10's `+0.0496`. Keep Cycle10 margin-8 as
the active candidate. The next high-margin mining pass should target a stronger
band such as margin 8-12 or add teacher weighting/calibration instead of simply
mixing all margin 6-12 samples equally.

Cycle14 mined 100 stricter margin 8-12 states. This was much slower: the first
eight 10-state chunks took about 6.3-6.7 minutes each, versus roughly 2 minutes
for margin 6-12. The distribution was stronger: min `8.26`, avg `9.81`, max
`11.54`, with 67 states at margin 9+ and 40 at margin 10+. Teacher score gap
averaged `3.57`. However, equal mixing still did not improve the deployed
candidate. The 600-sample ExtraTrees retrain had holdout regret `0.436`, made
no margin-8 overrides over the 200-paired seed slice (`0.000 EV/hand`), and
lost at margin 4 (`-0.161 EV/hand`). As a support model for Cycle10, threshold
4 removed the losing override in the trace but reduced approximate EV to
`+0.029`, still below Cycle10's `+0.0496`. Keep Cycle10 margin-8 active. The
next attempt should change training/calibration rather than only mining more:
try source weighting, action-score calibration, or a separate gate model trained
on override-level wins/losses.

Cycle15 added source weighting to the HU T3 sklearn trainer via
`--source-weight SOURCE=WEIGHT`. The goal was to test whether Cycle14 failed
mainly because the mined high-margin distribution was mixed too heavily. On the
Cycle10+Cycle14 600-sample file, downweighting `mined_state_rollout` to `0.25`
improved holdout regret versus equal mixing (`0.436 -> 0.340`) but scored
`-0.010 EV/hand` at margin 8 over the same 200 paired seed slice. Weight `0.5`
scored `+0.035 EV/hand` at margin 8, matching the active Cycle10 candidate on
that slice, but it still collapsed at margin 4 (`-0.173 EV/hand`). As a support
model for Cycle10, the `0.5` model preserved the Cycle10 trace EV at low support
thresholds (`+0.0496 EV/hand` for thresholds 0-6) but did not improve it; higher
thresholds reduced EV to `+0.0245`. Keep Cycle10 margin-8 active. Source
weighting alone is not enough; the remaining issue is likely calibration and
model expressiveness around rare overrides, not just raw data volume.

Cycle16 added the same `--source-weight` support to the streaming HU T3 Torch
trainer, including source-weighted ranking loss. A 1024/512/256 MLP with
`mined_state_rollout=0.5`, best-action weight `2.0`, and ranking loss `0.20`
trained on GPU but was weaker than the tree models: holdout avg regret was
`0.401`, and margin-8 seat-swap over the same 200 paired seed slice scored
`-0.118 EV/hand`. A smaller 256/128/64 plain MLP without ranking loss was even
worse, with holdout avg regret `1.040`. Do not promote these Torch models.
With only 600 HU samples, higher-capacity MLPs are not the next bottleneck fix;
the next useful direction is a calibrated gate/override model or more labeled
override data, not another unconstrained action-value retrain.

Cycle17 added a direct HU Turn3 override gate. The runtime now supports
`--hu-turn3-gate-*` and `--hu-turn3-min-gate-probability-*`; the gate sees the
HU candidate action, the self-board baseline action, predicted HU margin,
self-regret, row/royalty/discard features, and returns an accept probability.
`train_hu_turn3_gate` trains this from override traces, and
`analyze_hu_gate_trace` sweeps probability thresholds on a separate trace.

The first gate trained on Cycle11's 32 margin-8 override rows looked good
in-sample but did not improve the active Cycle10 candidate. Applied to the
Cycle10 margin-8 trace, threshold `0.5` kept the same trace EV
(`+0.0496 EV/hand`) and still kept the single losing override. Seat-swap over
the usual 200 paired seed slice also tied Cycle10 at `+0.035 EV/hand`.

To get more active-model labels, Cycle10 was traced at margin `4.0` over 200
paired seeds (`seed 2026065301`). Even at margin 4 it only made 12 overrides:
4 wins, 1 loss, 7 ties, average delta `+2.0`, paired score `+0.060 EV/hand`.
A stage2 logistic gate trained on Cycle11 plus this new trace improved the
same seed slice at margin 4 to `+0.125 EV/hand`, but failed to generalize to
the usual seed slice: margin4+gate scored `-0.005 EV/hand` over 200 paired
seeds. At margin 8, the same gate tied the active candidate at `+0.035
EV/hand`; trace analysis showed it actually reduced Cycle10's margin-8 trace EV
from `+0.0496` to `+0.0436` at threshold `0.5`. Do not promote the gate yet.
The gate infrastructure is useful, but it needs more active-model override
labels before lowering the adoption margin below 8.

Cycle18 added a resumable parallel override-trace runner:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\Run-HuTurn3OverrideTraceChunksParallel.ps1 -TotalGames 400 -ChunkSize 50 -MaxParallel 4 -Seed 2026065601 -HuTurn3MinMargin 4.0 -PredictionThreads 1 -OutputDir outputs\hu_turn3_stage1_cycle18_cycle10_margin4_trace_400_chunks -MergedDecisionOutput outputs\hu_turn3_stage1_cycle18_cycle10_margin4_override_trace_400.jsonl -MergedSummaryOutput outputs\hu_turn3_stage1_cycle18_cycle10_margin4_override_trace_400_summary.json
```

The runner writes one summary JSON and one decision JSONL per chunk, skips
completed chunks on rerun, then merges the decision rows and writes an aggregate
summary. A 20-game smoke produced 2 override rows and verified the merge path.
The 400-paired Cycle10 margin-4 run produced 32 override rows at a 4.0%
override rate, but the raw margin-4 policy lost on that seed range:
weighted paired score `-0.041 EV/hand`, 9 winning overrides / 7 losing /
16 ties, delta sum `-32.8`. A plain margin threshold sweep on the same trace
suggested margin 5 and 6 were better locally (`+0.045` and `+0.050 EV/hand`
trace approximation), but this did not survive seat-swap checks.

A stage3 logistic gate trained on Cycle11 plus Cycle17/Cycle18 active-model
margin-4 traces had 76 total override rows. Holdout was small but positive
under gate thresholds, with no selected losing holdout overrides. In seat-swap,
however, it did not beat the active Cycle10 margin-8 candidate. On the usual
200 paired seed slice, margin5+gate p0.7 scored `+0.0375 EV/hand`, only barely
above the active slice result and inside noise; on the second seed slice it
fell to `-0.005 EV/hand` while active margin8 scored `+0.045 EV/hand`.
Margin6+gate p0.7 simply tied active behavior on the usual slice (`+0.035
EV/hand`). Keep Cycle10 margin8 active. The next useful data pass should
collect more active-model margin4/5/6 traces across multiple seed ranges before
trusting any learned gate.

Cycle19 used the resumable trace runner for another 800 paired Cycle10
margin-4 seeds (`seed 2026066001`). This produced 77 override rows at a 4.8%
override rate. Raw margin4 was again unsafe on the full seed range:
weighted paired score `-0.025 EV/hand`, 22 winning overrides / 21 losing /
34 ties, delta sum `-40.0`. The simple margin sweep on this trace looked
locally promising: margin5 `+0.0746`, margin6 `+0.0679`, margin7 `+0.0799`,
and margin8 `+0.0736 EV/hand` by same-future trace approximation. That was
not enough evidence because earlier trace-local improvements failed seat-swap.

A stage4 logistic gate trained on Cycle11 plus Cycle17/Cycle18/Cycle19 had
153 total override rows. Holdout was still small but better populated
(31 rows); p0.6 kept 11 holdout overrides for delta `+85.2`, with 5 wins /
2 losses / 4 ties. In seat-swap, however, the candidate still did not beat
the active margin8 gate. On seed `2026062507`, margin5+gate p0.6 scored
`+0.0375 EV/hand` and margin6+gate p0.6 scored `+0.035 EV/hand`. On seed
`2026063501`, margin5+gate p0.6 dropped to `+0.0045 EV/hand`, while the active
Cycle10 margin8 candidate scored `+0.045 EV/hand`. Keep Cycle10 margin8 active.
The active-model low-margin gate likely needs much more data and possibly
seed-range holdout by construction; random row holdout is too weak for this
rare-override problem.

Use `analyze_hu_override_trace` to sweep confidence thresholds from an override
trace before spending another full seat-swap run:

```powershell
python -m ofc_regular.analyze_hu_override_trace --input outputs\hu_turn3_stage1_cycle11_direct_penalty_w025_1000_f32_extra_trees_margin8_override_trace_1000.jsonl --paired-seeds 1000 --thresholds 8,10,12,15,20,25 --output outputs\hu_turn3_stage1_cycle11_margin8_override_threshold_sweep.json
```

This confirmed the Cycle11 failure mode. At margin `8.0`, Cycle11 made 32
overrides versus Cycle10's 12, but the added overrides were poorly calibrated:
Cycle11 override delta averaged `-1.71 EV` with 7 wins / 10 losses. Raising the
threshold did not rescue it; margin `15.0` was still about `-0.0045 EV/hand`
from the trace approximation, and margin `20.0` was also negative.

A small hard-negative repair was also tried by converting the 10 losing
Cycle11 margin-8 overrides into HU teacher samples and repeating them 10x on
top of the Cycle10 500-sample data. That model had holdout avg regret `0.289`
and failed seat-swap badly at margin `8.0`: about `-0.38 EV/hand` over 200
paired seeds on the original seed range and `-0.42 EV/hand` on the second
range. This confirms that naive hard-negative repetition creates new bad
overrides instead of cleanly calibrating the model.

A two-model agreement gate was added for calibration experiments:
`--hu-turn3-support-*` plus `--hu-turn3-min-support-margin-*`. The primary HU
model must pass its own margin gate and the support model must also score the
chosen action above the self-board baseline by the support margin. Using
Cycle11 as primary at margin `8.0` and Cycle10 as support at support margin
`4.0` removed the observed bad Cycle11 overrides in trace analysis: 6 of 32
Cycle11 overrides survived, with 4 wins / 0 losses / 2 ties and trace
approximation `+0.0321 EV/hand`. Seat-swap matched that approximation over
1,000 paired seeds (`+0.0321 EV/hand`, CI `[-0.0075, +0.0717]`) and was also
positive-ish over 200 paired seeds on the second range (`+0.045 EV/hand`).
This is safer than Cycle11 alone, but still weaker than the Cycle10 margin-8
candidate, so it is recorded as
`configs/hu_turn3_stage1_cycle11_margin8_support_cycle10_4_experiment.json`
and not promoted.

Use `analyze_hu_support_gate_trace` to choose support margins from a trace:

```powershell
python -m ofc_regular.analyze_hu_support_gate_trace --input outputs\hu_turn3_stage1_cycle11_direct_penalty_w025_1000_f32_extra_trees_margin8_override_trace_1000.jsonl --support-model models\hu_turn3_stage1_cycle10_direct_penalty_w025_500_f32_extra_trees.pkl --paired-seeds 1000 --thresholds 0,2,4,6,8 --output outputs\hu_turn3_stage1_cycle11_margin8_support_cycle10_threshold_sweep.json
```

Use `evaluate_hu_candidate_config` to rerun a candidate or experiment without
hand-copying model paths and runtime gates:

```powershell
python -m ofc_regular.evaluate_hu_candidate_config --config configs\hu_turn3_stage1_cycle10_margin8_candidate.json --games 1000 --seed 2026062507 --prediction-threads 1 --progress-every 200 --output outputs\hu_turn3_cycle10_candidate_config_eval_1000.json
```
