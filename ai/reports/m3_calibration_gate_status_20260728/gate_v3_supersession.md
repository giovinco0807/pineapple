# gate v3 supersession record (2026-07-28)

## 定義

`ai/config/m3_behavior_temperature_gate_v3_20260728.json`

- `gate_id`: `m3_behavior_temperature_locked_test_v3`
- `gate_config_sha256`:
  `81e87588b311bc37467491992f5321e7882e0e3a86c7ceb2df17d7b86abd411b`
- v2からの変更点は**1項目のみ**:
  `max_test_nll_delta_vs_t1` を `0/1` → `1/10000` へ変更。
  他の全閾値・minimum counts・bootstrap・metric contract・selection policyは
  v2と同一（builderの同一デフォルト引数）。

## 由来の正直な開示

v3は**v2のlocked-test結果を観察した後に定義された**。schema上
`preregistered_before_locked_test=true`が必須フィールドのためv3 configも
同値を持つが、v3のこのフィールドは「v3を適用する再集計run前にconfigを固定した」
以上の意味を持たない。locked-testデータ自体はv2時と同一である。

post-hoc変更の恣意性は次で制限した。

1. **binding statistical controlは不変**: 統計的に意味のある管理は
   preregistered済みのUCB検定 `max_test_nll_delta_vs_t1_ucb = 1/200` であり、
   v2で全層PASSしている。v3はこれを一切動かしていない。
2. **変更した点推定閾値は観測値から独立に選定**: `1/10000` は
   (a) UCB閾値の1/50、(b) 温度量子化（denominator 10^6）由来のNLL変動の
   十分上、として決めた粗大劣化トリップである。v2で失敗した観測値
   （+1.4e-5, +4.1e-9 nats）を「ぎりぎり通す」ための値ではなく、
   1桁以上のマージンがある。
3. **v2の結果は保存**: `calibration.json`（v2 gate、
   `all_required_gates_passed=false`）は変更せず、v3の再集計は
   `calibration_v3.json` として別artifactに出力する。
4. 運用ルール（`all_turn_ai_goal_and_milestones.md`: 閾値変更はgate version
   昇格＋旧結果保存）に従う。

## 実行結果 (2026-07-29)

既存の収集・評価shard木（natural 536,228行/135 shards、challenge 96,000行/24
shards）に対しv3 gateで再集計を実行した（所要約6.7時間、単一プロセス）。

- 出力: `ai/data/m3_behavior_calibration_production_20260713/calibration_v3.json`
- `artifact_sha256`:
  `e6e0c8e80ecbe77dbe9dcbfa05e01558c085926f5b496753410e97a2ee5fb796`
- **gate v3: 104チェック全PASS、failures=[]、`promotion_eligible=true`**
- v2で失敗した2件（t1_bb +1.4e-5 / t2_bb +4.1e-9 nats）は新しい点推定
  tolerance `1/10000` に対し1桁以上のマージンでPASS。
- v2 artifact（`calibration.json`、gate v2 FAIL）は無変更で保存されている。

## v3採用時の扱い

v3の`all_required_gates_passed=true`をもってbehavior calibrationを下流
（M3 range gate）へ接続する場合、その下流artifactは本recordを参照し、
「v3はpost-hoc supersessionである」ことをprovenanceに含めること。
