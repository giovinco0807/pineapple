# M3a locked calibration 完了状態の監査 (2026-07-28)

## 事実

`ai/data/m3_behavior_calibration_production_20260713/calibration.json` は
既に存在し、`stage=calibration_complete`、`pipeline_complete=true`である。
milestoneドキュメント（2026-07-13版）の「natural評価とlocked calibrationは
実行中」という記述は古い。

- natural: 134,057 roots / 536,228 decisions / 135 shards、収集・評価とも
  complete。evaluation content SHA-256
  `f9468cc5443dcec6f8c77ffbaa49f4f3e44d34adfcff4d9c11a1badda5295570`
- challenge: 24,000 roots / 96,000 rows / 24 shards、収集・評価ともcomplete
- calibration artifact SHA-256
  `24fdd5d795eaa19a84509ed4af8fe896d2e1018c9d9134cf53120b07d50d2f97`

## gate v2の結果: 104チェック中2件失敗、`promotion_eligible=false`

失敗した2件はいずれも点推定のNLL delta（許容 `max_test_nll_delta_vs_t1 = 0/1`、
つまり「識別温度T=1に対しtest NLLが厳密に悪化ゼロ以下」）:

| check | observed (hex) | ≈十進 | 判定 |
|---|---|---|---|
| `quality.test.t1_bb.nll_delta_vs_t1` | `0x1.dc25e38700000p-17` | +1.42e-5 nats | FAIL |
| `quality.test.t2_bb.nll_delta_vs_t1` | `0x1.17b0620000000p-28` | +4.1e-9 nats | FAIL |

一方、同じ層の統計的検定 `nll_delta_vs_t1_ucb`（許容 `1/200`）は両層とも
PASSしており、t1_btn/t2_btnは点推定も負またはゼロでPASSしている。

## 解釈

校正温度はtest上で識別（T=1）と実質同等であり、失敗は「点推定が厳密に
0以下」という、UCB検定より強くノイズに脆い条件に起因する。+4.1e-9 natsは
浮動小数点の丸め水準であり、校正が実害を持つ証拠ではない。

## 推奨（gate version運用ルールに従う）

運用ルール「閾値や意味を変える場合はgate versionを上げ、旧gate結果を残す」に
基づき、gate v3を次のいずれかで定義することを推奨する。

- 案1: 点推定チェックを廃止しUCBチェック（既存、両層PASS）へ一本化する。
- 案2: 点推定許容を`0/1`から微小tolerance（例: `1/10000`）へ緩和する。
- 案3: selection policyを「dev改善が margin 未満なら identity(T=1) を選択」へ
  変更する（identityなら delta=0 で点推定も自明にPASS）。

いずれもv2結果（本fileが参照するartifact）は保存したまま、v3のconfig生成
コード・新artifact・監査を別途作る。**本レポートは記録であり、gate v2の
結果やcalibration artifactを変更していない。**
