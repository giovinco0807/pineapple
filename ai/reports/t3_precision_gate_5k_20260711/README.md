# T3 canonical Joker precision gate（2026-07-11）

## 結論

T3の本番候補プールは、現行union8 Top10に`near_full_expand_gap=5`を追加する構成へ更新した。方針をdevで固定してから、未使用のfinal holdout 1,000件を一度だけ評価し、BB/BTNともexact-best recall 100%、EV loss 0を確認した。処理はすべてローカルのRTX 2060 SUPERで実行し、GCPは使用していない。

追加のspecialist学習は行わなかった。ベースラインの7 missは、候補プールが全合法手まで残り2〜5手の局面だけだったため、残り5手以内なら全合法手へ拡張する決定論的な安全策でmine/dev上の全missを解消できた。モデルを少数missへ再適合させるより過学習リスクが小さく、追加候補数も上限付きである。7 miss中6件はJokerが見えている局面だった。

## データとexact teacher

- 生成seed: `20260713`
- 125 roots × 40 branches = 5,000局面（BB 2,500 / BTN 2,500）
- 一意なdecision: 5,000 / 5,000
- 旧20,000件および先行holdout 400件との重複: 0
- visible Joker: 3,860 / 5,000（77.2%）
- 現行canonical Joker ruleで評価: 5,000 rows / 82,053 legal candidates
- 欠落、index/count不一致、非有限スコア: 0

同じroot由来の枝がsplitをまたがないよう、root単位で分割した。

| split | roots | rows | 用途 |
|---|---:|---:|---|
| mine | 0–74 | 3,000 | miss検出 |
| dev | 75–99 | 1,000 | 方針選択 |
| final | 100–124 | 1,000 | 方針固定後の最終評価 |

## 候補プール評価

mine + dev 4,000件のベースラインunion8 Top10は、3,993 / 4,000、recall 0.99825、平均EV loss 0.0047348054、最大EV loss 8.6415128、EV loss > 0.1が6件、平均pool size 15.123だった。内訳はmineが2,995 / 3,000（最大loss 6.47755）、devが998 / 1,000（最大loss 8.64151）。

canonical action-value modelを加えたunion9は3,994 / 4,000までの改善に留まり、最大loss 8.6415128が残ったため採用しなかった。

選択したunion8 Top10 + `near_full_expand_gap=5`は、devで1,000 / 1,000、平均・最大EV loss 0、平均pool size 15.681だった。ここでconfigを固定し、その後final holdoutを評価した。

| final holdout | exact-best | EV loss mean / max | avg pool | unmatched |
|---|---:|---:|---:|---:|
| BB | 500 / 500 | 0 / 0 | 15.988 | 0 |
| BTN | 500 / 500 | 0 / 0 | 15.556 | 0 |
| 合計 | 1,000 / 1,000 | 0 / 0 | 15.772 | 0 |

## ローカルruntime A/B

dev先頭200件を、候補生成からRust exact rerankまで同じ条件で比較した。

| policy | exact-best | max loss | avg pool | elapsed mean | elapsed median |
|---|---:|---:|---:|---:|---:|
| baseline（gap=0） | 199 / 200 | 8.6415 | 15.660 | 167.261 ms | 89.76 ms |
| selected（gap=5） | 200 / 200 | 0 | 16.665 | 167.583 ms | 108.79 ms |

平均値の差は+0.322 ms（約+0.19%）で、この200件では実質的な平均速度低下は見られなかった。一方、中央値は89.76 msから108.79 msへ上昇したため、今後も平均とtail latencyの両方を監視する。

## 成果物

- 入力: `ai/data/t3_precision_gate_5k_20260711/t3_gate_input_5000.jsonl`
- exact出力: `ai/data/t3_precision_gate_5k_20260711/t3_gate_exact_5000.jsonl`
- teacher: `ai/data/t3_precision_gate_5k_20260711/t3_gate_teacher_5000.jsonl`
- データ監査: `ai/data/t3_precision_gate_5k_20260711/audit.json`
- root split: `ai/data/t3_precision_gate_5k_20260711/splits/`
- pool評価: `ai/data/t3_precision_gate_5k_20260711/eval/`
- runtime A/B: `ai/data/t3_precision_gate_5k_20260711/runtime_bench/`
- 採用config: `ai/config/t3_ev_loss_fresh_pool_20260607.json`
- 比較したunion9 config: `ai/config/t3_canonical_joker_pool_candidate_20260711.json`
- 監査script: `ai/tutor/audit_t3_gate_dataset.py`
- split script: `ai/tutor/split_t3_gate_by_root.py`
- pool evaluator: `ai/tutor/evaluate_t3_runtime_pool_teacher.py`

この結果は独立生成分布に対する実測保証であり、全局面に対する数学的証明ではない。ただし、未使用final holdoutを含むJoker-heavyな5,000局面では、採用policyの候補欠落によるEV lossは観測されなかった。
