# T2 seed52 clean external gate（2026-07-11）

## 結論

最新のT2診断候補であるmulti44 Top3 + pool switcherは、学習に使っていないseed52のcap100 100局面で100/100、平均・最大EV loss 0、miss 0だった。確認した4種類のswitcher（HGB 31/63、ExtraTrees、RandomForest）はすべて同じ結果だった。

seed52評価時に再生成された4つのswitcher artifactはseed51評価時のartifactとSHA-256がすべて一致した。評価データによって学習済みswitcherが変化していないことも確認済みである。

これにより、hard-negativeへ戻していないclean externalの累計は420件から520件へ増え、観測EV lossは引き続き0となった。seed52を追加学習へ使用していないため、この100件のholdout性は維持されている。

ただし、これは各T2 actionをfuture draw 100件で評価したcap100教師に対する実測であり、全future drawを列挙した数学的なfull-exact証明ではない。

## seed52データ

- 生成seed: `20260652`
- source: 400局面（BB 200 / BTN 200）
- source pseudo-eval: 394/400、Top1 98.5%、平均loss 0.0025768、最大loss 0.558963、pseudo miss 6件
- exact選択: 100局面（BB 50 / BTN 50）
- pseudo miss 6件をすべて含め、残りをposition-balanced controlから選択
- 強制選択source index: `305, 309, 353, 360, 380, 387`

## cap100ラベル監査

5チャンクが各20件ずつ完走し、`record_index=0..99`に欠落・重複はなかった。

- records: 100
- legal actions: 12〜27、全件で全legal actionを評価
- future draws per action: 100
- source Top1とcap100 Top1が一致: 44/100
- source Top1が変更: 56/100
- source Top1の平均cap100 regret: 0.693949
- source Top1の最大cap100 regret: 6.450001
- teacher変換: 100 records / 2,292 candidates
- skipped / invalid records / invalid candidates: 0 / 0 / 0

各局面のfull future drawsは5,984または7,140であり、今回の100はその一部である。このため、本書では`exact`ではなく`cap100`または`capped exact`と表記する。

## multi44 Top3評価

| switcher | groups | Top1 | mean EV loss | max EV loss | miss |
|---|---:|---:|---:|---:|---:|
| HGB depth 31 | 100 | 100% | 0 | 0 | 0 |
| HGB depth 63 | 100 | 100% | 0 | 0 | 0 |
| ExtraTrees depth 14 | 100 | 100% | 0 | 0 | 0 |
| RandomForest depth 14 | 100 | 100% | 0 | 0 | 0 |

全44候補源はTop3を提出し、そのunion内からswitcherが1手を選択した。pool upper boundも100/100、EV loss 0だった。

## clean external累計

seed45・49・50はmissを学習へ戻したためclean集計から除外する。

| データ | groups |
|---|---:|
| seed43 | 20 |
| seed44 | 50 |
| seed46 | 50 |
| seed47 | 100 |
| seed48 | 100 |
| seed51 | 100 |
| seed52 | 100 |
| 合計 | 520 |

multi44 Top3 + pool switcherはこの520件で520/520、平均・最大EV loss 0。これは強い実測根拠だが、未知分布への保証ではない。

## 次の作業

T2の追加hard-negative学習は一旦止める。次はruntime昇格準備である。

1. Dドライブ上にある44候補モデル、base/selectorモデル、switcherをportable bundleへ固定する。
2. 現行runtimeへmulti44候補生成とswitcher選択を追加する。
3. artifact hash・特徴量schema・selector順を検証する。
4. 同じ520件の再現テストと、5秒time budget内のlatency gateを通す。
5. その後にT1教師の再生成へ進む。

現在の`ai/config/t1t2_t3_union_runtime_20260607.json`は単一PyTorch action-value shortlistしか読み込まず、multi44 sklearn bundleのloaderを持たない。そのため、今回の合格だけでruntime defaultは変更していない。

## 成果物

- exact入力: `D:/ofc-pineapple-data/.../source400_seed20260652_pseudomiss6_balanced100_20260710/selected_source_pseudomiss6_balanced100_for_exact.jsonl`
- exact chunks: 同ディレクトリの`exact_cap100_balanced100_chunk00_19`〜`chunk80_99`
- 変換済み評価データ: `D:/ofc-pineapple-data/.../eval_data/seed20260652_pseudomiss6_balanced100_cap100_dim789`
- 評価summary: `D:/ofc-pineapple-data/.../t2_pool_switcher_multi44_top3_seed52_pseudomiss6_balanced100_external_20260711/summary.json`
- 診断設定: `ai/config/t2_top1_classifier_diag_20260624.json`

今回のチャンク監査、変換、selector score生成、multi44評価にはGCPを使用していない。
