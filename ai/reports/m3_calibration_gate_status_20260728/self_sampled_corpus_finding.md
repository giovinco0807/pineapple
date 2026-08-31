# 重要な発見: calibration corpusは自己サンプリングである (2026-07-29)

gate v3が104チェック全PASSしたため、その中身を検証した結果、
**このcalibrationはbehavior priorの形状を検証していない**ことが分かった。
gate通過をもってM3aの本来のブロッカーが解消したとは言えない。

## 確認したコード経路

1. `collect_hu_behavior_trace_shards.py:784`
   → `build_known_hu_policy_value_prior_dispatch(workspace_root)`
   （`frozen_behavior_torch.py:773`、既定 `temperature=Fraction(1,1)`）
2. `collect_hu_behavior_traces.py:431`
   → `behavior_model.action_distribution(information)` でその prior の分布を取得
3. 同 `:748` `sample_policy_evaluation(...)` でその分布から行動をサンプル
4. 同 `:824` そのサンプルを `observed_action_key` として記録
5. `behavior_temperature_calibration*` が、同じcheckpointのlogitの下で
   その「観測行動」の尤度を最大化する温度Tを推定

つまり**データ生成分布 = 評価対象モデルのT=1分布**である。

## 結果が機械的に説明できること

| 指標 | 観測値 | 自己サンプリング下での期待 |
|---|---|---|
| fit温度 t1_bb | 1.004445 | ≈1.0（生成温度そのもの） |
| fit温度 t2_btn | identity選択 | ≈1.0 |
| locked-test残差温度 | 0.9887〜1.0097 | ≈1.0 |
| NLL delta vs T=1 | +1.4e-5 / +4.1e-9 nats | ≈0（同一分布） |
| uniform比改善LCB | +0.12〜+1.20 nats | 必ず正（自分のサンプルを自分で予測） |
| marginal ECE | 0.0004〜0.004 | 極小（自己整合） |

v2で失敗した2件が「ノイズ水準」だったのも、真の温度が厳密に1.0で
サンプリング誤差だけが乗るためであり、当然の帰結だった。

## したがってgate v3が示すこと・示さないこと

**示すこと**: Q32量子化・温度パラメータ化された runtime が自身のサンプリング
分布を再現するという配線・数値の整合性。joker層別の診断値が退化していないこと。

**示さないこと**: behavior priorが実際のプレイ（あるいは何らかの外部基準）の
行動頻度に一致すること。corpusが同じpriorから生成されている以上、
**形状の誤りは原理的に検出できない**。

## 正典ドキュメントの元要件との関係

`all_turn_ai_goal_and_milestones.md` は以下を要求していた:

> 出力は観測行動頻度ではなくteacher EVを温度3でsoftmaxしたranking targetの
> logitであり、現状はsynthetic behavior priorに限定して
> `promotion_eligible=false`としている。**独立root splitの実行ログから
> observed-action calibrationを作り直す必要がある。**

本corpusは独立実行ログではなく自己サンプルであるため、**この要件は未充足**。
`calibration_v3.json`の`promotion_eligible=true`は
「gate v2/v3の定義を満たした」という意味であり、上記要件の充足を意味しない。

## 位置づけの修正

- gate v3の再集計自体は有効な作業であり、結果artifactは保持する。
- ただしM3aは「calibration gate通過」では閉じない。実質的な残作業は
  **behavior priorをどう定義するか**という設計判断である。

## 選択肢（要判断）

1. **自己整合を仕様として明示的に受け入れる**: 固定点反復では「相手は
   このpriorに従う」と仮定するので、自己整合温度は必要条件として妥当。
   その場合、gateの意味を「配線・数値整合の検証」と正直に再定義し、
   behavioral realismの主張を全artifactから外す。以後の下流は
   「仮定されたbehavior」であることをprovenanceへ明記する。
2. **独立corpusで作り直す**: 別方策（例: bottom-up BC、exact teacher greedy、
   人間ログ）で実行したtraceを観測行動として校正し直す。元要件に忠実だが、
   corpus生成からやり直しになる。
3. **behavior priorを別物へ差し替える**: ranking-logit priorではなく、
   observed-action頻度から直接学習したモデルを behavior として使う。

コスト最小は1、元要件に忠実なのは2または3。**この判断を経ずにT3教師の
大量生成へ進むと、生成物全体が「自己整合priorを仮定した結果」という
限定付きになる**ため、先に決めるべきである。
