# OFC Pineapple Tutor 強化プラン

## 方針

全分岐を事前計算して配るのではなく、各局面評価モデルを強くする。サービスでは高速モデル評価、必要時だけ高精度計算、キャッシュ、終盤の正確計算を組み合わせる。

## 教師データ

- T0/T1/T2 は高精度シミュレーションで教師ラベルを増やす。
- T3/T4 は可能な限りモンテカルロではなく全列挙で `EV / FL / AA / KK / QQ / trips / bust` を正確化する。
- 保存データには `board`, `dealt`, `legal_actions`, `chosen_action`, `EV`, `FL率`, `FL種別`, `bust率`, `sims`, `exact/estimated` を持たせる。
- 追加生成の初期目標は 1,000 ハンド相当。評価ズレが大きい局面を優先して増やす。

## モデル強化

- 候補生成モデルで合法配置を上位 10-20 候補へ絞る。
- 評価モデルで各候補の `EV / FL / bust / AA / KK / QQ / trips` を即時推定する。
- 主要指標は「最善手が top-N に入る率」と「EV/FL率/bust率の予測誤差」。
- T0/T1 はランキング精度重視、T3/T4 は正確計算で補正する。

## サービス設計

- UI は全分岐事前表示ではなく、ユーザーが見たい局面をオンデマンド評価する。
- 同じ局面はキャッシュし、再計算しない。
- 無料版は代表局面、低精度評価、回数制限。
- 有料版は高精度評価、ルートレビュー、保存済み解析、終盤正確値を提供する。

## 既存データの扱い

- `ai/data/tutor_route10_20260522/tutor_route10_review.json` は UI サンプルとして使う。
- `ai/data/tutor_route10_20260522/tutor_route10_merged.jsonl` は教師データ形式の確認用に使う。
- 表示上は `300 sims` と `96 sims`、および `estimated` であることを明記する。
- T3/T4 の FL 率は再計算し、可能なら正確値に置き換える。

## 実装メモ

- 終盤全列挙とオンデマンド評価 API は `ai/tutor/exact_late.py` と `t0_tutor_app.py` に実装する。
- 教師データ正規化は `ai/tutor/normalize_teacher_dataset.py` を使う。
- Route10 レビュー JSON の終盤正確値付与は `ai/tutor/enrich_route10_exact.py` を使う。
