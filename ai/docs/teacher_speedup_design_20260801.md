# 教師データ生成の高速化設計 (2026-08-01)

対象: `t3_vs_fl_lib.rs`(ライブラリ直接採点)と、その後継となる T2/T1/T0-vs-FL
のプレイアウトラベリング。レギュラー側が同種の最適化(RowMemo 10.7x、
FreeOutlookCache 2.3-3.5x、GCPフリート)の実装を開始しており、本設計は
その二本立て構成に揃える。

## 現状のコスト構造(実測ベース)

1 root ≈ 45行動 × 300ドロー × 20配置パターン × 219マッチ盤面の採点。
コストモデルで **score_mean の内側ループが 96.8%**。スレッド当たり実効
~160M ops/s しか出ておらず、これは 120k 件幅の library 配列(masks/values/
royalty/stay/busted が別 Vec)へ `matched: Vec<u32>` 経由で gather している
キャッシュミス支配の数字。演算自体は軽い(比較十数回)。

## Track A: 出力保存のローカル最適化(推定 3〜5x)

すべて「同一ドロー・同一マッチ集合・浮動小数の総和順序も固定」で
**出力バイト一致をパリティゲートで確認**してから採用する
(レギュラーの output-preserving 規律と同じ)。

### A1. ドロー単位の SoA スクラッチ(本命)
マッチ集合はドローごとに固定でパターン間共通。現在はパターンごと
(20回)に 120k 幅配列を gather し直している。ドロー冒頭で一度だけ
マッチ行を連続バッファ(values[3]·royalty·stay·busted を1構造体に
パック、16B/行)へコピーし、パターンループは連続スキャンにする。
219行 × 16B ≈ 3.5KB = L1 常駐。期待効果: 2〜3x。

### A2. ドロー単位の端末メモ
20パターンが生む hero 端末(values, royalty, busted, fl_count)は重複が
多い。ドロー内で端末をキーに score_mean をメモ化し、重複端末の再走査を
消す。期待効果: 1.3〜1.6x(重複率の実測が先)。

### A3. マッチ集合の busted/alive 事前分割
fl_busted 盤面のスコアは hero 端末だけの定数(6+royalty)。ドロー冒頭で
alive 群だけを A1 バッファに置き、busted 群は件数と royalty 和の2値に
畳む。stay 項も alive 群の定数和として先に畳める。期待効果: 1.1〜1.2x。

### A4. shortlist の root 単位ホイスト
seen 集合は行動に依存しない(board+draw+dead のみ)ため、shortlist を
root で1回計算し45行動で共有。全体の~0.4%なので効果は小さいが無料。

順序: A4 → A1 → A3 → A2(それぞれ独立にパリティ確認)。
20-root の計測ハーネスを先に用意し、各段の実測を記録する。

## Track B: レギュラーの GCP フリートの移植(壁時計 20〜40x)

レギュラー側で実証済みのフリート
(SPOT・create-only GCS・content-bindings・resume manifest、
実績 2,000 VM時間 ≈ $100-130、PREEMPTIBLE_CPUS 上限468)をそのまま使う。

移植に必要な条件は今日の checkpoint 改修で既に満たしている:
- root は seed+offset で決定的 → `--roots/--seed` の区間分割だけで分散可
- chunk_XXXXXXX.npz が再開単位 → シャード=chunk 区間、`already_done`
  述語にそのまま乗る
- 成果物はチャンク npz の集合 → 受信後ローカルで既存の merge を実行

配布物: t4_first_exact バイナリ + FL ライブラリ(120k 行 JSONL, ~40MB) +
python ドライバ + fl_ev.json。n2-highcpu-32 × 14台なら 30k root を
約40分 / ~$5-10。

## 適用先と優先順位

| 対象 | 規模感 | 打ち手 |
|---|---|---|
| T3-vs-FL 再生成(regretゲート失敗時のみ) | ~9h → A で ~2-3h | Track A |
| T2/T1/T0-vs-FL プレイアウトラベリング | T3 の 10〜50x | **A+B 必須** |
| M-D 通常側 T4/T3 再生成(FL EV v1 反映後) | 150k root 級 | Track B |

推奨: Track A を先に実装(数時間、全後続ランに効く)。Track B は
T2-vs-FL 教師の設計と同時に着手し、最初の本番を小シャードのリハーサル
から始める(レギュラーの運用手順に従う)。
