# T4先攻 厳密解 Rust化とparity検証 (2026-07-29)

## 背景

`ai/docs/roadmap_improvement_proposal_20260728.md`の順序修正に基づく作業。
T4先攻はT3探索が到達するたびに支払うノードであり、regular側が
`search.rs`で「先攻のリーフだけがコストを持つ。後攻は解決すべき不確実性が
残っていないので閉形式のまま」と記録しているとおり、学習評価器を入れるべき
最初の対象である。本クレートはその**教師ラベル生成器**にあたる。

## レガシークレートを使わない理由（既存判断の再確認）

`ai/reports/late_hu_status_20260712/README.md`が2026-07-12時点で既に
「レガシー`t4_exact_solver`は使用しない。スキーマ、Joker識別、52枚
ジェネレータ、**FL値の扱い**が正典ランタイムと異なるため」と記録している。

実測でも確認した。レガシー実装はFL進入時の価値に**カード枚数をそのまま**
使う（`cards as i32`）。

| FL種別 | 枚数 | 正典EV | レガシー | 影響 |
|---|---|---:|---:|---|
| QQ | 14 | 0.0 | +14.0 | 無価値なものを追う |
| KK | 15 | 10.7 | +15.0 | 過大 |
| AA | 16 | 29.9 | +16.0 | 過小 |
| トリップス | 17 | 63.5 | +17.0 | 約1/4に過小 |

さらにスコアが`i32`のため10.7/29.9/63.5を構造的に表現できない。
新クレートはFL EVを`ai/config/fl_ev.json`から読み、そのSHA-256を全出力へ
埋め込み、スコアを`f64`で扱う。レガシークレートには手を触れていない。

## 新クレート

`ai/rust_solver/t4_first_exact/`

- 意味論の基準は`ai/tutor/t4_bb_exact_resolver.py`（parity oracle）。
- 盤面評価は`ofc_core::evaluate_board_with_joker_constraint`（Python/Rust
  parity検証済みのボトムアップ制約）。
- 全合法手 × 全`C(n,3)`相手ドロー × 相手exact best response。
- 出力は全合法手のEV（`best`のみを返さない）。

## parity結果

`python -m ai.tutor.t4_first_exact_parity --roots 12`

| 指標 | 結果 |
|---|---|
| root数 | 12（Joker 0/1/2枚 = 4/6/2） |
| 合法手数 | 3〜6 |
| action key集合の不一致 | **0** |
| EV最大絶対差 | **0.0**（近似一致ではなく厳密一致） |
| Python所要 | 102.14秒 |
| Rust所要 | 3.69秒 |
| 高速化 | **27.7倍** |

単一fixture（`tests/test_t4_bb_exact_resolver.py`のBB/BTN盤面）でも
Python `-1.1831` に対しRust `-1.183077` で一致し、deck 26枚・
enumerated 2,600ドローも一致した。

## 次工程への含意

regular側は学習評価器の採否基準を「置き換える厳密解より速いこと」に置き、
バッチサイズ1で厳密解 約1,100μs に対しブースティング 9,006μs だったため
dense netを採用した。本クレートの実測は12 root 3.69秒 ≒ **1 rootあたり
約308ms**（rayonは1 root内の合法手方向のみ並列、root方向は逐次）。
T4先攻学習評価器はこの水準を超える必要がある。root方向の並列化と
相手応答のインクリメンタル評価にまだ余地がある。

## 非claim

本クレートは宣言uniform exchangeable restart belief下のexactであり、
behavior条件付きBayes posterior、Nash証明、未見局面への一般化のいずれも
主張しない。serving経路は変更していない。
