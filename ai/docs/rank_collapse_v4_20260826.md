# v4厳密スイープのランク畳み込み — 実装完了記録 (2026-08-26)

指示書: Fable「実装指示書: フラッシュ死亡盤面のランク畳み込み (v4厳密スイープ)」。
正典: [`suit_collapse_20260824.md`](suit_collapse_20260824.md)。
実装: `rank_collapse.rs` (新規) + `v4_first.rs` / `t3_first_hu.rs` / `t3_second_hu.rs` / `main.rs`。

ビルドは `CARGO_TARGET_DIR=C:/tmp/rank_collapse_target` で分離。
出荷中の `ai/rust_solver/target/release/t4_first_exact.exe` は**上書きしていない**。

全測定は同一マシン・`RAYON_NUM_THREADS=8`・同一 `ai/config/fl_ev.json`
(sha256 `4676ef8547c7376f73344c7e3724cd1c0df0edd2a042ae74e6d1d4b5515b2ff8`、
fl_ev(14)=6.57)。

---

## 1. Phase 0: v4_first の配置列挙が合法手の半分を落としていた

`v4_first::placement_patterns()` は `for b in a..3`、つまり**非順序**の行ペアしか
作っていなかった。呼び出し側は全て `pool[i]→pattern[0], pool[j]→pattern[1]` (i<j固定)
なので、空き枠が別行にあるとカード対の2通りの割当のうち1通りしか生成されない。
d3cca6c「Half of every placement was missing」と同じ欠陥クラスの再発。

修正は順序つき全列挙 (`for b in 0..3` + 容量検査)。ルール正典
`ai/engine/action_space.get_turn_actions` の `for pos0 in POSITIONS { for pos1 in POSITIONS }`
と同じ形になる。

### 1.1 再現と突き合わせ

指示書§2.1のフィクスチャ (空き = mid1/bot1、両盤面とも):

| | 決定モードの手数 | 判定 |
|---|---|---|
| 修正前バイナリ | **3** | `5h@bottom,6h@middle\|8d` 等が欠落 |
| Python 正典 `get_turn_actions` | **6** | — |
| 修正後バイナリ | **6** | 正典と**集合一致** (`scratchpad/p0_python_parity.py`) |

### 1.2 試験 (d3cca6c の規律: 生成器を共有した試験は欠陥に同意する)

`v4_first.rs` の `#[cfg(test)]` に3本。いずれも**独立列挙** `independent_row_pairs`
(9通りをカウンタから復号し、行長を容量と直接比較する。`open_slots` も
`placement_patterns` も使わない) を相手にする。

| 試験 | 内容 |
|---|---|
| `placement_patterns_reach_every_legal_final_board` | 空き6形状で**到達可能な最終盤面の集合一致** + 本数を定数でピン (2/1/2/1/2/1) + 重複なし |
| `probe_fixture_has_six_actions` | §2.1フィクスチャの手数 == 6 を定数 assert、両向きのキーの存在も assert |
| `amortised_state_value_matches_brute_force` | 既存のビット一致試験。**brute側の配置列挙を独立化**して維持 |

**検出力の証明**: 生成器を一時的に `a..3` に戻すと3本とも落ちる
(完全性 1≠2、本数 3≠6、brute 4.281 対 8.121)。独立化前の brute は
生成器を共有していたので、この欠陥に同意していた。

### 1.3 ラベル影響 (ゲートではなく報告義務)

`D:/ofc_data/hu/onpol2_requests/t3_btn_lap2.jsonl` 先頭50ルートを `--t3-second-hu` で
旧/新バイナリに流した。action キー集合は一致 (t3_second_hu 自身の列挙は元から全3×3)、
変わるのは値だけ。

| 量 | 旧 | 新 | 変化 |
|---|---|---|---|
| action毎の平均 \|Δ値\| | — | — | **2.6404** |
| 同 最大 \|Δ値\| | — | — | 9.8951 |
| best値の平均 | −4.8395 | −5.3525 | **−0.5130** |
| ルート内の散らばり (max−min) 平均 | 13.1044 | 14.8826 | **+1.7782** |
| argmax が実質的に動いたルート | — | — | **28.0%** |
| 前例 d3cca6c (fl_solver、同じ欠陥クラス) | — | — | best +0.696 / 散らばり 1.42→2.11 |

best が**下がる**のは向きとして正しい: `--t3-second-hu` のラベルは
`−V4(役割入替)` なので、修正で相手 (入替後の v4 hero) の max が倍の手集合の上の max
になり、その分ヒーローの取り分が減る。散らばりが広がるのは d3cca6c と同じ向き
(正しいラベルの方が手を区別する)。

### 1.4 運用上の注意

- serve挙動 (`hu_match` の T4先手席・T3後手席) が**無条件に**変わる。
  走行中・比較中の対戦測定をまたいで投入しないこと。lap 3 の世代境界で投入し、
  バイナリ版名を刻む (例 `20260826a`)。
- lap1/lap2 ラベルと lap2 判定 (union +0.125) はこの欠陥を両アーム共通で踏んでいる。
  順位は共通モードでおそらく生存するが、絶対値は要注意。再判定はオーナー裁量。
- `v3s_probe` の過去の probe 数値とは非互換 (probe も v4 経由)。

---

## 2. Phase 1: ランク畳み込み (`--rank-collapse`、既定 off)

### 2.1 適用範囲

`v4_first::solve_maybe_collapsed` が唯一の入口。畳むのは**状態モード・一様プール・
両盤面フラッシュ死亡**の3条件が揃ったときだけ:

- `draw` あり (決定モード) → 従来経路。カード単位の action キーを畳み込みは名乗れない。
- `discard` モデルあり → 従来経路。per-card 重みはランク類が捨てるもの。
- どちらかの盤面が生存 → 従来経路。両者が同じ29枚プールを食い合うので、片方の
  スート消費がもう片方のアウツを動かす (正典§4「プール結合の罠」)。

フラグを解釈するのは `--t3-first-hu` / `--t3-second-hu` / `--v4-first` の3モードのみ。
serve系・encode系・`self_play`・`v3s_probe` はコンパイル時に旧経路で固定。

**適格判定の拡張**: 空き0の完成行は常に適格 (評価がプール非依存の定数)。正典§1の
厳密規則は完成フラッシュ行を「生存」と数えるが、誰も足さない行にその問いは無意味。
拡張の価値は発火率 +16.5pt (58.8% → 75.3%、指示書§7.1の実測)。専用フィクスチャ
(§2.3-e) でカード単位列挙と突き合わせてある。

**Phase 0 が先でなければならなかった理由**: 畳み込みは具体的なカード対を類の代表対に
置き換えるが、プール順 (スート優先) と類順 (ランク順) は一致しないので、どちらが
「先」かは保存されない。`placement_patterns` が順序つき集合になった今、カードの入替は
パターンの入替で打ち消され、パターン列は入替で閉じているので min は同じ f64 になる。
非順序のままなら畳み込みは黙って別の問いに答えていた。

### 2.2 §6.1 凍結レーンのバイト回帰

参照は**同一ソースツリー・同一コンパイラで建てた baseline バイナリ**
(出荷 exe は 8/22 ビルドで joint_outlook.rs の 8/24 変更を含まないため、ビルド差の
交絡を避けてこちらを基準にした)。

| レーン | Phase 0 適用時 | Phase 1・フラグなし | Phase 1・`--rank-collapse` |
|---|---|---|---|
| `--hu-encode` (両席×4街) | IDENTICAL | IDENTICAL | IDENTICAL |
| `--joint-outlook` samples=400 | IDENTICAL | IDENTICAL | IDENTICAL |
| `--joint-outlook` samples=0 | IDENTICAL | IDENTICAL | IDENTICAL |
| `--encode-features` 104幅 (FL14 + joint) | IDENTICAL | IDENTICAL | IDENTICAL |
| `--encode-features` 110幅 (FL14 + allocation) | IDENTICAL | IDENTICAL | IDENTICAL |
| `--encode-features` 109幅 (`t3_second::joint_block`) | IDENTICAL | IDENTICAL | IDENTICAL |

v4系レーンのフラグoff証明 (Phase 0 バイナリ 対 Phase 1 バイナリ、フラグなし):

| レーン | 判定 |
|---|---|
| `--t3-second-hu` 200ルート | IDENTICAL |
| `--v4-first` 200状態 | IDENTICAL |
| `--t3-first-hu` 200ルート | IDENTICAL |

### 2.3 §6.2 強検査 (単体試験、`rank_collapse.rs`)

小プール (10〜12枚) で、試験ローカルの**カード単位素朴列挙** (`memo.terminal` +
`compose` を直接呼ぶ三重ループ。ランク類も多重集合重みも二項係数も使わない) と
突き合わせる。強検査は**三枚組値のビット多重集合一致**:
`{V_Tのビット → ΣW_T}` == `{draw_min のビット → 個数}`。

| ケース | 試験 | 結果 |
|---|---|---|
| (a) ジョーカーなし・両盤面死亡 | `jokerless_dead_boards_collapse` | ビット多重集合一致・ΣW_T == C(n−2,3)・状態値 \|Δ\| ≤ 1e-12 |
| (b) プールにジョーカー2枚 | `pool_jokers_form_their_own_class` | 同上 + 類分割 (自然ペア3 + ジョーカー類1) を assert |
| (c) 盤上ジョーカーが bottom (fast path) | `board_joker_in_bottom_fast_path` | 同上 |
| (d) 盤上ジョーカーが middle (slow path) | `board_joker_in_middle_slow_path` | 同上 |
| (e) 完成フラッシュ行を含む盤面 (拡張規則) | `completed_flush_row_collapses` | 同上。正典規則では「生存」であることも assert |
| (f) 生存対照 | `live_board_is_refused_and_suit_dependent` | `board_eligible == false`。さらにスート付け替えで生存盤の値が**動く** (検出力) / 死亡盤は**ビット不変** |
| (g) 適格判定そのもの | `row_eligibility_matches_the_canonical_rule` | 境界12例で `suit_parity.py::row_dead` と真理値一致。唯一の意図的乖離 (完成フラッシュ) を明示 assert |
| 配線 | `decision_mode_and_live_boards_stay_on_the_old_path` | draw ありでは発火カウンタが動かない・on/off の値が 1e-9 以内 |
| プール同一性 | `collapse_pool_matches_the_solver_pool` | `unseen_pool` == `solve_weighted` のインラインプール |

`cargo test --release -p t4_first_exact`: **24 passed / 2 failed**。失敗2件
(`t1_vs_fl`、`t2_vs_fl_pool`) は**本変更と無関係の既存不整合** — ディスク上の
プールが古い fl_ev テーブル `[0.0, 10.7, 29.9, 63.5]` で焼かれており、現行
`ai/config/fl_ev.json` の `[6.57, 16.61, 38.76, 70.07]` と `TableMismatch` になる。
両ファイルとも `v4_first` / `rank_collapse` / `placement_patterns` を参照しない。

### 2.4 §6.3 実ルートゲート (200ルート×3レーン)

同一バイナリのフラグ A/B。旧参照 = Phase 0 適用済み・フラグ off。

| レーン | ルート | 最大 \|Δ値\| | argmax regret | 発火率 | off | on | 速度 |
|---|---|---|---|---|---|---|---|
| `--t3-second-hu` (T3-BTN教師) | 200 | **4.377e-12** | **0.000e+00** | **2445/3288 = 74.4%** | 40.9 ms/root | 15.3 ms/root | **2.67x** |
| `--v4-first` (T3-BTN由来の11枚状態) | 200 | 2.444e-12 | 0.000e+00 | 133/200 = 66.5% | 2.7 ms/state | 1.6 ms/state | 1.71x |
| `--t3-first-hu` (T3-BB教師、16抽選) | 200 | 1.521e-12 | 0.000e+00 | **39702/51456 = 77.2%** | 837.5 ms/root | 388.1 ms/root | **2.16x** |

- 受け入れ限界 1e-9 に対して実測は **1e-12 台**。tolerance の緩和は一切していない。
- regret 形式の argmax は全ルートで**厳密ゼロ**。index が入れ替わったのは
  `--t3-second-hu` の 1 ルートのみで、旧経路の値でも同点 (regret 0) — [[t3-tie-misses]]
  の通り EV コストはない。
- 発火率 T3-BTN **74.4%** は事前実測 75.3%±3pt の中。拡張規則の入れ忘れなら ~59% に
  落ちて見えるはずで、そうなっていない。T3-BB **77.2%** は下界 66.6% を上回る
  (下界は相手9枚盤で代用したもの。実際の `opp_after` は11枚でより死んでいる)。
- 縮小率の実測 (lap2実ルート50本): ペア C(29,2)=406 → ~87 (**4.5x**)、
  三枚組 C(27,3)=2925 → ~399 (**6.9x**)。壁時計は類の bookkeeping と opp_terms 構築の
  残部で目減りし、**教師シャード単価で 2.2〜2.7x**。指示書§7.2の暫定 2.5〜3.5x の
  下端付近。

### 2.5 決定性

`RAYON_NUM_THREADS` を 1 と 8 で振って `--t3-second-hu` 200ルートを流し、
フラグ off / on の両方で**出力バイト一致**。類順はプール初出順、候補は index 収集、
状態スイープは逐次で、数値経路に HashMap の反復順は入っていない。

### 2.6 発火カウンタ

3モードの終了時に stderr へ1行 (stdout / `--output` は汚さない):

```
rank-collapse: fired 2445/3288 (74.4%) [on]
rank-collapse: fired 0/3288 (0.0%) [off]
```

off でも `[off]` を出す。lap 3 のランブックでは T3 教師コマンドに `--rank-collapse` を
明記し、この行で付け忘れを検知すること。「厳密ゼロ」は両向きに読む —
A/B が全ルート厳密ゼロなら、それは「差なし」ではなく「フラグが配線されていない」
かもしれない。

---

## 3. 何が畳めて何が畳めていないか

- **効いた**: lap 3 の T3 教師 (BB / BTN) の v4 厳密解。シャード単価 2.2〜2.7x。
- **効かない (今回の範囲外)**: `ai/tutor/hu_street_teacher.py` の境界207次元エンコード
  (サンプル400・凍結レーン)。T1 の「30シャード×1h」級の費用はここが占めており、
  本変更では1秒も減らない。減らす道は (a) 世代境界で「厳密ブロック化 + SubsetIter
  畳み込み」をコーパスごと採る判断、(b) serve への適用 (Phase 3)。
- **Phase 3 (未着手)**: serve (`hu_match` の T4先手/T3後手席・トレース艦隊) への配線は
  小change だが、serve挙動の変更は対戦でしか判定できない。同点入れ替えで挙動が
  変わり得るので digest では判定できない。ミラー対戦 A/B (2シードセット以上・
  プール判定) で「差 ≈ 0 かつ 速度 x 倍」を示してから。**着手はオーナー承認後。**

---

## 4. 逸脱

指示書からの逸脱は1点のみ。

- §4.4 は候補ごとの列挙完全性 `Σ_T W_T == C(n−2,3)` を `debug_assert` と書いているが、
  **実装は `assert!`** にした。`cargo test --release` も本番も debug_assertions は
  無効なので、debug_assert では誰も見ない。コストは候補あたり u64 比較1回で、
  スイープ本体に対して測定不能。弱める方向ではなく強める方向の差分。

---

## 5. 材料

- 指示書とスクリプト: `scratchpad/rank_collapse_brief.md`、`suit_parity.py`、
  `fire_rate.py`、`collapse_factor.py`、`v4probe.jsonl`
- 本作業で書いた検証スクリプト: `scratchpad/p0_python_parity.py` (Python正典突き合わせ)、
  `p0_label_impact.py` (§1.3)、`frozen_lane_regression.py` (§2.2)、
  `rank_collapse_gate.py` (§2.2 v4レーン + §2.4)
- Phase 0 時点のソース退避: `C:/tmp/p0_snapshot/` (P1 の差分を独立にレビューするため)
- ビルド: `C:/tmp/rank_collapse_target/`、バイナリ退避 `C:/tmp/rc_bin/{base,p0,p1}.exe`

一般化 (生存スートのバケット化。L=0 はその特殊例) は [`live_suit_bucket_20260826.md`](live_suit_bucket_20260826.md)、binstamp 20260826b。発火条件 |L|≤1 (L=2 は畳むと0.49xで遅いため素通り)。T3教師の総合は 2.67x → **3.31x** (BTN) / 2.16x → **2.40x** (BB)。
