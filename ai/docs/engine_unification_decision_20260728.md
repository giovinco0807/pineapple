# Decision Record: serving/学習エンジンの一本化 (2026-07-28)

Status: **accepted — B-phased**（2026-07-28 オーナー承認）
Decision owner: プロジェクトオーナー
Deadline: M4（T2 continuation）着手前
関連: `roadmap_improvement_proposal_20260728.md` 提案5

採用内容: B-phased。M3はjokerトラックで完走し、並行してregular側engineへ
jokerルールを移植する。移植の受け入れgate 1〜4がすべて通るまでM4のteacher生成を
unified engineで開始しない。gate失敗時はoption Aへfallbackする。

## 決定すべきこと

ジョーカーあり54枚HU AIのM4以降（continuation教師の大量生成、self-play、
最終serving）を、どのエンジン基盤で実行するか。現在は2トラックで
T4 exact、CFR、検証gate、Rust engineを重複実装しており、未決定のまま
M4〜M7を進めるとどちらかが捨て作業になる。

## 現状の実測インベントリ

### jokerトラック (`ai/`)

| 資産 | 実測 | 備考 |
|---|---|---|
| Python engine | 1,585行 (`game_engine` 605, `encoding` 379, `scoring` 242, `action_space` 238, `turn_order` 121) | canonical Joker制約置換評価。X1/X2ネイティブ |
| Rust crates | `ofc_core` 868行（joker対応Card型）＋ prob_engine / t3・t4 exact solver / t3・t4 generator / cfr_solver / backward / fl_solver | 6crate以上でX1/X2対応済み |
| M3 solver基盤 | multi-root MCCFR+、3313次元InfoSet encoder、teacher bundle v2、T4両席exact resolver | すべてjokerネイティブ、content-bound、進行中 |
| RL/batch engine | **なし** | packed batch、snapshot、seat両対応RL環境は未実装 |
| serving | 旧backend（`backend/ai_player.py`、legacy MCTS+VN系） | 近似均衡路線とは別物 |

### regularトラック (`regular-ofc-pineapple/`)

| 資産 | 実測 | 備考 |
|---|---|---|
| Python engine | 425行 (`cards` 42, `evaluator` 199, `rules` 96, `action_space` 88) | `RuleSet` dataclassでルールをパラメータ化済み（`include_jokers`フラグが既にfl_ev/solver configまで配線済み） |
| Rust `hu_rl_engine` | 2,992行 | scalar/batch同値、packed PyO3境界、snapshot lineage、hidden-discard redaction、10,000局Python/Rust parity（provenance v3）、512 lanesで約8,149 decisions/s |
| Rust `hu_m3_engine` | 約8,400行（`compact_scoring` 607, `scoring` 509, `cards` 199含む） | T4両席exact（first-seat p95 42.91ms）、belief、search、DLL SHA固定 |
| 検証文化 | 2,379テストregression、byte-exact parity、receipt/provenance | 情報漏洩監査（M1）済みの情報境界 |
| Joker対応 | **なし** | `CARD_TOKENS: [&str; 52]`、`ALL_CARDS: [Card; 52]`と52枚がindex levelで固定。evaluatorにjoker置換なし |

## 選択肢

### A: jokerトラック続行（現行スタックでM4〜M7を完走）

- 利点: 移植コストゼロ。M3の進行を止めない。全gate定義がそのまま使える。
- 欠点: M4以降のボトルネックである「continuation教師の大量playout」と
  「population self-play」に必要なRL/batch engineをjokerトラックへ新規実装する
  ことになり、regular側`hu_rl_engine`（2,992行＋parity harness）の再発明になる。
  serving契約（CPU推論時間gate）も旧backendから作り直しが必要。二重実装が恒久化する。

### B: 即時一本化（regular engineへjokerルールを移植してからM3を再開）

- 利点: 以後の実装がすべて片側に集約。
- 欠点: 進行中のM3a/M3b（content-boundなcalibration収集、teacher契約、72job smoke）
  が中断し、hash-bound artifactの再生成と再検証が必要になる。移植完了までjoker
  トラックの前進が止まる。リスクが前倒しで集中する。

### C: 逆方向移植（regular側のRL engineをjokerトラックへ移す）

- 欠点: `hu_rl_engine`のparity harness、receipt/provenance、情報境界監査は
  regularリポジトリの構造に結び付いており、切り出しは事実上の書き直しになる。
  regular側Python engineの方がルールパラメータ化が進んでいる（`RuleSet`）ため、
  ルールを動かすより計算基盤へルールを足す方が安い。不採用。

### B-phased（推奨）: M3はjokerトラックで完走、M4以降をunified engineへ

1. **今〜M3b完了**: jokerトラックは現行計画のまま進める（natural評価と
   locked calibration → teacher大量生成 → distillation → T3/T4 promotion gate）。
   移植作業はこれと並行して行い、M3を止めない。
2. **並行作業（M4着手前に完了）**: regular engineへjokerルールを移植する。
   - Python: `RuleSet`に`joker_chain`ルール（`include_jokers=True`、
     FL entry `{qq:14, kk:15, aa:16, trips:17}`、chain EV）を追加。
     `evaluator.py`へbottom-up制約置換評価を追加（参照実装は
     `ai/engine/game_engine.py`の`evaluate_board_with_joker_constraint`）。
   - Rust: `cards.rs`の52固定domain→54（`X1`/`X2`トークン、index拡張）、
     `scoring.rs`/`compact_scoring.rs`へjoker置換（参照は`ai/rust_solver/ofc_core`）、
     `seeded_deck.rs`と観測エンコーディングの幅、FL chain EV設定。
     触る実測量は概算1,700〜2,500行。
3. **M4以降**: continuation教師生成、self-play、servingはunified engineで実行。
   `ai/engine`のPython canonical評価器は削除せず、parity oracleとして恒久保持する。

## 移植の受け入れgate（B-phased採用時の必須条件）

1. canonical Jokerルール（bottom-up制約置換、base-15 tie-break）の
   Python(`ai/engine`) vs 移植先Python vs 移植先Rustの3-way完全一致。
   golden fixtureは`ai/reports/joker_rule_migration_20260711/`の20,000局面
   269,745候補と、M2の6層（BB/BTN × Joker 0/1/2）golden vectorを再利用する。
2. wheel straight（A2345=5-high）、X1/X2同時保持、FL entry/stay/chain EVの
   既知エッジケースを明示テスト化（過去の実バグに対応）。
3. 54枚化後の`hu_rl_engine` 10,000局scalar/batch parityをprovenance v3様式で再実行。
4. 既存52枚regularルールの全テストが無変更で通過（ルール追加が既存を壊さない）。
5. 移植完了までjokerトラックの正典はai/のまま。gate 1〜4がすべて通るまで
   M4のteacher生成をunified engineで開始しない（fail時はoption Aへfallback）。

## コストとリスクの見積

- 移植: 1〜2週（Rust joker置換が主工数。compact bitmask scoringへの置換組み込みが
  最難部で、golden vector駆動で進める）。timebox 2週、超過時はAへfallback。
- 回収: M4〜M7で必要な「高速full-hand playout」「population self-play」「serving
  runtime gate」の新規実装（jokerトラック単独なら概算2〜4週相当）が不要になり、
  二重メンテが恒久的に消える。
- 主リスク: joker置換のRust実装バグ。緩和はgate 1〜2のgolden 3-way parity。
- 可逆性: gate通過までjokerトラックを一切変更しないため、fallbackコストはゼロ
  （移植作業の破棄のみ）。

## 推奨

**B-phased を採用する。** 理由: (1) M3を止めずにリスクを並行化できる、
(2) 回収がM4開始時点から効く、(3) fallbackが無償、(4) regular側は
`RuleSet`パラメータ化と`include_jokers`配線が既にあり、設計上この拡張を
想定している。

## 採用手続き

オーナーが本recordのStatusを`accepted`（選択肢を明記）へ更新した時点で有効。
採用時は`all_turn_ai_goal_and_milestones.md`のM4前提へ本recordを参照として追記し、
移植作業はjokerトラックの進行と独立したブランチ/ディレクトリで行う。
