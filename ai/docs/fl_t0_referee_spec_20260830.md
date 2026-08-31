# 対FL T0審判 (--fl-t0-deep) 仕様 (2026-08-30)

目的: 相手がFantasylandのときのヒーローT0配置(本番= own_lap4/t0.bin の全232 argmax)を、
実プレイ深掘りで監査する。T0-BB通常戦で実証済みの「採掘→序列矯正」の輪の対FL版。

## 新Rustモード `--fl-t0-deep` (t4_first_exact)

入力フラグ:
- `--t0-cards "Ah,7c,X1,.."` ヒーローの5枚 (X1/X2正規化は t0_deep_eval と同じ)
- `--fl-opp-width N` 相手FLの枚数 (既定14)
- `--rollouts R` / `--self-play-seed S` / `--output`
- `--t0-keys FILE` 候補サブセット (無指定=232全部; キー規律は hu_match.rs:1314-1343 と同一)
- `--arm-a-own` own_lap4 3枚 / `--fl-ev-config`

1ロールアウト r の手順 (CRN: 同じrは全候補で同一の未来):
1. デッキからヒーロー5枚を除き、`deal_names(seed, r, 12 + width)` 相当で
   ヒーローの残り引き12枚と相手FLの width 枚を非交差に配る (hu_match.rs:1361-1380 の型)。
2. ヒーロー: 強制T0候補を置き、T1/T2は own_lap4 チューザ、T3/T4は現行の
   own-worth greedy (self_play.rs:242-299) — **本番の連鎖そのまま**を再生する
   (審判は本番を監査するので、賢い連鎖に差し替えないこと)。
   実装は play_roots::play_trace のT1/T2ループを「与えた5枚盤面から開始」で再利用
   (強制T0フックの追加が必要, play_roots.rs:223-310)。
3. 相手: `self_play::play_fl(opp_cards, width, table, Some(hero_finished))`
   (frontier best response, self_play.rs:371-400)。
4. 採点: `score_r = settle(hero, opp) + fl_ev[hero.entry_width] − (opp_stays ? fl_ev[14] : 0)`
   (fl_evはchainテーブル。相手stayの負担も候補依存なので入れる)。

出力: T0DeepRow と同じ形 {key, mean, scores[]} (hu_match.rs:736-755 再利用)。

`--rollouts 0`: own_lap4/t0.bin の全232ランキングを出力 (key, score, own_rank)。
本番に柵は無いので ranker/policy 列は不要。

## ドライバ `ai/tutor/fl_mine_local.py` (t0_mine_local.py を改造)

- ルート源: `D:/ofc_data/fl14_t0_requests_v1.jsonl` (正準1,000クラス) を
  `.multiplicity.json` の多重度降順 = 頻度順に処理。
- 配信pick: rollouts-0 の own_rank 1位。
- 候補場: own_rank 上位16 + 上段枚数ストラタ (t0_mine.py:137-149 と同型)。
- 選抜: 修正済み successive halving (means は totals 全候補から! t0_mine.py の修正版準拠)、
  シード帯 **850M + i*10**。
- 採点: 不一致のみ 4バッチ × `--score-rollouts 150`、シード帯 **860M + i*10 + j**。
  verdict/margin/ci は t0_mine.py:151-165 と同一規約。
- 出力: `D:/ofc_data/hu/fl_mine1/material.jsonl` (idはリクエストid t0c-NNNNN)。

## 検収 (Opusが実施)

1. cargo build (workspace 121テスト緑のまま)。ビルドは `--target-dir target_gate`
   (target/ は昇格再審が掴んでいる)。
2. スモーク: 1ルート×候補3×rollouts 8 で scores[] が候補間で同一長・CRN対
   (同一rの相手14枚が候補間で一致することをログで確認)。
3. `--rollouts 0` の own_rank 1位が、--fl-sim 系の play_trace が選ぶT0と一致 (3ルート)。
4. 速度実測 (秒/ロールアウト) を報告。

## 触ってはいけないもの

- ai/tutor/train_t0_eval_correction.py, train_t0_policy.py, t0_escalate.py (別作業が進行中)
- material.jsonl (t0_mine1) には書かない。fl_mine1 は新ディレクトリ。
- 昇格再審プロセスが D:/ofc_data/hu/t0_mine1/material.jsonl を随時書き換え中。
