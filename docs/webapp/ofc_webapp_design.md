# OFC Pineapple 対戦Webアプリ 設計図 v1

作成: 2026-07-29。対象: regular-ofc-pineapple(Jokerなし)トラックの学習済みAIと人間がスマホブラウザで対戦するアプリ。

## 0. 一枚図

```
[スマホブラウザ PWA]  ←HTTPS/JSON→  [Cloud Run コンテナ]
  React + TS                          FastAPI (Python 3.11)
  盤面UI / FL配置UI                     ├─ ゲーム状態機械 + ルール審判
  記録ビューア                          ├─ AIアセンブリ (街ごとに学習モデル or 本番ポリシー)
                                       │    ├─ libofc_hu_m3_engine.so (Rust, decide/score_final)
                                       │    ├─ libofc_stage3_feature_encoder.so (SHA固定)
                                       │    └─ 本番ポリシー (ofc_regular Python, T0/T1)
                                       ├─ regular_fl_solver (Rust CLI, AIのFL配置)
                                       └─ SQLite (GCSバックアップ) 全記録 + エクスポート
```

決定事項(発注者確認済み): クラウド(Cloud Run)/ AIのFLはRust実装(regular_fl_solver)を使用 / 200点持ち・0で終了・残点まで精算 / 記録はサーバー保存+エクスポート。

## 1. ゲームルール仕様(実装の正)

### 1.1 基本
- ヘッズアップ OFC Pineapple。トップ3枚・ミドル5枚・ボトム5枚。
- ストリート: T0=5枚(全配置)、T1〜T4=各3枚(2枚配置+1枚裏向き捨て)。
- 先攻が配置を確定してから後攻が配置(後攻は先攻の公開盤面を見られる)。
- **1ハンドごとにポジション(先攻/後攻)交代**。第1ハンドの割り当てはマッチ作成時に乱数で決定し記録する。
- スコアリング: 行ごと±1、スクープ±3、ロイヤリティ、ファウル処理は**すべてエンジンの `score_final` が唯一の正**。フロントは点数計算を再実装しない。

### 1.2 Fantasy Land (FL)
- 突入条件: ファウルせずトップ QQ=14枚 / KK=15枚 / AA=16枚 / トリップス=17枚。
- FL中: 該当枚数を一度に受け取り、13枚配置+残りを裏向き捨て。相手からは配置終了の事実だけ見え、ハンド終了時に公開。
- 継続(stay): FL中にファウルせず トップ=トリップス または ボトム=クアッズ以上。
- 両者FLも発生しうる(両者とも一括配置)。
- AI側のFL配置は `regular_fl_solver` を呼ぶ(列挙+ファウル回避+ロイヤリティ+stay価値の厳密最適化)。人間側はFL専用UI。
- FL EV定数はエンジンのScoringContextと一致させる(サーバー起動時にエンジンから取得し、フロントに配信)。

### 1.3 点数とマッチ進行
- 両者200点スタート。テーブルステークス: 1ハンドの授受は「負け側の残点」を上限に打ち切り(合計は常に400)。
- どちらかが0点になったらマッチ終了(FL持ち越しがあっても終了)。
- ハンド終了時、**次ハンドでどちらもFLでない場合のみ**「続行 / 終了」を選べる。どちらかがFLに入る場合は自動続行(FL消化義務)。
- 終了時はマッチサマリー(ハンド数、点数推移、FL回数、ロイヤリティ合計等)を表示。

## 2. AIアセンブリ(サーバー内)

ベンチマークで使っている合成構成と同一。**街ごとに最良の頭脳を差し替え可能**にする:

| 街 | 現在の頭脳 | 呼び出し |
|---|---|---|
| T0, T1 | 本番ポリシー (stage19_p0 相当) | Python `build_policy(...).choose_action_observation` |
| T2 先攻 | (学習中→完成後に learned へ) 当面は本番ポリシー | 同上 → 完成後 `decide` |
| T2 後攻 | 学習モデル | Rust `decide` リクエスト |
| T3 両席 | 学習モデル | Rust `decide` |
| T4 両席 | 厳密列挙 | Rust `decide` |
| FL | regular_fl_solver | CLI サブプロセス |

- 構成は `assembly.json`(街→evaluator種別+重みSHA)で宣言し、**AIの全決定に構成SHAを記録**する。新モデル完成時はこのファイルとweightsの差し替えだけで更新。
- 重みはT4M1形式 `.bin` + SHA-256固定。SHAが合わなければ起動拒否(既存エンジンの流儀)。

### レイテンシ見込み
decide(学習モデル): T4 ~3ms / T3 ~10ms / T2後攻 ~20ms。本番ポリシー(T0/T1): 数百ms。FLソルバー: 数百ms〜数秒。→ ユーザー体感は全て1手2秒以内。Cloud Runコールドスタート(コンテナ~1GB)は数秒〜十数秒: min-instances=0で運用し、フロントに「AI起動中」表示を用意。

## 3. API契約(FastAPI, すべてJSON)

認証: 全エンドポイントに `Authorization: Bearer <SHARED_TOKEN>`(環境変数)。個人利用前提の単一トークン。

```
POST /api/match                 … マッチ作成 {seed?} → {match_id, first_hand_positions}
GET  /api/match/{id}            … マッチ状態(点数、ハンド一覧、継続可否)
POST /api/match/{id}/hand       … 次ハンド開始 → {hand_id, positions, fl_status, 人間の初手配札}
GET  /api/hand/{id}             … ハンド状態(自分の盤面/手札、相手公開盤面、手番、街)
POST /api/hand/{id}/action      … 人間の配置 {placements:[[card,row]..], discards:[card..]}
                                   → サーバーが合法性検証(エンジンの合法手生成と照合)後に確定、
                                     続けてAIの応答が必要なら同期的にAI手番を進めて新状態を返す
POST /api/hand/{id}/fl-placement … 人間のFL一括配置(13枚+捨て)
POST /api/match/{id}/continue   … {continue: true|false}(両者非FL時のみ有効)
GET  /api/match/{id}/export     … マッチ全記録 JSONL ダウンロード
GET  /api/meta                  … assembly構成・重みSHA・ルール定数(FL EV等)・アプリ版
```

- 手番進行はサーバー主導の状態機械。**山札・配札はサーバーのみが知る**(シード付きRNG、デッキ順は記録するがAPIには出さない)。
- AIの思考が長い場合(FLソルバー)に備え、`/action` 応答に `ai_pending: true` + `GET /api/hand/{id}` ポーリング(1s)のフォールバックを用意。WebSocket/SSEは第1版では使わない。

## 4. データモデル(SQLite)

```
matches(id, created_at, seed, human_stack, ai_stack, status, first_positions,
        assembly_sha, app_version)
hands(id, match_id, index, positions(human=first|second), fl_human, fl_ai,
      deck_order_json, started_at, ended_at,
      result_json{row_wins, scoop, royalties, fouls, fl_entries, raw_score,
                  capped_score, stacks_after})
decisions(id, hand_id, actor(human|ai), street(T0..T4|FL), seat, dealt_json,
          placements_json, discards_json, think_ms, created_at,
          ai_meta_json{evaluator, weights_sha, scores_topk} | null)
```

- 「全て記録する」= 配札・全配置・捨て札・所要時間・AIの評価値上位k・スコア内訳・デッキ順・シードまで。後でそのまま学習/分析データにできる形(JSONLエクスポートは1行=1決定 + ハンドサマリー行)。
- SQLiteファイルは定期的(ハンド終了ごと)にGCSへコピー(Cloud Runのディスクは揮発のため)。起動時にGCSから復元。

## 5. フロントエンド

- React + TypeScript + Vite + Tailwind。PWA(ホーム画面追加、縦持ち専用)。同一Cloud Runから静的配信。
- 画面: ①マッチロビー(新規/再開/履歴) ②対局テーブル(相手盤面上・自盤面下・手札中央、カードはタップ→行タップで配置、取り消し可、確定ボタンで送信) ③FL配置(14〜17枚グリッド+3行スロット、ドラッグ or タップ) ④ハンド結果(行ごとの勝敗・ロイヤリティ内訳・点数移動アニメ・続行/終了ダイアログ) ⑤マッチサマリー ⑥履歴ビューア(ハンド再生: 記録から一手ずつ再現) ⑦エクスポート。
- 盤面状態は常にサーバーが正。フロントは操作前の軽い合法性チェック(行の空き数)のみ行い、確定はサーバー検証。

## 6. コンテナ / デプロイ

- ベース: debian-12-slim + Python 3.11。既存の艦隊ランタイム(runtime.tar.gz: エンジン.so、エンコーダ.so、weights 4本、ofc_regular一式)と同じ部材を利用し、`regular_fl_solver` バイナリ(linux向けに `cargo build --release`)を追加。
- Dockerfile はマルチステージ(rust builder → python runtime)。イメージにモデル重みを焼き込み、SHAを起動時検証。
- Cloud Run: region=asia-northeast1, 1 vCPU / 1GB, min-instances=0, max=1, concurrency=4。想定費用: 個人利用なら月数十円〜数百円。
- Secrets: SHARED_TOKEN は Secret Manager。GCSバケットは既存の学習用とは**別バケット**(記録用)を作る。

## 7. 実装マイルストーン(発注先AI向け)

1. **M-A**: ルール状態機械+エンジン統合+CLIでの1ハンド通し(FLなし)。エンジンの`score_final`と自前進行の整合をゴールデンテストで固定
2. **M-B**: REST API + SQLite記録 + 通常ハンドのフロント(スマホ実機で1マッチ)
3. **M-C**: FL(人間UI + regular_fl_solver統合 + stay/両者FL)
4. **M-D**: 点数精算・続行/終了フロー・マッチサマリー・履歴再生・エクスポート
5. **M-E**: Cloud Runデプロイ + GCS永続化 + PWA仕上げ

## 8. 未決事項(壁打ち継続用)

- 第1ハンドの先攻を乱数でなく人間選択にするか(現設計: 乱数+記録)
- 人間側の持ち時間/タイマー表示(現設計: なし、think_msの記録のみ)
- FL中の相手への進捗表示(現設計: 「配置完了」のみ。カード枚数も隠すか?)
- アンドゥ範囲(現設計: 確定送信前のみ可、送信後不可)
- AIの評価値を対局中に見せるか(現設計: 対局中は非表示、履歴再生では表示可) ← 練習用途なら表示トグルも
- T2先攻/T1/T0の学習モデル完成時の差し替え手順の自動化程度
```
