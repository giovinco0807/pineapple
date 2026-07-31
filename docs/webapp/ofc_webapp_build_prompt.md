# 発注プロンプト: OFC Pineapple 対戦Webアプリの実装

以下をそのまま実装担当AIに渡してください。設計図(ofc_webapp_design.md)を同梱すること。

---

あなたにはOFC Pineapple(Open Face Chinese Poker Pineapple、ヘッズアップ、Jokerなし)の対人戦Webアプリを実装してもらいます。同梱の設計図 `ofc_webapp_design.md` が仕様の正です。ここではあなたの作業範囲・touch禁止領域・受け入れ基準を定めます。

## リポジトリと既存部材(あなたは**利用するだけ**で、変更禁止)

リポジトリ: `regular-ofc-pineapple/`

1. **ゲームエンジン** `target/release/libofc_hu_m3_engine.so`(Rust製cdylib)
   - Pythonラッパー: `src/ofc_regular/hu_m3_rust.py` の `load_native_engine` / `evaluate_request`
   - 使うリクエストは2種:
     - `kind: "decide"` — 観測(ActorObservation.to_dict())+configを渡すと、その街のAIの手(placements/discards)と評価メタを返す。configの組み立ては `src/ofc_regular/hu_m3_rust.py` の `_joint_config_payload` と、参考実装 `scratchpad/integration_benchmark.py` の `decide_config` を踏襲(学習モデルのpath+sha256のペア。sha無しはエンジンが拒否する)
     - `kind: "score_final"` — 13枚完成盤面2つ+scoring を渡すとハンドスコア(hu_score)を返す。**点数計算は必ずこれを使う。自前実装禁止**
2. **本番ポリシー(T0/T1用)**: `src/ofc_regular/ai_profiles.py` の `ModelPaths` / `load_model_bundle` / `build_policy`(プロファイル `stage19_p0`)。特徴量エンコーダ `.so` はSHA固定でロード(`pinned_feature_encoder_library`、参考: integration_benchmark.py)
3. **FLソルバー** `rust/regular_fl_solver`(Rust CLI)— AIのFantasy Land一括配置に使う。`cargo build --release` してコンテナに同梱。入出力形式はソース(`src/main.rs`)を読んで確認し、薄いPythonアダプタを書くこと
4. **合法手生成**: `src/ofc_regular/action_space.py` の `generate_turn_actions` — 人間の手の合法性検証に使う
5. **観測の組み立て**: `src/ofc_regular/hu_infoset.py` の `ActorObservation`(参考: integration_benchmark.py の `observation_for`)
6. **モデル重み** 4本(`t4_model_v5.bin` / `t3_model_v2.bin` / `t3first_model_v1.bin` / `t2_model_v1.bin`)+各SHA。設計図§2のassembly構成で使用

## あなたが作るもの

- **バックエンド**: FastAPI(Python 3.11)。設計図§1のルール状態機械、§3のAPI、§4のSQLite記録+GCS永続化、§2のAIアセンブリ(街→頭脳のディスパッチ)
- **フロントエンド**: React+TS+Vite+Tailwind、スマホ縦持ちPWA。設計図§5の7画面
- **コンテナ/デプロイ**: マルチステージDockerfile、Cloud Run(asia-northeast1)デプロイ手順、Secret Manager経由のBearerトークン認証

## 絶対条件

1. 点数・ファウル・ロイヤリティ判定は `score_final` が唯一の正。フロントの表示用内訳もサーバー経由で取得
2. 人間の手はサーバー側で `generate_turn_actions` と照合して検証。不正な手は400で拒否
3. 山札はサーバーのみが保持。APIレスポンスに未公開カード・デッキ順・AIの手札を絶対に含めない(FL中の相手手札も)
4. AIの全決定に evaluator種別・重みSHA・思考時間・評価値上位3手 を記録(設計図§4 decisions.ai_meta_json)
5. ルール: 1ハンドごとに先攻後攻交代 / 両者200点・授受は負け側残点で打ち切り・0で強制マッチ終了 / ハンド終了時に次ハンド両者非FLの場合のみ「続行/終了」選択、FLが絡むなら自動続行
6. FL: 突入 QQ/KK/AA/トリップス=14/15/16/17枚、stay=トップトリップス or ボトムクアッズ以上。人間FLは専用UI(13枚配置+残り捨て)、AIのFLは regular_fl_solver
7. 全記録のJSONLエクスポート(1行=1決定+ハンドサマリー行)を `GET /api/match/{id}/export` で提供
8. 既存リポジトリのファイルは一切変更しない。新規コードは `webapp/` ディレクトリ以下に隔離。gitコミットはしない

## 受け入れ基準(この順で自己検証して報告)

1. ゴールデンテスト: シード固定で1マッチ(≥5ハンド、FL突入を1回以上含むシードを選ぶ)をヘッドレスで自動プレイ(人間側もランダム合法手)し、全ハンドのスコアが `score_final` と一致、点数遷移が精算ルールどおり、記録行数が決定数と一致
2. 合法性: 不正配置(行あふれ・重複カード・捨て札数違反)がすべて400になるテスト
3. FL: 人間FL配置(14/15/16/17枚各1ケース)がUI→API→記録まで通る。AI FLがソルバー出力と一致
4. スマホ実機(またはdevtoolsモバイルエミュレーション幅375px)で1マッチ完走のスクリーンショット
5. Cloud Runにデプロイし、コールドスタート込みで動作。`/api/meta` がassembly構成とSHAを返す
6. エクスポートJSONLをダウンロードし、1ハンドを完全再現(履歴再生画面)できる

## 進め方

設計図§7のマイルストーンM-A→M-Eの順。各マイルストーン完了時に動くもの+テスト結果を提示。不明点は推測で埋めず、選択肢と推奨を添えて質問すること。特にFLソルバーのCLI入出力形式と、既存Pythonポリシーの呼び出しシグネチャは、実物のソースを読んでから着手すること。
