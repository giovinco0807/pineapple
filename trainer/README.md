# OFC トレーナー（レギュラー）

https://ofc.olegsolvers.com/ を参考にした、レギュラーOFC Pineapple（ヘッズアップ・Jokerなし）のローカル学習用Webアプリ。

## 機能

- **トレーニング**: AI相手に1ハンドずつプレイ。各判断を全候補のEVランキングで採点。
  - 最善手なら自動続行 / 最善手を強制 / 候補1つの時はスキップ
  - しきい値以上のEVロスは自動でミス記録
- **エディタ**: 任意の局面（自分・相手の盤面、配札、死札）を作って候補EVランキングを表示
- **ミス復習**: 記録されたミスを同じ局面でリトライ
- **アカウント**: ヘッダーのセレクタで履歴を切り替え。ハンド履歴・ミス・統計はすべてアカウント単位。

## アカウント

ローカル専用ツールなのでログインではなく「名前付きの履歴バケツ」。

- ヘッダーの `＋` で作成、セレクタで切り替え、`🗑` でそのアカウントの履歴を全削除
- 選択中のIDはブラウザの localStorage に保存され、全APIに `X-Account-Id` ヘッダーで乗る
- ヘッダーが無い／未知のIDの場合は `default` アカウントにフォールバック（認証ではなく識別）
- アカウント導入前のDBは起動時に自動マイグレーションされ、既存の行は `default` に紐づく
- ハンドは**開始時**のアカウントに記録される（途中で切り替えても記録先は変わらない）
- `default` アカウントは削除不可

## 起動

```powershell
cd regular-ofc-pineapple
python -m uvicorn trainer.app:app --host 127.0.0.1 --port 8093
```

ブラウザで http://127.0.0.1:8093 を開く。

## 構成

- `app.py` — FastAPI（認証なし・ローカル用）
- `game.py` — トレーニングセッション（配札、AI対戦相手、バックグラウンドEV解析、採点）
- `evaluator.py` — ストリート別評価器（候補列挙 + EVランキング）
- `fl_ev.py` — FL EV定数を `configs/` から読む（ハードコード禁止）
- `store.py` — SQLite（アカウント・ミス・ハンド履歴、`trainer/data/trainer.sqlite3`）
  - `GET /api/accounts` / `POST /api/accounts` / `DELETE /api/accounts/{id}`
  - `GET /api/history?limit=N` — アカウントのハンド履歴
- `static/` — フロントエンド（素のJS・日本語UI）

## 評価の仕組み

各ストリートの全合法手を列挙し、EVで降順ランキング。ユーザーの手の順位とベストとのEV差で採点する。

| ストリート | 評価器 | 速度目安 |
|-----------|--------|---------|
| T0 | 純Python MC（共通乱数・全232候補） | fast 2秒 / standard 6.5秒 |
| T1（後攻） | m3エンジン `evaluate_t1`（学習済み重み） | 2.5秒 |
| T1（先行） | エンジン非対応 → MCフォールバック | 0.3秒 |
| T2 | m3エンジン `evaluate_t2` | 0.9〜2秒 |
| T3 | m3エンジン `evaluate_t3` | 0.1〜0.3秒 |
| T4 | m3エンジン exact enumeration | 即時 |

- エンジン: `target/release/ofc_hu_m3_engine.dll`（**`.windows-target` の古いDLLはT1/T2非対応なので使わない**）
- 学習済み重み: `rust/hu_m3_engine/tests/fixtures/*.bin`（webapp assemblyと同一SHA）
- エンジンのscoreは「サンプル信念下のテーブルポイントEV」。MCは「ランダム継続EV」なので、
  T0/T1先行とT1後攻以降でEVのスケールが微妙に異なる（同一ストリート内の比較は常に有効）。
- ヒーローの判断はカードが配られた瞬間からバックグラウンドで解析されるため、
  考えている間に評価が終わっているのが通常。

## AI対戦相手

同じ評価器のfast精度で最善手をプレイ（T1後攻以降は本番webapp相当の強さ）。

## FL（ファンタジーランド）

v1ではFL突入を検知して固定EV（+9.6、`configs/fl_ev_regular_v4_selfplay.json`）を結果に表示するのみ。
定数は `trainer/fl_ev.py` がこのconfigから読むので、configが更新されれば自動で追従する。

**要注意（未解決）**: この9.6が効くのは Python側（`evaluator.py` のMCと `game.py` の結果表示）だけ。
T1〜T4を担う `target/release/ofc_hu_m3_engine.dll` のビルドは 2026-08-05 23:49 で、
`rust/hu_m3_engine/src/infoset.rs` が `DEFAULT_FL_EV = 9.6` になった 08-06 19:59 より**古い**。
つまりエンジン内部は旧定数（10.227）のままの可能性が高く、
T0/T1先行（Python MC）とT1後攻以降（エンジン）でFLの評価がずれる。
解消には `cargo build --release -p ofc_hu_m3_engine` での再ビルドが必要。
FLハンド自体のプレイ・採点は未実装（`regular_fl_solver.exe --solve` で拡張可能）。
