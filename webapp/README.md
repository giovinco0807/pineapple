# OFC Pineapple 対戦Webアプリ

レギュラーOFC Pineapple（ヘッズアップ、Jokerなし）のWebアプリです。FastAPI、React PWA、Rustエンジン、Rust FLソルバーを1つのCloud Runコンテナで動かします。

## 実行構成

- API: `backend.ofc_webapp.api:app`
- 静的ファイル: `/app/static`（Viteのビルド結果）
- ゲームエンジン: `/app/native/libofc_hu_m3_engine.so`
- 特徴量エンコーダ: `/app/native/libofc_stage3_feature_encoder.so`
- FL: `/app/native/regular_fl_solver`
- AI候補上位3手の記録: `/app/native/ofc_webapp_decision_inspector`
- SQLite: `/tmp/ofc-webapp.sqlite3`
- 永続化: 専用GCSバケットの `sqlite/ofc-webapp.sqlite3`

ルール構成とAIの街別ディスパッチは `assembly.json` が正です。レギュラーの現行エンジン契約に合わせ、FL突入時の配札はQQ/KK/AA/トリップスのいずれも14枚です。

## コンテナをローカルで確認

Dockerのビルドコンテキストは必ずリポジトリ直下にします。

```powershell
docker build -f webapp/Dockerfile -t ofc-pineapple-webapp:local .
docker run --rm -p 8080:8080 `
  -e SHARED_TOKEN='replace-with-a-long-random-token' `
  -e OFC_GCS_BUCKET='' `
  ofc-pineapple-webapp:local
```

`Dockerfile.dockerignore` と `deploy/cloudbuild.gcloudignore` は許可リスト方式です。約5GBある研究・学習用の `models/` 全体は送信せず、assemblyで固定した実行用モデルだけをローカルDockerとCloud Buildへ渡します。

起動時に、4本の学習済み重み、`stage19_p0` の依存モデル、Rustライブラリ、FLソルバー、decision inspectorのSHA-256がすべて検証されます。1つでも欠落・不一致があればAPIを起動しません。

```powershell
$headers = @{ Authorization = 'Bearer replace-with-a-long-random-token' }
Invoke-RestMethod http://localhost:8080/api/meta -Headers $headers
```

## Cloud Runへ配備

前提:

- Google Cloud CLIがインストール済みで、対象プロジェクトへログイン済み
- 課金が有効
- Cloud Build、Cloud Run、Artifact Registry、Secret Manager、Cloud Storageを作成できる権限
- 学習データ用とは別の、Webアプリ記録専用GCSバケット名

最初にBearerトークンをSecret Managerへ登録します。次の例はトークンを画面に表示せず、一時ファイルも最後に削除します。

```powershell
$projectId = 'YOUR_PROJECT_ID'
$secretName = 'ofc-webapp-shared-token'
$tokenBytes = New-Object byte[] 32
$random = [Security.Cryptography.RandomNumberGenerator]::Create()
try {
  $random.GetBytes($tokenBytes)
}
finally {
  $random.Dispose()
}
$sharedToken = [Convert]::ToBase64String($tokenBytes)
$temporarySecret = New-TemporaryFile
try {
  [IO.File]::WriteAllText(
    $temporarySecret.FullName,
    $sharedToken,
    [Text.UTF8Encoding]::new($false)
  )
  gcloud secrets create $secretName `
    --project $projectId `
    --replication-policy automatic `
    --data-file $temporarySecret.FullName
}
finally {
  Remove-Item -LiteralPath $temporarySecret.FullName -Force
  $sharedToken = $null
}
```

すでにSecretがある場合は、新しいバージョンを追加します。

```powershell
gcloud secrets versions add ofc-webapp-shared-token `
  --project YOUR_PROJECT_ID `
  --data-file PATH_TO_TOKEN_FILE
```

その後、リポジトリ直下から配備スクリプトを実行します。

```powershell
.\webapp\deploy\deploy-cloud-run.ps1 `
  -ProjectId YOUR_PROJECT_ID `
  -RecordsBucket YOUR_GLOBALLY_UNIQUE_OFC_RECORDS_BUCKET
```

スクリプトは以下を設定します。

- region: `asia-northeast1`
- CPU / memory: `1 vCPU / 1Gi`
- min / max instances: `0 / 1`
- concurrency: `4`
- request timeout: `300秒`
- `SHARED_TOKEN`: Secret Managerのlatestバージョン
- GCS: 指定した専用記録バケット

Cloud Run自体は未認証アクセスを許可し、アプリの全APIをBearerトークンで保護します。ブラウザPWAから同一オリジンでAPIへアクセスするための構成です。

## モデルを更新する場合

4本の学習済みモデルは次のファイルだけをコンテナへ取り込みます。

- `rust/hu_m3_engine/tests/fixtures/t4_model_v5.bin`
- `rust/hu_m3_engine/tests/fixtures/t3_model_v2.bin`
- `rust/hu_m3_engine/tests/fixtures/t3first_model_v1.bin`
- `rust/hu_m3_engine/tests/fixtures/t2_model_v1.bin`

モデルを差し替えたら、`assembly.json` の対応するSHA-256も同時に更新してください。

```powershell
Get-FileHash -Algorithm SHA256 `
  rust/hu_m3_engine/tests/fixtures/t4_model_v5.bin
```

Rust生成物のSHAは、コンパイラ出力に対してイメージ作成時に `verify_assembly.py` が封印します。完成したイメージでは `assembly_sha256` も固定され、以後の起動ごとに再検証されます。実行中の構成は認証付き `GET /api/meta` で確認できます。

## Cloud Runでの確認

コールドスタートを含め、少なくとも次を確認します。

```powershell
$serviceUrl = gcloud run services describe ofc-pineapple-webapp `
  --project YOUR_PROJECT_ID `
  --region asia-northeast1 `
  --format 'value(status.url)'
$headers = @{ Authorization = 'Bearer YOUR_SHARED_TOKEN' }
Invoke-RestMethod "$serviceUrl/api/meta" -Headers $headers
```

`/api/meta` のassembly SHA、4本のweights SHA、`rules.name=regular`、`include_jokers=false`、Cloud Runの最新revisionを記録してから受け入れテストへ進みます。
