# Phase2 read failures: true root cause (2026-07-21)

## 要約

Step12d/12f/12gの2-VM canaryはいずれもPhase2 install冒頭の
`get_policy("project")`で`iam_https_transport_failed`となり停止した。
当初これをローカルネットワーク(USB Wi-Fi)の間欠障害と診断したが、
**それは誤診**であった。

真の根本原因は決定論的なコードバグ:

- 実行時のHTTPSクライアント`StdlibCloudHttpsClient`
  (`hu_m31_t3_step6d_rearm2_diagnostic_step12b_live_cloud_adapters_v2.py`)
  の`_ALLOWED_HOSTS`に`cloudresourcemanager.googleapis.com`が
  **欠落**していた
- project-level IAMポリシー読み書きはこのhostを使う
- 呼び出しは`parsed.hostname not in _ALLOWED_HOSTS`で
  `ValueError("live cloud request escaped HTTPS allowlist")`を
  **ネットワークに触れる前に即座に(0.0001s)**送出
- step11の`_request`が全例外を`iam_https_transport_failed`へ一括変換し、
  設定バグがネットワーク障害に偽装された

## 三つのallowlistの不一致

| allowlist | cloudresourcemanager | 用途 |
|---|---|---|
| step11 rest_iam_admin (`_ALLOWED_HOSTS` L46) | 含む | URL検証(合格) |
| preflight (`_ALLOWED_HOSTS` L136) | 含む | read-only preflight(合格していた) |
| **live StdlibCloudHttpsClient (L72)** | **欠落** | **実行時transport(拒否)** |

step11はURLを自分のallowlist(cloudresourcemanager許可)で検証して
通し、その後injectされたlive clientに渡す。live clientのallowlistが
拒否する。read-only preflightが通っていたのは、preflightが別の
(hostを含む)allowlistを使っていたため。

## なぜユニットテストで検出されなかったか

全ユニットテストはfake/mock HTTPクライアントを使うため、live client
のallowlistギャップはlive実行でのみ顕在化した。355+テストが緑でも
canaryは毎回同じ場所で死んだ。

## 誤診の証拠訂正

- 「26秒/失敗」= retry wrapperのsleep時間(接続タイムアウトではない)
- probe「IPv4/IPv6 47/47成功」= 生urllibがStdlibCloudHttpsClientの
  allowlistを迂回していたため
- health probe「project 0/8, iam 8/8, storage 8/8」= projectのみ
  決定論的にallowlist拒否(間欠ではない)
- `Get-NetAdapterPowerManagement`のデバイスエラー= 無関係(red herring)

## 修正

1. `cloudresourcemanager.googleapis.com`を
   `StdlibCloudHttpsClient._ALLOWED_HOSTS`へ追加(1行)
2. 回帰テスト(`test_..._live_cloud_adapters_v2.py`):
   - `rest_iam._ALLOWED_HOSTS <= subject._ALLOWED_HOSTS`
     (step11が許可する全hostをlive clientも許可する=drift防止)
   - 非allowlist hostは依然として拒否される(negative control)

## 検証

修正後、実admin clientで`get_policy("project")` = **5/5成功**
(修正前は0/8)。iam/storageも5/5。

## Step12e(別issue)は誤診ではない

Step12eの失敗はSA create直後のreadback 404(IAM eventual
consistency)であり、これは本物の別問題。Step12fのcreate-readback
pollがこれを正しく解決している。allowlistバグとは無関係。

## 保持した対策の位置づけ

Step12e/12g/12hで追加したread-only retry(deadline方式含む)は、
今回の真因ではなかったが、transientなtransport障害への
defense-in-depthとして有害ではないため保持する。ただし
**canaryが毎回死んでいた真の原因はこのallowlist 1行**である。

## 次

真の決定論的ブロッカーは除去された。Step12h canaryは初めて
Phase2 read → Phase2 IAM install → VM insert の未踏区間へ到達できる
見込み。実行にはユーザーのfresh authorizationを要する(費用・identity)。
