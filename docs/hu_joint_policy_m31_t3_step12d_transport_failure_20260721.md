# Step12d attempt0: transport failure audit (2026-07-21)

Step12d 2-VM lifecycle canary attempt0はNo-Goで安全に閉じた。
ユーザーのfresh explicit authorizationの下で実行された。

## 認可されていた構成

- candidate/reference: `c4-standard-8` 各1台
- attempt0のみ。attempt1、自動retry、3台目は禁止
- 事前にdry-run(cloud adapter非構築)で最終確認済み

## 何が起きたか

1. bootstrap source 3 objectsをgeneration-pinned upload(成功)
2. run-scoped controller service account作成(成功)
3. **token barrier成功**: token mint、Token Creator revoke、
   zero readback。新しい2フィールド
   (`token_creator_revoke_zero_readback`と同`_evidence_sha256`)を
   含むsuccess receiptが**修正済みPhase2 validatorをliveで通過**。
   Step12cの根本原因修正はlive環境で実証された
4. Phase2 install冒頭の`_require_initial_zero`によるread-only
   policy GETが`iam_https_transport_failed`で失敗
5. **最初のPhase2 IAM mutation前にfail closed**。VM insertは0回

根本原因分類:

```text
phase2_initial_zero_policy_get_iam_https_transport_failed_before_first_mutation
```

一過性のHTTPSトランスポート障害である。step11 REST IAM adapterは
設計上retryなしのfail-closedであり、read-only GETの一時障害でも
run全体が終了する。

## Runner内蔵cleanupと独立検証

runner内蔵cleanupは4項目中3項目を完了・検証:

- Token Creator残余ゼロ: 検証済み
- controller service account不在(GET 404): 検証済み
- VM/disk不在(GET 404): 検証済み
- `phase2_controller_then_worker_zero`: 同じtransport障害で
  読み取れず**未完了** → receiptは
  `mandatory_failure_cleanup_incomplete`

未完了の1項目は独立GET-only検証(gcloud)で補完し、cloudは
完全にクリーンであることを確定した:

- project IAM: Phase2の8 condition-title bindings = 0
- bucket IAM: 0
- worker SA actAs binding: 0
- controller SA: list一致0件、describeはdenied-or-not-exists
- 両VM: not found
- stage prefix objects: 0
- bootstrap source objects: 3件retained(Step12cと同じ扱い。
  結果objectではなくsource。認可された保持)

closeout receipt:
`outputs/hu_joint_policy/m31_t3_step6d/step12d_pair_v1_actual/transport_failure_closeout_receipt.json`

- 内部`receipt_sha256`:
  `dd33b2117b5bf92f5f75723f26951b665625a5677eb52e782e58b43eaf0e2a0e`
- ファイルSHA-256:
  `cdd0790b2dbdd3eadd1236e43c65eb57d7e17f2a020509e1398919a9a02836e6`

## Terminal宣言

Step12dのsigner、nonce、contract、source prefix、output root、
controller SA、VM/disk名、confirmation tokenはすべて消費済み
terminalであり、再利用・retry・resumeは禁止。次のcanaryは
Step12e相当の新しいversioned entrypointを要する。

## Step12eへの設計提言(2026-07-21 ユーザー承認・実装済み。下記は原提言)

1. **read-only GETの有界idempotent retry**: 今回の失敗はmutationでは
   なくread-only policy GETの一時障害。attempt0原則(pair/VM/mutation
   のretry禁止)を維持したまま、idempotentな読み取りに限り有界retry
   (例: 3回、指数backoff、合計30秒以内)を新contractで許可すれば、
   この故障モードを除去できる。mutationのretry禁止は不変
2. failure cleanupの検証読み取りも同じretryを適用し、
   `mandatory_failure_cleanup_incomplete`の頻度を下げる
3. その他の要件(producer/validator round-trip、machine-type
   preflight、disjointness、dry-run分離)はStep12dで実装済みの
   パターンを踏襲する

## Step12eローカル実装(2026-07-21、ユーザー認可済み)

上記提言のとおり実装した。Cloud実行は未実施で、別途fresh explicit
authorizationを要する。

- `src/ofc_regular/hu_m31_t3_step6d_rearm2_diagnostic_step12e_readonly_retry_v1.py`
  read-only `get_policy`限定の有界retry wrapper。事前登録値:
  `iam_https_transport_failed`のみretry可、最大3回、backoff 2s/8s。
  `add_binding`/`remove_binding`はいかなる場合もretryしない。
  凍結済みstep11 adapterは無変更。wrapper二重装着は拒否。
  retryイベントはreceiptへ記録される
- `src/ofc_regular/hu_m31_t3_step6d_rearm2_diagnostic_step12e_fresh_identity_v1.py`
  terminal 12b/12c/12d全3件のcloseout検証とidentity非一致。
  Step12d closeout receipt(`dd33b2...`)とdeployment binding
  (`cb94bb...`)をpin
- `scripts/run_hu_m31_t3_step6d_rearm2_diagnostic_step12e_pair_v1.py`
  新confirmation token `EXECUTE_STEP12E_DIRECT_V2_EXACT_PAIR_ATTEMPT0`。
  backend構築直後に`backend.iam_admin`をwrapperへ差し替え
  (token barrier・Phase2・cleanupの全policy GETをカバー)。
  STEP12E_FINAL/STEP12E_FAILUREにretryイベント数を記録
- テスト21件: wrapper単体8(retry予算・backoff・mutation非retry・
  非transport即時失敗・委譲・nesting拒否・事前登録値)、
  fresh identity 8、runner 5(旧token拒否・dry-run非構築・
  wrapper装着・両=全terminal immutable登録)

## 今回のcanaryが証明したこと

- Step12c schema fix: live通過(再発なし)
- machine-type 30720 MiB契約: preflight通過
- fail-closed動作: mutation前停止、bounded shutdown、no retry
- fresh identity境界: terminal 12b/12cとの完全非一致で実行
- 安全性は再実証されたが、性能・品質・AI強度の証拠ではない
