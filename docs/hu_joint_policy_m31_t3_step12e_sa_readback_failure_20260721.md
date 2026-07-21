# Step12e attempt0: SA create-readback failure audit (2026-07-21)

Step12e 2-VM lifecycle canary attempt0はNo-Goで安全に閉じた。
ユーザーのfresh explicit authorizationの下で実行された。

## 認可されていた構成

- candidate/reference: `c4-standard-8` 各1台
- attempt0のみ。attempt1、自動retry、3台目は禁止
- Step12e新装備: read-only IAM policy GETの有界retry
  (transport障害のみ、3回、backoff 2s/8s)

## 何が起きたか

1. bootstrap source 3 objectsをgeneration-pinned upload(成功)
2. run-scoped controller service accountのcreate POSTが200で成功
3. **直後のreadback GETが作成済みSAをまだ返さず**
   (`readback != created`)、
   `controller_service_account_create_readback_changed`でfail closed
4. token barrierには未到達。Phase2 IAM mutationは0回。VM insertは0回
5. read-only retry wrapperのretryイベントは0件
   (**Step12dのtransport故障モードは再発しなかった**)

根本原因分類:

```text
controller_service_account_create_readback_changed_iam_eventual_consistency_get404_suspected
```

IAMの結果整合性による既知の伝播遅延である。frozen adminは
**delete側には**有界の不在確認poll(`delete_created_and_wait_absent`)を
持つが、create側readbackは即時GET一発だった。この非対称が原因。
Step12dでは偶然伝播が間に合っていた。

## Runner内蔵cleanupと独立検証

今回はrunner内蔵cleanupが**全4項目を完了・検証**した
(`mandatory_failure_cleanup_verified`):

- phase2_not_installed / token_barrier_not_entered /
  controller_service_account_absent / exact_instances_and_disks_absent

独立GET-only検証(gcloud)でも確認:

- project IAM Phase2 bindings: 0
- controller SA: list一致0件
- 両VM: not found
- bootstrap source objects: 3件retained(認可された保持)

closeout receipt:
`outputs/hu_joint_policy/m31_t3_step6d/step12e_pair_v1_actual/sa_readback_failure_closeout_receipt.json`

- 内部`receipt_sha256`:
  `39020a11f21ac43aa37889c13311a9801f6a7905f8f859d0fff0aa5132da2f92`
- ファイルSHA-256:
  `4dc95545d58abfe70151320bfb8ceaf899cdb0cd1f1e1492c2dc7e59e863e07b`

## Terminal宣言

Step12eのsigner、nonce、contract、source prefix、output root、
controller SA、VM/disk名、confirmation tokenはすべて消費済み
terminalであり、再利用・retry・resumeは禁止。次のcanaryはStep12f。

## Step12fローカル実装(2026-07-21、実装・テスト済み)

- `src/ofc_regular/hu_m31_t3_step6d_rearm2_diagnostic_step12f_sa_create_poll_v1.py`
  frozen adminのサブクラス`CreateReadbackPollingControllerAdmin`。
  createのcontract(absence capability消費→POST 1回のみ→
  readback検証)は完全同一で、readbackが404の間だけ有界poll
  (最大8回、delays 2/4/8/8/8/15/15s、合計60s)。record内容が
  異なる場合は即時fail-closed(整合性違反はpollしない)。
  **2回目のcreateは絶対に行わない**。sealed receiptの形状は
  frozenと完全同一(schema-drift防止)で、poll証跡は
  STEP12F receiptに別途記録
- `src/ofc_regular/hu_m31_t3_step6d_rearm2_diagnostic_step12f_fresh_identity_v1.py`
  terminal 4件(12b/12c/12d/12e)のcloseout検証と非一致
- `scripts/run_hu_m31_t3_step6d_rearm2_diagnostic_step12f_pair_v1.py`
  token `EXECUTE_STEP12F_DIRECT_V2_EXACT_PAIR_ATTEMPT0`。
  両hardening wrapper(readonly retry + SA create poll)を装着
- テスト21件(poll admin 7 / runner 5 / fresh identity 9)、
  dry-run合格

## 残存する既知リスク(Step12f以降の候補)

Phase2 install後の**mutation直後policy readback**は依然として
即時GET比較であり、project policyの伝播遅延で同型の失敗があり得る
(未実測、live未到達領域)。frozen lifecycleの変更を伴うため、
発生した場合に別versionで対処する。

## この2回のcanaryが証明したこと

- fail-closed・cleanup・identity境界は3回連続で完全動作
- Step12c schema fix: live実証済み(12d)
- Step12d transport fix: 有効(12eでretryイベント0=再発なし)
- コストは実質ゼロ(VM未起動のまま)。canaryは想定どおり
  「安価に一過性故障モードを1つずつ炙り出す」働きをしている
