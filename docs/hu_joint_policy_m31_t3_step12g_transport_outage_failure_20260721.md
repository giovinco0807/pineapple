# Step12g attempt0: transport outage failure audit (2026-07-21)

Step12g 2-VM lifecycle canary attempt0はNo-Goで安全に閉じた。
ユーザーのfresh explicit authorizationの下で実行された。

## 何が起きたか

1. bootstrap source 3 objects upload成功
2. controller SA作成成功(create-readback poll発火0=即時可視)
3. **token barrier成功**(Step12c修正のlive通過4回目)
4. Phase2 initial-zeroのread-only GETで`iam_https_transport_failed`。
   **retry event 10件** = 2本のget_policyが各6回のv2予算(115s)を
   使い切った。fail closed
5. Phase2 IAM mutation 0、VM insert 0

根本原因分類:

```text
iam_https_transport_multiminute_outage_two_phase2_reads_each_exhausted_6attempt_115s_budget
```

Step12fより長いburstで、拡張済みのv2予算(6回/115s)でも足りなかった。

## 事後再現(2026-07-21)

runの直後にスクリプトで`get_policy(project)`を呼ぶと3.67sで
`iam_https_transport_failed`が再現、直後の生POSTは0.41sで成功。
局所的な間欠transport障害が実行時点で継続していたことを確認。

## 機能した対策(3層とも設計どおり)

- Step12c schema fix: token barrier live通過(4回目)
- Step12f SA create poll: 発火0(即時可視、故障モード再発なし)
- Step12g readonly retry v2: 10回retryしたが、局所障害が予算を上回った

## Runner内蔵cleanupと独立検証

cleanup 3/4完了(未完了はPhase2-zero読み取り、同じ障害)。
独立GET検証でcloud完全クリーン確定:

- project/worker SA/bucketのPhase2 bindings: すべて0
- controller SA不在、両VM not found
- bootstrap source objects: 3件retained
- Phase2 mutation 0、VM insert 0、費用実質ゼロ

closeout receipt:
`outputs/hu_joint_policy/m31_t3_step6d/step12g_pair_v1_actual/transport_outage_failure_closeout_receipt.json`

- 内部`receipt_sha256`:
  `43aa9cd4400fa54c8f5500af2bebbb2638fee4da07ee753d85b79272476e7006`
- ファイルSHA-256:
  `bf28b2849d4aca0e3fa0ed06753d14cbc92842afa2bb8a899538081298d32eb3`

## Terminal宣言

Step12gの全identityは消費済みterminal。次のcanaryはStep12h。

## Step12hローカル実装(2026-07-21、実装・テスト済み)

固定回数の予算は可変長の障害に負け続けるため、Step12hは
**deadline方式**に切替:

- `src/ofc_regular/hu_m31_t3_step6d_rearm2_diagnostic_step12h_readonly_retry_v3.py`
  idempotentなpolicy readを、**合計480秒**の期限まで再試行
  (個別backoffは60秒でcap)。480sはfrozen token barrierが既に
  許容する伝播上限`MAX_PROPAGATION_SECONDS`と同値で、恣意的な
  数字ではなく既存の先例に整合。mutation非retryは不変。
  v1/v2/v3の相互nesting拒否
- `src/ofc_regular/hu_m31_t3_step6d_rearm2_diagnostic_step12h_fresh_identity_v1.py`
  terminal 6件(12b〜12g)のcloseout検証と非一致
- `scripts/run_hu_m31_t3_step6d_rearm2_diagnostic_step12h_pair_v1.py`
  token `EXECUTE_STEP12H_DIRECT_V2_EXACT_PAIR_ATTEMPT0`。
  v3 deadline retry + Step12f SA create pollの両hardening装着

Step12gの~4.5分の総障害時間に対し、v3は各read呼び出しに最大480秒の
耐性を与えるため、同型のburstを乗り切る見込み。ただし局所network
自体の安定化(有線化)が本質的解決である点は変わらない。
