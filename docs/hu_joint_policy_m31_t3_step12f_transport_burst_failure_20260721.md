# Step12f attempt0: transport burst failure audit (2026-07-21)

Step12f 2-VM lifecycle canary attempt0はNo-Goで安全に閉じた。
ユーザーのfresh explicit authorizationの下で実行された。

## 何が起きたか(タイムライン、JST)

- 19:16:48 contracts生成
- 19:17:33 bootstrap source 3 objects upload成功
- 19:17:50 controller SA作成成功(**readbackは即時可視。
  Step12fのcreate-readback pollは発火不要=Step12e故障モード再発なし**)
- 19:19:55 **token barrier成功**(Step12c修正のlive通過3回目)
- 19:22:54 Phase2 initial-zeroのread-only GETで
  `iam_https_transport_failed`によりfail closed

Phase2の6回のGET試行が179秒を消費(retry sleep 20s除くと
**約26秒/失敗**)。retry wrapperは設計どおり動作し、1本目のGETは
2回のretryの末に成功、2本目のGETが3回目で予算(計10s)を使い切った。

根本原因分類:

```text
iam_https_transport_failed_burst_during_phase2_initial_zero_reads_retry_budget_10s_insufficient
```

## ネットワーク診断(2026-07-21実施)

- 事後probe(4分間、5秒間隔、IPv4/IPv6分離、
  cloudresourcemanager.googleapis.com): **47/47双方成功** →
  恒常的な経路・アドレスファミリ問題ではない
- 3ホスト(cloudresourcemanager/iam/storage)即時probe: 全成功
- 接続はUSB Wi-Fiアダプタ(BUFFALO WI-U3-2400XE2)。
  `Get-NetAdapterPowerManagement`が「デバイスが機能していません」を
  返した(ドライバレベルの不調の兆候)。WLAN切断イベントはなし
  (完全切断ではなくパケットロス/ストール型)
- 結論: **数分単位のepisodicなWi-Fiストール**。長時間run(6分超)
  だけが被弾し、短いCLI呼び出しは無事という観測と整合

## 機能した対策(3層とも実証済み)

1. Step12c schema fix: token barrier receiptがlive通過(3回目)
2. Step12e readonly retry: 1本目のGETバーストを実際に救済
3. Step12f SA create-readback poll: 今回は即時可視で発火不要
   (発火0はStep12e故障モードの不在証明)

## Runner内蔵cleanupと独立検証

cleanup 4項目中3項目完了(未完了はPhase2-zero読み取り、同じburst)。
独立GET検証でcloud完全クリーンを確定:

- project/worker SA/bucketのPhase2 bindings: すべて0
- controller SA: 不在。両VM: not found
- bootstrap source objects: 3件retained(認可された保持)
- Phase2 IAM mutation: 0回。VM insert: 0回。費用実質ゼロ

closeout receipt:
`outputs/hu_joint_policy/m31_t3_step6d/step12f_pair_v1_actual/transport_burst_failure_closeout_receipt.json`

- 内部`receipt_sha256`:
  `db0aa1bd3f69e3318b2e85a1c871c60ea3c4738d0a4b080c50d0da223c43aecd`
- ファイルSHA-256:
  `a3a9af38691eedfe4befaa9c9bb13bae4fb1c2069e42597245329cc15cf35ec8`

## Terminal宣言

Step12fの全identityは消費済みterminal。再利用・retry・resume禁止。
次のcanaryはStep12g。

## Step12gローカル実装(2026-07-21、実装・テスト済み)

- `src/ofc_regular/hu_m31_t3_step6d_rearm2_diagnostic_step12g_readonly_retry_v2.py`
  read予算を拡張: **最大6回、backoff 2/8/15/30/60s(sleep計115s)**。
  失敗接続自体の所要(~26s/回)を含め数分のburstを乗り切る設計。
  token barrierの480s伝播予算という先例に整合。mutation非retryは
  不変。新旧wrapperのnesting相互拒否
- `src/ofc_regular/hu_m31_t3_step6d_rearm2_diagnostic_step12g_fresh_identity_v1.py`
  terminal 5件(12b〜12f)のcloseout検証と非一致
- `scripts/run_hu_m31_t3_step6d_rearm2_diagnostic_step12g_pair_v1.py`
  token `EXECUTE_STEP12G_DIRECT_V2_EXACT_PAIR_ATTEMPT0`。
  v2 retry + Step12f SA create pollの両hardeningを装着

## 実行環境への推奨(ユーザー対応事項)

次のcanary実行前に、可能であれば:

1. **有線LAN接続に切り替える**(最も確実)
2. またはUSB Wi-Fiアダプタの挿し直し・ドライバ更新・
   USB省電力(selective suspend)の無効化

Step12gのretry予算は数分のburstに耐えるが、ネットワーク自体の
安定化が本質的な解決である。
