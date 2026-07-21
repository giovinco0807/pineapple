# Step12d entrypoint decision (M3.1 T3 lifecycle canary)

Status date: 2026-07-21

この文書は「次のCloud lifecycle canaryはStep12dである」という決定を
リポジトリ内のversioned文書として固定する。これまでこの決定は
引継ぎテキストにしか存在しなかった。

この文書は決定の記録であり、Cloud実行の認可ではない。Cloudで
candidate/reference pairを起動する前に、ユーザーのfresh explicit
authorizationが別途必要である。

## 1. 前提: Step12cはterminal

Step12c 2-VM lifecycle canaryはPhase2 validator schema failureで
No-Goとなり、安全に閉じた。詳細は
`docs/hu_joint_policy_m31_t3_step12c_phase2_schema_failure_20260721.md`
を参照。

以下のStep12c identityはすべて消費済みterminalであり、再利用禁止:

- signer
- nonce
- deployment/run/direct identity
- contract
- source prefix
- output root
- controller service account
- VM/disk名
- confirmation token

Step12c runnerのretry/resume/attempt1/第三VMは禁止。Step12cコード・
テスト・成果物はimmutable terminal evidenceとして保持し、変更しない。

closeout receipt:
`outputs/hu_joint_policy/m31_t3_step6d/step12c_pair_v1_actual/phase2_schema_failure_closeout_receipt.json`

- receipt内部の`receipt_sha256`フィールド(canonical self-hash):
  `31e675593ca168fc899a6dc34083a6f54c8258aabbb62dcd71906d56d77161ef`
- receiptファイル自体のSHA-256:
  `322eea21eb52bff767539127f29677951a074cf87aa4b0f37eace172ffd4235f`

両者は別の値である。検証時に混同しないこと。

## 2. 決定: 次はStep12d

次のCloud lifecycle canaryは、Step12d相当の新しいversioned
entrypointとして実装する。構成はStep12cで認可されていたものと同じ:

- candidate: `c4-standard-8` 1台
- reference: `c4-standard-8` 1台
- attempt0のみ。attempt1、自動retry、3台目は禁止

signer、nonce、contract、source prefix、output root、controller SA、
VM/disk名、confirmation tokenはすべて新規生成し、Step12b/Step12cとの
完全非一致をテストで固定する。

## 3. Step12d実装の必須要件(再発防止)

Step12cの失敗原因は、success receipt producerが追加した2フィールドを
Phase2 validatorのallowlistが認識しなかったというproducer/validator間の
schema driftである。また`c4-standard-8`のmemory契約が実値
`30720 MiB`ではなく誤った`32768 MiB`でハードコードされていた。
同型の失敗を防ぐため、Step12dでは以下を必須とする:

1. **Producer→validator round-tripテスト**
   receipt producerの実出力をそのままPhase2 validatorに通すテストを
   Step12d test suiteに含める。schema(フィールドallowlistとnested
   evidence構造)はproducerとvalidatorが同一モジュールを参照する
   single sourceとし、片側だけの変更をテストで検出する。
2. **Machine-type契約のpreflight照合**
   vCPU/memory等のmachine-type仕様はハードコード値をpinする場合でも、
   preflightでGCP APIの実値をGETして契約値と照合し、不一致なら
   fail closedする。
3. **Disjointnessテスト**
   Step12b/Step12cの全identity(signer、nonce、contract、prefix、
   output root、SA、VM/disk名)との非一致をテストで固定する。
4. **Dry-run分離**
   dry-runではCloud adapterを構築しないことをテストで確認する。
5. **既存Step12b/12c回帰の維持**
   Step12b/Step12c回帰(2026-07-21時点で275 passed)とprofile hashが
   不変であることを確認する。

fixture、tamper、cleanup、resume、receiver testsはStep12cと同水準以上
を維持する。

## 4. Quota並行作業(推奨)

Tokyo C4-family quotaは24 vCPUであり、canary合格後の次waveは現状では
最大`3 × c4-standard-8`(24 vCPU)に制限される。旧計画の
`20 × c4-standard-16 = 320 vCPU`は起動不能。

quota増枠申請は費用ゼロでリードタイムがあるため、Step12d実装と
並行して申請だけ先行させることを推奨する。ただしfanout設計の確定は
lifecycle canary合格後まで行わない。

## 5. R0b(exact-T4 Windows release runtime)は別トラック

R0bはT3 runtime promotionおよびfull-hand releaseのブロッカーだが、
Step12dとは独立に進められる。解決策は次のいずれか:

1. 元のpinned DLLを復旧してSHAを確認する
2. 新version contractとfresh seedsで検証をやり直す
   (rebuild candidateはsecond-seat p99 `2.3487 ms > 2.0 ms`でNo-Go)

2.を選ぶ場合、閾値2.0msの根拠(ウォームアップ条件・計測環境)を
新contractで明文化すること。

### 2026-07-21 ルート1(pinned DLL復旧)の探索結果: ローカルでは不可能

read-onlyで以下を探索したが、pinned SHA
`03d27825ccdf37ff52107207d8008df5b51235431c7274c604f922bc23c84d69`
に一致するコピーは存在しなかった:

- C:ドライブ全域・D:ドライブ全域のファイル名検索
  (`ofc_hu_m3_engine.dll`は計7ファイル存在、SHAはすべて不一致)
- ゴミ箱(Shell COM経由、該当なし)
- git全履歴(コミットされたことがない)
- r0 snapshot payload 2種(feature encoder DLLのみ。engine DLLは
  「Only the R0b Windows DLL role is deferred」のとおり未梱包)
- E:ドライブ(無関係のインストーラディスク)

発見された全コピーのSHA-256(いずれもpinと不一致):

| SHA-256先頭 | 場所 | ビルド日 |
|---|---|---|
| `00bd3b...` | `target/debug/`(+deps) | debug build |
| `7f30e3...` | `target/m30_build/release/deps/` | 2026-07-18 13:18 |
| `faf1ee...` | `target/m30_rebuild_20260718/release/`(+deps) | 2026-07-18 (rebuild候補) |
| `2ef66f...` | `%TEMP%/ofc_attempt05_target/release/`(+deps) | 2026-07-14 |

決定的な証拠: `target/m30_build/release/`内の全ファイル
(.pdb/.exp/.lib/.rlib/.fingerprint)が2026-07-18 13:17-13:19の
タイムスタンプを持つ。**元のpinned DLLは2026-07-18のm30_buildへの
再ビルドで上書き破壊された**と推定される。deps側は`7f30e3...`に
置換され、トップレベルDLLは削除されたまま残らなかった。

残るルート1の可能性(ユーザーのみ実行可能):

- VSS shadow copies(管理者権限で`vssadmin list shadows`)
- このマシン以外の外部バックアップ・別マシン

これらに元DLLがなければ、**ルート2(新versioned contractでの
再検証)が唯一の道**である。

### 2026-07-21 ルート2 v2検証ラダーの結果: No-Go(ただし原因特定済み)

事前登録: `configs/hu_joint_policy_m30_t4_runtime_v2_candidate.json`
(ゲート閾値は既存と完全同一、fresh seeds、失敗時再実行禁止を
結果を見る前に固定)。

結果:

1. focused tests: 20 passed
2. pilot100 (seed 2026072101): **全12ゲート合格**(後攻p99 1.933ms)。
   7/18のNo-Go(2.3487ms)は50サンプルでのp99=最大値という脆い計測に
   よる外れ値だったことが確定
3. pilot1000 (seed 2126072101): **後攻p99 total latencyのみ不合格**
   (3.046ms > 2.0ms)。semantic系9ゲートはすべて合格

事後分析(ゲートは変更していない。監査記録のみ):

- **native(DLL本体)の後攻p99は1.274msでゲート内**。超過分は
  Pythonラッパー側オーバーヘッドのスパイク(overhead p99 2.28ms、
  最悪2.9ms)であり、エンジン性能の問題ではない
- 候補DLLは先攻で歴史的バイナリの約3倍高速
  (p50 7.08ms vs 23.75ms、batch 331 vs 156 roots/s)

seedは両方消費済み。同一・追加seedでの再実行は禁止。

次の選択肢(ユーザーの明示認可が必要):

- **案A(推奨)**: v3プロトコルを新規事前登録し、後攻2.0msゲートを
  native latencyへ適用+total latencyには別途の緩い健全性上限
  (例: p99 ≤ 10ms)を設定して、fresh seedsで再検証する。
  根拠: contractがpinするのはDLLバイナリであり、Python側の
  オーバーヘッドは実行環境(Python版数等)に依存する
- **案B**: Pythonラッパーのオーバーヘッド(GC・アロケーション)を
  最適化し、ゲート無変更のままv3をfresh seedsで再検証する
- **案C**: VSS/外部バックアップで元DLL(`03d278...`)を復旧する
  (歴史的contractの凍結済み証拠がそのまま有効になる)

いずれもpost-hocのプロトコル変更・再試行にあたるため、AIが
勝手に実行してはならない。

### 2026-07-21 ユーザー認可により案Aを実行: v3ラダー全ゲート合格(R0b Go候補)

ユーザーが案A(後攻2.0msゲートをnative latencyへ適用+total
latency p99 ≤ 10msの健全性上限、fresh seedsで再検証)を明示選択した。

事前登録: `configs/hu_joint_policy_m30_t4_runtime_v3_candidate.json`
実装: `src/ofc_regular/validate_hu_m30_t4_runtime_v3.py`
(凍結済みv1 validatorは無変更。v1のpilot loopを再利用し、v3ゲート
ベクトルを適用。v1ゲートはartifact内に`v1_reference_gates`として保存)

結果(全ステップ合格):

1. v3 validatorユニットテスト: 8 passed
2. pilot100 (seed 2226072101): 全13ゲート合格
   (native p99 0.744ms、total p99 1.791ms)
3. pilot1000 (seed 2326072101): **全13ゲート合格**
   - 後攻native p99 **0.937ms** ≤ 2.0ms(max 1.109ms)
   - 後攻total p99 2.323ms ≤ 10ms(max 2.492ms)
   - 先攻total p50/p99/max 4.30/7.79/10.88ms(ゲート75/100ms)
   - batch 500.5 roots/s(歴史的contractは156 roots/s)
   - semantic系9ゲート(exact 2024列挙・Python parity・scalar/batch
     parity・決定性・regret非負等)すべて合格

**R0bはvalidationレベルでGo**。候補runtime
`hu_m30_t4_exact_both_seats_v3`(DLL SHA `faf1ee...`)は全証拠が
揃った。`current`・named profile・full replacementは一切変更して
いない。歴史的M3.0 contractのhashも無変更。

### 2026-07-21 ユーザー認可によりrelease review完了: R0b Go

ユーザーの明示認可により、v3 release contractを凍結した:
`configs/hu_joint_policy_m30_t4_runtime_v3.json`
(status: `complete_component_go_explicit_opt_in`)

- activationはv1と同じ明示opt-in方式: `--t4-mode-a/b m30_exact` +
  `--hu-t4-native-library`/`--hu-t4-native-sha256`でv3のpin
  (`faf1ee...`)をCLIから渡す
- named profile追加・`current`昇格・full replacementは引き続き別途
  ユーザー判断(remaining_no_goに明記)
- **T3 runtime promotion / full-hand releaseのR0bブロッカーは解消**

## 6. 実行順序

1. 本文書のとおりStep12d entrypointをローカル実装・テストする
   (Cloud認可不要) — **完了(2026-07-21、下記7参照)**
2. ユーザーからfresh explicit authorizationを得る
3. candidate/reference各1台、attempt0のみ実行する
4. heartbeat、DONE、self-delete、receive、VM/disk/IAM zeroを
   独立検証する
5. 合格時だけperformance-lockのquota/fanoutを再設計する

## 7. 2026-07-21 ローカル実装完了

実装ファイル:

- `src/ofc_regular/hu_m31_t3_step6d_rearm2_diagnostic_step12d_fresh_identity_v1.py`
  terminal Step12b **と** Step12c 両方のcloseout receipt検証・
  identity非一致(9フィールド+VM/disk名)・output root排他・
  両terminal treeのsnapshot
- `scripts/run_hu_m31_t3_step6d_rearm2_diagnostic_step12d_pair_v1.py`
  Step12b v2 protocolを再利用する新versioned entrypoint。
  新confirmation token `EXECUTE_STEP12D_DIRECT_V2_EXACT_PAIR_ATTEMPT0`、
  attempt0のみ、dry-runはcloud adapter非構築、両terminal rootを
  immutable登録、STEP12D_FINAL/STEP12D_FAILURE receipt

テスト(17 passed):

- `tests/test_hu_m31_t3_step6d_rearm2_diagnostic_step12d_fresh_identity_v1.py`
  実prepare_runのidentityが両terminalと非一致/12b・12cのkey・nonce
  再利用拒否/output root排他/terminal tree・closeout凍結/
  12c closeout改竄拒否
- `tests/test_hu_m31_t3_step6d_rearm2_diagnostic_step12d_pair_v1_runner.py`
  新token検証(12c tokenは拒否)/dry-run backend非構築/
  execute_once単発転送/両terminal root immutable登録
- `tests/test_hu_m31_t3_step6d_rearm2_diagnostic_step12d_roundtrip_v1.py`
  **セクション3必須要件の実装**:
  (1) 実producer(`run_token_creator_barrier`)出力をそのまま
  Phase2 validator(`_validate_token_barrier`)へ通すround-trip、
  producer全フィールド集合==validator allowlistの恒等、
  追加フィールドのfail-closed。
  (2) machine-type契約30720 MiB束縛と、旧誤値32768 MiB readbackの
  fail-closed

## 8. 2026-07-21 attempt0実行結果: No-Go(transport障害、mutation前fail-closed)

ユーザーのfresh explicit authorizationを得て、quota申請
(C4 asia-northeast1 24→128 vCPU、審査中)と並行してattempt0を実行。
dry-run合格後の本実行は、token barrier成功(**Step12c修正の
live実証**)の後、Phase2冒頭のread-only policy GETが
`iam_https_transport_failed`で失敗し、最初のIAM mutation前に
fail closedした。VM insertは0回。

独立GET検証でcloudの完全クリーンを確定済み。詳細・closeout receipt・
Step12e設計提言は
`docs/hu_joint_policy_m31_t3_step12d_transport_failure_20260721.md`
を参照。**Step12d identityは消費済みterminal。次のcanaryはStep12e。**
