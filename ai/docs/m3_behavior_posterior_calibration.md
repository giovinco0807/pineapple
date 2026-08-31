# M3 behavior posterior calibration

更新日: 2026-07-13

## 結論

既存のHU T1/T2 PolicyValueNet 4本は、54枚full-card rangeの配線と感度分析には
使えるが、M3 promotion用の行動尤度モデルにはしない。policy headは観測された
行動頻度ではなく、`softmax(teacher action EV / 3)`をListNet KLで近似している。
したがって現在の意味は`teacher_boltzmann_prior`であり、全checkpointを
`promotion_eligible=false`のまま保持する。

| turn/actor | checkpoint SHA-256 | val EV regret | val top-1 |
|---|---|---:|---:|
| T1 BB | `27dab713a658a5ede5637d2c96ae0d5330464b96738f562ae84ce7326c2562c6` | 0.41850 | 58.53% |
| T1 BTN | `064ab29967d4294f76f4ac844000b7fad97553eb341cfb54065f023bbfe7a32f` | 0.64632 | 51.34% |
| T2 BB | `fc47e5a7a02375c8d3fac32d04b0849d8e7b100fe5650007c8c7f8e6c7b69460` | 0.38200 | 49.31% |
| T2 BTN | `a4d5dfcff1811515a6b3db972b7f06e071f00666db49f3cefe8a0079b7e6c83b` | 0.43485 | 48.06% |

各5,000件でteacher soft targetへのKLを最小化する推論温度は
`1.0052 / 1.0016 / 0.9976 / 1.0017`だった。これは学習targetの再現確認であり、
実現行動の確率校正ではない。NPZの`actions`はteacher EVのargmaxなので、これを
観測行動と誤解したone-hot NLLの最適温度は`0.168 / 0.216 / 0.135 / 0.153`となる。

## 現artifactで証明できないこと

- 実際に採られたbehavior actionとhidden discardが保存されていない。
- 全rowをseed 42でshuffleした90/10 validationで、root-disjointではない。
- checkpoint選択とP99診断が同じvalidation領域を使う。
- root ID、worker seed namespace、dataset hash、generator引数、teacher hash、
  source hash、split commitmentがcheckpointにない。
- T1 BTNにfresh unused chunkがない。

既存データのsemantic wiring監査では、各role 10,000件についてlegal mask、turn、
positionの不一致は0で、Joker 0/1/2も存在した。一方、float16 EVのtieにより保存
`actions`と保存EVの再argmaxが各role 3〜7件ずれる。これらはranking学習データの
構造確認であって、posterior likelihoodの品質証明ではない。

## 新しいraw trace契約

1 decisionを`ofc_behavior_decision_log/v1`として保存する。最低限、以下を含める。

- `root_id`とcanonical initial deckのcommitment。
- worker/seed namespace、behavior population ID、policy/checkpoint hash。
- rules、`bb_first_v1`、物理Joker `X1`/`X2`、sampling contract。
- T1/T2、BB/BTN、既存`BehaviorInfoSet`とそのdigest。
- current draw、全legal action ID、27-slot mask。
- 実際にsample/selectionされたaction key、semantic index、hidden discard。
- source policyが確率分布を持つ場合は選択actionのsource probability。

人間、heuristic、特定self-play checkpoint、teacher softmaxは別の
`behavior_target_id`にし、同じ校正集合へ混ぜない。同一rootの全decisionは必ず同じ
splitへ入れる。raw full-private traceは検証・教師生成専用であり、servingの
public-information policy inputへは渡さない。

bootstrapでは、固定したstochastic priorから実際にactionをsampleし、その同一policy
hashに対するlikelihood再現性を確認できる。これはsampling/adapter校正の証明であり、
そのprior自体の戦略的な強さの証明ではない。M3 strength gateは別に通す。またT1/T2の
deployed policy hashが変わった時点で、そのbehavior targetに依存するrange、校正、
T3/T4 solver artifactをstaleとし、最終fixed-point反復で再生成する。

## 固定split

labelを見る前に次で固定する。

```text
bucket = uint64(SHA256("behavior-cal-v1" || root_id)[:8]) % 10000
0..6999    fit
7000..8499 dev
8500..9999 locked test
```

cell不足時もrootを別splitへ移さず、新しいrootを収集する。dataset manifestはraw file
hash/Merkle root、generatorとencoding/action source hash、policy hash、split salt、
role/Joker/legal-count別件数、split root-list commitment、overlap auditを固定する。

## 最初の校正方法

checkpoint logitsを凍結し、T1 BB、T1 BTN、T2 BB、T2 BTNごとにscalar temperatureを
fitのobserved-action NLLで決める。illegal actionは正規化から除外し、float64のstable
log-sum-expを使う。`T in [0.05, 20]`を決定論的に探索し、最終値は分母`10^6`の
`Fraction`へ固定して全metricを再計算する。posterior range側のepsilon smoothingは
校正objectiveへ混ぜない。

scalar temperatureがlocked testを通らない場合、testを見てaction biasを足さず、
manifest-richなfresh traceでpolicy headをobserved-action CE再学習する。

## 事前登録するgate案

- fit 50,000、dev 10,000、locked test 20,000 decisions以上を各roleで持つ。
- Joker challengeは`role × current-draw Joker 0/1/2`ごとに2,000以上。
- root overlap、duplicate root、mask mismatch、semantic roundtrip mismatch、illegal
  observed action、fallbackをすべて0にする。
- calibrated NLLのT=1比はpoint estimate `<= 0`、paired-root bootstrap 95% UCB
  `<= +0.005 nat/decision`。
- uniform legal比NLL改善の95% LCBはrole別`>= 0.02`、Joker cell別`>= 0.01`。
- multiclass marginal ECEはrole別`<= 0.03`、Joker cell別`<= 0.05`。
- Brier scoreのT=1比は`<= +0.002`。
- locked testだけで再推定する診断温度は`0.8 <= T_test <= 1.25`。
- 同一入力の再実行でartifact bytesとhashが一致する。

さらにfull-hand locked testで、真のhidden-discard historyのsupport率100%、uniform prior
より良いposterior log loss、BB/BTN × visible Joker 0/1/2の6層を確認する。

校正artifactは`ofc_behavior_temperature_calibration/v2`としてcheckpoint、dataset、
split、温度、optimizer、raw-derived metrics、gate config/result hashをcontent-boundに
する。`promotion_gate_m3_range`はこの校正artifactとgate resultのhashを独立再検証して
初めて`promotion_eligible=true`を認める。

## 実装順

1. raw trace schema、canonical verifier、root-hash splitを実装する。完了済み:
   `ai/tutor/behavior_calibration_contract.py`（専用22テスト）。
2. 4 roleのfresh full-private behavior traceをローカルsmokeで生成する。完了済み。
3. 大量収集前にmask/action/discard/root auditを通す。完了済み。append-only shard
   collectorでtargeted Joker challenge 24,000 rootsは完了し、natural 134,057 rootsは
   収集中。challengeのdirect-logit evaluationも24 shards・96,000 rowsを生成し、
   complete manifestを保存した。独立fresh readbackでも全入力・全出力を再計算して一致し、
   evaluation content SHA-256は
   `9d99cc235701d930879cba577022f3646d367a212488ad5655bf904405c56382`、
   manifest SHA-256は
   `82f24ba232ca71e49a68cbc0ecce11eb3933c22dd82ce65de1dbcdbbb2a03808`。
   収集・評価manifest自体は昇格claimを持たない。
4. scalar-temperature fitterとlocked-test gateを実装する。実装済み。productionでは
   challenge側はcollection/evaluationとも完全になった。natural側のcollectionと
   direct-logit evaluationが完全になるまでfail-closedで実行できない。
5. calibrated builderを既存priorとは別のopt-in routeとして追加する。実装済み。
   sharded builderは全入力をfresh再構築し、4 checkpointを再照合したうえで既存の
   fixed-point bootstrap dispatchを返すが、runtime自身は常にnon-promotingである。
6. 6層のM3 range evidenceを再生成し、full-card MCCFR holdoutへ進む。
