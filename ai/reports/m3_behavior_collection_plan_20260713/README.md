# M3 behavior collection production plan (2026-07-13)

このディレクトリは、大規模データを収集した結果ではなく、behavior temperature calibrationに必要な収集範囲の事前登録です。`plan.json`は、verified gate v2、collectorのroot identity、固定root-hash split、12-cell Joker cycleに結び付いた自己ハッシュ付きartifactです。

## 固定した収集範囲

| population | seed namespace | roots | logged decisions | shards |
|---|---|---:|---:|---:|
| natural | `m3-behavior-calibration-production-natural-20260713-v1` | 134,057 | 536,228 | 135 |
| targeted Joker challenge | `m3-behavior-calibration-production-joker-grid-20260713-v1` | 24,000 | 96,000（targeted 24,000） | 24 |

shard sizeは1,000 rootsです。naturalの最終shardだけ57 rootsになります。

naturalでは各rootから`T1 BB / T1 BTN / T2 BB / T2 BTN`へ各1 decisionが生じます。index 0から連続するroot IDを実際にhash splitへ通した結果、最小prefixは134,057 rootsでした。各roleのsplit件数は同一で、次のとおりです。

- fit: 93,870（要件50,000以上）
- dev: 20,187（要件10,000以上）
- locked test: 20,000（要件20,000以上）

直前の134,056-root prefixはlocked testが19,999件なので不合格です。このため134,057は、このnamespaceに対する最小の連続prefixです。

targeted challengeは`T1/T2 × BB/BTN × Joker 0/1/2`の12-cell cycleです。24,000 rootsで各cellが正確に2,000 targeted roots/decisionsになります。直前の23,999 rootsでは`t2_btn.joker_2`だけ1,999件なので不合格です。

## 容量見積り

保存済み12-root wiring smokeの実測JSONL byte数をroot数に線形外挿し、各fileをbyte単位で切り上げました。

- natural: 4,377,083,937 bytes
- targeted Joker challenge: 786,300,000 bytes
- 合計: 5,163,383,937 bytes（約5.164 GB、10進）

これは`decisions.jsonl`、restricted `roots.jsonl`、`model_evaluations.jsonl`だけの見積りです。manifest、calibration artifact、shard overhead、実データでの行長変動は含みません。圧縮も仮定していません。したがって容量保証ではありません。

## Content bindings

- temperature gate config SHA-256: `4f90642b47740785ea767aa80ce88055b429769756b657441299449c1e708569`
- natural root range SHA-256: `186091321553d789f7dcce1d9f8215f23de00e786dcfe40203a96d49d3073f2b`
- challenge root range SHA-256: `2bbc173c5a511689a336cf1902cd59ffe317b423284ba7e765d85691370f931c`
- plan SHA-256: `4aaa2d5720c4475fc9ecad5b9779ab1cededb567b8c88636fcc7f810392878af`

各shardにもroot index range、first/last root ID、root ID order commitment、splitまたはtarget-cell件数、自己ハッシュがあります。verifierはplanの集計値を信用せず、namespaceとindex範囲から全root IDを再生成して最小性とcommitmentを再導出します。

## 再生成と検証

```powershell
python -m ai.tutor.plan_m3_behavior_collection
python -m pytest tests/test_plan_m3_behavior_collection.py -q
```

このartifactが証明するのは必要件数を満たす決定論的な収集計画だけです。大規模収集の実行、temperature gateの合格、M3の戦略強度、policy/behavior modelのpromotionは一切主張しません。
