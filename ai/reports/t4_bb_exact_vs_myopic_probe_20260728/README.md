# T4 BB component probe: exact vs legacy myopic (2026-07-28)

診断専用。`promotion_eligible=false`、full-game strength claimなし。生成コマンド:

```
python -m ai.tutor.t4_bb_exact_vs_myopic_probe --out ai/reports/t4_bb_exact_vs_myopic_probe_20260728/report.json --workers 8
```

## 測定内容

同一rootのT4 BB決定について、2つの規則を宣言uniform exchangeable restart
belief下のexact EVで比較する。simulationなし（regretは全列挙による厳密値）。

- baseline `myopic`: レガシーplayout完成規則 `best_t4_completion(opponent=None)`
  （相手盤面・相手応答を無視してroyalty+自FL EVを最大化）
- candidate `exact`: `t4_bb_exact_uniform_deal_response_v1`
  （全C(26,3)相手ドロー × 相手exact best responseの平均を最大化）

root: seeded 54枚シャッフルのrandom_legal 40（盤面品質は無フィルタ、
Joker 0/1/2枚 = 30/24/6）+ FL-live fixture 20。

## 結果

| generator | roots | fire率 | mean regret | fired時mean | max regret |
|---|---|---|---|---|---|
| random_legal | 40 | 5.0% (2) | 0.0237 | 0.475 | 0.618 |
| fl_live | 20 | 0% | 0.0 | — | — |
| 全体 | 60 | 3.3% | 0.0158 | 0.475 | 0.618 |

fireした2件はいずれもline勝敗のかかったkicker/discard選択で、相手盤面を
見ないmyopicが検出できない状況だった（regret +0.33 / +0.62点）。FL-live
fixtureではmyopicとexactが完全一致（royalty+FL最大化がexactと整合する構造）。

## 解釈

1. **レガシーmyopic T4規則の取りこぼしは平均約0.016〜0.024点/hand**と小さい。
   joker_rule_migrationのT3分析（旧方策mean regret 0.029）と合わせて、
   **T3/T4の伸び代は合計でも0.1点/handのオーダー**である可能性が高い。
2. これは`roadmap_improvement_proposal_20260728.md`提案1の狙いどおりの
   投資判断材料になる: 全体benchmark（旧スタックでMCTS vs BC greedyが
   +12.7点/hand差）と比べ、T3/T4精緻化の限界効用は小さく、
   **伸び代はT0〜T2側に集中している**と推定される。M4〜M6（T2→T1→T0）への
   投資優先を支持する。
3. 限界: (a) regretは宣言belief測度下の値。(b) random rootは実プレイ分布と
   異なる（tactical局面の出現率が過小/過大の可能性）。(c) n=60。
   (d) T4 BTN側とT0〜T2の直接測定は未実施。本probeはpaired self-play
   strength probe（提案1本体）の代替ではない。
