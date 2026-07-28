# T4 BB FL EV感度パイロット (2026-07-28)

診断専用。`promotion_eligible=false`。生成コマンド:

```
python -m ai.tutor.t4_bb_fl_ev_sensitivity --out ai/reports/t4_bb_fl_ev_sensitivity_20260728/report.json --workers 8 --random-count 30
```

## 設定

- FL-live・バスト不能fixture: BB top QQ+空きスロット（middleストレート、
  bottomクワッズ）、BTN top AA+空きスロット（同構造）。すべての完成が合法なので、
  行動EVの差はline勝敗・royalty・FL EVのみから生じる。
- 36 root（hand-picked 6 + seeded random 30、Joker 0/1/2枚 = 26/8/2）、
  各rootで全合法6配置 × 全C(26,3)=2,600相手ドロー × 相手exact best response。
- FL EVスケール {0.8, 1.0, 1.2}。基準値は現行config
  `{14: 0.0, 15: 10.7, 16: 29.9, 17: 63.5}`。
- 実行時間435.6秒（8 worker）。

## 結果

| スケール | トップ行動変化 | mean regret | max regret | best EVシフト mean / max |
|---|---|---|---|---|
| 0.8 | 0 / 36 | 0.0 | 0.0 | +2.30 / +8.72 |
| 1.2 | 0 / 36 | 0.0 | 0.0 | -2.30 / -8.72 |

FL EVを±20%動かすとEVの水準は最大±8.7点動くが、**36 rootすべてで行動順位は
不変（baseline行動のregret厳密に0）**だった。FLが支配的なこのfixture群でも、
T4 BBの意思決定margin（例: trips-top +25.8 vs 非trips -51.9）はFL EV誤差より
はるかに大きい。

## 解釈と限界

1. この結果は「T4のexact決定は現行FL EV定数の±20%誤差に頑健」という証拠であり、
   `all_turn_ai_goal_and_milestones.md`のM7凍結条件へのFL EV要件を検討する際、
   T4層についてはprop. scalingの範囲で追加対策不要を支持する。
2. **一般化しない範囲**: (a) 単一のpublic prefix familyのみ。(b) バスト不能な
   構造のため、bust risk vs FLのトレードオフは測っていない。(c) FL EVが本当に
   効くのはFL追求自体を選ぶT0/T1と、range/solveを通じたT3であり、それは未測定
   （提案4の本体はT3 solveのTV距離測定のまま）。
3. **スケーリングの盲点**: 現行configは14枚FL entry（QQ top）のEVが0.0のため、
   比例スケーリングでは14枚entryの誤評価を検出できない（0×s=0）。T0/T1層の
   感度分析では加法摂動（例: 14枚entryへ+2〜+5点）を併用すべきである。
