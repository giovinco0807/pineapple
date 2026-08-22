# m7世代の訓練・書き出し・ゲート道具 — 救出コピー (2026-08-22)

これらは `t2_model_v2.bin` 他、現行ピンの .bin を作った実物のスクリプト。
元の置き場所は**リポジトリ外**だった: Claudeセッションのscratchpad
(`%LOCALAPPDATA%\Temp\claude\36087496-…` と `140fee39-…`)と
WSLの `~/ofc-m7/ablation/scripts/`。セッションscratchpadは揮発性のため、
2026-08-22にここへ救出した。出所の調査記録は
`docs/street_gate_leak_map_20260821.md` の続きの作業ログにある。

| ファイル | 役割 |
|---|---|
| `train_ship.py` | M7 Task-0凍結プロトコルの本体。安定ハッシュholdout、重み付きペアワイズ順位+0.1回帰、warm-start第1層再正規化 |
| `train_t2second_2048.sh` | t2_model_v2.bin を作った実コマンド(WSL CUDA venv、seed 997990041) |
| `m7v4_t3v7_train.py` | T3v7/T1リラベルで使われた訓練ハーネス(metadataの street 表記が T3 固定という既知の癖あり) |
| `ofcdata.py` | 全trainerが import する corpus loader / MLP / metrics / pack |
| `export_t4m1.py` | checkpoint → エンジン読み込み可能な T4M1 v1 .bin。clamp≠0 を拒否(DEBTガード) |
| `gate_match.py` | ミラー複式の本ゲート。`--swap-stem t2_second --swap-path <new.bin>` で1スロット差し替え |
| `gate_report.py` / `run_gate_t2.sh` | ゲート集計と6シャード駆動(20,004ディール) |
| `assemble_labels.py` | 艦隊の position_*.json → labels.jsonl(重複offset検査つき) |

実行環境の前提: 訓練は WSL の `/home/wner/ofc-t0-cuda-venv/bin/python3`(torch+CUDA)。
コーパスは `~/ofc-m7/t2_2048/<seat>/features_rs`(labelgen_feature_dump の出力)。
昇格基準は本ゲートのみ(held-out regret は基準にしない —
docs/hu_m7_roadmap_20260818.md:158 の裁定)。
