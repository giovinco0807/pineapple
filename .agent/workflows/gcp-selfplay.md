---
description: GCPでself-playループを高速実行する手順
---

# GCP Self-Play セットアップ

## 1. GCPプロジェクト準備

Google Cloud Console (https://console.cloud.google.com) にアクセス。

```bash
# gcloud CLIがなければインストール
# https://cloud.google.com/sdk/docs/install

# ログイン
gcloud auth login

# プロジェクト設定（既存プロジェクトを使うか新規作成）
gcloud config set project YOUR_PROJECT_ID
```

## 2. VMインスタンス作成

```bash
# 推奨: n2-highcpu-32 (32 vCPU, 32GB RAM, ~$1/h)
# 高速版: n2-highcpu-64 (64 vCPU, 64GB RAM, ~$2/h)
gcloud compute instances create ofc-selfplay \
  --zone=us-central1-a \
  --machine-type=n2-highcpu-32 \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=50GB \
  --boot-disk-type=pd-ssd
```

## 3. SSHで接続

```bash
gcloud compute ssh ofc-selfplay --zone=us-central1-a
```

## 4. 環境セットアップ（VM上で実行）

```bash
# 基本パッケージ
sudo apt update && sudo apt install -y python3-pip python3-venv git cargo

# Python仮想環境
python3 -m venv ~/venv
source ~/venv/bin/activate

# PyTorch CPU + numpy
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy
```

## 5. コードをアップロード

ローカルPCから実行:
```bash
# プロジェクト全体をアップロード
gcloud compute scp --recurse --zone=us-central1-a \
  c:\Users\Owner\.gemini\antigravity\scratch\ofc-pineapple \
  ofc-selfplay:~/ofc-pineapple
```

## 6. Rust FL Solverをビルド（VM上）

```bash
cd ~/ofc-pineapple/ai/rust_solver
cargo build --release
# バイナリ: target/release/fl_solver
```

## 7. Self-Playループ実行（VM上）

```bash
source ~/venv/bin/activate
cd ~/ofc-pineapple

# screenで実行（SSH切断しても続行）
screen -S selfplay

# 30ワーカーで実行（n2-highcpu-32の場合）
python ai/run_selfplay_loop.py \
  --iterations 5 \
  --start-iter 7 \
  --games 500 \
  --rollouts 50 \
  --top-k 30 \
  --workers 30 \
  --epochs 100 \
  --eval-games 50 \
  --eval-rollouts 50 \
  --base-model ai/models/selfplay_iter6/bc_policy_best.pt \
  --include data/ryo_aug_t18.jsonl data/selfplay_v3_500.jsonl \
    data/selfplay_iter1.jsonl data/selfplay_iter2.jsonl \
    data/selfplay_iter3.jsonl data/selfplay_iter4.jsonl \
    data/selfplay_iter5.jsonl data/selfplay_iter6.jsonl

# screenから抜ける: Ctrl+A, D
# 再接続: screen -r selfplay
```

## 8. 進捗確認（VM上）

```bash
python check_stats.py
```

## 9. 結果をダウンロード（ローカルPC）

```bash
# モデルとデータをダウンロード
gcloud compute scp --recurse --zone=us-central1-a \
  ofc-selfplay:~/ofc-pineapple/ai/models/selfplay_iter* \
  c:\Users\Owner\.gemini\antigravity\scratch\ofc-pineapple\ai\models\

gcloud compute scp --recurse --zone=us-central1-a \
  ofc-selfplay:~/ofc-pineapple/data/selfplay_iter*.jsonl \
  c:\Users\Owner\.gemini\antigravity\scratch\ofc-pineapple\data\
```

## 10. VM停止（課金停止）

```bash
gcloud compute instances stop ofc-selfplay --zone=us-central1-a

# 完全削除（不要になったら）
gcloud compute instances delete ofc-selfplay --zone=us-central1-a
```

## コスト見積

| マシン | 速度 | 5iter所要時間 | コスト |
|--------|------|-------------|--------|
| n2-highcpu-32 | ~12 g/min | ~4時間 | ~$4 |
| n2-highcpu-64 | ~25 g/min | ~2時間 | ~$4 |

> **重要**: 使い終わったら必ず`stop`か`delete`すること！
