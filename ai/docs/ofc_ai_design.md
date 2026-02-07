# OFC Pineapple AI 設計書

## 1. 概要

### 1.1 目的
OFC Pineapple の通常プレイにおいて、人間レベル以上の強さを持つAIを構築する。

### 1.2 アプローチ
```
Phase B: Behavior Cloning（RYOのプレイを模倣）
    ↓
Phase C: Self-Play + MCTS（自己対戦で人間を超える）
    ↓
Phase D: 対人戦（RYOと対戦、フィードバックループ）
```

### 1.3 AIが学ぶ2つの関数

| 名前 | 役割 | 入力 | 出力 |
|------|------|------|------|
| Policy Network | 「どう置くか」を決める | 盤面状態 | 各行動の確率分布 |
| Value Network | 「今の盤面はどれくらい有利か」を評価 | 盤面状態 | 期待スコア・各種確率 |

この2つを段階的に鍛えていく。

---

## 2. 状態表現（AIが見る世界）

### 2.1 AIが各ターンで知っている情報

```python
class Observation:
    # ===== 盤面情報 =====
    board_self: Board              # 自分のボード (top: 最大3枚, middle: 最大5枚, bottom: 最大5枚)
    board_opponent: Board          # 相手のボード (公開情報、通常プレイでは全て見える)
    
    # ===== 手札 =====
    dealt_cards: list[str]         # 今ターン配られたカード
                                   #   ターン0: 5枚
                                   #   ターン1-8: 3枚
    
    # ===== 記憶 =====
    known_discards_self: list[str] # 自分が過去に捨てたカード（全て覚えている）
    # ※ 相手の捨て札は見えない
    #    → 「見えていないカード」として間接的に推定
    
    # ===== メタ情報 =====
    turn: int                      # 現在のターン (0-8)
    is_btn: bool                   # 自分がBTN（先手）かどうか
    chips_self: int                # 自分の持ち点 (0-400)
    chips_opponent: int            # 相手の持ち点 (0-400)
    
    # ===== 派生情報（自動計算）=====
    unseen_cards: list[str]        # まだ見えていないカード
                                   # = 全54枚 - 両者の盤面 - 手札 - 自分の捨て札
    cards_remaining: int           # デッキ残り枚数
```

### 2.2 カードのエンコーディング（ベクトル化）

ニューラルネットワークは数値しか扱えないので、盤面をベクトル（数列）に変換する必要がある。

**カード一覧: 54枚**
```
通常カード 52枚:
  2h, 3h, 4h, ..., Ah  (ハート13枚)
  2d, 3d, 4d, ..., Ad  (ダイヤ13枚)
  2c, 3c, 4c, ..., Ac  (クラブ13枚)
  2s, 3s, 4s, ..., As  (スペード13枚)

ジョーカー 2枚:
  X1, X2
```

**各カードの状態: 9次元のone-hotベクトル**

1枚のカードは必ず以下の9箇所のどれかにある:

| インデックス | 意味 | 説明 |
|-------------|------|------|
| 0 | my_top | 自分のTopに配置済み |
| 1 | my_mid | 自分のMiddleに配置済み |
| 2 | my_bot | 自分のBottomに配置済み |
| 3 | opp_top | 相手のTopに配置済み |
| 4 | opp_mid | 相手のMiddleに配置済み |
| 5 | opp_bot | 相手のBottomに配置済み |
| 6 | in_hand | 今の手札にある |
| 7 | my_discard | 自分が過去に捨てた |
| 8 | unseen | どこにあるか分からない（デッキ or 相手の捨て札） |

**例: Ahが自分のTopにある場合**
```
Ah: [1, 0, 0, 0, 0, 0, 0, 0, 0]
     ↑my_top
```

**例: X1が手札にある場合**
```
X1: [0, 0, 0, 0, 0, 0, 1, 0, 0]
                        ↑in_hand
```

**例: 3cがどこにもない（デッキか相手の捨て札）**
```
3c: [0, 0, 0, 0, 0, 0, 0, 0, 1]
                              ↑unseen
```

### 2.3 状態ベクトルの全体構成

```
カード情報:  54枚 × 9次元 = 486次元
メタ情報:    4次元
───────────────────────
合計:        490次元
```

```python
def encode_state(obs: Observation) -> np.ndarray:
    """盤面を490次元のベクトルに変換"""
    
    # === カード情報: 54×9 の行列 ===
    card_matrix = np.zeros((54, 9))
    
    # 自分のボード
    for card in obs.board_self.top:
        card_matrix[card_to_idx(card)][0] = 1      # my_top
    for card in obs.board_self.middle:
        card_matrix[card_to_idx(card)][1] = 1      # my_mid
    for card in obs.board_self.bottom:
        card_matrix[card_to_idx(card)][2] = 1      # my_bot
    
    # 相手のボード
    for card in obs.board_opponent.top:
        card_matrix[card_to_idx(card)][3] = 1      # opp_top
    for card in obs.board_opponent.middle:
        card_matrix[card_to_idx(card)][4] = 1      # opp_mid
    for card in obs.board_opponent.bottom:
        card_matrix[card_to_idx(card)][5] = 1      # opp_bot
    
    # 手札
    for card in obs.dealt_cards:
        card_matrix[card_to_idx(card)][6] = 1      # in_hand
    
    # 自分の捨て札
    for card in obs.known_discards_self:
        card_matrix[card_to_idx(card)][7] = 1      # my_discard
    
    # 見えていないカード（残り全部）
    for card in obs.unseen_cards:
        card_matrix[card_to_idx(card)][8] = 1      # unseen
    
    # === メタ情報: 4次元 ===
    meta = np.array([
        obs.turn / 8.0,                  # ターン（0.0〜1.0に正規化）
        float(obs.is_btn),               # BTNなら1.0、BBなら0.0
        obs.chips_self / 200.0,          # 持ち点（正規化）
        obs.chips_opponent / 200.0,
    ])
    
    # === 結合: 486 + 4 = 490次元 ===
    return np.concatenate([card_matrix.flatten(), meta])

def card_to_idx(card: str) -> int:
    """カード文字列をインデックス(0-53)に変換"""
    if card == "X1": return 52
    if card == "X2": return 53
    
    ranks = "23456789TJQKA"
    suits = "hdcs"
    rank_idx = ranks.index(card[0])
    suit_idx = suits.index(card[1])
    return suit_idx * 13 + rank_idx
    # 0-12: ハート, 13-25: ダイヤ, 26-38: クラブ, 39-51: スペード
```

### 2.4 なぜこのエンコーディングなのか

**代替案1: 画像として表現（CNN）**
- ボードを画像のように2D配列にする
- メリット: 位置関係を学びやすい
- デメリット: カードゲームには不要な空間構造を持ち込む

**代替案2: カード列として表現（Transformer）**
- 各カードをトークンとして扱う
- メリット: カード間の関係性を柔軟に学べる
- デメリット: 学習データが大量に必要

**採用案: フラットベクトル（MLP）**
- シンプルで学習が早い
- OFCは盤面が小さい（最大26枚＝13枚×2人）ので十分
- Behavior Cloningに必要なデータ量が少なくて済む

---

## 3. 行動空間（AIが選べる手）

### 3.1 ターン0（初手: 5枚全配置）

5枚をTop(最大3)/Middle(最大5)/Bottom(最大5)に分配する全パターン。

```
配分パターン例（5枚を3箇所に分ける）:
  Top=0, Mid=0, Bot=5
  Top=0, Mid=1, Bot=4
  Top=0, Mid=2, Bot=3
  Top=0, Mid=3, Bot=2
  Top=0, Mid=4, Bot=1
  Top=0, Mid=5, Bot=0  ← Midは最大5なのでOK
  Top=1, Mid=0, Bot=4
  Top=1, Mid=1, Bot=3
  ...
  Top=3, Mid=2, Bot=0
  
配分パターン: 21通り
× 各パターン内の並べ方（どのカードをどこに置くか）
→ 初手の行動数: 数十〜数百通り
```

```python
def get_initial_actions(dealt_cards: list[str], board: Board) -> list[Action]:
    """初手5枚の全配置パターンを列挙"""
    actions = []
    cards = dealt_cards  # 5枚
    
    # Top: 0-3枚, Middle: 0-5枚, Bottom: 0-5枚 で合計5枚
    for top_count in range(min(4, 6)):  # Top最大3枚
        for mid_count in range(min(6, 6 - top_count)):
            bot_count = 5 - top_count - mid_count
            if bot_count > 5 or bot_count < 0:
                continue
            if top_count > 3:
                continue
            
            # この配分でのカードの割り当て全パターン
            for perm in itertools.permutations(cards):
                top_cards = list(perm[:top_count])
                mid_cards = list(perm[top_count:top_count+mid_count])
                bot_cards = list(perm[top_count+mid_count:])
                
                placements = (
                    [(c, "top") for c in top_cards] +
                    [(c, "middle") for c in mid_cards] +
                    [(c, "bottom") for c in bot_cards]
                )
                actions.append(Action(placements=placements, discard=None))
    
    # 重複排除（同じ位置内での順序は関係ない）
    return deduplicate(actions)
```

### 3.2 ターン1-8（3枚から2枚配置 + 1枚捨て）

```
手札3枚: C0, C1, C2

Step 1: 捨てる1枚を選ぶ → 3通り
Step 2: 残り2枚の配置先を選ぶ → 各3箇所(Top/Mid/Bot) × 2枚

組み合わせ:
  捨てC0: (C1→T, C2→T), (C1→T, C2→M), (C1→T, C2→B),
           (C1→M, C2→T), (C1→M, C2→M), (C1→M, C2→B),
           (C1→B, C2→T), (C1→B, C2→M), (C1→B, C2→B)  → 9通り
  捨てC1: 同様に9通り
  捨てC2: 同様に9通り

最大: 3 × 9 = 27通り
実際: 空き枠の制約で減る（Topが3枚埋まっていれば選べない等）
```

```python
def get_turn_actions(dealt_cards: list[str], board: Board) -> list[Action]:
    """通常ターンの全行動を列挙"""
    actions = []
    positions = ["top", "middle", "bottom"]
    limits = {"top": 3, "middle": 5, "bottom": 5}
    
    for discard_idx in range(3):
        discard = dealt_cards[discard_idx]
        remaining = [dealt_cards[i] for i in range(3) if i != discard_idx]
        
        for pos0 in positions:
            for pos1 in positions:
                # 空き枠チェック
                temp_counts = {
                    "top": len(board.top),
                    "middle": len(board.middle),
                    "bottom": len(board.bottom),
                }
                temp_counts[pos0] += 1
                if temp_counts[pos0] > limits[pos0]:
                    continue
                temp_counts[pos1] += 1
                if temp_counts[pos1] > limits[pos1]:
                    continue
                
                actions.append(Action(
                    placements=[(remaining[0], pos0), (remaining[1], pos1)],
                    discard=discard,
                ))
    
    return actions
```

### 3.3 行動のエンコーディング

Policy Networkの出力層は「各行動の確率」を返す。行動数が可変なので、マスキングで対応する。

```python
MAX_ACTIONS = 150  # 初手の最大行動数に合わせる（余裕を持って）

def encode_action(action: Action, valid_actions: list[Action]) -> int:
    """行動をインデックスに変換"""
    return valid_actions.index(action)

def create_action_mask(valid_actions: list[Action]) -> np.ndarray:
    """有効な行動だけ1、無効は0のマスクを作成"""
    mask = np.zeros(MAX_ACTIONS, dtype=bool)
    for i in range(len(valid_actions)):
        mask[i] = True
    return mask
```

---

## 4. Policy Network（方策ネットワーク）

### 4.1 役割

「この盤面で、どの行動が最も良いか」を確率分布として出力する。

### 4.2 アーキテクチャ

```
入力: 490次元（状態ベクトル）
  │
  ▼
Linear(490 → 512) + ReLU + Dropout(0.2)
  │
  │  ここで盤面の特徴を抽出
  │  「同スート4枚 → フラッシュ狙い」
  │  「相手TopにQQ → FL警戒」
  │
  ▼
Linear(512 → 256) + ReLU + Dropout(0.2)
  │
  │  より抽象的な戦略判断
  │  「バストリスクが高い → 安全策」
  │  「大差で勝ってる → 守る」
  │
  ▼
Linear(256 → MAX_ACTIONS)
  │
  │  各行動のスコア（logits）
  │
  ▼
Action Masking（無効行動を -inf に）
  │
  ▼
Softmax
  │
  ▼
出力: 各行動の確率分布
  例: [0.45, 0.25, 0.15, 0.08, 0.05, 0.02, ...]
       行動0  行動1  行動2  行動3  ...
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim=490, hidden1=512, hidden2=256, max_actions=150):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden2, max_actions),
        )
    
    def forward(self, state: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (batch, 490) 状態ベクトル
            valid_mask: (batch, MAX_ACTIONS) 有効行動マスク (True/False)
        Returns:
            action_probs: (batch, MAX_ACTIONS) 行動確率分布
        """
        logits = self.net(state)
        
        # 無効行動を -inf にして選ばれないようにする
        logits[~valid_mask] = float('-inf')
        
        # Softmaxで確率に変換
        action_probs = F.softmax(logits, dim=-1)
        return action_probs
    
    def select_action(self, state, valid_mask, temperature=1.0):
        """行動を選択する"""
        probs = self.forward(state, valid_mask)
        
        if temperature == 0:
            # 最善手を選択（実戦用）
            return torch.argmax(probs, dim=-1)
        else:
            # 確率的にサンプリング（探索用）
            # temperatureが低い → 最善手寄り
            # temperatureが高い → ランダム寄り
            return torch.multinomial(probs, 1).squeeze(-1)
```

### 4.3 Action Maskingの重要性

OFCでは盤面によって取れる行動が変わる。例えばTopが3枚埋まっていたらTopには置けない。

```
マスクなし: [0.30, 0.25, 0.20, 0.15, 0.10]  ← 行動2がTop配置で無効
マスクあり: [0.38, 0.31, 0.00, 0.19, 0.12]  ← 行動2が0になり、残りで再配分
```

これにより「ルール違反の手を選ぶ」ことが物理的に不可能になる。

---

## 5. Value Network（価値ネットワーク）

### 5.1 役割

「今の盤面がどれくらい有利か」を数値で評価する。MCTSで先読みする際に、末端の盤面を評価するのに使う。

### 5.2 2層構造

```
Layer 1（Behavior Cloning段階で学習）:
  ├── royalty_ev:  このハンドで最終的に得られるロイヤリティの期待値
  ├── bust_prob:   バストする確率 (0.0〜1.0)
  └── fl_prob:     FLにエントリーできる確率 (0.0〜1.0)

Layer 2（Self-Play段階で学習）:
  └── value:       最終的なチップ変動の期待値（対戦相手との相対評価込み）
```

**なぜ2層にするか:**
- Layer 1 は自分のボードだけ見れば推定できる → 少ないデータで学べる
- Layer 2 は相手との相対評価が必要 → 大量の対戦データが必要
- BCの段階ではLayer 1だけ、Self-PlayでLayer 2を追加

### 5.3 アーキテクチャ

```
入力: 490次元（状態ベクトル、Policy Networkと同じ）
  │
  ▼
Linear(490 → 512) + ReLU
  │
  ▼
Linear(512 → 256) + ReLU
  │
  ├──────────────────────────────────┐
  │                                  │
  ▼                                  ▼
Layer 1 ヘッド群                   Layer 2 ヘッド
  │                                  │
  ├── royalty_head → Linear(256→1)   └── value_head → Linear(256→1)
  │   → ロイヤリティ期待値               → チップ変動期待値
  │
  ├── bust_head → Linear(256→1) + Sigmoid
  │   → バスト確率 (0〜1)
  │
  └── fl_head → Linear(256→1) + Sigmoid
      → FL確率 (0〜1)
```

```python
class ValueNetwork(nn.Module):
    def __init__(self, input_dim=490, hidden1=512, hidden2=256):
        super().__init__()
        
        # 共通の特徴抽出部分
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
        )
        
        # Layer 1: 自己評価ヘッド（BC段階で学習）
        self.royalty_head = nn.Linear(hidden2, 1)   # ロイヤリティ期待値
        self.bust_head = nn.Linear(hidden2, 1)       # バスト確率
        self.fl_head = nn.Linear(hidden2, 1)         # FLエントリー確率
        
        # Layer 2: 対戦評価ヘッド（Self-Play段階で学習）
        self.value_head = nn.Linear(hidden2, 1)      # 最終スコア期待値
    
    def forward(self, state: torch.Tensor) -> dict:
        x = self.shared(state)
        return {
            "royalty_ev": self.royalty_head(x),                      # 実数
            "bust_prob": torch.sigmoid(self.bust_head(x)),           # 0〜1
            "fl_prob": torch.sigmoid(self.fl_head(x)),               # 0〜1
            "value": self.value_head(x),                             # 実数（正=有利, 負=不利）
        }
```

### 5.4 各ヘッドの使われ方

| ヘッド | 学習段階 | 教師データ | 使われる場面 |
|--------|----------|------------|-------------|
| royalty_ev | BC | ハンド終了時の実際のロイヤリティ | 配置判断の参考情報 |
| bust_prob | BC | バストしたかどうか (0/1) | 安全策 vs 攻撃策の判断 |
| fl_prob | BC | FL入りしたかどうか (0/1) | FL狙いの価値判断 |
| value | Self-Play | 対戦の最終チップ変動 | MCTSの盤面評価 |

---

## 6. Phase B: Behavior Cloning（模倣学習）

### 6.1 概要

RYOのプレイデータを「問題集」として使い、Policy NetworkとValue Network (Layer 1)を教師あり学習で鍛える。

```
入力: ターン開始時の盤面状態
正解（Policy）: RYOが実際に選んだ行動
正解（Value）:  ハンド終了時のロイヤリティ、バスト有無、FL入り有無
```

### 6.2 データの準備

WebアプリのSQLiteログから学習データを生成する。

```
SQLite (turns テーブル)
    │
    ▼
JSONL エクスポート
    │  1ターン = 1行
    │  {board_self, board_opponent, dealt_cards, action, ...}
    │
    ▼
前処理スクリプト (preprocess.py)
    │
    ├── states.npy        # (N, 490) 状態ベクトル
    ├── actions.npy       # (N,) 行動インデックス  
    ├── valid_masks.npy   # (N, MAX_ACTIONS) 有効行動マスク
    ├── royalties.npy     # (N,) ハンド終了時のロイヤリティ
    ├── busted.npy        # (N,) バストしたか (0/1)
    ├── fl_entry.npy      # (N,) FL入りしたか (0/1)
    └── metadata.json     # 統計情報
```

```python
def preprocess_turn(turn_log: dict, hand_result: dict) -> dict:
    """1ターンのログを学習データに変換"""
    
    # 状態のエンコード
    obs = build_observation(turn_log)
    state = encode_state(obs)
    
    # 有効行動の列挙
    if turn_log["turn"] == 0:
        valid_actions = get_initial_actions(obs.dealt_cards, obs.board_self)
    else:
        valid_actions = get_turn_actions(obs.dealt_cards, obs.board_self)
    
    # RYOの行動をインデックスに変換
    ryo_action = Action(
        placements=turn_log["action"]["placements"],
        discard=turn_log["action"].get("discard"),
    )
    action_idx = valid_actions.index(ryo_action)
    
    # マスク作成
    valid_mask = create_action_mask(valid_actions)
    
    # 教師データ（ハンド終了時の結果から）
    player = turn_log["player"]
    royalty = hand_result["royalties"][player]["total"]
    busted = hand_result["busted"][player]
    fl_entry = hand_result["fl_entry"][player]
    
    return {
        "state": state,           # (490,)
        "action_idx": action_idx, # int
        "valid_mask": valid_mask,  # (MAX_ACTIONS,)
        "royalty": royalty,        # float
        "busted": busted,         # 0 or 1
        "fl_entry": fl_entry,     # 0 or 1
    }
```

### 6.3 学習ループ

```python
def train_behavior_cloning(dataset, num_epochs=100, lr=1e-3, batch_size=64):
    policy_net = PolicyNetwork()
    value_net = ValueNetwork()
    
    policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    value_optimizer = torch.optim.Adam(value_net.parameters(), lr=lr)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(num_epochs):
        policy_losses = []
        value_losses = []
        
        for batch in dataloader:
            states = batch["state"]           # (B, 490)
            action_idx = batch["action_idx"]  # (B,)
            valid_mask = batch["valid_mask"]   # (B, MAX_ACTIONS)
            royalty = batch["royalty"]          # (B,)
            busted = batch["busted"]           # (B,)
            fl_entry = batch["fl_entry"]       # (B,)
            
            # === Policy Network 学習 ===
            pred_probs = policy_net(states, valid_mask)  # (B, MAX_ACTIONS)
            policy_loss = F.cross_entropy(
                torch.log(pred_probs + 1e-8),
                action_idx
            )
            
            policy_optimizer.zero_grad()
            policy_loss.backward()
            policy_optimizer.step()
            policy_losses.append(policy_loss.item())
            
            # === Value Network 学習 ===
            pred = value_net(states)
            value_loss = (
                F.mse_loss(pred["royalty_ev"].squeeze(), royalty)
                + F.binary_cross_entropy(pred["bust_prob"].squeeze(), busted.float())
                + F.binary_cross_entropy(pred["fl_prob"].squeeze(), fl_entry.float())
            )
            
            value_optimizer.zero_grad()
            value_loss.backward()
            value_optimizer.step()
            value_losses.append(value_loss.item())
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: "
                  f"Policy Loss={np.mean(policy_losses):.4f}, "
                  f"Value Loss={np.mean(value_losses):.4f}")
    
    return policy_net, value_net
```

### 6.4 評価指標

| 指標 | 意味 | 目標 |
|------|------|------|
| Top-1 Action Accuracy | RYOと同じ行動を選ぶ確率 | > 40% |
| Top-3 Action Accuracy | RYOの行動が確率上位3つに入る確率 | > 70% |
| Royalty MAE | ロイヤリティ予測の平均誤差 | < 3.0 |
| Bust AUC | バスト予測のAUC-ROC | > 0.85 |
| FL Entry AUC | FL予測のAUC-ROC | > 0.80 |

```python
def evaluate_bc(policy_net, value_net, test_dataset):
    """Behavior Cloningの評価"""
    correct_top1 = 0
    correct_top3 = 0
    total = 0
    
    for sample in test_dataset:
        probs = policy_net(sample["state"], sample["valid_mask"])
        predicted = torch.argsort(probs, descending=True)
        actual = sample["action_idx"]
        
        if predicted[0] == actual:
            correct_top1 += 1
        if actual in predicted[:3]:
            correct_top3 += 1
        total += 1
    
    print(f"Top-1 Accuracy: {correct_top1/total:.2%}")
    print(f"Top-3 Accuracy: {correct_top3/total:.2%}")
```

### 6.5 必要データ量

| データ量 | ハンド数 | ターン数(概算) | プレイ時間(概算) | 期待される精度 |
|---------|---------|--------------|----------------|--------------|
| 最小 | 500 | ~4,500 | ~42時間 | 基本パターンを学習 |
| 推奨 | 2,000 | ~18,000 | ~170時間 | 安定した模倣 |
| 理想 | 5,000 | ~45,000 | ~420時間 | 高精度な模倣 |

※ 1ハンド ≈ 9ターン × 2プレイヤー = 18ターンレコード
※ 1ハンド ≈ 5分（考慮時間含む）

### 6.6 BCの限界

- **天井がある**: どれだけ学習してもRYOのプレイレベルが上限
- **未経験局面に弱い**: RYOが遭遇しなかった状況では判断が不安定
- **局所最適**: RYOの「癖」も学んでしまう
- → だから次のSelf-Playが必要

---

## 7. Phase C: Self-Play + MCTS（自己対戦強化学習）

### 7.1 概要

BCで「そこそこ打てるAI」ができたら、そのAI同士を戦わせてさらに強くする。囲碁のAlphaGoと同じアプローチ。

### 7.2 MCTS（モンテカルロ木探索）とは

「先を読む」アルゴリズム。各ターンで、複数の未来をシミュレーションし、最も良い結果が期待できる行動を選ぶ。

**4つのステップを繰り返す:**

```
Step 1: 選択 (Select)
  木のルートから、UCB1スコアが最大の子ノードを辿っていく
  
Step 2: 展開 (Expand)
  葉ノードに達したら、新しい子ノードを1つ追加
  
Step 3: 評価 (Evaluate)
  新しいノードをValue Networkで評価
  
Step 4: 逆伝播 (Backpropagate)
  評価結果を親ノードに遡って更新

これを N 回繰り返し（例: 200回）、
最も訪問回数が多い行動を選択する。
```

**図解:**

```
                   現在の盤面 (Root)
                   訪問数: 200
                  /        |         \
          行動A           行動B          行動C
          訪問120         訪問55         訪問25
          平均+12.3       平均+8.7       平均+15.1
         /    \          /    \
     相手a   相手b    相手a   相手b
     +14     +10      +6     +11
      |
    次ターン
    行動X...

→ 行動A を選択（訪問数120が最多）
  ※ 行動Cの平均値が高いが訪問数が少ない
  ※ MCTSが探索を続ければ行動Cの訪問数も増えるかもしれない
```

### 7.3 UCB1: 探索と利用のバランス

どの子ノードを選ぶかを決める式:

```
UCB1(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))

Q(s, a)    = 行動aの平均評価値（これまでの探索結果）
P(s, a)    = Policy Networkが出した行動aの事前確率
N(s)       = 親ノードの訪問回数
N(s, a)    = 子ノードの訪問回数
c_puct     = 探索の度合い（ハイパーパラメータ、通常1.0〜2.0）
```

**直感的な意味:**
- **Q(s,a)** が高い → 過去の探索で良い結果が出た行動を選びやすい（利用）
- **P(s,a)** が高い → Policy Netが有望と判断した行動を選びやすい（事前知識）
- **N(s,a)** が少ない → まだ探索していない行動を選びやすい（探索）

Policy Networkが賢いほど、有望な手に探索リソースが集中し、効率が上がる。

### 7.4 MCTS実装

```python
class MCTSNode:
    def __init__(self, state, parent=None, action=None, prior=0.0):
        self.state = state              # Observation
        self.parent = parent
        self.action = action            # この位置に来るために取った行動
        self.prior = prior              # P(s,a): Policy Netの事前確率
        
        self.children = {}              # {action_idx: MCTSNode}
        self.visit_count = 0            # N(s,a)
        self.value_sum = 0.0            # W(s,a): 価値の合計
    
    @property
    def q_value(self):
        """Q(s,a) = W(s,a) / N(s,a)"""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def ucb_score(self, parent_visits, c_puct=1.5):
        """UCB1スコアを計算"""
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return self.q_value + exploration


class MCTS:
    def __init__(self, policy_net, value_net, game_engine,
                 num_simulations=200, c_puct=1.5):
        self.policy_net = policy_net
        self.value_net = value_net
        self.game_engine = game_engine
        self.num_simulations = num_simulations
        self.c_puct = c_puct
    
    def search(self, root_state: Observation) -> tuple[int, dict]:
        """
        MCTSを実行して最善行動とその分布を返す
        
        Returns:
            best_action_idx: 最善行動のインデックス
            action_probs: {action_idx: 確率} の辞書（学習用）
        """
        root = MCTSNode(state=root_state)
        
        # Policy Netで事前確率を計算
        self._expand(root)
        
        # num_simulations回のシミュレーション
        for _ in range(self.num_simulations):
            node = self._select(root)       # Step 1: 選択
            value = self._evaluate(node)     # Step 2+3: 展開+評価
            self._backpropagate(node, value) # Step 4: 逆伝播
        
        # 訪問回数に基づいて行動を選択
        action_visits = {
            action_idx: child.visit_count
            for action_idx, child in root.children.items()
        }
        total_visits = sum(action_visits.values())
        action_probs = {
            idx: count / total_visits
            for idx, count in action_visits.items()
        }
        
        best_action_idx = max(action_visits, key=action_visits.get)
        return best_action_idx, action_probs
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """UCB1スコアが最大の葉ノードまで辿る"""
        while node.children:
            node = max(
                node.children.values(),
                key=lambda child: child.ucb_score(node.visit_count, self.c_puct)
            )
        return node
    
    def _expand(self, node: MCTSNode):
        """葉ノードを展開: 全有効行動について子ノードを作成"""
        state_vec = encode_state(node.state)
        state_tensor = torch.FloatTensor(state_vec).unsqueeze(0)
        
        valid_actions = self.game_engine.get_valid_actions(node.state)
        valid_mask = create_action_mask(valid_actions)
        mask_tensor = torch.BoolTensor(valid_mask).unsqueeze(0)
        
        # Policy Netで事前確率を取得
        with torch.no_grad():
            priors = self.policy_net(state_tensor, mask_tensor).squeeze(0)
        
        for i, action in enumerate(valid_actions):
            if priors[i] > 0:
                child_state = self.game_engine.simulate_action(node.state, action)
                child = MCTSNode(
                    state=child_state,
                    parent=node,
                    action=i,
                    prior=priors[i].item(),
                )
                node.children[i] = child
    
    def _evaluate(self, node: MCTSNode) -> float:
        """ノードを評価: Value Networkで盤面の価値を推定"""
        if self.game_engine.is_hand_finished(node.state):
            # 終端ノード: 実際のスコアを計算
            return self.game_engine.compute_score(node.state)
        
        # 非終端ノード: Value Netで推定
        if not node.children:
            self._expand(node)
        
        state_vec = encode_state(node.state)
        state_tensor = torch.FloatTensor(state_vec).unsqueeze(0)
        
        with torch.no_grad():
            prediction = self.value_net(state_tensor)
        
        return prediction["value"].item()
    
    def _backpropagate(self, node: MCTSNode, value: float):
        """評価値を親ノードに遡って伝播"""
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            # 相手の手番では符号を反転
            value = -value
            node = node.parent
```

### 7.5 OFC特有のMCTSの課題と対策

**課題1: 不完全情報（相手の手札・捨て札が見えない）**

通常のMCTS（囲碁など）は完全情報ゲーム向け。OFCでは相手の手を読む必要がある。

```
対策: Information Set MCTS (IS-MCTS)
  
  各シミュレーションの冒頭で:
  1. unseen_cards からランダムに相手の捨て札を推定
  2. 残りからデッキを再構成
  3. その「仮定した世界」でシミュレーションを実行
  
  多数のシミュレーションを重ねることで、
  様々な可能性を平均的に考慮できる。
```

```python
def _select_with_determinization(self, root_state):
    """不完全情報を確定化してからシミュレーション"""
    # 見えていないカードをシャッフル
    unseen = list(root_state.unseen_cards)
    random.shuffle(unseen)
    
    # 相手の捨て札を推定割り当て
    opponent_discards_count = estimate_opponent_discards(root_state)
    estimated_opp_discards = unseen[:opponent_discards_count]
    remaining_deck = unseen[opponent_discards_count:]
    
    # この「世界」でシミュレーション
    determinized_state = root_state.copy()
    determinized_state.deck = remaining_deck
    determinized_state.opponent_discards = estimated_opp_discards
    
    return determinized_state
```

**課題2: 初手の行動空間が大きい**

5枚を3箇所に分配するパターンが多い。

```
対策: Progressive Widening
  
  最初は Policy Net の上位 K 個の行動だけ展開。
  訪問回数が増えるにつれて K を拡大。
  
  K = ceil(c * N^alpha)
    N: 親の訪問回数
    c, alpha: ハイパーパラメータ
```

**課題3: 相手の行動の扱い**

自分のターンではMCTSで探索するが、相手のターンはどうするか。

```
対策: 相手もPolicy Netで行動
  
  自分のターン: MCTSで深く探索
  相手のターン: Policy Net（相手モデル）で1手選択
  
  初期は自分と同じPolicy Netを使用。
  将来的にはRYOの特徴を学んだ専用モデルも可能。
```

### 7.6 Self-Playループ

```python
def self_play_iteration(policy_net, value_net, num_games=200, num_simulations=200):
    """Self-Play 1イテレーション"""
    
    game_engine = GameEngine()
    mcts = MCTS(policy_net, value_net, game_engine, num_simulations)
    
    all_trajectories = []
    
    for game_idx in range(num_games):
        # ゲーム初期化
        session = game_engine.new_session()
        trajectories = {0: [], 1: []}  # 各プレイヤーの軌跡
        
        while not session.is_finished():
            hand = session.new_hand()
            
            while not hand.is_finished():
                player = hand.current_player
                obs = hand.get_observation(player)
                
                # MCTSで行動選択
                action_idx, action_probs = mcts.search(obs)
                
                # 記録
                trajectories[player].append({
                    "state": encode_state(obs),
                    "action_probs": action_probs,   # MCTSの行動分布
                    "action_idx": action_idx,
                })
                
                # 行動を適用
                valid_actions = game_engine.get_valid_actions(obs)
                hand.apply_action(valid_actions[action_idx])
            
            # ハンド結果をチップに反映
            hand_result = hand.compute_result()
            session.apply_result(hand_result)
        
        # 最終スコアを各ステップに付与
        final_chips = session.get_chips()
        for player in [0, 1]:
            reward = (final_chips[player] - 200) / 200.0  # -1.0〜1.0 に正規化
            for step in trajectories[player]:
                step["reward"] = reward
        
        all_trajectories.extend(trajectories[0])
        all_trajectories.extend(trajectories[1])
    
    return all_trajectories


def train_from_self_play(policy_net, value_net, trajectories, lr=1e-4):
    """Self-Playのデータからネットワークを更新"""
    
    policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    value_optimizer = torch.optim.Adam(value_net.parameters(), lr=lr)
    
    for step in trajectories:
        state = torch.FloatTensor(step["state"]).unsqueeze(0)
        
        # === Policy 更新 ===
        # MCTSの行動分布を「より正確な正解」として学習
        # (BCではRYOの行動が正解、ここではMCTSの分布が正解)
        target_probs = torch.zeros(MAX_ACTIONS)
        for action_idx, prob in step["action_probs"].items():
            target_probs[action_idx] = prob
        
        valid_mask = torch.BoolTensor(create_action_mask_from_probs(step["action_probs"]))
        pred_probs = policy_net(state, valid_mask.unsqueeze(0)).squeeze(0)
        
        # KLダイバージェンス: 予測分布とMCTS分布のズレを最小化
        policy_loss = F.kl_div(
            torch.log(pred_probs + 1e-8),
            target_probs,
            reduction="batchmean"
        )
        
        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()
        
        # === Value 更新 ===
        pred = value_net(state)
        value_loss = F.mse_loss(
            pred["value"].squeeze(),
            torch.tensor(step["reward"])
        )
        
        value_optimizer.zero_grad()
        value_loss.backward()
        value_optimizer.step()


def run_self_play_training(policy_net, value_net, num_iterations=100):
    """Self-Play訓練メインループ"""
    
    for iteration in range(num_iterations):
        print(f"=== Iteration {iteration + 1}/{num_iterations} ===")
        
        # 1. Self-Playでデータ生成
        trajectories = self_play_iteration(
            policy_net, value_net,
            num_games=200,
            num_simulations=200,
        )
        print(f"  Generated {len(trajectories)} training steps")
        
        # 2. ネットワーク更新
        train_from_self_play(policy_net, value_net, trajectories)
        
        # 3. 評価（前のバージョンと対戦）
        if iteration % 10 == 0:
            win_rate = evaluate_vs_previous(policy_net, value_net)
            print(f"  Win rate vs previous: {win_rate:.1%}")
        
        # 4. チェックポイント保存
        if iteration % 20 == 0:
            save_checkpoint(policy_net, value_net, iteration)
```

### 7.7 Self-Playの設定

| パラメータ | 訓練時 | 実戦時 |
|-----------|--------|--------|
| MCTSシミュレーション数 | 200 | 500 |
| 1ターンあたりの時間 | ~0.5秒 | ~15秒 |
| temperature | 1.0（探索促進） | 0.1（ほぼ最善手） |
| c_puct | 1.5 | 1.0 |
| ゲーム数/イテレーション | 200 | - |
| 推定イテレーション数 | 50-100 | - |

---

## 8. 報酬設計

### 8.1 ハンド報酬

```python
from dataclasses import dataclass

@dataclass
class RewardConfig:
    """報酬の各係数（チューニング用）"""
    bust_penalty: float = -20.0      # バストペナルティ（-10〜-30で調整）
    fl_bonus: float = 15.0           # FLエントリーボーナス
    hand_weight: float = 0.7         # ハンド報酬の重み (α)
    session_weight: float = 0.3      # セッション報酬の重み (β)

# デフォルト設定
REWARD_CONFIG = RewardConfig()

def compute_hand_reward(hand_result: dict, player: int,
                        config: RewardConfig = REWARD_CONFIG) -> float:
    """1ハンド終了時の報酬を計算"""
    
    # ベース: チップ変動（最も重要なシグナル）
    chip_change = hand_result["actual_score"][player]
    
    # FLエントリーボーナス（FLの価値は非常に高い）
    fl_bonus = 0.0
    if hand_result["fl_entry"][player]:
        fl_bonus = config.fl_bonus
    
    # バスト ペナルティ（強いペナルティで回避を学習させる）
    bust_penalty = 0.0
    if hand_result["busted"][player]:
        bust_penalty = config.bust_penalty
    
    return chip_change + fl_bonus + bust_penalty
```

### 8.2 セッション報酬（Self-Play用）

```python
def compute_session_reward(final_chips: list[int], player: int) -> float:
    """セッション全体の報酬（-1.0〜1.0に正規化）"""
    return (final_chips[player] - 200) / 200.0
```

### 8.3 なぜハンド報酬とセッション報酬の両方が必要か

```
ハンド報酬のみ:
  ✓ 各ハンドの良い打ち方を学ぶ
  ✗ チップ状況に応じた戦略変更を学べない
     例: 大差で勝っている → 守り重視にすべき

セッション報酬のみ:
  ✓ 勝敗に直結した学習
  ✗ シグナルが疎（1セッション= 多数ハンド後にやっと報酬が来る）
  ✗ 学習が非常に遅い

両方を組み合わせ:
  reward = α * hand_reward + β * session_reward
  α = 0.7, β = 0.3  （ハンド報酬を重視しつつ、勝敗も考慮）
```

---

## 9. Phase D: 対人戦

### 9.1 AIプレイヤーの接続

AIをWebSocketクライアントとして実装し、人間と同じインターフェースで対戦する。

```python
import asyncio
import websockets
import json

class AIPlayer:
    def __init__(self, policy_net, value_net, use_mcts=True, mcts_simulations=500):
        self.policy_net = policy_net
        self.value_net = value_net
        self.use_mcts = use_mcts
        
        if use_mcts:
            self.mcts = MCTS(
                policy_net, value_net,
                game_engine=GameEngine(),
                num_simulations=mcts_simulations,
            )
        
        # ゲーム状態の追跡
        self.my_seat = None
        self.board_self = Board()
        self.board_opponent = Board()
        self.my_discards = []
    
    async def play(self, room_id: str, player_id: str):
        """ルームに接続して対戦"""
        uri = f"ws://localhost:8000/ws/{room_id}?player_id={player_id}"
        
        async with websockets.connect(uri) as ws:
            # ゲームスタートを送信
            await ws.send(json.dumps({"type": "game_start"}))
            
            async for raw_message in ws:
                message = json.loads(raw_message)
                response = self.handle_message(message)
                
                if response:
                    await ws.send(json.dumps(response))
                
                if message["type"] == "session_end":
                    break
    
    def handle_message(self, msg: dict) -> dict | None:
        """メッセージに応じて行動を返す"""
        
        if msg["type"] == "session_start":
            return None
        
        elif msg["type"] == "hand_start":
            self.my_seat = msg["your_seat"]
            self.board_self = Board()
            self.board_opponent = Board()
            self.my_discards = []
            return None
        
        elif msg["type"] == "deal":
            return self.decide_action(msg)
        
        elif msg["type"] == "opponent_placed":
            self.board_opponent = Board.from_dict(msg["opponent_board"])
            return None
        
        elif msg["type"] == "hand_end":
            self.update_from_result(msg["result"])
            # 次のハンドも自動で進む
            return None
        
        elif msg["type"] == "session_ended_awaiting_restart":
            # 自動で再戦
            return {"type": "game_start"}
        
        elif msg["type"] == "timeout":
            # タイムアウトされた（MCTSが遅すぎた場合）
            return None
        
        return None
    
    def decide_action(self, deal_msg: dict) -> dict:
        """配られたカードに対して行動を決定"""
        dealt_cards = deal_msg["cards"]
        turn = deal_msg["turn"]
        
        # Observationを構築
        obs = Observation(
            board_self=self.board_self,
            board_opponent=Board.from_dict(deal_msg["opponent_board"]),
            dealt_cards=dealt_cards,
            known_discards_self=self.my_discards,
            turn=turn,
            is_btn=deal_msg.get("is_btn", False),
            chips_self=deal_msg.get("chips_self", 200),
            chips_opponent=deal_msg.get("chips_opponent", 200),
        )
        
        # 行動選択
        valid_actions = get_valid_actions(obs)
        
        if self.use_mcts:
            action_idx, _ = self.mcts.search(obs)
        else:
            # MCTSなし: Policy Netから直接
            state = torch.FloatTensor(encode_state(obs)).unsqueeze(0)
            mask = torch.BoolTensor(create_action_mask(valid_actions)).unsqueeze(0)
            probs = self.policy_net(state, mask)
            action_idx = torch.argmax(probs).item()
        
        action = valid_actions[action_idx]
        
        # 自分のボードと捨て札を更新
        for card, pos in action.placements:
            getattr(self.board_self, pos).append(card)
        if action.discard:
            self.my_discards.append(action.discard)
        
        # WebSocketメッセージとして返す
        response = {
            "type": "place",
            "placements": action.placements,
        }
        if action.discard:
            response["discard"] = action.discard
        
        return response
```

### 9.2 強さの調整

MCTSのパラメータでAIの強さを制御できる。

| レベル | シミュレーション数 | 時間/ターン | 用途 |
|--------|-------------------|------------|------|
| 弱い | 10 | ~0.1秒 | デバッグ、初心者向け |
| 普通 | 100 | ~1秒 | カジュアル対戦 |
| 強い | 500 | ~5秒 | 本気の対戦 |
| 最強 | 2000 | ~15秒 | 分析用 |
| Policy Netのみ | 0 (MCTSなし) | ~0.01秒 | 高速対戦、大量データ生成 |

### 9.3 フィードバックループ

```
1. RYO vs AI で対戦
     │
     ▼
2. 対戦ログを分析
     │  - AIが負けたハンドの分析
     │  - AIの弱点パターンの特定
     │  - RYOの新しい戦略の検出
     │
     ▼
3. 対戦ログも学習データに追加
     │  - RYOの新しいプレイパターンでBC追加学習
     │  - AIの負けパターンをSelf-Playで重点学習
     │
     ▼
4. 改善されたAIで再対戦
     │
     ▼
5. 2に戻る
```

---

## 10. ヘッドレスゲームエンジン

Self-PlayではWebSocket不要の高速エンジンが必要。

```python
class GameEngine:
    """WebSocket不要のヘッドレスゲームエンジン"""
    
    def new_session(self) -> Session:
        """新しいセッションを開始"""
        return Session(chips=[200, 200])
    
    def new_hand(self, session: Session) -> Hand:
        """新しいハンドを開始"""
        deck = self.create_deck()  # 54枚
        random.shuffle(deck)
        return Hand(deck=deck, btn=session.btn_seat)
    
    def get_valid_actions(self, obs: Observation) -> list[Action]:
        """有効な行動を全て列挙"""
        if obs.turn == 0:
            return get_initial_actions(obs.dealt_cards, obs.board_self)
        else:
            return get_turn_actions(obs.dealt_cards, obs.board_self)
    
    def simulate_action(self, obs: Observation, action: Action) -> Observation:
        """行動を適用した後の状態を返す（元の状態は変更しない）"""
        new_obs = copy.deepcopy(obs)
        for card, pos in action.placements:
            getattr(new_obs.board_self, pos).append(card)
        if action.discard:
            new_obs.known_discards_self.append(action.discard)
        return new_obs
    
    def is_hand_finished(self, state) -> bool:
        """ハンドが終了したか"""
        board = state.board_self
        return (len(board.top) == 3 and
                len(board.middle) == 5 and
                len(board.bottom) == 5)
    
    def compute_score(self, hand) -> dict:
        """最終スコアを計算"""
        # ロイヤリティ、勝敗、スクープ、チップキャップ全て含む
        ...
    
    @staticmethod
    def create_deck() -> list[str]:
        ranks = "23456789TJQKA"
        suits = "hdcs"
        deck = [f"{r}{s}" for s in suits for r in ranks]
        deck.extend(["X1", "X2"])
        return deck
```

---

## 11. データフロー全体図

```
┌──────────────────────────────────────────────────────────────┐
│                        データ収集                            │
│                                                              │
│  RYOのプレイ(Webアプリ) → SQLite → JSONL → 前処理 → numpy   │
│                                                              │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                   Phase B: Behavior Cloning                  │
│                                                              │
│  numpy → 教師あり学習 → Policy Net v0 + Value Net v0        │
│                                                              │
│  正解: RYOの行動 + ハンド結果（ロイヤリティ/バスト/FL）      │
│  評価: Top-1 Accuracy > 40%, Bust AUC > 0.85                │
│                                                              │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                Phase C: Self-Play + MCTS                     │
│                                                              │
│  ┌─────────────────────────────────────────────┐             │
│  │  1. AI_vN vs AI_vN (MCTS付き, 200ゲーム)   │             │
│  │     │                                       │             │
│  │     ▼                                       │             │
│  │  2. 軌跡データ収集                          │             │
│  │     (状態, MCTSの行動分布, 最終報酬)        │             │
│  │     │                                       │             │
│  │     ▼                                       │             │
│  │  3. Policy Net + Value Net を更新           │             │
│  │     Policy: MCTSの分布を正解として学習      │             │
│  │     Value:  最終報酬を正解として学習         │             │
│  │     │                                       │             │
│  │     ▼                                       │             │
│  │  4. AI_v(N+1) が完成 → 1に戻る             │             │
│  └─────────────────────────────────────────────┘             │
│                                                              │
│  50-100イテレーションで収束                                  │
│                                                              │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                   Phase D: 対人戦                            │
│                                                              │
│  AI (WebSocket Client) vs RYO (Webアプリ)                    │
│     │                                                        │
│     ├── 対戦ログ収集                                         │
│     ├── 弱点分析                                             │
│     ├── 追加学習（BC + Self-Play）                           │
│     └── 再対戦                                               │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 12. 計算リソースの見積もり

### 12.1 Behavior Cloning

| 項目 | 値 |
|------|-----|
| データ量 | 500-5,000ハンド（4,500-45,000ターン） |
| エポック数 | 100 |
| バッチサイズ | 64 |
| 学習時間 | 10分-2時間（ローカルGPU） |
| 必要メモリ | 2-4GB |

### 12.2 Self-Play

| 項目 | 値 |
|------|-----|
| ゲーム数/イテレーション | 200 |
| MCTSシミュレーション/ターン | 200 |
| 1ターンの計算時間 | ~0.5秒 |
| 1ゲーム(~18ターン)の時間 | ~9秒 |
| 1イテレーションの時間 | ~30分 |
| 全イテレーション(100回) | ~50時間 |
| 必要GPU | 1× (RTX 3060以上推奨) |

### 12.3 高速化の選択肢

- **バッチ推論**: 複数ゲームのニューラルネット推論をバッチ化
- **並列Self-Play**: 複数プロセスで同時にゲームを実行
- **Rust MCTS**: MCTSのツリー探索部分をRustで実装（将来）
- **Policy Netのみモード**: MCTS不使用で高速に大量データ生成

---

## 13. ディレクトリ構成（AI部分）

```
ofc-pineapple/
├── ai/
│   ├── models/
│   │   ├── policy_net.py         # Policy Network定義
│   │   ├── value_net.py          # Value Network定義
│   │   └── checkpoints/          # 学習済みモデル保存先
│   │       ├── bc_policy_v1.pt
│   │       ├── bc_value_v1.pt
│   │       ├── sp_policy_v10.pt  # Self-Play 10イテレーション目
│   │       └── sp_value_v10.pt
│   │
│   ├── training/
│   │   ├── preprocess.py         # JSONL → numpy 変換
│   │   ├── behavior_cloning.py   # BC学習スクリプト
│   │   ├── self_play.py          # Self-Play実行
│   │   ├── evaluate.py           # モデル評価
│   │   └── config.yaml           # ハイパーパラメータ
│   │
│   ├── mcts/
│   │   ├── mcts.py               # MCTS本体
│   │   ├── node.py               # MCTSノード
│   │   └── is_mcts.py            # 不完全情報対応版
│   │
│   ├── engine/
│   │   ├── game_engine.py        # ヘッドレスゲームエンジン
│   │   ├── hand_eval.py          # ハンド評価
│   │   ├── scoring.py            # スコア計算
│   │   └── deck.py               # デッキ管理
│   │
│   ├── player/
│   │   ├── ai_player.py          # WebSocket AIプレイヤー
│   │   └── random_player.py      # ランダムプレイヤー（テスト用）
│   │
│   └── utils/
│       ├── encoding.py           # 状態エンコーディング
│       ├── action_space.py       # 行動空間
│       └── replay_buffer.py      # 経験リプレイバッファ
│
├── data/
│   ├── logs/                     # JONLログ
│   ├── processed/                # 前処理済みnumpy
│   └── self_play/                # Self-Playのデータ
│
└── scripts/
    ├── export_logs.py            # SQLite → JSONL
    ├── train_bc.sh               # BC学習実行
    ├── run_self_play.sh          # Self-Play実行
    └── play_vs_ai.py             # AI対戦起動
```
