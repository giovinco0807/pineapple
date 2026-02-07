# OFC Pineapple Web App 設計書

## 1. 概要

### 1.1 目的
OFC Pineapple の対人戦Webアプリケーションを構築し、全ターンのプレイログを収集する。
収集したログはAI（Behavior Cloning → Self-Play）の学習データとして使用する。

### 1.2 要件
- 人 vs 人のリアルタイム対戦（WebSocket）
- ジョーカー2枚入り54枚デッキ
- Fantasyland突入時は既存Rustソルバーで自動配置
- 全ターンのゲーム状態・行動をJSONLで保存
- 最初はRYOが2タブで両方操作

### 1.3 技術スタック

| レイヤー | 技術 | 理由 |
|---------|------|------|
| Frontend | React + TypeScript | カードUI操作、リアルタイム更新 |
| Backend | Python (FastAPI + WebSocket) | AI学習パイプラインとの統合 |
| DB | SQLite | 軽量、ファイルベースで移行が楽 |
| 通信 | WebSocket | リアルタイム双方向通信 |
| FLソルバー | Rust（既存） | stdin/stdout JSON経由で呼び出し |

---

## 2. アーキテクチャ

### 2.1 全体構成

```
┌─────────────────┐     WebSocket      ┌─────────────────────────┐
│   Browser Tab 1  │◄──────────────────►│                         │
│   (Player 1)     │                    │   FastAPI Server        │
└─────────────────┘                    │                         │
                                        │  ┌───────────────────┐  │
┌─────────────────┐     WebSocket      │  │  GameManager      │  │
│   Browser Tab 2  │◄──────────────────►│  │  - rooms{}        │  │
│   (Player 2)     │                    │  │  - game_state     │  │
└─────────────────┘                    │  │  - deck           │  │
                                        │  └───────┬───────────┘  │
                                        │          │              │
                                        │  ┌───────▼───────────┐  │
                                        │  │  LogWriter        │  │
                                        │  │  → SQLite         │  │
                                        │  │  → JSONL export   │  │
                                        │  └───────────────────┘  │
                                        │          │              │
                                        │  ┌───────▼───────────┐  │
                                        │  │  FL Solver (Rust)  │  │
                                        │  │  subprocess call   │  │
                                        │  └───────────────────┘  │
                                        └─────────────────────────┘
```

### 2.2 ディレクトリ構成

```
ofc-pineapple/
├── backend/
│   ├── main.py              # FastAPI エントリポイント
│   ├── game/
│   │   ├── __init__.py
│   │   ├── engine.py        # ゲームロジック（デッキ、配布、スコア計算）
│   │   ├── room.py          # ルーム管理、WebSocket接続
│   │   ├── models.py        # データモデル（Pydantic）
│   │   ├── scoring.py       # ロイヤリティ計算、バスト判定
│   │   └── hand_eval.py     # ハンド評価（ジョーカー対応）
│   ├── solver/
│   │   ├── fl_bridge.py     # Rustソルバー呼び出し
│   │   └── fl_solver        # Rustバイナリ
│   ├── log/
│   │   ├── writer.py        # ログ書き込み
│   │   └── exporter.py      # JSONL エクスポート
│   ├── db/
│   │   ├── schema.sql       # テーブル定義
│   │   └── connection.py    # SQLite接続
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   │   ├── Board.tsx         # ボード表示（Top/Middle/Bottom）
│   │   │   ├── CardHand.tsx      # 手札表示
│   │   │   ├── Card.tsx          # カード1枚の表示
│   │   │   ├── GameRoom.tsx      # ゲームルーム全体
│   │   │   ├── OpponentBoard.tsx # 相手ボード（公開情報）
│   │   │   ├── ScoreBoard.tsx    # スコア表示
│   │   │   └── Lobby.tsx         # ルーム選択・作成
│   │   ├── hooks/
│   │   │   └── useWebSocket.ts   # WebSocket管理
│   │   ├── types/
│   │   │   └── game.ts           # 型定義
│   │   └── utils/
│   │       └── cardUtils.ts      # カード表示ユーティリティ
│   └── package.json
├── data/
│   └── logs/                # JSONL出力先
└── README.md
```

---

## 3. データモデル

### 3.1 カード表現

```
通常カード: "{Rank}{Suit}"
  Rank: 2,3,4,5,6,7,8,9,T,J,Q,K,A
  Suit: h(hearts), d(diamonds), c(clubs), s(spades)
  例: "Ah", "Td", "2c"

ジョーカー: "X1", "X2"
```

※ FLソルバーとの互換性を維持するため、既存の表記に合わせる。

### 3.2 ボード状態

```python
class Board:
    top: list[str]       # 最大3枚
    middle: list[str]    # 最大5枚
    bottom: list[str]    # 最大5枚
```

### 3.3 ゲーム状態（2層構造）

**Session（ゲーム）**: 持ち点を賭けた1ゲーム単位

```python
class Session:
    session_id: str               # UUID
    room_id: str
    status: str                   # "waiting" | "active" | "finished"
    
    chips: list[int]              # 持ち点 [p0, p1] 初期値 [200, 200]
    hands_played: int             # プレイ済みハンド数
    btn_seat: int                 # 現在のBTN（0 or 1）、初回ランダム、以降交互
    
    # 終了条件
    winner: int | None            # 勝者（0 or 1 or None）
    finish_reason: str | None     # "chip_lead" | "bankrupt" | "manual"
```

**Hand（ハンド）**: 1ハンド（配布〜スコア確定）

```python
class HandState:
    hand_id: str                  # UUID
    session_id: str
    hand_number: int              # このセッション内の通し番号
    status: str                   # "dealing" | "playing" | "fantasyland" | "scoring" | "finished"
    turn: int                     # 0-8 (0=初手5枚、1-8=3枚配布)
    btn: int                      # 0 or 1 (BTNプレイヤー)
    deck: list[str]               # 残りデッキ（サーバーのみ保持）
    
    players: list[PlayerState]    # 2名分
    
    discards: list[list[str]]     # 各プレイヤーの捨て札（相手に非公開）
    
    current_player: int           # 現在の手番（0 or 1）
    dealt_cards: dict[int, list[str]]  # 各プレイヤーに配られたカード

class PlayerState:
    player_id: str
    board: Board
    is_fantasyland: bool          # FL中かどうか
    fl_card_count: int            # FL枚数（14-17）
    connected: bool               # WebSocket接続中
```

### 3.4 チップ（持ち点）ルール

```
- 初期持ち点: 各200点
- ハンド結果の反映: 
    raw_score = 列勝敗 + スクープ + ロイヤリティ差分
    
    # 持ち点キャップ: 相手の残り持ち点を超えては取れない
    actual_score = min(raw_score, opponent_chips)
    
    winner.chips += actual_score
    loser.chips  -= actual_score

- ゲーム終了条件（ハンド終了時に毎回判定）:
    条件A: いずれかの持ち点 ≤ 0 → 即終了（FL中でも）
    条件B: 両者ともFLに入っていない & 持ち点差 ≥ 40 → 次のハンド開始せず終了
    
    ※ 条件Aは最優先。FL Stayしていても持ち点0なら終了。
    ※ 条件Bは両者が通常プレイに戻った時点で初めて判定。
```
```

### 3.4 行動

```python
class Action:
    placements: list[tuple[str, str]]  # [(card, position), ...]
    discard: str | None                # 捨てカード（初手はNone）

# 初手（5枚全配置）:
#   placements: [("Ah","top"), ("Kd","top"), ("7s","middle"), ("8c","middle"), ("3h","bottom")]
#   discard: None

# 通常ターン（3枚から2枚配置、1枚捨て）:
#   placements: [("Td","middle"), ("5h","bottom")]
#   discard: "2c"
```

---

## 4. ゲームフロー

### 4.1 セッション（ゲーム）進行

```
1. ルーム作成 → P1参加 → P2参加
2. 両者が「ゲームスタート」ボタンを押す
3. セッション開始: 各プレイヤー 200点
4. BTN/BBをランダム決定（初回のみ）
5. ── ハンドループ ──
   a. ハンド開始前チェック:
      - いずれかの持ち点 ≤ 0 → ゲーム終了
      - 両者ともFLに入っていない & 持ち点差 ≥ 40 → ゲーム終了
   b. ハンドをプレイ（4.2参照）
   c. スコア計算 → チップ反映（持ち点キャップ適用）
   d. FLエントリー判定
   e. BTN/BB交代
   f. 5a に戻る
6. ゲーム終了 → 結果表示
7. 同じルームで「ゲームスタート」ボタンを押して次のセッション開始可能
```

### 4.2 ハンド（通常プレイ）

```
1. デッキ生成（52枚 + X1, X2 = 54枚）、シャッフル
2. ── ターン0（初手）──
   a. 各プレイヤーに5枚配布
   b. BTNプレイヤーが先に全5枚を配置
   c. BBプレイヤーが全5枚を配置
3. ── ターン1〜8 ──
   a. 各プレイヤーに3枚配布
   b. BTNプレイヤーが先に2枚配置、1枚捨て
   c. BBプレイヤーが2枚配置、1枚捨て
4. ── 全ターン終了 ──
   a. バスト判定（Top ≤ Middle ≤ Bottom の強さ順でなければバスト）
   b. ロイヤリティ計算
   c. 各列の勝敗判定（+1/-1/0）
   d. スクープ判定（3列全勝で+3ボーナス）
   e. チップ反映（持ち点キャップ: min(raw_score, opponent_chips)）
   f. FLエントリー判定
   g. ログ保存
```

### 4.3 Fantasyland

```
1. FL突入プレイヤーの枚数決定
   - Top QQ: 14枚
   - Top KK: 15枚
   - Top AA: 16枚
   - Top Trips: 17枚

2. FLプレイヤー:
   a. 枚数分のカードを配布
   b. Rustソルバーを呼び出し（JSON stdin/stdout）
   c. ソルバーの最適配置を自動適用
   d. 13枚配置、残り捨て

3. 非FLプレイヤー:
   a. 通常通りターン0〜8をプレイ
   b. FLプレイヤーのボードは見えない（配置完了まで非公開）

4. 両者完了後:
   a. FLプレイヤーのボードを公開
   b. スコア計算（通常と同じ）
   c. FL Stay判定
      - Bottom: Quads or Straight Flush → Stay
      - Top: Trips → Stay
   d. Stayした場合 → 再度FL（枚数再計算）
   e. Stayしない場合 → 通常プレイに戻る
```

### 4.4 手番の詳細（同時配置 vs 交互配置）

**採用: 交互配置（BTN先手）**

理由:
- 実際のOFC Pineappleのルールに準拠
- 相手の配置を見てから自分が配置する戦略性がある
- ログデータとして「相手の配置を見た上での判断」が記録される（AI学習に有用）

ただし初手は両者とも相手のボードが空なので、実質的に同時。
効率化のため、**初手のみ同時配置（両者の配置が揃ったら次へ進む）** とする。

---

## 5. API設計

### 5.1 REST API

```
POST   /api/rooms                  # ルーム作成
GET    /api/rooms                  # ルーム一覧
GET    /api/rooms/{room_id}        # ルーム状態
POST   /api/rooms/{room_id}/join   # ルーム参加

GET    /api/games/{game_id}/log    # ゲームログ取得
GET    /api/export?format=jsonl    # 全ログJSONLエクスポート
```

### 5.2 WebSocket メッセージ

**接続:**
```
ws://localhost:8000/ws/{room_id}?player_id={player_id}
```

**サーバー → クライアント:**

```jsonc
// ゲーム開始
{
  "type": "session_start",
  "session_id": "uuid",
  "chips": [200, 200]
}

// ハンド開始
{
  "type": "hand_start",
  "hand_id": "uuid",
  "hand_number": 1,
  "btn": 0,
  "your_seat": 0,
  "chips": [200, 200]
}

// カード配布
{
  "type": "deal",
  "turn": 0,
  "cards": ["Ah", "Kd", "7s", "8c", "3h"],
  "opponent_board": {
    "top": [], "middle": [], "bottom": []
  }
}

// 相手の配置完了（ボード更新）
{
  "type": "opponent_placed",
  "opponent_board": {
    "top": ["??", "??"],
    "middle": ["??", "??", "??"],
    "bottom": []
  }
  // 注: 通常プレイでは相手のカードは見える。
  //     FL中の相手ボードのみ "??" で隠す。
}

// 相手の配置完了（通常時、カード公開）
{
  "type": "opponent_placed",
  "opponent_board": {
    "top": ["Qs", "Qh"],
    "middle": ["9d", "Td", "Jd"],
    "bottom": ["As", "Ks"]
  }
}

// ゲーム終了
{
  "type": "hand_end",
  "result": {
    "boards": [
      {"top": [...], "middle": [...], "bottom": [...]},
      {"top": [...], "middle": [...], "bottom": [...]}
    ],
    "busted": [false, false],
    "royalties": [
      {"top": 9, "middle": 4, "bottom": 6, "total": 19},
      {"top": 1, "middle": 0, "bottom": 2, "total": 3}
    ],
    "line_results": [1, 1, -1],       // P0視点: Top勝ち, Mid勝ち, Bot負け
    "scoop": false,
    "raw_score": [20, -20],            // キャップ前の得失点
    "actual_score": [20, -20],         // キャップ後（min(raw, opponent_chips)）
    "chips": [220, 180],              // ハンド後の持ち点
    "fl_entry": [true, false],
    "fl_card_count": [15, 0]
  }
}

// セッション（ゲーム）終了
{
  "type": "session_end",
  "winner": 0,
  "final_chips": [240, 160],
  "hands_played": 7,
  "reason": "chip_lead"              // "chip_lead" | "bankrupt"
}

// FL自動配置結果
{
  "type": "fantasyland_result",
  "player": 0,
  "board": {
    "top": ["Ah", "Ad", "Ac"],
    "middle": ["7s", "8s", "9s", "Ts", "Js"],
    "bottom": ["Kh", "Kd", "Kc", "Ks", "2h"]
  },
  "discarded": ["3c", "4d"],
  "solver_score": 47,
  "fl_stay": true
}

// エラー
{
  "type": "error",
  "message": "Invalid placement: Top already has 3 cards"
}

// 待機中
{
  "type": "waiting_for_opponent"
}

// 両者揃い、スタート待ち
{
  "type": "ready_to_start",
  "players": ["player1_name", "player2_name"]
}

// セッション終了後、再戦待ち
{
  "type": "session_ended_awaiting_restart",
  "winner": 0,
  "final_chips": [240, 160]
}
```

**クライアント → サーバー:**

```jsonc
// ゲームスタート（両者が送信で開始）
{
  "type": "game_start"
}

```jsonc
// カード配置（初手）
{
  "type": "place",
  "placements": [
    ["Ah", "top"],
    ["Kd", "top"],
    ["7s", "middle"],
    ["8c", "middle"],
    ["3h", "bottom"]
  ]
}

// カード配置（通常ターン）
{
  "type": "place",
  "placements": [
    ["Td", "middle"],
    ["5h", "bottom"]
  ],
  "discard": "2c"
}
```

---

## 6. ログ設計

### 6.1 SQLiteスキーマ

```sql
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    room_id TEXT NOT NULL,
    started_at TEXT NOT NULL,         -- ISO 8601
    finished_at TEXT,
    player0_id TEXT NOT NULL,
    player1_id TEXT NOT NULL,
    initial_chips INTEGER NOT NULL DEFAULT 200,
    final_chips_p0 INTEGER,
    final_chips_p1 INTEGER,
    winner INTEGER,                   -- 0 or 1 or NULL
    finish_reason TEXT,               -- "chip_lead" | "bankrupt" | "manual"
    hands_played INTEGER DEFAULT 0
);

CREATE TABLE hands (
    hand_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    hand_number INTEGER NOT NULL,     -- セッション内の通し番号
    started_at TEXT NOT NULL,
    finished_at TEXT,
    btn INTEGER NOT NULL,             -- 0 or 1
    chips_before TEXT NOT NULL,       -- JSON: [p0_chips, p1_chips]
    chips_after TEXT,                 -- JSON: [p0_chips, p1_chips]
    raw_score TEXT,                   -- JSON: [raw_p0, raw_p1]
    actual_score TEXT,                -- JSON: [actual_p0, actual_p1] (キャップ後)
    result_detail TEXT                -- JSON: 終了時の詳細結果
);

CREATE TABLE turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hand_id TEXT NOT NULL REFERENCES hands(hand_id),
    turn INTEGER NOT NULL,           -- 0-8
    player INTEGER NOT NULL,         -- 0 or 1
    
    -- 配布時の状態
    board_self TEXT NOT NULL,        -- JSON: Board
    board_opponent TEXT NOT NULL,    -- JSON: Board
    dealt_cards TEXT NOT NULL,       -- JSON: ["Ah", "Kd", ...]
    known_discards TEXT NOT NULL,    -- JSON: 自分の捨て札リスト
    
    -- プレイヤーの行動
    action_placements TEXT NOT NULL, -- JSON: [["Ah","top"], ...]
    action_discard TEXT,             -- 捨てカード（初手はNULL）
    
    -- メタデータ
    think_time_ms INTEGER,           -- 考慮時間（ミリ秒）
    timestamp TEXT NOT NULL,         -- ISO 8601
    
    UNIQUE(hand_id, turn, player)
);

CREATE TABLE fantasyland (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hand_id TEXT NOT NULL REFERENCES hands(hand_id),
    player INTEGER NOT NULL,         -- 0 or 1
    card_count INTEGER NOT NULL,     -- 14-17
    dealt_cards TEXT NOT NULL,        -- JSON: 配られた全カード
    solver_placement TEXT NOT NULL,   -- JSON: ソルバーの配置結果
    solver_score REAL,               -- ソルバーの評価スコア
    fl_stay INTEGER NOT NULL,        -- 0 or 1
    timestamp TEXT NOT NULL
);
```

### 6.2 JSONLエクスポート形式

AIの学習データとして以下の形式でエクスポート:

```jsonc
// === 1ターン = 1行 ===
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "hand_id": "660e8400-e29b-41d4-a716-446655440001",
  "hand_number": 3,
  "turn": 3,
  "player": 0,
  "btn": 1,                                     // BTN seat (0 or 1)
  "is_btn": false,                               // このプレイヤーがBTNか
  "chips": [185, 215],                           // このターン時点の持ち点
  
  // 行動時の盤面（このプレイヤー視点）
  "board_self": {
    "top": ["Ah", "Ad"],
    "middle": ["7s", "8s", "9s"],
    "bottom": ["Kd", "Kc"]
  },
  "board_opponent": {
    "top": ["Qs"],
    "middle": ["Td", "Jd", "Qd"],
    "bottom": ["3c", "3d", "3h"]
  },
  
  // 配られたカード
  "dealt_cards": ["5h", "X1", "Tc"],
  
  // 既知の情報
  "known_discards_self": ["2c", "4d"],
  "visible_cards": ["Ah","Ad","7s","8s","9s","Kd","Kc",
                     "Qs","Td","Jd","Qd","3c","3d","3h",
                     "5h","X1","Tc"],
  
  // 行動
  "action": {
    "placements": [["5h", "middle"], ["Tc", "bottom"]],
    "discard": "X1"
  },
  
  // メタデータ
  "think_time_ms": 4200,
  "timestamp": "2026-02-07T15:30:00Z"
}
```

### 6.3 ログの活用フロー

```
[Webアプリ] → SQLite → JSONL export
                              ↓
                     [前処理スクリプト]
                              ↓
              ┌───────────────┼───────────────┐
              ↓               ↓               ↓
      状態ベクトル化     行動エンコード    報酬計算
              ↓               ↓               ↓
              └───────────────┼───────────────┘
                              ↓
                    [Behavior Cloning]
                              ↓
                    [Self-Play強化学習]
```

---

## 7. フロントエンド設計

### 7.1 画面構成

```
┌──────────────────────────────────────────────┐
│  OFC Pineapple                    Room: ABC  │
│  [You: 185 chips (BB)]  [Opp: 215 chips (BTN)]│
├──────────────────────────────────────────────┤
│                                              │
│  ┌─ 相手ボード（公開）─────────────────┐     │
│  │ Top:    [Qs][Qh][  ]               │     │
│  │ Middle: [9d][Td][Jd][  ][  ]       │     │
│  │ Bottom: [As][Ks][7h][  ][  ]       │     │
│  └─────────────────────────────────────┘     │
│                                              │
│  ──────────── 中央エリア ─────────────────    │
│                                              │
│  ┌─ 自分ボード ─────────────────────────┐    │
│  │ Top:    [Ah][Ad][  ]  ← ドロップ先   │    │
│  │ Middle: [7s][8s][9s][  ][  ]         │    │
│  │ Bottom: [Kd][Kc][  ][  ][  ]        │    │
│  └──────────────────────────────────────┘    │
│                                              │
│  ┌─ 手札 ──────────────────────────────┐     │
│  │ [5h]  [X1]  [Tc]                    │     │
│  │                          [Confirm]  │     │
│  └─────────────────────────────────────┘     │
│                                              │
│  捨て場: [ここにドラッグで捨て]              │
│                                              │
├──────────────────────────────────────────────┤
│  Hand #3  |  Turn 4/9  |  Thinking: 3.2s    │
└──────────────────────────────────────────────┘
```

### 7.2 操作方法

- **ドラッグ&ドロップ**: 手札のカードをTop/Middle/Bottom/捨て場にドラッグ
- **クリック配置**（代替）: カードをクリック → 配置先をクリック
- **Confirm ボタン**: 全カード配置後に確定
- **Undo**: 確定前なら配置をやり直し可能
- **ゲームスタートボタン**: セッション開始時・終了後に両者が押して開始

### 7.3 タイマー表示

```
- 各ターン20秒のカウントダウンを画面下部に表示
- 残り5秒で警告（色変化 or アニメーション）
- タイムアウト時は自動配置され、結果を表示
```

### 7.4 表示ルール

- 自分の手札: 常に表示
- 自分のボード: 常に表示
- 相手のボード: 通常時は公開、FL中は「??」で非表示
- 捨て札: 自分のもののみ表示（相手の捨て札は非公開）
- スコア: 両者の持ち点（chips）とBTN/BBを常時表示
- ロイヤリティ: ボード横にリアルタイム表示（現時点の暫定値）

---

## 8. スコア計算

### 8.1 ロイヤリティテーブル

```python
TOP_ROYALTIES = {
    # ペア
    "66": 1, "77": 2, "88": 3, "99": 4, "TT": 5,
    "JJ": 6, "QQ": 7, "KK": 8, "AA": 9,
    # トリップス
    "222": 10, "333": 11, "444": 12, "555": 13, "666": 14,
    "777": 15, "888": 16, "999": 17, "TTT": 18,
    "JJJ": 19, "QQQ": 20, "KKK": 21, "AAA": 22,
}

MIDDLE_ROYALTIES = {
    "trips": 2, "straight": 4, "flush": 8,
    "full_house": 12, "quads": 20,
    "straight_flush": 30, "royal_flush": 50,
}

BOTTOM_ROYALTIES = {
    "straight": 2, "flush": 4, "full_house": 6,
    "quads": 10, "straight_flush": 15, "royal_flush": 25,
}
```

### 8.2 スコア計算手順

```
1. バスト判定
   - hand_rank(Top) ≤ hand_rank(Middle) ≤ hand_rank(Bottom) でなければバスト
   - バストしたプレイヤーはロイヤリティ0、各列自動負け

2. 各列の勝敗（バストしていないプレイヤー同士）
   - 各列で hand_rank 比較
   - 勝ち: +1, 負け: -1, 引分: 0

3. スクープ判定
   - 3列全勝で追加+3（相手がバストした場合も3列勝ちなのでスクープ）

4. ロイヤリティ差分
   - 各プレイヤーのロイヤリティ合計を計算
   - 差分をスコアに加算

5. Raw スコア計算
   raw_score = 列勝敗の合計 + スクープボーナス + (自ロイヤリティ - 相手ロイヤリティ)

6. チップキャップ適用
   actual_score = min(raw_score, opponent_chips)
   ※ 相手の持ち点を超えては取れない（1ゲームで最大-200点）

7. チップ反映
   winner.chips  += actual_score
   loser.chips   -= actual_score
```

### 8.3 ジョーカー処理

ジョーカー（X1, X2）は万能カード（ワイルドカード）として機能:
- ハンド評価時に最も有利なカードとして扱う
- 例: [Ah, Kh, Qh, Jh, X1] → ロイヤルフラッシュ（X1 = Th）
- 既存のRustソルバーのジョーカー処理ロジックをPython側にも移植

---

## 9. FL ソルバー連携

### 9.1 呼び出しインターフェース

```python
import subprocess
import json

def solve_fantasyland(cards: list[str], joker_count: int) -> dict:
    """
    Rustソルバーを呼び出してFL最適配置を取得
    
    Args:
        cards: 配布カード（14-17枚）
        joker_count: ジョーカー枚数（0-2）
    
    Returns:
        {
            "top": ["Ah", "Ad", "Ac"],
            "middle": ["7s", "8s", "9s", "Ts", "Js"],
            "bottom": ["Kh", "Kd", "Kc", "Ks", "2h"],
            "discarded": ["3c", "4d"],
            "score": 47.0,
            "fl_stay": true,
            "royalties": {"top": 22, "middle": 30, "bottom": 10}
        }
    """
    input_data = json.dumps({
        "cards": cards,
        "joker_count": joker_count
    })
    
    result = subprocess.run(
        ["./solver/fl_solver"],
        input=input_data,
        capture_output=True,
        text=True,
        timeout=10
    )
    
    return json.loads(result.stdout)
```

### 9.2 FLプレイの流れ

```
1. FLプレイヤーにカード配布（14-17枚）
2. 非FLプレイヤーには通常通りターン進行
3. FLプレイヤー側:
   a. サーバーがRustソルバーを subprocess で呼び出し
   b. 最適配置結果を受け取り
   c. 自動的にボードに配置
   d. クライアントに結果を通知（ソルバースコア表示）
4. 非FLプレイヤーの全ターン完了を待つ
5. 両者完了後にスコア計算
```

---

## 10. 状態管理とバリデーション

### 10.1 サーバーサイドバリデーション

全ての配置操作はサーバーで検証する（クライアントを信用しない）:

```python
def validate_placement(state: GameState, player: int, action: Action) -> str | None:
    """不正な配置の場合エラーメッセージを返す"""
    
    board = state.players[player].board
    dealt = state.dealt_cards[player]
    
    # 1. 配置カードが配布カードに含まれているか
    placed = [c for c, _ in action.placements]
    if action.discard:
        used = placed + [action.discard]
    else:
        used = placed
    
    if sorted(used) != sorted(dealt):
        return "配置+捨てカードが配布カードと一致しません"
    
    # 2. 配置先の空きがあるか
    limits = {"top": 3, "middle": 5, "bottom": 5}
    temp_board = copy(board)
    for card, position in action.placements:
        current = getattr(temp_board, position)
        if len(current) >= limits[position]:
            return f"{position}はすでに{limits[position]}枚です"
        current.append(card)
    
    # 3. 初手は5枚配置、以降は2枚配置+1枚捨て
    if state.turn == 0:
        if len(action.placements) != 5 or action.discard is not None:
            return "初手は5枚全て配置してください"
    else:
        if len(action.placements) != 2 or action.discard is None:
            return "2枚配置、1枚捨ててください"
    
    return None  # OK
```

### 10.3 タイムアウト処理

```
- 各ターンの制限時間: 20秒
- カード配布時にサーバーがタイマー開始
- クライアントにも残り時間を通知（countdown表示用）
- 20秒超過時:
    1. サーバーが自動配置を実行
       - 手札を左から順にTop → Middle → Bottom の空き枠に配置
       - 余り1枚を捨て（通常ターン）
    2. 自動配置の結果をクライアントに通知
    3. ログには think_time_ms = -1（タイムアウト）と記録
```

**WebSocket メッセージ:**
```jsonc
// タイマー通知（カード配布と同時）
{
  "type": "timer_start",
  "seconds": 20
}

// タイムアウト発動
{
  "type": "timeout",
  "auto_placement": {
    "placements": [["5h", "top"], ["Tc", "middle"]],
    "discard": "X1"
  }
}
```

### 10.4 ゲーム状態遷移

``` → ACTIVE → FINISHED
                     │
                     ├── Hand: DEALING → PLAYING → SCORING → FINISHED
                     │                                │
                     │                     FL判定 → FANTASYLAND → SCORING
                     │                                │
                     │                     FL Stay → FANTASYLAND (繰返し)
                     │
                     ├── 終了判定: 差≥40 & 両者非FL → Session FINISHED
                     │             or chips≤0       → Session FINISHED
                     │
                     └── BTN/BB交代 → 次のHand
```

---

## 11. 実装フェーズ

### Phase 1: 最小動作版（MVP）
- バックエンド: ゲームエンジン + WebSocket + 基本API
- フロントエンド: 最低限のカード操作UI
- ログ: SQLite保存
- FL: なし（通常プレイのみ）
- **目標: 2タブで対戦してログが残る**

### Phase 2: FL統合
- Rustソルバー連携
- FL突入・Stay判定
- FL中の相手ボード非表示

### Phase 3: ログ活用
- JSONLエクスポート
- 前処理スクリプト（状態ベクトル化）
- Behavior Cloning への接続

### Phase 4: AI対戦
- AI プレイヤーをWebSocket クライアントとして接続
- BC → Self-Play → RYOとの対戦

---

## 12. 決定事項（旧・未決定事項）

1. **ルーム管理の永続性**: サーバー再起動でルームは消える前提。ログはSQLiteに永続化。
2. **プレイヤー認証**: ID手入力（ルーム参加時にプレイヤー名を入力）
3. **同時FL**: 両者ともソルバーで自動配置 → スコア計算
4. **タイムアウト**: 各ターン20秒。超過時は自動配置（ランダム or 左詰め）
5. **観戦機能**: 将来実装予定。WebSocket接続で観戦モード（読み取り専用）
6. **セッション間の連続性**: ゲーム終了後、同じルームで「ゲームスタート」ボタンを押して次のセッション開始
7. **FL中の終了判定**: 40点差チェックは両者非FL時のハンド開始前のみ。ただし持ち点が0以下になった場合はFL中でも即終了
