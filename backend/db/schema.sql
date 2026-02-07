-- OFC Pineapple Log Schema

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    room_id TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    player0_id TEXT NOT NULL,
    player1_id TEXT NOT NULL,
    initial_chips INTEGER NOT NULL DEFAULT 200,
    final_chips_p0 INTEGER,
    final_chips_p1 INTEGER,
    winner INTEGER,
    finish_reason TEXT,
    hands_played INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS hands (
    hand_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(session_id),
    hand_number INTEGER NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    btn INTEGER NOT NULL,
    chips_before TEXT NOT NULL,
    chips_after TEXT,
    raw_score TEXT,
    actual_score TEXT,
    result_detail TEXT
);

CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hand_id TEXT NOT NULL REFERENCES hands(hand_id),
    turn INTEGER NOT NULL,
    player INTEGER NOT NULL,
    
    board_self TEXT NOT NULL,
    board_opponent TEXT NOT NULL,
    dealt_cards TEXT NOT NULL,
    known_discards TEXT NOT NULL,
    
    action_placements TEXT NOT NULL,
    action_discard TEXT,
    
    think_time_ms INTEGER,
    timestamp TEXT NOT NULL,
    
    UNIQUE(hand_id, turn, player)
);

CREATE TABLE IF NOT EXISTS fantasyland (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hand_id TEXT NOT NULL REFERENCES hands(hand_id),
    player INTEGER NOT NULL,
    card_count INTEGER NOT NULL,
    dealt_cards TEXT NOT NULL,
    solver_placement TEXT NOT NULL,
    solver_score REAL,
    fl_stay INTEGER NOT NULL,
    timestamp TEXT NOT NULL
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_hands_session ON hands(session_id);
CREATE INDEX IF NOT EXISTS idx_turns_hand ON turns(hand_id);
CREATE INDEX IF NOT EXISTS idx_fl_hand ON fantasyland(hand_id);
