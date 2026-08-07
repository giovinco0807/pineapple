"""SQLite persistence for trainer mistakes and hand history."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_ACCOUNT = "default"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS mistakes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    session_id TEXT NOT NULL,
    hand_no INTEGER NOT NULL DEFAULT 0,
    street INTEGER NOT NULL,
    position TEXT NOT NULL,
    hero_board TEXT NOT NULL,
    opp_board TEXT NOT NULL,
    dealt TEXT NOT NULL,
    dead_cards TEXT NOT NULL,
    user_action TEXT NOT NULL,
    best_action TEXT NOT NULL,
    ev_loss REAL NOT NULL,
    user_rank INTEGER,
    candidates TEXT NOT NULL,
    evaluator TEXT NOT NULL DEFAULT '',
    retried INTEGER NOT NULL DEFAULT 0,
    solved INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_mistakes_created ON mistakes (created_at DESC);

CREATE TABLE IF NOT EXISTS hands (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    session_id TEXT NOT NULL,
    position TEXT NOT NULL,
    score REAL,
    total_ev_loss REAL,
    decisions INTEGER,
    mistakes INTEGER,
    summary TEXT
);
"""


class TrainerStore:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            self._migrate(conn)
        self.default_account_id = self.get_or_create_account(DEFAULT_ACCOUNT)["id"]

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------------
    # migration
    # ------------------------------------------------------------------

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Add account_id to pre-account databases and adopt existing rows.

        Rows written before accounts existed all belong to whoever was using
        the tool, so they are backfilled onto the default account rather than
        being orphaned.
        """
        for table in ("mistakes", "hands"):
            columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
            if "account_id" not in columns:
                conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN account_id INTEGER NOT NULL DEFAULT 1"
                )
        row = conn.execute("SELECT id FROM accounts ORDER BY id LIMIT 1").fetchone()
        if row is None:
            conn.execute(
                "INSERT INTO accounts (id, name, created_at) VALUES (1, ?, ?)",
                (DEFAULT_ACCOUNT, time.time()),
            )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_mistakes_account ON mistakes (account_id, created_at DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_hands_account ON hands (account_id, created_at DESC)"
        )

    # ------------------------------------------------------------------
    # accounts
    # ------------------------------------------------------------------

    def list_accounts(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT a.id, a.name, a.created_at,
                          (SELECT COUNT(*) FROM hands h WHERE h.account_id = a.id) AS hands,
                          (SELECT COUNT(*) FROM mistakes m WHERE m.account_id = a.id) AS mistakes
                   FROM accounts a ORDER BY a.id"""
            ).fetchall()
        return [dict(row) for row in rows]

    def get_or_create_account(self, name: str) -> Dict[str, Any]:
        name = (name or "").strip()
        if not name:
            raise ValueError("アカウント名を入力してください")
        if len(name) > 40:
            raise ValueError("アカウント名は40文字までです")
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT id, name FROM accounts WHERE name = ?", (name,)).fetchone()
            if row is not None:
                return dict(row)
            cur = conn.execute(
                "INSERT INTO accounts (name, created_at) VALUES (?, ?)", (name, time.time())
            )
            return {"id": int(cur.lastrowid), "name": name}

    def account_exists(self, account_id: int) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT 1 FROM accounts WHERE id = ?", (account_id,)).fetchone()
        return row is not None

    def delete_account(self, account_id: int) -> None:
        """Remove an account and everything recorded under it."""
        if account_id == self.default_account_id:
            raise ValueError("既定アカウントは削除できません")
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM mistakes WHERE account_id = ?", (account_id,))
            conn.execute("DELETE FROM hands WHERE account_id = ?", (account_id,))
            conn.execute("DELETE FROM accounts WHERE id = ?", (account_id,))

    def add_mistake(
        self,
        *,
        account_id: int,
        session_id: str,
        hand_no: int,
        street: int,
        position: str,
        hero_board: Dict[str, List[str]],
        opp_board: Dict[str, List[str]],
        dealt: List[str],
        dead_cards: List[str],
        user_action: Dict[str, Any],
        best_action: Dict[str, Any],
        ev_loss: float,
        user_rank: Optional[int],
        candidates: List[Dict[str, Any]],
        evaluator: str = "",
    ) -> int:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO mistakes
                   (created_at, account_id, session_id, hand_no, street, position, hero_board,
                    opp_board, dealt, dead_cards, user_action, best_action,
                    ev_loss, user_rank, candidates, evaluator)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    time.time(),
                    account_id,
                    session_id,
                    hand_no,
                    street,
                    position,
                    json.dumps(hero_board),
                    json.dumps(opp_board),
                    json.dumps(dealt),
                    json.dumps(dead_cards),
                    json.dumps(user_action),
                    json.dumps(best_action),
                    float(ev_loss),
                    user_rank,
                    json.dumps(candidates),
                    evaluator,
                ),
            )
            return int(cur.lastrowid)

    def list_mistakes(self, account_id: int, limit: int = 200) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT id, created_at, session_id, hand_no, street, position,
                          hero_board, opp_board, dealt, user_action, best_action,
                          ev_loss, user_rank, evaluator, retried, solved
                   FROM mistakes WHERE account_id = ?
                   ORDER BY created_at DESC LIMIT ?""",
                (account_id, limit),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            for key in ("hero_board", "opp_board", "dealt", "user_action", "best_action"):
                item[key] = json.loads(item[key])
            result.append(item)
        return result

    def get_mistake(self, mistake_id: int, account_id: int) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM mistakes WHERE id = ? AND account_id = ?",
                (mistake_id, account_id),
            ).fetchone()
        if row is None:
            return None
        item = dict(row)
        for key in (
            "hero_board",
            "opp_board",
            "dealt",
            "dead_cards",
            "user_action",
            "best_action",
            "candidates",
        ):
            item[key] = json.loads(item[key])
        return item

    def mark_retry(self, mistake_id: int, account_id: int, solved: bool) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """UPDATE mistakes SET retried = retried + 1, solved = ?
                   WHERE id = ? AND account_id = ?""",
                (1 if solved else 0, mistake_id, account_id),
            )

    def delete_mistake(self, mistake_id: int, account_id: int) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "DELETE FROM mistakes WHERE id = ? AND account_id = ?", (mistake_id, account_id)
            )

    def add_hand(
        self,
        *,
        account_id: int,
        session_id: str,
        position: str,
        score: Optional[float],
        total_ev_loss: Optional[float],
        decisions: int,
        mistakes: int,
        summary: Dict[str, Any],
    ) -> int:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO hands
                   (created_at, account_id, session_id, position, score, total_ev_loss,
                    decisions, mistakes, summary)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    time.time(),
                    account_id,
                    session_id,
                    position,
                    score,
                    total_ev_loss,
                    decisions,
                    mistakes,
                    json.dumps(summary),
                ),
            )
            return int(cur.lastrowid)

    def list_hands(self, account_id: int, limit: int = 200) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT id, created_at, session_id, position, score, total_ev_loss,
                          decisions, mistakes, summary
                   FROM hands WHERE account_id = ?
                   ORDER BY created_at DESC LIMIT ?""",
                (account_id, limit),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            try:
                item["summary"] = json.loads(item["summary"]) if item["summary"] else None
            except (TypeError, ValueError):
                item["summary"] = None
            result.append(item)
        return result

    def stats(self, account_id: int) -> Dict[str, Any]:
        with self._connect() as conn:
            hands = conn.execute(
                """SELECT COUNT(*) AS n, COALESCE(SUM(score), 0) AS score,
                          COALESCE(SUM(total_ev_loss), 0) AS ev_loss,
                          COALESCE(SUM(decisions), 0) AS decisions,
                          COALESCE(SUM(mistakes), 0) AS mistakes
                   FROM hands WHERE account_id = ?""",
                (account_id,),
            ).fetchone()
            open_mistakes = conn.execute(
                "SELECT COUNT(*) AS n FROM mistakes WHERE account_id = ? AND solved = 0",
                (account_id,),
            ).fetchone()
        return {
            "hands": hands["n"],
            "total_score": hands["score"],
            "total_ev_loss": hands["ev_loss"],
            "decisions": hands["decisions"],
            "mistakes": hands["mistakes"],
            "open_mistakes": open_mistakes["n"],
        }
