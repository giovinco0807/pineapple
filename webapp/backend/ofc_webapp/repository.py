"""SQLite persistence and JSONL replay export for the web game."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterator, Mapping
from uuid import uuid4

from ofc_regular.cards import validate_cards
from ofc_regular.state import Board

from .domain import (
    AppliedDecision,
    HandResult,
    HandState,
    HandStatus,
    MatchState,
    MatchStatus,
    Player,
    Seat,
    Street,
    Turn,
    board_to_dict,
    utc_now,
)
from .ports import AIMetadata


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DecisionRecord:
    id: str
    hand_id: str
    actor: Player
    street: Street
    seat: Seat
    dealt_cards: tuple[str, ...]
    placements: tuple[tuple[str, str], ...]
    discards: tuple[str, ...]
    think_ms: int
    created_at: str
    sequence_no: int
    ai_meta: AIMetadata | None = None

    def __post_init__(self) -> None:
        if not self.id or not self.hand_id:
            raise ValueError("decision id and hand id are required")
        if self.actor not in ("human", "ai"):
            raise ValueError("decision actor must be human or ai")
        if self.street not in ("T0", "T1", "T2", "T3", "T4", "FL"):
            raise ValueError("invalid decision street")
        if self.seat not in ("first", "second"):
            raise ValueError("invalid decision seat")
        if (
            isinstance(self.think_ms, bool)
            or not isinstance(self.think_ms, int)
            or self.think_ms < 0
        ):
            raise ValueError("think_ms must be a non-negative integer")
        if self.sequence_no < 0:
            raise ValueError("sequence_no cannot be negative")
        placed_cards = tuple(card for card, _row in self.placements)
        validate_cards(self.dealt_cards)
        validate_cards((*placed_cards, *self.discards))
        if set((*placed_cards, *self.discards)) != set(self.dealt_cards):
            raise ValueError("recorded action must account for every dealt card")
        if any(row not in ("top", "middle", "bottom") for _, row in self.placements):
            raise ValueError("recorded placement has an invalid row")
        if self.actor == "ai" and self.ai_meta is None:
            raise ValueError("every AI decision requires audit metadata")
        if self.actor == "human" and self.ai_meta is not None:
            raise ValueError("human decisions cannot carry AI metadata")

    @classmethod
    def from_applied(
        cls,
        decision: AppliedDecision,
        *,
        think_ms: int,
        ai_meta: AIMetadata | None = None,
        decision_id: str | None = None,
        created_at: str | None = None,
        hand_id: str,
    ) -> "DecisionRecord":
        return cls(
            id=decision_id or str(uuid4()),
            hand_id=hand_id,
            actor=decision.actor,
            street=decision.street,
            seat=decision.seat,
            dealt_cards=decision.dealt_cards,
            placements=decision.placements,
            discards=decision.discards,
            think_ms=think_ms,
            created_at=created_at or utc_now(),
            sequence_no=decision.sequence_no,
            ai_meta=ai_meta,
        )


class SQLiteRepository:
    """Small connection-per-operation repository suitable for FastAPI."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def initialize(self) -> None:
        with self._session() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode = WAL;
                PRAGMA foreign_keys = ON;

                CREATE TABLE IF NOT EXISTS schema_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS matches (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    human_stack INTEGER NOT NULL,
                    ai_stack INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    first_positions TEXT NOT NULL,
                    assembly_sha TEXT NOT NULL,
                    app_version TEXT NOT NULL,
                    state_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS hands (
                    id TEXT PRIMARY KEY,
                    match_id TEXT NOT NULL REFERENCES matches(id) ON DELETE CASCADE,
                    "index" INTEGER NOT NULL,
                    positions TEXT NOT NULL,
                    fl_human INTEGER NOT NULL,
                    fl_ai INTEGER NOT NULL,
                    deck_order_json TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    result_json TEXT,
                    state_json TEXT NOT NULL,
                    UNIQUE(match_id, "index")
                );

                CREATE TABLE IF NOT EXISTS decisions (
                    id TEXT PRIMARY KEY,
                    hand_id TEXT NOT NULL REFERENCES hands(id) ON DELETE CASCADE,
                    actor TEXT NOT NULL CHECK(actor IN ('human', 'ai')),
                    street TEXT NOT NULL CHECK(street IN ('T0','T1','T2','T3','T4','FL')),
                    seat TEXT NOT NULL CHECK(seat IN ('first', 'second')),
                    dealt_json TEXT NOT NULL,
                    placements_json TEXT NOT NULL,
                    discards_json TEXT NOT NULL,
                    think_ms INTEGER NOT NULL CHECK(think_ms >= 0),
                    created_at TEXT NOT NULL,
                    ai_meta_json TEXT,
                    sequence_no INTEGER NOT NULL,
                    UNIQUE(hand_id, sequence_no),
                    CHECK (
                        (actor = 'ai' AND ai_meta_json IS NOT NULL)
                        OR (actor = 'human' AND ai_meta_json IS NULL)
                    )
                );

                CREATE INDEX IF NOT EXISTS idx_hands_match
                    ON hands(match_id, "index");
                CREATE INDEX IF NOT EXISTS idx_decisions_hand
                    ON decisions(hand_id, sequence_no);
                """
            )
            connection.execute(
                """
                INSERT INTO schema_meta(key, value)
                VALUES ('schema_version', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (str(SCHEMA_VERSION),),
            )

    def save_match(self, match: MatchState) -> None:
        with self._session() as connection:
            self._upsert_match(connection, match)

    def get_match(self, match_id: str) -> MatchState | None:
        with self._session() as connection:
            row = connection.execute(
                "SELECT state_json FROM matches WHERE id = ?", (match_id,)
            ).fetchone()
        return None if row is None else _match_from_payload(_json_loads(row[0]))

    def list_matches(self) -> list[MatchState]:
        with self._session() as connection:
            rows = connection.execute(
                "SELECT state_json FROM matches ORDER BY created_at, id"
            ).fetchall()
        return [_match_from_payload(_json_loads(row[0])) for row in rows]

    def save_hand(self, hand: HandState) -> None:
        with self._session() as connection:
            self._upsert_hand(connection, hand)

    def get_hand(self, hand_id: str) -> HandState | None:
        with self._session() as connection:
            row = connection.execute(
                "SELECT state_json FROM hands WHERE id = ?", (hand_id,)
            ).fetchone()
        return None if row is None else _hand_from_payload(_json_loads(row[0]))

    def list_hands(self, match_id: str) -> list[HandState]:
        with self._session() as connection:
            rows = connection.execute(
                """
                SELECT state_json
                FROM hands
                WHERE match_id = ?
                ORDER BY "index"
                """,
                (match_id,),
            ).fetchall()
        return [_hand_from_payload(_json_loads(row[0])) for row in rows]

    def add_decision(self, decision: DecisionRecord) -> None:
        with self._session() as connection:
            self._insert_decision(connection, decision)

    def list_decisions(self, hand_id: str) -> list[DecisionRecord]:
        with self._session() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM decisions
                WHERE hand_id = ?
                ORDER BY sequence_no
                """,
                (hand_id,),
            ).fetchall()
        return [_decision_from_row(row) for row in rows]

    def count_decisions(self, hand_id: str) -> int:
        with self._session() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM decisions WHERE hand_id = ?",
                (hand_id,),
            ).fetchone()
        return int(row["count"])

    def save_transition(
        self,
        *,
        match: MatchState,
        hand: HandState,
        decision: DecisionRecord | None = None,
    ) -> None:
        """Atomically persist the latest state and optional action record."""

        if hand.match_id != match.id:
            raise ValueError("hand does not belong to match")
        if decision is not None and decision.hand_id != hand.id:
            raise ValueError("decision does not belong to hand")
        with self._session() as connection:
            self._upsert_match(connection, match)
            self._upsert_hand(connection, hand)
            if decision is not None:
                self._insert_decision(connection, decision)

    def export_jsonl(self, match_id: str) -> str:
        """Export one line per decision plus one summary per completed hand."""

        with self._session() as connection:
            match_row = connection.execute(
                "SELECT seed, assembly_sha, app_version FROM matches WHERE id = ?",
                (match_id,),
            ).fetchone()
            if match_row is None:
                raise KeyError(f"unknown match: {match_id}")
            hand_rows = connection.execute(
                """
                SELECT *
                FROM hands
                WHERE match_id = ?
                ORDER BY "index"
                """,
                (match_id,),
            ).fetchall()
            lines: list[str] = []
            for hand_row in hand_rows:
                decision_rows = connection.execute(
                    """
                    SELECT *
                    FROM decisions
                    WHERE hand_id = ?
                    ORDER BY sequence_no
                    """,
                    (hand_row["id"],),
                ).fetchall()
                for row in decision_rows:
                    lines.append(_json_dumps(_decision_export_payload(row)))
                if hand_row["result_json"] is not None:
                    lines.append(
                        _json_dumps(
                            {
                                "record_type": "hand_summary",
                                "schema_version": SCHEMA_VERSION,
                                "match_id": match_id,
                                "match_seed": match_row["seed"],
                                "assembly_sha": match_row["assembly_sha"],
                                "app_version": match_row["app_version"],
                                "hand_id": hand_row["id"],
                                "hand_index": hand_row["index"],
                                "positions": _json_loads(
                                    hand_row["positions"]
                                ),
                                "fl_human": bool(hand_row["fl_human"]),
                                "fl_ai": bool(hand_row["fl_ai"]),
                                "deck_order": _json_loads(
                                    hand_row["deck_order_json"]
                                ),
                                "started_at": hand_row["started_at"],
                                "ended_at": hand_row["ended_at"],
                                "result": _json_loads(
                                    hand_row["result_json"]
                                ),
                            }
                        )
                    )
        return "".join(f"{line}\n" for line in lines)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @contextmanager
    def _session(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def _upsert_match(
        connection: sqlite3.Connection, match: MatchState
    ) -> None:
        first_positions = match.positions_for_hand(0)
        connection.execute(
            """
            INSERT INTO matches(
                id, created_at, seed, human_stack, ai_stack, status,
                first_positions, assembly_sha, app_version, state_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                human_stack = excluded.human_stack,
                ai_stack = excluded.ai_stack,
                status = excluded.status,
                first_positions = excluded.first_positions,
                assembly_sha = excluded.assembly_sha,
                app_version = excluded.app_version,
                state_json = excluded.state_json
            """,
            (
                match.id,
                match.created_at,
                match.seed,
                match.stacks[0],
                match.stacks[1],
                match.status.value,
                _json_dumps(
                    {"human": first_positions[0], "ai": first_positions[1]}
                ),
                match.assembly_sha,
                match.app_version,
                _json_dumps(_match_to_payload(match)),
            ),
        )

    @staticmethod
    def _upsert_hand(
        connection: sqlite3.Connection, hand: HandState
    ) -> None:
        result_payload = (
            None
            if hand.result is None
            else _json_dumps(_result_to_payload(hand.result))
        )
        connection.execute(
            """
            INSERT INTO hands(
                id, match_id, "index", positions, fl_human, fl_ai,
                deck_order_json, started_at, ended_at, result_json, state_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                positions = excluded.positions,
                fl_human = excluded.fl_human,
                fl_ai = excluded.fl_ai,
                deck_order_json = excluded.deck_order_json,
                started_at = excluded.started_at,
                ended_at = excluded.ended_at,
                result_json = excluded.result_json,
                state_json = excluded.state_json
            """,
            (
                hand.id,
                hand.match_id,
                hand.index,
                _json_dumps(
                    {"human": hand.positions[0], "ai": hand.positions[1]}
                ),
                int(hand.fantasyland[0]),
                int(hand.fantasyland[1]),
                _json_dumps(list(hand.deck_order)),
                hand.started_at,
                hand.ended_at,
                result_payload,
                _json_dumps(_hand_to_payload(hand)),
            ),
        )

    @staticmethod
    def _insert_decision(
        connection: sqlite3.Connection, decision: DecisionRecord
    ) -> None:
        connection.execute(
            """
            INSERT INTO decisions(
                id, hand_id, actor, street, seat, dealt_json,
                placements_json, discards_json, think_ms, created_at,
                ai_meta_json, sequence_no
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                decision.id,
                decision.hand_id,
                decision.actor,
                decision.street,
                decision.seat,
                _json_dumps(list(decision.dealt_cards)),
                _json_dumps(
                    [[card, row] for card, row in decision.placements]
                ),
                _json_dumps(list(decision.discards)),
                decision.think_ms,
                decision.created_at,
                (
                    None
                    if decision.ai_meta is None
                    else _json_dumps(decision.ai_meta.to_dict())
                ),
                decision.sequence_no,
            ),
        )


def _match_to_payload(match: MatchState) -> dict[str, Any]:
    return {
        "id": match.id,
        "created_at": match.created_at,
        "seed": match.seed,
        "stacks": list(match.stacks),
        "status": match.status.value,
        "first_hand_first": match.first_hand_first,
        "assembly_sha": match.assembly_sha,
        "app_version": match.app_version,
        "hand_count": match.hand_count,
        "current_hand_id": match.current_hand_id,
        "pending_fantasyland": list(match.pending_fantasyland),
    }


def _match_from_payload(payload: Mapping[str, Any]) -> MatchState:
    return MatchState(
        id=str(payload["id"]),
        created_at=str(payload["created_at"]),
        seed=int(payload["seed"]),
        stacks=tuple(int(value) for value in payload["stacks"]),  # type: ignore[arg-type]
        status=MatchStatus(str(payload["status"])),
        first_hand_first=str(payload["first_hand_first"]),  # type: ignore[arg-type]
        assembly_sha=str(payload.get("assembly_sha", "")),
        app_version=str(payload.get("app_version", "")),
        hand_count=int(payload.get("hand_count", 0)),
        current_hand_id=(
            None
            if payload.get("current_hand_id") is None
            else str(payload["current_hand_id"])
        ),
        pending_fantasyland=tuple(
            bool(value) for value in payload.get("pending_fantasyland", (False, False))
        ),  # type: ignore[arg-type]
    )


def _hand_to_payload(hand: HandState) -> dict[str, Any]:
    return {
        "id": hand.id,
        "match_id": hand.match_id,
        "index": hand.index,
        "positions": list(hand.positions),
        "fantasyland": list(hand.fantasyland),
        "deck_order": list(hand.deck_order),
        "turns": [
            {
                "actor": turn.actor,
                "street": turn.street,
                "dealt_cards": list(turn.dealt_cards),
            }
            for turn in hand.turns
        ],
        "started_at": hand.started_at,
        "boards": [board_to_dict(board) for board in hand.boards],
        "private_discards": [
            list(cards) for cards in hand.private_discards
        ],
        "current_turn_index": hand.current_turn_index,
        "status": hand.status.value,
        "ended_at": hand.ended_at,
        "result": (
            None if hand.result is None else _result_to_payload(hand.result)
        ),
    }


def _hand_from_payload(payload: Mapping[str, Any]) -> HandState:
    raw_boards = payload["boards"]
    boards = tuple(
        Board.from_rows(
            top=board["top"],
            middle=board["middle"],
            bottom=board["bottom"],
        )
        for board in raw_boards
    )
    turns = tuple(
        Turn(
            actor=str(turn["actor"]),  # type: ignore[arg-type]
            street=str(turn["street"]),  # type: ignore[arg-type]
            dealt_cards=tuple(str(card) for card in turn["dealt_cards"]),
        )
        for turn in payload["turns"]
    )
    raw_result = payload.get("result")
    return HandState(
        id=str(payload["id"]),
        match_id=str(payload["match_id"]),
        index=int(payload["index"]),
        positions=tuple(str(value) for value in payload["positions"]),  # type: ignore[arg-type]
        fantasyland=tuple(bool(value) for value in payload["fantasyland"]),  # type: ignore[arg-type]
        deck_order=tuple(str(card) for card in payload["deck_order"]),
        turns=turns,
        started_at=str(payload["started_at"]),
        boards=boards,  # type: ignore[arg-type]
        private_discards=tuple(
            tuple(str(card) for card in cards)
            for cards in payload["private_discards"]
        ),  # type: ignore[arg-type]
        current_turn_index=int(payload["current_turn_index"]),
        status=HandStatus(str(payload["status"])),
        ended_at=(
            None
            if payload.get("ended_at") is None
            else str(payload["ended_at"])
        ),
        result=(
            None
            if raw_result is None
            else _result_from_payload(raw_result)
        ),
    )


def _result_to_payload(result: HandResult) -> dict[str, Any]:
    return {
        "raw_score": result.raw_score,
        "capped_score": result.capped_score,
        "breakdown": dict(result.breakdown),
        "stacks_after": list(result.stacks_after),
        "next_fantasyland": list(result.next_fantasyland),
    }


def _result_from_payload(payload: Mapping[str, Any]) -> HandResult:
    return HandResult(
        raw_score=int(payload["raw_score"]),
        capped_score=int(payload["capped_score"]),
        breakdown=dict(payload.get("breakdown", {})),
        stacks_after=tuple(int(value) for value in payload["stacks_after"]),  # type: ignore[arg-type]
        next_fantasyland=tuple(
            bool(value) for value in payload["next_fantasyland"]
        ),  # type: ignore[arg-type]
    )


def _decision_from_row(row: sqlite3.Row) -> DecisionRecord:
    raw_meta = (
        None
        if row["ai_meta_json"] is None
        else AIMetadata.from_dict(_json_loads(row["ai_meta_json"]))
    )
    return DecisionRecord(
        id=str(row["id"]),
        hand_id=str(row["hand_id"]),
        actor=str(row["actor"]),  # type: ignore[arg-type]
        street=str(row["street"]),  # type: ignore[arg-type]
        seat=str(row["seat"]),  # type: ignore[arg-type]
        dealt_cards=tuple(_json_loads(row["dealt_json"])),
        placements=tuple(
            (str(card), str(target_row))
            for card, target_row in _json_loads(row["placements_json"])
        ),
        discards=tuple(_json_loads(row["discards_json"])),
        think_ms=int(row["think_ms"]),
        created_at=str(row["created_at"]),
        sequence_no=int(row["sequence_no"]),
        ai_meta=raw_meta,
    )


def _decision_export_payload(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "record_type": "decision",
        "schema_version": SCHEMA_VERSION,
        "id": row["id"],
        "hand_id": row["hand_id"],
        "actor": row["actor"],
        "street": row["street"],
        "seat": row["seat"],
        "dealt": _json_loads(row["dealt_json"]),
        "placements": _json_loads(row["placements_json"]),
        "discards": _json_loads(row["discards_json"]),
        "think_ms": row["think_ms"],
        "created_at": row["created_at"],
        "sequence_no": row["sequence_no"],
        "ai_meta": (
            None
            if row["ai_meta_json"] is None
            else _json_loads(row["ai_meta_json"])
        ),
    }


def _json_dumps(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _json_loads(value: str) -> Any:
    return json.loads(value)


__all__ = [
    "DecisionRecord",
    "SCHEMA_VERSION",
    "SQLiteRepository",
]
