"""OFC Pineapple - Game Room Management"""
import asyncio
import time
from typing import Dict, Optional, List
from datetime import datetime

from fastapi import WebSocket

from .game_state import GameState

# Import SQLite logger
from .db.writer import LogWriter
db_logger = LogWriter("data/ofc_logs.db")

# Import FL solver
try:
    from .solver.fl_bridge import solve_fantasyland
    FL_SOLVER_AVAILABLE = True
except ImportError:
    FL_SOLVER_AVAILABLE = False
    def solve_fantasyland(cards):
        return None


class GameRoom:
    def __init__(self, room_id: str, is_ai_game: bool = False):
        self.room_id = room_id
        self.players: List[str] = []
        self.websockets: Dict[str, WebSocket] = {}
        self.game: Optional[GameState] = None
        self.start_votes = set()
        self.logs: List[dict] = []
        self.is_ai_game = is_ai_game
        self.ai_seat: int = 0  # AI joins first as seat 0, human as seat 1
        self.ai_lock = asyncio.Lock()  # prevent concurrent AI computations
        self.pending_next_hand = False
        self.next_hand_votes = set()
        self.last_activity = time.monotonic()

    async def broadcast(self, message: dict, exclude: Optional[str] = None):
        self.last_activity = time.monotonic()
        for player_id, ws in self.websockets.items():
            if player_id != exclude:
                try:
                    await ws.send_json(message)
                except Exception:
                    pass

    async def send_to_seat(self, seat: int, message: dict):
        if seat < len(self.players):
            player_id = self.players[seat]
            ws = self.websockets.get(player_id)
            if ws:
                try:
                    await ws.send_json(message)
                except Exception:
                    pass

    def log_turn(self, seat: int, placements: list, discard: Optional[str],
                 board_before: Optional[dict] = None, opp_board_before: Optional[dict] = None):
        if not self.game:
            return
        bs = board_before if board_before else {
            k: list(v) for k, v in self.game.boards[seat].items()
        }
        ob = opp_board_before if opp_board_before else {
            k: list(v) for k, v in self.game.boards[1-seat].items()
        }
        log_entry = {
            "session_id": self.game.session_id,
            "hand_id": self.game.hand_id,
            "hand_number": self.game.hands_played,
            "turn": self.game.turn,
            "player": seat,
            "btn": self.game.btn,
            "is_btn": seat == self.game.btn,
            "chips": self.game.chips.copy(),
            "board_self": bs,
            "board_opponent": ob,
            "dealt_cards": self.game.dealt_cards.get(seat, []),
            "known_discards_self": self.game.discards[seat].copy(),
            "action": {
                "placements": placements,
                "discard": discard
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        self.logs.append(log_entry)

        # Persist to SQLite
        try:
            db_logger.log_turn(
                hand_id=log_entry.get("hand_id", ""),
                turn=log_entry["turn"],
                player=seat,
                board_self=log_entry["board_self"],
                board_opponent=log_entry["board_opponent"],
                dealt_cards=log_entry["dealt_cards"],
                known_discards=log_entry["known_discards_self"],
                action_placements=placements,
                action_discard=discard,
                think_time_ms=0
            )
        except Exception as e:
            print(f"[WARN] Failed to log turn to SQLite: {e}")


rooms: Dict[str, GameRoom] = {}

ROOM_TTL_SECONDS = 3600  # 1 hour


async def cleanup_stale_rooms():
    """Periodically remove rooms with no activity for ROOM_TTL_SECONDS."""
    while True:
        await asyncio.sleep(300)  # check every 5 minutes
        now = time.monotonic()
        stale = [rid for rid, r in rooms.items()
                 if now - r.last_activity > ROOM_TTL_SECONDS and not r.websockets]
        for rid in stale:
            del rooms[rid]
        if stale:
            print(f"[CLEANUP] Removed {len(stale)} stale rooms: {stale}")
