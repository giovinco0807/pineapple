"""
SQLite Log Writer for OFC Pineapple
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import Optional


class LogWriter:
    def __init__(self, db_path: str = "data/ofc_logs.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        schema_path = Path(__file__).parent / "schema.sql"
        with open(schema_path) as f:
            schema = f.read()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(schema)
    
    def log_session_start(self, session_id: str, room_id: str, 
                          player0_id: str, player1_id: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO sessions (session_id, room_id, started_at, 
                                      player0_id, player1_id, initial_chips)
                VALUES (?, ?, ?, ?, ?, 200)
            """, (session_id, room_id, datetime.utcnow().isoformat(),
                  player0_id, player1_id))
    
    def log_session_end(self, session_id: str, winner: int, 
                        final_chips: list[int], reason: str, hands_played: int):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE sessions 
                SET finished_at = ?, winner = ?, final_chips_p0 = ?,
                    final_chips_p1 = ?, finish_reason = ?, hands_played = ?
                WHERE session_id = ?
            """, (datetime.utcnow().isoformat(), winner, final_chips[0],
                  final_chips[1], reason, hands_played, session_id))
    
    def log_hand_start(self, hand_id: str, session_id: str, 
                       hand_number: int, btn: int, chips: list[int]):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO hands (hand_id, session_id, hand_number, 
                                   started_at, btn, chips_before)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (hand_id, session_id, hand_number, 
                  datetime.utcnow().isoformat(), btn, json.dumps(chips)))
    
    def log_hand_end(self, hand_id: str, chips_after: list[int],
                     raw_score: list[int], actual_score: list[int], 
                     result_detail: dict):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE hands
                SET finished_at = ?, chips_after = ?, raw_score = ?,
                    actual_score = ?, result_detail = ?
                WHERE hand_id = ?
            """, (datetime.utcnow().isoformat(), json.dumps(chips_after),
                  json.dumps(raw_score), json.dumps(actual_score),
                  json.dumps(result_detail), hand_id))
    
    def log_turn(self, hand_id: str, turn: int, player: int,
                 board_self: dict, board_opponent: dict,
                 dealt_cards: list[str], known_discards: list[str],
                 action_placements: list, action_discard: Optional[str],
                 think_time_ms: int):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO turns (hand_id, turn, player, board_self,
                                   board_opponent, dealt_cards, known_discards,
                                   action_placements, action_discard,
                                   think_time_ms, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (hand_id, turn, player, json.dumps(board_self),
                  json.dumps(board_opponent), json.dumps(dealt_cards),
                  json.dumps(known_discards), json.dumps(action_placements),
                  action_discard, think_time_ms, datetime.utcnow().isoformat()))
    
    def log_fantasyland(self, hand_id: str, player: int, card_count: int,
                        dealt_cards: list[str], solver_placement: dict,
                        solver_score: float, fl_stay: bool):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO fantasyland (hand_id, player, card_count,
                                         dealt_cards, solver_placement,
                                         solver_score, fl_stay, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (hand_id, player, card_count, json.dumps(dealt_cards),
                  json.dumps(solver_placement), solver_score,
                  1 if fl_stay else 0, datetime.utcnow().isoformat()))
    
    def export_to_jsonl(self, output_path: str = "data/logs/training_data.jsonl"):
        """Export turns to JSONL for AI training (BC-compatible format)."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Load hand results (busted, royalties, fl_entry, raw_score)
            hand_results = {}
            hand_cursor = conn.execute("""
                SELECT hand_id, result_detail, raw_score
                FROM hands WHERE result_detail IS NOT NULL
            """)
            for hrow in hand_cursor:
                try:
                    detail = json.loads(hrow["result_detail"]) if hrow["result_detail"] else {}
                    raw = json.loads(hrow["raw_score"]) if hrow["raw_score"] else [0, 0]
                    hand_results[hrow["hand_id"]] = {
                        "busted": detail.get("busted", [False, False]),
                        "royalties": detail.get("royalties", [
                            {"top": 0, "middle": 0, "bottom": 0, "total": 0},
                            {"top": 0, "middle": 0, "bottom": 0, "total": 0},
                        ]),
                        "fl_entry": detail.get("fl_entry", [False, False]),
                        "raw_score": raw,
                    }
                except Exception:
                    pass

            cursor = conn.execute("""
                SELECT t.*, h.hand_number, h.btn, s.session_id
                FROM turns t
                JOIN hands h ON t.hand_id = h.hand_id
                JOIN sessions s ON h.session_id = s.session_id
                ORDER BY t.timestamp
            """)
            
            count = 0
            with open(output_path, 'w') as f:
                for row in cursor:
                    hid = row["hand_id"]
                    hr = hand_results.get(hid)
                    if not hr:
                        continue  # Skip turns without hand result

                    record = {
                        "turn_log": {
                            "turn": row["turn"],
                            "player": row["player"],
                            "is_btn": row["player"] == row["btn"],
                            "board_self": json.loads(row["board_self"]),
                            "board_opponent": json.loads(row["board_opponent"]),
                            "dealt_cards": json.loads(row["dealt_cards"]),
                            "discards_self": json.loads(row["known_discards"]),
                            "action": {
                                "placements": json.loads(row["action_placements"]),
                                "discard": row["action_discard"]
                            },
                        },
                        "hand_result": hr,
                    }
                    f.write(json.dumps(record) + "\n")
                    count += 1
        
        print(f"Exported {count} turns to {output_path}")
        return output_path
