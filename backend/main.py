"""
OFC Pineapple Web App - FastAPI Backend with Full Game Flow
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
import uuid
import random
import time
from typing import Dict, Optional, List
from datetime import datetime
import sys
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import SQLite logger
from db.writer import LogWriter
db_logger = LogWriter("data/ofc_logs.db")

# Import FL solver
try:
    from solver.fl_bridge import solve_fantasyland
    FL_SOLVER_AVAILABLE = True
except ImportError:
    FL_SOLVER_AVAILABLE = False
    def solve_fantasyland(cards):
        return None

app = FastAPI(title="OFC Pineapple", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== CARD CONSTANTS ==========
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
SUITS = ['h', 'd', 'c', 's']


def create_deck(include_jokers: bool = True) -> List[str]:
    deck = [f"{r}{s}" for s in SUITS for r in RANKS]
    if include_jokers:
        deck.extend(["X1", "X2"])
    random.shuffle(deck)
    return deck


# ========== GAME STATE ==========
class GameState:
    def __init__(self, room_id: str, players: List[str]):
        self.session_id = str(uuid.uuid4())
        self.room_id = room_id
        self.players = players
        self.chips = [200, 200]
        self.btn = random.randint(0, 1)
        self.hands_played = 0
        
        # Current hand state
        self.hand_id: Optional[str] = None
        self.deck: List[str] = []
        self.turn = 0
        self.current_player = 0
        self.boards = [
            {"top": [], "middle": [], "bottom": []},
            {"top": [], "middle": [], "bottom": []}
        ]
        self.dealt_cards: Dict[int, List[str]] = {}
        self.discards = [[], []]
        self.placed_this_turn = [False, False]
        self.turn_start_times = [0.0, 0.0]
        
        # FL state
        self.is_fantasyland = [False, False]
        self.fl_card_count = [0, 0]
    
    def start_hand(self) -> dict:
        """Start a new hand. Handle Fantasyland mode."""
        self.hand_id = str(uuid.uuid4())
        self.deck = create_deck(include_jokers=True)
        self.turn = 0
        self.current_player = self.btn
        self.boards = [
            {"top": [], "middle": [], "bottom": []},
            {"top": [], "middle": [], "bottom": []}
        ]
        self.dealt_cards = {}
        self.discards = [[], []]
        self.placed_this_turn = [False, False]
        self.hands_played += 1
        
        # Check FL mode - deal more cards to FL player
        card_idx = 0
        for seat in [0, 1]:
            if self.is_fantasyland[seat] and self.fl_card_count[seat] > 0:
                # FL player gets 14-17 cards
                fl_cards = self.fl_card_count[seat]
                self.dealt_cards[seat] = self.deck[card_idx:card_idx + fl_cards]
                card_idx += fl_cards
                print(f"[DEBUG] FL: Seat {seat} dealt {fl_cards} cards")
            else:
                # Normal player gets 5 cards
                self.dealt_cards[seat] = self.deck[card_idx:card_idx + 5]
                card_idx += 5
        self.deck = self.deck[card_idx:]
        
        return {
            "hand_id": self.hand_id,
            "hand_number": self.hands_played,
            "btn": self.btn,
            "chips": self.chips.copy(),
            "is_fantasyland": self.is_fantasyland.copy(),
            "fl_card_count": self.fl_card_count.copy()
        }
    
    def deal_turn(self) -> Dict[int, List[str]]:
        """Deal 3 cards to each player for regular turn. Skip FL players."""
        self.turn += 1
        self.placed_this_turn = [False, False]
        self.current_player = self.btn
        
        for seat in [0, 1]:
            if self.is_fantasyland[seat] and self.is_board_complete(seat):
                # FL player already done - skip dealing, mark as placed
                self.dealt_cards[seat] = []
                self.placed_this_turn[seat] = True
            else:
                self.dealt_cards[seat] = self.deck[:3]
                self.deck = self.deck[3:]
        
        return self.dealt_cards
    
    def is_board_complete(self, seat: int) -> bool:
        board = self.boards[seat]
        return len(board["top"]) == 3 and len(board["middle"]) == 5 and len(board["bottom"]) == 5
    
    def apply_placement(self, seat: int, placements: List[List[str]], 
                       discard: Optional[str] = None) -> bool:
        """Apply player's card placement."""
        board = self.boards[seat]
        
        for card, position in placements:
            if position == "top" and len(board["top"]) < 3:
                board["top"].append(card)
            elif position == "middle" and len(board["middle"]) < 5:
                board["middle"].append(card)
            elif position == "bottom" and len(board["bottom"]) < 5:
                board["bottom"].append(card)
            else:
                return False
        
        if discard:
            self.discards[seat].append(discard)
        
        self.placed_this_turn[seat] = True
        return True
    
    def is_turn_complete(self) -> bool:
        return all(self.placed_this_turn)
    
    def is_hand_complete(self) -> bool:
        for board in self.boards:
            if len(board["top"]) != 3 or len(board["middle"]) != 5 or len(board["bottom"]) != 5:
                return False
        return True
    
    def next_btn(self):
        self.btn = 1 - self.btn


# ========== ROOM MANAGEMENT ==========
class GameRoom:
    def __init__(self, room_id: str):
        self.room_id = room_id
        self.players: List[str] = []
        self.websockets: Dict[str, WebSocket] = {}
        self.game: Optional[GameState] = None
        self.start_votes = set()
        self.logs: List[dict] = []
    
    async def broadcast(self, message: dict, exclude: Optional[str] = None):
        for player_id, ws in self.websockets.items():
            if player_id != exclude:
                try:
                    await ws.send_json(message)
                except:
                    pass
    
    async def send_to_seat(self, seat: int, message: dict):
        if seat < len(self.players):
            player_id = self.players[seat]
            ws = self.websockets.get(player_id)
            if ws:
                try:
                    await ws.send_json(message)
                except:
                    pass
    
    def log_turn(self, seat: int, placements: list, discard: Optional[str],
                 board_before: Optional[dict] = None, opp_board_before: Optional[dict] = None):
        if not self.game:
            return
        # Use pre-captured board if provided, else current (deep copy)
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


# ========== API ROUTES ==========
@app.get("/")
async def root():
    return {"message": "OFC Pineapple Server", "status": "running"}


@app.post("/api/rooms")
async def create_room():
    room_id = str(uuid.uuid4())[:8]
    rooms[room_id] = GameRoom(room_id)
    return {"room_id": room_id}


@app.get("/api/rooms")
async def list_rooms():
    return {
        "rooms": [
            {"room_id": r.room_id, "players": len(r.players),
             "status": "waiting" if len(r.players) < 2 else "full"}
            for r in rooms.values()
        ]
    }


@app.get("/api/logs/{room_id}")
async def get_logs(room_id: str):
    room = rooms.get(room_id)
    if not room:
        return {"error": "Room not found"}
    return {"logs": room.logs}


@app.get("/api/export")
async def export_logs():
    all_logs = []
    for room in rooms.values():
        all_logs.extend(room.logs)
    return {"logs": all_logs, "count": len(all_logs)}


# ========== WEBSOCKET ==========
@app.websocket("/ws/{room_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str, player_id: Optional[str] = None):
    await websocket.accept()
    
    if room_id not in rooms:
        rooms[room_id] = GameRoom(room_id)
    room = rooms[room_id]
    
    if not player_id:
        player_id = str(uuid.uuid4())[:8]
    
    if len(room.players) >= 2 and player_id not in room.players:
        await websocket.send_json({"type": "error", "message": "Room is full"})
        await websocket.close()
        return
    
    if player_id not in room.players:
        room.players.append(player_id)
    room.websockets[player_id] = websocket
    seat = room.players.index(player_id)
    
    await websocket.send_json({
        "type": "connected",
        "room_id": room_id,
        "player_id": player_id,
        "seat": seat,
        "players_in_room": len(room.players)
    })
    
    await room.broadcast({
        "type": "player_joined",
        "player_id": player_id,
        "seat": seat,
        "players_in_room": len(room.players)
    }, exclude=player_id)
    
    if len(room.players) == 2:
        await room.broadcast({"type": "ready_to_start", "players": room.players})
    
    try:
        while True:
            data = await websocket.receive_json()
            await handle_message(room, player_id, seat, data)
    except WebSocketDisconnect:
        if player_id in room.websockets:
            del room.websockets[player_id]
        await room.broadcast({"type": "player_disconnected", "player_id": player_id})


async def handle_message(room: GameRoom, player_id: str, seat: int, data: dict):
    msg_type = data.get("type")
    
    if msg_type == "game_start":
        room.start_votes.add(player_id)
        
        if len(room.start_votes) >= 2:
            room.start_votes.clear()
            room.game = GameState(room.room_id, room.players)
            hand_info = room.game.start_hand()
            
            # Log session and hand start to DB
            try:
                db_logger.log_session_start(
                    room.game.session_id, room.room_id,
                    room.players[0], room.players[1] if len(room.players) > 1 else "")
                db_logger.log_hand_start(
                    room.game.hand_id, room.game.session_id,
                    room.game.hands_played, room.game.btn, room.game.chips)
            except Exception as e:
                print(f"[WARN] DB log session/hand start: {e}")
            
            await room.broadcast({
                "type": "session_start",
                "session_id": room.game.session_id,
                "chips": room.game.chips
            })
            
            # Send hand start to each player with their cards
            for s in [0, 1]:
                await room.send_to_seat(s, {
                    "type": "deal",
                    "turn": 0,
                    "cards": room.game.dealt_cards[s],
                    "your_seat": s,
                    "btn": room.game.btn,
                    "opponent_board": room.game.boards[1-s]
                })
        else:
            await room.broadcast({
                "type": "waiting_for_start",
                "votes": len(room.start_votes)
            })
    
    elif msg_type == "place":
        if not room.game:
            return
        
        placements = data.get("placements", [])
        discard = data.get("discard")
        
        # Capture pre-action board state (deep copy - inner lists too)
        board_before = {
            k: list(v) for k, v in room.game.boards[seat].items()
        }
        opp_board_before = {
            k: list(v) for k, v in room.game.boards[1-seat].items()
        }
        
        # Apply placement
        if room.game.apply_placement(seat, placements, discard):
            # Log with pre-action board state
            room.log_turn(seat, placements, discard,
                          board_before=board_before,
                          opp_board_before=opp_board_before)
            
            # Notify opponent (hide FL board)
            opponent_seat = 1 - seat
            if room.game.is_fantasyland[seat]:
                # FL: don't reveal board to opponent
                pass
            else:
                new_board = room.game.boards[seat]
                await room.send_to_seat(opponent_seat, {
                    "type": "opponent_placed",
                    "opponent_board": new_board
                })
            
            # Check if turn complete
            print(f"[DEBUG] Seat {seat} placed. placed_this_turn: {room.game.placed_this_turn}")
            if room.game.is_turn_complete():
                print(f"[DEBUG] Turn complete! turn={room.game.turn}, checking hand complete...")
                if room.game.is_hand_complete():
                    # Hand complete - score it
                    result = calculate_scores(room.game)
                    
                    # Log hand end to DB
                    try:
                        db_logger.log_hand_end(
                            room.game.hand_id, room.game.chips,
                            result["raw_score"], result["actual_score"], result)
                    except Exception as e:
                        print(f"[WARN] DB log hand end: {e}")
                    
                    await room.broadcast({
                        "type": "hand_end",
                        "result": result
                    })
                    
                    # Check session end
                    end_check = check_session_end(room.game)
                    if end_check:
                        await room.broadcast({
                            "type": "session_end",
                            **end_check
                        })
                    else:
                        # Wait for "next_hand" message from players
                        room.pending_next_hand = True
                        room.next_hand_votes = set()
                else:
                    # Deal next turn
                    room.game.deal_turn()
                    for s in [0, 1]:
                        await room.send_to_seat(s, {
                            "type": "deal",
                            "turn": room.game.turn,
                            "cards": room.game.dealt_cards[s],
                            "opponent_board": room.game.boards[1-s]
                        })
        else:
            await room.send_to_seat(seat, {"type": "error", "message": "Invalid placement"})
    
    elif msg_type == "next_hand":
        if not room.game or not getattr(room, 'pending_next_hand', False):
            return
        
        room.next_hand_votes = getattr(room, 'next_hand_votes', set())
        room.next_hand_votes.add(seat)
        
        if len(room.next_hand_votes) >= 2:
            room.pending_next_hand = False
            room.next_hand_votes = set()
            
            # Start next hand
            room.game.next_btn()
            hand_info = room.game.start_hand()
            
            # Log hand start to DB
            try:
                db_logger.log_hand_start(
                    room.game.hand_id, room.game.session_id,
                    room.game.hands_played, room.game.btn, room.game.chips)
            except Exception as e:
                print(f"[WARN] DB log hand start: {e}")
            
            # Auto-solve FL hands
            for s in [0, 1]:
                if room.game.is_fantasyland[s]:
                    fl_result = solve_fantasyland(room.game.dealt_cards[s])
                    if fl_result:
                        print(f"[DEBUG] FL Solver result for seat {s}: {fl_result}")
                        room.game.boards[s] = {
                            "top": fl_result["top"],
                            "middle": fl_result["middle"],
                            "bottom": fl_result["bottom"]
                        }
                        room.game.placed_this_turn[s] = True
                        room.log_turn(s, [[c, "auto"] for c in fl_result["top"] + fl_result["middle"] + fl_result["bottom"]], None)
            
            # Check if both players are already done (both FL)
            if room.game.placed_this_turn[0] and room.game.placed_this_turn[1]:
                # Both FL auto-solved - send fl_solved then immediately score
                for s in [0, 1]:
                    await room.send_to_seat(s, {
                        "type": "fl_solved",
                        "board": room.game.boards[s],
                        "message": "🎰 FL自動配置完了！"
                    })
                # Score immediately
                result = calculate_scores(room.game)
                
                # Log hand end to DB
                try:
                    db_logger.log_hand_end(
                        room.game.hand_id, room.game.chips,
                        result["raw_score"], result["actual_score"], result)
                except Exception as e:
                    print(f"[WARN] DB log hand end: {e}")
                
                await room.broadcast({
                    "type": "hand_end",
                    "result": result
                })
                end_check = check_session_end(room.game)
                if end_check:
                    await room.broadcast({"type": "session_end", **end_check})
                else:
                    room.pending_next_hand = True
                    room.next_hand_votes = set()
            else:
                # Only one or no FL - send messages normally
                for s in [0, 1]:
                    is_fl = room.game.is_fantasyland[s]
                    fl_cards = room.game.fl_card_count[s] if is_fl else 0
                    
                    if is_fl and room.game.placed_this_turn[s]:
                        await room.send_to_seat(s, {
                            "type": "fl_solved",
                            "board": room.game.boards[s],
                            "message": "🎰 FL自動配置完了！"
                        })
                    else:
                        await room.send_to_seat(s, {
                            "type": "deal",
                            "turn": 0,
                            "cards": room.game.dealt_cards[s],
                            "your_seat": s,
                            "btn": room.game.btn,
                            "opponent_board": {"top": [], "middle": [], "bottom": []},
                            "is_fantasyland": is_fl,
                            "fl_card_count": fl_cards
                        })


def _evaluate_with_joker_constraint(cards: list, expected_count: int, 
                                     max_value: int) -> int:
    """Evaluate hand with joker bust-prevention.
    
    Joker picks the strongest hand that doesn't exceed max_value.
    If no joker is present, or best hand is already ≤ max_value, returns normal eval.
    If even the weakest joker substitution exceeds max_value, player is genuinely busted.
    """
    jokers = [c for c in cards if c in ("X1", "X2")]
    if not jokers:
        return evaluate_hand(cards, expected_count)
    
    best_val = evaluate_hand(cards, expected_count)
    if best_val <= max_value:
        return best_val  # No constraint violation
    
    # Try all possible joker substitutions to find best ≤ max_value
    non_joker = [c for c in cards if c not in ("X1", "X2")]
    card_set = set(cards)
    
    RANKS = "23456789TJQKA"
    SUITS = "hdcs"
    
    best_constrained = -1
    
    if len(jokers) == 1:
        for r in RANKS:
            for s in SUITS:
                sub = r + s
                if sub in card_set:
                    continue
                test = non_joker + [sub]
                val = evaluate_hand(test, expected_count)
                if val <= max_value and val > best_constrained:
                    best_constrained = val
    else:
        # 2 jokers (rare but possible)
        all_subs = [r + s for r in RANKS for s in SUITS if r + s not in card_set]
        for i, sub1 in enumerate(all_subs):
            for sub2 in all_subs[i+1:]:
                test = non_joker + [sub1, sub2]
                val = evaluate_hand(test, expected_count)
                if val <= max_value and val > best_constrained:
                    best_constrained = val
    
    # If no valid substitution found, player is genuinely busted
    return best_constrained if best_constrained >= 0 else best_val


def calculate_scores(game: GameState) -> dict:
    """Calculate hand scores with full implementation."""
    from collections import Counter
    
    fl_entry = [False, False]
    fl_card_count = [0, 0]
    busted = [False, False]
    royalties = [
        {"top": 0, "middle": 0, "bottom": 0, "total": 0},
        {"top": 0, "middle": 0, "bottom": 0, "total": 0}
    ]
    
    # Evaluate hands with joker bust-prevention:
    # Joker picks the strongest hand that doesn't violate ordering.
    # Evaluate bottom-up: bottom (unconstrained), middle (≤ bottom), top (≤ middle)
    hand_values = [{}, {}]
    for seat in [0, 1]:
        board = game.boards[seat]
        
        # Bottom: unconstrained (always best hand)
        bot_val = evaluate_hand(board["bottom"], 5)
        
        # Middle: must be ≤ bottom
        mid_val = _evaluate_with_joker_constraint(
            board["middle"], 5, max_value=bot_val
        )
        
        # Top: must be ≤ middle
        top_val = _evaluate_with_joker_constraint(
            board["top"], 3, max_value=mid_val
        )
        
        hand_values[seat] = {"top": top_val, "middle": mid_val, "bottom": bot_val}
        
        # Check bust: top <= middle <= bottom (higher value = stronger hand)
        if top_val > mid_val or mid_val > bot_val:
            busted[seat] = True
        else:
            # Calculate royalties (only if not busted)
            royalties[seat]["top"] = get_top_royalty(board["top"])
            royalties[seat]["middle"] = get_middle_royalty(board["middle"])
            royalties[seat]["bottom"] = get_bottom_royalty(board["bottom"])
            royalties[seat]["total"] = royalties[seat]["top"] + royalties[seat]["middle"] + royalties[seat]["bottom"]
            
            # Check FL entry
            fl, cards = check_fl_entry(board["top"])
            fl_entry[seat] = fl
            fl_card_count[seat] = cards
    
    # Calculate line results (P0 perspective: +1 win, -1 loss, 0 tie)
    line_results = [0, 0, 0]
    if busted[0] and busted[1]:
        pass  # Both busted, no lines
    elif busted[0]:
        line_results = [-1, -1, -1]  # P0 loses all
    elif busted[1]:
        line_results = [1, 1, 1]  # P0 wins all
    else:
        # Compare each line
        for i, line in enumerate(["top", "middle", "bottom"]):
            if hand_values[0][line] > hand_values[1][line]:
                line_results[i] = 1
            elif hand_values[0][line] < hand_values[1][line]:
                line_results[i] = -1
    
    # Calculate total score
    line_total = sum(line_results)
    scoop = abs(line_total) == 3
    scoop_bonus = 3 if scoop else 0
    
    if busted[0] and not busted[1]:
        raw_score = [-6 - royalties[1]["total"], 6 + royalties[1]["total"]]
    elif busted[1] and not busted[0]:
        raw_score = [6 + royalties[0]["total"], -6 - royalties[0]["total"]]
    elif busted[0] and busted[1]:
        raw_score = [0, 0]
    else:
        p0_score = line_total + (scoop_bonus if line_total > 0 else -scoop_bonus if line_total < 0 else 0)
        p0_score += royalties[0]["total"] - royalties[1]["total"]
        raw_score = [p0_score, -p0_score]
    
    # Update chips
    game.chips[0] += raw_score[0]
    game.chips[1] += raw_score[1]
    
    # Update FL state in game
    game.is_fantasyland = fl_entry
    game.fl_card_count = fl_card_count
    
    # Get hand names for display
    hand_names = [{}, {}]
    for seat in [0, 1]:
        board = game.boards[seat]
        hand_names[seat] = {
            "top": get_hand_name(board["top"], 3),
            "middle": get_hand_name(board["middle"], 5),
            "bottom": get_hand_name(board["bottom"], 5)
        }
    
    result = {
        "boards": game.boards,
        "busted": busted,
        "royalties": royalties,
        "hand_names": hand_names,
        "line_results": line_results,
        "scoop": scoop,
        "raw_score": raw_score,
        "actual_score": raw_score,
        "chips": game.chips.copy(),
        "fl_entry": fl_entry,
        "fl_card_count": fl_card_count
    }
    print(f"[DEBUG] Score calculated: {result}")
    return result


def get_hand_name(cards: list, expected_count: int) -> str:
    """Get human-readable hand name."""
    val = evaluate_hand(cards, expected_count)
    if val == 0:
        return "---"
    if expected_count == 3:
        if val >= 4000:
            return "スリーカード"
        elif val >= 2000:
            rank = val - 2000
            rank_names = {2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',10:'T',11:'J',12:'Q',13:'K',14:'A'}
            return f"{rank_names.get(rank,'?')}のペア"
        else:
            return "ハイカード"
    else:
        if val >= 9000:
            return "ロイヤルフラッシュ" if val >= 9014 else "ストレートフラッシュ"
        elif val >= 8000:
            return "フォーカード"
        elif val >= 7000:
            return "フルハウス"
        elif val >= 6000:
            return "フラッシュ"
        elif val >= 5000:
            return "ストレート"
        elif val >= 4000:
            return "スリーカード"
        elif val >= 3000:
            return "ツーペア"
        elif val >= 2000:
            return "ワンペア"
        else:
            return "ハイカード"


def evaluate_hand(cards: list, expected_count: int) -> int:
    """Evaluate hand strength. Returns numeric value (higher = stronger)."""
    if len(cards) != expected_count:
        return 0
    
    RANK_VALUES = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
                   '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
    
    ranks = []
    suits = []
    jokers = 0
    for card in cards:
        if card in ["X1", "X2"]:
            jokers += 1
        else:
            ranks.append(RANK_VALUES.get(card[0], 0))
            suits.append(card[1])
    
    from collections import Counter
    rank_counts = Counter(ranks)
    suit_counts = Counter(suits)
    
    sorted_ranks = sorted(ranks, reverse=True)
    is_flush = len(suit_counts) == 1 and (len(suits) + jokers) == expected_count
    is_straight = check_straight(sorted_ranks, jokers)
    
    # 3-card hand evaluation
    if expected_count == 3:
        best_count = max(rank_counts.values()) if rank_counts else 0
        if best_count + jokers >= 3:
            return 4000 + max(ranks)  # Trips
        elif best_count + jokers >= 2:
            return 2000 + max(r for r, c in rank_counts.items() if c + jokers >= 2)  # Pair
        else:
            return sorted_ranks[0] if sorted_ranks else 0  # High card
    
    # 5-card hand evaluation
    best_count = max(rank_counts.values()) if rank_counts else 0
    
    # Royal/Straight Flush
    if is_flush and is_straight:
        return 9000 + max(ranks)
    # Four of a kind
    if best_count + jokers >= 4:
        return 8000 + max(r for r, c in rank_counts.items() if c >= 2)
    # Full house
    if (best_count >= 3 and len([r for r, c in rank_counts.items() if c >= 2]) >= 2) or \
       (best_count >= 2 and jokers >= 1 and len(rank_counts) <= 3):
        return 7000 + max(r for r, c in rank_counts.items() if c >= 3) if best_count >= 3 else 7000 + max(ranks)
    # Flush
    if is_flush:
        return 6000 + max(ranks)
    # Straight
    if is_straight:
        return 5000 + max(ranks)
    # Three of a kind
    if best_count + jokers >= 3:
        return 4000 + max(r for r, c in rank_counts.items() if c >= 2)
    # Two pair
    pairs = [r for r, c in rank_counts.items() if c >= 2]
    if len(pairs) >= 2:
        return 3000 + max(pairs)
    # One pair
    if best_count >= 2 or jokers >= 1:
        return 2000 + (max(pairs) if pairs else max(ranks))
    # High card
    return sorted_ranks[0] if sorted_ranks else 0


def check_straight(sorted_ranks: list, jokers: int = 0) -> bool:
    """Check if ranks form a straight, with joker wildcards."""
    total = len(sorted_ranks) + jokers
    if total != 5:
        return False

    if jokers == 0:
        # Original logic
        if sorted_ranks[0] - sorted_ranks[4] == 4 and len(set(sorted_ranks)) == 5:
            return True
        if sorted_ranks == [14, 5, 4, 3, 2]:
            return True
        return False

    # With jokers: check if unique ranks can form a 5-card straight
    # by filling gaps with jokers
    unique = sorted(set(sorted_ranks), reverse=True)
    if len(unique) != len(sorted_ranks):
        return False  # Duplicate ranks can't form a straight

    # Try each possible top of straight (14 down to 5)
    for high in range(14, 4, -1):
        needed = set(range(high, high - 5, -1))
        # Wheel: A-2-3-4-5
        if high == 5:
            needed = {14, 5, 4, 3, 2}
        present = needed & set(unique)
        missing = len(needed) - len(present)
        extra = len(set(unique) - needed)
        if missing <= jokers and extra == 0:
            return True

    return False


def get_top_royalty(cards: list) -> int:
    """Top row royalties: 66=1, 77=2, ... AA=9 (pairs). Trips: 222=10, ... AAA=22"""
    from collections import Counter
    RANK_VALUES = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
                   '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
    ranks = []
    jokers = 0
    for c in cards:
        if c in ["X1", "X2"]:
            jokers += 1
        else:
            ranks.append(RANK_VALUES.get(c[0], 0))
    rank_counts = Counter(ranks)
    
    # Check trips (with jokers) - highest rank first
    best_royalty = 0
    for r in sorted(rank_counts.keys(), reverse=True):
        count = rank_counts[r]
        if count + jokers >= 3:
            return 10 + (r - 2)  # 222=10, AAA=22
        if count + jokers >= 2 and r >= 6:
            royalty = r - 5  # 66=1, 77=2, ... AA=9
            best_royalty = max(best_royalty, royalty)
    
    return best_royalty


def get_middle_royalty(cards: list) -> int:
    """Middle row royalties: Trips=2, Straight=4, Flush=8, FH=12, Quads=20, SF=30, RF=50"""
    val = evaluate_hand(cards, 5)
    if val >= 9000:
        return 50 if val >= 9014 else 30  # RF or SF
    elif val >= 8000:
        return 20  # Quads
    elif val >= 7000:
        return 12  # FH
    elif val >= 6000:
        return 8  # Flush
    elif val >= 5000:
        return 4  # Straight
    elif val >= 4000:
        return 2  # Trips
    return 0


def get_bottom_royalty(cards: list) -> int:
    """Bottom row royalties: Straight=2, Flush=4, FH=6, Quads=10, SF=15, RF=25"""
    val = evaluate_hand(cards, 5)
    if val >= 9000:
        return 25 if val >= 9014 else 15  # RF or SF
    elif val >= 8000:
        return 10  # Quads
    elif val >= 7000:
        return 6  # FH
    elif val >= 6000:
        return 4  # Flush
    elif val >= 5000:
        return 2  # Straight
    return 0


def check_fl_entry(top_cards: list) -> tuple:
    """Check if top row qualifies for Fantasyland.
    Returns (qualifies: bool, card_count: int)
    QQ = 14, KK = 15, AA = 16, Trips = 17
    """
    if len(top_cards) < 3:
        return False, 0
    
    RANK_VALUES = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
                   '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
    
    ranks = []
    jokers = 0
    for card in top_cards:
        if card in ["X1", "X2"]:
            jokers += 1
        else:
            rank = card[0]
            ranks.append(RANK_VALUES.get(rank, 0))
    
    # Count ranks
    from collections import Counter
    rank_counts = Counter(ranks)
    
    # Check for trips (with jokers)
    for r, count in rank_counts.items():
        if count + jokers >= 3:
            return True, 17  # Trips = 17 cards in FL
    
    # Check for pairs QQ+
    for r, count in rank_counts.items():
        if count + jokers >= 2:
            if r == 12:  # QQ
                return True, 14
            elif r == 13:  # KK
                return True, 15
            elif r == 14:  # AA
                return True, 16
    
    return False, 0


def check_session_end(game: GameState) -> Optional[dict]:
    """Check if session should end."""
    if game.chips[0] <= 0:
        return {"winner": 1, "final_chips": game.chips, "hands_played": game.hands_played, "reason": "bankrupt"}
    if game.chips[1] <= 0:
        return {"winner": 0, "final_chips": game.chips, "hands_played": game.hands_played, "reason": "bankrupt"}
    if abs(game.chips[0] - game.chips[1]) >= 40:
        winner = 0 if game.chips[0] > game.chips[1] else 1
        return {"winner": winner, "final_chips": game.chips, "hands_played": game.hands_played, "reason": "chip_lead"}
    return None


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
