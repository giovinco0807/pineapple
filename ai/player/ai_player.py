"""
OFC Pineapple - AI Player (WebSocket Client)

Connects to the game server and plays using a trained PolicyNetwork.

Usage:
    python -m ai.player.ai_player --room <room_id> --model ai/models/checkpoints/bc_policy_best.pt
"""
import sys
import json
import asyncio
from pathlib import Path
from typing import Optional, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import numpy as np

from ai.engine.encoding import Board, Observation, encode_state
from ai.engine.action_space import (
    Action, get_initial_actions, get_turn_actions,
    create_action_mask, MAX_ACTIONS,
)
from ai.models.networks import PolicyNetwork


class AIPlayer:
    """AI player that connects via WebSocket and plays using PolicyNetwork."""

    def __init__(self, policy_path: Optional[str] = None,
                 temperature: float = 0.1, device: str = "cpu"):
        self.device = device
        self.temperature = temperature

        # Load policy network
        self.policy_net = PolicyNetwork().to(device)
        if policy_path and Path(policy_path).exists():
            self.policy_net.load_state_dict(
                torch.load(policy_path, map_location=device)
            )
            print(f"Loaded policy from {policy_path}")
        else:
            print("Using random (untrained) policy")
        self.policy_net.eval()

        # Game state
        self.my_seat: Optional[int] = None
        self.board_self = Board()
        self.board_opponent = Board()
        self.my_discards = []
        self.turn = 0
        self.chips_self = 200
        self.chips_opponent = 200
        self.is_btn = False

    def reset_hand(self):
        """Reset state for a new hand."""
        self.board_self = Board()
        self.board_opponent = Board()
        self.my_discards = []
        self.turn = 0

    def handle_message(self, msg: dict) -> Optional[dict]:
        """Process a server message and return a response (or None)."""
        msg_type = msg.get("type", "")

        if msg_type == "session_start":
            self.my_seat = msg.get("your_seat", 0)
            self.chips_self = msg.get("chips", [200, 200])[self.my_seat]
            self.chips_opponent = msg.get("chips", [200, 200])[1 - self.my_seat]
            return None

        elif msg_type == "deal":
            self.turn = msg.get("turn", 0)
            self.is_btn = (msg.get("btn", 0) == self.my_seat)

            if self.turn == 0:
                self.reset_hand()
                if "opponent_board" in msg:
                    self.board_opponent = Board.from_dict(msg["opponent_board"])

            return self._decide_action(msg["cards"])

        elif msg_type == "opponent_placed":
            if "opponent_board" in msg:
                self.board_opponent = Board.from_dict(msg["opponent_board"])
            return None

        elif msg_type == "hand_end":
            result = msg.get("result", {})
            if "chips" in result and self.my_seat is not None:
                self.chips_self = result["chips"][self.my_seat]
                self.chips_opponent = result["chips"][1 - self.my_seat]
            return {"type": "next_hand"}

        elif msg_type == "fl_solved":
            if "board" in msg:
                self.board_self = Board.from_dict(msg["board"])
            return None

        elif msg_type == "session_end":
            return None

        return None

    def _decide_action(self, dealt_cards: list) -> dict:
        """Use policy network to select an action."""
        obs = Observation(
            board_self=self.board_self.copy(),
            board_opponent=self.board_opponent.copy(),
            dealt_cards=dealt_cards,
            known_discards_self=list(self.my_discards),
            turn=self.turn,
            is_btn=self.is_btn,
            chips_self=self.chips_self,
            chips_opponent=self.chips_opponent,
        )

        # Get valid actions
        if self.turn == 0:
            valid_actions = get_initial_actions(dealt_cards, self.board_self)
        else:
            valid_actions = get_turn_actions(dealt_cards, self.board_self)

        if not valid_actions:
            print("[AIPlayer] No valid actions!")
            return {"type": "place", "placements": []}

        # Encode state
        state_vec = encode_state(obs)
        state_t = torch.FloatTensor(state_vec).unsqueeze(0).to(self.device)
        mask = create_action_mask(valid_actions)
        mask_t = torch.BoolTensor(mask).unsqueeze(0).to(self.device)

        # Select action
        with torch.no_grad():
            action_idx = self.policy_net.select_action(
                state_t, mask_t, temperature=self.temperature
            ).item()

        action_idx = min(action_idx, len(valid_actions) - 1)
        action = valid_actions[action_idx]

        # Update local state
        for card, pos in action.placements:
            getattr(self.board_self, pos).append(card)
        if action.discard:
            self.my_discards.append(action.discard)

        # Build response
        response: Dict = {
            "type": "place",
            "placements": [list(p) for p in action.placements],
        }
        if action.discard:
            response["discard"] = action.discard
        return response

    async def play(self, room_id: str, server_url: str = "ws://localhost:8080"):
        """Connect to server and play a full session."""
        try:
            import websockets
        except ImportError:
            print("pip install websockets")
            return

        uri = f"{server_url}/ws/{room_id}"
        print(f"Connecting to {uri}...")

        async with websockets.connect(uri) as ws:
            # Join
            await ws.send(json.dumps({"type": "join"}))

            async for raw in ws:
                msg = json.loads(raw)
                print(f"[AI] Received: {msg['type']}")

                response = self.handle_message(msg)
                if response:
                    print(f"[AI] Sending: {response['type']}")
                    await ws.send(json.dumps(response))

                if msg.get("type") == "session_end":
                    print("[AI] Session ended")
                    break


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run AI player")
    parser.add_argument("--room", required=True, help="Room ID to join")
    parser.add_argument("--model", default=None, help="Policy model path")
    parser.add_argument("--server", default="ws://localhost:8080", help="Server URL")
    parser.add_argument("--temperature", type=float, default=0.1)
    args = parser.parse_args()

    player = AIPlayer(policy_path=args.model, temperature=args.temperature)
    asyncio.run(player.play(args.room, args.server))
