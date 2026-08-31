"""OFC Pineapple - AI Opponent Module"""
import sys
from pathlib import Path

# Ensure parent paths are available for ai/ imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from ai.engine.encoding import Board, Observation, encode_state, STATE_DIM
from ai.engine.action_space import get_initial_actions, get_turn_actions, create_action_mask
from ai.engine.game_engine import (
    evaluate_hand, check_fl_entry,
    get_top_royalty, get_middle_royalty, get_bottom_royalty,
)
from ai.models.networks import PolicyNetwork, ValueNetwork
from ai.mcts.rollout_evaluator import RolloutEvaluator
from ai.mcts.ofc_mcts import OFC_MCTS, OFCMCTSConfig

AI_PLAYER_ID = "__ai__"

_AI_MODEL_PATH = Path(__file__).parent.parent / "ai" / "models" / "expectimax_bc_v3" / "bc_policy_best.pt"
_AI_VN_PATH = Path(__file__).parent.parent / "ai" / "models" / "value_v3" / "value_best.pt"
_ai_net = None
_ai_mcts = None  # Full MCTS engine (all turns)
_ai_re1 = None   # Fallback rollout evaluator


def init_ai():
    global _ai_net, _ai_mcts, _ai_re1
    if _ai_net is not None:
        return True
    if not _AI_MODEL_PATH.exists():
        print(f"[AI] Policy model not found: {_AI_MODEL_PATH}")
        return False
    try:
        # Auto-detect input_dim from checkpoint (backward compat: 520 or 522)
        ck = torch.load(str(_AI_MODEL_PATH), map_location='cpu', weights_only=True)
        sd = ck['model_state_dict'] if 'model_state_dict' in ck else ck
        saved_dim = sd.get('net.0.weight', sd.get('input_proj.0.weight', torch.empty(0))).shape[-1]
        _ai_net = PolicyNetwork(input_dim=saved_dim if saved_dim > 0 else STATE_DIM)
        _ai_net.load_state_dict(sd)
        _ai_net.eval()

        # Load Value Network
        _vn = None
        _ns = None
        if _AI_VN_PATH.exists():
            vk = torch.load(str(_AI_VN_PATH), map_location='cpu', weights_only=True)
            vsd = vk['model_state_dict'] if 'model_state_dict' in vk else vk
            vn_dim = vsd.get('shared.0.weight', torch.empty(0)).shape[-1]
            _vn = ValueNetwork(input_dim=vn_dim if vn_dim > 0 else STATE_DIM)
            _vn.load_state_dict(vsd)
            _vn.eval()
            _ns = vk.get('norm_stats', None)
            print(f"[AI] VN loaded: {_AI_VN_PATH}")
        else:
            print(f"[AI] VN not found, running without VN")

        # Rollout evaluator (used for T1+ decisions)
        # VN-truncated: play 1 turn ahead, batch VN eval, 500 rollouts
        _ai_re1 = RolloutEvaluator(
            policy_net=_ai_net, n_rollouts=200, top_k=25, device='cpu',
            value_net=_vn, vn_top_k=15, norm_stats=_ns)
        _ai_re1.bust_penalty = 0.0
        _ai_re1.vn_truncate_depth = 1   # 1-turn lookahead + VN eval
        _ai_re1.vn_truncate_n = 500     # more rollouts (cheaper per rollout)

        # MCTS engine (T0: MCTS search, T1+: truncated rollout)
        if _vn is not None:
            _ai_mcts = OFC_MCTS(
                policy_net=_ai_net, value_net=_vn,
                config=OFCMCTSConfig(num_simulations=500),
                device='cpu', norm_stats=_ns,
                t1_evaluator=_ai_re1)
            print(f"[AI] OFC_MCTS loaded (T0: 500 sims, T1+: VN-truncated d=1 r=500)")
        else:
            print(f"[AI] VN required for MCTS, falling back to rollout")

        print(f"[AI] Policy loaded: {_AI_MODEL_PATH}")
        return True
    except Exception as e:
        print(f"[AI] Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        _ai_net = None
        return False


# Try loading at import time
init_ai()


def ai_select_action_sync(game_state, ai_seat: int):
    """Select AI action synchronously. Called via asyncio.to_thread."""
    if _ai_net is None:
        return None

    board_dict = game_state.boards[ai_seat]
    opp_dict = game_state.boards[1 - ai_seat]
    board = Board.from_dict(board_dict)
    opp_board = Board.from_dict(opp_dict)
    cards = game_state.dealt_cards.get(ai_seat, [])
    turn = game_state.turn

    if not cards:
        return None

    obs = Observation(
        board_self=board,
        board_opponent=opp_board,
        dealt_cards=cards,
        known_discards_self=game_state.discards[ai_seat],
        turn=turn,
        is_btn=(ai_seat == game_state.btn),
        is_fl=getattr(game_state, 'is_fantasyland', [False, False])[ai_seat],
        opp_is_fl=getattr(game_state, 'is_fantasyland', [False, False])[1 - ai_seat],
    )

    card_count = len(board.top) + len(board.middle) + len(board.bottom)

    try:
        if _ai_mcts is not None:
            if card_count == 11:
                # Last turn: exact scoring (no search needed)
                valid = get_turn_actions(cards, board)
                if not valid:
                    return None
                if len(valid) == 1:
                    action = valid[0]
                else:
                    best_s = float('-inf')
                    best_a = valid[0]
                    for a in valid:
                        t = list(board.top); m = list(board.middle); b = list(board.bottom)
                        for c, p in a.placements:
                            if p == 'top': t.append(c)
                            elif p == 'middle': m.append(c)
                            else: b.append(c)
                        my_b = Board(top=t, middle=m, bottom=b)
                        s = RolloutEvaluator._compute_score(my_b, opp_board)
                        if s > best_s:
                            best_s = s
                            best_a = a
                    action = best_a
            else:
                # All turns (T0-T3): Full MCTS
                _, action = _ai_mcts.select_action(obs)
        else:
            # Fallback: rollout evaluator
            _, action = _ai_re1.select_action(obs)

        # Convert Action to backend format
        placements = [[c, p] for c, p in action.placements]
        discard = action.discard
        return {"placements": placements, "discard": discard}
    except Exception as e:
        print(f"[AI] Error selecting action: {e}")
        import traceback
        traceback.print_exc()
        return None
