"""Play one full HU hand with the production engine and print every placement.

The engine is constructed exactly as `backend/ai_player.py` does it, so the
per-turn decision path shown here is the production path:

    T0      OFC_MCTS PUCT search   (policy net = prior, value net = leaf eval)
    T1-T3   RolloutEvaluator       (1-turn lookahead + VN, VN top-k prefilter)
    T4      no search: every legal completion scored directly

Both seats are driven by the same engine, so this is a self-play walkthrough.

Usage:
    python -m ai.tutor.walkthrough_full_hand --seed 7
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch

import ai.tutor.exact_late as exact_late
from ai.engine.action_space import get_turn_actions
from ai.engine.encoding import ALL_CARDS, STATE_DIM, Board, Observation
from ai.mcts.ofc_mcts import OFC_MCTS, OFCMCTSConfig
from ai.mcts.rollout_evaluator import RolloutEvaluator
from ai.models.networks import PolicyNetwork, ValueNetwork

ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT / "ai" / "models" / "expectimax_bc_v3" / "bc_policy_best.pt"
VALUE_PATH = ROOT / "ai" / "models" / "value_v3" / "value_best.pt"


def build_engine(simulations: int):
    checkpoint = torch.load(str(POLICY_PATH), map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint)
    saved_dim = state.get(
        "net.0.weight", state.get("input_proj.0.weight", torch.empty(0))
    ).shape[-1]
    policy = PolicyNetwork(input_dim=saved_dim if saved_dim > 0 else STATE_DIM)
    policy.load_state_dict(state)
    policy.eval()

    value = None
    norm_stats = None
    if VALUE_PATH.exists():
        vk = torch.load(str(VALUE_PATH), map_location="cpu", weights_only=True)
        vsd = vk.get("model_state_dict", vk)
        vn_dim = vsd.get("shared.0.weight", torch.empty(0)).shape[-1]
        value = ValueNetwork(input_dim=vn_dim if vn_dim > 0 else STATE_DIM)
        value.load_state_dict(vsd)
        value.eval()
        norm_stats = vk.get("norm_stats", None)

    rollout = RolloutEvaluator(
        policy_net=policy,
        n_rollouts=200,
        top_k=25,
        device="cpu",
        value_net=value,
        vn_top_k=15,
        norm_stats=norm_stats,
    )
    rollout.bust_penalty = 0.0
    rollout.vn_truncate_depth = 1
    rollout.vn_truncate_n = 500

    if value is None:
        return None, rollout, "policy only (no value net): rollout for every turn"
    engine = OFC_MCTS(
        policy_net=policy,
        value_net=value,
        config=OFCMCTSConfig(num_simulations=simulations),
        device="cpu",
        norm_stats=norm_stats,
        t1_evaluator=rollout,
    )
    return engine, rollout, f"OFC_MCTS (T0: {simulations} sims, T1+: rollout)"


def show_board(label: str, board: Board) -> str:
    return (
        f"{label:<10} top {' '.join(board.top):<11} | "
        f"mid {' '.join(board.middle):<17} | bot {' '.join(board.bottom)}"
    )


def decide(engine, rollout, observation: Observation, board: Board, cards: list[str]):
    """Reproduce the production branch for one decision."""
    placed = len(board.top) + len(board.middle) + len(board.bottom)
    if engine is not None and placed == 11:
        # T4: no search, score every legal completion directly.
        valid = get_turn_actions(cards, board)
        best_score = float("-inf")
        best = valid[0]
        for action in valid:
            final = exact_late.apply_action(board, action)
            score = RolloutEvaluator._compute_score(final, observation.board_opponent)
            if score > best_score:
                best_score = score
                best = action
        return best, "T4 direct scoring (no model)"
    if engine is not None:
        _, action = engine.select_action(observation)
        return action, ("T0 MCTS search" if observation.turn == 0 else "rollout + VN")
    _, action = rollout.select_action(observation)
    return action, "rollout (fallback)"


def run(seed: int, simulations: int) -> None:
    engine, rollout, description = build_engine(simulations)
    print(f"engine: {description}")

    rng = random.Random(seed)
    deck = list(ALL_CARDS)
    rng.shuffle(deck)
    cursor = 0

    def deal(count: int) -> list[str]:
        nonlocal cursor
        cards = deck[cursor : cursor + count]
        cursor += count
        return cards

    boards = {"BB": Board(), "BTN": Board()}
    discards: dict[str, list[str]] = {"BB": [], "BTN": []}
    # BB acts first on every street.
    order = ("BB", "BTN")

    for turn in range(5):
        count = 5 if turn == 0 else 3
        hands = {seat: deal(count) for seat in order}
        print(f"\n{'=' * 92}\nTURN {turn}   " + "   ".join(
            f"{seat} draws [{' '.join(hands[seat])}]" for seat in order
        ))
        print("=" * 92)
        for seat in order:
            board = boards[seat]
            opponent = boards["BTN" if seat == "BB" else "BB"]
            observation = Observation(
                board_self=board,
                board_opponent=opponent,
                dealt_cards=hands[seat],
                known_discards_self=discards[seat],
                turn=turn,
                is_btn=(seat == "BTN"),
            )
            started = time.time()
            action, how = decide(engine, rollout, observation, board, hands[seat])
            elapsed = time.time() - started
            for card, row in action.placements:
                getattr(board, row).append(card)
            if action.discard:
                discards[seat].append(action.discard)
            placement = ", ".join(f"{card}->{row}" for card, row in action.placements)
            print(
                f"  {seat}: {placement}"
                + (f"  (discard {action.discard})" if action.discard else "")
            )
            print(f"       [{how}, {elapsed:.1f}s]")
            print("       " + show_board("", board).strip())

    print(f"\n{'=' * 92}\nFINAL\n{'=' * 92}")
    for seat in order:
        print(show_board(seat, boards[seat]))
    bb_metrics = exact_late.terminal_metrics(boards["BB"], boards["BTN"])
    btn_metrics = exact_late.terminal_metrics(boards["BTN"], boards["BB"])
    for seat, metrics in (("BB", bb_metrics), ("BTN", btn_metrics)):
        state = "BUST" if metrics["bust"] else f"royalty {metrics['royalty']:.0f}"
        fl = f", FL {metrics['fl_card_count']}" if metrics["fl_any"] else ""
        print(f"  {seat}: {state}{fl}")
    print(f"\n  BB score vs BTN (FL EV included): {bb_metrics['score']:+.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--simulations", type=int, default=500)
    args = parser.parse_args()
    run(args.seed, args.simulations)


if __name__ == "__main__":
    main()
