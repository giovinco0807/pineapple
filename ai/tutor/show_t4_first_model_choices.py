"""Show what each existing model places at a T4 first-seat decision.

Every candidate is scored by the same exact uniform-deal EV table, so the
comparison states not only what each model would place but how much EV that
choice gives up against the exact optimum.

Compared decision rules:

- ``exact``       : ai/tutor/t4_bb_exact_resolver.py (ground truth here)
- ``production``  : the T4 branch of backend/ai_player.py, which scores each
                    completion with RolloutEvaluator._compute_score against the
                    opponent's *current* board
- ``myopic``      : exact_late.best_t4_completion(opponent_board=None), the
                    legacy playout completion rule
- ``bc_v3``       : ai/models/expectimax_bc_v3 policy, greedy top-1
- ``bottomup_t3`` : ai/models/bottomup_t3 policy, greedy top-1

Usage:
    python -m ai.tutor.show_t4_first_model_choices
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

import ai.tutor.exact_late as exact_late
import ai.tutor.t4_bb_exact_vs_myopic_probe as probe
from ai.engine.action_space import get_turn_actions
from ai.engine.encoding import Board, Observation, encode_state
from ai.mcts.rollout_evaluator import RolloutEvaluator
from ai.models.networks import PolicyNetwork

POLICIES = {
    "bc_v3": Path("ai/models/expectimax_bc_v3/bc_policy_best.pt"),
    "bottomup_t3": Path("ai/models/bottomup_t3/bc_policy_best.pt"),
}


def _board(rows) -> Board:
    return Board(top=list(rows[0]), middle=list(rows[1]), bottom=list(rows[2]))


def load_policy(path: Path):
    if not path.is_file():
        return None
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint)
    first = next(iter(state.values()))
    input_dim = first.shape[1] if first.dim() == 2 else 520
    net = PolicyNetwork(input_dim=input_dim)
    net.load_state_dict(state)
    net.eval()
    return net


def policy_choice(net, root: dict, actions) -> str | None:
    """Greedy top-1 legal action under a behaviour-cloning policy."""
    if net is None:
        return None
    board = _board(root["bb_board"])
    observation = Observation(
        board_self=board,
        board_opponent=_board(root["btn_board"]),
        dealt_cards=list(root["draw"]),
        known_discards_self=list(root["bb_discards"]),
        turn=4,
        is_btn=False,
    )
    from ai.engine.action_space import get_action_from_semantic_index_if_valid

    legal: dict[int, str] = {}
    for index in range(27):
        action = get_action_from_semantic_index_if_valid(
            index, list(root["draw"]), board
        )
        if action is not None:
            legal[index] = exact_late.action_key(action)
    if not legal:
        return None
    # T1-8 use semantic indices inside the 250-wide action head.
    from ai.engine.action_space import MAX_ACTIONS

    mask = torch.zeros(1, MAX_ACTIONS, dtype=torch.bool)
    for index in legal:
        mask[0, index] = True
    with torch.no_grad():
        features = torch.tensor(
            encode_state(observation), dtype=torch.float32
        ).unsqueeze(0)
        logits = net(features, mask)
        if isinstance(logits, tuple):
            logits = logits[0]
        order = torch.argsort(logits.squeeze(0), descending=True).tolist()
    for index in order:
        if index in legal:
            return legal[index]
    return None


def production_choice(root: dict, actions) -> str:
    """The T4 branch of backend/ai_player.py, replicated exactly."""
    board = _board(root["bb_board"])
    opponent = _board(root["btn_board"])
    best_score = float("-inf")
    best = actions[0]
    for action in actions:
        final = exact_late.apply_action(board, action)
        score = RolloutEvaluator._compute_score(final, opponent)
        if score > best_score:
            best_score = score
            best = action
    return exact_late.action_key(best)


def describe(action_key: str) -> str:
    import json

    payload = json.loads(action_key)
    placed = ", ".join(f"{card}->{row}" for card, row in payload["placements"])
    return f"{placed}  (discard {payload['discard']})"


def show(root: dict, label: str, nets: dict) -> None:
    board = _board(root["bb_board"])
    actions = get_turn_actions(list(root["draw"]), board)
    exact = probe.exact_action_table(root)
    best_ev = max(exact.values())
    best_key = min(key for key, value in exact.items() if value == best_ev)

    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
    print(f"  BB (hero, 11):  top {root['bb_board'][0]}")
    print(f"                  mid {root['bb_board'][1]}")
    print(f"                  bot {root['bb_board'][2]}")
    print(f"  BTN (opp, 11):  top {root['btn_board'][0]}")
    print(f"                  mid {root['btn_board'][1]}")
    print(f"                  bot {root['btn_board'][2]}")
    print(f"  draw: {list(root['draw'])}   hero discards so far: {list(root['bb_discards'])}")
    jokers = sum(
        1
        for rows in (root["bb_board"], root["btn_board"])
        for row in rows
        for card in row
        if card in ("X1", "X2")
    ) + sum(1 for card in root["draw"] if card in ("X1", "X2"))
    print(f"  visible jokers: {jokers}   legal actions: {len(exact)}")

    print(f"\n  {'model':<14}{'placement':<44}{'exact EV':>10}{'regret':>9}")
    print(f"  {'-' * 76}")
    choices = {
        "exact": best_key,
        "production": production_choice(root, actions),
        "myopic": probe.myopic_action_id(root),
    }
    for name, net in nets.items():
        chosen = policy_choice(net, root, actions)
        if chosen is not None:
            choices[name] = chosen
    for name, key in choices.items():
        ev = exact.get(key)
        if ev is None:
            print(f"  {name:<14}{'(illegal / unavailable)':<44}")
            continue
        regret = best_ev - ev
        marker = "" if regret == 0 else "  <-- gives up EV"
        print(f"  {name:<14}{describe(key):<44}{ev:>10.3f}{regret:>9.3f}{marker}")

    print(f"\n  all legal actions by exact EV:")
    for key, value in sorted(exact.items(), key=lambda item: -item[1]):
        print(f"    {value:>8.3f}  {describe(key)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260762, 20260739])
    args = parser.parse_args()
    nets = {name: load_policy(path) for name, path in POLICIES.items()}
    for name, net in nets.items():
        if net is None:
            print(f"[warn] policy {name} not loaded ({POLICIES[name]})")
    for index, seed in enumerate(args.seeds, start=1):
        show(probe.sample_random_root(seed), f"Hand {index}  (seed {seed})", nets)


if __name__ == "__main__":
    main()
