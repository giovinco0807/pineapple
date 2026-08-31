"""
OFC Pineapple CFR - Evaluation & Analysis Tools

Provides:
1. CFR vs Random benchmark
2. CFR vs CFR self-play convergence check
3. Strategy analysis (top info sets, action distributions)
4. Checkpoint comparison
5. Detailed game replay with CFR decisions

Usage:
    python -m ai.cfr.evaluate_cfr --checkpoint ai/cfr/checkpoints/cfr_latest.pkl
    python -m ai.cfr.evaluate_cfr --checkpoint ai/cfr/checkpoints/cfr_latest.pkl --mode replay
"""
import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.cfr.ofc_cfr import OFC_CFR, evaluate_board_heuristic
from ai.cfr.game_state import create_initial_state, OFCState, NodeType
from ai.cfr.abstraction import compute_fl_potential, FLPotential
from ai.engine.game_engine import evaluate_hand, hand_category


# ─── Game Simulation ─────────────────────────────────────────────

def play_game(
    cfr: OFC_CFR,
    p0_mode: str = "cfr",   # "cfr", "random", "greedy"
    p1_mode: str = "random",
    verbose: bool = False,
) -> dict:
    """
    Play one complete OFC game with specified player modes.

    Returns dict with score, boards, and game trace.
    """
    state = create_initial_state()
    trace = []
    max_moves = 200

    for step in range(max_moves):
        nt = state.node_type
        if nt == NodeType.TERMINAL:
            break
        if nt == NodeType.CHANCE:
            state = state.deal_cards()
            continue

        player = int(nt)
        actions = state.get_legal_actions(player)
        if not actions:
            new_state = state.copy()
            if player == 0:
                new_state.placed = (True, new_state.placed[1])
            else:
                new_state.placed = (new_state.placed[0], True)
            state = new_state
            continue

        mode = p0_mode if player == 0 else p1_mode

        if mode == "cfr":
            try:
                idx, action = cfr.select_action(state, player)
            except (ValueError, Exception):
                action = random.choice(actions)
                idx = 0
        elif mode == "greedy":
            try:
                idx, action = cfr.select_action_greedy(state, player)
            except (ValueError, Exception):
                action = random.choice(actions)
                idx = 0
        else:  # random
            idx = random.randrange(len(actions))
            action = actions[idx]

        if verbose:
            acts, strat = cfr.get_strategy(state, player)
            top_3 = sorted(enumerate(strat), key=lambda x: x[1], reverse=True)[:3]
            trace.append({
                "turn": state.turn,
                "player": player,
                "action_idx": idx,
                "n_actions": len(actions),
                "top_3_probs": [(i, f"{p:.3f}") for i, p in top_3],
                "placements": [(c, pos) for c, pos in action.placements],
                "discard": action.discard,
            })

        state = state.apply_action(player, action)

    # Terminal evaluation
    if state.node_type == NodeType.TERMINAL:
        score = state.terminal_utility(0)
    else:
        score = evaluate_board_heuristic(
            state.boards[0], state.boards[1], state.btn == 0
        )

    result = {"score": score, "trace": trace}

    # Board summary
    for seat in range(2):
        b = state.boards[seat]
        result[f"p{seat}_top"] = b.top
        result[f"p{seat}_mid"] = b.middle
        result[f"p{seat}_bot"] = b.bottom

    return result


# ─── Benchmarks ──────────────────────────────────────────────────

def benchmark_vs_random(cfr: OFC_CFR, n_games: int = 100) -> dict:
    """CFR (greedy) vs Random. Returns stats."""
    scores = []
    for _ in range(n_games):
        result = play_game(cfr, p0_mode="greedy", p1_mode="random")
        scores.append(result["score"])

    scores = np.array(scores)
    return {
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "median": float(np.median(scores)),
        "min": float(scores.min()),
        "max": float(scores.max()),
        "win_rate": float(np.mean(scores > 0)),
        "n_games": n_games,
    }


def benchmark_self_play(cfr: OFC_CFR, n_games: int = 100) -> dict:
    """CFR vs CFR self-play. Should approach 0 mean."""
    scores = []
    for _ in range(n_games):
        result = play_game(cfr, p0_mode="cfr", p1_mode="cfr")
        scores.append(result["score"])

    scores = np.array(scores)
    return {
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "mean_abs": float(np.abs(scores).mean()),
        "n_games": n_games,
    }


def benchmark_cfr_vs_cfr_sampled(cfr: OFC_CFR, n_games: int = 100) -> dict:
    """Sampled CFR vs Greedy CFR to check strategy robustness."""
    scores = []
    for _ in range(n_games):
        result = play_game(cfr, p0_mode="cfr", p1_mode="greedy")
        scores.append(result["score"])

    scores = np.array(scores)
    return {
        "mean": float(scores.mean()),
        "std": float(scores.std()),
        "n_games": n_games,
    }


# ─── Strategy Analysis ──────────────────────────────────────────

def analyze_strategy(cfr: OFC_CFR, top_n: int = 20):
    """Analyze the most-visited information sets."""
    print(f"\n{'='*70}")
    print(f"Strategy Analysis -- Top {top_n} Information Sets by Visit Count")
    print(f"{'='*70}")

    sorted_keys = sorted(
        cfr.store.data.keys(),
        key=lambda k: cfr.store.data[k].reach_count,
        reverse=True,
    )

    for rank, key in enumerate(sorted_keys[:top_n], 1):
        info_data = cfr.store.data[key]
        n_actions = cfr.store._action_counts.get(key, 0)
        if n_actions == 0:
            continue

        avg = info_data.get_average_strategy(n_actions)
        curr = info_data.get_strategy(n_actions)

        # Entropy (strategy diversity measure)
        entropy = -np.sum(avg * np.log2(avg + 1e-10))
        max_entropy = np.log2(n_actions)
        diversity = entropy / max_entropy if max_entropy > 0 else 0

        # Top actions
        top_actions = sorted(enumerate(avg), key=lambda x: x[1], reverse=True)[:5]

        print(f"\n#{rank:>2} [{info_data.reach_count:>6,} visits] ({n_actions} actions)")
        print(f"   Key: {key[:80]}{'...' if len(key) > 80 else ''}")
        print(f"   Entropy: {entropy:.2f}/{max_entropy:.2f} (diversity={diversity:.2f})")
        print(f"   Top actions: ", end="")
        for idx, prob in top_actions:
            if prob > 0.01:
                print(f"a{idx}={prob:.3f} ", end="")
        print()

    # Summary stats
    n_sets = len(cfr.store.data)
    total_reach = sum(d.reach_count for d in cfr.store.data.values())
    n_actions_hist = Counter(cfr.store._action_counts.values())

    print(f"\n{'='*70}")
    print(f"Summary:")
    print(f"  Total info sets:     {n_sets:,}")
    print(f"  Total reach events:  {total_reach:,}")
    print(f"  Action count distribution: {dict(sorted(n_actions_hist.items()))}")
    print(f"{'='*70}")


def analyze_t0_decisions(cfr: OFC_CFR, n_samples: int = 5):
    """Sample some T0 decisions and show the CFR strategy."""
    print(f"\n{'='*70}")
    print("T0 Decision Analysis -- Sample Hands")
    print(f"{'='*70}")

    for i in range(n_samples):
        state = create_initial_state()
        player = 1 - state.btn  # Non-button acts first.

        print(f"\n--- Hand {i+1} ---")
        print(f"  P{player} hand: {' '.join(state.hands[player])}")

        actions, strategy = cfr.get_strategy(state, player)
        if not actions:
            print("  No actions available")
            continue

        # Show top 5 actions
        top_indices = np.argsort(strategy)[::-1][:5]
        for rank, idx in enumerate(top_indices):
            action = actions[idx]
            prob = strategy[idx]
            if prob < 0.01:
                break
            placements_str = ", ".join(
                f"{c}->{pos}" for c, pos in action.placements
            )
            disc_str = f" (disc: {action.discard})" if action.discard else ""
            print(f"  [{prob:>5.1%}] {placements_str}{disc_str}")


# ─── Game Replay ─────────────────────────────────────────────────

def replay_game(cfr: OFC_CFR):
    """Play and display a full game with CFR decisions."""
    print(f"\n{'='*70}")
    print("Game Replay -- CFR (greedy) vs Random")
    print(f"{'='*70}")

    result = play_game(cfr, p0_mode="greedy", p1_mode="random", verbose=True)

    for entry in result["trace"]:
        disc = entry.get('discard', '')
        disc_str = f" (disc: {disc})" if disc else ""
        placements_str = ", ".join(f"{c}->{p}" for c, p in entry['placements'])
        print(
            f"  T{entry['turn']} P{entry['player']}: "
            f"{placements_str}{disc_str}"
            f"  [{entry['n_actions']} actions, top: {entry['top_3_probs'][:2]}]"
        )

    print(f"\n  Final Boards:")
    for seat in range(2):
        print(f"    P{seat} Top:    {result[f'p{seat}_top']}")
        print(f"    P{seat} Middle: {result[f'p{seat}_mid']}")
        print(f"    P{seat} Bottom: {result[f'p{seat}_bot']}")
    print(f"\n  Score (P0): {result['score']:+.1f}")
    print(f"{'='*70}")


# ─── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OFC Pineapple CFR Evaluator")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to CFR checkpoint")
    parser.add_argument("--mode", type=str, default="full",
                        choices=["full", "benchmark", "analysis", "replay", "t0"],
                        help="Evaluation mode")
    parser.add_argument("--games", type=int, default=100,
                        help="Games per benchmark")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading checkpoint: {args.checkpoint}")
    cfr = OFC_CFR()
    cfr.load_checkpoint(args.checkpoint)
    cfr.print_stats()

    if args.mode in ("full", "benchmark"):
        print("\n--- Benchmark: CFR vs Random ---")
        t0 = time.time()
        vs_random = benchmark_vs_random(cfr, n_games=args.games)
        print(f"  Mean score:  {vs_random['mean']:>+.2f} +/- {vs_random['std']:.2f}")
        print(f"  Median:      {vs_random['median']:>+.2f}")
        print(f"  Win rate:    {vs_random['win_rate']:.1%}")
        print(f"  Range:       [{vs_random['min']:+.1f}, {vs_random['max']:+.1f}]")
        print(f"  Time:        {time.time()-t0:.1f}s")

        print("\n--- Benchmark: CFR Self-Play ---")
        t0 = time.time()
        vs_self = benchmark_self_play(cfr, n_games=args.games)
        print(f"  Mean score:  {vs_self['mean']:>+.2f} +/- {vs_self['std']:.2f}")
        print(f"  Mean |score|: {vs_self['mean_abs']:.2f}")
        print(f"  Time:        {time.time()-t0:.1f}s")

    if args.mode in ("full", "analysis"):
        analyze_strategy(cfr)

    if args.mode in ("full", "t0"):
        analyze_t0_decisions(cfr, n_samples=5)

    if args.mode in ("full", "replay"):
        replay_game(cfr)


if __name__ == "__main__":
    main()
