"""
OFC Pineapple CFR - Training Loop (v2)

Usage:
    # Quick local test (100 iterations)
    python -m ai.cfr.train_cfr --iterations 100

    # Medium training run
    python -m ai.cfr.train_cfr --iterations 10000 --t0-top-k 20 --max-depth 4

    # Resume from checkpoint
    python -m ai.cfr.train_cfr --iterations 50000 --resume checkpoints/cfr_latest.pkl
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.cfr.ofc_cfr import OFC_CFR, evaluate_board_heuristic
from ai.cfr.game_state import create_initial_state, NodeType


def play_one_game(cfr: OFC_CFR, p0_cfr: bool, p1_cfr: bool) -> float:
    """
    Play one complete game and return P0's score.
    
    For non-CFR players, use random action selection.
    Uses greedy selection for CFR player.
    """
    state = create_initial_state()
    max_moves = 200  # Safety limit

    for _ in range(max_moves):
        nt = state.node_type
        if nt == NodeType.TERMINAL:
            return state.terminal_utility(0)
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

        use_cfr = (player == 0 and p0_cfr) or (player == 1 and p1_cfr)
        if use_cfr:
            try:
                _, action = cfr.select_action_greedy(state, player)
            except ValueError:
                action = random.choice(actions)
        else:
            action = random.choice(actions)

        state = state.apply_action(player, action)

    # If we hit max_moves, use heuristic
    return evaluate_board_heuristic(
        state.boards[0], state.boards[1], state.btn == 0
    )


def evaluate_vs_random(cfr: OFC_CFR, n_games: int = 20) -> float:
    """CFR (P0) vs Random (P1). Returns avg score."""
    total = sum(play_one_game(cfr, True, False) for _ in range(n_games))
    return total / n_games


def evaluate_self_play(cfr: OFC_CFR, n_games: int = 10) -> float:
    """CFR vs CFR self-play. Should approach 0."""
    total = sum(play_one_game(cfr, True, True) for _ in range(n_games))
    return total / n_games


def main():
    parser = argparse.ArgumentParser(description="OFC Pineapple CFR Trainer")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--checkpoint-dir", type=str, default="ai/cfr/checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-games", type=int, default=20)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no-abstraction", action="store_true")
    parser.add_argument("--t0-top-k", type=int, default=20,
                        help="Max T0 actions to explore (default: 20)")
    parser.add_argument("--max-depth", type=int, default=4,
                        help="Max CFR tree depth (default: 4)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("OFC Pineapple CFR Trainer")
    print("=" * 60)
    print(f"  Iterations:     {args.iterations:,}")
    print(f"  T0 Top-K:       {args.t0_top_k}")
    print(f"  Max Depth:      {args.max_depth}")
    print(f"  Abstraction:    {'OFF' if args.no_abstraction else 'ON'}")
    print(f"  Eval every:     {args.eval_interval}")
    print(f"  Eval games:     {args.eval_games}")
    print(f"  Seed:           {args.seed}")
    if args.resume:
        print(f"  Resume from:    {args.resume}")
    print("=" * 60, flush=True)

    cfr = OFC_CFR(
        use_abstraction=not args.no_abstraction,
        t0_top_k=args.t0_top_k,
        max_cfr_depth=args.max_depth,
    )

    if args.resume and os.path.exists(args.resume):
        cfr.load_checkpoint(args.resume)
        print(f"Resumed from iteration {cfr.iteration}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    start_time = time.time()
    start_iter = cfr.iteration
    last_log_time = start_time
    log_entries = []

    try:
        for i in range(args.iterations):
            cfr.run_iteration()
            current_iter = cfr.iteration

            # Progress log every 10s
            now = time.time()
            if now - last_log_time >= 10.0 or (i + 1) == args.iterations:
                elapsed = now - start_time
                iters_done = current_iter - start_iter
                rate = iters_done / elapsed if elapsed > 0 else 0
                eta = (args.iterations - i - 1) / rate if rate > 0 else 0

                print(
                    f"  [{current_iter:>7,}] "
                    f"IS={cfr.store.size:>7,}  "
                    f"Nodes={cfr.total_nodes_visited:>9,}  "
                    f"Leaf={cfr.total_leaf_evals:>8,}  "
                    f"Regret={cfr.store.total_regret():>10,.0f}  "
                    f"{rate:>5.1f} it/s  "
                    f"ETA: {eta/60:>5.1f}min",
                    flush=True,
                )
                last_log_time = now

            # Evaluation
            if (i + 1) % args.eval_interval == 0:
                print(f"\n--- Eval @ iter {current_iter} ---")
                t_eval = time.time()
                vs_random = evaluate_vs_random(cfr, n_games=args.eval_games)
                vs_self = evaluate_self_play(cfr, n_games=min(args.eval_games // 2, 10))
                print(f"  vs Random: {vs_random:>+.2f} pts/game")
                print(f"  vs Self:   {vs_self:>+.2f} pts/game")
                print(f"  Eval time: {time.time()-t_eval:.1f}s\n", flush=True)

                log_entries.append({
                    "iteration": current_iter,
                    "info_sets": cfr.store.size,
                    "total_regret": cfr.store.total_regret(),
                    "vs_random": vs_random,
                    "vs_self": vs_self,
                    "elapsed": time.time() - start_time,
                })

            # Checkpoint
            if (i + 1) % args.checkpoint_interval == 0:
                ckpt = os.path.join(args.checkpoint_dir, f"cfr_iter_{current_iter}.pkl")
                cfr.save_checkpoint(ckpt)
                cfr.save_checkpoint(os.path.join(args.checkpoint_dir, "cfr_latest.pkl"))
                print(f"  Checkpoint: {ckpt}", flush=True)

    except KeyboardInterrupt:
        print("\n\nInterrupted. Saving...")

    # Final save
    final_path = os.path.join(args.checkpoint_dir, "cfr_latest.pkl")
    cfr.save_checkpoint(final_path)

    elapsed = time.time() - start_time
    total_iters = cfr.iteration - start_iter

    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)
    cfr.print_stats()
    print(f"  Total time:      {elapsed:.0f}s ({elapsed/60:.1f}min)")
    if elapsed > 0:
        print(f"  Iterations/sec:  {total_iters / elapsed:.2f}")
    print(f"  Checkpoint:      {final_path}")

    strategy_path = os.path.join(args.checkpoint_dir, "strategy_export.json")
    cfr.store.export_strategy(strategy_path, top_n=100)
    print(f"  Strategy export: {strategy_path}")

    if log_entries:
        log_path = os.path.join(args.checkpoint_dir, "training_log.json")
        with open(log_path, "w") as f:
            json.dump(log_entries, f, indent=2)
        print(f"  Training log:    {log_path}")

    print("=" * 60)


if __name__ == "__main__":
    main()
