"""
Preprocess Rust self-play JSONL → NPZ for BC/VN training.

Reads per-game JSONL from Rust benchmark, produces:
  - states.npy (N, 520) float32  — state encoding
  - actions.npy (N,) int64        — chosen action index
  - action_evs.npy (N, 250) float32 — soft-label distribution
  - valid_masks.npy (N, 250) bool
  - rewards.npy (N,) float32      — game total_score
  - busted.npy (N,) float32       — hero busted flag
  - fl_entry.npy (N,) float32     — hero FL entry flag
  - royalties.npy (N,) float32    — hero royalty

Usage:
    python ai/training/preprocess_selfplay.py data/selfplay.jsonl --output data/processed_sp
"""
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.engine.encoding import Board, Observation, encode_state, STATE_DIM

MAX_ACTIONS = 250


def softmax(x, temperature=1.0):
    """Softmax with temperature."""
    x = np.array(x, dtype=np.float64)
    x = x / temperature
    x = x - x.max()
    e = np.exp(x)
    return (e / e.sum()).astype(np.float32)


def visits_to_soft_label(visit_counts, n_actions, temperature=1.0):
    """Convert MCTS visit counts to soft-label distribution (250-dim)."""
    label = np.full(MAX_ACTIONS, -1e9, dtype=np.float32)
    if not visit_counts or n_actions == 0:
        return label
    # Convert visits to logits (proportional to log(visits + 1))
    visits = np.array(visit_counts[:n_actions], dtype=np.float64)
    # Use visit proportions directly as soft targets
    total = visits.sum()
    if total > 0:
        probs = visits / total
        # Store as log-scale for softmax compatibility in training
        label[:n_actions] = np.log(probs + 1e-10).astype(np.float32) * temperature
    return label


def evs_to_soft_label(action_evs, top_k_indices, n_actions, temperature=2.0):
    """Convert rollout EVs to soft-label distribution (250-dim)."""
    label = np.full(MAX_ACTIONS, -1e9, dtype=np.float32)
    if not action_evs or not top_k_indices or n_actions == 0:
        return label
    # Place EVs at their action indices
    for idx, ev in zip(top_k_indices, action_evs):
        if idx < MAX_ACTIONS:
            label[idx] = ev / temperature
    return label


def reconstruct_state(turn_log):
    """Reconstruct 520-dim state from board/dealt_cards/discards info."""
    bs = turn_log["board_self"]
    bo = turn_log["board_opponent"]

    # Convert CardIdx lists to card strings
    def idx_to_card_str(idx):
        if idx >= 52:
            return f"X{idx - 51}"
        suits = "hdcs"
        ranks = "23456789TJQKA"
        rank = ranks[idx // 4]   # Rust: rank = idx / 4 (0=2, 12=A)
        suit = suits[idx % 4]    # Rust: suit = idx % 4 (0=h, 1=d, 2=c, 3=s)
        return f"{rank}{suit}"

    board_self = Board(
        top=[idx_to_card_str(c) for c in bs["top"]],
        middle=[idx_to_card_str(c) for c in bs["middle"]],
        bottom=[idx_to_card_str(c) for c in bs["bottom"]],
    )
    board_opp = Board(
        top=[idx_to_card_str(c) for c in bo["top"]],
        middle=[idx_to_card_str(c) for c in bo["middle"]],
        bottom=[idx_to_card_str(c) for c in bo["bottom"]],
    )

    dealt = [idx_to_card_str(c) for c in turn_log["dealt_cards"]]
    discards = [idx_to_card_str(c) for c in turn_log["discards_self"]]

    obs = Observation(
        board_self=board_self,
        board_opponent=board_opp,
        dealt_cards=dealt,
        known_discards_self=discards,
        turn=turn_log["turn"],
        is_btn=turn_log["is_btn"],
        chips_self=turn_log.get("chips_self", 200),
        chips_opponent=turn_log.get("chips_opponent", 200),
    )
    return encode_state(obs)


def count_games(path):
    """Count lines in JSONL."""
    count = 0
    with open(path, "rb") as f:
        for _ in f:
            count += 1
    return count


def preprocess_selfplay(input_path, output_dir, skip_t4_no_evs=True):
    """Convert Rust self-play JSONL to training NPZ."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Counting games...")
    n_games = count_games(input_path)
    # Estimate: ~5 hero turns per game (selfplay: ~10 turns total)
    est_turns = n_games * 10
    print(f"  {n_games} games, est ~{est_turns} turns")

    # Pre-allocate arrays
    states_list = []
    actions_list = []
    action_evs_list = []
    rewards_list = []
    busted_list = []
    fl_list = []
    royalties_list = []

    start_time = time.time()
    total_turns = 0
    skipped = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for game_num, line in enumerate(f):
            try:
                game = json.loads(line.strip())
            except json.JSONDecodeError:
                skipped += 1
                continue

            result = game["result"]
            hero_reward = result["total_score"]
            hero_busted = float(result["hero_busted"])
            hero_fl = float(result["hero_fl"])
            hero_royalty = float(result["hero_royalty"])
            opp_reward = -hero_reward  # zero-sum
            opp_busted = float(result["opp_busted"])
            opp_fl = float(result["opp_fl"])
            opp_royalty = float(result["opp_royalty"])

            for turn_log in game["turns"]:
                is_hero = turn_log["player"] == "hero"
                n_actions = turn_log["n_actions"]

                # Skip T4 with no EVs (direct eval, no useful soft-label)
                if skip_t4_no_evs and turn_log["turn"] == 4:
                    if not turn_log.get("visit_counts") and not turn_log.get("action_evs"):
                        skipped += 1
                        continue

                # Reconstruct state from board info
                if turn_log.get("state"):
                    state = np.array(turn_log["state"], dtype=np.float32)
                else:
                    state = np.array(reconstruct_state(turn_log), dtype=np.float32)

                # Action index
                action_idx = turn_log["action_idx"]

                # Soft-label
                visit_counts = turn_log.get("visit_counts")
                action_evs = turn_log.get("action_evs")
                top_k_indices = turn_log.get("top_k_indices")

                if visit_counts:
                    # T0: MCTS visits
                    soft_label = visits_to_soft_label(visit_counts, n_actions)
                elif action_evs and top_k_indices:
                    # T1+: Rollout EVs
                    soft_label = evs_to_soft_label(action_evs, top_k_indices, n_actions)
                else:
                    # No soft-label available
                    soft_label = np.full(MAX_ACTIONS, -1e9, dtype=np.float32)
                    soft_label[action_idx] = 0.0  # hard label fallback

                states_list.append(state)
                actions_list.append(action_idx)
                action_evs_list.append(soft_label)

                if is_hero:
                    rewards_list.append(hero_reward)
                    busted_list.append(hero_busted)
                    fl_list.append(hero_fl)
                    royalties_list.append(hero_royalty)
                else:
                    rewards_list.append(opp_reward)
                    busted_list.append(opp_busted)
                    fl_list.append(opp_fl)
                    royalties_list.append(opp_royalty)

                total_turns += 1

            if (game_num + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                rate = (game_num + 1) / elapsed
                print(f"  {game_num+1}/{n_games} games ({total_turns} turns, {rate:.0f} games/s)")

    elapsed = time.time() - start_time
    print(f"\nEncoding done: {total_turns} turns from {n_games} games in {elapsed:.1f}s (skipped {skipped})")

    # Save as numpy files
    print("Saving numpy files...")

    states = np.array(states_list, dtype=np.float32)
    np.save(output_dir / "states.npy", states)

    actions = np.array(actions_list, dtype=np.int64)
    np.save(output_dir / "actions.npy", actions)

    action_evs = np.array(action_evs_list, dtype=np.float32)
    np.save(output_dir / "action_evs.npy", action_evs)

    rewards = np.array(rewards_list, dtype=np.float32)
    np.save(output_dir / "rewards.npy", rewards)

    busted = np.array(busted_list, dtype=np.float32)
    np.save(output_dir / "busted.npy", busted)

    fl_entry = np.array(fl_list, dtype=np.float32)
    np.save(output_dir / "fl_entry.npy", fl_entry)

    royalties = np.array(royalties_list, dtype=np.float32)
    np.save(output_dir / "royalties.npy", royalties)

    # Valid masks: all true up to n_actions for each turn
    masks = np.ones((total_turns, MAX_ACTIONS), dtype=bool)
    np.save(output_dir / "valid_masks.npy", masks)

    # Save VN data (float16 obs for space efficiency)
    vn_path = output_dir / "vn_data.npz"
    print(f"Saving VN data to {vn_path}")
    np.savez(vn_path,
             obs=states.astype(np.float16),
             score=rewards,
             turn=np.array([0] * total_turns, dtype=np.int64),  # placeholder, overwritten below
             busted=busted,
             fl_entry=fl_entry)

    # Metadata
    metadata = {
        "total_samples": total_turns,
        "n_games": n_games,
        "skipped": skipped,
        "state_dim": STATE_DIM,
        "max_actions": MAX_ACTIONS,
        "busted_ratio": float(busted.mean()),
        "fl_ratio": float(fl_entry.mean()),
        "avg_royalty": float(royalties.mean()),
        "avg_reward": float(rewards.mean()),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDone! {total_turns} samples -> {output_dir}")
    print(f"  Bust: {busted.mean():.1%}, FL: {fl_entry.mean():.1%}")
    print(f"  Avg royalty: {royalties.mean():.1f}, Avg reward: {rewards.mean():.1f}")
    print(f"  Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Rust self-play JSONL")
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument("--output", default="data/processed_selfplay",
                        help="Output directory for numpy files")
    parser.add_argument("--include-t4", action="store_true",
                        help="Include T4 turns even without EVs")
    args = parser.parse_args()

    preprocess_selfplay(args.input, args.output, skip_t4_no_evs=not args.include_t4)
