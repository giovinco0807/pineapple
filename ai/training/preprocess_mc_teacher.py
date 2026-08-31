"""
Preprocess MC teacher data (JSONL) → NPZ for VN v3 training.

Converts the MC teacher JSONL format (from generate_mc_teacher.py) into
numpy arrays suitable for train_value.py.

Features:
  - Dual training signals: MC best EV + final score
  - Suit symmetry augmentation (×4): s↔h↔d↔c permutations
  - Histogram features (optional): 522 → 822 dims via prob_engine histograms

Usage:
    python preprocess_mc_teacher.py data.jsonl --output data/processed
    python preprocess_mc_teacher.py data.jsonl --output data/processed --augment --hist
"""
import sys
import json
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.engine.encoding import Board, Observation, encode_state, STATE_DIM


# ============================================================
#  Suit Symmetry Augmentation
# ============================================================

# 4 suit permutations that preserve game value
SUIT_PERMS = [
    {'s': 's', 'h': 'h', 'd': 'd', 'c': 'c'},  # identity
    {'s': 'h', 'h': 's', 'd': 'c', 'c': 'd'},  # swap s↔h, d↔c
    {'s': 'd', 'h': 'c', 'd': 's', 'c': 'h'},  # swap s↔d, h↔c
    {'s': 'c', 'h': 'd', 'd': 'h', 'c': 's'},  # swap s↔c, h↔d
]


def permute_card(card: str, perm: dict) -> str:
    """Apply suit permutation to a single card."""
    if not card or card.startswith('X'):  # Jokers are suit-independent
        return card
    return card[0] + perm[card[1]]


def permute_cards(cards: list, perm: dict) -> list:
    """Apply suit permutation to a list of cards."""
    return [permute_card(c, perm) for c in cards]


def permute_hist(hist: list, perm: dict) -> list:
    """Apply suit permutation to histogram features.

    Histogram is 300 dims = 3 rows × 100 bins.
    Bins are suit-independent (they encode hand strength), so
    histogram permutation is NOT needed — the histogram is recomputed
    from the permuted board anyway. Return as-is for identity perm.
    For non-identity perms, we skip histogram (set to zeros).
    """
    if perm == SUIT_PERMS[0]:
        return hist
    # For augmented samples, histogram is not available
    # (would require re-running prob_engine)
    return None


# ============================================================
#  Main Preprocessing
# ============================================================

def encode_one_sample(board_dict, dealt, exclude, turn, hist_data=None, include_hist=False):
    """Encode one board state into a feature vector."""
    if turn == 0:
        board_self = Board(top=[], middle=[], bottom=[])
    else:
        board_self = Board(
            top=list(board_dict.get("top", [])),
            middle=list(board_dict.get("mid", [])),
            bottom=list(board_dict.get("bot", [])),
        )

    board_opp = Board(top=[], middle=[], bottom=[])

    obs = Observation(
        board_self=board_self,
        board_opponent=board_opp,
        dealt_cards=dealt,
        known_discards_self=exclude,
        turn=turn,
        is_btn=True,
    )

    prob_features = None
    if include_hist and hist_data is not None:
        prob_features = np.array(hist_data, dtype=np.float32)

    return encode_state(obs, prob_features=prob_features)


def preprocess_mc_teacher(input_path: str, output_dir: str,
                          augment: bool = False, include_hist: bool = False):
    """Convert MC teacher JSONL to NPZ format for VN training.

    Args:
        input_path: Path to merged JSONL file
        output_dir: Output directory for NPZ files
        augment: Enable suit symmetry augmentation (×4 data)
        include_hist: Include histogram features (522 → 822 dims)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_augs = 4 if augment else 1
    print(f"Reading {input_path}...")
    print(f"  Augmentation: {'ON (×4 suit perms)' if augment else 'OFF'}")
    print(f"  Histogram:    {'ON (822 dims)' if include_hist else 'OFF (522 dims)'}")
    start_time = time.time()

    # Load all records and group by hand_id
    hands = defaultdict(dict)
    n_lines = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line.strip())
            hand_id = rec["hand_id"]
            turn = rec["turn"]
            hands[hand_id][turn] = rec
            n_lines += 1

    elapsed = time.time() - start_time
    print(f"  {n_lines:,} lines, {len(hands):,} hands in {elapsed:.1f}s")

    # Count valid turn records
    n_base_samples = 0
    for hand_id, turns in hands.items():
        if -1 not in turns:
            continue
        for t in range(5):
            if t in turns and "candidates" in turns[t]:
                n_base_samples += 1

    n_samples = n_base_samples * n_augs
    print(f"  {n_base_samples:,} base samples × {n_augs} = {n_samples:,} total")

    # Allocate arrays
    obs_dim = 822 if include_hist else STATE_DIM  # 522 or 822
    obs_arr = np.zeros((n_samples, obs_dim), dtype=np.float32)
    score_arr = np.zeros(n_samples, dtype=np.float32)
    turn_arr = np.zeros(n_samples, dtype=np.int64)
    busted_arr = np.zeros(n_samples, dtype=np.float32)
    fl_arr = np.zeros(n_samples, dtype=np.float32)
    mc_ev_arr = np.zeros(n_samples, dtype=np.float32)
    mc_bust_arr = np.zeros(n_samples, dtype=np.float32)
    mc_fl_arr = np.zeros(n_samples, dtype=np.float32)

    idx = 0
    n_encode_fail = 0
    start_encode = time.time()

    for hand_id, turns in hands.items():
        if -1 not in turns:
            continue

        final = turns[-1]
        final_score = final.get("score", 0)
        is_busted = final.get("busted", False)
        is_fl = final.get("fl_entry", False)

        for t in range(5):
            if t not in turns:
                continue
            rec = turns[t]
            if "candidates" not in rec or not rec["candidates"]:
                continue

            # Extract MC stats
            candidates = rec["candidates"]
            best = candidates[0]
            if "mc" in best:
                mc = best["mc"]
                best_ev = mc.get("avg_score", 0.0)
                best_bust = mc.get("bust_rate", 0.0)
                best_fl = mc.get("fl_rate", 0.0)
            else:
                best_ev = best.get("ev", 0.0)
                best_bust = best.get("bust_prob", 0.0)
                best_fl = best.get("fl_rate", 0.0)

            # Board and card data
            board_dict = rec.get("board", {"top": [], "mid": [], "bot": []})
            dealt = list(rec.get("dealt", []))
            exclude = list(rec.get("exclude", []))
            hist_data = rec.get("hist_after", None)

            # Apply suit permutations
            perms = SUIT_PERMS[:n_augs]
            for perm in perms:
                try:
                    # Permute cards
                    p_board = {
                        "top": permute_cards(board_dict.get("top", []), perm),
                        "mid": permute_cards(board_dict.get("mid", []), perm),
                        "bot": permute_cards(board_dict.get("bot", []), perm),
                    }
                    p_dealt = permute_cards(dealt, perm)
                    p_exclude = permute_cards(exclude, perm)

                    # Histogram: only available for identity perm
                    p_hist = hist_data if perm == SUIT_PERMS[0] else None

                    state = encode_one_sample(
                        p_board, p_dealt, p_exclude, t,
                        hist_data=p_hist, include_hist=include_hist,
                    )

                    obs_arr[idx] = state
                    score_arr[idx] = final_score
                    turn_arr[idx] = t
                    busted_arr[idx] = float(is_busted)
                    fl_arr[idx] = float(is_fl)
                    mc_ev_arr[idx] = best_ev
                    mc_bust_arr[idx] = best_bust
                    mc_fl_arr[idx] = best_fl
                    idx += 1

                except Exception as e:
                    n_encode_fail += 1
                    if n_encode_fail <= 5:
                        print(f"  Warning: encode failed for hand {hand_id} turn {t}: {e}")

        if idx % 10000 == 0 and idx > 0:
            elapsed = time.time() - start_encode
            print(f"  {idx:,}/{n_samples:,} encoded ({elapsed:.1f}s, "
                  f"{idx/elapsed:.0f}/s)")

    # Trim to actual count
    actual = idx
    obs_arr = obs_arr[:actual]
    score_arr = score_arr[:actual]
    turn_arr = turn_arr[:actual]
    busted_arr = busted_arr[:actual]
    fl_arr = fl_arr[:actual]
    mc_ev_arr = mc_ev_arr[:actual]
    mc_bust_arr = mc_bust_arr[:actual]
    mc_fl_arr = mc_fl_arr[:actual]

    elapsed = time.time() - start_encode
    print(f"\nEncoding done: {actual:,} samples in {elapsed:.1f}s")
    if n_encode_fail > 0:
        print(f"  ({n_encode_fail} encoding failures)")

    # Save NPZ
    npz_path = output_dir / "mc_teacher.npz"
    print(f"Saving to {npz_path}...")
    np.savez_compressed(
        npz_path,
        obs=obs_arr,
        score=score_arr,
        turn=turn_arr,
        busted=busted_arr,
        fl_entry=fl_arr,
        mc_ev=mc_ev_arr,
        mc_bust=mc_bust_arr,
        mc_fl=mc_fl_arr,
    )

    # Statistics
    print(f"\n{'='*60}")
    print(f"DATASET STATISTICS")
    print(f"{'='*60}")
    print(f"  Total samples:       {actual:,}")
    print(f"  State dimension:     {obs_dim}")
    print(f"  Augmentation:        ×{n_augs}")
    print(f"  Score range:         [{score_arr.min():.1f}, {score_arr.max():.1f}]")
    print(f"  Score mean:          {score_arr.mean():.2f}")
    print(f"  Score std:           {score_arr.std():.2f}")
    print(f"  Bust rate:           {busted_arr.mean()*100:.1f}%")
    print(f"  FL rate:             {fl_arr.mean()*100:.1f}%")
    print(f"  MC best EV mean:     {mc_ev_arr.mean():.2f}")
    print(f"  MC bust prob mean:   {mc_bust_arr.mean()*100:.1f}%")
    print(f"  MC FL rate mean:     {mc_fl_arr.mean()*100:.1f}%")
    print(f"\n  Turn distribution:")
    for t in range(5):
        mask = turn_arr == t
        n = mask.sum()
        avg_ev = mc_ev_arr[mask].mean() if n > 0 else 0
        print(f"    T{t}: {n:,} samples  (avg MC EV: {avg_ev:.2f})")

    # Save metadata
    metadata = {
        "total_samples": actual,
        "state_dim": obs_dim,
        "augmentation": n_augs,
        "score_mean": float(score_arr.mean()),
        "score_std": float(score_arr.std()),
        "bust_rate": float(busted_arr.mean()),
        "fl_rate": float(fl_arr.mean()),
        "mc_ev_mean": float(mc_ev_arr.mean()),
        "turn_counts": {str(t): int((turn_arr == t).sum()) for t in range(5)},
        "encoding_failures": n_encode_fail,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.1f}s")
    print(f"Output: {npz_path} ({npz_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess MC teacher data for VN training")
    parser.add_argument("input", help="Path to merged JSONL file")
    parser.add_argument("--output", default="data/processed_mc",
                        help="Output directory for NPZ files")
    parser.add_argument("--augment", action="store_true",
                        help="Enable suit symmetry augmentation (×4)")
    parser.add_argument("--hist", action="store_true",
                        help="Include histogram features (522 → 822 dims)")
    args = parser.parse_args()
    preprocess_mc_teacher(args.input, args.output,
                          augment=args.augment, include_hist=args.hist)
