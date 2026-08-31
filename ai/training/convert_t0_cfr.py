"""
Convert Rust CFR T0 solver output to preprocess_fast.py-compatible JSONL.

The Rust CFR solver outputs:
  {"hand_idx": 0, "hand": "6s Ks 9d Jh 2c", "type": "NoPair_HiK",
   "n_placements": 232, "n_samples": 50, "nesting": [3,2,1],
   "placements": [{"p": "Top[Ks] Mid[6s 2c] Bot[9d Jh]", "ev": 30.573}, ...]}

preprocess_fast.py expects:
  {"turn_log": {"board_self": {top:[], mid:[], bot:[]}, "board_opponent": {...},
   "dealt_cards": [...], "discards_self": [], "turn": 0, "is_btn": true,
   "action": {"placements": [["Ks","top"], ["6s","middle"], ...], "discard": null}},
   "hand_result": {"royalties": {...}, "busted": {...}, "fl_entry": {...}}}

Supports:
  - Top-1 (best action only, default)
  - Top-K soft labels (--top-k 5: top 5 actions with EV-based probabilities)
  - Suit-permutation augmentation (--augment: 24x data)

Usage:
    python ai/training/convert_t0_cfr.py --input ai/data/t0_gcs/ --output ai/data/t0_converted.jsonl
    python ai/training/convert_t0_cfr.py --input ai/data/t0_gcs/ --output ai/data/t0_aug.jsonl --augment
    python ai/training/convert_t0_cfr.py --input ai/data/t0_gcs/ --output ai/data/t0_soft.jsonl --top-k 5
"""
import json
import re
import math
import itertools
import argparse
from pathlib import Path

SUIT_PERMS = list(itertools.permutations(['h', 'd', 'c', 's']))


def parse_placement_string(p_str: str) -> dict:
    """Parse 'Top[Ks] Mid[6s 2c] Bot[9d Jh]' -> {top: [Ks], middle: [6s, 2c], bottom: [9d, Jh]}"""
    result = {"top": [], "middle": [], "bottom": []}
    
    # Match Top[...], Mid[...], Bot[...]
    top_match = re.search(r'Top\[([^\]]*)\]', p_str)
    mid_match = re.search(r'Mid\[([^\]]*)\]', p_str)
    bot_match = re.search(r'Bot\[([^\]]*)\]', p_str)
    
    if top_match and top_match.group(1).strip():
        result["top"] = top_match.group(1).strip().split()
    if mid_match and mid_match.group(1).strip():
        result["middle"] = mid_match.group(1).strip().split()
    if bot_match and bot_match.group(1).strip():
        result["bottom"] = bot_match.group(1).strip().split()
    
    return result


def placement_to_action(placement: dict) -> list:
    """Convert {top: [Ks], middle: [6s, 2c], bottom: [9d, Jh]} -> [[Ks, top], [6s, middle], ...]"""
    action = []
    for card in placement["top"]:
        action.append([card, "top"])
    for card in placement["middle"]:
        action.append([card, "middle"])
    for card in placement["bottom"]:
        action.append([card, "bottom"])
    return action


def ev_to_soft_probs(evs: list, temperature: float = 1.0) -> list:
    """Convert EV values to softmax probabilities."""
    max_ev = max(evs)
    exps = [math.exp((ev - max_ev) / temperature) for ev in evs]
    total = sum(exps)
    return [e / total for e in exps]


def permute_card(card: str, mapping: dict) -> str:
    """Apply suit permutation to a single card."""
    if not card or len(card) < 2:
        return card
    rank, suit = card[:-1], card[-1]
    return rank + mapping.get(suit, suit)


def permute_cards(cards: list, mapping: dict) -> list:
    return [permute_card(c, mapping) for c in cards]


def permute_board(board: dict, mapping: dict) -> dict:
    return {
        'top': permute_cards(board.get('top', []), mapping),
        'middle': permute_cards(board.get('middle', []), mapping),
        'bottom': permute_cards(board.get('bottom', []), mapping),
    }


def permute_placements(placements: list, mapping: dict) -> list:
    return [[permute_card(p[0], mapping), p[1]] for p in placements]


def convert_cfr_to_bc(cfr_record: dict, top_k: int = 1,
                      temperature: float = 2.0) -> list:
    """Convert one CFR record to one or more BC training records.
    
    Args:
        cfr_record: One line from Rust CFR JSONL
        top_k: Number of top actions to include (1 = hard label, >1 = soft labels)
        temperature: Softmax temperature for EV-to-probability conversion
    
    Returns:
        List of BC-format dicts (usually 1, or top_k for soft label mode)
    """
    cards_str = cfr_record["hand"]
    dealt_cards = cards_str.split()
    
    placements = cfr_record.get("placements", [])
    if not placements:
        return []
    
    # Sort by EV descending (should already be sorted, but be safe)
    placements.sort(key=lambda x: x["ev"], reverse=True)
    
    # Use top-k placements
    top_placements = placements[:top_k]
    
    # Parse best placement for the action
    best = parse_placement_string(top_placements[0]["p"])
    best_action = placement_to_action(best)
    best_ev = top_placements[0]["ev"]
    
    # Empty boards (T0 = initial placement)
    empty_board = {"top": [], "middle": [], "bottom": []}
    
    # Build turn_log
    turn_log = {
        "board_self": empty_board,
        "board_opponent": empty_board,
        "dealt_cards": dealt_cards,
        "discards_self": [],
        "turn": 0,
        "player": 0,  # T0 data is single-player perspective
        "is_btn": True,  # T0 data doesn't have position info; default btn
        "action": {
            "placements": best_action,
            "discard": None,
        },
    }
    
    # Build hand_result (estimate from EV)
    # EV > 7 typically means FL entry is likely (royalties >= 7 = QQ+)
    estimated_fl = best_ev >= 7.0
    hand_result = {
        "royalties": {
            "0": {"total": max(0, best_ev)},
            "1": {"total": 0},
        },
        "busted": {"0": False, "1": False},
        "fl_entry": {"0": estimated_fl, "1": False},
    }
    
    records = [{"turn_log": turn_log, "hand_result": hand_result}]
    
    # For soft-label mode, add metadata about top-k EVs
    if top_k > 1 and len(top_placements) > 1:
        evs = [p["ev"] for p in top_placements]
        probs = ev_to_soft_probs(evs, temperature)
        
        # Store soft label info in the record
        soft_labels = []
        for p, prob in zip(top_placements, probs):
            parsed = parse_placement_string(p["p"])
            action = placement_to_action(parsed)
            soft_labels.append({
                "action": action,
                "ev": p["ev"],
                "prob": prob,
            })
        records[0]["soft_labels"] = soft_labels
    
    # Store all valid placements count for action masking
    records[0]["n_valid_actions"] = cfr_record.get("n_placements", len(placements))
    records[0]["cfr_ev"] = best_ev
    records[0]["hand_type"] = cfr_record.get("type", "unknown")
    
    return records


def process_files(input_path: str, output_path: str,
                  augment: bool = False, top_k: int = 1,
                  temperature: float = 2.0):
    """Process all JSONL files in input directory."""
    input_dir = Path(input_path)
    
    if input_dir.is_file():
        files = [input_dir]
    else:
        files = sorted(input_dir.glob("*.jsonl"))
    
    if not files:
        print(f"No JSONL files found in {input_path}")
        return
    
    perms = SUIT_PERMS if augment else [('h', 'd', 'c', 's')]
    print(f"Processing {len(files)} files")
    print(f"Suit permutations: {len(perms)}x")
    print(f"Top-K: {top_k}")
    
    total_in = 0
    total_out = 0
    skipped = 0
    hand_types = {}
    ev_stats = []
    
    with open(output_path, "w", encoding="utf-8") as out:
        for fpath in files:
            file_count = 0
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    total_in += 1
                    
                    try:
                        cfr_record = json.loads(line)
                    except json.JSONDecodeError as e:
                        skipped += 1
                        if skipped <= 5:
                            print(f"  [WARN] JSON decode error in {fpath.name}: {e}")
                        continue
                    
                    # Convert to BC format
                    bc_records = convert_cfr_to_bc(cfr_record, top_k, temperature)
                    if not bc_records:
                        skipped += 1
                        continue
                    
                    # Track stats
                    ht = cfr_record.get("type", "unknown")
                    hand_types[ht] = hand_types.get(ht, 0) + 1
                    best_ev = cfr_record["placements"][0]["ev"]
                    ev_stats.append(best_ev)
                    
                    # Apply suit permutations
                    for bc_rec in bc_records:
                        for perm in perms:
                            mapping = dict(zip(['h', 'd', 'c', 's'], perm))
                            
                            aug_rec = {
                                "turn_log": {
                                    "board_self": permute_board(
                                        bc_rec["turn_log"]["board_self"], mapping),
                                    "board_opponent": permute_board(
                                        bc_rec["turn_log"]["board_opponent"], mapping),
                                    "dealt_cards": permute_cards(
                                        bc_rec["turn_log"]["dealt_cards"], mapping),
                                    "discards_self": [],
                                    "turn": 0,
                                    "player": bc_rec["turn_log"].get("player", 0),
                                    "is_btn": bc_rec["turn_log"]["is_btn"],
                                    "action": {
                                        "placements": permute_placements(
                                            bc_rec["turn_log"]["action"]["placements"],
                                            mapping),
                                        "discard": None,
                                    },
                                },
                                "hand_result": bc_rec["hand_result"],
                            }
                            
                            # Carry forward metadata
                            if "soft_labels" in bc_rec:
                                aug_rec["soft_labels"] = [
                                    {
                                        "action": permute_placements(sl["action"], mapping),
                                        "ev": sl["ev"],
                                        "prob": sl["prob"],
                                    }
                                    for sl in bc_rec["soft_labels"]
                                ]
                            if "cfr_ev" in bc_rec:
                                aug_rec["cfr_ev"] = bc_rec["cfr_ev"]
                            if "hand_type" in bc_rec:
                                aug_rec["hand_type"] = bc_rec["hand_type"]
                            
                            out.write(json.dumps(aug_rec, ensure_ascii=False) + "\n")
                            total_out += 1
                            file_count += 1
            
            print(f"  {fpath.name}: {file_count} records output")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Conversion Summary:")
    print(f"  Input records:  {total_in}")
    print(f"  Output records: {total_out} ({len(perms)}x augmentation)")
    print(f"  Skipped:        {skipped}")
    print(f"\nHand Type Distribution:")
    for ht, count in sorted(hand_types.items(), key=lambda x: -x[1]):
        print(f"  {ht}: {count}")
    
    if ev_stats:
        print(f"\nEV Statistics:")
        print(f"  Mean: {sum(ev_stats)/len(ev_stats):.2f}")
        print(f"  Min:  {min(ev_stats):.2f}")
        print(f"  Max:  {max(ev_stats):.2f}")
        fl_count = sum(1 for ev in ev_stats if ev >= 7.0)
        print(f"  FL-likely (EV>=7): {fl_count}/{len(ev_stats)} ({100*fl_count/len(ev_stats):.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Rust CFR T0 data to BC training format")
    parser.add_argument("--input", default="ai/data/t0_gcs/",
                        help="Input directory or file")
    parser.add_argument("--output", default="ai/data/t0_converted.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--augment", action="store_true",
                        help="Apply 24x suit permutation augmentation")
    parser.add_argument("--top-k", type=int, default=1,
                        help="Number of top actions to include (1=hard, >1=soft labels)")
    parser.add_argument("--temperature", type=float, default=2.0,
                        help="Softmax temperature for EV-to-prob (soft label mode)")
    args = parser.parse_args()
    
    process_files(args.input, args.output, args.augment, args.top_k, args.temperature)
