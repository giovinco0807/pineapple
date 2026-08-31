"""
Convert Rust CFR T0 solver output to ranking-based training data.

Instead of extracting only the best action per hand, this converts ALL
placements with their EVs into training samples for ranking/regression learning.

Each hand with ~200 placements produces ~200 training samples, each labeled
with the CFR-evaluated EV. With 336 hands, this yields ~67,000 samples.
With suit augmentation (x24), approximately 1.6M samples.

Output format (JSONL):
  {"turn_log": {...}, "hand_result": {...}, "cfr_ev": 30.57, "hand_type": "..."}

Usage:
    python ai/training/convert_t0_allactions.py --input ai/data/t0_gcs/ --output ai/data/t0_allactions.jsonl
    python ai/training/convert_t0_allactions.py --input ai/data/t0_gcs/ --output ai/data/t0_allactions_aug.jsonl --augment
"""
import json
import re
import itertools
import argparse
from pathlib import Path


SUIT_PERMS = list(itertools.permutations(['h', 'd', 'c', 's']))


def parse_placement_string(p_str: str) -> dict:
    """Parse 'Top[Ks] Mid[6s 2c] Bot[9d Jh]' -> {top: [Ks], middle: [6s, 2c], bottom: [9d, Jh]}"""
    result = {"top": [], "middle": [], "bottom": []}
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
    action = []
    for card in placement["top"]:
        action.append([card, "top"])
    for card in placement["middle"]:
        action.append([card, "middle"])
    for card in placement["bottom"]:
        action.append([card, "bottom"])
    return action


def permute_card(card: str, mapping: dict) -> str:
    if not card or len(card) < 2:
        return card
    rank, suit = card[:-1], card[-1]
    return rank + mapping.get(suit, suit)


def permute_cards(cards: list, mapping: dict) -> list:
    return [permute_card(c, mapping) for c in cards]


def permute_placements(placements: list, mapping: dict) -> list:
    return [[permute_card(p[0], mapping), p[1]] for p in placements]


def convert_cfr_allactions(cfr_record: dict) -> list:
    """Convert one CFR record to multiple BC records (one per placement)."""
    cards_str = cfr_record["hand"]
    dealt_cards = cards_str.split()

    placements = cfr_record.get("placements", [])
    if not placements:
        return []

    hand_type = cfr_record.get("type", "unknown")
    empty_board = {"top": [], "middle": [], "bottom": []}

    records = []
    for p in placements:
        parsed = parse_placement_string(p["p"])
        action = placement_to_action(parsed)
        ev = p["ev"]

        turn_log = {
            "board_self": empty_board,
            "board_opponent": empty_board,
            "dealt_cards": dealt_cards,
            "discards_self": [],
            "turn": 0,
            "player": 0,
            "is_btn": True,
            "action": {
                "placements": action,
                "discard": None,
            },
        }

        estimated_fl = ev >= 7.0
        hand_result = {
            "royalties": {"0": {"total": max(0, ev)}, "1": {"total": 0}},
            "busted": {"0": False, "1": False},
            "fl_entry": {"0": estimated_fl, "1": False},
        }

        records.append({
            "turn_log": turn_log,
            "hand_result": hand_result,
            "cfr_ev": ev,
            "hand_type": hand_type,
        })

    return records


def process_files(input_path: str, output_path: str, augment: bool = False):
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

    total_hands = 0
    total_out = 0
    skipped = 0
    ev_stats = []
    hand_types = {}

    with open(output_path, "w", encoding="utf-8") as out:
        for fpath in files:
            file_count = 0
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        cfr_record = json.loads(line)
                    except json.JSONDecodeError:
                        skipped += 1
                        continue

                    bc_records = convert_cfr_allactions(cfr_record)
                    if not bc_records:
                        skipped += 1
                        continue

                    total_hands += 1
                    ht = cfr_record.get("type", "unknown")
                    hand_types[ht] = hand_types.get(ht, 0) + 1

                    for bc_rec in bc_records:
                        ev_stats.append(bc_rec["cfr_ev"])

                        for perm in perms:
                            mapping = dict(zip(['h', 'd', 'c', 's'], perm))
                            tl = bc_rec["turn_log"]

                            aug_rec = {
                                "turn_log": {
                                    "board_self": {"top": [], "middle": [], "bottom": []},
                                    "board_opponent": {"top": [], "middle": [], "bottom": []},
                                    "dealt_cards": permute_cards(tl["dealt_cards"], mapping),
                                    "discards_self": [],
                                    "turn": 0,
                                    "player": 0,
                                    "is_btn": True,
                                    "action": {
                                        "placements": permute_placements(
                                            tl["action"]["placements"], mapping),
                                        "discard": None,
                                    },
                                },
                                "hand_result": bc_rec["hand_result"],
                                "cfr_ev": bc_rec["cfr_ev"],
                                "hand_type": bc_rec["hand_type"],
                            }

                            out.write(json.dumps(aug_rec, ensure_ascii=False) + "\n")
                            total_out += 1
                            file_count += 1

            print(f"  {fpath.name}: {file_count} records output")

    # Stats
    print(f"\n{'='*60}")
    print(f"Conversion Summary:")
    print(f"  Input hands:    {total_hands}")
    print(f"  Input actions:  {len(ev_stats)}")
    print(f"  Output records: {total_out} ({len(perms)}x augmentation)")
    print(f"  Skipped:        {skipped}")
    print(f"\nHand Type Distribution:")
    for ht, count in sorted(hand_types.items(), key=lambda x: -x[1]):
        print(f"  {ht}: {count}")

    if ev_stats:
        import numpy as np
        evs = np.array(ev_stats)
        print(f"\nEV Statistics (all placements):")
        print(f"  Mean:   {evs.mean():.2f}")
        print(f"  Median: {np.median(evs):.2f}")
        print(f"  Std:    {evs.std():.2f}")
        print(f"  Min:    {evs.min():.2f}")
        print(f"  Max:    {evs.max():.2f}")
        print(f"  % with EV >= 7 (FL-likely): {(evs >= 7).sum()}/{len(evs)} ({100*(evs >= 7).mean():.1f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Rust CFR T0 data to all-action training format")
    parser.add_argument("--input", default="ai/data/t0_gcs/",
                        help="Input directory or file")
    parser.add_argument("--output", default="ai/data/t0_allactions.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--augment", action="store_true",
                        help="Apply 24x suit permutation augmentation")
    args = parser.parse_args()

    process_files(args.input, args.output, args.augment)
