"""
Convert Rust GameBatch output (T1-T4) to preprocess_fast.py-compatible JSONL.

The Rust GameBatch outputs:
  {"game_id":0, "turn":1, "board":"Top[Ks] Mid[6s 2c] Bot[9d Jh]",
   "hand":"Th 5d 3c", "n_actions":18, "n_samples":30,
   "actions":[{"a":"d:Th 5d→Middle 3c→Bottom","ev":25.3}, ...]}

preprocess_fast.py expects:
  {"turn_log": {"board_self": {top:[], mid:[], bot:[]},
   "dealt_cards": [...], "discards_self": [...], "turn": 1,
   "action": {"placements": [["5d","middle"],["3c","bottom"]], "discard": "Th"}},
   "hand_result": {"royalties": {...}, "busted": {...}, "fl_entry": {...}}}

Supports:
  - Top-1 (best action only, default)
  - Suit-permutation augmentation (--augment: 24x data)

Usage:
    python convert_turn_cfr.py --input game_batch_results.jsonl --output t1t4_converted.jsonl
    python convert_turn_cfr.py --input game_batch_results.jsonl --output t1t4_aug.jsonl --augment
"""
import json
import re
import itertools
import argparse
from pathlib import Path

SUIT_PERMS = list(itertools.permutations(['h', 'd', 'c', 's']))
ORIG_SUITS = ['h', 'd', 'c', 's']

ROW_MAP = {
    'Top': 'top',
    'Middle': 'middle',
    'Bottom': 'bottom',
}


def parse_board_string(board_str: str) -> dict:
    """Parse 'Top[Ks] Mid[6s 2c] Bot[9d Jh]' -> {top: [Ks], middle: [6s, 2c], bottom: [9d, Jh]}"""
    result = {"top": [], "middle": [], "bottom": []}
    
    top_match = re.search(r'Top\[([^\]]*)\]', board_str)
    mid_match = re.search(r'Mid\[([^\]]*)\]', board_str)
    bot_match = re.search(r'Bot\[([^\]]*)\]', board_str)
    
    if top_match and top_match.group(1).strip():
        result["top"] = [c for c in top_match.group(1).strip().split() if c != '-']
    if mid_match and mid_match.group(1).strip():
        result["middle"] = [c for c in mid_match.group(1).strip().split() if c != '-']
    if bot_match and bot_match.group(1).strip():
        result["bottom"] = [c for c in bot_match.group(1).strip().split() if c != '-']
    
    return result


def parse_action_string(action_str: str) -> dict:
    """Parse 'd:Th 5d→Middle 3c→Bottom' -> {discard: 'Th', placements: [['5d','middle'],['3c','bottom']]}
    
    Also handles Debug Row format: 'd:Th 5d→Middle 3c→Bottom'
    """
    result = {"discard": None, "placements": []}
    
    # Extract discard: 'd:Th' or 'd:Jo'
    d_match = re.match(r'd:(\S+)', action_str)
    if d_match:
        result["discard"] = d_match.group(1)
    
    # Extract placements: 'Card→Row' patterns
    # Row names from Rust Debug format: Top, Middle, Bottom
    place_pattern = re.findall(r'(\S+)→(\w+)', action_str)
    for card, row_debug in place_pattern:
        if card.startswith('d:'):
            continue
        row = ROW_MAP.get(row_debug, row_debug.lower())
        result["placements"].append([card, row])
    
    return result


def permute_card(card: str, mapping: dict) -> str:
    """Apply suit permutation to a single card."""
    if not card or len(card) < 2 or card in ('Jo', 'Joker'):
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


def convert_turn_record(record: dict) -> list:
    """Convert one Rust GameBatch JSONL line to BC training records.
    
    Returns list of BC-format dicts (1 per record, more with augmentation).
    """
    if "error" in record:
        return []
    
    turn = record.get("turn", 0)
    board_str = record.get("board", "")
    hand_str = record.get("hand", "")
    actions = record.get("actions", [])
    
    if not actions:
        return []
    
    # Parse board
    board = parse_board_string(board_str)
    
    # Parse hand
    dealt_cards = hand_str.split()
    
    # Best action (first in sorted list)
    best_action = parse_action_string(actions[0]["a"])
    best_ev = actions[0]["ev"]
    
    # Previous discards (not tracked in current format, use empty)
    discards_self = []
    
    # Build turn_log
    empty_board = {"top": [], "middle": [], "bottom": []}
    
    turn_log = {
        "board_self": board,
        "board_opponent": empty_board,
        "dealt_cards": dealt_cards,
        "discards_self": discards_self,
        "turn": turn,
        "player": 0,
        "is_btn": True,
        "action": {
            "placements": best_action["placements"],
            "discard": best_action["discard"],
        },
    }
    
    # Build hand_result
    estimated_fl = best_ev >= 7.0
    hand_result = {
        "royalties": {
            "0": {"total": max(0, best_ev)},
            "1": {"total": 0},
        },
        "busted": {"0": False, "1": False},
        "fl_entry": {"0": estimated_fl, "1": False},
    }
    
    result = {
        "turn_log": turn_log,
        "hand_result": hand_result,
        "cfr_ev": best_ev,
        "n_valid_actions": record.get("n_actions", len(actions)),
        "game_id": record.get("game_id", -1),
    }
    
    # Store all actions with EVs for analysis
    result["all_actions"] = [
        {"action": parse_action_string(a["a"]), "ev": a["ev"]}
        for a in actions
    ]
    
    return [result]


def process_files(input_path: str, output_path: str, augment: bool = False,
                  turns: list = None):
    """Process JSONL files and convert to BC format."""
    input_p = Path(input_path)
    
    if input_p.is_file():
        files = [input_p]
    else:
        files = sorted(input_p.glob("*.jsonl"))
    
    if not files:
        print(f"No JSONL files found in {input_path}")
        return
    
    perms = SUIT_PERMS if augment else [tuple(ORIG_SUITS)]
    
    total = 0
    skipped = 0
    turn_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    
    with open(output_path, 'w') as out_f:
        for fpath in files:
            print(f"Processing {fpath.name}...")
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        skipped += 1
                        continue
                    
                    # Filter by turn if specified
                    turn = record.get("turn", 0)
                    if turns and turn not in turns:
                        continue
                    
                    bc_records = convert_turn_record(record)
                    
                    for bc in bc_records:
                        for perm in perms:
                            mapping = dict(zip(ORIG_SUITS, perm))
                            
                            # Apply permutation
                            aug = json.loads(json.dumps(bc))
                            tl = aug["turn_log"]
                            tl["board_self"] = permute_board(tl["board_self"], mapping)
                            tl["board_opponent"] = permute_board(tl["board_opponent"], mapping)
                            tl["dealt_cards"] = permute_cards(tl["dealt_cards"], mapping)
                            tl["discards_self"] = permute_cards(tl["discards_self"], mapping)
                            
                            # Permute action
                            action = tl["action"]
                            action["placements"] = [
                                [permute_card(p[0], mapping), p[1]]
                                for p in action["placements"]
                            ]
                            if action["discard"]:
                                action["discard"] = permute_card(action["discard"], mapping)
                            
                            out_f.write(json.dumps(aug) + '\n')
                            total += 1
                            turn_counts[turn] = turn_counts.get(turn, 0) + 1
    
    print(f"\nDone: {total} records written to {output_path}")
    if augment:
        print(f"  ({total // 24} base × 24 suit permutations)")
    print(f"  Skipped: {skipped}")
    print(f"  By turn: {turn_counts}")


def main():
    parser = argparse.ArgumentParser(description="Convert Rust GameBatch T1-T4 output to BC format")
    parser.add_argument("--input", required=True, help="Input JSONL file or directory")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    parser.add_argument("--augment", action="store_true", help="Apply 24x suit permutation augmentation")
    parser.add_argument("--turns", type=str, default="", help="Comma-separated turn filter (e.g. '1,2')")
    args = parser.parse_args()
    
    turns_filter = None
    if args.turns:
        turns_filter = [int(t) for t in args.turns.split(",")]
    
    process_files(args.input, args.output, augment=args.augment, turns=turns_filter)


if __name__ == "__main__":
    main()
