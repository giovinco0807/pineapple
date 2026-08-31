"""
Convert human play data from SQLite to JSONL for preprocess_fast.py

Supports suit permutation augmentation (up to 24x data).

Usage:
    python ai/training/collect_human_data.py --db data/ofc_logs.db --output data/ryo_hands.jsonl
    python ai/training/collect_human_data.py --db data/ofc_logs.db --output data/ryo_aug.jsonl --augment
"""
import sqlite3
import json
import itertools
import argparse
from pathlib import Path

SUIT_PERMS = list(itertools.permutations(['h', 'd', 'c', 's']))


def permute_card(card: str, mapping: dict) -> str:
    """Apply suit permutation to a single card. Jokers unchanged."""
    if not card or len(card) < 2 or card[0] == 'X':
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


def collect(db_path: str, output_path: str, augment: bool = False):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Load hand results
    hand_results = {}
    for row in conn.execute("SELECT hand_id, result_detail, btn FROM hands WHERE result_detail IS NOT NULL"):
        try:
            detail = json.loads(row["result_detail"])
            hand_results[row["hand_id"]] = {"detail": detail, "btn": row["btn"]}
        except (json.JSONDecodeError, TypeError):
            pass

    print(f"Loaded {len(hand_results)} hands with results")

    perms = SUIT_PERMS if augment else [('h', 'd', 'c', 's')]  # identity only
    print(f"Suit permutations: {len(perms)}x")

    count = 0
    skipped = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for row in conn.execute(
            "SELECT hand_id, turn, player, board_self, board_opponent, "
            "dealt_cards, known_discards, action_placements, action_discard "
            "FROM turns ORDER BY hand_id, turn, player"
        ):
            hand_id = row["hand_id"]
            if hand_id not in hand_results:
                skipped += 1
                continue

            hr_info = hand_results[hand_id]
            detail = hr_info["detail"]
            btn = hr_info["btn"]
            player = row["player"]

            try:
                board_self = json.loads(row["board_self"])
                board_opp = json.loads(row["board_opponent"])
                dealt_cards = json.loads(row["dealt_cards"])
                placements_raw = json.loads(row["action_placements"])
                discards_self = json.loads(row["known_discards"]) if row["known_discards"] else []
                discard = row["action_discard"]

                # Normalize placements
                placements = []
                for p in placements_raw:
                    if isinstance(p, list):
                        placements.append(p)
                    elif isinstance(p, dict):
                        placements.append([p["card"], p["row"]])

                # Hand result
                royalties_data = detail.get("royalty", detail.get("royalties", {}))
                busted_data = detail.get("busted", [False, False])
                fl_data = detail.get("fl_entry", [False, False])

                if isinstance(royalties_data, list):
                    royalties = {
                        str(i): {"total": sum(r.values()) if isinstance(r, dict) else r}
                        for i, r in enumerate(royalties_data)
                    }
                elif isinstance(royalties_data, dict):
                    royalties = {}
                    for k, v in royalties_data.items():
                        royalties[str(k)] = {"total": v if isinstance(v, (int, float)) else sum(v.values()) if isinstance(v, dict) else 0}
                else:
                    royalties = {"0": {"total": 0}, "1": {"total": 0}}

                if isinstance(busted_data, list):
                    busted = {str(i): v for i, v in enumerate(busted_data)}
                else:
                    busted = {"0": False, "1": False}

                if isinstance(fl_data, list):
                    fl_entry = {str(i): v for i, v in enumerate(fl_data)}
                else:
                    fl_entry = {"0": False, "1": False}

                hand_result = {
                    "royalties": royalties,
                    "busted": busted,
                    "fl_entry": fl_entry,
                }

                # Apply each suit permutation
                for perm in perms:
                    mapping = dict(zip(['h', 'd', 'c', 's'], perm))

                    turn_log = {
                        "board_self": permute_board(board_self, mapping),
                        "board_opponent": permute_board(board_opp, mapping),
                        "dealt_cards": permute_cards(dealt_cards, mapping),
                        "discards_self": permute_cards(discards_self, mapping),
                        "turn": row["turn"],
                        "is_btn": player == btn,
                        "action": {
                            "placements": permute_placements(placements, mapping),
                            "discard": permute_card(discard, mapping) if discard else None,
                        },
                        "player": player,
                    }

                    record = {"turn_log": turn_log, "hand_result": hand_result}
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1

            except Exception as e:
                skipped += 1
                if skipped <= 5:
                    print(f"  [WARN] Skip {hand_id}/{row['turn']}/p{player}: {e}")

    conn.close()
    print(f"Exported {count} turns ({skipped} skipped) to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/ofc_logs.db")
    parser.add_argument("--output", default="data/ryo_hands.jsonl")
    parser.add_argument("--augment", action="store_true", help="Apply 24x suit permutation")
    args = parser.parse_args()
    collect(args.db, args.output, args.augment)
