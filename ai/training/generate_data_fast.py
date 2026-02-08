"""
OFC Pineapple - Fast Data Generator for Large-Scale Training

Optimized for 1M+ games. Uses simplified action selection
and batched writing for maximum throughput.

Usage:
    python ai/training/generate_data_fast.py --games 1000000 --output data/train_1m.jsonl
"""
import sys
import json
import random
import time
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.engine.encoding import ALL_CARDS
from ai.engine.game_engine import RANK_VALUES


def fast_game(deck: list) -> list:
    """Play one full hand with fast heuristic and collect samples."""
    btn = random.randint(0, 1)
    boards = [{"top": [], "middle": [], "bottom": []},
              {"top": [], "middle": [], "bottom": []}]
    discards = [[], []]
    samples = []

    # Deal initial 5 each
    hands = {0: deck[:5], 1: deck[5:10]}
    idx = 10

    # Turn 0: heuristic placement
    for seat in [btn, 1 - btn]:
        cards = hands[seat]
        board = boards[seat]
        board_before = {"top": list(board["top"]), "middle": list(board["middle"]),
                        "bottom": list(board["bottom"])}

        placements = heuristic_initial(cards, board)

        samples.append({
            "turn_log": {
                "turn": 0, "player": seat, "is_btn": seat == btn,
                "board_self": board_before,
                "board_opponent": {"top": [], "middle": [], "bottom": []},
                "dealt_cards": list(cards),
                "discards_self": [],
                "action": {"placements": placements, "discard": None},
            }
        })

        for card, pos in placements:
            board[pos].append(card)

    # Turns 1-8
    for turn in range(1, 9):
        complete = all(
            len(b["top"]) == 3 and len(b["middle"]) == 5 and len(b["bottom"]) == 5
            for b in boards
        )
        if complete:
            break

        hands = {0: deck[idx:idx + 3], 1: deck[idx + 3:idx + 6]}
        idx += 6

        for seat in [btn, 1 - btn]:
            cards = hands[seat]
            if not cards or len(cards) < 3:
                continue
            board = boards[seat]

            if (len(board["top"]) == 3 and len(board["middle"]) == 5
                    and len(board["bottom"]) == 5):
                continue

            board_before = {"top": list(board["top"]), "middle": list(board["middle"]),
                            "bottom": list(board["bottom"])}

            placements, discard = heuristic_turn(cards, board)

            samples.append({
                "turn_log": {
                    "turn": turn, "player": seat, "is_btn": seat == btn,
                    "board_self": board_before,
                    "board_opponent": boards[1 - seat].copy(),
                    "dealt_cards": list(cards),
                    "discards_self": list(discards[seat]),
                    "action": {"placements": placements, "discard": discard},
                }
            })

            for card, pos in placements:
                board[pos].append(card)
            if discard:
                discards[seat].append(discard)

    # Score
    hand_result = fast_score(boards)

    for s in samples:
        s["hand_result"] = hand_result

    return samples


def card_rank(card: str) -> int:
    if card in ("X1", "X2"):
        return 14
    return RANK_VALUES.get(card[0], 7)


def heuristic_initial(cards: list, board: dict) -> list:
    """Fast heuristic for turn 0: sort by rank, put low top, high bottom."""
    sorted_cards = sorted(cards, key=card_rank)

    # Find pairs
    ranks = [card_rank(c) for c in sorted_cards]
    rc = Counter(ranks)
    pairs = [(r, [c for c in sorted_cards if card_rank(c) == r])
             for r, cnt in rc.items() if cnt >= 2]

    placements = []
    used = set()

    if pairs:
        # Put best pair in bottom
        best_pair = max(pairs, key=lambda x: x[0])
        for c in best_pair[1][:2]:
            placements.append((c, "bottom"))
            used.add(c)

    remaining = [c for c in sorted_cards if c not in used]

    # Fill: lowest to top, rest to middle, overflow to bottom
    positions = []
    top_space = 3 - len(board["top"])
    mid_space = 5 - len(board["middle"])
    bot_space = 5 - len(board["bottom"]) - len([p for p in placements if p[1] == "bottom"])

    for c in remaining:
        if top_space > 0:
            positions.append("top")
            top_space -= 1
        elif mid_space > 0:
            positions.append("middle")
            mid_space -= 1
        elif bot_space > 0:
            positions.append("bottom")
            bot_space -= 1
        else:
            positions.append("middle")

    for c, p in zip(remaining, positions):
        placements.append((c, p))

    return placements


def heuristic_turn(cards: list, board: dict) -> tuple:
    """Fast heuristic for turns 1-8: keep pair-making cards, discard weakest."""
    sorted_cards = sorted(cards, key=card_rank)

    # Check if any dealt card pairs with a card already on the board
    board_ranks = set()
    for pos in ["top", "middle", "bottom"]:
        for c in board[pos]:
            if c not in ("X1", "X2"):
                board_ranks.add(card_rank(c))

    # Also check for pairs within the 3 dealt cards
    dealt_ranks = [card_rank(c) for c in sorted_cards]
    dealt_rc = Counter(dealt_ranks)
    internal_pair = [c for c in sorted_cards if dealt_rc[card_rank(c)] >= 2]
    board_pair = [c for c in sorted_cards
                  if c not in ("X1", "X2") and card_rank(c) in board_ranks]

    if internal_pair:
        # Keep the internal pair, discard the odd card
        pair_rank = card_rank(internal_pair[0])
        discard = next(c for c in sorted_cards if card_rank(c) != pair_rank)
        remaining = [c for c in sorted_cards if c != discard]
    elif board_pair:
        # Keep card that pairs with board, discard the weakest of the rest
        keep = board_pair[0]
        others = [c for c in sorted_cards if c != keep]
        discard = min(others, key=card_rank)
        remaining = [c for c in sorted_cards if c != discard]
    else:
        # No pairs possible: discard lowest rank
        discard = sorted_cards[0]
        remaining = sorted_cards[1:]

    placements = []
    for c in sorted(remaining, key=card_rank):
        r = card_rank(c)
        # Try to place strategically
        if r <= 8 and len(board["top"]) < 3:
            placements.append((c, "top"))
        elif r >= 10 and len(board["bottom"]) < 5:
            placements.append((c, "bottom"))
        elif len(board["middle"]) < 5:
            placements.append((c, "middle"))
        elif len(board["bottom"]) < 5:
            placements.append((c, "bottom"))
        elif len(board["top"]) < 3:
            placements.append((c, "top"))
        else:
            placements.append((c, "middle"))

    return placements, discard


def fast_score(boards: list) -> dict:
    """Quick scoring without full hand evaluation."""
    busted = [False, False]
    royalties = [{"top": 0, "middle": 0, "bottom": 0, "total": 0},
                 {"top": 0, "middle": 0, "bottom": 0, "total": 0}]
    fl_entry = [False, False]
    raw_score = [0, 0]

    for seat in [0, 1]:
        b = boards[seat]
        top_val = quick_eval(b["top"], 3)
        mid_val = quick_eval(b["middle"], 5)
        bot_val = quick_eval(b["bottom"], 5)

        if top_val > mid_val or mid_val > bot_val:
            busted[seat] = True
        else:
            royalties[seat]["top"] = quick_top_royalty(b["top"])
            royalties[seat]["middle"] = quick_mid_royalty(b["middle"])
            royalties[seat]["bottom"] = quick_bot_royalty(b["bottom"])
            royalties[seat]["total"] = (royalties[seat]["top"] +
                                        royalties[seat]["middle"] +
                                        royalties[seat]["bottom"])

            # Check FL entry: QQ+ (royalty >= 7) or trips (royalty >= 10)
            if royalties[seat]["top"] >= 7:  # QQ+
                fl_entry[seat] = True

    # Line comparison + scoop + royalty difference
    if busted[0] and busted[1]:
        raw_score = [0, 0]
    elif busted[0] and not busted[1]:
        raw_score = [-6 - royalties[1]["total"], 6 + royalties[1]["total"]]
    elif busted[1] and not busted[0]:
        raw_score = [6 + royalties[0]["total"], -6 - royalties[0]["total"]]
    else:
        # Both not busted: compare lines
        line_vals = [{}, {}]
        for seat in [0, 1]:
            b = boards[seat]
            line_vals[seat] = {
                "top": quick_eval(b["top"], 3),
                "middle": quick_eval(b["middle"], 5),
                "bottom": quick_eval(b["bottom"], 5),
            }
        line_total = 0
        for line in ["top", "middle", "bottom"]:
            if line_vals[0][line] > line_vals[1][line]:
                line_total += 1
            elif line_vals[0][line] < line_vals[1][line]:
                line_total -= 1
        scoop_bonus = 3 if abs(line_total) == 3 else 0
        p0 = line_total
        p0 += scoop_bonus if line_total > 0 else (-scoop_bonus if line_total < 0 else 0)
        p0 += royalties[0]["total"] - royalties[1]["total"]
        raw_score = [p0, -p0]

    return {
        "busted": busted,
        "royalties": royalties,
        "fl_entry": fl_entry,
        "raw_score": raw_score,
    }


_B = 15
_B5 = _B ** 5

def _qenc(cat, *ranks):
    val = cat
    for i in range(5):
        val = val * _B + (ranks[i] if i < len(ranks) else 0)
    return val

def _qstraight_high(sr, jokers=0):
    if jokers == 0:
        if sr == [14,5,4,3,2]: return 5
        return sr[0]
    unique = sorted(set(sr), reverse=True)
    for high in range(14, 4, -1):
        needed = set(range(high, high-5, -1))
        if high == 5: needed = {14,5,4,3,2}
        if len(needed - set(unique)) <= jokers and len(set(unique) - needed) == 0:
            return high
    return sr[0] if sr else 0

def quick_eval(cards: list, expected: int) -> int:
    """Fast hand evaluation with base-15 encoding."""
    if len(cards) != expected:
        return 0
    ranks = []; suits = []; jokers = 0
    for c in cards:
        if c in ("X1", "X2", "JK"): jokers += 1
        else:
            ranks.append(RANK_VALUES.get(c[0], 0))
            suits.append(c[1])
    if not ranks:
        return _qenc(3, 14)  # All jokers = trips
    rc = Counter(ranks)
    best = max(rc.values())
    sr = sorted(ranks, reverse=True)
    if expected == 3:
        if best + jokers >= 3:
            if best >= 3: r = max(r for r,c in rc.items() if c >= 3)
            elif best >= 2: r = max(r for r,c in rc.items() if c >= 2)
            else: r = sr[0]
            return _qenc(3, r)
        if best + jokers >= 2:
            if best >= 2:
                pr = max(r for r,c in rc.items() if c >= 2)
                k = sorted([r for r in ranks if r != pr], reverse=True)
            else: pr = sr[0]; k = sr[1:]
            return _qenc(1, pr, k[0] if k else 0)
        return _qenc(0, *sr)
    # 5-card
    sc = Counter(suits)
    is_flush = len(sc) == 1 and (len(suits) + jokers) == expected
    is_straight = _quick_check_straight(sr, jokers)
    pairs = sorted([r for r,c in rc.items() if c >= 2], reverse=True)
    if is_flush and is_straight:
        return _qenc(8, _qstraight_high(sr, jokers))
    if best + jokers >= 4:
        if best >= 4: qr = max(r for r,c in rc.items() if c >= 4)
        elif best >= 3: qr = max(r for r,c in rc.items() if c >= 3)
        else: qr = pairs[0] if pairs else sr[0]
        kicker = max((r for r in ranks if r != qr), default=0)
        return _qenc(7, qr, kicker)
    if best >= 3:
        tr = max(r for r,c in rc.items() if c >= 3)
        pc = [r for r,c in rc.items() if c >= 2 and r != tr]
        if pc: return _qenc(6, tr, max(pc))
    if jokers >= 1 and len(pairs) >= 2:
        return _qenc(6, pairs[0], pairs[1])
    if is_flush:
        return _qenc(5, *sr[:5])
    if is_straight:
        return _qenc(4, _qstraight_high(sr, jokers))
    if best + jokers >= 3:
        if best >= 3: tr = max(r for r,c in rc.items() if c >= 3)
        elif best >= 2: tr = max(r for r,c in rc.items() if c >= 2)
        else: tr = sr[0]
        k = sorted([r for r in ranks if r != tr], reverse=True)
        return _qenc(3, tr, k[0] if len(k)>0 else 0, k[1] if len(k)>1 else 0)
    if len(pairs) >= 2:
        kicker = max((r for r in ranks if r not in pairs[:2]), default=0)
        return _qenc(2, pairs[0], pairs[1], kicker)
    if best >= 2 or jokers >= 1:
        if pairs: pr=pairs[0]; k=sorted([r for r in ranks if r!=pr], reverse=True)
        else: pr=sr[0]; k=sr[1:]
        return _qenc(1, pr, k[0] if len(k)>0 else 0, k[1] if len(k)>1 else 0, k[2] if len(k)>2 else 0)
    return _qenc(0, *sr[:5])


def _quick_check_straight(sorted_ranks: list, jokers: int) -> bool:
    """Check if ranks can form a straight (with jokers)."""
    if len(sorted_ranks) + jokers < 5:
        return False
    unique = sorted(set(sorted_ranks), reverse=True)
    # Check A-high straight: A K Q J T
    if 14 in unique:
        unique_with_low = unique + [1]
    else:
        unique_with_low = unique

    for start_idx in range(len(unique_with_low)):
        high = unique_with_low[start_idx]
        needed = 0
        for i in range(5):
            if (high - i) not in unique_with_low:
                needed += 1
        if needed <= jokers:
            return True
    return False


def quick_mid_royalty(cards: list) -> int:
    """Quick middle row royalty."""
    val = quick_eval(cards, 5)
    cat = val // _B5
    r1 = (val // (_B ** 4)) % _B
    if cat == 8 and r1 == 14: return 50
    if cat == 8: return 30
    if cat == 7: return 20
    if cat == 6: return 12
    if cat == 5: return 8
    if cat == 4: return 4
    if cat == 3: return 2
    return 0


def quick_bot_royalty(cards: list) -> int:
    """Quick bottom row royalty."""
    val = quick_eval(cards, 5)
    cat = val // _B5
    r1 = (val // (_B ** 4)) % _B
    if cat == 8 and r1 == 14: return 25
    if cat == 8: return 15
    if cat == 7: return 10
    if cat == 6: return 6
    if cat == 5: return 4
    if cat == 4: return 2
    return 0


def quick_top_royalty(cards: list) -> int:
    """Quick top row royalty."""
    ranks = []
    jokers = 0
    for c in cards:
        if c in ("X1", "X2"):
            jokers += 1
        else:
            ranks.append(RANK_VALUES.get(c[0], 0))
    rc = Counter(ranks)
    for r in sorted(rc.keys(), reverse=True):
        if rc[r] + jokers >= 3:
            return 10 + (r - 2)
        if rc[r] + jokers >= 2 and r >= 6:
            return r - 5
    return 0


def generate_dataset(num_games: int, output_path: str):
    """Generate training data at scale."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    total_samples = 0
    total_busts = 0
    total_fl = 0
    errors = 0
    start_time = time.time()

    BATCH_SIZE = 10000
    buffer = []

    with open(output, "w", encoding="utf-8") as f:
        for i in range(num_games):
            try:
                deck = list(ALL_CARDS)
                random.shuffle(deck)
                samples = fast_game(deck)
                buffer.extend(samples)

                for s in samples:
                    p = s["turn_log"]["player"]
                    if s["hand_result"]["busted"][p]:
                        total_busts += 1
                    if s["hand_result"]["fl_entry"][p]:
                        total_fl += 1
                total_samples += len(samples)

            except Exception as e:
                errors += 1

            if len(buffer) >= BATCH_SIZE:
                for s in buffer:
                    f.write(json.dumps(s, ensure_ascii=False) + "\n")
                buffer.clear()

            if (i + 1) % 100000 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (num_games - i - 1) / rate
                print(f"  {i+1:,}/{num_games:,} games "
                      f"({total_samples:,} samples, "
                      f"{rate:.0f} games/s, "
                      f"ETA {eta/60:.1f}min)")

        # Flush remaining
        for s in buffer:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    elapsed = time.time() - start_time
    bust_rate = total_busts / max(total_samples, 1)
    fl_rate = total_fl / max(total_samples, 1)

    print(f"\n--- Generation Complete ---")
    print(f"  Games: {num_games - errors:,} ({errors} errors)")
    print(f"  Samples: {total_samples:,}")
    print(f"  Bust rate: {bust_rate:.1%}")
    print(f"  FL rate: {fl_rate:.1%}")
    print(f"  Time: {elapsed:.1f}s ({num_games/elapsed:.0f} games/s)")
    print(f"  Output: {output}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fast data generator")
    parser.add_argument("--games", type=int, default=1000000)
    parser.add_argument("--output", default="data/train_1m.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    print(f"Generating {args.games:,} games...")
    generate_dataset(args.games, args.output)
