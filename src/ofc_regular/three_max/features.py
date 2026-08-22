"""Feature encoding for 3-max street models.

Layout follows the heads-up 168-dim design, widened from two sides to three.
That shape is not arbitrary: heads-up measured feature blocks to be what binds
(a 6x parameter increase bought nothing, while adding the joint-outlook block
moved T3 regret from 0.279 to 0.0036), so the blocks are ported and the widths
are not economised.

    [0, 93)    three side blocks, 31 each: hero, then the two opponents in
               act-relative order (next to act first)
    [93, 201)  three outlook blocks, 36 each, same order
    [201, 221) two head-to-head blocks, 10 each: hero vs each opponent
    [221, 225) context

Street and seat are NOT features.  Models are selected per (street, seat) --
the owner's ruling of 2026-08-12, and the same choice heads-up made -- so both
would be constant inside any model that sees them.  The layout is shared across
seats precisely so a seat's weights can warm-start from another's.

The outlook blocks enumerate each side's two open slots against the WHOLE
unseen set, independently per side.  In a 3-max hand that set is small (16
cards at the BTN's T3) and three sides draw from it, so the enumeration
double-counts cards two sides cannot both receive.  This is deliberate and is
the heads-up convention: an outlook is a summary feature, not a probability the
label depends on, and the label is what carries the truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

from ..cards import RANK_VALUE, card_rank, card_suit
from ..evaluator import evaluate_3_card, evaluate_5_card
from ..state import ROW_CAPACITY, ROWS, Board
from .mc import Terminal, _board_terminal_for_test as board_terminal

FEATURE_SCHEMA = "regular_ofc_3max_features_v1"

SIDE_SIZE = 31
OUTLOOK_SIZE = 36
HEAD_TO_HEAD_SIZE = 10
CONTEXT_SIZE = 4
SIDES = 3
OPPONENTS = 2

DECK_BLOCK_SIZE = 20
CONTEST_SIZE = 8

SIDE_BLOCK_END = SIDE_SIZE * SIDES                       # 93
OUTLOOK_BLOCK_END = SIDE_BLOCK_END + OUTLOOK_SIZE * SIDES  # 201
HEAD_TO_HEAD_END = OUTLOOK_BLOCK_END + HEAD_TO_HEAD_SIZE * OPPONENTS  # 221
CONTEXT_END = HEAD_TO_HEAD_END + CONTEXT_SIZE            # 225
DECK_BLOCK_END = CONTEXT_END + DECK_BLOCK_SIZE           # 245
FEATURE_SIZE = DECK_BLOCK_END + CONTEST_SIZE * OPPONENTS # 261

CATEGORIES = 9
MAX_RANK = 14.0
# Cap the exact outlook enumeration and the head-to-head cross product; both
# stride deterministically rather than sample, so no RNG enters a feature.
MAX_COMPLETIONS = 190
MAX_FINISH_PAIRS = 24


def _straight_window(cards: Sequence[str]) -> float:
    """1.0 when the row's ranks all fit inside one five-card straight window."""
    if not cards:
        return 1.0
    ranks = {card_rank(card) for card in cards}
    if len(ranks) != len(cards):
        return 0.0
    for high in range(5, 15):
        if ranks <= set(range(high - 4, high + 1)):
            return 1.0
    if ranks <= {14, 2, 3, 4, 5}:
        return 1.0
    return 0.0


def _partial_value(cards: Sequence[str], row: str) -> tuple[int, tuple[int, ...]]:
    """Rank a row, complete or not, independently of placement order.

    A complete row goes through the real evaluator.  A partial row is
    summarised by its pairing structure alone -- slicing the first three cards
    would make the feature depend on the order the player happened to place
    them in, which is not information and is not stable.
    """
    if not cards:
        return (0, ())
    if row == "top" and len(cards) == 3:
        return evaluate_3_card(cards)
    if row != "top" and len(cards) == 5:
        return evaluate_5_card(cards)

    counts: dict[int, int] = {}
    for card in cards:
        rank = card_rank(card)
        counts[rank] = counts.get(rank, 0) + 1
    groups = sorted(((count, rank) for rank, count in counts.items()), reverse=True)
    largest = groups[0][0]
    if largest >= 4:
        category = 7  # quads
    elif largest == 3:
        category = 6 if len(groups) > 1 and groups[1][0] >= 2 else 3
    elif largest == 2:
        category = 2 if len(groups) > 1 and groups[1][0] == 2 else 1
    else:
        category = 0
    return (category, tuple(rank for _count, rank in groups))


def _row_block(cards: Sequence[str], row: str) -> list[float]:
    category, tiebreaks = _partial_value(cards, row)
    ranks = list(tiebreaks[:5]) + [0] * (5 - len(tiebreaks[:5]))
    suits: dict[str, int] = {}
    for card in cards:
        suit = card_suit(card)
        suits[suit] = suits.get(suit, 0) + 1
    return [
        category / 8.0,
        *[rank / MAX_RANK for rank in ranks],
        (ROW_CAPACITY[row] - len(cards)) / 5.0,
        (max(suits.values()) if suits else 0) / 5.0,
        _straight_window(cards),
    ]


def side_block(board: Board) -> list[float]:
    """31 dims describing one board on its own."""
    features: list[float] = []
    for row in ROWS:
        features.extend(_row_block(getattr(board, row), row))

    top = _partial_value(board.top, "top")
    middle = _partial_value(board.middle, "middle")
    bottom = _partial_value(board.bottom, "bottom")
    locked_foul = float(
        len(board.top) == 3 and len(board.middle) == 5 and top > middle
    ) or float(
        len(board.middle) == 5 and len(board.bottom) == 5 and middle > bottom
    )
    fl_possible = float(
        len(board.top) < 3
        or (evaluate_3_card(board.top)[0] >= 1 and _top_pair_rank(board.top) >= 12)
    )
    features.extend(
        [
            1.0 if top > middle else (0.0 if top < middle else 0.5),
            1.0 if middle > bottom else (0.0 if middle < bottom else 0.5),
            locked_foul,
            fl_possible,
        ]
    )
    return features


def _top_pair_rank(cards: Sequence[str]) -> int:
    if len(cards) != 3:
        return 0
    category, tiebreaks = evaluate_3_card(cards)
    return tiebreaks[0] if category >= 1 and tiebreaks else 0


def deck_block(unseen):
    """20 dims describing WHAT is left, not just how much.

    The encoder previously carried one deck feature -- the size of the unseen
    set -- and at a fixed street that is a constant, so it said nothing.  What
    the label depends on is composition: with sixteen cards unseen and nine of
    them about to be dealt, whether the ranks an opponent needs for a QQ+ top
    are still alive is a fact about the deck, and no side's outlook histogram
    can recover it.
    """
    rank_counts = [0] * 13
    suit_counts = [0] * 4
    for card in unseen:
        rank_counts[RANK_VALUE[card[0]] - 2] += 1
        suit_counts["hdcs".index(card[1])] += 1
    total = max(len(unseen), 1)
    pair_resource = sum(1 for count in rank_counts if count >= 2)
    broadway = sum(rank_counts[8:])   # T and above
    fl_ranks = sum(rank_counts[10:])  # Q, K, A -- the Fantasyland ranks
    return [
        *[count / 4.0 for count in rank_counts],
        *[count / 13.0 for count in suit_counts],
        pair_resource / 13.0,
        broadway / total,
        fl_ranks / total,
    ]


def _card_helpfulness(terminals, picks, unseen):
    """Per-card value and Fantasyland reach, read off completions already
    enumerated for the outlook block -- no extra hand evaluation."""
    value = {card: [] for card in unseen}
    entry = {card: [] for card in unseen}
    for terminal, pick in zip(terminals, picks):
        score = 0.0 if terminal.busted else 1.0 + terminal.royalty / 10.0
        for card in pick:
            if card in value:
                value[card].append(score)
                entry[card].append(1.0 if terminal.fl_entry else 0.0)
    mean_value = {
        card: (sum(v) / len(v) if v else 0.0) for card, v in value.items()
    }
    mean_entry = {
        card: (sum(v) / len(v) if v else 0.0) for card, v in entry.items()
    }
    return mean_value, mean_entry


def _top_half(scores):
    if not scores:
        return set()
    ordered = sorted(scores, key=lambda card: -scores[card])
    return set(ordered[: max(1, len(ordered) // 2)])


def contest_block(hero_value, hero_entry, opponent_value, opponent_entry, unseen):
    """8 dims: who the remaining cards belong to.

    The head-to-head block summarises OUTCOMES -- row win rates, scoop rates --
    which the two sides' outlooks can largely reconstruct, and ablation put its
    marginal contribution at about 8%.  This is what an outlook cannot express:
    the INTERSECTION of two sides' useful cards.  Nine of the sixteen unseen
    cards will be dealt and each goes to exactly one player, so a card both
    sides want is a different object from a card only one side wants.
    """
    total = max(len(unseen), 1)
    hero_good = _top_half(hero_value)
    opponent_good = _top_half(opponent_value)
    hero_fl = {card for card, rate in hero_entry.items() if rate > 0.0}
    opponent_fl = {card for card, rate in opponent_entry.items() if rate > 0.0}
    contested = hero_good & opponent_good
    return [
        len(contested) / total,
        len(hero_good - opponent_good) / total,
        len(opponent_good - hero_good) / total,
        len(contested) / max(len(hero_good | opponent_good), 1),
        len(hero_fl) / total,
        len(opponent_fl) / total,
        len(hero_fl & opponent_fl) / total,
        (len(hero_fl) - len(opponent_fl)) / total,
    ]


def _completions(board: Board, unseen: Sequence[str]) -> list[Terminal]:
    """Every way this board's open slots fill, strided if there are too many."""
    slots = [row for row in ROWS for _ in range(ROW_CAPACITY[row] - len(getattr(board, row)))]
    if not slots:
        return [board_terminal(board)]
    if len(slots) > 2:
        raise ValueError(
            f"outlook supports at most two open slots, board has {len(slots)}"
        )

    if len(slots) == 1:
        picks: list[tuple[str, ...]] = [(card,) for card in unseen]
    else:
        picks = list(combinations(unseen, 2))
    stride = max(1, len(picks) // MAX_COMPLETIONS)
    terminals: list[Terminal] = []
    for pick in picks[::stride][:MAX_COMPLETIONS]:
        filled = board.place(tuple(zip(pick, slots)))
        terminals.append(board_terminal(filled))
    return terminals


def _completions_with_picks(board: Board, unseen: Sequence[str]):
    """Completions plus which cards filled them, for the contested block."""
    slots = [
        row
        for row in ROWS
        for _ in range(ROW_CAPACITY[row] - len(getattr(board, row)))
    ]
    if not slots:
        return [board_terminal(board)], [()]
    if len(slots) > 2:
        # ``_completions`` has always refused this; this one was added later
        # without the guard, and without it the ``zip`` below silently fills
        # only the first two slots and hands an incomplete board to the
        # evaluator -- which fails several frames away with a message about
        # card counts.  Every side at (T3, BTN) has exactly two open slots, so
        # nothing has hit it; the BB seat, whose BTN opponent still has four,
        # hits it immediately.
        raise ValueError(
            f"outlook supports at most two open slots, board has {len(slots)}"
        )
    if len(slots) == 1:
        picks = [(card,) for card in unseen]
    else:
        picks = list(combinations(unseen, 2))
    stride = max(1, len(picks) // MAX_COMPLETIONS)
    chosen = picks[::stride][:MAX_COMPLETIONS]
    terminals = [
        board_terminal(board.place(tuple(zip(pick, slots)))) for pick in chosen
    ]
    return terminals, [tuple(pick) for pick in chosen]


def outlook_block(terminals: Sequence[Terminal], open_slots: int) -> list[float]:
    """36 dims summarising how a side's completions turn out."""
    total = max(len(terminals), 1)
    histogram = [[0.0] * CATEGORIES for _ in range(3)]
    fouls = 0
    entries = 0
    royalty = 0.0
    for terminal in terminals:
        for row_index, value in enumerate(terminal.values):
            histogram[row_index][min(value[0], CATEGORIES - 1)] += 1.0
        fouls += terminal.busted
        entries += terminal.fl_entry
        royalty += terminal.royalty

    features: list[float] = []
    for row_counts in histogram:
        features.extend(count / total for count in row_counts)
    features.extend(
        [
            open_slots / 5.0,
            open_slots / 5.0,
            open_slots / 5.0,
            fouls / total,
            1.0 if fouls == total else 0.0,
            1.0 - fouls / total,
            min(1.0, royalty / total / 10.0),
            entries / total,
            min(1.0, royalty / total / 30.0),
        ]
    )
    return features


def head_to_head_block(
    hero: Sequence[Terminal], opponent: Sequence[Terminal], fl_ev_14: float
) -> list[float]:
    """10 dims over a strided cross product of the two sides' completions."""
    hero_stride = max(1, len(hero) // MAX_FINISH_PAIRS)
    opponent_stride = max(1, len(opponent) // MAX_FINISH_PAIRS)
    hero_sample = hero[::hero_stride][:MAX_FINISH_PAIRS]
    opponent_sample = opponent[::opponent_stride][:MAX_FINISH_PAIRS]

    pairs = 0
    row_wins = [0.0, 0.0, 0.0]
    scoops = 0
    scooped = 0
    lines_total = 0.0
    hero_fouls = 0
    opponent_fouls = 0
    royalty_diff = 0.0
    fl_diff = 0.0
    for hero_terminal in hero_sample:
        for opponent_terminal in opponent_sample:
            pairs += 1
            hero_fouls += hero_terminal.busted
            opponent_fouls += opponent_terminal.busted
            royalty_diff += hero_terminal.royalty - opponent_terminal.royalty
            fl_diff += float(hero_terminal.fl_entry) - float(opponent_terminal.fl_entry)
            lines = 0
            for row_index, (hero_value, opponent_value) in enumerate(
                zip(hero_terminal.values, opponent_terminal.values)
            ):
                if hero_value > opponent_value:
                    row_wins[row_index] += 1.0
                    lines += 1
                elif hero_value < opponent_value:
                    lines -= 1
            lines_total += lines
            scoops += lines == 3
            scooped += lines == -3

    total = max(pairs, 1)
    return [
        *[wins / total for wins in row_wins],
        scoops / total,
        scooped / total,
        lines_total / total / 3.0,
        hero_fouls / total,
        opponent_fouls / total,
        max(-1.0, min(1.0, royalty_diff / total / 10.0)),
        max(-1.0, min(1.0, fl_diff / total)),
    ]


@dataclass(frozen=True)
class EncodedAction:
    features: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.features) != FEATURE_SIZE:
            raise ValueError(
                f"{FEATURE_SCHEMA} is {FEATURE_SIZE} dims, got {len(self.features)}"
            )


def encode(
    *,
    hero_board: Board,
    opponent_boards: Sequence[Board],
    dealt_cards: Sequence[str],
    hero_private_discards: Sequence[str],
    unseen: Sequence[str],
    fl_ev_14: float = 9.6,
) -> EncodedAction:
    """Encode one post-action 3-max state.

    ``hero_board`` is the board AFTER the candidate action, which is what makes
    this an action encoder: the model scores boards, not moves.
    """
    if len(opponent_boards) != OPPONENTS:
        raise ValueError(f"3-max encoding needs {OPPONENTS} opponent boards")

    boards = [hero_board, *opponent_boards]
    features: list[float] = []
    for board in boards:
        features.extend(side_block(board))

    completions = []
    picks = []
    for board in boards:
        terminals, used = _completions_with_picks(board, unseen)
        completions.append(terminals)
        picks.append(used)
    for board, terminals in zip(boards, completions):
        open_slots = 13 - board.card_count()
        features.extend(outlook_block(terminals, open_slots))

    for index in range(OPPONENTS):
        features.extend(
            head_to_head_block(completions[0], completions[index + 1], fl_ev_14)
        )

    ranks = [card_rank(card) for card in dealt_cards] or [0]
    features.extend(
        [
            max(ranks) / MAX_RANK,
            min(ranks) / MAX_RANK,
            len(unseen) / 52.0,
            len(hero_private_discards) / 4.0,
        ]
    )

    features.extend(deck_block(unseen))
    helpfulness = [
        _card_helpfulness(completions[index], picks[index], unseen)
        for index in range(SIDES)
    ]
    hero_value, hero_entry = helpfulness[0]
    for index in range(OPPONENTS):
        opponent_value, opponent_entry = helpfulness[index + 1]
        features.extend(
            contest_block(
                hero_value, hero_entry, opponent_value, opponent_entry, unseen
            )
        )
    return EncodedAction(tuple(features))


def encode_record_action(record: dict, action_index: int) -> EncodedAction:
    """Encode one action of a teacher JSONL record."""
    from ..cards import ALL_CARDS

    def board_of(payload: dict) -> Board:
        return Board.from_rows(payload["top"], payload["middle"], payload["bottom"])

    opponents = [board_of(payload) for payload in record["opponent_boards"]]
    action = record["actions"][action_index]
    hero_after = board_of(action["board"])

    seen = set(hero_after.all_cards())
    seen.update(record["hero_private_discards"])
    seen.update(action["discards"])
    for board in opponents:
        seen.update(board.all_cards())
    unseen = [card for card in ALL_CARDS if card not in seen]

    return encode(
        hero_board=hero_after,
        opponent_boards=opponents,
        dealt_cards=record["dealt"],
        hero_private_discards=record["hero_private_discards"],
        unseen=unseen,
    )
