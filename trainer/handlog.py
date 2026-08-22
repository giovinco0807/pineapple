"""Hand-history records: validation, replay, and batch grading.

The trainer's other two entry points grade a decision the moment it is made --
`game.py` while the user plays, `app.py`'s editor for a position typed in by
hand.  This module grades decisions that already happened: a hand transcribed
from a video or a live session, both seats, all five streets, replayed through
the same `evaluator.evaluate_position` ranking the trainer uses.

A record is deliberately the whole hand rather than a position.  Every
reconstruction below -- who has seen what, which discards are dead, whether the
opponent has replied yet -- is derivable from the deal order once the hero's
seat is known, and asking the user to re-derive it per street is exactly the
transcription error this module exists to avoid.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence

from trainer.game import candidate_key, rank_score, sorted_rows_key

logger = logging.getLogger("trainer.handlog")

ROWS = ("top", "middle", "bottom")
ROW_CAP = {"top": 3, "middle": 5, "bottom": 5}
STREET_DEAL = {0: 5, 1: 3, 2: 3, 3: 3, 4: 3}
STREET_PLACE = {0: 5, 1: 2, 2: 2, 3: 2, 4: 2}
RANKS = "23456789TJQKA"
SUITS = "shdc"

# Fantasyland: fourteen cards arrive at once and thirteen are set in a single
# action, so a seat in Fantasyland has no streets at all.  Fixed at 14 by this
# ruleset (`ScoringContext.fantasyland_cards`, which refuses any other value).
FL_CARDS = 14
FL_PLACE = 13

# One engine call at a time.  The Rust engine is loaded once per process behind
# its own lock, and T0 already costs 5-20 s; running two in parallel makes both
# slower and the progress bar meaningless.
_executor = ThreadPoolExecutor(max_workers=1)


# ----------------------------------------------------------------------
# validation
# ----------------------------------------------------------------------


def _check_card(card: Any, where: str) -> str:
    if not isinstance(card, str) or len(card) != 2 or card[0] not in RANKS or card[1] not in SUITS:
        raise ValueError(f"{where}: カード表記が不正です（{card!r}）。例: Ah, Td, 7s")
    return card


def validate_hand(hand: Dict[str, Any]) -> Dict[str, Any]:
    """Return a normalized record, or raise ValueError with a Japanese reason.

    The two seats are recorded asymmetrically because they are *seen*
    asymmetrically.  The hero knows all three cards they were dealt, so their
    street is dealt/placements/discard.  Of the opponent only the placements
    are visible -- their discard goes face down in regular OFC -- so their
    `dealt` is derived from what they placed, and their `discard` is optional,
    filled in afterwards only if the hand was later revealed.

    A side whose discard is unknown therefore normalizes to a short `dealt`
    (2 cards on T1-T4).  That is the flag every downstream reader uses: the
    opponent's own decision cannot be graded without knowing the third card,
    while every hero decision is unaffected, since the opponent's discard was
    never part of the hero's information set to begin with.
    """
    position = hand.get("hero_position")
    if position not in ("first", "second"):
        raise ValueError("hero_position は first / second のいずれかです")

    fl_in = hand.get("fl") or {}
    fl = {"hero": bool(fl_in.get("hero")), "opp": bool(fl_in.get("opp"))}
    normal_seats = [seat for seat in ("hero", "opp") if not fl[seat]]

    streets_in = hand.get("streets") or []
    if normal_seats and len(streets_in) != 5:
        raise ValueError(f"ストリートは5つ必要です（T0〜T4）。受け取ったのは {len(streets_in)} 個")
    if not normal_seats and streets_in:
        raise ValueError("両者ファンタジーランドのハンドにストリートはありません")

    seen: Dict[str, str] = {}  # card -> where it first appeared
    rows_by_seat = {"hero": {r: 0 for r in ROWS}, "opp": {r: 0 for r in ROWS}}
    streets: List[Dict[str, Any]] = []

    for turn, street_in in enumerate(streets_in):
        street: Dict[str, Any] = {"turn": turn}
        for seat in ("hero", "opp"):
            side_in = (street_in or {}).get(seat) or {}
            label = f"T{turn} {'自分' if seat == 'hero' else '相手'}"

            # A seat in Fantasyland took its whole hand before the streets
            # started; it has no per-street action to record.
            if fl[seat]:
                if side_in.get("placements") or side_in.get("dealt") or side_in.get("discard"):
                    raise ValueError(f"{label}: ファンタジーランド中の席にストリートの入力はできません")
                street[seat] = {"dealt": [], "placements": [], "discard": None}
                continue

            placements: List[List[str]] = []
            for entry in side_in.get("placements") or []:
                if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                    raise ValueError(f"{label}: 配置は [カード, 行] の形式です")
                card = _check_card(entry[0], f"{label} 配置")
                row = entry[1]
                if row not in ROW_CAP:
                    raise ValueError(f"{label}: 行は top/middle/bottom です（{row!r}）")
                placements.append([card, row])
            if len(placements) != STREET_PLACE[turn]:
                raise ValueError(
                    f"{label}: 配置は {STREET_PLACE[turn]} 枚必要です（{len(placements)} 枚）"
                )

            discard = side_in.get("discard")
            if turn == 0:
                if discard:
                    raise ValueError(f"{label}: T0 に捨て札はありません")
                discard = None
            elif discard:
                discard = _check_card(discard, f"{label} 捨て札")
            elif seat == "hero":
                raise ValueError(f"{label}: 捨て札が未入力です")
            else:
                discard = None

            dealt = [c for c, _ in placements] + ([discard] if discard else [])

            # Records written before the opponent's discard became optional
            # carry an explicit `dealt`; honour it as a cross-check rather than
            # silently preferring the derived list.
            given = side_in.get("dealt")
            if given:
                given = [_check_card(c, f"{label} 配札") for c in given]
                if sorted(given) != sorted(dealt):
                    raise ValueError(f"{label}: 配置＋捨て札が配札と一致しません")

            for card in dealt:
                if card in seen:
                    raise ValueError(f"{card} が重複しています（{seen[card]} と {label}）")
                seen[card] = label

            for _, row in placements:
                rows_by_seat[seat][row] += 1
                if rows_by_seat[seat][row] > ROW_CAP[row]:
                    raise ValueError(f"{label}: {row} が定員 {ROW_CAP[row]} を超えています")

            street[seat] = {"dealt": dealt, "placements": placements, "discard": discard}
        streets.append(street)

    fl_hands_in = hand.get("fl_hands") or {}
    fl_hands: Dict[str, Any] = {}
    for seat in ("hero", "opp"):
        label = f"FL {'自分' if seat == 'hero' else '相手'}"
        side_in = fl_hands_in.get(seat) or {}
        if not fl[seat]:
            if side_in.get("placements"):
                raise ValueError(f"{label}: FLでない席にFLの盤面は入力できません")
            continue

        placements = []
        rows_seen = {r: 0 for r in ROWS}
        for entry in side_in.get("placements") or []:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                raise ValueError(f"{label}: 配置は [カード, 行] の形式です")
            card = _check_card(entry[0], f"{label} 配置")
            row = entry[1]
            if row not in ROW_CAP:
                raise ValueError(f"{label}: 行は top/middle/bottom です（{row!r}）")
            rows_seen[row] += 1
            placements.append([card, row])
        if len(placements) != FL_PLACE:
            raise ValueError(f"{label}: 配置は13枚必要です（{len(placements)} 枚）")
        for row in ROWS:
            if rows_seen[row] != ROW_CAP[row]:
                raise ValueError(
                    f"{label}: {row} は {ROW_CAP[row]} 枚です（{rows_seen[row]} 枚）"
                )

        # Same asymmetry as a normal street, for the same reason: the hero knows
        # all fourteen cards they were dealt, while of the opponent only the
        # thirteen they set are ever visible.
        discard = side_in.get("discard")
        if discard:
            discard = _check_card(discard, f"{label} 捨て札")
        elif seat == "hero":
            raise ValueError(f"{label}: 捨て札（14枚目）が未入力です")
        else:
            discard = None

        dealt = [c for c, _ in placements] + ([discard] if discard else [])
        given = side_in.get("dealt")
        if given:
            given = [_check_card(c, f"{label} 配札") for c in given]
            if sorted(given) != sorted(dealt):
                raise ValueError(f"{label}: 配置＋捨て札が配札と一致しません")

        for card in dealt:
            if card in seen:
                raise ValueError(f"{card} が重複しています（{seen[card]} と {label}）")
            seen[card] = label

        fl_hands[seat] = {"dealt": dealt, "placements": placements, "discard": discard}

    for seat in ("hero", "opp"):
        if fl[seat] and seat not in fl_hands:
            raise ValueError(
                f"{'自分' if seat == 'hero' else '相手'}のファンタジーランドの盤面が未入力です"
            )

    return {
        "label": (hand.get("label") or "").strip(),
        "hero_position": position,
        "fl": fl,
        "streets": streets,
        "fl_hands": fl_hands,
        "note": (hand.get("note") or "").strip(),
    }


# ----------------------------------------------------------------------
# replay
# ----------------------------------------------------------------------


def _empty_board() -> Dict[str, List[str]]:
    return {"top": [], "middle": [], "bottom": []}


def _place(board: Dict[str, List[str]], placements: Sequence[Sequence[str]]) -> Dict[str, List[str]]:
    out = {r: list(board[r]) for r in ROWS}
    for card, row in placements:
        out[row].append(card)
    return out


def _gradable(turn: int, side: Dict[str, Any], discards_so_far: Sequence[str]) -> bool:
    """Can this decision be handed to the engine as a complete information set?

    Two conditions, and the second is the one that bites.  The actor's own three
    cards must be known -- that is the decision itself.  But the engine also
    checks the *count* of the actor's earlier discards against the street
    (`ActorObservation` rejects "T3/second expected ... 2, got 0"), so one
    missing discard makes every later street of that seat ungradable too, not
    just its own.  Left unchecked the engine refuses those streets and the
    evaluator quietly answers with the Monte-Carlo fallback instead, whose
    scores are on a different scale from the ones beside them in the table.
    """
    if len(side["dealt"]) != STREET_DEAL[turn]:
        return False
    return len(discards_so_far) == max(0, turn - 1)


def decision_contexts(
    hand: Dict[str, Any],
    include_opp: bool = False,
    turns: Optional[Sequence[int]] = None,
) -> List[Dict[str, Any]]:
    """Rebuild every decision in deal order, each with the state that faced it.

    The only subtlety is the opponent's board.  Within a street the first seat
    acts blind and the second seat acts having seen the reply, so the second
    seat's context includes the first seat's placements from that same street
    while the first seat's does not.

    An opponent street whose discard was never recorded is skipped rather than
    guessed: the decision it faced was a choice among three cards, and two of
    them do not describe it.  Hero streets are never affected -- the opponent's
    discard is not in the hero's information set.

    ``turns`` keeps only the streets named, and is how one spot gets the
    precision a whole hand cannot afford: a hand is ten decisions, and at the
    deep rung the opening alone is minutes.  It filters what is GRADED, never
    what is replayed -- every street is still walked in order so the boards and
    dead cards facing the kept turn are the ones that actually faced it.  A
    Fantasyland seat's single action is not a street and has no turn number;
    naming any turn drops it, since it is not the spot that was asked for.
    """
    hero_pos = hand["hero_position"]
    fl = hand.get("fl") or {"hero": False, "opp": False}
    boards = {"hero": _empty_board(), "opp": _empty_board()}
    discards: Dict[str, List[str]] = {"hero": [], "opp": []}
    order = ("hero", "opp") if hero_pos == "first" else ("opp", "hero")
    seat_position = {order[0]: "first", order[1]: "second"}
    wanted = None if turns is None else {int(t) for t in turns}

    contexts: List[Dict[str, Any]] = []

    # A Fantasyland seat's whole hand is one action, taken before any street.
    for seat in ("hero", "opp"):
        if not fl.get(seat):
            continue
        if seat != "hero" and not include_opp:
            continue
        if wanted is not None:
            continue
        side = (hand.get("fl_hands") or {}).get(seat)
        if not side:
            continue
        contexts.append(
            {
                "seat": seat,
                "kind": "fl",
                "turn": None,
                "position": seat_position[seat],
                "hero_board": _empty_board(),
                "opp_board": _empty_board(),
                "dealt": list(side["dealt"]),
                "dead": [],
                "action": {
                    "placements": [list(p) for p in side["placements"]],
                    "discard": side["discard"],
                },
            }
        )

    for street in hand["streets"]:
        turn = street["turn"]
        for seat in order:
            side = street[seat]
            other = "opp" if seat == "hero" else "hero"
            if fl.get(seat):
                continue
            # Streets played against a Fantasyland opponent are mostly not
            # gradable.  The m3 search is fed by a hidden-card belief sampler
            # whose geometry table covers only a normal opponent (`hu_belief.
            # _validate_decision_observation`), and it refuses outright:
            # "expected hero/opponent/hero-discards/opponent-discards=(7,7,1,1),
            # got (7,0,1,0)".  There is no coarser answer to fall back to --
            # the Monte-Carlo path deals the opponent a normal five-street hand,
            # which is a different opponent, not a rougher one.
            #
            # T0 acting first is the exception, and only there: both geometries
            # are (0,0,5,0) at that one node, so the request goes through and is
            # answered by the ordinary T0-first model.  That model does not know
            # the opponent is in Fantasyland -- measured and accepted as the
            # known approximation, `configs/fl_ev_regular_v4_selfplay.json`
            # provenance/m11_vsfl_models/t0_vs_fl -- so the decision is graded
            # and flagged rather than dropped.  The opening is the biggest
            # decision of the hand; an approximate grade beats none, labelled.
            vs_fl_approx = False
            if fl.get(other):
                if not (turn == 0 and seat_position[seat] == "first"):
                    continue
                vs_fl_approx = True
            keep_turn = wanted is None or turn in wanted
            if (
                keep_turn
                and (seat == "hero" or include_opp)
                and _gradable(turn, side, discards[seat])
            ):
                contexts.append(
                    {
                        "seat": seat,
                        "kind": "street",
                        "turn": turn,
                        "position": seat_position[seat],
                        # `hero_*` here is the acting seat, matching evaluate_position.
                        "hero_board": {r: list(boards[seat][r]) for r in ROWS},
                        # Nothing opposite when they are in Fantasyland: not a
                        # board we happen not to know, a board that does not
                        # exist face up.  The engine has a second geometry for
                        # exactly this and demands zero cards there.
                        "opp_board": (
                            _empty_board()
                            if fl.get(other)
                            else {r: list(boards[other][r]) for r in ROWS}
                        ),
                        "opp_in_fl": bool(fl.get(other)),
                        "vs_fl_approx": vs_fl_approx,
                        "dealt": list(side["dealt"]),
                        # Only the actor's own discards are known-dead; the
                        # opponent's go face down in regular OFC.
                        "dead": list(discards[seat]),
                        "action": {
                            "placements": [list(p) for p in side["placements"]],
                            "discard": side["discard"],
                        },
                    }
                )
            boards[seat] = _place(boards[seat], side["placements"])
            if side["discard"]:
                discards[seat].append(side["discard"])

    return contexts


def missing_opp_discards(hand: Dict[str, Any]) -> List[int]:
    """Turns whose opponent discard is unrecorded."""
    return [
        street["turn"]
        for street in hand["streets"]
        if street["turn"] > 0 and not street["opp"]["discard"]
    ]


def ungradable_turns(hand: Dict[str, Any], seat: str = "opp") -> List[int]:
    """Turns of `seat` that cannot be graded, including the knock-on ones.

    A missing discard costs its own street *and every later street of that
    seat*, because the engine checks the running discard count -- see
    `_gradable`.  The UI quotes this list, so it has to be the real one rather
    than just the streets whose own card is missing.
    """
    discards: List[str] = []
    out: List[int] = []
    for street in hand["streets"]:
        side = street[seat]
        if not _gradable(street["turn"], side, discards):
            out.append(street["turn"])
        if side["discard"]:
            discards.append(side["discard"])
    return out


def final_boards(hand: Dict[str, Any]) -> Dict[str, Dict[str, List[str]]]:
    fl = hand.get("fl") or {}
    fl_hands = hand.get("fl_hands") or {}
    boards = {"hero": _empty_board(), "opp": _empty_board()}
    for seat in ("hero", "opp"):
        if fl.get(seat) and fl_hands.get(seat):
            boards[seat] = _place(_empty_board(), fl_hands[seat]["placements"])
    for street in hand["streets"]:
        for seat in ("hero", "opp"):
            if fl.get(seat):
                continue
            boards[seat] = _place(boards[seat], street[seat]["placements"])
    return boards


def _fl_stay(board: Dict[str, List[str]]) -> bool:
    """Does this completed board keep a Fantasyland seat in Fantasyland?

    Re-entry is a different test from entry: entry needs QQ+ up top, staying
    needs trips up top or quads-or-better at the bottom.  A seat already in
    Fantasyland is therefore scored against the stay rule, and using the entry
    rule for it -- as the generic terminal scorer does -- would hand the FL
    bonus to any board with a queen pair, which is not what wins another one.
    """
    try:
        from ofc_regular.evaluator import evaluate_3_card, evaluate_5_card
        from ofc_regular.rules import check_fl_stay
    except Exception:  # pragma: no cover
        return False
    top = evaluate_3_card(tuple(board["top"]))
    middle = evaluate_5_card(tuple(board["middle"]))
    bottom = evaluate_5_card(tuple(board["bottom"]))
    if top > middle or middle > bottom:  # fouled boards keep nothing
        return False
    return bool(check_fl_stay(tuple(board["top"]), tuple(board["bottom"])).qualifies)


def hand_score(hand: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Actual heads-up result of the transcribed hand, from the hero's side.

    Uses the trainer evaluator's own terminal scorer so the number here is on
    the same scale as the EVs the candidate rankings are quoted in -- FL entry
    included as its fixed-point EV rather than as a separate flag.

    A seat that was ALREADY in Fantasyland is the exception: its next-hand
    Fantasyland is a stay, not an entry, so the flag is recomputed under the
    stay rule and the bonus applied by hand.
    """
    try:
        from trainer.evaluator import _board_terminal, _score_pair
        from trainer.fl_ev import FL_EV_14
    except Exception:  # pragma: no cover - evaluator import failure
        return None

    fl = hand.get("fl") or {}
    boards = final_boards(hand)
    terminal = {}
    for seat in ("hero", "opp"):
        b = boards[seat]
        if len(b["top"]) != 3 or len(b["middle"]) != 5 or len(b["bottom"]) != 5:
            return None
        terminal[seat] = _board_terminal(
            tuple(b["top"]), tuple(b["middle"]), tuple(b["bottom"])
        )

    hero, opp = terminal["hero"], terminal["opp"]
    score = _score_pair(hero, opp)

    # `_score_pair` has already applied the entry rule to both seats; for a seat
    # that was in Fantasyland, back that out and apply the stay rule instead.
    next_fl = {"hero": hero[3], "opp": opp[3]}
    for seat, sign in (("hero", 1.0), ("opp", -1.0)):
        if not fl.get(seat):
            continue
        entry_flag = terminal[seat][3]
        stay_flag = _fl_stay(boards[seat])
        next_fl[seat] = stay_flag
        if entry_flag != stay_flag:
            score += sign * FL_EV_14 * (1.0 if stay_flag else -1.0)

    return {
        "score": score,
        "hero_bust": hero[0],
        "opp_bust": opp[0],
        "hero_royalty": hero[2],
        "opp_royalty": opp[2],
        "hero_fl": next_fl["hero"],
        "opp_fl": next_fl["opp"],
        "hero_was_fl": bool(fl.get("hero")),
        "opp_was_fl": bool(fl.get("opp")),
        "hero_board": boards["hero"],
        "opp_board": boards["opp"],
    }


# ----------------------------------------------------------------------
# grading
# ----------------------------------------------------------------------


def grade_fl_decision(ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Grade a Fantasyland setting by exhaustively ranking the alternatives.

    Not the engine: the engine's action space is "place two of three onto a
    board", which cannot describe setting thirteen of fourteen at once, and it
    says so (`hero_in_fantasyland requires the FL observation schema`).  What
    ranks these is `fl_grade`, which enumerates the ~350k legal settings exactly
    and scores each on royalty plus the value of staying.

    That objective has no opponent in it, which is right for the decision --
    Fantasyland is set face down before anything opposite is visible -- but it
    also means the number here is not on the same scale as a street's EV loss.
    It is royalty points, plus one Fantasyland's worth for a setting that stays.
    """
    from trainer import fl_grade
    from trainer.fl_ev import FL_EV_14

    started = time.time()
    played_board = _place(_empty_board(), ctx["action"]["placements"])
    dealt = list(ctx["dealt"])

    if len(dealt) == FL_CARDS:
        result = fl_grade.rank_settings(
            dealt, stay_bonus=FL_EV_14, played=played_board, top_n=20
        )
        question = "14枚から13枚"
    else:
        # Only the thirteen they set are known; their real 14-choose-13 decision
        # is unrecoverable, so answer the narrower question and label it.
        result = fl_grade.rank_arrangements(played_board, stay_bonus=FL_EV_14, top_n=20)
        question = "既知の13枚の並べ方のみ"

    def _as_candidate(entry: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "action": {
                "placements": [[c, r] for r in ROWS for c in entry[r]],
                "discard": entry.get("discard"),
            },
            "board": {r: list(entry[r]) for r in ROWS},
            "metrics": {
                "ev": entry["score"],
                "rank_score": entry["score"],
                "ranked_by": "royalty+stay",
                "royalty": entry["royalty"],
                "stay": entry["stay"],
            },
            "key": sorted_rows_key({r: list(entry[r]) for r in ROWS}),
        }

    candidates = [_as_candidate(e) for e in result["candidates"]]
    played = _as_candidate(result["played"]) if result.get("played") else None
    best = _as_candidate(result["best"]) if result.get("best") else None

    # A fouled setting is absent from the ranking because the ranking is of
    # legal settings.  It is not "not found": it is worth nothing -- no royalty,
    # no stay, and the hand pays the foul on top -- so it is priced at zero
    # against the best rather than left blank.
    fouled = False
    rank = result.get("rank")
    ev_loss = result.get("ev_loss")
    if played is None and fl_grade.is_foul(played_board):
        fouled = True
        if best is not None:
            ev_loss = max(0.0, best["metrics"]["rank_score"])

    return {
        "seat": ctx["seat"],
        "kind": "fl",
        "turn": None,
        "position": ctx["position"],
        "hero_board": ctx["hero_board"],
        "opp_board": ctx["opp_board"],
        "dealt": dealt,
        "dead": [],
        "action": ctx["action"],
        "rank": rank,
        "ev_loss": ev_loss,
        "is_best": rank == 1,
        "tied_with_best": bool(rank and rank != 1 and (ev_loss or 0) <= 1e-9),
        "fouled": fouled,
        "ranked_by": "royalty+stay",
        "played": played,
        "best": best,
        "candidates": candidates,
        "candidate_count": result["legal_count"],
        "evaluator": f"fl_exact({question})",
        "elapsed": time.time() - started,
    }


# Streets where a small model penalty is re-checked with the search before it
# is shown, and the threshold under which that happens.
#
# Measured at T3 on 166 roots (`docs/t3_tie_misses_20260815.md`): the model
# charges a penalty on pairs the search puts together at 5.2 % of first-seat
# and 21.3 % of second-seat roots, and every one of those claims was under
# 0.30 (max 0.118 first, 0.272 second).  So 0.30 catches all of them, and
# fires on 41-46 % of T3 decisions -- about two seconds each at the `deep`
# rung, which is the whole added cost.
#
# T3 only, for now, and deliberately:
#   * T0 is minutes a decision, so confirming there would cost more than the
#     grade is worth;
#   * T1/T2 are 2-22 s and their tie-miss rate has not been measured, so there
#     is no threshold to justify -- add them here once it has been;
#   * T4 is decided by exact enumeration and has nothing to confirm.
CONFIRM_STREETS = {3}
CONFIRM_BELOW = 0.30
CONFIRM_PRECISION = "deep"


def grade_decision(
    ctx: Dict[str, Any],
    precision: str = "fast",
    evaluate: Optional[Callable[..., Dict[str, Any]]] = None,
    method: str = "model",
    confirm: bool = True,
) -> Dict[str, Any]:
    """Rank every legal action from `ctx` and locate the one actually played.

    When the model charges a penalty too small for it to actually resolve, the
    decision is re-graded with the search and the search's verdict is what is
    reported -- the model's claim is kept beside it rather than discarded.  See
    CONFIRM_STREETS above for where and why.
    """
    if ctx.get("kind") == "fl":
        return grade_fl_decision(ctx)

    if evaluate is None:
        from trainer import evaluator as trainer_evaluator

        evaluate = trainer_evaluator.evaluate_position

    started = time.time()
    result = evaluate(
        hero_board=ctx["hero_board"],
        opp_board=ctx["opp_board"],
        dealt=ctx["dealt"],
        dead=ctx["dead"],
        turn=ctx["turn"],
        position=ctx["position"],
        precision=precision,
        opp_in_fl=bool(ctx.get("opp_in_fl")),
        method=method,
    )
    candidates = result.get("candidates") or []

    played_board = _place(ctx["hero_board"], ctx["action"]["placements"])
    played_key = sorted_rows_key(played_board)

    rank = None
    played = None
    for i, cand in enumerate(candidates):
        if candidate_key(cand) == played_key:
            rank = i + 1
            played = cand
            break

    ev_loss = None
    is_best = False
    tied = False
    if played is not None and candidates:
        ev_loss = max(0.0, rank_score(candidates[0]) - rank_score(played))
        is_best = rank == 1
        # A ranked-but-zero-loss move is not the same thing as the best move:
        # at low sample counts the engine's scores are coarse enough that a
        # dozen actions can share the top value.  Calling those "best" would
        # read as approval of a move the engine merely could not separate.
        tied = not is_best and ev_loss <= 1e-9

    # The model said something small.  Ask the instrument that can tell a tie
    # from a near-tie, and report what it says.
    confirmation = None
    if (
        confirm
        and method == "model"
        and ctx["turn"] in CONFIRM_STREETS
        and ev_loss is not None
        and 0.0 < ev_loss < CONFIRM_BELOW
    ):
        checked = grade_decision(
            ctx,
            precision=CONFIRM_PRECISION,
            evaluate=evaluate,
            method="teacher",
            confirm=False,
        )
        if checked.get("ev_loss") is not None:
            confirmation = {
                "model_ev_loss": ev_loss,
                "model_rank": rank,
                "confirmed_by": "teacher",
                "confirm_precision": CONFIRM_PRECISION,
                # A penalty the model charged and the search did not.  Named
                # rather than inferred, because "the number got smaller" and
                # "there was no difference" read very differently to a user.
                "overturned": checked["ev_loss"] <= 1e-9,
            }
            checked.update(confirmation)
            checked["elapsed"] = time.time() - started
            return checked

    return {
        "seat": ctx["seat"],
        "kind": "street",
        "turn": ctx["turn"],
        "position": ctx["position"],
        "hero_board": ctx["hero_board"],
        "opp_board": ctx["opp_board"],
        "dealt": ctx["dealt"],
        "dead": ctx["dead"],
        "action": ctx["action"],
        "rank": rank,
        "ev_loss": ev_loss,
        "is_best": is_best,
        "tied_with_best": tied,
        "vs_fl_approx": bool(ctx.get("vs_fl_approx")),
        "ranked_by": ((candidates[0].get("metrics") or {}).get("ranked_by") if candidates else None),
        "played": played,
        "best": candidates[0] if candidates else None,
        "candidates": candidates[:10],
        "candidate_count": len(candidates),
        "evaluator": result.get("evaluator", ""),
        "elapsed": time.time() - started,
    }


def analyze_hand(
    hand: Dict[str, Any],
    precision: str = "fast",
    include_opp: bool = False,
    on_decision: Optional[Callable[[Dict[str, Any]], None]] = None,
    method: str = "model",
    turns: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    contexts = decision_contexts(hand, include_opp=include_opp, turns=turns)
    decisions: List[Dict[str, Any]] = []
    for ctx in contexts:
        try:
            graded = grade_decision(ctx, precision=precision, method=method)
        except Exception as exc:
            logger.exception("grading failed at T%s/%s", ctx["turn"], ctx["seat"])
            graded = {
                "seat": ctx["seat"],
                "turn": ctx["turn"],
                "position": ctx["position"],
                "action": ctx["action"],
                "dealt": ctx["dealt"],
                "hero_board": ctx["hero_board"],
                "opp_board": ctx["opp_board"],
                "error": str(exc),
                "rank": None,
                "ev_loss": None,
                "is_best": False,
                "candidates": [],
                "candidate_count": 0,
            }
        # Stamped on the row rather than only on the analysis, because a
        # targeted run merges into a hand graded at some other rung and the
        # merged list would otherwise claim one precision for all of it.
        graded.setdefault("precision", precision)
        graded.setdefault("method", method)
        decisions.append(graded)
        if on_decision:
            on_decision(graded)

    hero = [d for d in decisions if d["seat"] == "hero"]
    losses = [d["ev_loss"] for d in hero if d.get("ev_loss") is not None]
    return {
        "precision": precision,
        "method": method,
        "include_opp": include_opp,
        # Present only on a targeted run, so a stored analysis says on its face
        # that it covers one spot rather than the hand.  A reader that finds
        # this key must not treat the totals below as the hand's.
        **({"turns": sorted({int(t) for t in turns})} if turns is not None else {}),
        "decisions": decisions,
        "hero_total_ev_loss": sum(losses) if losses else 0.0,
        "hero_graded": len(losses),
        "hero_best": sum(1 for d in hero if d.get("is_best")),
        "hero_tied": sum(1 for d in hero if d.get("tied_with_best")),
        # Graded, but by a model that does not know the opponent is in
        # Fantasyland.  Named so the UI can say which rows carry the caveat.
        "vs_fl_approx_decisions": [
            {"seat": d["seat"], "turn": d["turn"]} for d in decisions if d.get("vs_fl_approx")
        ],
        # Only meaningful when include_opp was asked for; harmless otherwise.
        # Only about missing discards; when a seat is in Fantasyland the other
        # seat's streets are ungradable for a different reason, reported below.
        "opp_skipped_turns": (
            ungradable_turns(hand, "opp")
            if include_opp and not any((hand.get("fl") or {}).values())
            else []
        ),
        "fl": dict(hand.get("fl") or {"hero": False, "opp": False}),
        # Exactly one seat in Fantasyland: the other one played five streets
        # that no evaluator here will grade.
        "vs_fl_ungraded": bool(
            (hand.get("fl") or {}).get("hero") != (hand.get("fl") or {}).get("opp")
        ),
        # A decision the engine refused was answered by the Monte-Carlo
        # fallback, whose scores do not share a scale with the rows around it.
        # Identified by what actually answered rather than by what did not: the
        # Fantasyland grader is exact and deterministic and is not a fallback,
        # but its name is not "rust:" either.
        "fallback_decisions": [
            {"seat": d["seat"], "turn": d["turn"], "evaluator": d.get("evaluator", "")}
            for d in decisions
            if str(d.get("evaluator") or "").startswith("mc-")
        ],
        # A fouled Fantasyland setting has no rank because it is not a legal
        # setting, which is a different thing from the engine not listing it.
        "hero_unmatched": sum(
            1
            for d in hero
            if d.get("rank") is None and not d.get("error") and not d.get("fouled")
        ),
        "result": hand_score(hand),
    }


def _decision_slot(decision: Dict[str, Any]) -> tuple:
    """What makes two gradings the same decision: the seat and the street.

    ``turn`` is ``None`` for a Fantasyland seat's single action, which is a
    legitimate slot and distinct from every street.
    """
    return (decision.get("seat"), decision.get("turn"))


def merge_analysis(base: Optional[Dict[str, Any]], update: Dict[str, Any]) -> Dict[str, Any]:
    """Fold a targeted analysis into whatever the hand already carried.

    A targeted run grades one street.  Storing it whole would replace the
    hand's analysis with a single decision -- and since the hand list reads
    ``hero_total_ev_loss`` from it, the list would then show one street's loss
    as the hand's.  So the new decisions replace their own slots and nothing
    else, and the totals are recomputed over the merged list.

    The result deliberately does NOT claim a single precision.  Whatever rung
    each decision was graded at travels on the decision; the top level keeps
    the newest run's rung for display and sets ``mixed_precision`` when the
    rows disagree, so a reader is never told the opening got the deep search
    because the turn-two spot did.
    """
    if not base or not base.get("decisions"):
        return update

    merged = {**base}
    replacing = {_decision_slot(d): d for d in update.get("decisions", [])}
    decisions = [replacing.pop(_decision_slot(d), d) for d in base["decisions"]]
    # Slots the base never had -- an opponent street that became gradable once
    # its discard was filled in, say.  Appended in deal order rather than at
    # the end, so the list still reads as the hand was played.
    decisions.extend(replacing.values())
    decisions.sort(key=lambda d: (-1 if d.get("turn") is None else d["turn"],))

    hero = [d for d in decisions if d.get("seat") == "hero"]
    losses = [d["ev_loss"] for d in hero if d.get("ev_loss") is not None]
    rungs = {d.get("precision") for d in decisions if d.get("precision")}
    methods = {d.get("method") for d in decisions if d.get("method")}
    merged.update(
        {
            "decisions": decisions,
            "precision": update.get("precision", base.get("precision")),
            "method": update.get("method", base.get("method")),
            "mixed_precision": len(rungs) > 1 or len(methods) > 1,
            "precisions": sorted(r for r in rungs if r),
            "hero_total_ev_loss": sum(losses) if losses else 0.0,
            "hero_graded": len(losses),
            "hero_best": sum(1 for d in hero if d.get("is_best")),
            "hero_tied": sum(1 for d in hero if d.get("tied_with_best")),
        }
    )
    # A merged analysis covers the hand again, so the targeted marker from the
    # update must not survive into it.
    merged.pop("turns", None)
    return merged


# ----------------------------------------------------------------------
# background jobs
# ----------------------------------------------------------------------


class JobManager:
    """Runs analyses off the request thread so 15 hands can stream progress."""

    def __init__(self, max_jobs: int = 20):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._max_jobs = max_jobs

    def submit(
        self,
        items: List[Dict[str, Any]],
        precision: str,
        include_opp: bool,
        method: str = "model",
        on_hand_done: Optional[Callable[[Optional[int], Dict[str, Any], Dict[str, Any]], None]] = None,
        turns: Optional[Sequence[int]] = None,
    ) -> str:
        job_id = uuid.uuid4().hex[:12]
        # Count the decisions for real rather than assuming 5 or 10: an
        # opponent street with no recorded discard is not graded, and a
        # progress bar that counts it never reaches its own total.  Counted
        # under the same filter the run uses, or a targeted job's bar would
        # stop at a tenth.
        total = sum(
            len(decision_contexts(item["hand"], include_opp=include_opp, turns=turns))
            for item in items
        )
        job = {
            "id": job_id,
            "status": "running",
            "total": total,
            "done": 0,
            "hands_total": len(items),
            "hands_done": 0,
            "current": None,
            "results": [],
            "error": None,
        }
        with self._lock:
            self._jobs[job_id] = job
            # Evict finished jobs only.  The executor is single-worker, so with a
            # backlog the oldest *inserted* job is exactly the one still on the
            # worker -- dropping it 404s the poll of a run that is still going.
            for stale in [
                key
                for key, value in self._jobs.items()
                if value["status"] != "running" and key != job_id
            ]:
                if len(self._jobs) <= self._max_jobs:
                    break
                self._jobs.pop(stale, None)
        job["method"] = method
        job["turns"] = None if turns is None else sorted({int(t) for t in turns})
        _executor.submit(
            self._run, job, items, precision, include_opp, on_hand_done, method, turns
        )
        return job_id

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def _run(
        self, job, items, precision, include_opp, on_hand_done, method="model", turns=None
    ) -> None:
        try:
            for item in items:
                hand = item["hand"]
                job["current"] = {
                    "hand_id": item.get("id"),
                    "label": hand.get("label") or "",
                }

                def _tick(_graded, _job=job):
                    _job["done"] += 1

                analysis = analyze_hand(
                    hand,
                    precision=precision,
                    include_opp=include_opp,
                    on_decision=_tick,
                    method=method,
                    turns=turns,
                )
                job["hands_done"] += 1
                job["results"].append(
                    {"hand_id": item.get("id"), "label": hand.get("label") or "", "analysis": analysis}
                )
                if on_hand_done:
                    try:
                        on_hand_done(item.get("id"), hand, analysis)
                    except Exception:
                        logger.exception("analysis persistence failed")
            job["status"] = "done"
        except Exception as exc:
            logger.exception("analysis job failed")
            job["status"] = "error"
            job["error"] = str(exc)
        finally:
            job["current"] = None


jobs = JobManager()
