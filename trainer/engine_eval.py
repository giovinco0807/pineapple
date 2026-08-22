"""Rust m3 engine adapter: full candidate rankings for T1-T4.

Mirrors the production webapp runtime call pattern (JointExactConfig with the
learned weights, T4 exact enumeration) but keeps the ENTIRE ranked action
list instead of the webapp's top-3 audit rows.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from ofc_regular.cards import ALL_CARDS  # noqa: E402
from ofc_regular.hu_infoset import ActorObservation  # noqa: E402
from ofc_regular.hu_late_street_teacher import T4SearchConfig  # noqa: E402
from ofc_regular.hu_m3_rust import (  # noqa: E402
    HU_M3_REQUEST_SCHEMA,
    _joint_config_payload,
    evaluate_request,
    load_native_engine,
    t4_request,
    t0_request,
    t1_request,
    t2_request,
    t3_request,
)
from ofc_regular.hu_turn3_joint_exact_teacher import JointExactConfig  # noqa: E402
from ofc_regular.state import Board  # noqa: E402

logger = logging.getLogger("trainer.engine")

LIBRARY_PATH = _ROOT / "target" / "release" / "ofc_hu_m3_engine.dll"
WEIGHTS_DIR = _ROOT / "rust" / "hu_m3_engine" / "tests" / "fixtures"
# The full m7v5 pin set, strongest-first.  The engine refuses a street whose
# downstream evaluators are not all pinned -- T1 first needs the opponent's T1
# second reply, T0 needs every one of the seven below it -- so this table is
# what decides which streets the engine will serve at all.
#
# T4/T3/T2 are the 9.6-relabelled generation (v6 / v3 / v2 / v2 / v2); T1 and
# below are still labelled at the June constant, which is the M7 cascade's
# remaining work. The T2 pair landed 2026-08-08 on 25,000 positions at 2,048
# particles and passed its gates at +0.0171 (first) and +0.0295 (second) per
# hand over 20,004 mirrored deals each.
#
# The distilled `fast_*` images are deliberately NOT pinned.  They are a speed
# trade: the engine reaches its T0-T2 replies through them and reports those
# evaluators as "learned_fast".  A trainer grades rather than races, so the
# full-precision replies are the right side of that trade.
WEIGHT_FILES = {
    "t4": "t4_model_v6.bin",
    "t3_second": "t3_model_v3.bin",
    "t3_first": "t3first_model_v2.bin",
    # v3x16 (2026-08-22): v2 retrained with 3,600 on-policy mined positions and
    # the 121 confirmed misses oversampled x16; promoted on a two-seed-set gate,
    # pooled 40,008 deals, +0.0403 [+0.0103, +0.0704] vs v2.
    "t2_second": "t2_model_v3x16.bin",
    "t2_first": "t2first_model_v2.bin",
    "t1_second": "t1_model_v1.bin",
    "t1_first": "t1first_model_v1.bin",
    "t0_second": "t0_model_v1.bin",
    "t0_first": "t0first_model_v1.bin",
    # Distilled replies.  Pinned only because T0 is otherwise unusable: with
    # full-precision replies a T0-first ranking measured 292 s here, and the
    # opening is the one decision a trainer must answer while the user is still
    # looking at the screen.  The package notes name the T0 second-seat reply as
    # the one worth coarsening most -- 232 candidate boards where a turn reply
    # is twenty-seven.  T3/T4 have no distilled image and stay full-precision.
    "fast_t0_second": "fast_t0_second_v1.bin",
    "fast_t1_second": "fast_t1_second_v1.bin",
    "fast_t1_first": "fast_t1_first_v1.bin",
    "fast_t2_second": "fast_t2_second_v1.bin",
    "fast_t2_first": "fast_t2_first_v1.bin",
}

# (candidate_samples, evaluation_samples, downstream_t3_samples) per precision.
# The production webapp runs 1/1/1 for interactive latency.
# T0 is the deepest tree (232 candidate boards acting first, every street still
# ahead), so it stays at the cheapest rung on every precision -- "high" at T0
# would cost minutes per decision, not seconds.
SAMPLES = {
    "fast": {0: (1, 1, 1), 1: (1, 1, 1), 2: (1, 1, 1), 3: (1, 1, 1)},
    "standard": {0: (1, 1, 1), 1: (1, 1, 1), 2: (2, 4, 2), 3: (8, 32, 4)},
    "high": {0: (1, 1, 1), 1: (2, 2, 1), 2: (4, 8, 4), 3: (16, 64, 8)},
    # Offline grading of recorded hands, where nobody is waiting on the answer.
    # The rungs above leave T1 at one sample each way, and one sample is coarse
    # enough to be a real problem for grading rather than playing: measured on a
    # 27-candidate T1, (1,1,1) produced 8 distinct scores, so a fifth-ranked
    # move came back tied with the best.  (8,16,2) produced 20.  T0 stays at
    # (1,1,1) on every rung -- 232 candidate boards with every street still
    # ahead, so its cost is minutes, not seconds, the moment it is raised.
    "deep": {0: (1, 1, 1), 1: (8, 16, 2), 2: (8, 16, 4), 3: (16, 64, 8)},
    # Same, but paying for T0 as well.  Measured on this machine: T0-first at
    # (2,4,1) took 329 s against 40 s at (1,1,1), and separated 41 of its 232
    # candidates instead of 20.  That is the whole trade -- a better opening
    # grade for roughly eight times the wait, so it is a rung the user picks,
    # never a default.
    "deep_t0": {0: (2, 4, 1), 1: (8, 16, 2), 2: (8, 16, 4), 3: (16, 64, 8)},
}

# Which emitted score `sorted_index` was built from, per street.  See the note
# in _normalize_rows; the first field present wins, `ev` is the last resort.
RANK_FIELD_BY_TURN = {
    0: ("candidate_score", "ev"),
    1: ("ev",),
    2: ("ev",),
    # T3 moved to `ev` on 2026-08-15, and the rows are re-sorted by it below.
    # The engine still orders T3 by the candidate batch, but that statistic
    # does not converge: held at one root, raising `evaluation_samples`
    # 1,024 -> 8,192 -> 32,768 moved its top-two gap +2.875 -> +7.325 -> +6.950,
    # and raising `candidate_samples` 8 -> 512 moved it +7.325 -> +4.438, while
    # `ev` settled at +4.78 / +4.99 / +4.92 and +4.99 / +5.00 / +4.93.  Ranking
    # on the unstable one is why a user sees a T3 verdict change with the seed
    # (measured: four distinct best moves over eight seeds on one hand).
    #
    # Safe here in a way it is not at T0: this module sets no prefilter, so
    # every T3 row is scored on the same evaluation batch and their `ev`s are
    # comparable.  T4 is left on the candidate batch because nothing has
    # measured it; it decides by exact enumeration, so both fields should be
    # deterministic there, but "should be" is not a measurement.
    3: ("ev",),
    4: ("selection_score", "ev"),
}

_lock = threading.Lock()
_library = None
_weights: Optional[Dict[str, tuple]] = None


class EngineUnavailable(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_loaded():
    global _library, _weights
    with _lock:
        if _library is not None:
            return _library, _weights
        if not LIBRARY_PATH.is_file():
            raise EngineUnavailable(f"engine dll not found: {LIBRARY_PATH}")
        weights = {}
        for name, filename in WEIGHT_FILES.items():
            path = WEIGHTS_DIR / filename
            if not path.is_file():
                raise EngineUnavailable(f"learned weights missing: {path}")
            weights[name] = (str(path), _sha256(path))
        _library = load_native_engine(path=LIBRARY_PATH)
        _weights = weights
        logger.info("m3 engine loaded: %s", LIBRARY_PATH)
        return _library, _weights


def available() -> bool:
    try:
        _ensure_loaded()
        return True
    except Exception:
        return False


STREET_NAMES = {0: "T0", 1: "T1", 2: "T2", 3: "T3", 4: "T4"}


def _observation(
    hero_board: Dict[str, Sequence[str]],
    opp_board: Dict[str, Sequence[str]],
    dealt: Sequence[str],
    dead: Sequence[str],
    turn: int,
    position: str,
    opp_in_fl: bool = False,
) -> ActorObservation:
    """Build the information set the acting seat faced.

    `opp_in_fl` is not a hint, it is a different geometry: an opponent in
    Fantasyland took fourteen cards face down and places nothing where the hero
    can see it, so their public board is zero cards at EVERY street and the
    observation says so (`hu_infoset` enforces exactly that, and the flag is
    inside the fingerprint, so the position hashes differently from the same
    cards against a normal opponent).
    """
    return ActorObservation(
        hero_board=Board.from_rows(
            hero_board.get("top", ()), hero_board.get("middle", ()), hero_board.get("bottom", ())
        ),
        opponent_public_board=Board.from_rows(
            () if opp_in_fl else opp_board.get("top", ()),
            () if opp_in_fl else opp_board.get("middle", ()),
            () if opp_in_fl else opp_board.get("bottom", ()),
        ),
        dealt_cards=tuple(dealt),
        hero_private_discards=tuple(dead),
        seat=position,
        street=STREET_NAMES[turn],
        to_act_order=position,
        opponent_in_fantasyland=bool(opp_in_fl),
    )


def _joint_config(observation: ActorObservation, turn: int, precision: str) -> JointExactConfig:
    _, weights = _ensure_loaded()
    cand, evals, ds3 = SAMPLES.get(precision, SAMPLES["standard"]).get(turn, (1, 1, 1))
    seed = int(observation.fingerprint()[:16], 16) & ((1 << 63) - 1)
    return JointExactConfig(
        candidate_samples=cand,
        evaluation_samples=evals,
        downstream_t3_samples=ds3,
        downstream_t4_samples=0,
        seed=seed,
        run_id=f"trainer:{observation.fingerprint()[:16]}",
        seat=observation.seat,
        to_act_order=observation.to_act_order,
        learned_t4_model_path=weights["t4"][0],
        learned_t4_model_sha256=weights["t4"][1],
        learned_t3_second_model_path=weights["t3_second"][0],
        learned_t3_second_model_sha256=weights["t3_second"][1],
        learned_t3_first_model_path=weights["t3_first"][0],
        learned_t3_first_model_sha256=weights["t3_first"][1],
        learned_t2_second_model_path=weights["t2_second"][0],
        learned_t2_second_model_sha256=weights["t2_second"][1],
        learned_t2_first_model_path=weights["t2_first"][0],
        learned_t2_first_model_sha256=weights["t2_first"][1],
        learned_t1_second_model_path=weights["t1_second"][0],
        learned_t1_second_model_sha256=weights["t1_second"][1],
        learned_t1_first_model_path=weights["t1_first"][0],
        learned_t1_first_model_sha256=weights["t1_first"][1],
        learned_t0_second_model_path=weights["t0_second"][0],
        learned_t0_second_model_sha256=weights["t0_second"][1],
        learned_t0_first_model_path=weights["t0_first"][0],
        learned_t0_first_model_sha256=weights["t0_first"][1],
        fast_t0_second_model_path=weights["fast_t0_second"][0],
        fast_t0_second_model_sha256=weights["fast_t0_second"][1],
        fast_t1_second_model_path=weights["fast_t1_second"][0],
        fast_t1_second_model_sha256=weights["fast_t1_second"][1],
        fast_t1_first_model_path=weights["fast_t1_first"][0],
        fast_t1_first_model_sha256=weights["fast_t1_first"][1],
        fast_t2_second_model_path=weights["fast_t2_second"][0],
        fast_t2_second_model_sha256=weights["fast_t2_second"][1],
        fast_t2_first_model_path=weights["fast_t2_first"][0],
        fast_t2_first_model_sha256=weights["fast_t2_first"][1],
    )


def _normalize_rows(
    rows: List[Dict[str, Any]],
    hero_board: Dict[str, Sequence[str]],
    turn: int,
) -> List[Dict[str, Any]]:
    ordered = sorted(rows, key=lambda r: (int(r.get("sorted_index", 0)), str(r.get("action_key", ""))))
    candidates = []
    base = {r: list(hero_board.get(r, ())) for r in ("top", "middle", "bottom")}
    for row in ordered:
        placements = [[c, r] for c, r in row.get("placements", ())]
        discards = list(row.get("discards", ()) or ())
        board = {r: list(base[r]) for r in base}
        for card, target in placements:
            board[target].append(card)
        ev = row.get("score")
        if ev is None:
            ev = row.get("joint_ev")
        metrics = {"ev": float(ev)}
        for src, dst in (
            ("joint_ev", "joint_ev"),
            ("selection_score", "selection_score"),
            ("candidate_score", "candidate_score"),
        ):
            if row.get(src) is not None:
                metrics[dst] = float(row[src])
        # The quantity `sorted_index` was built from, which is street-dependent.
        #
        # Every street scores each action twice: once on the cheap candidate
        # batch the engine selects on, once on the evaluation batch.  Which one
        # the returned order reflects differs by street (search.rs): T0 sorts on
        # `selection_order` (candidate batch, emitted as `candidate_score`),
        # T1/T2 sort on `evaluation_order` (`score`, i.e. `ev`), and T3/T4 sort
        # on the candidate batch again (emitted as `selection_score`).
        #
        # Ranking a move by one of those and pricing its loss with the other is
        # not a rounding difference: measured at T1, the 5th-ranked action came
        # out at zero loss and was graded "best".  Nor can T0 simply switch to
        # `ev` -- its staged/raced search measures pruned rows on fewer
        # particles, so `score` is not comparable across T0 rows at all.
        ranked_by = next(
            (f for f in RANK_FIELD_BY_TURN.get(turn, ()) if metrics.get(f) is not None),
            "ev",
        )
        metrics["rank_score"] = metrics[ranked_by]
        metrics["ranked_by"] = ranked_by
        candidates.append(
            {
                "action": {"placements": placements, "discard": discards[0] if discards else None},
                "board": board,
                "metrics": metrics,
                "key": "|".join(",".join(sorted(board[r])) for r in ("top", "middle", "bottom")),
            }
        )
    # Rank by the number the rank is priced with.  For T0/T1/T2 that is the
    # field the engine already sorted by, so this reproduces `sorted_index`;
    # for T3 it is the change, since the engine's order there comes from the
    # statistic that does not converge.  The board key breaks ties so equal
    # rows come back in a fixed order rather than an arbitrary one.
    candidates.sort(key=lambda c: (-c["metrics"]["rank_score"], c["key"]))
    return candidates


_SOLVED_OPENINGS: Dict[str, Any] | None = None
_SOLVED_OPENINGS_LOADED = False


def _solved_openings_table() -> Dict[str, Any] | None:
    """The measured T0 first-seat table, loaded once, absent if it is not there."""

    global _SOLVED_OPENINGS, _SOLVED_OPENINGS_LOADED
    if not _SOLVED_OPENINGS_LOADED:
        _SOLVED_OPENINGS_LOADED = True
        try:
            from ofc_regular.hu_t0_solved_openings_v1 import DEFAULT_TABLE

            if DEFAULT_TABLE.is_file():
                _SOLVED_OPENINGS = json.loads(
                    DEFAULT_TABLE.read_text(encoding="utf-8")
                )
        except Exception:  # noqa: BLE001 - a missing table must never break play
            _SOLVED_OPENINGS = None
    return _SOLVED_OPENINGS


def _solved_opening(
    hero_board: Dict[str, Sequence[str]],
    opp_board: Dict[str, Sequence[str]],
    dealt: Sequence[str],
    dead: Sequence[str],
    turn: int,
    position: str,
) -> Dict[str, Any] | None:
    """The stored placement for this opening, if it is one that was measured.

    Only at T0 acting first with both boards empty and nothing discarded --
    which is the whole of what the table indexes, since that is the only
    position where the five cards ARE the information set and suit permutations
    map it to itself.  Anything else falls through to the engine.

    Only `settled` entries are played.  An `unresolved` row's pick is the
    argmax of estimates whose first-to-second gap does not exclude zero, so it
    carries the upward bias the maximum of several noisy means always carries,
    and three of the five sit at 1,024 particles where the gap's own scatter is
    0.462.  Such a row is the best action known, not one measured to beat the
    model's, so the model answers those.
    """

    if turn != 0 or position != "first" or dealt is None or len(dealt) != 5:
        return None
    if dead or any(hero_board.get(row) for row in ("top", "middle", "bottom")):
        return None
    if any(opp_board.get(row) for row in ("top", "middle", "bottom")):
        return None
    table = _solved_openings_table()
    if table is None:
        return None

    try:
        from ofc_regular.hu_t0_solved_openings_v1 import lookup

        answer = lookup(list(dealt), table)
    except Exception:  # noqa: BLE001
        return None
    if answer is None or answer["status"] != "settled":
        return None

    placements = [[card, row]
                  for row in ("top", "middle", "bottom")
                  for card in answer["best"][row]]
    # The relabelling has to reproduce the querent's own five cards. If it does
    # not, the entry is being read through the wrong permutation and playing it
    # would be an illegal action rather than a wrong one.
    if sorted(card for card, _ in placements) != sorted(dealt):
        return None

    return {
        "action": {"placements": placements, "discard": None},
        "evaluator": (f"table:t0_solved_openings("
                      f"{answer['status']},{answer['particles_behind_gap']}p)"),
        "solved_opening": {
            "canonical_hand": answer["canonical_hand"],
            "status": answer["status"],
            "best_ev": answer["best_ev"],
            "gap_to_second": answer["gap_to_second"],
            "particles": answer["particles_behind_gap"],
        },
    }


def decide_with_engine(
    *,
    hero_board: Dict[str, Sequence[str]],
    opp_board: Dict[str, Sequence[str]],
    dealt: Sequence[str],
    dead: Sequence[str],
    turn: int,
    position: str,
    precision: str = "fast",
) -> Dict[str, Any]:
    """The single action the learned model plays -- no ranking, no search.

    This is a different operation from `evaluate_with_engine`, not a cheaper
    setting of it.  `evaluate_*` runs the joint-exact TEACHER, the deep search
    whose output became these models' training labels; `decide` asks the model
    what it plays.  At T0 first seat that is 0.09 s against the teacher's 20 s,
    because one is a forward pass over the root fan and the other searches every
    street below it.

    The trainer needs the teacher for grading -- a rank and an EV gap require
    every candidate scored -- but the AI opponent only needs a move, and making
    it pay for a full ranking is what put a 19.5 s wall in front of every hand
    the hero played second.
    """
    solved = _solved_opening(hero_board, opp_board, dealt, dead, turn, position)
    if solved is not None:
        return solved

    library, _ = _ensure_loaded()
    observation = _observation(hero_board, opp_board, dealt, dead, turn, position)
    config = _joint_config(observation, turn, precision)
    request = {
        "schema": HU_M3_REQUEST_SCHEMA,
        "kind": "decide",
        "observation": observation.to_dict(),
        "observation_fingerprint": observation.fingerprint(),
        "config": _joint_config_payload(config),
    }
    response = evaluate_request(request, library=library)
    placements = [[c, r] for c, r in response.get("placements", ())]
    if not placements:
        raise EngineUnavailable(f"decide returned no placements for T{turn}")
    discards = list(response.get("discards", ()) or ())
    return {
        "action": {"placements": placements, "discard": discards[0] if discards else None},
        "evaluator": f"rust:decide({response.get('evaluator', '')})",
    }


# `rak1:<top>:<middle>:<bottom>:<discard>`, four 52-bit masks over ALL_CARDS.
_ACTION_KEY_PREFIX = "rak1"


def _decode_action_key(token: str) -> Dict[str, List[str]]:
    parts = token.split(":")
    if len(parts) != 5 or parts[0] != _ACTION_KEY_PREFIX:
        raise EngineUnavailable(f"unrecognised action key: {token!r}")
    out: Dict[str, List[str]] = {}
    for name, hexmask in zip(("top", "middle", "bottom", "discard"), parts[1:]):
        mask = int(hexmask, 16)
        out[name] = [ALL_CARDS[i] for i in range(len(ALL_CARDS)) if mask & (1 << i)]
    return out


def model_scores_with_engine(
    *,
    hero_board: Dict[str, Sequence[str]],
    opp_board: Dict[str, Sequence[str]],
    dealt: Sequence[str],
    dead: Sequence[str],
    turn: int,
    position: str,
    precision: str = "fast",
    opp_in_fl: bool = False,
) -> Dict[str, Any]:
    """Rank every legal action with the street's learned model, in one pass.

    A different question from `evaluate_with_engine`, not a cheaper setting of
    it.  That one runs the joint-exact teacher, which rolls every candidate out
    over sampled worlds; this asks the model those rollouts trained.

    For grading a recorded hand the model is the better instrument, and the
    reason is measured rather than aesthetic.  `docs/trainer_ranking_quality_
    20260808.md` put the teacher's T0 ranking under five seeds and got five
    different best openings, with a mean per-action spread of 25.87 points
    against a hand that settles for about +/-40 -- and narrowing the fan to ten
    and raising the world count to 256 did not converge either (its §2a).  The
    model's answer, by contrast, is a function of the position alone: no seed,
    no particles, same bytes in and same ranking out.

    T4 has no learned ranking and the engine says so; that street stays on its
    exact enumeration, which has no noise to remove.
    """
    if turn == 4:
        return evaluate_with_engine(
            hero_board=hero_board, opp_board=opp_board, dealt=dealt, dead=dead,
            turn=turn, position=position, precision=precision, opp_in_fl=opp_in_fl,
        )
    if turn not in (0, 1, 2, 3):
        raise EngineUnavailable("model ranking covers T0-T4 only")

    library, _ = _ensure_loaded()
    observation = _observation(
        hero_board, opp_board, dealt, dead, turn, position, opp_in_fl=opp_in_fl
    )
    config = _joint_config(observation, turn, precision)
    response = evaluate_request(
        {
            "schema": HU_M3_REQUEST_SCHEMA,
            "kind": "model_scores",
            "observation": observation.to_dict(),
            "observation_fingerprint": observation.fingerprint(),
            "config": _joint_config_payload(config),
        },
        library=library,
    )

    base = {r: list(hero_board.get(r, ())) for r in ("top", "middle", "bottom")}
    candidates = []
    for row in response.get("actions", ()):
        action = _decode_action_key(row["action_key"])
        board = {r: list(base[r]) + list(action[r]) for r in ("top", "middle", "bottom")}
        placements = [[c, r] for r in ("top", "middle", "bottom") for c in action[r]]
        score = float(row["score"])
        candidates.append(
            {
                "action": {
                    "placements": placements,
                    "discard": action["discard"][0] if action["discard"] else None,
                },
                "board": board,
                # One number, and the ranking is by it -- there is no second
                # measurement here to disagree with the order.
                "metrics": {"ev": score, "rank_score": score, "ranked_by": "model"},
                "key": "|".join(",".join(sorted(board[r])) for r in ("top", "middle", "bottom")),
            }
        )
    if not candidates:
        raise EngineUnavailable(f"model ranking returned no actions for T{turn}")
    return {"candidates": candidates, "evaluator": f"rust:model(T{turn})"}


def evaluate_with_engine(
    *,
    hero_board: Dict[str, Sequence[str]],
    opp_board: Dict[str, Sequence[str]],
    dealt: Sequence[str],
    dead: Sequence[str],
    turn: int,
    position: str,
    precision: str,
    opp_in_fl: bool = False,
) -> Dict[str, Any]:
    """Full ranked candidate list from the m3 engine (T0-T4)."""
    if turn not in (0, 1, 2, 3, 4):
        raise EngineUnavailable("engine evaluation covers T0-T4 only")
    library, _ = _ensure_loaded()
    observation = _observation(
        hero_board, opp_board, dealt, dead, turn, position, opp_in_fl=opp_in_fl
    )

    if turn == 4:
        seed = int(observation.fingerprint()[:16], 16) & ((1 << 63) - 1)
        request = t4_request(
            observation,
            config=T4SearchConfig(
                candidate_samples=0,
                evaluation_samples=0,
                seed=seed,
                run_id=f"trainer-t4:{observation.fingerprint()[:16]}",
            ),
        )
        label = "rust:t4_exact"
    else:
        config = _joint_config(observation, turn, precision)
        if turn == 3:
            request = t3_request(observation, config=config)
        elif turn == 2:
            request = t2_request(observation, config=config)
        elif turn == 1:
            request = t1_request(observation, config=config)
        else:
            request = t0_request(observation, config=config)
        label = f"rust:t{turn}({config.candidate_samples}/{config.evaluation_samples})"

    response = evaluate_request(request, library=library)
    rows = response.get("actions")
    if not rows:
        raise EngineUnavailable(f"engine returned no actions for T{turn}")
    return {
        "candidates": _normalize_rows(rows, hero_board, turn),
        "evaluator": label,
    }
