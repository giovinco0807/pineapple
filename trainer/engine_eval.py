"""Rust m3 engine adapter: full candidate rankings for T1-T4.

Mirrors the production webapp runtime call pattern (JointExactConfig with the
learned weights, T4 exact enumeration) but keeps the ENTIRE ranked action
list instead of the webapp's top-3 audit rows.
"""

from __future__ import annotations

import hashlib
import logging
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from ofc_regular.hu_infoset import ActorObservation  # noqa: E402
from ofc_regular.hu_late_street_teacher import T4SearchConfig  # noqa: E402
from ofc_regular.hu_m3_rust import (  # noqa: E402
    evaluate_request,
    load_native_engine,
    t4_request,
    t1_request,
    t2_request,
    t3_request,
)
from ofc_regular.hu_turn3_joint_exact_teacher import JointExactConfig  # noqa: E402
from ofc_regular.state import Board  # noqa: E402

logger = logging.getLogger("trainer.engine")

LIBRARY_PATH = _ROOT / "target" / "release" / "ofc_hu_m3_engine.dll"
WEIGHTS_DIR = _ROOT / "rust" / "hu_m3_engine" / "tests" / "fixtures"
WEIGHT_FILES = {
    "t4": "t4_model_v5.bin",
    "t3_second": "t3_model_v2.bin",
    "t3_first": "t3first_model_v1.bin",
    "t2_second": "t2_model_v1.bin",
    "t2_first": "t2first_model_v1.bin",
}

# (candidate_samples, evaluation_samples, downstream_t3_samples) per precision.
# The production webapp runs 1/1/1 for interactive latency.
SAMPLES = {
    "fast": {1: (1, 1, 1), 2: (1, 1, 1), 3: (1, 1, 1)},
    "standard": {1: (1, 1, 1), 2: (2, 4, 2), 3: (8, 32, 4)},
    "high": {1: (2, 2, 1), 2: (4, 8, 4), 3: (16, 64, 8)},
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
) -> ActorObservation:
    return ActorObservation(
        hero_board=Board.from_rows(
            hero_board.get("top", ()), hero_board.get("middle", ()), hero_board.get("bottom", ())
        ),
        opponent_public_board=Board.from_rows(
            opp_board.get("top", ()), opp_board.get("middle", ()), opp_board.get("bottom", ())
        ),
        dealt_cards=tuple(dealt),
        hero_private_discards=tuple(dead),
        seat=position,
        street=STREET_NAMES[turn],
        to_act_order=position,
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
    )


def _normalize_rows(
    rows: List[Dict[str, Any]],
    hero_board: Dict[str, Sequence[str]],
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
        candidates.append(
            {
                "action": {"placements": placements, "discard": discards[0] if discards else None},
                "board": board,
                "metrics": metrics,
                "key": "|".join(",".join(sorted(board[r])) for r in ("top", "middle", "bottom")),
            }
        )
    return candidates


def evaluate_with_engine(
    *,
    hero_board: Dict[str, Sequence[str]],
    opp_board: Dict[str, Sequence[str]],
    dealt: Sequence[str],
    dead: Sequence[str],
    turn: int,
    position: str,
    precision: str,
) -> Dict[str, Any]:
    """Full ranked candidate list from the m3 engine (T1-T4)."""
    if turn not in (1, 2, 3, 4):
        raise EngineUnavailable("engine evaluation covers T1-T4 only")
    library, _ = _ensure_loaded()
    observation = _observation(hero_board, opp_board, dealt, dead, turn, position)

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
        else:
            request = t1_request(observation, config=config)
        label = f"rust:t{turn}({config.candidate_samples}/{config.evaluation_samples})"

    response = evaluate_request(request, library=library)
    rows = response.get("actions")
    if not rows:
        raise EngineUnavailable(f"engine returned no actions for T{turn}")
    return {
        "candidates": _normalize_rows(rows, hero_board),
        "evaluator": label,
    }
