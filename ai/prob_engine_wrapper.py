"""
Python wrapper for Rust Probability Engine.

Calls prob_engine CLI via subprocess, returns parsed JSON results.
Supports three modes:
  - row: single row probability distribution
  - board: full board evaluation (bust, FL, royalty, EV)
  - candidates: evaluate all placement candidates for a turn
"""

import subprocess
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

_ext = ".exe" if sys.platform == "win32" else ""
PROB_ENGINE_PATH = Path(__file__).parent / "rust_solver" / "target" / "release" / f"prob_engine{_ext}"


def _cards_to_str(cards: List[str]) -> str:
    """Convert card list to comma-separated string."""
    return ",".join(cards) if cards else ""


def _run_prob_engine(args: List[str], engine_path: Optional[str] = None) -> Dict[str, Any]:
    """Run prob_engine with given arguments, return parsed JSON."""
    exe = engine_path or str(PROB_ENGINE_PATH)
    timeout = int(os.environ.get("PROB_ENGINE_TIMEOUT", "600"))
    result = subprocess.run(
        [exe] + args,
        capture_output=True, text=True, timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(f"prob_engine failed: {result.stderr}")
    return json.loads(result.stdout)


def evaluate_row(
    row_cards: List[str],
    row_type: str = "top",
    exclude: Optional[List[str]] = None,
    engine_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate a single row's probability distribution.

    Args:
        row_cards: Cards in the row (e.g., ["As", "Ah"])
        row_type: "top", "mid", or "bot"
        exclude: Known dead cards (opponent board, discards, etc.)

    Returns:
        dict with histogram, categories, expected_royalty, fl_rate, etc.
    """
    args = [
        "--mode", "row",
        "--row", _cards_to_str(row_cards),
        "--row-type", row_type,
    ]
    if exclude:
        args += ["--exclude", _cards_to_str(exclude)]
    return _run_prob_engine(args, engine_path)


def evaluate_board(
    top: List[str],
    mid: List[str],
    bot: List[str],
    exclude: Optional[List[str]] = None,
    turn: int = 1,
    position: str = "bb",
    engine_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate full board state.

    Args:
        top/mid/bot: Cards in each row
        exclude: Additional dead cards
        turn: Turn number (0-4)
        position: "btn", "bb", "fl_btn", "fl_bb", "fl_vs_fl", "fl_vs_normal"

    Returns:
        dict with bust_prob, fl_rate, expected_royalty, ev, per-row distributions
    """
    args = [
        "--mode", "board",
        "--top", _cards_to_str(top),
        "--mid", _cards_to_str(mid),
        "--bot", _cards_to_str(bot),
        "--turn", str(turn),
        "--position", position,
    ]
    if exclude:
        args += ["--exclude", _cards_to_str(exclude)]
    return _run_prob_engine(args, engine_path)


def evaluate_candidates(
    top: List[str],
    mid: List[str],
    bot: List[str],
    dealt: List[str],
    exclude: Optional[List[str]] = None,
    turn: int = 1,
    position: str = "bb",
    engine_path: Optional[str] = None,
    candidate_limit: int = 0,
    candidate_filter: str = "all",
) -> Dict[str, Any]:
    """Evaluate all placement candidates for a turn.

    Args:
        top/mid/bot: Current board cards
        dealt: Cards dealt this turn (3 for T1-T4, 5 for T0)
        exclude: Additional dead cards (opponent board, discards)
        turn: Turn number (0-4)
        position: Player position

    Returns:
        dict with candidates list sorted by EV descending.
        Each candidate has: placements, discard, ev, bust_prob, fl_rate, expected_royalty
    """
    args = [
        "--mode", "candidates",
        "--top", _cards_to_str(top),
        "--mid", _cards_to_str(mid),
        "--bot", _cards_to_str(bot),
        "--dealt", _cards_to_str(dealt),
        "--turn", str(turn),
        "--position", position,
    ]
    if exclude:
        args += ["--exclude", _cards_to_str(exclude)]
    if candidate_limit:
        args += ["--candidate-limit", str(candidate_limit)]
    if candidate_filter != "all":
        args += ["--candidate-filter", candidate_filter]
    return _run_prob_engine(args, engine_path)


def select_best_action(
    top: List[str],
    mid: List[str],
    bot: List[str],
    dealt: List[str],
    exclude: Optional[List[str]] = None,
    turn: int = 1,
    position: str = "bb",
    engine_path: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Select the best placement candidate.

    Returns:
        (best_candidate, full_result)
        best_candidate has: placements, discard, ev, bust_prob, fl_rate
    """
    result = evaluate_candidates(
        top, mid, bot, dealt, exclude, turn, position, engine_path)
    best = result["candidates"][0] if result["candidates"] else None
    return best, result


def evaluate_mc_t0(
    dealt: List[str],
    exclude: Optional[List[str]] = None,
    sims: int = 5,
    engine_path: Optional[str] = None,
    candidate_limit: int = 0,
    candidate_filter: str = "all",
) -> Dict[str, Any]:
    """Evaluate all T0 placement candidates via MC simulation.

    Args:
        dealt: 5 dealt cards for T0
        exclude: Dead cards (opponent board, discards)
        sims: Number of MC simulations per candidate

    Returns:
        dict with candidates list sorted by avg_score descending.
        Each candidate has: placements, discard, mc {avg_score, bust_rate, fl_rate, ...}
    """
    args = [
        "--mode", "mc",
        "--dealt", _cards_to_str(dealt),
        "--turn", "0",
        "--sims", str(sims),
    ]
    if exclude:
        args += ["--exclude", _cards_to_str(exclude)]
    if candidate_limit:
        args += ["--candidate-limit", str(candidate_limit)]
    if candidate_filter != "all":
        args += ["--candidate-filter", candidate_filter]
    return _run_prob_engine(args, engine_path)


def evaluate_t0_ladder(
    dealt: List[str],
    exclude: Optional[List[str]] = None,
    stage_sims: str = "2,5,10,25,50",
    stage_limits: str = "200,150,64,24,8",
    engine_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate T0 candidates with successive halving.

    Stage 0 evaluates all 232 legal T0 placements shallowly, then keeps the
    best stage_limits[0] candidates. Later stages re-evaluate only survivors
    with deeper simulation counts.
    """
    args = [
        "--mode", "t0_ladder",
        "--dealt", _cards_to_str(dealt),
        "--stage-sims", stage_sims,
        "--stage-limits", stage_limits,
    ]
    if exclude:
        args += ["--exclude", _cards_to_str(exclude)]
    return _run_prob_engine(args, engine_path)


def evaluate_board_mc(
    top: List[str],
    mid: List[str],
    bot: List[str],
    exclude: Optional[List[str]] = None,
    start_turn: int = 1,
    sims: int = 1000,
    engine_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate one fixed post-action board by MC continuation."""
    args = [
        "--mode", "board_mc",
        "--top", _cards_to_str(top),
        "--mid", _cards_to_str(mid),
        "--bot", _cards_to_str(bot),
        "--turn", str(start_turn),
        "--sims", str(sims),
    ]
    if exclude:
        args += ["--exclude", _cards_to_str(exclude)]
    return _run_prob_engine(args, engine_path)


def evaluate_board_recursive_mc(
    top: List[str],
    mid: List[str],
    bot: List[str],
    exclude: Optional[List[str]] = None,
    start_turn: int = 1,
    sims: int = 32,
    beam_width: int = 5,
    child_sims: int = 2,
    engine_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate one fixed post-action board with recursive T1-T3 rollouts."""
    args = [
        "--mode", "board_recursive_mc",
        "--top", _cards_to_str(top),
        "--mid", _cards_to_str(mid),
        "--bot", _cards_to_str(bot),
        "--turn", str(start_turn),
        "--sims", str(sims),
        "--recursive-beam", str(beam_width),
        "--recursive-child-sims", str(child_sims),
    ]
    if exclude:
        args += ["--exclude", _cards_to_str(exclude)]
    return _run_prob_engine(args, engine_path)


def evaluate_mc_candidates(
    top: List[str],
    mid: List[str],
    bot: List[str],
    dealt: List[str],
    exclude: Optional[List[str]] = None,
    turn: int = 1,
    sims: int = 5,
    engine_path: Optional[str] = None,
    candidate_limit: int = 0,
    candidate_filter: str = "all",
) -> Dict[str, Any]:
    """Evaluate all T1+ placement candidates via MC simulation.

    Each candidate placement is applied, then MC simulated from the next turn.
    Returns candidates sorted by avg_score descending with bust_rate/fl_rate.

    Args:
        top/mid/bot: Current board cards
        dealt: 3 dealt cards for T1-T4
        exclude: Dead cards (opponent board, discards)
        turn: Current turn number (1-4)
        sims: Number of MC simulations per candidate

    Returns:
        dict with candidates list, each having: placements, discard, mc {avg_score, bust_rate, fl_rate, ...}
    """
    args = [
        "--mode", "mc",
        "--top", _cards_to_str(top),
        "--mid", _cards_to_str(mid),
        "--bot", _cards_to_str(bot),
        "--dealt", _cards_to_str(dealt),
        "--turn", str(turn),
        "--sims", str(sims),
    ]
    if exclude:
        args += ["--exclude", _cards_to_str(exclude)]
    if candidate_limit:
        args += ["--candidate-limit", str(candidate_limit)]
    if candidate_filter != "all":
        args += ["--candidate-filter", candidate_filter]
    return _run_prob_engine(args, engine_path)


def get_feature_vector(board_eval: Dict[str, Any]) -> List[float]:
    """Extract feature vector from a board evaluation result.

    Feature layout (306 dims + 18 deck = 324 total):
      Top fine_hist:  100 dims
      Mid fine_hist:  100 dims
      Bot fine_hist:  100 dims
      FL rates (QQ, KK, AA, Trips): 4 dims
      Bust diagnostic (top>mid, mid>bot): 2 dims
      Deck composition (13 ranks + 4 suits + 1 joker): 18 dims (caller appends)
    """
    ev = board_eval.get("evaluation", board_eval)

    features = []

    # Fine histograms (100 bins × 3 rows = 300 dims)
    features.extend(ev["top"]["fine_hist"])
    features.extend(ev["mid"]["fine_hist"])
    features.extend(ev["bot"]["fine_hist"])

    # FL rates (4 dims)
    features.append(ev["top"].get("fl_qq_rate", 0.0) or 0.0)
    features.append(ev["top"].get("fl_kk_rate", 0.0) or 0.0)
    features.append(ev["top"].get("fl_aa_rate", 0.0) or 0.0)
    features.append(ev["top"].get("fl_trips_rate", 0.0) or 0.0)

    # Bust diagnostic (2 dims)
    features.append(ev["bust_top_mid"])
    features.append(ev["bust_mid_bot"])

    # Deck composition: caller should append 18-dim deck composition separately
    # (13 rank counts + 4 suit counts + 1 joker count, each normalized)

    return features  # 306 dims (caller adds 18 for deck = 324 total)


if __name__ == "__main__":
    # Quick test
    print(f"Engine: {PROB_ENGINE_PATH}")
    print(f"Exists: {PROB_ENGINE_PATH.exists()}")

    if PROB_ENGINE_PATH.exists():
        # Test board evaluation
        result = evaluate_board(
            top=["As"],
            mid=["Ks", "Kh"],
            bot=["Ah", "Ad"],
            turn=1,
            position="btn",
        )
        ev = result["evaluation"]
        print(f"\nBoard eval:")
        print(f"  EV: {ev['ev']:.2f}")
        print(f"  Bust: {ev['bust_prob']:.1%}")
        print(f"  FL rate: {ev['fl_rate']:.1%}")
        print(f"  Royalty: {ev['expected_royalty']:.2f}")

        # Test candidate evaluation
        result = evaluate_candidates(
            top=["As"],
            mid=["Ks", "Kh"],
            bot=["Ah", "Ad"],
            dealt=["Qs", "Js", "Ts"],
            turn=1,
            position="btn",
        )
        print(f"\nCandidates: {result['n_candidates']}")
        for i, c in enumerate(result["candidates"][:3]):
            print(f"  #{i+1}: EV={c['ev']:+.2f} bust={c['bust_prob']:.1%} "
                  f"FL={c['fl_rate']:.1%} | {c['placements']} discard={c['discard']}")

        # Test feature extraction
        board_result = evaluate_board(
            top=["As", "Qh"],
            mid=["Ks", "Kh", "Jd"],
            bot=["Ah", "Ad", "Tc"],
            turn=2,
        )
        features = get_feature_vector(board_result)
        print(f"\nFeature vector: {len(features)} dims (+ 18 deck = {len(features)+18} total)")
    else:
        print("Build first: cd ai/rust_solver && cargo build --release -p prob_engine")
