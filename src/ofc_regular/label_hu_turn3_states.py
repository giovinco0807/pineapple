"""Label mined HU Turn3 states with rollout teacher scores."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

from .ai_profiles import (
    DEFAULT_OPENING_MODEL,
    DEFAULT_TURN1_MODEL,
    DEFAULT_TURN2_MODEL,
    DEFAULT_TURN3_MODEL,
    ModelPaths,
    build_policy,
    load_model_bundle,
)
from .hu_self_play_teacher_data import (
    annotate_hu_turn3_reference_actions,
    build_hu_self_play_turn3_sample,
)
from .state import Board


def board_from_json(payload: dict[str, list[str]]) -> Board:
    return Board.from_rows(
        top=payload.get("top", ()),
        middle=payload.get("middle", ()),
        bottom=payload.get("bottom", ()),
    )


def discarded_cards_from_state(state: dict[str, Any], opponent_board: Board) -> tuple[str, ...]:
    if "discarded_cards" in state:
        return tuple(state["discarded_cards"])
    opponent_cards = set(opponent_board.all_cards())
    return tuple(card for card in state.get("dead_cards", ()) if card not in opponent_cards)


def visible_dead_cards_from_state(
    state: dict[str, Any],
    *,
    opponent_board: Board,
    discarded_cards: tuple[str, ...],
) -> tuple[str, ...]:
    if "visible_dead_cards" in state:
        return tuple(state["visible_dead_cards"])
    if "hero_private_discards" in state:
        return (*opponent_board.all_cards(), *tuple(state["hero_private_discards"]))
    return (*opponent_board.all_cards(), *discarded_cards)


def private_discards_from_state(
    state: dict[str, Any],
    key: str,
    *,
    fallback: tuple[str, ...],
) -> tuple[str, ...]:
    if key in state:
        return tuple(state[key])
    return fallback


def opponent_seat(hero_seat: str) -> str:
    return "second" if hero_seat == "first" else "first"


def label_hu_turn3_state(
    state: dict[str, Any],
    *,
    sample_id: int,
    policy_bundle: object,
    future_samples: int,
    opening_lookahead_samples: int,
    self_regret_penalty_weight: float,
    self_regret_free: float,
    rng: random.Random,
    policy_seed: int,
) -> dict | None:
    hero_seat = state.get("hero_seat", state.get("seat", "first"))
    board = board_from_json(state["board"])
    opponent_board = board_from_json(state["opponent_board"])
    discarded_cards = discarded_cards_from_state(state, opponent_board)
    visible_dead_cards = visible_dead_cards_from_state(
        state,
        opponent_board=opponent_board,
        discarded_cards=discarded_cards,
    )
    hero_private_discards = private_discards_from_state(
        state,
        "hero_private_discards",
        fallback=discarded_cards,
    )
    opponent_private_discards = private_discards_from_state(
        state,
        "opponent_private_discards",
        fallback=discarded_cards,
    )
    hero_policy = build_policy(
        "current",
        policy_bundle,
        seed=policy_seed,
        seat=hero_seat,
        opening_lookahead_samples=opening_lookahead_samples,
    )
    villain_policy = build_policy(
        "current",
        policy_bundle,
        seed=policy_seed + 1,
        seat=opponent_seat(hero_seat),
        opening_lookahead_samples=opening_lookahead_samples,
    )
    sample = build_hu_self_play_turn3_sample(
        sample_id=sample_id,
        board=board,
        dealt_cards=state["dealt"],
        opponent_board=opponent_board,
        dead_cards=discarded_cards,
        visible_dead_cards=visible_dead_cards,
        hero_private_discards=hero_private_discards,
        opponent_private_discards=opponent_private_discards,
        hero_seat=hero_seat,
        hero_policy=hero_policy,
        opponent_policy=villain_policy,
        self_regret_model=policy_bundle.turn3,
        self_regret_penalty_weight=self_regret_penalty_weight,
        self_regret_free=self_regret_free,
        future_samples=future_samples,
        rng=rng,
    )
    if sample is None:
        return None
    sample["source"] = "mined_state_rollout"
    sample["mined_state"] = {
        "state_id": state.get("state_id"),
        "seed": state.get("seed"),
        "hand_seed": state.get("hand_seed"),
        "hand_index": state.get("hand_index"),
        "player": state.get("player"),
        "visibility_model": state.get("visibility_model", "legacy_unspecified"),
        "discard_visibility": state.get("discard_visibility", "legacy_unspecified"),
    }
    if "selection" in state:
        sample["selection"] = state["selection"]
        annotate_hu_turn3_reference_actions(sample, state["selection"])
    return sample


def label_hu_turn3_states(
    *,
    input_path: Path,
    output: Path,
    seed: int,
    policy_bundle: object,
    future_samples: int,
    opening_lookahead_samples: int,
    min_score_gap: float,
    self_regret_penalty_weight: float,
    self_regret_free: float,
    max_states: int | None = None,
    progress_every: int = 0,
) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    read_states = 0
    written = 0
    skipped_empty = 0
    skipped_gap = 0
    started_at = time.time()
    with input_path.open("r", encoding="utf-8") as source, output.open("w", encoding="utf-8") as handle:
        for line in source:
            if max_states is not None and read_states >= max_states:
                break
            state = json.loads(line)
            sample = label_hu_turn3_state(
                state,
                sample_id=written,
                policy_bundle=policy_bundle,
                future_samples=future_samples,
                opening_lookahead_samples=opening_lookahead_samples,
                self_regret_penalty_weight=self_regret_penalty_weight,
                self_regret_free=self_regret_free,
                rng=rng,
                policy_seed=seed + read_states * 2,
            )
            read_states += 1
            if sample is None:
                skipped_empty += 1
                continue
            if sample["score_gap"] < min_score_gap:
                skipped_gap += 1
                continue
            handle.write(json.dumps(sample, separators=(",", ":")) + "\n")
            handle.flush()
            written += 1
            if progress_every > 0 and written % progress_every == 0:
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "read_states": read_states,
                            "samples": written,
                            "elapsed_seconds": time.time() - started_at,
                        },
                        separators=(",", ":"),
                    ),
                    flush=True,
                )
    return {
        "input": str(input_path),
        "output": str(output),
        "states_read": read_states,
        "samples": written,
        "skipped_empty": skipped_empty,
        "skipped_gap": skipped_gap,
        "seed": seed,
        "future_samples": future_samples,
        "min_score_gap": min_score_gap,
        "self_regret_penalty_weight": self_regret_penalty_weight,
        "self_regret_free": self_regret_free,
        "elapsed_seconds": time.time() - started_at,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--future-samples", type=int, default=16)
    parser.add_argument("--min-score-gap", type=float, default=0.0)
    parser.add_argument("--self-regret-penalty-weight", type=float, default=0.0)
    parser.add_argument("--self-regret-free", type=float, default=0.0)
    parser.add_argument("--max-states", type=int)
    parser.add_argument("--opening-lookahead-samples", type=int, default=64)
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--opening-model", type=Path, default=DEFAULT_OPENING_MODEL)
    parser.add_argument("--turn1-model", type=Path, default=DEFAULT_TURN1_MODEL)
    parser.add_argument("--turn2-model", type=Path, default=DEFAULT_TURN2_MODEL)
    parser.add_argument("--turn3-model", type=Path, default=DEFAULT_TURN3_MODEL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.future_samples < 0:
        raise SystemExit("--future-samples must be non-negative")
    if args.self_regret_penalty_weight < 0:
        raise SystemExit("--self-regret-penalty-weight must be non-negative")
    if args.self_regret_free < 0:
        raise SystemExit("--self-regret-free must be non-negative")
    paths = ModelPaths(
        opening=args.opening_model,
        turn1=args.turn1_model,
        turn2=args.turn2_model,
        turn3=args.turn3_model,
    )
    policy_bundle = load_model_bundle(paths, {"current"})
    summary = label_hu_turn3_states(
        input_path=args.input,
        output=args.output,
        seed=args.seed,
        policy_bundle=policy_bundle,
        future_samples=args.future_samples,
        opening_lookahead_samples=args.opening_lookahead_samples,
        min_score_gap=args.min_score_gap,
        self_regret_penalty_weight=args.self_regret_penalty_weight,
        self_regret_free=args.self_regret_free,
        max_states=args.max_states,
        progress_every=args.progress_every,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
