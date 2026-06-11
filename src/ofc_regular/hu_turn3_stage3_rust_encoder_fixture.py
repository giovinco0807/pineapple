"""Build compact fixtures for a future Rust HU Turn3 Stage3 feature encoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .action_space import Action
from .hu_turn3_model import CARD_INDEX, HU_FEATURE_DIM, ROWS
from .hu_turn3_stage3_feature_fast import FEATURE_SCHEMA_VERSION, build_hu_turn3_stage3_feature_matrix_batch
from .hu_turn3_stage3_feature_manifest import build_hu_turn3_stage3_feature_manifest
from .hu_turn3_stage3_feature_replay import load_stage3_feature_replay

ROW_INDEX = {"top": 0, "middle": 1, "bottom": 2}


def write_rust_encoder_fixture(
    *,
    input_path: Path,
    spec_output: Path,
    fixture_output: Path,
    limit_states: int = 0,
) -> dict[str, Any]:
    replay = load_stage3_feature_replay(input_path, limit_states=limit_states)
    if not replay.states:
        raise RuntimeError("feature replay has no states")

    expected = build_hu_turn3_stage3_feature_matrix_batch(
        replay.states,
        replay.actions_by_state,
        FEATURE_SCHEMA_VERSION,
        state_keys=replay.state_keys,
        include_action_encodings=False,
        encoder_mode="numpy_direct_full",
    )
    compact = _compact_arrays(replay.states, replay.actions_by_state)
    fixture_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        fixture_output,
        **compact,
        row_to_state_index=expected.row_to_state_index.astype(np.int32),
        row_to_action_index=expected.row_to_action_index.astype(np.int16),
        expected_features=expected.X.astype(np.float32),
        feature_column_count=np.asarray([HU_FEATURE_DIM], dtype=np.int32),
    )

    manifest = build_hu_turn3_stage3_feature_manifest()
    summary = {
        "input": str(input_path),
        "spec_output": str(spec_output),
        "fixture_output": str(fixture_output),
        "states": len(replay.states),
        "rows": int(expected.X.shape[0]),
        "feature_columns": int(expected.X.shape[1]),
        "feature_dtype": str(expected.X.dtype),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "manifest_version": manifest["version"],
        "compact_arrays": {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            for key, value in compact.items()
        },
    }
    _write_spec(spec_output, summary)
    return summary


def _compact_arrays(states: Sequence[Any], actions_by_state: Sequence[list[Action]]) -> dict[str, np.ndarray]:
    max_actions = max((len(actions) for actions in actions_by_state), default=0)
    max_placements = max(
        (len(action.placements) for actions in actions_by_state for action in actions),
        default=0,
    )
    max_discards = max(
        (len(action.discards) for actions in actions_by_state for action in actions),
        default=0,
    )
    state_count = len(states)
    action_counts = np.zeros(state_count, dtype=np.int16)
    hero_board_masks = np.zeros((state_count, 3), dtype=np.uint64)
    opponent_board_masks = np.zeros((state_count, 3), dtype=np.uint64)
    dead_card_masks = np.zeros(state_count, dtype=np.uint64)
    dealt_card_ids = np.full((state_count, 3), -1, dtype=np.int16)
    seat_ids = np.zeros(state_count, dtype=np.int8)
    order_ids = np.zeros(state_count, dtype=np.int8)
    action_placement_card_ids = np.full((state_count, max_actions, max_placements), -1, dtype=np.int16)
    action_placement_row_ids = np.full((state_count, max_actions, max_placements), -1, dtype=np.int8)
    action_discard_card_ids = np.full((state_count, max_actions, max_discards), -1, dtype=np.int16)

    for state_index, (state, actions) in enumerate(zip(states, actions_by_state)):
        action_counts[state_index] = len(actions)
        hero_board_masks[state_index] = _board_masks(state.board)
        opponent_board_masks[state_index] = _board_masks(state.opponent_board)
        dead_card_masks[state_index] = _cards_mask(state.dead_cards)
        for card_index, card in enumerate(state.dealt_cards[:3]):
            dealt_card_ids[state_index, card_index] = CARD_INDEX[card]
        seat_ids[state_index] = 1 if getattr(state, "seat", "first") == "second" else 0
        order_ids[state_index] = 1 if (getattr(state, "to_act_order", None) or "first") == "second" else 0
        for action_index, action in enumerate(actions):
            for placement_index, (card, row) in enumerate(action.placements):
                action_placement_card_ids[state_index, action_index, placement_index] = CARD_INDEX[card]
                action_placement_row_ids[state_index, action_index, placement_index] = ROW_INDEX[row]
            for discard_index, card in enumerate(action.discards):
                action_discard_card_ids[state_index, action_index, discard_index] = CARD_INDEX[card]

    return {
        "hero_board_masks": hero_board_masks,
        "opponent_board_masks": opponent_board_masks,
        "dead_card_masks": dead_card_masks,
        "dealt_card_ids": dealt_card_ids,
        "seat_ids": seat_ids,
        "order_ids": order_ids,
        "action_counts": action_counts,
        "action_placement_card_ids": action_placement_card_ids,
        "action_placement_row_ids": action_placement_row_ids,
        "action_discard_card_ids": action_discard_card_ids,
    }


def _board_masks(board: Any) -> np.ndarray:
    return np.asarray(
        [
            _cards_mask(board.top),
            _cards_mask(board.middle),
            _cards_mask(board.bottom),
        ],
        dtype=np.uint64,
    )


def _cards_mask(cards: Sequence[str]) -> np.uint64:
    mask = 0
    for card in cards:
        mask |= 1 << CARD_INDEX[card]
    return np.uint64(mask)


def _write_spec(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# HU Turn3 Stage3 Rust Encoder Input Spec",
        "",
        "Scope: Rust only builds the Stage3 reference feature matrix. HGB prediction, Stage3 action selection, Stage7 gate, JSONL writing, and shard orchestration stay in Python.",
        "",
        "Input arrays:",
        "- `hero_board_masks`: uint64 `[states, 3]`, row order top/middle/bottom.",
        "- `opponent_board_masks`: uint64 `[states, 3]`, row order top/middle/bottom.",
        "- `dead_card_masks`: uint64 `[states]`.",
        "- `dealt_card_ids`: int16 `[states, 3]`, card ids use the existing Python `CARD_INDEX` order.",
        "- `seat_ids`: int8 `[states]`, first=0/second=1.",
        "- `order_ids`: int8 `[states]`, first=0/second=1.",
        "- `action_counts`: int16 `[states]`.",
        "- `action_placement_card_ids`: int16 `[states, max_actions, max_placements]`, padded with -1.",
        "- `action_placement_row_ids`: int8 `[states, max_actions, max_placements]`, top=0/middle=1/bottom=2, padded with -1.",
        "- `action_discard_card_ids`: int16 `[states, max_actions, max_discards]`, padded with -1.",
        "",
        "Output:",
        "- `expected_features`: float32 `[rows, 1076]` for parity.",
        "- `row_to_state_index`: int32 `[rows]`.",
        "- `row_to_action_index`: int16 `[rows]`.",
        "",
        "Summary:",
        "```json",
        json.dumps(summary, ensure_ascii=False, indent=2),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--spec-output", type=Path, required=True)
    parser.add_argument("--fixture-output", type=Path, required=True)
    parser.add_argument("--limit-states", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = write_rust_encoder_fixture(
        input_path=args.input,
        spec_output=args.spec_output,
        fixture_output=args.fixture_output,
        limit_states=args.limit_states,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
