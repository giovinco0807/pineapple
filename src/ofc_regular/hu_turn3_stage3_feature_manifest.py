"""Feature manifest for HU Turn3 Stage3/reference feature matrices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .hu_turn3_model import (
    DEAD_OFFSET,
    GLOBAL_DIM,
    GLOBAL_OFFSET,
    HU_DERIVED_DIM,
    HU_DERIVED_OFFSET,
    HU_FEATURE_DIM,
    HU_MATCHUP_DIM,
    HU_MATCHUP_OFFSET,
    OPPONENT_OFFSET,
    OPP_RANK_COUNT_OFFSET,
    OPP_ROW_LEN_OFFSET,
    OPP_SUIT_COUNT_OFFSET,
    ORDER_OFFSET,
    SEAT_OFFSET,
)
from .turn3_model import (
    BASE_FEATURE_DIM,
    CURRENT_OFFSET,
    DEALT_OFFSET,
    DISCARD_OFFSET,
    FEATURE_DIM as SELF_FEATURE_DIM,
    GLOBAL_EXTRA_DIM,
    GLOBAL_EXTRA_OFFSET,
    NEXT_OFFSET,
    PLACEMENT_OFFSET,
    RANK_COUNT_OFFSET,
    ROW_EXTRA_DIM,
    ROW_EXTRA_OFFSET,
    ROW_LEN_OFFSET,
    SUIT_COUNT_OFFSET,
)

FEATURE_MANIFEST_VERSION = "hu_turn3_stage3_feature_manifest_v1"


def build_hu_turn3_stage3_feature_manifest() -> dict[str, Any]:
    features: list[dict[str, Any]] = []
    _add_range(features, CURRENT_OFFSET, 3 * 52, "current_board", "state_base")
    _add_range(features, DEALT_OFFSET, 52, "dealt_cards", "state_base")
    _add_range(features, PLACEMENT_OFFSET, 3 * 52, "placements", "action_encoding")
    _add_range(features, DISCARD_OFFSET, 52, "discards", "action_encoding")
    _add_range(features, NEXT_OFFSET, 3 * 52, "next_board", "after_board")
    _add_range(features, ROW_LEN_OFFSET, 3, "next_row_len", "after_board")
    _add_range(features, RANK_COUNT_OFFSET, 3 * 13, "next_rank_count", "after_board")
    _add_range(features, SUIT_COUNT_OFFSET, 3 * 4, "next_suit_count", "after_board")
    _add_range(features, BASE_FEATURE_DIM, SELF_FEATURE_DIM - BASE_FEATURE_DIM, "self_extra", "royalty_fl")

    _add_range(features, OPPONENT_OFFSET, 3 * 52, "opponent_board", "opponent_interaction")
    _add_range(features, DEAD_OFFSET, 52, "dead_cards", "outs_blocker")
    _add_range(features, SEAT_OFFSET, 2, "seat", "state_base")
    _add_range(features, ORDER_OFFSET, 2, "to_act_order", "state_base")
    _add_range(features, OPP_ROW_LEN_OFFSET, 3, "opponent_row_len", "opponent_interaction")
    _add_range(features, OPP_RANK_COUNT_OFFSET, 3 * 13, "opponent_rank_count", "opponent_interaction")
    _add_range(features, OPP_SUIT_COUNT_OFFSET, 3 * 4, "opponent_suit_count", "opponent_interaction")
    _add_range(features, GLOBAL_OFFSET, GLOBAL_DIM, "global_state", "global_summary")
    _add_range(features, HU_DERIVED_OFFSET, HU_DERIVED_DIM, "hu_derived", "royalty_fl")
    _add_range(features, HU_MATCHUP_OFFSET, HU_MATCHUP_DIM, "hu_matchup", "opponent_interaction")

    if len(features) != HU_FEATURE_DIM:
        raise ValueError(f"manifest feature count mismatch: {len(features)} != {HU_FEATURE_DIM}")
    for expected_index, feature in enumerate(features):
        if feature["index"] != expected_index:
            raise ValueError(f"manifest order mismatch at {expected_index}: {feature['index']}")
    return {
        "version": FEATURE_MANIFEST_VERSION,
        "feature_schema_version": "hu_turn3_stage3_fast_v1",
        "feature_count": HU_FEATURE_DIM,
        "feature_dtype": str(np.dtype(np.float32)),
        "feature_column_names": [feature["name"] for feature in features],
        "features": features,
        "groups": _group_counts(features),
        "self_feature_dim": SELF_FEATURE_DIM,
        "checks": {
            "missing_columns": 0,
            "extra_columns": 0,
            "hgb_order": "f0..f1075",
        },
    }


def write_hu_turn3_stage3_feature_manifest(path: str | Path) -> dict[str, Any]:
    manifest = build_hu_turn3_stage3_feature_manifest()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = write_hu_turn3_stage3_feature_manifest(args.output)
    print(json.dumps({"output": str(args.output), "feature_count": manifest["feature_count"]}, indent=2))


def _add_range(
    features: list[dict[str, Any]],
    start: int,
    length: int,
    scalar_key: str,
    source_type: str,
) -> None:
    for local_index in range(length):
        index = start + local_index
        features.append(
            {
                "name": f"f{index}",
                "index": index,
                "feature_group": scalar_key,
                "dtype": "float32",
                "source_type": source_type,
                "scalar_feature_key": f"{scalar_key}.{local_index}",
                "direct_fill_rule": _fill_rule_for(source_type),
                "constant_default": 0.0,
            }
        )


def _fill_rule_for(source_type: str) -> str:
    return {
        "state_base": "state_common_vector",
        "action_encoding": "action_direct_indices",
        "after_board": "after_board_direct_indices_and_counts",
        "row_summary": "cached_row_summary",
        "global_summary": "direct_counts_and_terminal_summary",
        "royalty_fl": "cached_royalty_fl_summary",
        "outs_blocker": "dead_card_mask",
        "candidate_vs_reference_delta": "reserved",
        "opponent_interaction": "opponent_and_matchup_summary",
        "other": "zero_default",
    }.get(source_type, "zero_default")


def _group_counts(features: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for feature in features:
        group = str(feature["source_type"])
        counts[group] = counts.get(group, 0) + 1
    return counts


if __name__ == "__main__":
    main()
