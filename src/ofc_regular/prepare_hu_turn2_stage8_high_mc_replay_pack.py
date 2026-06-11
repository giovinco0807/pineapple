"""Prepare a slim HU Turn2 Stage8 high-MC replay pack.

The Stage8 20k teacher file is several GB. GCP C4 replay only needs the
selected states plus their original teacher samples, so this utility extracts a
small replay-ready pack from an existing selected_teacher_states_deduped.jsonl.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .audit_hu_turn2_stage8_high_mc import (
    load_cache_metadata,
    load_selected_states_jsonl,
    load_teacher_rows_for_state_indices,
    write_jsonl,
)

DEFAULT_SELECTED = Path("outputs/evals/hu_turn2_stage8_c4_selected_high_mc_audit_mc4096/selected_teacher_states_deduped.jsonl")
DEFAULT_CACHE_DIR = Path("D:/ofc-pineapple-storage/regular-ofc-pineapple/feature_cache/hu_t2_stage8_20k_mc512")
DEFAULT_OUTPUT_DIR = Path("outputs/evals/hu_turn2_stage8_c4_high_mc_replay_pack")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-states-jsonl", type=Path, default=DEFAULT_SELECTED)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--replay-offset", type=int, default=0)
    parser.add_argument("--replay-count", type=int, default=50, help="0 means all states after offset.")
    return parser.parse_args()


def selected_slice(rows: list[dict[str, Any]], *, offset: int, count: int) -> list[dict[str, Any]]:
    if offset < 0:
        raise ValueError("replay offset must be non-negative")
    if count < 0:
        raise ValueError("replay count must be non-negative")
    return rows[offset:] if count == 0 else rows[offset : offset + count]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = load_selected_states_jsonl(args.selected_states_jsonl)
    replay_rows = selected_slice(selected, offset=args.replay_offset, count=args.replay_count)
    state_indices = {int(row["state_index"]) for row in replay_rows}
    metadata = load_cache_metadata(args.cache_dir)
    teacher_rows = load_teacher_rows_for_state_indices(metadata=metadata, repo_root=Path.cwd(), state_indices=state_indices)
    missing = sorted(state_indices.difference(teacher_rows))
    if missing:
        raise RuntimeError(f"teacher rows missing for selected states: {missing[:20]}")

    selected_path = args.output_dir / "selected_states.jsonl"
    samples_path = args.output_dir / "teacher_samples.jsonl"
    write_jsonl(selected_path, replay_rows)
    write_jsonl(samples_path, ({"state_index": state_index, "sample": teacher_rows[state_index]} for state_index in sorted(state_indices)))

    summary = {
        "selected_input": str(args.selected_states_jsonl),
        "cache_dir": str(args.cache_dir),
        "output_dir": str(args.output_dir),
        "replay_offset": args.replay_offset,
        "replay_count": args.replay_count,
        "selected_input_rows": len(selected),
        "replay_rows": len(replay_rows),
        "teacher_sample_rows": len(teacher_rows),
        "missing_teacher_rows": len(missing),
        "selected_states_jsonl": str(selected_path),
        "teacher_samples_jsonl": str(samples_path),
    }
    (args.output_dir / "replay_pack_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "replay_pack_summary.md").write_text(
        "\n".join(
            [
                "# HU Turn2 Stage8 C4 High-MC Replay Pack",
                "",
                "- scope: `analysis only`",
                "- production: `No-Go`",
                "- 50k teacher: `No-Go`",
                "- T1 training: `No-Go`",
                f"- selected input rows: `{len(selected)}`",
                f"- replay rows: `{len(replay_rows)}`",
                f"- teacher sample rows: `{len(teacher_rows)}`",
                f"- missing teacher rows: `{len(missing)}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
