"""Prepare leak-free mixed-MC datasets for HU T0 candidate training."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


MC1_SOURCE = "hu_turn0_terminal_rollout_mc1"
MC4_SOURCE = "hu_turn0_terminal_rollout_mc4"
ROWS = ("top", "middle", "bottom")


def _normalized_board(value: Any) -> dict[str, list[str]]:
    board = value if isinstance(value, dict) else {}
    return {row: sorted(str(card) for card in board.get(row, ())) for row in ROWS}


def canonical_t0_state_key(sample: dict[str, Any]) -> str:
    """Hash only observable decision-state fields, independent of action order."""
    payload = {
        "phase": sample.get("phase"),
        "seat": sample.get("seat"),
        "player": sample.get("player"),
        "board": _normalized_board(sample.get("board")),
        "opponent_board": _normalized_board(sample.get("opponent_board")),
        "dead_cards": sorted(str(card) for card in sample.get("dead_cards", ())),
        "dealt": sorted(str(card) for card in sample.get("dealt", ())),
        "to_act_order": sample.get("to_act_order"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def _read_rows(path: Path, *, source: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            sample = json.loads(line)
            if not sample.get("actions"):
                raise ValueError(f"{path}:{line_number} has no legal actions")
            if not str(sample.get("phase", "")).startswith("hu_turn0"):
                raise ValueError(f"{path}:{line_number} is not a HU T0 sample")
            original_source = sample.get("source") or sample.get("label_source")
            sample["teacher_source_original"] = original_source
            sample["source"] = source
            sample["teacher_mc_samples"] = 4 if source == MC4_SOURCE else 1
            sample["dataset_state_id"] = canonical_t0_state_key(sample)
            rows.append(sample)
    return rows


def _deduplicate(samples: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for sample in samples:
        key = str(sample["dataset_state_id"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(sample)
    return unique, len(samples) - len(unique)


def _seat_stratified_holdout(
    samples: Sequence[dict[str, Any]],
    *,
    fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 <= fraction < 1.0:
        raise ValueError("holdout fraction must be in [0, 1)")
    train: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    by_seat: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        by_seat.setdefault(str(sample.get("seat", "unknown")), []).append(sample)
    for seat in sorted(by_seat):
        group = list(by_seat[seat])
        seat_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{seat}".encode("ascii")).digest()[:8],
            "big",
        )
        random.Random(seat_seed).shuffle(group)
        count = int(round(len(group) * fraction))
        if fraction > 0.0 and len(group) > 1:
            count = max(1, min(len(group) - 1, count))
        holdout.extend(group[:count])
        train.extend(group[count:])
    return train, holdout


def _write_jsonl(path: Path, samples: Iterable[dict[str, Any]], *, split: str) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            row = dict(sample)
            row["dataset_split"] = split
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def _counts(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(samples),
        "sources": dict(sorted(Counter(str(row["source"]) for row in samples).items())),
        "seats": dict(sorted(Counter(str(row.get("seat", "unknown")) for row in samples).items())),
    }


def prepare_hu_turn0_candidate_dataset(
    *,
    mc1_input: Path,
    mc4_input: Path,
    output_dir: Path,
    mc1_holdout_fraction: float = 0.10,
    mc4_holdout_fraction: float = 0.20,
    seed: int = 2026103101,
) -> dict[str, Any]:
    mc1_rows, duplicate_mc1_internal = _deduplicate(
        _read_rows(mc1_input, source=MC1_SOURCE)
    )
    mc4_rows, duplicate_mc4_internal = _deduplicate(
        _read_rows(mc4_input, source=MC4_SOURCE)
    )
    if not mc1_rows or not mc4_rows:
        raise ValueError("both MC1 and MC4 inputs must contain T0 samples")

    # Prefer the higher-MC label if both inputs accidentally contain the same state.
    mc4_keys = {str(row["dataset_state_id"]) for row in mc4_rows}
    duplicate_mc1 = [row for row in mc1_rows if str(row["dataset_state_id"]) in mc4_keys]
    mc1_rows = [row for row in mc1_rows if str(row["dataset_state_id"]) not in mc4_keys]

    mc1_train, mc1_holdout = _seat_stratified_holdout(
        mc1_rows,
        fraction=mc1_holdout_fraction,
        seed=seed + 1,
    )
    mc4_train, mc4_holdout = _seat_stratified_holdout(
        mc4_rows,
        fraction=mc4_holdout_fraction,
        seed=seed + 4,
    )
    train = mc1_train + mc4_train
    random.Random(seed).shuffle(train)

    split_sets = {
        "train": {str(row["dataset_state_id"]) for row in train},
        "holdout_mc1": {str(row["dataset_state_id"]) for row in mc1_holdout},
        "holdout_mc4": {str(row["dataset_state_id"]) for row in mc4_holdout},
    }
    overlap = {
        "train_vs_mc1": len(split_sets["train"] & split_sets["holdout_mc1"]),
        "train_vs_mc4": len(split_sets["train"] & split_sets["holdout_mc4"]),
        "mc1_vs_mc4": len(split_sets["holdout_mc1"] & split_sets["holdout_mc4"]),
    }
    if any(overlap.values()):
        raise RuntimeError(f"state leakage across prepared splits: {overlap}")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_mixed_mc1_mc4.jsonl"
    mc1_holdout_path = output_dir / "holdout_mc1.jsonl"
    mc4_holdout_path = output_dir / "holdout_mc4.jsonl"
    _write_jsonl(train_path, train, split="train")
    _write_jsonl(mc1_holdout_path, mc1_holdout, split="holdout_mc1")
    _write_jsonl(mc4_holdout_path, mc4_holdout, split="holdout_mc4")

    summary = {
        "schema": "hu_turn0_candidate_dataset_v1",
        "seed": int(seed),
        "inputs": {"mc1": str(mc1_input), "mc4": str(mc4_input)},
        "source_labels": {"mc1": MC1_SOURCE, "mc4": MC4_SOURCE},
        "holdout_fractions": {
            "mc1": float(mc1_holdout_fraction),
            "mc4": float(mc4_holdout_fraction),
        },
        "input_counts": {"mc1": len(mc1_rows), "mc4": len(mc4_rows)},
        "duplicate_mc1_states_dropped": len(duplicate_mc1),
        "duplicate_within_source_dropped": {
            "mc1": duplicate_mc1_internal,
            "mc4": duplicate_mc4_internal,
        },
        "train": _counts(train),
        "holdout_mc1": _counts(mc1_holdout),
        "holdout_mc4": _counts(mc4_holdout),
        "split_overlap": overlap,
        "paths": {
            "train": str(train_path),
            "holdout_mc1": str(mc1_holdout_path),
            "holdout_mc4": str(mc4_holdout_path),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mc1-input", type=Path, required=True)
    parser.add_argument("--mc4-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mc1-holdout", type=float, default=0.10)
    parser.add_argument("--mc4-holdout", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=2026103101)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    summary = prepare_hu_turn0_candidate_dataset(
        mc1_input=args.mc1_input,
        mc4_input=args.mc4_input,
        output_dir=args.output_dir,
        mc1_holdout_fraction=args.mc1_holdout,
        mc4_holdout_fraction=args.mc4_holdout,
        seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
