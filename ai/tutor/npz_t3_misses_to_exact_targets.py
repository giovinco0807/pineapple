"""Convert T3 NPZ shortlist misses back to exact-teacher target JSONL.

The T3 oracle NPZ data stores pre-action decision states, while shortlist audit
misses reference converted reranker ``group_id`` values.  The converter that
created the reranker data processes NPZ rows in sorted file order and assigns
one sequential group id per decision.  This helper reverses that mapping for a
selected miss JSONL so hard cases can be regenerated with full exact labels,
including FL and bust targets.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.engine.encoding import Board
from ai.training.convert_t3_oracle_npz_to_reranker import decode_observation
from ai.tutor.exact_late import board_card_count, board_to_dict


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def npz_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No NPZ files found under {input_dir}")
    return files


def miss_rows(path: Path, max_records: int, min_pool_regret: float, sort_by_regret: bool) -> list[dict[str, Any]]:
    rows = []
    seen: set[int] = set()
    for row in iter_jsonl(path):
        group_id = row.get("group_id")
        if group_id is None:
            continue
        group_id_int = int(group_id)
        if group_id_int in seen:
            continue
        regret = float(row.get("hybrid_pool_regret", row.get("regret", 0.0)) or 0.0)
        if regret < min_pool_regret:
            continue
        row["_pool_regret"] = regret
        rows.append(row)
        seen.add(group_id_int)
    if sort_by_regret:
        rows.sort(key=lambda item: float(item.get("_pool_regret", 0.0)), reverse=True)
    else:
        rows.sort(key=lambda item: int(item["group_id"]))
    if max_records > 0:
        rows = rows[:max_records]
    return rows


def all_cards(board: Board) -> list[str]:
    return list(board.top) + list(board.middle) + list(board.bottom)


def target_from_state(
    raw_state: np.ndarray,
    group_id: int,
    npz_file: Path,
    npz_row: int,
    miss: dict[str, Any],
    decode_threshold: float,
) -> dict[str, Any]:
    obs = decode_observation(raw_state, threshold=decode_threshold)
    board = obs.board_self
    opponent = obs.board_opponent
    turn = int(obs.turn)
    if turn != 3:
        raise ValueError(f"group_id {group_id} decoded turn {turn}, expected 3")
    if board_card_count(board) != 9:
        raise ValueError(f"group_id {group_id} decoded {board_card_count(board)} hero cards, expected 9")
    if len(obs.dealt_cards) != 3:
        raise ValueError(f"group_id {group_id} decoded {len(obs.dealt_cards)} dealt cards, expected 3")

    return {
        "source": str(npz_file),
        "source_line": int(npz_row + 1),
        "turn": 3,
        "board": board_to_dict(board),
        "opponent_board": board_to_dict(opponent),
        "dealt": list(obs.dealt_cards),
        "known_discards": list(obs.known_discards_self),
        "exclude": list(obs.known_discards_self),
        "is_btn": bool(obs.is_btn),
        "position": "btn" if obs.is_btn else "bb",
        "reasons": [
            "t3_npz_pool_miss",
            "exact_relabel",
        ],
        "npz_group_id": int(group_id),
        "npz_file": str(npz_file),
        "npz_row": int(npz_row),
        "pool_miss_regret": float(miss.get("_pool_regret", 0.0)),
        "teacher_action_idx": int((miss.get("teacher_best") or {}).get("action_idx", -1)),
        "teacher_model_rank": int((miss.get("teacher_best") or {}).get("model_rank", -1)),
        "model_top1_action_idx": int((miss.get("model_top1") or {}).get("action_idx", -1)),
        "model_top1_teacher_score": float((miss.get("model_top1") or {}).get("teacher_score", 0.0) or 0.0),
        "oracle_group_key": json.dumps(
            {
                "source": str(npz_file),
                "group_id": int(group_id),
                "row": int(npz_row),
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        "used_cards_count": len(set(all_cards(board) + all_cards(opponent) + list(obs.dealt_cards) + list(obs.known_discards_self))),
    }


def build_targets(args: argparse.Namespace) -> dict[str, Any]:
    input_dir = Path(args.npz_input)
    misses = miss_rows(Path(args.misses), args.max_records, args.min_pool_regret, args.sort_by_regret)
    wanted = {int(row["group_id"]): row for row in misses}
    if not wanted:
        raise SystemExit("No miss rows selected")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    targets: list[dict[str, Any]] = []
    unresolved = set(wanted)
    base_group = 0
    for path in npz_files(input_dir):
        if not unresolved:
            break
        shard = np.load(path, mmap_mode="r")
        states = shard["states"]
        n_rows = int(states.shape[0])
        in_file = [gid for gid in sorted(unresolved) if base_group <= gid < base_group + n_rows]
        for group_id in in_file:
            npz_row = group_id - base_group
            target = target_from_state(
                states[npz_row],
                group_id,
                path,
                npz_row,
                wanted[group_id],
                args.decode_threshold,
            )
            targets.append(target)
            unresolved.remove(group_id)
        base_group += n_rows

    targets.sort(key=lambda item: int(item["npz_group_id"]))
    with output.open("w", encoding="utf-8") as handle:
        for target in targets:
            handle.write(json.dumps(target, ensure_ascii=False) + "\n")

    summary = {
        "npz_input": str(input_dir),
        "misses": str(args.misses),
        "output": str(output),
        "selected_misses": len(misses),
        "written": len(targets),
        "unresolved": sorted(int(x) for x in unresolved),
        "max_records": int(args.max_records),
        "min_pool_regret": float(args.min_pool_regret),
        "sort_by_regret": bool(args.sort_by_regret),
        "decode_threshold": float(args.decode_threshold),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert T3 NPZ pool misses to exact target JSONL")
    parser.add_argument("--npz-input", required=True)
    parser.add_argument("--misses", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--min-pool-regret", type=float, default=0.0)
    parser.add_argument("--sort-by-regret", action="store_true")
    parser.add_argument("--decode-threshold", type=float, default=0.5)
    args = parser.parse_args(list(argv) if argv is not None else None)
    print(json.dumps(build_targets(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
