"""Normalize recursive tutor JSONL into a stable teacher-data schema."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _board_from_candidate(candidate: Dict[str, Any]) -> Dict[str, List[str]]:
    board = candidate.get("board") or candidate.get("next_board") or {}
    return {
        "top": list(board.get("top", []) or []),
        "middle": list(board.get("middle", []) or board.get("mid", []) or []),
        "bottom": list(board.get("bottom", []) or board.get("bot", []) or []),
    }


def _metrics_from_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    recursive = candidate.get("recursive") or {}
    mc = recursive.get("mc") or {}
    components = candidate.get("target_components") or {}
    fl_types = mc.get("fl_type_rates") or {}
    return {
        "ev": float(components.get("ev", mc.get("avg_score", candidate.get("ev", 0.0)) or 0.0)),
        "score": float(candidate.get("target_score", components.get("ev", mc.get("avg_score", 0.0)) or 0.0)),
        "fl_rate": float(components.get("fl_any", mc.get("fl_rate", candidate.get("fl_rate", 0.0)) or 0.0)),
        "fl_type_rates": {
            "qq": float(components.get("fl_qq", fl_types.get("qq", 0.0)) or 0.0),
            "kk": float(components.get("fl_kk", fl_types.get("kk", 0.0)) or 0.0),
            "aa": float(components.get("fl_aa", fl_types.get("aa", 0.0)) or 0.0),
            "trips": float(components.get("fl_trips", fl_types.get("trips", 0.0)) or 0.0),
        },
        "bust_rate": float(components.get("bust", mc.get("bust_rate", candidate.get("bust_prob", 0.0)) or 0.0)),
        "sims": int(recursive.get("simulations", mc.get("simulations", 0)) or 0),
        "source": "estimated",
    }


def _action_from_candidate(candidate: Dict[str, Any]) -> Dict[str, Any]:
    placements = candidate.get("placements", [])
    if not placements:
        board = _board_from_candidate(candidate)
        placements = [
            [card, row]
            for row in ("top", "middle", "bottom")
            for card in board[row]
        ]
    return {
        "placements": placements,
        "discard": candidate.get("discard"),
        "label": candidate.get("action"),
    }


def normalize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    candidates = record.get("candidates", []) or []
    best = record.get("best") or (candidates[0] if candidates else {})
    return {
        "schema_version": 1,
        "source": "route10_recursive_teacher",
        "hand_index": record.get("hand_index"),
        "seat": record.get("seat"),
        "position": record.get("position"),
        "turn": 0,
        "board": {"top": [], "middle": [], "bottom": []},
        "opponent_board": record.get("opponent_board") or {},
        "dealt": record.get("dealt", []),
        "legal_actions": len(candidates),
        "chosen_action": _action_from_candidate(best),
        "chosen_board": _board_from_candidate(best),
        "metrics": _metrics_from_candidate(best),
        "settings": {
            "sims": record.get("sims"),
            "beam": record.get("beam"),
            "child_sims": record.get("child_sims"),
            "pool_size": record.get("pool_size"),
        },
        "candidate_metrics": [
            {
                "rank": idx,
                "action": _action_from_candidate(candidate),
                "board": _board_from_candidate(candidate),
                "metrics": _metrics_from_candidate(candidate),
            }
            for idx, candidate in enumerate(candidates, start=1)
        ],
    }


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def normalize_file(input_path: Path, output_path: Path) -> Dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    candidate_count = 0
    with output_path.open("w", encoding="utf-8") as out:
        for record in iter_jsonl(input_path):
            normalized = normalize_record(record)
            out.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            count += 1
            candidate_count += normalized["legal_actions"]
    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "records": count,
        "candidate_metrics": candidate_count,
        "schema_version": 1,
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize recursive tutor JSONL")
    parser.add_argument("input")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    summary = normalize_file(Path(args.input), Path(args.output))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
