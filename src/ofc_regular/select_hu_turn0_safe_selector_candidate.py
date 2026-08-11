"""Select one preregistered T0 safe-selector candidate from OOF MC metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, Any], key: str) -> float:
    return float(row.get(key, 0.0))


def _int(row: dict[str, Any], key: str) -> int:
    return int(float(row.get(key, 0)))


def select_candidate(
    candidates: list[dict[str, Any]],
    *,
    min_fires: int,
    min_safe_positives: int,
    max_hard_negative_rate: float,
    min_mc_delta_ci95_low: float,
    max_p95_mc_loss: float,
) -> dict[str, Any]:
    evaluated: list[dict[str, Any]] = []
    for candidate in candidates:
        rows = _read_csv(Path(candidate["threshold_sweep"]))
        best_model = str(candidate["best_model"])
        for row in rows:
            if str(row.get("model")) != best_model:
                continue
            checks = {
                "minimum_fires": _int(row, "fires") >= min_fires,
                "minimum_safe_positives": _int(row, "safe_positive_count")
                >= min_safe_positives,
                "maximum_hard_negative_rate": _float(row, "hard_negative_rate")
                <= max_hard_negative_rate,
                "positive_mc_delta_lcb": _float(row, "mc_delta_ci95_low")
                >= min_mc_delta_ci95_low,
                "maximum_p95_mc_loss": _float(row, "p95_mc_loss") <= max_p95_mc_loss,
            }
            evaluated.append(
                {
                    **candidate,
                    **row,
                    "model": candidate["model"],
                    "estimator_name": row.get("model"),
                    "eligible": all(checks.values()),
                    "checks": checks,
                }
            )
    eligible = [row for row in evaluated if row["eligible"]]
    eligible.sort(
        key=lambda row: (
            _float(row, "mc_delta_ci95_low"),
            _int(row, "safe_positive_count"),
            _float(row, "precision_safe_positive"),
            float(row.get("best_oof_average_precision", 0.0)),
            -_float(row, "threshold"),
            str(row.get("run")),
        ),
        reverse=True,
    )
    selected = eligible[0] if eligible else None
    return {
        "schema": "hu_turn0_stage19_safe_selector_selection_v1",
        "decision": "Go" if selected is not None else "No-Go",
        "selected": selected,
        "eligible_count": len(eligible),
        "evaluated_count": len(evaluated),
        "constraints": {
            "min_fires": min_fires,
            "min_safe_positives": min_safe_positives,
            "max_hard_negative_rate": max_hard_negative_rate,
            "min_mc_delta_ci95_low": min_mc_delta_ci95_low,
            "max_p95_mc_loss": max_p95_mc_loss,
        },
        "runtime_status": "preregistered_for_fresh_whole_game_holdout"
        if selected is not None
        else "selector_not_supported_by_oof_mc_labels",
        "evaluated": evaluated,
    }


def _write_grid(path: Path, rows: list[dict[str, Any]]) -> None:
    flattened = []
    for row in rows:
        flattened.append(
            {
                key: value
                for key, value in row.items()
                if key not in {"checks"}
            }
            | {
                f"check_{key}": value
                for key, value in (row.get("checks") or {}).items()
            }
        )
    if not flattened:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-fires", type=int, default=10)
    parser.add_argument("--min-safe-positives", type=int, default=5)
    parser.add_argument("--max-hard-negative-rate", type=float, default=0.10)
    parser.add_argument("--min-mc-delta-ci95-low", type=float, default=0.0)
    parser.add_argument("--max-p95-mc-loss", type=float, default=10.0)
    args = parser.parse_args()
    candidates = json.loads(args.training_candidates.read_text(encoding="utf-8-sig"))
    if not isinstance(candidates, list):
        raise SystemExit("--training-candidates must contain a JSON list")
    result = select_candidate(
        candidates,
        min_fires=args.min_fires,
        min_safe_positives=args.min_safe_positives,
        max_hard_negative_rate=args.max_hard_negative_rate,
        min_mc_delta_ci95_low=args.min_mc_delta_ci95_low,
        max_p95_mc_loss=args.max_p95_mc_loss,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "selection.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_grid(args.output_dir / "candidate_grid.csv", result["evaluated"])
    print(json.dumps({key: value for key, value in result.items() if key != "evaluated"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
