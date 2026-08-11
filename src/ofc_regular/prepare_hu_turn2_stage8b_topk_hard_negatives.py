"""Prepare replay-ready hard negatives from HU T2 Stage8b TopK+MC logs.

This is a diagnostic/prep tool. It does not train a model, launch teacher
generation, or approve any runtime configuration.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .teacher import DEFAULT_FL_EV


DEFAULT_OUTPUT_DIR = Path("outputs/evals/hu_turn2_stage8b_topk_hard_negatives")
REALIZED_DELTA_METRIC_SOURCE = "realized_fired_whole_game_delta"
CONFIRM_DELTA_METRIC_ROLE = "gate_diagnostic_only"
CONFIRM_DELTA_PERFORMANCE_CLAIM_ALLOWED = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--decision-log",
        type=Path,
        action="append",
        required=True,
        help="Runtime decision JSONL from evaluate_hu_turn2_stage8b_topk_mc_rerank. Can be repeated.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--false-positive-threshold", type=float, default=0.0)
    parser.add_argument("--neutral-overconfirm-threshold", type=float, default=1.0)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def action_signature(action: dict[str, Any] | None) -> str:
    if not action:
        return ""
    placements = tuple(sorted((str(card), str(row)) for card, row in action.get("placements", ())))
    discards = tuple(sorted(str(card) for card in action.get("discards", ())))
    return json.dumps({"placements": placements, "discards": discards}, sort_keys=True)


def state_signature(row: dict[str, Any]) -> str:
    return json.dumps(
        {
            "hand_seed": row.get("hand_seed"),
            "seat": row.get("seat"),
            "hero_board": row.get("hero_board"),
            "opponent_board": row.get("opponent_board"),
            "cards_to_place": sorted(str(card) for card in row.get("cards_to_place", ())),
        },
        sort_keys=True,
    )


def has_replay_fields(row: dict[str, Any]) -> bool:
    required = (
        "hero_board",
        "opponent_board",
        "cards_to_place",
        "baseline_action",
        "final_action",
    )
    return all(row.get(key) for key in required) and bool(
        row.get("hero_private_discards") or row.get("visible_dead_cards")
    )


def replay_blocker(row: dict[str, Any]) -> str:
    missing = []
    for key in (
        "hero_board",
        "opponent_board",
        "cards_to_place",
        "baseline_action",
        "final_action",
    ):
        if not row.get(key):
            missing.append(key)
    if not row.get("hero_private_discards") and not row.get("visible_dead_cards"):
        missing.append("hero_visible_discard")
    return ";".join(missing)


def classify_row(row: dict[str, Any], *, false_positive_threshold: float, neutral_threshold: float) -> str:
    if not row.get("override_fired"):
        return "not_fired"
    if not row.get("realized_delta_valid"):
        return "realized_delta_missing"
    realized = safe_float(row.get("realized_candidate_seat_delta"))
    confirm = safe_float(row.get("rerank_delta"))
    predicted = safe_float(row.get("predicted_delta"))
    if realized < false_positive_threshold:
        if predicted < 0.0:
            return "false_positive_negative_model_delta"
        return "false_positive_confirm_gate"
    if realized == 0.0 and confirm >= neutral_threshold:
        return "neutral_overconfirm"
    if realized > 0.0:
        return "realized_positive"
    return "other"


def replay_payload(row: dict[str, Any], *, source_log: str, diagnosis: str) -> dict[str, Any]:
    risk_probability = row.get("local_ev_risk_probability", row.get("risk_probability"))
    return {
        "schema": "hu_turn2_stage8b_topk_hard_negative_v1",
        "source_log": source_log,
        "replay_origin_group": "topk_mc_false_positive" if diagnosis.startswith("false_positive") else diagnosis,
        "diagnosis": diagnosis,
        "replay_ready": has_replay_fields(row),
        "replay_blocker": replay_blocker(row),
        "hand_seed": row.get("hand_seed"),
        "hand_id": row.get("hand_id"),
        "game_id": row.get("game_id"),
        "seat": row.get("seat"),
        "seat_swap": row.get("seat_swap"),
        "config_id": row.get("config_id"),
        "state_signature": state_signature(row),
        "action_signature": action_signature(row.get("final_action")),
        "baseline_action_signature": action_signature(row.get("baseline_action")),
        "hero_board": row.get("hero_board"),
        "opponent_board": row.get("opponent_board"),
        "dead_cards": row.get("dead_cards"),
        "visible_dead_cards": row.get("visible_dead_cards"),
        "hero_private_discards": row.get("hero_private_discards"),
        "opponent_private_discards": row.get("opponent_private_discards"),
        "cards_to_place": row.get("cards_to_place"),
        "baseline_action": row.get("baseline_action"),
        "candidate_action": row.get("final_action"),
        "baseline_action_index": safe_int(row.get("baseline_action_index"), -1),
        "candidate_action_index": safe_int(row.get("final_action_index"), -1),
        "rerank_best_index": safe_int(row.get("rerank_best_index"), -1),
        "realized_delta": safe_float(row.get("realized_candidate_seat_delta")),
        "confirm_delta": safe_float(row.get("rerank_delta")),
        "confirm_delta_se": safe_float(row.get("rerank_delta_se")),
        "stage_a_delta": safe_float(row.get("stage_a_delta")),
        "predicted_delta": safe_float(row.get("predicted_delta")),
        "gate_probability": safe_float(row.get("gate_probability")),
        "candidate_ev_rank": safe_int(row.get("candidate_ev_rank"), -1),
        "risk_probability": risk_probability,
        "local_ev_risk_probability": risk_probability,
        "local_ev_risk_enabled": row.get("local_ev_risk_enabled"),
        "local_ev_risk_threshold": row.get("local_ev_risk_threshold"),
        "local_ev_risk_rank_min": row.get("local_ev_risk_rank_min"),
        "local_ev_risk_rank_max": row.get("local_ev_risk_rank_max"),
        "local_ev_risk_rank_guard_passed": row.get("local_ev_risk_rank_guard_passed"),
        "local_ev_risk_vetoed": row.get("local_ev_risk_vetoed"),
        "confirm_count": safe_int(row.get("confirm_delta_count"), 0),
        "hard_negative_label": 1 if diagnosis.startswith("false_positive") else 0,
        "safe_lcb196_label": "negative" if diagnosis.startswith("false_positive") else "gray",
        # Names the FL EV these rows must be relabelled under.  Derived, not
        # written out: a stale literal here would send a reader to relabel
        # against a superseded constant.
        "training_note": (
            "replay or feature-generate this state/action under FL EV "
            f"{DEFAULT_FL_EV[14]} before using as a supervised label"
        ),
    }


def extract_rows(
    logs: Iterable[tuple[str, list[dict[str, Any]]]],
    *,
    false_positive_threshold: float = 0.0,
    neutral_overconfirm_threshold: float = 1.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    false_positives: list[dict[str, Any]] = []
    neutral_overconfirm: list[dict[str, Any]] = []
    all_fired: list[dict[str, Any]] = []
    for source_log, rows in logs:
        for row in rows:
            diagnosis = classify_row(
                row,
                false_positive_threshold=false_positive_threshold,
                neutral_threshold=neutral_overconfirm_threshold,
            )
            if not row.get("override_fired"):
                continue
            payload = replay_payload(row, source_log=source_log, diagnosis=diagnosis)
            all_fired.append(payload)
            if diagnosis.startswith("false_positive"):
                false_positives.append(payload)
            elif diagnosis == "neutral_overconfirm":
                neutral_overconfirm.append(payload)
    return (
        dedupe_replay_rows(false_positives),
        dedupe_replay_rows(neutral_overconfirm),
        dedupe_replay_rows(all_fired),
    )


def dedupe_replay_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("state_signature", "")), str(row.get("action_signature", "")))
        current = by_key.get(key)
        if current is None or safe_float(row.get("realized_delta")) < safe_float(current.get("realized_delta")):
            by_key[key] = row
    return sorted(
        by_key.values(),
        key=lambda item: (
            safe_float(item.get("realized_delta")),
            -safe_float(item.get("confirm_delta")),
            str(item.get("hand_seed")),
        ),
    )


def summary_rows(false_positives: list[dict[str, Any]], neutral: list[dict[str, Any]], all_fired: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for label, rows in (("false_positive", false_positives), ("neutral_overconfirm", neutral), ("all_fired_deduped", all_fired)):
        values = [safe_float(row.get("realized_delta")) for row in rows]
        confirms = [safe_float(row.get("confirm_delta")) for row in rows]
        output.append(
            {
                "group": label,
                "rows": len(rows),
                "replay_ready_rows": sum(1 for row in rows if row.get("replay_ready")),
                "realized_mean": sum(values) / len(values) if values else 0.0,
                "realized_sum": sum(values),
                "confirm_mean": sum(confirms) / len(confirms) if confirms else 0.0,
                "realized_delta_metric_source": REALIZED_DELTA_METRIC_SOURCE,
                "confirm_delta_metric_role": CONFIRM_DELTA_METRIC_ROLE,
                "confirm_delta_performance_claim_allowed": CONFIRM_DELTA_PERFORMANCE_CLAIM_ALLOWED,
                "diagnosis_counts": json.dumps(Counter(str(row.get("diagnosis")) for row in rows), sort_keys=True),
            }
        )
    return output


def write_summary_markdown(path: Path, summary: list[dict[str, Any]]) -> None:
    lines = [
        "# HU T2 Stage8b TopK Hard Negative Pack",
        "",
        "This is a replay/training-prep artifact only. It does not approve 50k teacher generation, T1, P2 fixed status, or production runtime.",
        "`confirm_mean` is a gate diagnostic only; use `realized_mean` for fired whole-game loss evidence.",
        "",
        "| group | rows | replay-ready | realized mean | confirm mean | diagnoses |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in summary:
        lines.append(
            "| {group} | {rows} | {ready} | {realized:.4f} | {confirm:.4f} | `{diagnoses}` |".format(
                group=row["group"],
                rows=row["rows"],
                ready=row["replay_ready_rows"],
                realized=safe_float(row["realized_mean"]),
                confirm=safe_float(row["confirm_mean"]),
                diagnoses=row["diagnosis_counts"],
            )
        )
    lines.extend(
        [
            "",
            "## Recommended Use",
            "",
            "- Use `topk_false_positive_hard_negatives.jsonl` as hard-negative replay input for the next T2 model/gate iteration.",
            "- Do not train directly from confirm MC deltas. Regenerate or replay labels under the current FL EV objective first.",
            "- Current evidence still says T2 Stage8b TopK+MC is No-Go for production, T1, and 50k teacher.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logs = [(str(path), read_jsonl(path)) for path in args.decision_log]
    false_positives, neutral, all_fired = extract_rows(
        logs,
        false_positive_threshold=args.false_positive_threshold,
        neutral_overconfirm_threshold=args.neutral_overconfirm_threshold,
    )
    summary = summary_rows(false_positives, neutral, all_fired)
    write_jsonl(args.output_dir / "topk_false_positive_hard_negatives.jsonl", false_positives)
    write_jsonl(args.output_dir / "topk_neutral_overconfirm_candidates.jsonl", neutral)
    write_jsonl(args.output_dir / "topk_all_fired_deduped.jsonl", all_fired)
    write_csv(args.output_dir / "topk_hard_negative_summary.csv", summary)
    write_summary_markdown(args.output_dir / "topk_hard_negative_summary.md", summary)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "false_positive_hard_negatives": len(false_positives),
                "neutral_overconfirm": len(neutral),
                "all_fired_deduped": len(all_fired),
                "production": "No-Go",
                "fifty_k_teacher": "No-Go",
                "t1": "No-Go",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
