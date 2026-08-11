"""Prepare HU T2 Stage8c TopK+confirm distillation rows.

This creates a lightweight-runtime training artifact from realized
TopK+confirm decision logs. Confirm MC values are copied as runtime features
only; the supervised label is based on the realized fired-hand result or on a
candidate being rejected by the confirm gate.

The artifact is not a production approval. It is a bridge from the validated
offline/search policy to a trainable runtime gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_OUTPUT_DIR = Path("outputs/training/hu_turn2_stage8c_topk_confirm_distillation")
OUTPUT_NAME = "topk_confirm_distillation_rows.jsonl"
PRIMARY_LABEL_SOURCE = "realized_fired_delta_or_confirm_gate_rejection"
CONFIRM_DELTA_METRIC_ROLE = "runtime_feature_and_gate_diagnostic_only"
CONFIRM_DELTA_PERFORMANCE_CLAIM_ALLOWED = False
POSITIVE_USE = "topk_confirm_realized_positive"
LOSS_USE = "topk_confirm_realized_loss"
REJECTED_USE = "topk_confirm_rejected"
FIRE_SELECTOR_REJECTED_USE = "topk_confirm_fire_selector_rejected"
TOPK_EMPTY_USE = "topk_confirm_topk_empty"
TRAINING_USES = (POSITIVE_USE, LOSS_USE, REJECTED_USE, FIRE_SELECTOR_REJECTED_USE, TOPK_EMPTY_USE)
REJECTED_REASONS = {"below_confirm_delta", "below_confirm_se"}
FIRE_SELECTOR_REJECTED_REASONS = {"below_fire_selector_threshold"}


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
    parser.add_argument(
        "--positive-threshold",
        type=float,
        default=0.0,
        help="Realized fired-hand delta must be greater than this to be a positive label.",
    )
    parser.add_argument(
        "--max-rejected-negatives",
        type=int,
        default=5000,
        help="Maximum confirm-rejected negative rows to keep after deterministic sampling.",
    )
    parser.add_argument(
        "--max-fire-selector-negatives",
        type=int,
        default=5000,
        help="Maximum fire-selector rejected negative rows to keep after deterministic sampling.",
    )
    parser.add_argument(
        "--max-topk-empty-negatives",
        type=int,
        default=3000,
        help="Maximum topk_empty negative rows to keep after deterministic sampling.",
    )
    parser.add_argument("--sample-seed", type=int, default=2026064519)
    parser.add_argument(
        "--include-topk-empty",
        action="store_true",
        help="Also use Stage8b top1 candidates from topk_empty rows as easy negatives.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def present(value: Any) -> bool:
    return value not in (None, "", [], {})


def action_signature(action: dict[str, Any] | None) -> str:
    if not action:
        return ""
    placements = tuple(sorted((str(card), str(row)) for card, row in action.get("placements", ())))
    discards = tuple(sorted(str(card) for card in action.get("discards", ())))
    return json.dumps({"placements": placements, "discards": discards}, sort_keys=True)


def candidate_from_row(row: dict[str, Any], *, include_topk_empty: bool) -> tuple[dict[str, Any] | None, int, str]:
    if truthy(row.get("override_fired")):
        return row.get("final_action") or row.get("rerank_best_action"), safe_int(row.get("final_action_index"), -1), "fired"
    reason = str(row.get("no_override_reason") or "")
    if reason in REJECTED_REASONS | FIRE_SELECTOR_REJECTED_REASONS and row.get("rerank_best_action"):
        return row.get("rerank_best_action"), safe_int(row.get("rerank_best_index"), -1), reason
    if reason in FIRE_SELECTOR_REJECTED_REASONS:
        candidates = [item for item in row.get("stage8c_fire_selector_candidates") or [] if item.get("action")]
        if candidates:
            best = max(candidates, key=lambda item: safe_float(item.get("probability")))
            return best.get("action"), safe_int(best.get("action_index"), -1), reason
    if reason == "topk_empty" and include_topk_empty and row.get("stage8b_top1_action"):
        return row.get("stage8b_top1_action"), safe_int(row.get("stage8b_top1_index"), -1), reason
    return None, -1, reason


def missing_training_fields(row: dict[str, Any]) -> list[str]:
    required = (
        "hero_board",
        "opponent_board",
        "dead_cards",
        "cards_to_place",
        "baseline_action",
        "candidate_action",
    )
    return [field for field in required if not present(row.get(field))]


def training_ready(row: dict[str, Any]) -> bool:
    return not missing_training_fields(row)


def label_for_row(row: dict[str, Any], *, positive_threshold: float) -> tuple[int, str] | None:
    if truthy(row.get("override_fired")) and truthy(row.get("realized_delta_valid")):
        realized = safe_float(row.get("realized_candidate_seat_delta"))
        if realized > positive_threshold:
            return 1, POSITIVE_USE
        return 0, LOSS_USE
    reason = str(row.get("no_override_reason") or "")
    if reason in REJECTED_REASONS:
        return 0, REJECTED_USE
    if reason in FIRE_SELECTOR_REJECTED_REASONS:
        return 0, FIRE_SELECTOR_REJECTED_USE
    if reason == "topk_empty":
        return 0, TOPK_EMPTY_USE
    return None


def source_state_key(row: dict[str, Any]) -> str:
    return json.dumps(
        {
            "hand_seed": row.get("hand_seed"),
            "seat": row.get("seat"),
            "seat_swap": row.get("seat_swap"),
            "hero_board": row.get("hero_board"),
            "opponent_board": row.get("opponent_board"),
            "cards_to_place": sorted(str(card) for card in row.get("cards_to_place", ())),
        },
        sort_keys=True,
    )


def distillation_payload(
    row: dict[str, Any],
    *,
    source_log: str,
    label: int,
    training_use: str,
    candidate_action: dict[str, Any],
    candidate_index: int,
    candidate_source: str,
) -> dict[str, Any]:
    stage8c_candidate = next(
        (
            item
            for item in row.get("stage8c_fire_selector_candidates") or []
            if safe_int(item.get("action_index"), -9999) == safe_int(candidate_index, -1)
        ),
        {},
    )
    realized = safe_float(row.get("realized_candidate_seat_delta"))
    confirm_delta = row.get("confirm_delta")
    if confirm_delta in (None, ""):
        confirm_delta = row.get("rerank_delta")
    confirm_delta_se = row.get("confirm_delta_se")
    if confirm_delta_se in (None, ""):
        confirm_delta_se = row.get("rerank_delta_se")
    payload = {
        "schema": "hu_turn2_stage8c_topk_confirm_distillation_v1",
        "source_log": source_log,
        "config_id": row.get("config_id", ""),
        "hand_seed": row.get("hand_seed"),
        "hand_id": row.get("hand_id"),
        "game_id": row.get("game_id"),
        "seed": row.get("seed"),
        "seat": row.get("seat"),
        "seat_swap": row.get("seat_swap"),
        "street": row.get("street", "T2"),
        "turn": row.get("turn", "T2"),
        "recommended_training_use": training_use,
        "topk_distill_label_id": int(label),
        "use_for_topk_confirm_fire_head": 1,
        "risk_target_group": training_use,
        "candidate_source": candidate_source,
        "override_fired": truthy(row.get("override_fired")),
        "no_override_reason": row.get("no_override_reason", ""),
        "realized_delta": realized,
        "realized_candidate_seat_delta": realized,
        "realized_delta_valid": truthy(row.get("realized_delta_valid")),
        "realized_delta_observed": truthy(row.get("override_fired")) and truthy(row.get("realized_delta_valid")),
        "realized_loss": max(0.0, -realized),
        "confirm_delta": safe_float(confirm_delta),
        "confirm_delta_se": safe_float(confirm_delta_se),
        "confirm_delta_count": safe_int(row.get("confirm_delta_count"), 0),
        "confirm_paired_delta_summary": row.get("confirm_paired_delta_summary"),
        "stage_a_delta": safe_float(row.get("stage_a_delta")),
        "stage_a_delta_se": safe_float(row.get("stage_a_delta_se")),
        "predicted_delta": safe_float(stage8c_candidate.get("predicted_delta"), safe_float(row.get("predicted_delta"))),
        "gate_probability": safe_float(stage8c_candidate.get("gate_probability"), safe_float(row.get("gate_probability"))),
        "candidate_ev_rank": safe_int(stage8c_candidate.get("candidate_ev_rank"), safe_int(row.get("candidate_ev_rank"), 9999)),
        "stage8c_fire_selector_probability": safe_float(
            stage8c_candidate.get("probability"),
            safe_float(row.get("stage8c_fire_selector_probability")),
        ),
        "stage8c_fire_selector_passed": bool(stage8c_candidate.get("passed", False)),
        "candidate_ev_rank_max": safe_int(row.get("candidate_ev_rank_max"), 9999),
        "model_score": safe_float(stage8c_candidate.get("model_score"), safe_float(row.get("model_score"))),
        "topk_score": safe_float(row.get("topk_score")),
        "top_k": safe_int(row.get("top_k")),
        "mc_samples": safe_int(row.get("mc_samples")),
        "confirm_mc_samples": safe_int(row.get("confirm_mc_samples")),
        "confirm_se_multiplier": safe_float(row.get("confirm_se_multiplier")),
        "hero_board": row.get("hero_board"),
        "opponent_board": row.get("opponent_board"),
        "dead_cards": row.get("dead_cards"),
        "visible_dead_cards": row.get("visible_dead_cards"),
        "true_dead_cards": row.get("true_dead_cards"),
        "hero_private_discards": row.get("hero_private_discards"),
        "opponent_private_discards": row.get("opponent_private_discards"),
        "cards_to_place": row.get("cards_to_place"),
        "baseline_action": row.get("baseline_action"),
        "candidate_action": candidate_action,
        "baseline_index": safe_int(row.get("baseline_action_index"), -1),
        "candidate_index": safe_int(candidate_index, -1),
        "baseline_action_index": safe_int(row.get("baseline_action_index"), -1),
        "candidate_action_index": safe_int(candidate_index, -1),
        "rerank_best_index": safe_int(row.get("rerank_best_index"), -1),
        "stage8b_top1_index": safe_int(row.get("stage8b_top1_index"), -1),
        "state_signature": source_state_key(row),
        "action_signature": action_signature(candidate_action),
        "baseline_action_signature": action_signature(row.get("baseline_action")),
        "training_ready": True,
        "missing_training_fields": "",
        "production_p2_fixed": "No-Go",
        "teacher_50k": "No-Go",
        "t1_training": "No-Go",
    }
    missing = missing_training_fields(payload)
    payload["training_ready"] = not missing
    payload["missing_training_fields"] = ",".join(missing)
    # These fields keep the generic Stage8c trainer from treating the row as a
    # local replay label.
    payload["local_replay_status"] = ""
    payload["local_replay_action_mapping_status"] = ""
    return payload


def dedupe_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("state_signature", "")), str(row.get("action_signature", "")))
        current = by_key.get(key)
        if current is None:
            by_key[key] = row
            continue
        current_label = safe_int(current.get("topk_distill_label_id"))
        new_label = safe_int(row.get("topk_distill_label_id"))
        if new_label > current_label:
            by_key[key] = row
            continue
        if new_label == current_label and safe_float(row.get("realized_delta")) > safe_float(current.get("realized_delta")):
            by_key[key] = row
    return sorted(by_key.values(), key=lambda row: (str(row.get("hand_seed")), str(row.get("seat")), str(row.get("action_signature"))))


def sample_group(rows: list[dict[str, Any]], *, limit: int, seed: int) -> list[dict[str, Any]]:
    if limit < 0 or len(rows) <= limit:
        return rows
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    keep = set(indices[:limit])
    return [row for index, row in enumerate(rows) if index in keep]


def build_rows(
    logs: Iterable[tuple[str, Iterable[dict[str, Any]]]],
    *,
    positive_threshold: float,
    max_rejected_negatives: int,
    max_fire_selector_negatives: int,
    max_topk_empty_negatives: int,
    sample_seed: int,
    include_topk_empty: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    positives: list[dict[str, Any]] = []
    losses: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    fire_selector_rejected: list[dict[str, Any]] = []
    topk_empty: list[dict[str, Any]] = []
    skipped = Counter()
    source_rows: list[dict[str, Any]] = []
    for source_log, rows in logs:
        input_count = 0
        for row in rows:
            input_count += 1
            candidate, candidate_index, candidate_source = candidate_from_row(row, include_topk_empty=include_topk_empty)
            if candidate is None:
                skipped["no_candidate"] += 1
                continue
            if action_signature(candidate) == action_signature(row.get("baseline_action")):
                skipped["candidate_same_as_baseline"] += 1
                continue
            target = label_for_row(row, positive_threshold=positive_threshold)
            if target is None:
                skipped["no_label"] += 1
                continue
            label, training_use = target
            payload = distillation_payload(
                row,
                source_log=source_log,
                label=label,
                training_use=training_use,
                candidate_action=candidate,
                candidate_index=candidate_index,
                candidate_source=candidate_source,
            )
            if not training_ready(payload):
                skipped[f"not_training_ready:{payload.get('missing_training_fields', '')}"] += 1
                continue
            if training_use == POSITIVE_USE:
                positives.append(payload)
            elif training_use == LOSS_USE:
                losses.append(payload)
            elif training_use == REJECTED_USE:
                rejected.append(payload)
            elif training_use == FIRE_SELECTOR_REJECTED_USE:
                fire_selector_rejected.append(payload)
            elif training_use == TOPK_EMPTY_USE:
                topk_empty.append(payload)
        source_rows.append({"source_log": source_log, "rows": input_count})

    positives = dedupe_rows(positives)
    losses = dedupe_rows(losses)
    rejected = sample_group(dedupe_rows(rejected), limit=max_rejected_negatives, seed=sample_seed + 1)
    fire_selector_rejected = sample_group(
        dedupe_rows(fire_selector_rejected),
        limit=max_fire_selector_negatives,
        seed=sample_seed + 2,
    )
    topk_empty = sample_group(dedupe_rows(topk_empty), limit=max_topk_empty_negatives, seed=sample_seed + 3)
    rows = sorted(
        positives + losses + rejected + fire_selector_rejected + topk_empty,
        key=lambda row: (
            str(row.get("recommended_training_use")),
            str(row.get("hand_seed")),
            str(row.get("seat")),
            str(row.get("action_signature")),
        ),
    )
    source_rows.append({"source_log": "_skipped", "rows": sum(skipped.values()), "counts": json.dumps(dict(skipped), sort_keys=True)})
    return rows, source_rows


def summary_rows(rows: list[dict[str, Any]], source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    output.append({"metric": "source_logs", "value": sum(1 for row in source_rows if row.get("source_log") != "_skipped")})
    output.append({"metric": "input_rows", "value": sum(safe_int(row.get("rows")) for row in source_rows if row.get("source_log") != "_skipped")})
    output.append({"metric": "training_rows", "value": len(rows)})
    output.append({"metric": "primary_label_source", "value": PRIMARY_LABEL_SOURCE})
    output.append({"metric": "confirm_delta_metric_role", "value": CONFIRM_DELTA_METRIC_ROLE})
    output.append({"metric": "confirm_delta_performance_claim_allowed", "value": CONFIRM_DELTA_PERFORMANCE_CLAIM_ALLOWED})
    output.append({"metric": "positive_rows", "value": sum(safe_int(row.get("topk_distill_label_id")) for row in rows)})
    output.append({"metric": "negative_rows", "value": sum(1 - safe_int(row.get("topk_distill_label_id")) for row in rows)})
    output.append({"metric": "training_ready_rows", "value": sum(1 for row in rows if row.get("training_ready"))})
    output.append({"metric": "realized_delta_mean", "value": sum(safe_float(row.get("realized_delta")) for row in rows) / len(rows) if rows else 0.0})
    for key, count in sorted(Counter(str(row.get("recommended_training_use")) for row in rows).items()):
        output.append({"metric": f"recommended_training_use.{key}", "value": count})
    for key, count in sorted(Counter(str(row.get("seat")) for row in rows).items()):
        output.append({"metric": f"seat.{key}", "value": count})
    for row in source_rows:
        if row.get("source_log") == "_skipped":
            output.append({"metric": "skipped_counts", "value": row.get("counts", "{}")})
    return output


def breakdown_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    groups: list[tuple[str, str, list[dict[str, Any]]]] = [("overall", "all", rows)]
    for field in ("recommended_training_use", "seat", "candidate_source", "no_override_reason"):
        for value in sorted({str(row.get(field, "")) for row in rows}):
            groups.append((field, value, [row for row in rows if str(row.get(field, "")) == value]))
    for field, value, subset in groups:
        deltas = [safe_float(row.get("realized_delta")) for row in subset]
        output.append(
            {
                "group_field": field,
                "group_value": value,
                "rows": len(subset),
                "positive_rows": sum(safe_int(row.get("topk_distill_label_id")) for row in subset),
                "positive_rate": (
                    sum(safe_int(row.get("topk_distill_label_id")) for row in subset) / len(subset) if subset else 0.0
                ),
                "realized_delta_mean": sum(deltas) / len(deltas) if deltas else 0.0,
                "realized_loss_count": sum(1 for value in deltas if value < 0.0),
                "confirm_delta_mean": (
                    sum(safe_float(row.get("confirm_delta")) for row in subset) / len(subset) if subset else 0.0
                ),
                "confirm_delta_metric_role": CONFIRM_DELTA_METRIC_ROLE,
                "confirm_delta_performance_claim_allowed": CONFIRM_DELTA_PERFORMANCE_CLAIM_ALLOWED,
                "primary_label_source": PRIMARY_LABEL_SOURCE,
                "predicted_delta_mean": (
                    sum(safe_float(row.get("predicted_delta")) for row in subset) / len(subset) if subset else 0.0
                ),
            }
        )
    return output


def write_markdown(path: Path, rows: list[dict[str, Any]], summary: list[dict[str, Any]]) -> None:
    values = {str(row["metric"]): row["value"] for row in summary}
    lines = [
        "# HU T2 Stage8c TopK+Confirm Distillation Prep",
        "",
        "This artifact is for lightweight runtime distillation only. It does not approve production, P2 fixed status, 50k teacher generation, or T1.",
        "",
        "Labels use realized fired-hand deltas for fired rows and confirm-gate rejection for rejected candidates. Confirm MC means are runtime features, not performance claims.",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for row in summary:
        lines.append(f"| {row['metric']} | {row['value']} |")
    lines.extend(
        [
            "",
            "## Recommended Next Step",
            "",
            "- Train with `target_mode=topk_confirm_fire` as a smoke first.",
            "- Keep TopK+confirm as an offline/search policy until a lightweight model reproduces the signal in realized per-fire validation.",
            "- Do not mix these rows with old teacher-EV LCB labels for adoption decisions.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    logs = [(str(path), iter_jsonl(path)) for path in args.decision_log]
    rows, source_rows = build_rows(
        logs,
        positive_threshold=args.positive_threshold,
        max_rejected_negatives=args.max_rejected_negatives,
        max_fire_selector_negatives=args.max_fire_selector_negatives,
        max_topk_empty_negatives=args.max_topk_empty_negatives,
        sample_seed=args.sample_seed,
        include_topk_empty=args.include_topk_empty,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = summary_rows(rows, source_rows)
    write_jsonl(args.output_dir / OUTPUT_NAME, rows)
    write_csv(args.output_dir / "topk_confirm_distillation_summary.csv", summary)
    write_csv(args.output_dir / "topk_confirm_distillation_breakdown.csv", breakdown_rows(rows))
    write_markdown(args.output_dir / "topk_confirm_distillation_summary.md", rows, summary)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "rows": len(rows),
                "positive_rows": sum(safe_int(row.get("topk_distill_label_id")) for row in rows),
                "negative_rows": sum(1 - safe_int(row.get("topk_distill_label_id")) for row in rows),
                "production": "No-Go",
                "p2_fixed": "No-Go",
                "fifty_k_teacher": "No-Go",
                "t1": "No-Go",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
