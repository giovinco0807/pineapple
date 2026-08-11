"""Merge independent replay labels back into Stage8c TopK distillation rows."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .extract_hu_turn2_stage8c_topk_replay_targets import row_key, safe_float, safe_int, truthy


DEFAULT_OUTPUT_DIR = Path("outputs/training/hu_turn2_stage8c_topk_confirm_replay_labeled_distillation")
OUTPUT_NAME = "topk_confirm_replay_labeled_distillation_rows.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distillation-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--replay-source-jsonl", type=Path, required=True)
    parser.add_argument("--replay-summary-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_jsonl_many(paths: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for path in paths:
        file_rows = read_jsonl(path)
        rows.extend(file_rows)
        sources.append({"path": str(path), "rows": len(file_rows)})
    return rows, sources


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


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
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def replay_label(summary_row: dict[str, str]) -> str:
    if str(summary_row.get("safe_lcb196_label", "")).lower() == "positive":
        return "topk_confirm_replay_positive"
    if safe_int(summary_row.get("hard_negative_label")) or str(summary_row.get("safe_lcb196_label", "")).lower() == "negative":
        return "topk_confirm_replay_negative"
    return "topk_confirm_replay_gray"


def replay_bucket(summary_row: dict[str, str]) -> str:
    label = str(summary_row.get("safe_lcb196_label", "")).lower()
    if label == "positive":
        return "local_positive_lcb"
    if label == "negative" or safe_int(summary_row.get("hard_negative_label")):
        return "local_negative"
    if safe_float(summary_row.get("delta_for_label")) > 0.0:
        return "local_positive_gray"
    if safe_float(summary_row.get("delta_for_label")) < 0.0:
        return "local_negative_delta"
    return "local_neutral"


def source_for_summary(summary_row: dict[str, str], replay_source_rows: list[dict[str, Any]]) -> dict[str, Any]:
    index = safe_int(summary_row.get("row_index"), -1)
    if 0 <= index < len(replay_source_rows):
        return replay_source_rows[index]
    return {}


def build_replay_label_index(
    replay_source_rows: list[dict[str, Any]],
    replay_summary_rows: list[dict[str, str]],
) -> dict[tuple[str, str, str, str, str, str, str, str], dict[str, Any]]:
    output: dict[tuple[str, str, str, str, str, str, str, str], dict[str, Any]] = {}
    for summary in replay_summary_rows:
        source = source_for_summary(summary, replay_source_rows)
        if not source:
            continue
        future_samples = safe_int(summary.get("future_samples"))
        replay_basis = f"mc{future_samples}_independent_replay" if future_samples > 0 else "independent_replay"
        if summary.get("status") != "ok" or summary.get("action_mapping_status") != "ok":
            label = "topk_confirm_replay_failed"
        else:
            label = replay_label(summary)
        output[row_key(source)] = {
            "recommended_training_use": label,
            "topk_distill_label_id": 1 if label == "topk_confirm_replay_positive" else 0,
            "risk_target_group": label,
            "replay_label_source": "independent_mc_replay",
            "replay_label_basis": replay_basis,
            "local_replay_status": summary.get("status", ""),
            "local_replay_action_mapping_status": summary.get("action_mapping_status", ""),
            "local_replay_future_samples": future_samples,
            "local_replay_delta": safe_float(summary.get("delta_for_label")),
            "local_replay_delta_se": safe_float(summary.get("delta_standard_error_for_label")),
            "local_replay_lcb196": safe_float(summary.get("replay_delta_lcb196")),
            "local_replay_lcb164": safe_float(summary.get("replay_delta_lcb164")),
            "local_replay_label": summary.get("safe_lcb196_label", ""),
            "local_replay_bucket": replay_bucket(summary),
            "hard_negative_label": safe_int(summary.get("hard_negative_label")),
            "realized_delta": safe_float(summary.get("delta_for_label")),
            "realized_candidate_seat_delta": safe_float(summary.get("delta_for_label")),
            "realized_delta_valid": True,
            "realized_delta_observed": True,
            "realized_delta_basis": replay_basis,
            "realized_loss": max(0.0, -safe_float(summary.get("delta_for_label"))),
            "observed_performance_claim": "No",
        }
    return output


def merge_rows(
    distillation_rows: list[dict[str, Any]],
    replay_label_index: dict[tuple[str, str, str, str, str, str, str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output: list[dict[str, Any]] = []
    replacements: list[dict[str, Any]] = []
    for row in distillation_rows:
        replacement = replay_label_index.get(row_key(row))
        if replacement is None:
            output.append(row)
            continue
        merged = dict(row)
        original_use = str(merged.get("recommended_training_use", ""))
        merged.update(replacement)
        merged["original_recommended_training_use"] = original_use
        merged["replay_label_merged"] = True
        output.append(merged)
        replacements.append(
            {
                "hand_seed": merged.get("hand_seed"),
                "seat": merged.get("seat"),
                "original_recommended_training_use": original_use,
                "recommended_training_use": merged.get("recommended_training_use"),
                "local_replay_delta": merged.get("local_replay_delta"),
                "local_replay_lcb196": merged.get("local_replay_lcb196"),
                "local_replay_label": merged.get("local_replay_label"),
                "candidate_source": merged.get("candidate_source"),
                "risk_prediction_split": merged.get("risk_prediction_split", ""),
                "local_replay_future_samples": merged.get("local_replay_future_samples"),
                "replay_label_basis": merged.get("replay_label_basis"),
                "realized_delta_basis": merged.get("realized_delta_basis"),
            }
        )
    return output, replacements


def summary_rows(rows: list[dict[str, Any]], replacements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    uses = Counter(str(row.get("recommended_training_use", "")) for row in rows)
    replay_uses = Counter(str(row.get("recommended_training_use", "")) for row in replacements)
    replay_basis_counts = Counter(str(row.get("replay_label_basis", "")) for row in replacements)
    replay_sample_counts = Counter(str(row.get("local_replay_future_samples", "")) for row in replacements)
    output = [
        {"metric": "rows", "value": len(rows)},
        {"metric": "replay_label_replacements", "value": len(replacements)},
        {
            "metric": "replay_positive_replacements",
            "value": replay_uses.get("topk_confirm_replay_positive", 0),
        },
        {
            "metric": "replay_negative_replacements",
            "value": replay_uses.get("topk_confirm_replay_negative", 0),
        },
        {
            "metric": "replay_gray_replacements",
            "value": replay_uses.get("topk_confirm_replay_gray", 0),
        },
        {
            "metric": "trainable_topk_confirm_fire_rows",
            "value": sum(
                1
                for row in rows
                if str(row.get("recommended_training_use", ""))
                in {
                    "topk_confirm_realized_positive",
                    "topk_confirm_realized_loss",
                    "topk_confirm_rejected",
                    "topk_confirm_topk_empty",
                    "topk_confirm_replay_positive",
                    "topk_confirm_replay_negative",
                }
            ),
        },
    ]
    for key, value in sorted(uses.items()):
        output.append({"metric": f"recommended_use.{key}", "value": value})
    for key, value in sorted(replay_basis_counts.items()):
        if key:
            output.append({"metric": f"replay_label_basis.{key}", "value": value})
    for key, value in sorted(replay_sample_counts.items()):
        if key:
            output.append({"metric": f"local_replay_future_samples.{key}", "value": value})
    return output


def write_summary(path: Path, rows: list[dict[str, Any]], replacements: list[dict[str, Any]]) -> None:
    values = {str(row["metric"]): row["value"] for row in summary_rows(rows, replacements)}
    basis_lines = [
        f"- `{key.removeprefix('replay_label_basis.')}`: `{value}`"
        for key, value in sorted(values.items())
        if key.startswith("replay_label_basis.")
    ]
    lines = [
        "# HU T2 Stage8c TopK Confirm Replay-Labeled Distillation",
        "",
        "Replay labels replace selected unknown placeholder labels. This is a training-prep artifact, not a production gate.",
        "",
        f"- Rows: `{values.get('rows', 0)}`",
        f"- Replay replacements: `{values.get('replay_label_replacements', 0)}`",
        f"- Replay positives: `{values.get('replay_positive_replacements', 0)}`",
        f"- Replay negatives: `{values.get('replay_negative_replacements', 0)}`",
        f"- Replay gray/excluded: `{values.get('replay_gray_replacements', 0)}`",
        f"- Trainable topk_confirm_fire rows: `{values.get('trainable_topk_confirm_fire_rows', 0)}`",
        "",
        "Replay label basis:",
        *(basis_lines or ["- none"]),
        "",
        "- Production / P2 fixed / 50k / T1: `No-Go`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(
    path: Path,
    *,
    args: argparse.Namespace,
    distillation_sources: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    replay_source_rows: list[dict[str, Any]],
    replay_summary_rows: list[dict[str, str]],
    replay_index: dict[tuple[str, str, str, str, str, str, str, str], dict[str, Any]],
    replacements: list[dict[str, Any]],
) -> None:
    values = {str(row["metric"]): row["value"] for row in summary_rows(rows, replacements)}
    manifest = {
        "schema": "hu_turn2_stage8c_topk_confirm_replay_labeled_distillation_v1",
        "distillation_sources": distillation_sources,
        "replay_source_jsonl": str(args.replay_source_jsonl),
        "replay_summary_csv": str(args.replay_summary_csv),
        "output_dir": str(args.output_dir),
        "output_jsonl": OUTPUT_NAME,
        "rows": len(rows),
        "replay_source_rows": len(replay_source_rows),
        "replay_summary_rows": len(replay_summary_rows),
        "replay_index_rows": len(replay_index),
        "replacements": len(replacements),
        "replay_positive_replacements": values.get("replay_positive_replacements", 0),
        "replay_negative_replacements": values.get("replay_negative_replacements", 0),
        "replay_gray_replacements": values.get("replay_gray_replacements", 0),
        "trainable_topk_confirm_fire_rows": values.get("trainable_topk_confirm_fire_rows", 0),
        "replay_label_basis_counts": {
            key.removeprefix("replay_label_basis."): value
            for key, value in values.items()
            if key.startswith("replay_label_basis.")
        },
        "local_replay_future_sample_counts": {
            key.removeprefix("local_replay_future_samples."): value
            for key, value in values.items()
            if key.startswith("local_replay_future_samples.")
        },
        "production_p2_fixed": "No-Go",
        "teacher_50k": "No-Go",
        "t1_training": "No-Go",
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    distillation_rows, distillation_sources = read_jsonl_many(args.distillation_jsonl)
    replay_source_rows = read_jsonl(args.replay_source_jsonl)
    replay_summary_rows = read_csv(args.replay_summary_csv)
    replay_index = build_replay_label_index(replay_source_rows, replay_summary_rows)
    rows, replacements = merge_rows(distillation_rows, replay_index)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / OUTPUT_NAME, rows)
    write_csv(args.output_dir / "topk_confirm_replay_label_replacements.csv", replacements)
    write_csv(args.output_dir / "topk_confirm_replay_labeled_summary.csv", summary_rows(rows, replacements))
    write_summary(args.output_dir / "topk_confirm_replay_labeled_summary.md", rows, replacements)
    write_manifest(
        args.output_dir / "topk_confirm_replay_labeled_manifest.json",
        args=args,
        distillation_sources=distillation_sources,
        rows=rows,
        replay_source_rows=replay_source_rows,
        replay_summary_rows=replay_summary_rows,
        replay_index=replay_index,
        replacements=replacements,
    )
    print(
        json.dumps(
            {
                "schema": "hu_turn2_stage8c_topk_confirm_replay_labeled_distillation_v1",
                "rows": len(rows),
                "replacements": len(replacements),
                "output_dir": str(args.output_dir),
                "production_p2_fixed": "No-Go",
                "teacher_50k": "No-Go",
                "t1_training": "No-Go",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
