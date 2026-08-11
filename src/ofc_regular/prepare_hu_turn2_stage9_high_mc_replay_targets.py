"""Prepare high-MC replay targets for HU T2 Stage9 candidate generation.

This converts Stage9 candidate-generator audit artifacts into the CSV format
consumed by ``replay_hu_turn2_event_high_mc``.  It is a label-improvement tool:
the output is for independent high-MC replay before another Stage9 training
round, not for seat-swap or production claims.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_OUTPUT_DIR = Path("outputs/training/hu_turn2_stage9_high_mc_replay_targets")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-name", default="stage9_high_mc_replay_candidates.csv")
    parser.add_argument("--split", action="append", default=[], help="Split to include. Defaults to all splits.")
    parser.add_argument("--hard-miss-limit", type=int, default=250)
    parser.add_argument("--hard-negative-limit", type=int, default=250)
    parser.add_argument("--near-boundary-limit", type=int, default=100)
    parser.add_argument(
        "--near-boundary-max-regret",
        type=float,
        default=0.25,
        help="Select states with 0 < top5_oracle_regret <= this value as near-boundary oracle checks.",
    )
    parser.add_argument("--max-total", type=int, default=0)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


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


def split_allowed(row: dict[str, Any], splits: set[str] | None) -> bool:
    return splits is None or str(row.get("split", "")) in splits


def _base_target(row: dict[str, Any], *, reason: str, priority: float) -> dict[str, Any]:
    state_index = safe_int(row.get("state_index"), -1)
    return {
        "candidate_id": f"stage9:{reason}:state{state_index}:cand{safe_int(row.get('candidate_action_local_index'), -1)}",
        "priority": priority,
        "reason": reason,
        "state_index": state_index,
        "sample_id": row.get("sample_id", ""),
        "source": row.get("source_bucket", ""),
        "source_group": row.get("bucket_group") or row.get("run_bucket") or row.get("source_bucket", ""),
        "run_bucket": row.get("run_bucket", ""),
        "seat": row.get("seat", ""),
        "position": row.get("seat", ""),
        "split": row.get("split", ""),
        "pilot_gate_label": row.get("pilot_gate_label", ""),
        "baseline_action_local_index": safe_int(row.get("baseline_index"), -1),
        "candidate_action_local_index": safe_int(row.get("candidate_action_local_index"), -1),
        "teacher_best_action_local_index": safe_int(row.get("oracle_best_index"), -1),
        "teacher_gain_mean_current_mc512": safe_float(row.get("teacher_gain_mean_current_mc512")),
        "teacher_gain_stderr_proxy_current_mc512": row.get("teacher_gain_stderr_proxy_current_mc512", ""),
        "oracle_best_delta_vs_baseline": safe_float(row.get("oracle_best_delta_vs_baseline")),
        "model_top1_delta_vs_baseline": safe_float(row.get("model_top1_delta_vs_baseline")),
        "model_top1_regret": safe_float(row.get("model_top1_regret")),
        "top5_oracle_regret": safe_float(row.get("top5_oracle_regret")),
        "hard_miss_regret": safe_float(row.get("hard_miss_regret")),
        "production_p2_fixed": "No-Go",
        "teacher_50k": "No-Go",
        "t1_training": "No-Go",
        "observed_performance_claim": "No",
    }


def hard_miss_target(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["candidate_action_local_index"] = safe_int(row.get("oracle_best_index"), -1)
    out["teacher_gain_mean_current_mc512"] = safe_float(row.get("oracle_best_delta_vs_baseline"))
    priority = safe_float(row.get("hard_miss_regret")) + 0.25 * safe_float(row.get("oracle_best_delta_vs_baseline"))
    target = _base_target(out, reason="stage9_hard_miss_oracle_best", priority=priority)
    target["source_artifact"] = "hard_miss_states.jsonl"
    return target


def hard_negative_target(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["candidate_action_local_index"] = safe_int(row.get("hard_negative_index"), -1)
    out["teacher_gain_mean_current_mc512"] = safe_float(row.get("hard_negative_delta_vs_baseline"))
    priority = safe_float(row.get("hard_negative_loss_vs_baseline"))
    target = _base_target(out, reason="stage9_model_topk_hard_negative", priority=priority)
    target["source_artifact"] = "hard_negative_actions.jsonl"
    target["hard_negative_delta_vs_baseline"] = safe_float(row.get("hard_negative_delta_vs_baseline"))
    target["hard_negative_loss_vs_baseline"] = safe_float(row.get("hard_negative_loss_vs_baseline"))
    return target


def near_boundary_target(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["candidate_action_local_index"] = safe_int(row.get("oracle_best_index"), -1)
    out["teacher_gain_mean_current_mc512"] = safe_float(row.get("oracle_best_delta_vs_baseline"))
    priority = safe_float(row.get("top5_oracle_regret"))
    target = _base_target(out, reason="stage9_top5_near_boundary_oracle_check", priority=priority)
    target["source_artifact"] = "candidate_generator_state_rows.csv"
    return target


def dedupe_targets(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[int, int, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            safe_int(row.get("state_index"), -1),
            safe_int(row.get("candidate_action_local_index"), -1),
            str(row.get("reason", "")),
        )
        previous = best.get(key)
        if previous is None or safe_float(row.get("priority")) > safe_float(previous.get("priority")):
            best[key] = row
    return sorted(best.values(), key=lambda item: (-safe_float(item.get("priority")), safe_int(item.get("state_index"), -1)))


def select_stage9_replay_targets(
    *,
    hard_miss_rows: list[dict[str, Any]],
    hard_negative_rows: list[dict[str, Any]],
    state_rows: list[dict[str, Any]],
    splits: set[str] | None,
    hard_miss_limit: int,
    hard_negative_limit: int,
    near_boundary_limit: int,
    near_boundary_max_regret: float,
    max_total: int = 0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    hard_miss_targets = [
        hard_miss_target(row)
        for row in hard_miss_rows
        if split_allowed(row, splits) and safe_int(row.get("oracle_best_index"), -1) >= 0
    ]
    hard_miss_targets = dedupe_targets(hard_miss_targets)[: max(0, hard_miss_limit)]
    selected.extend(hard_miss_targets)

    hard_negative_targets = [
        hard_negative_target(row)
        for row in hard_negative_rows
        if split_allowed(row, splits) and safe_int(row.get("hard_negative_index"), -1) >= 0
    ]
    hard_negative_targets = dedupe_targets(hard_negative_targets)[: max(0, hard_negative_limit)]
    selected.extend(hard_negative_targets)

    near_boundary_targets = []
    if near_boundary_limit > 0 and near_boundary_max_regret > 0.0:
        for row in state_rows:
            regret = safe_float(row.get("top5_oracle_regret"))
            if (
                split_allowed(row, splits)
                and 0.0 < regret <= near_boundary_max_regret
                and safe_int(row.get("oracle_best_index"), -1) >= 0
            ):
                near_boundary_targets.append(near_boundary_target(row))
    near_boundary_targets = dedupe_targets(near_boundary_targets)[:near_boundary_limit]
    selected.extend(near_boundary_targets)

    selected = dedupe_targets(selected)
    if max_total > 0:
        selected = selected[:max_total]

    counts = Counter(str(row.get("reason", "")) for row in selected)
    split_counts = Counter(str(row.get("split", "")) for row in selected)
    for reason, count in sorted(counts.items()):
        audit_rows.append({"group_type": "reason", "group": reason, "targets": count})
    for split, count in sorted(split_counts.items()):
        audit_rows.append({"group_type": "split", "group": split, "targets": count})
    return selected, audit_rows


def write_summary(path: Path, *, manifest: dict[str, Any], audit_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# HU T2 Stage9 High-MC Replay Targets",
        "",
        "This artifact prepares independent high-MC replay targets for candidate-generator labels.",
        "It is not seat-swap evidence and does not approve production, 50k teacher, or T1.",
        "",
        f"- audit dir: `{manifest['audit_dir']}`",
        f"- target csv: `{manifest['target_csv']}`",
        f"- targets: `{manifest['targets']}`",
        f"- splits: `{manifest['splits']}`",
        "",
        "## Breakdown",
        "",
    ]
    if audit_rows:
        for row in audit_rows:
            lines.append(f"- {row['group_type']} `{row['group']}`: `{row['targets']}`")
    else:
        lines.append("- no targets")
    lines.extend(
        [
            "",
            "## Next Commands",
            "",
            "Readiness only:",
            "",
            "```powershell",
            (
                "python -m ofc_regular.replay_hu_turn2_event_high_mc "
                f"--candidates {manifest['target_csv']} "
                f"--cache-dir {manifest['cache_dir']} "
                f"--output-dir {manifest['output_dir']}\\readiness "
                "--readiness-only --t3-continuation stage7_m5_r10"
            ),
            "```",
            "",
            "High-MC replay after readiness is clean:",
            "",
            "```powershell",
            (
                "python -m ofc_regular.replay_hu_turn2_event_high_mc "
                f"--candidates {manifest['target_csv']} "
                f"--cache-dir {manifest['cache_dir']} "
                f"--output-dir {manifest['output_dir']}\\mc512 "
                "--mc-samples 512 --seed-mode offset_by_mc --t3-continuation stage7_m5_r10 "
                "--prediction-threads 1"
            ),
            "```",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    audit_dir = args.audit_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = {str(value) for value in args.split if str(value)} or None

    manifest_path = audit_dir / "candidate_generator_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    hard_miss_rows = read_jsonl(audit_dir / "hard_miss_states.jsonl")
    hard_negative_rows = read_jsonl(audit_dir / "hard_negative_actions.jsonl")
    state_rows = read_csv(audit_dir / "candidate_generator_state_rows.csv")

    targets, audit_rows = select_stage9_replay_targets(
        hard_miss_rows=hard_miss_rows,
        hard_negative_rows=hard_negative_rows,
        state_rows=state_rows,
        splits=splits,
        hard_miss_limit=args.hard_miss_limit,
        hard_negative_limit=args.hard_negative_limit,
        near_boundary_limit=args.near_boundary_limit,
        near_boundary_max_regret=args.near_boundary_max_regret,
        max_total=args.max_total,
    )
    target_csv = output_dir / args.target_name
    write_csv(target_csv, targets)
    write_csv(output_dir / "stage9_high_mc_replay_target_breakdown.csv", audit_rows)
    out_manifest = {
        "schema": "hu_turn2_stage9_high_mc_replay_targets_v1",
        "audit_dir": str(audit_dir),
        "output_dir": str(output_dir),
        "target_csv": str(target_csv),
        "cache_dir": manifest.get("cache_dir", ""),
        "model": manifest.get("model", ""),
        "score_head": manifest.get("score_head", ""),
        "splits": sorted(splits) if splits is not None else "all",
        "targets": len(targets),
        "hard_miss_limit": args.hard_miss_limit,
        "hard_negative_limit": args.hard_negative_limit,
        "near_boundary_limit": args.near_boundary_limit,
        "near_boundary_max_regret": args.near_boundary_max_regret,
        "max_total": args.max_total,
        "production_p2_fixed": "No-Go",
        "teacher_50k": "No-Go",
        "t1_training": "No-Go",
    }
    write_json(output_dir / "stage9_high_mc_replay_manifest.json", out_manifest)
    write_summary(output_dir / "stage9_high_mc_replay_summary.md", manifest=out_manifest, audit_rows=audit_rows)
    print(json.dumps(out_manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
