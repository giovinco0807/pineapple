"""Analyze HU Turn2 Stage8 broad-pass teacher data."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

BUCKET_FILES = (
    "natural",
    "predicted_high_regret_from_pool",
    "predicted_low_margin_from_pool",
    "predicted_teacher_disagreement_from_pool",
    "random_off_policy",
)
POOL_FILES = {
    "predicted_high_regret": "predicted_high_regret_candidates.jsonl",
    "predicted_low_margin": "predicted_low_margin_candidates.jsonl",
    "predicted_teacher_disagreement": "predicted_teacher_disagreement_candidates.jsonl",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-total", type=int, default=20_000)
    parser.add_argument("--expected-per-bucket", type=int, default=4_000)
    parser.add_argument("--future-samples", type=int, default=512)
    parser.add_argument("--summary-dir", type=Path)
    parser.add_argument("--pool-dir", type=Path)
    parser.add_argument("--run-name", default="")
    return parser.parse_args()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def percentile(values: Iterable[float], q: float) -> float:
    arr = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, q * 100.0))


def mean(values: Iterable[float]) -> float:
    xs = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(xs) / len(xs)) if xs else 0.0


def action_signature(action: dict[str, Any]) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
    body = action.get("action") if isinstance(action.get("action"), dict) else action
    placements = tuple(sorted((str(card), str(row)) for card, row in body.get("placements", ())))
    discards = tuple(sorted(str(card) for card in body.get("discards", ())))
    return placements, discards


def action_is_legal(record: dict[str, Any], action: Any) -> bool:
    actions = record.get("actions", ())
    legal = record.get("legal_actions", ()) or actions
    if isinstance(action, int):
        return 0 <= action < len(actions)
    if not isinstance(action, dict):
        return False
    if "original_index" in action:
        original = safe_int(action.get("original_index"), -999)
        if any(safe_int(candidate.get("original_index"), -1) == original for candidate in actions):
            return True
    signature = action_signature(action)
    return any(action_signature(candidate) == signature for candidate in legal)


def actual_flags(record: dict[str, Any]) -> dict[str, bool | str]:
    delta = safe_float(record.get("delta_best_vs_baseline"))
    margin = safe_float(record.get("best_margin"))
    disagreement = bool(record.get("teacher_distribution_metrics", {}).get("baseline_disagreement", False))
    high_regret = bool(record.get("actual_high_regret", delta >= 1.0))
    low_margin = bool(record.get("actual_low_margin", margin <= 0.25))
    teacher_disagreement = bool(record.get("actual_teacher_disagreement", disagreement))
    bucket = (
        "actual_high_regret"
        if high_regret
        else "actual_low_margin"
        if low_margin
        else "actual_teacher_disagreement"
        if teacher_disagreement
        else "actual_other"
    )
    return {
        "actual_high_regret": high_regret,
        "actual_low_margin": low_margin,
        "actual_teacher_disagreement": teacher_disagreement,
        "actual_bucket": str(record.get("actual_bucket") or record.get("actual_candidate_bucket") or bucket),
    }


def read_jsonl(path: Path, run_bucket: str = "") -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if run_bucket:
                record["_run_bucket"] = run_bucket
            records.append(record)
    return records


def read_teacher_records(teacher: Path, output_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    sidecars = [(output_dir / f"{bucket}.jsonl", bucket) for bucket in BUCKET_FILES]
    if all(path.exists() for path, _bucket in sidecars):
        for path, bucket in sidecars:
            records.extend(read_jsonl(path, bucket))
        return records
    return read_jsonl(teacher, "")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    keys.append(key)
                    seen.add(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def signal_row(group: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    deltas = [safe_float(r.get("delta_best_vs_baseline")) for r in records]
    margins = [safe_float(r.get("best_margin")) for r in records]
    se = [safe_float(r.get("SE_delta_best_vs_baseline") or r.get("SE_delta")) for r in records]
    labels = Counter(str(r.get("teacher_label") or r.get("gate_label") or "unknown") for r in records)
    return {
        "group": group,
        "records": len(records),
        "positive": labels.get("positive", 0),
        "gray": labels.get("gray", 0),
        "negative": labels.get("negative", 0),
        "delta_mean": mean(deltas),
        "delta_p50": percentile(deltas, 0.50),
        "delta_p90": percentile(deltas, 0.90),
        "delta_p95": percentile(deltas, 0.95),
        "delta_p99": percentile(deltas, 0.99),
        "baseline_avg_regret": mean(deltas),
        "baseline_p50_regret": percentile(deltas, 0.50),
        "baseline_p90_regret": percentile(deltas, 0.90),
        "baseline_p95_regret": percentile(deltas, 0.95),
        "baseline_p99_regret": percentile(deltas, 0.99),
        "margin_mean": mean(margins),
        "margin_p25": percentile(margins, 0.25),
        "margin_p50": percentile(margins, 0.50),
        "margin_p75": percentile(margins, 0.75),
        "margin_p90": percentile(margins, 0.90),
        "delta_ge_2se_rate": ratio((1 for d, s in zip(deltas, se) if s > 0 and d >= 2.0 * s), len(records)),
        "delta_ge_2_5se_rate": ratio((1 for d, s in zip(deltas, se) if s > 0 and d >= 2.5 * s), len(records)),
        "delta_ge_3se_rate": ratio((1 for d, s in zip(deltas, se) if s > 0 and d >= 3.0 * s), len(records)),
    }


def ratio(count_or_iter: Any, total: int) -> float:
    if total <= 0:
        return 0.0
    if isinstance(count_or_iter, int):
        return float(count_or_iter) / total
    return float(sum(1 for _ in count_or_iter)) / total


def bucket_breakdown(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("_run_bucket") or record.get("source_bucket_actual") or record.get("source_bucket") or "unknown")].append(record)
    rows: list[dict[str, Any]] = []
    for bucket in sorted(grouped):
        group = grouped[bucket]
        flags = [actual_flags(record) for record in group]
        labels = Counter(str(record.get("teacher_label") or "unknown") for record in group)
        rows.append(
            {
                "bucket": bucket,
                "records": len(group),
                "positive": labels.get("positive", 0),
                "gray": labels.get("gray", 0),
                "negative": labels.get("negative", 0),
                "first": sum(1 for r in group if r.get("seat") == "first"),
                "second": sum(1 for r in group if r.get("seat") == "second"),
                "actual_high_regret_rate": mean(f["actual_high_regret"] for f in flags),
                "actual_low_margin_rate": mean(f["actual_low_margin"] for f in flags),
                "actual_teacher_disagreement_rate": mean(f["actual_teacher_disagreement"] for f in flags),
                "delta_mean": mean(safe_float(r.get("delta_best_vs_baseline")) for r in group),
                "margin_mean": mean(safe_float(r.get("best_margin")) for r in group),
                "seconds_mean": mean(safe_float(r.get("profiling", {}).get("seconds_total")) for r in group),
            }
        )
    return rows


def position_breakdown(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for seat in ("first", "second", "unknown"):
        group = [record for record in records if str(record.get("seat", "unknown")) == seat]
        if not group and seat != "unknown":
            continue
        row = signal_row(seat, group)
        row["first_second_ratio"] = ""
        rows.append(row)
    first = sum(1 for record in records if record.get("seat") == "first")
    second = sum(1 for record in records if record.get("seat") == "second")
    rows.append({"group": "first_second_balance", "records": len(records), "first_second_ratio": first / max(second, 1)})
    return rows


def state_profile(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for index, record in enumerate(records):
        profile = record.get("profiling", {})
        rows.append(
            {
                "state_index": index,
                "run_bucket": record.get("_run_bucket", ""),
                "source_bucket": record.get("source_bucket", ""),
                "source_bucket_requested": record.get("source_bucket_requested", ""),
                "source_bucket_actual": record.get("source_bucket_actual", ""),
                "predicted_bucket": record.get("predicted_bucket") or record.get("candidate_pool_predicted_bucket", ""),
                "actual_bucket": actual_flags(record)["actual_bucket"],
                "seat": record.get("seat", ""),
                "teacher_label": record.get("teacher_label", ""),
                "legal_actions": len(record.get("actions", ())),
                "rollout_count": record.get("rollout_count", ""),
                "delta_best_vs_baseline": safe_float(record.get("delta_best_vs_baseline")),
                "best_margin": safe_float(record.get("best_margin")),
                "SE_delta": safe_float(record.get("SE_delta_best_vs_baseline") or record.get("SE_delta")),
                "reference_margin_raw": safe_float(record.get("reference_margin_raw") or record.get("baseline_model_margin")),
                "predicted_delta": record.get("predicted_delta", ""),
                "seconds_total": safe_float(profile.get("seconds_total")),
                "rollout_seconds": safe_float(profile.get("rollout_seconds")),
                "t3_continuation_total_seconds": safe_float(profile.get("t3_continuation_total_seconds")),
                "stage3_feature_generation_time": safe_float(profile.get("stage3_feature_generation_time")),
                "final_turn_decision_seconds": safe_float(profile.get("final_turn_decision_seconds")),
                "raw_t3_states": safe_float(profile.get("raw_t3_states")),
                "stage3_fallback_recomputed_count": safe_float(profile.get("stage3_fallback_recomputed_count")),
                "memory_peak_mb": safe_float(profile.get("memory_peak_mb")),
                "stage3_feature_mode": profile.get("stage3_feature_mode", ""),
            }
        )
    return rows


def correctness(records: list[dict[str, Any]], expected_total: int, expected_per_bucket: int, future_samples: int) -> dict[str, Any]:
    bucket_counts = Counter(str(record.get("_run_bucket") or "unknown") for record in records)
    invalid_states = 0
    invalid_actions = 0
    digest_mismatch = 0
    finite_failures = 0
    rollout_mismatch = 0
    action_legal_failures = 0
    stage3_fallback_recompute = 0.0
    rust_direct_count = 0
    for record in records:
        if record.get("phase") != "hu_turn2_7card":
            invalid_states += 1
        actions = record.get("actions", ())
        root_digest = record.get("common_random_future_digest")
        digests = {action.get("common_random_future_digest") for action in actions}
        if not root_digest or len(digests) != 1 or root_digest not in digests:
            digest_mismatch += 1
        numeric_values = [
            safe_float(record.get("delta_best_vs_baseline"), float("nan")),
            safe_float(record.get("best_margin"), float("nan")),
            safe_float(record.get("SE_delta_best_vs_baseline"), float("nan")),
        ]
        for action in actions:
            numeric_values.append(safe_float(action.get("score") or action.get("ev"), float("nan")))
            numeric_values.append(safe_float(action.get("ev_standard_error") or action.get("standard_error"), float("nan")))
            if safe_int(action.get("rollout_count") or action.get("future_count"), -1) != future_samples:
                rollout_mismatch += 1
        if not all(math.isfinite(value) for value in numeric_values):
            finite_failures += 1
        if safe_int(record.get("rollout_count"), -1) != future_samples:
            rollout_mismatch += 1
        for key in ("best_action", "second_best_action", "baseline_action", "reference_action", "fallback_action"):
            if key in record and not action_is_legal(record, record.get(key)):
                action_legal_failures += 1
        profile = record.get("profiling", {})
        stage3_fallback_recompute += safe_float(profile.get("stage3_fallback_recomputed_count"))
        if profile.get("stage3_feature_mode") == "rust_direct":
            rust_direct_count += 1
    missing = max(expected_total - len(records), 0)
    bad_bucket_counts = {
        bucket: count for bucket, count in bucket_counts.items() if bucket in BUCKET_FILES and count != expected_per_bucket
    }
    ok = (
        len(records) == expected_total
        and missing == 0
        and not bad_bucket_counts
        and invalid_states == 0
        and invalid_actions == 0
        and digest_mismatch == 0
        and finite_failures == 0
        and rollout_mismatch == 0
        and action_legal_failures == 0
        and stage3_fallback_recompute == 0.0
        and rust_direct_count == len(records)
    )
    return {
        "ok": ok,
        "records": len(records),
        "expected_total": expected_total,
        "missing": missing,
        "bucket_counts": dict(bucket_counts),
        "bad_bucket_counts": bad_bucket_counts,
        "invalid_states": invalid_states,
        "invalid_actions": invalid_actions,
        "common_random_future_digest_mismatch": digest_mismatch,
        "finite_failures": finite_failures,
        "rollout_mismatch": rollout_mismatch,
        "action_legal_failures": action_legal_failures,
        "stage3_fallback_recomputed_count": stage3_fallback_recompute,
        "rust_direct_records": rust_direct_count,
    }


def enrichment_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("_run_bucket") or "unknown")].append(record)
    for bucket, group in sorted(grouped.items()):
        flags = [actual_flags(record) for record in group]
        rows.append(
            {
                "bucket": bucket,
                "records": len(group),
                "actual_high_regret_count": sum(1 for f in flags if f["actual_high_regret"]),
                "actual_high_regret_rate": mean(f["actual_high_regret"] for f in flags),
                "actual_low_margin_count": sum(1 for f in flags if f["actual_low_margin"]),
                "actual_low_margin_rate": mean(f["actual_low_margin"] for f in flags),
                "actual_teacher_disagreement_count": sum(1 for f in flags if f["actual_teacher_disagreement"]),
                "actual_teacher_disagreement_rate": mean(f["actual_teacher_disagreement"] for f in flags),
            }
        )
    return rows


def read_pool_stats(pool_dir: Path | None) -> list[dict[str, Any]]:
    if pool_dir is None:
        return []
    rows = []
    for bucket, filename in POOL_FILES.items():
        path = pool_dir / filename
        records = read_jsonl(path)
        seats = Counter(str(record.get("seat", "unknown")) for record in records)
        rows.append(
            {
                "pool": bucket,
                "path": str(path),
                "records": len(records),
                "first": seats.get("first", 0),
                "second": seats.get("second", 0),
                "first_second_ratio": seats.get("first", 0) / max(seats.get("second", 0), 1),
                "predicted_delta_mean": mean(safe_float(record.get("predicted_delta_vs_baseline")) for record in records),
                "predicted_margin_mean": mean(safe_float(record.get("predicted_margin")) for record in records),
            }
        )
    return rows


def read_attempt_stats(summary_dir: Path | None, pool_dir: Path | None) -> list[dict[str, Any]]:
    rows = []
    paths: list[Path] = []
    if summary_dir and summary_dir.exists():
        paths.extend(sorted(summary_dir.glob("*.json")))
    if pool_dir and pool_dir.exists():
        paths.extend(sorted(pool_dir.glob("*.summary.json")))
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        attempt = payload.get("attempt_profile", {})
        rows.append(
            {
                "file": str(path),
                "source_bucket": payload.get("source_bucket", ""),
                "source_bucket_requested": payload.get("source_bucket_requested", ""),
                "source_bucket_actual": payload.get("source_bucket_actual", ""),
                "samples": payload.get("samples", ""),
                "attempts": payload.get("attempts", ""),
                "elapsed_seconds": safe_float(payload.get("elapsed_seconds")),
                "seconds_per_state": safe_float(payload.get("seconds_per_state")),
                "attempt_count": safe_int(attempt.get("attempt_count")),
                "accepted_count": safe_int(attempt.get("accepted_count")),
                "skipped_count": safe_int(attempt.get("skipped_count")),
                "candidate_pool_written": safe_int(payload.get("candidate_pool_written")),
                "candidate_pool_prefilter": payload.get("candidate_pool_prefilter", ""),
            }
        )
    return rows


def estimate_rows(records: list[dict[str, Any]], attempt_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    profile_seconds = mean(safe_float(record.get("profiling", {}).get("seconds_total")) for record in records)
    wall_seconds = mean(row["seconds_per_state"] for row in attempt_rows if safe_float(row.get("seconds_per_state")) > 0.0)
    base = wall_seconds or profile_seconds
    rows = []
    for workers in (1, 20, 50, 100):
        rows.append(
            {
                "basis": "wall_seconds_per_written" if wall_seconds else "profile_seconds_per_state",
                "seconds_per_state": base,
                "target_states": 50_000,
                "workers": workers,
                "estimated_hours": 50_000 * base / max(workers, 1) / 3600.0,
            }
        )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("_run_bucket") or "unknown")].append(record)
    for bucket, group in sorted(grouped.items()):
        seconds = mean(safe_float(record.get("profiling", {}).get("seconds_total")) for record in group)
        rows.append(
            {
                "basis": f"bucket_profile:{bucket}",
                "seconds_per_state": seconds,
                "target_states": 10_000,
                "workers": 20,
                "estimated_hours": 10_000 * seconds / 20.0 / 3600.0,
            }
        )
    return rows


def write_markdown(
    output_dir: Path,
    *,
    records: list[dict[str, Any]],
    check: dict[str, Any],
    signal: dict[str, Any],
    bucket_rows: list[dict[str, Any]],
    position_rows: list[dict[str, Any]],
    estimate: list[dict[str, Any]],
    run_name: str,
) -> None:
    first = next((row for row in position_rows if row.get("group") == "first"), {})
    second = next((row for row in position_rows if row.get("group") == "second"), {})
    first_second_ok = bool(first and second and 0.5 <= (float(first.get("records", 0)) / max(float(second.get("records", 0)), 1.0)) <= 2.0)
    enough_signal = signal.get("positive", 0) > 0 and signal.get("gray", 0) > 0 and signal.get("negative", 0) > 0
    go_feature_cache = bool(check["ok"] and enough_signal and first_second_ok)
    rows = [
        "# HU Turn2 Stage8 20k MC512 Broad Pass",
        "",
        f"- run_name: `{run_name}`",
        f"- records: `{len(records)}`",
        f"- correctness_ok: `{check['ok']}`",
        f"- missing: `{check['missing']}`",
        f"- common_random_future_digest_mismatch: `{check['common_random_future_digest_mismatch']}`",
        f"- action_legal_failures: `{check['action_legal_failures']}`",
        f"- Stage3 fallback recompute: `{check['stage3_fallback_recomputed_count']}`",
        f"- positive/gray/negative: `{signal.get('positive', 0)} / {signal.get('gray', 0)} / {signal.get('negative', 0)}`",
        f"- delta_mean: `{float(signal.get('delta_mean', 0.0)):.4f}`",
        f"- margin_mean: `{float(signal.get('margin_mean', 0.0)):.4f}`",
        f"- delta>=2SE rate: `{float(signal.get('delta_ge_2se_rate', 0.0)):.4f}`",
        f"- first records: `{first.get('records', 0)}`",
        f"- second records: `{second.get('records', 0)}`",
        "",
        "## Bucket Counts",
        "",
        "| bucket | records | positive | gray | negative | high_regret_rate | low_margin_rate | disagreement_rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in bucket_rows:
        rows.append(
            "| {bucket} | {records} | {positive} | {gray} | {negative} | {actual_high_regret_rate:.4f} | {actual_low_margin_rate:.4f} | {actual_teacher_disagreement_rate:.4f} |".format(
                **row
            )
        )
    rows.extend(
        [
            "",
            "## Go / No-Go",
            "",
            "GO for feature cache and Stage8 broad training." if go_feature_cache else "NO-GO for feature cache until the listed correctness/signal issue is fixed.",
            "",
            "Do not move to T1 or production candidate deployment from this result alone.",
            "",
            "## 50k Estimate",
            "",
            "| basis | workers | hours |",
            "|---|---:|---:|",
        ]
    )
    for row in estimate:
        if int(row.get("target_states", 0)) == 50_000:
            rows.append(f"| {row['basis']} | {row['workers']} | {float(row['estimated_hours']):.2f} |")
    (output_dir / "summary.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (output_dir / "go_nogo.md").write_text(
        "\n".join(
            [
                "# Go / No-Go",
                "",
                f"- clean_20k: `{check['ok']}`",
                f"- enough_teacher_signal: `{enough_signal}`",
                f"- first_second_ok: `{first_second_ok}`",
                f"- feature_cache_go: `{go_feature_cache}`",
                f"- stage8_broad_training_go: `{go_feature_cache}`",
                "- 50k_full_pass: `defer until broad training/calibration result`",
                "- selected_high_mc_refinement: `evaluate after Stage8 broad calibration`",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    records = read_teacher_records(args.teacher, output_dir)
    check = correctness(records, args.expected_total, args.expected_per_bucket, args.future_samples)
    bucket_rows = bucket_breakdown(records)
    position_rows = position_breakdown(records)
    signal_rows = [signal_row("all", records)]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("_run_bucket") or "unknown")].append(record)
    for bucket, group in sorted(grouped.items()):
        signal_rows.append(signal_row(bucket, group))
    enrich_rows = enrichment_rows(records)
    pool_rows = read_pool_stats(args.pool_dir)
    attempt_rows = read_attempt_stats(args.summary_dir, args.pool_dir)
    estimate = estimate_rows(records, attempt_rows)

    (output_dir / "correctness.json").write_text(json.dumps(check, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(output_dir / "bucket_breakdown.csv", bucket_rows)
    write_csv(output_dir / "position_breakdown.csv", position_rows)
    write_csv(output_dir / "teacher_signal_stats.csv", signal_rows)
    write_csv(output_dir / "actual_bucket_enrichment.csv", enrich_rows)
    write_csv(output_dir / "pool_stats.csv", pool_rows)
    write_csv(output_dir / "attempt_acceptance_stats.csv", attempt_rows)
    write_csv(output_dir / "fifty_k_estimates.csv", estimate)
    write_csv(output_dir / "state_profile.csv", state_profile(records))
    write_markdown(
        output_dir,
        records=records,
        check=check,
        signal=signal_rows[0],
        bucket_rows=bucket_rows,
        position_rows=position_rows,
        estimate=estimate,
        run_name=args.run_name,
    )
    print(json.dumps({"records": len(records), "correctness_ok": check["ok"], "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
