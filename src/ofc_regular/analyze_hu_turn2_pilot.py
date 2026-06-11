"""Analyze HU Turn2 pilot teacher data and MC stability."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mc128", type=Path, required=True)
    parser.add_argument("--mc512", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--requested-mc128", type=int, default=0)
    parser.add_argument("--requested-mc512", type=int, default=0)
    return parser.parse_args()


def read_records(path: Path) -> list[dict[str, Any]]:
    paths = []
    if path.is_dir():
        paths = sorted(path.rglob("*.jsonl"))
    else:
        paths = [path]
    records = []
    for item in paths:
        if not item.exists():
            continue
        with item.open("r", encoding="utf-8-sig") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
    return records


def q(values: Iterable[float], quantile: float) -> float:
    materialized = np.asarray([float(value) for value in values], dtype=np.float64)
    if materialized.size == 0:
        return 0.0
    return float(np.quantile(materialized, quantile))


def mean(values: Iterable[float]) -> float:
    materialized = [float(value) for value in values]
    return float(sum(materialized) / len(materialized)) if materialized else 0.0


def record_key(record: dict[str, Any]) -> str:
    payload = {
        "source_bucket": record.get("source_bucket"),
        "hand_seed": record.get("hand_seed"),
        "player": record.get("player"),
        "board": record.get("board"),
        "opponent_board": record.get("opponent_board"),
        "dealt": record.get("dealt"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def action_key(action: dict[str, Any]) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
    return (
        tuple((str(card), str(row)) for card, row in action.get("placements", ())),
        tuple(str(card) for card in action.get("discards", ())),
    )


def summarize_dataset(name: str, records: list[dict[str, Any]], requested: int) -> list[dict[str, Any]]:
    action_counts = [len(record.get("actions", ())) for record in records]
    regrets = [float(record.get("delta_best_vs_baseline", 0.0)) for record in records]
    margins = [float(record.get("best_margin", 0.0)) for record in records]
    se_actions = [
        float(action.get("ev_standard_error", 0.0))
        for record in records
        for action in record.get("actions", ())
    ]
    se_deltas = [float(record.get("SE_delta_best_vs_baseline", 0.0)) for record in records]
    source_counts = Counter(str(record.get("source_bucket", "unknown")) for record in records)
    missing = sum(int(record.get("missing", 0) or 0) for record in records)
    invalid_states = sum(1 for record in records if record.get("phase") != "hu_turn2_7card")
    invalid_actions = sum(
        1
        for record in records
        for action in record.get("actions", ())
        if not np.isfinite(float(action.get("score", 0.0))) or not np.isfinite(float(action.get("ev_standard_error", 0.0)))
    )
    rows = [
        row(name, "samples_requested", requested or len(records)),
        row(name, "samples_written", len(records)),
        row(name, "missing", missing),
        row(name, "failed_states", max((requested or len(records)) - len(records), 0)),
        row(name, "invalid_states", invalid_states),
        row(name, "invalid_actions", invalid_actions),
        row(name, "legal_actions_mean", mean(action_counts)),
        row(name, "legal_actions_median", q(action_counts, 0.5)),
        row(name, "legal_actions_p90", q(action_counts, 0.9)),
        row(name, "legal_actions_p99", q(action_counts, 0.99)),
        row(name, "legal_actions_max", max(action_counts) if action_counts else 0),
        row(name, "baseline_avg_regret", mean(regrets)),
        row(name, "baseline_p50_regret", q(regrets, 0.5)),
        row(name, "baseline_p90_regret", q(regrets, 0.9)),
        row(name, "baseline_p95_regret", q(regrets, 0.95)),
        row(name, "baseline_p99_regret", q(regrets, 0.99)),
        row(name, "best_second_margin_mean", mean(margins)),
        row(name, "best_second_margin_median", q(margins, 0.5)),
        row(name, "best_second_margin_p25", q(margins, 0.25)),
        row(name, "best_second_margin_p75", q(margins, 0.75)),
        row(name, "best_second_margin_p90", q(margins, 0.9)),
        row(name, "delta_best_vs_baseline_mean", mean(regrets)),
        row(name, "delta_best_vs_baseline_median", q(regrets, 0.5)),
        row(name, "delta_best_vs_baseline_p90", q(regrets, 0.9)),
        row(name, "delta_best_vs_baseline_p95", q(regrets, 0.95)),
        row(name, "delta_best_vs_baseline_p99", q(regrets, 0.99)),
        row(name, "positive_rate_delta_ge_0_25", rate(regrets, 0.25)),
        row(name, "positive_rate_delta_ge_0_35", rate(regrets, 0.35)),
        row(name, "positive_rate_delta_ge_0_50", rate(regrets, 0.50)),
        row(name, "positive_rate_delta_ge_1_00", rate(regrets, 1.00)),
        row(name, "action_ev_se_mean", mean(se_actions)),
        row(name, "action_ev_se_median", q(se_actions, 0.5)),
        row(name, "action_ev_se_p90", q(se_actions, 0.9)),
        row(name, "action_ev_se_p95", q(se_actions, 0.95)),
        row(name, "action_ev_se_p99", q(se_actions, 0.99)),
        row(name, "delta_se_mean", mean(se_deltas)),
        row(name, "delta_se_median", q(se_deltas, 0.5)),
        row(name, "delta_se_p90", q(se_deltas, 0.9)),
        row(name, "delta_se_p95", q(se_deltas, 0.95)),
        row(name, "delta_se_p99", q(se_deltas, 0.99)),
        row(name, "delta_ge_2se_rate", se_rate(records, 2.0)),
        row(name, "delta_ge_2_5se_rate", se_rate(records, 2.5)),
        row(name, "delta_ge_3se_rate", se_rate(records, 3.0)),
        row(name, "t3_stage7_fired_rate", mean(record.get("downstream_features", {}).get("t3_stage7_override_rate", 0.0) for record in records)),
    ]
    for source, count in sorted(source_counts.items()):
        rows.append(row(name, f"source_count_{source}", count))
    return rows


def row(dataset: str, metric: str, value: Any) -> dict[str, Any]:
    return {"dataset": dataset, "metric": metric, "value": value}


def rate(values: list[float], threshold: float) -> float:
    return sum(1 for value in values if value >= threshold) / len(values) if values else 0.0


def se_rate(records: list[dict[str, Any]], multiplier: float) -> float:
    if not records:
        return 0.0
    count = 0
    for record in records:
        delta = float(record.get("delta_best_vs_baseline", 0.0))
        se = float(record.get("SE_delta_best_vs_baseline", 0.0))
        if se > 0.0 and delta >= multiplier * se:
            count += 1
    return count / len(records)


def stability_rows(mc128: list[dict[str, Any]], mc512: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key_512 = {record_key(record): record for record in mc512}
    matched = [(record, by_key_512[record_key(record)]) for record in mc128 if record_key(record) in by_key_512]
    rows = []
    if not matched:
        return [row("stability", "matched_states", 0)]
    sign_matches = []
    top1_matches = []
    top3_overlaps = []
    ev_corrs = []
    delta_pairs = []
    bucket_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for a, b in matched:
        delta_a = float(a.get("delta_best_vs_baseline", 0.0))
        delta_b = float(b.get("delta_best_vs_baseline", 0.0))
        sign_match = sign(delta_a) == sign(delta_b)
        sign_matches.append(sign_match)
        bucket = margin_bucket(max(abs(delta_a), abs(delta_b)))
        bucket_counts[bucket][0] += int(sign_match)
        bucket_counts[bucket][1] += 1
        top1_matches.append(best_original_index(a) == best_original_index(b))
        top3_a = set(top_original_indices(a, 3))
        top3_b = set(top_original_indices(b, 3))
        top3_overlaps.append(len(top3_a & top3_b) / max(len(top3_a | top3_b), 1))
        ev_a, ev_b = matched_action_evs(a, b)
        if len(ev_a) >= 2:
            ev_corrs.append(corr(ev_a, ev_b))
        delta_pairs.append((delta_a, delta_b))
    rows.extend(
        [
            row("stability", "matched_states", len(matched)),
            row("stability", "sign_agreement_delta_best_vs_baseline", mean(sign_matches)),
            row("stability", "rank_top1_agreement", mean(top1_matches)),
            row("stability", "top3_overlap_jaccard_mean", mean(top3_overlaps)),
            row("stability", "ev_per_action_correlation_mean", mean(ev_corrs)),
            row("stability", "delta_vs_baseline_correlation", corr([a for a, _ in delta_pairs], [b for _, b in delta_pairs])),
        ]
    )
    for bucket, (matches, total) in sorted(bucket_counts.items()):
        rows.append(row("stability", f"sign_agreement_bucket_{bucket}", matches / total if total else 0.0))
        rows.append(row("stability", f"sign_count_bucket_{bucket}", total))
    return rows


def sign(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def margin_bucket(value: float) -> str:
    if value < 0.05:
        return "lt_0_05"
    if value < 0.25:
        return "0_05_0_25"
    if value < 0.5:
        return "0_25_0_50"
    if value < 1.0:
        return "0_50_1_00"
    return "ge_1_00"


def best_original_index(record: dict[str, Any]) -> int:
    actions = record.get("actions", ())
    return int(actions[0].get("original_index", -1)) if actions else -1


def top_original_indices(record: dict[str, Any], count: int) -> list[int]:
    return [int(action.get("original_index", -1)) for action in record.get("actions", ())[:count]]


def matched_action_evs(a: dict[str, Any], b: dict[str, Any]) -> tuple[list[float], list[float]]:
    b_actions = {action_key(action): float(action.get("score", 0.0)) for action in b.get("actions", ())}
    values_a = []
    values_b = []
    for action in a.get("actions", ()):
        key = action_key(action)
        if key in b_actions:
            values_a.append(float(action.get("score", 0.0)))
            values_b.append(b_actions[key])
    return values_a, values_b


def corr(a: Iterable[float], b: Iterable[float]) -> float:
    array_a = np.asarray(list(a), dtype=np.float64)
    array_b = np.asarray(list(b), dtype=np.float64)
    if array_a.size < 2 or array_b.size < 2 or np.std(array_a) < 1e-12 or np.std(array_b) < 1e-12:
        return 0.0
    return float(np.corrcoef(array_a, array_b)[0, 1])


def speed_profile(records: list[dict[str, Any]]) -> dict[str, Any]:
    totals: Counter[str] = Counter()
    for record in records:
        for key, value in record.get("profiling", {}).items():
            if isinstance(value, (int, float)):
                totals[key] += float(value)
    state_count = len(records)
    future_count = sum(int(record.get("rollout_count", 0) or 0) for record in records)
    action_count = sum(len(record.get("actions", ())) for record in records)
    return {
        "states": state_count,
        "future_samples": future_count,
        "legal_actions": action_count,
        "seconds_per_state": totals.get("seconds_total", 0.0) / max(state_count, 1),
        "seconds_per_future_sample": totals.get("rollout_seconds", 0.0) / max(future_count, 1),
        "seconds_per_legal_action": totals.get("rollout_seconds", 0.0) / max(action_count, 1),
        "timing_totals": dict(totals),
        "timing_per_state": {key: value / max(state_count, 1) for key, value in totals.items()},
        "bottleneck": bottleneck_from_totals(totals),
    }


def bottleneck_from_totals(totals: Counter[str]) -> str:
    candidates = {
        key: value
        for key, value in totals.items()
        if key not in {"seconds_total"}
    }
    return max(candidates.items(), key=lambda item: item[1])[0] if candidates else "unknown"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    path: Path,
    *,
    mc128: list[dict[str, Any]],
    mc512: list[dict[str, Any]],
    stats_rows: list[dict[str, Any]],
    stability: list[dict[str, Any]],
    speed: dict[str, Any],
) -> None:
    metrics = {(item["dataset"], item["metric"]): item["value"] for item in [*stats_rows, *stability]}
    go = bool(
        metrics.get(("mc512", "missing"), 1) == 0
        and metrics.get(("stability", "matched_states"), 0) > 0
        and metrics.get(("stability", "sign_agreement_bucket_ge_1_00"), 0) >= 0.8
        and metrics.get(("mc512", "positive_rate_delta_ge_0_25"), 0) > 0.01
        and speed.get("seconds_per_state", 999.0) < 60.0
    )
    lines = [
        "# HU Turn2 Pilot Summary",
        "",
        f"- MC128 samples: `{len(mc128)}`",
        f"- MC512 samples: `{len(mc512)}`",
        f"- matched states: `{metrics.get(('stability', 'matched_states'), 0)}`",
        f"- MC512 missing: `{metrics.get(('mc512', 'missing'), 0)}`",
        f"- MC512 positive delta>=0.25 rate: `{float(metrics.get(('mc512', 'positive_rate_delta_ge_0_25'), 0.0)):.4f}`",
        f"- delta sign agreement: `{float(metrics.get(('stability', 'sign_agreement_delta_best_vs_baseline'), 0.0)):.4f}`",
        f"- high-margin sign agreement: `{float(metrics.get(('stability', 'sign_agreement_bucket_ge_1_00'), 0.0)):.4f}`",
        f"- seconds/state: `{float(speed.get('seconds_per_state', 0.0)):.4f}`",
        f"- bottleneck: `{speed.get('bottleneck', 'unknown')}`",
        "",
        "## Go / No-Go",
        "",
        "GO for 50k x MC4096." if go else "NO-GO for 50k x MC4096 with the current Python path.",
        "",
        "If NO-GO, prioritize batch/Rust rollout and two-stage MC512 -> MC4096 refinement before full teacher generation.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    mc128 = read_records(args.mc128)
    mc512 = read_records(args.mc512)
    stats = [
        *summarize_dataset("mc128", mc128, args.requested_mc128),
        *summarize_dataset("mc512", mc512, args.requested_mc512),
    ]
    stability = stability_rows(mc128, mc512)
    speed = {
        "mc128": speed_profile(mc128),
        "mc512": speed_profile(mc512),
    }
    combined_speed = speed_profile([*mc128, *mc512])
    speed["combined"] = combined_speed
    write_csv(output_dir / "hu_turn2_pilot_stats.csv", stats)
    write_csv(output_dir / "hu_turn2_mc128_vs_mc512_stability.csv", stability)
    (output_dir / "hu_turn2_speed_profile.json").write_text(json.dumps(speed, indent=2) + "\n", encoding="utf-8")
    write_markdown(
        output_dir / "hu_turn2_pilot_summary.md",
        mc128=mc128,
        mc512=mc512,
        stats_rows=stats,
        stability=stability,
        speed=combined_speed,
    )
    (output_dir / "hu_turn2_go_nogo.md").write_text(
        (output_dir / "hu_turn2_pilot_summary.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    print(json.dumps({"mc128": len(mc128), "mc512": len(mc512), "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
