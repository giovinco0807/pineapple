"""Compare T0 MC32/MC512 pair labels and build selector training datasets."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL") from exc
    return rows


def _index(rows: Iterable[dict[str, Any]], *, name: str) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        target_id = str(row.get("target_id") or "")
        if not target_id:
            raise ValueError(f"{name} row is missing target_id")
        if target_id in output:
            raise ValueError(f"duplicate {name} target_id: {target_id}")
        output[target_id] = row
    return output


def _finite_float(value: Any) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite replay value: {value!r}")
    return parsed


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    x_mean = mean(xs)
    y_mean = mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_norm = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_norm = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    return numerator / (x_norm * y_norm) if x_norm > 0.0 and y_norm > 0.0 else 0.0


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare_dataset(
    *,
    mc32_rows: list[dict[str, Any]],
    mc512_rows: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    mc32 = _index(mc32_rows, name="MC32")
    mc512 = _index(mc512_rows, name="MC512")
    missing_from_mc32 = sorted(set(mc512) - set(mc32))
    if missing_from_mc32:
        raise ValueError(f"MC512 targets missing from MC32: {len(missing_from_mc32)}")

    label_matrix: Counter[tuple[str, str]] = Counter()
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    deltas32: list[float] = []
    deltas512: list[float] = []
    sign_agree = 0
    label_agree = 0
    comparison_rows: list[dict[str, Any]] = []
    for target_id in sorted(mc512):
        low = mc32[target_id]
        high = mc512[target_id]
        low_delta = _finite_float(low.get("candidate_delta_vs_baseline"))
        high_delta = _finite_float(high.get("candidate_delta_vs_baseline"))
        low_label = str(low.get("safe_override_label"))
        high_label = str(high.get("safe_override_label"))
        low_sign = 1 if low_delta > 0.0 else -1 if low_delta < 0.0 else 0
        high_sign = 1 if high_delta > 0.0 else -1 if high_delta < 0.0 else 0
        sign_agree += int(low_sign == high_sign)
        label_agree += int(low_label == high_label)
        label_matrix[(low_label, high_label)] += 1
        deltas32.append(low_delta)
        deltas512.append(high_delta)
        source = str(high.get("source_config_id") or "unknown")
        comparison = {
            "target_id": target_id,
            "source_config_id": source,
            "seat": high.get("seat"),
            "predicted_margin": high.get("predicted_margin"),
            "mc32_label": low_label,
            "mc512_label": high_label,
            "mc32_delta": low_delta,
            "mc512_delta": high_delta,
            "delta_shift": high_delta - low_delta,
            "sign_agreement": low_sign == high_sign,
            "label_agreement": low_label == high_label,
        }
        comparison_rows.append(comparison)
        by_source[source].append(comparison)

    broad_refined: list[dict[str, Any]] = []
    for target_id in sorted(mc32):
        if target_id in mc512:
            row = dict(mc512[target_id])
            row["selector_label_source"] = "mc512_refinement"
            row["selector_label_future_samples"] = 512
        else:
            row = dict(mc32[target_id])
            row["selector_label_source"] = "mc32_broad"
            row["selector_label_future_samples"] = 32
        broad_refined.append(row)

    high_mc_rows = []
    for target_id in sorted(mc512):
        row = dict(mc512[target_id])
        row["selector_label_source"] = "mc512_refinement"
        row["selector_label_future_samples"] = 512
        high_mc_rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "selector_high_mc_only.jsonl", high_mc_rows)
    _write_jsonl(output_dir / "selector_broad_with_mc512_overrides.jsonl", broad_refined)
    _write_csv(output_dir / "mc32_vs_mc512_rows.csv", comparison_rows)
    transition_rows = [
        {"mc32_label": low, "mc512_label": high, "count": count}
        for (low, high), count in sorted(label_matrix.items())
    ]
    _write_csv(output_dir / "label_transition.csv", transition_rows)
    source_rows = []
    for source, rows in sorted(by_source.items()):
        source_rows.append(
            {
                "source_config_id": source,
                "rows": len(rows),
                "sign_agreement_rate": sum(row["sign_agreement"] for row in rows) / len(rows),
                "label_agreement_rate": sum(row["label_agreement"] for row in rows) / len(rows),
                "mc512_positive": sum(row["mc512_label"] == "positive" for row in rows),
                "mc512_negative": sum(row["mc512_label"] == "negative" for row in rows),
                "mc512_gray": sum(row["mc512_label"] == "gray" for row in rows),
                "mc512_delta_mean": mean(row["mc512_delta"] for row in rows),
            }
        )
    _write_csv(output_dir / "source_breakdown.csv", source_rows)

    mc32_positive_ids = {
        target_id for target_id, row in mc32.items() if row.get("safe_override_label") == "positive"
    }
    refined_mc32_positive_ids = mc32_positive_ids & set(mc512)
    survived_positive = sum(
        mc512[target_id].get("safe_override_label") == "positive"
        for target_id in refined_mc32_positive_ids
    )
    summary = {
        "schema": "hu_turn0_safe_selector_dataset_v1",
        "mc32_rows": len(mc32),
        "mc512_rows": len(mc512),
        "mc512_missing_from_mc32": len(missing_from_mc32),
        "sign_agreement_rate": sign_agree / len(mc512) if mc512 else 0.0,
        "label_agreement_rate": label_agree / len(mc512) if mc512 else 0.0,
        "delta_pearson": _pearson(deltas32, deltas512),
        "mc32_positive_refined": len(refined_mc32_positive_ids),
        "mc32_positive_survived_mc512": survived_positive,
        "mc32_positive_survival_rate": (
            survived_positive / len(refined_mc32_positive_ids)
            if refined_mc32_positive_ids
            else 0.0
        ),
        "mc512_labels": dict(
            sorted(Counter(str(row.get("safe_override_label")) for row in mc512.values()).items())
        ),
        "broad_refined_rows": len(broad_refined),
        "high_mc_only_rows": len(high_mc_rows),
        "action_mapping_verified": all(
            bool(row.get("action_mapping_verified")) for row in mc512.values()
        ),
        "common_random_futures_verified": all(
            bool(row.get("common_random_futures_verified")) for row in mc512.values()
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mc32", type=Path, required=True)
    parser.add_argument("--mc512", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = prepare_dataset(
        mc32_rows=read_jsonl(args.mc32),
        mc512_rows=read_jsonl(args.mc512),
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
