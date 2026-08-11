"""Pool independent HU T0 paired-replay chunks into one high-MC label."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _signature(action: dict[str, Any]) -> str:
    return json.dumps(
        {
            "placements": action.get("placements") or (),
            "discards": action.get("discards") or (),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _pooled_mean_se(rows: list[dict[str, Any]]) -> tuple[int, float, float]:
    counts = [int(row["future_samples"]) for row in rows]
    means = [float(row["candidate_delta_vs_baseline"]) for row in rows]
    ses = [float(row["candidate_delta_se_vs_baseline"]) for row in rows]
    total = sum(counts)
    if total <= 0:
        raise ValueError("pooled replay count must be positive")
    mean = sum(count * value for count, value in zip(counts, means)) / total
    m2 = 0.0
    for count, chunk_mean, chunk_se in zip(counts, means, ses):
        if count <= 0 or not math.isfinite(chunk_mean) or not math.isfinite(chunk_se):
            raise ValueError("invalid chunk mean/SE")
        sample_variance = chunk_se * chunk_se * count
        m2 += max(0, count - 1) * sample_variance
        m2 += count * (chunk_mean - mean) ** 2
    variance = m2 / max(total - 1, 1)
    return total, mean, math.sqrt(max(0.0, variance) / total)


def combine_chunks(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    chunks = list(rows)
    if not chunks:
        raise ValueError("no T0 replay chunks")
    first = chunks[0]
    target_id = str(first.get("target_id") or "")
    baseline_index = int(first["baseline_action_index"])
    candidate_index = int(first["candidate_action_index"])
    baseline_signature = _signature(first["baseline_action"])
    candidate_signature = _signature(first["candidate_action"])
    for row in chunks:
        if str(row.get("target_id") or "") != target_id:
            raise ValueError("replay chunks contain multiple target_id values")
        if int(row["baseline_action_index"]) != baseline_index:
            raise ValueError("baseline action index changed across chunks")
        if int(row["candidate_action_index"]) != candidate_index:
            raise ValueError("candidate action index changed across chunks")
        if _signature(row["baseline_action"]) != baseline_signature:
            raise ValueError("baseline action mapping changed across chunks")
        if _signature(row["candidate_action"]) != candidate_signature:
            raise ValueError("candidate action mapping changed across chunks")
        if not row.get("common_random_futures_verified"):
            raise ValueError("chunk did not verify common random futures")
        if not row.get("action_mapping_verified"):
            raise ValueError("chunk did not verify action mapping")

    total, delta, delta_se = _pooled_mean_se(chunks)
    baseline_ev = sum(
        int(row["future_samples"]) * float(row["baseline_ev"]) for row in chunks
    ) / total
    candidate_ev = sum(
        int(row["future_samples"]) * float(row["candidate_ev"]) for row in chunks
    ) / total
    lcb164 = delta - 1.64 * delta_se
    lcb196 = delta - 1.96 * delta_se
    ucb196 = delta + 1.96 * delta_se
    if lcb196 > 0.0:
        label, label_id = "positive", 1
    elif ucb196 < 0.0 or delta <= -0.25:
        label, label_id = "negative", 0
    else:
        label, label_id = "gray", -1
    digests = [str(row["common_random_future_digest"]) for row in chunks]
    combined_digest = hashlib.sha256("|".join(digests).encode("utf-8")).hexdigest()
    output = dict(first)
    output.update(
        {
            "future_samples": total,
            "replay_seed": int(first["replay_seed"]),
            "replay_seed_chunks": [int(row["replay_seed"]) for row in chunks],
            "future_seed_chunks": [int(row["future_seed"]) for row in chunks],
            "chunk_future_samples": [int(row["future_samples"]) for row in chunks],
            "chunk_count": len(chunks),
            "pooled_from_independent_chunks": True,
            "common_random_future_digest": combined_digest,
            "common_random_future_digests": digests,
            "common_random_futures_verified": True,
            "action_mapping_verified": True,
            "baseline_ev": baseline_ev,
            "candidate_ev": candidate_ev,
            "candidate_delta_vs_baseline": delta,
            "candidate_delta_se_vs_baseline": delta_se,
            "candidate_delta_z_vs_baseline": delta / max(delta_se, 1.0e-9),
            "candidate_delta_lcb164": lcb164,
            "candidate_delta_lcb196": lcb196,
            "candidate_delta_ucb196": ucb196,
            "safe_override_label": label,
            "safe_override_label_id": label_id,
            "replay_seconds": sum(float(row.get("replay_seconds", 0.0)) for row in chunks),
        }
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args()
    rows = [row for path in args.input for row in read_jsonl(path)]
    combined = combine_chunks(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(combined, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema": "hu_turn0_replay_chunk_pool_v1",
        "target_id": combined["target_id"],
        "chunk_count": combined["chunk_count"],
        "future_samples": combined["future_samples"],
        "candidate_delta_vs_baseline": combined["candidate_delta_vs_baseline"],
        "candidate_delta_se_vs_baseline": combined[
            "candidate_delta_se_vs_baseline"
        ],
        "safe_override_label": combined["safe_override_label"],
        "common_random_futures_verified": combined[
            "common_random_futures_verified"
        ],
        "action_mapping_verified": combined["action_mapping_verified"],
    }
    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
