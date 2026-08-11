"""Build a feature cache for HU Turn2 pilot teacher data.

This cache is intentionally pilot-oriented: it preserves state-level split
metadata and all action-row labels needed to validate EV, delta, rank, and
selective-override gate training before larger teacher generation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .action_key import ACTION_KEY_SCHEMA, action_key_from_payload
from .hu_infoset import (
    OBSERVATION_SCHEMA,
    POLICY_FEATURE_SAMPLE_SCHEMA,
    ScoringContext,
    actor_observation_from_record,
    policy_feature_sample_from_record,
)
from .hu_turn3_model import HU_FEATURE_DIM, sample_to_matrix

FEATURE_CACHE_SCHEMA = "hu_turn2_pilot_feature_cache_v2"
FEATURE_VALUE_SCHEMA = "hu_turn2_hu_action_features_v1"


def _digest_payload(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


REGULAR_RULES_DIGEST = _digest_payload(
    {
        "rule_set": "heads_up_regular_ofc_pineapple_v1",
        "rows": {"top": 3, "middle": 5, "bottom": 5},
        "middle_trips_royalty": 2,
        "fantasyland_cards": 14,
        "scoring": ScoringContext().to_dict(),
    }
)

RUN_BUCKET_FILES = (
    "natural.jsonl",
    "predicted_high_regret_from_pool.jsonl",
    "predicted_low_margin_from_pool.jsonl",
    "predicted_teacher_disagreement_from_pool.jsonl",
    "random_off_policy.jsonl",
)

SPLIT_NAME_TO_ID = {"train": 0, "val": 1, "test": 2}
SPLIT_ID_TO_NAME = {value: key for key, value in SPLIT_NAME_TO_ID.items()}
GATE_LABEL_TO_ID = {"negative": 0, "gray": 1, "positive": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="Merged HU T2 pilot teacher JSONL. Can be repeated to combine teacher/replay files.",
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--bucket-sidecar-dir",
        type=Path,
        help="Directory containing per-bucket JSONL files. Defaults to the input parent when present.",
    )
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--seed", type=int, default=2026061701)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-records", type=int)
    return parser.parse_args()


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _count_jsonl(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def resolve_single_input_file(input_path: Path, sidecar_dir: Path | None) -> list[tuple[Path, str]]:
    """Prefer per-bucket sidecars when they exactly cover the merged input."""

    if input_path.is_dir():
        files = [(path, path.stem) for path in sorted(input_path.glob("*.jsonl"))]
        if not files:
            raise FileNotFoundError(f"no jsonl files under {input_path}")
        return files

    candidate_dir = sidecar_dir or input_path.parent
    sidecars = [(candidate_dir / name, Path(name).stem) for name in RUN_BUCKET_FILES]
    if all(path.exists() for path, _bucket in sidecars):
        merged_count = _count_jsonl(input_path)
        sidecar_count = sum(_count_jsonl(path) for path, _bucket in sidecars)
        if merged_count == sidecar_count:
            return sidecars
    return [(input_path, input_path.stem)]


def resolve_input_files(input_paths: Iterable[Path], sidecar_dir: Path | None) -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    for input_path in input_paths:
        files.extend(resolve_single_input_file(input_path, sidecar_dir))
    return files


def stable_state_hash(sample: dict[str, Any]) -> str:
    """Actor-information-set identity; replay truth cannot affect this hash."""
    return actor_observation_from_record(sample).fingerprint()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def action_record_mapping_digests(
    actions: Iterable[dict[str, Any]],
) -> tuple[str, str]:
    rows = list(actions)
    indices = [action_original_index(action) for action in rows]
    if any(index is None for index in indices) or sorted(indices) != list(range(len(rows))):
        raise ValueError("feature cache requires a complete legal original-index mapping")
    ordered_rows = [
        row
        for _index, row in sorted(
            zip((int(index) for index in indices), rows), key=lambda item: item[0]
        )
    ]
    tokens = [action_key_from_payload(action).to_token() for action in ordered_rows]
    if len(tokens) != len(set(tokens)):
        raise ValueError("duplicate ActionKey in teacher legal actions")
    order_digest = hashlib.sha256("\n".join(tokens).encode("ascii")).hexdigest()
    set_digest = hashlib.sha256(
        "\n".join(sorted(tokens)).encode("ascii")
    ).hexdigest()
    return set_digest, order_digest


def cache_identity_payload(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": metadata.get("schema"),
        "observation_schema": metadata.get("observation_schema"),
        "policy_feature_sample_schema": metadata.get("policy_feature_sample_schema"),
        "action_key_schema": metadata.get("action_key_schema"),
        "feature_value_schema": metadata.get("feature_value_schema"),
        "rules_digest": metadata.get("rules_digest"),
        "feature_dim": metadata.get("feature_dim"),
        "feature_dtype": metadata.get("feature_dtype"),
        "state_count": metadata.get("state_count"),
        "action_count": metadata.get("action_count"),
        "input_files": metadata.get("input_files"),
        "observation_fingerprint_digest": metadata.get(
            "observation_fingerprint_digest"
        ),
        "legal_action_mapping_digest": metadata.get(
            "legal_action_mapping_digest"
        ),
    }


def action_original_index(action: Any) -> int | None:
    if isinstance(action, int):
        return int(action)
    if isinstance(action, dict) and action.get("original_index") is not None:
        return int(action["original_index"])
    if isinstance(action, dict) and isinstance(action.get("action"), dict):
        return action_original_index(action["action"])
    return None


def action_signature(action: dict[str, Any]) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
    if isinstance(action.get("action"), dict):
        action = action["action"]
    return (
        tuple(sorted((str(card), str(row)) for card, row in action.get("placements", ()))),
        tuple(sorted(str(card) for card in action.get("discards", ()))),
    )


def action_position(actions: list[dict[str, Any]], action: Any, *, fallback: int = 0) -> int:
    original = action_original_index(action)
    if isinstance(action, dict):
        wanted = action_signature(action)
        has_payload = bool(wanted[0] or wanted[1])
        explicit_key = action.get("canonical_action_key", action.get("action_key"))
        if has_payload:
            payload_key = action_key_from_payload(
                action["action"] if isinstance(action.get("action"), dict) else action
            ).to_token()
            if explicit_key is not None and explicit_key != payload_key:
                raise ValueError("action key disagrees with action payload")
            for index, candidate in enumerate(actions):
                if action_signature(candidate) == wanted:
                    if original is not None and action_original_index(candidate) != original:
                        raise ValueError(
                            "action original_index disagrees with semantic payload"
                        )
                    return index
            raise ValueError("semantic action is missing from legal action records")
    if original is not None:
        for index, candidate in enumerate(actions):
            if action_original_index(candidate) == original:
                return index
    return min(max(fallback, 0), max(len(actions) - 1, 0))


def pilot_gate_label(sample: dict[str, Any]) -> str:
    delta = float(sample.get("delta_best_vs_baseline", 0.0) or 0.0)
    se = float(sample.get("SE_delta_best_vs_baseline", 0.0) or 0.0)
    if delta >= 0.35 and (se <= 0.0 or delta >= 2.5 * se):
        return "positive"
    if delta <= 0.05:
        return "negative"
    return "gray"


def _coerce_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _fl_ev_from_canonical_key(canonical_key: Any, *, card_count: int = 14) -> float | None:
    if not isinstance(canonical_key, list) or not canonical_key:
        return None
    maybe_fl_ev = canonical_key[-1]
    if not isinstance(maybe_fl_ev, list):
        return None
    for item in maybe_fl_ev:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        try:
            count = int(item[0])
        except (TypeError, ValueError):
            continue
        if count == card_count:
            return _coerce_float(item[1])
    return None


def sample_fl_ev_14(sample: dict[str, Any]) -> float | None:
    """Infer the 14-card FL EV used by a teacher record, when it is recorded."""

    for key in ("fl_ev_14", "fantasyland_ev_14"):
        value = _coerce_float(sample.get(key))
        if value is not None:
            return value
    fl_ev = sample.get("fl_ev")
    if isinstance(fl_ev, dict):
        value = _coerce_float(fl_ev.get("14") if "14" in fl_ev else fl_ev.get(14))
        if value is not None:
            return value
    profiling = sample.get("profiling")
    if isinstance(profiling, dict):
        for row in profiling.get("_final_turn_slow_states", ()) or ():
            if not isinstance(row, dict):
                continue
            metadata = row.get("metadata")
            if not isinstance(metadata, dict):
                continue
            value = _fl_ev_from_canonical_key(metadata.get("canonical_key"))
            if value is not None:
                return value
    return None


def scoring_objective_metadata(samples: list[tuple[dict[str, Any], str]]) -> dict[str, Any]:
    values: Counter[str] = Counter()
    numeric_by_key: dict[str, float] = {}
    missing = 0
    for sample, _run_bucket in samples:
        value = sample_fl_ev_14(sample)
        if value is None:
            missing += 1
        else:
            key = f"{value:.17g}"
            values[key] += 1
            numeric_by_key.setdefault(key, value)
    numeric_values = sorted(numeric_by_key.values())
    return {
        "fl_ev_14": numeric_values[0] if len(numeric_values) == 1 and missing == 0 else None,
        "fl_ev_14_values": {key: int(count) for key, count in sorted(values.items())},
        "fl_ev_14_missing_records": missing,
        "fl_ev_14_recorded_records": len(samples) - missing,
        "fl_ev_14_status": "unique" if len(numeric_values) == 1 and missing == 0 else ("missing" if not numeric_values else "mixed_or_partial"),
    }


def margin_bucket(value: float) -> str:
    if value < 0.05:
        return "lt_0_05"
    if value < 0.10:
        return "0_05_0_10"
    if value < 0.25:
        return "0_10_0_25"
    if value < 0.50:
        return "0_25_0_50"
    if value < 1.00:
        return "0_50_1_00"
    return "ge_1_00"


def state_metadata(sample: dict[str, Any], *, run_bucket: str, state_index: int) -> dict[str, Any]:
    metrics = sample.get("teacher_distribution_metrics") or {}
    delta = float(sample.get("delta_best_vs_baseline", 0.0) or 0.0)
    margin = float(sample.get("best_margin", sample.get("score_gap", 0.0)) or 0.0)
    label = pilot_gate_label(sample)
    actual_high_regret = delta >= 1.0
    actual_low_margin = margin < 0.25
    actual_teacher_disagreement = bool(metrics.get("baseline_disagreement", False))
    bucket_group = run_bucket
    if run_bucket.startswith("predicted_") and run_bucket.endswith("_from_pool"):
        bucket_group = run_bucket.removesuffix("_from_pool")
    predicted_bucket = bucket_group
    return {
        "state_index": state_index,
        "state_hash": stable_state_hash(sample),
        "state_id": sample.get("state_id"),
        "sample_id": sample.get("sample_id"),
        "hand_id": sample.get("hand_id"),
        "seed": sample.get("seed"),
        "hand_seed": sample.get("hand_seed"),
        "run_bucket": run_bucket,
        "bucket_group": bucket_group,
        "predicted_bucket": predicted_bucket,
        "source_bucket": sample.get("source_bucket"),
        "source_bucket_requested": sample.get("source_bucket_requested"),
        "source_bucket_actual": sample.get("source_bucket_actual"),
        "seat": sample.get("seat", "unknown"),
        "to_act_order": sample.get("to_act_order", "unknown"),
        "teacher_label": sample.get("teacher_label"),
        "pilot_gate_label": label,
        "pilot_gate_label_id": GATE_LABEL_TO_ID[label],
        "actual_high_regret": actual_high_regret,
        "actual_low_margin": actual_low_margin,
        "actual_teacher_disagreement": actual_teacher_disagreement,
        "delta_best_vs_baseline": delta,
        "delta_best_vs_reference": float(sample.get("delta_best_vs_reference", 0.0) or 0.0),
        "SE_delta_best_vs_baseline": float(sample.get("SE_delta_best_vs_baseline", 0.0) or 0.0),
        "best_margin": margin,
        "margin_bucket": margin_bucket(margin),
        "baseline_model_margin": float(sample.get("baseline_model_margin", 0.0) or 0.0),
        "rollout_count": int(sample.get("rollout_count", 0) or 0),
        "t3_continuation": sample.get("t3_continuation"),
        "continuation_policy_T3": sample.get("continuation_policy_T3"),
    }


def t3_continuation_metadata(rows: list[dict[str, Any]]) -> dict[str, Any]:
    mode_counts = Counter(str(row.get("t3_continuation") or "legacy_unspecified") for row in rows)
    policy_counts = Counter(str(row.get("continuation_policy_T3") or "legacy_unspecified") for row in rows)
    primary_mode = next(iter(mode_counts)) if len(mode_counts) == 1 else "mixed"
    primary_policy = next(iter(policy_counts)) if len(policy_counts) == 1 else "mixed"
    return {
        "t3_continuation": primary_mode,
        "continuation_policy_T3": primary_policy,
        "t3_continuation_counts": dict(mode_counts),
        "continuation_policy_T3_counts": dict(policy_counts),
    }


def split_states(
    metadata: list[dict[str, Any]],
    *,
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> np.ndarray:
    if train_fraction <= 0.0 or val_fraction < 0.0 or train_fraction + val_fraction >= 1.0:
        raise ValueError("split fractions must satisfy train>0, val>=0, train+val<1")
    groups: dict[str, list[int]] = defaultdict(list)
    for item in metadata:
        key = "|".join(
            [
                str(item["bucket_group"]),
                str(item["seat"]),
                str(item["pilot_gate_label"]),
            ]
        )
        groups[key].append(int(item["state_index"]))

    split = np.full(len(metadata), SPLIT_NAME_TO_ID["train"], dtype=np.int8)
    rng = np.random.default_rng(seed)
    for indices in groups.values():
        shuffled = np.asarray(indices, dtype=np.int64)
        rng.shuffle(shuffled)
        n = int(shuffled.size)
        val_n = int(round(n * val_fraction))
        test_n = int(round(n * (1.0 - train_fraction - val_fraction)))
        if n >= 3:
            val_n = max(val_n, 1)
            test_n = max(test_n, 1)
        if val_n + test_n >= n:
            overflow = val_n + test_n - n + 1
            test_n = max(0, test_n - overflow)
        split[shuffled[:val_n]] = SPLIT_NAME_TO_ID["val"]
        split[shuffled[val_n : val_n + test_n]] = SPLIT_NAME_TO_ID["test"]
    return split


def summarize_split(metadata: list[dict[str, Any]], split: np.ndarray) -> dict[str, Any]:
    summary: dict[str, Any] = {"splits": {}}
    for split_id, name in SPLIT_ID_TO_NAME.items():
        indices = np.where(split == split_id)[0]
        rows = [metadata[int(index)] for index in indices]
        summary["splits"][name] = {
            "states": int(indices.size),
            "bucket_counts": dict(Counter(str(row["bucket_group"]) for row in rows)),
            "run_bucket_counts": dict(Counter(str(row["run_bucket"]) for row in rows)),
            "seat_counts": dict(Counter(str(row["seat"]) for row in rows)),
            "gate_label_counts": dict(Counter(str(row["pilot_gate_label"]) for row in rows)),
        }
    return summary


def finite_array(name: str, values: np.ndarray) -> None:
    if not np.isfinite(values).all():
        bad = int(np.size(values) - np.isfinite(values).sum())
        raise ValueError(f"{name} contains {bad} non-finite values")


def write_metadata_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_state_metadata_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.max_records is not None and args.max_records <= 0:
        raise SystemExit("--max-records must be positive")
    cache_dir = args.cache_dir.resolve()
    if cache_dir.exists() and any(cache_dir.iterdir()) and not args.force:
        raise SystemExit(f"cache dir is not empty: {cache_dir} (use --force)")
    cache_dir.mkdir(parents=True, exist_ok=True)

    started_at = time.time()
    input_files = resolve_input_files(args.input, args.bucket_sidecar_dir)
    input_file_metadata = [
        {
            "path": str(path.resolve()),
            "run_bucket": bucket,
            "sha256": sha256_file(path),
        }
        for path, bucket in input_files
    ]
    samples: list[tuple[dict[str, Any], str]] = []
    action_counts: list[int] = []
    for path, run_bucket in input_files:
        for sample in _iter_jsonl(path):
            samples.append((sample, run_bucket))
            action_counts.append(len(sample.get("actions", ())))
            if args.max_records is not None and len(samples) >= args.max_records:
                break
        if args.max_records is not None and len(samples) >= args.max_records:
            break
    if not samples:
        raise SystemExit("no HU T2 teacher samples")
    if any(count <= 0 for count in action_counts):
        raise SystemExit("teacher sample with no actions")

    state_count = len(samples)
    offsets = np.zeros(state_count + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(np.asarray(action_counts, dtype=np.int64))
    action_count = int(offsets[-1])
    dtype = np.dtype(args.dtype)

    features_path = cache_dir / f"features.{args.dtype}.mmap"
    features = np.memmap(features_path, dtype=dtype, mode="w+", shape=(action_count, HU_FEATURE_DIM))
    ev_targets = np.memmap(cache_dir / "target_ev.float32.mmap", dtype=np.float32, mode="w+", shape=(action_count,))
    delta_baseline = np.memmap(cache_dir / "target_delta_baseline.float32.mmap", dtype=np.float32, mode="w+", shape=(action_count,))
    delta_reference = np.memmap(cache_dir / "target_delta_reference.float32.mmap", dtype=np.float32, mode="w+", shape=(action_count,))
    rank_score = np.memmap(cache_dir / "target_rank_score.float32.mmap", dtype=np.float32, mode="w+", shape=(action_count,))
    se = np.memmap(cache_dir / "action_ev_se.float32.mmap", dtype=np.float32, mode="w+", shape=(action_count,))
    action_original_indices = np.memmap(cache_dir / "action_original_index.int16.mmap", dtype=np.int16, mode="w+", shape=(action_count,))

    state_metadata_rows: list[dict[str, Any]] = []
    baseline_action_index = np.zeros(state_count, dtype=np.int16)
    reference_action_index = np.zeros(state_count, dtype=np.int16)
    fallback_action_index = np.zeros(state_count, dtype=np.int16)
    best_action_index = np.zeros(state_count, dtype=np.int16)
    second_best_action_index = np.zeros(state_count, dtype=np.int16)
    gate_label_id = np.zeros(state_count, dtype=np.int8)

    source_counts: Counter[str] = Counter()
    run_bucket_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    total_rollout_count = Counter()
    observation_fingerprints: list[str] = []
    legal_mapping_rows: list[str] = []

    for state_index, (sample, run_bucket) in enumerate(samples):
        actions = list(sample.get("actions", ()))
        if sample.get("action_key_schema") != ACTION_KEY_SCHEMA:
            raise ValueError(
                f"missing/unsupported ActionKey schema at state {state_index}"
            )
        action_set_digest, action_order_digest = action_record_mapping_digests(actions)
        if sample.get("legal_action_set_digest") != action_set_digest:
            raise ValueError(f"legal action set digest mismatch at state {state_index}")
        if sample.get("legal_action_order_digest") != action_order_digest:
            raise ValueError(f"legal action order digest mismatch at state {state_index}")
        observation_fingerprint = stable_state_hash(sample)
        observation_fingerprints.append(observation_fingerprint)
        legal_mapping_rows.append(
            f"{observation_fingerprint}|{action_set_digest}|{action_order_digest}"
        )
        policy_sample = policy_feature_sample_from_record(sample)
        sample_features, sample_targets = sample_to_matrix(policy_sample)
        if sample_features.shape != (len(actions), HU_FEATURE_DIM):
            raise ValueError(f"feature shape mismatch at state {state_index}: {sample_features.shape}")
        start = int(offsets[state_index])
        end = int(offsets[state_index + 1])
        features[start:end] = sample_features.astype(dtype, copy=False)
        targets = sample_targets.astype(np.float32, copy=False)
        ev_targets[start:end] = targets
        baseline_pos = action_position(actions, sample.get("baseline_action"), fallback=0)
        reference_pos = action_position(actions, sample.get("reference_action"), fallback=baseline_pos)
        fallback_pos = action_position(actions, sample.get("fallback_action"), fallback=baseline_pos)
        order = np.argsort(-targets.astype(np.float64))
        best_pos = int(order[0])
        second_pos = int(order[1]) if order.size > 1 else best_pos
        baseline_ev = float(targets[baseline_pos])
        reference_ev = float(targets[reference_pos])
        best_ev = float(targets[best_pos])
        for local_index, action in enumerate(actions):
            absolute = start + local_index
            ev = float(targets[local_index])
            delta_baseline[absolute] = float(action.get("delta_vs_baseline", ev - baseline_ev))
            delta_reference[absolute] = float(action.get("delta_vs_reference", ev - reference_ev))
            rank_score[absolute] = float(ev - best_ev)
            se[absolute] = float(action.get("ev_standard_error", action.get("standard_error", 0.0)) or 0.0)
            original_index = action_original_index(action)
            action_original_indices[absolute] = int(local_index if original_index is None else original_index)

        meta = state_metadata(sample, run_bucket=run_bucket, state_index=state_index)
        state_metadata_rows.append(meta)
        baseline_action_index[state_index] = baseline_pos
        reference_action_index[state_index] = reference_pos
        fallback_action_index[state_index] = fallback_pos
        best_action_index[state_index] = best_pos
        second_best_action_index[state_index] = second_pos
        gate_label_id[state_index] = int(meta["pilot_gate_label_id"])
        source_counts[str(sample.get("source_bucket", "unknown"))] += 1
        run_bucket_counts[run_bucket] += 1
        label_counts[str(meta["pilot_gate_label"])] += 1
        total_rollout_count[int(sample.get("rollout_count", 0) or 0)] += 1

    features.flush()
    ev_targets.flush()
    delta_baseline.flush()
    delta_reference.flush()
    rank_score.flush()
    se.flush()
    action_original_indices.flush()

    finite_array("features", np.asarray(features, dtype=np.float32))
    for name, array in [
        ("target_ev", ev_targets),
        ("target_delta_baseline", delta_baseline),
        ("target_delta_reference", delta_reference),
        ("target_rank_score", rank_score),
        ("action_ev_se", se),
    ]:
        finite_array(name, np.asarray(array))

    state_split = split_states(
        state_metadata_rows,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    split_summary = summarize_split(state_metadata_rows, state_split)

    np.save(cache_dir / "sample_offsets.npy", offsets)
    np.save(cache_dir / "sample_action_counts.npy", np.asarray(action_counts, dtype=np.int16))
    np.save(cache_dir / "state_split.npy", state_split)
    np.save(cache_dir / "baseline_action_index.npy", baseline_action_index)
    np.save(cache_dir / "reference_action_index.npy", reference_action_index)
    np.save(cache_dir / "fallback_action_index.npy", fallback_action_index)
    np.save(cache_dir / "best_action_index.npy", best_action_index)
    np.save(cache_dir / "second_best_action_index.npy", second_best_action_index)
    np.save(cache_dir / "gate_label_id.npy", gate_label_id)
    write_metadata_jsonl(cache_dir / "state_metadata.jsonl", state_metadata_rows)
    write_state_metadata_csv(cache_dir / "state_metadata.csv", state_metadata_rows)
    (cache_dir / "split_summary.json").write_text(
        json.dumps(split_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    metadata = {
        "schema": FEATURE_CACHE_SCHEMA,
        "observation_schema": OBSERVATION_SCHEMA,
        "policy_feature_sample_schema": POLICY_FEATURE_SAMPLE_SCHEMA,
        "action_key_schema": ACTION_KEY_SCHEMA,
        "feature_value_schema": FEATURE_VALUE_SCHEMA,
        "rules_digest": REGULAR_RULES_DIGEST,
        "input": str(args.input[-1]),
        "inputs": [str(path) for path in args.input],
        "input_files": input_file_metadata,
        "cache_dir": str(cache_dir),
        "feature_dim": HU_FEATURE_DIM,
        "feature_dtype": args.dtype,
        "state_count": state_count,
        "action_count": action_count,
        "observation_fingerprint_digest": hashlib.sha256(
            "\n".join(observation_fingerprints).encode("ascii")
        ).hexdigest(),
        "legal_action_mapping_digest": hashlib.sha256(
            "\n".join(legal_mapping_rows).encode("ascii")
        ).hexdigest(),
        "run_bucket_counts": dict(run_bucket_counts),
        "source_bucket_counts": dict(source_counts),
        "pilot_gate_label_counts": dict(label_counts),
        "rollout_count_distribution": dict(total_rollout_count),
        "t3_continuation_metadata": t3_continuation_metadata(state_metadata_rows),
        "scoring_objective": scoring_objective_metadata(samples),
        "split_summary": split_summary,
        "train_fraction": args.train_fraction,
        "val_fraction": args.val_fraction,
        "test_fraction": 1.0 - args.train_fraction - args.val_fraction,
        "seed": args.seed,
        "elapsed_seconds": time.time() - started_at,
        "validation": {
            "feature_nan_or_inf": False,
            "target_nan_or_inf": False,
            "state_split_unit": "state",
            "all_legal_actions_present": True,
        },
    }
    metadata["cache_manifest_digest"] = _digest_payload(
        cache_identity_payload(metadata)
    )
    (cache_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
