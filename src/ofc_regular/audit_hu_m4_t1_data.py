"""Fail-closed audit for M4 T1-second teacher shards and locked splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .action_key import action_key, legal_action_set_digest
from .action_space import generate_turn_actions
from .generate_hu_m4_t1_data import M4_T1_DATA_SCHEMA
from .hu_infoset import ActorObservation
from .hu_m4_t1_teacher import M4_PAIRED_DELTA_SUMMARY_SCHEMA


M4_T1_AUDIT_SCHEMA = "hu_m4_t1_second_data_audit_v1"
M4_T1_LEGACY_DATA_SCHEMA = "hu_m4_t1_second_training_sample_v1"
_FORBIDDEN_RECORD_KEYS = {
    "opponent_private_discards",
    "true_opponent_private_discards",
    "true_dead_cards",
    "replay_truth",
    "world_state",
    "future_cards",
    "draw_order",
}


def read_and_audit_shard(
    path: str | Path,
    *,
    expected_split: str,
    require_paired_delta: bool = True,
) -> dict[str, Any]:
    source = Path(path)
    records: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{source}:{line_number}: record must be a mapping")
            _audit_record(
                row,
                expected_split=expected_split,
                location=f"{source}:{line_number}",
                require_paired_delta=require_paired_delta,
            )
            records.append(row)
    if not records:
        raise ValueError(f"M4 shard has no records: {source}")
    seeds = [int(row["hand_seed"]) for row in records]
    fingerprints = [str(row["observation_fingerprint"]) for row in records]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"M4 shard has duplicate hand seeds: {source}")
    if len(set(fingerprints)) != len(fingerprints):
        raise ValueError(f"M4 shard has duplicate observation fingerprints: {source}")
    return {
        "path": str(source),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "split": expected_split,
        "records": len(records),
        "hand_seeds": seeds,
        "observation_fingerprints": fingerprints,
        "legal_action_counts": dict(
            Counter(int(row["legal_action_count"]) for row in records)
        ),
        "candidate_evaluation_rng_overlap": 0,
        "paired_delta_records": sum(
            row.get("schema") == M4_T1_DATA_SCHEMA for row in records
        ),
        "legacy_v1_records": sum(
            row.get("schema") == M4_T1_LEGACY_DATA_SCHEMA for row in records
        ),
        "hidden_truth_records": 0,
        "teacher_value_status": "diagnostic_not_match_EV",
    }


def audit_locked_splits(
    *,
    train: Sequence[str | Path],
    calibration: Sequence[str | Path],
    locked_holdout: Sequence[str | Path],
    require_paired_delta: bool = True,
) -> dict[str, Any]:
    inputs = {
        "train": tuple(train),
        "calibration": tuple(calibration),
        "locked_holdout": tuple(locked_holdout),
    }
    if any(not paths for paths in inputs.values()):
        raise ValueError("train, calibration, and locked_holdout must all be explicit")
    for split, paths in inputs.items():
        normalized_paths = [
            os.path.normcase(str(Path(path).resolve())) for path in paths
        ]
        duplicates = _duplicate_values(normalized_paths)
        if duplicates:
            raise ValueError(
                f"M4 {split} inputs contain duplicate shard paths: {duplicates}"
            )

    groups = {
        "train": [
            read_and_audit_shard(
                path,
                expected_split="train",
                require_paired_delta=require_paired_delta,
            )
            for path in inputs["train"]
        ],
        "calibration": [
            read_and_audit_shard(
                path,
                expected_split="calibration",
                require_paired_delta=require_paired_delta,
            )
            for path in inputs["calibration"]
        ],
        "locked_holdout": [
            read_and_audit_shard(
                path,
                expected_split="locked_holdout",
                require_paired_delta=require_paired_delta,
            )
            for path in inputs["locked_holdout"]
        ],
    }
    for split, shards in groups.items():
        duplicate_seeds = _duplicate_values(
            [seed for shard in shards for seed in shard["hand_seeds"]]
        )
        if duplicate_seeds:
            raise ValueError(
                f"M4 {split} shards contain duplicate hand_seed values: "
                f"{duplicate_seeds}"
            )
        duplicate_fingerprints = _duplicate_values(
            [
                value
                for shard in shards
                for value in shard["observation_fingerprints"]
            ]
        )
        if duplicate_fingerprints:
            raise ValueError(
                f"M4 {split} shards contain duplicate observation_fingerprint "
                f"values: {duplicate_fingerprints}"
            )
    seed_sets = {
        split: {
            seed for shard in shards for seed in shard["hand_seeds"]
        }
        for split, shards in groups.items()
    }
    fingerprint_sets = {
        split: {
            value
            for shard in shards
            for value in shard["observation_fingerprints"]
        }
        for split, shards in groups.items()
    }
    overlap: dict[str, dict[str, int]] = {}
    split_names = tuple(groups)
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1 :]:
            key = f"{left}__{right}"
            overlap[key] = {
                "hand_seeds": len(seed_sets[left] & seed_sets[right]),
                "observation_fingerprints": len(
                    fingerprint_sets[left] & fingerprint_sets[right]
                ),
            }
    if any(value for pair in overlap.values() for value in pair.values()):
        raise ValueError(f"M4 locked split leakage detected: {overlap}")
    return {
        "schema": M4_T1_AUDIT_SCHEMA,
        "status": "pass",
        "teacher_value_status": "diagnostic_not_match_EV",
        "paired_delta_mode": (
            "required_v2" if require_paired_delta else "legacy_v1_read_only_allowed"
        ),
        "splits": {
            split: {
                "records": sum(shard["records"] for shard in shards),
                "shards": shards,
            }
            for split, shards in groups.items()
        },
        "cross_split_overlap": overlap,
        "gates": {
            "hidden_discard_safe": True,
            "all_legal_actions_mapped": True,
            "candidate_evaluation_rng_disjoint": True,
            "paired_common_future_delta_contract": all(
                shard["legacy_v1_records"] == 0
                for shards in groups.values()
                for shard in shards
            ),
            "within_split_shard_paths_unique": True,
            "within_split_hand_seeds_unique": True,
            "within_split_fingerprints_unique": True,
            "seed_ranges_disjoint": True,
            "fingerprints_disjoint": True,
            "holdout_threshold_search_allowed": False,
            "current_profile_resolved": False,
        },
    }


def _duplicate_values(values: Sequence[Any]) -> list[Any]:
    return sorted(value for value, count in Counter(values).items() if count > 1)


def _audit_record(
    row: Mapping[str, Any],
    *,
    expected_split: str,
    location: str,
    require_paired_delta: bool,
) -> None:
    data_schema = row.get("schema")
    allowed_schemas = (
        {M4_T1_DATA_SCHEMA}
        if require_paired_delta
        else {M4_T1_DATA_SCHEMA, M4_T1_LEGACY_DATA_SCHEMA}
    )
    if data_schema not in allowed_schemas:
        raise ValueError(f"{location}: unsupported schema")
    if row.get("split") != expected_split:
        raise ValueError(f"{location}: split mismatch")
    if row.get("teacher_value_status") != "diagnostic_not_match_EV":
        raise ValueError(f"{location}: teacher score status is unsafe")
    unsafe = _FORBIDDEN_RECORD_KEYS & set(row)
    if unsafe:
        raise ValueError(f"{location}: hidden/replay fields present: {sorted(unsafe)}")
    encoded = json.dumps(row, sort_keys=True)
    if "opponent_private_discards" in encoded:
        raise ValueError(f"{location}: opponent private discard field present")
    provenance = row.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"{location}: provenance missing")
    if provenance.get("current_profile_resolved") is not False:
        raise ValueError(f"{location}: current profile resolution is forbidden")

    observation_payload = row.get("policy_observation")
    if not isinstance(observation_payload, Mapping):
        raise ValueError(f"{location}: policy_observation missing")
    observation = ActorObservation.from_dict(observation_payload)
    if (observation.street, observation.seat, observation.to_act_order) != (
        "T1",
        "second",
        "second",
    ):
        raise ValueError(f"{location}: not a second-seat T1 observation")
    if observation.fingerprint() != row.get("observation_fingerprint"):
        raise ValueError(f"{location}: observation fingerprint mismatch")
    if list(observation.legacy_dead_cards()) != list(row.get("dead_cards", ())):
        raise ValueError(f"{location}: visible dead-card adapter mismatch")

    legal = generate_turn_actions(observation.hero_board, observation.dealt_cards)
    if int(row.get("legal_action_count", -1)) != len(legal):
        raise ValueError(f"{location}: legal action count mismatch")
    if row.get("legal_action_set_digest") != legal_action_set_digest(legal):
        raise ValueError(f"{location}: legal action set digest mismatch")
    action_rows = row.get("actions")
    if not isinstance(action_rows, Sequence) or isinstance(action_rows, (str, bytes)):
        raise ValueError(f"{location}: action rows missing")
    tokens = [str(action.get("action_key", "")) for action in action_rows]
    legal_tokens = {action_key(action).to_token() for action in legal}
    if len(tokens) != len(legal) or set(tokens) != legal_tokens:
        raise ValueError(f"{location}: action mapping is incomplete or duplicated")
    scores = [float(action.get("score", float("nan"))) for action in action_rows]
    ses = [float(action.get("score_se", float("nan"))) for action in action_rows]
    if any(not math.isfinite(value) for value in scores):
        raise ValueError(f"{location}: non-finite teacher score")
    if any(not math.isfinite(value) or value < 0.0 for value in ses):
        raise ValueError(f"{location}: invalid teacher standard error")
    if scores != sorted(scores, reverse=True):
        raise ValueError(f"{location}: independent-evaluation rows not score-sorted")
    baseline_index = int(row.get("baseline_action_row_index", -1))
    if not 0 <= baseline_index < len(tokens):
        raise ValueError(f"{location}: baseline row index invalid")
    if tokens[baseline_index] != row.get("baseline_action_key"):
        raise ValueError(f"{location}: baseline ActionKey mapping mismatch")
    search_config = row.get("search_config")
    if not isinstance(search_config, Mapping):
        raise ValueError(f"{location}: search_config missing")
    expected_rng_counts: dict[str, int] = {}
    for phase in ("candidate", "evaluation"):
        value = search_config.get(f"{phase}_samples")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                f"{location}: search_config {phase}_samples must be a "
                "positive integer"
            )
        expected_rng_counts[phase] = value

    if data_schema == M4_T1_DATA_SCHEMA:
        paired_contract = row.get("paired_delta_contract")
        if not isinstance(paired_contract, Mapping):
            raise ValueError(f"{location}: paired-delta contract missing")
        if paired_contract.get("schema") != M4_PAIRED_DELTA_SUMMARY_SCHEMA:
            raise ValueError(f"{location}: paired-delta contract schema mismatch")
        if paired_contract.get("baseline_action_key") != tokens[baseline_index]:
            raise ValueError(f"{location}: paired-delta baseline ActionKey mismatch")
        if paired_contract.get("common_evaluation_futures") is not True:
            raise ValueError(f"{location}: paired common futures are not verified")
        if (
            paired_contract.get("evaluation_samples")
            != expected_rng_counts["evaluation"]
        ):
            raise ValueError(f"{location}: paired-delta evaluation count mismatch")

        baseline_score = scores[baseline_index]
        declared_baseline_score = float(
            row.get("baseline_teacher_score", float("nan"))
        )
        if not math.isfinite(declared_baseline_score) or not math.isclose(
            declared_baseline_score, baseline_score, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise ValueError(f"{location}: baseline teacher score mismatch")
        for action_index, action in enumerate(action_rows):
            if not isinstance(action, Mapping):
                raise ValueError(f"{location}: action row must be a mapping")
            summary = _audit_paired_delta_summary(
                action.get("paired_delta_vs_baseline"),
                expected_count=expected_rng_counts["evaluation"],
                location=f"{location}:action={action_index}",
            )
            delta = float(action.get("delta_vs_baseline", float("nan")))
            delta_se = float(action.get("delta_se_vs_baseline", float("nan")))
            if not math.isfinite(delta) or not math.isclose(
                delta, summary["mean"], rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError(f"{location}: paired-delta mean/flat field mismatch")
            if not math.isfinite(delta_se) or delta_se < 0.0 or not math.isclose(
                delta_se,
                summary["standard_error"],
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(f"{location}: paired-delta SE/flat field mismatch")
            marginal_delta = scores[action_index] - baseline_score
            if not math.isclose(
                delta, marginal_delta, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError(f"{location}: paired delta disagrees with score means")
            if action_index == baseline_index and any(
                float(summary[key]) != 0.0
                for key in (
                    "mean",
                    "standard_error",
                    "std",
                    "min",
                    "p01",
                    "p05",
                    "p25",
                    "p50",
                    "p75",
                    "p95",
                    "p99",
                    "max",
                    "lt0_rate",
                    "le_neg6_rate",
                    "le_neg12_rate",
                    "le_neg20_rate",
                )
            ):
                raise ValueError(f"{location}: baseline paired delta is not exactly zero")

    candidate_rng = _rng_digest_list(
        row.get("candidate_rng_key_digests"),
        location=location,
        phase="candidate",
    )
    evaluation_rng = _rng_digest_list(
        row.get("evaluation_rng_key_digests"),
        location=location,
        phase="evaluation",
    )
    if len(candidate_rng) != expected_rng_counts["candidate"]:
        raise ValueError(
            f"{location}: candidate RNG digest count does not match "
            "search_config candidate_samples"
        )
    if len(evaluation_rng) != expected_rng_counts["evaluation"]:
        raise ValueError(
            f"{location}: evaluation RNG digest count does not match "
            "search_config evaluation_samples"
        )
    if set(candidate_rng) & set(evaluation_rng):
        raise ValueError(f"{location}: candidate/evaluation RNGs are not disjoint")


def _audit_paired_delta_summary(
    value: Any, *, expected_count: int, location: str
) -> dict[str, float | int | str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location}: paired-delta summary missing")
    if value.get("schema") != M4_PAIRED_DELTA_SUMMARY_SCHEMA:
        raise ValueError(f"{location}: paired-delta summary schema mismatch")
    count = value.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or count != expected_count:
        raise ValueError(f"{location}: paired-delta summary count mismatch")
    numeric_keys = (
        "mean",
        "standard_error",
        "std",
        "min",
        "p01",
        "p05",
        "p25",
        "p50",
        "p75",
        "p95",
        "p99",
        "max",
        "lt0_rate",
        "le_neg6_rate",
        "le_neg12_rate",
        "le_neg20_rate",
    )
    result: dict[str, float | int | str] = {
        "schema": M4_PAIRED_DELTA_SUMMARY_SCHEMA,
        "count": count,
    }
    for key in numeric_keys:
        number = float(value.get(key, float("nan")))
        if not math.isfinite(number):
            raise ValueError(f"{location}: paired-delta {key} is non-finite")
        result[key] = number
    if float(result["standard_error"]) < 0.0 or float(result["std"]) < 0.0:
        raise ValueError(f"{location}: paired-delta dispersion is negative")
    for key in ("lt0_rate", "le_neg6_rate", "le_neg12_rate", "le_neg20_rate"):
        if not 0.0 <= float(result[key]) <= 1.0:
            raise ValueError(f"{location}: paired-delta rate is outside [0, 1]")
    ordered = [
        float(result[key])
        for key in (
            "min",
            "p01",
            "p05",
            "p25",
            "p50",
            "p75",
            "p95",
            "p99",
            "max",
        )
    ]
    if ordered != sorted(ordered):
        raise ValueError(f"{location}: paired-delta quantiles are not monotone")
    return result


def _rng_digest_list(value: Any, *, location: str, phase: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{location}: {phase} RNG digest list missing")
    digests = tuple(str(item) for item in value)
    if not digests or any(not digest for digest in digests):
        raise ValueError(f"{location}: {phase} RNG digest list is empty or invalid")
    if len(set(digests)) != len(digests):
        raise ValueError(f"{location}: {phase} RNG digest list contains duplicates")
    return digests


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, action="append", required=True)
    parser.add_argument("--calibration", type=Path, action="append", required=True)
    parser.add_argument("--locked-holdout", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-legacy-v1",
        action="store_true",
        help="Read-only audit of frozen pre-M4.2 rows without paired deltas.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = audit_locked_splits(
        train=args.train,
        calibration=args.calibration,
        locked_holdout=args.locked_holdout,
        require_paired_delta=not args.allow_legacy_v1,
    )
    _atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["audit_locked_splits", "read_and_audit_shard"]
