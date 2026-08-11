from __future__ import annotations

import json
from pathlib import Path
from copy import deepcopy

import pytest

from ofc_regular.audit_hu_m4_t1_data import audit_locked_splits, read_and_audit_shard
from ofc_regular.generate_hu_m4_t1_data import M4_T1_DATA_SCHEMA


def _paired_summary(delta: float, *, count: int) -> dict:
    return {
        "schema": "hu_m4_paired_delta_summary_v1",
        "count": count,
        "mean": delta,
        "standard_error": 0.0,
        "std": 0.0,
        "min": delta,
        "p01": delta,
        "p05": delta,
        "p25": delta,
        "p50": delta,
        "p75": delta,
        "p95": delta,
        "p99": delta,
        "max": delta,
        "lt0_rate": float(delta < 0.0),
        "le_neg6_rate": float(delta <= -6.0),
        "le_neg12_rate": float(delta <= -12.0),
        "le_neg20_rate": float(delta <= -20.0),
    }


def _upgrade_to_v2(row: dict) -> dict:
    upgraded = deepcopy(row)
    upgraded["schema"] = M4_T1_DATA_SCHEMA
    baseline_index = int(upgraded["baseline_action_row_index"])
    baseline_score = float(upgraded["actions"][baseline_index]["score"])
    count = int(upgraded["search_config"]["evaluation_samples"])
    baseline_key = str(upgraded["baseline_action_key"])
    upgraded["paired_delta_contract"] = {
        "schema": "hu_m4_paired_delta_summary_v1",
        "baseline_action_key": baseline_key,
        "common_evaluation_futures": True,
        "evaluation_samples": count,
    }
    for action in upgraded["actions"]:
        delta = float(action["score"]) - baseline_score
        action["delta_vs_baseline"] = delta
        action["delta_se_vs_baseline"] = 0.0
        action["paired_delta_vs_baseline"] = _paired_summary(delta, count=count)
    return upgraded


def _source_row() -> dict:
    path = Path("outputs/hu_joint_policy/m4_complete/correctness_smoke.jsonl")
    if not path.exists():
        pytest.skip("local M4 smoke artifact is not present")
    return _upgrade_to_v2(json.loads(path.read_text(encoding="utf-8")))


def _write(path: Path, row: dict, *, split: str, seed: int) -> None:
    payload = dict(row)
    payload["split"] = split
    payload["hand_seed"] = seed
    payload["root_id"] = f"{split}-{seed}"
    # A different seed normally implies a different observation. Tests only
    # exercise cross-split seed accounting, so keep each path to one record.
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_v1_smoke_artifact_fails_closed_and_v2_fixture_passes_audit(
    tmp_path: Path,
) -> None:
    path = Path("outputs/hu_joint_policy/m4_complete/correctness_smoke.jsonl")
    if not path.exists():
        pytest.skip("local M4 smoke artifact is not present")
    with pytest.raises(ValueError, match="unsupported schema"):
        read_and_audit_shard(path, expected_split="correctness_smoke")
    legacy_report = read_and_audit_shard(
        path,
        expected_split="correctness_smoke",
        require_paired_delta=False,
    )
    assert legacy_report["legacy_v1_records"] == 1
    assert legacy_report["paired_delta_records"] == 0
    upgraded = tmp_path / "v2.jsonl"
    upgraded.write_text(json.dumps(_source_row()) + "\n", encoding="utf-8")
    report = read_and_audit_shard(upgraded, expected_split="correctness_smoke")
    assert report["records"] == 1
    assert report["paired_delta_records"] == 1
    assert report["hidden_truth_records"] == 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("missing", "summary missing"),
        ("count", "count mismatch"),
        ("nonfinite", "non-finite"),
        ("baseline", "baseline paired delta is not exactly zero"),
    ],
)
def test_paired_delta_tampering_fails_closed(
    tmp_path: Path, mutation: str, message: str
) -> None:
    row = _source_row()
    baseline_index = int(row["baseline_action_row_index"])
    if mutation == "missing":
        row["actions"][0].pop("paired_delta_vs_baseline")
    elif mutation == "count":
        row["actions"][0]["paired_delta_vs_baseline"]["count"] += 1
    elif mutation == "nonfinite":
        row["actions"][0]["paired_delta_vs_baseline"]["p95"] = float("nan")
    else:
        summary = row["actions"][baseline_index]["paired_delta_vs_baseline"]
        summary["std"] = 0.25
    path = tmp_path / f"bad-paired-{mutation}.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        read_and_audit_shard(path, expected_split="correctness_smoke")


def test_forbidden_opponent_private_field_fails_closed(tmp_path: Path) -> None:
    row = _source_row()
    row["opponent_private_discards"] = ["As"]
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hidden/replay"):
        read_and_audit_shard(path, expected_split="correctness_smoke")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("candidate_samples", 0, "positive integer"),
        ("evaluation_samples", "1", "positive integer"),
        ("candidate_samples", 2, "digest count does not match"),
        ("evaluation_samples", 2, "paired-delta evaluation count mismatch"),
    ],
)
def test_search_config_rng_count_tampering_fails_closed(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    row = _source_row()
    row["search_config"] = dict(row["search_config"])
    row["search_config"][field] = value
    # These legacy action fields are intentionally not the source of truth.
    assert row["actions"][0]["future_count"] == 0
    path = tmp_path / f"tampered-{field}-{value}.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        read_and_audit_shard(path, expected_split="correctness_smoke")


def test_locked_split_seed_overlap_is_rejected(tmp_path: Path) -> None:
    row = _source_row()
    train = tmp_path / "train.jsonl"
    calibration = tmp_path / "cal.jsonl"
    holdout = tmp_path / "hold.jsonl"
    _write(train, row, split="train", seed=11)
    _write(calibration, row, split="calibration", seed=11)
    _write(holdout, row, split="locked_holdout", seed=13)
    with pytest.raises(ValueError, match="leakage"):
        audit_locked_splits(
            train=[train], calibration=[calibration], locked_holdout=[holdout]
        )


def test_duplicate_shard_path_within_split_is_rejected(tmp_path: Path) -> None:
    row = _source_row()
    train = tmp_path / "train.jsonl"
    calibration = tmp_path / "cal.jsonl"
    holdout = tmp_path / "hold.jsonl"
    _write(train, row, split="train", seed=21)
    _write(calibration, row, split="calibration", seed=22)
    _write(holdout, row, split="locked_holdout", seed=23)

    with pytest.raises(ValueError, match="duplicate shard paths"):
        audit_locked_splits(
            train=[train, train],
            calibration=[calibration],
            locked_holdout=[holdout],
        )


def test_duplicate_hand_seed_across_train_shards_is_rejected(tmp_path: Path) -> None:
    row = _source_row()
    train_a = tmp_path / "train_a.jsonl"
    train_b = tmp_path / "train_b.jsonl"
    calibration = tmp_path / "cal.jsonl"
    holdout = tmp_path / "hold.jsonl"
    _write(train_a, row, split="train", seed=31)
    _write(train_b, row, split="train", seed=31)
    _write(calibration, row, split="calibration", seed=32)
    _write(holdout, row, split="locked_holdout", seed=33)

    with pytest.raises(ValueError, match="duplicate hand_seed"):
        audit_locked_splits(
            train=[train_a, train_b],
            calibration=[calibration],
            locked_holdout=[holdout],
        )


def test_duplicate_fingerprint_across_train_shards_is_rejected(tmp_path: Path) -> None:
    row = _source_row()
    train_a = tmp_path / "train_a.jsonl"
    train_b = tmp_path / "train_b.jsonl"
    calibration = tmp_path / "cal.jsonl"
    holdout = tmp_path / "hold.jsonl"
    _write(train_a, row, split="train", seed=41)
    _write(train_b, row, split="train", seed=42)
    _write(calibration, row, split="calibration", seed=43)
    _write(holdout, row, split="locked_holdout", seed=44)

    with pytest.raises(ValueError, match="duplicate observation_fingerprint"):
        audit_locked_splits(
            train=[train_a, train_b],
            calibration=[calibration],
            locked_holdout=[holdout],
        )
