from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from ofc_regular import hu_m43_attempt02_contract as contract


REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN = REPO_ROOT / "configs" / "hu_joint_policy_m43_attempt02.json"
ATTEMPT01_ROOT = (
    REPO_ROOT
    / "outputs"
    / "hu_joint_policy"
    / "m43_spot"
    / "regular-hu-m43-c2e16-pilot200-20260713-1810"
)
ATTEMPT01_CONTRACT = (
    ATTEMPT01_ROOT / "m43_t1_second_attempt01" / "data_contract.json"
)
ATTEMPT01_TRAINING_MANIFEST = (
    REPO_ROOT
    / "outputs"
    / "hu_joint_policy"
    / "m43_fold_model_runs"
    / "regular-hu-m43-fold-model-20260713-2052"
    / "training_manifest.json"
)
ATTEMPT01_TRAIN = ATTEMPT01_ROOT / "train.jsonl"
ATTEMPT01_CALIBRATION = ATTEMPT01_ROOT / "calibration.jsonl"
INHERITED_LOCKED = ATTEMPT01_ROOT / "locked_holdout.jsonl"


def _preflight_kwargs() -> dict[str, Path]:
    return {
        "plan_path": PLAN,
        "repo_root": REPO_ROOT,
        "attempt01_data_contract": ATTEMPT01_CONTRACT,
        "attempt01_training_manifest": ATTEMPT01_TRAINING_MANIFEST,
        "attempt01_train": ATTEMPT01_TRAIN,
        "attempt01_calibration": ATTEMPT01_CALIBRATION,
        "inherited_locked": INHERITED_LOCKED,
    }


@pytest.fixture(scope="module")
def sealed_preflight(tmp_path_factory: pytest.TempPathFactory):
    original_read_jsonl = contract._read_jsonl

    def reject_locked_parse(path: Path):
        assert Path(path).resolve() != INHERITED_LOCKED.resolve(), (
            "preflight/finalize must never parse inherited locked JSONL"
        )
        return original_read_jsonl(path)

    contract._read_jsonl = reject_locked_parse
    try:
        receipt = contract.build_preflight_receipt(**_preflight_kwargs())
        destination = tmp_path_factory.mktemp("m43-attempt02") / "preflight.json"
        destination.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        yield receipt, destination
    finally:
        contract._read_jsonl = original_read_jsonl


def test_frozen_plan_has_300_fresh_roots_and_is_plan_driven() -> None:
    plan = contract.load_and_validate_attempt02_plan(PLAN)
    assert plan["fresh_splits"]["train"]["roots"] == 200
    assert plan["fresh_splits"]["calibration"]["roots"] == 100
    assert plan["budget"]["fresh_roots"] == 300
    assert plan["budget"]["fresh_shards"] == 30
    assert plan["teacher_search"]["candidate_samples"] == 2
    assert plan["teacher_search"]["evaluation_samples"] == 64
    assert plan["calibration_partition"]["safety_fit_roots"] == 50
    assert plan["calibration_partition"]["threshold_lock_roots"] == 50

    # Counts are consumed from the immutable plan, not hidden 100/60 literals.
    alternate = copy.deepcopy(plan)
    alternate["fresh_splits"]["train"]["roots"] = 190
    alternate["budget"]["fresh_roots"] = 290
    alternate["budget"]["fresh_shards"] = 29
    for row in alternate["root_population"]:
        row["roots"]["train"] = 38
    contract._validate_plan(alternate)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda value: value["teacher_search"].__setitem__(
                "evaluation_samples", 16
            ),
            "c2/e64",
        ),
        (
            lambda value: value["freshness_exclusions"]["expected_union"].__setitem__(
                "unique_identities", 411
            ),
            "412",
        ),
        (
            lambda value: value["inherited_locked"].__setitem__(
                "classification", "fresh"
            ),
            "fixture",
        ),
    ],
)
def test_plan_rejects_weakened_frozen_boundaries(mutator, message: str) -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    mutator(plan)
    with pytest.raises(ValueError, match=message):
        contract._validate_plan(plan)


def test_preflight_seals_union_and_keeps_locked_unopened(sealed_preflight) -> None:
    receipt, _ = sealed_preflight
    assert receipt["schema"] == contract.M43_ATTEMPT02_PREFLIGHT_SCHEMA
    assert receipt["status"] == "pass_frozen_before_fresh_generation"
    assert receipt["exclusion_union"] == {
        **receipt["exclusion_union"],
        "source_records": 412,
        "unique_identities": 412,
        "unique_hand_seeds": 412,
        "unique_observation_fingerprints": 412,
        "identity_sha256": contract.EXCLUSION_UNION_IDENTITY_SHA256,
    }
    inherited = receipt["inherited_locked"]
    assert inherited["classification"] == "inherited_unopened"
    assert inherited["content_parse_count"] == 0
    assert inherited["model_evaluation_count"] == 0
    assert inherited["source_search"] == {
        "candidate_samples": 2,
        "evaluation_samples": 16,
    }
    assert inherited["fresh_search"] == {
        "candidate_samples": 2,
        "evaluation_samples": 64,
    }
    assert receipt["receipt_sha256"] == contract._self_digest(
        receipt, "receipt_sha256"
    )


def test_global_marker_discovery_is_identity_scoped(tmp_path: Path) -> None:
    other = tmp_path / "other" / "M43_LOCKED_CONSUMED.json"
    other.parent.mkdir()
    other.write_text(json.dumps({"identity_sha256": "a" * 64}), encoding="utf-8")
    assert contract._find_consumption_markers(
        tmp_path, contract.INHERITED_LOCKED_IDENTITY_SHA256
    ) == []
    consumed = tmp_path / "consumed" / "M43_LOCKED_CONSUMED.json"
    consumed.parent.mkdir()
    consumed.write_text(
        json.dumps(
            {"locked_identity_sha256": contract.INHERITED_LOCKED_IDENTITY_SHA256}
        ),
        encoding="utf-8",
    )
    assert contract._find_consumption_markers(
        tmp_path, contract.INHERITED_LOCKED_IDENTITY_SHA256
    ) == ["consumed/M43_LOCKED_CONSUMED.json"]


def _write_fresh_shards(tmp_path: Path, plan: dict):
    result: dict[str, list[Path]] = {}
    roots_per_shard = int(plan["budget"]["roots_per_shard"])
    for split in ("train", "calibration"):
        spec = plan["fresh_splits"][split]
        roots = int(spec["roots"])
        profile_quota = {
            row["profile"]: int(row["roots"][split])
            for row in plan["root_population"]
        }
        profiles = [
            profile
            for profile in contract._PROFILES
            for _ in range(profile_quota[profile])
        ]
        rows = []
        for index in range(roots):
            seed = int(spec["seed_start"]) + int(spec["seed_stride"]) * index
            fingerprint = hashlib.sha256(
                f"attempt02-test\0{split}\0{index}".encode()
            ).hexdigest()
            rows.append(
                {
                    "split": split,
                    "hand_seed": seed,
                    "observation_fingerprint": fingerprint,
                    "provenance": {
                        "current_profile_resolved": False,
                        "root_profile": profiles[index],
                    },
                    "search_config": {
                        "candidate_samples": 2,
                        "evaluation_samples": 64,
                        "candidate_seed": int(spec["candidate_seed_start"])
                        + int(spec["seed_stride"]) * (index // roots_per_shard),
                        "evaluation_seed": int(spec["evaluation_seed_start"])
                        + int(spec["seed_stride"]) * (index // roots_per_shard),
                        "child_policy_seed": int(spec["child_policy_seed_start"])
                        + int(spec["seed_stride"]) * (index // roots_per_shard),
                    },
                }
            )
        paths = []
        for shard_index in range(roots // roots_per_shard):
            path = tmp_path / f"{split}_{shard_index:03d}.jsonl"
            shard_rows = rows[
                shard_index
                * roots_per_shard : (shard_index + 1)
                * roots_per_shard
            ]
            path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in shard_rows),
                encoding="utf-8",
            )
            paths.append(path)
        result[split] = paths
    return result


def test_finalize_enforces_fresh_counts_and_partitions_50_50(
    sealed_preflight, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, receipt_path = sealed_preflight
    plan = contract.load_and_validate_attempt02_plan(PLAN)
    shards = _write_fresh_shards(tmp_path, plan)

    def structural_audit(path, *, expected_split, require_paired_delta):
        rows = contract._read_jsonl(Path(path))
        assert require_paired_delta is True
        assert all(row["split"] == expected_split for row in rows)
        return {"records": len(rows), "paired_delta_records": len(rows)}

    monkeypatch.setattr(contract, "read_and_audit_shard", structural_audit)
    sealed = contract.finalize_fresh_data(
        **_preflight_kwargs(),
        preflight_receipt=receipt_path,
        train=shards["train"],
        calibration=shards["calibration"],
    )
    assert sealed["schema"] == contract.M43_ATTEMPT02_DATA_CONTRACT_SCHEMA
    assert sealed["status"] == (
        "pass_fresh_train_calibration_sealed_inherited_locked_unopened"
    )
    assert sealed["preflight"]["receipt_sha256"] == receipt["receipt_sha256"]
    assert sealed["fresh_splits"]["train"]["records"] == 200
    assert sealed["fresh_splits"]["calibration"]["records"] == 100
    partition = sealed["calibration_partition"]
    assert partition["safety_fit"]["records"] == 50
    assert partition["threshold_lock"]["records"] == 50
    assert partition["overlap"] == 0
    assert partition["inherited_locked_used"] is False
    assert sealed["inherited_locked"]["content_parse_count"] == 0
    assert sealed["contract_sha256"] == contract._self_digest(
        sealed, "contract_sha256"
    )
