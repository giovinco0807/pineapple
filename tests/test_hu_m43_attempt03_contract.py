from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from ofc_regular import hu_m43_attempt02_contract as attempt02_contract
from ofc_regular import hu_m43_attempt03_contract as contract


REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN = REPO_ROOT / "configs" / "hu_joint_policy_m43_attempt03.json"
ATTEMPT02_ROOT = (
    REPO_ROOT
    / "outputs"
    / "hu_joint_policy"
    / "m43_attempt02_teacher"
    / "regular-hu-m43-attempt02-c2e64-pilot300-20260713-2201"
)
ATTEMPT02_CALIBRATION = ATTEMPT02_ROOT / "calibration.jsonl"
INHERITED_LOCKED = (
    REPO_ROOT
    / "outputs"
    / "hu_joint_policy"
    / "m43_spot"
    / "regular-hu-m43-c2e16-pilot200-20260713-1810"
    / "locked_holdout.jsonl"
)


@pytest.fixture(scope="module")
def sealed_preflight(tmp_path_factory: pytest.TempPathFactory):
    original_new = contract._read_jsonl
    original_old = attempt02_contract._read_jsonl
    forbidden = {
        ATTEMPT02_CALIBRATION.resolve(),
        INHERITED_LOCKED.resolve(),
    }

    def guarded(path: Path):
        assert Path(path).resolve() not in forbidden, (
            "Attempt03 must not parse sealed calibration or inherited locked JSONL"
        )
        return original_new(path)

    def guarded_old(path: Path):
        assert Path(path).resolve() not in forbidden, (
            "Attempt02 helper must keep Attempt03 holdouts opaque"
        )
        return original_old(path)

    contract._read_jsonl = guarded
    attempt02_contract._read_jsonl = guarded_old
    try:
        receipt = contract.build_preflight_receipt(
            plan_path=PLAN, repo_root=REPO_ROOT
        )
        path = tmp_path_factory.mktemp("m43-attempt03") / "preflight.json"
        path.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        yield receipt, path
    finally:
        contract._read_jsonl = original_new
        attempt02_contract._read_jsonl = original_old


def test_plan_freezes_700_fresh_roots_roles_and_seed_namespaces() -> None:
    plan = contract.load_and_validate_attempt03_plan(PLAN)
    assert plan["budget"]["fresh_roots"] == 700
    assert plan["budget"]["fresh_shards"] == 70
    assert plan["fresh_splits"]["train.fit"]["roots"] == 500
    assert plan["fresh_splits"]["train.precal_holdout"]["roots"] == 200
    assert plan["fresh_splits"]["train.fit"]["seed_start"] == 9106071901
    assert (
        plan["fresh_splits"]["train.precal_holdout"]["seed_start"]
        == 9506071901
    )
    assert plan["teacher_search"] == {
        "candidate_samples": 2,
        "evaluation_samples": 64,
        "common_random_futures": True,
        "candidate_evaluation_rng_disjoint": True,
        "candidate_seed_namespace": "hu-m43-attempt03-candidate-v1",
        "evaluation_seed_namespace": "hu-m43-attempt03-evaluation-v1",
        "child_policy_seed_namespace": "hu-m43-attempt03-child-v1",
        "namespace_domain_separation": "split_shard",
        "batch_child_selectors": True,
        "native_batch_threads": 4,
    }
    for row in plan["root_population"]:
        assert row["roots"]["train.fit"] == 100
        assert row["roots"]["train.precal_holdout"] == 40


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda value: value["fresh_splits"]["train.fit"].__setitem__(
                "seed_start", 9106071902
            ),
            "seed_start",
        ),
        (
            lambda value: value["teacher_search"].__setitem__(
                "evaluation_samples", 16
            ),
            "c2/e64",
        ),
        (
            lambda value: value["attempt02_provenance"][
                "sealed_calibration"
            ].__setitem__("classification", "fit"),
            "unopened calibration",
        ),
        (
            lambda value: value["model_contract"]["proposal_head"].__setitem__(
                "trees", 181
            ),
            "proposal head",
        ),
        (
            lambda value: value["precalibration_gate"].__setitem__(
                "eligible_fires_min", 0
            ),
            "eligible_fires_min",
        ),
        (
            lambda value: value["activation_guards"].__setitem__(
                "runtime_policy_activated", True
            ),
            "activation guard",
        ),
    ],
)
def test_plan_rejects_mutated_frozen_boundaries(mutator, message: str) -> None:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    mutator(plan)
    with pytest.raises(ValueError, match=message):
        contract._validate_plan(plan)


def test_preflight_binds_attempt02_and_keeps_holdouts_unopened(
    sealed_preflight,
) -> None:
    receipt, _ = sealed_preflight
    assert receipt["schema"] == contract.M43_ATTEMPT03_PREFLIGHT_SCHEMA
    assert receipt["status"] == "pass_frozen_before_fresh_generation"
    assert receipt["freshness_exclusions"]["unique_records"] == 752
    assert receipt["fit_source"] == {
        "classification": "inherited_attempt02_train_fit",
        "records": 200,
        "identity_sha256": (
            "623112de0b7a8af1357782f6f80a09de8dfc370bcf8faadb3e5c23ec03d80372"
        ),
        "content_parse_count": 1,
        "allowed_use": "ranker_fit_and_grouped_crossfit_only",
    }
    assert receipt["sealed_calibration"]["records"] == 100
    assert receipt["sealed_calibration"]["jsonl_content_parse_count"] == 0
    assert receipt["sealed_calibration"]["model_evaluation_count"] == 0
    assert receipt["sealed_calibration"]["roles"]["safety_fit"]["records"] == 50
    assert (
        receipt["sealed_calibration"]["roles"]["threshold_lock"]["records"]
        == 50
    )
    assert receipt["inherited_locked"]["jsonl_content_parse_count"] == 0
    assert receipt["attempt02"]["terminal_status"] == "no_go_precalibration"
    assert receipt["runtime"] == {
        "current_profile_resolved": False,
        "current_profile_changed": False,
        "policy_activated": False,
        "full_replacement": False,
        "large_scale_authorized": False,
    }
    assert receipt["receipt_sha256"] == contract._self_digest(
        receipt, "receipt_sha256"
    )


def _write_fresh_shards(tmp_path: Path, plan: dict) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    profiles = (
        "stage19_p0",
        "stage9f_p2",
        "stage7_m5_r10",
        "stage3_baseline",
        "random_exact_final",
    )
    for role in ("train.fit", "train.precal_holdout"):
        spec = plan["fresh_splits"][role]
        paths: list[Path] = []
        for shard_index in range(spec["roots"] // 10):
            rows = []
            for offset in range(10):
                index = shard_index * 10 + offset
                fingerprint = hashlib.sha256(
                    f"attempt03-test\0{role}\0{index}".encode()
                ).hexdigest()
                rows.append(
                    {
                        "split": "train",
                        "hand_seed": spec["seed_start"]
                        + spec["seed_stride"] * index,
                        "observation_fingerprint": fingerprint,
                        "provenance": {
                            "current_profile_resolved": False,
                            "root_profile": profiles[offset // 2],
                        },
                        "search_config": {
                            "candidate_samples": 2,
                            "evaluation_samples": 64,
                            "candidate_seed": spec["candidate_seed_start"]
                            + spec["seed_stride"] * shard_index,
                            "evaluation_seed": spec["evaluation_seed_start"]
                            + spec["seed_stride"] * shard_index,
                            "child_policy_seed": spec["child_policy_seed_start"]
                            + spec["seed_stride"] * shard_index,
                        },
                    }
                )
            path = tmp_path / f"{role.replace('.', '_')}_{shard_index:03d}.jsonl"
            path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                encoding="utf-8",
            )
            paths.append(path)
        result[role] = paths
    return result


def test_finalize_seals_fit700_and_one_shot_precal200(
    sealed_preflight, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, receipt_path = sealed_preflight
    plan = contract.load_and_validate_attempt03_plan(PLAN)
    shards = _write_fresh_shards(tmp_path, plan)

    def structural_audit(path, *, expected_split, require_paired_delta):
        rows = contract._read_jsonl(Path(path))
        assert expected_split == "train"
        assert require_paired_delta is True
        return {"records": len(rows), "paired_delta_records": len(rows)}

    monkeypatch.setattr(contract, "read_and_audit_shard", structural_audit)
    sealed = contract.finalize_fresh_data(
        plan_path=PLAN,
        repo_root=REPO_ROOT,
        preflight_receipt=receipt_path,
        train_fit=shards["train.fit"],
        train_precal_holdout=shards["train.precal_holdout"],
    )
    assert sealed["schema"] == contract.M43_ATTEMPT03_DATA_CONTRACT_SCHEMA
    assert sealed["status"] == (
        "pass_fresh_fit_and_one_shot_precal_sealed_calibration_locked_unopened"
    )
    assert sealed["preflight"]["receipt_sha256"] == receipt["receipt_sha256"]
    assert sealed["fit"]["records"] == 700
    assert sealed["fit"]["sources"]["attempt02.train"]["records"] == 200
    assert sealed["fit"]["sources"]["train.fit"]["records"] == 500
    assert set(sealed["fit"]["profile_counts"].values()) == {140}
    assert sealed["precal_holdout"]["records"] == 200
    assert set(sealed["precal_holdout"]["profile_counts"].values()) == {40}
    assert sealed["precal_holdout"]["model_evaluation_count"] == 0
    assert sealed["precal_holdout"]["consumption_status"] == "unconsumed_sealed"
    assert sealed["sealed_calibration"]["jsonl_content_parse_count"] == 0
    assert sealed["inherited_locked"]["jsonl_content_parse_count"] == 0
    assert sealed["freshness"]["unique_prior_and_fresh_records"] == 1452
    assert sealed["contract_sha256"] == contract._self_digest(
        sealed, "contract_sha256"
    )


def test_immutable_writer_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    contract._write_immutable_json(path, {"status": "first"})
    with pytest.raises(FileExistsError):
        contract._write_immutable_json(path, {"status": "second"})
