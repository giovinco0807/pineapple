from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from ofc_regular import hu_m31_t3_abr_cli_v1 as subject
from ofc_regular import hu_m31_t3_abr_teacher_v1 as abr_teacher
from ofc_regular import hu_m31_t3_abr_v1 as abr
from ofc_regular import hu_m31_t3_step6d_locked_promotion_v1 as promotion


def _write(path: Path, value: dict[str, Any]) -> str:
    path.write_bytes(promotion.canonical_bytes(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_raw_example_document_is_canonical_and_tamper_evident(
    tmp_path: Path,
) -> None:
    examples = [
        {
            "example_id": "dev-0001",
            "source_seed": 123,
            "observation": {"public": True},
            "family_action_values": {"family": [0.0]},
        }
    ]
    path = tmp_path / "raw.json"
    first = subject.write_raw_examples_document(
        raw_examples=examples, output_path=path
    )
    second = subject.write_raw_examples_document(
        raw_examples=examples, output_path=path
    )
    assert first == second
    assert first["locked_seed_training_allowed"] is False
    assert first["opponent_private_discards_used"] is False
    assert first["realized_deck_tail_used"] is False
    assert path.read_bytes() == promotion.canonical_bytes(first)

    changed = dict(first)
    changed["example_count"] = 2
    with pytest.raises(ValueError, match="contract"):
        subject.validate_raw_examples_document(changed)


def test_three_bundle_publish_is_atomic_and_resume_replays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = {"kind": "plan"}
    candidate = {"kind": "candidate"}
    dataset = {"kind": "dataset"}
    plan_path = tmp_path / "plan.json"
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    candidate_path = candidate_root / "manifest.json"
    dataset_path = tmp_path / "dataset.json"
    plan_sha = _write(plan_path, plan)
    candidate_sha = _write(candidate_path, candidate)
    dataset_sha = _write(dataset_path, dataset)
    monkeypatch.setattr(
        subject.promotion,
        "validate_locked_promotion_plan",
        lambda value: dict(value),
    )
    monkeypatch.setattr(
        subject.abr,
        "validate_development_dataset",
        lambda value, **kwargs: dict(value),
    )
    writes: list[Path] = []
    frozen_bundle = {
        "bundle_identity_sha256": "a" * 64,
        "family_count": 3,
        "families": [
            {"response_id": response_id}
            for response_id in abr.RESPONSE_IDS
        ],
    }

    def write_bundle(*, output_directory, **kwargs):
        root = Path(output_directory)
        root.mkdir()
        writes.append(root)
        (root / "bundle.json").write_bytes(
            promotion.canonical_bytes(frozen_bundle)
        )
        return dict(frozen_bundle)

    monkeypatch.setattr(subject.abr, "write_policy_bundle", write_bundle)
    monkeypatch.setattr(
        subject.abr,
        "validate_policy_bundle",
        lambda value, **kwargs: dict(value),
    )
    output = tmp_path / "abr-bundle"
    first = subject.train_three_policy_bundle(
        promotion_plan_path=plan_path,
        expected_promotion_plan_sha256=plan_sha,
        candidate_bundle_directory=candidate_root,
        expected_candidate_manifest_sha256=candidate_sha,
        development_dataset_path=dataset_path,
        expected_development_dataset_sha256=dataset_sha,
        production_provenance_required=False,
        output_directory=output,
        torch=object(),
    )
    second = subject.train_three_policy_bundle(
        promotion_plan_path=plan_path,
        expected_promotion_plan_sha256=plan_sha,
        candidate_bundle_directory=candidate_root,
        expected_candidate_manifest_sha256=candidate_sha,
        development_dataset_path=dataset_path,
        expected_development_dataset_sha256=dataset_sha,
        production_provenance_required=False,
        output_directory=output,
        torch=object(),
    )
    assert first["status"] == "complete"
    assert second["status"] == "resume_complete"
    assert first["abr_bundle_file_sha256"] == second["abr_bundle_file_sha256"]
    assert len(writes) == 1
    assert output.is_dir()
    assert not list(tmp_path.glob(".abr-bundle.build-*"))


def test_bundle_resume_rejects_partial_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path = tmp_path / "plan.json"
    candidate_root = tmp_path / "candidate"
    candidate_root.mkdir()
    candidate_path = candidate_root / "manifest.json"
    dataset_path = tmp_path / "dataset.json"
    plan_sha = _write(plan_path, {"kind": "plan"})
    candidate_sha = _write(candidate_path, {"kind": "candidate"})
    dataset_sha = _write(dataset_path, {"kind": "dataset"})
    monkeypatch.setattr(
        subject.promotion,
        "validate_locked_promotion_plan",
        lambda value: dict(value),
    )
    monkeypatch.setattr(
        subject.abr,
        "validate_development_dataset",
        lambda value, **kwargs: dict(value),
    )
    output = tmp_path / "partial"
    output.mkdir()
    with pytest.raises(ValueError, match="bundle"):
        subject.train_three_policy_bundle(
            promotion_plan_path=plan_path,
            expected_promotion_plan_sha256=plan_sha,
            candidate_bundle_directory=candidate_root,
            expected_candidate_manifest_sha256=candidate_sha,
            development_dataset_path=dataset_path,
            expected_development_dataset_sha256=dataset_sha,
            production_provenance_required=False,
            output_directory=output,
            torch=object(),
        )


def test_production_build_receipt_binds_t3_only_runtime_composition() -> None:
    count = abr_teacher.MAX_DEVELOPMENT_PAIRS
    teacher_receipt = {
        "production_training_authorized": True,
        "pair_count": count,
        "example_count": count * 2,
        "paired_count_per_response": {
            response_id: count for response_id in abr.RESPONSE_IDS
        },
        "root_count_per_response": {
            response_id: count * 2 for response_id in abr.RESPONSE_IDS
        },
        "teacher_receipt_identity_sha256": "1" * 64,
    }
    dataset = {
        "example_count": count * 2,
        "response_ids": list(abr.RESPONSE_IDS),
        "dataset_identity_sha256": "2" * 64,
    }
    bundle = {
        "family_count": len(abr.RESPONSE_IDS),
        "families": [
            {"response_id": response_id}
            for response_id in abr.RESPONSE_IDS
        ],
        "bundle_identity_sha256": "3" * 64,
    }
    receipt = subject.build_production_build_receipt(
        teacher_receipt=teacher_receipt,
        teacher_receipt_file_sha256="4" * 64,
        raw_examples_file_sha256="5" * 64,
        development_dataset=dataset,
        development_dataset_file_sha256="6" * 64,
        bundle=bundle,
        bundle_file_sha256="7" * 64,
    )
    validated = subject.validate_production_build_receipt(
        receipt,
        bundle_file_sha256="7" * 64,
        bundle_identity_sha256="3" * 64,
    )
    assert validated["training_street"] == "T3"
    assert validated["runtime_street_composition"] == {
        "T0": "stage19_p0",
        "T1": "stage18_p1",
        "T2": "stage9f_p2",
        "T3": "artifact_bound_learned_abr",
        "T4": "hu_m3_t4_exact_both_seats_v1",
    }
    assert validated["response_scientific_roles"][
        "greedy_search_response"
    ] == "direct_hu_score_approximate_best_response"
    changed = dict(receipt)
    changed["training_street"] = "T2"
    identity = dict(changed)
    identity.pop("production_build_receipt_identity_sha256")
    changed["production_build_receipt_identity_sha256"] = (
        promotion.canonical_sha256(identity)
    )
    with pytest.raises(ValueError, match="receipt changed"):
        subject.validate_production_build_receipt(
            changed,
            bundle_file_sha256="7" * 64,
            bundle_identity_sha256="3" * 64,
        )
