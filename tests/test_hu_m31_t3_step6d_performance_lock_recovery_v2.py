from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from ofc_regular import run_hu_m31_t3_step6d_performance_v2 as subject
from ofc_regular.cards import ALL_CARDS
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.hu_m31_t3_step6d_contract import SEED_SCHEDULES, planned_seed_values
from ofc_regular.state import Board


REPO_ROOT = Path(__file__).resolve().parents[1]
ACCEPTED_CANDIDATE_SHA256 = (
    "4050e04b22d7943da9e1691402a2b34cf3d3781041a44557f35e2dfcf366d55d"
)
ACCEPTED_REFERENCE_SHA256 = (
    "3f831615d9e751b8e1f266f14dd2009903b3ac2a4094f7e47616e0aa372a01b0"
)


def _recovery_contract() -> dict[str, Any]:
    return subject.build_run_contract(
        candidate_library_sha256=ACCEPTED_CANDIDATE_SHA256,
        reference_library_sha256=ACCEPTED_REFERENCE_SHA256,
        variant=subject.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_VARIANT,
    )


def _observations() -> tuple[ActorObservation, ActorObservation]:
    first_hero = ALL_CARDS[:9]
    first_opponent = ALL_CARDS[9:18]
    first = ActorObservation(
        hero_board=Board.from_rows(
            top=first_hero[:2],
            middle=first_hero[2:6],
            bottom=first_hero[6:],
        ),
        opponent_public_board=Board.from_rows(
            top=first_opponent[:2],
            middle=first_opponent[2:6],
            bottom=first_opponent[6:],
        ),
        dealt_cards=ALL_CARDS[18:21],
        hero_private_discards=ALL_CARDS[21:23],
        seat="first",
        street="T3",
        to_act_order="first",
    )
    second_hero = ALL_CARDS[:9]
    second_opponent = ALL_CARDS[9:20]
    second = ActorObservation(
        hero_board=Board.from_rows(
            top=second_hero[:2],
            middle=second_hero[2:6],
            bottom=second_hero[6:],
        ),
        opponent_public_board=Board.from_rows(
            top=second_opponent[:3],
            middle=second_opponent[3:8],
            bottom=second_opponent[8:],
        ),
        dealt_cards=ALL_CARDS[20:23],
        hero_private_discards=ALL_CARDS[23:25],
        seat="second",
        street="T3",
        to_act_order="second",
    )
    return first, second


def test_recovery_seed_contract_is_fresh_complete_and_deterministic() -> None:
    seed_contract = subject.candidate02_performance_lock_recovery_seed_contract()
    rows = [
        seed
        for index in subject.CONTRACT_HAND_INDICES
        for seed in subject.candidate02_performance_lock_recovery_seed_values(
            index
        ).values()
    ]
    recovery = set(rows)
    development = {
        seed
        for index in subject.CONTRACT_HAND_INDICES
        for seed in subject.candidate02_seed_values(index).values()
    }
    performance_lock_v1 = {
        seed
        for index in subject.CONTRACT_HAND_INDICES
        for seed in subject.candidate02_performance_lock_seed_values(index).values()
    }

    assert seed_contract["schema"].endswith(
        "performance_lock_recovery_seed_schedule_v2"
    )
    assert seed_contract["schedule"] == (
        subject.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_SCHEDULE
    )
    assert seed_contract["role"] == subject.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_ROLE
    assert seed_contract["namespace_bases"] == {
        "hand": 700_108_071_901,
        "behavior": 701_108_071_901,
        "candidate": 702_108_071_901,
        "evaluation": 703_108_071_901,
        "child": 704_108_071_901,
        "confirmation": 705_108_071_901,
    }
    assert seed_contract["seed_set_sha256"] == (
        "0df66657f263eb6de858706060620b4212e0e60173a0488bf0ddb3454c143aaa"
    )
    assert len(rows) == len(recovery) == 600
    assert (min(rows), max(rows)) == (700_108_071_901, 705_207_072_198)
    assert recovery.isdisjoint(planned_seed_values())
    assert recovery.isdisjoint(development)
    assert recovery.isdisjoint(performance_lock_v1)
    assert seed_contract["existing_step6d_schedule_overlap_counts"] == {
        schedule.name: 0 for schedule in SEED_SCHEDULES
    }
    assert seed_contract["existing_step6d_union_overlap_count"] == 0
    assert seed_contract["candidate02_development_overlap_count"] == 0
    assert seed_contract["performance_lock_v1_overlap_count"] == 0

    profiles: Counter[str] = Counter()
    for index in subject.CONTRACT_HAND_INDICES:
        row = subject.candidate02_performance_lock_recovery_schedule_row(index)
        assert row == subject.candidate02_performance_lock_recovery_schedule_row(index)
        assert row["root_indices"] == [index * 2, index * 2 + 1]
        assert row["profile"] == subject.v1.behavior_profile_for_index(index)
        assert row["seeds"] == (
            subject.candidate02_performance_lock_recovery_seed_values(index)
        )
        profiles[row["profile"]] += 1
    assert set(profiles.values()) == {20}


def test_recovery_contract_is_distinct_locked_and_fail_closed() -> None:
    contract = _recovery_contract()
    required_exports = {
        "CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_VARIANT",
        "CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_SCHEDULE",
        "CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_ROLE",
        "CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_RUN_ID",
        "CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_SEED_SET_SHA256",
        "candidate02_performance_lock_recovery_seed_values",
        "candidate02_performance_lock_recovery_seed_contract",
        "candidate02_performance_lock_recovery_schedule_row",
    }

    assert required_exports.issubset(subject.__all__)
    assert contract["schema"] == (
        subject.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_RUN_CONTRACT_SCHEMA
    )
    assert contract["step6d_run_id"] == (
        subject.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_RUN_ID
    )
    assert contract["schedule"] == (
        subject.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_SCHEDULE
    )
    assert contract["candidate_variant"] == (
        subject.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_VARIANT
    )
    assert contract["tail_hand_indices"] == []
    assert contract["seed_contract"] == (
        subject.candidate02_performance_lock_recovery_seed_contract()
    )
    assert subject.validate_run_contract(contract) == contract
    assert _recovery_contract() == contract
    assert subject.canonical_sha256(contract) == (
        "f02e8845401eef92ba617f2c832204bdc74c20aa9e91810c52bf99687c4886b5"
    )
    assert contract["contract_canonical_sha256"] == (
        "be575881b4d03bdb501cf1146c1936da3475df11addcfb6aa9bde077324c7599"
    )

    for candidate_sha256, reference_sha256 in (
        ("a" * 64, ACCEPTED_REFERENCE_SHA256),
        (ACCEPTED_CANDIDATE_SHA256, "b" * 64),
    ):
        with pytest.raises(ValueError, match="accepted binary hashes"):
            subject.build_run_contract(
                candidate_library_sha256=candidate_sha256,
                reference_library_sha256=reference_sha256,
                variant=subject.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_VARIANT,
            )

    tampered_seed = deepcopy(contract)
    tampered_seed["seed_contract"]["seed_min"] += 1
    for tampered in (
        dict(contract, schedule=subject.CANDIDATE02_PERFORMANCE_LOCK_SCHEDULE),
        dict(contract, contract_canonical_sha256="0" * 64),
        dict(contract, candidate_library_sha256="a" * 64),
        dict(contract, reference_library_sha256="b" * 64),
        tampered_seed,
    ):
        with pytest.raises(ValueError, match="shared run contract changed"):
            subject.validate_run_contract(tampered)


def test_recovery_does_not_change_candidate02_or_lock_v1_contracts() -> None:
    development = subject.build_run_contract(
        candidate_library_sha256=ACCEPTED_CANDIDATE_SHA256,
        reference_library_sha256=ACCEPTED_REFERENCE_SHA256,
        variant=subject.CANDIDATE02_VARIANT,
    )
    performance_lock_v1 = subject.build_run_contract(
        candidate_library_sha256=ACCEPTED_CANDIDATE_SHA256,
        reference_library_sha256=ACCEPTED_REFERENCE_SHA256,
        variant=subject.CANDIDATE02_PERFORMANCE_LOCK_VARIANT,
    )

    assert subject.canonical_sha256(development) == (
        "92a87f1544a73104d5256db39b022094c8c7f7b11f77bd19dcf1ab1442581ebd"
    )
    assert subject.canonical_sha256(performance_lock_v1) == (
        "e73c2b06279c1f1e91c38b2887ee465acf85f8f1072afef636512283252c34a4"
    )
    assert development["seed_contract"]["seed_set_sha256"] == (
        "173cd8d27fe918cab4552b89bbf5fd7929a3e0bd6955929bd5d4cc7d299f0c16"
    )
    assert performance_lock_v1["seed_contract"]["seed_set_sha256"] == (
        "b5a37a8f96d2995b9020ef568a3737ba8f6203c55794e7a2a98055a92ef179ab"
    )


def test_recovery_manifest_source_and_done_schemas_are_isolated() -> None:
    contract = _recovery_contract()
    manifest = subject.build_shard_manifest(
        run_contract=contract,
        source_role="candidate",
        work_hand_indices=[2, 6],
    )
    old_lock = subject.build_run_contract(
        candidate_library_sha256=ACCEPTED_CANDIDATE_SHA256,
        reference_library_sha256=ACCEPTED_REFERENCE_SHA256,
        variant=subject.CANDIDATE02_PERFORMANCE_LOCK_VARIANT,
    )
    old_manifest = subject.build_shard_manifest(
        run_contract=old_lock,
        source_role="candidate",
        work_hand_indices=[2, 6],
    )

    assert manifest["schema"] == (
        subject.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_SHARD_MANIFEST_SCHEMA
    )
    assert manifest["run_contract_digest"] == subject.canonical_sha256(contract)
    assert subject.validate_shard_manifest(manifest) == manifest
    assert subject._source_hand_schema(contract) == (
        subject.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_SOURCE_HAND_SCHEMA
    )
    assert subject._done_schema(contract) == (
        subject.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_DONE_SCHEMA
    )
    assert old_manifest["schema"] == subject.SHARD_MANIFEST_SCHEMA
    assert subject._source_hand_schema(old_lock) == (
        subject.CANDIDATE02_PERFORMANCE_LOCK_SOURCE_HAND_SCHEMA
    )
    assert subject._done_schema(old_lock) == (
        subject.CANDIDATE02_PERFORMANCE_LOCK_DONE_SCHEMA
    )

    with pytest.raises(ValueError, match="shard manifest changed"):
        subject.validate_shard_manifest(
            dict(manifest, schema=subject.SHARD_MANIFEST_SCHEMA)
        )


def test_recovery_root_materialization_is_deterministic_and_resume_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _recovery_contract()
    observations = _observations()
    calls: list[tuple[int, int, str]] = []

    monkeypatch.setattr(
        subject.v1, "load_model_bundle", lambda *args, **kwargs: object()
    )

    def fake_generate_behavior_t3_roots(
        *, hand_seed: int, behavior_seed: int, profile: str, bundle: object
    ) -> tuple[ActorObservation, ActorObservation]:
        assert bundle is not None
        calls.append((hand_seed, behavior_seed, profile))
        return observations

    monkeypatch.setattr(
        subject.v1,
        "generate_behavior_t3_roots",
        fake_generate_behavior_t3_roots,
    )
    output_a = tmp_path / "a"
    output_b = tmp_path / "b"
    roots_a = subject._materialize_candidate02_performance_lock_recovery_roots(
        repository_root=REPO_ROOT,
        output_dir=output_a,
        indices=[2],
    )
    roots_b = subject._materialize_candidate02_performance_lock_recovery_roots(
        repository_root=REPO_ROOT,
        output_dir=output_b,
        indices=[2],
    )
    path_a = output_a / "roots" / "hand_002.json"
    path_b = output_b / "roots" / "hand_002.json"

    assert roots_a == roots_b
    assert path_a.read_bytes() == path_b.read_bytes()
    assert roots_a[0]["schema"] == (
        subject.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_ROOT_SCHEMA
    )
    assert roots_a[0]["schedule"] == (
        subject.CANDIDATE02_PERFORMANCE_LOCK_RECOVERY_SCHEDULE
    )
    assert roots_a[0]["seeds"] == (
        subject.candidate02_performance_lock_recovery_seed_values(2)
    )
    first, second = subject._validate_root_artifact(contract, roots_a[0], index=2)
    assert (first.seat, second.seat) == ("first", "second")
    assert calls == [
        (
            roots_a[0]["seeds"]["hand"],
            roots_a[0]["seeds"]["behavior"],
            roots_a[0]["profile"],
        ),
        (
            roots_a[0]["seeds"]["hand"],
            roots_a[0]["seeds"]["behavior"],
            roots_a[0]["profile"],
        ),
    ]

    def reject_regeneration(**kwargs: Any) -> tuple[ActorObservation, ActorObservation]:
        raise AssertionError(f"resume regenerated a locked root: {kwargs}")

    monkeypatch.setattr(
        subject.v1,
        "generate_behavior_t3_roots",
        reject_regeneration,
    )
    resumed = subject._materialize_candidate02_performance_lock_recovery_roots(
        repository_root=REPO_ROOT,
        output_dir=output_a,
        indices=[2],
    )
    assert resumed == roots_a
    assert path_a.read_bytes() == path_b.read_bytes()


def test_recovery_root_materialization_routes_only_to_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _recovery_contract()
    sentinel = [{"hand_index": 2, "source": "recovery"}]
    calls: list[tuple[Path, Path, tuple[int, ...]]] = []

    def fake_recovery_materializer(**kwargs: Any) -> list[dict[str, Any]]:
        calls.append(
            (
                kwargs["repository_root"],
                kwargs["output_dir"],
                tuple(kwargs["indices"]),
            )
        )
        return sentinel

    monkeypatch.setattr(
        subject,
        "_materialize_candidate02_performance_lock_recovery_roots",
        fake_recovery_materializer,
    )
    result = subject._materialize_roots(
        contract=contract,
        repository_root=REPO_ROOT,
        output_dir=tmp_path,
        indices=[2],
    )
    assert result is sentinel
    assert calls == [(REPO_ROOT, tmp_path, (2,))]
