from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from ofc_regular import run_hu_m31_t3_step6d_performance_v2 as subject
from ofc_regular.cards import ALL_CARDS
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.state import Board


REPO_ROOT = Path(__file__).resolve().parents[1]


def _contract() -> dict[str, Any]:
    return subject.build_run_contract(
        candidate_library_sha256="a" * 64,
        reference_library_sha256="b" * 64,
    )


def _candidate02_contract() -> dict[str, Any]:
    return subject.build_run_contract(
        candidate_library_sha256="a" * 64,
        reference_library_sha256="b" * 64,
        variant=subject.CANDIDATE02_VARIANT,
    )


def _candidate02_tail_v2_contract() -> dict[str, Any]:
    return subject.build_run_contract(
        candidate_library_sha256="a" * 64,
        reference_library_sha256="b" * 64,
        variant=subject.CANDIDATE02_TAIL_V2_VARIANT,
    )


def _candidate02_performance_lock_contract() -> dict[str, Any]:
    return subject.build_run_contract(
        candidate_library_sha256="a" * 64,
        reference_library_sha256="b" * 64,
        variant=subject.CANDIDATE02_PERFORMANCE_LOCK_VARIANT,
    )


def test_shared_contract_digest_excludes_source_role_and_work_subset() -> None:
    contract = _contract()
    candidate = subject.build_shard_manifest(
        run_contract=contract,
        source_role="candidate",
        work_hand_indices=[2, 6],
    )
    reference = subject.build_shard_manifest(
        run_contract=contract,
        source_role="reference",
        work_hand_indices=[20, 50],
    )
    assert candidate["run_contract_digest"] == reference["run_contract_digest"]
    assert candidate["run_contract"] == reference["run_contract"] == contract
    assert "source_role" not in contract
    assert "work_hand_indices" not in contract
    assert contract["contract_hand_indices"] == list(range(100))
    assert contract["tail_hand_indices"] == list(subject.TAIL_HAND_INDICES)


def test_candidate01_contract_digest_remains_byte_compatible() -> None:
    contract = subject.build_run_contract(
        candidate_library_sha256=(
            "4fd193237878c223b2729fa637c79884b392b50ae51950c9c53073d5264cd86f"
        ),
        reference_library_sha256=(
            "3f831615d9e751b8e1f266f14dd2009903b3ac2a4094f7e47616e0aa372a01b0"
        ),
    )
    assert subject.canonical_sha256(contract) == (
        "e5b8c77d7f48a59fcd0221022bd2146c16d18b52c99d6827ed9252d13b9a4d72"
    )


def test_candidate02_contract_binds_fresh_globally_disjoint_seed_set() -> None:
    contract = _candidate02_contract()
    seeds = [
        value
        for index in subject.CONTRACT_HAND_INDICES
        for value in subject.candidate02_seed_values(index).values()
    ]
    old_seeds = {
        value
        for index in subject.CONTRACT_HAND_INDICES
        for value in subject.v1.performance_seed_values(index).values()
    }
    assert contract["schema"] == subject.CANDIDATE02_RUN_CONTRACT_SCHEMA
    assert contract["candidate_variant"] == subject.CANDIDATE02_VARIANT
    assert contract["schedule"] == subject.CANDIDATE02_SCHEDULE
    assert contract["step6d_run_id"] == subject.CANDIDATE02_RUN_ID
    assert contract["seed_contract"] == subject.candidate02_seed_contract()
    assert len(seeds) == len(set(seeds)) == 600
    assert min(seeds) > 680_607_073_398
    assert set(seeds).isdisjoint(old_seeds)


def test_candidate02_contract_fails_closed_on_variant_or_seed_tamper() -> None:
    contract = _candidate02_contract()
    with pytest.raises(ValueError, match="shared run contract changed"):
        subject.validate_run_contract(dict(contract, candidate_variant="candidate01"))
    tampered = dict(contract)
    tampered["seed_contract"] = dict(
        contract["seed_contract"],
        seed_min=contract["seed_contract"]["seed_min"] + 1,
    )
    with pytest.raises(ValueError, match="shared run contract changed"):
        subject.validate_run_contract(tampered)
    with pytest.raises(ValueError, match="variant"):
        subject.build_run_contract(
            candidate_library_sha256="a" * 64,
            reference_library_sha256="b" * 64,
            variant="candidate03",
        )


def test_candidate02_v1_contract_digest_remains_byte_compatible() -> None:
    contract = subject.build_run_contract(
        candidate_library_sha256=(
            "4050e04b22d7943da9e1691402a2b34cf3d3781041a44557f35e2dfcf366d55d"
        ),
        reference_library_sha256=(
            "3f831615d9e751b8e1f266f14dd2009903b3ac2a4094f7e47616e0aa372a01b0"
        ),
        variant=subject.CANDIDATE02_VARIANT,
    )
    assert subject.canonical_sha256(contract) == (
        "92a87f1544a73104d5256db39b022094c8c7f7b11f77bd19dcf1ab1442581ebd"
    )


def test_candidate02_performance_lock_contract_binds_preregistered_schedule() -> None:
    contract = _candidate02_performance_lock_contract()
    seeds = [
        value
        for index in subject.CONTRACT_HAND_INDICES
        for value in subject.candidate02_performance_lock_seed_values(index).values()
    ]
    development = {
        value
        for index in subject.CONTRACT_HAND_INDICES
        for value in subject.candidate02_seed_values(index).values()
    }
    assert (
        contract["schema"] == subject.CANDIDATE02_PERFORMANCE_LOCK_RUN_CONTRACT_SCHEMA
    )
    assert contract["candidate_variant"] == subject.CANDIDATE02_PERFORMANCE_LOCK_VARIANT
    assert contract["schedule"] == "performance_lock"
    assert contract["step6d_run_id"] == subject.CANDIDATE02_PERFORMANCE_LOCK_RUN_ID
    assert contract["tail_hand_indices"] == []
    assert (
        contract["seed_contract"]
        == subject.candidate02_performance_lock_seed_contract()
    )
    assert contract["seed_contract"]["role"] == (
        "one_shot_performance_qualification_only"
    )
    assert contract["seed_contract"]["namespace_bases"] == {
        "hand": 490_108_071_901,
        "behavior": 491_108_071_901,
        "candidate": 492_108_071_901,
        "evaluation": 493_108_071_901,
        "child": 494_108_071_901,
        "confirmation": 495_108_071_901,
    }
    assert contract["seed_contract"]["seed_stride"] == 1_000_003
    assert contract["seed_contract"]["seed_set_sha256"] == (
        "b5a37a8f96d2995b9020ef568a3737ba8f6203c55794e7a2a98055a92ef179ab"
    )
    assert len(seeds) == len(set(seeds)) == 600
    assert (min(seeds), max(seeds)) == (490_108_071_901, 495_207_072_198)
    assert set(seeds).isdisjoint(development)


def test_candidate02_performance_lock_real_contract_digest_is_frozen() -> None:
    contract = subject.build_run_contract(
        candidate_library_sha256=(
            "4050e04b22d7943da9e1691402a2b34cf3d3781041a44557f35e2dfcf366d55d"
        ),
        reference_library_sha256=(
            "3f831615d9e751b8e1f266f14dd2009903b3ac2a4094f7e47616e0aa372a01b0"
        ),
        variant=subject.CANDIDATE02_PERFORMANCE_LOCK_VARIANT,
    )
    assert subject.canonical_sha256(contract) == (
        "e73c2b06279c1f1e91c38b2887ee465acf85f8f1072afef636512283252c34a4"
    )


def test_candidate02_performance_lock_contract_tamper_fails_closed() -> None:
    contract = _candidate02_performance_lock_contract()
    mutations = [
        dict(contract, schedule=subject.CANDIDATE02_SCHEDULE),
        dict(contract, tail_hand_indices=[0]),
        dict(
            contract,
            seed_contract=dict(
                contract["seed_contract"],
                role="repeatable_performance_engineering_only",
            ),
        ),
    ]
    for value in mutations:
        with pytest.raises(ValueError, match="shared run contract changed"):
            subject.validate_run_contract(value)


def test_candidate02_tail_v2_contract_binds_reselection_and_reuses_roots_seeds() -> (
    None
):
    from ofc_regular import (
        select_hu_m31_t3_step6d_candidate02_tail_v2 as selector,
    )

    contract = _candidate02_tail_v2_contract()
    candidate02_v1 = _candidate02_contract()
    assert contract["schema"] == subject.CANDIDATE02_TAIL_V2_RUN_CONTRACT_SCHEMA
    assert contract["candidate_variant"] == subject.CANDIDATE02_TAIL_V2_VARIANT
    assert contract["step6d_run_id"] == subject.CANDIDATE02_TAIL_V2_RUN_ID
    assert contract["schedule"] == candidate02_v1["schedule"]
    assert contract["seed_contract"] == candidate02_v1["seed_contract"]
    assert contract["tail_hand_indices"] == list(
        subject.CANDIDATE02_TAIL_V2_TAIL_HAND_INDICES
    )
    assert (
        contract["selection_manifest_sha256"]
        == subject.CANDIDATE02_TAIL_V2_SELECTION_MANIFEST_SHA256
    )
    assert subject.CANDIDATE02_TAIL_V2_HEAVY_HAND_INDICES == (
        selector.HEAVY_HAND_INDICES
    )
    assert subject.CANDIDATE02_TAIL_V2_RANDOM_HAND_INDICES == (
        selector.RANDOM_HAND_INDICES
    )
    assert subject.CANDIDATE02_TAIL_V2_PRIOR_EXPOSED_HAND_INDICES == (
        selector.PRIOR_RUNTIME_EXPOSED_HAND_INDICES
    )
    assert contract["contract_canonical_sha256"] != (
        candidate02_v1["contract_canonical_sha256"]
    )


def test_candidate02_tail_v2_contract_and_work_tamper_fail_closed() -> None:
    contract = _candidate02_tail_v2_contract()
    with pytest.raises(ValueError, match="shared run contract changed"):
        subject.validate_run_contract(
            dict(contract, selection_manifest_sha256="0" * 64)
        )
    with pytest.raises(ValueError, match="selected tail"):
        subject.build_shard_manifest(
            run_contract=contract,
            source_role="candidate",
            work_hand_indices=[2],
        )
    manifest = subject.build_shard_manifest(
        run_contract=contract,
        source_role="candidate",
        work_hand_indices=[0, 4],
    )
    manifest["work_hand_indices"] = [0, 2]
    with pytest.raises(ValueError, match="selected tail"):
        subject.validate_shard_manifest(manifest)


def test_contract_is_exact_1x16_and_binds_both_distinct_binaries() -> None:
    contract = _contract()
    assert contract["allocation"] == {
        "workers": 1,
        "rayon_threads_per_worker": 16,
    }
    with pytest.raises(ValueError, match="1 worker x 16"):
        subject.build_run_contract(
            candidate_library_sha256="a" * 64,
            reference_library_sha256="b" * 64,
            workers=2,
            rayon_threads_per_worker=8,
        )
    with pytest.raises(ValueError, match="distinct"):
        subject.build_run_contract(
            candidate_library_sha256="a" * 64,
            reference_library_sha256="a" * 64,
        )
    with pytest.raises(ValueError, match="all 100"):
        subject.build_run_contract(
            candidate_library_sha256="a" * 64,
            reference_library_sha256="b" * 64,
            contract_hand_indices=range(10),
        )


@pytest.mark.parametrize(
    ("role", "indices", "message"),
    [
        ("other", [2], "source_role"),
        ("candidate", [2, 2], "unique"),
        ("candidate", [6, 2], "sorted"),
        ("candidate", [100], "0..99"),
        ("candidate", [], "must not be empty"),
    ],
)
def test_shard_manifest_fails_closed(
    role: str, indices: list[int], message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        subject.build_shard_manifest(
            run_contract=_contract(),
            source_role=role,
            work_hand_indices=indices,
        )


def test_contract_and_manifest_reject_unknown_fields() -> None:
    contract = _contract()
    bad_contract = dict(contract, source_role="candidate")
    with pytest.raises(ValueError, match="keys changed"):
        subject.validate_run_contract(bad_contract)
    manifest = subject.build_shard_manifest(
        run_contract=contract,
        source_role="candidate",
        work_hand_indices=[2],
    )
    with pytest.raises(ValueError, match="keys changed"):
        subject.validate_shard_manifest(dict(manifest, fallback=True))


def _first_21x21_observation() -> ActorObservation:
    # top/middle/bottom counts 2/4/3 leave 1/1/2 slots: 21 legal actions.
    hero = ALL_CARDS[:9]
    opponent = ALL_CARDS[9:18]
    return ActorObservation(
        hero_board=Board.from_rows(top=hero[:2], middle=hero[2:6], bottom=hero[6:]),
        opponent_public_board=Board.from_rows(
            top=opponent[:2], middle=opponent[2:6], bottom=opponent[6:]
        ),
        dealt_cards=ALL_CARDS[18:21],
        hero_private_discards=ALL_CARDS[21:23],
        seat="first",
        street="T3",
        to_act_order="first",
    )


def _candidate02_root() -> dict[str, Any]:
    row = subject.candidate02_schedule_row(2)
    first = _first_21x21_observation()
    hero = ALL_CARDS[:9]
    opponent = ALL_CARDS[9:20]
    second = ActorObservation(
        hero_board=Board.from_rows(top=hero[:2], middle=hero[2:6], bottom=hero[6:]),
        opponent_public_board=Board.from_rows(
            top=opponent[:3], middle=opponent[3:8], bottom=opponent[8:]
        ),
        dealt_cards=ALL_CARDS[20:23],
        hero_private_discards=ALL_CARDS[23:25],
        seat="second",
        street="T3",
        to_act_order="second",
    )
    observations = (first, second)
    return {
        "schema": subject.CANDIDATE02_ROOT_SCHEMA,
        "contract_canonical_sha256": (subject._candidate02_contract_anchor_sha256()),
        "schedule": subject.CANDIDATE02_SCHEDULE,
        "schedule_row_sha256": subject.canonical_sha256(row),
        "hand_index": 2,
        "root_indices": row["root_indices"],
        "profile": row["profile"],
        "seeds": row["seeds"],
        "budget": row["budget"],
        "observations": [
            {
                "root_index": row["root_indices"][offset],
                "seat": observation.seat,
                "observation_fingerprint": observation.fingerprint(),
                "observation": observation.to_dict(),
            }
            for offset, observation in enumerate(observations)
        ],
        "current_profile_resolved": False,
        "opponent_private_discards_used": False,
        "training_eligible": False,
    }


def _candidate02_performance_lock_root() -> dict[str, Any]:
    row = subject.candidate02_performance_lock_schedule_row(2)
    first = _first_21x21_observation()
    hero = ALL_CARDS[:9]
    opponent = ALL_CARDS[9:20]
    second = ActorObservation(
        hero_board=Board.from_rows(top=hero[:2], middle=hero[2:6], bottom=hero[6:]),
        opponent_public_board=Board.from_rows(
            top=opponent[:3], middle=opponent[3:8], bottom=opponent[8:]
        ),
        dealt_cards=ALL_CARDS[20:23],
        hero_private_discards=ALL_CARDS[23:25],
        seat="second",
        street="T3",
        to_act_order="second",
    )
    observations = (first, second)
    return {
        "schema": subject.CANDIDATE02_PERFORMANCE_LOCK_ROOT_SCHEMA,
        "contract_canonical_sha256": (
            subject._candidate02_performance_lock_contract_anchor_sha256()
        ),
        "schedule": subject.CANDIDATE02_PERFORMANCE_LOCK_SCHEDULE,
        "schedule_row_sha256": subject.canonical_sha256(row),
        "hand_index": 2,
        "root_indices": row["root_indices"],
        "profile": row["profile"],
        "seeds": row["seeds"],
        "budget": row["budget"],
        "observations": [
            {
                "root_index": row["root_indices"][offset],
                "seat": observation.seat,
                "observation_fingerprint": observation.fingerprint(),
                "observation": observation.to_dict(),
            }
            for offset, observation in enumerate(observations)
        ],
        "current_profile_resolved": False,
        "opponent_private_discards_used": False,
        "training_eligible": False,
    }


def test_candidate02_root_schema_and_seed_provenance_fail_closed() -> None:
    contract = _candidate02_contract()
    root = _candidate02_root()
    first, second = subject._validate_root_artifact(contract, root, index=2)
    assert (first.seat, second.seat) == ("first", "second")
    assert root["seeds"] == subject.candidate02_seed_values(2)
    tampered = dict(root)
    tampered["seeds"] = dict(root["seeds"], hand=root["seeds"]["hand"] + 1)
    with pytest.raises(ValueError, match="root provenance changed"):
        subject._validate_root_artifact(contract, tampered, index=2)
    legacy = _contract()
    with pytest.raises(ValueError, match="root schema changed"):
        subject._validate_root_artifact(legacy, root, index=2)


def test_candidate02_performance_lock_root_has_distinct_provenance() -> None:
    contract = _candidate02_performance_lock_contract()
    root = _candidate02_performance_lock_root()
    first, second = subject._validate_root_artifact(contract, root, index=2)
    assert (first.seat, second.seat) == ("first", "second")
    assert root["seeds"] == subject.candidate02_performance_lock_seed_values(2)
    assert root["schema"] != subject.CANDIDATE02_ROOT_SCHEMA
    assert root["contract_canonical_sha256"] != (
        subject._candidate02_contract_anchor_sha256()
    )

    tampered = dict(root, schedule=subject.CANDIDATE02_SCHEDULE)
    with pytest.raises(ValueError, match="performance-lock root provenance changed"):
        subject._validate_root_artifact(contract, tampered, index=2)
    with pytest.raises(ValueError, match="root schema changed"):
        subject._validate_root_artifact(_candidate02_contract(), root, index=2)
    hidden = deepcopy(root)
    hidden["observations"][0]["observation"]["opponent_private_discards"] = []
    with pytest.raises(ValueError, match="opponent_private_discards"):
        subject._validate_root_artifact(contract, hidden, index=2)


def test_candidate02_performance_lock_materialization_routing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _candidate02_performance_lock_contract()
    sentinel = [{"hand_index": 2, "source": "lock"}]
    calls: list[tuple[Path, Path, tuple[int, ...]]] = []

    def fake_lock_materializer(**kwargs: Any) -> list[dict[str, Any]]:
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
        "_materialize_candidate02_performance_lock_roots",
        fake_lock_materializer,
    )
    result = subject._materialize_roots(
        contract=contract,
        repository_root=REPO_ROOT,
        output_dir=tmp_path,
        indices=[2],
    )
    assert result is sentinel
    assert calls == [(REPO_ROOT, tmp_path, (2,))]


def test_geometry_records_explicit_first_seat_21x21_matrix() -> None:
    observation = _first_21x21_observation()
    portable = {
        "action_values": [{} for _ in range(21)],
        "child_information_set_count": 143_640,
    }
    geometry = subject._source_geometry(observation, portable)
    assert geometry == {
        "hero_legal_action_count": 21,
        "opponent_response_legal_action_count": 21,
        "first_seat_action_matrix_rows": 21,
        "first_seat_action_matrix_columns": 21,
        "first_seat_action_matrix_cells": 441,
        "child_information_set_count": 143_640,
        "candidate_q_count": 21,
        "evaluation_q_count": 21,
    }


def test_write_once_is_canonical_and_never_replaces(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    subject._write_once(path, {"b": 2, "a": 1})
    assert path.read_bytes() == b'{"a":1,"b":2}\n'
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        subject._write_once(path, {"a": 999})
    assert path.read_bytes() == b'{"a":1,"b":2}\n'


@pytest.mark.parametrize(
    "variant",
    [
        subject.CANDIDATE01_VARIANT,
        subject.CANDIDATE02_VARIANT,
        subject.CANDIDATE02_TAIL_V2_VARIANT,
        subject.CANDIDATE02_PERFORMANCE_LOCK_VARIANT,
    ],
)
def test_source_shard_interrupt_resume_and_done_are_role_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    candidate = tmp_path / "candidate.so"
    reference_bytes = b"reference"
    candidate_bytes = b"candidate"
    candidate.write_bytes(candidate_bytes)
    contract = subject.build_run_contract(
        candidate_library_sha256=hashlib.sha256(candidate_bytes).hexdigest(),
        reference_library_sha256=hashlib.sha256(reference_bytes).hexdigest(),
        variant=variant,
    )
    manifest = subject.build_shard_manifest(
        run_contract=contract,
        source_role="candidate",
        work_hand_indices=(
            [0, 4] if variant == subject.CANDIDATE02_TAIL_V2_VARIANT else [2, 6]
        ),
    )

    def fake_roots(**kwargs: Any) -> list[dict[str, Any]]:
        values = []
        for index in kwargs["indices"]:
            value = {"hand_index": index, "root": f"root-{index}"}
            path = kwargs["output_dir"] / "roots" / f"hand_{index:03d}.json"
            if path.exists():
                value = subject._read_canonical(path)
            else:
                subject._write_once(path, value)
            values.append(value)
        return values

    calls: list[tuple[int, str, Path]] = []

    def fake_run(**kwargs: Any) -> dict[str, Any]:
        calls.append(
            (
                kwargs["root"]["hand_index"],
                kwargs["source_role"],
                kwargs["library_path"],
            )
        )
        return {
            "hand_index": kwargs["root"]["hand_index"],
            "source_role": kwargs["source_role"],
            "run_contract_digest": kwargs["run_contract_digest"],
        }

    def fake_validate(value: Any, **kwargs: Any) -> dict[str, Any]:
        payload = dict(value)
        if (
            payload.get("hand_index") != kwargs["root"]["hand_index"]
            or payload.get("source_role") != kwargs["source_role"]
            or payload.get("run_contract_digest") != kwargs["run_contract_digest"]
        ):
            raise ValueError("fake source hand was tampered")
        return payload

    monkeypatch.setattr(subject.v1, "validate_performance_contract", lambda path: {})
    monkeypatch.setattr(subject.v1, "_materialize_roots", fake_roots)
    monkeypatch.setattr(subject, "_materialize_candidate02_roots", fake_roots)
    monkeypatch.setattr(
        subject, "_materialize_candidate02_performance_lock_roots", fake_roots
    )
    monkeypatch.setattr(subject, "_run_source_hand", fake_run)
    monkeypatch.setattr(subject, "_validate_source_hand", fake_validate)

    output = tmp_path / "output"
    interrupted = subject.run_source_shard(
        repository_root=REPO_ROOT,
        output_dir=output,
        shard_manifest=manifest,
        library_path=candidate,
        stop_after_hands=1,
    )
    assert interrupted["status"] == "interrupted_for_resume"
    expected_indices = (
        [0, 4] if variant == subject.CANDIDATE02_TAIL_V2_VARIANT else [2, 6]
    )
    assert interrupted["pending_hand_indices"] == [expected_indices[1]]
    assert not (output / "DONE.json").exists()
    assert calls == [(expected_indices[0], "candidate", candidate.resolve())]

    complete = subject.run_source_shard(
        repository_root=REPO_ROOT,
        output_dir=output,
        shard_manifest=manifest,
        library_path=candidate,
    )
    assert complete["status"] == "complete_source_isolated_shard"
    expected_done_schema = {
        subject.CANDIDATE01_VARIANT: subject.DONE_SCHEMA,
        subject.CANDIDATE02_VARIANT: subject.CANDIDATE02_DONE_SCHEMA,
        subject.CANDIDATE02_TAIL_V2_VARIANT: (subject.CANDIDATE02_TAIL_V2_DONE_SCHEMA),
        subject.CANDIDATE02_PERFORMANCE_LOCK_VARIANT: (
            subject.CANDIDATE02_PERFORMANCE_LOCK_DONE_SCHEMA
        ),
    }[variant]
    assert complete["schema"] == expected_done_schema
    assert complete["artifact_count"] == 4
    assert complete["completed_hand_indices"] == expected_indices
    assert calls == [
        (expected_indices[0], "candidate", candidate.resolve()),
        (expected_indices[1], "candidate", candidate.resolve()),
    ]
    assert all(
        record["source_role"] == "candidate" for record in complete["artifact_manifest"]
    )
    assert not any(
        "reference" in record["path"] for record in complete["artifact_manifest"]
    )

    repeated = subject.run_source_shard(
        repository_root=REPO_ROOT,
        output_dir=output,
        shard_manifest=manifest,
        library_path=candidate,
    )
    assert repeated == complete
    assert len(calls) == 2

    hand = output / "hands" / "candidate" / f"hand_{expected_indices[0]:03d}.json"
    hand.write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="canonical|tampered"):
        subject.run_source_shard(
            repository_root=REPO_ROOT,
            output_dir=output,
            shard_manifest=manifest,
            library_path=candidate,
        )


def test_runner_has_no_cloud_profile_or_batch_entrypoint() -> None:
    source = (
        REPO_ROOT / "src/ofc_regular/run_hu_m31_t3_step6d_performance_v2.py"
    ).read_text(encoding="utf-8")
    assert "gcloud" not in source.casefold()
    assert "set_current" not in source
    assert ".solve_many(" not in source
    assert '"training_eligible": True' not in source
