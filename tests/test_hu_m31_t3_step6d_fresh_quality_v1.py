from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ofc_regular.cards import create_deck
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.state import Board
import ofc_regular.hu_m31_t3_step6d_fresh_quality_v1 as subject


_AUTHORIZATION = {
    "schema": subject.PERFORMANCE_RECEIPT_SCHEMA,
    "status": "qualified",
    "decision": subject.QUALIFIED_DECISION,
    "receipt_sha256": "a" * 64,
    "performance_lock_qualified": True,
    "quality_pilot_authorized": True,
    "performance_lock_finalized": True,
    "one_shot_lock_consumed": True,
    "current_profile_changed": False,
}


def _plan(monkeypatch: pytest.MonkeyPatch) -> dict:
    monkeypatch.setattr(
        subject,
        "_load_performance_authorization",
        lambda _path: dict(_AUTHORIZATION),
    )
    return subject.build_fresh_quality_plan(
        performance_receipt_path=Path("unused.json")
    )


def _observations(pair_index: int, phase: str):
    deck = create_deck(shuffle=False)
    if phase == subject.CONFIRMATION_PHASE:
        deck = list(reversed(deck))
    offset = pair_index % len(deck)
    deck = deck[offset:] + deck[:offset]
    first_board = Board.from_rows(
        top=deck[0:2],
        middle=deck[2:6],
        bottom=deck[6:9],
    )
    second_board = Board.from_rows(
        top=deck[9:11],
        middle=deck[11:15],
        bottom=deck[15:18],
    )
    first = ActorObservation(
        hero_board=first_board,
        opponent_public_board=second_board,
        dealt_cards=tuple(deck[20:23]),
        hero_private_discards=tuple(deck[18:20]),
        seat="first",
        street="T3",
        to_act_order="first",
    )
    first_after = Board.from_rows(
        top=first_board.top,
        middle=(*first_board.middle, deck[20]),
        bottom=(*first_board.bottom, deck[21]),
    )
    second = ActorObservation(
        hero_board=second_board,
        opponent_public_board=first_after,
        dealt_cards=tuple(deck[25:28]),
        hero_private_discards=tuple(deck[23:25]),
        seat="second",
        street="T3",
        to_act_order="second",
    )
    return first, second


def test_frozen_schedule_is_50_plus_separate_5_and_seed_disjoint():
    rows = subject.schedule_rows()
    assert len(rows) == 55
    assert [row["root_indices"] for row in rows[:50]] == [
        [index * 2, index * 2 + 1] for index in range(50)
    ]
    assert [row["root_indices"] for row in rows[50:]] == [
        [100 + index * 2, 101 + index * 2] for index in range(5)
    ]
    assert {
        profile: sum(
            row["profile"] == profile and row["phase"] == subject.PRIMARY_PHASE
            for row in rows
        )
        for profile in subject.M31_T3_BEHAVIOR_PROFILES
    } == {profile: 10 for profile in subject.M31_T3_BEHAVIOR_PROFILES}
    assert {
        profile: sum(
            row["profile"] == profile and row["phase"] == subject.CONFIRMATION_PHASE
            for row in rows
        )
        for profile in subject.M31_T3_BEHAVIOR_PROFILES
    } == {profile: 1 for profile in subject.M31_T3_BEHAVIOR_PROFILES}
    seed_contract = subject.build_seed_contract()
    assert seed_contract["primary_seed_count"] == 300
    assert seed_contract["confirmation_seed_count"] == 30
    assert seed_contract["all_quality_seed_count"] == 330
    assert seed_contract["primary_confirmation_disjoint"] is True
    assert seed_contract["legacy_step6c_disjoint"] is True
    assert seed_contract["performance_v4_disjoint"] is True


def test_plan_is_exact_and_confirmation_is_not_primary_subset(
    monkeypatch: pytest.MonkeyPatch,
):
    plan = _plan(monkeypatch)
    assert subject.validate_fresh_quality_plan(plan) == plan
    assert plan["primary"]["paired_hand_count"] == 50
    assert plan["primary"]["root_count"] == 100
    assert plan["confirmation"]["paired_hand_count"] == 5
    assert plan["confirmation"]["root_count"] == 10
    assert plan["confirmation"]["budget"] == {
        "candidate_samples": 8,
        "evaluation_samples": 128,
        "downstream_t3_samples": 4,
        "downstream_t4_samples": 0,
    }
    assert (
        plan["execution_contract"]["primary_and_confirmation_root_populations_disjoint"]
        is True
    )
    assert plan["scientific_boundaries"]["current_profile_changed"] is False


def test_plan_rejects_unknown_field_and_unqualified_receipt(
    monkeypatch: pytest.MonkeyPatch,
):
    plan = _plan(monkeypatch)
    plan["unknown"] = True
    with pytest.raises(ValueError, match="fields changed"):
        subject.validate_fresh_quality_plan(plan)

    denied = dict(_AUTHORIZATION)
    denied["quality_pilot_authorized"] = False
    monkeypatch.setattr(
        subject, "_load_performance_authorization", lambda _path: denied
    )
    with pytest.raises(PermissionError, match="has not authorized"):
        subject.build_fresh_quality_plan(performance_receipt_path=Path("unused.json"))


def test_root_is_actor_observation_only_and_fail_closed_on_hidden_or_unknown(
    monkeypatch: pytest.MonkeyPatch,
):
    plan = _plan(monkeypatch)
    row = subject.schedule_row(subject.PRIMARY_PHASE, 0)
    root = subject._root_value(
        plan=plan,
        row=row,
        observations=_observations(0, subject.PRIMARY_PHASE),
    )
    assert subject.validate_root(root, plan=plan) == root
    assert root["opponent_private_discards_used"] is False
    assert all(
        "opponent_private_discards" not in record["observation"]
        for record in root["observations"]
    )

    hidden = json.loads(json.dumps(root))
    hidden["observations"][0]["observation"]["opponent_private_discards"] = []
    with pytest.raises(ValueError):
        subject.validate_root(hidden, plan=plan)

    unknown = dict(root)
    unknown["realized_deck_tail"] = []
    with pytest.raises(ValueError, match="fields changed"):
        subject.validate_root(unknown, plan=plan)


def test_materialize_seal_package_are_create_only_and_exact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    plan = _plan(monkeypatch)
    plan_path = tmp_path / "plan.json"
    subject._write_once(plan_path, plan)
    monkeypatch.setattr(
        subject.step6d_v1, "load_model_bundle", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        subject.step6d_v1, "_absolute_model_paths", lambda _root: object()
    )
    monkeypatch.setattr(
        subject,
        "_generate_observations",
        lambda *, repository_root, row, bundle: _observations(
            row["pair_index"], row["phase"]
        ),
    )
    root_dir = tmp_path / "materialized"
    receipt_path = tmp_path / "materialization.json"
    seal_path = tmp_path / "seal.json"
    receipt, seal = subject.materialize_fresh_quality_roots(
        repository_root=tmp_path,
        plan_path=plan_path,
        performance_receipt_path=tmp_path / "performance.json",
        output_directory=root_dir,
        materialization_output=receipt_path,
        seal_output=seal_path,
    )
    assert receipt["paired_hand_count"] == 55
    assert receipt["root_count"] == 110
    assert receipt["observation_count"] == 110
    assert len(receipt["root_records"]) == 55
    assert seal["root_count"] == 110
    jobs = subject.build_job_descriptors(plan=plan, seal=seal, root_directory=root_dir)
    assert len(jobs) == 15
    assert sum(job["phase"] == subject.PRIMARY_PHASE for job in jobs) == 10
    assert sum(job["phase"] == subject.CONFIRMATION_PHASE for job in jobs) == 5

    package = subject.create_fresh_quality_package(
        plan_path=plan_path,
        materialization_path=receipt_path,
        seal_path=seal_path,
        performance_receipt_path=tmp_path / "performance.json",
        output_directory=tmp_path / "package",
        archive_path=tmp_path / "package.zip",
    )
    assert package["manifest"]["job_count"] == 15
    assert package["manifest"]["root_file_count"] == 55
    assert package["ready"]["cloud_execution_started"] is False
    assert Path(package["archive_path"]).is_file()
    assert package["validated"]["archive"]["sha256"] == package["archive_sha256"]

    with pytest.raises(FileExistsError, match="create-only"):
        subject.materialize_fresh_quality_roots(
            repository_root=tmp_path,
            plan_path=plan_path,
            performance_receipt_path=tmp_path / "performance.json",
            output_directory=root_dir,
            materialization_output=tmp_path / "other.json",
            seal_output=tmp_path / "other-seal.json",
        )

    packaged_root = tmp_path / "package" / "roots" / "primary" / "pair_000.json"
    packaged_root.write_bytes(packaged_root.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash/size"):
        subject.validate_fresh_quality_package(
            package_directory=tmp_path / "package",
            performance_receipt_path=tmp_path / "performance.json",
            archive_path=tmp_path / "package.zip",
        )


def test_materialization_scopes_the_exact_feature_encoder_binding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    encoder = (tmp_path / "accepted-feature-encoder.so").resolve()
    encoder.write_bytes(b"accepted-feature-encoder")
    encoder_sha = hashlib.sha256(encoder.read_bytes()).hexdigest()
    monkeypatch.setattr(subject, "ACCEPTED_FEATURE_ENCODER_SHA256", encoder_sha)
    plan = _plan(monkeypatch)
    plan_path = tmp_path / "plan.json"
    subject._write_once(plan_path, plan)
    monkeypatch.setattr(
        subject.step6d_v1, "load_model_bundle", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        subject.step6d_v1, "_absolute_model_paths", lambda _root: object()
    )
    original = subject.stage3_feature_rust._library_path()
    observed: list[Path] = []

    def generate(*, repository_root, row, bundle):
        del repository_root, bundle
        observed.append(subject.stage3_feature_rust._library_path())
        return _observations(row["pair_index"], row["phase"])

    monkeypatch.setattr(subject, "_generate_observations", generate)
    materialization, _seal = subject.materialize_fresh_quality_roots(
        repository_root=tmp_path,
        plan_path=plan_path,
        performance_receipt_path=tmp_path / "performance.json",
        output_directory=tmp_path / "materialized",
        materialization_output=tmp_path / "materialization.json",
        seal_output=tmp_path / "seal.json",
        feature_encoder_path=encoder,
    )

    assert materialization["root_count"] == 110
    assert observed == [encoder] * 55
    assert subject.stage3_feature_rust._library_path() == original


def test_materialization_replay_detects_root_tamper(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    plan = _plan(monkeypatch)
    root_dir = tmp_path / "materialized"
    for row in subject.schedule_rows():
        root = subject._root_value(
            plan=plan,
            row=row,
            observations=_observations(row["pair_index"], row["phase"]),
        )
        subject._write_once(
            root_dir / subject._root_relative_path(row["phase"], row["pair_index"]),
            root,
        )
    receipt = subject.build_materialization_receipt(plan=plan, root_directory=root_dir)
    target = root_dir / "roots" / "primary" / "pair_000.json"
    changed = json.loads(target.read_text(encoding="ascii"))
    changed["unknown"] = True
    target.write_bytes(subject.canonical_bytes(changed))
    with pytest.raises(ValueError):
        subject.validate_materialization_receipt(receipt, plan=plan, replay_roots=True)
