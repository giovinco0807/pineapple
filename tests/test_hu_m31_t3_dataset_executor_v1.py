from __future__ import annotations

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import pytest

from ofc_regular.action_key import (
    ACTION_KEY_SCHEMA,
    ActionKey,
    action_key,
    canonicalize_actions,
    legal_action_set_digest,
    ordered_action_mapping_digest,
)
from ofc_regular.action_space import generate_turn_actions
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular import hu_m31_t3_dataset_contract_v1 as dataset
from ofc_regular import hu_m31_t3_dataset_executor_v1 as subject
from ofc_regular import hu_m31_t3_step6d_fresh_quality_gate_v1 as quality_gate
from ofc_regular import run_hu_m31_t3_step6c_shard as step6c


def _load_existing_certificate_builder() -> Any:
    """Reuse the independently tested exact Step6c certificate fixture."""

    path = Path(__file__).with_name("test_hu_m31_t3_step6c_runner.py")
    name = "_m31_dataset_step6c_certificate_fixture"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Step6c certificate fixture cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_STEP6C_FIXTURE = _load_existing_certificate_builder()


def _canonical_gate(path: Path, *, passed: bool = True) -> dict[str, Any]:
    value = {
        "schema": quality_gate.GATE_SCHEMA,
        "status": "pass" if passed else "no_go",
        "decision": (
            "fresh_quality_pass_open_25_paired_data_shard_only"
            if passed
            else "fresh_quality_no_go_no_same_seed_threshold_reselection"
        ),
        "merge": {"synthetic": True},
        "merge_sha256": "a" * 64,
        "thresholds": {},
        "gates": {},
        "all_gates_passed": passed,
        "quality_pilot_passed": passed,
        "data_pilot_25_paired_authorized": passed,
        "full_9000_paired_fanout_authorized": False,
        "training_eligible": False,
        "promotion_evidence": False,
        "current_profile_changed": False,
        "teacher_values_are_realized_match_ev": False,
    }
    path.write_bytes(subject.canonical_bytes(value))
    return value


def _patch_gate_replay(
    monkeypatch: pytest.MonkeyPatch, gate: Mapping[str, Any]
) -> list[bool]:
    replay_flags: list[bool] = []

    def validate(
        value: Mapping[str, Any], *, replay_sources: bool
    ) -> dict[str, Any]:
        replay_flags.append(replay_sources)
        assert dict(value) == dict(gate)
        return deepcopy(dict(gate))

    monkeypatch.setattr(
        subject.quality_gate, "validate_fresh_quality_gate_value", validate
    )
    return replay_flags


def _observation(index: int, seat: str) -> ActorObservation:
    return _STEP6C_FIXTURE._observation(20_000 + index, seat)


def _root_generator(
    pair: Mapping[str, Any], bundle: object | None
) -> tuple[ActorObservation, ActorObservation]:
    del bundle
    index = int(pair["global_pair_index"])
    return _observation(index, "first"), _observation(index, "second")


def _baseline(
    observation: ActorObservation,
    pair: Mapping[str, Any],
    bundle: object | None,
) -> str:
    del pair, bundle
    actions = canonicalize_actions(
        generate_turn_actions(observation.hero_board, observation.dealt_cards)
    )
    return action_key(actions[-1]).to_token()


def _fast_decision(
    observation: ActorObservation,
    pair: Mapping[str, Any],
    budget: Mapping[str, int],
    evaluation_seed_key: str,
    library_path: Path | None,
) -> dict[str, Any]:
    del library_path
    actions = canonicalize_actions(
        generate_turn_actions(observation.hero_board, observation.dealt_cards)
    )
    values = []
    for index, action in enumerate(actions):
        token = action_key(action).to_token()
        q = float(len(actions) - index)
        values.append(
            {
                "action_key": token,
                "selection_ev": q,
                "evaluation_ev": q - (0.125 if evaluation_seed_key == "confirmation" else 0.0),
            }
        )
    selected = values[0]["action_key"]
    seed = int(pair["seeds"][evaluation_seed_key])
    return {
        "action_key_schema": ACTION_KEY_SCHEMA,
        "legal_action_set_digest": legal_action_set_digest(actions),
        "legal_action_order_digest": ordered_action_mapping_digest(actions),
        "selected_action_key": selected,
        "action_values": values,
        "search_contract_digest": f"{seed:064x}"[-64:],
        "semantic_result_digest": f"{seed + 1:064x}"[-64:],
        "result_digest": f"{seed + 2:064x}"[-64:],
        "candidate_rng_digest": f"{int(pair['seeds']['candidate']) + 3:064x}"[-64:],
        "evaluation_rng_digest": f"{seed + 4:064x}"[-64:],
        "_budget": dict(budget),
        "_evaluation_seed_key": evaluation_seed_key,
    }


def _patch_fast_certificate_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[str, str, int]]:
    calls: list[tuple[str, str, int]] = []

    def validate(
        decision: Mapping[str, Any],
        *,
        observation: ActorObservation,
        seeds: Mapping[str, int],
        budget: Mapping[str, int],
        evaluation_seed_key: str,
        native_library_sha256: str,
    ) -> step6c.DecisionEvidence:
        assert native_library_sha256 == dataset.ACCEPTED_CANDIDATE_LIBRARY_SHA256
        assert decision["_budget"] == dict(budget)
        assert decision["_evaluation_seed_key"] == evaluation_seed_key
        actions = canonicalize_actions(
            generate_turn_actions(observation.hero_board, observation.dealt_cards)
        )
        tokens = [action_key(action).to_token() for action in actions]
        assert [row["action_key"] for row in decision["action_values"]] == tokens
        selection = {
            row["action_key"]: float(row["selection_ev"])
            for row in decision["action_values"]
        }
        evaluation = {
            row["action_key"]: float(row["evaluation_ev"])
            for row in decision["action_values"]
        }
        candidate_keys = frozenset(
            f"candidate:{seeds['candidate']}:{index}"
            for index in range(int(budget["candidate_samples"]))
        )
        evaluation_keys = frozenset(
            f"{evaluation_seed_key}:{seeds[evaluation_seed_key]}:{index}"
            for index in range(int(budget["evaluation_samples"]))
        )
        calls.append(
            (
                observation.seat,
                evaluation_seed_key,
                int(budget["evaluation_samples"]),
            )
        )
        return step6c.DecisionEvidence(
            candidate_keys=candidate_keys,
            evaluation_keys=evaluation_keys,
            selection_by_key=selection,
            evaluation_by_key=evaluation,
        )

    monkeypatch.setattr(subject.step6c, "_validate_decision_payload", validate)
    return calls


def _authorization_for_unit(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": subject.AUTHORIZATION_SCHEMA,
        "status": "qualified_fresh_quality_bound_to_smoke_shard",
        "plan_sha256": subject.canonical_sha256(plan),
        "shard_id": dataset.SMOKE_SHARD_ID,
        "fresh_quality_gate_schema": quality_gate.GATE_SCHEMA,
        "fresh_quality_gate_sha256": "a" * 64,
        "fresh_quality_merge_sha256": "b" * 64,
        "fresh_quality_decision": (
            "fresh_quality_pass_open_25_paired_data_shard_only"
        ),
        "source_replayed": True,
        "data_pilot_25_paired_authorized": True,
        "full_9000_paired_fanout_authorized": False,
        "current_profile_registry_sha256": (
            dataset.CURRENT_PROFILE_REGISTRY_SHA256
        ),
        "current_profile_changed": False,
    }


def test_real_step6c_replay_maps_evaluation_q_delta_rank_and_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = dataset.build_dataset_plan()
    pair = dataset.pair_contract(plan, "train", 0)
    authorization = _authorization_for_unit(plan)
    observations = (_observation(0, "first"), _observation(0, "second"))

    # The accepted Candidate02 binary differs from the old Step6c candidate,
    # while the decision schema/certificate semantics are intentionally shared.
    monkeypatch.setattr(
        step6c,
        "EXPECTED_NATIVE_LIBRARY_SHA256",
        dataset.ACCEPTED_CANDIDATE_LIBRARY_SHA256,
    )

    def search(
        observation: ActorObservation,
        pair_value: Mapping[str, Any],
        budget: Mapping[str, int],
        evaluation_seed_key: str,
        library_path: Path | None,
    ) -> dict[str, Any]:
        del budget, library_path
        return _STEP6C_FIXTURE._decision(
            observation,
            seeds=dict(pair_value["seeds"]),
            confirmation=evaluation_seed_key == "confirmation",
            selected_regret=0.5,
        )

    candidate = subject._build_pair_evidence(
        plan=plan,
        authorization=authorization,
        pair=pair,
        observations=observations,
        search_adapter=search,
        baseline_adapter=_baseline,
        bundle=None,
        library_path=None,
    )
    evidence, result = subject.validate_pair_evidence(
        candidate,
        plan=plan,
        authorization=authorization,
        expected_local_pair_index=0,
    )
    assert result["confirmation_required"] is True
    assert [row["seat"] for row in result["rows"]] == ["first", "second"]
    for row in result["rows"]:
        teacher = row["teacher"]
        baseline = teacher["baseline_action_key"]
        target_by_key = {
            target["action_key"]: target for target in teacher["action_targets"]
        }
        assert target_by_key[baseline]["primary_delta"] == pytest.approx(0.0)
        assert sorted(
            target["primary_rank"] for target in teacher["action_targets"]
        ) == list(range(len(teacher["action_targets"])))
        assert sorted(
            target["confirmation_rank"] for target in teacher["action_targets"]
        ) == list(range(len(teacher["action_targets"])))

    tampered = deepcopy(evidence)
    tampered["rows"][0]["primary_decision"]["action_values"][0][
        "evaluation_ev"
    ] += 1.0
    with pytest.raises(ValueError, match="certificate|regret|selected-action"):
        subject.validate_pair_evidence(
            tampered,
            plan=plan,
            authorization=authorization,
            expected_local_pair_index=0,
        )


def test_unqualified_gate_stops_before_smoke_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = dataset.build_dataset_plan()
    gate_path = tmp_path / "quality_no_go.json"
    gate = _canonical_gate(gate_path, passed=False)
    flags = _patch_gate_replay(monkeypatch, gate)
    root_called = False

    def forbidden_root(*args: Any, **kwargs: Any) -> Any:
        nonlocal root_called
        root_called = True
        raise AssertionError("root generation must not start")

    shard = tmp_path / "shard"
    with pytest.raises(PermissionError, match="does not authorize"):
        subject.run_smoke_shard(
            plan=plan,
            shard_directory=shard,
            fresh_quality_gate_path=gate_path,
            root_generator=forbidden_root,
            search_adapter=_fast_decision,
            baseline_adapter=_baseline,
        )
    assert flags == [True]
    assert root_called is False
    assert not (shard / subject.AUTHORIZATION_NAME).exists()
    assert not (shard / "pairs").exists()


def test_25_pair_resume_done_last_pure_merge_and_immutable_parquet_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = dataset.build_dataset_plan()
    gate_path = tmp_path / "quality_pass.json"
    gate = _canonical_gate(gate_path, passed=True)
    replay_flags = _patch_gate_replay(monkeypatch, gate)
    certificate_calls = _patch_fast_certificate_replay(monkeypatch)
    search_calls: list[tuple[int, str, str]] = []
    shard = tmp_path / "shard"

    def search(
        observation: ActorObservation,
        pair: Mapping[str, Any],
        budget: Mapping[str, int],
        evaluation_seed_key: str,
        library_path: Path | None,
    ) -> dict[str, Any]:
        assert (shard / subject.AUTHORIZATION_NAME).is_file()
        search_calls.append(
            (
                int(pair["local_pair_index"]),
                observation.seat,
                evaluation_seed_key,
            )
        )
        return _fast_decision(
            observation, pair, budget, evaluation_seed_key, library_path
        )

    first = subject.run_smoke_shard(
        plan=plan,
        shard_directory=shard,
        fresh_quality_gate_path=gate_path,
        root_generator=_root_generator,
        search_adapter=search,
        baseline_adapter=_baseline,
        max_new_pairs=7,
    )
    assert first["status"] == "partial_safe_to_resume"
    assert first["resume"]["completed_pair_count"] == 7
    assert first["resume"]["pending_pair_count"] == 18
    assert not (shard / subject.DONE_NAME).exists()

    second = subject.run_smoke_shard(
        plan=plan,
        shard_directory=shard,
        fresh_quality_gate_path=gate_path,
        root_generator=_root_generator,
        search_adapter=search,
        baseline_adapter=_baseline,
    )
    assert second["status"] == "complete_done_published_last"
    assert second["done"]["pair_count"] == 25
    assert second["done"]["root_count"] == 50
    assert second["done"]["confirmation_pair_count"] == 3
    assert len(search_calls) == 56  # 50 primary + 6 locked confirmations.
    assert sum(call[2] == "confirmation" for call in search_calls) == 6
    assert all(replay_flags)

    done_path = shard / subject.DONE_NAME
    other_files = [
        path
        for path in shard.rglob("*")
        if path.is_file() and path != done_path
    ]
    assert done_path.stat().st_mtime_ns >= max(
        path.stat().st_mtime_ns for path in other_files
    )
    before = {
        path.relative_to(shard).as_posix(): subject._file_sha256(path)
        for path in shard.rglob("*")
        if path.is_file()
    }
    search_count = len(search_calls)
    third = subject.run_smoke_shard(
        plan=plan,
        shard_directory=shard,
        fresh_quality_gate_path=gate_path,
        root_generator=_root_generator,
        search_adapter=search,
        baseline_adapter=_baseline,
    )
    assert third["status"] == "already_complete"
    assert len(search_calls) == search_count
    assert before == {
        path.relative_to(shard).as_posix(): subject._file_sha256(path)
        for path in shard.rglob("*")
        if path.is_file()
    }

    merge_path = tmp_path / "merge" / subject.SMOKE_MERGE_NAME
    merge = subject.write_smoke_merge(
        plan=plan,
        shard_directory=shard,
        fresh_quality_gate_path=gate_path,
        output_path=merge_path,
    )
    assert merge["paired_hand_count"] == 25
    assert merge["root_count"] == 50
    assert merge["confirmation_pair_count"] == 3
    assert merge["action_row_count"] > 50
    assert merge["rng_overlap_count"] == 0
    assert merge["full_9000_paired_fanout_authorized"] is False

    # Exercise the immutable export lifecycle without making pyarrow a unit
    # test dependency.  A separate optional test below exercises the real
    # Apache Parquet codec when it is installed.
    stored_rows: dict[str, list[dict[str, Any]]] = {}

    def fake_write(path: Path, rows: Any) -> None:
        if path.exists():
            raise FileExistsError(path)
        normalized = [dict(row) for row in rows]
        stored_rows[str(path.resolve())] = normalized
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"PAR1" + subject.canonical_bytes(normalized) + b"PAR1")

    def fake_read(path: Path) -> list[dict[str, Any]]:
        return deepcopy(stored_rows[str(path.resolve())])

    monkeypatch.setattr(subject, "_write_parquet_once", fake_write)
    monkeypatch.setattr(subject, "_read_parquet_rows", fake_read)
    export_dir = tmp_path / "parquet"
    manifest = subject.write_parquet_export(
        plan=plan,
        shard_directory=shard,
        fresh_quality_gate_path=gate_path,
        smoke_merge_path=merge_path,
        output_directory=export_dir,
    )
    assert manifest["row_count"] == merge["action_row_count"]
    assert manifest["parquet_file"] == subject.PARQUET_FILE_NAME
    assert manifest["immutable"] is True
    assert manifest["training_eligible"] is False
    manifest_bytes = (export_dir / subject.PARQUET_MANIFEST_NAME).read_bytes()
    parquet_bytes = (export_dir / subject.PARQUET_FILE_NAME).read_bytes()
    assert subject.validate_parquet_export(
        plan=plan,
        shard_directory=shard,
        fresh_quality_gate_path=gate_path,
        smoke_merge_path=merge_path,
        output_directory=export_dir,
    ) == manifest
    assert (export_dir / subject.PARQUET_MANIFEST_NAME).read_bytes() == manifest_bytes
    assert (export_dir / subject.PARQUET_FILE_NAME).read_bytes() == parquet_bytes

    # Resume/merge replays retained full certificates, rather than trusting
    # the reduced pair JSON alone.
    evidence_path = subject.evidence_artifact_path(shard, 7)
    tampered = json.loads(evidence_path.read_text(encoding="ascii"))
    tampered["rows"][0]["primary_decision"]["action_values"][0][
        "evaluation_ev"
    ] += 100.0
    evidence_path.write_bytes(subject.canonical_bytes(tampered))
    with pytest.raises(ValueError):
        subject.build_smoke_merge(
            plan=plan,
            shard_directory=shard,
            fresh_quality_gate_path=gate_path,
        )
    assert certificate_calls


def test_smoke_gate_opens_replayable_non_smoke_shard_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = dataset.build_dataset_plan()
    gate_path = tmp_path / "quality_pass.json"
    gate = _canonical_gate(gate_path, passed=True)
    replay_flags = _patch_gate_replay(monkeypatch, gate)
    _patch_fast_certificate_replay(monkeypatch)
    smoke = tmp_path / "train-0000"
    subject.run_smoke_shard(
        plan=plan,
        shard_directory=smoke,
        fresh_quality_gate_path=gate_path,
        root_generator=_root_generator,
        search_adapter=_fast_decision,
        baseline_adapter=_baseline,
    )
    smoke_gate_path = tmp_path / "dataset_smoke_gate.json"
    smoke_gate = dataset.write_smoke_gate_receipt(
        plan=plan,
        smoke_shard_directory=smoke,
        output_path=smoke_gate_path,
    )
    assert smoke_gate["decision"] == "open_remaining_8975_paired_fanout"

    shard = tmp_path / "train-0001"
    first = subject.run_dataset_shard(
        plan=plan,
        shard_id="train-0001",
        shard_directory=shard,
        fresh_quality_gate_path=gate_path,
        smoke_gate_receipt_path=smoke_gate_path,
        smoke_shard_directory=smoke,
        root_generator=_root_generator,
        search_adapter=_fast_decision,
        baseline_adapter=_baseline,
        max_new_pairs=4,
    )
    assert first["status"] == "partial_safe_to_resume"
    assert first["resume"]["completed_pair_indices"] == [25, 26, 27, 28]
    authorization = json.loads(
        (shard / subject.AUTHORIZATION_NAME).read_text(encoding="ascii")
    )
    assert authorization["schema"] == subject.FULL_AUTHORIZATION_SCHEMA
    assert authorization["shard_id"] == "train-0001"
    assert authorization["full_9000_paired_fanout_authorized"] is True
    assert authorization["dataset_smoke_gate_sha256"] == subject._file_sha256(
        smoke_gate_path
    )

    completed = subject.run_dataset_shard(
        plan=plan,
        shard_id="train-0001",
        shard_directory=shard,
        fresh_quality_gate_path=gate_path,
        smoke_gate_receipt_path=smoke_gate_path,
        smoke_shard_directory=smoke,
        root_generator=_root_generator,
        search_adapter=_fast_decision,
        baseline_adapter=_baseline,
    )
    assert completed["status"] == "complete_done_published_last"
    assert completed["done"]["pair_count"] == 25
    assert completed["done"]["pair_records"][0]["local_pair_index"] == 25
    assert completed["done"]["pair_records"][-1]["local_pair_index"] == 49
    assert all(replay_flags)

    tampered_gate = tmp_path / "tampered_smoke_gate.json"
    changed = deepcopy(smoke_gate)
    changed["decision"] = "stop_before_full_fanout"
    tampered_gate.write_bytes(subject.canonical_bytes(changed))
    with pytest.raises(ValueError):
        subject.run_dataset_shard(
            plan=plan,
            shard_id="train-0002",
            shard_directory=tmp_path / "train-0002",
            fresh_quality_gate_path=gate_path,
            smoke_gate_receipt_path=tampered_gate,
            smoke_shard_directory=smoke,
            root_generator=_root_generator,
            search_adapter=_fast_decision,
            baseline_adapter=_baseline,
        )
    assert not (tmp_path / "train-0002" / "pairs").exists()


def test_real_pyarrow_parquet_export_when_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("pyarrow")
    plan = dataset.build_dataset_plan()
    gate_path = tmp_path / "quality_pass.json"
    gate = _canonical_gate(gate_path, passed=True)
    _patch_gate_replay(monkeypatch, gate)
    _patch_fast_certificate_replay(monkeypatch)
    shard = tmp_path / "shard"
    subject.run_smoke_shard(
        plan=plan,
        shard_directory=shard,
        fresh_quality_gate_path=gate_path,
        root_generator=_root_generator,
        search_adapter=_fast_decision,
        baseline_adapter=_baseline,
    )
    merge_path = tmp_path / "merge" / subject.SMOKE_MERGE_NAME
    merge = subject.write_smoke_merge(
        plan=plan,
        shard_directory=shard,
        fresh_quality_gate_path=gate_path,
        output_path=merge_path,
    )
    export_dir = tmp_path / "parquet"
    manifest = subject.write_parquet_export(
        plan=plan,
        shard_directory=shard,
        fresh_quality_gate_path=gate_path,
        smoke_merge_path=merge_path,
        output_directory=export_dir,
    )
    parquet = export_dir / subject.PARQUET_FILE_NAME
    assert parquet.read_bytes()[:4] == b"PAR1"
    assert parquet.read_bytes()[-4:] == b"PAR1"
    assert manifest["row_count"] == merge["action_row_count"]
    assert subject.validate_parquet_export(
        plan=plan,
        shard_directory=shard,
        fresh_quality_gate_path=gate_path,
        smoke_merge_path=merge_path,
        output_directory=export_dir,
    ) == manifest
