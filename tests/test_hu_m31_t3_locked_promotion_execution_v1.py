from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from ofc_regular import hu_m31_t3_locked_promotion_execution_v1 as subject
from ofc_regular import hu_m31_t3_step6d_locked_promotion_v1 as promotion
from ofc_regular.action_key import ACTION_KEY_SCHEMA, ActionKey


def _threshold_lock(model_id: str, model_sha256: str) -> dict[str, Any]:
    return {
        "schema": promotion.THRESHOLD_LOCK_SCHEMA,
        "status": "locked_no_holdout_reselection",
        "model_artifact_id": model_id,
        "model_sha256": model_sha256,
        "state_action_input_schema_sha256": "b" * 64,
        "action_key_schema": ACTION_KEY_SCHEMA,
        "seat_thresholds": {"first": 0.80, "second": 0.82},
        "seat_enabled": {"first": True, "second": True},
        "source_training_threshold_lock_sha256": "c" * 64,
        "source_checkpoint_bundle_identity_sha256": "d" * 64,
        "locked_on_threshold_holdout_only": True,
        "model_change_requires_new_lock": True,
        "teacher_ev_lcb_is_runtime_gate": False,
        "top1_accuracy_is_promotion_gate": False,
        "current_profile_changed": False,
        "runtime_activated": False,
    }


def _make_plan(tmp_path: Path) -> dict[str, Any]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    model_id = "street-policy-net-v1-execution-test"
    model = tmp_path / "manifest.json"
    model.write_bytes(b"synthetic immutable model manifest")
    model_sha = promotion.sha256_file(model)
    lock = tmp_path / "threshold.json"
    lock.write_bytes(
        promotion.canonical_bytes(_threshold_lock(model_id, model_sha))
    )
    registry = tmp_path / "ai_profiles.py"
    registry.write_bytes(b"synthetic immutable explicit policy registry")
    runtime = tmp_path / "runtime-closure.tar"
    runtime.write_bytes(b"synthetic immutable evaluation runtime closure")
    return promotion.build_locked_promotion_plan(
        plan_id="m31-t3-locked-execution-test",
        model_artifact_id=model_id,
        model_path=model,
        expected_model_sha256=model_sha,
        threshold_lock_path=lock,
        expected_threshold_lock_sha256=promotion.sha256_file(lock),
        policy_registry_path=registry,
        expected_policy_registry_sha256=promotion.sha256_file(registry),
        evaluation_runtime_closure_path=runtime,
        expected_evaluation_runtime_closure_sha256=(
            promotion.sha256_file(runtime)
        ),
    )


def _row(
    *,
    plan: dict[str, Any],
    schedule: str,
    entity_id: str,
    seed_index: int,
    seat: str,
) -> dict[str, Any]:
    seeds = promotion.seed_values(schedule, seed_index)
    action = ActionKey().to_token()
    return {
        "schema": promotion.ROW_SCHEMA,
        "schedule": schedule,
        "entity_id": entity_id,
        "seed_index": seed_index,
        "hand_seed": seeds["hand"],
        "actor_policy_seed": seeds["actor_policy"],
        "opponent_policy_seed": seeds["opponent_policy"],
        "evaluation_seed": seeds["evaluation"],
        "child_seed": seeds["child"],
        "confirmation_seed": seeds["confirmation"],
        "seat": seat,
        "action_key_schema": ACTION_KEY_SCHEMA,
        "candidate_action_key": action,
        "baseline_action_key": action,
        "legal_action_mapping_sha256": "e" * 64,
        "candidate_trajectory_sha256": "f" * 64,
        "baseline_trajectory_sha256": "f" * 64,
        "score_perspective": "candidate_hero_hu_score",
        "candidate_score": 0.0,
        "baseline_score": 0.0,
        "delta": 0.0,
        "override_log_valid": True,
        "override_fired": False,
        "nonfire_action_key_identical": True,
        "nonfire_trajectory_identical": True,
        "nonfire_cancellation_valid": True,
        "model_sha256": plan["artifact_binding"]["model"]["sha256"],
        "threshold_lock_sha256": plan["artifact_binding"][
            "threshold_lock"
        ]["sha256"],
        "policy_registry_sha256": plan["artifact_binding"][
            "policy_registry"
        ]["sha256"],
        "evaluation_runtime_closure_sha256": plan["artifact_binding"][
            "evaluation_runtime_closure"
        ]["sha256"],
        "counterfactual_basis": (
            "same_hand_role_policy_seeds_physical_seat_v1"
        ),
        "teacher_values_used": False,
        "opponent_private_discards_used": False,
        "current_profile_resolved": False,
        "current_profile_changed": False,
    }


class _FakeRunner:
    def __init__(self, plan: dict[str, Any], *, offset: int = 0) -> None:
        self.plan = plan
        self.offset = offset
        self.calls: list[dict[str, Any]] = []

    def run_shard(
        self,
        *,
        schedule: str,
        entity_id: str,
        seed_indices,
        shard_id: str,
        output_path: Path,
    ) -> dict[str, Any]:
        indices = tuple(seed_indices)
        self.calls.append(
            {
                "schedule": schedule,
                "entity_id": entity_id,
                "seed_indices": indices,
                "shard_id": shard_id,
                "output_path": Path(output_path),
            }
        )
        rows = [
            _row(
                plan=self.plan,
                schedule=schedule,
                entity_id=entity_id,
                seed_index=index + self.offset,
                seat=seat,
            )
            for index in indices
            for seat in promotion.SEATS
        ]
        rows.sort(key=promotion._row_key)
        return promotion.write_evaluation_shard(
            plan=self.plan,
            shard_id=shard_id,
            rows=rows,
            output_path=output_path,
        )


def test_execution_plan_freezes_exact_full_grid(tmp_path: Path) -> None:
    plan = _make_plan(tmp_path)
    value = subject.build_execution_plan(promotion_plan=plan)

    assert value["work_item_count"] == 260
    assert value["population_work_item_count"] == 200
    assert value["abr_work_item_count"] == 60
    assert value["coverage"] == {
        "population_opponents": 5,
        "population_paired_seeds_per_opponent": 1_000,
        "abr_families": 3,
        "abr_paired_seeds_per_response": 500,
        "paired_seed_count": 6_500,
        "row_count": 13_000,
        "first_rows": 6_500,
        "second_rows": 6_500,
        "coverage_sha256": value["coverage"]["coverage_sha256"],
    }
    assert value["work_items"][0]["work_id"] == (
        "population-stage19-p0-0000-0024"
    )
    assert value["work_items"][199]["work_id"] == (
        "population-random-exact-final-0975-0999"
    )
    assert value["work_items"][200]["work_id"] == (
        "abr-greedy-search-response-0000-0024"
    )
    assert value["work_items"][-1]["work_id"] == (
        "abr-royalty-denial-response-0475-0499"
    )
    assert len(
        {item["work_id"] for item in value["work_items"]}
    ) == 260
    assert len(
        {item["coverage_sha256"] for item in value["work_items"]}
    ) == 260
    assert value["cloud_execution_started"] is False
    assert value["promotion_decision_applied"] is False
    assert value["named_profile_added"] is False
    assert value["current_profile_changed"] is False
    assert value["runtime_activated"] is False
    assert value["full_replacement_enabled"] is False
    assert subject.validate_execution_plan(
        value, promotion_plan=plan
    ) == value


def test_execution_plan_is_canonical_write_once_and_cli_readable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plan = _make_plan(tmp_path)
    plan_path = tmp_path / "promotion-plan.json"
    plan_path.write_bytes(promotion.canonical_bytes(plan))
    output = tmp_path / "execution-plan.json"

    assert subject.main(
        [
            "prepare",
            "--promotion-plan",
            str(plan_path),
            "--output",
            str(output),
        ]
    ) == 0
    first_stdout = capsys.readouterr().out
    stored = subject._read_canonical(output, "execution plan")
    assert stored == subject.build_execution_plan(promotion_plan=plan)
    assert first_stdout.strip() == promotion.canonical_bytes(
        stored
    ).decode("ascii")

    assert subject.main(
        [
            "prepare",
            "--promotion-plan",
            str(plan_path),
            "--output",
            str(output),
        ]
    ) == 0
    assert capsys.readouterr().out == first_stdout
    output.write_bytes(output.read_bytes() + b"\n")
    assert subject.main(
        [
            "prepare",
            "--promotion-plan",
            str(plan_path),
            "--output",
            str(output),
        ]
    ) == 2
    assert "artifact changed" in capsys.readouterr().err


def test_execution_plan_tampering_fails_source_replay(
    tmp_path: Path,
) -> None:
    plan = _make_plan(tmp_path)
    execution = subject.build_execution_plan(promotion_plan=plan)

    missing = deepcopy(execution)
    missing["work_items"].pop()
    with pytest.raises(ValueError, match="source replay"):
        subject.validate_execution_plan(missing, promotion_plan=plan)

    reseeded = deepcopy(execution)
    reseeded["work_items"][0]["seed_index_start"] = 1
    with pytest.raises(ValueError, match="source replay"):
        subject.validate_execution_plan(reseeded, promotion_plan=plan)

    current = deepcopy(execution)
    current["work_items"][0]["entity_id"] = "current"
    with pytest.raises(ValueError, match="source replay"):
        subject.validate_execution_plan(current, promotion_plan=plan)

    activated = deepcopy(execution)
    activated["runtime_activated"] = True
    with pytest.raises(ValueError, match="source replay"):
        subject.validate_execution_plan(activated, promotion_plan=plan)


def test_run_work_item_dispatches_exact_range_and_both_seats(
    tmp_path: Path,
) -> None:
    plan = _make_plan(tmp_path)
    execution = subject.build_execution_plan(promotion_plan=plan)
    runner = _FakeRunner(plan)
    work_id = "population-stage9f-p2-0050-0074"

    shard = subject.run_work_item(
        runner=runner,
        promotion_plan=plan,
        execution_plan=execution,
        work_id=work_id,
        shard_directory=tmp_path / "shards",
    )

    assert len(runner.calls) == 1
    call = runner.calls[0]
    assert call["schedule"] == promotion.LOCKED_POPULATION
    assert call["entity_id"] == "stage9f_p2"
    assert call["seed_indices"] == tuple(range(50, 75))
    assert call["shard_id"] == work_id
    assert call["output_path"].name == f"{work_id}.json"
    assert shard["row_count"] == 50
    assert {
        (row["seed_index"], row["seat"]) for row in shard["rows"]
    } == {
        (index, seat)
        for index in range(50, 75)
        for seat in promotion.SEATS
    }

    same = subject.run_work_item(
        runner=runner,
        promotion_plan=plan,
        execution_plan=execution,
        work_id=work_id,
        shard_directory=tmp_path / "shards",
    )
    assert same == shard


def test_run_work_item_rejects_wrong_coverage_or_plan(
    tmp_path: Path,
) -> None:
    plan = _make_plan(tmp_path)
    execution = subject.build_execution_plan(promotion_plan=plan)
    with pytest.raises(ValueError, match="frozen work item"):
        subject.run_work_item(
            runner=_FakeRunner(plan, offset=25),
            promotion_plan=plan,
            execution_plan=execution,
            work_id="abr-greedy-search-response-0000-0024",
            shard_directory=tmp_path / "wrong-coverage",
        )

    other_plan = _make_plan(tmp_path / "other")
    other_plan["plan_id"] = "m31-t3-other-locked-execution-test"
    with pytest.raises(ValueError, match="bound to this"):
        subject.run_work_item(
            runner=_FakeRunner(other_plan),
            promotion_plan=plan,
            execution_plan=execution,
            work_id="abr-greedy-search-response-0000-0024",
            shard_directory=tmp_path / "wrong-plan",
        )


def test_exact_shard_collection_rejects_missing_and_extra_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _make_plan(tmp_path)
    execution = subject.build_execution_plan(promotion_plan=plan)
    directory = tmp_path / "shards"
    directory.mkdir()
    for item in execution["work_items"]:
        (directory / item["output_filename"]).write_bytes(b"{}")

    monkeypatch.setattr(
        subject,
        "validate_work_item_shard",
        lambda **kwargs: kwargs["shard"],
    )
    paths = subject.collect_exact_shard_paths(
        promotion_plan=plan,
        execution_plan=execution,
        shard_directory=directory,
    )
    assert [path.name for path in paths] == [
        item["output_filename"] for item in execution["work_items"]
    ]

    paths[0].unlink()
    with pytest.raises(ValueError, match="missing or extra"):
        subject.collect_exact_shard_paths(
            promotion_plan=plan,
            execution_plan=execution,
            shard_directory=directory,
        )
    paths[0].write_bytes(b"{}")
    (directory / "unexpected.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="missing or extra"):
        subject.collect_exact_shard_paths(
            promotion_plan=plan,
            execution_plan=execution,
            shard_directory=directory,
        )
