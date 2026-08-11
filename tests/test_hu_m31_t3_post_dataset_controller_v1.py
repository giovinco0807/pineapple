from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ofc_regular import hu_m31_t3_locked_promotion_execution_v1 as execution
from ofc_regular import hu_m31_t3_post_dataset_controller_v1 as subject
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


def _plan(tmp_path: Path) -> dict[str, Any]:
    model_id = "street-policy-net-v1-post-dataset-test"
    model = tmp_path / "manifest.json"
    model.write_bytes(b"model")
    model_sha = promotion.sha256_file(model)
    lock = tmp_path / "threshold.json"
    lock.write_bytes(
        promotion.canonical_bytes(_threshold_lock(model_id, model_sha))
    )
    registry = tmp_path / "ai_profiles.py"
    registry.write_bytes(b"registry")
    closure = tmp_path / "closure.tar"
    closure.write_bytes(b"closure")
    return promotion.build_locked_promotion_plan(
        plan_id="m31-t3-post-dataset-test",
        model_artifact_id=model_id,
        model_path=model,
        expected_model_sha256=model_sha,
        threshold_lock_path=lock,
        expected_threshold_lock_sha256=promotion.sha256_file(lock),
        policy_registry_path=registry,
        expected_policy_registry_sha256=promotion.sha256_file(registry),
        evaluation_runtime_closure_path=closure,
        expected_evaluation_runtime_closure_sha256=(
            promotion.sha256_file(closure)
        ),
    )


def _row(
    plan: dict[str, Any],
    *,
    schedule: str,
    entity_id: str,
    seed_index: int,
    seat: str,
) -> dict[str, Any]:
    seeds = promotion.seed_values(schedule, seed_index)
    action = ActionKey().to_token()
    binding = plan["artifact_binding"]
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
        "model_sha256": binding["model"]["sha256"],
        "threshold_lock_sha256": binding["threshold_lock"]["sha256"],
        "policy_registry_sha256": binding["policy_registry"]["sha256"],
        "evaluation_runtime_closure_sha256": binding[
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


class _Runner:
    def __init__(self, plan: dict[str, Any]) -> None:
        self.plan = plan
        self.calls: list[str] = []

    def run_shard(
        self,
        *,
        schedule: str,
        entity_id: str,
        seed_indices,
        shard_id: str,
        output_path,
    ) -> dict[str, Any]:
        assert output_path is None
        self.calls.append(shard_id)
        rows = [
            _row(
                self.plan,
                schedule=schedule,
                entity_id=entity_id,
                seed_index=index,
                seat=seat,
            )
            for index in seed_indices
            for seat in promotion.SEATS
        ]
        rows.sort(key=promotion._row_key)
        return promotion.build_evaluation_shard(
            plan=self.plan, shard_id=shard_id, rows=rows
        )


def test_bounded_batch_is_atomic_and_resumes_in_frozen_order(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path)
    frozen = execution.build_execution_plan(promotion_plan=plan)
    shards = tmp_path / "shards"
    empty = subject.inspect_evaluation_progress(
        promotion_plan=plan,
        execution_plan=frozen,
        shard_directory=shards,
    )
    assert empty["complete_work_item_count"] == 0
    assert empty["pending_work_item_count"] == 260
    runner = _Runner(plan)
    first = subject.run_pending_work_items(
        runner=runner,
        promotion_plan=plan,
        execution_plan=frozen,
        shard_directory=shards,
        max_items=2,
    )
    assert first["status"] == "resume_required"
    assert first["complete_work_item_count"] == 2
    assert first["pending_work_item_count"] == 258
    assert first["selected_work_ids"] == [
        frozen["work_items"][0]["work_id"],
        frozen["work_items"][1]["work_id"],
    ]
    second = subject.run_pending_work_items(
        runner=runner,
        promotion_plan=plan,
        execution_plan=frozen,
        shard_directory=shards,
        max_items=1,
    )
    assert second["selected_work_ids"] == [
        frozen["work_items"][2]["work_id"]
    ]
    assert second["complete_work_item_count"] == 3
    assert runner.calls == [
        item["work_id"] for item in frozen["work_items"][:3]
    ]
    assert len(list(shards.glob("*.json"))) == 3
    assert not list(shards.glob("*.tmp"))


def test_progress_rejects_extra_or_tampered_shards(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    frozen = execution.build_execution_plan(promotion_plan=plan)
    shards = tmp_path / "shards"
    runner = _Runner(plan)
    subject.run_pending_work_items(
        runner=runner,
        promotion_plan=plan,
        execution_plan=frozen,
        shard_directory=shards,
        max_items=1,
    )
    (shards / "extra.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="extra files"):
        subject.inspect_evaluation_progress(
            promotion_plan=plan,
            execution_plan=frozen,
            shard_directory=shards,
        )
    (shards / "extra.json").unlink()
    only = next(shards.glob("*.json"))
    only.write_bytes(b"{}")
    with pytest.raises(ValueError, match="fields changed|canonical"):
        subject.inspect_evaluation_progress(
            promotion_plan=plan,
            execution_plan=frozen,
            shard_directory=shards,
        )


def test_registration_authorization_is_dormant_and_pass_gated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _plan(tmp_path)
    frozen = execution.build_execution_plan(promotion_plan=plan)
    progress = {
        "status": "complete",
        "complete_work_item_count": 260,
        "accepted_row_count": 13_000,
    }
    merge = {"kind": "merge"}
    gate = {
        "status": "pass",
        "all_gates_passed": True,
        "scientific_promotion_passed": True,
        "separate_opt_in_profile_candidate_authorized": True,
        "gates": {"all": True},
        "named_profile_added": False,
        "current_profile_changed": False,
        "runtime_activated": False,
        "full_replacement_enabled": False,
    }
    monkeypatch.setattr(
        subject,
        "inspect_evaluation_progress",
        lambda **kwargs: dict(progress),
    )
    monkeypatch.setattr(
        subject.execution,
        "build_closeout",
        lambda **kwargs: (dict(merge), dict(gate)),
    )
    monkeypatch.setattr(
        subject.promotion,
        "validate_locked_promotion_gate",
        lambda *args, **kwargs: dict(gate),
    )
    result = subject.build_opt_in_registration_authorization(
        promotion_plan=plan,
        execution_plan=frozen,
        shard_directory=tmp_path / "shards",
        merge=merge,
        gate=gate,
    )
    assert result["registration_authorized"] is True
    assert result["registration_applied"] is False
    assert result["named_profile_added"] is False
    assert result["current_profile_changed"] is False
    assert result["runtime_activated"] is False

    failed = dict(gate)
    failed["status"] = "no_go"
    failed["all_gates_passed"] = False
    failed["scientific_promotion_passed"] = False
    monkeypatch.setattr(
        subject.execution,
        "build_closeout",
        lambda **kwargs: (dict(merge), dict(failed)),
    )
    monkeypatch.setattr(
        subject.promotion,
        "validate_locked_promotion_gate",
        lambda *args, **kwargs: dict(failed),
    )
    with pytest.raises(PermissionError, match="did not authorize"):
        subject.build_opt_in_registration_authorization(
            promotion_plan=plan,
            execution_plan=frozen,
            shard_directory=tmp_path / "shards",
            merge=merge,
            gate=failed,
        )
