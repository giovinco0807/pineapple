from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

import ofc_regular.run_hu_m43_attempt07_preflight as preflight
from ofc_regular.hu_m43_attempt06_teacher import ATTEMPT06_FROZEN_MODEL_SHA256
from ofc_regular.hu_m43_attempt07_contract import (
    M43_ATTEMPT07_PLAN_SHA256,
    load_and_validate_attempt07_plan,
)
from ofc_regular.hu_m43_attempt07_teacher import (
    ATTEMPT07_SOLVER_ID,
    ATTEMPT07_TEACHER_SCHEMA,
)


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT_PLAN = ROOT / "configs" / "hu_joint_policy_m43_attempt07_preflight.json"
ATTEMPT07_PLAN = ROOT / "configs" / "hu_joint_policy_m43_attempt07.json"
SOURCE = ROOT / (
    "outputs/hu_joint_policy/m43_attempt06_search_quality/"
    "regular-hu-m43-attempt06-preflight-final-20260714-1154/"
    "merged/teacher.jsonl"
)


def _source_row(index: int = 0) -> dict:
    rows = [json.loads(line) for line in SOURCE.read_text().splitlines() if line]
    return rows[index]


def _fake_teacher(observation, baseline_action_key, config) -> dict:
    return {
        "status": "ok",
        "schema": ATTEMPT07_TEACHER_SCHEMA,
        "solver_id": ATTEMPT07_SOLVER_ID,
        "street": "T1",
        "seat": "second",
        "to_act_order": "second",
        "observation_fingerprint": observation.fingerprint(),
        "policy_observation": observation.to_dict(),
        "baseline_action_key": baseline_action_key,
        "search_config": {
            "screen_seed": config.screen_seed,
            "rerank_seed": config.rerank_seed,
            "veto_seed": config.veto_seed,
            "assessment_seed": config.assessment_seed,
            "child_policy_seed": config.child_policy_seed,
            "batch_child_selectors": config.batch_child_selectors,
        },
        "continuation_policy": {
            "t2_policy_id": "stage9f_p2",
            "t2_resolution": "explicit_profile_never_current",
        },
        "frozen_candidate_generator": {
            "artifact_sha256": ATTEMPT06_FROZEN_MODEL_SHA256,
            "runtime_authorized": False,
        },
        "teacher_value_status": "diagnostic_not_match_EV",
        "runtime_gate_allowed": False,
        "profile_activation_allowed": False,
        "current_profile_resolved": False,
        "development_only": True,
        # Deliberate value-bearing payload: the harness must hash, not export it.
        "arms": {"R32_V64": {"mean": 999.0, "selected_action_key": "secret"}},
        "assessment": {"sample_best_score": 123.0},
    }


def _mock_heavy(monkeypatch, calls: list[dict] | None = None) -> None:
    seen = calls if calls is not None else []

    monkeypatch.setattr(
        preflight,
        "load_model_bundle",
        lambda _paths, *, profiles: seen.append({"profiles": set(profiles)})
        or object(),
    )
    monkeypatch.setattr(
        preflight,
        "build_policy",
        lambda profile, _bundle, **kwargs: {
            "profile": profile,
            "seat": kwargs["seat"],
            "seed": kwargs["seed"],
        },
    )
    monkeypatch.setattr(preflight, "_require_stage9f_p2_policies", lambda _: None)

    class Ranker:
        @staticmethod
        def load(_path, *, expected_sha256):
            assert expected_sha256 == ATTEMPT06_FROZEN_MODEL_SHA256
            return object()

    monkeypatch.setattr(preflight, "FrozenAttempt06LambdaRanker", Ranker)

    def evaluate(
        observation,
        *,
        baseline_action_key,
        ranker,
        t2_policies,
        config,
    ):
        del ranker, t2_policies
        seen.append(
            {
                "batch": config.batch_child_selectors,
                "seeds": {
                    "screen": config.screen_seed,
                    "rerank": config.rerank_seed,
                    "veto": config.veto_seed,
                    "assessment": config.assessment_seed,
                    "child": config.child_policy_seed,
                },
            }
        )
        return _fake_teacher(observation, baseline_action_key, config)

    monkeypatch.setattr(preflight, "evaluate_attempt07_t1_second", evaluate)


def test_preflight_plan_and_five_seed_slices_are_exact_and_disjoint() -> None:
    frozen = preflight.load_preflight_plan(PREFLIGHT_PLAN)
    attempt07 = load_and_validate_attempt07_plan(ATTEMPT07_PLAN)
    schedules = preflight.validate_preflight_seed_disjointness(frozen, attempt07)
    assert frozen["target"]["plan_sha256"] == M43_ATTEMPT07_PLAN_SHA256
    assert frozen["source"]["allowed_source_root_indices"] == [0, 1, 2]
    assert frozen["source"]["new_root_generation_allowed"] is False
    assert frozen["operational_go_no_go"] == {
        "all_five_done_metadata_required": True,
        "proof_aggregate_go_required": True,
        "batch_elapsed_seconds_per_root_max": 900.0,
        "scalar_elapsed_seconds_root0_max": 3600.0,
        "scalar_to_root0_batch_median_speedup_min": 1.25,
        "root0_batch_replicate_elapsed_ratio_max": 2.0,
        "peak_rss_bytes_per_job_max": 12_884_901_888,
        "operational_metrics_are_policy_science_input": False,
        "threshold_reselection_after_results_allowed": False,
    }
    assert set(schedules) == {"screen", "rerank", "veto", "assessment", "child"}
    assert all(len(values) == 3 for values in schedules.values())
    assert preflight.attempt07_preflight_seeds(2) == {
        name: base + 2 * 1_000_003
        for name, base in preflight.ATTEMPT07_PREFLIGHT_SEED_BASES.items()
    }
    with pytest.raises(ValueError, match="0, 1, 2"):
        preflight.attempt07_preflight_seeds(3)


def test_preflight_seed_overlap_is_fail_closed(monkeypatch) -> None:
    frozen = preflight.load_preflight_plan(PREFLIGHT_PLAN)
    attempt07 = load_and_validate_attempt07_plan(ATTEMPT07_PLAN)
    collided = dict(preflight.ATTEMPT07_PREFLIGHT_SEED_BASES)
    collided["screen"] = 60_106_071_901  # Attempt07 development hand root 0.
    monkeypatch.setattr(preflight, "ATTEMPT07_PREFLIGHT_SEED_BASES", collided)
    with pytest.raises(ValueError, match="overlaps"):
        preflight.validate_preflight_seed_disjointness(frozen, attempt07)


def test_source_hash_wrapper_teacher_baseline_and_observation_are_verified(
    tmp_path: Path,
) -> None:
    row, observation = preflight.load_source_row(SOURCE, 0)
    assert row["root_index"] == 0
    assert row["teacher"]["observation_fingerprint"] == observation.fingerprint()
    assert row["baseline_action_key"] == row["teacher"]["baseline_action_key"]

    changed = tmp_path / "changed.jsonl"
    changed.write_bytes(SOURCE.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="source SHA-256"):
        preflight.load_source_row(changed, 0)

    bad_wrapper = copy.deepcopy(row)
    bad_wrapper["baseline_action_key"] = row["teacher"]["selected_action_key"]
    with pytest.raises(ValueError):
        preflight.validate_source_row(bad_wrapper, 0)

    bad_teacher = copy.deepcopy(row)
    bad_teacher["teacher"]["runtime_gate_allowed"] = True
    with pytest.raises(ValueError):
        preflight.validate_source_row(bad_teacher, 0)

    bad_observation = copy.deepcopy(row)
    bad_observation["policy_observation"]["opponent_private_discards"] = ["As"]
    with pytest.raises((ValueError, TypeError)):
        preflight.validate_source_row(bad_observation, 0)


def test_mocked_preflight_output_is_atomic_canonical_deterministic_and_redacted(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[dict] = []
    _mock_heavy(monkeypatch, calls)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    row1 = preflight.run_preflight_root(source_root_index=1, output=first)
    row2 = preflight.run_preflight_root(source_root_index=1, output=second)
    assert row1 == row2
    assert first.read_bytes() == second.read_bytes() == preflight.canonical_json_bytes(row1)
    assert first.read_bytes().endswith(b"\n")
    encoded = first.read_text()
    assert '"arms"' not in encoded
    assert "sample_best_score" not in encoded
    assert '"mean"' not in encoded
    assert '"selected_action_key"' not in encoded
    assert "timestamp" not in encoded
    assert "unix" not in encoded
    assert row1["contract"]["plan_sha256"] == M43_ATTEMPT07_PLAN_SHA256
    assert row1["result_proof"]["teacher_values_exported"] is False
    assert row1["result_proof"]["arm_details_exported"] is False
    assert row1["science_boundary"] == {
        "arm_selection_allowed": False,
        "fit_allowed": False,
        "threshold_selection_allowed": False,
        "runtime_activation_allowed": False,
        "current_profile_resolved": False,
        "current_profile_mutated": False,
        "fresh_seed_or_root_opened": False,
    }
    assert all(call.get("profiles") == {"stage9f_p2"} for call in calls if "profiles" in call)


def test_scalar_and_batch_runs_bind_mode_and_frozen_seeds(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[dict] = []
    _mock_heavy(monkeypatch, calls)
    scalar = preflight.run_preflight_root(
        source_root_index=2,
        output=tmp_path / "scalar.json",
        batch_child_selectors=False,
    )
    batch = preflight.run_preflight_root(
        source_root_index=2,
        output=tmp_path / "batch.json",
        batch_child_selectors=True,
    )
    expected_seeds = preflight.attempt07_preflight_seeds(2)
    assert scalar["execution"]["batch_child_selectors"] is False
    assert batch["execution"]["batch_child_selectors"] is True
    assert scalar["contract"]["seeds"] == batch["contract"]["seeds"] == expected_seeds
    assert scalar["execution"]["run_id"] == batch["execution"]["run_id"]
    assert (
        scalar["result_proof"]["semantic_parity_sha256"]
        == batch["result_proof"]["semantic_parity_sha256"]
    )
    assert (
        scalar["result_proof"]["opaque_teacher_sha256"]
        != batch["result_proof"]["opaque_teacher_sha256"]
    )
    assert scalar["execution"]["native_batch_threads"] == 4
    assert batch["execution"]["native_batch_threads"] == 4
    evaluator_calls = [call for call in calls if "batch" in call]
    assert [call["batch"] for call in evaluator_calls] == [False, True]
    assert all(call["seeds"] == expected_seeds for call in evaluator_calls)


def test_output_may_not_alias_any_immutable_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="aliases"):
        preflight.run_preflight_root(source_root_index=0, output=SOURCE)
    parser = preflight._parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--source-root-index", "3", "--output", str(tmp_path / "x")])


def test_output_is_no_clobber(tmp_path: Path, monkeypatch) -> None:
    _mock_heavy(monkeypatch)
    output = tmp_path / "proof.json"
    output.write_text("owned\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        preflight.run_preflight_root(source_root_index=0, output=output)
    assert output.read_text(encoding="utf-8") == "owned\n"


def test_semantic_parity_hash_keeps_all_value_fields() -> None:
    row = _source_row(0)
    observation = preflight.validate_source_row(row, 0)

    class Config:
        screen_seed = 1
        rerank_seed = 2
        veto_seed = 3
        assessment_seed = 4
        child_policy_seed = 5
        batch_child_selectors = False

    scalar = _fake_teacher(observation, row["baseline_action_key"], Config())
    batch = copy.deepcopy(scalar)
    batch["search_config"]["batch_child_selectors"] = True
    assert preflight._semantic_parity_sha256(scalar) == preflight._semantic_parity_sha256(
        batch
    )
    batch["assessment"]["sample_best_score"] += 1.0
    assert preflight._semantic_parity_sha256(scalar) != preflight._semantic_parity_sha256(
        batch
    )


def test_scalar_temporarily_removes_parent_batch_threads_and_restores_it() -> None:
    previous = os.environ.get("OFC_HU_M3_BATCH_THREADS")
    os.environ["OFC_HU_M3_BATCH_THREADS"] = "17"
    try:
        with preflight._native_batch_threads(False, 4):
            assert "OFC_HU_M3_BATCH_THREADS" not in os.environ
        assert os.environ["OFC_HU_M3_BATCH_THREADS"] == "17"
        with preflight._native_batch_threads(True, 4):
            assert os.environ["OFC_HU_M3_BATCH_THREADS"] == "4"
        assert os.environ["OFC_HU_M3_BATCH_THREADS"] == "17"
    finally:
        if previous is None:
            os.environ.pop("OFC_HU_M3_BATCH_THREADS", None)
        else:
            os.environ["OFC_HU_M3_BATCH_THREADS"] = previous
