from __future__ import annotations

import copy
import hashlib
import json
from itertools import combinations
from pathlib import Path

import pytest

import ofc_regular.run_hu_m43_attempt08_preflight as preflight
from ofc_regular.hu_m43_attempt08_contract import (
    M43_ATTEMPT08_PLAN_SHA256,
    enumerate_attempt06_known_seed_schedules,
    enumerate_attempt07_known_seed_schedules,
    enumerate_attempt07_preflight_known_seed_schedules,
    enumerate_attempt08_seed_schedules,
    load_and_validate_attempt08_plan,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "hu_joint_policy_m43_attempt08.json"
PREFLIGHT = ROOT / "configs" / "hu_joint_policy_m43_attempt08_preflight.json"


def test_preflight_plan_is_hash_frozen_to_consumed_roots_and_c4_limits() -> None:
    plan = preflight.load_preflight_plan(PREFLIGHT)
    assert hashlib.sha256(PREFLIGHT.read_bytes()).hexdigest() == (
        preflight.ATTEMPT08_PREFLIGHT_PLAN_SHA256
    )
    assert plan["source"]["allowed_source_root_indices"] == [0, 1, 2]
    assert plan["source"]["source_already_consumed_development_evidence"] is True
    assert plan["source"]["new_root_generation_allowed"] is False
    assert plan["target"]["plan_sha256"] == M43_ATTEMPT08_PLAN_SHA256
    assert plan["execution"]["machine_type"] == "c4-highmem-4"
    assert plan["execution"]["development200_authorized"] is False
    gates = plan["correctness_go_no_go"]
    assert gates["teacher_elapsed_seconds_each_run_max"] == 2400.0
    assert gates["process_peak_rss_bytes_each_run_max"] == 28 * 1024**3


def test_preflight_slots_are_exact_root0_ab_scalar_root1_root2_batch() -> None:
    assert preflight.ATTEMPT08_PREFLIGHT_SLOTS == {
        "root0_batch_a": (0, True),
        "root0_batch_b": (0, True),
        "root0_scalar": (0, False),
        "root1_batch": (1, True),
        "root2_batch": (2, True),
    }
    parser = preflight._parser()
    args = parser.parse_args(
        ["--slot", "root0_batch_a", "--output", "proof.json"]
    )
    assert args.slot == "root0_batch_a"
    with pytest.raises(SystemExit):
        parser.parse_args(["--slot", "root3_batch", "--output", "proof.json"])


def test_preflight_six_seed_slices_are_pairwise_disjoint_and_fresh() -> None:
    proof_plan = preflight.load_preflight_plan(PREFLIGHT)
    target = load_and_validate_attempt08_plan(PLAN)
    schedules = preflight.validate_preflight_seed_disjointness(proof_plan, target)
    assert set(schedules) == {
        "hand",
        "rerank",
        "veto",
        "stress",
        "assessment",
        "child",
    }
    assert schedules["hand"] == tuple(
        90_108_071_901 + 1_000_003 * index for index in range(3)
    )
    assert schedules["child"] == tuple(
        95_108_071_901 + 1_000_003 * index for index in range(3)
    )
    for left, right in combinations(schedules, 2):
        assert set(schedules[left]).isdisjoint(schedules[right])

    prior = (
        enumerate_attempt06_known_seed_schedules(),
        enumerate_attempt07_known_seed_schedules(),
        enumerate_attempt07_preflight_known_seed_schedules(),
    )
    current = {
        population: enumerate_attempt08_seed_schedules(target, population=population)
        for population in ("development", "future_audit")
    }
    for values in schedules.values():
        for group in prior:
            for old in group.values():
                assert set(values).isdisjoint(old)
        for group in current.values():
            for reserved in group.values():
                assert set(values).isdisjoint(reserved)


def test_preflight_seed_proof_rejects_prior_collision(monkeypatch) -> None:
    collided = dict(preflight.ATTEMPT08_PREFLIGHT_SEED_BASES)
    collided["hand"] = 71_106_071_901
    monkeypatch.setattr(preflight, "ATTEMPT08_PREFLIGHT_SEED_BASES", collided)
    with pytest.raises(ValueError, match="overlaps attempt07_preflight"):
        preflight.validate_preflight_seed_disjointness(
            preflight.load_preflight_plan(PREFLIGHT),
            load_and_validate_attempt08_plan(PLAN),
        )


def test_only_source_roots_zero_one_two_are_accepted() -> None:
    for root in range(3):
        row, observation = preflight.load_source_row(preflight.DEFAULT_SOURCE_PATH, root)
        assert row["root_index"] == root
        assert row["policy_observation"] == observation.to_dict()
    with pytest.raises(ValueError):
        preflight.attempt08_preflight_seeds(3)
    with pytest.raises(ValueError):
        preflight.attempt08_preflight_seeds(True)


def test_semantic_hash_normalizes_only_scalar_batch_flag() -> None:
    scalar = {
        "search_config": {"batch_child_selectors": False, "seed": 9},
        "decision": {"fired": False},
    }
    batch = copy.deepcopy(scalar)
    batch["search_config"]["batch_child_selectors"] = True
    assert preflight._semantic_parity_sha256(scalar) == (
        preflight._semantic_parity_sha256(batch)
    )
    batch["decision"]["fired"] = True
    assert preflight._semantic_parity_sha256(scalar) != (
        preflight._semantic_parity_sha256(batch)
    )


def _mock_heavy(monkeypatch, times: list[float]) -> None:
    monkeypatch.setattr(
        preflight, "validate_runtime_semantic_anchor", lambda **_kwargs: None
    )
    monkeypatch.setattr(
        preflight,
        "validate_expected_runtime_fingerprint",
        lambda: preflight.ATTEMPT08_EXPECTED_RUNTIME_FINGERPRINT_SHA256,
    )
    monkeypatch.setattr(preflight, "load_model_bundle", lambda *a, **k: object())
    monkeypatch.setattr(
        preflight,
        "build_policy",
        lambda *a, **k: object(),
    )
    monkeypatch.setattr(preflight, "_require_stage9f_p2_policies", lambda value: None)
    monkeypatch.setattr(
        preflight.FrozenAttempt08LambdaRanker,
        "load",
        lambda *a, **k: object(),
    )

    def evaluate(_observation, *, config, **_kwargs):
        return {
            "search_config": {
                "batch_child_selectors": config.batch_child_selectors,
                "fixed": "same",
            },
            "conditional_phase_shape": "R_V_optional_X_optional_A",
        }

    monkeypatch.setattr(preflight, "evaluate_attempt08_t1_second", evaluate)
    monkeypatch.setattr(
        preflight,
        "_validate_attempt08_result",
        lambda *a, **k: {
            "exact_actionkey_reference_parity_verified": True,
            "hidden_information_safety_verified": True,
            "rng_domain_separation_verified": True,
            "conditional_X_A_skip_contract_verified": True,
        },
    )
    values = iter(times)
    monkeypatch.setattr(preflight.time, "perf_counter", lambda: next(values))
    monkeypatch.setattr(preflight, "_process_peak_rss_bytes", lambda: 123_456_789)


def test_mocked_runner_is_redacted_and_proves_batch_and_cross_mode_hashes(
    tmp_path: Path, monkeypatch
) -> None:
    _mock_heavy(monkeypatch, [10.0, 22.5, 30.0, 42.5, 50.0, 65.0])
    batch_a = preflight.run_preflight_slot(
        slot="root0_batch_a", output=tmp_path / "batch-a.json"
    )
    batch_b = preflight.run_preflight_slot(
        slot="root0_batch_b", output=tmp_path / "batch-b.json"
    )
    scalar = preflight.run_preflight_slot(
        slot="root0_scalar", output=tmp_path / "scalar.json"
    )
    assert batch_a["result_proof"]["opaque_teacher_sha256"] == batch_b[
        "result_proof"
    ]["opaque_teacher_sha256"]
    assert batch_a["result_proof"]["semantic_parity_sha256"] == scalar[
        "result_proof"
    ]["semantic_parity_sha256"]
    assert batch_a["result_proof"]["opaque_teacher_sha256"] != scalar[
        "result_proof"
    ]["opaque_teacher_sha256"]
    assert batch_a["execution"]["teacher_elapsed_seconds"] == 12.5
    assert batch_a["execution"]["process_peak_rss_bytes"] == 123_456_789
    assert batch_a["science_boundary"]["development200_authorized"] is False
    raw = (tmp_path / "batch-a.json").read_bytes()
    assert raw == preflight.canonical_json_bytes(batch_a)
    for forbidden in (
        b'"selected_action_key"',
        b'"raw_paired_deltas',
        b'"paired_delta_vs_baseline"',
        b'"legal_actions"',
    ):
        assert forbidden not in raw


def test_runner_is_atomic_no_clobber_and_rejects_input_alias(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "owned.json"
    output.write_text("owned\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        preflight.run_preflight_slot(slot="root1_batch", output=output)
    assert output.read_text(encoding="utf-8") == "owned\n"
    with pytest.raises(ValueError, match="aliases"):
        preflight.run_preflight_slot(
            slot="root1_batch", output=preflight.DEFAULT_SOURCE_PATH
        )
