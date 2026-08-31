from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest

import ai.tutor.run_t3_bb_fixed_point_smoke as smoke
from ai.tutor.t3_bb_range_evidence import EVIDENCE_SCHEMA as RANGE_EVIDENCE_SCHEMA
from ai.tutor.t3_bb_candidate_queries import Q32_DENOMINATOR
from ai.tutor.t3_bb_fixed_point_runtime import (
    build_t3_bb_candidate_policy_artifact,
    read_t3_bb_candidate_policy_artifact,
)


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_REPORT = (
    WORKSPACE_ROOT / "ai" / "reports" / "m3_behavior_calibration_smoke_20260713"
)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_sha256(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.fixture(scope="module")
def fixed_point_smoke(tmp_path_factory):
    output = tmp_path_factory.mktemp("t3-bb-fixed-point") / "report"
    kwargs = {
        "workspace_root": WORKSPACE_ROOT,
        "output_dir": output,
        "calibration_report_dir": CALIBRATION_REPORT,
    }

    first = smoke.run_t3_bb_fixed_point_smoke(**kwargs)
    first_tree = _tree_sha256(output)
    second = smoke.run_t3_bb_fixed_point_smoke(**kwargs)
    second_tree = _tree_sha256(output)

    assert second == first
    assert second["report_sha256"] == first["report_sha256"]
    assert second_tree == first_tree
    assert smoke.verify_t3_bb_fixed_point_smoke_output(
        output,
        calibration_report_dir=CALIBRATION_REPORT,
        workspace_root=WORKSPACE_ROOT,
    ) == second
    return output, second


def test_real_mccfr_assets_and_canonical_six_root_query_chain(fixed_point_smoke):
    output, report = fixed_point_smoke

    assert report["artifact_kind"] == "real_full_card_mccfr_fixed_point_wiring_smoke"
    assert report["strata"] == [
        "bb_joker0",
        "bb_joker1",
        "bb_joker2",
        "btn_joker0",
        "btn_joker1",
        "btn_joker2",
    ]
    assert report["query_count"] == 6
    assert report["candidate_count"] == 3
    assert report["transition_count"] == 2
    assert report["solver_seeds"] == [20260731, 20260732]
    assert report["iterations_per_job"] == 1
    assert report["max_particles"] == 1
    assert report["raw_iteration_row_count"] == 24
    assert report["candidate_semantic_tables_identical"] is True
    assert report["raw_calibration_fully_reverified"] is True
    assert report["initial_t3_policy_prior_used"] is False
    assert report["initial_t3_uniform_use_scope"] == "query_discovery_only"
    assert report["missing_query_fallback_allowed"] is False

    checkpoints = sorted(output.rglob("checkpoint.json"))
    strategies = sorted(output.rglob("average-strategy.json"))
    restricted_ranges = sorted(output.rglob("restricted-range.json"))
    assert len(checkpoints) == len(strategies) == len(restricted_ranges) == 72

    for path in checkpoints:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
        assert checkpoint["format"] == "full_card_dynamic_mccfr_checkpoint_v1"
        assert checkpoint["payload"]["completed_iterations"] == 1
        assert checkpoint["payload"]["seed"] in report["solver_seeds"]
    for path in strategies:
        strategy = json.loads(path.read_text(encoding="utf-8"))
        assert strategy["schema"] == "ofc_full_card_public_strategy/v1"
    for path in restricted_ranges:
        restricted = json.loads(path.read_text(encoding="utf-8"))
        assert restricted["schema"] == RANGE_EVIDENCE_SCHEMA
        assert restricted["restricted_hidden_information"] is True

    for link in report["candidate_chain"]:
        assert (
            link["query_bundle_manifest_sha256"]
            != link["evaluation_bundle_manifest_sha256"]
        )


def test_candidate_tables_are_exact_q32_and_missing_query_is_fail_closed(
    fixed_point_smoke,
):
    output, report = fixed_point_smoke
    candidate = read_t3_bb_candidate_policy_artifact(
        output / "candidate-000" / "candidate-policy.json"
    )
    manifest = candidate["model_manifest"]
    probabilities = manifest["probabilities"]
    assert len(probabilities) == report["query_count"]
    for row in probabilities.values():
        exact = [Fraction(value) for value in row.values()]
        assert sum(exact, Fraction(0, 1)) == 1
        assert all((value * Q32_DENOMINATOR).denominator == 1 for value in exact)

    missing_digest = sorted(probabilities)[0]
    incomplete = {
        digest: {
            action_id: Fraction(value)
            for action_id, value in row.items()
        }
        for digest, row in probabilities.items()
        if digest != missing_digest
    }
    rebuilt = build_t3_bb_candidate_policy_artifact(
        incomplete,
        checkpoint_sha256=manifest["checkpoint_sha256"],
        solver_manifest_sha256=manifest["solver_manifest_sha256"],
        range_builder_source_sha256=manifest["range_builder_source_sha256"],
        model_id=manifest["model_id"],
    )
    expected_queries = {digest: None for digest in probabilities}
    with pytest.raises(ValueError, match="candidate query coverage mismatch: missing="):
        smoke._assert_candidate_query_coverage(rebuilt, expected_queries)


def test_v2_gate_replays_all_assets_but_cannot_promote(fixed_point_smoke):
    output, report = fixed_point_smoke
    gate_result = json.loads(
        (output / "fixed-point-result.json").read_text(encoding="utf-8")
    )

    assert gate_result["schema"] == "ofc_m3_t3_bb_fixed_point_gate_result/v2"
    assert gate_result["derived_metrics"] is not None
    assert gate_result["passed"] is False
    assert gate_result["promotion_eligible"] is False
    assert report["v2_fresh_asset_replay_completed"] is True
    for field in (
        "promotion_eligible",
        "fixed_point_promotion_passed",
        "fixed_point_converged_claimed",
        "strategic_strength_evaluated",
        "strategic_strength_claimed",
        "diagnostic_zero_drift_is_strength_evidence",
        "exact_exploitability_computed",
        "all_turn_ai_complete",
        "production_gate_passed",
        "t3_bb_likelihood_binding_emitted",
    ):
        assert report[field] is False
    assert report["independent_strength_gate_required"] is True

    failures = "\n".join(report["fixed_point_gate_failures"])
    assert "independent seeds: 2 < 3" in failures
    assert "roots per stratum" in failures
    assert "transition rounds: 2 < 3" in failures
    assert "trailing production-converged rounds: 0 < 3" in failures
    assert "production promotion claim is false" in failures
    assert len(report["diagnostic_round_metrics"]) == 2
    for row in report["diagnostic_round_metrics"]:
        assert row["converged"] is False
        assert row["max_previous_to_current_policy_tv"] == "0/1"
        assert row["max_previous_to_current_btn_posterior_tv"] == "0/1"
        assert row["max_same_root_cross_seed_policy_tv"] == "1/1"
        assert row["max_same_root_cross_seed_btn_posterior_tv"] == "1/1"


@pytest.mark.parametrize(
    ("seeds", "transitions", "message"),
    [
        ((20260731,), 2, "at least two independent seeds"),
        ((20260731, 20260732), 1, "at least two transitions"),
    ],
)
def test_smoke_input_floor_is_fail_closed(seeds, transitions, message):
    with pytest.raises(ValueError, match=message):
        smoke._validate_inputs(
            seeds=seeds,
            transitions=transitions,
            iterations=1,
            max_particles=1,
            max_infosets=100_000,
            epsilon=Fraction(1, 1000),
        )
