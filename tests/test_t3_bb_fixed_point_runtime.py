from __future__ import annotations

import copy
import json
from dataclasses import replace
from fractions import Fraction

import pytest

from ai.tutor.calibrated_behavior_manifest_bridge import (
    build_calibrated_behavior_bridge,
)
from ai.tutor.calibrated_behavior_runtime import (
    build_verified_calibrated_hu_t1_t2_behavior_dispatch,
)
from ai.tutor.calibrated_behavior_bootstrap import (
    build_fixed_point_bootstrap_calibrated_hu_t1_t2_behavior_dispatch,
)
from ai.tutor.behavior_temperature_calibration import (
    build_behavior_temperature_calibration,
)
from ai.tutor.promotion_gate_m3_full_card_strength import (
    T3_BB_LIKELIHOOD_METHOD,
    T3_BB_LIKELIHOOD_SCHEMA,
    canonical_sha256,
    self_hash,
)
from ai.tutor.t3_bb_fixed_point_runtime import (
    COMPOSITE_MODEL_TYPE,
    ITERATION_MODEL_TYPE,
    build_t3_bb_candidate_policy_artifact,
    build_fixed_point_iteration_behavior_dispatch,
    build_verified_fixed_point_t3_bb_behavior_model,
    build_verified_m3_behavior_likelihood_dispatch,
    read_t3_bb_candidate_policy_artifact,
    verify_t3_bb_candidate_policy_artifact,
)
from test_behavior_temperature_calibration import (
    _dataset,
    _small_gate,
    _t1_bb_information,
)
from test_calibrated_behavior_manifest_bridge import (
    TRAINING_ROOTS,
    _promoted_fixture,
)
from test_calibrated_behavior_runtime import _install_fake_loaders
from test_calibrated_behavior_bootstrap import (
    _install_fake_loaders as _install_fake_bootstrap_loaders,
)


def _t3_information(actor: str = "bb"):
    return replace(_t1_bb_information(), turn=3, actor=actor)


def _artifact():
    information = _t3_information()
    probability = Fraction(1, information.legal_action_count)
    artifact = build_t3_bb_candidate_policy_artifact(
        {
            information.digest(): {
                action_id: probability
                for action_id in information.legal_action_ids
            }
        },
        checkpoint_sha256=canonical_sha256({"fixture": "checkpoint"}),
        solver_manifest_sha256=canonical_sha256({"fixture": "solver"}),
        range_builder_source_sha256=canonical_sha256({"fixture": "range"}),
    )
    return information, artifact


def _binding(artifact):
    manifest = artifact["model_manifest"]
    binding = {
        "schema": T3_BB_LIKELIHOOD_SCHEMA,
        "route": "t3_bb",
        "consumer_root_actor": "btn",
        "consumer_phase": "t3_second",
        "method": T3_BB_LIKELIHOOD_METHOD,
        "promotion_eligible": True,
        "fixed_point_converged": True,
        "candidate_policy_artifact_sha256": artifact["artifact_sha256"],
        "candidate_policy_checkpoint_sha256": manifest["checkpoint_sha256"],
        "fixed_point_evidence_sha256": canonical_sha256({"fixture": "evidence"}),
        "fixed_point_gate_result_sha256": canonical_sha256({"fixture": "result"}),
        "fixed_point_config_sha256": canonical_sha256({"fixture": "config"}),
        "range_builder_source_sha256": manifest["range_builder_source_sha256"],
        "solver_manifest_sha256": manifest["solver_manifest_sha256"],
    }
    binding["binding_sha256"] = self_hash(binding, "binding_sha256")
    return binding


def _t1_t2_runtime(monkeypatch, binding):
    records, rows, challenge_records, challenge_rows, calibration = (
        _promoted_fixture()
    )
    bridge = build_calibrated_behavior_bridge(
        records,
        rows,
        calibration,
        training_root_ids=TRAINING_ROOTS,
        t3_bb_likelihood_binding=binding,
        challenge_records=challenge_records,
        challenge_evaluation_rows=challenge_rows,
    )
    _install_fake_loaders(monkeypatch, bridge)
    runtime = build_verified_calibrated_hu_t1_t2_behavior_dispatch(
        records,
        rows,
        calibration,
        bridge,
        training_root_ids=TRAINING_ROOTS,
        t3_bb_likelihood_binding=binding,
        challenge_records=challenge_records,
        challenge_evaluation_rows=challenge_rows,
        workspace_root=".",
    )
    return runtime


def test_candidate_artifact_is_exact_content_addressed_and_no_fallback():
    information, artifact = _artifact()
    binding = _binding(artifact)

    assert verify_t3_bb_candidate_policy_artifact(
        artifact, t3_bb_likelihood_binding=binding
    ) == artifact
    model = build_verified_fixed_point_t3_bb_behavior_model(
        artifact, t3_bb_likelihood_binding=binding
    )
    assert model.model_sha256 == artifact["artifact_sha256"]
    distribution = model.action_distribution(information)
    assert distribution.source == "table"
    assert distribution.used_fallback is False
    assert set(distribution.probabilities) == set(information.legal_action_ids)
    assert sum(distribution.probabilities.values(), Fraction(0, 1)) == 1

    with pytest.raises(KeyError, match="exactly the T3-BB route"):
        model.action_distribution(replace(information, actor="btn"))


def test_artifact_or_binding_tamper_fails_closed():
    _information, artifact = _artifact()
    binding = _binding(artifact)

    tampered = copy.deepcopy(artifact)
    digest = next(iter(tampered["model_manifest"]["probabilities"]))
    action = next(iter(tampered["model_manifest"]["probabilities"][digest]))
    tampered["model_manifest"]["probabilities"][digest][action] = "0/1"
    with pytest.raises(ValueError, match="sum exactly to one"):
        verify_t3_bb_candidate_policy_artifact(tampered)

    wrong_binding = copy.deepcopy(binding)
    wrong_binding["candidate_policy_checkpoint_sha256"] = "1" * 64
    wrong_binding["binding_sha256"] = self_hash(
        wrong_binding, "binding_sha256"
    )
    with pytest.raises(ValueError, match="candidate_policy_checkpoint_sha256"):
        verify_t3_bb_candidate_policy_artifact(
            artifact, t3_bb_likelihood_binding=wrong_binding
        )


def test_strict_reader_rejects_duplicate_json_keys(tmp_path):
    path = tmp_path / "candidate.json"
    path.write_text(
        '{"schema":"one","schema":"two","model_manifest":{},'
        '"artifact_sha256":"' + "0" * 64 + '"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate key"):
        read_t3_bb_candidate_policy_artifact(path)


def test_composite_runtime_closes_m3_t3_root_route_without_overclaim(
    monkeypatch,
):
    information, artifact = _artifact()
    binding = _binding(artifact)
    t1_t2 = _t1_t2_runtime(monkeypatch, binding)

    composite = build_verified_m3_behavior_likelihood_dispatch(
        t1_t2,
        artifact,
        t3_bb_likelihood_binding=binding,
    )
    manifest = dict(composite.model_manifest)
    assert manifest["model_type"] == COMPOSITE_MODEL_TYPE
    assert manifest["promotion_eligible"] is True
    assert manifest["m3_t3_root_behavior_complete"] is True
    assert manifest["all_turn_behavior_complete"] is False
    assert manifest["t4_posterior_routes_complete"] is False
    assert manifest["t3_bb_candidate_policy_artifact_sha256"] == artifact[
        "artifact_sha256"
    ]
    assert composite.model_sha256 == canonical_sha256(manifest)
    assert composite.action_distribution(information).source == "table"
    assert composite.action_distribution(_t1_bb_information()).source == "model"

    with pytest.raises(KeyError, match="T3-BTN/T4"):
        composite.action_distribution(_t3_information(actor="btn"))

    changed = copy.deepcopy(binding)
    changed["fixed_point_evidence_sha256"] = "2" * 64
    changed["binding_sha256"] = self_hash(changed, "binding_sha256")
    with pytest.raises(ValueError, match="different T3-BB binding"):
        build_verified_m3_behavior_likelihood_dispatch(
            t1_t2,
            artifact,
            t3_bb_likelihood_binding=changed,
        )


def test_unfinished_iteration_dispatch_uses_candidate_without_fake_binding(
    monkeypatch,
):
    information, artifact = _artifact()
    records, rows, challenge_records, challenge_rows = _dataset()
    calibration = build_behavior_temperature_calibration(
        records,
        rows,
        challenge_records=challenge_records,
        challenge_evaluation_rows=challenge_rows,
        gate_config=_small_gate(),
    )
    _install_fake_bootstrap_loaders(monkeypatch, calibration)
    bootstrap = build_fixed_point_bootstrap_calibrated_hu_t1_t2_behavior_dispatch(
        records,
        rows,
        calibration,
        challenge_records=challenge_records,
        challenge_evaluation_rows=challenge_rows,
        workspace_root=".",
    )

    iteration = build_fixed_point_iteration_behavior_dispatch(
        bootstrap, artifact
    )
    manifest = dict(iteration.model_manifest)
    assert manifest["model_type"] == ITERATION_MODEL_TYPE
    assert manifest["promotion_eligible"] is False
    assert manifest["fixed_point_iteration_only"] is True
    assert manifest["fixed_point_converged"] is False
    assert manifest["strategic_strength_evaluated"] is False
    assert manifest["t3_bb_candidate_policy_artifact_sha256"] == artifact[
        "artifact_sha256"
    ]
    assert iteration.action_distribution(information).source == "table"
    assert iteration.action_distribution(_t1_bb_information()).source == "model"
    with pytest.raises(KeyError, match="no route"):
        iteration.action_distribution(_t3_information(actor="btn"))

    with pytest.raises(TypeError, match="fixed-point bootstrap runtime"):
        build_fixed_point_iteration_behavior_dispatch(object(), artifact)
