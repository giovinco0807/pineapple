from __future__ import annotations

import copy
from fractions import Fraction

import pytest

from ai.tutor.behavior_temperature_calibration import (
    build_behavior_temperature_calibration,
)
from ai.tutor.calibrated_behavior_manifest_bridge import (
    BRIDGE_SCHEMA,
    build_calibrated_behavior_bridge,
    build_root_partition_manifest,
    verify_calibrated_behavior_bridge,
)
from ai.tutor.promotion_gate_m3_full_card_strength import (
    CALIBRATED_BEHAVIOR_KEYS,
    T3_BB_LIKELIHOOD_METHOD,
    T3_BB_LIKELIHOOD_SCHEMA,
    _validate_behavior,
    canonical_sha256,
    root_identity_commitment_sha256,
    self_hash,
)
from test_behavior_temperature_calibration import _dataset, _small_gate


TRAINING_ROOTS = ("behavior-training-root-0", "behavior-training-root-1")


def _t3_binding() -> dict:
    binding = {
        "schema": T3_BB_LIKELIHOOD_SCHEMA,
        "route": "t3_bb",
        "consumer_root_actor": "btn",
        "consumer_phase": "t3_second",
        "method": T3_BB_LIKELIHOOD_METHOD,
        "promotion_eligible": True,
        "fixed_point_converged": True,
        "candidate_policy_artifact_sha256": canonical_sha256(
            {"fixture": "candidate-policy"}
        ),
        "candidate_policy_checkpoint_sha256": canonical_sha256(
            {"fixture": "candidate-checkpoint"}
        ),
        "fixed_point_evidence_sha256": canonical_sha256(
            {"fixture": "fixed-point-evidence"}
        ),
        "fixed_point_gate_result_sha256": canonical_sha256(
            {"fixture": "fixed-point-result"}
        ),
        "fixed_point_config_sha256": canonical_sha256(
            {"fixture": "fixed-point-config"}
        ),
        "range_builder_source_sha256": canonical_sha256(
            {"fixture": "range-builder-source"}
        ),
        "solver_manifest_sha256": canonical_sha256(
            {"fixture": "solver-manifest"}
        ),
    }
    binding["binding_sha256"] = self_hash(binding, "binding_sha256")
    return binding


def _promoted_fixture():
    records, rows, challenge_records, challenge_rows = _dataset()
    artifact = build_behavior_temperature_calibration(
        records,
        rows,
        challenge_records=challenge_records,
        challenge_evaluation_rows=challenge_rows,
        gate_config=_small_gate(),
    )
    assert artifact["promotion_eligible"] is True
    return records, rows, challenge_records, challenge_rows, artifact


def _bridge():
    records, rows, challenge_records, challenge_rows, artifact = _promoted_fixture()
    bridge = build_calibrated_behavior_bridge(
        records,
        rows,
        artifact,
        training_root_ids=TRAINING_ROOTS,
        t3_bb_likelihood_binding=_t3_binding(),
        challenge_records=challenge_records,
        challenge_evaluation_rows=challenge_rows,
    )
    return records, rows, challenge_records, challenge_rows, artifact, bridge


def test_bridge_is_exact_strength_contract_and_includes_every_challenge_root():
    records, rows, challenge_records, challenge_rows, artifact, bridge = _bridge()

    assert bridge["schema"] == BRIDGE_SCHEMA
    assert bridge["strategic_strength_evaluated"] is False
    assert bridge["strategic_strength_claimed"] is False
    behavior = bridge["calibrated_behavior_manifest"]
    assert set(behavior) == set(CALIBRATED_BEHAVIOR_KEYS)
    assert bridge["calibrated_behavior_sha256"] == canonical_sha256(behavior)
    assert behavior["calibration_artifact_sha256"] == artifact["artifact_sha256"]
    assert (
        behavior["calibration_gate_config_sha256"]
        == artifact["gate_config"]["gate_config_sha256"]
    )
    assert (
        behavior["calibration_gate_result_sha256"]
        == artifact["gate_result"]["gate_result_sha256"]
    )
    assert set(behavior["role_temperatures"]) == {
        "t1_bb",
        "t1_btn",
        "t2_bb",
        "t2_btn",
    }
    for role, value in behavior["role_temperatures"].items():
        source = artifact["temperatures"][role]["final_temperature"]
        expected = Fraction(source["numerator"], source["denominator"])
        assert value == f"{expected.numerator}/{expected.denominator}"
    assert behavior["t3_bb_likelihood_binding"] == _t3_binding()

    partitions = bridge["root_partitions"]
    assert behavior["training_root_partition_sha256"] == partitions["training"][
        "manifest_sha256"
    ]
    assert behavior["calibration_root_partition_sha256"] == partitions[
        "calibration"
    ]["manifest_sha256"]
    calibration_commitments = set(partitions["calibration"]["root_commitments"])
    assert {
        root_identity_commitment_sha256(record["root_id"])
        for record in challenge_records
    } <= calibration_commitments
    assert {
        root_identity_commitment_sha256(record["root_id"]) for record in records
    } <= calibration_commitments

    # Exercise the strength gate's own behavior validator, not just duplicated
    # assertions in the bridge.
    partition_hashes = {
        name: partition["manifest_sha256"] for name, partition in partitions.items()
    }
    failures: list[str] = []
    _validate_behavior(
        behavior,
        claimed_sha256=bridge["calibrated_behavior_sha256"],
        approved_sha256=bridge["calibrated_behavior_sha256"],
        partition_hashes=partition_hashes,
        failures=failures,
    )
    assert failures == []
    assert verify_calibrated_behavior_bridge(
        records,
        rows,
        artifact,
        bridge,
        training_root_ids=TRAINING_ROOTS,
        t3_bb_likelihood_binding=_t3_binding(),
        challenge_records=challenge_records,
        challenge_evaluation_rows=challenge_rows,
    ) == bridge


def test_bridge_rejects_missing_semantics_or_changed_t3_bb_fixed_point_binding():
    records, rows, challenge_records, challenge_rows, artifact, bridge = _bridge()
    prior = _t3_binding()
    prior["method"] = "frozen_teacher_ranking_prior_v1"
    prior["binding_sha256"] = self_hash(prior, "binding_sha256")
    with pytest.raises(ValueError, match="method"):
        build_calibrated_behavior_bridge(
            records,
            rows,
            artifact,
            training_root_ids=TRAINING_ROOTS,
            t3_bb_likelihood_binding=prior,
            challenge_records=challenge_records,
            challenge_evaluation_rows=challenge_rows,
        )

    changed = _t3_binding()
    changed["candidate_policy_artifact_sha256"] = canonical_sha256(
        {"fixture": "different-candidate-policy"}
    )
    changed["binding_sha256"] = self_hash(changed, "binding_sha256")
    with pytest.raises(ValueError, match="does not match raw inputs"):
        verify_calibrated_behavior_bridge(
            records,
            rows,
            artifact,
            bridge,
            training_root_ids=TRAINING_ROOTS,
            t3_bb_likelihood_binding=changed,
            challenge_records=challenge_records,
            challenge_evaluation_rows=challenge_rows,
        )


def test_source_artifact_tamper_and_resigned_bridge_tamper_fail_closed():
    records, rows, challenge_records, challenge_rows, artifact, bridge = _bridge()

    tampered_artifact = copy.deepcopy(artifact)
    tampered_artifact["temperatures"]["t1_bb"]["final_temperature"][
        "numerator"
    ] += 1
    tampered_artifact["artifact_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in tampered_artifact.items()
            if key != "artifact_sha256"
        }
    )
    with pytest.raises(ValueError, match="does not match raw inputs"):
        build_calibrated_behavior_bridge(
            records,
            rows,
            tampered_artifact,
            training_root_ids=TRAINING_ROOTS,
            t3_bb_likelihood_binding=_t3_binding(),
            challenge_records=challenge_records,
            challenge_evaluation_rows=challenge_rows,
        )

    tampered_bridge = copy.deepcopy(bridge)
    tampered_bridge["calibrated_behavior_manifest"]["model_type"] += "-tampered"
    tampered_bridge["calibrated_behavior_sha256"] = canonical_sha256(
        tampered_bridge["calibrated_behavior_manifest"]
    )
    tampered_bridge["bridge_sha256"] = self_hash(tampered_bridge, "bridge_sha256")
    with pytest.raises(ValueError, match="does not match raw inputs"):
        verify_calibrated_behavior_bridge(
            records,
            rows,
            artifact,
            tampered_bridge,
            training_root_ids=TRAINING_ROOTS,
            t3_bb_likelihood_binding=_t3_binding(),
            challenge_records=challenge_records,
            challenge_evaluation_rows=challenge_rows,
        )


def test_unpromoted_artifact_and_training_calibration_overlap_are_rejected():
    records, rows, challenge_records, challenge_rows = _dataset(dev_observed="last")
    unpromoted = build_behavior_temperature_calibration(
        records,
        rows,
        challenge_records=challenge_records,
        challenge_evaluation_rows=challenge_rows,
        gate_config=_small_gate(require_nonidentity_temperature=True),
    )
    assert unpromoted["promotion_eligible"] is False
    with pytest.raises(ValueError, match="not promotion eligible"):
        build_calibrated_behavior_bridge(
            records,
            rows,
            unpromoted,
            training_root_ids=TRAINING_ROOTS,
            t3_bb_likelihood_binding=_t3_binding(),
            challenge_records=challenge_records,
            challenge_evaluation_rows=challenge_rows,
        )

    records, rows, challenge_records, challenge_rows, promoted = _promoted_fixture()
    with pytest.raises(ValueError, match="partitions overlap"):
        build_calibrated_behavior_bridge(
            records,
            rows,
            promoted,
            training_root_ids=[records[0]["root_id"]],
            t3_bb_likelihood_binding=_t3_binding(),
            challenge_records=challenge_records,
            challenge_evaluation_rows=challenge_rows,
        )


@pytest.mark.parametrize("field", ["temperatures", "role_model_bindings"])
def test_missing_or_extra_calibration_role_is_rejected_as_source_tamper(field):
    records, rows, challenge_records, challenge_rows, artifact = _promoted_fixture()
    for mutation in ("missing", "extra"):
        tampered = copy.deepcopy(artifact)
        if mutation == "missing":
            del tampered[field]["t2_btn"]
        else:
            tampered[field]["t3_bb"] = copy.deepcopy(tampered[field]["t1_bb"])
        tampered["artifact_sha256"] = canonical_sha256(
            {key: value for key, value in tampered.items() if key != "artifact_sha256"}
        )
        with pytest.raises(ValueError, match="does not match raw inputs"):
            build_calibrated_behavior_bridge(
                records,
                rows,
                tampered,
                training_root_ids=TRAINING_ROOTS,
                t3_bb_likelihood_binding=_t3_binding(),
                challenge_records=challenge_records,
                challenge_evaluation_rows=challenge_rows,
            )


def test_partition_builder_uses_identity_domain_and_rejects_duplicates():
    partition = build_root_partition_manifest(
        purpose="training", root_ids=["root-b", "root-a"]
    )
    assert partition["root_commitments"] == sorted(
        [
            root_identity_commitment_sha256("root-a"),
            root_identity_commitment_sha256("root-b"),
        ]
    )
    assert partition["manifest_sha256"] == self_hash(partition, "manifest_sha256")
    with pytest.raises(ValueError, match="duplicate"):
        build_root_partition_manifest(
            purpose="calibration", root_ids=["same-root", "same-root"]
        )
    with pytest.raises(TypeError, match="sequence of complete root ID"):
        build_root_partition_manifest(purpose="training", root_ids="one-root")
