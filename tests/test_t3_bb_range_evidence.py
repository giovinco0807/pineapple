from __future__ import annotations

import copy
from fractions import Fraction

import pytest

from ai.tutor.promotion_gate_m3_full_card_strength import canonical_sha256
from ai.tutor.t3_bb_range_evidence import (
    build_restricted_range_evidence,
    read_restricted_range_evidence,
    verify_restricted_range_evidence,
    write_restricted_range_evidence,
)
from ai.tutor.t3_hu_full_card_range import (
    UniformLegalBehaviorModel,
    build_history_weighted_full_card_range,
)
from ai.tutor.t3_hu_reduced_fixtures import compile_canonical_reduced_fixture


def _fixture():
    compiled = compile_canonical_reduced_fixture("btn", 1)
    observation = compiled.root.branches[0].child.infoset_key
    full_range = build_history_weighted_full_card_range(
        observation,
        UniformLegalBehaviorModel(),
        epsilon=Fraction(0, 1),
        max_particles=2,
        seed=20260713,
    )
    artifact = build_restricted_range_evidence(
        observation,
        full_range,
        root_id="restricted-range-fixture-btn-joker1",
        root_commitment_sha256=canonical_sha256({"fixture": "root"}),
        round_index=1,
        solver_seed=73,
    )
    return observation, full_range, artifact


def _resign(artifact):
    unsigned = copy.deepcopy(artifact)
    unsigned.pop("artifact_sha256", None)
    artifact["artifact_sha256"] = canonical_sha256(unsigned)


def test_restricted_range_artifact_replays_physical_posterior_and_hashes(tmp_path):
    observation, full_range, artifact = _fixture()
    verified, audit = verify_restricted_range_evidence(artifact)

    assert verified == artifact
    assert audit["verified"] is True
    assert audit["observation_digest"] == observation.digest()
    assert audit["range_content_sha256"] == full_range.range_content_sha256
    assert audit["range_build_sha256"] == full_range.range_build_sha256
    assert audit["particle_count"] == 2
    assert sum(
        (Fraction(value) for value in audit["posterior_weights"].values()),
        Fraction(0, 1),
    ) == 1

    path = write_restricted_range_evidence(tmp_path / "range.json", artifact)
    reread, reread_audit = read_restricted_range_evidence(path)
    assert reread == artifact
    assert reread_audit == audit


def test_rehashed_posterior_weight_tamper_fails_full_replay():
    _observation, _full_range, artifact = _fixture()
    tampered = copy.deepcopy(artifact)
    tampered["particles"][0]["weight"] = "1/1"
    _resign(tampered)
    with pytest.raises(ValueError, match="weights do not sum exactly to one"):
        verify_restricted_range_evidence(tampered)


def test_rehashed_hidden_particle_tamper_fails_commitment_or_partition():
    _observation, _full_range, artifact = _fixture()
    tampered = copy.deepcopy(artifact)
    tampered["particles"][0]["undealt_cards"][0] = tampered["particles"][0][
        "undealt_cards"
    ][1]
    _resign(tampered)
    with pytest.raises(ValueError):
        verify_restricted_range_evidence(tampered)


def test_observation_and_manifest_tamper_fail_even_when_resigned():
    _observation, _full_range, artifact = _fixture()

    changed_observation = copy.deepcopy(artifact)
    changed_observation["observation"]["current_draw"] = list(
        reversed(changed_observation["observation"]["current_draw"])
    )
    _resign(changed_observation)
    with pytest.raises(ValueError, match="not canonical"):
        verify_restricted_range_evidence(changed_observation)

    changed_manifest = copy.deepcopy(artifact)
    changed_manifest["build_manifest"]["behavior_query_count"] += 1
    _resign(changed_manifest)
    with pytest.raises(ValueError, match="range build manifest hash mismatch"):
        verify_restricted_range_evidence(changed_manifest)
