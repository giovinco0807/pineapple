from __future__ import annotations

import copy
import gc
import hashlib
import inspect
import json
import shutil
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType

import pytest

import ai.tutor.calibrated_behavior_sharded_bootstrap as sharded_bootstrap
from ai.tutor.behavior_calibration_contract import canonical_json, canonical_sha256
from ai.tutor.behavior_logit_evaluator_torch import (
    build_known_hu_policy_value_logit_evaluators,
)
from ai.tutor.behavior_temperature_calibration import build_temperature_gate_config
from ai.tutor.behavior_temperature_calibration_shards import (
    build_sharded_behavior_temperature_calibration,
    write_sharded_behavior_temperature_calibration,
)
from ai.tutor.calibrated_behavior_bootstrap import (
    BOOTSTRAP_MODEL_TYPE,
    FixedPointBootstrapCalibratedHuT1T2BehaviorDispatch,
)
from ai.tutor.calibrated_behavior_runtime import EXPECTED_ROUTES, RUNTIME_SCHEMA
from ai.tutor.collect_hu_behavior_trace_shards import (
    collect_sharded_behavior_traces,
)
from ai.tutor.collect_hu_behavior_traces import (
    DETERMINISTIC_JOKER_CYCLE,
    NATURAL_UNIFORM_SHUFFLE,
    TARGETED_JOKER_CHALLENGE,
    BehaviorTraceCollectionConfig,
)
from ai.tutor.evaluate_hu_behavior_trace_shards import (
    evaluate_hu_behavior_trace_shards,
)
from ai.tutor.frozen_behavior_torch import (
    TurnActorBehaviorDispatch,
    build_known_hu_policy_value_prior_dispatch,
)
from ai.tutor.t3_bb_fixed_point_runtime import (
    FixedPointIterationBehaviorDispatch,
    build_fixed_point_iteration_behavior_dispatch,
    build_t3_bb_candidate_policy_artifact,
)
from ai.tutor.t3_hu_full_card_range import BehaviorDistribution
from test_behavior_temperature_calibration import _t1_bb_information
from test_behavior_temperature_calibration_shards import (
    CHALLENGE_ID,
    _collect_and_evaluate,
    _evaluators,
    _small_gate_config,
)


build_from_shards = getattr(
    sharded_bootstrap,
    "build_fixed_point_bootstrap_calibrated_hu_t1_t2_behavior_dispatch_from_shards",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@pytest.fixture(scope="module")
def sharded_evidence(tmp_path_factory):
    root = tmp_path_factory.mktemp("sharded-bootstrap")
    evaluators = _evaluators()
    paths = _collect_and_evaluate(root, evaluators, natural_roots=32)
    artifact = build_sharded_behavior_temperature_calibration(
        paths["natural_collection"],
        paths["natural_evaluation"],
        paths["challenge_collection"],
        paths["challenge_evaluation"],
        evaluators,
        gate_config=_small_gate_config(),
        scratch_dir=root / "build-scratch",
    )
    artifact_path = write_sharded_behavior_temperature_calibration(
        artifact, root / "calibration.json"
    )
    return root, paths, evaluators, artifact, artifact_path


class _FakeRuntimeChild:
    bindings = None

    def __init__(
        self,
        checkpoint_path,
        *,
        model_id,
        supported_turns,
        supported_actors,
        training_scope_id,
        temperature,
        quantization_denominator,
        expected_checkpoint_sha256,
    ):
        del checkpoint_path, training_scope_id
        turn = tuple(supported_turns)[0]
        actor = tuple(supported_actors)[0]
        role = f"t{turn}_{actor}"
        binding = self.bindings[role]
        self._model_id = model_id
        self._manifest = {
            "schema": RUNTIME_SCHEMA,
            "model_id": model_id,
            "model_type": "torch_policyvalue_boltzmann_q32",
            "position_contract_version": "bb_first_v1",
            "rules_version": sharded_bootstrap.RULES_VERSION,
            "checkpoint_sha256": expected_checkpoint_sha256,
            "adapter_source_sha256": binding["adapter_source_sha256"],
            "temperature": str(temperature),
            "quantization_denominator": quantization_denominator,
            "supported_turns": [turn],
            "supported_actors": [actor],
            "promotion_eligible": False,
        }

    @property
    def model_id(self):
        return self._model_id

    @property
    def model_manifest(self):
        return MappingProxyType(copy.deepcopy(self._manifest))

    @property
    def model_sha256(self):
        return canonical_sha256(self._manifest)

    def action_distribution(self, information):
        probability = Fraction(1, information.legal_action_count)
        return BehaviorDistribution(
            information_digest=information.digest(),
            probabilities=MappingProxyType(
                {
                    action_id: probability
                    for action_id in information.legal_action_ids
                }
            ),
            source="model",
            used_fallback=False,
        )


def _install_fake_runtime(monkeypatch, artifact, evaluators):
    assets = {}
    for turn, actor in EXPECTED_ROUTES:
        role = f"t{turn}_{actor}"
        assets[(turn, actor)] = {
            "relative_path": f"synthetic/{role}.pt",
            "checkpoint_sha256": artifact["role_model_bindings"][role][
                "checkpoint_sha256"
            ],
        }
    loader_calls = []

    def load(root, *, quantization_denominator):
        loader_calls.append((Path(root).resolve(), quantization_denominator))
        return evaluators

    _FakeRuntimeChild.bindings = artifact["role_model_bindings"]
    monkeypatch.setattr(sharded_bootstrap, "KNOWN_HU_POLICY_VALUE_ASSETS", assets)
    monkeypatch.setattr(
        sharded_bootstrap,
        "APPROVED_HU_POLICY_VALUE_CHECKPOINTS",
        {route: asset["checkpoint_sha256"] for route, asset in assets.items()},
    )
    monkeypatch.setattr(
        sharded_bootstrap,
        "build_known_hu_policy_value_logit_evaluators",
        load,
    )
    monkeypatch.setattr(
        sharded_bootstrap, "TorchPolicyValueBehaviorModel", _FakeRuntimeChild
    )
    return loader_calls


def _build_fake(monkeypatch, evidence, **kwargs):
    root, paths, evaluators, artifact, artifact_path = evidence
    loader_calls = _install_fake_runtime(monkeypatch, artifact, evaluators)
    model = build_from_shards(
        paths["natural_collection"],
        paths["natural_evaluation"],
        paths["challenge_collection"],
        paths["challenge_evaluation"],
        artifact_path,
        workspace_root=root,
        scratch_dir=root / "runtime-scratch",
        **kwargs,
    )
    return model, loader_calls


def test_sharded_runtime_freshly_binds_all_inputs_and_stays_nonpromotable(
    monkeypatch, sharded_evidence
):
    root, _paths, _evaluators_value, artifact, artifact_path = sharded_evidence
    model, loader_calls = _build_fake(monkeypatch, sharded_evidence)

    assert isinstance(model, FixedPointBootstrapCalibratedHuT1T2BehaviorDispatch)
    manifest = dict(model.model_manifest)
    assert manifest["schema"] == RUNTIME_SCHEMA
    assert manifest["model_type"] == BOOTSTRAP_MODEL_TYPE
    assert manifest["calibration_verified_from_raw"] is True
    assert manifest["calibration_verified_from_shards"] is True
    assert manifest["bounded_memory_verification"] is True
    assert manifest["raw_json_corpus_retained_in_memory"] is False
    assert manifest["source_calibration_promotion_eligible"] is artifact[
        "promotion_eligible"
    ]
    assert manifest["source_calibration_all_required_gates_passed"] is artifact[
        "gate_result"
    ]["all_required_gates_passed"]
    assert manifest["source_calibration_gate_failure_count"] == len(
        artifact["gate_result"]["failures"]
    )
    assert manifest["promotion_eligible"] is False
    assert manifest["fixed_point_bootstrap_only"] is True
    assert manifest["strategic_strength"] is False
    assert manifest["strategic_strength_evaluated"] is False
    assert manifest["strategic_strength_claimed"] is False
    assert manifest["no_fallback"] is True
    assert manifest["calibration_artifact_sha256"] == artifact["artifact_sha256"]
    assert manifest["calibration_artifact_file_sha256"] == _sha256_file(
        artifact_path
    )
    assert manifest["sharded_input_hash_binding_sha256"] == canonical_sha256(
        manifest["sharded_input_hash_bindings"]
    )
    for dataset in ("natural", "challenge"):
        expected = artifact["input_bindings"][dataset]
        bound = manifest["sharded_input_hash_bindings"][dataset]
        assert bound["collection"] == {
            "content_sha256": expected["collection"][
                "collection_content_sha256"
            ],
            "layout_sha256": expected["collection"]["layout_sha256"],
            "manifest_sha256": expected["collection"]["manifest_sha256"],
            "ordered_entry_chain_sha256": expected["collection"][
                "ordered_shard_entry_chain_sha256"
            ],
        }
        assert bound["evaluation"]["content_sha256"] == expected["evaluation"][
            "evaluation_content_sha256"
        ]
        assert bound["evaluation"]["layout_sha256"] == expected["evaluation"][
            "layout_sha256"
        ]
        assert bound["evaluation"]["manifest_sha256"] == expected["evaluation"][
            "manifest_sha256"
        ]
    assert [row["role"] for row in manifest["routes"]] == [
        "t1_bb",
        "t1_btn",
        "t2_bb",
        "t2_btn",
    ]
    assert loader_calls == [
        (root.resolve(), sharded_bootstrap.DEFAULT_QUANTIZATION_DENOMINATOR)
    ]
    distribution = model.action_distribution(_t1_bb_information())
    assert distribution.source == "model"
    assert distribution.used_fallback is False
    assert model.model_sha256 == canonical_sha256(dict(model.model_manifest))

    source = inspect.getsource(
        build_from_shards
    )
    assert "read_sharded_behavior_temperature_calibration" in source
    assert "read_text" not in source
    assert "read_bytes" not in source


def test_runtime_is_directly_compatible_with_fixed_point_iteration_dispatch(
    monkeypatch, sharded_evidence
):
    model, _calls = _build_fake(monkeypatch, sharded_evidence)
    candidate = build_t3_bb_candidate_policy_artifact(
        {"a" * 64: {"only-action": Fraction(1, 1)}},
        checkpoint_sha256="b" * 64,
        solver_manifest_sha256="c" * 64,
        range_builder_source_sha256="d" * 64,
    )
    iteration = build_fixed_point_iteration_behavior_dispatch(model, candidate)
    assert isinstance(iteration, FixedPointIterationBehaviorDispatch)
    assert iteration.model_manifest["promotion_eligible"] is False
    assert iteration.model_manifest["fixed_point_iteration_only"] is True
    assert (
        iteration.model_manifest["t1_t2_bootstrap_model_sha256"]
        == model.model_sha256
    )


def test_artifact_shard_and_input_path_drift_fail_before_runtime_children(
    monkeypatch, sharded_evidence, tmp_path
):
    root, paths, evaluators, artifact, artifact_path = sharded_evidence
    loader_calls = _install_fake_runtime(monkeypatch, artifact, evaluators)

    tampered_artifact = tmp_path / "tampered-calibration.json"
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    payload["artifact_sha256"] = "0" * 64
    tampered_artifact.write_bytes((canonical_json(payload) + "\n").encode("utf-8"))
    with pytest.raises(ValueError, match="artifact SHA-256 mismatch"):
        build_from_shards(
            paths["natural_collection"],
            paths["natural_evaluation"],
            paths["challenge_collection"],
            paths["challenge_evaluation"],
            tampered_artifact,
            workspace_root=root,
        )

    copied = {}
    for name, source_path in paths.items():
        destination = tmp_path / name
        shutil.copytree(source_path, destination)
        copied[name] = destination
    top = json.loads(
        (copied["natural_collection"] / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    decisions = (
        copied["natural_collection"]
        / top["shards"][0]["name"]
        / "decisions.jsonl"
    )
    decisions.write_bytes(decisions.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA-256 mismatch|canonical JSONL"):
        build_from_shards(
            copied["natural_collection"],
            copied["natural_evaluation"],
            copied["challenge_collection"],
            copied["challenge_evaluation"],
            artifact_path,
            workspace_root=root,
        )

    with pytest.raises(ValueError):
        build_from_shards(
            paths["natural_collection"],
            paths["challenge_evaluation"],
            paths["challenge_collection"],
            paths["natural_evaluation"],
            artifact_path,
            workspace_root=root,
        )
    # The evaluator/checkpoint loader ran, but no runtime child can be created
    # until the artifact and every supplied path have passed fresh verification.
    assert len(loader_calls) == 3


def test_nonpromoted_source_is_rejected_when_production_is_required(
    monkeypatch, sharded_evidence
):
    root, paths, evaluators, _artifact, _artifact_path = sharded_evidence
    strict_config = build_temperature_gate_config(
        gate_id="test-sharded-bootstrap-deliberately-nonpromoted-v1",
        min_fit_decisions_per_role=10_000,
        min_dev_decisions_per_role=10_000,
        min_test_decisions_per_role=10_000,
        min_challenge_decisions_per_role_joker=10_000,
        min_roots_per_split_role=10_000,
        min_challenge_roots_per_role_joker=10_000,
        challenge_root_namespace_prefix=f"{CHALLENGE_ID}/",
        bootstrap_replicates=20,
        bootstrap_seed="test-sharded-bootstrap-nonpromoted-v1",
        ece_bins=5,
    )
    artifact = build_sharded_behavior_temperature_calibration(
        paths["natural_collection"],
        paths["natural_evaluation"],
        paths["challenge_collection"],
        paths["challenge_evaluation"],
        evaluators,
        gate_config=strict_config,
        scratch_dir=root / "strict-scratch",
    )
    assert artifact["promotion_eligible"] is False
    artifact_path = write_sharded_behavior_temperature_calibration(
        artifact, root / "nonpromoted-calibration.json"
    )
    loader_calls = _install_fake_runtime(monkeypatch, artifact, evaluators)
    with pytest.raises(ValueError, match="production sharded bootstrap requires"):
        build_from_shards(
            paths["natural_collection"],
            paths["natural_evaluation"],
            paths["challenge_collection"],
            paths["challenge_evaluation"],
            artifact_path,
            workspace_root=root,
            require_promoted_source=True,
            scratch_dir=root / "strict-runtime-scratch",
        )
    assert len(loader_calls) == 1


def test_real_four_checkpoint_small_end_to_end(tmp_path):
    workspace_root = Path.cwd().resolve()
    checkpoint_paths = [
        workspace_root / "ai/data/t1_hu_bb_gcp_2m_model/t2_policyvalue_model.pt",
        workspace_root / "ai/data/t1_hu_btn_gcp_2m_model/t2_policyvalue_model.pt",
        workspace_root / "ai/data/t2_hu_bb_gcp_2m_model/t2_policyvalue_model.pt",
        workspace_root / "ai/data/t2_hu_btn_gcp_2m_model/t2_policyvalue_model.pt",
    ]
    if not all(path.is_file() for path in checkpoint_paths):
        pytest.skip("all four known HU policy-value checkpoints are required")

    natural_collection = tmp_path / "real-natural-collection"
    natural_evaluation = tmp_path / "real-natural-evaluation"
    challenge_collection = tmp_path / "real-challenge-collection"
    challenge_evaluation = tmp_path / "real-challenge-evaluation"
    dispatch = build_known_hu_policy_value_prior_dispatch(workspace_root)
    collect_sharded_behavior_traces(
        natural_collection,
        BehaviorTraceCollectionConfig(
            seed_namespace="real-sharded-bootstrap-natural-v1",
            root_sampling_mode=NATURAL_UNIFORM_SHUFFLE,
        ),
        dispatch,
        total_root_target=1,
        shard_size=1,
    )
    collect_sharded_behavior_traces(
        challenge_collection,
        BehaviorTraceCollectionConfig(
            seed_namespace="real-sharded-bootstrap-challenge-v1",
            root_sampling_mode=TARGETED_JOKER_CHALLENGE,
            challenge_id=CHALLENGE_ID,
            challenge_deck_source=DETERMINISTIC_JOKER_CYCLE,
        ),
        dispatch,
        total_root_target=12,
        shard_size=6,
    )
    del dispatch
    gc.collect()
    evaluators = build_known_hu_policy_value_logit_evaluators(workspace_root)
    evaluate_hu_behavior_trace_shards(
        natural_collection, natural_evaluation, evaluators
    )
    evaluate_hu_behavior_trace_shards(
        challenge_collection, challenge_evaluation, evaluators
    )
    artifact = build_sharded_behavior_temperature_calibration(
        natural_collection,
        natural_evaluation,
        challenge_collection,
        challenge_evaluation,
        evaluators,
        gate_config=_small_gate_config(),
        scratch_dir=tmp_path / "calibration-scratch",
    )
    artifact_path = write_sharded_behavior_temperature_calibration(
        artifact, tmp_path / "calibration.json"
    )
    del evaluators
    gc.collect()

    model = build_from_shards(
        natural_collection,
        natural_evaluation,
        challenge_collection,
        challenge_evaluation,
        artifact_path,
        workspace_root=workspace_root,
        scratch_dir=tmp_path / "runtime-scratch",
    )
    manifest = model.model_manifest
    assert isinstance(model, FixedPointBootstrapCalibratedHuT1T2BehaviorDispatch)
    assert manifest["promotion_eligible"] is False
    assert manifest["fixed_point_bootstrap_only"] is True
    assert len(manifest["routes"]) == 4
    expected_checkpoint_by_role = {
        role: _sha256_file(path)
        for role, path in zip(
            ("t1_bb", "t1_btn", "t2_bb", "t2_btn"),
            checkpoint_paths,
            strict=True,
        )
    }
    assert {
        row["role"]: row["checkpoint_sha256"] for row in manifest["routes"]
    } == expected_checkpoint_by_role
    distribution = model.action_distribution(_t1_bb_information())
    assert distribution.used_fallback is False
    assert sum(distribution.probabilities.values(), Fraction(0, 1)) == 1
