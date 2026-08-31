import copy
import gc
import json
import shutil
from pathlib import Path

import pytest

from ai.tutor.behavior_calibration_contract import canonical_json, canonical_sha256
from ai.tutor.behavior_logit_evaluator_torch import (
    EVALUATOR_VERSION,
    build_known_hu_policy_value_logit_evaluators,
)
from ai.tutor.behavior_temperature_calibration import (
    build_behavior_temperature_calibration,
    build_temperature_gate_config,
)
from ai.tutor.behavior_temperature_calibration_shards import (
    SHARDED_CALIBRATION_SCHEMA,
    build_sharded_behavior_temperature_calibration,
    read_sharded_behavior_temperature_calibration,
    verify_sharded_behavior_temperature_calibration,
    write_sharded_behavior_temperature_calibration,
)
from ai.tutor.collect_hu_behavior_trace_shards import (
    collect_sharded_behavior_traces,
)
from ai.tutor.collect_hu_behavior_traces import (
    DETERMINISTIC_JOKER_CYCLE,
    NATURAL_UNIFORM_SHUFFLE,
    TARGETED_JOKER_CHALLENGE,
    BehaviorTraceCollectionConfig,
    read_behavior_trace_dataset,
)
from ai.tutor.evaluate_hu_behavior_trace_shards import (
    EXPECTED_INPUT_POLICY_ID,
    evaluate_hu_behavior_trace_shards,
)
from ai.tutor.frozen_behavior_torch import (
    TurnActorBehaviorDispatch,
    build_known_hu_policy_value_prior_dispatch,
)
from ai.tutor.t3_hu_full_card_range import UniformLegalBehaviorModel


ROUTES = ((1, "bb"), (1, "btn"), (2, "bb"), (2, "btn"))
CHALLENGE_ID = "m3-joker-challenge-v1"
CHALLENGE_PREFIX = f"{CHALLENGE_ID}/"


class _FakeDirectLogitEvaluator:
    def __init__(self, turn, actor, *, model_salt="v1", evaluator_salt="v1"):
        self.turn = turn
        self.actor = actor
        self._behavior = UniformLegalBehaviorModel(
            model_id=f"sharded-calibration-t{turn}-{actor}-{model_salt}"
        )
        route = {"turn": turn, "actor": actor, "salt": evaluator_salt}
        self.checkpoint_sha256 = canonical_sha256(
            {"kind": "checkpoint", **route}
        )
        self.row_extractor_sha256 = canonical_sha256(
            {"kind": "extractor", **route}
        )
        self.adapter_source_sha256 = canonical_sha256(
            {"kind": "adapter", **route}
        )

    @property
    def model_id(self):
        return self._behavior.model_id

    @property
    def model_manifest(self):
        return self._behavior.model_manifest

    @property
    def model_sha256(self):
        return self._behavior.model_sha256

    @property
    def evaluator_manifest(self):
        return {
            "schema": "ofc_behavior_logit_evaluator/v1",
            "evaluator_version": EVALUATOR_VERSION,
            "checkpoint_sha256": self.checkpoint_sha256,
            "model_sha256": self.model_sha256,
            "row_extractor_sha256": self.row_extractor_sha256,
            "adapter_source_sha256": self.adapter_source_sha256,
            "temperature": "1/1",
            "device": "cpu",
            "logit_contract": (
                "checkpoint_policy_head_pre_temperature_pre_quantization"
            ),
        }

    def action_distribution(self, information):
        return self._behavior.action_distribution(information)

    def pre_temperature_legal_logits(self, information, legal_action_ids):
        assert legal_action_ids == information.legal_action_ids
        offset = self.turn + (1 if self.actor == "btn" else 0)
        return [
            float(index - offset) / 8.0 for index in range(len(legal_action_ids))
        ]


def _evaluators(*, model_salt="v1", evaluator_salt="v1"):
    return {
        route: _FakeDirectLogitEvaluator(
            *route, model_salt=model_salt, evaluator_salt=evaluator_salt
        )
        for route in ROUTES
    }


def _small_gate_config():
    return build_temperature_gate_config(
        gate_id="test-sharded-temperature-v1",
        min_fit_decisions_per_role=1,
        min_dev_decisions_per_role=1,
        min_test_decisions_per_role=1,
        min_challenge_decisions_per_role_joker=1,
        min_roots_per_split_role=1,
        min_challenge_roots_per_role_joker=1,
        challenge_root_namespace_prefix=CHALLENGE_PREFIX,
        bootstrap_replicates=20,
        bootstrap_seed="test-sharded-temperature-bootstrap-v1",
        ece_bins=5,
    )


def _collect_and_evaluate(root: Path, evaluators, *, natural_roots=32):
    dispatch = TurnActorBehaviorDispatch(
        evaluators, model_id=EXPECTED_INPUT_POLICY_ID
    )
    natural_collection = root / "natural-collection"
    natural_evaluation = root / "natural-evaluation"
    challenge_collection = root / "challenge-collection"
    challenge_evaluation = root / "challenge-evaluation"
    collect_sharded_behavior_traces(
        natural_collection,
        BehaviorTraceCollectionConfig(
            seed_namespace="sharded-calibration-natural-test-v1",
            root_sampling_mode=NATURAL_UNIFORM_SHUFFLE,
        ),
        dispatch,
        total_root_target=natural_roots,
        shard_size=7,
    )
    collect_sharded_behavior_traces(
        challenge_collection,
        BehaviorTraceCollectionConfig(
            seed_namespace="sharded-calibration-challenge-test-v1",
            root_sampling_mode=TARGETED_JOKER_CHALLENGE,
            challenge_id=CHALLENGE_ID,
            challenge_deck_source=DETERMINISTIC_JOKER_CYCLE,
        ),
        dispatch,
        total_root_target=12,
        shard_size=5,
    )
    evaluate_hu_behavior_trace_shards(
        natural_collection, natural_evaluation, evaluators
    )
    evaluate_hu_behavior_trace_shards(
        challenge_collection, challenge_evaluation, evaluators
    )
    return {
        "natural_collection": natural_collection,
        "natural_evaluation": natural_evaluation,
        "challenge_collection": challenge_collection,
        "challenge_evaluation": challenge_evaluation,
    }


@pytest.fixture(scope="module")
def sharded_fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("sharded-temperature")
    evaluators = _evaluators()
    paths = _collect_and_evaluate(root, evaluators)
    artifact = build_sharded_behavior_temperature_calibration(
        paths["natural_collection"],
        paths["natural_evaluation"],
        paths["challenge_collection"],
        paths["challenge_evaluation"],
        evaluators,
        gate_config=_small_gate_config(),
        scratch_dir=root / "scratch",
    )
    return root, paths, evaluators, artifact


def _flat_rows(collection_dir: Path, evaluation_dir: Path):
    top = json.loads((collection_dir / "manifest.json").read_text(encoding="utf-8"))
    evaluation_top = json.loads(
        (evaluation_dir / "manifest.json").read_text(encoding="utf-8")
    )
    records = []
    rows = []
    for entry in top["shards"]:
        dataset = read_behavior_trace_dataset(
            collection_dir / entry["name"] / "decisions.jsonl"
        )
        records.extend(dataset.records)
    for entry in evaluation_top["shards"]:
        path = evaluation_dir / entry["name"] / "evaluations.jsonl"
        rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    return records, rows


def _copied_paths(base_paths, target: Path):
    result = {}
    for key, source in base_paths.items():
        destination = target / key
        shutil.copytree(source, destination)
        result[key] = destination
    return result


def test_sharded_builder_is_exactly_v2_for_temperatures_metrics_and_gate(
    sharded_fixture,
):
    _root, paths, _evaluators_value, artifact = sharded_fixture
    natural_records, natural_rows = _flat_rows(
        paths["natural_collection"], paths["natural_evaluation"]
    )
    challenge_records, challenge_rows = _flat_rows(
        paths["challenge_collection"], paths["challenge_evaluation"]
    )
    legacy = build_behavior_temperature_calibration(
        natural_records,
        natural_rows,
        challenge_records=challenge_records,
        challenge_evaluation_rows=challenge_rows,
        gate_config=_small_gate_config(),
    )

    assert artifact["schema"] == SHARDED_CALIBRATION_SCHEMA
    assert artifact["temperatures"] == legacy["temperatures"]
    assert artifact["metrics"] == legacy["metrics"]
    assert artifact["gate_result"] == legacy["gate_result"]
    assert artifact["promotion_eligible"] is artifact["gate_result"][
        "promotion_eligible"
    ]
    assert artifact["input_bindings"]["natural"]["collection"][
        "collection_content_sha256"
    ]
    assert artifact["input_bindings"]["challenge"]["evaluation"][
        "evaluator_set_sha256"
    ]


def test_canonical_save_and_full_fresh_readback(sharded_fixture):
    root, paths, evaluators, artifact = sharded_fixture
    output = root / "published" / "calibration.json"
    assert write_sharded_behavior_temperature_calibration(artifact, output) == output.resolve()
    assert output.read_bytes().endswith(b"\n")
    rebuilt = read_sharded_behavior_temperature_calibration(
        output,
        paths["natural_collection"],
        paths["natural_evaluation"],
        paths["challenge_collection"],
        paths["challenge_evaluation"],
        evaluators,
        scratch_dir=root / "readback-scratch",
    )
    assert rebuilt == artifact
    assert not list((root / "readback-scratch").iterdir())


def test_promotion_cannot_be_self_declared(sharded_fixture):
    _root, paths, evaluators, artifact = sharded_fixture
    bad = copy.deepcopy(artifact)
    bad["promotion_eligible"] = not bad["gate_result"]["promotion_eligible"]
    unsigned = dict(bad)
    unsigned.pop("artifact_sha256")
    bad["artifact_sha256"] = canonical_sha256(unsigned)
    with pytest.raises(ValueError, match="derived solely from the gate"):
        verify_sharded_behavior_temperature_calibration(
            paths["natural_collection"],
            paths["natural_evaluation"],
            paths["challenge_collection"],
            paths["challenge_evaluation"],
            evaluators,
            bad,
        )


def test_evaluation_tamper_and_evaluator_drift_fail_closed(
    sharded_fixture, tmp_path
):
    _root, base_paths, evaluators, _artifact = sharded_fixture
    paths = _copied_paths(base_paths, tmp_path / "copy")
    top = json.loads(
        (paths["natural_evaluation"] / "manifest.json").read_text(encoding="utf-8")
    )
    row_path = (
        paths["natural_evaluation"]
        / top["shards"][0]["name"]
        / "evaluations.jsonl"
    )
    rows = [json.loads(line) for line in row_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["logits_f64_hex"][0] = float(123.0).hex()
    row_path.write_bytes(
        ("\n".join(canonical_json(row) for row in rows) + "\n").encode("utf-8")
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        build_sharded_behavior_temperature_calibration(
            paths["natural_collection"],
            paths["natural_evaluation"],
            paths["challenge_collection"],
            paths["challenge_evaluation"],
            evaluators,
            gate_config=_small_gate_config(),
        )

    with pytest.raises(ValueError, match="evaluator/checkpoint binding drift"):
        build_sharded_behavior_temperature_calibration(
            base_paths["natural_collection"],
            base_paths["natural_evaluation"],
            base_paths["challenge_collection"],
            base_paths["challenge_evaluation"],
            _evaluators(evaluator_salt="drift"),
            gate_config=_small_gate_config(),
        )


def test_gap_orphan_and_immutable_output_fail_closed(sharded_fixture, tmp_path):
    _root, base_paths, evaluators, artifact = sharded_fixture
    paths = _copied_paths(base_paths, tmp_path / "copy")
    top = json.loads(
        (paths["challenge_evaluation"] / "manifest.json").read_text(encoding="utf-8")
    )
    removed = paths["challenge_evaluation"] / top["shards"][1]["name"]
    shutil.rmtree(removed)
    with pytest.raises((FileNotFoundError, ValueError)):
        build_sharded_behavior_temperature_calibration(
            paths["natural_collection"],
            paths["natural_evaluation"],
            paths["challenge_collection"],
            paths["challenge_evaluation"],
            evaluators,
            gate_config=_small_gate_config(),
        )

    orphan_paths = _copied_paths(base_paths, tmp_path / "orphan-copy")
    source = orphan_paths["natural_evaluation"] / "evaluation-shard-000000"
    orphan = orphan_paths["natural_evaluation"] / "evaluation-shard-999999"
    shutil.copytree(source, orphan)
    with pytest.raises(ValueError, match="orphan"):
        build_sharded_behavior_temperature_calibration(
            orphan_paths["natural_collection"],
            orphan_paths["natural_evaluation"],
            orphan_paths["challenge_collection"],
            orphan_paths["challenge_evaluation"],
            evaluators,
            gate_config=_small_gate_config(),
        )

    output = tmp_path / "immutable.json"
    write_sharded_behavior_temperature_calibration(artifact, output)
    different = copy.deepcopy(artifact)
    different["computation_contract"]["combined_jsonl_present"] = True
    unsigned = dict(different)
    unsigned.pop("artifact_sha256")
    different["artifact_sha256"] = canonical_sha256(unsigned)
    with pytest.raises(FileExistsError, match="immutable"):
        write_sharded_behavior_temperature_calibration(different, output)


def test_incomplete_evaluation_is_never_accepted(sharded_fixture, tmp_path):
    _root, base_paths, evaluators, _artifact = sharded_fixture
    natural_collection = tmp_path / "natural-collection"
    shutil.copytree(base_paths["natural_collection"], natural_collection)
    partial_evaluation = tmp_path / "partial-natural-evaluation"
    partial = evaluate_hu_behavior_trace_shards(
        natural_collection,
        partial_evaluation,
        evaluators,
        max_new_shards=1,
    )
    assert partial.evaluation_complete is False
    with pytest.raises(ValueError, match="must cover the complete collection"):
        build_sharded_behavior_temperature_calibration(
            natural_collection,
            partial_evaluation,
            base_paths["challenge_collection"],
            base_paths["challenge_evaluation"],
            evaluators,
            gate_config=_small_gate_config(),
        )


def test_real_four_checkpoint_small_e2e(tmp_path):
    checkpoint_paths = [
        Path("ai/data/t1_hu_bb_gcp_2m_model/t2_policyvalue_model.pt"),
        Path("ai/data/t1_hu_btn_gcp_2m_model/t2_policyvalue_model.pt"),
        Path("ai/data/t2_hu_bb_gcp_2m_model/t2_policyvalue_model.pt"),
        Path("ai/data/t2_hu_btn_gcp_2m_model/t2_policyvalue_model.pt"),
    ]
    if not all(path.is_file() for path in checkpoint_paths):
        pytest.skip("all four known HU policy-value checkpoints are required")

    natural_collection = tmp_path / "real-natural-collection"
    natural_evaluation = tmp_path / "real-natural-evaluation"
    challenge_collection = tmp_path / "real-challenge-collection"
    challenge_evaluation = tmp_path / "real-challenge-evaluation"
    dispatch = build_known_hu_policy_value_prior_dispatch(Path.cwd())
    collect_sharded_behavior_traces(
        natural_collection,
        BehaviorTraceCollectionConfig(
            seed_namespace="real-sharded-calibration-natural-v1",
            root_sampling_mode=NATURAL_UNIFORM_SHUFFLE,
        ),
        dispatch,
        total_root_target=1,
        shard_size=1,
    )
    collect_sharded_behavior_traces(
        challenge_collection,
        BehaviorTraceCollectionConfig(
            seed_namespace="real-sharded-calibration-challenge-v1",
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
    evaluators = build_known_hu_policy_value_logit_evaluators(Path.cwd())
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
        scratch_dir=tmp_path / "scratch",
    )
    assert artifact["schema"] == SHARDED_CALIBRATION_SCHEMA
    assert len(artifact["role_model_bindings"]) == 4
    assert all(
        binding is not None and len(binding["checkpoint_sha256"]) == 64
        for binding in artifact["role_model_bindings"].values()
    )
    assert artifact["input_bindings"]["natural"]["evaluation"]["row_count"] == 4
    assert artifact["input_bindings"]["challenge"]["evaluation"]["row_count"] == 48
