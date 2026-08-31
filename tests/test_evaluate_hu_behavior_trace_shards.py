import gc
import json
import shutil
from pathlib import Path

import pytest

import ai.tutor.collect_hu_behavior_trace_shards as input_shards_module
import ai.tutor.evaluate_hu_behavior_trace_shards as shard_evaluator_module
from ai.tutor.behavior_calibration_contract import canonical_json, canonical_sha256
from ai.tutor.behavior_logit_evaluator_torch import (
    EVALUATOR_VERSION,
    build_known_hu_policy_value_logit_evaluators,
)
from ai.tutor.collect_hu_behavior_trace_shards import (
    collect_sharded_behavior_traces,
)
from ai.tutor.collect_hu_behavior_traces import (
    NATURAL_UNIFORM_SHUFFLE,
    BehaviorTraceCollectionConfig,
)
from ai.tutor.evaluate_hu_behavior_trace_shards import (
    EXPECTED_INPUT_POLICY_ID,
    _StreamAggregate,
    _build_top_manifest,
    _evaluator_bindings,
    _input_collection_binding,
    _source_hashes,
    evaluate_hu_behavior_trace_shards,
    read_sharded_behavior_trace_evaluation,
    resume_hu_behavior_trace_shard_evaluation,
)
from ai.tutor.frozen_behavior_torch import (
    TurnActorBehaviorDispatch,
    build_known_hu_policy_value_prior_dispatch,
)
from ai.tutor.t3_hu_full_card_range import UniformLegalBehaviorModel


ROUTES = ((1, "bb"), (1, "btn"), (2, "bb"), (2, "btn"))


class _FakeDirectLogitEvaluator:
    def __init__(self, turn, actor, *, model_salt="v1", evaluator_salt="v1"):
        self.turn = turn
        self.actor = actor
        self._behavior = UniformLegalBehaviorModel(
            model_id=f"test-t{turn}-{actor}-{model_salt}"
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
        return [float(index - offset) / 8.0 for index in range(len(legal_action_ids))]


def _evaluators(*, model_salt="v1", evaluator_salt="v1"):
    return {
        route: _FakeDirectLogitEvaluator(
            *route, model_salt=model_salt, evaluator_salt=evaluator_salt
        )
        for route in ROUTES
    }


def _fixture(tmp_path, *, roots=3, shard_size=2, namespace="eval-shards/v1"):
    evaluators = _evaluators()
    dispatch = TurnActorBehaviorDispatch(
        evaluators, model_id=EXPECTED_INPUT_POLICY_ID
    )
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "evaluation"
    config = BehaviorTraceCollectionConfig(
        seed_namespace=namespace,
        root_sampling_mode=NATURAL_UNIFORM_SHUFFLE,
    )
    collection = collect_sharded_behavior_traces(
        input_dir,
        config,
        dispatch,
        total_root_target=roots,
        shard_size=shard_size,
    )
    return collection, input_dir, output_dir, evaluators


def _published_bytes(output_dir):
    return {
        str(path.relative_to(output_dir)): path.read_bytes()
        for path in output_dir.rglob("*")
        if path.is_file() and "evaluation-shard-" in str(path.parent)
    }


def test_one_to_one_append_resume_and_complete_census(tmp_path):
    collection, input_dir, output_dir, evaluators = _fixture(tmp_path)

    first = evaluate_hu_behavior_trace_shards(
        input_dir, output_dir, evaluators, max_new_shards=1
    )
    assert first.shard_count == 1
    assert first.row_count == 8
    assert first.evaluation_complete is False
    before = _published_bytes(output_dir)

    completed = resume_hu_behavior_trace_shard_evaluation(
        input_dir, output_dir, evaluators
    )
    assert completed.shard_count == collection.shard_count == 2
    assert completed.row_count == collection.decision_count == 12
    assert completed.evaluation_complete is True
    assert completed.manifest["promotion_eligible"] is False
    assert completed.manifest["raw_evaluation_only"] is True
    assert completed.manifest["input_collection"]["collection_content_sha256"] == (
        collection.manifest["collection_content_sha256"]
    )
    assert completed.manifest["input_collection"]["layout_sha256"] == (
        collection.manifest["layout_sha256"]
    )
    assert [
        row["entry_sha256"]
        for row in completed.manifest["input_collection"]["shards"]
    ] == [row["entry_sha256"] for row in collection.manifest["shards"]]
    assert len(completed.manifest["evaluator_routes"]) == 4
    assert all(
        len(route["evaluator_manifest_sha256"]) == 64
        and len(route["checkpoint_sha256"]) == 64
        and len(route["model_sha256"]) == 64
        and len(route["row_extractor_sha256"]) == 64
        and len(route["adapter_source_sha256"]) == 64
        for route in completed.manifest["evaluator_routes"]
    )
    assert completed.manifest["census"]["row_counts_by_role"] == {
        "t1_bb": 3,
        "t1_btn": 3,
        "t2_bb": 3,
        "t2_btn": 3,
    }
    assert sum(
        completed.manifest["census"]["row_counts_by_split"].values()
    ) == 12
    assert sum(
        completed.manifest["census"]["row_counts_by_joker"].values()
    ) == 12
    assert set(before).issubset(_published_bytes(output_dir))
    for name, payload in before.items():
        assert _published_bytes(output_dir)[name] == payload
    assert not (output_dir / "evaluations.jsonl").exists()
    assert not (output_dir / "decisions.jsonl").exists()

    before_no_op = _published_bytes(output_dir)
    no_op = resume_hu_behavior_trace_shard_evaluation(
        input_dir, output_dir, evaluators
    )
    assert no_op.manifest == completed.manifest
    assert _published_bytes(output_dir) == before_no_op


def test_each_row_is_direct_logit_bound_and_shard_manifest_is_last_marker(tmp_path):
    collection, input_dir, output_dir, evaluators = _fixture(
        tmp_path, roots=2, shard_size=2, namespace="eval-shards/rows-v1"
    )
    result = evaluate_hu_behavior_trace_shards(input_dir, output_dir, evaluators)
    entry = result.manifest["shards"][0]
    shard_dir = output_dir / entry["name"]
    rows = [
        json.loads(line)
        for line in (shard_dir / "evaluations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert len(rows) == collection.decision_count == 8
    assert all(row["promotion_eligible"] is False for row in rows)
    assert all(
        row["logit_contract"]
        == "frozen_checkpoint_pre_temperature_selected_legal_logits_float64"
        for row in rows
    )
    assert all(
        all(isinstance(value, str) and value.startswith(("0x", "-0x")) for value in row["logits_f64_hex"])
        for row in rows
    )
    shard_manifest = json.loads(
        (shard_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert shard_manifest["raw_evaluation_only"] is True
    assert shard_manifest["promotion_eligible"] is False
    assert shard_manifest["input_shard"]["entry_sha256"] == (
        collection.manifest["shards"][0]["entry_sha256"]
    )
    assert shard_manifest["evaluations_artifact"]["file_sha256"] == entry[
        "evaluations_file_sha256"
    ]


def test_tampered_evaluation_row_blocks_read_and_resume(tmp_path):
    _, input_dir, output_dir, evaluators = _fixture(
        tmp_path, namespace="eval-shards/tamper-v1"
    )
    result = evaluate_hu_behavior_trace_shards(input_dir, output_dir, evaluators)
    path = output_dir / result.manifest["shards"][0]["name"] / "evaluations.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["visible_joker_count"] = (rows[0]["visible_joker_count"] + 1) % 3
    path.write_bytes(
        ("\n".join(canonical_json(row) for row in rows) + "\n").encode("utf-8")
    )

    with pytest.raises(ValueError, match="evaluation row SHA-256 mismatch"):
        read_sharded_behavior_trace_evaluation(input_dir, output_dir, evaluators)
    with pytest.raises(ValueError, match="evaluation row SHA-256 mismatch"):
        resume_hu_behavior_trace_shard_evaluation(input_dir, output_dir, evaluators)


def test_evaluator_and_input_config_drift_fail_closed(tmp_path):
    _, input_dir, output_dir, evaluators = _fixture(
        tmp_path, namespace="eval-shards/drift-v1"
    )
    evaluate_hu_behavior_trace_shards(input_dir, output_dir, evaluators)

    checkpoint_drift = _evaluators(evaluator_salt="changed-checkpoint")
    with pytest.raises(ValueError, match="evaluator/checkpoint binding drift"):
        read_sharded_behavior_trace_evaluation(
            input_dir, output_dir, checkpoint_drift
        )

    _, other_input, _, _ = _fixture(
        tmp_path / "other",
        namespace="eval-shards/different-collection-config-v1",
    )
    with pytest.raises(ValueError, match="input collection binding drift"):
        read_sharded_behavior_trace_evaluation(
            other_input, output_dir, evaluators
        )


def test_orphan_gap_and_partial_staging_contracts(tmp_path):
    collection, input_dir, output_dir, evaluators = _fixture(
        tmp_path, namespace="eval-shards/orphan-v1"
    )
    result = evaluate_hu_behavior_trace_shards(input_dir, output_dir, evaluators)

    partial = output_dir / ".evaluation-shard-000002.crash.partial"
    partial.mkdir()
    (partial / "evaluations.jsonl").write_text("truncated", encoding="utf-8")
    assert read_sharded_behavior_trace_evaluation(
        input_dir, output_dir, evaluators
    ).evaluation_complete

    orphan = output_dir / "evaluation-shard-000002"
    shutil.copytree(output_dir / "evaluation-shard-000000", orphan)
    with pytest.raises(ValueError, match="orphan"):
        read_sharded_behavior_trace_evaluation(input_dir, output_dir, evaluators)
    shutil.rmtree(orphan)

    evaluator_bindings, evaluator_set_sha, _ = _evaluator_bindings(evaluators)
    bad_top = _build_top_manifest(
        input_binding=_input_collection_binding(collection),
        evaluator_bindings=evaluator_bindings,
        evaluator_set_sha256=evaluator_set_sha,
        source_hashes=_source_hashes(),
        entries=[result.manifest["shards"][1]],
        aggregate=_StreamAggregate.from_top_manifest(result.manifest),
    )
    (output_dir / "manifest.json").write_bytes(
        (canonical_json(bad_top) + "\n").encode("utf-8")
    )
    with pytest.raises(ValueError, match="gap, duplicate, or out-of-order"):
        read_sharded_behavior_trace_evaluation(input_dir, output_dir, evaluators)


def test_source_drift_is_checked_before_resume(tmp_path, monkeypatch):
    _, input_dir, output_dir, evaluators = _fixture(
        tmp_path, roots=1, shard_size=1, namespace="eval-shards/source-v1"
    )
    evaluate_hu_behavior_trace_shards(input_dir, output_dir, evaluators)
    original = _source_hashes()
    changed = dict(original)
    changed["sharded_evaluation_source_sha256"] = "f" * 64
    monkeypatch.setattr(shard_evaluator_module, "_source_hashes", lambda: changed)

    with pytest.raises(ValueError, match="source hash binding mismatch"):
        resume_hu_behavior_trace_shard_evaluation(input_dir, output_dir, evaluators)


def test_growing_input_collection_is_explicitly_rejected(tmp_path):
    collection, input_dir, output_dir, evaluators = _fixture(
        tmp_path, roots=1, shard_size=1, namespace="eval-shards/growing-v1"
    )
    incomplete = input_shards_module._build_top_manifest(
        config=collection.config,
        policy=collection.manifest["policy"],
        shard_size=collection.manifest["shard_size"],
        requested_total_root_target=2,
        shards=collection.manifest["shards"],
        aggregate=input_shards_module._Aggregate.from_manifest(collection.manifest),
    )
    (input_dir / "manifest.json").write_bytes(
        (canonical_json(incomplete) + "\n").encode("utf-8")
    )

    with pytest.raises(ValueError, match="complete and frozen"):
        evaluate_hu_behavior_trace_shards(input_dir, output_dir, evaluators)


def test_real_two_root_direct_logit_shard_smoke(tmp_path):
    checkpoint_paths = [
        Path("ai/data/t1_hu_bb_gcp_2m_model/t2_policyvalue_model.pt"),
        Path("ai/data/t1_hu_btn_gcp_2m_model/t2_policyvalue_model.pt"),
        Path("ai/data/t2_hu_bb_gcp_2m_model/t2_policyvalue_model.pt"),
        Path("ai/data/t2_hu_btn_gcp_2m_model/t2_policyvalue_model.pt"),
    ]
    if not all(path.is_file() for path in checkpoint_paths):
        pytest.skip("all four known HU policy-value checkpoints are required")
    input_dir = tmp_path / "real-input"
    output_dir = tmp_path / "real-evaluation"
    dispatch = build_known_hu_policy_value_prior_dispatch(Path.cwd())
    collect_sharded_behavior_traces(
        input_dir,
        BehaviorTraceCollectionConfig(
            seed_namespace="eval-shards/real-two-root-v1",
            root_sampling_mode=NATURAL_UNIFORM_SHUFFLE,
        ),
        dispatch,
        total_root_target=2,
        shard_size=2,
    )
    del dispatch
    gc.collect()
    # Load the separately constructed direct-logit readers, matching the real
    # collector CLI -> evaluator CLI handoff instead of reusing model objects.
    evaluators = build_known_hu_policy_value_logit_evaluators(Path.cwd())

    result = evaluate_hu_behavior_trace_shards(
        input_dir, output_dir, evaluators
    )
    assert result.evaluation_complete is True
    assert result.shard_count == 1
    assert result.row_count == 8
    assert result.manifest["census"]["row_counts_by_role"] == {
        "t1_bb": 2,
        "t1_btn": 2,
        "t2_bb": 2,
        "t2_btn": 2,
    }
