import json
import shutil
from pathlib import Path

import pytest

from ai.tutor.behavior_calibration_contract import canonical_json, canonical_sha256
from ai.tutor.collect_hu_behavior_trace_shards import (
    TOP_MANIFEST_NAME,
    _Aggregate,
    _build_top_manifest,
    _ordered_shard_chain,
    collect_sharded_behavior_traces,
    read_sharded_behavior_trace_collection,
    resume_sharded_behavior_trace_collection,
)
from ai.tutor.collect_hu_behavior_traces import (
    DETERMINISTIC_JOKER_CYCLE,
    NATURAL_UNIFORM_SHUFFLE,
    TARGETED_JOKER_CHALLENGE,
    BehaviorTraceCollectionConfig,
    read_behavior_trace_dataset,
)
from ai.tutor.t3_hu_full_card_range import UniformLegalBehaviorModel


def _natural(namespace: str = "shard-tests/natural-v1"):
    return BehaviorTraceCollectionConfig(
        seed_namespace=namespace,
        root_sampling_mode=NATURAL_UNIFORM_SHUFFLE,
    )


def _targeted(namespace: str = "shard-tests/challenge-v1"):
    return BehaviorTraceCollectionConfig(
        seed_namespace=namespace,
        root_sampling_mode=TARGETED_JOKER_CHALLENGE,
        challenge_id="shard-joker-challenge-v1",
        challenge_deck_source=DETERMINISTIC_JOKER_CYCLE,
    )


def _rewrite_top(output: Path, manifest):
    (output / TOP_MANIFEST_NAME).write_text(
        canonical_json(manifest) + "\n", encoding="utf-8"
    )


def _rebuild_with_shards(output: Path, shards):
    old = json.loads((output / TOP_MANIFEST_NAME).read_text(encoding="utf-8"))
    config = BehaviorTraceCollectionConfig.from_canonical_dict(
        old["collection_config"]
    )
    rebuilt = _build_top_manifest(
        config=config,
        policy=old["policy"],
        shard_size=old["shard_size"],
        requested_total_root_target=old["requested_total_root_target"],
        shards=shards,
        aggregate=_Aggregate.from_manifest(old),
    )
    _rewrite_top(output, rebuilt)


def test_two_then_three_matches_one_shot_boundary_independent_content(tmp_path):
    config = _natural()
    model = UniformLegalBehaviorModel()
    incremental_dir = tmp_path / "incremental"
    one_shot_dir = tmp_path / "one-shot"

    first = collect_sharded_behavior_traces(
        incremental_dir,
        config,
        model,
        total_root_target=2,
        shard_size=10,
    )
    assert first.root_count == 2
    assert first.shard_count == 1
    resumed = resume_sharded_behavior_trace_collection(
        incremental_dir,
        model,
        total_root_target=5,
        shard_size=10,
        expected_config=config,
    )
    one_shot = collect_sharded_behavior_traces(
        one_shot_dir,
        config,
        model,
        total_root_target=5,
        shard_size=10,
    )

    assert resumed.root_count == one_shot.root_count == 5
    assert resumed.decision_count == one_shot.decision_count == 20
    assert resumed.shard_count == 2
    assert one_shot.shard_count == 1
    assert (
        resumed.manifest["global_content_commitments"]
        == one_shot.manifest["global_content_commitments"]
    )
    assert resumed.manifest["census"] == one_shot.manifest["census"]
    assert (
        resumed.manifest["collection_content_sha256"]
        == one_shot.manifest["collection_content_sha256"]
    )
    # Physical layouts intentionally bind different immutable shard boundaries.
    assert resumed.manifest["layout_sha256"] != one_shot.manifest["layout_sha256"]
    assert resumed.manifest["promotion_eligible"] is False
    assert resumed.manifest["raw_collection_only"] is True
    assert resumed.manifest["counters"]["policy_query_count"] == 20
    assert resumed.manifest["counters"]["model_evaluation_count"] == 20
    assert resumed.manifest["counters"]["elapsed_collection_runtime_ns"] >= 0
    assert resumed.manifest["counters"]["shard_artifact_bytes"] > 0

    ranges = []
    for entry in resumed.manifest["shards"]:
        shard = read_behavior_trace_dataset(
            incremental_dir / entry["name"] / "decisions.jsonl"
        )
        ranges.append(
            (shard.root_index_start, shard.root_index_stop_exclusive)
        )
        assert shard.manifest["promotion_eligible"] is False
    assert ranges == [(0, 2), (2, 5)]
    assert not (incremental_dir / "decisions.jsonl").exists()
    assert not (incremental_dir / "roots.jsonl").exists()


def test_targeted_mode_census_and_append_only_resume_contracts(tmp_path):
    output = tmp_path / "targeted"
    config = _targeted()
    model = UniformLegalBehaviorModel()
    initial = collect_sharded_behavior_traces(
        output,
        config,
        model,
        total_root_target=3,
        shard_size=2,
    )
    assert initial.shard_count == 2
    assert sum(
        initial.manifest["census"]["challenge_target_root_counts"].values()
    ) == 3

    shard_hashes = {
        entry["name"]: {
            name: (output / entry["name"] / name).read_bytes()
            for name in ("decisions.jsonl", "roots.jsonl", "manifest.json")
        }
        for entry in initial.manifest["shards"]
    }
    resumed = resume_sharded_behavior_trace_collection(
        output,
        model,
        total_root_target=4,
        shard_size=2,
        expected_config=config,
    )
    assert resumed.shard_count == 3
    assert [(entry["root_index_start"], entry["root_index_stop_exclusive"]) for entry in resumed.manifest["shards"]] == [
        (0, 2),
        (2, 3),
        (3, 4),
    ]
    for shard_name, files in shard_hashes.items():
        for name, before in files.items():
            assert (output / shard_name / name).read_bytes() == before

    with pytest.raises(ValueError, match="shard size"):
        resume_sharded_behavior_trace_collection(
            output,
            model,
            total_root_target=5,
            shard_size=3,
            expected_config=config,
        )
    with pytest.raises(ValueError, match="resume config"):
        resume_sharded_behavior_trace_collection(
            output,
            model,
            total_root_target=5,
            shard_size=2,
            expected_config=_targeted("shard-tests/changed-config-v1"),
        )
    with pytest.raises(ValueError, match="behavior policy"):
        resume_sharded_behavior_trace_collection(
            output,
            UniformLegalBehaviorModel(model_id="changed-uniform-policy-v1"),
            total_root_target=5,
            shard_size=2,
            expected_config=config,
        )


def test_partial_staging_is_ignored_but_finalized_orphan_is_rejected(tmp_path):
    output = tmp_path / "interrupt"
    config = _natural("shard-tests/interruption-v1")
    model = UniformLegalBehaviorModel()
    collect_sharded_behavior_traces(
        output, config, model, total_root_target=1, shard_size=1
    )

    partial = output / ".shard-000001.crash.partial"
    partial.mkdir()
    (partial / "decisions.jsonl").write_text("truncated", encoding="utf-8")
    assert read_sharded_behavior_trace_collection(output).root_count == 1

    orphan = output / "shard-000001"
    shutil.copytree(output / "shard-000000", orphan)
    with pytest.raises(ValueError, match="orphan"):
        read_sharded_behavior_trace_collection(output)


def test_decision_tamper_fails_before_resume_and_does_not_rewrite_shards(tmp_path):
    output = tmp_path / "tamper"
    config = _natural("shard-tests/tamper-v1")
    model = UniformLegalBehaviorModel()
    result = collect_sharded_behavior_traces(
        output, config, model, total_root_target=2, shard_size=1
    )
    untouched = (output / "shard-000001" / "decisions.jsonl").read_bytes()
    path = output / "shard-000000" / "decisions.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["visible_joker_count"] = (rows[0]["visible_joker_count"] + 1) % 3
    path.write_text(
        "\n".join(canonical_json(row) for row in rows) + "\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="record SHA-256 mismatch"):
        resume_sharded_behavior_trace_collection(
            output,
            model,
            total_root_target=3,
            shard_size=1,
            expected_config=config,
        )
    assert (output / "shard-000001" / "decisions.jsonl").read_bytes() == untouched
    assert len(result.manifest["shards"]) == 2


@pytest.mark.parametrize("mutation", ["gap", "duplicate"])
def test_authenticated_gap_and_duplicate_shard_metadata_fail_closed(tmp_path, mutation):
    output = tmp_path / mutation
    config = _natural(f"shard-tests/{mutation}-v1")
    model = UniformLegalBehaviorModel()
    result = collect_sharded_behavior_traces(
        output, config, model, total_root_target=3, shard_size=2
    )
    shards = [dict(entry) for entry in result.manifest["shards"]]
    if mutation == "gap":
        shards[1]["root_index_start"] = 1
    else:
        shards[1]["index"] = 0
        shards[1]["name"] = "shard-000000"
    unsigned = dict(shards[1])
    unsigned.pop("entry_sha256")
    shards[1]["entry_sha256"] = canonical_sha256(unsigned)
    _rebuild_with_shards(output, shards)

    expected = "gap, overlap" if mutation == "gap" else "duplicate/out-of-order"
    with pytest.raises(ValueError, match=expected):
        read_sharded_behavior_trace_collection(output)


def test_top_manifest_binds_all_ordered_shard_artifact_hashes(tmp_path):
    output = tmp_path / "bindings"
    result = collect_sharded_behavior_traces(
        output,
        _natural("shard-tests/bindings-v1"),
        UniformLegalBehaviorModel(),
        total_root_target=2,
        shard_size=1,
    )
    assert result.manifest["ordered_shard_entry_chain_sha256"] == _ordered_shard_chain(
        result.manifest["shards"]
    )
    assert all(
        len(digest) == 64 for digest in result.manifest["source_hashes"].values()
    )
    for entry in result.manifest["shards"]:
        assert len(entry["collection_content_sha256"]) == 64
        assert len(entry["shard_manifest_sha256"]) == 64
        assert len(entry["decisions_file_sha256"]) == 64
        assert len(entry["hidden_roots_file_sha256"]) == 64
        assert len(entry["shard_manifest_file_sha256"]) == 64
        assert sum(entry["census"]["root_counts_by_split"].values()) == 1
        assert sum(entry["census"]["decision_counts_by_cell"].values()) == 4
        assert entry["artifact_sizes"]["total_bytes"] == sum(
            entry["artifact_sizes"][key]
            for key in (
                "decisions_bytes",
                "hidden_roots_bytes",
                "shard_manifest_bytes",
            )
        )
