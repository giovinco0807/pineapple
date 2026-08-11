from __future__ import annotations

import json

import pytest

from ofc_regular.aggregate_hu_turn0_fired_replay import aggregate_replay


def _row(target_id: str) -> dict:
    return {
        "target_id": target_id,
        "future_samples": 32,
        "common_random_futures_verified": True,
        "action_mapping_verified": True,
        "candidate_delta_vs_baseline": 1.0,
        "candidate_delta_se_vs_baseline": 0.25,
        "candidate_delta_lcb196": 0.51,
        "safe_override_label": "positive",
        "seat": "first",
        "source_config_id": "cfg",
        "replay_actions": [
            {"rollout_count": 32},
            {"rollout_count": 32},
        ],
    }


def _write_shard(root, prefix: str, row: dict) -> None:
    path = root / prefix
    path.mkdir(parents=True)
    (path / "DONE").write_text("complete\n", encoding="utf-8")
    (path / "teacher.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (path / "summary.json").write_text(
        json.dumps({"written": 1, "missing": 0, "elapsed_seconds": 2.0}) + "\n",
        encoding="utf-8",
    )


def test_aggregate_replay_validates_and_merges_unique_targets(tmp_path) -> None:
    root = tmp_path / "results"
    _write_shard(root, "s0", _row("a"))
    _write_shard(root, "s1", _row("b"))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps({"shard": index, "samples": 1, "output_prefix": prefix})
            for index, prefix in enumerate(("s0", "s1"))
        )
        + "\n",
        encoding="utf-8",
    )

    result = aggregate_replay(
        results_root=root,
        shard_manifest=manifest,
        output_dir=tmp_path / "out",
    )

    assert result["records"] == 2
    assert result["completed_shards"] == 2
    assert result["label_counts"] == {"positive": 2}
    assert result["common_random_futures_verified"] is True


def test_aggregate_replay_rejects_duplicate_targets(tmp_path) -> None:
    root = tmp_path / "results"
    _write_shard(root, "s0", _row("same"))
    _write_shard(root, "s1", _row("same"))
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps({"shard": 0, "samples": 1, "output_prefix": "s0"})
        + "\n"
        + json.dumps({"shard": 1, "samples": 1, "output_prefix": "s1"})
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="duplicate replay target ids"):
        aggregate_replay(
            results_root=root,
            shard_manifest=manifest,
            output_dir=tmp_path / "out",
        )
