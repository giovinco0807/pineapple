import json
from pathlib import Path

from ofc_regular.aggregate_hu_turn2_stage9f_profile_canary import (
    aggregate_summaries,
    merge_decisions,
)


def _write_shard(path: Path, *, shard: int, t1_count: int) -> None:
    path.mkdir(parents=True)
    (path / "DONE").write_text("", encoding="utf-8")
    summary = {
        "profile_a": "stage9f_p2_hu_t1_topk_confirm",
        "profile_b": "stage9f_p2",
        "paired_seeds": 10,
        "hands": 20,
        "avg_score_per_hand_for_a": 0.5,
        "topk_decisions_written": 40,
        "topk_realized_delta_count": 38,
        "topk_realized_override_count": 1,
        "topk_realized_override_delta_mean": 2.0,
        "topk_non_fired_nonzero_count": 0,
        "hu_turn1_decisions_written": t1_count,
        "hu_turn1_realized_delta_count": t1_count,
        "hu_turn1_realized_override_count": shard + 1,
        "hu_turn1_non_fired_final_mismatch_count": 0,
    }
    (path / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False),
        encoding="utf-8",
    )
    (path / "topk_decisions.jsonl").write_text(
        json.dumps({"shard": shard, "kind": "topk"}) + "\n",
        encoding="utf-8",
    )
    (path / "hu_turn1_decisions.jsonl").write_text(
        json.dumps({"shard": shard, "kind": "t1"}) + "\n",
        encoding="utf-8",
    )


def test_aggregate_profile_canary_merges_optional_hu_turn1_decisions(tmp_path: Path):
    shard0 = tmp_path / "results" / "shard0"
    shard1 = tmp_path / "results" / "shard1"
    _write_shard(shard0, shard=0, t1_count=20)
    _write_shard(shard1, shard=1, t1_count=22)

    summary = aggregate_summaries([shard0, shard1])
    assert summary["paired_seeds"] == 20
    assert summary["hands"] == 40
    assert summary["topk_decisions_written"] == 80
    assert summary["hu_turn1_decisions_written"] == 42
    assert summary["hu_turn1_realized_override_count"] == 3
    assert summary["hu_turn1_non_fired_final_mismatch_count"] == 0

    topk_path = tmp_path / "out" / "topk.jsonl"
    t1_path = tmp_path / "out" / "t1.jsonl"
    assert merge_decisions([shard0, shard1], topk_path, file_name="topk_decisions.jsonl") == 2
    assert merge_decisions([shard0, shard1], t1_path, file_name="hu_turn1_decisions.jsonl") == 2
    assert '"kind": "topk"' in topk_path.read_text(encoding="utf-8")
    assert '"kind": "t1"' in t1_path.read_text(encoding="utf-8")
