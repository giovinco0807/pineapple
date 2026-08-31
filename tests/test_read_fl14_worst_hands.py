import json

from ai.tutor.encode_fl14_teacher import stable_root_id
from ai.tutor.read_fl14_worst_hands import action_keys


def test_action_keys_separates_reused_local_root_by_global_deal_id(tmp_path):
    labels = tmp_path / "labels.jsonl"
    records = [
        {"id": "deal-a", "root": 7, "actions": []},
        {"id": "deal-b", "root": 7, "actions": []},
    ]
    labels.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")

    # The global ids prevent the reused worker-local ordinal from merging the
    # two decisions, so either can be retrieved independently.
    first = action_keys(labels, "t2", {stable_root_id(records[0])})
    second = action_keys(labels, "t2", {stable_root_id(records[1])})
    assert set(first) == {stable_root_id(records[0])}
    assert set(second) == {stable_root_id(records[1])}
