import copy
import hashlib
import json
from pathlib import Path

import pytest

import ai.tutor.merge_fl14_teacher_labels as merger
from ai.tutor.encode_fl14_teacher import split_of
from ai.tutor.merge_fl14_teacher_labels import merge_teacher_labels


def _id_for(split: str, start: int = 1) -> int:
    value = start
    while split_of(value, "t2") != split:
        value += 1
    return value


def _record(root_id: int, values=(0.0, 1.0), *, board="Ac||2d,3d,4d,5d,6d"):
    return {
        "id": str(root_id),
        "root": 0,
        "stream": 0,
        "opponents": 60,
        "board": board,
        "dead": "7c",
        "draw": "8c,9c,Tc",
        "actions": [
            {"action_key": "action-a", "value": values[0], "t3_draws": 48},
            {"action_key": "action-b", "value": values[1], "t3_draws": 48},
        ],
    }


def _high(record: dict, values, *, reverse=False):
    out = copy.deepcopy(record)
    out["opponents"] = 120
    for action, value in zip(out["actions"], values):
        action["value"] = value
        action["t3_draws"] = 96
    if reverse:
        out["actions"].reverse()
    return out


def _write(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )


def _merge(tmp_path: Path, base: list[dict], streams: list[list[dict]], *, split="fit", suffix=""):
    base_path = tmp_path / f"base{suffix}.jsonl"
    relabel_paths = []
    _write(base_path, base)
    for index, records in enumerate(streams):
        path = tmp_path / f"high{suffix}-{index}.jsonl"
        _write(path, records)
        relabel_paths.append(path)
    output = tmp_path / f"merged{suffix}.jsonl"
    manifest = tmp_path / f"merged{suffix}.manifest.json"
    result = merge_teacher_labels(
        base_path=base_path,
        relabel_paths=relabel_paths,
        output_path=output,
        manifest_path=manifest,
        street="t2",
        expected_split=split,
    )
    return output, manifest, result


def test_merge_replaces_by_action_key_and_is_deterministic(tmp_path):
    root = _id_for("fit")
    untouched = _id_for("fit", root + 1)
    base = [_record(root), _record(untouched, values=(8.0, 9.0))]
    first = _high(base[0], (4.0, 2.0), reverse=True)
    second = _high(base[0], (6.0, 4.0))

    output, _manifest_path, manifest = _merge(
        tmp_path, base, [[first], [second]], suffix="-a"
    )
    reversed_output, _, _ = _merge(
        tmp_path, base, [[second], [first]], suffix="-b"
    )
    records = [json.loads(line) for line in output.read_text().splitlines()]

    assert output.read_bytes() == reversed_output.read_bytes()
    assert [record["id"] for record in records] == [str(root), str(untouched)]
    assert [action["action_key"] for action in records[0]["actions"]] == [
        "action-a",
        "action-b",
    ]
    assert [action["value"] for action in records[0]["actions"]] == [5.0, 3.0]
    assert records[0]["opponents"] == 120
    assert records[0]["actions"][0]["t3_draws"] == 96
    assert records[1] == base[1]
    assert manifest["append_allowed"] is False
    assert manifest["replacement_roots"] == 1
    assert manifest["relabel_best_action_agreement_roots"] == 1
    assert manifest["output"]["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()


def test_merge_rejects_append_and_split_leakage(tmp_path):
    fit_root = _id_for("fit")
    other_fit = _id_for("fit", fit_root + 1)
    base = [_record(fit_root)]
    extra = _record(other_fit)
    with pytest.raises(ValueError, match="append is forbidden"):
        _merge(tmp_path, base, [[extra], [extra]], suffix="-append")

    dev_root = _id_for("dev")
    dev = _record(dev_root)
    with pytest.raises(ValueError, match="outside requested fit"):
        _merge(tmp_path, [dev], [[_high(dev, (1, 2))], [_high(dev, (1, 2))]], suffix="-split")


@pytest.mark.parametrize("fault", ["context", "missing-action", "extra-action"])
def test_merge_rejects_context_and_action_set_drift(tmp_path, fault):
    root = _id_for("fit")
    base = _record(root)
    first = _high(base, (1.0, 2.0))
    second = _high(base, (2.0, 3.0))
    if fault == "context":
        second["board"] = "Ad||2d,3d,4d,5d,6d"
        match = "context differs"
    elif fault == "missing-action":
        second["actions"].pop()
        match = "action set differs"
    else:
        second["actions"].append(
            {"action_key": "action-extra", "value": 0.0, "t3_draws": 96}
        )
        match = "action set differs"
    with pytest.raises(ValueError, match=match):
        _merge(tmp_path, [base], [[first], [second]], suffix=f"-{fault}")


def test_merge_rejects_duplicate_ids_actions_and_stream_root_drift(tmp_path):
    root = _id_for("fit")
    second_root = _id_for("fit", root + 1)
    base = _record(root)
    duplicate_action = _high(base, (1.0, 2.0))
    duplicate_action["actions"][1]["action_key"] = "action-a"
    with pytest.raises(ValueError, match="duplicate action_key"):
        _merge(
            tmp_path,
            [base],
            [[duplicate_action], [_high(base, (1.0, 2.0))]],
            suffix="-duplicate-action",
        )

    with pytest.raises(ValueError, match="duplicate id"):
        _merge(
            tmp_path,
            [base, copy.deepcopy(base)],
            [[_high(base, (1.0, 2.0))], [_high(base, (1.0, 2.0))]],
            suffix="-duplicate-id",
        )

    extra = _record(second_root)
    with pytest.raises(ValueError, match="root set differs"):
        _merge(
            tmp_path,
            [base, extra],
            [[_high(base, (1.0, 2.0))], [_high(base, (1.0, 2.0)), _high(extra, (2.0, 3.0))]],
            suffix="-root-drift",
        )


def test_manifest_atomic_replace_failure_preserves_previous_file(tmp_path, monkeypatch):
    root = _id_for("fit")
    base = _record(root)
    manifest_path = tmp_path / "merged-atomic.manifest.json"
    manifest_path.write_text("previous manifest\n", encoding="utf-8")
    real_replace = merger.os.replace

    def fail_manifest_replace(source, destination):
        if Path(destination) == manifest_path:
            raise OSError("simulated manifest replace failure")
        return real_replace(source, destination)

    monkeypatch.setattr(merger.os, "replace", fail_manifest_replace)
    with pytest.raises(OSError, match="simulated manifest replace failure"):
        _merge(
            tmp_path,
            [base],
            [[_high(base, (1.0, 2.0))], [_high(base, (2.0, 3.0))]],
            suffix="-atomic",
        )

    assert manifest_path.read_text(encoding="utf-8") == "previous manifest\n"
    assert not list(tmp_path.glob(f".{manifest_path.name}.*.tmp"))
