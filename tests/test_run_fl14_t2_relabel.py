import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import ai.tutor.run_fl14_t2_relabel as runner


def _root(index: int) -> dict:
    suits = ("s", "h", "d", "c")
    suit = suits[index % len(suits)]
    return {
        "id": f"root-{index}",
        "rows": [
            [f"A{suit}"],
            [f"2{suit}", f"3{suit}", f"4{suit}"],
            [f"5{suit}", f"6{suit}", f"7{suit}"],
        ],
        "dead": [f"J{suit}"],
        "draw": [f"8{suit}", f"9{suit}", f"T{suit}"],
    }


def _write_roots(path: Path, count: int) -> list[dict]:
    roots = [_root(index) for index in range(count)]
    path.write_text(
        "".join(json.dumps(root, separators=(",", ":")) + "\n" for root in roots),
        encoding="utf-8",
    )
    return roots


def _arg(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def _fake_success(calls: list[list[str]]):
    def fake_run(command, *, cwd, stdout, stderr, timeout, check):
        del cwd, timeout, check
        command = list(command)
        calls.append(command)
        roots_path = Path(_arg(command, "--roots-file"))
        out_dir = Path(_arg(command, "--out-dir"))
        root_offset = int(_arg(command, "--root-offset"))
        stream_offset = int(_arg(command, "--stream-offset"))
        opponents = int(_arg(command, "--opponents"))
        t3_draws = int(_arg(command, "--t3-draws"))
        roots = [
            runner.validate_root(json.loads(line), label="fake")
            for line in roots_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        lines = []
        for local_index, root in enumerate(roots):
            actions = []
            for action_index, signature in enumerate(
                sorted(runner.expected_action_signatures(root))
            ):
                *rows, discard = signature
                key = "|".join([*(",".join(row) for row in rows), discard])
                actions.append(
                    {
                        "action_key": key,
                        "value": float(action_index),
                        "t3_draws": t3_draws,
                    }
                )
            global_root = root_offset + local_index
            lines.append(
                json.dumps(
                    {
                        "id": root["id"],
                        "root": global_root,
                        "stream": runner.stream_of(global_root, stream_offset),
                        "opponents": opponents,
                        "board": "|".join(",".join(row) for row in root["rows"]),
                        "dead": ",".join(root["dead"]),
                        "draw": ",".join(root["draw"]),
                        "actions": actions,
                    },
                    separators=(",", ":"),
                )
            )
        (out_dir / "t2_labels.jsonl").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        stdout.write(f"wrote {len(lines)} roots\n")
        stderr.write("solver diagnostic\n")
        stdout.flush()
        stderr.flush()
        return SimpleNamespace(returncode=0)

    return fake_run


def _config(tmp_path: Path, *, roots: int = 3, shard_size: int = 2, **kwargs):
    roots_path = tmp_path / "roots.jsonl"
    _write_roots(roots_path, roots)
    solver = tmp_path / "fl_solver.exe"
    pool = tmp_path / "fl14.jfl1"
    solver.write_bytes(b"fake solver v1")
    pool.write_bytes(b"fake pool v1")
    return runner.RunConfig(
        roots=roots_path,
        output_dir=tmp_path / "out",
        solver=solver,
        pool=pool,
        shard_size=shard_size,
        opponents=120,
        t3_draws=96,
        t4_draws=24,
        stream_offset=900000,
        workspace_root=tmp_path,
        **kwargs,
    )


def test_shards_are_content_bound_merged_in_order_and_skipped_on_rerun(
    tmp_path, monkeypatch
):
    config = _config(tmp_path)
    calls: list[list[str]] = []
    monkeypatch.setattr(runner.subprocess, "run", _fake_success(calls))

    result = runner.run(config)

    assert result["status"] == "complete"
    assert len(calls) == 2
    assert [_arg(command, "--root-offset") for command in calls] == ["0", "2"]
    merged = config.output_dir / "t2_labels.jsonl"
    merged_rows = [json.loads(line) for line in merged.read_text().splitlines()]
    assert [row["id"] for row in merged_rows] == ["root-0", "root-1", "root-2"]
    assert [row["root"] for row in merged_rows] == [0, 1, 2]
    for entry in result["shards"]:
        shard = json.loads(Path(entry["manifest"]).read_text(encoding="utf-8"))
        attempt = shard["attempts"][-1]
        assert shard["status"] == "complete"
        assert attempt["started_at_utc"]
        assert attempt["finished_at_utc"]
        assert attempt["command"][1] == "teach-t2"
        assert attempt["stdout"]["sha256"]
        assert attempt["stderr"]["sha256"]
        assert shard["validation"]["lines"] == shard["records"]
        assert shard["validation"]["actions"] > 0

    again = runner.run(config)

    assert again["status"] == "complete"
    assert len(calls) == 2, "verified completed shards must not run again"
    assert again["merged_output"]["sha256"] == runner.sha256_file(merged)


def test_max_shards_leaves_no_merge_and_next_run_resumes_only_incomplete_shards(
    tmp_path, monkeypatch
):
    limited = _config(tmp_path, roots=3, shard_size=1, max_shards=1)
    calls: list[list[str]] = []
    monkeypatch.setattr(runner.subprocess, "run", _fake_success(calls))

    first = runner.run(limited)

    assert first["status"] == "incomplete"
    assert len(calls) == 1
    assert not (limited.output_dir / "t2_labels.jsonl").exists()

    resumed = runner.run(
        runner.RunConfig(
            **{**limited.__dict__, "max_shards": 0}
        )
    )

    assert resumed["status"] == "complete"
    assert len(calls) == 3
    assert [_arg(command, "--root-offset") for command in calls] == ["0", "1", "2"]


def test_failed_shard_retries_in_a_new_attempt_without_truncating_the_first(
    tmp_path, monkeypatch
):
    config = _config(tmp_path, roots=1, shard_size=1)

    def fail(command, *, cwd, stdout, stderr, timeout, check):
        del cwd, timeout, check
        out_dir = Path(_arg(list(command), "--out-dir"))
        (out_dir / "t2_labels.jsonl").write_text("partial output\n", encoding="utf-8")
        stdout.write("partial stdout\n")
        stderr.write("intentional failure\n")
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(runner.subprocess, "run", fail)
    with pytest.raises(runner.RelabelError, match="exited with code 7"):
        runner.run(config)
    shard_dir = config.output_dir / "shards" / "shard_00000" / "attempts"
    first_output = shard_dir / "attempt_0001" / "t2_labels.jsonl"
    assert first_output.read_text(encoding="utf-8") == "partial output\n"

    calls: list[list[str]] = []
    monkeypatch.setattr(runner.subprocess, "run", _fake_success(calls))
    result = runner.run(config)

    assert result["status"] == "complete"
    assert first_output.read_text(encoding="utf-8") == "partial output\n"
    assert (shard_dir / "attempt_0002" / "t2_labels.jsonl").is_file()
    shard = json.loads(
        (config.output_dir / "shards" / "shard_00000" / "manifest.json").read_text()
    )
    assert [attempt["status"] for attempt in shard["attempts"]] == [
        "failed",
        "complete",
    ]


def test_completed_output_tamper_fails_before_solver_can_run_again(tmp_path, monkeypatch):
    config = _config(tmp_path, roots=1, shard_size=1)
    calls: list[list[str]] = []
    monkeypatch.setattr(runner.subprocess, "run", _fake_success(calls))
    result = runner.run(config)
    shard = json.loads(Path(result["shards"][0]["manifest"]).read_text())
    completed = Path(shard["completed_output"]["path"])
    completed.write_text(completed.read_text() + "tampered\n", encoding="utf-8")

    with pytest.raises(
        runner.RelabelError, match=r"completed output: (byte count|SHA256) mismatch"
    ):
        runner.run(config)

    assert len(calls) == 1


def test_exact_untracked_merge_is_recovered_after_atomic_rename_window(
    tmp_path, monkeypatch
):
    config = _config(tmp_path, roots=2, shard_size=1)
    calls: list[list[str]] = []
    monkeypatch.setattr(runner.subprocess, "run", _fake_success(calls))
    runner.run(config)
    top_path = config.output_dir / "manifest.json"
    top = json.loads(top_path.read_text(encoding="utf-8"))
    top["status"] = "running"
    top["merged_output"] = None
    top_path.write_text(json.dumps(top), encoding="utf-8")

    recovered = runner.run(config)

    assert recovered["status"] == "complete"
    assert recovered["merged_output"]["recovered_after_atomic_rename"] is True
    assert len(calls) == 2


def test_output_validation_rejects_missing_actions_and_changed_context(tmp_path):
    root = runner.validate_root(_root(0), label="root")
    output = tmp_path / "bad.jsonl"
    output.write_text(
        json.dumps(
            {
                "id": root["id"],
                "root": 0,
                "stream": 0,
                "opponents": 120,
                "board": "Ah|2s,3s,4s|5s,6s,7s",
                "dead": "Js",
                "draw": "8s,9s,Ts",
                "actions": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(runner.RelabelError, match="board context differs"):
        runner.validate_output(
            output,
            [root],
            root_offset=0,
            opponents=120,
            t3_draws=96,
            stream_offset=0,
        )


def test_nonempty_untracked_output_directory_is_never_reused(tmp_path):
    config = _config(tmp_path, roots=1, shard_size=1)
    config.output_dir.mkdir()
    sentinel = config.output_dir / "t2_labels.jsonl"
    sentinel.write_text("do not truncate\n", encoding="utf-8")

    with pytest.raises(runner.RelabelError, match="non-empty"):
        runner.run(config)

    assert sentinel.read_text(encoding="utf-8") == "do not truncate\n"
