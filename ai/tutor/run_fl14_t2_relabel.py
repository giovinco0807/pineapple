"""Run ``fl_solver teach-t2`` in restartable, content-bound shards.

The Rust command opens ``t2_labels.jsonl`` with ``File::create``.  Calling it
again on the same directory therefore destroys a useful partial result.  This
runner never reuses an attempt directory: completed shards are verified and
skipped, while an incomplete shard gets a new attempt directory.  Only a fully
verified set of shards is merged, in source-root order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


RUN_SCHEMA = "ofc_fl14_t2_relabel_run/v1"
SHARD_SCHEMA = "ofc_fl14_t2_relabel_shard/v1"
CAPACITY = (3, 5, 5)
JOKER_NAMES = {"X", "X1", "X2", "JK"}
RANKS = "23456789TJQKA"
SUITS = "shdc"


class RelabelError(RuntimeError):
    """A fail-closed relabel orchestration or validation error."""


@dataclass(frozen=True)
class RunConfig:
    roots: Path
    output_dir: Path
    solver: Path
    pool: Path
    shard_size: int = 25
    opponents: int = 120
    t3_draws: int = 96
    t4_draws: int = 24
    stream_offset: int = 0
    workspace_root: Path = Path.cwd()
    timeout_seconds: float = 0.0
    max_shards: int = 0
    keep_going: bool = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise RelabelError(f"refusing to replace existing temporary file: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RelabelError(f"cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise RelabelError(f"{label} {path} is not a JSON object")
    return value


def _normal_card(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise RelabelError(f"{label}: card is not a string: {value!r}")
    if value in JOKER_NAMES:
        return "X"
    if len(value) != 2 or value[0] not in RANKS or value[1] not in SUITS:
        raise RelabelError(f"{label}: invalid card {value!r}")
    return value


def _card_list(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, list):
        raise RelabelError(f"{label}: expected a card list")
    return [_normal_card(card, label=label) for card in value]


def _text_cards(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, str):
        raise RelabelError(f"{label}: expected comma-separated cards")
    if not value:
        return []
    return [_normal_card(card, label=label) for card in value.split(",")]


def _canonical(cards: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted(cards))


def validate_root(record: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise RelabelError(f"{label}: root is not an object")
    identifier = record.get("id")
    if not isinstance(identifier, str) or not identifier:
        raise RelabelError(f"{label}: id must be a non-empty string")
    if '"' in identifier or "\\" in identifier:
        raise RelabelError(f"{label}: id cannot contain quote or backslash")
    raw_rows = record.get("rows")
    if not isinstance(raw_rows, list) or len(raw_rows) != 3:
        raise RelabelError(f"{label}: expected exactly three rows")
    rows = [
        _card_list(row, label=f"{label}.rows[{index}]")
        for index, row in enumerate(raw_rows)
    ]
    for index, (row, capacity) in enumerate(zip(rows, CAPACITY)):
        if len(row) > capacity:
            raise RelabelError(
                f"{label}.rows[{index}]: {len(row)} cards exceeds {capacity}"
            )
    if sum(map(len, rows)) != 7:
        raise RelabelError(f"{label}: T2 root must have seven placed cards")
    dead = _card_list(record.get("dead"), label=f"{label}.dead")
    draw = _card_list(record.get("draw"), label=f"{label}.draw")
    if len(dead) != 1:
        raise RelabelError(f"{label}: T2 root must have one dead card")
    if len(draw) != 3:
        raise RelabelError(f"{label}: T2 root must have a three-card draw")
    all_cards = [card for row in rows for card in row] + dead + draw
    naturals = [card for card in all_cards if card != "X"]
    if len(naturals) != len(set(naturals)):
        raise RelabelError(f"{label}: a natural card is repeated")
    if all_cards.count("X") > 2:
        raise RelabelError(f"{label}: more than two jokers are present")
    return {
        "id": identifier,
        "rows": rows,
        "dead": dead,
        "draw": draw,
    }


def read_roots(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    compact_lines: list[str] = []
    identifiers: set[str] = set()
    try:
        handle = path.open("r", encoding="utf-8-sig")
    except OSError as error:
        raise RelabelError(f"cannot read roots {path}: {error}") from error
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise RelabelError(f"{path}:{line_number}: {error}") from error
            record = validate_root(raw, label=f"{path}:{line_number}")
            if record["id"] in identifiers:
                raise RelabelError(f"{path}:{line_number}: duplicate id {record['id']!r}")
            identifiers.add(record["id"])
            records.append(record)
            # Preserve the accepted spellings in the solver input, but make the
            # shard bytes independent of whitespace in the source file.
            compact_lines.append(
                json.dumps(raw, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    if not records:
        raise RelabelError(f"roots file is empty: {path}")
    return records, compact_lines


def mix64(value: int) -> int:
    mask = (1 << 64) - 1
    z = (value + 0x9E37_79B9_7F4A_7C15) & mask
    z = ((z ^ (z >> 30)) * 0xBF58_476D_1CE4_E5B9) & mask
    z = ((z ^ (z >> 27)) * 0x94D0_49BB_1331_11EB) & mask
    return (z ^ (z >> 31)) & mask


def stream_of(root: int, offset: int) -> int:
    return root if offset == 0 else mix64(root ^ offset)


ActionSignature = tuple[
    tuple[str, ...], tuple[str, ...], tuple[str, ...], str
]


def _action_signature(rows: Sequence[Sequence[str]], discard: str) -> ActionSignature:
    return (
        _canonical(rows[0]),
        _canonical(rows[1]),
        _canonical(rows[2]),
        discard,
    )


def expected_action_signatures(root: Mapping[str, Any]) -> set[ActionSignature]:
    rows = [list(row) for row in root["rows"]]
    draw = list(root["draw"])
    rooms = [capacity - len(row) for row, capacity in zip(rows, CAPACITY)]
    expected: set[ActionSignature] = set()
    for discard_index in range(3):
        kept = [index for index in range(3) if index != discard_index]
        for first in range(3):
            for second in range(3):
                need = [0, 0, 0]
                need[first] += 1
                need[second] += 1
                if any(need[index] > rooms[index] for index in range(3)):
                    continue
                after = [list(row) for row in rows]
                after[first].append(draw[kept[0]])
                after[second].append(draw[kept[1]])
                expected.add(_action_signature(after, draw[discard_index]))
    return expected


def _validate_action(
    action: Any,
    *,
    root: Mapping[str, Any],
    label: str,
    t3_draws: int,
) -> ActionSignature:
    if not isinstance(action, dict):
        raise RelabelError(f"{label}: action is not an object")
    key = action.get("action_key")
    if not isinstance(key, str):
        raise RelabelError(f"{label}: missing action_key")
    parts = key.split("|")
    if len(parts) != 4:
        raise RelabelError(f"{label}: malformed action_key {key!r}")
    rows = [
        _text_cards(parts[index], label=f"{label}.action_key.rows[{index}]")
        for index in range(3)
    ]
    discard_cards = _text_cards(parts[3], label=f"{label}.action_key.discard")
    if len(discard_cards) != 1:
        raise RelabelError(f"{label}: action must discard exactly one card")
    for index, (row, capacity) in enumerate(zip(rows, CAPACITY)):
        if len(row) > capacity:
            raise RelabelError(f"{label}: action row {index} exceeds capacity")
    if sum(map(len, rows)) != 9:
        raise RelabelError(f"{label}: action must leave nine placed cards")
    value = action.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RelabelError(f"{label}: action value is not numeric")
    if not math.isfinite(float(value)):
        raise RelabelError(f"{label}: action value is not finite")
    if action.get("t3_draws") != t3_draws:
        raise RelabelError(
            f"{label}: t3_draws {action.get('t3_draws')!r} != {t3_draws}"
        )
    return _action_signature(rows, discard_cards[0])


def validate_output(
    path: Path,
    roots: Sequence[Mapping[str, Any]],
    *,
    root_offset: int,
    opponents: int,
    t3_draws: int,
    stream_offset: int,
) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise RelabelError(f"cannot read solver output {path}: {error}") from error
    if not payload.endswith(b"\n"):
        raise RelabelError(f"solver output does not end in a newline: {path}")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RelabelError(f"solver output is not UTF-8: {path}: {error}") from error
    lines = text.splitlines()
    if any(not line.strip() for line in lines):
        raise RelabelError(f"solver output contains a blank line: {path}")
    if len(lines) != len(roots):
        raise RelabelError(
            f"solver output has {len(lines)} lines, expected {len(roots)}: {path}"
        )
    action_total = 0
    for local_index, (line, expected_root) in enumerate(zip(lines, roots)):
        label = f"{path}:{local_index + 1}"
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise RelabelError(f"{label}: {error}") from error
        if not isinstance(record, dict):
            raise RelabelError(f"{label}: output record is not an object")
        if str(record.get("id")) != expected_root["id"]:
            raise RelabelError(
                f"{label}: id {record.get('id')!r} != {expected_root['id']!r}"
            )
        global_root = root_offset + local_index
        if record.get("root") != global_root:
            raise RelabelError(
                f"{label}: root {record.get('root')!r} != {global_root}"
            )
        expected_stream = stream_of(global_root, stream_offset)
        if record.get("stream") != expected_stream:
            raise RelabelError(
                f"{label}: stream {record.get('stream')!r} != {expected_stream}"
            )
        if record.get("opponents") != opponents:
            raise RelabelError(
                f"{label}: opponents {record.get('opponents')!r} != {opponents}"
            )
        board = record.get("board")
        if not isinstance(board, str) or len(board.split("|")) != 3:
            raise RelabelError(f"{label}: malformed board")
        board_rows = [
            _text_cards(part, label=f"{label}.board[{index}]")
            for index, part in enumerate(board.split("|"))
        ]
        if [_canonical(row) for row in board_rows] != [
            _canonical(row) for row in expected_root["rows"]
        ]:
            raise RelabelError(f"{label}: board context differs from the input root")
        dead = _text_cards(record.get("dead"), label=f"{label}.dead")
        draw = _text_cards(record.get("draw"), label=f"{label}.draw")
        if _canonical(dead) != _canonical(expected_root["dead"]):
            raise RelabelError(f"{label}: dead-card context differs from the input root")
        if _canonical(draw) != _canonical(expected_root["draw"]):
            raise RelabelError(f"{label}: draw context differs from the input root")
        actions = record.get("actions")
        if not isinstance(actions, list) or not actions:
            raise RelabelError(f"{label}: actions must be a non-empty list")
        actual = {
            _validate_action(
                action,
                root=expected_root,
                label=f"{label}.actions[{index}]",
                t3_draws=t3_draws,
            )
            for index, action in enumerate(actions)
        }
        if len(actual) != len(actions):
            raise RelabelError(f"{label}: duplicate action key/signature")
        expected = expected_action_signatures(expected_root)
        if actual != expected:
            raise RelabelError(
                f"{label}: action set differs: "
                f"missing={len(expected - actual)}, extra={len(actual - expected)}"
            )
        action_total += len(actions)
    return {
        "lines": len(lines),
        "actions": action_total,
        "first_id": roots[0]["id"] if roots else None,
        "last_id": roots[-1]["id"] if roots else None,
        "root_start": root_offset,
        "root_stop": root_offset + len(roots),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _verify_artifact(value: Any, *, label: str) -> Path:
    if not isinstance(value, dict):
        raise RelabelError(f"{label}: artifact metadata is missing")
    path = Path(value.get("path", ""))
    if not path.is_file():
        raise RelabelError(f"{label}: artifact is missing: {path}")
    if path.stat().st_size != value.get("bytes"):
        raise RelabelError(f"{label}: byte count mismatch")
    if sha256_file(path) != value.get("sha256"):
        raise RelabelError(f"{label}: SHA256 mismatch")
    return path


def _resolved(config: RunConfig) -> RunConfig:
    roots = config.roots.resolve(strict=True)
    solver = config.solver.resolve(strict=True)
    pool = config.pool.resolve(strict=True)
    workspace = config.workspace_root.resolve(strict=True)
    output = config.output_dir.resolve()
    if not roots.is_file() or not solver.is_file() or not pool.is_file():
        raise RelabelError("roots, solver, and pool must be files")
    if not workspace.is_dir():
        raise RelabelError("workspace root must be a directory")
    if config.shard_size <= 0:
        raise RelabelError("shard-size must be positive")
    for name, value in (
        ("opponents", config.opponents),
        ("t3-draws", config.t3_draws),
        ("t4-draws", config.t4_draws),
    ):
        if value <= 0:
            raise RelabelError(f"{name} must be positive")
    if not 0 <= config.stream_offset < 2**64:
        raise RelabelError("stream-offset must fit u64")
    if config.timeout_seconds < 0 or config.max_shards < 0:
        raise RelabelError("timeout-seconds and max-shards cannot be negative")
    return RunConfig(
        roots=roots,
        output_dir=output,
        solver=solver,
        pool=pool,
        shard_size=config.shard_size,
        opponents=config.opponents,
        t3_draws=config.t3_draws,
        t4_draws=config.t4_draws,
        stream_offset=config.stream_offset,
        workspace_root=workspace,
        timeout_seconds=config.timeout_seconds,
        max_shards=config.max_shards,
        keep_going=config.keep_going,
    )


def _settings(config: RunConfig) -> dict[str, Any]:
    return {
        "shard_size": config.shard_size,
        "opponents": config.opponents,
        "t3_draws": config.t3_draws,
        "t4_draws": config.t4_draws,
        "stream_offset": config.stream_offset,
    }


def _shard_dir(output_dir: Path, index: int) -> Path:
    return output_dir / "shards" / f"shard_{index:05d}"


def _initialize(
    config: RunConfig,
    roots: Sequence[Mapping[str, Any]],
    compact_lines: Sequence[str],
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    output_dir = config.output_dir
    if output_dir.exists():
        entries = list(output_dir.iterdir())
        if entries:
            raise RelabelError(
                f"output-dir is non-empty and has no run manifest: {output_dir}"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    shards: list[dict[str, Any]] = []
    for index, start in enumerate(range(0, len(roots), config.shard_size)):
        stop = min(start + config.shard_size, len(roots))
        directory = _shard_dir(output_dir, index)
        directory.mkdir(parents=True, exist_ok=False)
        input_path = directory / "roots.jsonl"
        with input_path.open("xb") as handle:
            handle.write("".join(compact_lines[start:stop]).encode("utf-8"))
        input_artifact = _artifact(input_path)
        shard = {
            "schema": SHARD_SCHEMA,
            "status": "planned",
            "index": index,
            "root_start": start,
            "root_stop": stop,
            "records": stop - start,
            "input": input_artifact,
            "source_roots_sha256": bindings["roots"]["sha256"],
            "solver_sha256": bindings["solver"]["sha256"],
            "pool_sha256": bindings["pool"]["sha256"],
            "settings": _settings(config),
            "attempts": [],
            "completed_output": None,
            "created_at_utc": utc_now(),
            "updated_at_utc": utc_now(),
        }
        shard_manifest = directory / "manifest.json"
        atomic_write_json(shard_manifest, shard)
        shards.append(
            {
                "index": index,
                "root_start": start,
                "root_stop": stop,
                "records": stop - start,
                "directory": str(directory),
                "manifest": str(shard_manifest),
                "input_sha256": input_artifact["sha256"],
                "status": "planned",
            }
        )
    now = utc_now()
    manifest = {
        "schema": RUN_SCHEMA,
        "status": "planned",
        "created_at_utc": now,
        "updated_at_utc": now,
        "bindings": dict(bindings),
        "settings": _settings(config),
        "records": len(roots),
        "shards": shards,
        "merged_output": None,
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    return manifest


def _load_or_initialize(
    config: RunConfig,
    roots: Sequence[Mapping[str, Any]],
    compact_lines: Sequence[str],
    bindings: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_path = config.output_dir / "manifest.json"
    if not manifest_path.exists():
        return _initialize(config, roots, compact_lines, bindings)
    manifest = _load_json(manifest_path, label="run manifest")
    if manifest.get("schema") != RUN_SCHEMA:
        raise RelabelError(f"unexpected run manifest schema: {manifest.get('schema')!r}")
    if manifest.get("bindings") != bindings:
        raise RelabelError("roots/solver/pool binding changed; use a new output-dir")
    if manifest.get("settings") != _settings(config):
        raise RelabelError("relabel settings changed; use a new output-dir")
    if manifest.get("records") != len(roots):
        raise RelabelError("run manifest root count changed")
    expected_shards = (len(roots) + config.shard_size - 1) // config.shard_size
    if len(manifest.get("shards", [])) != expected_shards:
        raise RelabelError("run manifest shard count changed")
    return manifest


def _verify_shard_static(
    config: RunConfig,
    entry: Mapping[str, Any],
    shard: Mapping[str, Any],
    bindings: Mapping[str, Any],
) -> None:
    if shard.get("schema") != SHARD_SCHEMA:
        raise RelabelError(f"shard {entry.get('index')}: unexpected schema")
    for key in ("index", "root_start", "root_stop", "records"):
        if shard.get(key) != entry.get(key):
            raise RelabelError(f"shard {entry.get('index')}: {key} changed")
    if shard.get("settings") != _settings(config):
        raise RelabelError(f"shard {entry.get('index')}: settings changed")
    for key in ("roots", "solver", "pool"):
        binding_key = f"{key if key != 'roots' else 'source_roots'}_sha256"
        if shard.get(binding_key) != bindings[key]["sha256"]:
            raise RelabelError(f"shard {entry.get('index')}: {key} binding changed")
    input_path = Path(shard.get("input", {}).get("path", ""))
    if not input_path.is_file():
        raise RelabelError(f"shard {entry.get('index')}: input is missing")
    actual = sha256_file(input_path)
    if actual != entry.get("input_sha256") or actual != shard["input"].get("sha256"):
        raise RelabelError(f"shard {entry.get('index')}: input SHA256 mismatch")


def _command(config: RunConfig, shard: Mapping[str, Any], attempt_dir: Path) -> list[str]:
    return [
        str(config.solver),
        "teach-t2",
        "--roots-file",
        str(Path(shard["input"]["path"])),
        "--out-dir",
        str(attempt_dir),
        "--pool",
        str(config.pool),
        "--opponents",
        str(config.opponents),
        "--t3-draws",
        str(config.t3_draws),
        "--t4-draws",
        str(config.t4_draws),
        "--stream-offset",
        str(config.stream_offset),
        "--root-offset",
        str(shard["root_start"]),
    ]


def _persist_attempt(
    manifest_path: Path,
    shard: dict[str, Any],
    attempt: dict[str, Any],
    *,
    status: str,
) -> None:
    attempts = list(shard.get("attempts", []))
    attempts.append(attempt)
    shard["attempts"] = attempts
    shard["status"] = status
    shard["updated_at_utc"] = utc_now()
    atomic_write_json(manifest_path, shard)


def _run_shard(
    config: RunConfig,
    entry: Mapping[str, Any],
    roots: Sequence[Mapping[str, Any]],
    shard: dict[str, Any],
    manifest_path: Path,
) -> dict[str, Any]:
    attempt_index = len(shard.get("attempts", [])) + 1
    directory = Path(entry["directory"])
    attempt_dir = directory / "attempts" / f"attempt_{attempt_index:04d}"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    stdout_path = attempt_dir / "stdout.log"
    stderr_path = attempt_dir / "stderr.log"
    output_path = attempt_dir / "t2_labels.jsonl"
    command = _command(config, shard, attempt_dir)
    started_at = utc_now()
    started = time.monotonic()
    attempt: dict[str, Any] = {
        "attempt": attempt_index,
        "status": "running",
        "directory": str(attempt_dir),
        "command": command,
        "cwd": str(config.workspace_root),
        "started_at_utc": started_at,
        "finished_at_utc": None,
        "elapsed_seconds": None,
        "returncode": None,
        "stdout": {"path": str(stdout_path), "sha256": None, "bytes": None},
        "stderr": {"path": str(stderr_path), "sha256": None, "bytes": None},
        "output": None,
        "validation": None,
        "error": None,
        "runtime": {"RAYON_NUM_THREADS": os.environ.get("RAYON_NUM_THREADS")},
    }
    # Persist the attempt before the subprocess starts.  If this process dies,
    # the next invocation preserves this directory and starts a new attempt.
    _persist_attempt(manifest_path, shard, attempt, status="running")
    # _persist_attempt appended a copy; replace that running copy on completion.
    shard["attempts"] = shard["attempts"][:-1]
    returncode: int | None = None
    error_text: str | None = None
    try:
        with stdout_path.open("x", encoding="utf-8") as stdout_handle, stderr_path.open(
            "x", encoding="utf-8"
        ) as stderr_handle:
            completed = subprocess.run(
                command,
                cwd=str(config.workspace_root),
                stdout=stdout_handle,
                stderr=stderr_handle,
                timeout=config.timeout_seconds or None,
                check=False,
            )
        returncode = int(completed.returncode)
    except subprocess.TimeoutExpired as error:
        error_text = f"solver timed out after {error.timeout} seconds"
    except OSError as error:
        error_text = f"cannot run solver: {error}"
    attempt["finished_at_utc"] = utc_now()
    attempt["elapsed_seconds"] = time.monotonic() - started
    attempt["returncode"] = returncode
    for key, path in (("stdout", stdout_path), ("stderr", stderr_path)):
        if path.exists():
            attempt[key] = _artifact(path)
    if error_text is None and returncode != 0:
        error_text = f"solver exited with code {returncode}"
    if error_text is None:
        try:
            validation = validate_output(
                output_path,
                roots[entry["root_start"] : entry["root_stop"]],
                root_offset=entry["root_start"],
                opponents=config.opponents,
                t3_draws=config.t3_draws,
                stream_offset=config.stream_offset,
            )
        except RelabelError as error:
            error_text = str(error)
        else:
            attempt["validation"] = validation
            attempt["output"] = _artifact(output_path)
    if error_text is not None:
        attempt["status"] = "failed"
        attempt["error"] = error_text
        if output_path.exists():
            attempt["output"] = _artifact(output_path)
        _persist_attempt(manifest_path, shard, attempt, status="failed")
        raise RelabelError(f"shard {entry['index']} failed: {error_text}")
    attempt["status"] = "complete"
    attempts = list(shard.get("attempts", []))
    attempts.append(attempt)
    shard["attempts"] = attempts
    shard["status"] = "complete"
    shard["completed_output"] = attempt["output"]
    shard["validation"] = attempt["validation"]
    shard["updated_at_utc"] = utc_now()
    # Status, completed output, and validation become visible together.  A
    # crash must never leave a shard claiming complete without its commitment.
    atomic_write_json(manifest_path, shard)
    return shard


def _verify_complete_shard(
    config: RunConfig,
    entry: Mapping[str, Any],
    roots: Sequence[Mapping[str, Any]],
    shard: Mapping[str, Any],
) -> dict[str, Any]:
    completed = shard.get("completed_output")
    output_path = _verify_artifact(
        completed, label=f"shard {entry['index']} completed output"
    )
    attempts = shard.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise RelabelError(f"shard {entry['index']}: attempts are missing")
    attempt = attempts[-1]
    if not isinstance(attempt, dict) or attempt.get("status") != "complete":
        raise RelabelError(f"shard {entry['index']}: final attempt is not complete")
    if attempt.get("returncode") != 0:
        raise RelabelError(f"shard {entry['index']}: completed return code is not zero")
    attempt_dir = Path(attempt.get("directory", ""))
    if attempt.get("command") != _command(config, shard, attempt_dir):
        raise RelabelError(f"shard {entry['index']}: completed command changed")
    for key in ("stdout", "stderr", "output"):
        _verify_artifact(attempt.get(key), label=f"shard {entry['index']} {key}")
    if attempt.get("output") != completed:
        raise RelabelError(f"shard {entry['index']}: completed output commitment changed")
    validation = validate_output(
        output_path,
        roots[entry["root_start"] : entry["root_stop"]],
        root_offset=entry["root_start"],
        opponents=config.opponents,
        t3_draws=config.t3_draws,
        stream_offset=config.stream_offset,
    )
    if validation != shard.get("validation"):
        raise RelabelError(f"shard {entry['index']}: validation manifest changed")
    return validation


def _update_top_entry(
    top: dict[str, Any], index: int, *, status: str, output_sha256: str | None = None
) -> None:
    entry = top["shards"][index]
    entry["status"] = status
    if output_sha256 is not None:
        entry["output_sha256"] = output_sha256
    top["updated_at_utc"] = utc_now()


def _merge(
    config: RunConfig,
    top: dict[str, Any],
    roots: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    target = config.output_dir / "t2_labels.jsonl"
    existing = top.get("merged_output")
    if target.exists():
        validation = validate_output(
            target,
            roots,
            root_offset=0,
            opponents=config.opponents,
            t3_draws=config.t3_draws,
            stream_offset=config.stream_offset,
        )
        # Recover the narrow crash window after the atomic rename but before
        # the top manifest update.  Adoption is allowed only if the bytes are
        # exactly the ordered concatenation of every committed shard.
        expected = hashlib.sha256()
        expected_bytes = 0
        for entry in top["shards"]:
            shard = _load_json(Path(entry["manifest"]), label="shard manifest")
            source = _verify_artifact(
                shard.get("completed_output"),
                label=f"shard {entry['index']} completed output",
            )
            with source.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    expected.update(block)
                    expected_bytes += len(block)
        actual_sha = sha256_file(target)
        if actual_sha != expected.hexdigest() or target.stat().st_size != expected_bytes:
            raise RelabelError(f"refusing to overwrite untracked merged output: {target}")
        if isinstance(existing, dict):
            if existing.get("path") != str(target) or existing.get("sha256") != actual_sha:
                raise RelabelError(f"merged output manifest mismatch: {target}")
            if existing.get("validation") != validation:
                raise RelabelError(f"merged output validation changed: {target}")
            return existing
        artifact = _artifact(target)
        artifact["validation"] = validation
        artifact["merged_at_utc"] = utc_now()
        artifact["recovered_after_atomic_rename"] = True
        return artifact
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise RelabelError(f"refusing to replace existing merge temporary: {temporary}")
    try:
        with temporary.open("xb") as destination:
            for entry in top["shards"]:
                shard = _load_json(Path(entry["manifest"]), label="shard manifest")
                source = Path(shard["completed_output"]["path"])
                with source.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        destination.write(block)
            destination.flush()
            os.fsync(destination.fileno())
        validation = validate_output(
            temporary,
            roots,
            root_offset=0,
            opponents=config.opponents,
            t3_draws=config.t3_draws,
            stream_offset=config.stream_offset,
        )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    artifact = _artifact(target)
    artifact["validation"] = validation
    artifact["merged_at_utc"] = utc_now()
    return artifact


def run(config: RunConfig) -> dict[str, Any]:
    config = _resolved(config)
    roots, compact_lines = read_roots(config.roots)
    bindings = {
        "roots": _artifact(config.roots),
        "solver": _artifact(config.solver),
        "pool": _artifact(config.pool),
    }
    top = _load_or_initialize(config, roots, compact_lines, bindings)
    top_path = config.output_dir / "manifest.json"
    runnable = 0
    failures: list[str] = []
    for entry in top["shards"]:
        shard_path = Path(entry["manifest"])
        shard = _load_json(shard_path, label="shard manifest")
        _verify_shard_static(config, entry, shard, bindings)
        if shard.get("status") == "complete":
            validation = _verify_complete_shard(config, entry, roots, shard)
            _update_top_entry(
                top, entry["index"], status="complete", output_sha256=validation["sha256"]
            )
            atomic_write_json(top_path, top)
            continue
        if config.max_shards and runnable >= config.max_shards:
            continue
        runnable += 1
        try:
            shard = _run_shard(config, entry, roots, shard, shard_path)
        except RelabelError as error:
            failures.append(str(error))
            _update_top_entry(top, entry["index"], status="failed")
            top["status"] = "failed"
            top["failures"] = list(failures)
            atomic_write_json(top_path, top)
            if not config.keep_going:
                raise
        else:
            validation = _verify_complete_shard(config, entry, roots, shard)
            _update_top_entry(
                top, entry["index"], status="complete", output_sha256=validation["sha256"]
            )
            top["status"] = "running"
            atomic_write_json(top_path, top)
    complete = all(entry.get("status") == "complete" for entry in top["shards"])
    if failures:
        top["status"] = "failed"
        top["failures"] = failures
    elif complete:
        top["merged_output"] = _merge(config, top, roots)
        top["status"] = "complete"
        top["completed_at_utc"] = utc_now()
        top.pop("failures", None)
    else:
        top["status"] = "incomplete"
        top.pop("failures", None)
    top["updated_at_utc"] = utc_now()
    atomic_write_json(top_path, top)
    return top


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--solver", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--shard-size", type=int, default=25)
    parser.add_argument("--opponents", type=int, default=120)
    parser.add_argument("--t3-draws", type=int, default=96)
    parser.add_argument("--t4-draws", type=int, default=24)
    parser.add_argument("--stream-offset", type=int, default=0)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--timeout-seconds", type=float, default=0.0)
    parser.add_argument(
        "--max-shards",
        type=int,
        default=0,
        help="run at most this many incomplete shards in this invocation; 0 is all",
    )
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    config = RunConfig(
        roots=args.roots,
        output_dir=args.out_dir,
        solver=args.solver,
        pool=args.pool,
        shard_size=args.shard_size,
        opponents=args.opponents,
        t3_draws=args.t3_draws,
        t4_draws=args.t4_draws,
        stream_offset=args.stream_offset,
        workspace_root=args.workspace_root,
        timeout_seconds=args.timeout_seconds,
        max_shards=args.max_shards,
        keep_going=args.keep_going,
    )
    try:
        result = run(config)
    except RelabelError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if result["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
