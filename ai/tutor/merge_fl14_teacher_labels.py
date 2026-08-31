"""Replace low-sample FL14 teacher values with averaged relabel streams.

Merging happens before feature encoding because the encoded NPZ files retain
root ids but not action keys.  A relabel is therefore allowed to replace an
existing ``(stable root id, action_key)`` only; appending roots or duplicating
the low- and high-sample versions would make within-root regret ambiguous.

Example:
    python -m ai.tutor.merge_fl14_teacher_labels \
        --base D:/ofc_data/lap3/t2_labels.jsonl \
        --relabel D:/ofc_data/lap3/high_a/t2_labels.jsonl \
        --relabel D:/ofc_data/lap3/high_b/t2_labels.jsonl \
        --street t2 --split fit \
        --output D:/ofc_data/lap3/t2_labels_hq_fit.jsonl
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Iterable

from ai.tutor.encode_fl14_teacher import split_of, stable_root_id

SCHEMA = "ofc_fl14_teacher_label_merge/v1"
CONTEXT_FIELDS = ("board", "dead", "draw")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_id(record: dict, path: Path, line_number: int) -> tuple[int, str]:
    if "id" not in record:
        raise ValueError(f"{path}:{line_number}: record has no global id")
    raw = str(record["id"])
    return stable_root_id(record), raw


def _action_index(record: dict, path: Path, line_number: int) -> dict[str, dict]:
    actions = record.get("actions")
    if not isinstance(actions, list) or not actions:
        raise ValueError(f"{path}:{line_number}: record has no actions")
    indexed: dict[str, dict] = {}
    for position, action in enumerate(actions):
        key = action.get("action_key")
        if not isinstance(key, str) or not key:
            raise ValueError(
                f"{path}:{line_number}: action {position} has no action_key"
            )
        if key in indexed:
            raise ValueError(
                f"{path}:{line_number}: duplicate action_key {key!r}"
            )
        value = action.get("value")
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(
                f"{path}:{line_number}: action {key!r} has a non-finite value"
            )
        indexed[key] = action
    return indexed


def read_records(path: Path) -> tuple[list[dict], dict[int, dict]]:
    """Read one JSONL source and reject duplicate/colliding decision ids."""
    ordered: list[dict] = []
    indexed: dict[int, dict] = {}
    raw_ids: dict[int, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            root, raw = _source_id(record, path, line_number)
            previous = raw_ids.get(root)
            if previous is not None:
                detail = "duplicate id" if previous == raw else "stable id collision"
                raise ValueError(
                    f"{path}:{line_number}: {detail}: {previous!r} and {raw!r}"
                )
            _action_index(record, path, line_number)
            raw_ids[root] = raw
            indexed[root] = record
            ordered.append(record)
    if not ordered:
        raise ValueError(f"{path}: no records")
    return ordered, indexed


def _same_context(base: dict, relabel: dict) -> bool:
    return all(base.get(field) == relabel.get(field) for field in CONTEXT_FIELDS)


def _common_metadata(
    actions: list[dict], field: str, *, root: int, action_key: str
) -> object | None:
    values = [action.get(field) for action in actions]
    present = [value is not None for value in values]
    if any(present) and not all(present):
        raise ValueError(
            f"root {root} action {action_key!r}: relabel streams disagree on {field} presence"
        )
    if not any(present):
        return None
    if any(value != values[0] for value in values[1:]):
        raise ValueError(
            f"root {root} action {action_key!r}: relabel streams disagree on {field}"
        )
    return values[0]


def _write_jsonl_atomic(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _write_text_atomic(path: Path, text: str) -> None:
    """Replace a small text artifact without exposing a partial write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def merge_teacher_labels(
    *,
    base_path: Path,
    relabel_paths: list[Path],
    output_path: Path,
    manifest_path: Path,
    street: str,
    expected_split: str,
) -> dict:
    """Validate 2+ complete relabel streams and replace matching base values."""
    if len(relabel_paths) < 2:
        raise ValueError("at least two independent relabel streams are required")
    if street not in ("t2", "t3"):
        raise ValueError("street must be t2 or t3")
    if expected_split not in ("fit", "dev", "test"):
        raise ValueError("split must be fit, dev, or test")

    inputs = [base_path, *relabel_paths]
    resolved_inputs = [path.resolve(strict=True) for path in inputs]
    if len(set(resolved_inputs)) != len(resolved_inputs):
        raise ValueError("base and relabel inputs must be distinct files")
    if output_path.resolve() in set(resolved_inputs):
        raise ValueError("output must not overwrite an input")
    if manifest_path.resolve() in set(resolved_inputs) or manifest_path.resolve() == output_path.resolve():
        raise ValueError("manifest must be distinct from inputs and output")

    base_order, base_by_root = read_records(resolved_inputs[0])
    streams = [read_records(path)[1] for path in resolved_inputs[1:]]
    expected_roots = set(streams[0])
    for position, stream in enumerate(streams[1:], 2):
        missing = sorted(expected_roots - set(stream))
        extra = sorted(set(stream) - expected_roots)
        if missing or extra:
            raise ValueError(
                f"relabel stream {position} root set differs: missing={missing[:5]} extra={extra[:5]}"
            )
    extra_from_base = sorted(expected_roots - set(base_by_root))
    if extra_from_base:
        raise ValueError(
            f"append is forbidden; relabel roots are absent from base: {extra_from_base[:5]}"
        )

    replacements: dict[int, dict] = {}
    replacement_actions = 0
    best_action_agreement = 0
    for root in sorted(expected_roots):
        actual_split = split_of(root, street)
        if actual_split != expected_split:
            raise ValueError(
                f"root {root} belongs to {actual_split}, outside requested {expected_split} split"
            )
        base = base_by_root[root]
        relabels = [stream[root] for stream in streams]
        for relabel in relabels:
            if str(relabel["id"]) != str(base["id"]):
                raise ValueError(f"root {root}: raw global id differs from base")
            if not _same_context(base, relabel):
                raise ValueError(f"root {root}: board/dead/draw context differs")

        base_actions = _action_index(base, resolved_inputs[0], 0)
        action_maps = [
            _action_index(record, path, 0)
            for record, path in zip(relabels, resolved_inputs[1:])
        ]
        base_keys = set(base_actions)
        for position, action_map in enumerate(action_maps, 1):
            missing = sorted(base_keys - set(action_map))
            extra = sorted(set(action_map) - base_keys)
            if missing or extra:
                raise ValueError(
                    f"root {root} relabel stream {position} action set differs: "
                    f"missing={missing[:5]} extra={extra[:5]}"
                )

        opponents = [record.get("opponents") for record in relabels]
        if any(value != opponents[0] for value in opponents[1:]):
            raise ValueError(f"root {root}: relabel streams disagree on opponents")

        merged = copy.deepcopy(base)
        if opponents[0] is not None:
            merged["opponents"] = opponents[0]
        for action in merged["actions"]:
            key = action["action_key"]
            high_actions = [action_map[key] for action_map in action_maps]
            action["value"] = math.fsum(
                float(high_action["value"]) for high_action in high_actions
            ) / len(high_actions)
            t3_draws = _common_metadata(
                high_actions, "t3_draws", root=root, action_key=key
            )
            if t3_draws is not None:
                action["t3_draws"] = t3_draws
        merged["label_merge"] = {
            "aggregation": "equal_mean_relabel_streams",
            "base_value_included": False,
            "schema": SCHEMA,
            "streams": len(streams),
        }
        winners = [
            max(action_map, key=lambda key: float(action_map[key]["value"]))
            for action_map in action_maps
        ]
        if all(winner == winners[0] for winner in winners[1:]):
            best_action_agreement += 1
        replacements[root] = merged
        replacement_actions += len(base_keys)

    output_records = [
        replacements.get(stable_root_id(record), record) for record in base_order
    ]
    _write_jsonl_atomic(output_path, output_records)

    sorted_roots = "".join(f"{root}\n" for root in sorted(expected_roots)).encode()
    sources = [
        {
            "path": str(path),
            "records": len(base_order) if position == 0 else len(streams[position - 1]),
            "role": "base" if position == 0 else "relabel",
            "sha256": sha256_file(path),
        }
        for position, path in enumerate(resolved_inputs)
    ]
    manifest = {
        "append_allowed": False,
        "context_fields": list(CONTEXT_FIELDS),
        "expected_split": expected_split,
        "output": {
            "path": str(output_path.resolve()),
            "records": len(output_records),
            "sha256": sha256_file(output_path),
        },
        "replacement_actions": replacement_actions,
        "replacement_root_ids_sha256": hashlib.sha256(sorted_roots).hexdigest(),
        "replacement_roots": len(expected_roots),
        "relabel_best_action_agreement_roots": best_action_agreement,
        "relabel_streams": len(streams),
        "schema": SCHEMA,
        "sources": sources,
        "street": street,
        "value_aggregation": "equal_mean_relabel_streams_base_excluded",
    }
    _write_text_atomic(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument(
        "--relabel",
        type=Path,
        action="append",
        required=True,
        help="independent high-sample JSONL; pass at least twice",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="default: <output>.manifest.json",
    )
    parser.add_argument("--street", choices=("t2", "t3"), default="t2")
    parser.add_argument("--split", choices=("fit", "dev", "test"), required=True)
    args = parser.parse_args()
    manifest_path = args.manifest or args.output.with_suffix(
        args.output.suffix + ".manifest.json"
    )
    manifest = merge_teacher_labels(
        base_path=args.base,
        relabel_paths=args.relabel,
        output_path=args.output,
        manifest_path=manifest_path,
        street=args.street,
        expected_split=args.split,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
