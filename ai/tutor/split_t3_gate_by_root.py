"""Split branch-expanded T3 gate JSONL into root-disjoint partitions.

Rows produced directly by ``build_branch_expanded_t3_targets`` carry their
root in ``branch.root_index``.  The exact-teacher converter intentionally
keeps a smaller context and currently drops ``branch``; in that format the
one-based ``source_line`` still identifies the root.  This splitter supports
both representations and writes each original JSON line unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, TextIO


DEFAULT_ROOT_SPECS = {
    "mine": "0-74",
    "dev": "75-99",
    "final": "100-124",
}


def _integer(value: Any, *, field: str, line_number: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer at line {line_number}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer at line {line_number}") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"{field} must be an integer at line {line_number}")
    if isinstance(value, str) and value.strip() != str(parsed):
        raise ValueError(f"{field} must be an integer at line {line_number}")
    return parsed


def root_index(record: Mapping[str, Any], *, line_number: int) -> int:
    """Return the zero-based generation root for an input or teacher row."""
    branch = record.get("branch")
    if isinstance(branch, Mapping) and branch.get("root_index") is not None:
        root = _integer(
            branch.get("root_index"), field="branch.root_index", line_number=line_number
        )
    elif record.get("source_line") is not None:
        source_line = _integer(
            record.get("source_line"), field="source_line", line_number=line_number
        )
        if source_line < 1:
            raise ValueError(f"source_line must be >= 1 at line {line_number}")
        root = source_line - 1
    else:
        raise ValueError(
            f"missing branch.root_index and source_line at line {line_number}"
        )
    if root < 0:
        raise ValueError(f"root index must be >= 0 at line {line_number}")
    return root


def position(record: Mapping[str, Any]) -> str:
    value = record.get("position")
    if value is not None:
        normalized = str(value).strip().lower()
        if normalized in {"bb", "btn"}:
            return normalized
        return normalized or "unknown"
    if "is_btn" in record:
        return "btn" if bool(record.get("is_btn")) else "bb"
    return "unknown"


def parse_root_spec(spec: str) -> set[int]:
    """Parse comma-separated roots and inclusive ranges (for example 0-74)."""
    roots: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            raise ValueError(f"empty root token in {spec!r}")
        if "-" in token:
            parts = token.split("-")
            if len(parts) != 2:
                raise ValueError(f"invalid root range {token!r}")
            try:
                start, end = (int(part.strip()) for part in parts)
            except ValueError as exc:
                raise ValueError(f"invalid root range {token!r}") from exc
            if start < 0 or end < start:
                raise ValueError(f"invalid root range {token!r}")
            roots.update(range(start, end + 1))
        else:
            try:
                root = int(token)
            except ValueError as exc:
                raise ValueError(f"invalid root {token!r}") from exc
            if root < 0:
                raise ValueError(f"invalid root {token!r}")
            roots.add(root)
    if not roots:
        raise ValueError("root specification must not be empty")
    return roots


def validate_root_partitions(partitions: Mapping[str, set[int]]) -> None:
    owners: dict[int, str] = {}
    for split_name, roots in partitions.items():
        if not roots:
            raise ValueError(f"{split_name} root set must not be empty")
        for root in roots:
            previous = owners.get(root)
            if previous is not None:
                raise ValueError(
                    f"root {root} is assigned to both {previous} and {split_name}"
                )
            owners[root] = split_name


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _output_summary(path: Path, records: int, roots: set[int], positions: Counter[str]) -> dict[str, Any]:
    return {
        "path": str(path),
        "records": records,
        "positions": {
            "bb": int(positions.get("bb", 0)),
            "btn": int(positions.get("btn", 0)),
            **{
                key: int(value)
                for key, value in sorted(positions.items())
                if key not in {"bb", "btn"}
            },
        },
        "root_count": len(roots),
        "roots": sorted(roots),
        "sha256": _sha256(path),
    }


def _close_all(handles: Iterable[TextIO]) -> None:
    for handle in handles:
        handle.close()


def split_jsonl(
    input_path: Path,
    output_dir: Path,
    *,
    prefix: str | None = None,
    root_specs: Mapping[str, str] = DEFAULT_ROOT_SPECS,
    expected_counts: Mapping[str, int] | None = None,
    summary_path: Path | None = None,
) -> dict[str, Any]:
    """Split ``input_path`` transactionally and return its reproducibility summary."""
    partitions = {name: parse_root_spec(spec) for name, spec in root_specs.items()}
    if set(partitions) != {"mine", "dev", "final"}:
        raise ValueError("root_specs must define exactly mine, dev, and final")
    validate_root_partitions(partitions)

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    prefix = prefix or input_path.stem
    if not prefix or Path(prefix).name != prefix:
        raise ValueError("prefix must be a non-empty file-name component")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        name: output_dir / f"{prefix}.{name}.jsonl" for name in partitions
    }
    temp_paths = {
        name: path.with_name(path.name + ".tmp") for name, path in output_paths.items()
    }
    handles: dict[str, TextIO] = {}
    counts: Counter[str] = Counter()
    positions: dict[str, Counter[str]] = {name: Counter() for name in partitions}
    observed_roots: dict[str, set[int]] = {name: set() for name in partitions}
    root_owner = {
        root: split_name
        for split_name, roots in partitions.items()
        for root in roots
    }
    input_records = 0

    try:
        handles = {
            name: path.open("w", encoding="utf-8", newline="")
            for name, path in temp_paths.items()
        }
        with input_path.open("r", encoding="utf-8-sig", newline="") as source:
            for line_number, raw_line in enumerate(source, 1):
                if not raw_line.strip():
                    continue
                try:
                    record = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid JSON in {input_path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(record, dict):
                    raise ValueError(
                        f"expected JSON object in {input_path}:{line_number}"
                    )
                root = root_index(record, line_number=line_number)
                split_name = root_owner.get(root)
                if split_name is None:
                    raise ValueError(
                        f"root {root} at line {line_number} is outside all partitions"
                    )
                handles[split_name].write(raw_line)
                counts[split_name] += 1
                positions[split_name][position(record)] += 1
                observed_roots[split_name].add(root)
                input_records += 1
        _close_all(handles.values())
        handles.clear()

        actual_counts = {
            "total": input_records,
            **{name: int(counts[name]) for name in partitions},
        }
        for name, expected in (expected_counts or {}).items():
            if name not in actual_counts:
                raise ValueError(f"unknown expected-count key {name!r}")
            if actual_counts[name] != int(expected):
                raise ValueError(
                    f"expected {name}={int(expected)}, got {actual_counts[name]}"
                )

        for name in partitions:
            temp_paths[name].replace(output_paths[name])

        outputs = {
            name: _output_summary(
                output_paths[name], int(counts[name]), observed_roots[name], positions[name]
            )
            for name in partitions
        }
        summary = {
            "input": str(input_path),
            "input_sha256": _sha256(input_path),
            "records": input_records,
            "root_partitions": {
                name: sorted(roots) for name, roots in partitions.items()
            },
            "outputs": outputs,
            "checks": {
                "configured_roots_disjoint": True,
                "observed_roots_disjoint": (
                    observed_roots["mine"].isdisjoint(observed_roots["dev"])
                    and observed_roots["mine"].isdisjoint(observed_roots["final"])
                    and observed_roots["dev"].isdisjoint(observed_roots["final"])
                ),
                "all_records_assigned": sum(counts.values()) == input_records,
                "expected_counts": {
                    key: int(value) for key, value in (expected_counts or {}).items()
                },
            },
        }
        if not summary["checks"]["observed_roots_disjoint"]:
            raise AssertionError("observed roots overlap between splits")
        if not summary["checks"]["all_records_assigned"]:
            raise AssertionError("not all input records were assigned")

        destination = summary_path or output_dir / f"{prefix}.split_summary.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        summary["summary_path"] = str(destination)
        return summary
    finally:
        _close_all(handles.values())
        for temp_path in temp_paths.values():
            if temp_path.exists():
                temp_path.unlink()


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix")
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--mine-roots", default=DEFAULT_ROOT_SPECS["mine"])
    parser.add_argument("--dev-roots", default=DEFAULT_ROOT_SPECS["dev"])
    parser.add_argument("--final-roots", default=DEFAULT_ROOT_SPECS["final"])
    parser.add_argument("--expect-total", type=int)
    parser.add_argument("--expect-mine", type=int)
    parser.add_argument("--expect-dev", type=int)
    parser.add_argument("--expect-final", type=int)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    expected_counts = {
        name: value
        for name, value in {
            "total": args.expect_total,
            "mine": args.expect_mine,
            "dev": args.expect_dev,
            "final": args.expect_final,
        }.items()
        if value is not None
    }
    summary = split_jsonl(
        args.input,
        args.output_dir,
        prefix=args.prefix,
        root_specs={
            "mine": args.mine_roots,
            "dev": args.dev_roots,
            "final": args.final_roots,
        },
        expected_counts=expected_counts,
        summary_path=args.summary,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
