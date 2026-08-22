"""Assemble labelgen position files into one labels.jsonl.

Each position_*.json is a single JSON object with no trailing newline, so they
cannot be concatenated with cat -- doing so runs two objects together on one
line. Read them individually instead.

Offsets must be unique. A repeated offset is only tolerated when the two files
agree exactly, which is what a re-run shard produces (seeds are a function of
the offset, so a shard that ran twice writes identical labels). A repeated
offset with different content means two runs used overlapping hand seeds, which
would silently double-count positions, and is refused.

Usage: assemble_labels.py <raw-dir> <out.jsonl>
"""
import json
import pathlib
import sys


def main():
    raw = pathlib.Path(sys.argv[1])
    out = pathlib.Path(sys.argv[2])

    seen: dict[int, str] = {}
    canon: dict[int, str] = {}
    files = 0
    for path in sorted(raw.rglob("position_*.json")):
        files += 1
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            raise SystemExit(f"{path}: empty")
        record = json.loads(text)
        offset = record["offset"]
        key = json.dumps(record, sort_keys=True, separators=(",", ":"))
        if offset in canon and canon[offset] != key:
            raise SystemExit(
                f"offset {offset} appears twice with different content "
                f"({path}); overlapping hand seeds would double-count positions"
            )
        canon[offset] = key
        seen[offset] = text

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as stream:
        for offset in sorted(seen):
            stream.write(seen[offset])
            stream.write("\n")

    duplicates = files - len(seen)
    print(f"files {files}  unique offsets {len(seen)}  "
          f"exact duplicates {duplicates}")
    print(f"offset range {min(seen)}..{max(seen)}")
    missing = [o for o in range(min(seen), max(seen) + 1) if o not in seen]
    if missing:
        print(f"gaps inside the range: {len(missing)} "
              f"(first few: {missing[:10]})")


if __name__ == "__main__":
    main()
