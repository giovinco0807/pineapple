"""Run active-teacher generation in restartable JSONL chunks.

This is a thin orchestration layer around ``ai.training.generate_active_teacher``.
It is meant for slow high-confidence relabeling runs, where a monolithic command
is too fragile.  Each chunk has its own input, output, summary, and ``.done``
marker; completed chunks are skipped on resume.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


def parse_turns(raw: str) -> set[int]:
    return {int(part) for part in raw.split(",") if part.strip()}


def iter_targets(path: Path, turns: set[int]) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if int(item.get("turn", -1)) not in turns:
                continue
            item.setdefault("chunk_source_line", line_no)
            yield item


def chunked(items: list[dict[str, Any]], size: int) -> Iterable[tuple[int, list[dict[str, Any]]]]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    for start in range(0, len(items), size):
        yield start // size, items[start : start + size]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8-sig") as f:
        return sum(1 for line in f if line.strip())


def load_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def merge_done_chunks(args: argparse.Namespace, chunk_dir: Path, total_chunks: int) -> dict[str, Any]:
    merged_output = Path(args.merged_output) if args.merged_output else chunk_dir / "merged.jsonl"
    merged_output.parent.mkdir(parents=True, exist_ok=True)
    aggregate = Counter()
    written = 0
    done_chunks = 0

    with merged_output.open("w", encoding="utf-8") as dst:
        for chunk_idx in range(total_chunks):
            stem = f"chunk_{chunk_idx:05d}"
            done_path = chunk_dir / f"{stem}.done"
            output_path = chunk_dir / f"{stem}.jsonl"
            summary_path = output_path.with_suffix(".summary.json")
            if not done_path.exists() or not output_path.exists():
                continue
            done_chunks += 1
            summary = load_summary(summary_path)
            aggregate["targets"] += int(summary.get("targets", 0) or 0)
            aggregate["written"] += int(summary.get("written", 0) or 0)
            aggregate["skipped"] += int(summary.get("skipped", 0) or 0)
            for turn, count in (summary.get("turns") or {}).items():
                aggregate[f"turn_{turn}"] += int(count)
            for mode, count in (summary.get("eval_modes") or {}).items():
                aggregate[f"eval_mode_{mode}"] += int(count)
            with output_path.open("r", encoding="utf-8-sig") as src:
                for line in src:
                    if line.strip():
                        dst.write(line if line.endswith("\n") else line + "\n")
                        written += 1

    report = {
        "input": args.input,
        "chunk_dir": str(chunk_dir),
        "merged_output": str(merged_output),
        "total_chunks": total_chunks,
        "done_chunks": done_chunks,
        "merged_records": written,
        "aggregate": dict(aggregate),
    }
    (chunk_dir / "status.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def run_chunk(args: argparse.Namespace, chunk_dir: Path, chunk_idx: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    stem = f"chunk_{chunk_idx:05d}"
    input_path = chunk_dir / f"{stem}.input.jsonl"
    output_path = chunk_dir / f"{stem}.jsonl"
    done_path = chunk_dir / f"{stem}.done"
    meta_path = chunk_dir / f"{stem}.meta.json"

    if done_path.exists() and not args.force:
        return {"chunk": chunk_idx, "status": "skipped_done", "output": str(output_path), "written": count_jsonl(output_path)}

    write_jsonl(input_path, rows)
    cmd = [
        args.python,
        "-m",
        "ai.training.generate_active_teacher",
        str(input_path),
        "--output",
        str(output_path),
        "--sims",
        str(args.sims),
        "--turns",
        args.turns,
        "--mc-turns",
        args.mc_turns,
        "--workers",
        str(args.workers),
        "--progress-every",
        str(args.progress_every),
    ]
    if args.engine_path:
        cmd.extend(["--engine-path", args.engine_path])
    if args.batch_engine:
        cmd.append("--batch-engine")
        cmd.extend(["--batch-timeout", str(args.batch_timeout)])
    if args.print_errors:
        cmd.append("--print-errors")
    if args.dry_run:
        return {"chunk": chunk_idx, "status": "dry_run", "cmd": cmd, "rows": len(rows)}

    started = time.time()
    completed = subprocess.run(
        cmd,
        cwd=str(Path.cwd()),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=args.chunk_timeout,
    )
    elapsed_s = time.time() - started
    meta = {
        "chunk": chunk_idx,
        "input": str(input_path),
        "output": str(output_path),
        "rows": len(rows),
        "cmd": cmd,
        "returncode": completed.returncode,
        "elapsed_s": round(elapsed_s, 3),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    if completed.returncode != 0:
        return {**meta, "status": "failed", "written": count_jsonl(output_path)}

    summary_path = output_path.with_suffix(".summary.json")
    summary = load_summary(summary_path)
    done_path.write_text(json.dumps({**meta, "summary": summary}, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "chunk": chunk_idx,
        "status": "done",
        "output": str(output_path),
        "written": int(summary.get("written", count_jsonl(output_path)) or 0),
        "elapsed_s": round(elapsed_s, 3),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    chunk_dir = Path(args.output_dir)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    turns = parse_turns(args.turns)
    targets = list(iter_targets(Path(args.input), turns))
    if args.max_records:
        targets = targets[: args.max_records]
    chunks = list(chunked(targets, args.chunk_size))
    if args.max_chunks:
        runnable = chunks[: args.max_chunks]
    else:
        runnable = chunks

    print(
        json.dumps(
            {
                "input": args.input,
                "targets": len(targets),
                "turns": sorted(turns),
                "chunk_size": args.chunk_size,
                "total_chunks": len(chunks),
                "running_chunks": len(runnable),
                "output_dir": str(chunk_dir),
                "sims": args.sims,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    results = []
    for chunk_idx, rows in runnable:
        result = run_chunk(args, chunk_dir, chunk_idx, rows)
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if result.get("status") == "failed" and not args.keep_going:
            break

    status = merge_done_chunks(args, chunk_dir, len(chunks))
    status["last_run"] = results
    (chunk_dir / "last_run.json").write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(status, indent=2, ensure_ascii=False), flush=True)
    return status


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run active teacher generation in restartable chunks")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--merged-output", default="")
    parser.add_argument("--turns", default="1,2")
    parser.add_argument("--mc-turns", default="1,2")
    parser.add_argument("--sims", type=int, default=3000)
    parser.add_argument("--chunk-size", type=int, default=2)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--engine-path", default="")
    parser.add_argument("--batch-engine", action="store_true")
    parser.add_argument("--batch-timeout", type=int, default=86400)
    parser.add_argument("--chunk-timeout", type=int, default=86400)
    parser.add_argument("--print-errors", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    run(args)


if __name__ == "__main__":
    main()
