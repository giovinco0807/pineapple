"""Run oracle-first teacher generation for late and T2 tutor positions.

This is the slow/accurate entry point.  It deliberately separates label
creation from runtime speed work:

* T3/T4 rows are evaluated with full exact late-turn enumeration.
* T2 rows are evaluated with the Rust capped-exact solver at multiple caps.
* Stable T2 labels are emitted only when the selected cap pair agrees on Top1.

The generated files remain compatible with the existing action-value teacher
conversion scripts.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

from ai.tutor import build_t2_stable_oracle_labels
from ai.tutor import generate_exact_late_teacher
from ai.tutor import run_t2_exact_oracle


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for index, line in enumerate(f):
            if not line.strip():
                continue
            record = json.loads(line)
            if isinstance(record, dict):
                record.setdefault("oracle_first_original_index", index)
            records.append(record)
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def parse_int_csv(value: str) -> list[int]:
    out: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def apply_record_window(records: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.record_indices.strip():
        indices = parse_int_csv(args.record_indices)
        return [records[index] for index in indices if 0 <= index < len(records)]
    start = max(int(args.record_offset), 0)
    if args.record_limit > 0:
        return records[start : start + int(args.record_limit)]
    return records[start:]


def cap_suffix(cap: int) -> str:
    return "exact" if cap == 0 else f"cap{cap}"


def t2_exact_path(out_dir: Path, cap: int, limit: int) -> Path:
    return out_dir / "t2_caps" / f"t2_oracle_{cap_suffix(cap)}_limit{limit}.jsonl"


def run_late_exact(args: argparse.Namespace, input_path: Path, out_dir: Path) -> dict[str, Any] | None:
    if args.skip_late:
        return None
    turns = set(parse_int_csv(args.late_turns))
    records = apply_record_window(
        [record for record in load_jsonl(input_path) if int(record.get("turn", -1)) in turns],
        args,
    )
    late_input = out_dir / "inputs" / "late_t3t4_input.jsonl"
    late_count = write_jsonl(late_input, records)
    output = out_dir / "late_exact" / "exact_late_teacher.jsonl"
    if late_count == 0:
        return {
            "input": str(late_input),
            "output": str(output),
            "targets": 0,
            "written": 0,
            "skipped": 0,
            "source": "exact_late",
            "status": "no_matching_turns",
        }
    if args.dry_run:
        return {
            "input": str(late_input),
            "output": str(output),
            "targets": late_count,
            "written": 0,
            "skipped": 0,
            "source": "exact_late",
            "status": "dry_run",
        }
    return generate_exact_late_teacher.generate(
        SimpleNamespace(
            input=str(late_input),
            output=str(output),
            turns=args.late_turns,
            workers=args.late_workers,
            max_records=args.late_max_records,
            progress_every=args.progress_every,
            print_errors=args.print_errors,
        )
    )


def write_t2_report(out_dir: Path, cap: int, limit: int, report: dict[str, Any]) -> dict[str, str]:
    prefix = out_dir / "t2_caps" / f"t2_oracle_{cap_suffix(cap)}_limit{limit}"
    summary_path = prefix.with_suffix(".summary.json")
    markdown_path = prefix.with_suffix(".summary.md")
    misses_path = prefix.with_suffix(".misses.jsonl")
    with misses_path.open("w", encoding="utf-8") as f:
        for miss in report["misses"]:
            f.write(json.dumps(miss, ensure_ascii=False, separators=(",", ":")) + "\n")
    report["misses_path"] = str(misses_path)
    summary_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    run_t2_exact_oracle.write_markdown(markdown_path, report)
    return {
        "summary": str(summary_path),
        "markdown": str(markdown_path),
        "misses": str(misses_path),
    }


def run_t2_cap(
    args: argparse.Namespace,
    input_path: Path,
    out_dir: Path,
    cap: int,
) -> dict[str, Any]:
    exact_path = t2_exact_path(out_dir, cap, args.t2_limit)
    exact_path.parent.mkdir(parents=True, exist_ok=True)
    reused_existing = bool(args.skip_existing and exact_path.exists())
    if args.dry_run:
        return {
            "cap": cap,
            "exact_output": str(exact_path),
            "status": "dry_run",
        }
    if not reused_existing:
        run_t2_exact_oracle.run_solver(
            SimpleNamespace(
                input=str(input_path),
                skip=0,
                limit=args.t2_limit,
                top_n=args.t2_top_n,
                t2_draw_limit=cap,
                source_candidate_top_k=0,
                manifest=args.t2_manifest,
                binary=args.t2_binary,
                fl_config=args.fl_config,
            ),
            exact_path,
        )
    report = run_t2_exact_oracle.compare(input_path, exact_path, miss_top_k=args.miss_top_k)
    report_files = write_t2_report(out_dir, cap, args.t2_limit, report)
    return {
        "cap": cap,
        "exact_output": str(exact_path),
        "records": int(report.get("records", 0)),
        "same_top1": int(report.get("same_top1", 0)),
        "changed_top1": int(report.get("changed_top1", 0)),
        "same_top1_rate": float(report.get("same_top1_rate", 0.0)),
        "avg_exact_elapsed_ms": float(report.get("avg_exact_elapsed_ms", 0.0)),
        "avg_estimated_full_t2_elapsed_ms_from_cap": float(
            report.get("avg_estimated_full_t2_elapsed_ms_from_cap", 0.0)
        ),
        "max_exact_regret_of_source_top1": float(report.get("max_exact_regret_of_source_top1", 0.0)),
        "report_files": report_files,
        "status": "reused" if reused_existing else "ran",
    }


def choose_stable_caps(caps: list[int], args: argparse.Namespace) -> tuple[int, int] | None:
    if args.stable_weak_cap >= 0 or args.stable_strong_cap >= 0:
        if args.stable_weak_cap < 0 or args.stable_strong_cap < 0:
            raise ValueError("Both --stable-weak-cap and --stable-strong-cap are required when either is set")
        return args.stable_weak_cap, args.stable_strong_cap
    finite_caps = [cap for cap in caps if cap > 0]
    if len(finite_caps) < 2:
        return None
    return finite_caps[-2], finite_caps[-1]


def build_stable_t2(
    args: argparse.Namespace,
    source_path: Path,
    out_dir: Path,
    caps: list[int],
) -> dict[str, Any] | None:
    if args.skip_t2 or args.dry_run:
        return None
    pair = choose_stable_caps(caps, args)
    if pair is None:
        return None
    weak_cap, strong_cap = pair
    weak = t2_exact_path(out_dir, weak_cap, args.t2_limit)
    strong = t2_exact_path(out_dir, strong_cap, args.t2_limit)
    output = out_dir / "t2_stable" / f"stable_t2_oracle_{cap_suffix(weak_cap)}_{cap_suffix(strong_cap)}.jsonl"
    unstable = output.with_suffix(".unstable.jsonl")
    return build_t2_stable_oracle_labels.build(
        SimpleNamespace(
            source=str(source_path),
            weak=str(weak),
            strong=str(strong),
            output=str(output),
            unstable_output=str(unstable),
            max_records=args.stable_max_records,
        )
    )


def run_t2(args: argparse.Namespace, input_path: Path, out_dir: Path) -> dict[str, Any] | None:
    if args.skip_t2:
        return None
    records = apply_record_window(
        [record for record in load_jsonl(input_path) if int(record.get("turn", -1)) == 2],
        args,
    )
    t2_input = out_dir / "inputs" / "t2_input.jsonl"
    t2_count = write_jsonl(t2_input, records)
    if t2_count == 0:
        return {
            "input": str(t2_input),
            "targets": 0,
            "status": "no_matching_turns",
            "caps": [],
            "stable": None,
        }
    caps = parse_int_csv(args.t2_caps)
    cap_summaries = [run_t2_cap(args, t2_input, out_dir, cap) for cap in caps]
    stable = build_stable_t2(args, t2_input, out_dir, caps)
    return {
        "input": str(t2_input),
        "targets": t2_count,
        "caps": cap_summaries,
        "stable": stable,
        "status": "dry_run" if args.dry_run else "ok",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.time()
    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "input": str(input_path),
        "out_dir": str(out_dir),
        "mode": "oracle_first",
        "dry_run": bool(args.dry_run),
        "record_indices": args.record_indices,
        "record_offset": int(args.record_offset),
        "record_limit": int(args.record_limit),
        "late_exact": run_late_exact(args, input_path, out_dir),
        "t2": run_t2(args, input_path, out_dir),
        "elapsed_s": 0.0,
    }
    summary["elapsed_s"] = round(time.time() - start, 3)
    summary_path = out_dir / "oracle_first_batch.summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary["summary"] = str(summary_path)
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run oracle-first exact/stable teacher generation")
    parser.add_argument("--input", required=True, help="Source teacher/target JSONL")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing T2 cap JSONL files")
    parser.add_argument("--record-indices", default="", help="Comma-separated indices after turn filtering")
    parser.add_argument("--record-offset", type=int, default=0, help="Offset after turn filtering")
    parser.add_argument("--record-limit", type=int, default=0, help="0 means all rows after offset")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--print-errors", action="store_true")

    parser.add_argument("--skip-late", action="store_true")
    parser.add_argument("--late-turns", default="3,4")
    parser.add_argument("--late-workers", type=int, default=1)
    parser.add_argument("--late-max-records", type=int, default=0)

    parser.add_argument("--skip-t2", action="store_true")
    parser.add_argument("--t2-caps", default="100,200,500")
    parser.add_argument("--t2-limit", type=int, default=0, help="0 means all T2 rows")
    parser.add_argument("--t2-top-n", type=int, default=24)
    parser.add_argument("--t2-manifest", default="ai/rust_solver/t3_exact_solver/Cargo.toml")
    parser.add_argument("--t2-binary", default="")
    parser.add_argument("--fl-config", default="ai/config/fl_ev.json")
    parser.add_argument("--miss-top-k", type=int, default=5)
    parser.add_argument("--stable-weak-cap", type=int, default=-1)
    parser.add_argument("--stable-strong-cap", type=int, default=-1)
    parser.add_argument("--stable-max-records", type=int, default=0)

    args = parser.parse_args(list(argv) if argv is not None else None)
    print(json.dumps(run(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
