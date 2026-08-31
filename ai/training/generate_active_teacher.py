"""Generate MC/exact teacher labels for active-teacher target states.

Input is produced by `extract_active_teacher_targets.py`.  Output is compatible
with `convert_action_value_teacher.py`: one decision record with ranked
candidates per input target.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

from ai.prob_engine_wrapper import (
    PROB_ENGINE_PATH,
    evaluate_candidates,
    evaluate_mc_candidates,
    evaluate_mc_t0,
    evaluate_t0_ladder,
)


def _row(board: dict, *names: str) -> list[str]:
    out: list[str] = []
    for name in names:
        out.extend(board.get(name, []) or [])
    return out


def _card_count(board: dict) -> int:
    return len(_row(board, "top")) + len(_row(board, "mid", "middle")) + len(_row(board, "bot", "bottom"))


def load_targets(args: argparse.Namespace) -> list[dict]:
    include_turns = {int(t) for t in args.turns.split(",") if t.strip()}
    targets: list[dict] = []
    with Path(args.input).open("r", encoding="utf-8-sig") as f:
        for line in f:
            try:
                target = json.loads(line)
            except json.JSONDecodeError:
                continue
            if int(target.get("turn", -1)) not in include_turns:
                continue
            targets.append(target)
    if args.shuffle:
        random.Random(args.seed).shuffle(targets)
    if args.max_records:
        targets = targets[: args.max_records]
    return targets


def evaluate_target(payload: tuple[dict, int, set[int], str | None, int, str, bool, str, str]) -> dict | None:
    (
        target,
        sims,
        mc_turns,
        engine_path,
        candidate_limit,
        candidate_filter,
        t0_ladder,
        stage_sims,
        stage_limits,
    ) = payload
    turn = int(target.get("turn", -1))
    board = target.get("board", {})
    opponent_board = target.get("opponent_board") or target.get("board_opponent") or {}
    top = _row(board, "top")
    mid = _row(board, "mid", "middle")
    bot = _row(board, "bot", "bottom")
    dealt = list(target.get("dealt", []))
    exclude = list(target.get("exclude", []))
    known_discards = list(target.get("known_discards", []))
    position = "btn" if bool(target.get("is_btn", False)) else "bb"

    if turn < 0 or not dealt:
        return None

    start = time.time()
    if turn == 0 and t0_ladder:
        result = evaluate_t0_ladder(
            dealt=dealt,
            exclude=exclude,
            stage_sims=stage_sims,
            stage_limits=stage_limits,
            engine_path=engine_path,
        )
        eval_mode = "t0_ladder"
    elif turn == 0 and turn in mc_turns:
        result = evaluate_mc_t0(
            dealt=dealt,
            exclude=exclude,
            sims=sims,
            engine_path=engine_path,
            candidate_limit=candidate_limit,
            candidate_filter=candidate_filter,
        )
        eval_mode = f"mc{sims}"
    elif turn in mc_turns and _card_count(board) < 11:
        result = evaluate_mc_candidates(
            top=top,
            mid=mid,
            bot=bot,
            dealt=dealt,
            exclude=exclude,
            turn=turn,
            sims=sims,
            engine_path=engine_path,
            candidate_limit=candidate_limit if turn == 0 else 0,
            candidate_filter=candidate_filter if turn == 0 else "all",
        )
        eval_mode = f"mc{sims}"
    else:
        result = evaluate_candidates(
            top=top,
            mid=mid,
            bot=bot,
            dealt=dealt,
            exclude=exclude,
            turn=turn,
            position=position,
            engine_path=engine_path,
            candidate_limit=candidate_limit if turn == 0 else 0,
            candidate_filter=candidate_filter if turn == 0 else "all",
        )
        eval_mode = "exact"

    candidates = result.get("candidates") or []
    if not candidates:
        return None

    record = {
        "source": target.get("source"),
        "source_line": target.get("source_line"),
        "active_reasons": list(target.get("reasons", [])),
        "turn": turn,
        "board": {"top": top, "mid": mid, "bot": bot},
        "opponent_board": {
            "top": _row(opponent_board, "top"),
            "mid": _row(opponent_board, "mid", "middle"),
            "bot": _row(opponent_board, "bot", "bottom"),
        },
        "dealt": dealt,
        "known_discards": known_discards,
        "exclude": exclude,
        "is_btn": bool(target.get("is_btn", False)),
        "position": position,
        "n_candidates": len(candidates),
        "candidates": candidates,
        "best_idx": 0,
        "eval_mode": eval_mode,
        "elapsed_s": round(time.time() - start, 3),
    }
    if "stages" in result:
        record["ladder_stages"] = result.get("stages", [])
        record["initial_candidates"] = result.get("initial_candidates")
        record["final_candidates"] = result.get("final_candidates")
        record["returned_candidates"] = result.get("returned_candidates", len(candidates))
    return record


def batch_request_for_target(
    target: dict,
    index: int,
    sims: int,
    mc_turns: set[int],
    candidate_limit: int,
    candidate_filter: str,
    t0_ladder: bool,
    stage_sims: str,
    stage_limits: str,
) -> dict | None:
    turn = int(target.get("turn", -1))
    board = target.get("board", {})
    dealt = list(target.get("dealt", []))
    if turn < 0 or not dealt:
        return None
    mode = "t0_ladder" if turn == 0 and t0_ladder else ("mc" if turn in mc_turns else "candidates")
    return {
        "id": index,
        "mode": mode,
        "top": ",".join(_row(board, "top")),
        "mid": ",".join(_row(board, "mid", "middle")),
        "bot": ",".join(_row(board, "bot", "bottom")),
        "dealt": ",".join(dealt),
        "exclude": ",".join(list(target.get("exclude", []))),
        "turn": turn,
        "position": "btn" if bool(target.get("is_btn", False)) else "bb",
        "sims": sims,
        "stage_sims": stage_sims,
        "stage_limits": stage_limits,
        "candidate_limit": candidate_limit if turn == 0 else 0,
        "candidate_filter": candidate_filter if turn == 0 else "all",
    }


def active_record_from_batch(target: dict, result: dict, elapsed_s: float) -> dict | None:
    turn = int(target.get("turn", -1))
    board = target.get("board", {})
    opponent_board = target.get("opponent_board") or target.get("board_opponent") or {}
    top = _row(board, "top")
    mid = _row(board, "mid", "middle")
    bot = _row(board, "bot", "bottom")
    dealt = list(target.get("dealt", []))
    candidates = result.get("candidates") or []
    if not candidates:
        return None
    record = {
        "source": target.get("source"),
        "source_line": target.get("source_line"),
        "active_reasons": list(target.get("reasons", [])),
        "turn": turn,
        "board": {"top": top, "mid": mid, "bot": bot},
        "opponent_board": {
            "top": _row(opponent_board, "top"),
            "mid": _row(opponent_board, "mid", "middle"),
            "bot": _row(opponent_board, "bot", "bottom"),
        },
        "dealt": dealt,
        "known_discards": list(target.get("known_discards", [])),
        "exclude": list(target.get("exclude", [])),
        "is_btn": bool(target.get("is_btn", False)),
        "position": "btn" if bool(target.get("is_btn", False)) else "bb",
        "n_candidates": len(candidates),
        "candidates": candidates,
        "best_idx": 0,
        "eval_mode": (
            "t0_ladder"
            if "stages" in result
            else ("mc" + str(result.get("simulations_per_candidate", "")) if "simulations_per_candidate" in result else "exact")
        ),
        "elapsed_s": round(elapsed_s, 3),
    }
    if "stages" in result:
        record["ladder_stages"] = result.get("stages", [])
        record["initial_candidates"] = result.get("initial_candidates")
        record["final_candidates"] = result.get("final_candidates")
        record["returned_candidates"] = result.get("returned_candidates", len(candidates))
    return record


def generate_batch(args: argparse.Namespace, targets: list[dict], mc_turns: set[int], engine_path: str | None) -> dict:
    output = Path(args.output)
    request_path = output.with_suffix(".batch_requests.jsonl")
    response_path = output.with_suffix(".batch_responses.jsonl")
    engine = engine_path or str(PROB_ENGINE_PATH)

    request_count = 0
    with request_path.open("w", encoding="utf-8") as f:
        for i, target in enumerate(targets):
            req = batch_request_for_target(
                target,
                i,
                args.sims,
                mc_turns,
                args.candidate_limit,
                args.candidate_filter,
                args.t0_ladder,
                args.stage_sims,
                args.stage_limits,
            )
            if req is None:
                continue
            f.write(json.dumps(req, ensure_ascii=False) + "\n")
            request_count += 1

    cmd = [
        engine,
        "--mode",
        "batch",
        "--input",
        str(request_path),
        "--output",
        str(response_path),
    ]
    print(f"Batch engine: {engine}")
    print(f"Batch requests: {request_count:,} -> {request_path}")
    start = time.time()
    completed = subprocess.run(cmd, capture_output=True, text=True, timeout=args.batch_timeout)
    elapsed = time.time() - start
    if completed.returncode != 0:
        raise RuntimeError(f"batch prob_engine failed: {completed.stderr}")
    if completed.stderr.strip():
        print(completed.stderr.strip())

    stats = {
        "input": args.input,
        "output": str(output),
        "request_path": str(request_path),
        "response_path": str(response_path),
        "targets": len(targets),
        "requests": request_count,
        "written": 0,
        "skipped": 0,
        "batch_errors": 0,
        "sims": args.sims,
        "mc_turns": sorted(mc_turns),
        "candidate_limit": args.candidate_limit,
        "candidate_filter": args.candidate_filter,
        "t0_ladder": bool(args.t0_ladder),
        "stage_sims": args.stage_sims,
        "stage_limits": args.stage_limits,
        "turns": {},
        "eval_modes": {},
        "elapsed_s": round(elapsed, 3),
        "batch_engine": engine,
    }

    with response_path.open("r", encoding="utf-8") as src, output.open("w", encoding="utf-8") as dst:
        for line in src:
            try:
                response = json.loads(line)
            except json.JSONDecodeError:
                stats["skipped"] += 1
                continue
            idx = int(response.get("id", -1))
            if idx < 0 or idx >= len(targets) or not response.get("ok"):
                stats["batch_errors"] += 1
                if args.print_errors:
                    print(f"  batch error id={idx}: {response.get('error')}", flush=True)
                continue
            result = response.get("result") or {}
            elapsed_ms = float(result.get("elapsed_ms", 0.0))
            record = active_record_from_batch(targets[idx], result, elapsed_ms / 1000.0)
            if record is None:
                stats["skipped"] += 1
                continue
            dst.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["written"] += 1
            turn_key = str(record["turn"])
            mode = record["eval_mode"]
            stats["turns"][turn_key] = stats["turns"].get(turn_key, 0) + 1
            stats["eval_modes"][mode] = stats["eval_modes"].get(mode, 0) + 1

    return stats


def generate(args: argparse.Namespace) -> None:
    targets = load_targets(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    mc_turns = {int(t) for t in args.mc_turns.split(",") if t.strip()}
    engine_path = args.engine_path or None

    print(f"Active targets: {len(targets):,}")
    print(f"Output: {output}")
    print(f"Sims: {args.sims}")
    print(f"MC turns: {sorted(mc_turns)}")
    print(f"Workers: {args.workers}")
    print(f"Candidate limit: {args.candidate_limit}")
    print(f"Candidate filter: {args.candidate_filter}")
    print(f"T0 ladder: {args.t0_ladder}")
    if args.t0_ladder:
        print(f"Stage sims: {args.stage_sims}")
        print(f"Stage limits: {args.stage_limits}")

    if args.batch_engine:
        stats = generate_batch(args, targets, mc_turns, engine_path)
        with output.with_suffix(".summary.json").open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"Done: written={stats['written']:,} skipped={stats['skipped']:,} batch_errors={stats['batch_errors']:,} elapsed={stats['elapsed_s']:.1f}s")
        print(f"Summary: {output.with_suffix('.summary.json')}")
        return

    stats = {
        "input": args.input,
        "output": str(output),
        "targets": len(targets),
        "written": 0,
        "skipped": 0,
        "sims": args.sims,
        "mc_turns": sorted(mc_turns),
        "candidate_limit": args.candidate_limit,
        "candidate_filter": args.candidate_filter,
        "t0_ladder": bool(args.t0_ladder),
        "stage_sims": args.stage_sims,
        "stage_limits": args.stage_limits,
        "turns": {},
        "eval_modes": {},
        "elapsed_s": 0.0,
    }
    start = time.time()

    work = [
        (
            target,
            args.sims,
            mc_turns,
            engine_path,
            args.candidate_limit,
            args.candidate_filter,
            args.t0_ladder,
            args.stage_sims,
            args.stage_limits,
        )
        for target in targets
    ]
    with output.open("w", encoding="utf-8") as dst:
        if args.workers <= 1:
            iterator = (evaluate_target(item) for item in work)
            for i, record in enumerate(iterator, start=1):
                if record is None:
                    stats["skipped"] += 1
                    continue
                dst.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["written"] += 1
                turn_key = str(record["turn"])
                mode = record["eval_mode"]
                stats["turns"][turn_key] = stats["turns"].get(turn_key, 0) + 1
                stats["eval_modes"][mode] = stats["eval_modes"].get(mode, 0) + 1
                if i % args.progress_every == 0:
                    dst.flush()
                    elapsed = time.time() - start
                    print(f"  {i:,}/{len(work):,} written={stats['written']:,} skipped={stats['skipped']:,} ({elapsed:.0f}s)", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(evaluate_target, item) for item in work]
                for i, future in enumerate(as_completed(futures), start=1):
                    try:
                        record = future.result()
                    except Exception as exc:
                        stats["skipped"] += 1
                        if args.print_errors:
                            print(f"  error: {exc}", flush=True)
                        continue
                    if record is None:
                        stats["skipped"] += 1
                        continue
                    dst.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stats["written"] += 1
                    turn_key = str(record["turn"])
                    mode = record["eval_mode"]
                    stats["turns"][turn_key] = stats["turns"].get(turn_key, 0) + 1
                    stats["eval_modes"][mode] = stats["eval_modes"].get(mode, 0) + 1
                    if i % args.progress_every == 0:
                        dst.flush()
                        elapsed = time.time() - start
                        print(f"  {i:,}/{len(work):,} written={stats['written']:,} skipped={stats['skipped']:,} ({elapsed:.0f}s)", flush=True)

    stats["elapsed_s"] = round(time.time() - start, 3)
    with output.with_suffix(".summary.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"Done: written={stats['written']:,} skipped={stats['skipped']:,} elapsed={stats['elapsed_s']:.1f}s")
    print(f"Summary: {output.with_suffix('.summary.json')}")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate teacher labels for active target states")
    parser.add_argument("input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--sims", type=int, default=20)
    parser.add_argument("--turns", default="0,1,2,3,4")
    parser.add_argument("--mc-turns", default="0,1,2",
                        help="Turns evaluated with MC. Other turns use exact candidate evaluation.")
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--engine-path", default=None)
    parser.add_argument("--print-errors", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--candidate-limit", type=int, default=0)
    parser.add_argument("--candidate-filter", default="all", choices=["all", "t0_fl_route"])
    parser.add_argument("--t0-ladder", action="store_true",
                        help="For T0 records, use successive halving instead of plain MC")
    parser.add_argument("--stage-sims", default="2,5,10,25,50",
                        help="Comma-separated T0 ladder sims per stage")
    parser.add_argument("--stage-limits", default="200,150,64,24,8",
                        help="Comma-separated T0 ladder keep counts per stage")
    parser.add_argument("--batch-engine", action="store_true",
                        help="Use prob_engine JSONL batch mode instead of one subprocess per target")
    parser.add_argument("--batch-timeout", type=int, default=86400)
    args = parser.parse_args(argv)
    generate(args)


if __name__ == "__main__":
    main()
