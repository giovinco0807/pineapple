"""Convert candidate-level weak groups back into active-teacher targets."""
from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


SUITS = "hdcs"


def row(board: dict[str, Any], *names: str) -> list[str]:
    out: list[str] = []
    for name in names:
        out.extend(board.get(name, []) or [])
    return [str(card) for card in out if card]


def teacher_board(board: dict[str, Any] | None) -> dict[str, list[str]]:
    board = board or {}
    return {
        "top": row(board, "top"),
        "mid": row(board, "mid", "middle"),
        "bot": row(board, "bot", "bottom"),
    }


def flat_cards(board: dict[str, Any]) -> list[str]:
    return row(board, "top") + row(board, "mid", "middle") + row(board, "bot", "bottom")


def unique_cards(cards: Iterable[str], blocked: set[str] | None = None) -> list[str]:
    blocked = blocked or set()
    seen: set[str] = set()
    out: list[str] = []
    for card in cards:
        card = str(card)
        if not card or card in seen or card in blocked:
            continue
        seen.add(card)
        out.append(card)
    return out


def topk_regret(item: dict[str, Any], topk: int) -> float:
    values = item.get("topk_exact_rerank_regret") or {}
    return float(values.get(str(topk), values.get(topk, 0.0)))


def candidate_score(candidate: dict[str, Any]) -> float | None:
    if "target_score" in candidate:
        return float(candidate["target_score"])
    if "score" in candidate:
        return float(candidate["score"])
    if "ev" in candidate:
        return float(candidate["ev"])
    recursive = candidate.get("recursive") or {}
    recursive_mc = recursive.get("mc") or {}
    if "avg_score" in recursive_mc:
        return float(recursive_mc["avg_score"])
    metrics = candidate.get("metrics") or {}
    if "score" in metrics:
        return float(metrics["score"])
    if "ev" in metrics:
        return float(metrics["ev"])
    mc = candidate.get("mc") or {}
    if "avg_score" in mc:
        return float(mc["avg_score"])
    if "score" in mc:
        return float(mc["score"])
    return None


def candidate_sims(candidate: dict[str, Any]) -> int | None:
    recursive = candidate.get("recursive") or {}
    recursive_mc = recursive.get("mc") or {}
    for key in ("simulations", "n_rollouts", "samples"):
        if key in recursive_mc:
            return int(recursive_mc[key])
    mc = candidate.get("mc") or {}
    for key in ("simulations", "n_rollouts", "samples"):
        if key in mc:
            return int(mc[key])
    return None


def teacher_margin(record: dict[str, Any]) -> dict[str, Any]:
    candidates = record.get("candidates") or []
    scored: list[tuple[int, float]] = []
    for idx, candidate in enumerate(candidates):
        score = candidate_score(candidate)
        if score is not None:
            scored.append((idx, score))
    if not scored:
        return {
            "teacher_score": None,
            "teacher_second_score": None,
            "teacher_margin": None,
            "teacher_sims": None,
        }
    scored.sort(key=lambda item: item[1], reverse=True)
    best_idx = int(record.get("best_idx", scored[0][0]))
    best_score = next((score for idx, score in scored if idx == best_idx), scored[0][1])
    second_score = next((score for idx, score in scored if idx != best_idx), None)
    margin = None if second_score is None else float(best_score - second_score)
    sims = candidate_sims(candidates[best_idx]) if 0 <= best_idx < len(candidates) else None
    return {
        "teacher_score": float(best_score),
        "teacher_second_score": None if second_score is None else float(second_score),
        "teacher_margin": margin,
        "teacher_sims": sims,
    }


def load_weak_group_ids(args: argparse.Namespace) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    turns = {int(t) for t in args.turns.split(",") if t.strip()}
    with Path(args.weak_groups).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            turn = int(item.get("turn", -1))
            rank = int(item.get("rank", 0))
            t_regret = topk_regret(item, args.topk)
            if turn not in turns:
                continue
            if rank <= args.min_rank and t_regret < args.min_topk_regret:
                continue
            out[int(item["group_id"])] = item
            if args.max_groups and len(out) >= args.max_groups:
                break
    return out


def parse_turns(raw: str) -> set[int]:
    return {int(t) for t in raw.split(",") if t.strip()}


def iter_flat_records(path: Path, source_turns: set[int]):
    with path.open("r", encoding="utf-8") as src:
        for source_line, line in enumerate(src):
            record = json.loads(line)
            turn = int(record.get("turn", -1))
            candidates = record.get("candidates") or []
            if turn not in source_turns or not candidates:
                continue
            yield record, source_line, None


def iter_recursive_records(path: Path, source_turns: set[int]):
    from argparse import Namespace

    from ai.training.convert_recursive_teacher_to_reranker import decision_candidates

    helper_args = Namespace(include_turns=source_turns, position="bb")
    with path.open("r", encoding="utf-8") as src:
        for source_line, line in enumerate(src):
            if not line.strip():
                continue
            record = json.loads(line)
            decision_index = 0
            for decision in decision_candidates(record, helper_args):
                candidates = decision.get("candidates") or []
                if int(decision.get("turn", -1)) not in source_turns or not candidates:
                    continue
                expanded = dict(decision)
                expanded["source"] = str(path)
                expanded["source_line"] = source_line
                expanded["source_decision_index"] = decision_index
                expanded["source_hand_index"] = record.get("hand_index")
                expanded["eval_mode"] = record.get("eval_mode") or "recursive_estimated"
                yield expanded, source_line, decision_index
                decision_index += 1


def iter_teacher_records(args: argparse.Namespace, source_turns: set[int]):
    path = Path(args.teacher_labels)
    fmt = args.teacher_format
    if fmt == "auto":
        first = ""
        with path.open("r", encoding="utf-8") as src:
            for line in src:
                if line.strip():
                    first = line
                    break
        item = json.loads(first) if first else {}
        fmt = "flat" if "turn" in item and "candidates" in item else "recursive"
    if fmt == "recursive":
        yield from iter_recursive_records(path, source_turns)
    elif fmt == "flat":
        yield from iter_flat_records(path, source_turns)
    else:
        raise ValueError(f"Unsupported teacher format: {fmt}")


def card_permuter(mapping: dict[str, str]):
    def convert(card: str) -> str:
        card = str(card)
        if len(card) < 2 or card.startswith("X"):
            return card
        suit = card[-1]
        if suit not in mapping:
            return card
        return card[:-1] + mapping[suit]

    return convert


def map_cards(cards: Iterable[str], convert) -> list[str]:
    return [convert(str(card)) for card in cards if card]


def map_board(board: dict[str, list[str]], convert) -> dict[str, list[str]]:
    return {
        "top": map_cards(board.get("top", []), convert),
        "mid": map_cards(board.get("mid", []), convert),
        "bot": map_cards(board.get("bot", []), convert),
    }


def suit_mappings(mode: str) -> list[dict[str, str]]:
    if mode == "none":
        return [{s: s for s in SUITS}]
    return [dict(zip(SUITS, perm)) for perm in itertools.permutations(SUITS)]


def target_from_record(record: dict[str, Any], weak: dict[str, Any]) -> dict[str, Any]:
    board = teacher_board(record.get("board") or {})
    opponent = teacher_board(record.get("opponent_board") or record.get("board_opponent") or {})
    dealt = unique_cards(record.get("dealt", []) or [])
    own_cards = set(flat_cards(board)) | set(dealt)
    known = unique_cards(record.get("known_discards", []) or [], blocked=own_cards)
    exclude = unique_cards(flat_cards(opponent) + known, blocked=own_cards)
    reasons = list(record.get("active_reasons", []) or [])
    if "top20_miss" not in reasons and int(weak.get("rank", 0)) > 20:
        reasons.append("top20_miss")
    if "top20_rerank_regret" not in reasons and topk_regret(weak, 20) > 0:
        reasons.append("top20_rerank_regret")
    weak_topk = int(weak.get("selection_topk", 20))
    if weak_topk != 20:
        miss_reason = f"top{weak_topk}_miss"
        regret_reason = f"top{weak_topk}_rerank_regret"
        if miss_reason not in reasons and int(weak.get("rank", 0)) > weak_topk:
            reasons.append(miss_reason)
        if regret_reason not in reasons and topk_regret(weak, weak_topk) > 0:
            reasons.append(regret_reason)
    if "high_regret" not in reasons and float(weak.get("regret", 0.0)) >= 2.0:
        reasons.append("high_regret")
    margin_info = teacher_margin(record)
    eval_mode = str(record.get("eval_mode") or "")
    exact_teacher = "exact" in eval_mode.lower()
    if not exact_teacher and margin_info["teacher_margin"] is not None and "estimated_teacher" not in reasons:
        reasons.append("estimated_teacher")
    if (
        margin_info["teacher_margin"] is not None
        and margin_info["teacher_margin"] >= float(weak.get("min_teacher_margin", 0.0))
        and "high_confidence_teacher_margin" not in reasons
    ):
        reasons.append("high_confidence_teacher_margin")
    min_teacher_sims = int(weak.get("min_teacher_sims", 0) or 0)
    if (
        min_teacher_sims > 0
        and margin_info["teacher_sims"] is not None
        and int(margin_info["teacher_sims"]) >= min_teacher_sims
        and "high_confidence_teacher_sims" not in reasons
    ):
        reasons.append("high_confidence_teacher_sims")

    return {
        "source": str(record.get("source") or "weak_group"),
        "source_line": record.get("source_line"),
        "source_decision_index": record.get("source_decision_index"),
        "source_hand_index": record.get("source_hand_index"),
        "turn": int(record.get("turn", -1)),
        "board": board,
        "opponent_board": opponent,
        "dealt": dealt,
        "known_discards": known,
        "exclude": exclude,
        "is_btn": bool(record.get("is_btn", str(record.get("position", "")).lower() == "btn")),
        "player": 0,
        "reasons": reasons,
        "weak_group_id": int(weak.get("group_id", -1)),
        "weak_rank": int(weak.get("rank", 0)),
        "weak_regret": float(weak.get("regret", 0.0)),
        "weak_topk_rerank_regret": {
            str(k): topk_regret(weak, k) for k in sorted({1, 3, 5, 10, 20, weak_topk})
        },
        "teacher_eval_mode": eval_mode,
        "teacher_is_exact": exact_teacher,
        **margin_info,
    }


def requires_margin(record: dict[str, Any], mode: str) -> bool:
    if mode == "all":
        return True
    eval_mode = str(record.get("eval_mode") or "").lower()
    return "exact" not in eval_mode


def target_key(target: dict[str, Any]) -> str:
    stable = {
        "turn": target.get("turn"),
        "board": target.get("board"),
        "opponent_board": target.get("opponent_board"),
        "dealt": target.get("dealt"),
        "known_discards": target.get("known_discards"),
        "exclude": target.get("exclude"),
        "is_btn": target.get("is_btn"),
    }
    return json.dumps(stable, sort_keys=True, separators=(",", ":"))


def expand_suits(target: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, mapping in enumerate(suit_mappings(mode)):
        convert = card_permuter(mapping)
        copied = dict(target)
        copied["board"] = map_board(target["board"], convert)
        copied["opponent_board"] = map_board(target["opponent_board"], convert)
        copied["dealt"] = map_cards(target["dealt"], convert)
        copied["known_discards"] = map_cards(target.get("known_discards", []), convert)
        copied["exclude"] = map_cards(target.get("exclude", []), convert)
        copied["suit_permutation"] = "".join(mapping[s] for s in SUITS)
        copied["augmentation_index"] = idx
        out.append(copied)
    return out


def convert(args: argparse.Namespace) -> dict[str, Any]:
    weak_by_group = load_weak_group_ids(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    seen: set[str] = set()
    written = 0

    source_turns = parse_turns(args.source_turns)
    converted_group_id = 0
    with output.open("w", encoding="utf-8") as dst:
        for record, source_line, _decision_index in iter_teacher_records(args, source_turns):
            if converted_group_id not in weak_by_group:
                converted_group_id += 1
                continue
            record.setdefault("source_line", source_line)
            weak = weak_by_group[converted_group_id]
            weak = dict(weak)
            weak["selection_topk"] = int(args.topk)
            weak["min_teacher_margin"] = float(args.min_teacher_margin)
            weak["min_teacher_sims"] = int(args.min_teacher_sims)
            margin_info = teacher_margin(record)
            margin = margin_info.get("teacher_margin")
            if (
                args.min_teacher_margin > 0.0
                and requires_margin(record, args.margin_applies_to)
                and (margin is None or float(margin) < args.min_teacher_margin)
            ):
                stats["skipped_low_teacher_margin"] += 1
                converted_group_id += 1
                continue
            sims = margin_info.get("teacher_sims")
            if (
                args.min_teacher_sims > 0
                and requires_margin(record, args.teacher_sims_applies_to)
                and (sims is None or int(sims) < args.min_teacher_sims)
            ):
                stats["skipped_low_teacher_sims"] += 1
                converted_group_id += 1
                continue
            target = target_from_record(record, weak)
            if int(target.get("turn", -1)) != int(weak.get("turn", -2)):
                stats["turn_mismatch_skipped"] += 1
                converted_group_id += 1
                continue
            for expanded in expand_suits(target, args.suit_permutations):
                key = target_key(expanded)
                if key in seen:
                    continue
                seen.add(key)
                dst.write(json.dumps(expanded, ensure_ascii=False) + "\n")
                written += 1
                stats[f"turn_{expanded['turn']}"] += 1
                for reason in expanded.get("reasons", []):
                    stats[f"reason_{reason}"] += 1
            converted_group_id += 1

    summary = {
        "weak_groups": str(args.weak_groups),
        "teacher_labels": str(args.teacher_labels),
        "output": str(output),
        "selected_groups": len(weak_by_group),
        "written": written,
        "source_turns": args.source_turns,
        "teacher_format": args.teacher_format,
        "topk": int(args.topk),
        "min_rank": int(args.min_rank),
        "min_topk_regret": float(args.min_topk_regret),
        "min_teacher_margin": float(args.min_teacher_margin),
        "margin_applies_to": args.margin_applies_to,
        "min_teacher_sims": int(args.min_teacher_sims),
        "teacher_sims_applies_to": args.teacher_sims_applies_to,
        "max_groups": int(args.max_groups),
        "suit_permutations": args.suit_permutations,
        "counts": dict(stats),
    }
    output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Convert weak reranker groups to active-teacher targets")
    parser.add_argument("--weak-groups", required=True)
    parser.add_argument("--teacher-labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--turns", default="2")
    parser.add_argument("--source-turns", default="0,1,2,3,4",
                        help="Turns included when the source reranker data was converted")
    parser.add_argument("--teacher-format", choices=("flat", "recursive", "auto"), default="flat",
                        help="Input teacher label format. recursive matches convert_recursive_teacher_to_reranker.")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--min-rank", type=int, default=20)
    parser.add_argument("--min-topk-regret", type=float, default=1e-9)
    parser.add_argument("--min-teacher-margin", type=float, default=0.0,
                        help="Skip targets whose teacher best is closer than this to second best")
    parser.add_argument("--margin-applies-to", choices=("estimated", "all"), default="estimated",
                        help="Apply teacher margin filtering to estimated labels only or all labels")
    parser.add_argument("--min-teacher-sims", type=int, default=0,
                        help="Skip estimated/all targets whose best candidate has fewer simulations")
    parser.add_argument("--teacher-sims-applies-to", choices=("estimated", "all"), default="estimated",
                        help="Apply simulation-count filtering to estimated labels only or all labels")
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument("--suit-permutations", choices=("none", "all"), default="all")
    args = parser.parse_args(list(argv) if argv is not None else None)
    print(json.dumps(convert(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
