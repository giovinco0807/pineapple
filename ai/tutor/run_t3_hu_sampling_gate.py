"""Select and run a balanced two-seed T3 HU sampling pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from ai.engine.turn_order import POSITION_CONTRACT_VERSION, normalize_position
from ai.tutor.audit_t3_gate_dataset import _joker_count, decision_signature, iter_jsonl
from ai.tutor.split_t3_gate_by_root import root_index
from ai.tutor.t3_hu_sampled import evaluate_t3_hu_sampled


SCHEMA = "ofc_hu_gate/v1"
METHOD = "hu_sampled_pimc_exact_t4"
INFORMATION_MODEL = "pimc_determinization_v1"


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{label} line {line_number} is not a JSON object")
                rows.append(row)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {label} JSONL at {path}:{exc.lineno}: {exc.msg}") from exc
    return rows


def _sanitize_public_action_history(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Keep only information that was public when each prior action completed."""
    raw_history = (
        record.get("public_action_history")
        if "public_action_history" in record
        else record.get("trace")
    ) or []
    if not isinstance(raw_history, list):
        raise ValueError("public action history must be a list")
    history: list[dict[str, Any]] = []
    row_aliases = {"mid": "middle", "bot": "bottom"}
    for index, raw_entry in enumerate(raw_history):
        if not isinstance(raw_entry, dict):
            raise ValueError(f"public action history entry {index} is not an object")
        raw_actor = raw_entry.get("actor") or raw_entry.get("seat") or raw_entry.get("position")
        has_actor_flag = "is_btn" in raw_entry and raw_entry.get("is_btn") is not None
        if raw_actor in (None, "") and not has_actor_flag:
            raise ValueError(f"public action history entry {index} is missing actor")
        if "turn" not in raw_entry:
            raise ValueError(f"public action history entry {index} is missing turn")
        actor_flag = raw_entry["is_btn"] if has_actor_flag else None
        actor = normalize_position(raw_actor, is_btn=actor_flag)
        action = raw_entry.get("action") if isinstance(raw_entry.get("action"), dict) else {}
        raw_placements = raw_entry.get("placements", action.get("placements")) or []
        if not isinstance(raw_placements, list):
            raise ValueError(f"public action history entry {index} placements must be a list")
        placements: list[list[str]] = []
        for placement in raw_placements:
            if not isinstance(placement, (list, tuple)) or len(placement) != 2:
                raise ValueError(f"public action history entry {index} has invalid placement")
            card, row = placement
            row_name = row_aliases.get(str(row), str(row))
            if row_name not in {"top", "middle", "bottom"}:
                raise ValueError(f"public action history entry {index} has invalid row {row!r}")
            placements.append([str(card), row_name])
        history.append(
            {
                "turn": int(raw_entry.get("turn", 0)),
                "actor": actor,
                "placements": placements,
            }
        )
    return history


def _hash_order(*parts: Any) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _board_cards(board: Any) -> set[str]:
    if not isinstance(board, dict):
        return set()
    cards: set[str] = set()
    for key in ("top", "middle", "mid", "bottom", "bot"):
        cards.update(str(card) for card in (board.get(key) or []))
    return cards


def sanitize_position(record: dict[str, Any], *, source_index: int, root: int) -> dict[str, Any]:
    is_btn = record["is_btn"] if "is_btn" in record else None
    position = normalize_position(record.get("position"), is_btn=is_btn)
    board = record.get("board") or {}
    opponent = record.get("opponent_board") or {}
    dealt = list(record.get("dealt") or [])
    known_self = list(record.get("known_discards_self") or record.get("known_discards") or [])
    known = _board_cards(board) | _board_cards(opponent) | set(dealt) | set(known_self)
    public_exclude = [str(card) for card in (record.get("public_exclude") or [])]
    ambiguous_exclude: list[str] = []
    for card in record.get("exclude") or []:
        text = str(card)
        if text not in known and text not in public_exclude:
            ambiguous_exclude.append(text)
    if ambiguous_exclude:
        raise ValueError(
            "ambiguous exclude cards may reveal opponent private discards; "
            "use public_exclude for synthetic public removals: "
            f"{ambiguous_exclude}"
        )
    joker_count = _joker_count(record)
    return {
        "schema": SCHEMA,
        "turn": 3,
        "actor": position,
        "position": position,
        "is_btn": position == "btn",
        "first_actor": "bb",
        "position_contract_version": POSITION_CONTRACT_VERSION,
        "board": board,
        "opponent_board": opponent,
        "dealt": dealt,
        "known_discards_self": known_self,
        "public_exclude": public_exclude,
        "public_action_history": _sanitize_public_action_history(record),
        "pilot": {
            "decision_id": decision_signature(record),
            "cluster_id": f"root:{root}",
            "root_index": root,
            "source_index": source_index,
            "visible_joker_count": joker_count,
            "stratum": f"{position}_joker{joker_count}",
        },
    }


def select_balanced_positions(
    records: Sequence[dict[str, Any]],
    *,
    selection_seed: int = 20260712,
    max_root: int = 99,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    annotated: list[tuple[dict[str, Any], int, int]] = []
    for source_index, record in enumerate(records):
        root = root_index(record, line_number=source_index + 1)
        if root <= max_root:
            annotated.append((record, source_index, root))

    targets = {
        "bb_joker0": 25,
        "btn_joker0": 25,
        "bb_joker1": 13,
        "bb_joker2": 12,
        "btn_joker1": 13,
        "btn_joker2": 12,
    }
    groups: dict[str, dict[int, list[tuple[dict[str, Any], int, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for item in annotated:
        record, _source_index, root = item
        is_btn = record["is_btn"] if "is_btn" in record else None
        position = normalize_position(record.get("position"), is_btn=is_btn)
        joker_count = _joker_count(record)
        key = f"{position}_joker{joker_count}"
        if key in targets:
            groups[key][root].append(item)
    for key, by_root in groups.items():
        for root, items in by_root.items():
            items.sort(key=lambda item: _hash_order(selection_seed, key, root, decision_signature(item[0])))

    root_usage: Counter[int] = Counter()
    selected: list[dict[str, Any]] = []
    selected_counts: Counter[str] = Counter()
    # Scarce no-Joker strata go first; Joker strata then maximize remaining root diversity.
    order = ["btn_joker0", "bb_joker0", "bb_joker1", "btn_joker1", "bb_joker2", "btn_joker2"]
    for key in order:
        target = targets[key]
        by_root = groups.get(key) or {}
        offsets: Counter[int] = Counter()
        while selected_counts[key] < target:
            eligible = [root for root, items in by_root.items() if offsets[root] < len(items)]
            if not eligible:
                raise ValueError(f"insufficient rows for {key}: {selected_counts[key]}/{target}")
            eligible.sort(
                key=lambda root: (
                    root_usage[root],
                    offsets[root],
                    _hash_order(selection_seed, key, root),
                )
            )
            root = eligible[0]
            record, source_index, _ = by_root[root][offsets[root]]
            offsets[root] += 1
            root_usage[root] += 1
            selected_counts[key] += 1
            selected.append(sanitize_position(record, source_index=source_index, root=root))

    selected.sort(
        key=lambda row: (
            row["pilot"]["stratum"],
            _hash_order(selection_seed, row["pilot"]["decision_id"]),
        )
    )
    summary = {
        "schema": SCHEMA,
        "selection_seed": int(selection_seed),
        "max_root": int(max_root),
        "selected": len(selected),
        "strata": dict(sorted(selected_counts.items())),
        "unique_roots": len(root_usage),
        "max_rows_per_root": max(root_usage.values(), default=0),
        "root_usage": {str(root): count for root, count in sorted(root_usage.items())},
    }
    return selected, summary


def load_sanitized_positions(path: Path) -> list[dict[str, Any]]:
    """Load a prior positions file while reapplying public-history allowlists."""
    positions: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, row in enumerate(_read_jsonl(path, label="positions"), start=1):
        if row.get("schema") != SCHEMA or int(row.get("turn", -1)) != 3:
            raise ValueError(f"positions line {line_number} is not a sanitized T3 HU position")
        pilot = row.get("pilot")
        if not isinstance(pilot, dict) or not str(pilot.get("decision_id") or ""):
            raise ValueError(f"positions line {line_number} is missing pilot.decision_id")
        decision_id = str(pilot["decision_id"])
        if decision_id in seen_ids:
            raise ValueError(f"duplicate position decision_id: {decision_id}")
        seen_ids.add(decision_id)
        flag = row["is_btn"] if "is_btn" in row else None
        position = normalize_position(row.get("position"), is_btn=flag)
        if row.get("actor") in (None, ""):
            raise ValueError(f"positions line {line_number} is missing actor")
        actor = normalize_position(row.get("actor"))
        if actor != position:
            raise ValueError(f"positions line {line_number} actor/position mismatch")
        normalized = dict(row)
        normalized["position"] = position
        normalized["actor"] = position
        normalized["is_btn"] = position == "btn"
        normalized["first_actor"] = "bb"
        normalized["public_action_history"] = _sanitize_public_action_history(row)
        normalized.pop("trace", None)
        positions.append(normalized)
    return positions


def _comparison_priority(row: dict[str, Any]) -> tuple[float, str]:
    decision_id = str(row.get("decision_id") or "")
    if not decision_id:
        raise ValueError("comparison is missing decision_id")
    try:
        max_regret = max(
            float(row["cross_regret_a_to_b"]),
            float(row["cross_regret_b_to_a"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"comparison {decision_id} is missing valid cross-regret values") from exc
    return max_regret, decision_id


def refine_positions(
    positions: Sequence[dict[str, Any]],
    *,
    position_filter: str = "all",
    limit: int = 0,
    comparisons: Sequence[dict[str, Any]] | None = None,
    disagreements_only: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Filter/rank a reusable position set deterministically for refinement."""
    if position_filter not in {"all", "bb", "btn"}:
        raise ValueError("position_filter must be one of: all, bb, btn")
    if int(limit) < 0:
        raise ValueError("limit must be non-negative")
    if disagreements_only and comparisons is None:
        raise ValueError("--disagreements-only requires --comparisons-input")

    requested = [
        position
        for position in positions
        if position_filter == "all" or position.get("position") == position_filter
    ]
    after_position_filter = len(requested)
    ranked_by = "source_order"
    disagreement_count: int | None = None
    if comparisons is not None:
        by_id: dict[str, dict[str, Any]] = {}
        for comparison in comparisons:
            _max_regret, decision_id = _comparison_priority(comparison)
            if decision_id in by_id:
                raise ValueError(f"duplicate comparison decision_id: {decision_id}")
            if not isinstance(comparison.get("strict_top1_agreement"), bool):
                raise ValueError(f"comparison {decision_id} has invalid strict_top1_agreement")
            by_id[decision_id] = comparison
        missing = [
            str(position["pilot"]["decision_id"])
            for position in requested
            if str(position["pilot"]["decision_id"]) not in by_id
        ]
        if missing:
            raise ValueError(f"comparisons are missing requested decision IDs: {missing[:3]}")
        if disagreements_only:
            requested = [
                position
                for position in requested
                if not by_id[str(position["pilot"]["decision_id"])]["strict_top1_agreement"]
            ]
        disagreement_count = len(requested)
        requested.sort(
            key=lambda position: (
                -_comparison_priority(by_id[str(position["pilot"]["decision_id"])])[0],
                str(position["pilot"]["decision_id"]),
            )
        )
        ranked_by = "max_cross_regret_desc_then_decision_id"

    before_limit = len(requested)
    if int(limit) > 0:
        requested = requested[: int(limit)]
    details = {
        "source_positions": len(positions),
        "position_filter": position_filter,
        "after_position_filter": after_position_filter,
        "comparisons_supplied": comparisons is not None,
        "disagreements_only": bool(disagreements_only),
        "after_disagreement_filter": disagreement_count,
        "ranked_by": ranked_by,
        "before_limit": before_limit,
        "limit": int(limit),
        "requested_positions": len(requested),
    }
    return requested, details


def _percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * p) - 1))
    return ordered[index]


def _run_seed(
    positions: Sequence[dict[str, Any]],
    *,
    seed: int,
    btn_outer: int,
    bb_outer: int,
    bb_inner: int,
    rust_solver_path: str | Path | None,
    rust_timeout_s: float,
    output_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    normalized_solver = str(Path(rust_solver_path).resolve()) if rust_solver_path else None
    run_config = {
        "schema": SCHEMA,
        "method": METHOD,
        "information_model": INFORMATION_MODEL,
        "seed": int(seed),
        "btn_outer": int(btn_outer),
        "bb_outer": int(bb_outer),
        "bb_inner": int(bb_inner),
        "rust_solver_path": normalized_solver,
        "rust_timeout_s": float(rust_timeout_s),
    }
    requested_ids = [str(position["pilot"]["decision_id"]) for position in positions]
    if len(set(requested_ids)) != len(requested_ids):
        raise ValueError("requested positions contain duplicate decision IDs")

    results: list[dict[str, Any]] = []
    legacy_config_rows = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        results = _read_jsonl(output_path, label=f"seed {seed} run")
        if len(results) > len(positions):
            raise ValueError(
                f"existing seed {seed} run has {len(results)} rows for {len(positions)} requested positions"
            )
        existing_ids: list[str] = []
        for row_index, (result, position) in enumerate(zip(results, positions), start=1):
            pilot = result.get("pilot")
            decision_id = str(pilot.get("decision_id") or "") if isinstance(pilot, dict) else ""
            existing_ids.append(decision_id)
            stored_run_config = result.get("pilot_run_config")
            if stored_run_config is not None and stored_run_config != run_config:
                raise ValueError(f"existing seed {seed} run config mismatch at row {row_index}")
            if stored_run_config is None:
                legacy_config_rows += 1
            expected_outer = int(btn_outer if position["position"] == "btn" else bb_outer)
            expected_inner = int(bb_inner if position["position"] == "bb" else 0)
            if (
                result.get("schema") != SCHEMA
                or result.get("method") != METHOD
                or result.get("information_model") != INFORMATION_MODEL
                or result.get("hu_exact") is not False
                or result.get("inner_t4_exact") is not True
                or int(result.get("seed", -1)) != int(seed)
                or int(result.get("outer_samples", -1)) != expected_outer
                or int(result.get("response_inner_samples", -1)) != expected_inner
                or result.get("position") != position["position"]
            ):
                raise ValueError(f"existing seed {seed} result metadata mismatch at row {row_index}")
            if int(result.get("legal_actions", -1)) != int(result.get("evaluated_actions", -2)):
                raise ValueError(f"existing seed {seed} candidate coverage mismatch at row {row_index}")
            action_keys = [candidate.get("action_key") for candidate in result.get("candidates") or []]
            if len(action_keys) != int(result["legal_actions"]) or len(set(action_keys)) != len(action_keys):
                raise ValueError(f"existing seed {seed} action-key coverage mismatch at row {row_index}")
        if existing_ids != requested_ids[: len(existing_ids)]:
            raise ValueError(f"existing seed {seed} decision IDs are not a requested-position prefix")

    existing_rows = len(results)
    mode = "a" if output_path.exists() else "w"
    with output_path.open(mode, encoding="utf-8") as handle:
        for index, position in enumerate(positions[existing_rows:], start=existing_rows + 1):
            actor = position["position"]
            result = evaluate_t3_hu_sampled(
                position,
                seed=seed,
                outer_samples=btn_outer if actor == "btn" else bb_outer,
                response_inner_samples=bb_inner,
                rust_solver_path=rust_solver_path,
                rust_timeout_s=rust_timeout_s,
            )
            result["pilot"] = position["pilot"]
            result["pilot_run_config"] = run_config
            expected_outer = int(btn_outer if actor == "btn" else bb_outer)
            expected_inner = int(bb_inner if actor == "bb" else 0)
            if (
                int(result.get("seed", -1)) != int(seed)
                or int(result.get("outer_samples", -1)) != expected_outer
                or int(result.get("response_inner_samples", -1)) != expected_inner
                or result.get("position") != actor
            ):
                raise RuntimeError(
                    f"sampled HU result metadata mismatch for {position['pilot']['decision_id']}"
                )
            if result["legal_actions"] != result["evaluated_actions"]:
                raise RuntimeError(
                    f"candidate coverage failure for {position['pilot']['decision_id']}: "
                    f"{result['evaluated_actions']}/{result['legal_actions']}"
                )
            action_keys = [candidate["action_key"] for candidate in result["candidates"]]
            if len(set(action_keys)) != result["legal_actions"]:
                raise RuntimeError("sampled HU result contains missing or duplicate action keys")
            handle.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            results.append(result)
            print(
                f"seed={seed} {index}/{len(positions)} {actor} "
                f"actions={result['legal_actions']} elapsed_ms={result['elapsed_ms']:.1f}",
                flush=True,
            )
    resume_details = {
        "seed": int(seed),
        "output": str(output_path),
        "requested_rows": len(positions),
        "existing_rows": existing_rows,
        "resumed": existing_rows > 0,
        "complete_before_run": existing_rows == len(positions),
        "new_rows": len(positions) - existing_rows,
        "legacy_rows_without_run_config": legacy_config_rows,
        "run_config": run_config,
    }
    return results, resume_details


def _candidate_map(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {candidate["action_key"]: candidate for candidate in result.get("candidates") or []}


def compare_seed_runs(
    run_a: Sequence[dict[str, Any]],
    run_b: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_id_b = {result["pilot"]["decision_id"]: result for result in run_b}
    comparisons: list[dict[str, Any]] = []
    for a in run_a:
        decision_id = a["pilot"]["decision_id"]
        b = by_id_b[decision_id]
        map_a = _candidate_map(a)
        map_b = _candidate_map(b)
        if set(map_a) != set(map_b):
            raise RuntimeError(f"seed action-set mismatch for {decision_id}")
        best_a = a["best"]["action_key"]
        best_b = b["best"]["action_key"]
        score_a_in_b = float(map_b[best_a]["metrics"]["score"])
        score_b_in_a = float(map_a[best_b]["metrics"]["score"])
        best_score_a = float(map_a[best_a]["metrics"]["score"])
        best_score_b = float(map_b[best_b]["metrics"]["score"])
        se_best_a_b = float(map_b[best_a]["metrics"]["standard_error"])
        se_best_b = float(map_b[best_b]["metrics"]["standard_error"])
        se_best_b_a = float(map_a[best_b]["metrics"]["standard_error"])
        se_best_a = float(map_a[best_a]["metrics"]["standard_error"])
        regret_a_to_b = max(0.0, best_score_b - score_a_in_b)
        regret_b_to_a = max(0.0, best_score_a - score_b_in_a)
        compatible = (
            regret_a_to_b <= 1.96 * math.sqrt(se_best_a_b**2 + se_best_b**2)
            and regret_b_to_a <= 1.96 * math.sqrt(se_best_b_a**2 + se_best_a**2)
        )
        shared_drift = [
            abs(float(map_a[key]["metrics"]["score"]) - float(map_b[key]["metrics"]["score"]))
            for key in map_a
        ]
        comparisons.append(
            {
                "decision_id": decision_id,
                "cluster_id": a["pilot"]["cluster_id"],
                "stratum": a["pilot"]["stratum"],
                "position": a["position"],
                "visible_joker_count": a["pilot"]["visible_joker_count"],
                "strict_top1_agreement": best_a == best_b,
                "ci_compatible_agreement": bool(compatible),
                "cross_regret_a_to_b": regret_a_to_b,
                "cross_regret_b_to_a": regret_b_to_a,
                "candidate_ev_drift_mean": statistics.fmean(shared_drift),
                "candidate_ev_drift_max": max(shared_drift, default=0.0),
                "runtime_ms_a": float(a["elapsed_ms"]),
                "runtime_ms_b": float(b["elapsed_ms"]),
                "legal_actions": int(a["legal_actions"]),
            }
        )

    def aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        n = len(rows)
        regrets = [
            max(float(row["cross_regret_a_to_b"]), float(row["cross_regret_b_to_a"]))
            for row in rows
        ]
        drifts = [float(row["candidate_ev_drift_mean"]) for row in rows]
        runtimes = [float(row["runtime_ms_a"]) for row in rows] + [
            float(row["runtime_ms_b"]) for row in rows
        ]
        return {
            "positions": n,
            "strict_top1_agreement": sum(bool(row["strict_top1_agreement"]) for row in rows) / max(n, 1),
            "ci_compatible_agreement": sum(bool(row["ci_compatible_agreement"]) for row in rows) / max(n, 1),
            "cross_regret": {
                "mean": statistics.fmean(regrets) if regrets else 0.0,
                "p95": _percentile(regrets, 0.95),
                "max": max(regrets, default=0.0),
            },
            "candidate_ev_drift": {
                "mean": statistics.fmean(drifts) if drifts else 0.0,
                "p95": _percentile(drifts, 0.95),
                "max": max(drifts, default=0.0),
            },
            "runtime_ms": {
                "p50": _percentile(runtimes, 0.50),
                "p95": _percentile(runtimes, 0.95),
                "max": max(runtimes, default=0.0),
                "over_5s": sum(runtime > 5000.0 for runtime in runtimes),
            },
        }

    strata = sorted({row["stratum"] for row in comparisons})
    positions = sorted({row["position"] for row in comparisons})
    summary = {
        "schema": SCHEMA,
        "overall": aggregate(comparisons),
        "by_position": {
            position: aggregate([row for row in comparisons if row["position"] == position])
            for position in positions
        },
        "by_stratum": {
            stratum: aggregate([row for row in comparisons if row["stratum"] == stratum])
            for stratum in strata
        },
    }
    return comparisons, summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    positions_input_value = str(getattr(args, "positions_input", "") or "")
    input_value = str(getattr(args, "input", "") or "")
    if positions_input_value:
        source_path = Path(positions_input_value)
        selected = load_sanitized_positions(source_path)
        source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        selection_summary = {
            "schema": SCHEMA,
            "mode": "positions_input",
            "input": str(source_path),
            "input_sha256": source_hash,
            "selected": len(selected),
        }
    else:
        if not input_value:
            raise ValueError("--input is required unless --positions-input is supplied")
        source_path = Path(input_value)
        records = list(iter_jsonl(source_path))
        selected, selection_summary = select_balanced_positions(
            records,
            selection_seed=args.selection_seed,
            max_root=args.max_root,
        )
        source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
        selection_summary["mode"] = "balanced_selection"
        selection_summary["input"] = str(source_path)
        selection_summary["input_sha256"] = source_hash

    comparisons_input_value = str(getattr(args, "comparisons_input", "") or "")
    comparisons = None
    comparisons_hash = None
    if comparisons_input_value:
        comparisons_path = Path(comparisons_input_value)
        comparisons = _read_jsonl(comparisons_path, label="comparisons")
        comparisons_hash = hashlib.sha256(comparisons_path.read_bytes()).hexdigest()
    elif bool(getattr(args, "disagreements_only", False)):
        raise ValueError("--disagreements-only requires --comparisons-input")

    selected, refinement_summary = refine_positions(
        selected,
        position_filter=str(getattr(args, "position_filter", "all") or "all"),
        limit=int(getattr(args, "limit", 0) or 0),
        comparisons=comparisons,
        disagreements_only=bool(getattr(args, "disagreements_only", False)),
    )
    if not selected:
        raise ValueError("position/refinement filters produced no requested positions")
    refinement_summary.update(
        {
            "positions_input": positions_input_value or None,
            "comparisons_input": comparisons_input_value or None,
            "comparisons_input_sha256": comparisons_hash,
        }
    )
    selection_summary["refinement"] = refinement_summary
    (output_dir / "positions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in selected),
        encoding="utf-8",
    )
    (output_dir / "selection_summary.json").write_text(
        json.dumps(selection_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    seeds = [int(part) for part in str(args.eval_seeds).split(",") if part.strip()]
    if len(seeds) != 2:
        raise ValueError("--eval-seeds requires exactly two comma-separated integers")
    runs: list[list[dict[str, Any]]] = []
    resume_details: list[dict[str, Any]] = []
    started = time.perf_counter()
    for seed in seeds:
        seed_run, seed_resume = _run_seed(
            selected,
            seed=seed,
            btn_outer=args.btn_outer,
            bb_outer=args.bb_outer,
            bb_inner=args.bb_inner,
            rust_solver_path=args.rust_solver or None,
            rust_timeout_s=args.rust_timeout_s,
            output_path=output_dir / "runs" / f"seed_{seed}.jsonl",
        )
        runs.append(seed_run)
        resume_details.append(seed_resume)
    comparisons, summary = compare_seed_runs(runs[0], runs[1])
    summary.update(
        {
            "input": str(source_path),
            "input_sha256": source_hash,
            "selection": selection_summary,
            "refinement": refinement_summary,
            "resume": {
                "runs": resume_details,
                "total_existing_rows": sum(item["existing_rows"] for item in resume_details),
                "total_new_rows": sum(item["new_rows"] for item in resume_details),
            },
            "eval_seeds": seeds,
            "btn_outer": args.btn_outer,
            "bb_outer": args.bb_outer,
            "bb_inner": args.bb_inner,
            "elapsed_s": time.perf_counter() - started,
            "method": METHOD,
            "information_model": INFORMATION_MODEL,
            "hu_exact": False,
            "inner_t4_exact": True,
        }
    )
    (output_dir / "comparisons.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in comparisons),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run balanced T3 HU sampled two-seed gate")
    parser.add_argument("--input", default="", help="Source gate JSONL for a new balanced selection")
    parser.add_argument(
        "--positions-input",
        default="",
        help="Reuse an already-sanitized positions.jsonl instead of selecting from --input",
    )
    parser.add_argument(
        "--comparisons-input",
        default="",
        help="Prior comparisons.jsonl used for deterministic regret-ranked refinement",
    )
    parser.add_argument(
        "--disagreements-only",
        action="store_true",
        help="Keep only prior strict Top1 disagreements (requires --comparisons-input)",
    )
    parser.add_argument("--position-filter", choices=["all", "bb", "btn"], default="all")
    parser.add_argument("--limit", type=int, default=0, help="Limit positions after all filters (0 keeps all)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--selection-seed", type=int, default=20260712)
    parser.add_argument("--eval-seeds", default="20260712,20260713")
    parser.add_argument("--max-root", type=int, default=99)
    parser.add_argument("--btn-outer", type=int, default=8)
    parser.add_argument("--bb-outer", type=int, default=2)
    parser.add_argument("--bb-inner", type=int, default=1)
    parser.add_argument("--rust-solver", default="")
    parser.add_argument("--rust-timeout-s", type=float, default=300.0)
    args = parser.parse_args(list(argv) if argv is not None else None)
    print(json.dumps(run(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
