"""Postmortem for HU Turn2 Stage8 C3 No-Go.

This analyzer is intentionally diagnostic-only. It reads the C3 runtime
decision logs and summarizes why the runtime proxy gate failed to turn the
C1f teacher-oracle signal into larger seat-swap EV.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_C3_DIR = Path("outputs/evals/hu_turn2_stage8_c3_larger_seat_swap")
DEFAULT_C2_DIR = Path("outputs/evals/hu_turn2_stage8_c2_small")
DEFAULT_C1F_DIR = Path("outputs/hu_turn2_stage1_pilot_training_c1f_expanded_calibration")
DEFAULT_OUTPUT_DIR = Path("outputs/evals/hu_turn2_stage8_c3_postmortem")
RUNTIME_DECISION_FILE = "hu_turn2_stage8_20k_runtime_decisions.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c3-dir", type=Path, default=DEFAULT_C3_DIR)
    parser.add_argument("--c2-dir", type=Path, default=DEFAULT_C2_DIR)
    parser.add_argument("--c1f-dir", type=Path, default=DEFAULT_C1F_DIR)
    parser.add_argument("--gcp-run-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-loss-limit", type=int, default=100)
    return parser.parse_args()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_cards(cards: Any) -> list[str]:
    if not isinstance(cards, list):
        return []
    return sorted(str(card) for card in cards)


def canonical_board(board: Any) -> dict[str, list[str]]:
    if not isinstance(board, dict):
        return {"top": [], "middle": [], "bottom": []}
    return {
        "top": canonical_cards(board.get("top")),
        "middle": canonical_cards(board.get("middle")),
        "bottom": canonical_cards(board.get("bottom")),
    }


def state_key(row: dict[str, Any]) -> str:
    """Config-independent key for comparing fired sets across C3 configs."""

    payload = {
        "hand_id": row.get("hand_id"),
        "paired_index": row.get("paired_index"),
        "seat_swap": row.get("seat_swap"),
        "seat": row.get("seat"),
        "hero_board": canonical_board(row.get("hero_board")),
        "opponent_board": canonical_board(row.get("opponent_board")),
        "cards_to_place": canonical_cards(row.get("cards_to_place")),
        "dead_cards": canonical_cards(row.get("dead_cards")),
    }
    return canonical_json(payload)


def action_signature(action: Any) -> str:
    if not isinstance(action, dict):
        return canonical_json(action)
    placements = action.get("placements")
    if isinstance(placements, list):
        placement_items = sorted((str(item[0]), str(item[1])) for item in placements if isinstance(item, (list, tuple)) and len(item) >= 2)
    else:
        placement_items = []
    payload = {
        "placements": placement_items,
        "discards": canonical_cards(action.get("discards")),
    }
    return canonical_json(payload)


def config_to_proxy_id(config_id: str) -> str:
    parts = {}
    for token in config_id.split("_"):
        if token.startswith("m"):
            parts["m"] = token[1:]
        elif token.startswith("g"):
            parts["g"] = token[1:]
    if "m" not in parts or "g" not in parts:
        return config_id
    return f"A_m{parts['m']}_g{parts['g']}"


def discover_gcp_run_dir(c3_dir: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    run_name_path = c3_dir / "gcp_run_name.txt"
    if not run_name_path.exists():
        raise FileNotFoundError(f"missing {run_name_path}; pass --gcp-run-dir explicitly")
    run_name = run_name_path.read_text(encoding="utf-8").strip()
    if not run_name:
        raise RuntimeError(f"{run_name_path} is empty")
    return Path("outputs/gcp_runs") / run_name


def load_runtime_decisions(gcp_run_dir: Path) -> list[dict[str, Any]]:
    paths = sorted((gcp_run_dir / "results").rglob(RUNTIME_DECISION_FILE))
    if not paths and gcp_run_dir.exists():
        paths = sorted(gcp_run_dir.rglob(RUNTIME_DECISION_FILE))
    decisions: list[dict[str, Any]] = []
    for path in paths:
        for row in iter_jsonl(path) or ():
            row["_source_path"] = str(path)
            row["_state_key"] = state_key(row)
            row["_stage8_action_signature"] = action_signature(row.get("stage8_action"))
            row["_final_action_signature"] = action_signature(row.get("final_action"))
            decisions.append(row)
    if not decisions:
        raise RuntimeError(f"no runtime decisions found below {gcp_run_dir}")
    return decisions


def seat_swap_by_config(c3_dir: Path) -> dict[str, dict[str, str]]:
    return {str(row.get("config_id", "")): row for row in read_csv(c3_dir / "c3_seat_swap_results.csv") if row.get("config_id")}


def fired_overlap_rows(decisions: list[dict[str, Any]], seat_swap_rows: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in decisions:
        by_config[str(row.get("config_id", ""))].append(row)

    output: list[dict[str, Any]] = []
    fired_by_config: dict[str, dict[str, dict[str, Any]]] = {}
    for config_id, rows in sorted(by_config.items()):
        fired = {str(row["_state_key"]): row for row in rows if bool(row.get("override_fired"))}
        fired_by_config[config_id] = fired
        seats = Counter(str(row.get("seat", "")) for row in fired.values())
        metrics = seat_swap_rows.get(config_id, {})
        output.append(
            {
                "row_type": "config_summary",
                "config_id": config_id,
                "fired_count": len([row for row in rows if bool(row.get("override_fired"))]),
                "unique_fired_states": len(fired),
                "decision_count": len(rows),
                "override_rate": len(fired) / max(len(rows), 1),
                "first_fired": seats.get("first", 0),
                "second_fired": seats.get("second", 0),
                "ev_per_hand": safe_float(metrics.get("aggregate_ev_per_hand")),
                "ci95_low": safe_float(metrics.get("ci95_low_seed_means")),
                "ci95_high": safe_float(metrics.get("ci95_high_seed_means")),
            }
        )

    configs = sorted(fired_by_config)
    for index, config_a in enumerate(configs):
        for config_b in configs[index + 1 :]:
            fired_a = fired_by_config[config_a]
            fired_b = fired_by_config[config_b]
            keys_a = set(fired_a)
            keys_b = set(fired_b)
            shared = keys_a.intersection(keys_b)
            union = keys_a.union(keys_b)
            same_action = sum(
                1
                for key in shared
                if fired_a[key].get("_stage8_action_signature") == fired_b[key].get("_stage8_action_signature")
            )
            first_overlap = sum(1 for key in shared if str(fired_a[key].get("seat")) == "first")
            second_overlap = sum(1 for key in shared if str(fired_a[key].get("seat")) == "second")
            metrics_a = seat_swap_rows.get(config_a, {})
            metrics_b = seat_swap_rows.get(config_b, {})
            output.append(
                {
                    "row_type": "pairwise_overlap",
                    "config_a": config_a,
                    "config_b": config_b,
                    "fired_a": len(keys_a),
                    "fired_b": len(keys_b),
                    "intersection_states": len(shared),
                    "union_states": len(union),
                    "jaccard_overlap": len(shared) / max(len(union), 1),
                    "same_action_overlap": same_action,
                    "same_action_overlap_rate": same_action / max(len(shared), 1),
                    "first_overlap": first_overlap,
                    "second_overlap": second_overlap,
                    "ev_per_hand_a": safe_float(metrics_a.get("aggregate_ev_per_hand")),
                    "ev_per_hand_b": safe_float(metrics_b.get("aggregate_ev_per_hand")),
                    "ev_diff_a_minus_b": safe_float(metrics_a.get("aggregate_ev_per_hand"))
                    - safe_float(metrics_b.get("aggregate_ev_per_hand")),
                }
            )
    return output


def no_override_reason_rows(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_config: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in decisions:
        by_config[str(row.get("config_id", ""))].append(row)
    output: list[dict[str, Any]] = []
    for config_id, rows in sorted(by_config.items()):
        counts = Counter(
            "override_fired" if bool(row.get("override_fired")) else str(row.get("no_override_reason", "unknown"))
            for row in rows
        )
        total = len(rows)
        for reason, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
            output.append(
                {
                    "config_id": config_id,
                    "no_override_reason": reason,
                    "count": count,
                    "rate": count / max(total, 1),
                    "decision_count": total,
                    "reason_group": reason_group(reason),
                }
            )
    return output


def reason_group(reason: str) -> str:
    return {
        "below_stage8_margin": "below_predicted_delta",
        "below_gate_threshold": "below_gate",
        "same_as_baseline": "same_as_baseline",
        "override_fired": "override_fired",
        "illegal_candidate": "illegal_or_safety",
        "nan_prediction": "illegal_or_safety",
        "feature_failed": "model_or_feature_failure",
        "model_load_failed": "model_or_feature_failure",
    }.get(reason, "other")


def classify_fired(row: dict[str, Any]) -> str:
    score = safe_float(row.get("candidate_seat_score"))
    predicted_delta = safe_float(row.get("predicted_delta"))
    gate = safe_float(row.get("gate_probability"))
    if score < -5.0:
        return "tail_loss_audit_required"
    if score < 0.0 and predicted_delta >= 2.5 and gate >= 0.90:
        return "false_positive_gate"
    if score < 0.0:
        return "runtime_loss"
    if predicted_delta < 3.0:
        return "low_margin_noise"
    return "positive_or_neutral_runtime"


def fired_state_quality_rows(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in decisions:
        if not bool(row.get("override_fired")):
            continue
        output.append(
            {
                "config_id": row.get("config_id"),
                "state_key": row.get("_state_key"),
                "seed": row.get("seed"),
                "hand_id": row.get("hand_id"),
                "hand_seed": row.get("hand_seed"),
                "paired_index": row.get("paired_index"),
                "seat": row.get("seat"),
                "seat_swap": row.get("seat_swap"),
                "predicted_delta": safe_float(row.get("predicted_delta")),
                "gate_probability": safe_float(row.get("gate_probability")),
                "predicted_ev_margin": "",
                "candidate_rank": "",
                "reference_margin_raw": safe_float(row.get("reference_margin_raw")),
                "model_score": safe_float(row.get("model_score")),
                "candidate_seat_score": safe_float(row.get("candidate_seat_score")),
                "realized_delta_proxy": safe_float(row.get("candidate_seat_score")),
                "baseline_action_index": safe_int(row.get("baseline_action_index"), -1),
                "stage8_action_index": safe_int(row.get("stage8_action_index"), -1),
                "final_action_index": safe_int(row.get("final_action_index"), -1),
                "teacher_ev_gain_available": 0,
                "source_bucket_available": 0,
                "actual_bucket_available": 0,
                "failure_label": classify_fired(row),
            }
        )
    output.sort(key=lambda item: (str(item["config_id"]), safe_float(item["candidate_seat_score"])))
    return output


def replay_ready(row: dict[str, Any]) -> bool:
    required = ("hero_board", "opponent_board", "dead_cards", "cards_to_place", "baseline_action", "stage8_action")
    return all(row.get(field) not in (None, "", []) for field in required)


def top_loss_audit_rows(decisions: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in decisions:
        if bool(row.get("override_fired")):
            key = (str(row.get("_state_key")), str(row.get("_stage8_action_signature")))
            grouped[key].append(row)

    rows: list[dict[str, Any]] = []
    for group in grouped.values():
        worst = min(group, key=lambda item: safe_float(item.get("candidate_seat_score")))
        rows.append(
            {
                "configs": sorted({str(item.get("config_id")) for item in group}),
                "seed": worst.get("seed"),
                "hand_id": worst.get("hand_id"),
                "hand_seed": worst.get("hand_seed"),
                "paired_index": worst.get("paired_index"),
                "seat": worst.get("seat"),
                "seat_swap": worst.get("seat_swap"),
                "candidate_seat_score": safe_float(worst.get("candidate_seat_score")),
                "predicted_delta": safe_float(worst.get("predicted_delta")),
                "gate_probability": safe_float(worst.get("gate_probability")),
                "reference_margin_raw": safe_float(worst.get("reference_margin_raw")),
                "model_score": safe_float(worst.get("model_score")),
                "failure_label": classify_fired(worst),
                "replay_ready": replay_ready(worst),
                "selected_mc4096_8192_reason": selected_high_mc_reason(worst),
                "hero_board": worst.get("hero_board"),
                "opponent_board": worst.get("opponent_board"),
                "dead_cards": worst.get("dead_cards"),
                "cards_to_place": worst.get("cards_to_place"),
                "baseline_action": worst.get("baseline_action"),
                "stage8_action": worst.get("stage8_action"),
                "final_action": worst.get("final_action"),
                "state_key": worst.get("_state_key"),
            }
        )
    rows.sort(key=lambda item: safe_float(item.get("candidate_seat_score")))
    return rows[:limit]


def selected_high_mc_reason(row: dict[str, Any]) -> str:
    score = safe_float(row.get("candidate_seat_score"))
    if score < 0.0:
        return "c3_runtime_override_loss"
    if safe_float(row.get("gate_probability")) >= 0.90 and safe_float(row.get("predicted_delta")) >= 2.5:
        return "c3_fired_high_confidence"
    return "c3_fired_boundary"


def oracle_vs_proxy_gap_rows(
    *,
    c2_dir: Path,
    c1f_dir: Path,
    c3_rows: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    c2_overlap = read_csv(c2_dir / "proxy_vs_oracle_overlap.csv")
    c2_proxy = read_csv(c2_dir / "runtime_proxy_filter_eval.csv")
    c1f_metrics = read_csv(c1f_dir / "c1f_candidate_metrics.csv")
    c1f_by_candidate = {row.get("candidate_id", ""): row for row in c1f_metrics}
    c2_proxy_by_id = {row.get("proxy_id", ""): row for row in c2_proxy}
    c3_by_proxy = {config_to_proxy_id(config_id): row for config_id, row in c3_rows.items()}
    target_proxy_ids = sorted(c3_by_proxy)

    output: list[dict[str, Any]] = []
    for row in c2_overlap:
        proxy_id = str(row.get("proxy_id", ""))
        if proxy_id not in target_proxy_ids:
            continue
        oracle_id = str(row.get("oracle_candidate_id", ""))
        c3 = c3_by_proxy.get(proxy_id, {})
        c2 = c2_proxy_by_id.get(proxy_id, {})
        c1f = c1f_by_candidate.get(oracle_id, {})
        output.append(
            {
                "comparison_scope": "c2_heldout_proxy_vs_oracle_plus_c3_runtime_result",
                "oracle_candidate_id": oracle_id,
                "proxy_id": proxy_id,
                "c3_config_id": c3.get("config_id", ""),
                "c3_direct_oracle_match_available": 0,
                "oracle_fires_c2": safe_int(row.get("oracle_fires")),
                "proxy_fires_c2": safe_int(row.get("proxy_fires")),
                "intersection_c2": safe_int(row.get("intersection")),
                "precision_vs_oracle_c2": safe_float(row.get("precision_vs_oracle")),
                "recall_vs_oracle_c2": safe_float(row.get("recall_vs_oracle")),
                "jaccard_overlap_c2": safe_float(row.get("jaccard_overlap")),
                "proxy_avg_gain_c2": safe_float(c2.get("avg_gain")),
                "proxy_fp_rate_c2": safe_float(c2.get("false_positive_rate")),
                "proxy_unbiased_fires_c2": safe_int(c2.get("unbiased_fires")),
                "proxy_first_fires_c2": safe_int(c2.get("first_fires")),
                "proxy_second_fires_c2": safe_int(c2.get("second_fires")),
                "oracle_fires_c1f": safe_int(c1f.get("fires")),
                "oracle_avg_gain_c1f": safe_float(c1f.get("avg_gain")),
                "oracle_fp_rate_c1f": safe_float(c1f.get("false_positive_rate")),
                "c3_ev_per_hand": safe_float(c3.get("aggregate_ev_per_hand")),
                "c3_ci95_low": safe_float(c3.get("ci95_low_seed_means")),
                "c3_override_count": safe_int(c3.get("override_count")),
                "c3_runtime_override_rate": safe_float(c3.get("runtime_override_rate")),
                "gap_interpretation": "runtime_proxy_has_only_indirect_oracle_evidence_in_c3_logs",
            }
        )
    if not output:
        for proxy_id in target_proxy_ids:
            c3 = c3_by_proxy.get(proxy_id, {})
            output.append(
                {
                    "comparison_scope": "c3_runtime_only",
                    "proxy_id": proxy_id,
                    "c3_config_id": c3.get("config_id", ""),
                    "c3_direct_oracle_match_available": 0,
                    "c3_ev_per_hand": safe_float(c3.get("aggregate_ev_per_hand")),
                    "c3_ci95_low": safe_float(c3.get("ci95_low_seed_means")),
                    "c3_override_count": safe_int(c3.get("override_count")),
                    "c3_runtime_override_rate": safe_float(c3.get("runtime_override_rate")),
                    "gap_interpretation": "c2_proxy_vs_oracle_overlap_missing",
                }
            )
    output.sort(
        key=lambda item: (
            str(item.get("proxy_id", "")),
            str(item.get("oracle_candidate_id", "")),
        )
    )
    return output


def write_label_design(path: Path) -> None:
    lines = [
        "# Stage8b Safe Override Label Design",
        "",
        "C1f showed that teacher-EV LCB filters are strong, but C3 showed the current runtime proxy does not reproduce that advantage in seat-swap EV. Stage8b should learn a deployable confidence target instead of using teacher LCB directly at runtime.",
        "",
        "## Labels",
        "",
        "- `teacher_delta_lcb_196 = teacher_delta_candidate_vs_baseline - 1.96 * SE_delta_candidate_vs_baseline`",
        "- `teacher_delta_lcb_164 = teacher_delta_candidate_vs_baseline - 1.64 * SE_delta_candidate_vs_baseline`",
        "- `safe_lcb196_label = 1` when `teacher_delta_lcb_196 > 0`, otherwise 0 for confident negatives and low-weight gray for uncertain rows",
        "- `safe_lcb164_label = 1` when `teacher_delta_lcb_164 > 0`, otherwise 0 for confident negatives and low-weight gray for uncertain rows",
        "- `hard_negative_label = 1` when current runtime proxy confidence is high but teacher or high-MC delta is `<= 0`",
        "- `gray_label = 1` for near-zero or high-SE rows that should receive low gate-loss weight",
        "",
        "## Runtime Safety",
        "",
        "Teacher EV, teacher gain LCB, MC512 EV per action, and actual bucket labels remain analysis-only. Production runtime should use only model outputs and runtime-observable state/action fields.",
        "",
        "## Required Cache Columns",
        "",
        "- `teacher_delta_candidate_vs_baseline`",
        "- `SE_delta_candidate_vs_baseline`",
        "- `teacher_delta_lcb_196`",
        "- `teacher_delta_lcb_164`",
        "- `safe_lcb196_label`",
        "- `safe_lcb164_label`",
        "- `hard_negative_label`",
        "- `gray_label`",
        "- `hard_negative_source`",
        "",
        "## Selected High-MC Use",
        "",
        "MC4096/8192 should be used to upgrade labels for C3 fired losses, near-threshold rows, oracle-positive/proxy-missed rows, and high-confidence proxy false positives. It is diagnostic and training-label work, not production approval.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_training_plan(path: Path) -> None:
    lines = [
        "# HU Turn2 Stage8b Training Plan",
        "",
        "Stage8b should not be a wider threshold search over the current proxy. It should add a deployable confidence head trained to predict safe override labels derived from teacher LCB and selected high-MC replay.",
        "",
        "## Model",
        "",
        "- Keep existing EV, delta, and ranking/listwise heads.",
        "- Add `safe_override_head` for `safe_lcb196_label` or `safe_lcb164_label`.",
        "- Optionally add a separate `hard_negative_head` or use a high-weight negative term in the safe head.",
        "",
        "## Loss",
        "",
        "- Huber losses for EV and delta stay in place.",
        "- Pairwise/listwise ranking loss stays in place.",
        "- Add BCE or focal loss for `safe_override_head`.",
        "- Weight hard negatives higher than ordinary negatives.",
        "- Give gray rows low or zero gate-loss weight.",
        "",
        "## Runtime Gate",
        "",
        "Candidate gate for validation only:",
        "",
        "```text",
        "override if",
        "  predicted_delta >= m",
        "  and safe_override_probability >= p",
        "  and candidate_rank <= k",
        "  and optional predicted_EV_margin >= pm",
        "```",
        "",
        "Recommended grid:",
        "",
        "- `m`: 2.0, 2.25, 2.5, 2.75",
        "- `p`: 0.80, 0.85, 0.90, 0.95",
        "- `k`: 1, 2, 3, 5",
        "- `pm`: none, 0.25, 0.50",
        "",
        "## Evaluation Gates",
        "",
        "1. Holdout teacher cache: low FP, positive avg gain, p95/p99 loss controlled.",
        "2. Proxy-vs-oracle overlap: safe head improves recall without losing precision.",
        "3. C3-style seat-swap: non-overlapping seeds, `--seed-stride`, T3 Stage7 m5_r10 fixed.",
        "4. Only after a positive larger seat-swap should selected MC4096/8192 and C4 be used for promotion evidence.",
        "",
        "50k teacher, T1, and production remain No-Go until Stage8b passes these gates.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_recommended_next_step(path: Path, *, overlap_rows: list[dict[str, Any]], no_override_rows: list[dict[str, Any]], top_loss_count: int) -> None:
    m25 = next((row for row in overlap_rows if row.get("row_type") == "config_summary" and row.get("config_id") == "m2.5_r0_g0.9"), {})
    no_override = [row for row in no_override_rows if row.get("config_id") == "m2.5_r0_g0.9"]
    dominant_reason = max(no_override, key=lambda row: safe_int(row.get("count")), default={})
    lines = [
        "# C3 Postmortem Recommended Next Step",
        "",
        "## Decision",
        "",
        "- C3 larger seat-swap: `No-Go`",
        "- production / P2 fixed: `No-Go`",
        "- 50k teacher: `No-Go`",
        "- T1 training: `No-Go`",
        "- next: `Stage8b safe_override/confidence head design and selected high-MC diagnostic labels`",
        "",
        "## Main Finding",
        "",
        f"- main candidate `m2.5_r0_g0.9` fired `{safe_int(m25.get('fired_count'))}` unique states with override rate `{safe_float(m25.get('override_rate')):.4f}`.",
        f"- dominant no-override reason: `{dominant_reason.get('no_override_reason', '')}`.",
        f"- selected top-loss audit states written: `{top_loss_count}`.",
        "",
        "The C1f oracle LCB signal is not deployable directly. The current runtime proxy underfires and does not reproduce the oracle advantage strongly enough in C3 seat-swap.",
        "",
        "## Next Implementation Command",
        "",
        "```powershell",
        ".\\scripts\\Run-HuTurn2Stage8C3Postmortem.ps1",
        "```",
        "",
        "Equivalent direct command:",
        "",
        "```powershell",
        "python -m ofc_regular.analyze_hu_turn2_stage8_c3_postmortem `",
        "  --c3-dir outputs/evals/hu_turn2_stage8_c3_larger_seat_swap `",
        "  --output-dir outputs/evals/hu_turn2_stage8_c3_postmortem",
        "```",
        "",
        "After review, build Stage8b labels on the existing 20k teacher EV cache and selected MC4096/8192 replay set. Do not start 50k teacher or T1.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(path: Path, *, decisions: list[dict[str, Any]], seat_swap: dict[str, dict[str, str]], output_dir: Path) -> None:
    rows = [
        "| config | EV/hand | CI low | CI high | overrides | override rate | teacher avg gain | FP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for config_id, row in sorted(seat_swap.items()):
        rows.append(
            "| {config} | {ev:.4f} | {low:.4f} | {high:.4f} | {overrides} | {rate:.4f} | {gain:.4f} | {fp:.4f} |".format(
                config=config_id,
                ev=safe_float(row.get("aggregate_ev_per_hand")),
                low=safe_float(row.get("ci95_low_seed_means")),
                high=safe_float(row.get("ci95_high_seed_means")),
                overrides=safe_int(row.get("override_count")),
                rate=safe_float(row.get("runtime_override_rate")),
                gain=safe_float(row.get("avg_gain_on_override")),
                fp=safe_float(row.get("false_positive_override_rate")),
            )
        )
    lines = [
        "# HU Turn2 Stage8 C3 No-Go Postmortem",
        "",
        "C3 is a decision No-Go. This postmortem is diagnostic only and does not authorize production, 50k teacher, T1, or P2 fixation.",
        "",
        "- T3 continuation: `Stage7_candidate_A_m5_r10` fixed",
        "- Stage8 mode: `HU T2 selective override`, not full replacement",
        f"- runtime decisions read: `{len(decisions)}`",
        f"- output directory: `{output_dir}`",
        "",
        "## C3 Result",
        "",
        *rows,
        "",
        "## Interpretation",
        "",
        "The C1f teacher oracle LCB filters were strong on teacher EV, but the C3 runtime proxy gate did not reproduce that signal in larger seat-swap. The current threshold family also underfires: the best candidate overrides less than 1% of T2 decisions, so even good individual teacher gains have little aggregate EV impact.",
        "",
        "## Produced Artifacts",
        "",
        "- `c3_fired_overlap.csv`",
        "- `c3_no_override_reason.csv`",
        "- `c3_fired_state_quality.csv`",
        "- `c3_top_loss_audit_states.jsonl`",
        "- `oracle_vs_proxy_gap.csv`",
        "- `safe_override_label_design.md`",
        "- `stage8b_training_plan.md`",
        "- `recommended_next_step.md`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gcp_run_dir = discover_gcp_run_dir(args.c3_dir, args.gcp_run_dir)
    decisions = load_runtime_decisions(gcp_run_dir)
    seat_swap = seat_swap_by_config(args.c3_dir)

    overlap = fired_overlap_rows(decisions, seat_swap)
    reasons = no_override_reason_rows(decisions)
    fired_quality = fired_state_quality_rows(decisions)
    top_loss = top_loss_audit_rows(decisions, args.top_loss_limit)
    gap = oracle_vs_proxy_gap_rows(c2_dir=args.c2_dir, c1f_dir=args.c1f_dir, c3_rows=seat_swap)

    write_csv(args.output_dir / "c3_fired_overlap.csv", overlap)
    write_csv(args.output_dir / "c3_no_override_reason.csv", reasons)
    write_csv(args.output_dir / "c3_fired_state_quality.csv", fired_quality)
    write_jsonl(args.output_dir / "c3_top_loss_audit_states.jsonl", top_loss)
    write_csv(args.output_dir / "oracle_vs_proxy_gap.csv", gap)
    write_label_design(args.output_dir / "safe_override_label_design.md")
    write_training_plan(args.output_dir / "stage8b_training_plan.md")
    write_recommended_next_step(
        args.output_dir / "recommended_next_step.md",
        overlap_rows=overlap,
        no_override_rows=reasons,
        top_loss_count=len(top_loss),
    )
    write_summary(args.output_dir / "c3_postmortem_summary.md", decisions=decisions, seat_swap=seat_swap, output_dir=args.output_dir)

    print(
        json.dumps(
            {
                "runtime_decisions": len(decisions),
                "configs": sorted({str(row.get("config_id", "")) for row in decisions}),
                "fired_rows": len(fired_quality),
                "top_loss_rows": len(top_loss),
                "output_dir": str(args.output_dir),
                "production": "No-Go",
                "teacher_50k": "No-Go",
                "t1": "No-Go",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
