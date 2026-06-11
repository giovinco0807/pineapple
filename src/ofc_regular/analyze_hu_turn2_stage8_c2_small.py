"""Gate C2-small heldout and seat-swap validation for HU Turn2 Stage8.

C2-small is validation-only. It separates teacher-oracle filters from runtime
proxy filters because teacher EV LCB is not available at runtime.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .analyze_hu_turn2_gate_c1d_threshold_repair import threshold_passes_strategy
from .analyze_hu_turn2_gate_c1f_expanded_calibration import (
    attach_c1e_fields,
    fixed_candidate_specs,
    load_c1e_index,
    metric_summary,
    safe_float,
    safe_int,
)
from .analyze_hu_turn2_pilot_calibration import load_model, rows_for_split, write_csv
from .evaluate_hu_turn2_stage8_seat_swap import (
    DEFAULT_STAGE3_REFERENCE,
    DEFAULT_STAGE7_MODEL,
    DEFAULT_T2_STAGE8_MODEL,
    ModelParts,
    aggregate_seed_rows,
    evaluate_config_seed,
    load_parts,
    parse_seeds,
    runtime_decision_buckets,
    runtime_position_breakdown,
    summarize_values,
    write_jsonl,
)
from .ai_profiles import DEFAULT_OPENING_MODEL, DEFAULT_TURN1_MODEL, DEFAULT_TURN2_MODEL, DEFAULT_TURN3_MODEL
from .hu_turn2_stage8_runtime import HuTurn2Stage8RuntimeConfig
from .play_ai import _prediction_thread_context
from .train_hu_turn2_pilot_model import load_cache, predict_all, target_matrix
from .train_torch_action_value import select_device


DEFAULT_CACHE_DIR = Path("D:/ofc-pineapple-storage/regular-ofc-pineapple/feature_cache/hu_t2_stage8_20k_mc512")
DEFAULT_MODEL = DEFAULT_T2_STAGE8_MODEL
DEFAULT_C1E_DIR = Path("outputs/hu_turn2_stage1_pilot_training_c1e_teacher_ev_feature_cache")
DEFAULT_C1F_DIR = Path("outputs/hu_turn2_stage1_pilot_training_c1f_expanded_calibration")
DEFAULT_OUTPUT_DIR = Path("outputs/evals/hu_turn2_stage8_c2_small")
ORACLE_CANDIDATES = (
    "confidence_lcb196_m2p5_r0_g0p7",
    "confidence_lcb164_m2p5_r0_g0p7",
    "source_filtered_lcb196",
    "confidence_lcb164_m2p0_r0_g0p75",
    "best_diagnostic_m2p5_r0_g0p7",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--c1e-dir", type=Path, default=DEFAULT_C1E_DIR)
    parser.add_argument("--c1f-dir", type=Path, default=DEFAULT_C1F_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--run-seat-swap", action="store_true")
    parser.add_argument("--games-per-seed", type=int, default=300)
    parser.add_argument("--seeds", default="2026061801,2026061802,2026061803")
    parser.add_argument("--seed-stride", type=int, default=1_000_000)
    parser.add_argument("--opening-model", type=Path, default=DEFAULT_OPENING_MODEL)
    parser.add_argument("--turn1-model", type=Path, default=DEFAULT_TURN1_MODEL)
    parser.add_argument("--turn2-baseline-model", type=Path, default=DEFAULT_TURN2_MODEL)
    parser.add_argument("--turn3-model", type=Path, default=DEFAULT_TURN3_MODEL)
    parser.add_argument("--hu-turn3-stage7-model", type=Path, default=DEFAULT_STAGE7_MODEL)
    parser.add_argument("--hu-turn3-reference-model", type=Path, default=DEFAULT_STAGE3_REFERENCE)
    parser.add_argument("--hu-turn2-stage8-model", type=Path, default=DEFAULT_T2_STAGE8_MODEL)
    parser.add_argument("--prediction-threads", type=int, default=1)
    parser.add_argument("--opening-lookahead-samples", type=int, default=64)
    parser.add_argument("--progress-every", type=int, default=0)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def candidate_specs_by_id() -> dict[str, dict[str, Any]]:
    return {str(spec["candidate_id"]): spec for spec in fixed_candidate_specs()}


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def build_rows(cache_dir: Path, model_path: Path, c1e_dir: Path, *, device_name: str, batch_size: int) -> list[dict[str, Any]]:
    import torch

    from .analyze_hu_turn2_gate_c1_followup import enriched_state_rows, load_action_original_indices

    device = select_device(torch, device_name)
    cache = load_cache(cache_dir)
    net, stats, _payload = load_model(torch, model_path, device)
    predictions = predict_all(torch, net, cache, stats, device, batch_size)
    targets = target_matrix(cache)
    action_original_indices = load_action_original_indices(cache_dir, int(cache["metadata"]["action_count"]))
    rows = enriched_state_rows(cache, predictions, targets, action_original_indices)
    attach_c1e_fields(rows, load_c1e_index(c1e_dir))
    offsets = cache["offsets"]
    for row in rows:
        state_index = safe_int(row["state_index"])
        start = int(offsets[state_index])
        end = int(offsets[state_index + 1])
        pred = np.asarray(predictions[start:end], dtype=np.float64)
        candidate = safe_int(row["candidate_action_local_index"])
        delta_values = pred[:, 1]
        ev_values = pred[:, 0]
        delta_order = np.argsort(-delta_values, kind="mergesort")
        ev_order = np.argsort(-ev_values, kind="mergesort")
        delta_rank = int(np.where(delta_order == candidate)[0][0]) + 1
        ev_rank = int(np.where(ev_order == candidate)[0][0]) + 1
        delta_second = float(delta_values[delta_order[1]]) if delta_order.size > 1 else float(delta_values[candidate])
        row["model_candidate_delta_rank"] = delta_rank
        row["model_candidate_ev_rank"] = ev_rank
        row["predicted_delta_margin_top1_top2"] = float(delta_values[delta_order[0]] - delta_second)
        row["predicted_ev_margin"] = safe_float(row.get("predicted_EV_margin_top1_top2"))
        row["residual_risk_proxy"] = max(0.0, 1.0 - safe_float(row.get("gate_probability"))) + max(
            0.0, 1.0 - safe_float(row.get("predicted_ev_margin"))
        )
    return rows


def heldout_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return rows_for_split(rows, "holdout")


def group_eval_rows(rows: list[dict[str, Any]], fired: list[dict[str, Any]]) -> dict[str, Any]:
    summary = metric_summary("group", "group", rows, fired)
    return {
        "rows": len(rows),
        "fires": summary["fires"],
        "fire_rate": summary["fire_rate"],
        "avg_gain": summary["avg_gain"],
        "median_gain": summary["median_gain"],
        "false_positive_count": summary["false_positive_count"],
        "false_positive_rate": summary["false_positive_rate"],
        "avg_false_positive_cost": summary["avg_false_positive_cost"],
        "p90_loss": summary["p90_loss"],
        "p95_loss": summary["p95_loss"],
        "p99_loss": summary["p99_loss"],
        "max_loss": summary["max_loss"],
        "positive_fire_count": summary["positive_fire_count"],
        "negative_fire_count": summary["negative_fire_count"],
        "missed_positive_count": summary["missed_positive_count"],
    }


def teacher_oracle_eval(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specs = candidate_specs_by_id()
    output: list[dict[str, Any]] = []
    subsets = {
        "heldout": rows,
        "unbiased_heldout": [row for row in rows if row.get("c1e_split") == "unbiased"],
        "enriched_heldout": [row for row in rows if row.get("c1e_split") == "enriched"],
    }
    for candidate_id in ORACLE_CANDIDATES:
        spec = specs[candidate_id]
        for split_name, subset in subsets.items():
            fired = [row for row in subset if threshold_passes_strategy(row, spec)]
            output.append(
                {
                    "filter_type": "teacher_oracle",
                    "candidate_id": candidate_id,
                    "split": split_name,
                    **group_eval_rows(subset, fired),
                    "first_fires": sum(1 for row in fired if row.get("position") == "first"),
                    "second_fires": sum(1 for row in fired if row.get("position") == "second"),
                    "source_counts": json.dumps(Counter(str(row.get("run_bucket", "unknown")) for row in fired), sort_keys=True),
                }
            )
    return output


def actual_bucket_breakdown(rows: list[dict[str, Any]], specs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    groupers = {
        "c1e_split": lambda row: row.get("c1e_split", "unknown"),
        "run_bucket": lambda row: row.get("run_bucket", "unknown"),
        "actual_high_regret": lambda row: str(bool(row.get("actual_high_regret"))),
        "actual_low_margin": lambda row: str(bool(row.get("actual_low_margin"))),
        "actual_teacher_disagreement": lambda row: str(bool(row.get("actual_teacher_disagreement"))),
    }
    for candidate_id in ORACLE_CANDIDATES:
        spec = specs[candidate_id]
        fired_all = [row for row in rows if threshold_passes_strategy(row, spec)]
        for group_type, getter in groupers.items():
            for group in sorted({str(getter(row)) for row in rows}):
                subset = [row for row in rows if str(getter(row)) == group]
                fired = [row for row in fired_all if str(getter(row)) == group]
                output.append(
                    {
                        "candidate_id": candidate_id,
                        "group_type": group_type,
                        "group": group,
                        **group_eval_rows(subset, fired),
                    }
                )
    return output


def proxy_passes(row: dict[str, Any], config: dict[str, Any]) -> bool:
    if safe_int(row.get("candidate_is_baseline")):
        return False
    if safe_float(row.get("predicted_delta_vs_baseline")) < safe_float(config["min_margin"]):
        return False
    if safe_float(row.get("gate_probability")) < safe_float(config["gate_threshold"]):
        return False
    if safe_float(row.get("reference_margin_raw")) < safe_float(config.get("reference_min_margin", 0.0)):
        return False
    family = config["proxy_family"]
    if family in {"predicted_delta_gate_ev_margin", "runtime_lcb_proxy", "source_filtered_runtime_proxy"}:
        if safe_float(row.get("predicted_ev_margin")) < safe_float(config.get("predicted_ev_margin_min", 0.0)):
            return False
    if family == "predicted_delta_gate_rank_guard":
        if safe_int(row.get("model_candidate_ev_rank"), 999) > safe_int(config.get("model_candidate_rank_max"), 999):
            return False
    if family == "runtime_lcb_proxy":
        lcb = safe_float(row.get("predicted_delta_vs_baseline")) - safe_float(config.get("lambda", 1.0)) * safe_float(
            row.get("residual_risk_proxy")
        )
        if lcb < safe_float(config["min_margin"]):
            return False
    if family == "source_filtered_runtime_proxy":
        if row.get("run_bucket") == "random_off_policy" and row.get("position") == "second":
            return False
    return True


def proxy_grid() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    margins = (2.0, 2.25, 2.5, 2.75, 3.0)
    gates = (0.70, 0.75, 0.80, 0.85, 0.90)
    ev_margins = (0.25, 0.50, 0.75, 1.00)
    ranks = (1, 2, 3, 5)
    lambdas = (0.5, 1.0, 1.64, 1.96)
    for m in margins:
        for g in gates:
            configs.append({"proxy_id": f"A_m{m:g}_g{g:g}", "proxy_family": "predicted_delta_gate", "min_margin": m, "gate_threshold": g, "reference_min_margin": 0.0})
            for pm in ev_margins:
                configs.append(
                    {
                        "proxy_id": f"B_m{m:g}_g{g:g}_pm{pm:g}",
                        "proxy_family": "predicted_delta_gate_ev_margin",
                        "min_margin": m,
                        "gate_threshold": g,
                        "reference_min_margin": 0.0,
                        "predicted_ev_margin_min": pm,
                    }
                )
            for k in ranks:
                configs.append(
                    {
                        "proxy_id": f"C_m{m:g}_g{g:g}_k{k}",
                        "proxy_family": "predicted_delta_gate_rank_guard",
                        "min_margin": m,
                        "gate_threshold": g,
                        "reference_min_margin": 0.0,
                        "model_candidate_rank_max": k,
                    }
                )
            for lam in lambdas:
                configs.append(
                    {
                        "proxy_id": f"D_m{m:g}_g{g:g}_l{lam:g}",
                        "proxy_family": "runtime_lcb_proxy",
                        "min_margin": m,
                        "gate_threshold": g,
                        "reference_min_margin": 0.0,
                        "predicted_ev_margin_min": 0.25,
                        "lambda": lam,
                    }
                )
            configs.append(
                {
                    "proxy_id": f"E_m{m:g}_g{g:g}",
                    "proxy_family": "source_filtered_runtime_proxy",
                    "min_margin": m,
                    "gate_threshold": g,
                    "reference_min_margin": 0.0,
                    "predicted_ev_margin_min": 0.25,
                }
            )
    return configs


def proxy_eval(rows: list[dict[str, Any]], configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for config in configs:
        fired = [row for row in rows if proxy_passes(row, config)]
        summary = group_eval_rows(rows, fired)
        output.append(
            {
                **config,
                **summary,
                "unbiased_fires": sum(1 for row in fired if row.get("c1e_split") == "unbiased"),
                "enriched_fires": sum(1 for row in fired if row.get("c1e_split") == "enriched"),
                "first_fires": sum(1 for row in fired if row.get("position") == "first"),
                "second_fires": sum(1 for row in fired if row.get("position") == "second"),
                "runtime_implemented": int(config["proxy_family"] == "predicted_delta_gate"),
            }
        )
    return output


def overlap_rows(rows: list[dict[str, Any]], oracle_specs: dict[str, dict[str, Any]], proxy_configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    state_ids = {safe_int(row["state_index"]) for row in rows}
    for oracle_id in ORACLE_CANDIDATES:
        oracle = {safe_int(row["state_index"]) for row in rows if threshold_passes_strategy(row, oracle_specs[oracle_id])}
        for config in proxy_configs:
            proxy = {safe_int(row["state_index"]) for row in rows if proxy_passes(row, config)}
            inter = oracle & proxy
            union = oracle | proxy
            output.append(
                {
                    "oracle_candidate_id": oracle_id,
                    "proxy_id": config["proxy_id"],
                    "proxy_family": config["proxy_family"],
                    "oracle_fires": len(oracle),
                    "proxy_fires": len(proxy),
                    "intersection": len(inter),
                    "precision_vs_oracle": len(inter) / max(len(proxy), 1),
                    "recall_vs_oracle": len(inter) / max(len(oracle), 1),
                    "jaccard_overlap": len(inter) / max(len(union), 1),
                    "rows": len(state_ids),
                }
            )
    return output


def choose_proxy_candidates(proxy_rows: list[dict[str, Any]], overlap: list[dict[str, Any]], limit: int = 4) -> list[dict[str, Any]]:
    best_overlap: dict[str, dict[str, Any]] = {}
    for row in overlap:
        if row["oracle_candidate_id"] not in {"confidence_lcb196_m2p5_r0_g0p7", "confidence_lcb164_m2p5_r0_g0p7", "source_filtered_lcb196"}:
            continue
        current = best_overlap.get(str(row["proxy_id"]))
        if current is None or safe_float(row["jaccard_overlap"]) > safe_float(current["jaccard_overlap"]):
            best_overlap[str(row["proxy_id"])] = row
    candidates = []
    for row in proxy_rows:
        if not safe_int(row.get("runtime_implemented")):
            continue
        if safe_int(row.get("fires")) < 15:
            continue
        if safe_int(row.get("unbiased_fires")) == 0 or safe_int(row.get("first_fires")) == 0 or safe_int(row.get("second_fires")) == 0:
            continue
        if safe_float(row.get("false_positive_rate")) > 0.05:
            continue
        if safe_float(row.get("avg_gain")) <= 0.0:
            continue
        item = dict(row)
        item.update(
            {
                "best_oracle_candidate_id": best_overlap.get(str(row["proxy_id"]), {}).get("oracle_candidate_id", ""),
                "best_jaccard_overlap": best_overlap.get(str(row["proxy_id"]), {}).get("jaccard_overlap", 0.0),
                "best_precision_vs_oracle": best_overlap.get(str(row["proxy_id"]), {}).get("precision_vs_oracle", 0.0),
                "best_recall_vs_oracle": best_overlap.get(str(row["proxy_id"]), {}).get("recall_vs_oracle", 0.0),
            }
        )
        candidates.append(item)
    candidates.sort(
        key=lambda row: (
            -safe_float(row.get("best_jaccard_overlap")),
            safe_float(row.get("false_positive_rate")),
            -safe_float(row.get("avg_gain")),
            -safe_int(row.get("fires")),
        )
    )
    selected: list[dict[str, Any]] = []
    seen: set[tuple[float, float, float]] = set()
    for row in candidates:
        triple = (safe_float(row["min_margin"]), safe_float(row["reference_min_margin"]), safe_float(row["gate_threshold"]))
        if triple in seen:
            continue
        seen.add(triple)
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def seat_swap_config_id(row: dict[str, Any]) -> str:
    return f"{safe_float(row['min_margin']):g}/{safe_float(row.get('reference_min_margin', 0.0)):g}/{safe_float(row['gate_threshold']):g}"


def run_seat_swap(
    *,
    args: argparse.Namespace,
    parts: ModelParts,
    selected: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    configs = [
        HuTurn2Stage8RuntimeConfig(
            min_margin=safe_float(row["min_margin"]),
            reference_min_margin=safe_float(row.get("reference_min_margin", 0.0)),
            gate_threshold=safe_float(row["gate_threshold"]),
        )
        for row in selected
    ]
    seeds = parse_seeds(args.seeds)
    seed_rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    position_scores: list[dict[str, Any]] = []
    with _prediction_thread_context(args.prediction_threads):
        for config in configs:
            for seed in seeds:
                summary, decision_rows, scores = evaluate_config_seed(
                    config=config,
                    seed=seed,
                    seed_stride=args.seed_stride,
                    games=args.games_per_seed,
                    parts=parts,
                    opening_lookahead_samples=args.opening_lookahead_samples,
                    progress_every=args.progress_every,
                )
                seed_rows.append(summary)
                decisions.extend(decision_rows)
                position_scores.extend(scores)
    aggregate = aggregate_seed_rows(seed_rows)
    return aggregate, seed_rows, position_scores, decisions


def seat_swap_failure_rows(decisions: list[dict[str, Any]], limit: int = 30) -> list[dict[str, Any]]:
    losses = [row for row in decisions if row.get("override_fired") and safe_float(row.get("candidate_seat_score")) < 0.0]
    losses.sort(key=lambda row: safe_float(row.get("candidate_seat_score")))
    return losses[:limit]


def write_summary(
    output_dir: Path,
    *,
    oracle_rows: list[dict[str, Any]],
    proxy_rows: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    seat_swap_rows: list[dict[str, Any]],
    c2_status: str,
    blockers: list[str],
    elapsed: float,
    run_seat_swap_enabled: bool,
) -> None:
    lines = [
        "# HU Turn2 Stage8 C2-Small Validation",
        "",
        "C2-small is validation-only. It does not authorize 50k teacher, T1, production training, or production runtime changes.",
        "",
        f"- C2-small: `{c2_status}`",
        f"- blockers: `{';'.join(blockers) if blockers else 'none'}`",
        f"- seat_swap_run: `{run_seat_swap_enabled}`",
        f"- elapsed seconds: `{elapsed:.2f}`",
        "",
        "## Teacher Oracle Heldout",
        "",
        "| candidate | fires | FP rate | avg gain | unbiased fires | enriched fires | first | second |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in [row for row in oracle_rows if row["split"] == "heldout"]:
        lines.append(
            "| {candidate_id} | {fires} | {false_positive_rate:.4f} | {avg_gain:.4f} | {unbiased} | {enriched} | {first} | {second} |".format(
                candidate_id=row["candidate_id"],
                fires=safe_int(row["fires"]),
                false_positive_rate=safe_float(row["false_positive_rate"]),
                avg_gain=safe_float(row["avg_gain"]),
                unbiased=sum(
                    safe_int(item["fires"])
                    for item in oracle_rows
                    if item["candidate_id"] == row["candidate_id"] and item["split"] == "unbiased_heldout"
                ),
                enriched=sum(
                    safe_int(item["fires"])
                    for item in oracle_rows
                    if item["candidate_id"] == row["candidate_id"] and item["split"] == "enriched_heldout"
                ),
                first="see position csv",
                second="see position csv",
            )
        )
    lines.extend(
        [
            "",
            "## Runtime Proxy Candidates",
            "",
            "| proxy | fires | FP rate | avg gain | unbiased | first | second | overlap |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in selected:
        lines.append(
            "| {proxy_id} | {fires} | {false_positive_rate:.4f} | {avg_gain:.4f} | {unbiased_fires} | {first_fires} | {second_fires} | {best_jaccard_overlap:.4f} |".format(
                **row
            )
        )
    if seat_swap_rows:
        lines.extend(
            [
                "",
                "## Seat-Swap Small",
                "",
                "| config | EV/hand | CI low | CI high | override rate |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in sorted(seat_swap_rows, key=lambda item: safe_float(item.get("aggregate_ev_per_hand")), reverse=True):
            lines.append(
                "| {config_id} | {aggregate_ev_per_hand:.4f} | {ci95_low_seed_means:.4f} | {ci95_high_seed_means:.4f} | {runtime_override_rate:.4f} |".format(
                    **row
                )
            )
    lines.extend(
        [
            "",
            "## Decisions",
            "",
            f"- C3 / larger seat-swap: `{c2_status}`",
            "- 50k teacher: `No-Go`",
            "- T1 training: `No-Go`",
            "- production training: `No-Go`",
        ]
    )
    (output_dir / "c2_small_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "c2_small_recommended_next_step.md").write_text(
        "\n".join(
            [
                "# C2-Small Recommended Next Step",
                "",
                f"- C3 / larger seat-swap: `{c2_status}`",
                f"- blockers: `{';'.join(blockers) if blockers else 'none'}`",
                "- 50k teacher: `No-Go`",
                "- selected MC4096/8192 refinement: `consider for proxy-fired and near-threshold rows before production`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (output_dir / "go_nogo_for_c3.md").write_text(
        "\n".join(
            [
                "# Go / No-Go For C3",
                "",
                f"- C3: `{c2_status}`",
                f"- blockers: `{';'.join(blockers) if blockers else 'none'}`",
                "- 50k teacher: `No-Go`",
                "- T1 training: `No-Go`",
                "- production training: `No-Go`",
                "",
            ]
        ),
        encoding="utf-8",
    )


def c2_status(
    *,
    selected: list[dict[str, Any]],
    seat_swap_rows: list[dict[str, Any]],
    run_seat_swap_enabled: bool,
) -> tuple[str, list[str]]:
    blockers: list[str] = []
    if not selected:
        blockers.append("no_runtime_proxy_candidate")
    if selected and all(safe_float(row.get("false_positive_rate")) > 0.05 for row in selected):
        blockers.append("runtime_proxy_fp_too_high")
    if selected and all(safe_int(row.get("unbiased_fires")) == 0 for row in selected):
        blockers.append("runtime_proxy_unbiased_zero")
    if run_seat_swap_enabled:
        if not seat_swap_rows:
            blockers.append("seat_swap_not_completed")
        elif all(safe_float(row.get("aggregate_ev_per_hand")) < 0.0 for row in seat_swap_rows):
            blockers.append("seat_swap_all_negative")
        elif all(safe_float(row.get("runtime_override_rate")) <= 0.0 for row in seat_swap_rows):
            blockers.append("seat_swap_override_zero")
    return ("Go" if not blockers else "No-Go", blockers)


def main() -> int:
    args = parse_args()
    started_at = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.prediction_threads > 0:
        os.environ.setdefault("OMP_NUM_THREADS", str(args.prediction_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(args.prediction_threads))
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(args.prediction_threads))

    rows = build_rows(args.cache_dir, args.model, args.c1e_dir, device_name=args.device, batch_size=args.batch_size)
    heldout = heldout_rows(rows)
    specs = candidate_specs_by_id()
    oracle_rows = teacher_oracle_eval(heldout)
    source_rows = actual_bucket_breakdown(heldout, specs)
    proxy_configs = proxy_grid()
    proxy_rows = proxy_eval(heldout, proxy_configs)
    overlap = overlap_rows(heldout, specs, proxy_configs)
    selected = choose_proxy_candidates(proxy_rows, overlap, limit=4)

    seat_swap_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    position_scores: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    if args.run_seat_swap and selected:
        parts = load_parts(args)
        seat_swap_rows, seed_rows, position_scores, decisions = run_seat_swap(args=args, parts=parts, selected=selected)

    status, blockers = c2_status(selected=selected, seat_swap_rows=seat_swap_rows, run_seat_swap_enabled=args.run_seat_swap)

    write_csv(args.output_dir / "teacher_oracle_filter_eval.csv", oracle_rows)
    write_csv(args.output_dir / "runtime_proxy_filter_eval.csv", proxy_rows)
    write_csv(args.output_dir / "proxy_vs_oracle_overlap.csv", overlap)
    write_csv(args.output_dir / "c2_small_source_breakdown.csv", source_rows)
    write_csv(args.output_dir / "c2_small_seat_swap_results.csv", seat_swap_rows)
    write_csv(args.output_dir / "c2_small_seed_breakdown.csv", seed_rows)
    write_csv(args.output_dir / "c2_small_position_breakdown.csv", runtime_position_breakdown(position_scores, decisions) if decisions else [])
    write_jsonl(args.output_dir / "c2_small_failure_top30.jsonl", seat_swap_failure_rows(decisions))
    if decisions:
        write_csv(args.output_dir / "c2_small_runtime_decisions_sample.csv", decisions[:2000])
        write_csv(args.output_dir / "c2_small_runtime_no_override_reasons.csv", runtime_decision_buckets(decisions))
    write_summary(
        args.output_dir,
        oracle_rows=oracle_rows,
        proxy_rows=proxy_rows,
        selected=selected,
        seat_swap_rows=seat_swap_rows,
        c2_status=status,
        blockers=blockers,
        elapsed=time.perf_counter() - started_at,
        run_seat_swap_enabled=args.run_seat_swap,
    )
    print(
        json.dumps(
            {
                "gate": "C2-small",
                "heldout_rows": len(heldout),
                "runtime_proxy_candidates": [row["proxy_id"] for row in selected],
                "seat_swap_run": args.run_seat_swap,
                "c2_status": status,
                "blockers": blockers,
                "output_dir": str(args.output_dir),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
