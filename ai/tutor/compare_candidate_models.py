"""Compare action-value candidate models across fixed guardrail datasets.

This script wraps the existing action-value evaluators and writes one compact
comparison report.  It is intended to stop us from accepting a model that fixes
one mined weak spot while quietly regressing broad T1/T2/T3 behavior.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ai.tutor.evaluate_action_value_dataset import evaluate as evaluate_dataset
from ai.tutor.evaluate_action_value_suit_ensemble import evaluate as evaluate_suit_ensemble
from ai.tutor.evaluate_teacher_model import resolve_model_path


DEFAULT_MODEL = (
    "current",
    "ai/models/candidate_runs/tutor-route10-top20-prune-active8x-ft-20260523/model/action_value_best.pt",
)
DEFAULT_TOP_NS = "1,3,5,10,20,24"


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    name: str
    path: str
    mode: str = "standard"
    guardrail: str = "broad"
    block_size: int = 24
    ensemble_size: int = 0


DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        key="t1_holdout",
        name="T1 holdout MC300",
        path="ai/data/tutor_eval_holdout_20260523/reranker_t1_mc300_500",
    ),
    DatasetSpec(
        key="t2_holdout",
        name="T2 holdout MC300",
        path="ai/data/tutor_eval_holdout_20260523/reranker_t2_mc300_500",
    ),
    DatasetSpec(
        key="t3_exact",
        name="T3 exact holdout",
        path="ai/data/tutor_eval_holdout_20260523/reranker_t3_exact_500",
    ),
    DatasetSpec(
        key="t2_g300_eval",
        name="T2 G300 eval MC300",
        path="ai/data/tutor_t2_continued_mining_20260524/reranker_eval_g300_t2_500_mc300",
    ),
    DatasetSpec(
        key="t2_g1000_eval",
        name="T2 G1000 eval MC300",
        path="ai/data/tutor_t2_continued_mining_20260524/reranker_eval_g1000_t2_500_mc300",
    ),
    DatasetSpec(
        key="t2_joint200",
        name="T2 joint unseen MC300",
        path="ai/data/tutor_t2_continued_mining_20260524/reranker_joint_unseen_t2_200_mc300",
    ),
    DatasetSpec(
        key="t2_top24_stress_suit24",
        name="T2 top24 stress suit24 MC1000",
        path="ai/data/tutor_t2_continued_mining_20260524/reranker_eval_g300_t2_top24_miss_suit24_mc1000",
        guardrail="hard",
    ),
    DatasetSpec(
        key="t2_top20_probe_suit24",
        name="T2 top20 probe suit24 MC1000",
        path="ai/data/tutor_t2_continued_mining_20260524/reranker_eval_g1000_t2_top20_miss_probe_mc1000_suit24",
        guardrail="hard",
    ),
    DatasetSpec(
        key="t2_p99_top20_suit24",
        name="T2 p99 top20 miss suit24 MC1000",
        path="ai/data/tutor_t2_broader_mining_20260524/reranker_p99_t2_top20_miss4_suit24_mc1000",
        guardrail="hard",
    ),
    DatasetSpec(
        key="t2_top20_probe_suit24_ens8",
        name="T2 top20 probe suit24 ensemble8",
        path="ai/data/tutor_t2_continued_mining_20260524/reranker_eval_g1000_t2_top20_miss_probe_mc1000_suit24",
        mode="suit_ensemble",
        guardrail="ensemble",
        ensemble_size=8,
    ),
    DatasetSpec(
        key="t2_p99_top20_suit24_ens8",
        name="T2 p99 top20 miss suit24 ensemble8",
        path="ai/data/tutor_t2_broader_mining_20260524/reranker_p99_t2_top20_miss4_suit24_mc1000",
        mode="suit_ensemble",
        guardrail="ensemble",
        ensemble_size=8,
    ),
)


def slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return cleaned or "item"


def parse_model(value: str) -> tuple[str, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
        label = label.strip()
    else:
        path = value
        label = Path(path).parent.parent.name if Path(path).name == "action_value_best.pt" else Path(path).stem
    if not label:
        raise ValueError(f"Model label is empty in {value!r}")
    model_path = Path(resolve_model_path(path.strip()))
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    return slug(label), model_path


def parse_models(values: list[str]) -> list[tuple[str, Path]]:
    if not values:
        values = [f"{DEFAULT_MODEL[0]}={DEFAULT_MODEL[1]}"]
    models = [parse_model(value) for value in values]
    labels = [label for label, _path in models]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ValueError(f"Duplicate model labels after slugging: {', '.join(duplicates)}")
    return models


def select_datasets(value: str | None) -> list[DatasetSpec]:
    wanted = [part.strip() for part in value.split(",") if part.strip()] if value else []
    specs_by_key = {spec.key: spec for spec in DATASETS}
    if wanted:
        unknown = [key for key in wanted if key not in specs_by_key]
        if unknown:
            raise ValueError(
                "Unknown dataset key(s): "
                + ", ".join(unknown)
                + "\nAvailable: "
                + ", ".join(specs_by_key)
            )
        specs = [specs_by_key[key] for key in wanted]
    else:
        specs = list(DATASETS)

    existing: list[DatasetSpec] = []
    missing: list[str] = []
    for spec in specs:
        if Path(spec.path).exists():
            existing.append(spec)
        else:
            missing.append(f"{spec.key} ({spec.path})")
    if missing:
        print("Skipping missing datasets: " + "; ".join(missing), file=sys.stderr)
    if not existing:
        raise FileNotFoundError("No selected datasets exist")
    return existing


def run_one(
    output_dir: Path,
    spec: DatasetSpec,
    model_label: str,
    model_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    out_dir = output_dir / "evals" / spec.key / model_label
    existing = out_dir / "summary.json"
    if existing.exists() and not args.force:
        return json.loads(existing.read_text(encoding="utf-8"))

    eval_args = argparse.Namespace(
        data=spec.path,
        model=str(model_path),
        output=str(out_dir),
        device=args.device,
        batch_size=args.batch_size,
        limit_samples=args.limit_samples,
        weak_groups=args.weak_groups,
        top_ns=args.top_ns,
        block_size=spec.block_size,
        ensemble_size=spec.ensemble_size or spec.block_size,
    )
    if spec.mode == "suit_ensemble":
        return evaluate_suit_ensemble(eval_args)
    return evaluate_dataset(eval_args)


def metric(report: dict[str, Any], group: str, top_n: int) -> float:
    stats = report.get("overall", {})
    section = stats.get(group, {})
    return float(section.get(f"top_{top_n}", 0.0))


def top_recall(report: dict[str, Any], top_n: int) -> float:
    return metric(report, "top_recall", top_n)


def top_regret(report: dict[str, Any], top_n: int) -> float:
    return metric(report, "topk_exact_rerank_avg_regret", top_n)


def decide(
    spec: DatasetSpec,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    is_baseline: bool,
) -> tuple[str, list[str]]:
    if is_baseline:
        return "BASELINE", []

    b_top20 = top_recall(baseline, 20)
    b_top24 = top_recall(baseline, 24)
    c_top20 = top_recall(candidate, 20)
    c_top24 = top_recall(candidate, 24)
    b_t20reg = top_regret(baseline, 20)
    b_t24reg = top_regret(baseline, 24)
    c_t20reg = top_regret(candidate, 20)
    c_t24reg = top_regret(candidate, 24)

    reasons: list[str] = []
    if spec.guardrail == "broad":
        if c_top24 - b_top24 < -0.005:
            reasons.append(f"top24 {100.0 * (c_top24 - b_top24):+.1f}pp")
        if c_t24reg - b_t24reg > 0.010:
            reasons.append(f"t24reg {c_t24reg - b_t24reg:+.3f}")
        if c_top20 - b_top20 < -0.010:
            reasons.append(f"top20 {100.0 * (c_top20 - b_top20):+.1f}pp")
        if c_t20reg - b_t20reg > 0.050:
            reasons.append(f"t20reg {c_t20reg - b_t20reg:+.3f}")
    elif spec.guardrail in {"hard", "ensemble"}:
        if c_top24 - b_top24 < -0.010:
            reasons.append(f"top24 {100.0 * (c_top24 - b_top24):+.1f}pp")
        if c_t24reg - b_t24reg > 0.050:
            reasons.append(f"t24reg {c_t24reg - b_t24reg:+.3f}")
    else:
        raise ValueError(f"Unknown guardrail: {spec.guardrail}")

    return ("REJECT" if reasons else "PASS"), reasons


def pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def signed_pct(value: float) -> str:
    return f"{100.0 * value:+.1f}pp"


def signed_float(value: float) -> str:
    return f"{value:+.3f}"


def build_rows(
    models: list[tuple[str, Path]],
    specs: list[DatasetSpec],
    reports: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    baseline_label = models[0][0]
    rows: list[dict[str, Any]] = []
    for spec in specs:
        baseline = reports[spec.key][baseline_label]
        for label, model_path in models:
            report = reports[spec.key][label]
            verdict, reasons = decide(spec, baseline, report, is_baseline=(label == baseline_label))
            row = {
                "dataset_key": spec.key,
                "dataset": spec.name,
                "mode": spec.mode,
                "guardrail": spec.guardrail,
                "model": label,
                "model_path": str(model_path),
                "decisions": int(report.get("overall", {}).get("decisions", report.get("blocks", 0))),
                "candidates": int(report.get("overall", {}).get("candidates", 0)),
                "top1": top_recall(report, 1),
                "top3": top_recall(report, 3),
                "top5": top_recall(report, 5),
                "top10": top_recall(report, 10),
                "top20": top_recall(report, 20),
                "top24": top_recall(report, 24),
                "t10reg": top_regret(report, 10),
                "t20reg": top_regret(report, 20),
                "t24reg": top_regret(report, 24),
                "verdict": verdict,
                "reasons": reasons,
            }
            if label == baseline_label:
                row.update(
                    {
                        "d_top1": 0.0,
                        "d_top3": 0.0,
                        "d_top5": 0.0,
                        "d_top10": 0.0,
                        "d_top20": 0.0,
                        "d_top24": 0.0,
                        "d_t10reg": 0.0,
                        "d_t20reg": 0.0,
                        "d_t24reg": 0.0,
                    }
                )
            else:
                row.update(
                    {
                        "d_top1": row["top1"] - top_recall(baseline, 1),
                        "d_top3": row["top3"] - top_recall(baseline, 3),
                        "d_top5": row["top5"] - top_recall(baseline, 5),
                        "d_top10": row["top10"] - top_recall(baseline, 10),
                        "d_top20": row["top20"] - top_recall(baseline, 20),
                        "d_top24": row["top24"] - top_recall(baseline, 24),
                        "d_t10reg": row["t10reg"] - top_regret(baseline, 10),
                        "d_t20reg": row["t20reg"] - top_regret(baseline, 20),
                        "d_t24reg": row["t24reg"] - top_regret(baseline, 24),
                    }
                )
            rows.append(row)
    return rows


def write_comparison(output_dir: Path, payload: dict[str, Any]) -> None:
    (output_dir / "comparison.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        "# Candidate Model Comparison",
        "",
        f"- created: {payload['created_at']}",
        f"- baseline: `{payload['baseline_label']}`",
        "",
        "| dataset | mode | guardrail | model | decisions | top10 | d_top10 | top20 | d_top20 | top24 | d_top24 | r10 | d_r10 | r20 | d_r20 | r24 | d_r24 | verdict |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["rows"]:
        reason = ""
        if row["reasons"]:
            reason = " (" + "; ".join(row["reasons"]) + ")"
        lines.append(
            "| "
            + " | ".join(
                [
                    row["dataset"],
                    row["mode"],
                    row["guardrail"],
                    row["model"],
                    f"{row['decisions']:,}",
                    pct(row["top10"]),
                    signed_pct(row["d_top10"]),
                    pct(row["top20"]),
                    signed_pct(row["d_top20"]),
                    pct(row["top24"]),
                    signed_pct(row["d_top24"]),
                    f"{row['t10reg']:.3f}",
                    signed_float(row["d_t10reg"]),
                    f"{row['t20reg']:.3f}",
                    signed_float(row["d_t20reg"]),
                    f"{row['t24reg']:.3f}",
                    signed_float(row["d_t24reg"]),
                    row["verdict"] + reason,
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Top-N Detail",
            "",
            "| dataset | model | top1 | d_top1 | top3 | d_top3 | top5 | d_top5 | top10 | d_top10 | top20 | d_top20 | top24 | d_top24 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in payload["rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["dataset"],
                    row["model"],
                    pct(row["top1"]),
                    signed_pct(row["d_top1"]),
                    pct(row["top3"]),
                    signed_pct(row["d_top3"]),
                    pct(row["top5"]),
                    signed_pct(row["d_top5"]),
                    pct(row["top10"]),
                    signed_pct(row["d_top10"]),
                    pct(row["top20"]),
                    signed_pct(row["d_top20"]),
                    pct(row["top24"]),
                    signed_pct(row["d_top24"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Verdict By Model",
            "",
            "| model | verdict | rejected datasets |",
            "|---|---|---|",
        ]
    )
    for item in payload["model_verdicts"]:
        rejected = ", ".join(item["rejected_datasets"]) if item["rejected_datasets"] else "-"
        lines.append(f"| {item['model']} | {item['verdict']} | {rejected} |")

    (output_dir / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def model_verdicts(models: list[tuple[str, Path]], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, (label, model_path) in enumerate(models):
        if index == 0:
            result.append(
                {
                    "model": label,
                    "model_path": str(model_path),
                    "verdict": "BASELINE",
                    "rejected_datasets": [],
                }
            )
            continue
        rejected = [row["dataset_key"] for row in rows if row["model"] == label and row["verdict"] == "REJECT"]
        result.append(
            {
                "model": label,
                "model_path": str(model_path),
                "verdict": "REJECT" if rejected else "PASS",
                "rejected_datasets": rejected,
            }
        )
    return result


def default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("ai/data") / f"tutor_model_compare_{stamp}"


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare candidate action-value reranker checkpoints")
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model to compare. Use label=path. The first model is the baseline.",
    )
    parser.add_argument("--datasets", default=None, help="Comma-separated dataset keys. Defaults to all existing.")
    parser.add_argument("--output", default=None, help="Output directory for per-model evals and comparison report.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--weak-groups", type=int, default=100)
    parser.add_argument("--top-ns", default=DEFAULT_TOP_NS)
    parser.add_argument("--force", action="store_true", help="Re-run evals even when summary.json already exists.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any non-baseline model is rejected.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    output_dir = Path(args.output) if args.output else default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    models = parse_models(args.model)
    specs = select_datasets(args.datasets)

    reports: dict[str, dict[str, dict[str, Any]]] = {}
    for spec in specs:
        reports[spec.key] = {}
        for label, model_path in models:
            print(f"Evaluating {label} on {spec.key} ({spec.mode})", file=sys.stderr)
            reports[spec.key][label] = run_one(output_dir, spec, label, model_path, args)

    rows = build_rows(models, specs, reports)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "baseline_label": models[0][0],
        "models": [{"label": label, "path": str(path)} for label, path in models],
        "datasets": [spec.__dict__ for spec in specs],
        "rows": rows,
        "model_verdicts": model_verdicts(models, rows),
    }
    write_comparison(output_dir, payload)
    print(json.dumps(payload["model_verdicts"], indent=2, ensure_ascii=False))
    print(f"Wrote {output_dir / 'comparison.md'}")

    if args.strict and any(item["verdict"] == "REJECT" for item in payload["model_verdicts"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
