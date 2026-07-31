"""Prepare a T3 data-vs-capacity experiment.

The current late-turn exact path can create many T3 labels, but T3 model
accuracy has not moved enough to assume data volume is the only issue.  This
runner writes a reproducible command plan for comparing:

* more/diverse exact T3 data with the current reranker architecture, and
* a larger unfrozen model trained on the same T3 action-value data.

By default this script only writes a manifest and PowerShell command file.  Use
``--execute`` only when you intentionally want to start the selected stages.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_TARGET_INPUT = (
    "ai/data/t0_proxy_holdout_20260529/"
    "turn_routed_t0top128_mc30_10000_targets_t3t4_20260529.jsonl"
)
DEFAULT_RERANKER_DATA = (
    "ai/data/t0_proxy_holdout_20260529/"
    "exact_late_reranker_t3t4_10000_20260529"
)
DEFAULT_BASE_CHECKPOINT = (
    "ai/models/candidate_runs/"
    "tutor-route10-t3-exact10000-adapter64-20260529/model/action_value_best.pt"
)
DEFAULT_HOLDOUT = (
    "ai/data/tutor_eval_holdout_20260523/teacher_labels_t3_exact_500.jsonl"
)
DEFAULT_T2_INPUT = (
    "ai/data/hybrid_t1t2_active_20260531/"
    "oracle_first_next2_rows40_49_cap100_20260603/inputs/t2_input.jsonl"
)
DEFAULT_T2_EXACT = (
    "ai/data/hybrid_t1t2_active_20260531/"
    "oracle_first_next2_rows40_49_cap100_20260603/"
    "t2_caps/t2_oracle_cap100_limit10.jsonl"
)


@dataclass(frozen=True)
class Stage:
    name: str
    args: list[str]

    def powershell(self) -> str:
        return " ".join(ps_quote(arg) for arg in self.args)

    def bash(self) -> str:
        return " ".join(shlex.quote(arg) for arg in self.args)


def ps_quote(value: str) -> str:
    if value == "":
        return "''"
    if all(ch.isalnum() or ch in ".:_/-\\" for ch in value):
        return value
    return "'" + value.replace("'", "''") + "'"


def path_text(path: str | Path) -> str:
    return str(path).replace("\\", "/")


def build_stages(args: argparse.Namespace) -> tuple[list[Stage], dict[str, str]]:
    out_dir = Path(args.out_dir)
    input_path = out_dir / "inputs" / "t3_input.jsonl"
    teacher_path = out_dir / "exact" / "t3_exact_teacher.jsonl"
    generated_data_dir = out_dir / "reranker_t3_exact"
    reranker_data = generated_data_dir if args.generate_labels else Path(args.reranker_data)
    baseline_model_dir = out_dir / "models" / "current_arch_unfrozen"
    capacity_model_dir = out_dir / "models" / "large_capacity"
    eval_dir = out_dir / "eval"
    bench_dir = out_dir / "t2_bench"

    py = args.python
    stages: list[Stage] = []
    if args.generate_labels:
        stages.extend(
            [
                Stage(
                    "select_t3_inputs",
                    [
                        py,
                        "-m",
                        "ai.tutor.select_diverse_oracle_inputs",
                        "--input",
                        args.target_input,
                        "--output",
                        path_text(input_path),
                        "--turns",
                        "3",
                        "--limit",
                        str(args.exact_records),
                        "--max-per-source",
                        str(args.max_per_source),
                        "--max-per-rank-group",
                        str(args.max_per_rank_group),
                    ],
                ),
                Stage(
                    "generate_exact_t3_labels",
                    [
                        py,
                        "-m",
                        "ai.tutor.generate_exact_late_teacher",
                        path_text(input_path),
                        "--output",
                        path_text(teacher_path),
                        "--turns",
                        "3",
                        "--workers",
                        str(args.exact_workers),
                        "--progress-every",
                        str(args.progress_every),
                    ],
                ),
                Stage(
                    "convert_exact_t3_to_reranker",
                    [
                        py,
                        "-m",
                        "ai.training.convert_action_value_teacher",
                        path_text(teacher_path),
                        "--output",
                        path_text(generated_data_dir),
                        "--turns",
                        "3",
                        "--regular-max-candidates",
                        "27",
                        "--state-dim",
                        "520",
                        "--teacher-best-weight",
                        "0.5",
                        "--gap-weight",
                        "0.5",
                        "--fl-positive-weight",
                        "2.0",
                        "--fl-shape-but-foul-weight",
                        "1.5",
                    ],
                ),
            ]
        )

    common_train = [
        "--data",
        path_text(reranker_data),
        "--train-turns",
        "3",
        "--epochs",
        str(args.epochs),
        "--max-seconds",
        str(args.max_seconds),
        "--batch-size",
        str(args.batch_size),
        "--weight-decay",
        "0.0001",
        "--score-weight",
        "1.0",
        "--bust-weight",
        "0.05",
        "--fl-weight",
        "0.55",
        "--fl-type-weight",
        "0.75",
        "--ranking-weight",
        "0.25",
        "--ranking-temperature",
        "2.5",
        "--ranking-batches-per-epoch",
        str(args.ranking_batches_per_epoch),
        "--group-batch-size",
        "32",
        "--topk-rank-weight",
        "0.8",
        "--target-topk",
        "10",
        "--topk-rank-temperature",
        "0.25",
        "--selection-metric",
        "topk",
        "--val-frac",
        "0.1",
        "--eval-max-samples",
        str(args.eval_max_samples),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
    ]
    stages.append(
        Stage(
            "train_current_arch_unfrozen",
            [
                py,
                "-m",
                "ai.training.train_action_value_reranker",
                *common_train,
                "--lr",
                str(args.lr),
                "--save-dir",
                path_text(baseline_model_dir),
                "--init-checkpoint",
                args.base_checkpoint,
                "--normalization-source",
                "checkpoint",
            ],
        )
    )
    stages.append(
        Stage(
            "train_large_capacity",
            [
                py,
                "-m",
                "ai.training.train_action_value_reranker",
                *common_train,
                "--lr",
                str(args.capacity_lr),
                "--save-dir",
                path_text(capacity_model_dir),
                "--hidden",
                str(args.capacity_hidden),
                "--n-blocks",
                str(args.capacity_blocks),
                "--dropout",
                str(args.capacity_dropout),
                "--turn-specific-heads",
                "--turn-specific-adapters",
                "--adapter-dim",
                str(args.capacity_adapter_dim),
                "--normalization-source",
                "data",
            ],
        )
    )

    model_outputs = {
        "current_arch": baseline_model_dir / "action_value_best.pt",
        "large_capacity": capacity_model_dir / "action_value_best.pt",
    }
    for label, model_path in model_outputs.items():
        stages.append(
            Stage(
                f"eval_{label}_t3_holdout",
                [
                    py,
                    "-m",
                    "ai.tutor.evaluate_teacher_model",
                    args.holdout,
                    "--model",
                    path_text(model_path),
                    "--output",
                    path_text(eval_dir / label),
                    "--turns",
                    "3",
                    "--device",
                    "cpu",
                ],
            )
        )
        stages.append(
            Stage(
                f"bench_{label}_t2_via_t3",
                [
                    py,
                    "-m",
                    "ai.tutor.benchmark_t2_t3_value_model",
                    "--input",
                    args.t2_input,
                    "--model",
                    path_text(model_path),
                    "--exact",
                    args.t2_exact,
                    "--output-dir",
                    path_text(bench_dir / label),
                    "--limit",
                    str(args.t2_limit),
                    "--max-candidates",
                    str(args.t2_max_candidates),
                    "--draw-limit",
                    str(args.t2_draw_limit),
                    "--batch-size",
                    str(args.t2_batch_size),
                    "--device",
                    "cpu",
                ],
            )
        )

    paths = {
        "out_dir": path_text(out_dir),
        "reranker_data": path_text(reranker_data),
        "current_arch_model": path_text(model_outputs["current_arch"]),
        "large_capacity_model": path_text(model_outputs["large_capacity"]),
        "t3_holdout_eval_dir": path_text(eval_dir),
        "t2_bench_dir": path_text(bench_dir),
        "commands": path_text(out_dir / "commands.ps1"),
        "commands_sh": path_text(out_dir / "commands.sh"),
        "manifest": path_text(out_dir / "manifest.json"),
    }
    return stages, paths


def write_outputs(out_dir: Path, stages: list[Stage], paths: dict[str, str], args: argparse.Namespace) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    commands_path = out_dir / "commands.ps1"
    bash_path = out_dir / "commands.sh"
    manifest_path = out_dir / "manifest.json"
    lines = [
        "$ErrorActionPreference = 'Stop'",
        "",
    ]
    for stage in stages:
        lines.append(f"Write-Host '== {stage.name} =='")
        lines.append(stage.powershell())
        lines.append("")
    commands_path.write_text("\n".join(lines), encoding="utf-8")

    bash_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
    ]
    for stage in stages:
        bash_lines.append(f"echo '== {stage.name} =='")
        bash_lines.append(stage.bash())
        bash_lines.append("")
    bash_path.write_text("\n".join(bash_lines), encoding="utf-8", newline="\n")

    manifest = {
        "description": "T3 data-vs-capacity experiment plan",
        "generate_labels": bool(args.generate_labels),
        "paths": paths,
        "stages": [{"name": stage.name, "args": stage.args} for stage in stages],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def execute_stages(stages: Iterable[Stage]) -> None:
    for stage in stages:
        print(f"== {stage.name} ==", flush=True)
        print(" ".join(shlex.quote(arg) for arg in stage.args), flush=True)
        subprocess.run(stage.args, check=True)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare/run T3 data-vs-capacity experiments")
    parser.add_argument("--out-dir", default="ai/data/hybrid_t1t2_active_20260531/t3_data_capacity_20260603")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--generate-labels", action="store_true", help="Generate a fresh exact T3 dataset before training")
    parser.add_argument("--target-input", default=DEFAULT_TARGET_INPUT)
    parser.add_argument("--reranker-data", default=DEFAULT_RERANKER_DATA)
    parser.add_argument("--exact-records", type=int, default=5000)
    parser.add_argument("--exact-workers", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--max-per-source", type=int, default=1)
    parser.add_argument("--max-per-rank-group", type=int, default=1)
    parser.add_argument("--base-checkpoint", default=DEFAULT_BASE_CHECKPOINT)
    parser.add_argument("--holdout", default=DEFAULT_HOLDOUT)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--max-seconds", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--ranking-batches-per-epoch", type=int, default=128)
    parser.add_argument("--eval-max-samples", type=int, default=200000)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--capacity-lr", type=float, default=3e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--capacity-hidden", type=int, default=768)
    parser.add_argument("--capacity-blocks", type=int, default=5)
    parser.add_argument("--capacity-dropout", type=float, default=0.1)
    parser.add_argument("--capacity-adapter-dim", type=int, default=128)
    parser.add_argument("--t2-input", default=DEFAULT_T2_INPUT)
    parser.add_argument("--t2-exact", default=DEFAULT_T2_EXACT)
    parser.add_argument("--t2-limit", type=int, default=3)
    parser.add_argument("--t2-max-candidates", type=int, default=10)
    parser.add_argument("--t2-draw-limit", type=int, default=30)
    parser.add_argument("--t2-batch-size", type=int, default=4096)
    parser.add_argument("--execute", action="store_true", help="Run stages after writing commands and manifest")
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    stages, paths = build_stages(args)
    out_dir = Path(args.out_dir)
    write_outputs(out_dir, stages, paths, args)
    print(json.dumps({"paths": paths, "stages": [stage.name for stage in stages]}, indent=2), flush=True)
    if args.execute:
        execute_stages(stages)


if __name__ == "__main__":
    main()
