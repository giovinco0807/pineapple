"""Build exact T3 EV teacher data and train BB/BTN EV rerankers.

This is an orchestration script for local, D-drive-first T3 model work:

1. Select/copy T3 input rows.
2. Run the Rust exact T3/T4 solver with the current FL EV config.
3. Convert Rust output to action-value teacher JSONL.
4. Convert teacher JSONL to candidate-level numpy arrays.
5. Train separate BB and BTN action-value rerankers.
6. Evaluate TopK + exact-rerank EV loss on the generated exact teacher.

The script writes commands and a manifest even when --execute is not supplied,
so long runs remain reproducible.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable


DEFAULT_OUT_DIR = Path("D:/ofc-pineapple-data/t3_ev_precision_20260614")
DEFAULT_SOURCE_INPUT = Path(
    "ai/data/t3_relabel_jokerfix_20k_20260605/inputs/t3_input_20000.jsonl"
)
DEFAULT_RUST_EXE = Path("ai/rust_solver/target/release/t3_exact_solver.exe")
DEFAULT_FL_CONFIG = Path("ai/config/fl_ev.json")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def rel_to_repo(path: Path) -> Path:
    if path.is_absolute():
        return path
    return repo_root() / path


def path_text(path: Path) -> str:
    return str(path)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def run_command(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    execute: bool,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    rendered = " ".join(f'"{part}"' if " " in part else part for part in command)
    record: dict[str, Any] = {
        "command": command,
        "rendered": rendered,
        "cwd": str(cwd),
        "log": str(log_path),
        "executed": bool(execute),
        "returncode": None,
        "elapsed_s": 0.0,
    }
    if not execute:
        return record

    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {rendered}\n\n")
        log.flush()
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    record["returncode"] = int(proc.returncode)
    record["elapsed_s"] = time.time() - start
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {rendered}\nLog: {log_path}")
    return record


def select_input(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    source = rel_to_repo(Path(args.source_input))
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.execute and (args.force or not output.exists()):
        if args.use_diverse_selector:
            command = [
                sys.executable,
                "-m",
                "ai.tutor.select_diverse_oracle_inputs",
                "--input",
                path_text(source),
                "--output",
                path_text(output),
                "--turns",
                "3",
                "--limit",
                str(args.limit),
                "--max-per-source",
                str(args.max_per_source),
                "--max-per-rank-group",
                str(args.max_per_rank_group),
            ]
            run_command(
                command,
                cwd=repo_root(),
                log_path=args.out_dir / "logs" / "01_select_input.log",
                execute=True,
            )
        else:
            written = 0
            with source.open("r", encoding="utf-8-sig") as src, output.open("w", encoding="utf-8") as dst:
                for line in src:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if int(row.get("turn", -1)) != 3:
                        continue
                    dst.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    written += 1
                    if args.limit > 0 and written >= args.limit:
                        break
            write_json(
                output.with_suffix(".summary.json"),
                {
                    "input": str(source),
                    "output": str(output),
                    "turns": [3],
                    "limit": int(args.limit),
                    "selected": written,
                    "selector": "copy_first_matching_t3",
                },
            )
    elif not output.exists():
        # Dry-run convenience: write a small manifest sidecar without copying.
        write_json(
            output.with_suffix(".summary.json"),
            {
                "input": str(source),
                "output": str(output),
                "turns": [3],
                "limit": int(args.limit),
                "selected": 0,
                "selector": "planned",
            },
        )
    return {
        "source_input": str(source),
        "selected_input": str(output),
        "selected_rows": count_jsonl(output),
    }


def build_commands(args: argparse.Namespace) -> dict[str, Any]:
    out_dir: Path = args.out_dir
    selected_input = out_dir / "inputs" / f"t3_input_{args.limit}.jsonl"
    rust_output = out_dir / "exact" / "t3_exact.rust.jsonl"
    teacher_output = out_dir / "teacher" / "t3_exact_teacher.jsonl"
    reranker_dir = out_dir / "reranker_data" / "t3_exact_ev"
    model_bb = out_dir / "models" / "t3-ev-exact-bb"
    model_btn = out_dir / "models" / "t3-ev-exact-btn"
    eval_output = out_dir / "eval" / "t3_ev_exact_models.json"

    python = sys.executable
    rust_exe = rel_to_repo(Path(args.rust_exe))
    fl_config = rel_to_repo(Path(args.fl_config))

    commands: dict[str, list[str]] = {
        "rust_exact": [
            path_text(rust_exe),
            "--input",
            path_text(selected_input),
            "--output",
            path_text(rust_output),
            "--top-n",
            str(args.rust_top_n),
            "--fl-config",
            path_text(fl_config),
        ],
        "convert_teacher": [
            python,
            "-m",
            "ai.tutor.convert_rust_t3_exact_teacher",
            "--input",
            path_text(selected_input),
            "--rust-output",
            path_text(rust_output),
            "--output",
            path_text(teacher_output),
        ],
        "convert_reranker": [
            python,
            "-m",
            "ai.training.convert_action_value_teacher",
            path_text(teacher_output),
            "--output",
            path_text(reranker_dir),
            "--turns",
            "3",
            "--state-dim",
            str(args.state_dim),
            "--regular-max-candidates",
            "0",
            "--teacher-best-weight",
            str(args.teacher_best_weight),
            "--gap-weight",
            str(args.gap_weight),
            "--fl-positive-weight",
            str(args.fl_positive_weight),
            "--fl-shape-but-foul-weight",
            str(args.fl_shape_but_foul_weight),
            "--clean-fl-weight",
            str(args.clean_fl_weight),
            "--max-sample-weight",
            str(args.max_sample_weight),
        ],
        "train_bb": [
            python,
            "-m",
            "ai.training.train_action_value_reranker",
            "--data",
            path_text(reranker_dir),
            "--save-dir",
            path_text(model_bb),
            "--train-turns",
            "3",
            "--train-position",
            "bb",
            "--epochs",
            str(args.epochs),
            "--max-seconds",
            str(args.max_seconds),
            "--batch-size",
            str(args.batch_size),
            "--hidden",
            str(args.hidden),
            "--n-blocks",
            str(args.n_blocks),
            "--dropout",
            str(args.dropout),
            "--score-weight",
            str(args.score_weight),
            "--bust-weight",
            str(args.bust_weight),
            "--fl-weight",
            str(args.fl_weight),
            "--fl-type-weight",
            str(args.fl_type_weight),
            "--ranking-weight",
            str(args.ranking_weight),
            "--topk-rank-weight",
            str(args.topk_rank_weight),
            "--topk-margin-weight",
            str(args.topk_margin_weight),
            "--target-topk",
            str(args.target_topk),
            "--selection-metric",
            args.selection_metric,
            "--eval-max-samples",
            str(args.eval_max_samples),
            "--device",
            args.device,
            "--seed",
            str(args.seed),
        ],
        "train_btn": [
            python,
            "-m",
            "ai.training.train_action_value_reranker",
            "--data",
            path_text(reranker_dir),
            "--save-dir",
            path_text(model_btn),
            "--train-turns",
            "3",
            "--train-position",
            "btn",
            "--epochs",
            str(args.epochs),
            "--max-seconds",
            str(args.max_seconds),
            "--batch-size",
            str(args.batch_size),
            "--hidden",
            str(args.hidden),
            "--n-blocks",
            str(args.n_blocks),
            "--dropout",
            str(args.dropout),
            "--score-weight",
            str(args.score_weight),
            "--bust-weight",
            str(args.bust_weight),
            "--fl-weight",
            str(args.fl_weight),
            "--fl-type-weight",
            str(args.fl_type_weight),
            "--ranking-weight",
            str(args.ranking_weight),
            "--topk-rank-weight",
            str(args.topk_rank_weight),
            "--topk-margin-weight",
            str(args.topk_margin_weight),
            "--target-topk",
            str(args.target_topk),
            "--selection-metric",
            args.selection_metric,
            "--eval-max-samples",
            str(args.eval_max_samples),
            "--device",
            args.device,
            "--seed",
            str(args.seed + 1),
        ],
        "evaluate": [
            python,
            "-m",
            "ai.tutor.evaluate_t3_exact_teacher_models",
            "--teacher",
            path_text(teacher_output),
            "--model",
            f"ev_exact={model_bb / 'action_value_best.pt'},{model_btn / 'action_value_best.pt'}",
            "--topks",
            args.topks,
            "--output",
            path_text(eval_output),
            "--device",
            args.device,
            "--batch-size",
            str(args.eval_batch_size),
        ],
    }
    return {
        "selected_input": selected_input,
        "rust_output": rust_output,
        "teacher_output": teacher_output,
        "reranker_dir": reranker_dir,
        "model_bb": model_bb,
        "model_btn": model_btn,
        "eval_output": eval_output,
        "commands": commands,
    }


def write_commands_ps1(out_dir: Path, commands: dict[str, list[str]]) -> None:
    lines = [
        "$ErrorActionPreference = 'Stop'",
        f"Set-Location -LiteralPath {json.dumps(str(repo_root()))}",
        "",
    ]
    for name, command in commands.items():
        rendered = " ".join(f'"{part}"' if " " in part else part for part in command)
        lines.append(f"Write-Host '== {name} =='")
        lines.append(rendered)
        lines.append("")
    (out_dir / "commands.ps1").write_text("\n".join(lines), encoding="utf-8")


def summarize_outputs(paths: dict[str, Any]) -> dict[str, Any]:
    selected_input: Path = paths["selected_input"]
    rust_output: Path = paths["rust_output"]
    teacher_output: Path = paths["teacher_output"]
    reranker_dir: Path = paths["reranker_dir"]
    model_bb: Path = paths["model_bb"]
    model_btn: Path = paths["model_btn"]
    eval_output: Path = paths["eval_output"]

    summary: dict[str, Any] = {
        "selected_rows": count_jsonl(selected_input),
        "rust_rows": count_jsonl(rust_output),
        "teacher_rows": count_jsonl(teacher_output),
        "reranker_metadata": None,
        "bb_training": None,
        "btn_training": None,
        "evaluation": None,
    }
    metadata_path = reranker_dir / "metadata.json"
    if metadata_path.exists():
        summary["reranker_metadata"] = json.loads(metadata_path.read_text(encoding="utf-8"))
    for key, model_dir in (("bb_training", model_bb), ("btn_training", model_btn)):
        path = model_dir / "summary.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            summary[key] = {
                "n_samples": data.get("n_samples"),
                "train_samples": data.get("train_samples"),
                "val_samples": data.get("val_samples"),
                "best_epoch": data.get("best_epoch"),
                "final": data.get("final"),
            }
    if eval_output.exists():
        summary["evaluation"] = json.loads(eval_output.read_text(encoding="utf-8"))
    return summary


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dir = Path(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.force and args.execute and args.clean and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
        args.out_dir.mkdir(parents=True, exist_ok=True)

    built = build_commands(args)
    write_commands_ps1(args.out_dir, built["commands"])

    manifest: dict[str, Any] = {
        "out_dir": str(args.out_dir),
        "created_at_unix": time.time(),
        "execute": bool(args.execute),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "paths": {
            key: str(value)
            for key, value in built.items()
            if key != "commands"
        },
        "commands": built["commands"],
        "steps": [],
    }
    write_json(args.out_dir / "manifest.json", manifest)

    selection = select_input(args, built["selected_input"])
    manifest["selection"] = selection
    write_json(args.out_dir / "manifest.json", manifest)

    if args.execute:
        env = os.environ.copy()
        env.setdefault("PYTHONUTF8", "1")
        for name in ("rust_exact", "convert_teacher", "convert_reranker", "train_bb", "train_btn", "evaluate"):
            command = built["commands"][name]
            log_path = args.out_dir / "logs" / f"{len(manifest['steps']) + 2:02d}_{name}.log"
            # Reuse existing heavy outputs unless --force was requested.
            target = {
                "rust_exact": built["rust_output"],
                "convert_teacher": built["teacher_output"],
                "convert_reranker": built["reranker_dir"] / "metadata.json",
                "train_bb": built["model_bb"] / "action_value_best.pt",
                "train_btn": built["model_btn"] / "action_value_best.pt",
                "evaluate": built["eval_output"],
            }[name]
            if target.exists() and not args.force:
                manifest["steps"].append(
                    {
                        "name": name,
                        "skipped": True,
                        "reason": "target_exists",
                        "target": str(target),
                    }
                )
                write_json(args.out_dir / "manifest.json", manifest)
                continue
            if target.suffix:
                target.parent.mkdir(parents=True, exist_ok=True)
            else:
                target.mkdir(parents=True, exist_ok=True)
            result = run_command(
                command,
                cwd=repo_root(),
                log_path=log_path,
                execute=True,
                env=env,
            )
            result["name"] = name
            manifest["steps"].append(result)
            write_json(args.out_dir / "manifest.json", manifest)

    manifest["summary"] = summarize_outputs(built)
    write_json(args.out_dir / "manifest.json", manifest)
    return manifest


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run local D-drive T3 EV exact teacher and model pipeline")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--source-input", type=Path, default=DEFAULT_SOURCE_INPUT)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--use-diverse-selector", action="store_true")
    parser.add_argument("--max-per-source", type=int, default=1)
    parser.add_argument("--max-per-rank-group", type=int, default=0)
    parser.add_argument("--rust-exe", type=Path, default=DEFAULT_RUST_EXE)
    parser.add_argument("--fl-config", type=Path, default=DEFAULT_FL_CONFIG)
    parser.add_argument("--rust-top-n", type=int, default=64)
    parser.add_argument("--state-dim", type=int, default=520)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max-seconds", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--n-blocks", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--score-weight", type=float, default=2.0)
    parser.add_argument("--bust-weight", type=float, default=0.2)
    parser.add_argument("--fl-weight", type=float, default=0.3)
    parser.add_argument("--fl-type-weight", type=float, default=0.3)
    parser.add_argument("--ranking-weight", type=float, default=0.4)
    parser.add_argument("--topk-rank-weight", type=float, default=1.0)
    parser.add_argument("--topk-margin-weight", type=float, default=0.2)
    parser.add_argument("--target-topk", type=int, default=10)
    parser.add_argument("--selection-metric", choices=("regret", "topk"), default="topk")
    parser.add_argument("--teacher-best-weight", type=float, default=0.75)
    parser.add_argument("--gap-weight", type=float, default=1.0)
    parser.add_argument("--fl-positive-weight", type=float, default=1.0)
    parser.add_argument("--fl-shape-but-foul-weight", type=float, default=0.5)
    parser.add_argument("--clean-fl-weight", type=float, default=0.5)
    parser.add_argument("--max-sample-weight", type=float, default=10.0)
    parser.add_argument("--eval-max-samples", type=int, default=200_000)
    parser.add_argument("--topks", default="1,3,5,8,10,15,20")
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260614)
    args = parser.parse_args(list(argv) if argv is not None else None)

    manifest = run_pipeline(args)
    print(json.dumps(manifest["summary"], indent=2, ensure_ascii=False))
    print(f"Manifest: {args.out_dir / 'manifest.json'}")
    print(f"Commands: {args.out_dir / 'commands.ps1'}")


if __name__ == "__main__":
    main()
