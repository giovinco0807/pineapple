from pathlib import Path

from ai.tutor.run_t3_data_capacity_experiment import build_stages, parse_args, write_outputs


def test_default_plan_uses_existing_t3_reranker_data(tmp_path: Path) -> None:
    args = parse_args(["--out-dir", str(tmp_path), "--python", "py"])

    stages, paths = build_stages(args)
    current_stage = next(stage for stage in stages if stage.name == "train_current_arch_unfrozen")
    capacity_stage = next(stage for stage in stages if stage.name == "train_large_capacity")

    assert [stage.name for stage in stages] == [
        "train_current_arch_unfrozen",
        "train_large_capacity",
        "eval_current_arch_t3_holdout",
        "bench_current_arch_t2_via_t3",
        "eval_large_capacity_t3_holdout",
        "bench_large_capacity_t2_via_t3",
    ]
    assert paths["reranker_data"].endswith("exact_late_reranker_t3t4_10000_20260529")
    assert any("--init-checkpoint" in stage.args for stage in stages)
    assert current_stage.args[current_stage.args.index("--lr") + 1] == "2e-06"
    assert capacity_stage.args[capacity_stage.args.index("--lr") + 1] == "0.0003"


def test_generate_labels_plan_adds_exact_data_stages(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--out-dir",
            str(tmp_path),
            "--python",
            "py",
            "--generate-labels",
            "--exact-records",
            "123",
        ]
    )

    stages, paths = build_stages(args)

    assert [stage.name for stage in stages[:3]] == [
        "select_t3_inputs",
        "generate_exact_t3_labels",
        "convert_exact_t3_to_reranker",
    ]
    exact_stage = stages[1]
    assert "--input" not in exact_stage.args
    assert "t3_input.jsonl" in exact_stage.args[3]
    assert "reranker_t3_exact" in paths["reranker_data"]
    assert "123" in stages[0].args


def test_write_outputs_creates_manifest_and_powershell(tmp_path: Path) -> None:
    args = parse_args(["--out-dir", str(tmp_path), "--python", "py"])
    stages, paths = build_stages(args)

    write_outputs(tmp_path, stages, paths, args)

    manifest = tmp_path / "manifest.json"
    commands = tmp_path / "commands.ps1"
    commands_sh = tmp_path / "commands.sh"
    assert manifest.exists()
    assert commands.exists()
    assert commands_sh.exists()
    text = commands.read_text(encoding="utf-8")
    assert "train_current_arch_unfrozen" in text
    assert "benchmark_t2_t3_value_model" in text
    bash_text = commands_sh.read_text(encoding="utf-8")
    assert "set -euo pipefail" in bash_text
    assert "benchmark_t2_t3_value_model" in bash_text
