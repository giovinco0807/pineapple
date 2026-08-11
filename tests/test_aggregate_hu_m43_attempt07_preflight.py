from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

import ofc_regular.aggregate_hu_m43_attempt07_preflight as aggregate
import ofc_regular.run_hu_m43_attempt07_preflight as preflight
from ofc_regular.hu_m43_attempt06_teacher import (
    ATTEMPT06_FROZEN_MODEL_SHA256,
    ATTEMPT06_SHARD_ROW_SCHEMA,
    ATTEMPT06_TEACHER_SCHEMA,
)
from ofc_regular.hu_m43_attempt07_contract import (
    AI_PROFILES_SHA256,
    M43_ATTEMPT07_PLAN_SHA256,
)
from ofc_regular.hu_m43_attempt07_teacher import (
    ATTEMPT07_SOLVER_ID,
    ATTEMPT07_TEACHER_SCHEMA,
)


def _proof_row(
    root_index: int,
    *,
    batch: bool,
    opaque_sha256: str,
    semantic_sha256: str,
) -> dict:
    source_row, observation = preflight.load_source_row(
        preflight.DEFAULT_SOURCE_PATH, root_index
    )
    seeds = preflight.attempt07_preflight_seeds(root_index)
    return {
        "schema": preflight.ATTEMPT07_PREFLIGHT_ROW_SCHEMA,
        "status": preflight.ATTEMPT07_PREFLIGHT_STATUS,
        "source": {
            "merged_sha256": preflight.ATTEMPT07_PREFLIGHT_SOURCE_SHA256,
            "source_root_index": root_index,
            "source_row_sha256": preflight._sha256_value(source_row),
            "wrapper_schema": ATTEMPT06_SHARD_ROW_SCHEMA,
            "teacher_schema": ATTEMPT06_TEACHER_SCHEMA,
            "observation_fingerprint": observation.fingerprint(),
            "baseline_action_key": source_row["baseline_action_key"],
            "new_root_generated": False,
        },
        "contract": {
            "plan_sha256": M43_ATTEMPT07_PLAN_SHA256,
            "preflight_plan_sha256": aggregate._sha256_file(
                preflight.DEFAULT_PREFLIGHT_PLAN_PATH
            ),
            "model_sha256": ATTEMPT06_FROZEN_MODEL_SHA256,
            "ai_profiles_sha256": AI_PROFILES_SHA256,
            "t2_policy_id": "stage9f_p2",
            "t2_resolution": "explicit_profile_never_current",
            "seeds": seeds,
            "continuation_policy_seeds": {
                "first": seeds["child"],
                "second": seeds["child"] + 1,
            },
        },
        "execution": {
            "batch_child_selectors": batch,
            "native_batch_threads": 4,
            "run_id": (
                f"attempt07-preflight:source-root={root_index}:"
                f"obs={observation.fingerprint()}"
            ),
        },
        "result_proof": {
            "teacher_schema": ATTEMPT07_TEACHER_SCHEMA,
            "solver_id": ATTEMPT07_SOLVER_ID,
            "opaque_teacher_sha256": opaque_sha256,
            "opaque_teacher_sha256_purpose": (
                "determinism_only_not_arm_selection"
            ),
            "semantic_parity_sha256": semantic_sha256,
            "semantic_parity_sha256_purpose": (
                "scalar_batch_exact_parity_after_normalizing_only_"
                "batch_execution_metadata"
            ),
            "teacher_values_exported": False,
            "arm_details_exported": False,
        },
        "science_boundary": {
            "arm_selection_allowed": False,
            "fit_allowed": False,
            "threshold_selection_allowed": False,
            "runtime_activation_allowed": False,
            "current_profile_resolved": False,
            "current_profile_mutated": False,
            "fresh_seed_or_root_opened": False,
        },
    }


def _write(path: Path, row: dict) -> None:
    path.write_bytes(preflight.canonical_json_bytes(row))


def _valid_inputs(tmp_path: Path) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    specs = {
        "root0_batch_a": _proof_row(
            0, batch=True, opaque_sha256="a" * 64, semantic_sha256="c" * 64
        ),
        "root0_batch_b": _proof_row(
            0, batch=True, opaque_sha256="a" * 64, semantic_sha256="c" * 64
        ),
        "root0_scalar": _proof_row(
            0, batch=False, opaque_sha256="b" * 64, semantic_sha256="c" * 64
        ),
        "root1_batch": _proof_row(
            1, batch=True, opaque_sha256="d" * 64, semantic_sha256="e" * 64
        ),
        "root2_batch": _proof_row(
            2, batch=True, opaque_sha256="f" * 64, semantic_sha256="1" * 64
        ),
    }
    paths: dict[str, Path] = {}
    for label, row in specs.items():
        path = tmp_path / f"{label}.json"
        _write(path, row)
        paths[label] = path
    return paths


def _run(tmp_path: Path, paths: dict[str, Path], **overrides):
    values = {**paths, "output": tmp_path / "aggregate.json"}
    values.update(overrides)
    return aggregate.aggregate_preflight_proofs(**values)


def test_valid_five_proofs_go_without_exposing_values_or_arm_details(
    tmp_path: Path,
) -> None:
    paths = _valid_inputs(tmp_path)
    report = _run(tmp_path, paths)
    output = tmp_path / "aggregate.json"
    assert report["decision"] == "go"
    assert report["reasons"] == ["all_preflight_proof_gates_passed"]
    assert report["valid_proof_count"] == 5
    assert report["root_coverage"] == [0, 1, 2]
    assert all(report["proof_gates"].values())
    assert report["proof_file_sha256"]["root0_batch_a"] == report[
        "proof_file_sha256"
    ]["root0_batch_b"]
    assert report["cross_mode_opaque_teacher_hash_compared"] is False
    assert report["science_boundary"]["current_profile_mutated"] is False
    assert report["science_boundary"]["done_metadata_is_science_input"] is False
    raw = output.read_bytes()
    assert raw == preflight.canonical_json_bytes(report)
    assert b'"arms":' not in raw
    assert b'"assessment":' not in raw
    assert b'"opaque_teacher_sha256":' not in raw


def test_noncanonical_or_extra_value_bearing_proof_is_no_go(tmp_path: Path) -> None:
    paths = _valid_inputs(tmp_path)
    paths["root1_batch"].write_bytes(paths["root1_batch"].read_bytes() + b"\n")
    report = _run(tmp_path, paths)
    assert report["decision"] == "no_go"
    assert "proof_validation_failed:root1_batch" in report["reasons"]
    assert report["valid_proof_count"] == 4
    assert report["proof_gates"]["all_five_canonical_proofs_valid"] is False

    paths = _valid_inputs(tmp_path / "extra")
    row = json.loads(paths["root2_batch"].read_bytes())
    row["arms"] = {"R32_V64": {"mean": 999.0}}
    _write(paths["root2_batch"], row)
    report = _run(
        tmp_path / "extra",
        paths,
        output=tmp_path / "extra" / "aggregate-extra.json",
    )
    assert report["decision"] == "no_go"
    assert "proof_validation_failed:root2_batch" in report["reasons"]


def test_batch_a_b_requires_complete_canonical_and_opaque_determinism(
    tmp_path: Path,
) -> None:
    paths = _valid_inputs(tmp_path)
    changed = json.loads(paths["root0_batch_b"].read_bytes())
    changed["result_proof"]["opaque_teacher_sha256"] = "9" * 64
    _write(paths["root0_batch_b"], changed)
    report = _run(tmp_path, paths)
    assert report["decision"] == "no_go"
    assert "root0_batch_canonical_determinism_failed" in report["reasons"]
    assert "root0_batch_opaque_determinism_failed" in report["reasons"]
    assert report["proof_gates"][
        "root0_batch_a_b_canonical_rows_identical"
    ] is False


def test_batch_row_and_opaque_determinism_are_separate_gates(tmp_path: Path) -> None:
    paths = _valid_inputs(tmp_path)
    changed = json.loads(paths["root0_batch_b"].read_bytes())
    changed["result_proof"]["semantic_parity_sha256"] = "7" * 64
    _write(paths["root0_batch_b"], changed)
    report = _run(tmp_path, paths)
    assert report["decision"] == "no_go"
    assert report["proof_gates"][
        "root0_batch_a_b_canonical_rows_identical"
    ] is False
    assert report["proof_gates"]["root0_batch_opaque_mode_determinism"] is True


def test_scalar_batch_uses_semantic_hash_not_cross_mode_opaque_hash(
    tmp_path: Path,
) -> None:
    paths = _valid_inputs(tmp_path)
    # Different scalar/batch opaque hashes are expected and never compared.
    good = _run(tmp_path, paths, output=tmp_path / "good.json")
    assert good["decision"] == "go"
    changed = json.loads(paths["root0_scalar"].read_bytes())
    changed["result_proof"]["semantic_parity_sha256"] = "8" * 64
    _write(paths["root0_scalar"], changed)
    bad = _run(tmp_path, paths, output=tmp_path / "bad.json")
    assert bad["decision"] == "no_go"
    assert "root0_scalar_batch_semantic_parity_failed" in bad["reasons"]
    assert bad["cross_mode_opaque_teacher_hash_compared"] is False


@pytest.mark.parametrize(
    ("label", "mutate"),
    [
        (
            "root1_batch",
            lambda row: row["source"].__setitem__("source_root_index", 0),
        ),
        (
            "root2_batch",
            lambda row: row["execution"].__setitem__(
                "batch_child_selectors", False
            ),
        ),
        (
            "root0_scalar",
            lambda row: row["contract"]["seeds"].__setitem__("screen", 1),
        ),
        (
            "root0_batch_a",
            lambda row: row["contract"].__setitem__("plan_sha256", "0" * 64),
        ),
    ],
)
def test_root_mode_seed_and_hash_tampering_fail_closed(
    tmp_path: Path,
    label: str,
    mutate,
) -> None:
    paths = _valid_inputs(tmp_path)
    row = json.loads(paths[label].read_bytes())
    mutate(row)
    _write(paths[label], row)
    report = _run(tmp_path, paths)
    assert report["decision"] == "no_go"
    assert f"proof_validation_failed:{label}" in report["reasons"]


def test_done_metadata_projects_only_elapsed_and_rss_outside_science(
    tmp_path: Path,
) -> None:
    paths = _valid_inputs(tmp_path)
    done = tmp_path / "DONE.json"
    done.write_text(
        json.dumps(
            {
                "elapsed_seconds": 12.5,
                "peak_rss_bytes": 123456,
                "teacher_mean": 999.0,
                "secret_arm": "R32_V64",
                "finished_at": "never-export-this",
            }
        ),
        encoding="utf-8",
    )
    report = _run(
        tmp_path,
        paths,
        done_metadata={"root0_batch_a": done},
    )
    assert report["decision"] == "go"
    operations = report["operational_diagnostics"]
    assert operations["science_decision_input"] is False
    assert operations["jobs"] == {
        "root0_batch_a": {
            "elapsed_seconds": 12.5,
            "peak_rss_bytes": 123456,
        }
    }
    encoded = json.dumps(report, sort_keys=True)
    assert "teacher_mean" not in encoded
    assert "secret_arm" not in encoded
    assert "finished_at" not in encoded

    done.write_text('{"elapsed_seconds":"bad"}', encoding="utf-8")
    second = _run(
        tmp_path,
        paths,
        output=tmp_path / "with-invalid-done.json",
        done_metadata={"root0_batch_a": done},
    )
    assert second["decision"] == "go"
    assert second["operational_diagnostics"]["status"] == (
        "partial_invalid_not_used_for_science"
    )


def test_no_clobber_and_input_alias_rejection_happen_before_publish(
    tmp_path: Path,
) -> None:
    paths = _valid_inputs(tmp_path)
    output = tmp_path / "owned.json"
    output.write_text("owned\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        _run(tmp_path, paths, output=output)
    assert output.read_text(encoding="utf-8") == "owned\n"
    with pytest.raises(ValueError, match="aliases"):
        _run(tmp_path, paths, output=paths["root0_scalar"])
    idempotent = tmp_path / "aggregate.json"
    first = _run(tmp_path, paths, output=idempotent)
    before = idempotent.read_bytes()
    second = _run(tmp_path, paths, output=idempotent)
    assert second == first
    assert idempotent.read_bytes() == before


def test_five_proof_artifacts_must_be_physically_distinct(tmp_path: Path) -> None:
    paths = _valid_inputs(tmp_path)
    paths["root0_batch_b"] = paths["root0_batch_a"]
    report = _run(tmp_path, paths)
    assert report["decision"] == "no_go"
    assert "five_proof_artifact_paths_not_distinct" in report["reasons"]
    assert report["proof_gates"]["five_proof_artifact_paths_distinct"] is False


def test_report_proof_hashes_are_canonical_file_hashes(tmp_path: Path) -> None:
    paths = _valid_inputs(tmp_path)
    report = _run(tmp_path, paths)
    for label, path in paths.items():
        assert report["proof_file_sha256"][label] == hashlib.sha256(
            path.read_bytes()
        ).hexdigest()


def test_cli_requires_all_five_frozen_slots_and_parses_done_metadata(
    tmp_path: Path,
) -> None:
    paths = _valid_inputs(tmp_path)
    argv: list[str] = []
    for label, path in paths.items():
        argv.extend([f"--{label.replace('_', '-')}", str(path)])
    argv.extend(
        [
            "--output",
            str(tmp_path / "aggregate.json"),
            "--done-metadata",
            f"root0_batch_a={tmp_path / 'DONE.json'}",
        ]
    )
    args = aggregate._parser().parse_args(argv)
    assert args.root0_batch_a == str(paths["root0_batch_a"])
    assert args.root2_batch == str(paths["root2_batch"])
    parsed = aggregate._parse_done_metadata(args.done_metadata)
    assert parsed == {"root0_batch_a": tmp_path / "DONE.json"}
    with pytest.raises(ValueError, match="SLOT=PATH"):
        aggregate._parse_done_metadata(["unknown=x.json"])
