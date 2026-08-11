from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

import ofc_regular.aggregate_hu_m43_attempt08_preflight as aggregate
import ofc_regular.run_hu_m43_attempt08_preflight as preflight
from ofc_regular.hu_m43_attempt06_teacher import (
    ATTEMPT06_SHARD_ROW_SCHEMA,
    ATTEMPT06_TEACHER_SCHEMA,
)
from ofc_regular.hu_m43_attempt08_contract import (
    AI_PROFILES_SHA256,
    ATTEMPT08_LAMBDA_MODEL_SHA256,
    M43_ATTEMPT08_PLAN_SCHEMA,
    M43_ATTEMPT08_PLAN_SHA256,
)
from ofc_regular.hu_m43_attempt08_teacher import (
    ATTEMPT08_SOLVER_ID,
    ATTEMPT08_TEACHER_SCHEMA,
)
from ofc_regular.hu_m43_attempt08_runtime_anchor_contract import (
    ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SHA256,
    ATTEMPT08_RUNTIME_SOURCE_CLOSURE_SHA256,
)
from ofc_regular.hu_m43_attempt08_runtime_anchor import (
    ATTEMPT08_RUNTIME_REQUIREMENTS_SHA256,
)
from ofc_regular.hu_m43_attempt08_runtime_identity import (
    ATTEMPT08_EXPECTED_RUNTIME_FINGERPRINT_SHA256,
    ATTEMPT08_GCP_IMAGE_ID,
    ATTEMPT08_GCP_IMAGE_NAME,
)


def _proof(
    slot: str,
    *,
    opaque: str,
    semantic: str,
    elapsed: float,
    rss: int,
) -> dict:
    root, batch = preflight.ATTEMPT08_PREFLIGHT_SLOTS[slot]
    source_row, observation = preflight.load_source_row(
        preflight.DEFAULT_SOURCE_PATH, root
    )
    seeds = preflight.attempt08_preflight_seeds(root)
    return {
        "schema": preflight.ATTEMPT08_PREFLIGHT_PROOF_SCHEMA,
        "status": preflight.ATTEMPT08_PREFLIGHT_PROOF_STATUS,
        "slot": slot,
        "source": {
            "merged_sha256": preflight.ATTEMPT07_PREFLIGHT_SOURCE_SHA256,
            "source_root_index": root,
            "source_row_sha256": preflight._sha256_value(source_row),
            "wrapper_schema": ATTEMPT06_SHARD_ROW_SCHEMA,
            "teacher_schema": ATTEMPT06_TEACHER_SCHEMA,
            "observation_fingerprint": observation.fingerprint(),
            "source_already_consumed": True,
            "new_root_generated": False,
        },
        "contract": {
            "attempt08_plan_schema": M43_ATTEMPT08_PLAN_SCHEMA,
            "attempt08_plan_sha256": M43_ATTEMPT08_PLAN_SHA256,
            "preflight_plan_schema": preflight.ATTEMPT08_PREFLIGHT_PLAN_SCHEMA,
            "preflight_plan_sha256": preflight.ATTEMPT08_PREFLIGHT_PLAN_SHA256,
            "model_sha256": ATTEMPT08_LAMBDA_MODEL_SHA256,
            "ai_profiles_sha256": AI_PROFILES_SHA256,
            "runtime_semantic_anchor_sha256": (
                ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SHA256
            ),
            "runtime_source_closure_sha256": (
                ATTEMPT08_RUNTIME_SOURCE_CLOSURE_SHA256
            ),
            "runtime_requirements_sha256": ATTEMPT08_RUNTIME_REQUIREMENTS_SHA256,
            "gcp_image_name": ATTEMPT08_GCP_IMAGE_NAME,
            "gcp_image_id": ATTEMPT08_GCP_IMAGE_ID,
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
                f"attempt08-preflight:source-root={root}:"
                f"obs={observation.fingerprint()}"
            ),
            "teacher_elapsed_seconds": elapsed,
            "process_peak_rss_bytes": rss,
            "runtime_fingerprint_sha256": (
                ATTEMPT08_EXPECTED_RUNTIME_FINGERPRINT_SHA256
            ),
            "measurement_scope": (
                "elapsed_is_teacher_call_only_rss_is_one_root_process_high_water"
            ),
        },
        "result_proof": {
            "teacher_schema": ATTEMPT08_TEACHER_SCHEMA,
            "solver_id": ATTEMPT08_SOLVER_ID,
            "opaque_teacher_sha256": opaque,
            "semantic_parity_sha256": semantic,
            "exact_actionkey_reference_parity_verified": True,
            "hidden_information_safety_verified": True,
            "rng_domain_separation_verified": True,
            "conditional_X_A_skip_contract_verified": True,
            "teacher_action_or_value_details_exported": False,
        },
        "science_boundary": {
            "proof_only_not_policy_science": True,
            "development200_authorized": False,
            "future_audit_authorized": False,
            "fit_allowed": False,
            "threshold_selection_allowed": False,
            "runtime_activation_allowed": False,
            "current_profile_resolved": False,
            "current_profile_mutated": False,
            "fresh_seed_or_root_opened": False,
        },
    }


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(preflight.canonical_json_bytes(payload))


def _valid_inputs(tmp_path: Path) -> dict[str, Path]:
    specs = {
        "root0_batch_a": _proof(
            "root0_batch_a", opaque="a" * 64, semantic="b" * 64, elapsed=100.0, rss=1_000
        ),
        "root0_batch_b": _proof(
            "root0_batch_b", opaque="a" * 64, semantic="b" * 64, elapsed=110.0, rss=1_100
        ),
        "root0_scalar": _proof(
            "root0_scalar", opaque="c" * 64, semantic="b" * 64, elapsed=210.0, rss=1_200
        ),
        "root1_batch": _proof(
            "root1_batch", opaque="d" * 64, semantic="e" * 64, elapsed=90.0, rss=1_300
        ),
        "root2_batch": _proof(
            "root2_batch", opaque="f" * 64, semantic="1" * 64, elapsed=95.0, rss=1_400
        ),
    }
    paths: dict[str, Path] = {}
    for slot, payload in specs.items():
        path = tmp_path / f"{slot}.json"
        _write(path, payload)
        paths[slot] = path
    return paths


def _run(tmp_path: Path, paths: dict[str, Path], **overrides):
    values = {
        **paths,
        "output": tmp_path / "aggregate.json",
        "spot_operational_evidence_sha256": "9" * 64,
    }
    values.update(overrides)
    return aggregate.aggregate_preflight_proofs(**values)


def test_valid_five_proofs_go_and_aggregate_redacts_teacher_hashes_and_details(
    tmp_path: Path,
) -> None:
    paths = _valid_inputs(tmp_path)
    report = _run(tmp_path, paths)
    assert report["decision"] == "go"
    assert report["valid_proof_count"] == 5
    assert report["root_coverage"] == [0, 1, 2]
    assert all(report["proof_gates"].values())
    assert all(report["operational_gates"].values())
    assert report["operational_diagnostics"][
        "root0_scalar_to_batch_median_speedup_diagnostic"
    ] == 2.0
    assert report["science_boundary"]["development200_authorized"] is False
    raw = (tmp_path / "aggregate.json").read_bytes()
    assert raw == preflight.canonical_json_bytes(report)
    for forbidden in (
        b'"opaque_teacher_sha256"',
        b'"semantic_parity_sha256"',
        b'"selected_action_key"',
        b'"raw_paired_deltas',
        b'"paired_delta_vs_baseline"',
    ):
        assert forbidden not in raw


def test_root0_batch_hash_mismatch_and_scalar_semantic_mismatch_are_no_go(
    tmp_path: Path,
) -> None:
    paths = _valid_inputs(tmp_path)
    changed = json.loads(paths["root0_batch_b"].read_bytes())
    changed["result_proof"]["opaque_teacher_sha256"] = "9" * 64
    _write(paths["root0_batch_b"], changed)
    report = _run(tmp_path, paths)
    assert report["decision"] == "no_go"
    assert report["proof_gates"][
        "root0_batch_a_b_exact_teacher_determinism"
    ] is False

    second_dir = tmp_path / "semantic"
    second = _valid_inputs(second_dir)
    changed = json.loads(second["root0_scalar"].read_bytes())
    changed["result_proof"]["semantic_parity_sha256"] = "8" * 64
    _write(second["root0_scalar"], changed)
    report = _run(second_dir, second)
    assert report["decision"] == "no_go"
    assert report["proof_gates"]["root0_scalar_batch_semantic_parity"] is False


@pytest.mark.parametrize(
    ("slot", "field", "value", "gate"),
    [
        (
            "root1_batch",
            "teacher_elapsed_seconds",
            2400.0001,
            "teacher_elapsed_seconds_each_run_max_2400",
        ),
        (
            "root2_batch",
            "process_peak_rss_bytes",
            28 * 1024**3 + 1,
            "process_peak_rss_bytes_each_run_max_28GiB",
        ),
    ],
)
def test_operational_limits_are_fail_closed(
    tmp_path: Path, slot: str, field: str, value, gate: str
) -> None:
    paths = _valid_inputs(tmp_path)
    row = json.loads(paths[slot].read_bytes())
    row["execution"][field] = value
    _write(paths[slot], row)
    report = _run(tmp_path, paths)
    assert report["decision"] == "no_go"
    assert report["operational_gates"][gate] is False


def test_noncanonical_extra_or_value_bearing_proof_fails_closed(
    tmp_path: Path,
) -> None:
    paths = _valid_inputs(tmp_path)
    row = json.loads(paths["root2_batch"].read_bytes())
    row["teacher_action_details"] = {"mean": 999.0}
    _write(paths["root2_batch"], row)
    report = _run(tmp_path, paths)
    assert report["decision"] == "no_go"
    assert "gate_failed:all_five_canonical_proofs_valid" in report["reasons"]

    second_dir = tmp_path / "noncanonical"
    second = _valid_inputs(second_dir)
    second["root1_batch"].write_bytes(second["root1_batch"].read_bytes() + b"\n")
    report = _run(second_dir, second)
    assert report["decision"] == "no_go"
    assert "gate_failed:all_five_canonical_proofs_valid" in report["reasons"]


@pytest.mark.parametrize(
    ("block", "field"),
    [
        ("execution", "measurement_scope"),
        ("result_proof", "rng_domain_separation_verified"),
    ],
)
def test_truncated_nested_proof_contract_fails_closed(
    tmp_path: Path, block: str, field: str
) -> None:
    paths = _valid_inputs(tmp_path)
    row = json.loads(paths["root1_batch"].read_bytes())
    row[block].pop(field)
    _write(paths["root1_batch"], row)
    report = _run(tmp_path, paths)
    assert report["decision"] == "no_go"
    assert report["valid_proof_count"] == 4
    assert "gate_failed:all_five_canonical_proofs_valid" in report["reasons"]

def test_proof_paths_are_distinct_and_output_is_no_clobber(tmp_path: Path) -> None:
    paths = _valid_inputs(tmp_path)
    paths["root0_batch_b"] = paths["root0_batch_a"]
    report = _run(tmp_path, paths)
    assert report["decision"] == "no_go"
    assert report["proof_gates"]["five_proof_artifact_paths_distinct"] is False

    second_dir = tmp_path / "owned"
    second = _valid_inputs(second_dir)
    output = second_dir / "aggregate.json"
    output.write_text("owned\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        _run(second_dir, second, output=output)
    assert output.read_text(encoding="utf-8") == "owned\n"
    with pytest.raises(ValueError, match="aliases"):
        _run(second_dir, second, output=second["root0_scalar"])


def test_aggregate_internal_tampering_fails_validation(tmp_path: Path) -> None:
    paths = _valid_inputs(tmp_path)
    report = _run(tmp_path, paths)
    changed = copy.deepcopy(report)
    changed["proof_evidence_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        aggregate.validate_preflight_aggregate(changed)
    changed = copy.deepcopy(report)
    changed["operational_gates"][
        "teacher_elapsed_seconds_each_run_max_2400"
    ] = False
    with pytest.raises(ValueError):
        aggregate.validate_preflight_aggregate(changed)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value.__setitem__("proof_file_sha256", {}),
        lambda value: value.__setitem__("proof_gates", {}),
        lambda value: value.__setitem__("operational_gates", {}),
        lambda value: value.__setitem__("valid_proof_count", 0),
        lambda value: value.__setitem__("root_coverage", []),
        lambda value: value["operational_diagnostics"].__setitem__("jobs", {}),
        lambda value: value["operational_diagnostics"].__setitem__(
            "teacher_elapsed_seconds_max", 0.0
        ),
        lambda value: value.__setitem__("reasons", []),
        lambda value: value["contract"].__setitem__("unexpected", False),
    ],
)
def test_forged_go_with_empty_truncated_or_inconsistent_evidence_is_rejected(
    tmp_path: Path, mutator
) -> None:
    paths = _valid_inputs(tmp_path)
    report = _run(tmp_path, paths)
    changed = copy.deepcopy(report)
    mutator(changed)
    with pytest.raises(ValueError):
        aggregate.validate_preflight_aggregate(changed)


def test_report_proof_hashes_bind_actual_canonical_files(tmp_path: Path) -> None:
    paths = _valid_inputs(tmp_path)
    report = _run(tmp_path, paths)
    for slot, path in paths.items():
        assert report["proof_file_sha256"][slot] == hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
