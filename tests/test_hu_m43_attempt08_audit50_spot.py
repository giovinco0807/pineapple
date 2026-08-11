from __future__ import annotations

import json
from pathlib import Path

import pytest

from ofc_regular import hu_m43_attempt08_audit50_spot as spot
from ofc_regular.hu_m43_attempt08_audit50_contract import (
    AUDIT50_PLAN_SHA256,
    DEVELOPMENT_FUTURE_RUNNER_SHA256,
    TOTAL_SHARDS,
    build_audit50_schedule,
    canonical_json_bytes,
    schedule_bytes,
    sha256_file,
    write_canonical_json,
)


ROOT = Path(__file__).resolve().parents[1]
DEV_RUN = (
    ROOT
    / "outputs/gcp_runs"
    / "regular-hu-m43-attempt08-development200-finalprop-20260714-215952"
)


def _done_run(tmp_path: Path) -> tuple[Path, Path, Path]:
    run = tmp_path / "run"
    run.mkdir()
    write_canonical_json(run / "manifest.json", {"run_name": "audit50-test"})
    (run / "shards_manifest.jsonl").write_bytes(
        schedule_bytes(build_audit50_schedule())
    )
    launch = {
        "audit_open_authorization_sha256": "a" * 64,
        "core_authorization_sha256": "b" * 64,
    }
    write_canonical_json(run / "launch_authorization.json", launch)
    write_canonical_json(run / "audit_open_authorization.json", {"test": True})
    done_root = tmp_path / "done"
    done_root.mkdir()
    for shard, spec in enumerate(build_audit50_schedule()):
        payload = {
            "schema": spot.DONE_SCHEMA,
            "status": "complete_done_published_last",
            "run_name": "audit50-test",
            "shard": shard,
            "root_index": spec["root_index"],
            "root_profile": spec["root_profile"],
            "manifest_sha256": sha256_file(run / "manifest.json"),
            "launch_authorization_sha256": sha256_file(
                run / "launch_authorization.json"
            ),
            "audit_open_authorization_sha256": "a" * 64,
            "core_authorization_sha256": "b" * 64,
            "audit_plan_sha256": AUDIT50_PLAN_SHA256,
            "spec_sha256": spot.canonical_sha256(spec),
            "resume_commit_sha256": "c" * 64,
            "files": {name: "d" * 64 for name in spot._COMMIT_FILES},
            "teacher_values_are_realized_match_ev": False,
            "selector_executed": False,
            "fit_performed": False,
            "threshold_selected": False,
            "current_profile_mutated": False,
            "runtime_policy_activated": False,
        }
        write_canonical_json(done_root / f"DONE-{shard:03d}.json", payload)
    return run, run / "launch_authorization.json", done_root


def test_done_only_claim_requires_exact_50_without_opening_teacher(
    tmp_path: Path,
) -> None:
    run, launch, done_root = _done_run(tmp_path)
    assert len(
        spot.validate_done_set(
            run_dir=run, authorization_path=launch, done_root=done_root
        )
    ) == TOTAL_SHARDS
    output = tmp_path / "CONSUMED.json"
    claim = spot.claim_complete_output(
        run_dir=run,
        authorization_path=launch,
        done_root=done_root,
        output=output,
    )
    assert claim["all_done_markers_verified"] is True
    assert claim["result_objects_addressed_when_claimed"] is False
    assert claim["remote_claim_required_before_content_read"] is True
    assert not (tmp_path / "teacher.jsonl").exists()


def test_done_set_gap_and_done_identity_tamper_fail_closed(tmp_path: Path) -> None:
    run, launch, done_root = _done_run(tmp_path)
    (done_root / "DONE-049.json").unlink()
    with pytest.raises(ValueError, match="exactly 50"):
        spot.validate_done_set(
            run_dir=run, authorization_path=launch, done_root=done_root
        )


def test_frozen_future_runner_hash_is_unchanged() -> None:
    if not DEV_RUN.is_dir():
        pytest.skip("frozen development package is not present")
    frozen = (
        DEV_RUN
        / "package_src/src/ofc_regular/run_hu_m43_attempt08_future_audit.py"
    )
    workspace = ROOT / "src/ofc_regular/run_hu_m43_attempt08_future_audit.py"
    assert sha256_file(frozen) == DEVELOPMENT_FUTURE_RUNNER_SHA256
    assert frozen.read_bytes() == workspace.read_bytes()


def test_direct_selector_cli_is_disabled() -> None:
    from ofc_regular import select_hu_m43_attempt08_audit50 as selector

    with pytest.raises(SystemExit, match="claimed receive lifecycle"):
        selector.main()


def _authorization_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, Path, dict[str, object]]:
    run = tmp_path / "audit-run"
    development = tmp_path / "development-run"
    run.mkdir()
    development.mkdir()
    manifest = {
        "run_name": "audit50-test",
        "development_run_name": "development-test",
        "development_manifest_sha256": "1" * 64,
        "development_launch_authorization_sha256": "2" * 64,
        "development_source_closure_sha256": "3" * 64,
        "development_source_zip_sha256": "4" * 64,
        "development_package_tree_sha256": "5" * 64,
        "runtime_semantic_anchor_sha256": "6" * 64,
        "runtime_source_closure_sha256": "7" * 64,
        "runtime_fingerprint_sha256": "8" * 64,
        "runtime_requirements_sha256": "9" * 64,
        "schedule_sha256": "a" * 64,
        "overlay_source_closure_sha256": "b" * 64,
        "overlay_source_zip_sha256": "c" * 64,
        "startup_sha256": "d" * 64,
    }
    write_canonical_json(run / "manifest.json", {"test": True})
    write_canonical_json(
        development / "manifest.json",
        {"development_open_authorization_sha256": "e" * 64},
    )
    decision = tmp_path / "decision.json"
    receipt = tmp_path / "receipt.json"
    freeze = tmp_path / "freeze.json"
    write_canonical_json(
        decision,
        {
            "decision": "go",
            "status": "go_write_separate_search_freeze_only",
            "fit_performed": False,
            "threshold_selected": False,
            "current_profile_mutated": False,
            "runtime_policy_activated": False,
        },
    )
    write_canonical_json(
        receipt,
        {
            "decision": "go",
            "status": "single_frozen_gate_evaluation_complete",
            "selector_executed": True,
            "merge_receipt_sha256": "f" * 64,
            "merged_sha256": "0" * 64,
            "selector_claim_sha256": "1" * 64,
            "remote_selector_claim_sha256": "2" * 64,
            "fit_performed": False,
            "threshold_selected": False,
            "current_profile_mutated": False,
            "runtime_policy_activated": False,
        },
    )
    write_canonical_json(
        freeze,
        {
            "schema": "hu_m43_attempt08_development_pass_freeze_v1",
            "status": "development_go_frozen_without_future_audit_authorization",
            "run_name": "development-test",
            "package_manifest_sha256": "1" * 64,
            "launch_authorization_sha256": "2" * 64,
            "development_decision_sha256": sha256_file(decision),
            "selector_receipt_sha256": sha256_file(receipt),
            "plan_sha256": spot.SOURCE_PLAN_SHA256,
            "model_sha256": spot.LAMBDA_MODEL_SHA256,
            "ai_profiles_sha256": spot.AI_PROFILES_SHA256,
            "search_freeze_authorized": True,
            "future_audit_authorized": False,
            "fit_performed": False,
            "threshold_selected": False,
            "current_profile_mutated": False,
            "runtime_policy_activated": False,
        },
    )
    return run, development, freeze, decision, receipt, manifest


def test_authorize_audit_builds_outer_envelope_around_compatible_core(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, development, freeze, decision, receipt, manifest = _authorization_fixture(
        tmp_path
    )
    monkeypatch.setattr(spot, "validate_package", lambda *a, **k: manifest)
    monkeypatch.setattr(spot, "_validate_core_with_frozen_runner", lambda **k: None)
    outer = spot.authorize_audit50(
        run_dir=run,
        development_run_dir=development,
        development_pass_freeze_path=freeze,
        development_decision_path=decision,
        selector_receipt_path=receipt,
        core_output=run / "core_audit_authorization.json",
        outer_output=run / "audit_open_authorization.json",
    )
    assert outer["audit_plan_sha256"] == AUDIT50_PLAN_SHA256
    assert outer["audit_authorized"] is True
    assert outer["audit_started"] is False
    assert outer["fit_performed"] is False
    assert outer["current_profile_mutated"] is False
    core = spot.load_json_mapping(
        run / "core_audit_authorization.json", "core"
    )
    assert core["schema"] == spot.CORE_AUTH_SCHEMA
    assert core["run_name"] == "development-test"
    launch_path = run / "launch_authorization.json"
    launch = spot.authorize_launch(
        run_dir=run,
        development_run_dir=development,
        output=launch_path,
    )
    assert launch["spot_authorized"] is True
    assert launch["audit_started"] is False
    assert (
        spot.validate_launch(
            run_dir=run,
            development_run_dir=development,
            authorization_path=launch_path,
        )
        == launch
    )


def test_no_go_cannot_leave_authorization_inputs_in_audit_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, development, freeze, decision, receipt, manifest = _authorization_fixture(
        tmp_path
    )
    payload = spot.load_json_mapping(decision, "decision")
    payload["decision"] = "no_go"
    decision.write_bytes(canonical_json_bytes(payload))
    monkeypatch.setattr(spot, "validate_package", lambda *a, **k: manifest)
    with pytest.raises(ValueError, match="exact development GO chain"):
        spot.authorize_audit50(
            run_dir=run,
            development_run_dir=development,
            development_pass_freeze_path=freeze,
            development_decision_path=decision,
            selector_receipt_path=receipt,
            core_output=run / "core_audit_authorization.json",
            outer_output=run / "audit_open_authorization.json",
        )
    assert not (run / "development_pass_freeze.json").exists()
    assert not (run / "core_audit_authorization.json").exists()


def _shard_material_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path, dict[str, object], dict[str, object]]:
    run = tmp_path / "run"
    development = tmp_path / "development"
    output = tmp_path / "result"
    run.mkdir()
    development.mkdir()
    output.mkdir()
    write_canonical_json(run / "manifest.json", {"run_name": "audit50-test"})
    write_canonical_json(run / "launch_authorization.json", {"test": "launch"})
    write_canonical_json(run / "audit_open_authorization.json", {"test": "audit"})
    write_canonical_json(run / "core_audit_authorization.json", {"test": "core"})
    spec: dict[str, object] = {
        "root_index": 200,
        "root_profile": "stage19_p0",
    }
    launch: dict[str, object] = {
        "run_name": "audit50-test",
        "manifest_sha256": sha256_file(run / "manifest.json"),
        "audit_open_authorization_sha256": sha256_file(
            run / "audit_open_authorization.json"
        ),
        "core_authorization_sha256": sha256_file(
            run / "core_audit_authorization.json"
        ),
    }
    root_claim = {"spec_sha256": spot.canonical_sha256(spec)}
    monkeypatch.setattr(spot, "validate_launch", lambda *a, **k: launch)
    monkeypatch.setattr(spot, "_spec", lambda *a, **k: spec)
    monkeypatch.setattr(
        spot,
        "_validate_claims_before_root",
        lambda *a, **k: ({"claim": "global"}, root_claim),
    )
    monkeypatch.setattr(
        spot, "_validate_teacher_row_semantics", lambda *a, **k: {"valid": True}
    )
    write_canonical_json(output / "teacher.jsonl", {"root_index": 200})
    teacher_sha = sha256_file(output / "teacher.jsonl")
    write_canonical_json(output / "global_claim.json", {"claim": "global"})
    write_canonical_json(output / "root_claim.json", root_claim)
    runner = {
        "schema": "hu_m43_attempt08_future_audit_summary_v1",
        "status": "complete",
        "population": "future_audit",
        "root_index": 200,
        "output_sha256": teacher_sha,
        "generator_elapsed_seconds": 1.5,
        "generator_peak_rss_bytes": 1024,
        "teacher_values_are_realized_match_ev": False,
        "fit_performed": False,
        "threshold_selected": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    checkpoint = {
        "schema": spot.CHECKPOINT_SCHEMA,
        "status": "one_root_complete",
        "run_name": "audit50-test",
        "shard": 0,
        "root_index": 200,
        "completed_roots": 1,
        "target_roots": 1,
        "teacher_sha256": teacher_sha,
        "manifest_sha256": sha256_file(run / "manifest.json"),
        "launch_authorization_sha256": sha256_file(
            run / "launch_authorization.json"
        ),
        "audit_open_authorization_sha256": sha256_file(
            run / "audit_open_authorization.json"
        ),
        "deterministic_same_root_recompute_allowed": True,
        "alternate_root_seed_allowed": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    heartbeat = {
        "schema": spot.HEARTBEAT_SCHEMA,
        "status": "root_complete",
        "run_name": "audit50-test",
        "shard": 0,
        "root_index": 200,
        "completed_roots": 1,
        "target_roots": 1,
        "teacher_sha256": teacher_sha,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    summary = {
        "schema": spot.SUMMARY_SCHEMA,
        "status": "complete",
        "run_name": "audit50-test",
        "shard": 0,
        "root_index": 200,
        "root_profile": "stage19_p0",
        "teacher_sha256": teacher_sha,
        "runner_summary": runner,
        "elapsed_seconds": 2.0,
        "teacher_values_are_realized_match_ev": False,
        "fit_performed": False,
        "threshold_selected": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    write_canonical_json(output / "checkpoint.json", checkpoint)
    write_canonical_json(output / "heartbeat.json", heartbeat)
    write_canonical_json(output / "generator_summary.json", summary)
    (output / "run.log").write_text(
        json.dumps(runner, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "time.txt").write_text(
        "elapsed_seconds=2.000000000\n", encoding="ascii"
    )
    write_canonical_json(
        output / "boot_image_evidence.json",
        {
            "schema": "hu_m43_attempt08_audit50_boot_image_evidence_v1",
            "run_name": "audit50-test",
            "shard": 0,
            "instance_name": "audit50-worker",
            "disk_name": "audit50-disk",
            "source_image": spot.GCP_IMAGE_SELF_LINK,
            "source_image_id": spot.GCP_IMAGE_ID,
        },
    )
    files = {name: sha256_file(output / name) for name in spot._COMMIT_FILES}
    write_canonical_json(
        output / "resume_commit.json",
        spot._resume_commit_payload(
            run_dir=run,
            shard=0,
            launch=launch,
            files=files,
            root_claim_sha256=sha256_file(output / "root_claim.json"),
        ),
    )
    return run, development, output, launch, spec


@pytest.mark.parametrize(
    "artifact",
    [
        "teacher.jsonl",
        "checkpoint.json",
        "heartbeat.json",
        "generator_summary.json",
        "time.txt",
    ],
)
def test_restored_material_tamper_fails_before_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, artifact: str
) -> None:
    run, development, output, _, _ = _shard_material_fixture(tmp_path, monkeypatch)
    path = output / artifact
    if artifact == "time.txt":
        path.write_text("elapsed_seconds=9.000000000\n", encoding="ascii")
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        value["tampered"] = True
        path.write_bytes(canonical_json_bytes(value))
    with pytest.raises(ValueError):
        spot.complete_shard(
            run_dir=run,
            development_run_dir=development,
            shard=0,
            directory=output,
        )
    assert not (output / "DONE.json").exists()


def test_resume_commit_rejects_non_allowlisted_object_before_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, development, output, _, _ = _shard_material_fixture(tmp_path, monkeypatch)
    commit_path = output / "resume_commit.json"
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    del commit["files"]["teacher.jsonl"]
    commit["files"]["../teacher.jsonl"] = "a" * 64
    commit_path.write_bytes(canonical_json_bytes(commit))
    with pytest.raises(ValueError, match="resume commit changed"):
        spot.validate_resume_commit_file(
            run_dir=run,
            development_run_dir=development,
            shard=0,
            commit_path=commit_path,
        )


def _selector_binding_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, Path]]:
    audit_run = tmp_path / "audit-run"
    receive = tmp_path / "receive"
    selector_root = receive / "selector"
    merged_root = receive / "merged"
    selector_source = (
        audit_run / "overlay_src/src/ofc_regular/select_hu_m43_attempt08_audit50.py"
    )
    selector_source.parent.mkdir(parents=True)
    selector_source.write_text("# frozen selector\n", encoding="utf-8")
    selector_root.mkdir(parents=True)
    merged_root.mkdir(parents=True)
    write_canonical_json(audit_run / "manifest.json", {"run_name": "audit50-test"})
    merged = merged_root / "teacher.jsonl"
    merged.write_bytes(
        b"".join(
            canonical_json_bytes({"root_index": 200 + shard})
            for shard in range(TOTAL_SHARDS)
        )
    )
    merge_receipt = merged_root / "merge_receipt.json"
    write_canonical_json(
        merge_receipt,
        {
            "schema": spot.MERGE_SCHEMA,
            "status": "exact_50_roots_merged_without_selection",
            "roots": TOTAL_SHARDS,
            "root_index_first": 200,
            "root_index_last": 249,
            "merged_sha256": sha256_file(merged),
        },
    )
    claim = {
        "schema": spot.SELECTOR_CLAIM_SCHEMA,
        "status": "claimed_before_single_frozen_audit_gate_evaluation",
        "run_name": "audit50-test",
        "selector_source_sha256": sha256_file(selector_source),
        "gate_evaluation_count_before_claim": 0,
        "selector_executed": False,
        "core_authorization_sha256": "a" * 64,
        "audit_open_authorization_sha256": "b" * 64,
        "merged_sha256": sha256_file(merged),
        "merge_receipt_sha256": sha256_file(merge_receipt),
        "canonical_merged_path": str(merged.resolve()),
        "canonical_merge_receipt_path": str(merge_receipt.resolve()),
    }
    claim_path = selector_root / "CLAIM.json"
    remote_path = selector_root / "REMOTE_CLAIM.json"
    write_canonical_json(claim_path, claim)
    write_canonical_json(remote_path, claim)
    return audit_run, receive, {
        "merged": merged,
        "merge_receipt": merge_receipt,
        "claim": claim_path,
        "remote_claim": remote_path,
        "execution": selector_root / "EXECUTION_STARTED.json",
        "decision": selector_root / "decision.json",
        "receipt": selector_root / "decision_receipt.json",
    }


@pytest.mark.parametrize("artifact", ["merged", "merge_receipt"])
def test_selector_rehashes_claimed_inputs_before_execution(
    tmp_path: Path, artifact: str
) -> None:
    audit_run, _, paths = _selector_binding_fixture(tmp_path)
    path = paths[artifact]
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="input changed after claim"):
        spot.execute_selector_once(
            run_dir=audit_run,
            development_run_dir=tmp_path / "development",
            selector_claim_path=paths["claim"],
            remote_selector_claim_path=paths["remote_claim"],
        )
    assert not paths["execution"].exists()


def _write_selector_completion(paths: dict[str, Path]) -> None:
    claim_sha = sha256_file(paths["claim"])
    remote_sha = sha256_file(paths["remote_claim"])
    merged_sha = sha256_file(paths["merged"])
    merge_receipt_sha = sha256_file(paths["merge_receipt"])
    execution = {
        "schema": spot.SELECTOR_EXECUTION_SCHEMA,
        "status": "single_frozen_gate_evaluation_started",
        "selector_claim_sha256": claim_sha,
        "remote_selector_claim_sha256": remote_sha,
        "merged_sha256": merged_sha,
        "merge_receipt_sha256": merge_receipt_sha,
        "selector_executed": False,
    }
    write_canonical_json(paths["execution"], execution)
    decision = {
        "decision": "go",
        "source": {
            "input_jsonl_sha256": merged_sha,
            "merge_receipt_sha256": merge_receipt_sha,
        },
        "decision_contract": {
            "gate_evaluation_count": 1,
            "audit_rows_used_for_fit": False,
        },
    }
    write_canonical_json(paths["decision"], decision)
    write_canonical_json(
        paths["receipt"],
        {
            "schema": spot.SELECTOR_RECEIPT_SCHEMA,
            "status": "single_frozen_audit_gate_evaluation_complete",
            "selector_claim_sha256": claim_sha,
            "remote_selector_claim_sha256": remote_sha,
            "execution_marker_sha256": sha256_file(paths["execution"]),
            "merged_sha256": merged_sha,
            "merge_receipt_sha256": merge_receipt_sha,
            "decision_sha256": sha256_file(paths["decision"]),
            "decision": "go",
            "gate_evaluation_count": 1,
            "selector_executed": True,
            "audit_rows_used_for_fit": False,
            "fit_performed": False,
            "threshold_selected": False,
            "current_profile_mutated": False,
            "runtime_policy_activated": False,
        },
    )


def test_selector_completion_rehashes_merged_content(
    tmp_path: Path,
) -> None:
    _, receive, paths = _selector_binding_fixture(tmp_path)
    _write_selector_completion(paths)
    assert spot.validate_selector_completion(run_dir=receive)["decision"] == "go"
    paths["merged"].write_bytes(paths["merged"].read_bytes() + b" ")
    with pytest.raises(ValueError, match="input changed after claim"):
        spot.validate_selector_completion(run_dir=receive)


def test_selector_completion_binds_decision_source_to_claimed_merged_sha(
    tmp_path: Path,
) -> None:
    _, receive, paths = _selector_binding_fixture(tmp_path)
    _write_selector_completion(paths)
    decision = json.loads(paths["decision"].read_text(encoding="utf-8"))
    decision["source"]["input_jsonl_sha256"] = "f" * 64
    paths["decision"].write_bytes(canonical_json_bytes(decision))
    receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
    receipt["decision_sha256"] = sha256_file(paths["decision"])
    paths["receipt"].write_bytes(canonical_json_bytes(receipt))
    with pytest.raises(ValueError, match="completed selector lifecycle changed"):
        spot.validate_selector_completion(run_dir=receive)
