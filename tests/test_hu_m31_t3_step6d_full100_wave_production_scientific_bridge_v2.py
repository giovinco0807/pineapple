from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_full100_wave_production_scientific_bridge_v2 as subject,
)


RUN_NAME = "regular-hu-m31-c02-full100-production-science-test"
PROFILE_BYTES = b"# pinned profile fixture\n"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


@pytest.fixture
def local_paths(tmp_path: Path) -> subject.BridgePathsV2:
    run = (tmp_path / "run").resolve()
    accepted = run / "accepted-results"
    for directory in (
        accepted,
        run / "receiver",
        run / "control",
        run / "cleanup",
        run / "phase-a",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    (accepted / "immutable.json").write_bytes(b"immutable\n")
    plan = run / "phase-a" / "wave_plan.json"
    plan.write_bytes(b"{}\n")
    profile = (tmp_path / "repo" / "src" / "ofc_regular" / "ai_profiles.py").resolve()
    profile.parent.mkdir(parents=True)
    profile.write_bytes(PROFILE_BYTES)
    return subject.BridgePathsV2(
        run_root=run,
        wave_plan=plan,
        accepted_root=accepted,
        merge_view_root=run / "scientific-merge-view",
        receipt_output=run / "scientific-gate-receipt.json",
        profile_path=profile,
        receiver_root=run / "receiver",
        control_root=run / "control",
        cleanup_root=run / "cleanup",
    )


def _execution(paths: subject.BridgePathsV2) -> subject.ExecutionEvidenceV2:
    namespace = "execution-000-aaaaaaaaaaaa"
    receipt_path = paths.receiver_root / namespace / "receiver_receipt.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_bytes(b"{}\n")
    journal = paths.control_root / namespace / "controller-journal"
    journal.mkdir(parents=True, exist_ok=True)
    return subject.ExecutionEvidenceV2(
        ordinal=0,
        namespace=namespace,
        pre_attempt_ledger={"ledger_sha256": "a" * 64},
        resume_plan={"resume_sha256": "b" * 64},
        post_attempt_ledger={"ledger_sha256": "c" * 64},
        request={
            "content_binding": {
                "immutable_content_prefix": "content/test",
                "content_payload_sha256": "d" * 64,
                "outer_manifest_sha256": "e" * 64,
            }
        },
        receiver_receipt={
            "status": "all_jobs_complete_exact_inventory_accepted"
        },
        receiver_receipt_path=receipt_path,
        controller_journal_dir=journal,
    )


def _inputs(paths: subject.BridgePathsV2) -> subject.ProductionBridgeInputsV2:
    execution = _execution(paths)
    return subject.ProductionBridgeInputsV2(
        paths=paths,
        wave_plan={
            "run_name": RUN_NAME,
            "execution_identity_sha256": "f" * 64,
            "schedule_sha256": "1" * 64,
        },
        final_attempt_ledger={"ledger_sha256": "c" * 64},
        executions=(execution,),
        final_execution=execution,
        expected_startup_sha256="2" * 64,
        content_payload_sha256="d" * 64,
        outer_manifest_sha256="e" * 64,
    )


def test_validate_bridge_paths_requires_explicit_disjoint_write_once_paths(
    local_paths: subject.BridgePathsV2,
) -> None:
    checked = subject.validate_bridge_paths(
        run_root=local_paths.run_root,
        wave_plan_path=local_paths.wave_plan,
        accepted_root=local_paths.accepted_root,
        merge_view_root=local_paths.merge_view_root,
        receipt_output_path=local_paths.receipt_output,
        profile_path=local_paths.profile_path,
    )
    assert checked == local_paths

    with pytest.raises(ValueError, match="absolute"):
        subject.validate_bridge_paths(
            run_root="relative-run",
            wave_plan_path=local_paths.wave_plan,
            accepted_root=local_paths.accepted_root,
            merge_view_root=local_paths.merge_view_root,
            receipt_output_path=local_paths.receipt_output,
            profile_path=local_paths.profile_path,
        )
    with pytest.raises(ValueError, match="overlaps immutable"):
        subject.validate_bridge_paths(
            run_root=local_paths.run_root,
            wave_plan_path=local_paths.wave_plan,
            accepted_root=local_paths.accepted_root,
            merge_view_root=local_paths.accepted_root / "view",
            receipt_output_path=local_paths.receipt_output,
            profile_path=local_paths.profile_path,
        )
    local_paths.receipt_output.write_bytes(b"existing\n")
    with pytest.raises(FileExistsError, match="write-once"):
        subject.validate_bridge_paths(
            run_root=local_paths.run_root,
            wave_plan_path=local_paths.wave_plan,
            accepted_root=local_paths.accepted_root,
            merge_view_root=local_paths.merge_view_root,
            receipt_output_path=local_paths.receipt_output,
            profile_path=local_paths.profile_path,
        )


def test_namespace_inventory_rejects_any_unexpected_top_level_entry(
    tmp_path: Path,
) -> None:
    root = (tmp_path / "receiver").resolve()
    (root / "execution-000-aaaaaaaaaaaa").mkdir(parents=True)
    assert list(subject._namespace_dirs(root, "receiver")) == [
        "execution-000-aaaaaaaaaaaa"
    ]
    (root / "unexpected.txt").write_bytes(b"x")
    with pytest.raises(ValueError, match="unexpected top-level"):
        subject._namespace_dirs(root, "receiver")


def test_production_receive_receipt_is_bound_to_exact_execution() -> None:
    pre = {"ledger_sha256": "a" * 64, "transitions": [{}]}
    resume = {"resume_sha256": "b" * 64}
    post = {"ledger_sha256": "c" * 64}
    receiver = {
        "status": "wave_results_recorded_next_wave_ready",
        "receipt_sha256": "d" * 64,
        "next_resume_plan": {"resume_sha256": "e" * 64},
        "accepted_job_ids": ["candidate-shard-00", "reference-shard-00"],
        "failed_job_ids": [],
    }
    body = {
        "schema": subject.production_v2.PRODUCTION_RECEIVE_SCHEMA,
        "status": receiver["status"],
        "run_name": RUN_NAME,
        "wave_index": 0,
        "execution_ordinal": 0,
        "input_attempt_ledger_sha256": pre["ledger_sha256"],
        "input_resume_plan_sha256": resume["resume_sha256"],
        "execution_namespace": "execution-000-aaaaaaaaaaaa",
        "poll_receipt_sha256": "f" * 64,
        "receiver_receipt_sha256": receiver["receipt_sha256"],
        "next_attempt_ledger_sha256": post["ledger_sha256"],
        "next_resume_plan_sha256": "e" * 64,
        "accepted_job_ids": receiver["accepted_job_ids"],
        "failed_job_ids": [],
        "lifecycle_proof_replayed_inside_one_shot_adapter": True,
        "terminal_observations_derived_from_pinned_gcs_and_closeout": True,
        "acceptance_create_only": True,
        "vm_lifecycle_mutation_performed": False,
        "current_profile_changed": False,
    }
    value = {
        **body,
        "receipt_sha256": subject.wave_v2.canonical_sha256(body),
    }
    execution = {
        "pre_attempt_ledger": pre,
        "resume_plan": resume,
        "post_attempt_ledger": post,
        "wave_index": 0,
    }
    assert subject._validate_production_receipt(
        value,
        plan={"run_name": RUN_NAME},
        execution=execution,
        namespace="execution-000-aaaaaaaaaaaa",
        receiver_receipt=receiver,
    ) == value
    forged = dict(value)
    forged["execution_namespace"] = "execution-001-bbbbbbbbbbbb"
    forged["receipt_sha256"] = subject.wave_v2.canonical_sha256(
        {key: item for key, item in forged.items() if key != "receipt_sha256"}
    )
    with pytest.raises(ValueError, match="execution binding"):
        subject._validate_production_receipt(
            forged,
            plan={"run_name": RUN_NAME},
            execution=execution,
            namespace="execution-000-aaaaaaaaaaaa",
            receiver_receipt=receiver,
        )


def test_preflight_replays_lifecycle_and_exact_440_object_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    local_paths: subject.BridgePathsV2,
) -> None:
    inputs = _inputs(local_paths)
    lifecycle = {"chain_sha256": "3" * 64}
    snapshot = {
        "snapshot_sha256": "4" * 64,
        "accepted_job_count": 20,
        "accepted_object_count": 440,
    }

    class Lifecycle:
        def load_validated_lifecycle_chain(self, **_: Any) -> dict[str, Any]:
            return lifecycle

    class Accepted:
        def load_accepted_results(self, **_: Any) -> dict[str, Any]:
            return snapshot

    monkeypatch.setattr(
        subject, "build_production_adapters", lambda _inputs: (Accepted(), Lifecycle())
    )
    monkeypatch.setattr(
        subject.bridge_v2,
        "validate_validated_lifecycle_chain",
        lambda **kwargs: kwargs["value"],
    )
    monkeypatch.setattr(
        subject.bridge_v2,
        "validate_accepted_results_snapshot",
        lambda **kwargs: SimpleNamespace(snapshot=kwargs["value"]),
    )
    receipt = subject.preflight_production_bridge(
        inputs,
        profile_sha256=_sha(PROFILE_BYTES),
        accepted_tree_snapshot={"file_count": 440, "tree_sha256": "5" * 64},
    )
    assert receipt["status"] == "ready_for_write_once_scientific_merge"
    assert receipt["accepted_object_count"] == 440
    assert receipt["cloud_network_authorized"] is False
    assert receipt["current_profile_changed"] is False


def test_run_preflight_pins_profile_and_accepted_tree_without_outputs(
    monkeypatch: pytest.MonkeyPatch,
    local_paths: subject.BridgePathsV2,
) -> None:
    inputs = _inputs(local_paths)
    monkeypatch.setattr(subject, "validate_bridge_paths", lambda **_: local_paths)
    monkeypatch.setattr(
        subject,
        "prepare_production_bridge_inputs",
        lambda **_: inputs,
    )
    expected = {
        "schema": subject.PREFLIGHT_SCHEMA,
        "status": "ready_for_write_once_scientific_merge",
        "run_name": RUN_NAME,
        "preflight_sha256": "6" * 64,
    }
    monkeypatch.setattr(
        subject,
        "preflight_production_bridge",
        lambda *_args, **_kwargs: expected,
    )
    result = subject.run_production_scientific_bridge(
        mode="preflight",
        run_root=local_paths.run_root,
        wave_plan_path=local_paths.wave_plan,
        accepted_root=local_paths.accepted_root,
        merge_view_root=local_paths.merge_view_root,
        receipt_output_path=local_paths.receipt_output,
        profile_path=local_paths.profile_path,
        expected_profile_sha256=_sha(PROFILE_BYTES),
        expected_run_name=RUN_NAME,
    )
    assert result == expected
    assert not local_paths.merge_view_root.exists()
    assert not local_paths.receipt_output.exists()
    assert local_paths.profile_path.read_bytes() == PROFILE_BYTES
    assert (local_paths.accepted_root / "immutable.json").read_bytes() == b"immutable\n"


def test_profile_pin_mismatch_stops_before_input_discovery(
    monkeypatch: pytest.MonkeyPatch,
    local_paths: subject.BridgePathsV2,
) -> None:
    monkeypatch.setattr(subject, "validate_bridge_paths", lambda **_: local_paths)
    discovered = False

    def discover(**_: Any) -> subject.ProductionBridgeInputsV2:
        nonlocal discovered
        discovered = True
        return _inputs(local_paths)

    monkeypatch.setattr(subject, "prepare_production_bridge_inputs", discover)
    with pytest.raises(PermissionError, match="explicit SHA-256 pin"):
        subject.run_production_scientific_bridge(
            mode="preflight",
            run_root=local_paths.run_root,
            wave_plan_path=local_paths.wave_plan,
            accepted_root=local_paths.accepted_root,
            merge_view_root=local_paths.merge_view_root,
            receipt_output_path=local_paths.receipt_output,
            profile_path=local_paths.profile_path,
            expected_profile_sha256="0" * 64,
            expected_run_name=RUN_NAME,
        )
    assert discovered is False


def test_run_merge_uses_fresh_adapters_and_replays_written_receipt(
    monkeypatch: pytest.MonkeyPatch,
    local_paths: subject.BridgePathsV2,
) -> None:
    inputs = _inputs(local_paths)
    monkeypatch.setattr(subject, "validate_bridge_paths", lambda **_: local_paths)
    monkeypatch.setattr(
        subject, "prepare_production_bridge_inputs", lambda **_: inputs
    )
    monkeypatch.setattr(
        subject,
        "preflight_production_bridge",
        lambda *_args, **_kwargs: {"status": "ready"},
    )
    adapter_calls: list[object] = []

    def adapters(_inputs: object) -> tuple[object, object]:
        pair = (object(), object())
        adapter_calls.append(pair)
        return pair

    monkeypatch.setattr(subject, "build_production_adapters", adapters)
    scientific = {
        "status": "pass",
        "decision": "full100_wave_v2_go_open_one_shot_performance_lock_only",
        "all_gates_passed": True,
        "run_name": RUN_NAME,
        "receipt_sha256": "7" * 64,
    }

    def merge(**kwargs: Any) -> dict[str, Any]:
        assert kwargs["accepted_results_adapter"] is adapter_calls[0][0]
        assert kwargs["lifecycle_chain_adapter"] is adapter_calls[0][1]
        assert kwargs["merge_view_root"] == local_paths.merge_view_root
        assert kwargs["receipt_output_path"] == local_paths.receipt_output
        local_paths.merge_view_root.mkdir()
        (local_paths.merge_view_root / "MERGE_VIEW.json").write_bytes(b"{}\n")
        local_paths.receipt_output.write_bytes(_canonical(scientific))
        return scientific

    monkeypatch.setattr(
        subject.bridge_v2, "merge_and_write_scientific_gate_receipt", merge
    )
    monkeypatch.setattr(
        subject.bridge_v2,
        "validate_scientific_gate_receipt",
        lambda **_: scientific,
    )
    result = subject.run_production_scientific_bridge(
        mode="merge",
        run_root=local_paths.run_root,
        wave_plan_path=local_paths.wave_plan,
        accepted_root=local_paths.accepted_root,
        merge_view_root=local_paths.merge_view_root,
        receipt_output_path=local_paths.receipt_output,
        profile_path=local_paths.profile_path,
        expected_profile_sha256=_sha(PROFILE_BYTES),
        expected_run_name=RUN_NAME,
    )
    assert result == scientific
    assert len(adapter_calls) == 1
    assert local_paths.profile_path.read_bytes() == PROFILE_BYTES


def test_profile_and_accepted_tree_post_drift_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    local_paths: subject.BridgePathsV2,
) -> None:
    inputs = _inputs(local_paths)
    monkeypatch.setattr(subject, "validate_bridge_paths", lambda **_: local_paths)
    monkeypatch.setattr(
        subject, "prepare_production_bridge_inputs", lambda **_: inputs
    )

    def mutate_profile(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        local_paths.profile_path.write_bytes(b"changed\n")
        return {"status": "ready"}

    monkeypatch.setattr(subject, "preflight_production_bridge", mutate_profile)
    with pytest.raises(RuntimeError, match="profile source changed"):
        subject.run_production_scientific_bridge(
            mode="preflight",
            run_root=local_paths.run_root,
            wave_plan_path=local_paths.wave_plan,
            accepted_root=local_paths.accepted_root,
            merge_view_root=local_paths.merge_view_root,
            receipt_output_path=local_paths.receipt_output,
            profile_path=local_paths.profile_path,
            expected_profile_sha256=_sha(PROFILE_BYTES),
            expected_run_name=RUN_NAME,
        )

    local_paths.profile_path.write_bytes(PROFILE_BYTES)

    def mutate_tree(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        (local_paths.accepted_root / "extra.json").write_bytes(b"extra\n")
        return {"status": "ready"}

    monkeypatch.setattr(subject, "preflight_production_bridge", mutate_tree)
    with pytest.raises(RuntimeError, match="accepted-results tree changed"):
        subject.run_production_scientific_bridge(
            mode="preflight",
            run_root=local_paths.run_root,
            wave_plan_path=local_paths.wave_plan,
            accepted_root=local_paths.accepted_root,
            merge_view_root=local_paths.merge_view_root,
            receipt_output_path=local_paths.receipt_output,
            profile_path=local_paths.profile_path,
            expected_profile_sha256=_sha(PROFILE_BYTES),
            expected_run_name=RUN_NAME,
        )


def test_main_returns_one_for_valid_scientific_no_go(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        subject,
        "run_production_scientific_bridge",
        lambda **_: {
            "status": "no_go",
            "run_name": RUN_NAME,
            "decision": "full100_wave_v2_no_go_performance_lock_closed",
            "all_gates_passed": False,
            "receipt_sha256": "8" * 64,
        },
    )
    absolute = tmp_path.resolve()
    code = subject.main(
        [
            "--mode",
            "merge",
            "--run-root",
            str(absolute),
            "--wave-plan",
            str(absolute / "plan.json"),
            "--accepted-root",
            str(absolute / "accepted"),
            "--merge-view-root",
            str(absolute / "view"),
            "--receipt-output",
            str(absolute / "receipt.json"),
            "--profile-path",
            str(absolute / "profile.py"),
            "--expected-profile-sha256",
            "9" * 64,
            "--expected-run-name",
            RUN_NAME,
        ]
    )
    assert code == 1
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "no_go"
    assert output["all_gates_passed"] is False


def test_thin_script_only_delegates_to_production_main() -> None:
    script = Path(
        "scripts/run_hu_m31_t3_step6d_full100_wave_production_scientific_bridge_v2.py"
    ).read_text(encoding="utf-8")
    assert "production_scientific_bridge_v2 import" in script
    assert "raise SystemExit(main())" in script
