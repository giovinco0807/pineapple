"""Production-local composition for the full100 wave-v2 scientific bridge.

This command is intentionally cloud-incapable.  It discovers every receiver
execution from the final attempt ledger, replays the controller-owned lifecycle
proofs from their immutable journals, validates the exact 440-object accepted
tree, and only then delegates to the write-once scientific bridge.

``preflight`` performs all transport and lifecycle validation without creating
a merge view.  ``merge`` repeats that preflight with fresh one-shot adapters,
then creates only the explicitly supplied merge-view and gate-receipt paths.
The explicitly supplied profile and accepted-results tree are hashed before and
after either operation and must remain byte-identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import hu_m31_t3_step6d_full100_wave_controller_v2 as controller_v2
from . import hu_m31_t3_step6d_full100_wave_plan_v2 as wave_v2
from . import hu_m31_t3_step6d_full100_wave_production_receiver_v2 as production_v2
from . import hu_m31_t3_step6d_full100_wave_result_receiver_v2 as receiver_v2
from . import hu_m31_t3_step6d_full100_wave_scientific_bridge_v2 as bridge_v2
from . import hu_m31_t3_step6d_full100_wave_science_registry_v2 as science_registry


PREFLIGHT_SCHEMA = (
    "hu_m31_t3_step6d_full100_wave_production_scientific_preflight_v2"
)
CLI_RESULT_SCHEMA = (
    "hu_m31_t3_step6d_full100_wave_production_scientific_cli_result_v2"
)
_EXECUTION_NAMESPACE = re.compile(r"^execution-[0-9]{3}-[0-9a-f]{12}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PRODUCTION_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "wave_index",
        "execution_ordinal",
        "input_attempt_ledger_sha256",
        "input_resume_plan_sha256",
        "execution_namespace",
        "poll_receipt_sha256",
        "receiver_receipt_sha256",
        "next_attempt_ledger_sha256",
        "next_resume_plan_sha256",
        "accepted_job_ids",
        "failed_job_ids",
        "lifecycle_proof_replayed_inside_one_shot_adapter",
        "terminal_observations_derived_from_pinned_gcs_and_closeout",
        "acceptance_create_only",
        "vm_lifecycle_mutation_performed",
        "current_profile_changed",
        "receipt_sha256",
    }
)


@dataclass(frozen=True)
class BridgePathsV2:
    """Resolved local paths authorized for one bridge invocation."""

    run_root: Path
    wave_plan: Path
    accepted_root: Path
    merge_view_root: Path
    receipt_output: Path
    profile_path: Path
    receiver_root: Path
    control_root: Path
    cleanup_root: Path


@dataclass(frozen=True)
class ExecutionEvidenceV2:
    """One exact execution transition and its immutable producer evidence."""

    ordinal: int
    namespace: str
    pre_attempt_ledger: dict[str, Any]
    resume_plan: dict[str, Any]
    post_attempt_ledger: dict[str, Any]
    request: dict[str, Any]
    receiver_receipt: dict[str, Any]
    receiver_receipt_path: Path
    controller_journal_dir: Path


@dataclass(frozen=True)
class ProductionBridgeInputsV2:
    """Complete validated local input set for preflight or merge."""

    paths: BridgePathsV2
    wave_plan: dict[str, Any]
    final_attempt_ledger: dict[str, Any]
    executions: tuple[ExecutionEvidenceV2, ...]
    final_execution: ExecutionEvidenceV2
    expected_startup_sha256: str
    content_payload_sha256: str
    outer_manifest_sha256: str


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required plain file is missing or unsafe: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_read(path: Path, label: str) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} is not an absolute plain file")
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical JSON") from exc
    if not isinstance(value, dict) or raw != bridge_v2.canonical_bytes(value):
        raise ValueError(f"{label} is not canonical LF JSON")
    return value


def _plain_existing_dir(path: Path, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} must be an absolute non-symlink directory")
    return path.resolve()


def _plain_existing_file(path: Path, label: str) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be an absolute non-symlink file")
    return path.resolve()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _reject_symlink_ancestors(path: Path, *, stop: Path, label: str) -> None:
    cursor = path
    while cursor != stop:
        if cursor.exists() and cursor.is_symlink():
            raise ValueError(f"{label} contains a symlink ancestor")
        parent = cursor.parent
        if parent == cursor:
            raise ValueError(f"{label} escapes the run root")
        cursor = parent


def validate_bridge_paths(
    *,
    run_root: str | Path,
    wave_plan_path: str | Path,
    accepted_root: str | Path,
    merge_view_root: str | Path,
    receipt_output_path: str | Path,
    profile_path: str | Path,
) -> BridgePathsV2:
    """Resolve the exact local read/write boundary and reject path aliasing."""

    raw_run = Path(run_root)
    raw_plan = Path(wave_plan_path)
    raw_accepted = Path(accepted_root)
    raw_view = Path(merge_view_root)
    raw_receipt = Path(receipt_output_path)
    raw_profile = Path(profile_path)
    if any(
        not value.is_absolute()
        for value in (
            raw_run,
            raw_plan,
            raw_accepted,
            raw_view,
            raw_receipt,
            raw_profile,
        )
    ):
        raise ValueError("all production scientific bridge paths must be absolute")

    root = _plain_existing_dir(raw_run, "run root")
    plan = _plain_existing_file(raw_plan, "wave plan")
    accepted = _plain_existing_dir(raw_accepted, "accepted-results root")
    profile = _plain_existing_file(raw_profile, "profile source")
    view = raw_view.resolve()
    receipt = raw_receipt.resolve()
    if not _inside(plan, root) or not _inside(accepted, root):
        raise ValueError("wave plan and accepted-results root must be inside run root")
    if not _inside(view, root) or not _inside(receipt, root):
        raise ValueError("scientific outputs must remain inside the explicit run root")
    if view == receipt or view in receipt.parents or receipt in view.parents:
        raise ValueError("merge-view and receipt output paths must be disjoint")
    receiver_root = root / "receiver"
    control_root = root / "control"
    cleanup_root = root / "cleanup"
    for required, label in (
        (receiver_root, "receiver root"),
        (control_root, "controller root"),
        (cleanup_root, "cleanup root"),
    ):
        _plain_existing_dir(required, label)
    protected = (
        accepted,
        receiver_root.resolve(),
        control_root.resolve(),
        cleanup_root.resolve(),
        plan,
    )
    for output, label in ((view, "merge-view"), (receipt, "receipt output")):
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"{label} is write-once and already exists")
        _reject_symlink_ancestors(output.parent, stop=root, label=label)
        if any(output == item or item in output.parents or output in item.parents for item in protected):
            raise ValueError(f"{label} overlaps immutable bridge input")
    return BridgePathsV2(
        run_root=root,
        wave_plan=plan,
        accepted_root=accepted,
        merge_view_root=view,
        receipt_output=receipt,
        profile_path=profile,
        receiver_root=receiver_root.resolve(),
        control_root=control_root.resolve(),
        cleanup_root=cleanup_root.resolve(),
    )


def _tree_snapshot(root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        if item.is_symlink():
            raise ValueError("accepted-results tree contains a symlink")
        if item.is_dir():
            continue
        if not item.is_file():
            raise ValueError("accepted-results tree contains a non-file object")
        rows.append(
            {
                "path": item.relative_to(root).as_posix(),
                "bytes": item.stat().st_size,
                "sha256": _sha256_file(item),
            }
        )
    return {
        "file_count": len(rows),
        "tree_sha256": bridge_v2.canonical_sha256(rows),
    }


def _namespace_dirs(root: Path, label: str) -> dict[str, Path]:
    entries = list(root.iterdir())
    result: dict[str, Path] = {}
    for entry in entries:
        if (
            entry.is_symlink()
            or not entry.is_dir()
            or _EXECUTION_NAMESPACE.fullmatch(entry.name) is None
            or entry.name in result
        ):
            raise ValueError(f"{label} contains an unexpected top-level entry")
        result[entry.name] = entry.resolve()
    if not result:
        raise ValueError(f"{label} contains no execution evidence")
    return result


def _validate_production_receipt(
    value: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    execution: Mapping[str, Any],
    namespace: str,
    receiver_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = deepcopy(dict(value))
    if set(receipt) != _PRODUCTION_RECEIPT_KEYS:
        raise ValueError("production receive receipt fields changed")
    digest = receipt.pop("receipt_sha256", None)
    if digest != wave_v2.canonical_sha256(receipt):
        raise ValueError("production receive receipt digest changed")
    receipt["receipt_sha256"] = digest
    pre = execution["pre_attempt_ledger"]
    resume = execution["resume_plan"]
    post = execution["post_attempt_ledger"]
    next_resume = receiver_receipt.get("next_resume_plan")
    if (
        receipt.get("schema") != production_v2.PRODUCTION_RECEIVE_SCHEMA
        or receipt.get("status") != receiver_receipt.get("status")
        or receipt.get("run_name") != plan["run_name"]
        or receipt.get("wave_index") != execution["wave_index"]
        or receipt.get("execution_ordinal") != len(pre["transitions"]) - 1
        or receipt.get("input_attempt_ledger_sha256") != pre["ledger_sha256"]
        or receipt.get("input_resume_plan_sha256") != resume["resume_sha256"]
        or receipt.get("execution_namespace") != namespace
        or receipt.get("receiver_receipt_sha256")
        != receiver_receipt.get("receipt_sha256")
        or receipt.get("next_attempt_ledger_sha256") != post["ledger_sha256"]
        or receipt.get("next_resume_plan_sha256")
        != (None if next_resume is None else next_resume.get("resume_sha256"))
        or receipt.get("accepted_job_ids")
        != receiver_receipt.get("accepted_job_ids")
        or receipt.get("failed_job_ids") != receiver_receipt.get("failed_job_ids")
        or receipt.get("lifecycle_proof_replayed_inside_one_shot_adapter") is not True
        or receipt.get("terminal_observations_derived_from_pinned_gcs_and_closeout")
        is not True
        or receipt.get("acceptance_create_only") is not True
        or receipt.get("vm_lifecycle_mutation_performed") is not False
        or receipt.get("current_profile_changed") is not False
    ):
        raise ValueError("production receive receipt execution binding changed")
    return receipt


def _discover_execution_evidence(
    *, paths: BridgePathsV2, plan: Mapping[str, Any]
) -> tuple[tuple[ExecutionEvidenceV2, ...], dict[str, Any]]:
    receiver_dirs = _namespace_dirs(paths.receiver_root, "receiver root")
    raw_receipts: dict[str, tuple[dict[str, Any], Path]] = {}
    final_names: list[str] = []
    for namespace, directory in receiver_dirs.items():
        receipt_path = _plain_existing_file(
            directory / "receiver_receipt.json", "receiver receipt"
        )
        receipt = receiver_v2.validate_receiver_receipt(
            plan, _canonical_read(receipt_path, "receiver receipt")
        )
        raw_receipts[namespace] = (receipt, receipt_path)
        if receipt.get("status") == "all_jobs_complete_exact_inventory_accepted":
            final_names.append(namespace)
    if len(final_names) != 1:
        raise ValueError("exactly one final all-jobs-complete receiver receipt is required")
    final_receipt = raw_receipts[final_names[0]][0]
    final_ledger = wave_v2.validate_attempt_ledger(
        plan, final_receipt["attempt_ledger"]
    )
    next_resume = final_receipt.get("next_resume_plan")
    if (
        not isinstance(next_resume, Mapping)
        or next_resume.get("all_jobs_complete") is not True
        or Path(str(final_receipt.get("local_destination"))).resolve()
        != paths.accepted_root
    ):
        raise ValueError("final receiver receipt is not bound to accepted-results root")

    contexts = bridge_v2._execution_transition_contexts(plan, final_ledger)
    expected_names = [
        production_v2.execution_namespace(
            context["pre_attempt_ledger"], context["resume_plan"]
        )
        for context in contexts
    ]
    if len(expected_names) != len(set(expected_names)) or set(expected_names) != set(
        receiver_dirs
    ):
        raise ValueError("receiver execution namespace coverage differs from final ledger")
    control_dirs = _namespace_dirs(paths.control_root, "controller root")
    cleanup_dirs = _namespace_dirs(paths.cleanup_root, "cleanup root")
    if set(control_dirs) != set(expected_names) or set(cleanup_dirs) != set(
        expected_names
    ):
        raise ValueError("controller/cleanup execution coverage differs from final ledger")

    evidence: list[ExecutionEvidenceV2] = []
    content_bindings: set[tuple[str, str, str]] = set()
    for ordinal, (context, namespace) in enumerate(
        zip(contexts, expected_names, strict=True)
    ):
        control = control_dirs[namespace]
        cleanup = cleanup_dirs[namespace]
        input_ledger = wave_v2.validate_attempt_ledger(
            plan,
            _canonical_read(
                control / "inputs" / "attempt_ledger.json",
                "controller input attempt ledger",
            ),
        )
        resume = wave_v2.validate_resume_plan(
            plan,
            input_ledger,
            _canonical_read(
                control / "inputs" / "resume_plan.json",
                "controller input resume plan",
            ),
        )
        if (
            input_ledger != context["pre_attempt_ledger"]
            or resume != context["resume_plan"]
        ):
            raise ValueError("controller input differs from final ledger execution context")
        journal = _plain_existing_dir(
            control / "controller-journal", "controller journal"
        )
        request = production_v2.validate_production_receive_request(
            _canonical_read(cleanup / "receiver_request.json", "receiver request"),
            expected_execution_namespace=namespace,
            expected_controller_journal_dir=journal,
        )
        receipt, receipt_path = raw_receipts[namespace]
        if receipt.get("attempt_ledger") != context["post_attempt_ledger"]:
            raise ValueError("receiver post-ledger differs from final execution context")
        production_receipt = _canonical_read(
            receiver_dirs[namespace] / "production_receive_receipt.json",
            "production receive receipt",
        )
        _validate_production_receipt(
            production_receipt,
            plan=plan,
            execution=context,
            namespace=namespace,
            receiver_receipt=receipt,
        )
        content = request["content_binding"]
        content_bindings.add(
            (
                content["immutable_content_prefix"],
                content["content_payload_sha256"],
                content["outer_manifest_sha256"],
            )
        )
        evidence.append(
            ExecutionEvidenceV2(
                ordinal=ordinal,
                namespace=namespace,
                pre_attempt_ledger=input_ledger,
                resume_plan=resume,
                post_attempt_ledger=deepcopy(dict(context["post_attempt_ledger"])),
                request=request,
                receiver_receipt=receipt,
                receiver_receipt_path=receipt_path,
                controller_journal_dir=journal,
            )
        )
    if len(content_bindings) != 1:
        raise ValueError("immutable content binding differs across executions")
    return tuple(evidence), final_ledger


def prepare_production_bridge_inputs(
    *, paths: BridgePathsV2, expected_run_name: str
) -> ProductionBridgeInputsV2:
    """Validate and bind every local producer input without writing outputs."""

    if not isinstance(expected_run_name, str) or not expected_run_name:
        raise ValueError("expected run name is required")
    plan = wave_v2.validate_wave_plan(
        _canonical_read(paths.wave_plan, "wave plan")
    )
    if plan.get("run_name") != expected_run_name:
        raise PermissionError("explicit run-name confirmation differs from wave plan")
    executions, final_ledger = _discover_execution_evidence(paths=paths, plan=plan)
    if not executions:
        raise ValueError("final ledger contains no executed waves")
    final_execution = executions[-1]
    if (
        final_execution.receiver_receipt.get("status")
        != "all_jobs_complete_exact_inventory_accepted"
    ):
        raise ValueError("last execution is not the final accepted inventory")
    content = final_execution.request["content_binding"]
    return ProductionBridgeInputsV2(
        paths=paths,
        wave_plan=plan,
        final_attempt_ledger=final_ledger,
        executions=executions,
        final_execution=final_execution,
        expected_startup_sha256=science_registry.resolve_startup_sha256(plan),
        content_payload_sha256=content["content_payload_sha256"],
        outer_manifest_sha256=content["outer_manifest_sha256"],
    )


def _replay_callback(
    inputs: ProductionBridgeInputsV2,
    execution: ExecutionEvidenceV2,
) -> Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]:
    payload, launch_bundle, startup, validate_bundle = production_v2._validated_material(
        wave_plan=inputs.wave_plan,
        attempt_ledger=execution.pre_attempt_ledger,
        resume_plan=execution.resume_plan,
        request=execution.request,
    )
    controller = controller_v2.Full100WaveControllerV2(
        journal_dir=execution.controller_journal_dir,
        wave_plan=inputs.wave_plan,
        attempt_ledger=execution.pre_attempt_ledger,
        resume_plan=execution.resume_plan,
        create_journal=False,
    )
    proof_box: dict[str, dict[str, Any]] = {}
    adapter = production_v2._lifecycle_adapter(
        controller=controller,
        request=payload,
        launch_bundle=launch_bundle,
        startup=startup,
        validate_bundle=validate_bundle,
        proof_box=proof_box,
    )

    def replay(
        callback_plan: Mapping[str, Any], callback_ledger: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        if wave_v2.validate_wave_plan(callback_plan) != inputs.wave_plan:
            raise ValueError("lifecycle callback wave plan changed")
        if (
            wave_v2.validate_attempt_ledger(inputs.wave_plan, callback_ledger)
            != inputs.final_attempt_ledger
        ):
            raise ValueError("lifecycle callback final ledger changed")
        return adapter.validate(
            wave_plan=inputs.wave_plan,
            attempt_ledger=execution.pre_attempt_ledger,
            resume_plan=execution.resume_plan,
            receiver_observed_at_utc=payload["observed_at_utc"],
        )

    return replay


def build_production_adapters(
    inputs: ProductionBridgeInputsV2,
) -> tuple[
    bridge_v2.ReceiverReceiptAcceptedResultsAdapterV2,
    bridge_v2.ControllerReceiverLifecycleChainAdapterV2,
]:
    """Build fresh one-shot adapters for one preflight or merge replay."""

    callbacks = [_replay_callback(inputs, row) for row in inputs.executions]
    lifecycle = bridge_v2.ControllerReceiverLifecycleChainAdapterV2(
        controller_replay_callbacks=callbacks,
        receiver_receipts=[row.receiver_receipt for row in inputs.executions],
        receiver_validator=lambda plan, receipt: receiver_v2.validate_receiver_receipt(
            plan, receipt
        ),
    )
    final = inputs.final_execution
    accepted = bridge_v2.ReceiverReceiptAcceptedResultsAdapterV2(
        receiver_receipt=final.receiver_receipt,
        receiver_receipt_path=final.receiver_receipt_path,
        expected_startup_sha256=inputs.expected_startup_sha256,
        content_payload_sha256=inputs.content_payload_sha256,
        outer_manifest_sha256=inputs.outer_manifest_sha256,
    )
    return accepted, lifecycle


def preflight_production_bridge(
    inputs: ProductionBridgeInputsV2,
    *,
    profile_sha256: str,
    accepted_tree_snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay all lifecycle and accepted-result validation without writing."""

    accepted, lifecycle_adapter = build_production_adapters(inputs)
    lifecycle = bridge_v2.validate_validated_lifecycle_chain(
        wave_plan=inputs.wave_plan,
        attempt_ledger=inputs.final_attempt_ledger,
        value=lifecycle_adapter.load_validated_lifecycle_chain(
            wave_plan=inputs.wave_plan,
            attempt_ledger=inputs.final_attempt_ledger,
        ),
    )
    snapshot = accepted.load_accepted_results(
        wave_plan=inputs.wave_plan,
        attempt_ledger=inputs.final_attempt_ledger,
        validated_lifecycle_chain=lifecycle,
    )
    evidence = bridge_v2.validate_accepted_results_snapshot(
        wave_plan=inputs.wave_plan,
        attempt_ledger=inputs.final_attempt_ledger,
        validated_lifecycle_chain=lifecycle,
        value=snapshot,
    )
    body = {
        "schema": PREFLIGHT_SCHEMA,
        "status": "ready_for_write_once_scientific_merge",
        "run_name": inputs.wave_plan["run_name"],
        "execution_identity_sha256": inputs.wave_plan[
            "execution_identity_sha256"
        ],
        "wave_plan_sha256": inputs.wave_plan["schedule_sha256"],
        "final_attempt_ledger_sha256": inputs.final_attempt_ledger[
            "ledger_sha256"
        ],
        "execution_count": len(inputs.executions),
        "execution_namespaces": [row.namespace for row in inputs.executions],
        "accepted_snapshot_sha256": evidence.snapshot["snapshot_sha256"],
        "accepted_job_count": evidence.snapshot["accepted_job_count"],
        "accepted_object_count": evidence.snapshot["accepted_object_count"],
        "validated_lifecycle_chain_sha256": lifecycle["chain_sha256"],
        "profile_sha256": profile_sha256,
        "accepted_tree_file_count": accepted_tree_snapshot["file_count"],
        "accepted_tree_sha256": accepted_tree_snapshot["tree_sha256"],
        "merge_view_root": str(inputs.paths.merge_view_root),
        "receipt_output_path": str(inputs.paths.receipt_output),
        "outputs_absent": (
            not inputs.paths.merge_view_root.exists()
            and not inputs.paths.receipt_output.exists()
        ),
        "cloud_network_authorized": False,
        "vm_lifecycle_mutation_performed": False,
        "accepted_tree_modified": False,
        "current_profile_changed": False,
    }
    if (
        body["accepted_job_count"] != 20
        or body["accepted_object_count"] != 440
        or body["outputs_absent"] is not True
    ):
        raise ValueError("production scientific preflight coverage changed")
    return {**body, "preflight_sha256": bridge_v2.canonical_sha256(body)}


def run_production_scientific_bridge(
    *,
    mode: str,
    run_root: str | Path,
    wave_plan_path: str | Path,
    accepted_root: str | Path,
    merge_view_root: str | Path,
    receipt_output_path: str | Path,
    profile_path: str | Path,
    expected_profile_sha256: str,
    expected_run_name: str,
) -> dict[str, Any]:
    """Run a profile/accepted-tree-pinned preflight or write-once merge."""

    if mode not in {"preflight", "merge"}:
        raise ValueError("production scientific bridge mode changed")
    if (
        not isinstance(expected_profile_sha256, str)
        or _SHA256.fullmatch(expected_profile_sha256) is None
    ):
        raise ValueError("expected profile SHA-256 must be lowercase hex")
    paths = validate_bridge_paths(
        run_root=run_root,
        wave_plan_path=wave_plan_path,
        accepted_root=accepted_root,
        merge_view_root=merge_view_root,
        receipt_output_path=receipt_output_path,
        profile_path=profile_path,
    )
    profile_before = _sha256_file(paths.profile_path)
    if profile_before != expected_profile_sha256:
        raise PermissionError("profile source differs from explicit SHA-256 pin")
    tree_before = _tree_snapshot(paths.accepted_root)
    try:
        inputs = prepare_production_bridge_inputs(
            paths=paths, expected_run_name=expected_run_name
        )
        preflight = preflight_production_bridge(
            inputs,
            profile_sha256=profile_before,
            accepted_tree_snapshot=tree_before,
        )
        if mode == "preflight":
            return preflight

        # One-shot lifecycle adapters cannot be reused after preflight.  Build
        # a fresh set and repeat producer validation inside the actual bridge.
        accepted, lifecycle = build_production_adapters(inputs)
        receipt = bridge_v2.merge_and_write_scientific_gate_receipt(
            wave_plan=inputs.wave_plan,
            attempt_ledger=inputs.final_attempt_ledger,
            accepted_results_adapter=accepted,
            lifecycle_chain_adapter=lifecycle,
            merge_view_root=paths.merge_view_root,
            receipt_output_path=paths.receipt_output,
        )
        replayed = bridge_v2.validate_scientific_gate_receipt(
            receipt_path=paths.receipt_output
        )
        if replayed != receipt:
            raise ValueError("written scientific receipt differs from replay")
        return receipt
    finally:
        profile_after = _sha256_file(paths.profile_path)
        tree_after = _tree_snapshot(paths.accepted_root)
        if profile_after != profile_before:
            raise RuntimeError("profile source changed during scientific bridge")
        if tree_after != tree_before:
            raise RuntimeError("accepted-results tree changed during scientific bridge")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "merge"), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--wave-plan", type=Path, required=True)
    parser.add_argument("--accepted-root", type=Path, required=True)
    parser.add_argument("--merge-view-root", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    parser.add_argument("--profile-path", type=Path, required=True)
    parser.add_argument("--expected-profile-sha256", required=True)
    parser.add_argument("--expected-run-name", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_production_scientific_bridge(
        mode=args.mode,
        run_root=args.run_root,
        wave_plan_path=args.wave_plan,
        accepted_root=args.accepted_root,
        merge_view_root=args.merge_view_root,
        receipt_output_path=args.receipt_output,
        profile_path=args.profile_path,
        expected_profile_sha256=args.expected_profile_sha256,
        expected_run_name=args.expected_run_name,
    )
    compact = {
        "schema": CLI_RESULT_SCHEMA,
        "mode": args.mode,
        "status": result["status"],
        "run_name": result["run_name"],
        "decision": result.get("decision"),
        "all_gates_passed": result.get("all_gates_passed"),
        "receipt_output_path": str(args.receipt_output.resolve()),
        "result_sha256": result.get("receipt_sha256", result.get("preflight_sha256")),
        "current_profile_changed": False,
    }
    print(json.dumps(compact, sort_keys=True, separators=(",", ":")))
    if args.mode == "merge" and result.get("all_gates_passed") is not True:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "BridgePathsV2",
    "CLI_RESULT_SCHEMA",
    "ExecutionEvidenceV2",
    "PREFLIGHT_SCHEMA",
    "ProductionBridgeInputsV2",
    "build_production_adapters",
    "main",
    "preflight_production_bridge",
    "prepare_production_bridge_inputs",
    "run_production_scientific_bridge",
    "validate_bridge_paths",
]
