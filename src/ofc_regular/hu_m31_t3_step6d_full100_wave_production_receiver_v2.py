"""Production composition boundary for full100 wave-v2 result receipt.

This module deliberately composes existing, independently validated contracts.
It does not create, inspect, or delete a VM.  A receive run is allowed to write
only controller-owned ``ACCEPTED.json`` objects and immutable local output.

The lifecycle proof is not an input.  It is replayed from the write-once
controller journal and producer receipts by
``Full100WaveControllerV2.validate_lifecycle_closeout_chain`` inside the exact
one-shot adapter required by the result receiver.  Terminal observations are
then derived from that proof and a generation-pinned GCS snapshot; callers
cannot supply a terminal reason or raw proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from . import hu_m31_t3_step6d_full100_wave_controller_v2 as controller_v2
from . import hu_m31_t3_step6d_full100_wave_gce_adapter_v2 as gce_v2
from . import hu_m31_t3_step6d_full100_wave_launch_bundle_v2 as bundle_v2
from . import hu_m31_t3_step6d_full100_wave_plan_v2 as wave_v2
from . import hu_m31_t3_step6d_full100_wave_result_gcs_adapter_v2 as gcs_v2
from . import hu_m31_t3_step6d_full100_wave_result_receiver_v2 as receiver_v2
from . import hu_m31_t3_step6d_full100_wave_science_registry_v2 as science_registry
from .hu_m31_t3_step6d_performance_development_v2_cloud_execution_v1 import (
    _stdlib_http_request,
)


PRODUCTION_POLL_SCHEMA = (
    "hu_m31_t3_step6d_full100_wave_production_poll_v2"
)
PRODUCTION_RECEIVE_SCHEMA = (
    "hu_m31_t3_step6d_full100_wave_production_receive_v2"
)
PRODUCTION_RECEIVE_REQUEST_SCHEMA = (
    "hu_m31_t3_step6d_full100_wave_production_receive_request_v2"
)
JOURNAL_DIR_MODE_EXACT = "exact"

_LAUNCH_VALIDATION_KEYS = frozenset(
    {
        "outer_manifest",
        "quota_receipt",
        "persistent_claim_receipt",
        "planned_mapping_receipt",
        "prelaunch_authorization",
        "raw_claim_nonce",
        "current_time_utc",
        "runtime_preflight_receipt",
        "runtime_gcp_read_receipt",
        "gcp_read_receipt",
        "service_account_actas_receipt",
        "worker_identity_plan",
        "worker_identity_inventory_receipt",
        "worker_identity_act_as_receipt",
        "project_iam_scan_receipt",
        "worker_iam_plan",
        "worker_iam_prepare_receipt",
        "worker_iam_install_receipt",
        "worker_iam_readback_receipt",
    }
)
_CONTENT_BINDING_KEYS = frozenset(
    {
        "immutable_content_prefix",
        "content_payload_sha256",
        "outer_manifest_sha256",
    }
)
_REQUEST_PAYLOAD_KEYS = frozenset(
    {
        "closeout_event_sha256",
        "launch_bundle",
        "launch_validation",
        "startup_script_path",
        "gce_create_receipt",
        "gce_delete_receipt",
        "worker_iam_cleanup_receipt",
        "content_binding",
        "observed_at_utc",
    }
)
_REQUEST_KEYS = frozenset(
    {
        *_REQUEST_PAYLOAD_KEYS,
        "schema",
        "execution_namespace",
        "controller_journal_dir",
        "journal_dir_mode",
        "credentials_from_environment_only",
        "current_profile_changed",
        "request_sha256",
    }
)
_EXECUTION_NAMESPACE = re.compile(r"^execution-[0-9]{3}-[0-9a-f]{12}$")


def _read_json(path: str | Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not readable JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _clone_exact(
    value: Mapping[str, Any], expected: frozenset[str], label: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{label} fields changed")
    try:
        clone = json.loads(
            json.dumps(
                dict(value),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError):
        raise ValueError(f"{label} is not strict JSON") from None
    if not isinstance(clone, dict):  # pragma: no cover - guarded above
        raise ValueError(f"{label} is not an object")
    return clone


def _exact_controller_journal_dir(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} is not a path")
    requested = Path(value)
    if not requested.is_absolute():
        raise ValueError(f"{label} must be absolute")
    try:
        resolved = requested.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{label} does not exist") from exc
    if requested.is_symlink() or not resolved.is_dir():
        raise ValueError(f"{label} is not a plain directory")
    if str(resolved) != value:
        raise ValueError(f"{label} is not a canonical exact path")
    return str(resolved)


def build_production_receive_request(
    *,
    execution_namespace: str,
    controller_journal_dir: str | Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal the exact cleanup-to-receiver handoff for one execution.

    The controller journal path is deliberately exact.  It is not a parent
    directory to which the execution namespace may be appended later.
    """

    if (
        not isinstance(execution_namespace, str)
        or _EXECUTION_NAMESPACE.fullmatch(execution_namespace) is None
    ):
        raise ValueError("production receive execution namespace changed")
    material = _clone_exact(
        payload, _REQUEST_PAYLOAD_KEYS, "production receive request payload"
    )
    core = {
        "schema": PRODUCTION_RECEIVE_REQUEST_SCHEMA,
        "execution_namespace": execution_namespace,
        "controller_journal_dir": _exact_controller_journal_dir(
            str(controller_journal_dir),
            "controller journal directory",
        ),
        "journal_dir_mode": JOURNAL_DIR_MODE_EXACT,
        **material,
        "credentials_from_environment_only": True,
        "current_profile_changed": False,
    }
    return {**core, "request_sha256": wave_v2.canonical_sha256(core)}


def validate_production_receive_request(
    value: Mapping[str, Any],
    *,
    expected_execution_namespace: str | None = None,
    expected_controller_journal_dir: str | Path | None = None,
) -> dict[str, Any]:
    request = _clone_exact(value, _REQUEST_KEYS, "production receive request")
    digest = request.pop("request_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or digest != wave_v2.canonical_sha256(request)
    ):
        raise ValueError("production receive request digest changed")
    if (
        request.get("schema") != PRODUCTION_RECEIVE_REQUEST_SCHEMA
        or request.get("journal_dir_mode") != JOURNAL_DIR_MODE_EXACT
        or request.get("credentials_from_environment_only") is not True
        or request.get("current_profile_changed") is not False
        or not isinstance(request.get("execution_namespace"), str)
        or _EXECUTION_NAMESPACE.fullmatch(request["execution_namespace"]) is None
    ):
        raise ValueError("production receive request safety binding changed")
    exact_journal = _exact_controller_journal_dir(
        request["controller_journal_dir"], "controller journal directory"
    )
    if (
        expected_execution_namespace is not None
        and request["execution_namespace"] != expected_execution_namespace
    ):
        raise ValueError("production receive request namespace changed")
    if expected_controller_journal_dir is not None:
        expected_journal = _exact_controller_journal_dir(
            str(expected_controller_journal_dir),
            "expected controller journal directory",
        )
        if exact_journal != expected_journal:
            raise ValueError("production receive request journal path changed")
    request["controller_journal_dir"] = exact_journal
    request["request_sha256"] = digest
    return request


def resolve_production_receive_journal_dir(
    *,
    request: Mapping[str, Any],
    journal_dir: str | Path,
    journal_dir_mode: str,
    expected_execution_namespace: str | None = None,
) -> Path:
    """Resolve only the exact journal path sealed by cleanup.

    This intentionally has no namespaced-root compatibility branch.  The
    launch controller journal lives below the execution root as
    ``controller-journal`` and must be consumed at that exact path.
    """

    checked = validate_production_receive_request(
        request, expected_execution_namespace=expected_execution_namespace
    )
    if journal_dir_mode != JOURNAL_DIR_MODE_EXACT:
        raise PermissionError("production receive requires exact journal-dir mode")
    exact = _exact_controller_journal_dir(
        str(journal_dir),
        "CLI controller journal directory",
    )
    if exact != checked["controller_journal_dir"]:
        raise ValueError("CLI controller journal directory differs from sealed request")
    return Path(exact)


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def execution_namespace(
    attempt_ledger: Mapping[str, Any], resume_plan: Mapping[str, Any]
) -> str:
    """Return the sole filesystem namespace for one execution attempt.

    A base wave index can repeat when a failed candidate/reference pair moves
    from a00 to a01.  The transition ordinal plus resume digest disambiguates
    that retry while remaining deterministic across an idempotent rerun.
    """

    transitions = attempt_ledger.get("transitions")
    digest = resume_plan.get("resume_sha256")
    if not isinstance(transitions, list) or not transitions:
        raise ValueError("attempt ledger transitions are missing")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("resume digest changed")
    return f"execution-{len(transitions) - 1:03d}-{digest[:12]}"


def _namespaced(root: str | Path, namespace: str) -> Path:
    base = Path(root)
    return base if base.name == namespace else base / namespace


def _write_once_json(path: Path, value: Mapping[str, Any]) -> None:
    raw = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(raw)
    except FileExistsError:
        try:
            existing = path.read_bytes()
        except OSError as exc:
            raise RuntimeError(f"write-once output {path.name} is unreadable") from exc
        if existing != raw:
            raise FileExistsError(
                f"write-once output {path.name} already contains different bytes"
            ) from None


def _read_optional_json(path: Path, label: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} is not a plain file")
    return _read_json(path, label)


def _write_poll_checkpoint(
    output_root: str | Path,
    *,
    namespace: str,
    receipt: Mapping[str, Any],
) -> Path:
    digest = receipt.get("receipt_sha256")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or wave_v2.canonical_sha256(
            {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        )
        != digest
    ):
        raise ValueError("poll checkpoint receipt digest changed")
    path = _namespaced(output_root, namespace) / "polls" / f"{digest}.json"
    _write_once_json(path, receipt)
    return path


def _receiver_idempotency_view(value: Mapping[str, Any]) -> dict[str, Any]:
    clone = deepcopy(dict(value))
    clone.pop("receipt_sha256", None)
    attempts = clone.get("attempt_results")
    if not isinstance(attempts, list):
        raise ValueError("receiver receipt attempt results are missing")
    for row in attempts:
        if not isinstance(row, dict):
            raise ValueError("receiver receipt attempt result changed")
        # This is the only expected first-run/rerun difference: on the second
        # run the exact marker already exists and is read back instead of
        # created.  Its generation, bytes, SHA, DONE and transition must still
        # be identical and are intentionally retained in this comparison.
        row["acceptance_create_performed"] = False
    return clone


def _no_gce_network(*args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    raise PermissionError("production receiver never authorizes GCE network access")


def _validated_material(
    *,
    wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any],
    resume_plan: Mapping[str, Any],
    request: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], bytes, Callable[[Mapping[str, Any]], Mapping[str, Any]]]:
    plan = wave_v2.validate_wave_plan(wave_plan)
    expected_startup_sha256 = science_registry.resolve_startup_sha256(plan)
    expected_namespace = execution_namespace(attempt_ledger, resume_plan)
    payload = validate_production_receive_request(
        request, expected_execution_namespace=expected_namespace
    )
    validation = _clone_exact(
        payload["launch_validation"],
        _LAUNCH_VALIDATION_KEYS,
        "launch validation evidence",
    )
    _clone_exact(
        payload["content_binding"], _CONTENT_BINDING_KEYS, "content binding"
    )
    startup_path = Path(payload["startup_script_path"])
    if not startup_path.is_file() or startup_path.is_symlink():
        raise ValueError("startup script path is not a plain existing file")
    startup = startup_path.read_bytes()
    if hashlib.sha256(startup).hexdigest() != expected_startup_sha256:
        raise ValueError("startup script bytes differ from frozen launch hash")

    def validate_bundle(value: Mapping[str, Any]) -> Mapping[str, Any]:
        return bundle_v2.validate_launch_bundle(
            wave_plan=plan,
            attempt_ledger=attempt_ledger,
            resume_plan=resume_plan,
            expected_startup_sha256=expected_startup_sha256,
            value=value,
            **validation,
        )

    launch_bundle = deepcopy(dict(validate_bundle(payload["launch_bundle"])))
    content = payload["content_binding"]
    manifest = validation["outer_manifest"]
    if (
        content["immutable_content_prefix"] != manifest["content_prefix"]
        or content["content_payload_sha256"]
        != manifest["content_payload_sha256"]
        or content["outer_manifest_sha256"] != manifest["manifest_sha256"]
    ):
        raise ValueError("content binding differs from validated outer package")
    return payload, launch_bundle, startup, validate_bundle


def _lifecycle_adapter(
    *,
    controller: controller_v2.Full100WaveControllerV2,
    request: Mapping[str, Any],
    launch_bundle: Mapping[str, Any],
    startup: bytes,
    validate_bundle: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    proof_box: dict[str, dict[str, Any]],
    expected_proof_sha256: str | None = None,
) -> receiver_v2.ValidatedLifecycleProofAdapterV2:
    validation = request["launch_validation"]
    image = validation["runtime_preflight_receipt"]["image"]
    common = {
        "launch_bundle": launch_bundle,
        "launch_bundle_validator": validate_bundle,
        "active_image_self_link": image["self_link"],
        "active_image_identity_sha256": image["image_identity_sha256"],
        "expected_image_digest": controller.wave_plan["runtime_binding"][
            "image_digest"
        ],
        "startup_script_bytes": startup,
        "expected_startup_sha256": science_registry.resolve_startup_sha256(
            controller.wave_plan
        ),
        "requester": _no_gce_network,
    }
    create_adapter = gce_v2.GceWavePhaseBAdapter(mode="create", **common)
    delete_adapter = gce_v2.GceWavePhaseBAdapter(
        mode="delete", create_receipt=request["gce_create_receipt"], **common
    )
    absence_adapter = gce_v2.GceWavePhaseBAdapter(
        mode="absence",
        create_receipt=request["gce_create_receipt"],
        delete_receipt=request["gce_delete_receipt"],
        **common,
    )
    content = request["content_binding"]

    def replay() -> Mapping[str, Any]:
        if proof_box:
            raise RuntimeError("lifecycle replay callback was reused")
        proof = controller.validate_lifecycle_closeout_chain(
            closeout_event_sha256=request["closeout_event_sha256"],
            launch_bundle=launch_bundle,
            launch_bundle_validator=validate_bundle,
            gce_create_adapter=create_adapter,
            gce_delete_adapter=delete_adapter,
            gce_absence_adapter=absence_adapter,
            quota_receipt=validation["quota_receipt"],
            persistent_claim_receipt=validation["persistent_claim_receipt"],
            planned_mapping_receipt=validation["planned_mapping_receipt"],
            prelaunch_authorization=validation["prelaunch_authorization"],
            worker_iam_plan=validation["worker_iam_plan"],
            immutable_content_prefix=content["immutable_content_prefix"],
            content_payload_sha256=content["content_payload_sha256"],
            outer_manifest_sha256=content["outer_manifest_sha256"],
            worker_iam_prepare_receipt=validation[
                "worker_iam_prepare_receipt"
            ],
            worker_iam_install_receipt=validation[
                "worker_iam_install_receipt"
            ],
            worker_iam_readback_receipt=validation[
                "worker_iam_readback_receipt"
            ],
        )
        if (
            expected_proof_sha256 is not None
            and proof.get("proof_sha256") != expected_proof_sha256
        ):
            raise RuntimeError("lifecycle proof changed between closeout and receive")
        proof_box["proof"] = deepcopy(dict(proof))
        return proof

    return receiver_v2.ValidatedLifecycleProofAdapterV2(replay)


class _LifecycleDerivedTerminals(Sequence[Mapping[str, Any]]):
    """Lazy sequence populated only after the one-shot proof replay executes."""

    def __init__(
        self,
        *,
        resume_plan: Mapping[str, Any],
        proof_box: Mapping[str, Mapping[str, Any]],
        pinned_paths: frozenset[str],
    ) -> None:
        self._selected = deepcopy(list(resume_plan["selected_attempts"]))
        self._proof_box = proof_box
        self._pinned_paths = pinned_paths

    def __len__(self) -> int:
        return len(self._selected)

    def _rows(self) -> list[dict[str, Any]]:
        proof = self._proof_box.get("proof")
        if not isinstance(proof, Mapping):
            raise RuntimeError("terminal derivation preceded lifecycle proof replay")
        mappings = proof.get("selected_instance_mapping")
        if not isinstance(mappings, list) or len(mappings) != len(self._selected):
            raise ValueError("lifecycle selected mapping is missing")
        rows: list[dict[str, Any]] = []
        for selected, mapping in zip(self._selected, mappings, strict=True):
            if mapping.get("job_id") != selected["job_id"]:
                raise ValueError("lifecycle selected mapping order changed")
            done_path = f"{selected['artifact_prefix']}/DONE.json"
            if mapping.get("exact_instance_created") is False:
                reason = "create_missing_before_done"
            elif done_path in self._pinned_paths:
                reason = "done_observed"
            else:
                # Closeout proves the exact instance and disk are gone.  The
                # generic worker-failed classification avoids inventing an
                # unobserved spot-loss or timeout cause.
                reason = "worker_failed_before_done"
            rows.append(
                {
                    "job_id": selected["job_id"],
                    "source_role": selected["source_role"],
                    "attempt_id": selected["attempt_id"],
                    "instance_id": selected["instance_id"],
                    "terminal_reason": reason,
                }
            )
        return rows

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        return iter(self._rows())

    def __getitem__(self, index: int | slice) -> Any:
        return self._rows()[index]


def _poll_store(
    *,
    plan: Mapping[str, Any],
    ledger: Mapping[str, Any],
    resume: Mapping[str, Any],
    requester: Callable[..., Any],
) -> tuple[gcs_v2.GcsResultStoreV2, dict[str, Any]]:
    store = gcs_v2.GcsResultStoreV2(
        mode="read",
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        requester=requester,
    )
    return store, store.poll_selected()


def production_poll(
    *,
    wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any],
    resume_plan: Mapping[str, Any],
    requester: Callable[..., Any] = _stdlib_http_request,
) -> dict[str, Any]:
    """Perform only generation-pinned GET/list operations for a selected wave."""

    plan = wave_v2.validate_wave_plan(wave_plan)
    ledger = wave_v2.validate_attempt_ledger(plan, attempt_ledger)
    resume = wave_v2.validate_resume_plan(plan, ledger, resume_plan)
    namespace = execution_namespace(ledger, resume)
    _, poll = _poll_store(
        plan=plan, ledger=ledger, resume=resume, requester=requester
    )
    body = {
        "schema": PRODUCTION_POLL_SCHEMA,
        "status": "selected_wave_generation_pinned_read_only",
        "run_name": plan["run_name"],
        "wave_index": resume["resume_wave_index"],
        "execution_namespace": namespace,
        "attempt_ledger_sha256": ledger["ledger_sha256"],
        "resume_plan_sha256": resume["resume_sha256"],
        "poll_receipt": poll,
        "acceptance_create_authorized": False,
        "vm_lifecycle_mutation_performed": False,
        "current_profile_changed": False,
    }
    return {**body, "receipt_sha256": wave_v2.canonical_sha256(body)}


def production_receive(
    *,
    journal_dir: str | Path,
    journal_dir_mode: str = JOURNAL_DIR_MODE_EXACT,
    wave_plan: Mapping[str, Any],
    attempt_ledger: Mapping[str, Any],
    resume_plan: Mapping[str, Any],
    request: Mapping[str, Any],
    destination: str | Path,
    output_dir: str | Path,
    project_id: str,
    zone: str,
    allow_accept_create: bool = False,
    confirm_run_name: str | None = None,
    requester: Callable[..., Any] = _stdlib_http_request,
) -> dict[str, Any]:
    """Replay closeout, receive one wave, and write immutable local outputs."""

    plan = wave_v2.validate_wave_plan(wave_plan)
    ledger = wave_v2.validate_attempt_ledger(plan, attempt_ledger)
    resume = wave_v2.validate_resume_plan(plan, ledger, resume_plan)
    if allow_accept_create is not True:
        raise PermissionError("production receive requires explicit ACCEPTED create")
    if confirm_run_name != plan["run_name"]:
        raise PermissionError("production receive confirmation does not match the wave")
    namespace = execution_namespace(ledger, resume)
    payload, launch_bundle, startup, validate_bundle = _validated_material(
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        request=request,
    )
    exact_journal_dir = resolve_production_receive_journal_dir(
        request=payload,
        journal_dir=journal_dir,
        journal_dir_mode=journal_dir_mode,
        expected_execution_namespace=namespace,
    )
    controller = controller_v2.Full100WaveControllerV2(
        journal_dir=exact_journal_dir,
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        create_journal=False,
    )

    # Establish quiescence before reading result objects.  This first
    # one-shot adapter proves closeout from the journal and producer receipts;
    # it does not accept a caller-built proof and cannot touch GCE.
    closeout_box: dict[str, dict[str, Any]] = {}
    closeout_adapter = _lifecycle_adapter(
        controller=controller,
        request=payload,
        launch_bundle=launch_bundle,
        startup=startup,
        validate_bundle=validate_bundle,
        proof_box=closeout_box,
    )
    closeout_proof = closeout_adapter.validate(
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        receiver_observed_at_utc=payload["observed_at_utc"],
    )
    _, poll = _poll_store(
        plan=plan, ledger=ledger, resume=resume, requester=requester
    )
    pinned_paths = frozenset(row["path"] for row in poll["records"])
    proof_box: dict[str, dict[str, Any]] = {}
    lifecycle_adapter = _lifecycle_adapter(
        controller=controller,
        request=payload,
        launch_bundle=launch_bundle,
        startup=startup,
        validate_bundle=validate_bundle,
        proof_box=proof_box,
        expected_proof_sha256=closeout_proof["proof_sha256"],
    )
    terminals = _LifecycleDerivedTerminals(
        resume_plan=resume,
        proof_box={"proof": closeout_proof},
        pinned_paths=pinned_paths,
    )
    accept_store = gcs_v2.GcsResultStoreV2(
        mode="accept",
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        requester=requester,
    )
    receipt = receiver_v2.receive_wave_results(
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        job_bootstraps=[
            row["bootstrap"] for row in launch_bundle["bootstrap_inventory"]
        ],
        terminal_observations=terminals,
        lifecycle_proof_adapter=lifecycle_adapter,
        worker_iam_plan=payload["launch_validation"]["worker_iam_plan"],
        worker_iam_prepare_receipt=payload["launch_validation"][
            "worker_iam_prepare_receipt"
        ],
        worker_iam_install_receipt=payload["launch_validation"][
            "worker_iam_install_receipt"
        ],
        worker_iam_readback_receipt=payload["launch_validation"][
            "worker_iam_readback_receipt"
        ],
        worker_iam_cleanup_receipt=payload["worker_iam_cleanup_receipt"],
        project_id=project_id,
        zone=zone,
        observed_at_utc=payload["observed_at_utc"],
        store=accept_store,
        destination=destination,
        readback_source="gcloud_readback",
    )
    receiver_v2.validate_receiver_receipt(plan, receipt)
    output = _namespaced(output_dir, namespace)
    receiver_path = output / "receiver_receipt.json"
    existing_receiver = _read_optional_json(receiver_path, "receiver receipt")
    if existing_receiver is not None:
        checked_existing = receiver_v2.validate_receiver_receipt(
            plan, existing_receiver
        )
        if _receiver_idempotency_view(checked_existing) != _receiver_idempotency_view(
            receipt
        ):
            raise FileExistsError(
                "write-once receiver receipt differs from revalidated result"
            )
        receipt = checked_existing
    _write_once_json(receiver_path, receipt)
    _write_once_json(output / "attempt_ledger.json", receipt["attempt_ledger"])
    if receipt["next_resume_plan"] is not None:
        _write_once_json(output / "resume_plan.json", receipt["next_resume_plan"])
    body = {
        "schema": PRODUCTION_RECEIVE_SCHEMA,
        "status": receipt["status"],
        "run_name": plan["run_name"],
        "wave_index": resume["resume_wave_index"],
        # ``wave_index`` may repeat for an a01 pair retry.  These three fields
        # are therefore the execution-attempt identity; the index alone is
        # never used to name or resume an execution.
        "execution_ordinal": len(ledger["transitions"]) - 1,
        "input_attempt_ledger_sha256": ledger["ledger_sha256"],
        "input_resume_plan_sha256": resume["resume_sha256"],
        "execution_namespace": namespace,
        "poll_receipt_sha256": poll["receipt_sha256"],
        "receiver_receipt_sha256": receipt["receipt_sha256"],
        "next_attempt_ledger_sha256": receipt["attempt_ledger"]["ledger_sha256"],
        "next_resume_plan_sha256": (
            None
            if receipt["next_resume_plan"] is None
            else receipt["next_resume_plan"]["resume_sha256"]
        ),
        "accepted_job_ids": deepcopy(receipt["accepted_job_ids"]),
        "failed_job_ids": deepcopy(receipt["failed_job_ids"]),
        "lifecycle_proof_replayed_inside_one_shot_adapter": True,
        "terminal_observations_derived_from_pinned_gcs_and_closeout": True,
        "acceptance_create_only": True,
        "vm_lifecycle_mutation_performed": False,
        "current_profile_changed": False,
    }
    summary = {**body, "receipt_sha256": wave_v2.canonical_sha256(body)}
    summary_path = output / "production_receive_receipt.json"
    existing_summary = _read_optional_json(
        summary_path, "production receive receipt"
    )
    if existing_summary is not None:
        digest = existing_summary.pop("receipt_sha256", None)
        if (
            set(existing_summary) != set(body)
            or digest != wave_v2.canonical_sha256(existing_summary)
            or existing_summary.get("schema") != PRODUCTION_RECEIVE_SCHEMA
            or existing_summary.get("run_name") != plan["run_name"]
            or existing_summary.get("execution_namespace") != namespace
            or existing_summary.get("input_attempt_ledger_sha256")
            != ledger["ledger_sha256"]
            or existing_summary.get("input_resume_plan_sha256")
            != resume["resume_sha256"]
            or existing_summary.get("receiver_receipt_sha256")
            != receipt["receipt_sha256"]
            or existing_summary.get("next_attempt_ledger_sha256")
            != receipt["attempt_ledger"]["ledger_sha256"]
            or existing_summary.get("next_resume_plan_sha256")
            != (
                None
                if receipt["next_resume_plan"] is None
                else receipt["next_resume_plan"]["resume_sha256"]
            )
        ):
            raise FileExistsError(
                "write-once production receipt differs from revalidated result"
            )
        return {**existing_summary, "receipt_sha256": digest}
    _write_once_json(summary_path, summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("poll", "receive"), required=True)
    parser.add_argument("--wave-plan", type=Path, required=True)
    parser.add_argument("--attempt-ledger", type=Path, required=True)
    parser.add_argument("--resume-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--journal-dir", type=Path)
    parser.add_argument(
        "--journal-dir-mode", choices=(JOURNAL_DIR_MODE_EXACT,)
    )
    parser.add_argument("--request", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--project-id")
    parser.add_argument("--zone")
    parser.add_argument("--allow-accept-create", action="store_true")
    parser.add_argument("--confirm-run-name")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan = _read_json(args.wave_plan, "wave plan")
    ledger = _read_json(args.attempt_ledger, "attempt ledger")
    resume = _read_json(args.resume_plan, "resume plan")
    validated = wave_v2.validate_wave_plan(plan)
    if args.mode == "poll":
        if args.allow_accept_create:
            raise PermissionError("poll mode cannot authorize ACCEPTED create")
        receipt = production_poll(
            wave_plan=validated,
            attempt_ledger=ledger,
            resume_plan=resume,
        )
        namespace = execution_namespace(ledger, resume)
        _write_poll_checkpoint(
            args.output_dir, namespace=namespace, receipt=receipt
        )
    else:
        if args.allow_accept_create is not True:
            raise PermissionError("receive requires explicit --allow-accept-create")
        if args.confirm_run_name != validated["run_name"]:
            raise PermissionError("--confirm-run-name must exactly match the wave")
        missing = [
            name
            for name in (
                "journal_dir",
                "journal_dir_mode",
                "request",
                "destination",
                "project_id",
                "zone",
            )
            if getattr(args, name) is None
        ]
        if missing:
            raise ValueError("receive arguments are incomplete")
        receipt = production_receive(
            journal_dir=args.journal_dir,
            journal_dir_mode=args.journal_dir_mode,
            wave_plan=validated,
            attempt_ledger=ledger,
            resume_plan=resume,
            request=_read_json(args.request, "production receive request"),
            destination=args.destination,
            output_dir=args.output_dir,
            project_id=args.project_id,
            zone=args.zone,
            allow_accept_create=True,
            confirm_run_name=args.confirm_run_name,
        )
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "JOURNAL_DIR_MODE_EXACT",
    "PRODUCTION_POLL_SCHEMA",
    "PRODUCTION_RECEIVE_SCHEMA",
    "PRODUCTION_RECEIVE_REQUEST_SCHEMA",
    "build_production_receive_request",
    "execution_namespace",
    "main",
    "production_poll",
    "production_receive",
    "resolve_production_receive_journal_dir",
    "validate_production_receive_request",
]
