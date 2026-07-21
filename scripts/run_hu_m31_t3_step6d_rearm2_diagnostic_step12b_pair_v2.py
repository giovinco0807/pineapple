#!/usr/bin/env python3
"""Run the explicitly authorized Step12b direct-v2 exact pair once.

The default mode is offline dry-run.  Cloud mutation requires both
``--execute`` and the exact confirmation string.  The live path is deliberately
one-shot: candidate attempt0 plus reference attempt0, with no attempt1, retry,
or third VM surface.

The ephemeral RSA private key and OAuth bearer tokens remain process-local.
Only public, digest-bound audit evidence is written to the fresh output root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_cloud_worker_adapter
    as worker_adapter,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_gce_transport_10c2_v1
    as payload_transport,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_plan as payload_plan,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step11_controller_v1
    as step11_controller,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step11_rest_iam_admin_v1
    as rest_iam,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_bootstrap_source_v2
    as bootstrap_source,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_bootstrap_source_content_v2
    as bootstrap_source_content,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_deployment_contract_v2
    as deployment_v2,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_external_authorization_v2
    as external_auth,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_external_preflight_gate_v2
    as external_preflight,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_live_cloud_adapters_v2
    as live_cloud,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_live_preflight_collectors_v2
    as live_collectors,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_pair_cloud_controller_v2
    as pair_controller,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_phase2_iam_lifecycle_v2
    as phase2_lifecycle,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_phase2_iam_plan_v2
    as phase2_iam,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_result_receiver_v2
    as result_receiver,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_run_scoped_controller_sa_v2
    as controller_sa,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_startup_loader_v2
    as startup_loader,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_token_barrier_v1
    as token_barrier,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_vm_prebootstrap_v2
    as vm_prebootstrap,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
STEP11_ROOT = (
    REPO_ROOT
    / "outputs"
    / "hu_joint_policy"
    / "m31_t3_step6d"
    / "step11_one_vm_v12_fix3_actual"
)
PACKAGE_DIR = (
    REPO_ROOT
    / "outputs"
    / "hu_joint_policy"
    / "m31_t3_step6d"
    / "rearm2_diagnostic_cloud_worker_package_v1"
)
PACKAGE_PROVISION_RECEIPT_PATH = (
    STEP11_ROOT / "package_provision_receipt.json"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "outputs"
    / "hu_joint_policy"
    / "m31_t3_step6d"
    / "step12b_pair_v2_actual"
)
POLICY_REGISTRY_PATH = REPO_ROOT / "src" / "ofc_regular" / "ai_profiles.py"
EXPECTED_POLICY_REGISTRY_SHA256 = (
    "d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3"
)
EXECUTION_CONFIRMATION = "EXECUTE_STEP12B_DIRECT_V2_EXACT_PAIR_ATTEMPT0"
AUTHORIZATION_WINDOW_SECONDS = phase2_iam.MAX_AUTHORIZATION_WINDOW_SECONDS
RUNNER_RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_pair_v2_actual_runner_receipt_v1"
)
FAILURE_RECEIPT_SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_pair_v2_actual_runner_failure_v1"
)

_IMMUTABLE_ROOTS = {
    STEP11_ROOT.resolve(),
    PACKAGE_DIR.resolve(),
    (
        REPO_ROOT
        / "outputs"
        / "hu_joint_policy"
        / "m31_t3_step6d"
        / "step12_pair_v1_actual"
    ).resolve(),
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sealed(body: Mapping[str, Any]) -> dict[str, Any]:
    copied = dict(body)
    return {**copied, "receipt_sha256": _canonical_sha256(copied)}


def _validated_sealed_receipt(
    value: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is not a receipt mapping")
    checked = dict(value)
    supplied = checked.pop("receipt_sha256", None)
    if (
        not isinstance(supplied, str)
        or len(supplied) != 64
        or any(character not in "0123456789abcdef" for character in supplied)
        or supplied != _canonical_sha256(checked)
    ):
        raise ValueError(f"{label} sealed digest changed")
    restored = {**checked, "receipt_sha256": supplied}
    _assert_no_secret_surface(restored)
    return restored


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"required regular JSON file is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"required JSON object changed: {path}")
    return value


def _policy_registry_sha256() -> str:
    if (
        not POLICY_REGISTRY_PATH.is_file()
        or POLICY_REGISTRY_PATH.is_symlink()
    ):
        raise RuntimeError("current/profile registry is not a regular file")
    return hashlib.sha256(POLICY_REGISTRY_PATH.read_bytes()).hexdigest()


def _assert_policy_registry_unchanged(
    expected: str = EXPECTED_POLICY_REGISTRY_SHA256,
) -> str:
    observed = _policy_registry_sha256()
    if observed != expected:
        raise RuntimeError("current/profile registry hash changed")
    return observed


def _assert_no_secret_surface(value: Any, path: str = "$") -> None:
    """Reject raw credential/private-key surfaces before durable writes."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string field")
            lowered = key.lower()
            if lowered in {
                "access_token",
                "authorization_header",
                "private_key",
                "private_key_pem",
                "raw_token",
            } and child is not False and child is not None:
                raise ValueError(f"{path}.{key} is a secret field")
            _assert_no_secret_surface(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_no_secret_surface(child, f"{path}[{index}]")
    elif isinstance(value, str):
        if (
            "-----BEGIN PRIVATE KEY-----" in value
            or value.lower().startswith("bearer ")
        ):
            raise ValueError(f"{path} contains secret material")


def _exclusive_write_json(path: Path, value: Mapping[str, Any]) -> None:
    _assert_no_secret_surface(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = _canonical_bytes(dict(value)) + b"\n"
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()


def _safe_output_root(path: str | Path) -> Path:
    root = Path(path).resolve()
    if root in _IMMUTABLE_ROOTS:
        raise PermissionError("immutable Step11/old Step12 output root selected")
    for immutable in _IMMUTABLE_ROOTS:
        if immutable in root.parents:
            raise PermissionError("output escaped into immutable artifact tree")
    return root


@dataclass(frozen=True)
class PreparedRun:
    signer: Any
    controller_public_key_record: Mapping[str, Any]
    run_nonce: str
    candidate_payload_contract: Mapping[str, Any]
    reference_payload_contract: Mapping[str, Any]
    deployment_contract: Mapping[str, Any]
    runtime_source_files: Mapping[str, str]
    source_plan: Mapping[str, Any]
    phase2_iam_plan: Mapping[str, Any]
    package_provision_receipt: Mapping[str, Any]
    issued_at_unix_seconds: int
    expires_at_unix_seconds: int
    policy_registry_sha256: str


def prepare_run(
    *,
    now_unix_seconds: int | None = None,
    run_nonce: str | None = None,
    signer: Any | None = None,
) -> PreparedRun:
    """Build every offline contract without constructing a cloud adapter."""

    profile_sha = _assert_policy_registry_unchanged()
    now = int(time.time()) if now_unix_seconds is None else now_unix_seconds
    if type(now) is not int or now <= 0:
        raise ValueError("runner wall clock changed")
    checked_nonce = secrets.token_hex(32) if run_nonce is None else run_nonce
    if (
        not isinstance(checked_nonce, str)
        or len(checked_nonce) != 64
        or any(character not in "0123456789abcdef" for character in checked_nonce)
    ):
        raise ValueError("runner nonce changed")
    controller_signer = (
        step11_controller.generate_ephemeral_controller_key(key_size=3_072)
        if signer is None
        else signer
    )
    public_record = dict(controller_signer.public_record)
    stage1 = _read_json(STEP11_ROOT / "transport_contract.json")
    stage1_done = _read_json(STEP11_ROOT / "late_done_envelope.json")
    stage1_receive = worker_adapter.build_receive(
        stage1["adapter_preview"], done_records=[stage1_done]
    )
    wheel = next(
        row
        for row in stage1["outer_package_manifest"]["objects"]
        if row["kind"] == "offline_numpy_cp311_manylinux_x86_64_wheel"
    )
    payloads = [
        payload_transport.build_job_contract(
            package_dir=PACKAGE_DIR,
            stage_id=payload_plan.STAGE2_ID,
            job_id=job_id,
            attempt_index=0,
            offline_wheel_record=wheel,
            controller_public_key_record=public_record,
            prerequisite_stage1_preview=stage1["adapter_preview"],
            prerequisite_stage1_receive=stage1_receive,
        )
        for job_id in payload_plan.STAGE2_JOB_IDS
    ]
    if len(payloads) != 2:
        raise ValueError("Step12b exact payload pair changed")
    runtime_sources = {
        logical_path: (
            REPO_ROOT / "src" / Path(logical_path)
        ).read_text(encoding="utf-8")
        for logical_path in bootstrap_source.REQUIRED_RUNTIME_SOURCE_PATHS
    }
    if set(runtime_sources) != set(
        bootstrap_source.REQUIRED_RUNTIME_SOURCE_PATHS
    ):
        raise ValueError("runtime source closure changed while being read")
    content_binding = (
        bootstrap_source_content.build_bootstrap_source_content_binding(
            runtime_source_files=runtime_sources,
            candidate_payload_contract=payloads[0],
            reference_payload_contract=payloads[1],
        )
    )
    deployment = deployment_v2.build_deployment_contract(
        candidate_payload_contract=payloads[0],
        reference_payload_contract=payloads[1],
        controller_public_key_record=public_record,
        run_nonce=checked_nonce,
        bootstrap_source_content_binding=content_binding,
    )
    source_plan = bootstrap_source.build_bootstrap_source_plan(
        deployment_contract=deployment,
        candidate_payload_contract=payloads[0],
        reference_payload_contract=payloads[1],
        controller_public_key_record=public_record,
        run_nonce=checked_nonce,
        runtime_source_files=runtime_sources,
    )
    expires = now + AUTHORIZATION_WINDOW_SECONDS
    phase2_plan = phase2_iam.build_step12b_phase2_iam_plan(
        deployment,
        candidate_payload_contract=payloads[0],
        reference_payload_contract=payloads[1],
        controller_public_key_record=public_record,
        run_nonce=checked_nonce,
        issued_at_unix_seconds=now,
        expires_at_unix_seconds=expires,
    )
    package_receipt = _read_json(PACKAGE_PROVISION_RECEIPT_PATH)
    # The owning external gate later performs the full immutable 16-object
    # validation.  This early check prevents accidentally selecting another
    # package receipt.
    if (
        package_receipt.get("object_count") != 16
        or package_receipt.get("receipt_sha256")
        != "d1994fb4c759d437dfbcd4cddbabb625a85e830ca5d3c6f95750c969972c97d6"
    ):
        raise ValueError("immutable Step11 package receipt changed")
    return PreparedRun(
        signer=controller_signer,
        controller_public_key_record=public_record,
        run_nonce=checked_nonce,
        candidate_payload_contract=payloads[0],
        reference_payload_contract=payloads[1],
        deployment_contract=deployment,
        runtime_source_files=runtime_sources,
        source_plan=source_plan,
        phase2_iam_plan=phase2_plan,
        package_provision_receipt=package_receipt,
        issued_at_unix_seconds=now,
        expires_at_unix_seconds=expires,
        policy_registry_sha256=profile_sha,
    )


def dry_run_receipt(prepared: PreparedRun) -> dict[str, Any]:
    deployment = prepared.deployment_contract
    body = {
        "schema": RUNNER_RECEIPT_SCHEMA,
        "status": "offline_dry_run_only_no_cloud_adapter_constructed",
        "deployment_contract_sha256": deployment[
            "deployment_contract_sha256"
        ],
        "run_identity_sha256": deployment["run_identity_sha256"],
        "direct_stage_identity_sha256": deployment[
            "direct_stage_identity_sha256"
        ],
        "instance_names": [
            row["instance_name"] for row in deployment["instances"]
        ],
        "source_roles": list(deployment["source_roles"]),
        "machine_type": deployment_v2.ACTUAL_MACHINE_TYPE,
        "vm_count": 2,
        "attempt_index": 0,
        "attempt1_authorized": False,
        "third_vm_authorized": False,
        "automatic_retry_authorized": False,
        "runtime_source_file_count": len(prepared.runtime_source_files),
        "source_object_count": bootstrap_source.SOURCE_OBJECT_COUNT,
        "phase2_binding_count": (
            phase2_iam.CONTROLLER_BINDING_COUNT
            + phase2_iam.WORKER_BINDING_COUNT
        ),
        "policy_registry_sha256": prepared.policy_registry_sha256,
        "output_root_created": False,
        "cloud_adapter_constructed": False,
        "cloud_mutation_performed": False,
        "current_profile_changed": False,
    }
    return _sealed(body)


class _LivePhase2Teardown:
    def __init__(self, backend: "LiveExecutionBackend") -> None:
        self._backend = backend
        self.controller_zero: (
            phase2_lifecycle.ControllerBindingsZeroCapability | None
        ) = None
        self.failure_cleanup_complete = False

    def remove_controller_bindings_after_claims(
        self,
        *,
        installed: phase2_lifecycle.Phase2IamInstalledCapability,
        claim_cas_readback_receipts: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        pair_claims = (
            phase2_lifecycle.validate_pair_claim_cas_readbacks_capability(
                deployment_contract=self._backend.prepared.deployment_contract,
                claim_cas_readback_receipts=claim_cas_readback_receipts,
            )
        )
        self.controller_zero = (
            phase2_lifecycle.remove_step12b_phase2_controller_bindings(
                iam_admin=self._backend.iam_admin,
                installed=installed,
                pair_claims=pair_claims,
                **self._backend.phase2_context(),
            )
        )
        return dict(self.controller_zero.phase2_zero_receipt)

    def cleanup_phase2_iam_on_failure(
        self, *, failure_evidence_sha256: str
    ) -> Mapping[str, Any]:
        receipt = phase2_lifecycle.cleanup_step12b_phase2_iam_on_failure(
            iam_admin=self._backend.iam_admin,
            failure_evidence_sha256=failure_evidence_sha256,
            **self._backend.phase2_context(),
        )
        self.failure_cleanup_complete = True
        return receipt


class LiveExecutionBackend:
    """Real adapters, instantiated only after the exact CLI confirmation."""

    def __init__(self, prepared: PreparedRun) -> None:
        self.prepared = prepared
        self.user_tokens = live_cloud.GcloudUserAccessTokenSource()
        self.http = live_cloud.StdlibCloudHttpsClient()
        deployment = prepared.deployment_contract
        self.instance_names = tuple(
            row["instance_name"] for row in deployment["instances"]
        )
        self.user_compute = live_cloud.ExactPairComputeClient(
            token_source=self.user_tokens,
            instance_names=self.instance_names,
            principal=token_barrier.DEFAULT_INITIATING_PRINCIPAL,
            credential_kind="user",
            http_client=self.http,
            allow_insert=False,
        )
        identity = controller_sa.identity_from_deployment(
            deployment,
            candidate_payload_contract=prepared.candidate_payload_contract,
            reference_payload_contract=prepared.reference_payload_contract,
            controller_public_key_record=prepared.controller_public_key_record,
            run_nonce=prepared.run_nonce,
        )
        self.controller_sa_admin = (
            controller_sa.RunScopedControllerServiceAccountAdmin(
                http_client=self.http,
                user_token_source=self.user_tokens,
                identity=identity,
            )
        )
        self.iam_admin = rest_iam.Step11RestIamAdmin(
            http_client=self.http,
            user_token_source=self.user_tokens,
            project=payload_transport.PROJECT,
            bucket=payload_transport.BUCKET,
            worker_service_account=payload_transport.WORKER_SERVICE_ACCOUNT,
            controller_service_account=identity.email,
        )
        self.source_provision: (
            bootstrap_source.ValidatedBootstrapSourceProvision | None
        ) = None
        self.role_manifests: dict[str, dict[str, Any]] = {}
        self.created: Any | None = None
        self.barrier: token_barrier.TokenBarrierOutcome | None = None
        self.token_creator_binding: (
            token_barrier.TokenCreatorBinding | None
        ) = None
        self.token_barrier_entered = False
        self.installed: (
            phase2_lifecycle.Phase2IamInstalledCapability | None
        ) = None
        self.phase2_install_entered = False
        self.teardown = _LivePhase2Teardown(self)
        self.worker_zero: (
            phase2_lifecycle.WorkerBindingsZeroCapability | None
        ) = None
        self.pair_controller_entered = False
        self._monotonic = time.monotonic
        self._sleep = time.sleep

    def phase2_context(self) -> dict[str, Any]:
        return {
            "phase2_iam_plan": self.prepared.phase2_iam_plan,
            "deployment_contract": self.prepared.deployment_contract,
            "candidate_payload_contract": (
                self.prepared.candidate_payload_contract
            ),
            "reference_payload_contract": (
                self.prepared.reference_payload_contract
            ),
            "controller_public_key_record": (
                self.prepared.controller_public_key_record
            ),
            "run_nonce": self.prepared.run_nonce,
        }

    def provision_sources(
        self,
    ) -> tuple[
        bootstrap_source.ValidatedBootstrapSourceProvision,
        dict[str, dict[str, Any]],
    ]:
        store = live_cloud.GenerationPinnedBootstrapSourceStore(
            source_plan=self.prepared.source_plan,
            http_client=self.http,
            user_token_source=self.user_tokens,
        )
        self.source_provision = bootstrap_source.provision_bootstrap_sources(
            source_plan=self.prepared.source_plan,
            prefix_observer=store,
            writer=store,
            reader=store,
            **{
                key: value
                for key, value in self.phase2_context().items()
                if key != "phase2_iam_plan"
            },
        )
        self.role_manifests = {
            job_id: bootstrap_source.build_role_bootstrap_manifest(
                source_plan=self.prepared.source_plan,
                validated_provision=self.source_provision,
                external_job_id=job_id,
                **{
                    key: value
                    for key, value in self.phase2_context().items()
                    if key != "phase2_iam_plan"
                },
            )
            for job_id in self.prepared.deployment_contract[
                "selected_job_ids"
            ]
        }
        return self.source_provision, dict(self.role_manifests)

    def create_controller_service_account(self) -> Any:
        absence = self.controller_sa_admin.require_absent()
        self.created = self.controller_sa_admin.create(absence=absence)
        return self.created

    def run_token_barrier(self) -> token_barrier.TokenBarrierOutcome:
        controller_email = self.prepared.deployment_contract[
            "controller_service_account"
        ]["email"]
        generator = live_cloud.IamCredentialsControllerTokenGenerator(
            controller_service_account=controller_email,
            http_client=self.http,
            user_token_source=self.user_tokens,
        )
        binding = token_barrier.build_token_creator_binding(
            controller_service_account=controller_email,
            expires_at_rfc3339=self.prepared.phase2_iam_plan[
                "authorization_window"
            ]["expires_at_rfc3339"],
        )
        self.token_creator_binding = binding
        # Mark the boundary before the add request.  An unknown transport
        # outcome must be treated as a possibly-live residual binding.
        self.token_barrier_entered = True
        self.barrier = token_barrier.run_token_creator_barrier(
            admin=self.iam_admin,
            token_generator=generator,
            binding=binding,
            now_monotonic=time.monotonic,
            now_unix_seconds=lambda: int(time.time()),
            sleep=time.sleep,
        )
        return self.barrier

    def _collect_live(self) -> dict[str, Any]:
        return live_collectors.collect_live_readback_receipt(
            self.prepared.deployment_contract,
            phase2_iam_plan=self.prepared.phase2_iam_plan,
            http_client=self.http,
            token_source=self.user_tokens,
            observed_at_unix_seconds=int(time.time()),
        )

    @staticmethod
    def _lifecycle_role_readbacks(
        live_receipt: Mapping[str, Any],
        plan: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        raw_roles = live_receipt["provider_readbacks"]["custom_roles"]
        rows = []
        for requirement in plan["custom_role_readback_contract"][
            "requirements"
        ]:
            raw = raw_roles[requirement["purpose"]]
            rows.append(
                {
                    "name": raw["name"],
                    "stage": raw["stage"],
                    "included_permissions": sorted(
                        raw["includedPermissions"]
                    ),
                    "deleted": raw.get("deleted", False),
                    "get_status": 200,
                    "readback_complete": True,
                }
            )
        return rows

    def install_phase2(
        self,
    ) -> phase2_lifecycle.Phase2IamInstalledCapability:
        if self.barrier is None:
            raise RuntimeError("token barrier has not completed")
        preinstall_live = self._collect_live()
        roles = self._lifecycle_role_readbacks(
            preinstall_live, self.prepared.phase2_iam_plan
        )
        # Mark immediately before the first Phase2 IAM mutation.  The owning
        # lifecycle already attempts cleanup, but an exception/unknown outcome
        # must still trigger an idempotent outer cleanup and zero readback.
        self.phase2_install_entered = True
        self.installed = phase2_lifecycle.install_step12b_phase2_iam(
            iam_admin=self.iam_admin,
            token_barrier_outcome=self.barrier,
            custom_role_readbacks=roles,
            observed_at_unix_seconds=int(time.time()),
            **self.phase2_context(),
        )
        return self.installed

    def collect_and_mint_external_preflight(
        self,
    ) -> tuple[dict[str, Any], Any, dict[str, Any]]:
        if (
            self.installed is None
            or self.created is None
            or self.barrier is None
            or self.source_provision is None
            or len(self.role_manifests) != 2
        ):
            raise RuntimeError("external preflight prerequisites are incomplete")
        phase2_readback = (
            phase2_lifecycle.get_step12b_phase2_exact_readback_receipt(
                self.installed,
                **self.phase2_context(),
            )
        )
        observed = max(
            int(time.time()), phase2_readback["observed_at_unix_seconds"]
        )
        prefix = live_collectors.collect_direct_v2_prefix_empty_receipt(
            self.prepared.deployment_contract,
            http_client=self.http,
            token_source=self.user_tokens,
            observed_at_unix_seconds=observed,
        )
        compute = live_collectors.collect_compute_absence_receipt(
            self.prepared.deployment_contract,
            http_client=self.http,
            token_source=self.user_tokens,
            observed_at_unix_seconds=observed,
        )
        live = live_collectors.collect_live_readback_receipt(
            self.prepared.deployment_contract,
            phase2_iam_plan=self.prepared.phase2_iam_plan,
            http_client=self.http,
            token_source=self.user_tokens,
            observed_at_unix_seconds=max(int(time.time()), observed),
        )
        kwargs = {
            "deployment_contract": self.prepared.deployment_contract,
            "candidate_payload_contract": (
                self.prepared.candidate_payload_contract
            ),
            "reference_payload_contract": (
                self.prepared.reference_payload_contract
            ),
            "controller_public_key_record": (
                self.prepared.controller_public_key_record
            ),
            "run_nonce": self.prepared.run_nonce,
            "package_provision_receipt": (
                self.prepared.package_provision_receipt
            ),
            "direct_v2_prefix_empty_receipt": prefix,
            "compute_absence_receipt": compute,
            "live_readback_receipt": live,
            "phase2_iam_plan": self.prepared.phase2_iam_plan,
            "phase2_iam_readback_receipt": phase2_readback,
            "controller_service_account_create_receipt": (
                self.created.receipt
            ),
            "token_barrier_outcome": self.barrier,
            "bootstrap_source_plan": self.prepared.source_plan,
            "validated_bootstrap_source_provision": (
                self.source_provision
            ),
            "role_bootstrap_manifests": self.role_manifests,
            "source_bytes": external_preflight.load_expected_source_bytes(),
            "current_profile_bytes": POLICY_REGISTRY_PATH.read_bytes(),
            "now_unix_seconds": int(time.time()),
        }
        gate = external_preflight.build_external_preflight_gate_receipt(
            **kwargs
        )
        capability = external_preflight.mint_validated_external_preflight(
            **kwargs
        )
        auth_receipt = (
            external_auth.get_validated_external_preflight_receipt(
                capability,
                deployment_contract=self.prepared.deployment_contract,
            )
        )
        return gate, capability, dict(auth_receipt)

    def prepare_role_launches(
        self,
        *,
        validated_external_preflight: Any,
        external_preflight_receipt: Mapping[str, Any],
    ) -> list[pair_controller.PreparedRoleLaunch]:
        if self.source_provision is None or len(self.role_manifests) != 2:
            raise RuntimeError("role source manifests are incomplete")
        now = int(time.time())
        package_generations = self.prepared.package_provision_receipt[
            "package_generations"
        ]
        verifier = payload_transport.RsaSha256ControllerTrustVerifier(
            self.prepared.controller_public_key_record
        )
        startup_script = startup_loader.build_startup_loader_source()
        payloads = (
            self.prepared.candidate_payload_contract,
            self.prepared.reference_payload_contract,
        )
        rows = []
        for position, (job_id, selected_payload) in enumerate(
            zip(
                self.prepared.deployment_contract["selected_job_ids"],
                payloads,
                strict=True,
            )
        ):
            authorization = external_auth.build_external_authorization(
                deployment_contract=self.prepared.deployment_contract,
                candidate_payload_contract=payloads[0],
                reference_payload_contract=payloads[1],
                controller_public_key_record=(
                    self.prepared.controller_public_key_record
                ),
                run_nonce=self.prepared.run_nonce,
                external_job_id=job_id,
                package_generations=package_generations,
                validated_external_preflight=validated_external_preflight,
                issued_unix_seconds=now,
                expires_unix_seconds=self.prepared.expires_at_unix_seconds,
                signer=self.prepared.signer,
                nonce=secrets.token_hex(32),
            )
            manifest = self.role_manifests[job_id]
            metadata = vm_prebootstrap.build_role_initial_metadata(
                startup_script=startup_script,
                deployment_contract=self.prepared.deployment_contract,
                selected_payload_contract=selected_payload,
                controller_public_key_record=(
                    self.prepared.controller_public_key_record
                ),
                authorization=authorization,
                package_generations=package_generations,
                external_preflight_receipt=external_preflight_receipt,
                run_nonce=self.prepared.run_nonce,
                external_job_id=job_id,
                role_bootstrap_manifest=manifest,
                verifier=verifier,
                now_unix_seconds=now,
            )
            rows.append(
                pair_controller.PreparedRoleLaunch(
                    external_job_id=job_id,
                    authorization=authorization,
                    package_generations=package_generations,
                    external_preflight_receipt=external_preflight_receipt,
                    initial_metadata_values=metadata,
                    startup_script_bytes=startup_script.encode("utf-8"),
                    bootstrap_role_manifest=manifest,
                    claim_nonce=secrets.token_hex(32),
                )
            )
        if len(rows) != 2 or position != 1:
            raise AssertionError("exact role launch pair changed")
        return rows

    def run_pair_controller(
        self,
        *,
        role_launches: Sequence[pair_controller.PreparedRoleLaunch],
    ) -> dict[str, Any]:
        if (
            self.barrier is None
            or self.installed is None
            or self.created is None
            or self.source_provision is None
        ):
            raise RuntimeError("pair controller prerequisites are incomplete")
        controller_compute = live_cloud.ExactPairComputeClient(
            token_source=self.barrier.token_source(),
            instance_names=self.instance_names,
            principal=self.prepared.deployment_contract[
                "controller_service_account"
            ]["principal"],
            credential_kind="fixed_nonrefreshing_controller",
            http_client=self.http,
            allow_insert=True,
        )
        self.pair_controller_entered = True
        now = int(time.time())
        return pair_controller.run_exact_pair_attempt0(
            deployment_contract=self.prepared.deployment_contract,
            candidate_payload_contract=(
                self.prepared.candidate_payload_contract
            ),
            reference_payload_contract=(
                self.prepared.reference_payload_contract
            ),
            controller_public_key_record=(
                self.prepared.controller_public_key_record
            ),
            run_nonce=self.prepared.run_nonce,
            source_plan=self.prepared.source_plan,
            source_provision=self.source_provision,
            role_launches=role_launches,
            phase2_iam_plan=self.prepared.phase2_iam_plan,
            phase2_installed=self.installed,
            controller_sa_created=self.created,
            controller_client=controller_compute,
            user_compute_client=self.user_compute,
            phase2_teardown=self.teardown,
            controller_sa_admin=self.controller_sa_admin,
            signer=self.prepared.signer,
            request_ids=[str(uuid.uuid4()) for _ in range(6)],
            now_unix_seconds=now,
            pair_release_issued_unix_seconds=now,
            pair_release_nonce=secrets.token_hex(32),
        )

    def receive_pair(self, *, destination_root: Path) -> dict[str, Any]:
        jobs = self.prepared.deployment_contract["remote_layout"]["jobs"]
        result_prefixes = [
            prefix
            for row in jobs
            for prefix in (
                row["result_prefix"],
                row["heartbeat_uris"][0].rsplit("/", 1)[0],
            )
        ]
        store = live_cloud.GenerationPinnedResultStore(
            result_prefixes=result_prefixes,
            http_client=self.http,
            user_token_source=self.user_tokens,
        )
        return result_receiver.wait_for_pair_and_receive(
            deployment_contract=self.prepared.deployment_contract,
            candidate_payload_contract=(
                self.prepared.candidate_payload_contract
            ),
            reference_payload_contract=(
                self.prepared.reference_payload_contract
            ),
            controller_public_key_record=(
                self.prepared.controller_public_key_record
            ),
            run_nonce=self.prepared.run_nonce,
            store=store,
            instance_observer=self.user_compute,
            destination_root=destination_root,
        )

    def remove_worker_bindings_after_done(
        self, *, pair_receive_receipt: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        if (
            self.installed is None
            or self.teardown.controller_zero is None
        ):
            raise RuntimeError("controller bindings are not proven zero")
        self.worker_zero = (
            phase2_lifecycle.remove_step12b_phase2_worker_bindings_final(
                iam_admin=self.iam_admin,
                installed=self.installed,
                controller_zero=self.teardown.controller_zero,
                completion_kind="pair_done_readback",
                completion_evidence_sha256=pair_receive_receipt[
                    "receipt_sha256"
                ],
                **self.phase2_context(),
            )
        )
        return dict(self.worker_zero.receipt)

    def ensure_compute_absent(self) -> Mapping[str, Any]:
        receipt = self.user_compute.cleanup_exact_instances_and_disks(
            instance_names=self.instance_names,
            disk_names=self.instance_names,
        )
        if (
            not isinstance(receipt, Mapping)
            or receipt.get("exact_cleanup_complete") is not True
            or receipt.get("instance_final_statuses") != [404, 404]
            or receipt.get("disk_final_statuses") != [404, 404]
        ):
            raise RuntimeError(
                "exact instance/disk GET-404 cleanup proof changed"
            )
        return receipt

    def _get_policy_allow_deleted_controller_zero(
        self, target: rest_iam.PolicyTarget
    ) -> tuple[Mapping[str, Any] | None, bool]:
        try:
            return self.iam_admin.get_policy(target), False
        except rest_iam.RestIamAdminError as error:
            if (
                target is rest_iam.PolicyTarget.CONTROLLER_SERVICE_ACCOUNT
                and error.status_code == 404
            ):
                # A deleted run-scoped controller service account has no IAM
                # policy surface.  Its exact policy GET-404 is therefore a
                # zero-membership readback, not a cleanup failure.
                return None, True
            raise

    def _verify_phase2_principals_absent(self) -> dict[str, Any]:
        plan = self.prepared.phase2_iam_plan
        bindings = [
            *plan["phase2_bindings"]["controller"],
            *plan["phase2_bindings"]["worker"],
        ]
        principals = {
            plan["principals"]["controller_principal"],
            plan["principals"]["worker_principal"],
        }
        targets = sorted({row["target"] for row in bindings})
        policy_shas: dict[str, str | None] = {}
        deleted_target_get404s: list[str] = []
        failed_targets: list[str] = []
        residual_targets: list[str] = []
        for target in targets:
            checked_target = rest_iam.PolicyTarget(target)
            try:
                policy, deleted = (
                    self._get_policy_allow_deleted_controller_zero(
                        checked_target
                    )
                )
            except BaseException:
                failed_targets.append(target)
                continue
            if deleted:
                policy_shas[target] = None
                deleted_target_get404s.append(target)
                continue
            if policy is None:
                failed_targets.append(target)
                continue
            raw_bindings = (
                policy["bindings"] if "bindings" in policy else []
            )
            if not isinstance(raw_bindings, list):
                failed_targets.append(target)
                continue
            malformed = False
            for raw in raw_bindings:
                members = raw.get("members") if isinstance(
                    raw, Mapping
                ) else None
                if not isinstance(members, list):
                    malformed = True
                    break
                if any(member in principals for member in members):
                    residual_targets.append(target)
            if malformed:
                failed_targets.append(target)
                continue
            policy_shas[target] = _canonical_sha256(dict(policy))
        if failed_targets:
            raise RuntimeError("Phase2 zero readback incomplete")
        if residual_targets:
            raise RuntimeError(
                "Phase2 principal binding remained after cleanup"
            )
        return _sealed(
            {
                "status": "phase2_controller_and_worker_principals_zero",
                "targets": targets,
                "policy_readback_sha256s": policy_shas,
                "deleted_target_get404s": deleted_target_get404s,
                "targeted_principal_binding_count": 0,
                "readback_complete": True,
                "access_token_stored": False,
                "authorization_header_stored": False,
                "current_profile_changed": False,
            }
        )

    def _cleanup_all_phase2_and_verify(
        self, *, failure_evidence_sha256: str
    ) -> dict[str, Any]:
        cleanup: Mapping[str, Any] | None = None
        try:
            cleanup = phase2_lifecycle.cleanup_step12b_phase2_iam_on_failure(
                iam_admin=self.iam_admin,
                failure_evidence_sha256=failure_evidence_sha256,
                **self.phase2_context(),
            )
        except BaseException:
            # An unknown mutation outcome is resolved only by the complete
            # policy readback below; never skip that readback.
            cleanup = None
        readback = self._verify_phase2_principals_absent()
        return _sealed(
            {
                "status": "phase2_idempotent_cleanup_and_zero_readback",
                "lifecycle_cleanup_returned": cleanup is not None,
                "lifecycle_cleanup_receipt_sha256": (
                    cleanup.get("receipt_sha256")
                    if cleanup is not None
                    else None
                ),
                "zero_readback_receipt_sha256": readback[
                    "receipt_sha256"
                ],
                "all_phase2_principals_zero": True,
                "current_profile_changed": False,
            }
        )

    def _cleanup_token_creator_residual(self) -> dict[str, Any]:
        binding = self.token_creator_binding
        if not self.token_barrier_entered or binding is None:
            return _sealed(
                {
                    "status": "token_barrier_not_entered",
                    "token_creator_member_count": 0,
                    "cloud_mutation_performed": False,
                    "current_profile_changed": False,
                }
            )
        result: Any | None = None
        try:
            result = self.iam_admin.remove_binding(
                binding.target,
                role=binding.role,
                member=binding.member,
                condition=binding.condition,
            )
        except BaseException:
            # Resolve an unknown remove outcome with the mandatory GET below.
            result = None
        changed = getattr(result, "changed", None)
        attempts = getattr(result, "attempts", None)
        if result is not None and (
            type(changed) is not bool
            or type(attempts) is not int
            or not 1 <= attempts <= 8
        ):
            raise RuntimeError("TokenCreator residual cleanup changed")
        verified = self._verify_token_creator_absent()
        return _sealed(
            {
                "status": "token_creator_residual_absent",
                "binding_role": binding.role,
                "binding_member": binding.member,
                "remove_call_returned": result is not None,
                "remove_changed": changed,
                "remove_attempts": attempts,
                "token_creator_member_count": 0,
                "zero_readback_receipt_sha256": verified[
                    "receipt_sha256"
                ],
                "readback_complete": True,
                "access_token_stored": False,
                "authorization_header_stored": False,
                "current_profile_changed": False,
            }
        )

    def _verify_token_creator_absent(self) -> dict[str, Any]:
        binding = self.token_creator_binding
        if not self.token_barrier_entered or binding is None:
            return _sealed(
                {
                    "status": "token_barrier_not_entered",
                    "token_creator_member_count": 0,
                    "readback_complete": True,
                    "cloud_mutation_performed": False,
                    "current_profile_changed": False,
                }
            )
        expected_present = [
            {
                "role": binding.role,
                "members": [binding.member],
                "condition": dict(binding.condition),
            }
        ]
        monotonic = getattr(self, "_monotonic", time.monotonic)
        sleeper = getattr(self, "_sleep", time.sleep)
        started = float(monotonic())
        observations: list[str] = []
        delays: list[float] = []
        stale_count = 0
        poll_count = 0
        policy_sha256: str | None = None
        deleted_target = False
        while True:
            poll_count += 1
            policy, deleted = (
                self._get_policy_allow_deleted_controller_zero(
                    binding.target
                )
            )
            if deleted:
                targeted: list[dict[str, Any]] = []
                deleted_target = True
            else:
                if policy is None:
                    raise AssertionError(
                        "live IAM policy unexpectedly missing"
                    )
                raw_bindings = (
                    policy["bindings"]
                    if "bindings" in policy
                    else []
                )
                if not isinstance(raw_bindings, list):
                    raise RuntimeError(
                        "TokenCreator cleanup policy changed"
                    )
                targeted = []
                for raw in raw_bindings:
                    members = raw.get("members") if isinstance(
                        raw, Mapping
                    ) else None
                    if not isinstance(members, list):
                        raise RuntimeError(
                            "TokenCreator cleanup members changed"
                        )
                    if binding.member in members:
                        targeted.append(dict(raw))
                policy_sha256 = _canonical_sha256(dict(policy))
            observations.append(_canonical_sha256(targeted))
            if targeted == []:
                break
            if targeted != expected_present:
                raise RuntimeError(
                    "unexpected TokenCreator residual binding"
                )
            stale_count += 1
            elapsed = float(monotonic()) - started
            if elapsed >= token_barrier.MAX_PROPAGATION_SECONDS:
                raise RuntimeError(
                    "TokenCreator zero readback propagation timeout"
                )
            delay = float(
                token_barrier.REVOKE_READBACK_DELAYS_SECONDS[
                    min(
                        stale_count - 1,
                        len(
                            token_barrier.REVOKE_READBACK_DELAYS_SECONDS
                        )
                        - 1,
                    )
                ]
            )
            delay = min(
                delay,
                token_barrier.MAX_PROPAGATION_SECONDS - elapsed,
            )
            if delay <= 0:
                raise RuntimeError(
                    "TokenCreator zero readback propagation timeout"
                )
            delays.append(delay)
            sleeper(delay)
        elapsed = float(monotonic()) - started
        return _sealed(
            {
                "status": (
                    "token_creator_zero_deleted_controller_target"
                    if deleted_target
                    else "token_creator_residual_absent"
                ),
                "binding_role": binding.role,
                "binding_member": binding.member,
                "token_creator_member_count": 0,
                "target_policy_get_status": (
                    404 if deleted_target else 200
                ),
                "policy_readback_sha256": policy_sha256,
                "poll_count": poll_count,
                "stale_present_count": stale_count,
                "observation_sha256s": observations,
                "sleep_delays_seconds": delays,
                "elapsed_seconds": round(elapsed, 6),
                "maximum_seconds": (
                    token_barrier.MAX_PROPAGATION_SECONDS
                ),
                "second_add_performed": False,
                "controller_token_reminted": False,
                "readback_complete": True,
                "access_token_stored": False,
                "authorization_header_stored": False,
                "current_profile_changed": False,
            }
        )

    def collect_final_zero_readback(self) -> dict[str, Any]:
        """Independent success-path zero proof after every teardown."""

        phase2 = self._verify_phase2_principals_absent()
        token = self._verify_token_creator_absent()
        if self.controller_sa_admin.get() is not None:
            raise RuntimeError(
                "controller service account final GET was not 404"
            )
        controller = _sealed(
            {
                "status": (
                    "run_scoped_controller_service_account_get404"
                ),
                "final_get_status": 404,
                "readback_verified": True,
                "current_profile_changed": False,
            }
        )
        compute = (
            self.user_compute.verify_exact_instances_and_disks_absent(
                instance_names=self.instance_names,
                disk_names=self.instance_names,
            )
        )
        if (
            compute.get("all_four_targets_get404_verified") is not True
            or compute.get("instance_final_statuses") != [404, 404]
            or compute.get("disk_final_statuses") != [404, 404]
        ):
            raise RuntimeError("final exact compute zero proof changed")
        return _sealed(
            {
                "status": "all_live_surfaces_independently_zero",
                "phase2_zero_receipt_sha256": phase2["receipt_sha256"],
                "token_creator_zero_receipt_sha256": token[
                    "receipt_sha256"
                ],
                "controller_service_account_zero_receipt_sha256": (
                    controller["receipt_sha256"]
                ),
                "compute_zero_receipt_sha256": compute[
                    "receipt_sha256"
                ],
                "phase2_controller_and_worker_principals_zero": True,
                "token_creator_zero": True,
                "controller_service_account_get404": True,
                "instance_final_statuses": [404, 404],
                "disk_final_statuses": [404, 404],
                "readback_only": True,
                "cloud_mutation_performed": False,
                "current_profile_changed": False,
            }
        )

    def _cleanup_controller_sa_and_verify(self) -> dict[str, Any]:
        cleanup: Mapping[str, Any] | None = None
        try:
            candidate = (
                self.controller_sa_admin.cleanup_delete_if_present()
            )
            if isinstance(candidate, Mapping):
                cleanup = candidate
        except BaseException:
            # Resolve an unknown delete outcome only with the mandatory GET.
            cleanup = None
        if self.controller_sa_admin.get() is not None:
            raise RuntimeError(
                "controller service account remained after cleanup"
            )
        return _sealed(
            {
                "status": (
                    "controller_service_account_absent_get404_verified"
                ),
                "cleanup_call_returned": cleanup is not None,
                "cleanup_receipt_sha256": (
                    cleanup.get("receipt_sha256")
                    if cleanup is not None
                    else None
                ),
                "final_get_status": 404,
                "readback_verified": True,
                "current_profile_changed": False,
            }
        )

    def _cleanup_compute_and_verify(self) -> dict[str, Any]:
        cleanup = self.ensure_compute_absent()
        if (
            not isinstance(cleanup, Mapping)
            or cleanup.get("exact_cleanup_complete") is not True
            or cleanup.get("instance_final_statuses") != [404, 404]
            or cleanup.get("disk_final_statuses") != [404, 404]
        ):
            raise RuntimeError(
                "exact instance/disk failure cleanup proof changed"
            )
        return _sealed(
            {
                "status": "exact_instances_and_disks_get404_verified",
                "cleanup_receipt_sha256": cleanup.get("receipt_sha256"),
                "instance_final_statuses": [404, 404],
                "disk_final_statuses": [404, 404],
                "readback_verified": True,
                "current_profile_changed": False,
            }
        )

    def cleanup_failure(
        self, *, failure_evidence_sha256: str
    ) -> dict[str, Any]:
        """Idempotent recovery with mandatory zero/GET-404 readbacks."""

        records: list[dict[str, Any]] = []

        def record(operation: str, callback: Any) -> None:
            try:
                value = callback()
                digest = (
                    value.get("receipt_sha256")
                    if isinstance(value, Mapping)
                    else None
                )
                records.append(
                    {
                        "operation": operation,
                        "completed": True,
                        "receipt_sha256": (
                            digest
                            if isinstance(digest, str)
                            else _canonical_sha256(dict(value))
                        ),
                    }
                )
            except BaseException:
                records.append(
                    {
                        "operation": operation,
                        "completed": False,
                        "receipt_sha256": None,
                    }
                )

        if self.phase2_install_entered:
            record(
                "phase2_controller_then_worker_zero",
                lambda: self._cleanup_all_phase2_and_verify(
                    failure_evidence_sha256=failure_evidence_sha256,
                ),
            )
        else:
            records.append(
                {
                    "operation": "phase2_not_installed",
                    "completed": True,
                    "receipt_sha256": None,
                }
            )
        if self.token_barrier_entered:
            record(
                "token_creator_residual_zero",
                self._cleanup_token_creator_residual,
            )
        else:
            records.append(
                {
                    "operation": "token_barrier_not_entered",
                    "completed": True,
                    "receipt_sha256": None,
                }
            )
        record(
            "controller_service_account_absent",
            self._cleanup_controller_sa_and_verify,
        )
        record(
            "exact_instances_and_disks_absent",
            self._cleanup_compute_and_verify,
        )
        mandatory_complete = all(
            record_row["completed"] for record_row in records
        )
        body = {
            "schema": FAILURE_RECEIPT_SCHEMA,
            "status": (
                "mandatory_failure_cleanup_verified"
                if mandatory_complete
                else "mandatory_failure_cleanup_incomplete"
            ),
            "failure_evidence_sha256": failure_evidence_sha256,
            "records": records,
            "records_sha256": _canonical_sha256(records),
            "cleanup_order": [
                "phase2_controller_then_worker_zero",
                "token_creator_residual_zero",
                "controller_service_account_absent",
                "exact_instances_and_disks_absent",
            ],
            "mandatory_verification_complete": mandatory_complete,
            "phase2_principals_zero_verified": records[0]["completed"],
            "token_creator_zero_verified": records[1]["completed"],
            "controller_service_account_get404_verified": records[2][
                "completed"
            ],
            "exact_instances_and_disks_get404_verified": records[3][
                "completed"
            ],
            "automatic_retry_performed": False,
            "attempt1_authorized": False,
            "third_vm_authorized": False,
            "current_profile_changed": False,
        }
        return _sealed(body)


def _artifact_summary(prepared: PreparedRun) -> dict[str, Any]:
    deployment = prepared.deployment_contract
    return _sealed(
        {
            "schema": RUNNER_RECEIPT_SCHEMA,
            "status": "exact_pair_offline_contracts_frozen",
            "deployment_contract_sha256": deployment[
                "deployment_contract_sha256"
            ],
            "source_plan_sha256": prepared.source_plan[
                "source_plan_sha256"
            ],
            "phase2_iam_plan_sha256": prepared.phase2_iam_plan[
                "plan_sha256"
            ],
            "package_provision_receipt_sha256": (
                prepared.package_provision_receipt["receipt_sha256"]
            ),
            "controller_public_key_sha256": _canonical_sha256(
                prepared.controller_public_key_record
            ),
            "runtime_source_file_count": len(
                prepared.runtime_source_files
            ),
            "runtime_source_paths": sorted(prepared.runtime_source_files),
            "instance_names": [
                row["instance_name"] for row in deployment["instances"]
            ],
            "source_roles": list(deployment["source_roles"]),
            "vm_count": 2,
            "attempt_index": 0,
            "attempt1_authorized": False,
            "third_vm_authorized": False,
            "automatic_retry_authorized": False,
            "private_key_stored": False,
            "access_token_stored": False,
            "current_profile_changed": False,
        }
    )


def execute_prepared(
    *,
    prepared: PreparedRun,
    backend: Any,
    output_root: str | Path,
) -> dict[str, Any]:
    """Execute the high-level one-shot sequence using an injected backend."""

    root = _safe_output_root(output_root)
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"Step12b output root is not fresh: {root}")
    if not root.parent.is_dir() or root.parent.is_symlink():
        raise FileNotFoundError("Step12b output parent must already exist")
    _assert_policy_registry_unchanged(prepared.policy_registry_sha256)
    root.mkdir()
    _exclusive_write_json(
        root / "offline_contract_summary.json",
        _artifact_summary(prepared),
    )
    _exclusive_write_json(
        root / "deployment_contract.json",
        dict(prepared.deployment_contract),
    )
    _exclusive_write_json(
        root / "controller_public_key.json",
        dict(prepared.controller_public_key_record),
    )
    _exclusive_write_json(
        root / "phase2_iam_plan.json",
        dict(prepared.phase2_iam_plan),
    )
    stage = "source_provision"
    pair_controller_entered = False
    try:
        source_provision, manifests = backend.provision_sources()
        source_receipt = source_provision.receipt()
        _exclusive_write_json(
            root / "bootstrap_source_provision_receipt.json",
            source_receipt,
        )
        _exclusive_write_json(
            root / "role_bootstrap_manifest_digests.json",
            _sealed(
                {
                    "manifests": [
                        {
                            "external_job_id": job_id,
                            "source_role": manifests[job_id][
                                "source_role"
                            ],
                            "role_manifest_sha256": manifests[job_id][
                                "role_manifest_sha256"
                            ],
                        }
                        for job_id in prepared.deployment_contract[
                            "selected_job_ids"
                        ]
                    ],
                    "manifest_count": 2,
                    "opponent_payload_embedded_in_role_metadata": False,
                }
            ),
        )
        stage = "controller_service_account_create"
        created = backend.create_controller_service_account()
        _exclusive_write_json(
            root / "controller_service_account_create_receipt.json",
            dict(created.receipt),
        )
        stage = "token_barrier"
        barrier = backend.run_token_barrier()
        _exclusive_write_json(
            root / "token_barrier_receipt.json", dict(barrier.receipt)
        )
        stage = "phase2_install"
        installed = backend.install_phase2()
        _exclusive_write_json(
            root / "phase2_install_receipt.json",
            dict(installed.install_receipt),
        )
        stage = "postmutation_external_preflight"
        gate, capability, auth_preflight_receipt = (
            backend.collect_and_mint_external_preflight()
        )
        _exclusive_write_json(
            root / "external_preflight_gate_receipt.json", gate
        )
        _exclusive_write_json(
            root / "external_authorization_preflight_receipt.json",
            auth_preflight_receipt,
        )
        stage = "role_launch_prepare"
        launches = backend.prepare_role_launches(
            validated_external_preflight=capability,
            external_preflight_receipt=auth_preflight_receipt,
        )
        launch_summary = _sealed(
            {
                "external_job_ids": [
                    row.external_job_id for row in launches
                ],
                "authorization_sha256s": [
                    external_auth.canonical_sha256(row.authorization)
                    for row in launches
                ],
                "initial_metadata_sha256s": [
                    _canonical_sha256(dict(row.initial_metadata_values))
                    for row in launches
                ],
                "launch_count": 2,
                "authorization_body_stored": False,
                "metadata_body_stored": False,
                "private_key_stored": False,
                "access_token_stored": False,
            }
        )
        _exclusive_write_json(
            root / "role_launch_summary.json", launch_summary
        )
        stage = "exact_pair_attempt0"
        pair_controller_entered = True
        pair_receipt = backend.run_pair_controller(role_launches=launches)
        _exclusive_write_json(
            root / "pair_controller_receipt.json", pair_receipt
        )
        stage = "pair_result_receive"
        receive = backend.receive_pair(
            destination_root=root / "received"
        )
        _exclusive_write_json(
            root / "pair_receive_receipt.json", receive
        )
        stage = "worker_iam_cleanup"
        worker_cleanup = backend.remove_worker_bindings_after_done(
            pair_receive_receipt=receive
        )
        _exclusive_write_json(
            root / "worker_iam_cleanup_receipt.json",
            worker_cleanup,
        )
        stage = "compute_absence"
        compute_cleanup = backend.ensure_compute_absent()
        _exclusive_write_json(
            root / "final_compute_absence_receipt.json",
            compute_cleanup,
        )
        stage = "final_independent_zero_readback"
        final_zero = backend.collect_final_zero_readback()
        _exclusive_write_json(
            root / "final_independent_zero_readback_receipt.json",
            final_zero,
        )
        final_profile_sha = _assert_policy_registry_unchanged(
            prepared.policy_registry_sha256
        )
        body = {
            "schema": RUNNER_RECEIPT_SCHEMA,
            "status": (
                "candidate_and_reference_attempt0_received_and_cleaned"
            ),
            "deployment_contract_sha256": prepared.deployment_contract[
                "deployment_contract_sha256"
            ],
            "pair_controller_receipt_sha256": pair_receipt[
                "receipt_sha256"
            ],
            "pair_receive_receipt_sha256": receive["receipt_sha256"],
            "worker_iam_cleanup_receipt_sha256": worker_cleanup[
                "receipt_sha256"
            ],
            "compute_absence_receipt_sha256": compute_cleanup[
                "receipt_sha256"
            ],
            "final_independent_zero_readback_receipt_sha256": (
                final_zero["receipt_sha256"]
            ),
            "final_independent_zero_readback_verified": True,
            "instance_names": [
                row["instance_name"]
                for row in prepared.deployment_contract["instances"]
            ],
            "vm_count": 2,
            "attempt_index": 0,
            "attempt1_authorized": False,
            "third_vm_authorized": False,
            "automatic_retry_performed": False,
            "old_package_write_count": 0,
            "old_result_write_count": 0,
            "result_object_count": 88,
            "policy_registry_sha256_before": (
                prepared.policy_registry_sha256
            ),
            "policy_registry_sha256_after": final_profile_sha,
            "private_key_stored": False,
            "access_token_stored": False,
            "current_profile_changed": False,
        }
        final = _sealed(body)
        _exclusive_write_json(root / "FINAL.json", final)
        return final
    except BaseException as primary:
        evidence = _canonical_sha256(
            {
                "deployment_contract_sha256": prepared.deployment_contract[
                    "deployment_contract_sha256"
                ],
                "failure_stage": stage,
                "exception_type": type(primary).__name__,
            }
        )
        cleanup: Mapping[str, Any] | None = None
        try:
            # The pair core's cleanup is intentionally followed by this
            # idempotent recovery.  It never launches or retries; it resolves
            # unknown core-cleanup outcomes with mandatory zero/404 readbacks.
            cleanup = backend.cleanup_failure(
                failure_evidence_sha256=evidence
            )
        except BaseException:
            # Preserve the primary failure and write an honest incomplete
            # cleanup receipt.  The live backend normally returns a receipt
            # with per-surface verification even when one surface is unproven.
            cleanup = None
        validated_cleanup: dict[str, Any] | None = None
        if isinstance(cleanup, Mapping):
            try:
                validated_cleanup = _validated_sealed_receipt(
                    cleanup,
                    label="outer failure cleanup",
                )
                _exclusive_write_json(
                    root / "outer_failure_cleanup_receipt.json",
                    validated_cleanup,
                )
            except BaseException:
                validated_cleanup = None
        pair_failure_receipt: dict[str, Any] | None = None
        if (
            pair_controller_entered
            and isinstance(
                primary, pair_controller.ExactPairCloudControllerError
            )
        ):
            try:
                pair_failure_receipt = _validated_sealed_receipt(
                    primary.receipt,
                    label="pair controller failure",
                )
                _exclusive_write_json(
                    root / "pair_controller_failure_receipt.json",
                    pair_failure_receipt,
                )
            except BaseException:
                pair_failure_receipt = None
        token_failure_receipt: dict[str, Any] | None = None
        if isinstance(primary, token_barrier.TokenBarrierFailure):
            try:
                token_failure_receipt = _validated_sealed_receipt(
                    primary.receipt,
                    label="token barrier failure",
                )
                _exclusive_write_json(
                    root / "token_barrier_failure_receipt.json",
                    token_failure_receipt,
                )
            except BaseException:
                token_failure_receipt = None
        phase2_failure_receipt: dict[str, Any] | None = None
        if isinstance(primary, phase2_lifecycle.Phase2IamLifecycleError):
            # The phase2 sealed receipt carries only non-sensitive detail
            # (failure_reason plus an underlying step11 error code/status/
            # operation); persist it so a set/readback failure is diagnosable.
            try:
                phase2_failure_receipt = _validated_sealed_receipt(
                    primary.receipt,
                    label="phase2 install failure",
                )
                _exclusive_write_json(
                    root / "phase2_install_failure_receipt.json",
                    phase2_failure_receipt,
                )
            except BaseException:
                phase2_failure_receipt = None
        profile_after = _policy_registry_sha256()
        outer_cleanup_verified = (
            validated_cleanup is not None
            and validated_cleanup.get(
                "mandatory_verification_complete"
            )
            is True
        )
        unverified_operations = (
            [
                row.get("operation")
                for row in validated_cleanup.get("records", [])
                if isinstance(row, Mapping)
                and row.get("completed") is not True
                and isinstance(row.get("operation"), str)
            ]
            if validated_cleanup is not None
            else []
        )
        outer_cleanup_path = root / "outer_failure_cleanup_receipt.json"
        pair_failure_path = root / "pair_controller_failure_receipt.json"
        token_failure_path = root / "token_barrier_failure_receipt.json"
        failure_body = {
            "schema": FAILURE_RECEIPT_SCHEMA,
            "status": "step12b_exact_pair_stopped_without_retry",
            "deployment_contract_sha256": prepared.deployment_contract[
                "deployment_contract_sha256"
            ],
            "failure_stage": stage,
            "exception_type": type(primary).__name__,
            "failure_evidence_sha256": evidence,
            "pair_controller_owned_cleanup": (
                pair_controller_entered
                and isinstance(
                    primary,
                    pair_controller.ExactPairCloudControllerError,
                )
            ),
            "outer_cleanup_always_attempted": True,
            "outer_cleanup_call_returned": isinstance(cleanup, Mapping),
            "outer_cleanup_verified": outer_cleanup_verified,
            "outer_failure_cleanup_receipt_path": (
                "outer_failure_cleanup_receipt.json"
                if validated_cleanup is not None
                else None
            ),
            "outer_failure_cleanup_receipt_exists": (
                validated_cleanup is not None
                and outer_cleanup_path.is_file()
            ),
            "outer_cleanup_unverified_operations": unverified_operations,
            "outer_cleanup_receipt_sha256": (
                validated_cleanup.get("receipt_sha256")
                if validated_cleanup is not None
                else None
            ),
            "pair_controller_failure_receipt_path": (
                "pair_controller_failure_receipt.json"
                if pair_failure_receipt is not None
                else None
            ),
            "pair_controller_failure_receipt_exists": (
                pair_failure_receipt is not None
                and pair_failure_path.is_file()
            ),
            "pair_controller_failure_receipt_sha256": (
                pair_failure_receipt.get("receipt_sha256")
                if pair_failure_receipt is not None
                else None
            ),
            "token_barrier_failure_receipt_path": (
                "token_barrier_failure_receipt.json"
                if token_failure_receipt is not None
                else None
            ),
            "token_barrier_failure_receipt_exists": (
                token_failure_receipt is not None
                and token_failure_path.is_file()
            ),
            "token_barrier_failure_receipt_sha256": (
                token_failure_receipt.get("receipt_sha256")
                if token_failure_receipt is not None
                else None
            ),
            "automatic_retry_performed": False,
            "attempt1_authorized": False,
            "third_vm_authorized": False,
            "policy_registry_sha256_before": (
                prepared.policy_registry_sha256
            ),
            "policy_registry_sha256_after": profile_after,
            "current_profile_changed": (
                profile_after != prepared.policy_registry_sha256
            ),
            "exception_message_stored": False,
            "private_key_stored": False,
            "access_token_stored": False,
        }
        try:
            _exclusive_write_json(
                root / "FAILURE.json", _sealed(failure_body)
            )
        except BaseException:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Step12b direct-v2 exact candidate/reference attempt0 runner"
        )
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="enable the one-shot cloud mutation path",
    )
    parser.add_argument(
        "--confirm",
        default="",
        help="exact execution confirmation; ignored in dry-run mode",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="fresh audit output root (execute mode only)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.execute and args.confirm != EXECUTION_CONFIRMATION:
        raise PermissionError(
            "exact Step12b execution confirmation is missing"
        )
    prepared = prepare_run()
    if not args.execute:
        sys.stdout.write(
            json.dumps(
                dry_run_receipt(prepared),
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        return 0
    backend = LiveExecutionBackend(prepared)
    receipt = execute_prepared(
        prepared=prepared,
        backend=backend,
        output_root=args.output_root,
    )
    sys.stdout.write(
        json.dumps(
            {
                "status": receipt["status"],
                "receipt_sha256": receipt["receipt_sha256"],
                "output_root": str(Path(args.output_root).resolve()),
                "instance_names": receipt["instance_names"],
                "attempt_index": 0,
                "automatic_retry_performed": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
