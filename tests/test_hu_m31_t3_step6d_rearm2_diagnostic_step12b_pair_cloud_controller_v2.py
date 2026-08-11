from __future__ import annotations

import copy
import hashlib
import json
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_cloud_worker_adapter as adapter,
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
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_pair_cloud_controller_v2
    as subject,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_pair_release_v2
    as pair_release,
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
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_run_scoped_controller_sa_v2
    as controller_sa,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_vm_metadata_v2
    as vm_metadata,
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
RUN_NONCE = "a1" * 32
ISSUED = 1_900_200_000
EXPIRES = ISSUED + 3_600
NOW = ISSUED + 120
RELEASE_NONCE = "ef" * 32
RAW_TOKEN = "never-serialize-this-controller-token"
REQUEST_IDS = tuple(
    str(uuid.UUID(int=position + 1, version=4))
    for position in range(6)
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


class _SourceStore:
    def __init__(self) -> None:
        self.objects: dict[str, tuple[int, bytes]] = {}

    def list_objects(self, *, prefix: str) -> list[str]:
        return []

    def conditional_create(
        self,
        *,
        uri: str,
        content: bytes,
        if_generation_match: int,
    ) -> dict[str, Any]:
        assert if_generation_match == 0
        generation = 1_900_200_000_000_000 + len(self.objects)
        self.objects[uri] = (generation, content)
        return {
            "uri": uri,
            "generation": generation,
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
            "created": True,
        }

    def generation_pinned_get(
        self, *, uri: str, generation: int
    ) -> bytes:
        stored_generation, content = self.objects[uri]
        assert stored_generation == generation
        return content


@pytest.fixture(scope="module")
def context() -> dict[str, Any]:
    signer = step11_controller.generate_ephemeral_controller_key(
        key_size=2_048
    )
    public_record = dict(signer.public_record)
    stage1 = _read_json(STEP11_ROOT / "transport_contract.json")
    done = _read_json(STEP11_ROOT / "late_done_envelope.json")
    stage1_receive = adapter.build_receive(
        stage1["adapter_preview"], done_records=[done]
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
    runtime_sources = {
        path: (REPO_ROOT / "src" / Path(path)).read_text(
            encoding="utf-8"
        )
        for path in bootstrap_source.REQUIRED_RUNTIME_SOURCE_PATHS
    }
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
        run_nonce=RUN_NONCE,
        bootstrap_source_content_binding=content_binding,
    )
    preflight_receipt = (
        external_auth._build_external_preflight_receipt_record(
            deployment_contract=deployment,
            readonly_preflight_receipt_sha256=hashlib.sha256(
                b"readonly"
            ).hexdigest(),
            iam_capacity_gate_receipt_sha256=hashlib.sha256(
                b"iam-capacity"
            ).hexdigest(),
            package_provision_receipt_sha256=hashlib.sha256(
                b"package"
            ).hexdigest(),
            deployment_source_sha256=hashlib.sha256(
                b"deployment"
            ).hexdigest(),
            alias_bridge_source_sha256=hashlib.sha256(
                b"bridge"
            ).hexdigest(),
            prebootstrap_source_sha256=hashlib.sha256(
                b"prebootstrap"
            ).hexdigest(),
            startup_source_sha256=hashlib.sha256(
                b"startup"
            ).hexdigest(),
            controller_source_sha256=hashlib.sha256(
                b"controller"
            ).hexdigest(),
        )
    )
    validated_preflight = (
        external_auth._mint_validated_external_preflight_after_gate(
            preflight_receipt,
            deployment_contract=deployment,
            gate_validation_seal=external_auth._VALIDATED_PREFLIGHT_SEAL,
        )
    )
    inventory = payloads[0]["remote_layout"]["package_inventory"][
        "records"
    ]
    generations = {
        row["uri"]: 1_900_200_100_000_000 + position
        for position, row in enumerate(inventory)
    }
    authorizations = [
        external_auth.build_external_authorization(
            deployment_contract=deployment,
            candidate_payload_contract=payloads[0],
            reference_payload_contract=payloads[1],
            controller_public_key_record=public_record,
            run_nonce=RUN_NONCE,
            external_job_id=job_id,
            package_generations=generations,
            validated_external_preflight=validated_preflight,
            issued_unix_seconds=ISSUED,
            expires_unix_seconds=EXPIRES,
            signer=signer,
            nonce=f"{position + 11:02x}" * 32,
        )
        for position, job_id in enumerate(
            deployment["selected_job_ids"]
        )
    ]
    source_plan = bootstrap_source.build_bootstrap_source_plan(
        deployment_contract=deployment,
        candidate_payload_contract=payloads[0],
        reference_payload_contract=payloads[1],
        controller_public_key_record=public_record,
        run_nonce=RUN_NONCE,
        runtime_source_files=runtime_sources,
    )
    store = _SourceStore()
    source_provision = bootstrap_source.provision_bootstrap_sources(
        source_plan=source_plan,
        deployment_contract=deployment,
        candidate_payload_contract=payloads[0],
        reference_payload_contract=payloads[1],
        controller_public_key_record=public_record,
        run_nonce=RUN_NONCE,
        prefix_observer=store,
        writer=store,
        reader=store,
    )
    manifests = [
        bootstrap_source.build_role_bootstrap_manifest(
            source_plan=source_plan,
            validated_provision=source_provision,
            deployment_contract=deployment,
            candidate_payload_contract=payloads[0],
            reference_payload_contract=payloads[1],
            controller_public_key_record=public_record,
            run_nonce=RUN_NONCE,
            external_job_id=job_id,
        )
        for job_id in deployment["selected_job_ids"]
    ]
    phase2_plan = phase2_iam.build_step12b_phase2_iam_plan(
        deployment,
        candidate_payload_contract=payloads[0],
        reference_payload_contract=payloads[1],
        controller_public_key_record=public_record,
        run_nonce=RUN_NONCE,
        issued_at_unix_seconds=ISSUED,
        expires_at_unix_seconds=EXPIRES,
    )
    custom_role_readbacks = [
        {
            "name": row["name"],
            "stage": row["stage"],
            "included_permissions": list(row["included_permissions"]),
            "deleted": False,
            "get_status": 200,
            "readback_complete": True,
        }
        for row in phase2_plan["custom_role_readback_contract"][
            "requirements"
        ]
    ]
    binding_readbacks = [
        {
            key: copy.deepcopy(row[key])
            for key in (
                "purpose",
                "target",
                "resource",
                "role",
                "member",
                "condition",
                "binding_sha256",
            )
        }
        | {
            "member_occurrences": 1,
            "readback_complete": True,
        }
        for row in [
            *phase2_plan["phase2_bindings"]["controller"],
            *phase2_plan["phase2_bindings"]["worker"],
        ]
    ]
    exact_readback = (
        phase2_iam.build_step12b_phase2_iam_readback_receipt(
            phase2_plan,
            deployment_contract=deployment,
            candidate_payload_contract=payloads[0],
            reference_payload_contract=payloads[1],
            controller_public_key_record=public_record,
            run_nonce=RUN_NONCE,
            observed_at_unix_seconds=ISSUED + 30,
            custom_role_readbacks=custom_role_readbacks,
            binding_readbacks=binding_readbacks,
            unexpected_targeted_bindings=[],
        )
    )
    install_body = {
        "schema": phase2_lifecycle.INSTALL_RECEIPT_SCHEMA,
        "exact_readback_receipt_sha256": exact_readback[
            "receipt_sha256"
        ],
    }
    install_receipt = {
        **install_body,
        "receipt_sha256": phase2_iam.canonical_sha256(install_body),
    }
    installed = phase2_lifecycle.Phase2IamInstalledCapability(
        deployment_contract_sha256=deployment[
            "deployment_contract_sha256"
        ],
        phase2_iam_plan_sha256=phase2_plan["plan_sha256"],
        token_barrier_receipt_sha256="1" * 64,
        exact_readback_receipt_sha256=exact_readback[
            "receipt_sha256"
        ],
        install_receipt=install_receipt,
        _exact_readback_receipt_bytes=phase2_iam.canonical_bytes(
            exact_readback
        ),
        _seal=phase2_lifecycle._INSTALLED_SEAL,
    )
    create_body = {
        "schema": controller_sa.SCHEMA,
        "status": "run_scoped_controller_service_account_created",
        "project": deployment["controller_service_account"]["project"],
        "account_id": deployment["controller_service_account"][
            "account_id"
        ],
        "email": deployment["controller_service_account"]["email"],
        "provider": {
            "name": (
                f"projects/{deployment['controller_service_account']['project']}"
                "/serviceAccounts/"
                f"{deployment['controller_service_account']['email']}"
            ),
            "project_id": deployment["controller_service_account"][
                "project"
            ],
            "unique_id": "123456789012345678901",
            "email": deployment["controller_service_account"]["email"],
            "disabled": False,
        },
        "create_call_count": 1,
        "create_retry_count": 0,
        "readback_verified": True,
        "cloud_mutation_performed": True,
    }
    create_receipt = {
        **create_body,
        "receipt_sha256": controller_sa.canonical_sha256(create_body),
    }
    created = type("CreatedCapability", (), {})()
    created.receipt = create_receipt
    return {
        "signer": signer,
        "public_record": public_record,
        "payloads": payloads,
        "deployment": deployment,
        "preflight_receipt": preflight_receipt,
        "generations": generations,
        "authorizations": authorizations,
        "source_plan": source_plan,
        "source_provision": source_provision,
        "manifests": manifests,
        "phase2_plan": phase2_plan,
        "installed": installed,
        "created": created,
    }


class _MetadataAdapter:
    def validate_initial(
        self,
        *,
        values: Mapping[str, str],
        startup_script_bytes: bytes,
        role_manifest: Mapping[str, Any],
    ) -> dict[str, Any]:
        assert values[vm_metadata.STARTUP_KEY].encode() == startup_script_bytes
        assert values[subject.ROLE_BOOTSTRAP_MANIFEST_KEY] == (
            bootstrap_source.canonical_bytes(role_manifest).decode("ascii")
        )
        return {
            "metadata_budget_receipt_sha256": subject.canonical_sha256(
                dict(values)
            )
        }

    def add_claim(
        self,
        *,
        initial_values: Mapping[str, str],
        claim_value: str,
    ) -> tuple[dict[str, str], dict[str, Any]]:
        values = {
            **dict(initial_values),
            vm_metadata.POSTCREATE_CLAIM_KEY: claim_value,
        }
        return values, {
            "metadata_budget_receipt_sha256": subject.canonical_sha256(
                values
            )
        }

    def add_release(
        self,
        *,
        initial_values: Mapping[str, str],
        claim_value: str,
        release_value: str,
    ) -> tuple[dict[str, str], dict[str, Any]]:
        values = {
            **dict(initial_values),
            vm_metadata.POSTCREATE_CLAIM_KEY: claim_value,
            vm_metadata.PAIR_RELEASE_KEY: release_value,
        }
        return values, {
            "metadata_budget_receipt_sha256": subject.canonical_sha256(
                values
            )
        }


def _role_launches(
    context: dict[str, Any],
) -> list[subject.PreparedRoleLaunch]:
    rows = []
    for position, (job_id, instance, authorization, manifest) in enumerate(
        zip(
            context["deployment"]["selected_job_ids"],
            context["deployment"]["instances"],
            context["authorizations"],
            context["manifests"],
            strict=True,
        )
    ):
        startup = (
            "#!/usr/bin/env python3\n"
            f"# {job_id}\n"
            "print('bootstrap')\n"
        ).encode()
        values = {
            vm_metadata.STARTUP_KEY: startup.decode(),
            vm_metadata.EXTERNAL_AUTHORIZATION_KEY: (
                external_auth.canonical_bytes(authorization).decode("ascii")
            ),
            vm_metadata.RUN_NONCE_KEY: RUN_NONCE,
            vm_metadata.EXTERNAL_JOB_ID_KEY: job_id,
            vm_metadata.SOURCE_ROLE_KEY: instance["source_role"],
            subject.ROLE_BOOTSTRAP_MANIFEST_KEY: (
                bootstrap_source.canonical_bytes(manifest).decode("ascii")
            ),
        }
        rows.append(
            subject.PreparedRoleLaunch(
                external_job_id=job_id,
                authorization=authorization,
                package_generations=context["generations"],
                external_preflight_receipt=context[
                    "preflight_receipt"
                ],
                initial_metadata_values=values,
                startup_script_bytes=startup,
                bootstrap_role_manifest=manifest,
                claim_nonce=f"{position + 21:02x}" * 32,
            )
        )
    return rows


class _CloudState:
    def __init__(
        self,
        *,
        fail_insert_at: int | None = None,
        fail_claim_cas_at: int | None = None,
        fail_release_cas_at: int | None = None,
        duplicate_provider_ids: bool = False,
    ) -> None:
        self.instances: dict[str, dict[str, Any]] = {}
        self.log: list[dict[str, Any]] = []
        self.insert_count = 0
        self.claim_set_count = 0
        self.release_set_count = 0
        self.fail_insert_at = fail_insert_at
        self.fail_claim_cas_at = fail_claim_cas_at
        self.fail_release_cas_at = fail_release_cas_at
        self.duplicate_provider_ids = duplicate_provider_ids

    def provider_from_body(
        self, name: str, body: Mapping[str, Any]
    ) -> dict[str, Any]:
        return {
            "id": str(
                987_654_321_001
                if self.duplicate_provider_ids
                else 987_654_321_000 + self.insert_count
            ),
            "name": name,
            "selfLink": (
                "https://www.googleapis.com/compute/v1/projects/"
                f"{subject.PROJECT}/zones/{subject.ZONE}/instances/{name}"
            ),
            "zone": (
                "https://www.googleapis.com/compute/v1/projects/"
                f"{subject.PROJECT}/zones/{subject.ZONE}"
            ),
            "machineType": body["machineType"],
            "serviceAccounts": copy.deepcopy(body["serviceAccounts"]),
            "networkInterfaces": copy.deepcopy(body["networkInterfaces"]),
            "scheduling": copy.deepcopy(body["scheduling"]),
            "disks": [
                {
                    "boot": True,
                    "autoDelete": True,
                    "source": (
                        "https://www.googleapis.com/compute/v1/projects/"
                        f"{subject.PROJECT}/zones/{subject.ZONE}/disks/{name}"
                    ),
                }
            ],
            "metadata": {
                "fingerprint": f"initialFingerprint{self.insert_count}",
                "items": copy.deepcopy(body["metadata"]["items"]),
            },
        }


class _ComputeClient:
    def __init__(
        self,
        state: _CloudState,
        *,
        actor: str,
        controller_principal: str,
    ) -> None:
        self.state = state
        self.actor = actor
        self.project = subject.PROJECT
        self.zone = subject.ZONE
        self.principal = (
            controller_principal
            if actor == "controller"
            else "user:test@example.com"
        )
        self.credential_kind = (
            "fixed_nonrefreshing_controller"
            if actor == "controller"
            else "user"
        )
        self.raw_token = RAW_TOKEN

    def insert_instance(
        self,
        *,
        instance_name: str,
        request_id: str,
        body: Mapping[str, Any],
    ) -> subject.ComputeMutationResult:
        assert self.actor == "controller"
        self.state.insert_count += 1
        self.state.log.append(
            {
                "actor": self.actor,
                "operation": "insert",
                "name": instance_name,
                "request_id": request_id,
                "body": copy.deepcopy(dict(body)),
            }
        )
        if self.state.fail_insert_at == self.state.insert_count:
            return subject.ComputeMutationResult(
                500, instance_name, request_id, None, False
            )
        self.state.instances[instance_name] = (
            self.state.provider_from_body(instance_name, body)
        )
        return subject.ComputeMutationResult(
            202,
            instance_name,
            request_id,
            f"insert-op-{self.state.insert_count}",
            True,
        )

    def get_instance(
        self, *, instance_name: str
    ) -> Mapping[str, Any] | None:
        self.state.log.append(
            {
                "actor": self.actor,
                "operation": "get",
                "name": instance_name,
            }
        )
        value = self.state.instances.get(instance_name)
        return None if value is None else copy.deepcopy(value)

    def set_metadata(
        self,
        *,
        instance_name: str,
        request_id: str,
        expected_fingerprint: str,
        values: Mapping[str, str],
    ) -> subject.ComputeMutationResult:
        operation = (
            "claim_set_metadata"
            if self.actor == "controller"
            else "release_set_metadata"
        )
        if self.actor == "controller":
            self.state.claim_set_count += 1
            ordinal = self.state.claim_set_count
            fail = self.state.fail_claim_cas_at == ordinal
        else:
            self.state.release_set_count += 1
            ordinal = self.state.release_set_count
            fail = self.state.fail_release_cas_at == ordinal
        self.state.log.append(
            {
                "actor": self.actor,
                "operation": operation,
                "name": instance_name,
                "request_id": request_id,
                "expected_fingerprint": expected_fingerprint,
            }
        )
        provider = self.state.instances[instance_name]
        if fail or provider["metadata"]["fingerprint"] != expected_fingerprint:
            return subject.ComputeMutationResult(
                412, instance_name, request_id, None, False
            )
        provider["metadata"] = {
            "fingerprint": f"{self.actor}Fingerprint{ordinal:02d}",
            "items": subject._metadata_items(values),
        }
        return subject.ComputeMutationResult(
            202,
            instance_name,
            request_id,
            f"{operation}-op-{ordinal}",
            True,
        )

    def cleanup_exact_instances_and_disks(
        self,
        *,
        instance_names: Sequence[str],
        disk_names: Sequence[str],
    ) -> Mapping[str, Any]:
        assert self.actor == "user"
        self.state.log.append(
            {
                "actor": "user",
                "operation": "cleanup_compute",
                "instance_names": list(instance_names),
                "disk_names": list(disk_names),
            }
        )
        for name in instance_names:
            self.state.instances.pop(name, None)
        body = {
            "instance_names": list(instance_names),
            "disk_names": list(disk_names),
            "instance_final_statuses": [404 for _ in instance_names],
            "disk_final_statuses": [404 for _ in disk_names],
            "exact_cleanup_complete": True,
        }
        return {
            **body,
            "receipt_sha256": subject.canonical_sha256(body),
        }


class _Phase2Teardown:
    def __init__(
        self,
        context: dict[str, Any],
        log: list[dict[str, Any]],
        *,
        fail_remove: bool = False,
    ) -> None:
        self.context = context
        self.log = log
        self.fail_remove = fail_remove

    def remove_controller_bindings_after_claims(
        self,
        *,
        installed: Any,
        claim_cas_readback_receipts: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        self.log.append(
            {
                "actor": "user",
                "operation": "phase2_controller_remove",
                "claim_count": len(claim_cas_readback_receipts),
            }
        )
        if self.fail_remove:
            raise RuntimeError("injected teardown failure " + RAW_TOKEN)
        identities = self.context["phase2_plan"][
            "controller_release_removal_group"
        ]["binding_identities"]
        readbacks = [
            {
                **copy.deepcopy(dict(row)),
                "controller_member_count": 0,
                "controller_member_present": False,
                "readback_complete": True,
            }
            for row in identities
        ]
        candidate, reference = self.context["payloads"]
        return pair_release.build_phase2_zero_receipt(
            deployment_contract=self.context["deployment"],
            candidate_payload_contract=candidate,
            reference_payload_contract=reference,
            controller_public_key_record=self.context["public_record"],
            run_nonce=RUN_NONCE,
            phase2_iam_plan=self.context["phase2_plan"],
            binding_readbacks=readbacks,
        )

    def cleanup_phase2_iam_on_failure(
        self, *, failure_evidence_sha256: str
    ) -> Mapping[str, Any]:
        self.log.append(
            {
                "actor": "user",
                "operation": "cleanup_iam",
                "failure_evidence_sha256": failure_evidence_sha256,
            }
        )
        body = {
            "controller_before_worker": True,
            "targeted_binding_count": 0,
        }
        return {
            **body,
            "receipt_sha256": subject.canonical_sha256(body),
        }


class _ControllerSaAdmin:
    def __init__(
        self,
        context: dict[str, Any],
        log: list[dict[str, Any]],
        *,
        fail_success_delete: bool = False,
    ) -> None:
        self.context = context
        self.log = log
        self.fail_success_delete = fail_success_delete

    def _delete_receipt(self, *, success_path: bool) -> dict[str, Any]:
        deployment = self.context["deployment"]
        created = self.context["created"].receipt
        body = {
            "schema": controller_sa.SCHEMA,
            "status": (
                "run_scoped_controller_service_account_deleted_and_absent"
            ),
            "project": deployment["controller_service_account"]["project"],
            "account_id": deployment["controller_service_account"][
                "account_id"
            ],
            "email": deployment["controller_service_account"]["email"],
            "provider_unique_id": created["provider"]["unique_id"],
            "controller_create_receipt_sha256": (
                created["receipt_sha256"] if success_path else None
            ),
            "delete_call_count": 1,
            "delete_retry_count": 0,
            "absence_get_count": 1,
            "final_get_status": 404,
            "readback_verified": True,
            "success_path_teardown_evidence": success_path,
            "cloud_mutation_performed": True,
        }
        return {
            **body,
            "receipt_sha256": controller_sa.canonical_sha256(body),
        }

    def delete_created_and_wait_absent(
        self, *, created: Any
    ) -> Mapping[str, Any]:
        self.log.append(
            {
                "actor": "user",
                "operation": "delete_controller_sa_success",
            }
        )
        if self.fail_success_delete:
            raise RuntimeError("injected SA delete failure " + RAW_TOKEN)
        assert created is self.context["created"]
        return self._delete_receipt(success_path=True)

    def cleanup_delete_if_present(self) -> Mapping[str, Any]:
        self.log.append(
            {
                "actor": "user",
                "operation": "cleanup_controller_sa",
            }
        )
        body = {
            "controller_sa_absent": True,
            "cleanup_complete": True,
        }
        return {
            **body,
            "receipt_sha256": subject.canonical_sha256(body),
        }


def _run(
    context: dict[str, Any],
    *,
    state: _CloudState | None = None,
    role_launches: Sequence[subject.PreparedRoleLaunch] | None = None,
    request_ids: Sequence[str] = REQUEST_IDS,
    fail_remove: bool = False,
    fail_success_delete: bool = False,
) -> tuple[dict[str, Any], _CloudState]:
    cloud = _CloudState() if state is None else state
    controller = _ComputeClient(
        cloud,
        actor="controller",
        controller_principal=context["deployment"][
            "controller_service_account"
        ]["principal"],
    )
    user = _ComputeClient(
        cloud,
        actor="user",
        controller_principal=context["deployment"][
            "controller_service_account"
        ]["principal"],
    )
    teardown = _Phase2Teardown(
        context, cloud.log, fail_remove=fail_remove
    )
    sa_admin = _ControllerSaAdmin(
        context, cloud.log, fail_success_delete=fail_success_delete
    )
    candidate, reference = context["payloads"]
    receipt = subject.run_exact_pair_attempt0(
        deployment_contract=context["deployment"],
        candidate_payload_contract=candidate,
        reference_payload_contract=reference,
        controller_public_key_record=context["public_record"],
        run_nonce=RUN_NONCE,
        source_plan=context["source_plan"],
        source_provision=context["source_provision"],
        role_launches=(
            _role_launches(context)
            if role_launches is None
            else role_launches
        ),
        phase2_iam_plan=context["phase2_plan"],
        phase2_installed=context["installed"],
        controller_sa_created=context["created"],
        controller_client=controller,
        user_compute_client=user,
        phase2_teardown=teardown,
        controller_sa_admin=sa_admin,
        signer=context["signer"],
        request_ids=request_ids,
        now_unix_seconds=NOW,
        pair_release_issued_unix_seconds=NOW + 1,
        pair_release_nonce=RELEASE_NONCE,
        metadata_adapter=_MetadataAdapter(),
    )
    return receipt, cloud


def test_success_has_exact_two_attempt0_inserts_claims_and_releases(
    context: dict[str, Any],
) -> None:
    receipt, cloud = _run(context)
    assert receipt["schema"] == subject.FINAL_RECEIPT_SCHEMA
    assert receipt["insert_request_count"] == 2
    assert receipt["claim_cas_request_count"] == 2
    assert receipt["release_cas_request_count"] == 2
    assert receipt["controller_provider_get_count"] == 4
    assert receipt["user_provider_get_count"] == 2
    assert receipt["attempt_index"] == 0
    assert receipt["vm_count"] == 2
    assert receipt["machine_type"] == "c4-standard-8"
    assert receipt["attempt1_authorized"] is False
    assert receipt["third_vm_authorized"] is False
    assert receipt["automatic_retry_performed"] is False
    assert receipt["user_only_compute_after_controller_teardown"] is True
    assert len(set(receipt["request_ids"])) == 6

    controller_ops = [
        row["operation"]
        for row in cloud.log
        if row["actor"] == "controller"
    ]
    assert controller_ops == [
        "insert",
        "get",
        "insert",
        "get",
        "claim_set_metadata",
        "get",
        "claim_set_metadata",
        "get",
    ]
    user_release_ops = [
        row["operation"]
        for row in cloud.log
        if row["actor"] == "user"
        and row["operation"] in {"release_set_metadata", "get"}
    ]
    assert user_release_ops == [
        "release_set_metadata",
        "get",
        "release_set_metadata",
        "get",
    ]
    remove_index = next(
        index
        for index, row in enumerate(cloud.log)
        if row["operation"] == "phase2_controller_remove"
    )
    delete_index = next(
        index
        for index, row in enumerate(cloud.log)
        if row["operation"] == "delete_controller_sa_success"
    )
    release_index = next(
        index
        for index, row in enumerate(cloud.log)
        if row["operation"] == "release_set_metadata"
    )
    assert remove_index < delete_index < release_index
    assert RAW_TOKEN not in json.dumps(receipt, sort_keys=True)

    for insert in [
        row for row in cloud.log if row["operation"] == "insert"
    ]:
        body = insert["body"]
        assert body["name"] in receipt["instance_names"]
        assert body["scheduling"]["provisioningModel"] == "SPOT"
        assert body["networkInterfaces"][0]["accessConfigs"] == []
        assert body["serviceAccounts"] == [
            {
                "email": subject.WORKER_SERVICE_ACCOUNT,
                "scopes": [subject.OAUTH_SCOPE],
            }
        ]
        assert body["disks"][0]["initializeParams"]["diskName"] == (
            body["name"]
        )


def test_first_insert_success_second_failure_cleans_in_required_order(
    context: dict[str, Any],
) -> None:
    cloud = _CloudState(fail_insert_at=2)
    with pytest.raises(subject.ExactPairCloudControllerError) as captured:
        _run(context, state=cloud)
    receipt = captured.value.receipt
    assert receipt["failure_stage"] == "insert_1"
    assert receipt["failure_reason"] == "insert_non2xx"
    assert [row["operation"] for row in receipt["cleanup_records"]] == [
        "phase2_iam_revoke",
        "controller_service_account_delete",
        "exact_instance_and_disk_cleanup",
    ]
    assert [row["order"] for row in receipt["cleanup_records"]] == [
        1,
        2,
        3,
    ]
    assert len(
        [row for row in cloud.log if row["operation"] == "insert"]
    ) == 2
    assert not any(
        row["operation"] == "release_set_metadata" for row in cloud.log
    )
    cleanup = next(
        row for row in cloud.log if row["operation"] == "cleanup_compute"
    )
    assert cleanup["instance_names"] == [
        row["instance_name"] for row in context["deployment"]["instances"]
    ]
    assert cleanup["disk_names"] == cleanup["instance_names"]
    assert RAW_TOKEN not in json.dumps(receipt)


def test_role_swap_is_rejected_before_any_compute_call(
    context: dict[str, Any],
) -> None:
    roles = _role_launches(context)
    swapped = [
        subject.PreparedRoleLaunch(
            **{
                **roles[position].__dict__,
                "authorization": roles[1 - position].authorization,
            }
        )
        for position in range(2)
    ]
    cloud = _CloudState()
    with pytest.raises(ValueError):
        _run(context, state=cloud, role_launches=swapped)
    assert cloud.log == []


def test_duplicate_request_id_is_rejected_before_any_compute_call(
    context: dict[str, Any],
) -> None:
    ids = list(REQUEST_IDS)
    ids[-1] = ids[0]
    cloud = _CloudState()
    with pytest.raises(ValueError, match="collided"):
        _run(context, state=cloud, request_ids=ids)
    assert cloud.log == []


def test_duplicate_provider_instance_id_is_rejected_before_claim(
    context: dict[str, Any],
) -> None:
    cloud = _CloudState(duplicate_provider_ids=True)
    with pytest.raises(subject.ExactPairCloudControllerError) as captured:
        _run(context, state=cloud)
    assert (
        captured.value.receipt["failure_reason"]
        == "provider_instance_ids_collided"
    )
    assert len(
        [row for row in cloud.log if row["operation"] == "insert"]
    ) == 2
    assert not any(
        row["operation"] == "claim_set_metadata" for row in cloud.log
    )


def test_claim_cas_412_has_no_reget_reseal_or_retry(
    context: dict[str, Any],
) -> None:
    cloud = _CloudState(fail_claim_cas_at=1)
    with pytest.raises(subject.ExactPairCloudControllerError) as captured:
        _run(context, state=cloud)
    assert captured.value.receipt["failure_reason"] == "claim_cas_cas_412"
    claim_index = next(
        index
        for index, row in enumerate(cloud.log)
        if row["operation"] == "claim_set_metadata"
    )
    later_controller = [
        row
        for row in cloud.log[claim_index + 1 :]
        if row["actor"] == "controller"
    ]
    assert later_controller == []
    assert len(
        [
            row
            for row in cloud.log
            if row["operation"] == "claim_set_metadata"
        ]
    ) == 1
    assert not any(
        row["operation"] in {
            "phase2_controller_remove",
            "release_set_metadata",
        }
        for row in cloud.log
    )


@pytest.mark.parametrize(
    ("fail_remove", "fail_success_delete"),
    [(True, False), (False, True)],
)
def test_release_is_withheld_until_iam_zero_and_sa_get404(
    context: dict[str, Any],
    fail_remove: bool,
    fail_success_delete: bool,
) -> None:
    cloud = _CloudState()
    with pytest.raises(subject.ExactPairCloudControllerError):
        _run(
            context,
            state=cloud,
            fail_remove=fail_remove,
            fail_success_delete=fail_success_delete,
        )
    assert not any(
        row["operation"] == "release_set_metadata" for row in cloud.log
    )
    if fail_remove:
        assert not any(
            row["operation"] == "delete_controller_sa_success"
            for row in cloud.log[
                : next(
                    i
                    for i, row in enumerate(cloud.log)
                    if row["operation"] == "cleanup_iam"
                )
            ]
        )


def test_release_cas_412_is_single_shot_and_uses_user_only(
    context: dict[str, Any],
) -> None:
    cloud = _CloudState(fail_release_cas_at=1)
    with pytest.raises(subject.ExactPairCloudControllerError) as captured:
        _run(context, state=cloud)
    assert captured.value.receipt["failure_reason"] == (
        "pair_release_cas_cas_412"
    )
    release_calls = [
        row
        for row in cloud.log
        if row["operation"] == "release_set_metadata"
    ]
    assert len(release_calls) == 1
    assert release_calls[0]["actor"] == "user"
    first_release = cloud.log.index(release_calls[0])
    assert not any(
        row["actor"] == "controller"
        for row in cloud.log[first_release:]
    )
