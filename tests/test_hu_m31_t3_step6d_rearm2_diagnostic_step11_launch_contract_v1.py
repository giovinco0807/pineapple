from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_cloud_package as package_builder,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_gce_transport_10c2_v1
    as transport,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_plan as plan,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step11_launch_contract_v1 as subject,
)


_TEST_RSA_N_HEX = (
    "a3d0294654b6af7e3117defa88604a05f914837ca47ac7265606fc2ece35b3f"
    "89c06dabe879a731df9614b50707eae9c3fc3ad8e02d595b28bc19c9a1fe4"
    "ec48a9b72018dad5f544e3e512836986b34eaa3d1c6f1da67c5a0f088276e"
    "ab85bb5a782b8a46c6e98e1967e860c8c47216b584feff4d82ff73bd8cc0e"
    "a50244bf24d8d799b70d2db589122c3278409ed541994a80641591fb9def1f"
    "6e8746306e15fdc3d9044bee17414fc23c0b8f0a17650fa525d0af7d0558c"
    "405ede44ff12217bda719decb8ffaecbfb1b5f4c3f8f309707180d40f11067"
    "936a3c2bf597445c1f9b7b82594944377fc1e76bb999ed17d14cabc41e0b5"
    "f97b937a3a4166469237"
)


@pytest.fixture(scope="module")
def public_key() -> dict[str, Any]:
    return transport.build_rsa_public_key_record(modulus_hex=_TEST_RSA_N_HEX)


@pytest.fixture(scope="module")
def direct_contract(
    tmp_path_factory: pytest.TempPathFactory,
    public_key: dict[str, Any],
) -> dict[str, Any]:
    package = tmp_path_factory.mktemp("step11-launch-package") / "package"
    package_builder.build_package(output_dir=package)
    wheel = {
        "path": f"wheels/{transport.EXPECTED_NUMPY_WHEEL_FILENAME}",
        "uri_suffix": f"wheels/{transport.EXPECTED_NUMPY_WHEEL_FILENAME}",
        "sha256": transport.EXPECTED_NUMPY_WHEEL_SHA256,
        "bytes": transport.EXPECTED_NUMPY_WHEEL_BYTES,
        "mode": "0644",
        "kind": "offline_numpy_cp311_manylinux_x86_64_wheel",
    }
    return transport.build_job_contract(
        package_dir=package,
        stage_id=plan.STAGE1_ID,
        job_id=plan.STAGE1_JOB_IDS[0],
        offline_wheel_record=wheel,
        controller_public_key_record=public_key,
    )


@pytest.fixture
def prebootstrap(tmp_path: Path) -> Path:
    path = tmp_path / "prebootstrap.py"
    path.write_text(
        "from __future__ import annotations\nraise SystemExit(0)\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


@pytest.fixture
def launch_contract(
    direct_contract: dict[str, Any],
    public_key: dict[str, Any],
    prebootstrap: Path,
) -> dict[str, Any]:
    return subject.build_launch_contract(
        transport_contract=direct_contract,
        controller_public_key_record=public_key,
        prebootstrap_path=prebootstrap,
    )


def test_freezes_exact_one_candidate_vm(
    launch_contract: dict[str, Any],
) -> None:
    assert launch_contract["stage_id"] == plan.STAGE1_ID
    assert launch_contract["selected_job_ids"] == ["candidate-shard-00"]
    assert launch_contract["source_roles"] == ["candidate"]
    assert launch_contract["vm_count"] == 1
    assert launch_contract["attempt_index"] == 0
    assert launch_contract["target"]["machine_type"] == "c4-standard-16"


def test_freezes_exact_image_spot_runtime_and_deletion(
    launch_contract: dict[str, Any],
) -> None:
    target = launch_contract["target"]
    assert target["image_name"] == transport.IMAGE_NAME
    assert target["image_id"] == transport.IMAGE_ID
    assert target["image_self_link"] == transport.IMAGE_SELF_LINK
    assert target["image_family_resolution_permitted"] is False
    body = launch_contract["instance_insert"]["static_body"]
    assert body["deletionProtection"] is False
    assert body["disks"][0]["autoDelete"] is True
    assert body["disks"][0]["initializeParams"]["sourceImage"] == (
        transport.IMAGE_SELF_LINK
    )
    assert body["scheduling"] == {
        "provisioningModel": "SPOT",
        "instanceTerminationAction": "DELETE",
        "automaticRestart": False,
        "onHostMaintenance": "TERMINATE",
        "maxRunDuration": {"seconds": "4200", "nanos": 0},
    }


def test_freezes_no_external_ip_default_tokyo_subnet_and_nat(
    launch_contract: dict[str, Any],
) -> None:
    body = launch_contract["instance_insert"]["static_body"]
    interface = body["networkInterfaces"][0]
    assert interface["accessConfigs"] == []
    assert interface["network"].endswith("/global/networks/default")
    assert interface["subnetwork"].endswith(
        "/regions/asia-northeast1/subnetworks/default"
    )
    egress = launch_contract["network_egress_contract"]
    assert egress["external_ip_permitted"] is False
    assert egress["cloud_nat_required"] is True
    assert egress["nat_router_name"] == (
        "ofc-t3-nat-router-asia-northeast1"
    )
    assert egress["nat_name"] == "ofc-t3-nat-asia-northeast1"
    assert egress["nat_source_subnetwork_ip_ranges"] == (
        "ALL_SUBNETWORKS_ALL_IP_RANGES"
    )


def test_freezes_dedicated_service_account_and_cloud_platform_scope(
    launch_contract: dict[str, Any],
) -> None:
    service_accounts = launch_contract["instance_insert"]["static_body"][
        "serviceAccounts"
    ]
    assert service_accounts == [
        {
            "email": transport.WORKER_SERVICE_ACCOUNT,
            "scopes": ["https://www.googleapis.com/auth/cloud-platform"],
        }
    ]
    permission = launch_contract["permission_contract"]
    assert permission["worker_required_iam_permissions"] == [
        "compute.instances.delete",
        "storage.objects.create",
        "storage.objects.get",
    ]
    assert permission["worker_permission_proof_required_before_insert"] is True


def test_metadata_allowlist_and_postcreate_claim_release_are_exact(
    launch_contract: dict[str, Any],
) -> None:
    metadata = launch_contract["metadata_from_file_contract"]
    assert metadata["initial_exact_keys"] == [
        "startup-script",
        "ofc-step11-transport-contract",
        "ofc-step11-controller-authorization",
        "ofc-step11-controller-public-key",
        "ofc-step11-prebootstrap",
    ]
    assert metadata["claim_key"] == "ofc-step11-controller-claim"
    assert metadata["claim_key"] not in metadata["initial_exact_keys"]
    assert metadata["claim_present_at_insert"] is False
    assert metadata["claim_release_phase"] == "post_instance_insert_get"
    assert metadata["claim_release_requires_provider_instance_id"] is True
    assert metadata["claim_release_requires_metadata_fingerprint_cas"] is True
    assert metadata["claim_release_may_add_only_claim_key"] is True
    assert metadata["claim_release_must_preserve_all_initial_metadata"] is True


def test_public_key_is_contract_bound_and_private_key_is_absent(
    launch_contract: dict[str, Any],
    public_key: dict[str, Any],
) -> None:
    trust = launch_contract["controller_trust"]
    assert trust["key_id"] == public_key["key_id"]
    assert trust["public_key_sha256"] == transport.canonical_sha256(public_key)
    assert trust["private_key_embedded"] is False
    assert trust["shared_secret_present"] is False
    assert "outside_workspace_vm_and_gcs" in trust["private_key_location"]


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("vm_count",), 2),
        (("source_roles",), ["reference"]),
        (("target", "machine_type"), "c4-standard-8"),
        (
            ("target", "image_self_link"),
            "https://example.invalid/replaced-image",
        ),
        (
            ("instance_insert", "static_body", "scheduling", "provisioningModel"),
            "STANDARD",
        ),
        (
            (
                "instance_insert",
                "static_body",
                "scheduling",
                "maxRunDuration",
                "seconds",
            ),
            "4201",
        ),
        (
            (
                "instance_insert",
                "static_body",
                "networkInterfaces",
                0,
                "accessConfigs",
            ),
            [{"type": "ONE_TO_ONE_NAT"}],
        ),
        (
            (
                "instance_insert",
                "static_body",
                "serviceAccounts",
                0,
                "scopes",
            ),
            ["https://www.googleapis.com/auth/compute.readonly"],
        ),
        (
            ("metadata_from_file_contract", "claim_present_at_insert"),
            True,
        ),
        (
            ("authorization_boundary", "vm_created"),
            True,
        ),
    ],
)
def test_validator_rejects_launch_contract_tamper(
    launch_contract: dict[str, Any],
    direct_contract: dict[str, Any],
    public_key: dict[str, Any],
    prebootstrap: Path,
    path: tuple[Any, ...],
    replacement: Any,
) -> None:
    changed = deepcopy(launch_contract)
    cursor: Any = changed
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement
    with pytest.raises(ValueError, match="launch contract changed"):
        subject.validate_launch_contract(
            changed,
            transport_contract=direct_contract,
            controller_public_key_record=public_key,
            prebootstrap_path=prebootstrap,
        )


def test_rejects_transport_without_pinned_controller_key(
    direct_contract: dict[str, Any],
    public_key: dict[str, Any],
    prebootstrap: Path,
) -> None:
    changed = deepcopy(direct_contract)
    changed["authorization_contract"]["controller_public_key_sha256"] = None
    changed["authorization_contract"]["controller_key_id"] = None
    with pytest.raises(ValueError, match="public key is not pinned"):
        subject.build_launch_contract(
            transport_contract=changed,
            controller_public_key_record=public_key,
            prebootstrap_path=prebootstrap,
        )


def test_rejects_attempt1_and_reference_stage(
    direct_contract: dict[str, Any],
    public_key: dict[str, Any],
    prebootstrap: Path,
) -> None:
    attempt1 = deepcopy(direct_contract)
    attempt1["adapter_preview"]["attempt_index"] = 1
    with pytest.raises(ValueError):
        subject.build_launch_contract(
            transport_contract=attempt1,
            controller_public_key_record=public_key,
            prebootstrap_path=prebootstrap,
        )


def test_freeze_is_canonical_write_once_and_validate_cli(
    direct_contract: dict[str, Any],
    public_key: dict[str, Any],
    prebootstrap: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    direct_path = tmp_path / "transport.json"
    key_path = tmp_path / "public-key.json"
    output = tmp_path / "launch.json"
    direct_path.write_bytes(subject.canonical_bytes(direct_contract))
    key_path.write_bytes(subject.canonical_bytes(public_key))
    assert (
        subject.main(
            [
                "freeze",
                "--transport-contract",
                str(direct_path),
                "--controller-public-key",
                str(key_path),
                "--prebootstrap",
                str(prebootstrap),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    frozen = json.loads(output.read_bytes())
    assert output.read_bytes() == subject.canonical_bytes(frozen)
    with pytest.raises(FileExistsError):
        subject.freeze_launch_contract(
            transport_contract=direct_contract,
            controller_public_key_record=public_key,
            prebootstrap_path=prebootstrap,
            output=output,
        )
    assert (
        subject.main(
            [
                "validate",
                "--transport-contract",
                str(direct_path),
                "--controller-public-key",
                str(key_path),
                "--prebootstrap",
                str(prebootstrap),
                "--contract",
                str(output),
            ]
        )
        == 0
    )
    rows = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.strip()
    ]
    assert rows[-1]["launch_ready"] is False


def test_stage0_shell_has_bounded_failure_trap_and_postcreate_claim_poll() -> None:
    script = subject.DEFAULT_STARTUP.read_text(encoding="utf-8")
    assert "set -Eeuo pipefail" in script
    assert "trap finish EXIT" in script
    assert "request_shutdown_once" in script
    assert "/usr/bin/timeout \"$FAILURE_SHUTDOWN_SECONDS\"" in script
    assert "/sbin/shutdown -h now" in script
    assert "/usr/bin/timeout 300" in script
    assert "/usr/bin/apt-get update" in script
    assert "python3-venv" in script
    for key in subject.INITIAL_METADATA_FROM_FILE_KEYS[1:]:
        assert f'"{key}"' in script
    assert '"ofc-step11-controller-claim"' in script
    assert "CLAIM_WAIT_SECONDS=300" in script
    assert "--controller-public-key" in script
    assert "OFC_DIAGNOSTIC_10C2_AUTHORIZED_WORKER=1" in script
    assert "gcloud " not in script
    assert "gsutil " not in script

