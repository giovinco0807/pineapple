from __future__ import annotations

import json
import urllib.parse
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import pytest

from ofc_regular import hu_m31_t3_step6d_full100_wave_gce_adapter_v2 as subject
from ofc_regular import hu_m31_t3_step6d_full100_wave_launch_bundle_v2 as bundle_v2
from ofc_regular import hu_m31_t3_step6d_full100_wave_package_v2 as package_v2


NOW = "2026-07-22T04:00:00Z"
IMAGE_SHA = "9dd85299f559ea3b143b1a764a9c69e0e535672036c2b45bf1cff25b88da3c0d"
IMAGE_LINK = (
    "https://www.googleapis.com/compute/v1/projects/debian-cloud/global/images/"
    "debian-12-bookworm-v20260721"
)
CREATE_REQUESTS = {
    "f100wv2-c00-a00": "11111111-1111-4111-8111-111111111111",
    "f100wv2-r00-a00": "22222222-2222-4222-8222-222222222222",
}
DELETE_REQUESTS = {
    "f100wv2-c00-a00": "33333333-3333-4333-8333-333333333333",
    "f100wv2-r00-a00": "44444444-4444-4444-8444-444444444444",
}
ORPHAN_DELETE_REQUESTS = {
    "f100wv2-c00-a00": "55555555-5555-4555-8555-555555555555",
    "f100wv2-r00-a00": "66666666-6666-4666-8666-666666666666",
}


def _bootstrap(
    *, job: str, role: str, attempt: str, instance: str, account: str
) -> dict[str, Any]:
    core = {
        "job_id": job,
        "source_role": role,
        "attempt_id": attempt,
        "instance_name": instance,
        "worker_principal": account,
    }
    return {**core, "bootstrap_sha256": package_v2.canonical_sha256(core)}


def _bundle(
    identities: list[tuple[str, str, str, str, str]] | None = None,
) -> dict[str, Any]:
    if identities is None:
        identities = [
            (
                "candidate-00",
                "candidate",
                "a00",
                "f100wv2-c00-a00",
                f"f100wv2-c00@{subject.PROJECT}.iam.gserviceaccount.com",
            ),
            (
                "reference-00",
                "reference",
                "a00",
                "f100wv2-r00-a00",
                f"f100wv2-r00@{subject.PROJECT}.iam.gserviceaccount.com",
            ),
        ]
    inventory = []
    for job, role, attempt, instance, account in identities:
        bootstrap = _bootstrap(
            job=job,
            role=role,
            attempt=attempt,
            instance=instance,
            account=account,
        )
        inventory.append(
            {
                "job_id": job,
                "source_role": role,
                "attempt_id": attempt,
                "instance_id": instance,
                "service_account": account,
                "bootstrap_sha256": bootstrap["bootstrap_sha256"],
                "bootstrap": bootstrap,
            }
        )
    core = {
        "schema": bundle_v2.LAUNCH_BUNDLE_SCHEMA,
        "status": bundle_v2.LAUNCH_BUNDLE_STATUS,
        "run_name": "regular-hu-m31-c02-full100-wave-v2-test",
        "execution_identity_sha256": "1" * 64,
        "wave_plan_sha256": "2" * 64,
        "project_id": subject.PROJECT,
        "zone": subject.ZONE,
        "wave_index": 0,
        "current_time_utc": "2026-07-22T03:59:00Z",
        "selected_vm_count": len(inventory),
        "selected_job_ids": [row["job_id"] for row in inventory],
        "selected_source_roles": [row["source_role"] for row in inventory],
        "selected_attempt_ids": [row["attempt_id"] for row in inventory],
        "selected_instance_ids": [row["instance_id"] for row in inventory],
        "selected_service_accounts": [row["service_account"] for row in inventory],
        "bootstrap_inventory": inventory,
        "cloud_create_authorized": True,
        "exact_selected_create_authorized": True,
        "one_shot": True,
        "reuse_authorized": False,
        "additional_create_authorized": False,
        "unlisted_instance_create_authorized": False,
        "cloud_started": False,
    }
    return {**core, "bundle_sha256": bundle_v2.canonical_sha256(core)}


class ValidatingCallback:
    def __init__(self, expected: Mapping[str, Any]) -> None:
        self.expected = deepcopy(dict(expected))
        self.calls = 0

    def __call__(self, value: Mapping[str, Any]) -> Mapping[str, Any]:
        self.calls += 1
        if dict(value) != self.expected:
            raise ValueError("upstream evidence validation rejected bundle")
        return deepcopy(self.expected)


class FakeGce:
    def __init__(self) -> None:
        self.instances: dict[str, dict[str, Any]] = {}
        self.disks: dict[str, dict[str, Any]] = {}
        self.calls: list[tuple[str, str, bytes | None]] = []
        self.insert_count = 0
        self.fail_insert_index: int | None = None
        self.fail_status = 409
        self.malformed_operation = False
        self.keep_disk_on_delete = False
        self.drop_instance_delete_after_commit = False
        self.instance_delete_status: int | None = None
        self.drop_disk_delete_after_commit = False

    @staticmethod
    def _response(status: int, value: Mapping[str, Any] | None = None) -> subject.HttpResponse:
        return subject.HttpResponse(
            status=status,
            body=b"" if value is None else json.dumps(value).encode("utf-8"),
            headers={},
        )

    @staticmethod
    def _target(name: str, collection: str = "instances") -> str:
        return (
            f"https://www.googleapis.com/compute/v1/projects/{subject.PROJECT}/zones/"
            f"{subject.ZONE}/{collection}/{name}"
        )

    def _operation(
        self, *, kind: str, name: str, target_id: str, index: int,
        collection: str = "instances",
    ) -> dict[str, Any]:
        if self.malformed_operation:
            return {
                "name": f"operation-{kind}-{index}",
                "id": str(5000 + index),
                "status": "DONE",
                "operationType": "wrong",
                "targetLink": self._target(name, collection),
                "targetId": target_id,
            }
        return {
            "name": f"operation-{kind}-{index}",
            "id": str(5000 + index),
            "status": "DONE",
            "operationType": kind,
            "targetLink": self._target(name, collection),
            "targetId": target_id,
        }

    def __call__(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_seconds: int,
    ) -> subject.HttpResponse:
        assert headers["Authorization"] == "Bearer " + "t" * 40
        assert 1 <= timeout_seconds <= 600
        self.calls.append((method, url, body))
        parsed = urllib.parse.urlparse(url)
        parts = parsed.path.split("/")
        if "/instances/" in parsed.path:
            name = urllib.parse.unquote(parts[-1])
            if method == "GET":
                value = self.instances.get(name)
                return self._response(404 if value is None else 200, value)
            if method == "DELETE":
                if self.instance_delete_status is not None:
                    return self._response(
                        self.instance_delete_status, {"error": "explicit failure"}
                    )
                current = self.instances.pop(name, None)
                if current is None:
                    return self._response(404)
                if not self.keep_disk_on_delete:
                    self.disks.pop(name, None)
                elif name in self.disks:
                    self.disks[name]["users"] = []
                if self.drop_instance_delete_after_commit:
                    self.drop_instance_delete_after_commit = False
                    raise ConnectionError("instance delete response was lost")
                return self._response(
                    200,
                    self._operation(
                        kind="delete",
                        name=name,
                        target_id=str(current["id"]),
                        index=len(self.calls),
                    ),
                )
        if "/disks/" in parsed.path:
            name = urllib.parse.unquote(parts[-1])
            if method == "GET":
                value = self.disks.get(name)
                return self._response(404 if value is None else 200, value)
            if method == "DELETE":
                current = self.disks.pop(name, None)
                if current is None:
                    return self._response(404)
                if self.drop_disk_delete_after_commit:
                    raise ConnectionError("disk delete response was lost")
                return self._response(
                    200,
                    self._operation(
                        kind="delete",
                        name=name,
                        target_id=str(current["id"]),
                        index=len(self.calls),
                        collection="disks",
                    ),
                )
        if parsed.path.endswith("/instances") and method == "POST":
            self.insert_count += 1
            if self.fail_insert_index == self.insert_count:
                return self._response(self.fail_status)
            assert body is not None
            spec = json.loads(body)
            name = spec["name"]
            if name in self.instances:
                return self._response(409)
            instance_id = str(1000 + self.insert_count)
            disk_id = str(2000 + self.insert_count)
            instance_self = self._target(name)
            disk_self = (
                f"https://www.googleapis.com/compute/v1/projects/{subject.PROJECT}/zones/"
                f"{subject.ZONE}/disks/{name}"
            )
            instance = {
                "name": name,
                "id": instance_id,
                "selfLink": instance_self,
                "status": "PROVISIONING",
                "machineType": spec["machineType"],
                "deletionProtection": spec["deletionProtection"],
                "canIpForward": spec["canIpForward"],
                "labels": deepcopy(spec["labels"]),
                "scheduling": deepcopy(spec["scheduling"]),
                "disks": [
                    {
                        "boot": True,
                        "autoDelete": True,
                        "interface": subject.BOOT_DISK_INTERFACE,
                        "deviceName": name,
                        "source": disk_self,
                    }
                ],
                "networkInterfaces": deepcopy(spec["networkInterfaces"]),
                "serviceAccounts": deepcopy(spec["serviceAccounts"]),
                "metadata": deepcopy(spec["metadata"]),
            }
            disk = {
                "name": name,
                "id": disk_id,
                "selfLink": disk_self,
                "type": spec["disks"][0]["initializeParams"]["diskType"],
                "sizeGb": spec["disks"][0]["initializeParams"]["diskSizeGb"],
                "sourceImage": spec["disks"][0]["initializeParams"]["sourceImage"],
                "labels": deepcopy(spec["disks"][0]["initializeParams"]["labels"]),
                "status": "READY",
                "users": [instance_self],
            }
            self.instances[name] = instance
            self.disks[name] = disk
            return self._response(
                200,
                self._operation(
                    kind="insert",
                    name=name,
                    target_id=instance_id,
                    index=self.insert_count,
                ),
            )
        raise AssertionError(f"unexpected fake request: {method} {url}")


@pytest.fixture()
def fixture(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setenv(subject.TOKEN_ENV, "t" * 40)
    bundle = _bundle()
    validator = ValidatingCallback(bundle)
    fake = FakeGce()
    startup = Path(
        "scripts/startup_hu_m31_t3_step6d_full100_wave_v2.sh"
    ).read_bytes()
    return {
        "bundle": bundle,
        "validator": validator,
        "fake": fake,
        "startup": startup,
    }


def _adapter(
    fixture: Mapping[str, Any],
    mode: str,
    *,
    create_receipt: Mapping[str, Any] | None = None,
    delete_receipt: Mapping[str, Any] | None = None,
) -> subject.GceWavePhaseBAdapter:
    return subject.GceWavePhaseBAdapter(
        mode=mode,
        launch_bundle=fixture["bundle"],
        launch_bundle_validator=fixture["validator"],
        active_image_self_link=IMAGE_LINK,
        active_image_identity_sha256=IMAGE_SHA,
        expected_image_digest=f"sha256:{IMAGE_SHA}",
        startup_script_bytes=fixture["startup"],
        requester=fixture["fake"],
        create_receipt=create_receipt,
        delete_receipt=delete_receipt,
        sleeper=lambda _: None,
    )


def _create(fixture: Mapping[str, Any]) -> dict[str, Any]:
    return _adapter(fixture, "create").create_selected(
        request_ids=CREATE_REQUESTS,
        observed_at_utc=NOW,
    )


def _delete(
    fixture: Mapping[str, Any], create: Mapping[str, Any]
) -> dict[str, Any]:
    return _adapter(fixture, "delete", create_receipt=create).delete_owned(
        request_ids=DELETE_REQUESTS,
        orphan_disk_request_ids=ORPHAN_DELETE_REQUESTS,
        observed_at_utc="2026-07-22T04:01:00Z",
    )


def _reseal(value: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(value))
    result["receipt_sha256"] = subject.canonical_sha256(
        {key: val for key, val in result.items() if key != "receipt_sha256"}
    )
    return result


def _partial_without(
    create: Mapping[str, Any], instance_name: str
) -> dict[str, Any]:
    result = deepcopy(dict(create))
    result["status"] = "partial_exact_owned_gce_create"
    result["rows"] = [
        row for row in result["rows"] if row["instance_name"] != instance_name
    ]
    result["created_instance_count"] = len(result["rows"])
    result["create_complete"] = False
    return _reseal(result)


def test_requires_real_validator_callback_and_exact_image_startup(
    fixture: dict[str, Any],
) -> None:
    kwargs = {
        "mode": "create",
        "launch_bundle": fixture["bundle"],
        "active_image_self_link": IMAGE_LINK,
        "active_image_identity_sha256": IMAGE_SHA,
        "expected_image_digest": f"sha256:{IMAGE_SHA}",
        "startup_script_bytes": fixture["startup"],
        "requester": fixture["fake"],
    }
    with pytest.raises(PermissionError, match="validator"):
        subject.GceWavePhaseBAdapter(launch_bundle_validator=None, **kwargs)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="expected wave digest"):
        subject.GceWavePhaseBAdapter(
            launch_bundle_validator=fixture["validator"],
            **{**kwargs, "expected_image_digest": "sha256:" + "8" * 64},
        )
    with pytest.raises(ValueError, match="frozen hash"):
        subject.GceWavePhaseBAdapter(
            launch_bundle_validator=fixture["validator"],
            **{**kwargs, "startup_script_bytes": fixture["startup"] + b"\n"},
        )


def test_exact_spec_is_c4_spot_private_pinned_and_metadata_closed(
    fixture: dict[str, Any],
) -> None:
    adapter = _adapter(fixture, "create")
    bodies = adapter.expected_instance_bodies
    assert len(bodies) == 2
    assert bodies[0]["machineType"].endswith("/machineTypes/c4-standard-16")
    assert bodies[0]["scheduling"] == {
        "provisioningModel": "SPOT",
        "instanceTerminationAction": "DELETE",
        "automaticRestart": False,
        "onHostMaintenance": "TERMINATE",
        "maxRunDuration": {"seconds": "4500", "nanos": 0},
    }
    disk = bodies[0]["disks"][0]
    assert disk["autoDelete"] is True
    assert disk["initializeParams"]["sourceImage"] == IMAGE_LINK
    assert disk["initializeParams"]["diskType"].endswith("/hyperdisk-balanced")
    assert disk["initializeParams"]["diskName"] == bodies[0]["name"]
    assert disk["initializeParams"]["labels"] == bodies[0]["labels"]
    assert bodies[0]["networkInterfaces"][0]["accessConfigs"] == []
    assert bodies[0]["serviceAccounts"][0]["email"] != bodies[1]["serviceAccounts"][0]["email"]
    assert {row["key"] for row in bodies[0]["metadata"]["items"]} == {
        "job-bootstrap-b64", "startup-script"
    }
    assert bodies[0]["labels"]["ofc-bundle"] == fixture["bundle"]["bundle_sha256"][:32]


def test_missing_token_before_create_or_delete_is_not_transport_ambiguity(
    fixture: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    create_receipt = _create(fixture)
    create_adapter = _adapter(fixture, "create")
    delete_adapter = _adapter(
        fixture, "delete", create_receipt=create_receipt
    )
    fixture["fake"].calls.clear()
    monkeypatch.delenv(subject.TOKEN_ENV)

    mutation_calls = (
        (
            create_adapter,
            "POST",
            (
                f"https://compute.googleapis.com/compute/v1/projects/{subject.PROJECT}/"
                f"zones/{subject.ZONE}/instances?requestId="
                "77777777-7777-4777-8777-777777777777"
            ),
        ),
        (
            delete_adapter,
            "DELETE",
            (
                f"https://compute.googleapis.com/compute/v1/projects/{subject.PROJECT}/"
                f"zones/{subject.ZONE}/instances/f100wv2-c00-a00?requestId="
                "88888888-8888-4888-8888-888888888888"
            ),
        ),
    )
    for adapter, method, url in mutation_calls:
        with pytest.raises(PermissionError, match=subject.TOKEN_ENV) as caught:
            adapter._http(  # noqa: SLF001 - exercise the transport boundary itself
                method=method,
                url=url,
                body=b"{}" if method == "POST" else None,
                content_type="application/json" if method == "POST" else None,
                allowed_statuses=(200, 404),
            )
        assert not isinstance(caught.value, subject.GcePhaseBTransportError)

    assert fixture["fake"].calls == []


def test_create_and_separate_status_readback_are_exact_and_one_shot(
    fixture: dict[str, Any],
) -> None:
    adapter = _adapter(fixture, "create")
    receipt = adapter.create_selected(request_ids=CREATE_REQUESTS, observed_at_utc=NOW)
    assert receipt["create_complete"] is True
    assert receipt["created_instance_count"] == 2
    assert [row["request_id"] for row in receipt["rows"]] == list(CREATE_REQUESTS.values())
    with pytest.raises(PermissionError, match="consumed"):
        adapter.create_selected(request_ids=CREATE_REQUESTS, observed_at_utc=NOW)
    posts = [call for call in fixture["fake"].calls if call[0] == "POST"]
    assert len(posts) == 2
    assert all("requestId=" in url for _, url, _ in posts)

    status_adapter = _adapter(fixture, "read-status", create_receipt=receipt)
    status = status_adapter.read_status(observed_at_utc=NOW)
    assert status["instance_count"] == 2
    assert status["all_specs_exact"] is True
    assert status_adapter.validate_status_receipt(status) == status
    assert subject.validate_status_receipt(status_adapter, status) == status


def test_single_selected_job_preserves_exact_create_status_delete_absence_contract(
    fixture: dict[str, Any],
) -> None:
    identity = (
        "candidate-shard-00",
        "candidate",
        "a00",
        "f100wv2-c00-a00",
        f"f100wv2-c00@{subject.PROJECT}.iam.gserviceaccount.com",
    )
    bundle = _bundle([identity])
    single = {
        **fixture,
        "bundle": bundle,
        "validator": ValidatingCallback(bundle),
    }
    name = identity[3]
    create_requests = {name: "71111111-1111-4111-8111-111111111111"}
    delete_requests = {name: "73333333-3333-4333-8333-333333333333"}
    orphan_requests = {name: "75555555-5555-4555-8555-555555555555"}

    create = _adapter(single, "create").create_selected(
        request_ids=create_requests,
        observed_at_utc=NOW,
    )
    assert create["expected_instance_count"] == 1
    assert create["created_instance_count"] == 1
    assert create["create_complete"] is True
    status = _adapter(single, "read-status", create_receipt=create).read_status(
        observed_at_utc=NOW
    )
    assert status["instance_count"] == 1
    assert status["rows"][0]["instance_name"] == name

    delete = _adapter(single, "delete", create_receipt=create).delete_owned(
        request_ids=delete_requests,
        orphan_disk_request_ids=orphan_requests,
        observed_at_utc="2026-07-22T04:01:00Z",
    )
    assert delete["selected_instance_count"] == 1
    assert delete["scanned_selected_names"] == [name]
    absence = _adapter(
        single,
        "absence",
        create_receipt=create,
        delete_receipt=delete,
    ).verify_absence(observed_at_utc="2026-07-22T04:02:00Z")
    assert absence["checked_instance_count"] == 1
    assert absence["absent_instance_names"] == [name]
    assert absence["all_instances_absent"] is True
    assert absence["all_boot_disks_absent"] is True


@pytest.mark.parametrize(
    "drift",
    ["labels", "service_account", "metadata", "network", "spot", "disk_image", "provider_id"],
)
def test_read_status_rejects_provider_drift(
    fixture: dict[str, Any], drift: str
) -> None:
    create = _create(fixture)
    name = create["rows"][0]["instance_name"]
    instance = fixture["fake"].instances[name]
    disk = fixture["fake"].disks[name]
    if drift == "labels":
        instance["labels"]["ofc-bundle"] = "0" * 32
    elif drift == "service_account":
        instance["serviceAccounts"][0]["email"] = fixture["bundle"]["selected_service_accounts"][1]
    elif drift == "metadata":
        instance["metadata"]["items"].append({"key": "extra", "value": "x"})
    elif drift == "network":
        instance["networkInterfaces"][0]["accessConfigs"] = [{"natIP": "1.2.3.4"}]
    elif drift == "spot":
        instance["scheduling"]["provisioningModel"] = "STANDARD"
    elif drift == "disk_image":
        disk["sourceImage"] = IMAGE_LINK + "-changed"
    else:
        instance["id"] = "99999"
    with pytest.raises(RuntimeError, match="drifted|changed"):
        _adapter(fixture, "read-status", create_receipt=create).read_status(
            observed_at_utc=NOW
        )


def test_preempted_auto_deleted_instance_and_disk_are_terminal_absent(
    fixture: dict[str, Any],
) -> None:
    create = _create(fixture)
    name = create["rows"][0]["instance_name"]
    fixture["fake"].instances.pop(name)
    fixture["fake"].disks.pop(name)
    status = _adapter(
        fixture, "read-status", create_receipt=create
    ).read_status(observed_at_utc=NOW)
    assert status["rows"][0]["status"] == "ABSENT"
    assert status["rows"][0]["provider_instance_id"] == create["rows"][0][
        "provider_instance_id"
    ]
    assert status["rows"][1]["status"] == "PROVISIONING"


def test_preemption_partial_instance_disk_absence_is_drift(
    fixture: dict[str, Any],
) -> None:
    create = _create(fixture)
    name = create["rows"][0]["instance_name"]
    fixture["fake"].instances.pop(name)
    with pytest.raises(RuntimeError, match="only instance or boot disk absent"):
        _adapter(fixture, "read-status", create_receipt=create).read_status(
            observed_at_utc=NOW
        )


def test_partial_create_receipt_on_409_can_delete_and_verify_absence(
    fixture: dict[str, Any],
) -> None:
    fixture["fake"].fail_insert_index = 2
    with pytest.raises(subject.GceCreateIncompleteError) as caught:
        _adapter(fixture, "create").create_selected(
            request_ids=CREATE_REQUESTS,
            observed_at_utc=NOW,
        )
    partial = caught.value.partial_receipt
    assert partial is not None
    assert partial["create_complete"] is False
    assert partial["created_instance_count"] == 1
    delete = _delete(fixture, partial)
    assert delete["delete_operation_count"] == 1
    absence_adapter = _adapter(
        fixture, "absence", create_receipt=partial, delete_receipt=delete
    )
    absence = absence_adapter.verify_absence(
        observed_at_utc="2026-07-22T04:02:00Z"
    )
    assert absence["all_instances_absent"] is True
    assert absence["all_boot_disks_absent"] is True


def test_orphan_exact_owned_boot_disk_is_deleted_and_bound_to_receipt(
    fixture: dict[str, Any],
) -> None:
    complete = _create(fixture)
    name = complete["rows"][0]["instance_name"]
    partial = _partial_without(complete, name)
    fixture["fake"].instances.pop(name)
    fixture["fake"].disks[name]["users"] = []

    delete = _delete(fixture, partial)
    assert delete["selected_instance_count"] == 2
    assert delete["scanned_selected_names"] == fixture["bundle"][
        "selected_instance_ids"
    ]
    assert delete["orphan_boot_disk_count"] == 1
    orphan = delete["orphan_boot_disk_rows"][0]
    assert orphan["instance_name"] == name
    assert orphan["provider_boot_disk_id"] == complete["rows"][0][
        "provider_boot_disk_id"
    ]
    assert orphan["already_absent"] is False
    assert orphan["recovered_after_transport_ambiguity"] is False
    assert name not in fixture["fake"].disks

    absence = _adapter(
        fixture, "absence", create_receipt=partial, delete_receipt=delete
    ).verify_absence(observed_at_utc="2026-07-22T04:02:00Z")
    assert absence["absent_instance_names"] == fixture["bundle"][
        "selected_instance_ids"
    ]


@pytest.mark.parametrize("drift", ["labels", "source_image"])
def test_orphan_disk_foreign_label_or_spec_is_never_deleted(
    fixture: dict[str, Any], drift: str
) -> None:
    complete = _create(fixture)
    name = complete["rows"][0]["instance_name"]
    partial = _partial_without(complete, name)
    fixture["fake"].instances.pop(name)
    disk = fixture["fake"].disks[name]
    disk["users"] = []
    if drift == "labels":
        disk["labels"]["ofc-bundle"] = "0" * 32
    else:
        disk["sourceImage"] = IMAGE_LINK + "-foreign"
    with pytest.raises(RuntimeError, match="ownership specification drifted"):
        _delete(fixture, partial)
    assert name in fixture["fake"].disks


def test_orphan_disk_unknown_provider_identity_is_never_deleted(
    fixture: dict[str, Any],
) -> None:
    complete = _create(fixture)
    name = complete["rows"][0]["instance_name"]
    partial = _partial_without(complete, name)
    fixture["fake"].instances.pop(name)
    fixture["fake"].disks[name]["users"] = []
    fixture["fake"].disks[name]["id"] = "0"
    with pytest.raises(RuntimeError, match="provider id changed"):
        _delete(fixture, partial)
    assert name in fixture["fake"].disks


def test_selected_instance_without_disk_fails_closed_before_any_delete(
    fixture: dict[str, Any],
) -> None:
    complete = _create(fixture)
    name = complete["rows"][0]["instance_name"]
    partial = _partial_without(complete, name)
    fixture["fake"].disks.pop(name)
    with pytest.raises(RuntimeError, match="exists without"):
        _delete(fixture, partial)
    assert name in fixture["fake"].instances


def test_orphan_disk_delete_response_loss_uses_get_only_reconciliation(
    fixture: dict[str, Any],
) -> None:
    complete = _create(fixture)
    name = complete["rows"][0]["instance_name"]
    partial = _partial_without(complete, name)
    fixture["fake"].instances.pop(name)
    fixture["fake"].disks[name]["users"] = []
    fixture["fake"].drop_disk_delete_after_commit = True
    fixture["fake"].calls.clear()

    delete = _delete(fixture, partial)
    orphan = delete["orphan_boot_disk_rows"][0]
    assert orphan["recovered_after_transport_ambiguity"] is True
    assert orphan["already_absent"] is False
    assert orphan["operation_name"] is None
    disk_deletes = [
        call for call in fixture["fake"].calls
        if call[0] == "DELETE" and "/disks/" in call[1]
    ]
    assert len(disk_deletes) == 1


def test_instance_delete_response_loss_never_reposts_and_recovers_by_get(
    fixture: dict[str, Any],
) -> None:
    create = _create(fixture)
    first = create["rows"][0]["instance_name"]
    fixture["fake"].drop_instance_delete_after_commit = True
    fixture["fake"].calls.clear()
    delete = _delete(fixture, create)
    row = delete["rows"][0]
    assert row["instance_name"] == first
    assert row["recovered_after_transport_ambiguity"] is True
    assert row["operation_name"] is None
    assert sum(
        method == "DELETE" and f"/instances/{first}" in url
        for method, url, _ in fixture["fake"].calls
    ) == 1


def test_instance_delete_response_loss_with_orphan_continues_exact_disk_cleanup(
    fixture: dict[str, Any],
) -> None:
    create = _create(fixture)
    first = create["rows"][0]["instance_name"]
    fixture["fake"].keep_disk_on_delete = True
    fixture["fake"].drop_instance_delete_after_commit = True
    delete = _delete(fixture, create)
    assert delete["rows"][0]["recovered_after_transport_ambiguity"] is True
    assert delete["orphan_boot_disk_count"] == 1
    assert delete["orphan_boot_disk_rows"][0]["request_id"] == (
        ORPHAN_DELETE_REQUESTS[first]
    )
    assert first not in fixture["fake"].disks


def test_explicit_instance_delete_failure_is_not_transport_recovery(
    fixture: dict[str, Any],
) -> None:
    create = _create(fixture)
    first = create["rows"][0]["instance_name"]
    fixture["fake"].instance_delete_status = 500
    with pytest.raises(RuntimeError, match="status 500"):
        _delete(fixture, create)
    assert first in fixture["fake"].instances


def test_reconcile_delete_all_absent_is_get_only_and_returns_delete_receipt(
    fixture: dict[str, Any],
) -> None:
    create = _create(fixture)
    fixture["fake"].instances.clear()
    fixture["fake"].disks.clear()
    fixture["fake"].calls.clear()
    adapter = _adapter(fixture, "reconcile-delete", create_receipt=create)
    receipt = adapter.reconcile_delete(
        request_ids=DELETE_REQUESTS,
        observed_at_utc="2026-07-22T04:01:00Z",
    )
    assert receipt["all_boot_disks_absent"] is True
    assert receipt["orphan_cleanup_required"] is False
    assert receipt["recovered_delete_receipt"]["already_absent_count"] == 2
    assert all(method == "GET" for method, _, _ in fixture["fake"].calls)


def test_reconcile_delete_returns_exact_partial_orphan_without_mutation(
    fixture: dict[str, Any],
) -> None:
    create = _create(fixture)
    orphan_name = create["rows"][0]["instance_name"]
    fixture["fake"].instances.clear()
    for name in list(fixture["fake"].disks):
        if name == orphan_name:
            fixture["fake"].disks[name]["users"] = []
        else:
            fixture["fake"].disks.pop(name)
    fixture["fake"].calls.clear()
    receipt = _adapter(
        fixture, "reconcile-delete", create_receipt=create
    ).reconcile_delete(
        request_ids=DELETE_REQUESTS,
        observed_at_utc="2026-07-22T04:01:00Z",
    )
    assert receipt["orphan_cleanup_required"] is True
    assert receipt["orphan_boot_disk_count"] == 1
    assert receipt["recovered_delete_receipt"] is None
    assert all(method == "GET" for method, _, _ in fixture["fake"].calls)


def test_reconcile_delete_fails_unresolved_when_instance_remains(
    fixture: dict[str, Any],
) -> None:
    create = _create(fixture)
    with pytest.raises(RuntimeError, match="unresolved"):
        _adapter(
            fixture, "reconcile-delete", create_receipt=create
        ).reconcile_delete(
            request_ids=DELETE_REQUESTS,
            observed_at_utc="2026-07-22T04:01:00Z",
        )


def test_post_commit_visibility_gap_is_reconciled_get_only_then_fully_deleted(
    fixture: dict[str, Any],
) -> None:
    # Model a process that durably saw only the first row even though both POSTs
    # committed server-side.  The restart path receives the durable request IDs
    # and the one known row, but is forbidden to POST again.
    complete = _create(fixture)
    partial = deepcopy(complete)
    partial["status"] = "partial_exact_owned_gce_create"
    partial["rows"] = partial["rows"][:1]
    partial["created_instance_count"] = 1
    partial["create_complete"] = False
    partial = _reseal(partial)
    validator = _adapter(fixture, "reconcile-create", create_receipt=partial)
    assert validator.validate_create_receipt(partial) == partial

    fixture["fake"].calls.clear()
    reconciled = validator.reconcile_create(
        request_ids=CREATE_REQUESTS,
        observed_at_utc="2026-07-22T04:00:01Z",
    )
    assert reconciled["created_instance_count"] == 2
    assert reconciled["create_complete"] is False
    assert reconciled["rows"][0]["recovered_after_insert_failure"] is False
    assert reconciled["rows"][1]["recovered_after_insert_failure"] is True
    assert all(method == "GET" for method, _, _ in fixture["fake"].calls)

    delete = _delete(fixture, reconciled)
    absence = _adapter(
        fixture, "absence", create_receipt=reconciled, delete_receipt=delete
    ).verify_absence(observed_at_utc="2026-07-22T04:02:00Z")
    assert absence["checked_instance_count"] == 2
    assert absence["absent_instance_names"] == fixture["bundle"][
        "selected_instance_ids"
    ]


def test_unknown_create_intent_reconcile_scans_every_selected_name_without_post(
    fixture: dict[str, Any],
) -> None:
    _create(fixture)
    fixture["fake"].calls.clear()
    reconciled = _adapter(fixture, "reconcile-create").reconcile_create(
        request_ids=CREATE_REQUESTS,
        observed_at_utc="2026-07-22T04:00:01Z",
    )
    assert reconciled["created_instance_count"] == 2
    assert all(row["recovered_after_insert_failure"] for row in reconciled["rows"])
    assert all(method == "GET" for method, _, _ in fixture["fake"].calls)


def test_existing_same_name_is_never_adopted(fixture: dict[str, Any]) -> None:
    adapter = _adapter(fixture, "create")
    body = adapter.expected_instance_bodies[0]
    name = body["name"]
    fixture["fake"].instances[name] = {"id": "999"}
    with pytest.raises(FileExistsError, match="never adopted"):
        adapter.create_selected(request_ids=CREATE_REQUESTS, observed_at_utc=NOW)


@pytest.mark.parametrize("status", [409, 412])
def test_create_rejects_collision_and_precondition_status(
    fixture: dict[str, Any], status: int
) -> None:
    fixture["fake"].fail_insert_index = 1
    fixture["fake"].fail_status = status
    with pytest.raises(subject.GceCreateIncompleteError) as caught:
        _adapter(fixture, "create").create_selected(
            request_ids=CREATE_REQUESTS,
            observed_at_utc=NOW,
        )
    assert caught.value.partial_receipt is not None
    assert caught.value.partial_receipt["created_instance_count"] == 0


def test_malformed_operation_is_recovered_as_exact_owned_partial_receipt(
    fixture: dict[str, Any],
) -> None:
    fixture["fake"].malformed_operation = True
    with pytest.raises(subject.GceCreateIncompleteError) as caught:
        _adapter(fixture, "create").create_selected(
            request_ids=CREATE_REQUESTS,
            observed_at_utc=NOW,
        )
    assert caught.value.partial_receipt is not None
    assert caught.value.partial_receipt["created_instance_count"] == 1
    assert caught.value.partial_receipt["rows"][0]["recovered_after_insert_failure"] is True
    assert caught.value.partial_receipt["rows"][0]["operation_name"] is None
    fixture["fake"].malformed_operation = False
    delete = _delete(fixture, caught.value.partial_receipt)
    absence = _adapter(
        fixture,
        "absence",
        create_receipt=caught.value.partial_receipt,
        delete_receipt=delete,
    ).verify_absence(observed_at_utc="2026-07-22T04:02:00Z")
    assert absence["all_instances_absent"] is True


def test_delete_rejects_request_id_reuse_and_ownership_drift(
    fixture: dict[str, Any],
) -> None:
    create = _create(fixture)
    with pytest.raises(ValueError, match="reused"):
        _adapter(fixture, "delete", create_receipt=create).delete_owned(
            request_ids=CREATE_REQUESTS,
            observed_at_utc=NOW,
        )
    name = create["rows"][0]["instance_name"]
    fixture["fake"].instances[name]["labels"]["ofc-bundle"] = "0" * 32
    with pytest.raises(RuntimeError, match="drifted"):
        _delete(fixture, create)


def test_absence_rejects_remaining_boot_disk(fixture: dict[str, Any]) -> None:
    create = _create(fixture)
    fixture["fake"].keep_disk_on_delete = True
    delete = _delete(fixture, create)
    adapter = _adapter(
        fixture, "absence", create_receipt=create, delete_receipt=delete
    )
    with pytest.raises(RuntimeError, match="absence"):
        adapter.verify_absence(observed_at_utc=NOW)


def test_receipts_reject_resealed_provider_and_bundle_tampering(
    fixture: dict[str, Any],
) -> None:
    create = _create(fixture)
    adapter = _adapter(fixture, "read-status", create_receipt=create)
    forged = deepcopy(create)
    forged["rows"][0]["provider_instance_id"] = "999999"
    with pytest.raises(ValueError, match="digest"):
        adapter.validate_create_receipt(forged)
    forged = deepcopy(create)
    forged["bundle_sha256"] = "8" * 64
    forged = _reseal(forged)
    with pytest.raises(ValueError, match="contract"):
        adapter.validate_create_receipt(forged)


def test_duplicate_or_unlisted_inventory_is_rejected_after_callback_validation(
    fixture: dict[str, Any],
) -> None:
    changed = deepcopy(fixture["bundle"])
    changed["bootstrap_inventory"][1] = deepcopy(changed["bootstrap_inventory"][0])
    changed["selected_instance_ids"][1] = changed["selected_instance_ids"][0]
    changed["selected_job_ids"][1] = changed["selected_job_ids"][0]
    changed["selected_source_roles"][1] = changed["selected_source_roles"][0]
    changed["selected_attempt_ids"][1] = changed["selected_attempt_ids"][0]
    changed["selected_service_accounts"][1] = changed["selected_service_accounts"][0]
    core = {key: value for key, value in changed.items() if key != "bundle_sha256"}
    changed["bundle_sha256"] = bundle_v2.canonical_sha256(core)
    local_fixture = {**fixture, "bundle": changed, "validator": ValidatingCallback(changed)}
    with pytest.raises(ValueError, match="duplicated"):
        _adapter(local_fixture, "create")


def test_shared_attempt_ids_and_repeated_source_roles_are_valid_wave_attributes(
    fixture: dict[str, Any],
) -> None:
    identities = [
        (
            f"{role}-{index:02d}",
            role,
            "a00",
            f"f100wv2-{role[0]}{index:02d}-a00",
            f"f100wv2-{role[0]}{index:02d}@{subject.PROJECT}.iam.gserviceaccount.com",
        )
        for index in range(2)
        for role in ("candidate", "reference")
    ]
    bundle = _bundle(identities)
    local_fixture = {
        **fixture,
        "bundle": bundle,
        "validator": ValidatingCallback(bundle),
    }

    adapter = _adapter(local_fixture, "create")

    assert len(adapter.expected_instance_bodies) == 4
    assert bundle["selected_attempt_ids"] == ["a00"] * 4
    assert bundle["selected_source_roles"] == [
        "candidate",
        "reference",
        "candidate",
        "reference",
    ]


@pytest.mark.parametrize(
    ("row_key", "vector_key"),
    [
        ("job_id", "selected_job_ids"),
        ("instance_id", "selected_instance_ids"),
        ("service_account", "selected_service_accounts"),
    ],
)
def test_identity_fields_remain_globally_unique(
    fixture: dict[str, Any], row_key: str, vector_key: str
) -> None:
    changed = deepcopy(fixture["bundle"])
    changed["bootstrap_inventory"][1][row_key] = changed[
        "bootstrap_inventory"
    ][0][row_key]
    changed[vector_key][1] = changed[vector_key][0]
    core = {key: value for key, value in changed.items() if key != "bundle_sha256"}
    changed["bundle_sha256"] = bundle_v2.canonical_sha256(core)
    local_fixture = {
        **fixture,
        "bundle": changed,
        "validator": ValidatingCallback(changed),
    }

    with pytest.raises(ValueError, match="duplicated"):
        _adapter(local_fixture, "create")


def test_parallel_inventory_vector_order_must_match_rows(
    fixture: dict[str, Any],
) -> None:
    changed = deepcopy(fixture["bundle"])
    changed["selected_job_ids"] = list(reversed(changed["selected_job_ids"]))
    core = {key: value for key, value in changed.items() if key != "bundle_sha256"}
    changed["bundle_sha256"] = bundle_v2.canonical_sha256(core)
    local_fixture = {
        **fixture,
        "bundle": changed,
        "validator": ValidatingCallback(changed),
    }

    with pytest.raises(ValueError, match="inventory vectors changed"):
        _adapter(local_fixture, "create")
