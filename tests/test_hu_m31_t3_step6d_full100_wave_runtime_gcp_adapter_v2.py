from __future__ import annotations

import base64
import copy
import hashlib
import json

import pytest

from ofc_regular import hu_m31_t3_step6d_full100_wave_plan_v2 as wave_v2
from ofc_regular import hu_m31_t3_step6d_full100_wave_controller_v2 as controller_v2
from ofc_regular import hu_m31_t3_step6d_full100_wave_runtime_gcp_adapter_v2 as subject
from ofc_regular import hu_m31_t3_step6d_full100_wave_runtime_preflight_v2 as runtime_v2
from ofc_regular.hu_m31_t3_step6d_performance_development_v2_cloud_execution_v1 import (
    HttpResponse,
)


OBSERVED = "2026-07-22T01:00:00Z"
CURRENT = "2026-07-22T01:00:04Z"


def _observations() -> dict[str, dict]:
    image_name = "debian-12-bookworm-v20260721"
    return {
        "image_family": {
            "id": "9021508813201755912",
            "name": image_name,
            "family": "debian-12",
            "status": "READY",
            "architecture": "X86_64",
            "guestOsFeatures": [
                {"type": "GVNIC"},
                {"type": "SEV_CAPABLE"},
                {"type": "UEFI_COMPATIBLE"},
                {"type": "VIRTIO_SCSI_MULTIQUEUE"},
            ],
            "storageLocations": ["asia-northeast1"],
            "selfLink": (
                "https://www.googleapis.com/compute/v1/projects/debian-cloud/"
                f"global/images/{image_name}"
            ),
        },
        "machine_type": {
            "id": "16001",
            "name": runtime_v2.MACHINE_TYPE,
            "guestCpus": 16,
            "memoryMb": 61_440,
            "architecture": "X86_64",
            "zone": runtime_v2.ZONE_SELF_LINK,
            "selfLink": runtime_v2.MACHINE_TYPE_SELF_LINK,
        },
        "network": {
            "id": "10001",
            "name": runtime_v2.NETWORK_NAME,
            "selfLink": runtime_v2.NETWORK_SELF_LINK,
        },
        "subnetwork": {
            "id": "10002",
            "name": runtime_v2.SUBNETWORK_NAME,
            "region": runtime_v2.REGION_SELF_LINK,
            "network": runtime_v2.NETWORK_SELF_LINK,
            "selfLink": runtime_v2.SUBNETWORK_SELF_LINK,
        },
        "router_nat": {
            "id": "10003",
            "name": runtime_v2.NAT_ROUTER_NAME,
            "region": runtime_v2.REGION_SELF_LINK,
            "network": runtime_v2.NETWORK_SELF_LINK,
            "selfLink": runtime_v2.ROUTER_SELF_LINK,
            "nats": [
                {
                    "name": runtime_v2.NAT_NAME,
                    "natIpAllocateOption": "AUTO_ONLY",
                    "sourceSubnetworkIpRangesToNat": (
                        "ALL_SUBNETWORKS_ALL_IP_RANGES"
                    ),
                }
            ],
        },
        "bucket": {
            "id": runtime_v2.BUCKET,
            "name": runtime_v2.BUCKET,
            "projectNumber": runtime_v2.PROJECT_NUMBER,
            "location": runtime_v2.BUCKET_LOCATION,
            "locationType": "region",
            "iamConfiguration": {
                "uniformBucketLevelAccess": {"enabled": True}
            },
        },
    }


def _plan() -> dict:
    image = _observations()["image_family"]
    return wave_v2.build_wave_plan(
        run_name="regular-hu-m31-c02-f100wv2-runtime-live-001",
        identity_salt="1234567890abcdef1234567890abcdef",
        package_sha256="2" * 64,
        image_digest="sha256:" + runtime_v2.derive_image_identity_sha256(image),
    )


class FakeRequester:
    def __init__(self, observations: dict[str, dict] | None = None) -> None:
        self.observations = _observations() if observations is None else observations
        self.calls: list[tuple[str, str, dict[str, str], bytes | None, int]] = []

    def __call__(self, method, url, headers, body, timeout):
        self.calls.append((method, url, dict(headers), body, timeout))
        by_url = {endpoint: name for name, endpoint in subject.ENDPOINTS}
        payload = json.dumps(
            self.observations[by_url[url]],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        return HttpResponse(status=200, body=payload, headers={})


def _read(monkeypatch, requester: FakeRequester | None = None) -> tuple[dict, FakeRequester]:
    monkeypatch.setenv(subject.TOKEN_ENV, "test-token-1234567890-abcdef")
    fake = FakeRequester() if requester is None else requester
    receipt = subject.RuntimeGcpReadAdapterV2(
        wave_plan=_plan(), requester=fake
    ).read(observed_at_utc=OBSERVED, current_utc=CURRENT)
    return receipt, fake


def _reseal(receipt: dict) -> None:
    receipt["receipt_sha256"] = subject.canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )


def test_live_collector_calls_exactly_six_gets_and_binds_raw_bodies(monkeypatch) -> None:
    receipt, requester = _read(monkeypatch)
    assert len(requester.calls) == 6
    assert [(call[0], call[1]) for call in requester.calls] == [
        ("GET", url) for _, url in subject.ENDPOINTS
    ]
    assert all(call[2]["Authorization"].startswith("Bearer ") for call in requester.calls)
    assert all(call[3] is None for call in requester.calls)
    assert all(call[4] == subject.HTTP_TIMEOUT_SECONDS for call in requester.calls)
    assert receipt["http_get_count"] == receipt["exact_endpoint_count"] == 6
    assert receipt["runtime_preflight_receipt_sha256"] == receipt[
        "runtime_preflight_receipt"
    ]["receipt_sha256"]
    assert subject.validate_runtime_gcp_read_receipt(
        wave_plan=_plan(), value=receipt, current_utc=CURRENT
    ) == receipt


def test_wrong_method_and_endpoint_are_rejected_before_network(monkeypatch) -> None:
    monkeypatch.setenv(subject.TOKEN_ENV, "test-token-1234567890-abcdef")
    requester = FakeRequester()
    adapter = subject.RuntimeGcpReadAdapterV2(wave_plan=_plan(), requester=requester)
    with pytest.raises(PermissionError, match="GET-only"):
        adapter._request(method="POST", url=subject.IMAGE_FAMILY_URL)
    with pytest.raises(PermissionError, match="exact endpoint"):
        adapter._request(method="GET", url=subject.IMAGE_FAMILY_URL + "&alt=media")
    assert requester.calls == []


def test_missing_token_is_local_permission_error_not_transport_loss(monkeypatch) -> None:
    monkeypatch.delenv(subject.TOKEN_ENV, raising=False)
    requester = FakeRequester()
    adapter = subject.RuntimeGcpReadAdapterV2(wave_plan=_plan(), requester=requester)
    with pytest.raises(PermissionError, match=subject.TOKEN_ENV):
        adapter._request(method="GET", url=subject.IMAGE_FAMILY_URL)
    assert requester.calls == []


def test_stale_live_receipt_is_rejected(monkeypatch) -> None:
    receipt, _ = _read(monkeypatch)
    with pytest.raises(ValueError, match="changed or expired"):
        subject.validate_runtime_gcp_read_receipt(
            wave_plan=_plan(),
            value=receipt,
            current_utc=receipt["runtime_preflight_receipt"]["expires_at_utc"],
        )


@pytest.mark.parametrize("tamper", ["url", "body", "nested"])
def test_resealed_endpoint_body_or_nested_runtime_tamper_fails(
    monkeypatch, tamper: str
) -> None:
    receipt, _ = _read(monkeypatch)
    forged = copy.deepcopy(receipt)
    if tamper == "url":
        forged["request_rows"][0]["url"] += "&alt=media"
    elif tamper == "body":
        body = base64.b64decode(forged["request_rows"][0]["response_body_base64"])
        replacement = body.replace(b'"READY"', b'"FAILED"')
        forged["request_rows"][0]["response_body_base64"] = base64.b64encode(
            replacement
        ).decode("ascii")
        forged["request_rows"][0]["response_bytes"] = len(replacement)
        forged["request_rows"][0]["response_sha256"] = hashlib.sha256(
            replacement
        ).hexdigest()
    else:
        forged["runtime_preflight_receipt"]["read_only_observation"] = False
    _reseal(forged)
    with pytest.raises(ValueError):
        subject.validate_runtime_gcp_read_receipt(
            wave_plan=_plan(), value=forged, current_utc=CURRENT
        )


def test_duplicate_json_key_and_oversized_body_are_rejected(monkeypatch) -> None:
    monkeypatch.setenv(subject.TOKEN_ENV, "test-token-1234567890-abcdef")

    def duplicate_requester(method, url, headers, body, timeout):
        del method, url, headers, body, timeout
        return HttpResponse(status=200, body=b'{"id":"1","id":"2"}', headers={})

    with pytest.raises(RuntimeError, match="strict JSON"):
        subject.RuntimeGcpReadAdapterV2(
            wave_plan=_plan(), requester=duplicate_requester
        ).read(observed_at_utc=OBSERVED, current_utc=CURRENT)

    def oversized_requester(method, url, headers, body, timeout):
        del method, url, headers, body, timeout
        return HttpResponse(
            status=200, body=b"x" * (subject.MAX_RESPONSE_BYTES + 1), headers={}
        )

    with pytest.raises(RuntimeError, match="body bound"):
        subject.RuntimeGcpReadAdapterV2(
            wave_plan=_plan(), requester=oversized_requester
        ).read(observed_at_utc=OBSERVED, current_utc=CURRENT)


def test_controller_runtime_gcp_read_requires_opt_in_and_records_producer_receipt(
    tmp_path, monkeypatch
) -> None:
    plan = _plan()
    transition = wave_v2.build_observed_transition(
        plan,
        project_id=runtime_v2.PROJECT,
        zone=runtime_v2.ZONE,
        observed_at_utc=OBSERVED,
        previous_transition_digest=None,
        attempt_history=wave_v2.empty_attempt_history(plan),
    )
    ledger = wave_v2.build_attempt_ledger(plan, transitions=[transition])
    resume = wave_v2.build_resume_plan(plan, attempt_ledger=ledger)
    controller = controller_v2.Full100WaveControllerV2(
        journal_dir=tmp_path / "journal",
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        create_journal=True,
    )
    request = {
        "operation_key": "runtime-live-read",
        "predecessor_event_sha256": None,
        "observed_at_utc": OBSERVED,
        "current_utc": CURRENT,
    }
    monkeypatch.setenv(subject.TOKEN_ENV, "test-token-1234567890-abcdef")
    requester = FakeRequester()
    with pytest.raises(PermissionError, match="allow_cloud_read"):
        controller_v2.execute_mode_request(
            controller=controller,
            mode="runtime-gcp-read",
            request=request,
            requester=requester,
        )
    assert requester.calls == []
    event = controller_v2.execute_mode_request(
        controller=controller,
        mode="runtime-gcp-read",
        request=request,
        requester=requester,
        allow_cloud_read=True,
    )
    live = event.value["output"]["runtime_gcp_read_receipt"]
    runtime = event.value["output"]["runtime_preflight_receipt"]
    assert event.value["phase"] == "runtime-gcp-read"
    assert live["runtime_preflight_receipt"] == runtime
    assert event.value["evidence"]["runtime_gcp_read_receipt_sha256"] == live[
        "receipt_sha256"
    ]


def test_controller_cli_runtime_gcp_read_requires_exact_run_confirmation(
    tmp_path,
) -> None:
    plan = _plan()
    transition = wave_v2.build_observed_transition(
        plan,
        project_id=runtime_v2.PROJECT,
        zone=runtime_v2.ZONE,
        observed_at_utc=OBSERVED,
        previous_transition_digest=None,
        attempt_history=wave_v2.empty_attempt_history(plan),
    )
    ledger = wave_v2.build_attempt_ledger(plan, transitions=[transition])
    resume = wave_v2.build_resume_plan(plan, attempt_ledger=ledger)
    request = {
        "operation_key": "runtime-live-cli",
        "predecessor_event_sha256": None,
        "observed_at_utc": OBSERVED,
        "current_utc": CURRENT,
    }
    paths = {}
    for name, value in (
        ("wave", plan),
        ("ledger", ledger),
        ("resume", resume),
        ("request", request),
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        paths[name] = path
    with pytest.raises(SystemExit, match="confirm-run-name"):
        controller_v2.main(
            [
                "--journal-dir",
                str(tmp_path / "journal"),
                "--wave-plan",
                str(paths["wave"]),
                "--attempt-ledger",
                str(paths["ledger"]),
                "--resume-plan",
                str(paths["resume"]),
                "--mode",
                "runtime-gcp-read",
                "--request",
                str(paths["request"]),
                "--allow-cloud-read",
            ]
        )
