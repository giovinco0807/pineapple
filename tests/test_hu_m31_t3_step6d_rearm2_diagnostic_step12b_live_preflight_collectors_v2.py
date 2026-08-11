from __future__ import annotations

import json
from typing import Any, Mapping

import pytest

from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_gce_transport_10c2_v1
    as transport,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step11_rest_iam_admin_v1
    as rest_iam,
)
from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_live_preflight_collectors_v2
    as subject,
)


TOKEN = "test-token-" + "x" * 64
NAMES = ("ofc-m31-s2b-candidate-a0", "ofc-m31-s2b-reference-a0")
NOW = 1_900_300_000


class _Token:
    def access_token(self) -> str:
        return TOKEN


class _Http:
    def __init__(self, callback: Any) -> None:
        self.callback = callback
        self.calls: list[tuple[str, str]] = []

    def request(
        self,
        *,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_seconds: int,
    ) -> rest_iam.HttpResponse:
        assert method == "GET"
        assert body is None
        assert headers["Authorization"] == f"Bearer {TOKEN}"
        self.calls.append((method, url))
        return self.callback(url)


def _response(status: int, value: Any) -> rest_iam.HttpResponse:
    return rest_iam.HttpResponse(
        status,
        json.dumps(value, sort_keys=True).encode(),
        {},
    )


def _deployment() -> dict[str, Any]:
    return {
        "deployment_contract_sha256": "a" * 64,
        "direct_stage_identity_sha256": "b" * 64,
        "remote_layout": {
            "stage_prefix": (
                f"gs://{transport.BUCKET}/hu-m31-r2diag-direct-v2/"
                f"stages/{'c' * 64}"
            )
        },
        "instances": [
            {"instance_name": name} for name in NAMES
        ],
    }


def test_direct_prefix_empty_uses_complete_get_pages() -> None:
    http = _Http(lambda _: _response(200, {}))
    receipt = subject.collect_direct_v2_prefix_empty_receipt(
        _deployment(),
        http_client=http,
        token_source=_Token(),
        observed_at_unix_seconds=NOW,
    )
    assert receipt["object_count"] == 0
    assert receipt["page_count"] == 1
    assert receipt["cloud_mutation_performed"] is False
    assert len(http.calls) == 1
    assert "hu-m31-r2diag-direct-v2" in http.calls[0][1]


def test_compute_absence_gets_pair_disks_and_full_zone_history() -> None:
    def callback(url: str) -> rest_iam.HttpResponse:
        if "/operations?" in url:
            return _response(200, {"items": []})
        return _response(404, {"error": {"code": 404}})

    http = _Http(callback)
    receipt = subject.collect_compute_absence_receipt(
        _deployment(),
        http_client=http,
        token_source=_Token(),
        observed_at_unix_seconds=NOW,
    )
    assert receipt["provider_get_404_count"] == 4
    assert receipt["operation_name_history_count"] == 0
    assert len(http.calls) == 5


def test_compute_absence_rejects_prior_target_link() -> None:
    target = (
        "https://www.googleapis.com/compute/v1/projects/"
        f"{transport.PROJECT}/zones/{transport.ZONE}/instances/{NAMES[0]}"
    )

    def callback(url: str) -> rest_iam.HttpResponse:
        if "/operations?" in url:
            return _response(200, {"items": [{"targetLink": target}]})
        return _response(404, {"error": {"code": 404}})

    with pytest.raises(FileExistsError, match="history"):
        subject.collect_compute_absence_receipt(
            _deployment(),
            http_client=_Http(callback),
            token_source=_Token(),
            observed_at_unix_seconds=NOW,
        )


def test_live_readback_collector_routes_fixed_get_surfaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    role_name = (
        f"projects/{transport.PROJECT}/roles/"
        "ofcM31T3ObjectReaderV1"
    )

    def callback(url: str) -> rest_iam.HttpResponse:
        if "/services?" in url:
            return _response(
                200,
                {
                    "services": [
                        {
                            "state": "ENABLED",
                            "config": {"name": "compute.googleapis.com"},
                        }
                    ]
                },
            )
        if "iam.googleapis.com" in url:
            return _response(200, {"name": role_name})
        if "/machineTypes/" in url:
            return _response(200, {"name": "c4-standard-8"})
        if "/routers/" in url:
            return _response(200, {"name": "router"})
        if "/regions/" in url:
            return _response(200, {"name": "asia-northeast1"})
        raise AssertionError(url)

    monkeypatch.setattr(
        subject,
        "_capacity_observations",
        lambda *args, **kwargs: {"capacity": "read"},
    )
    captured: dict[str, Any] = {}

    def builder(*args: Any, **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return {"receipt_sha256": "d" * 64}

    monkeypatch.setattr(
        subject.external_preflight,
        "build_live_readback_receipt",
        builder,
    )
    receipt = subject.collect_live_readback_receipt(
        _deployment(),
        phase2_iam_plan={
            "custom_role_readback_contract": {
                "requirements": [
                    {"purpose": "package_read", "name": role_name}
                ]
            }
        },
        http_client=_Http(callback),
        token_source=_Token(),
        observed_at_unix_seconds=NOW,
    )
    assert receipt == {"receipt_sha256": "d" * 64}
    assert captured["machine_type_readback"]["name"] == "c4-standard-8"
    assert captured["custom_role_readbacks"] == {
        "package_read": {"name": role_name}
    }
    assert captured["authoritative_capacity_observations"] == {
        "capacity": "read"
    }
