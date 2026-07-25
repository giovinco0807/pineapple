from __future__ import annotations

import copy
import hashlib
import json
import urllib.parse
from typing import Any, Mapping

import pytest

from ofc_regular import hu_m31_t3_step6d_full100_wave_cloud_v2 as cloud_v2
from ofc_regular import hu_m31_t3_step6d_full100_wave_gcp_adapter_v2 as subject
from ofc_regular import hu_m31_t3_step6d_full100_wave_plan_v2 as wave_v2
from ofc_regular import hu_m31_t3_step6d_full100_wave_worker_iam_v2 as iam_v2


CONTENT_SHA = "4" * 64
MANIFEST_SHA = "5" * 64
CONTENT_PREFIX = f"{iam_v2.IMMUTABLE_CONTENT_PREFIX_ROOT}/{CONTENT_SHA}"
NONCE = "12345678-1234-4234-9234-1234567890ab"
TOKEN_A = "fixture-token-a-1234567890"
TOKEN_B = "fixture-token-b-1234567890"
OBSERVED = "2026-07-22T02:00:01Z"
EXPIRES = "2026-07-22T02:05:01Z"
ISSUED = 1_800_000_000


def _content_kwargs() -> dict[str, str]:
    return {
        "immutable_content_prefix": CONTENT_PREFIX,
        "content_payload_sha256": CONTENT_SHA,
        "outer_manifest_sha256": MANIFEST_SHA,
    }


@pytest.fixture(scope="module")
def evidence() -> tuple[dict, dict, dict, dict]:
    plan = wave_v2.build_wave_plan(
        run_name="regular-hu-m31-c02-f100wv2-gcpa-001",
        identity_salt="1234567890abcdef1234567890abcdef",
        package_sha256="2" * 64,
        image_digest="sha256:" + "3" * 64,
    )
    transition = wave_v2.build_observed_transition(
        plan,
        project_id="ofc-project-123",
        zone=subject.ZONE,
        observed_at_utc="2026-07-22T02:00:00Z",
        previous_transition_digest=None,
        attempt_history=wave_v2.empty_attempt_history(plan),
    )
    ledger = wave_v2.build_attempt_ledger(plan, transitions=[transition])
    resume = wave_v2.build_resume_plan(plan, attempt_ledger=ledger)
    iam = iam_v2.build_worker_iam_plan(
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        wave_index=0,
        **_content_kwargs(),
        issued_at_unix_seconds=ISSUED,
    )
    return plan, ledger, resume, iam


def _base_policy() -> dict[str, Any]:
    return {
        "kind": "storage#policy",
        "resourceId": f"projects/_/buckets/{subject.BUCKET}",
        "version": 3,
        "etag": "BwWInitialEtag==",
        "bindings": [
            {
                "role": "roles/storage.objectViewer",
                "members": ["user:unrelated@example.com"],
            }
        ],
        "auditConfigs": [],
    }


class LocalPolicyBackend:
    def __init__(self, policy: Mapping[str, Any]) -> None:
        self.policy = copy.deepcopy(dict(policy))

    def get_bucket_policy(self) -> Mapping[str, Any]:
        return copy.deepcopy(self.policy)

    def set_bucket_policy(self, *, policy: Mapping[str, Any]) -> Mapping[str, Any]:
        self.policy = copy.deepcopy(dict(policy))
        self.policy["etag"] = "BwWLocalSet=="
        return copy.deepcopy(self.policy)


class FakeHttp:
    def __init__(self) -> None:
        self.calls: list[
            tuple[str, str, dict[str, str], bytes | None, int]
        ] = []
        self.instance_collision: str | None = None
        self.disk_collision: str | None = None
        self.pagination = False
        self.unknown_spot_quota = False
        self.wrong_region_scope = False
        self.regional_unknown_top_level = False
        self.regional_unknown_row_field = False
        self.regional_missing_quotas = False
        self.service_account_404: str | None = None
        self.quota_404 = False
        self.actas_denied: str | None = None
        self.actas_extra_field = False
        self.provider_permission_denied: str | None = None
        self.provider_permission_extra_field = False
        self.custom_role_fault: str | None = None
        self.claim_collision = False
        self.claim_drop_after_commit = False
        self.claim_objects: dict[str, dict[str, Any]] = {}
        self.result_list_pagination = False
        self.policy = _base_policy()
        self.iam_race = False
        self.iam_drop_after_commit = False
        self.iam_reorder_bindings = False
        self.iam_omit_empty_audit_configs = False
        self.iam_semantic_drift = False
        self.iam_canonicalize_unconditional_version_one = False
        self.iam_put_count = 0
        self.provider_secret_body = "provider-secret-echo"

    @staticmethod
    def _response(
        status: int, value: Mapping[str, Any] | bytes
    ) -> subject.HttpResponse:
        raw = value if isinstance(value, bytes) else json.dumps(value).encode()
        return subject.HttpResponse(status, raw, {})

    def __call__(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout: int,
    ) -> subject.HttpResponse:
        self.calls.append((method, url, dict(headers), body, timeout))
        parsed = urllib.parse.urlparse(url)
        query = urllib.parse.parse_qs(parsed.query)

        if parsed.netloc == "cloudquotas.googleapis.com":
            quota_id = parsed.path.rsplit("/", 1)[-1]
            common = {
                "name": (
                    "projects/123456789/locations/global/services/"
                    f"compute.googleapis.com/quotaInfos/{quota_id}"
                ),
                "quotaId": quota_id,
                "service": "compute.googleapis.com",
                "isPrecise": True,
                "containerType": "PROJECT",
                "metricUnit": "1",
            }
            if quota_id == subject.C4_QUOTA_ID:
                return self._response(
                    200,
                    {
                        **common,
                        "metric": subject.C4_QUOTA_METRIC,
                        "quotaDisplayName": "C4 CPUs per VM family",
                        "quotaIncreaseEligibility": {"isEligible": True},
                        "dimensions": ["region", "vm_family"],
                        "dimensionsInfos": [
                            {
                                "dimensions": {
                                    "region": subject.REGION,
                                    "vm_family": subject.C4_QUOTA_VM_FAMILY,
                                },
                                "applicableLocations": [subject.REGION],
                                "details": {"value": "128"},
                            }
                        ],
                    },
                )
            if quota_id == subject.GLOBAL_QUOTA_ID:
                return self._response(
                    200,
                    {
                        **common,
                        "metric": subject.GLOBAL_QUOTA_METRIC,
                        "dimensions": [],
                        "dimensionsInfos": [
                            {
                                "applicableLocations": ["global"],
                                "details": {"value": "500"},
                            }
                        ],
                    },
                )

        aggregated = "/aggregated/"
        if parsed.netloc == "compute.googleapis.com" and aggregated in parsed.path:
            assert query == {
                "maxResults": ["500"],
                "returnPartialSuccess": ["false"],
                "includeAllScopes": ["true"],
            }
            collection = parsed.path.rsplit("/", 1)[-1]
            kinds = {
                "instances": "compute#instanceAggregatedList",
                "reservations": "compute#reservationAggregatedList",
                "nodeGroups": "compute#nodeGroupAggregatedList",
                "futureReservations": (
                    "compute#futureReservationsAggregatedListResponse"
                ),
            }
            scopes = [
                "global",
                f"regions/{subject.REGION}",
                f"zones/{subject.ZONE}",
            ]
            value: dict[str, Any] = {
                "kind": kinds[collection],
                "id": f"projects/{subject.PROJECT}/aggregated/{collection}",
                "selfLink": (
                    "https://www.googleapis.com/compute/v1/projects/"
                    f"{subject.PROJECT}/aggregated/{collection}"
                ),
                "items": {
                    scope: {
                        "warning": {
                            "code": "NO_RESULTS_ON_PAGE",
                            "message": "No results",
                            "data": [{"key": "scope", "value": scope}],
                        }
                    }
                    for scope in scopes
                },
            }
            if collection == "futureReservations":
                value["etag"] = "future-reservations-etag"
            return self._response(200, value)

        region_path = (
            f"/compute/v1/projects/{subject.PROJECT}/regions/{subject.REGION}"
        )
        project_path = f"/compute/v1/projects/{subject.PROJECT}"
        if parsed.netloc == "compute.googleapis.com" and parsed.path == region_path:
            assert query == {"fields": ["name,quotas"]}
            if self.quota_404:
                return self._response(404, {"error": self.provider_secret_body})
            quotas = [
                {"metric": "CPUS", "limit": 500, "usage": 0},
            ]
            if not self.unknown_spot_quota:
                quotas.append(
                    {"metric": "PREEMPTIBLE_CPUS", "limit": 468, "usage": 32}
                )
            if self.regional_unknown_row_field:
                quotas[0]["unknown"] = "provider-shape-drift"
            value: dict[str, Any] = {
                "name": "wrong-region" if self.wrong_region_scope else subject.REGION,
                "quotas": quotas,
            }
            if self.regional_missing_quotas:
                del value["quotas"]
            if self.regional_unknown_top_level:
                value["unknown"] = "provider-shape-drift"
            if self.pagination:
                value["nextPageToken"] = "forbidden-page"
            return self._response(200, value)
        if parsed.netloc == "compute.googleapis.com" and parsed.path == project_path:
            return self._response(
                200,
                {
                    "name": subject.PROJECT,
                    "quotas": [
                        {"metric": "CPUS_ALL_REGIONS", "limit": 500, "usage": 20}
                    ],
                },
            )
        instance_marker = f"/zones/{subject.ZONE}/instances/"
        if parsed.netloc == "compute.googleapis.com" and instance_marker in parsed.path:
            name = urllib.parse.unquote(parsed.path.rsplit("/", 1)[-1])
            if name == self.instance_collision:
                return self._response(200, {"name": name})
            return self._response(404, {"error": "not-found"})
        disk_marker = f"/zones/{subject.ZONE}/disks/"
        if parsed.netloc == "compute.googleapis.com" and disk_marker in parsed.path:
            name = urllib.parse.unquote(parsed.path.rsplit("/", 1)[-1])
            if name == self.disk_collision:
                return self._response(200, {"name": name})
            return self._response(404, {"error": "not-found"})
        if (
            parsed.netloc == "cloudresourcemanager.googleapis.com"
            and parsed.path.endswith(":testIamPermissions")
        ):
            assert method == "POST"
            assert body == subject.canonical_bytes(
                {"permissions": list(subject.PROVIDER_PROJECT_PERMISSIONS)}
            )
            granted = list(subject.PROVIDER_PROJECT_PERMISSIONS)
            if self.provider_permission_denied is not None:
                granted.remove(self.provider_permission_denied)
            result: dict[str, Any] = {"permissions": granted}
            if self.provider_permission_extra_field:
                result["unexpected"] = "provider-shape-drift"
            return self._response(200, result)
        if (
            parsed.netloc == "iam.googleapis.com"
            and parsed.path.endswith(":testIamPermissions")
        ):
            assert method == "POST"
            assert body == subject.canonical_bytes(
                {"permissions": [subject.ACT_AS_PERMISSION]}
            )
            encoded = parsed.path.rsplit("/", 1)[-1].removesuffix(
                ":testIamPermissions"
            )
            email = urllib.parse.unquote(encoded)
            result: dict[str, Any] = {
                "permissions": (
                    []
                    if email == self.actas_denied
                    else [subject.ACT_AS_PERMISSION]
                )
            }
            if self.actas_extra_field:
                result["unexpected"] = "provider-shape-drift"
            return self._response(200, result)
        role_marker = f"/v1/projects/{subject.PROJECT}/roles/"
        if parsed.netloc == "iam.googleapis.com" and parsed.path.startswith(
            role_marker
        ):
            role_id = urllib.parse.unquote(parsed.path[len(role_marker) :])
            role_name = f"projects/{subject.PROJECT}/roles/{role_id}"
            expected = dict(subject.CUSTOM_ROLE_EXPECTATIONS).get(role_name)
            if expected is None:
                return self._response(404, {"error": self.provider_secret_body})
            value: dict[str, Any] = {
                "name": role_name,
                "includedPermissions": list(expected),
                "stage": "GA",
                "deleted": False,
                "etag": f"role-etag-{role_id}",
            }
            if role_name == subject.CUSTOM_ROLE_EXPECTATIONS[0][0]:
                if self.custom_role_fault == "extra_permission":
                    value["includedPermissions"].append("storage.objects.list")
                elif self.custom_role_fault == "missing_permission":
                    value["includedPermissions"] = []
                elif self.custom_role_fault == "stage":
                    value["stage"] = "BETA"
                elif self.custom_role_fault == "deleted":
                    value["deleted"] = True
            return self._response(200, value)
        if parsed.netloc == "iam.googleapis.com" and "/serviceAccounts/" in parsed.path:
            email = urllib.parse.unquote(parsed.path.rsplit("/", 1)[-1])
            if email == self.service_account_404:
                return self._response(404, {"error": self.provider_secret_body})
            unique_id = str(10**20 + len(self.calls))
            return self._response(
                200,
                {
                    "name": (
                        f"projects/{subject.PROJECT}/serviceAccounts/{unique_id}"
                    ),
                    "projectId": subject.PROJECT,
                    "uniqueId": unique_id,
                    "email": email,
                    "disabled": False,
                },
            )
        if parsed.netloc == "storage.googleapis.com" and parsed.path.startswith(
            f"/upload/storage/v1/b/{subject.BUCKET}/o"
        ):
            assert method == "POST"
            name = query["name"][0]
            if self.claim_collision or name in self.claim_objects:
                return self._response(412, {"error": self.provider_secret_body})
            assert body is not None
            row = {
                "name": name,
                "generation": "1001",
                "etag": "claim-etag-1001",
                "size": str(len(body)),
                "payload": body,
            }
            self.claim_objects[name] = row
            if self.claim_drop_after_commit:
                self.claim_drop_after_commit = False
                raise ConnectionError("fake claim response lost after commit")
            return self._response(
                200,
                {key: value for key, value in row.items() if key != "payload"},
            )
        list_path = f"/storage/v1/b/{subject.BUCKET}/o"
        if parsed.netloc == "storage.googleapis.com" and parsed.path == list_path:
            assert method == "GET"
            prefix = query["prefix"][0]
            page_token = query.get("pageToken", [None])[0]
            matching = [
                {
                    key: value for key, value in row.items()
                    if key in {"name", "generation", "etag", "size"}
                }
                for name, row in sorted(self.claim_objects.items())
                if name.startswith(prefix)
            ]
            if self.result_list_pagination and page_token is None:
                return self._response(200, {"items": [], "nextPageToken": "page-2"})
            if page_token not in {None, "page-2"}:
                raise AssertionError("unexpected fake result page token")
            return self._response(200, {"items": matching})
        object_marker = f"/storage/v1/b/{subject.BUCKET}/o/"
        if parsed.netloc == "storage.googleapis.com" and parsed.path.startswith(
            object_marker
        ):
            name = urllib.parse.unquote(parsed.path[len(object_marker) :])
            row = self.claim_objects.get(name)
            if row is None:
                return self._response(404, {"error": self.provider_secret_body})
            if query.get("alt") == ["media"]:
                return self._response(200, row["payload"])
            return self._response(
                200,
                {key: value for key, value in row.items() if key != "payload"},
            )
        iam_path = f"/storage/v1/b/{subject.BUCKET}/iam"
        if parsed.netloc == "storage.googleapis.com" and parsed.path == iam_path:
            if method == "GET":
                return self._response(200, self.policy)
            assert method == "PUT"
            if self.iam_race:
                return self._response(412, {"error": self.provider_secret_body})
            assert body is not None
            supplied = json.loads(body)
            assert supplied["etag"] == self.policy["etag"]
            self.policy = supplied
            self.iam_put_count += 1
            self.policy["etag"] = f"BwWAfterHttpSet{self.iam_put_count}=="
            if self.iam_reorder_bindings:
                self.policy["bindings"] = list(reversed(self.policy["bindings"]))
            if self.iam_omit_empty_audit_configs:
                assert self.policy.get("auditConfigs") == []
                self.policy.pop("auditConfigs")
            if (
                self.iam_canonicalize_unconditional_version_one
                and not any(
                    "condition" in binding
                    for binding in self.policy["bindings"]
                )
            ):
                self.policy["version"] = 1
            if self.iam_semantic_drift:
                self.policy["bindings"][0]["members"].append(
                    "user:provider-drift@example.com"
                )
            if self.iam_drop_after_commit:
                self.iam_drop_after_commit = False
                raise ConnectionError("fake IAM PUT response lost after commit")
            return self._response(200, self.policy)
        raise AssertionError(f"unexpected fake HTTP request: {method} {url}")


def _fake_object(fake: FakeHttp, path: str) -> None:
    fake.claim_objects[path] = {
        "name": path,
        "generation": str(9000 + len(fake.claim_objects)),
        "etag": f"etag-{len(fake.claim_objects)}",
        "size": "2",
        "payload": b"{}",
    }


def _adapter(
    evidence: tuple[dict, dict, dict, dict],
    fake: FakeHttp,
    *,
    mode: str,
    gcp_read: Mapping[str, Any] | None = None,
    prepare: Mapping[str, Any] | None = None,
    install: Mapping[str, Any] | None = None,
    readback: Mapping[str, Any] | None = None,
) -> subject.GcpWavePhaseAAdapter:
    plan, ledger, resume, iam = evidence
    return subject.GcpWavePhaseAAdapter(
        mode=mode,
        project=subject.PROJECT,
        region=subject.REGION,
        zone=subject.ZONE,
        bucket=subject.BUCKET,
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        **_content_kwargs(),
        iam_plan=iam if mode != "claim" else None,
        gcp_read_receipt=gcp_read,
        prepare_receipt=prepare,
        install_receipt=install,
        readback_receipt=readback,
        requester=fake,
    )


def _prepare_receipt(
    evidence: tuple[dict, dict, dict, dict]
) -> dict[str, Any]:
    plan, ledger, resume, iam = evidence
    return iam_v2.prepare_worker_iam(
        iam_plan=iam,
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        **_content_kwargs(),
        backend=LocalPolicyBackend(_base_policy()),
    )


def test_read_mode_gets_exact_scope_and_builds_bound_receipt(
    monkeypatch: pytest.MonkeyPatch,
    evidence: tuple[dict, dict, dict, dict],
) -> None:
    monkeypatch.setenv(subject.TOKEN_ENV, TOKEN_A)
    fake = FakeHttp()
    adapter = _adapter(evidence, fake, mode="read")
    receipt = adapter.read_prelaunch(
        observed_at_utc=OBSERVED, expires_at_utc=EXPIRES
    )
    plan, ledger, resume, iam = evidence
    assert subject.validate_read_receipt(
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        **_content_kwargs(),
        iam_plan=iam,
        value=receipt,
    ) == receipt
    assert receipt["provider_quota_metrics"] == {
        "c4": subject.C4_QUOTA_METRIC,
        "c4_quota_id": subject.C4_QUOTA_ID,
        "c4_project_number": "123456789",
        "c4_usage_inventory_sha256": receipt["provider_quota_metrics"][
            "c4_usage_inventory_sha256"
        ],
        "spot": "PREEMPTIBLE_CPUS",
        "global": subject.GLOBAL_QUOTA_METRIC,
        "global_quota_id": subject.GLOBAL_QUOTA_ID,
        "global_usage_inventory_sha256": receipt["provider_quota_metrics"][
            "global_usage_inventory_sha256"
        ],
    }
    assert len(receipt["provider_quota_metrics"]["c4_usage_inventory_sha256"]) == 64
    assert (
        len(receipt["provider_quota_metrics"]["global_usage_inventory_sha256"])
        == 64
    )
    assert receipt["selected_vm_count"] == 8
    assert receipt["http_get_count"] == 49
    assert receipt["custom_role_count"] == len(subject.CUSTOM_ROLE_EXPECTATIONS)
    result_preflight = receipt["selected_result_preflight_receipt"]
    assert result_preflight["attempt_ledger_sha256"] == ledger["ledger_sha256"]
    assert result_preflight["resume_plan_sha256"] == resume["resume_sha256"]
    assert result_preflight["selected_attempt_count"] == 8
    assert result_preflight["http_get_count"] == 16
    assert receipt["observed_at_utc"] == result_preflight["observed_at_utc"]
    assert receipt["observed_at_utc"] == receipt["quota_receipt"]["observed_at_utc"]
    assert (
        receipt["observed_at_utc"]
        == receipt["planned_mapping_receipt"]["observed_at_utc"]
    )
    assert receipt["all_custom_roles_exact_ga_not_deleted"] is True
    assert receipt["custom_roles"] == [
        {
            "role_name": role,
            "included_permissions": list(permissions),
            "stage": "GA",
            "deleted": False,
            "etag_sha256": receipt["custom_roles"][index]["etag_sha256"],
        }
        for index, (role, permissions) in enumerate(subject.CUSTOM_ROLE_EXPECTATIONS)
    ]
    assert receipt["custom_roles_sha256"] == subject.canonical_sha256(
        receipt["custom_roles"]
    )
    assert all(method == "GET" and body is None for method, _, _, body, _ in fake.calls)
    assert all(headers["Authorization"] == f"Bearer {TOKEN_A}" for _, _, headers, _, _ in fake.calls)
    assert all(
        subject.PROJECT in url or f"/b/{subject.BUCKET}/" in url
        for _, url, _, _, _ in fake.calls
    )
    regional_calls = [
        url
        for method, url, _, body, _ in fake.calls
        if method == "GET"
        and body is None
        and urllib.parse.urlparse(url).path
        == f"/compute/v1/projects/{subject.PROJECT}/regions/{subject.REGION}"
    ]
    assert regional_calls == [
        "https://compute.googleapis.com/compute/v1/projects/"
        f"{subject.PROJECT}/regions/{subject.REGION}?fields=name,quotas"
    ]
    assert TOKEN_A not in subject.canonical_bytes(receipt).decode("ascii")
    assert fake.provider_secret_body not in subject.canonical_bytes(receipt).decode(
        "ascii"
    )

    tampered = copy.deepcopy(receipt)
    tampered["provider_quota_metrics"]["unrecognized"] = "OTHER_CPUS"
    unsigned = {key: value for key, value in tampered.items() if key != "receipt_sha256"}
    tampered["receipt_sha256"] = subject.canonical_sha256(unsigned)
    with pytest.raises(ValueError, match="provider quota metric fields"):
        subject.validate_read_receipt(
            wave_plan=plan,
            attempt_ledger=ledger,
            resume_plan=resume,
            **_content_kwargs(),
            iam_plan=iam,
            value=tampered,
        )

    stale_preflight = copy.deepcopy(receipt)
    stale_preflight["selected_result_preflight_receipt"]["observed_at_utc"] = (
        "2026-07-22T01:55:01Z"
    )
    nested = stale_preflight["selected_result_preflight_receipt"]
    nested_core = {
        key: value for key, value in nested.items() if key != "receipt_sha256"
    }
    nested["receipt_sha256"] = subject.canonical_sha256(nested_core)
    stale_core = {
        key: value
        for key, value in stale_preflight.items()
        if key != "receipt_sha256"
    }
    stale_preflight["receipt_sha256"] = subject.canonical_sha256(stale_core)
    with pytest.raises(ValueError, match="observation-time evidence chain"):
        subject.validate_read_receipt(
            wave_plan=plan,
            attempt_ledger=ledger,
            resume_plan=resume,
            **_content_kwargs(),
            iam_plan=iam,
            value=stale_preflight,
        )


def test_result_preflight_allows_prior_unselected_wave_results(
    monkeypatch: pytest.MonkeyPatch,
    evidence: tuple[dict, dict, dict, dict],
) -> None:
    monkeypatch.setenv(subject.TOKEN_ENV, TOKEN_A)
    plan = evidence[0]
    prior_job = plan["waves"][1]["job_ids"][0]
    prior_prefix = plan["artifact_contract"]["attempt_path_template"].format(
        job_id=prior_job, attempt_id="a00"
    )
    fake = FakeHttp()
    _fake_object(fake, f"{prior_prefix}/DONE.json")
    _fake_object(
        fake,
        plan["artifact_contract"]["job_acceptance_path_template"].format(
            job_id=prior_job
        ),
    )
    receipt = _adapter(evidence, fake, mode="read").read_prelaunch(
        observed_at_utc=OBSERVED, expires_at_utc=EXPIRES
    )
    assert receipt["selected_result_preflight_receipt"][
        "all_selected_artifact_prefixes_absent"
    ] is True


@pytest.mark.parametrize("stale", ["HEARTBEAT.json", "DONE.json", "ACCEPTED.json"])
def test_result_preflight_rejects_stale_selected_attempt_or_acceptance(
    monkeypatch: pytest.MonkeyPatch,
    evidence: tuple[dict, dict, dict, dict],
    stale: str,
) -> None:
    monkeypatch.setenv(subject.TOKEN_ENV, TOKEN_A)
    plan, _, resume, _ = evidence
    selected = resume["selected_attempts"][0]
    path = (
        plan["artifact_contract"]["job_acceptance_path_template"].format(
            job_id=selected["job_id"]
        )
        if stale == "ACCEPTED.json"
        else f"{selected['artifact_prefix']}/{stale}"
    )
    fake = FakeHttp()
    _fake_object(fake, path)
    with pytest.raises(FileExistsError, match="prefix|ACCEPTED"):
        _adapter(evidence, fake, mode="read").read_prelaunch(
            observed_at_utc=OBSERVED, expires_at_utc=EXPIRES
        )


def test_result_preflight_supports_paginated_empty_prefix_lists(
    monkeypatch: pytest.MonkeyPatch,
    evidence: tuple[dict, dict, dict, dict],
) -> None:
    monkeypatch.setenv(subject.TOKEN_ENV, TOKEN_A)
    fake = FakeHttp()
    fake.result_list_pagination = True
    receipt = _adapter(evidence, fake, mode="read").read_prelaunch(
        observed_at_utc=OBSERVED, expires_at_utc=EXPIRES
    )
    result = receipt["selected_result_preflight_receipt"]
    assert result["pagination_observed"] is True
    assert result["http_get_count"] == 24
    assert all(row["list_page_count"] == 2 for row in result["rows"])


def test_a01_result_preflight_ignores_same_job_a00_attempt_objects(
    monkeypatch: pytest.MonkeyPatch,
    evidence: tuple[dict, dict, dict, dict],
) -> None:
    monkeypatch.setenv(subject.TOKEN_ENV, TOKEN_A)
    plan = evidence[0]
    baseline = wave_v2.build_observed_transition(
        plan,
        project_id="ofc-project-123",
        zone=subject.ZONE,
        observed_at_utc="2026-07-22T02:00:00Z",
        previous_transition_digest=None,
        attempt_history=wave_v2.empty_attempt_history(plan),
    )
    history = copy.deepcopy(baseline["attempt_history"])
    instances: dict[str, str] = {}
    for pair in plan["waves"][0]["candidate_reference_pairs"]:
        instances[pair["candidate_job_id"]] = pair[
            "candidate_attempt_instance_ids"
        ]["a00"]
        instances[pair["reference_job_id"]] = pair[
            "reference_attempt_instance_ids"
        ]["a00"]
    wave0_jobs = set(plan["waves"][0]["job_ids"])
    for row in history:
        if row["job_id"] in wave0_jobs:
            row["attempts"].append(
                {
                    "attempt_id": "a00",
                    "instance_id": instances[row["job_id"]],
                    "launch_receipt_sha256": hashlib.sha256(
                        f"failed-{row['job_id']}".encode()
                    ).hexdigest(),
                    "terminal_status": "failed",
                }
            )
    failed = wave_v2.build_observed_transition(
        plan,
        project_id="ofc-project-123",
        zone=subject.ZONE,
        observed_at_utc="2026-07-22T02:00:01Z",
        previous_transition_digest=baseline["transition_digest"],
        attempt_history=history,
    )
    ledger = wave_v2.build_attempt_ledger(
        plan,
        transitions=[baseline, failed],
        consumed_transition_digests=[baseline["transition_digest"]],
    )
    resume = wave_v2.build_resume_plan(plan, attempt_ledger=ledger)
    assert all(row["attempt_id"] == "a01" for row in resume["selected_attempts"])
    iam = iam_v2.build_worker_iam_plan(
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        wave_index=0,
        **_content_kwargs(),
        issued_at_unix_seconds=ISSUED,
    )
    retry_evidence = (plan, ledger, resume, iam)
    fake = FakeHttp()
    for selected in resume["selected_attempts"]:
        _fake_object(
            fake,
            f"{selected['artifact_prefix'].replace('/a01', '/a00')}/DONE.json",
        )
    receipt = _adapter(retry_evidence, fake, mode="read").read_prelaunch(
        observed_at_utc=OBSERVED, expires_at_utc=EXPIRES
    )
    result = receipt["selected_result_preflight_receipt"]
    assert result["attempt_ledger_sha256"] == ledger["ledger_sha256"]
    assert result["resume_plan_sha256"] == resume["resume_sha256"]


@pytest.mark.parametrize(
    "fault", ["extra_permission", "missing_permission", "stage", "deleted"]
)
def test_phase_a_read_rejects_custom_role_permission_or_lifecycle_drift(
    monkeypatch: pytest.MonkeyPatch,
    evidence: tuple[dict, dict, dict, dict],
    fault: str,
) -> None:
    monkeypatch.setenv(subject.TOKEN_ENV, TOKEN_A)
    fake = FakeHttp()
    fake.custom_role_fault = fault
    with pytest.raises(RuntimeError, match="custom role"):
        _adapter(evidence, fake, mode="read").read_prelaunch(
            observed_at_utc=OBSERVED,
            expires_at_utc=EXPIRES,
        )


@pytest.mark.parametrize(
    ("attribute", "message"),
    [
        ("pagination", "pagination"),
        ("unknown_spot_quota", "missing or ambiguous"),
        ("wrong_region_scope", "scope changed"),
        ("regional_unknown_top_level", "top-level fields changed"),
        ("regional_unknown_row_field", "quota entry fields changed"),
        ("regional_missing_quotas", "top-level fields changed"),
        ("quota_404", "status 404"),
    ],
)
def test_read_quota_pagination_unknown_and_404_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    evidence: tuple[dict, dict, dict, dict],
    attribute: str,
    message: str,
) -> None:
    monkeypatch.setenv(subject.TOKEN_ENV, TOKEN_A)
    fake = FakeHttp()
    setattr(fake, attribute, True)
    with pytest.raises((RuntimeError, ValueError), match=message) as error:
        _adapter(evidence, fake, mode="read").read_prelaunch(
            observed_at_utc=OBSERVED, expires_at_utc=EXPIRES
        )
    assert TOKEN_A not in str(error.value)
    assert fake.provider_secret_body not in str(error.value)


def test_read_instance_disk_and_service_account_collision_semantics(
    monkeypatch: pytest.MonkeyPatch,
    evidence: tuple[dict, dict, dict, dict],
) -> None:
    monkeypatch.setenv(subject.TOKEN_ENV, TOKEN_A)
    selected = evidence[2]["selected_attempts"][0]["instance_id"]
    fake = FakeHttp()
    fake.instance_collision = selected
    with pytest.raises(FileExistsError, match="instance"):
        _adapter(evidence, fake, mode="read").read_prelaunch(
            observed_at_utc=OBSERVED, expires_at_utc=EXPIRES
        )
    fake = FakeHttp()
    fake.disk_collision = selected
    with pytest.raises(FileExistsError, match="disk"):
        _adapter(evidence, fake, mode="read").read_prelaunch(
            observed_at_utc=OBSERVED, expires_at_utc=EXPIRES
        )
    fake = FakeHttp()
    fake.service_account_404 = evidence[3]["workers"][0]["service_account"]
    with pytest.raises(RuntimeError, match="status 404"):
        _adapter(evidence, fake, mode="read").read_prelaunch(
            observed_at_utc=OBSERVED, expires_at_utc=EXPIRES
        )


def test_actas_check_is_exact_post_only_and_receipt_bound(
    monkeypatch: pytest.MonkeyPatch,
    evidence: tuple[dict, dict, dict, dict],
) -> None:
    monkeypatch.setenv(subject.TOKEN_ENV, TOKEN_A)
    read_fake = FakeHttp()
    read = _adapter(evidence, read_fake, mode="read").read_prelaunch(
        observed_at_utc=OBSERVED,
        expires_at_utc=EXPIRES,
    )
    fake = FakeHttp()
    adapter = _adapter(
        evidence,
        fake,
        mode="actas-check",
        gcp_read=read,
    )
    receipt = adapter.check_service_account_act_as(
        checked_at_utc=OBSERVED,
        expires_at_utc=EXPIRES,
    )
    plan, ledger, resume, iam = evidence
    assert subject.validate_service_account_actas_receipt(
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        **_content_kwargs(),
        iam_plan=iam,
        gcp_read_receipt=read,
        value=receipt,
        now_utc="2026-07-22T02:00:02Z",
    ) == receipt
    assert receipt["gcp_read_receipt_sha256"] == read["receipt_sha256"]
    assert receipt["http_post_count"] == 9
    assert len(fake.calls) == 9
    assert receipt["provider_project_permissions_granted"] == list(
        subject.PROVIDER_PROJECT_PERMISSIONS
    )
    assert "compute.disks.setLabels" in receipt[
        "provider_project_permissions_granted"
    ]
    assert all(method == "POST" for method, *_ in fake.calls)
    assert all(
        urllib.parse.urlparse(url).path.endswith(":testIamPermissions")
        for _, url, _, _, _ in fake.calls
    )
    actas_body = subject.canonical_bytes(
        {"permissions": [subject.ACT_AS_PERMISSION]}
    )
    provider_body = subject.canonical_bytes(
        {"permissions": list(subject.PROVIDER_PROJECT_PERMISSIONS)}
    )
    assert [body for _, _, _, body, _ in fake.calls].count(actas_body) == 8
    assert [body for _, _, _, body, _ in fake.calls].count(provider_body) == 1
    assert all(
        headers["Authorization"] == f"Bearer {TOKEN_A}"
        and headers["Content-Type"] == "application/json"
        for _, _, headers, _, _ in fake.calls
    )
    assert TOKEN_A not in subject.canonical_bytes(receipt).decode("ascii")


@pytest.mark.parametrize(
    "response_drift", ["denied", "extra", "provider_denied", "provider_extra"]
)
def test_actas_check_denial_or_response_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    evidence: tuple[dict, dict, dict, dict],
    response_drift: str,
) -> None:
    monkeypatch.setenv(subject.TOKEN_ENV, TOKEN_A)
    read = _adapter(evidence, FakeHttp(), mode="read").read_prelaunch(
        observed_at_utc=OBSERVED,
        expires_at_utc=EXPIRES,
    )
    fake = FakeHttp()
    if response_drift == "denied":
        fake.actas_denied = evidence[3]["workers"][0]["service_account"]
        expected = PermissionError
    elif response_drift == "extra":
        fake.actas_extra_field = True
        expected = RuntimeError
    elif response_drift == "provider_denied":
        fake.provider_permission_denied = "compute.disks.setLabels"
        expected = PermissionError
    else:
        fake.provider_permission_extra_field = True
        expected = RuntimeError
    with pytest.raises(expected) as error:
        _adapter(
            evidence,
            fake,
            mode="actas-check",
            gcp_read=read,
        ).check_service_account_act_as(
            checked_at_utc=OBSERVED,
            expires_at_utc=EXPIRES,
        )
    assert TOKEN_A not in str(error.value)
    assert fake.provider_secret_body not in str(error.value)


def test_claim_mode_is_exact_create_only_and_token_is_per_request(
    monkeypatch: pytest.MonkeyPatch,
    evidence: tuple[dict, dict, dict, dict],
) -> None:
    plan, ledger, resume, _iam = evidence
    fake = FakeHttp()
    adapter = _adapter(evidence, fake, mode="claim")
    monkeypatch.setenv(subject.TOKEN_ENV, TOKEN_A)
    claim = cloud_v2.create_persistent_atomic_launch_claim(
        plan,
        ledger,
        resume,
        immutable_content_sha256=CONTENT_SHA,
        project_id=subject.PROJECT,
        zone=subject.ZONE,
        claim_nonce=NONCE,
        claimed_at_utc="2026-07-22T02:00:02Z",
        backend=adapter,
    )
    assert [method for method, *_ in fake.calls] == ["POST", "GET", "GET"]
    upload = fake.calls[0]
    assert upload[3] == cloud_v2.canonical_bytes(claim["claim_payload"])
    query = urllib.parse.parse_qs(urllib.parse.urlparse(upload[1]).query)
    assert query["ifGenerationMatch"] == ["0"]
    assert query["uploadType"] == ["media"]
    assert upload[2]["Authorization"] == f"Bearer {TOKEN_A}"

    monkeypatch.setenv(subject.TOKEN_ENV, TOKEN_B)
    adapter.get_object(object_name=claim["object_path"])
    assert fake.calls[-1][2]["Authorization"] == f"Bearer {TOKEN_B}"
    assert TOKEN_A not in subject.canonical_bytes(claim).decode("ascii")
    assert TOKEN_B not in subject.canonical_bytes(claim).decode("ascii")

    with pytest.raises(FileExistsError) as error:
        cloud_v2.create_persistent_atomic_launch_claim(
            plan,
            ledger,
            resume,
            immutable_content_sha256=CONTENT_SHA,
            project_id=subject.PROJECT,
            zone=subject.ZONE,
            claim_nonce=NONCE,
            claimed_at_utc="2026-07-22T02:00:02Z",
            backend=adapter,
        )
    assert TOKEN_B not in str(error.value)
    assert fake.provider_secret_body not in str(error.value)


def test_claim_post_response_loss_recovers_by_exact_object_get_without_repost(
    monkeypatch: pytest.MonkeyPatch,
    evidence: tuple[dict, dict, dict, dict],
) -> None:
    plan, ledger, resume, _iam = evidence
    fake = FakeHttp()
    fake.claim_drop_after_commit = True
    adapter = _adapter(evidence, fake, mode="claim")
    monkeypatch.setenv(subject.TOKEN_ENV, TOKEN_A)
    claim = cloud_v2.create_persistent_atomic_launch_claim(
        plan,
        ledger,
        resume,
        immutable_content_sha256=CONTENT_SHA,
        project_id=subject.PROJECT,
        zone=subject.ZONE,
        claim_nonce=NONCE,
        claimed_at_utc="2026-07-22T02:00:02Z",
        backend=adapter,
    )
    methods = [method for method, *_ in fake.calls]
    assert methods == ["POST", "GET", "GET", "GET", "GET"]
    assert methods.count("POST") == 1
    assert claim["claim_payload"] == json.loads(
        fake.claim_objects[claim["object_path"]]["payload"]
    )
    assert claim["generation"] == "1001"
    assert claim["etag"] == "claim-etag-1001"


def test_bucket_iam_install_cleanup_are_exact_single_cas(
    monkeypatch: pytest.MonkeyPatch,
    evidence: tuple[dict, dict, dict, dict],
) -> None:
    monkeypatch.setenv(subject.TOKEN_ENV, TOKEN_A)
    plan, ledger, resume, iam = evidence
    fake = FakeHttp()
    prepare = _prepare_receipt(evidence)
    install_adapter = _adapter(
        evidence, fake, mode="bucket-iam-install", prepare=prepare
    )
    install = iam_v2.install_worker_iam(
        iam_plan=iam,
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        **_content_kwargs(),
        prepare_receipt=prepare,
        backend=install_adapter,
    )
    readback = iam_v2.readback_worker_iam(
        iam_plan=iam,
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        **_content_kwargs(),
        prepare_receipt=prepare,
        install_receipt=install,
        backend=install_adapter,
    )
    cleanup_adapter = _adapter(
        evidence,
        fake,
        mode="bucket-iam-cleanup",
        prepare=prepare,
        install=install,
        readback=readback,
    )
    cleanup = iam_v2.cleanup_worker_iam(
        iam_plan=iam,
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        **_content_kwargs(),
        prepare_receipt=prepare,
        install_receipt=install,
        readback_receipt=readback,
        backend=cleanup_adapter,
    )
    puts = [call for call in fake.calls if call[0] == "PUT"]
    assert len(puts) == 2
    assert all(
        urllib.parse.urlparse(url).path
        == f"/storage/v1/b/{subject.BUCKET}/iam"
        for _, url, _, _, _ in puts
    )
    assert all(body is not None for _, _, _, body, _ in puts)
    assert cleanup["cleanup_complete"] is True
    assert TOKEN_A not in subject.canonical_bytes(cleanup).decode("ascii")


def test_cleanup_accepts_only_real_gcs_unconditional_version_canonicalization(
    monkeypatch: pytest.MonkeyPatch,
    evidence: tuple[dict, dict, dict, dict],
) -> None:
    monkeypatch.setenv(subject.TOKEN_ENV, TOKEN_A)
    plan, ledger, resume, iam = evidence
    fake = FakeHttp()
    fake.iam_canonicalize_unconditional_version_one = True
    prepare = _prepare_receipt(evidence)
    install = iam_v2.install_worker_iam(
        iam_plan=iam,
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        **_content_kwargs(),
        prepare_receipt=prepare,
        backend=_adapter(
            evidence, fake, mode="bucket-iam-install", prepare=prepare
        ),
    )
    readback = iam_v2.readback_worker_iam(
        iam_plan=iam,
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        **_content_kwargs(),
        prepare_receipt=prepare,
        install_receipt=install,
        backend=_adapter(
            evidence,
            fake,
            mode="bucket-iam-readback",
            prepare=prepare,
            install=install,
        ),
    )
    cleanup = iam_v2.cleanup_worker_iam(
        iam_plan=iam,
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        **_content_kwargs(),
        prepare_receipt=prepare,
        install_receipt=install,
        readback_receipt=readback,
        backend=_adapter(
            evidence,
            fake,
            mode="bucket-iam-cleanup",
            prepare=prepare,
            install=install,
            readback=readback,
        ),
    )
    assert cleanup["cleanup_complete"] is True
    assert fake.iam_put_count == 2
    assert fake.policy["version"] == 1
    assert not any("condition" in row for row in fake.policy["bindings"])


@pytest.mark.parametrize(
    "drift",
    ["binding", "audit", "kind", "resource", "condition", "version"],
)
def test_provider_version_canonicalization_never_masks_semantic_drift(
    drift: str,
) -> None:
    supplied = _base_policy()
    result = copy.deepcopy(supplied)
    result["etag"] = "BwWCanonicalized=="
    result["version"] = 1
    assert subject._provider_policy_response_matches_exact_mutation(
        supplied=supplied, result=result
    )
    if drift == "binding":
        result["bindings"][0]["members"].append("user:drift@example.com")
    elif drift == "audit":
        result["auditConfigs"] = [{"service": "storage.googleapis.com"}]
    elif drift == "kind":
        result["kind"] = "storage#otherPolicy"
    elif drift == "resource":
        result["resourceId"] += "-drift"
    elif drift == "condition":
        condition = {"title": "drift", "expression": "request.time < timestamp(\"2030-01-01T00:00:00Z\")"}
        supplied["bindings"][0]["condition"] = copy.deepcopy(condition)
        result["bindings"][0]["condition"] = condition
    else:
        result["version"] = 2
    assert not subject._provider_policy_response_matches_exact_mutation(
        supplied=supplied, result=result
    )


def test_bucket_iam_put_accepts_provider_order_and_empty_field_normalization(
    monkeypatch: pytest.MonkeyPatch,
    evidence: tuple[dict, dict, dict, dict],
) -> None:
    monkeypatch.setenv(subject.TOKEN_ENV, TOKEN_A)
    plan, ledger, resume, iam = evidence
    fake = FakeHttp()
    fake.iam_reorder_bindings = True
    fake.iam_omit_empty_audit_configs = True
    prepare = _prepare_receipt(evidence)
    install = iam_v2.install_worker_iam(
        iam_plan=iam,
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        **_content_kwargs(),
        prepare_receipt=prepare,
        backend=_adapter(
            evidence, fake, mode="bucket-iam-install", prepare=prepare
        ),
    )
    assert install["install_complete"] is True
    assert fake.iam_put_count == 1
    assert "auditConfigs" not in fake.policy


def test_bucket_iam_put_still_rejects_provider_semantic_drift(
    monkeypatch: pytest.MonkeyPatch,
    evidence: tuple[dict, dict, dict, dict],
) -> None:
    monkeypatch.setenv(subject.TOKEN_ENV, TOKEN_A)
    plan, ledger, resume, iam = evidence
    fake = FakeHttp()
    fake.iam_semantic_drift = True
    prepare = _prepare_receipt(evidence)
    with pytest.raises(RuntimeError, match="readback changed"):
        iam_v2.install_worker_iam(
            iam_plan=iam,
            wave_plan=plan,
            attempt_ledger=ledger,
            resume_plan=resume,
            **_content_kwargs(),
            prepare_receipt=prepare,
            backend=_adapter(
                evidence, fake, mode="bucket-iam-install", prepare=prepare
            ),
        )
    assert fake.iam_put_count == 1


def test_bucket_iam_install_and_cleanup_response_loss_reconcile_get_only(
    monkeypatch: pytest.MonkeyPatch,
    evidence: tuple[dict, dict, dict, dict],
) -> None:
    monkeypatch.setenv(subject.TOKEN_ENV, TOKEN_A)
    plan, ledger, resume, iam = evidence
    fake = FakeHttp()
    prepare = _prepare_receipt(evidence)
    install_adapter = _adapter(
        evidence, fake, mode="bucket-iam-install", prepare=prepare
    )
    fake.iam_drop_after_commit = True
    with pytest.raises(subject.GcpPhaseATransportError):
        iam_v2.install_worker_iam(
            iam_plan=iam,
            wave_plan=plan,
            attempt_ledger=ledger,
            resume_plan=resume,
            **_content_kwargs(),
            prepare_receipt=prepare,
            backend=install_adapter,
        )
    calls_after_install = len(fake.calls)
    recover_install_adapter = _adapter(
        evidence,
        fake,
        mode="bucket-iam-reconcile-install",
        prepare=prepare,
    )
    recovered_install = iam_v2.reconcile_install_worker_iam(
        iam_plan=iam,
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        **_content_kwargs(),
        prepare_receipt=prepare,
        backend=recover_install_adapter,
    )
    assert recovered_install["recovered_after_transport_ambiguity"] is True
    assert [call[0] for call in fake.calls[calls_after_install:]] == ["GET"]
    readback = iam_v2.readback_worker_iam(
        iam_plan=iam,
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        **_content_kwargs(),
        prepare_receipt=prepare,
        install_receipt=recovered_install,
        backend=recover_install_adapter,
    )
    cleanup_adapter = _adapter(
        evidence,
        fake,
        mode="bucket-iam-cleanup",
        prepare=prepare,
        install=recovered_install,
        readback=readback,
    )
    fake.iam_drop_after_commit = True
    with pytest.raises(subject.GcpPhaseATransportError):
        iam_v2.cleanup_worker_iam(
            iam_plan=iam,
            wave_plan=plan,
            attempt_ledger=ledger,
            resume_plan=resume,
            **_content_kwargs(),
            prepare_receipt=prepare,
            install_receipt=recovered_install,
            readback_receipt=readback,
            backend=cleanup_adapter,
        )
    calls_after_cleanup = len(fake.calls)
    recover_cleanup_adapter = _adapter(
        evidence,
        fake,
        mode="bucket-iam-reconcile-cleanup",
        prepare=prepare,
        install=recovered_install,
        readback=readback,
    )
    recovered_cleanup = iam_v2.reconcile_cleanup_worker_iam(
        iam_plan=iam,
        wave_plan=plan,
        attempt_ledger=ledger,
        resume_plan=resume,
        **_content_kwargs(),
        prepare_receipt=prepare,
        install_receipt=recovered_install,
        readback_receipt=readback,
        backend=recover_cleanup_adapter,
    )
    assert recovered_cleanup["recovered_after_outcome_ambiguity"] is True
    assert recovered_cleanup["removed_binding_count"] == 0
    assert recovered_cleanup["set_attempt_count"] == 0
    assert recovered_cleanup["cloud_mutation_performed"] is False
    assert recovered_cleanup["source_mutation_outcome"] == "unknown"
    assert recovered_cleanup["cleanup_complete"] is True
    assert [call[0] for call in fake.calls[calls_after_cleanup:]] == ["GET"]
    assert fake.iam_put_count == 2


def test_bucket_iam_etag_race_wrong_scope_and_mode_escape(
    monkeypatch: pytest.MonkeyPatch,
    evidence: tuple[dict, dict, dict, dict],
) -> None:
    monkeypatch.setenv(subject.TOKEN_ENV, TOKEN_A)
    plan, ledger, resume, iam = evidence
    fake = FakeHttp()
    fake.iam_race = True
    prepare = _prepare_receipt(evidence)
    adapter = _adapter(
        evidence, fake, mode="bucket-iam-install", prepare=prepare
    )
    with pytest.raises(iam_v2.WorkerIamCasError, match="CAS") as error:
        iam_v2.install_worker_iam(
            iam_plan=iam,
            wave_plan=plan,
            attempt_ledger=ledger,
            resume_plan=resume,
            **_content_kwargs(),
            prepare_receipt=prepare,
            backend=adapter,
        )
    assert TOKEN_A not in str(error.value)
    assert fake.provider_secret_body not in str(error.value)
    assert len([call for call in fake.calls if call[0] == "PUT"]) == 1

    with pytest.raises(ValueError, match="fixed project scope"):
        subject.GcpWavePhaseAAdapter(
            mode="read",
            project="wrong-project-123",
            region=subject.REGION,
            zone=subject.ZONE,
            bucket=subject.BUCKET,
            wave_plan=plan,
            attempt_ledger=ledger,
            resume_plan=resume,
            **_content_kwargs(),
            iam_plan=iam,
            requester=FakeHttp(),
        )

    read_adapter = _adapter(evidence, FakeHttp(), mode="read")
    with pytest.raises(PermissionError, match="claim mode"):
        read_adapter.put_if_absent(object_name="x", payload=b"x")
    with pytest.raises(PermissionError, match="actAs check mode"):
        read_adapter.check_service_account_act_as(
            checked_at_utc=OBSERVED,
            expires_at_utc=EXPIRES,
        )
    with pytest.raises(PermissionError, match="Phase B"):
        read_adapter.create_instance(specification={})
    claim_adapter = _adapter(evidence, FakeHttp(), mode="claim")
    with pytest.raises(PermissionError, match="IAM mode"):
        claim_adapter.get_bucket_policy()
    with pytest.raises(PermissionError, match="read mode"):
        claim_adapter.read_prelaunch(
            observed_at_utc=OBSERVED, expires_at_utc=EXPIRES
        )
    with pytest.raises(PermissionError, match="requires IAM plan and GCP read"):
        _adapter(evidence, FakeHttp(), mode="actas-check")
