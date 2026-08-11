"""Exact lifecycle for the fresh Step 12b controller service account.

The controller identity is derived by the offline deployment contract.  This
module can only create, inspect, and delete that one identity through injected
user credentials.  It never retries a mutation and never accepts the legacy
shared controller service account.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import time
import urllib.parse
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_canary_gce_transport_10c2_v1
    as transport,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step11_rest_iam_admin_v1
    as rest_iam,
)
from . import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12b_deployment_contract_v2
    as deployment_v2,
)


SCHEMA = (
    "hu_m31_t3_step6d_rearm2_diagnostic_"
    "step12b_run_scoped_controller_sa_lifecycle_v2"
)
LEGACY_CONTROLLER_SERVICE_ACCOUNT = (
    f"ofc-m31-t3-controller@{transport.PROJECT}.iam.gserviceaccount.com"
)
MAX_READBACK_WAIT_SECONDS = 120
READBACK_POLL_SECONDS = 2.0

_ACCOUNT_ID = re.compile(r"^[a-z][a-z0-9-]{4,28}[a-z0-9]$")
_EMAIL = re.compile(
    r"^(?P<account>[a-z][a-z0-9-]{4,28}[a-z0-9])@"
    r"(?P<project>[a-z][a-z0-9-]{4,61}[a-z0-9])"
    r"\.iam\.gserviceaccount\.com$"
)
_DECIMAL_ID = re.compile(r"^[1-9][0-9]{5,31}$")
_RUN_TAG = re.compile(r"^[0-9a-f]{12}$")


class UserTokenSource(Protocol):
    def access_token(self) -> str: ...


class JsonHttpsClient(Protocol):
    def request(
        self,
        *,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout_seconds: int,
    ) -> rest_iam.HttpResponse: ...


@dataclass(frozen=True)
class ControllerServiceAccountIdentity:
    project: str
    account_id: str
    email: str
    principal: str


class RunScopedControllerServiceAccountError(RuntimeError):
    def __init__(self, code: str, *, status_code: int | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.status_code = status_code

    def __str__(self) -> str:
        return self.code


class _FreshAbsenceCapability:
    __slots__ = ("_owner", "_consumed", "identity", "receipt")

    def __init__(
        self,
        *,
        owner: object,
        identity: ControllerServiceAccountIdentity,
        receipt: Mapping[str, Any],
    ) -> None:
        self._owner = owner
        self._consumed = False
        self.identity = identity
        self.receipt = copy.deepcopy(dict(receipt))

    def __repr__(self) -> str:
        return "<_FreshAbsenceCapability verified_get_404=True>"


class _CreatedControllerCapability:
    __slots__ = ("_owner", "identity", "provider", "receipt")

    def __init__(
        self,
        *,
        owner: object,
        identity: ControllerServiceAccountIdentity,
        provider: Mapping[str, Any],
        receipt: Mapping[str, Any],
    ) -> None:
        self._owner = owner
        self.identity = identity
        self.provider = copy.deepcopy(dict(provider))
        self.receipt = copy.deepcopy(dict(receipt))

    def __repr__(self) -> str:
        return (
            "<_CreatedControllerCapability "
            f"unique_id={self.provider.get('unique_id')!r}>"
        )


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    body = copy.deepcopy(dict(value))
    return {**body, "receipt_sha256": canonical_sha256(body)}


def validate_controller_identity(
    value: Mapping[str, Any],
) -> ControllerServiceAccountIdentity:
    if not isinstance(value, Mapping):
        raise ValueError("Step12b controller service-account record changed")
    expected_fields = {
        "project",
        "account_id",
        "email",
        "principal",
        "derived_from_run_tag",
        "run_scoped",
        "legacy_shared_controller_reused",
    }
    if set(value) != expected_fields:
        raise ValueError("Step12b controller service-account fields changed")
    project = value["project"]
    account_id = value["account_id"]
    email = value["email"]
    principal = value["principal"]
    match = _EMAIL.fullmatch(email) if isinstance(email, str) else None
    if (
        project != transport.PROJECT
        or not isinstance(account_id, str)
        or _ACCOUNT_ID.fullmatch(account_id) is None
        or len(account_id) > 30
        or match is None
        or match.group("account") != account_id
        or match.group("project") != project
        or principal != f"serviceAccount:{email}"
        or not isinstance(value["derived_from_run_tag"], str)
        or _RUN_TAG.fullmatch(value["derived_from_run_tag"]) is None
        or account_id
        != f"ofc-m31-s2b-{value['derived_from_run_tag']}"
        or value["run_scoped"] is not True
        or value["legacy_shared_controller_reused"] is not False
        or email == LEGACY_CONTROLLER_SERVICE_ACCOUNT
    ):
        raise ValueError("Step12b controller service-account identity changed")
    return ControllerServiceAccountIdentity(
        project=project,
        account_id=account_id,
        email=email,
        principal=principal,
    )


def identity_from_deployment(
    deployment_contract: Mapping[str, Any],
    *,
    candidate_payload_contract: Mapping[str, Any],
    reference_payload_contract: Mapping[str, Any],
    controller_public_key_record: Mapping[str, Any],
    run_nonce: str,
) -> ControllerServiceAccountIdentity:
    checked = deployment_v2.validate_deployment_contract(
        deployment_contract,
        candidate_payload_contract=candidate_payload_contract,
        reference_payload_contract=reference_payload_contract,
        controller_public_key_record=controller_public_key_record,
        run_nonce=run_nonce,
    )
    return validate_controller_identity(
        checked["controller_service_account"]
    )


class RunScopedControllerServiceAccountAdmin:
    """User-credential REST client bound to one fresh service account."""

    def __init__(
        self,
        *,
        http_client: JsonHttpsClient,
        user_token_source: UserTokenSource,
        identity: ControllerServiceAccountIdentity,
    ) -> None:
        if identity.project != transport.PROJECT:
            raise ValueError("Step12b controller project changed")
        self._http = http_client
        self._tokens = user_token_source
        self.identity = identity
        self._capability_owner = object()

    @property
    def _collection_url(self) -> str:
        return (
            "https://iam.googleapis.com/v1/projects/"
            f"{self.identity.project}/serviceAccounts"
        )

    @property
    def _resource_url(self) -> str:
        email = urllib.parse.quote(self.identity.email, safe="")
        return (
            "https://iam.googleapis.com/v1/projects/"
            f"{self.identity.project}/serviceAccounts/{email}"
        )

    def _request(
        self,
        *,
        method: str,
        url: str,
        body: Mapping[str, Any] | None = None,
    ) -> rest_iam.HttpResponse:
        token = self._tokens.access_token()
        if (
            not isinstance(token, str)
            or not token
            or len(token) > 16_384
            or any(character.isspace() for character in token)
        ):
            raise ValueError("user access token shape changed")
        return self._http.request(
            method=method,
            url=url,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
                **(
                    {"Content-Type": "application/json"}
                    if body is not None
                    else {}
                ),
            },
            body=canonical_bytes(body) if body is not None else None,
            timeout_seconds=60,
        )

    @staticmethod
    def _json(response: rest_iam.HttpResponse, operation: str) -> dict[str, Any]:
        if len(response.body) > 1_048_576:
            raise RunScopedControllerServiceAccountError(
                f"{operation}_response_too_large",
                status_code=response.status_code,
            )
        try:
            value = json.loads(response.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise RunScopedControllerServiceAccountError(
                f"{operation}_response_changed",
                status_code=response.status_code,
            ) from None
        if not isinstance(value, dict):
            raise RunScopedControllerServiceAccountError(
                f"{operation}_response_changed",
                status_code=response.status_code,
            )
        return value

    def get(self) -> dict[str, Any] | None:
        response = self._request(method="GET", url=self._resource_url)
        if response.status_code == 404:
            return None
        if response.status_code != 200:
            raise RunScopedControllerServiceAccountError(
                "controller_service_account_get_failed",
                status_code=response.status_code,
            )
        return self._validate_provider_record(
            self._json(response, "controller_service_account_get")
        )

    def _validate_provider_record(
        self, value: Mapping[str, Any]
    ) -> dict[str, Any]:
        name = value.get("name")
        project_id = value.get("projectId")
        unique_id = value.get("uniqueId")
        email = value.get("email")
        disabled = value.get("disabled", False)
        expected_name = (
            f"projects/{self.identity.project}/serviceAccounts/"
            f"{self.identity.email}"
        )
        if (
            name != expected_name
            or project_id != self.identity.project
            or email != self.identity.email
            or not isinstance(unique_id, str)
            or _DECIMAL_ID.fullmatch(unique_id) is None
            or disabled is not False
        ):
            raise RunScopedControllerServiceAccountError(
                "controller_service_account_provider_identity_changed"
            )
        return {
            "name": name,
            "project_id": project_id,
            "unique_id": unique_id,
            "email": email,
            "disabled": False,
        }

    def require_absent(self) -> _FreshAbsenceCapability:
        if self.get() is not None:
            raise RunScopedControllerServiceAccountError(
                "controller_service_account_freshness_failed"
            )
        receipt = _seal(
            {
                "schema": SCHEMA,
                "status": "run_scoped_controller_service_account_absent",
                "project": self.identity.project,
                "account_id": self.identity.account_id,
                "email": self.identity.email,
                "get_status": 404,
                "cloud_mutation_performed": False,
            }
        )
        return _FreshAbsenceCapability(
            owner=self._capability_owner,
            identity=self.identity,
            receipt=receipt,
        )

    def create(
        self, *, absence: _FreshAbsenceCapability
    ) -> _CreatedControllerCapability:
        """Create exactly once after this admin produced a GET-404 capability."""

        if (
            not isinstance(absence, _FreshAbsenceCapability)
            or absence._owner is not self._capability_owner
            or absence._consumed
            or absence.identity != self.identity
            or absence.receipt.get("status")
            != "run_scoped_controller_service_account_absent"
            or absence.receipt.get("get_status") != 404
        ):
            raise PermissionError(
                "fresh controller service-account absence proof is missing"
            )
        # Consume before the POST.  An unknown transport outcome must never be
        # converted into a second create attempt.
        absence._consumed = True

        response = self._request(
            method="POST",
            url=self._collection_url,
            body={
                "accountId": self.identity.account_id,
                "serviceAccount": {
                    "displayName": "OFC M31 Step12b run-scoped controller",
                    "description": (
                        "Fresh attempt0 controller; deleted before pair release."
                    ),
                },
            },
        )
        if response.status_code != 200:
            raise RunScopedControllerServiceAccountError(
                "controller_service_account_create_failed",
                status_code=response.status_code,
            )
        created = self._validate_provider_record(
            self._json(response, "controller_service_account_create")
        )
        readback = self.get()
        if readback != created:
            raise RunScopedControllerServiceAccountError(
                "controller_service_account_create_readback_changed"
            )
        receipt = _seal(
            {
                "schema": SCHEMA,
                "status": "run_scoped_controller_service_account_created",
                "project": self.identity.project,
                "account_id": self.identity.account_id,
                "email": self.identity.email,
                "provider": created,
                "create_call_count": 1,
                "create_retry_count": 0,
                "readback_verified": True,
                "cloud_mutation_performed": True,
            }
        )
        return _CreatedControllerCapability(
            owner=self._capability_owner,
            identity=self.identity,
            provider=created,
            receipt=receipt,
        )

    def delete_created_and_wait_absent(
        self,
        *,
        created: _CreatedControllerCapability,
        now: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        maximum_wait_seconds: int = MAX_READBACK_WAIT_SECONDS,
    ) -> dict[str, Any]:
        """Prove success-path teardown of the exact identity created by us."""

        if maximum_wait_seconds != MAX_READBACK_WAIT_SECONDS:
            raise ValueError("Step12b service-account wait bound changed")
        if (
            not isinstance(created, _CreatedControllerCapability)
            or created._owner is not self._capability_owner
            or created.identity != self.identity
            or created.receipt.get("status")
            != "run_scoped_controller_service_account_created"
        ):
            raise PermissionError(
                "created controller service-account capability is missing"
            )
        before = self.get()
        if before is None:
            raise RunScopedControllerServiceAccountError(
                "created_controller_service_account_disappeared_before_teardown"
            )
        if before != created.provider:
            raise RunScopedControllerServiceAccountError(
                "created_controller_service_account_unique_id_changed"
            )
        return self._delete_present_and_wait_absent(
            before=before,
            now=now,
            sleep=sleep,
            maximum_wait_seconds=maximum_wait_seconds,
            success_path=True,
            create_receipt_sha256=created.receipt["receipt_sha256"],
        )

    def cleanup_delete_if_present(
        self,
        *,
        now: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        maximum_wait_seconds: int = MAX_READBACK_WAIT_SECONDS,
    ) -> dict[str, Any]:
        """Idempotent failure-path cleanup after an attempted create."""

        if maximum_wait_seconds != MAX_READBACK_WAIT_SECONDS:
            raise ValueError("Step12b service-account wait bound changed")
        before = self.get()
        if before is None:
            return _seal(
                {
                    "schema": SCHEMA,
                    "status": "run_scoped_controller_service_account_absent",
                    "project": self.identity.project,
                    "account_id": self.identity.account_id,
                    "email": self.identity.email,
                    "provider_unique_id": None,
                    "controller_create_receipt_sha256": None,
                    "delete_call_count": 0,
                    "delete_retry_count": 0,
                    "final_get_status": 404,
                    "readback_verified": True,
                    "success_path_teardown_evidence": False,
                    "cloud_mutation_performed": False,
                }
            )
        return self._delete_present_and_wait_absent(
            before=before,
            now=now,
            sleep=sleep,
            maximum_wait_seconds=maximum_wait_seconds,
            success_path=False,
            create_receipt_sha256=None,
        )

    def _delete_present_and_wait_absent(
        self,
        *,
        before: Mapping[str, Any],
        now: Callable[[], float],
        sleep: Callable[[float], None],
        maximum_wait_seconds: int,
        success_path: bool,
        create_receipt_sha256: str | None,
    ) -> dict[str, Any]:
        response = self._request(method="DELETE", url=self._resource_url)
        if response.status_code not in {200, 204}:
            raise RunScopedControllerServiceAccountError(
                "controller_service_account_delete_failed",
                status_code=response.status_code,
            )
        started = float(now())
        get_count = 0
        while True:
            get_count += 1
            if self.get() is None:
                break
            elapsed = float(now()) - started
            if elapsed >= maximum_wait_seconds:
                raise RunScopedControllerServiceAccountError(
                    "controller_service_account_delete_readback_timeout"
                )
            sleep(
                min(
                    READBACK_POLL_SECONDS,
                    maximum_wait_seconds - elapsed,
                )
            )
        return _seal(
            {
                "schema": SCHEMA,
                "status": (
                    "run_scoped_controller_service_account_deleted_and_absent"
                ),
                "project": self.identity.project,
                "account_id": self.identity.account_id,
                "email": self.identity.email,
                "provider_unique_id": before["unique_id"],
                "controller_create_receipt_sha256": (
                    create_receipt_sha256
                ),
                "delete_call_count": 1,
                "delete_retry_count": 0,
                "absence_get_count": get_count,
                "final_get_status": 404,
                "readback_verified": True,
                "success_path_teardown_evidence": success_path,
                "cloud_mutation_performed": True,
            }
        )


__all__ = [
    "ControllerServiceAccountIdentity",
    "LEGACY_CONTROLLER_SERVICE_ACCOUNT",
    "MAX_READBACK_WAIT_SECONDS",
    "RunScopedControllerServiceAccountAdmin",
    "RunScopedControllerServiceAccountError",
    "SCHEMA",
    "canonical_bytes",
    "canonical_sha256",
    "identity_from_deployment",
    "validate_controller_identity",
]
