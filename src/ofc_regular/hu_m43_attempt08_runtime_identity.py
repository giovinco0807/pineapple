"""Stable external-runtime identity used by Attempt08 preflight and development."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from typing import Any


ATTEMPT08_GCP_IMAGE_NAME = "debian-12-bookworm-v20260609"
ATTEMPT08_GCP_IMAGE_ID = "1449487925682397051"
ATTEMPT08_GCP_IMAGE_SELF_LINK = (
    "projects/debian-cloud/global/images/debian-12-bookworm-v20260609"
)
ATTEMPT08_RUNTIME_FINGERPRINT_SCHEMA = "hu_m43_attempt08_runtime_fingerprint_v1"
_PACKAGES = (
    "filelock",
    "fsspec",
    "jinja2",
    "iniconfig",
    "joblib",
    "lightgbm",
    "markupsafe",
    "mpmath",
    "networkx",
    "numpy",
    "packaging",
    "pip",
    "pluggy",
    "pygments",
    "pytest",
    "scikit-learn",
    "scipy",
    "sympy",
    "setuptools",
    "threadpoolctl",
    "torch",
    "typing-extensions",
    "wheel",
)
ATTEMPT08_RUNTIME_REQUIREMENTS_SHA256 = (
    "a0ce16d1ab481528ac53f2e94fd9037af7f94004a38b8cf8116537bbf4277c68"
)
_EXPECTED_PYTHON = {
    "implementation": "CPython",
    "version": "3.11.2",
    "machine": "x86_64",
    "system": "Linux",
}
_EXPECTED_PACKAGES = {
    "filelock": "3.18.0",
    "fsspec": "2025.5.1",
    "jinja2": "3.1.6",
    "iniconfig": "2.1.0",
    "joblib": "1.4.2",
    "numpy": "2.2.6",
    "packaging": "25.0",
    "pip": "26.1.2",
    "pluggy": "1.6.0",
    "pygments": "2.19.2",
    "pytest": "8.4.1",
    "scipy": "1.15.3",
    "scikit-learn": "1.8.0",
    "lightgbm": "4.6.0",
    "markupsafe": "3.0.2",
    "mpmath": "1.3.0",
    "networkx": "3.4.2",
    "sympy": "1.13.1",
    "setuptools": "83.0.0",
    "threadpoolctl": "3.6.0",
    "torch": "2.6.0+cpu",
    "typing-extensions": "4.13.2",
    "wheel": "0.46.1",
}


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def runtime_fingerprint_payload(*, requirements_sha256: str) -> dict[str, Any]:
    if (
        not isinstance(requirements_sha256, str)
        or len(requirements_sha256) != 64
        or any(character not in "0123456789abcdef" for character in requirements_sha256)
    ):
        raise ValueError("Attempt08 runtime requirements SHA-256 is invalid")
    return {
        "schema": ATTEMPT08_RUNTIME_FINGERPRINT_SCHEMA,
        "gcp_image": {
            "name": ATTEMPT08_GCP_IMAGE_NAME,
            "id": ATTEMPT08_GCP_IMAGE_ID,
            "self_link": ATTEMPT08_GCP_IMAGE_SELF_LINK,
        },
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "machine": platform.machine().lower(),
            "system": platform.system(),
        },
        "packages": {
            package: importlib.metadata.version(package) for package in _PACKAGES
        },
        "requirements_sha256": requirements_sha256,
    }


def runtime_fingerprint_sha256(*, requirements_sha256: str) -> str:
    return hashlib.sha256(
        canonical_json_bytes(
            runtime_fingerprint_payload(requirements_sha256=requirements_sha256)
        )
    ).hexdigest()


def expected_runtime_fingerprint_payload() -> dict[str, Any]:
    return {
        "schema": ATTEMPT08_RUNTIME_FINGERPRINT_SCHEMA,
        "gcp_image": {
            "name": ATTEMPT08_GCP_IMAGE_NAME,
            "id": ATTEMPT08_GCP_IMAGE_ID,
            "self_link": ATTEMPT08_GCP_IMAGE_SELF_LINK,
        },
        "python": dict(_EXPECTED_PYTHON),
        "packages": dict(_EXPECTED_PACKAGES),
        "requirements_sha256": ATTEMPT08_RUNTIME_REQUIREMENTS_SHA256,
    }


ATTEMPT08_EXPECTED_RUNTIME_FINGERPRINT_SHA256 = (
    "8c2cd111bc4e70096ff4f974f684ad146e94329871328e5b5db5d3426256c218"
)


def validate_expected_runtime_fingerprint() -> str:
    try:
        actual = runtime_fingerprint_payload(
            requirements_sha256=ATTEMPT08_RUNTIME_REQUIREMENTS_SHA256
        )
    except importlib.metadata.PackageNotFoundError as error:
        raise ValueError("Attempt08 external runtime fingerprint changed") from error
    expected = expected_runtime_fingerprint_payload()
    if actual != expected:
        raise ValueError("Attempt08 external runtime fingerprint changed")
    digest = hashlib.sha256(canonical_json_bytes(actual)).hexdigest()
    if digest != ATTEMPT08_EXPECTED_RUNTIME_FINGERPRINT_SHA256:
        raise ValueError("Attempt08 expected runtime fingerprint digest changed")
    return digest


__all__ = [
    "ATTEMPT08_GCP_IMAGE_ID",
    "ATTEMPT08_GCP_IMAGE_NAME",
    "ATTEMPT08_GCP_IMAGE_SELF_LINK",
    "ATTEMPT08_RUNTIME_FINGERPRINT_SCHEMA",
    "ATTEMPT08_EXPECTED_RUNTIME_FINGERPRINT_SHA256",
    "ATTEMPT08_RUNTIME_REQUIREMENTS_SHA256",
    "expected_runtime_fingerprint_payload",
    "runtime_fingerprint_payload",
    "runtime_fingerprint_sha256",
    "validate_expected_runtime_fingerprint",
]
