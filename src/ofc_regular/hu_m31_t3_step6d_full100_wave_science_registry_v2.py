"""Strict science-plan dispatch for the full100 wave-v2 transport.

The wave transport must not infer scientific semantics from directory names,
job contents, or a validator that happens to accept a similar shape.  This
module maps each exact frozen plan schema to one immutable implementation
contract.  Implementations are imported lazily so the already-frozen
development plan remains usable while a later science kind is being built.

No registry entry can be added at runtime.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


DEVELOPMENT_SCIENCE_KIND = "development"
PERFORMANCE_LOCK_V4_SCIENCE_KIND = "performance_lock_v4"

DEVELOPMENT_PLAN_SCHEMA = "hu_m31_t3_step6d_candidate02_full100_plan_v1"
PERFORMANCE_LOCK_V4_PLAN_SCHEMA = (
    "hu_m31_t3_step6d_candidate02_performance_lock_plan_v4"
)

DEVELOPMENT_PLAN_SCOPE = "full_performance_development"
PERFORMANCE_LOCK_V4_PLAN_SCOPE = "one_shot_performance_lock_v4"

DEVELOPMENT_EXECUTION_SCOPE = "repeatable_performance_development_transport_only"
PERFORMANCE_LOCK_V4_EXECUTION_SCOPE = "one_shot_performance_lock_v4"
STARTUP_CANARY_EXECUTION_SCOPE = (
    "startup_canary_candidate_shard_00_a00_transport_diagnostic_only"
)

DEVELOPMENT_WAVE_STATUS = "local_8_8_4_contract_ready_cloud_not_authorized"
DEVELOPMENT_WAVE_DECISION = (
    "reuse_frozen_full100_science_with_fresh_quota_bounded_transport_only"
)
PERFORMANCE_LOCK_V4_WAVE_STATUS = "frozen_preregistered_plan_roots_unopened"
PERFORMANCE_LOCK_V4_WAVE_DECISION = (
    "run009_performance_go_freezes_fresh_one_shot_performance_lock_v4_only"
)

DEVELOPMENT_PACKAGE_SCHEMA = "hu_m31_t3_step6d_full100_spot_package_v1"
DEVELOPMENT_PACKAGE_SOURCE_NAME = (
    "ofc_regular_hu_m31_t3_step6d_full100_v1_source.zip"
)
DEVELOPMENT_STARTUP_RELATIVE_PATH = (
    "scripts/startup_hu_m31_t3_step6d_full100_wave_v2.sh"
)
DEVELOPMENT_STARTUP_SHA256 = (
    "204a12c40b56dda643b7228e1687818a618bf1cb4a1f50359187b113c82a7d87"
)
PERFORMANCE_LOCK_V4_PACKAGE_SCHEMA = (
    "hu_m31_t3_step6d_performance_lock_v4_spot_package_v1"
)
PERFORMANCE_LOCK_V4_PACKAGE_SOURCE_NAME = (
    "ofc_regular_hu_m31_t3_step6d_performance_lock_v4_source.zip"
)
PERFORMANCE_LOCK_V4_STARTUP_RELATIVE_PATH = (
    "scripts/startup_hu_m31_t3_step6d_performance_lock_v4_wave_v2.sh"
)
PERFORMANCE_LOCK_V4_STARTUP_SHA256 = (
    "80c489030a67536f3584d062e16b265486659be2fd1a38b2bfe03b4a4ec0d68b"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_FACADE_KEYS = frozenset(
    {
        "schema",
        "run_name",
        "source_name",
        "source_sha256",
        "source_bytes",
        "plan_sha256",
        "run_contract_digest",
        "job_manifests",
    }
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"


@dataclass(frozen=True)
class ScienceDescriptor:
    """An immutable exact-schema binding to one scientific plan module."""

    science_kind: str
    plan_schema: str
    plan_scope: str
    execution_scope: str
    wave_status: str
    wave_decision: str
    module_name: str
    validator_name: str
    plan_sha256_name: str
    run_contract_digest_name: str
    default_plan_path: Path | None
    default_plan_path_name: str | None
    startup_canary_allowed: bool
    legacy_development_identity: bool
    package_module_name: str
    package_schema: str
    package_source_name: str
    package_manifest_name: str
    package_ready_name: str
    package_validator_name: str
    package_canonical_bytes_name: str
    startup_relative_path: str
    startup_sha256: str
    merge_module_name: str
    merge_function_name: str
    merge_schema_name: str
    merge_scope_name: str
    merge_variant_name: str
    merge_lineage_path_names: tuple[str, ...]
    merge_validator_name: str | None
    merge_generic_key: str | None

    def plan_module(self) -> Any:
        """The scientific plan module bound to this lineage."""

        return self._module()

    def merge_module(self) -> Any:
        """The scientific merger bound to this lineage.

        The scientific bridge must dispatch through this rather than importing
        one merger directly; the two lineages validate their plans with
        different schemas and a hardcoded merger fails closed on the other.
        """

        qualified = f"{__package__}.{self.merge_module_name}"
        try:
            return importlib.import_module(qualified)
        except ImportError as exc:
            raise ValueError(
                f"scientific merger is unavailable for {self.science_kind}"
            ) from exc

    def _module(self) -> Any:
        qualified = f"{__package__}.{self.module_name}"
        try:
            module = importlib.import_module(qualified)
        except ImportError as exc:
            raise ValueError(
                f"science implementation is unavailable for {self.science_kind}"
            ) from exc
        if (
            getattr(module, "PLAN_SCHEMA", None) != self.plan_schema
            or getattr(module, "PLAN_SCOPE", None) != self.plan_scope
        ):
            raise ValueError(
                f"science implementation identity changed for {self.science_kind}"
            )
        return module

    def _digest(self, attribute: str, label: str) -> str:
        value = getattr(self._module(), attribute, None)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ValueError(f"{label} changed for {self.science_kind}")
        return value

    @property
    def plan_sha256(self) -> str:
        return self._digest(self.plan_sha256_name, "science plan SHA-256")

    @property
    def run_contract_digest(self) -> str:
        return self._digest(
            self.run_contract_digest_name, "science run-contract digest"
        )

    def resolved_default_plan_path(self) -> Path:
        if self.default_plan_path is not None:
            return self.default_plan_path
        if self.default_plan_path_name is None:
            raise ValueError(
                f"default science plan path is unavailable for {self.science_kind}"
            )
        value = getattr(self._module(), self.default_plan_path_name, None)
        if not isinstance(value, (str, Path)):
            raise ValueError(
                f"default science plan path changed for {self.science_kind}"
            )
        return Path(value)

    def validate_plan(self, value: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise ValueError("scientific plan must be an object")
        module = self._module()
        validator = getattr(module, self.validator_name, None)
        if not callable(validator):
            raise ValueError(
                f"science validator changed for {self.science_kind}"
            )
        validated = validator(value)
        if not isinstance(validated, Mapping):
            raise ValueError(
                f"science validator returned a non-object for {self.science_kind}"
            )
        payload = deepcopy(dict(validated))
        if (
            payload.get("schema") != self.plan_schema
            or payload.get("scope") != self.plan_scope
            or hashlib.sha256(_canonical_bytes(payload)).hexdigest()
            != self.plan_sha256
        ):
            raise ValueError(
                f"validated science plan identity changed for {self.science_kind}"
            )
        return payload

    def _package_module(self) -> Any:
        qualified = f"{__package__}.{self.package_module_name}"
        try:
            module = importlib.import_module(qualified)
        except ImportError as exc:
            raise ValueError(
                f"package facade is unavailable for {self.science_kind}"
            ) from exc
        if (
            getattr(module, "PACKAGE_SCHEMA", None) != self.package_schema
            or getattr(module, "SOURCE_NAME", None) != self.package_source_name
            or getattr(module, "MANIFEST_NAME", None) != self.package_manifest_name
            or getattr(module, "READY_NAME", None) != self.package_ready_name
        ):
            raise ValueError(
                f"package facade identity changed for {self.science_kind}"
            )
        if (
            not callable(getattr(module, self.package_validator_name, None))
            or not callable(
                getattr(module, self.package_canonical_bytes_name, None)
            )
        ):
            raise ValueError(
                f"package facade callable changed for {self.science_kind}"
            )
        return module

    def validate_package_manifest_facade(
        self, value: Mapping[str, Any]
    ) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise ValueError("scientific package manifest must be an object")
        payload = deepcopy(dict(value))
        if (
            not _PACKAGE_FACADE_KEYS.issubset(payload)
            or payload.get("schema") != self.package_schema
            or payload.get("source_name") != self.package_source_name
            or not isinstance(payload.get("run_name"), str)
            or not isinstance(payload.get("job_manifests"), list)
        ):
            raise ValueError(
                f"package manifest facade changed for {self.science_kind}"
            )
        return payload

    def validate_package(self, run_dir: str | Path) -> dict[str, Any]:
        module = self._package_module()
        validator = getattr(module, self.package_validator_name)
        return self.validate_package_manifest_facade(validator(run_dir))

    def package_canonical_bytes(self, value: Any) -> bytes:
        module = self._package_module()
        canonicalizer = getattr(module, self.package_canonical_bytes_name)
        raw = canonicalizer(value)
        if not isinstance(raw, bytes):
            raise ValueError(
                f"package canonicalizer changed for {self.science_kind}"
            )
        return raw

    def resolved_startup_path(self) -> Path:
        relative = Path(self.startup_relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(
                f"startup path changed for {self.science_kind}"
            )
        target = (_REPO_ROOT / relative).resolve()
        if (
            target.is_symlink()
            or not target.is_file()
            or target.relative_to(_REPO_ROOT.resolve()).as_posix()
            != relative.as_posix()
            or _SHA256.fullmatch(self.startup_sha256) is None
            or hashlib.sha256(target.read_bytes()).hexdigest()
            != self.startup_sha256
        ):
            raise ValueError(
                f"startup bytes changed for {self.science_kind}"
            )
        return target

    def allows_execution_scope(self, value: Any) -> bool:
        return value == self.execution_scope or (
            self.startup_canary_allowed
            and value == STARTUP_CANARY_EXECUTION_SCOPE
        )


_DESCRIPTORS = (
    ScienceDescriptor(
        science_kind=DEVELOPMENT_SCIENCE_KIND,
        plan_schema=DEVELOPMENT_PLAN_SCHEMA,
        plan_scope=DEVELOPMENT_PLAN_SCOPE,
        execution_scope=DEVELOPMENT_EXECUTION_SCOPE,
        wave_status=DEVELOPMENT_WAVE_STATUS,
        wave_decision=DEVELOPMENT_WAVE_DECISION,
        module_name="hu_m31_t3_step6d_candidate02_full100_plan",
        validator_name="validate_full100_plan",
        plan_sha256_name="FULL100_PLAN_SHA256",
        run_contract_digest_name="FULL_RUN_CONTRACT_DIGEST",
        default_plan_path=(
            _REPO_ROOT
            / "outputs/hu_joint_policy/m31_t3_step6d/candidate02_development/"
            "full100_plan_v1.json"
        ),
        default_plan_path_name=None,
        startup_canary_allowed=True,
        legacy_development_identity=True,
        package_module_name="hu_m31_t3_step6d_full100_spot_v1",
        package_schema=DEVELOPMENT_PACKAGE_SCHEMA,
        package_source_name=DEVELOPMENT_PACKAGE_SOURCE_NAME,
        package_manifest_name="manifest.json",
        package_ready_name="PACKAGE_READY.json",
        package_validator_name="validate_package",
        package_canonical_bytes_name="canonical_bytes",
        startup_relative_path=DEVELOPMENT_STARTUP_RELATIVE_PATH,
        startup_sha256=DEVELOPMENT_STARTUP_SHA256,
        merge_module_name="merge_hu_m31_t3_step6d_candidate02_full100",
        merge_function_name="merge_candidate02_full100",
        merge_schema_name="MERGE_SCHEMA",
        merge_scope_name="FULL_SCOPE",
        merge_variant_name="CANDIDATE02_VARIANT",
        merge_lineage_path_names=(),
        merge_validator_name=None,
        merge_generic_key=None,
    ),
    ScienceDescriptor(
        science_kind=PERFORMANCE_LOCK_V4_SCIENCE_KIND,
        plan_schema=PERFORMANCE_LOCK_V4_PLAN_SCHEMA,
        plan_scope=PERFORMANCE_LOCK_V4_PLAN_SCOPE,
        execution_scope=PERFORMANCE_LOCK_V4_EXECUTION_SCOPE,
        wave_status=PERFORMANCE_LOCK_V4_WAVE_STATUS,
        wave_decision=PERFORMANCE_LOCK_V4_WAVE_DECISION,
        module_name="hu_m31_t3_step6d_candidate02_performance_lock_v4_plan",
        validator_name="validate_performance_lock_v4_plan",
        plan_sha256_name="PLAN_SHA256",
        run_contract_digest_name="RUN_CONTRACT_DIGEST",
        default_plan_path=None,
        default_plan_path_name="DEFAULT_PLAN_PATH",
        startup_canary_allowed=False,
        legacy_development_identity=False,
        package_module_name=(
            "hu_m31_t3_step6d_performance_lock_v4_spot_package"
        ),
        package_schema=PERFORMANCE_LOCK_V4_PACKAGE_SCHEMA,
        package_source_name=PERFORMANCE_LOCK_V4_PACKAGE_SOURCE_NAME,
        package_manifest_name="manifest.json",
        package_ready_name="PACKAGE_READY.json",
        package_validator_name="validate_package",
        package_canonical_bytes_name="canonical_bytes",
        startup_relative_path=PERFORMANCE_LOCK_V4_STARTUP_RELATIVE_PATH,
        startup_sha256=PERFORMANCE_LOCK_V4_STARTUP_SHA256,
        merge_module_name=(
            "merge_hu_m31_t3_step6d_candidate02_performance_lock_v4"
        ),
        merge_function_name="merge_candidate02_performance_lock_v4",
        merge_schema_name="MERGE_SCHEMA",
        merge_scope_name="SCOPE",
        merge_variant_name="CANDIDATE02_PERFORMANCE_LOCK_V4_VARIANT",
        # The v4 merger replays frozen lineage the wave plan does not carry.
        merge_lineage_path_names=(
            "DEFAULT_MATERIALIZATION_RECEIPT_PATH",
            "DEFAULT_ROOT_SEAL_PATH",
        ),
        # The v4 summary wraps a development-shaped summary; downstream
        # gate checks read the inner one, the wrapper validates itself.
        merge_validator_name="validate_candidate02_performance_lock_v4_value",
        merge_generic_key="generic_merge",
    ),
)

_BY_SCHEMA = MappingProxyType({row.plan_schema: row for row in _DESCRIPTORS})
_BY_KIND = MappingProxyType({row.science_kind: row for row in _DESCRIPTORS})
_STARTUP_RELATIVE_PATHS = MappingProxyType(
    {row.startup_sha256: row.startup_relative_path for row in _DESCRIPTORS}
)
if (
    len(_BY_SCHEMA) != len(_DESCRIPTORS)
    or len(_BY_KIND) != len(_DESCRIPTORS)
    or len(_STARTUP_RELATIVE_PATHS) != len(_DESCRIPTORS)
):
    raise RuntimeError("science descriptor registry contains duplicate identities")


def startup_relative_paths_by_sha256() -> Mapping[str, str]:
    """Every registered startup script path, keyed by its pinned SHA-256.

    Each startup script verifies the object name it was staged under, so
    callers that stage one must resolve the name from this registry rather
    than keeping their own copy of the lineage table.
    """

    return _STARTUP_RELATIVE_PATHS


def descriptor_for_schema(value: Any) -> ScienceDescriptor:
    if not isinstance(value, str):
        raise ValueError("scientific plan schema is missing")
    descriptor = _BY_SCHEMA.get(value)
    if descriptor is None:
        raise ValueError(f"unsupported scientific plan schema: {value}")
    return descriptor


def descriptor_for_kind(value: Any) -> ScienceDescriptor:
    if not isinstance(value, str):
        raise ValueError("science kind is missing")
    descriptor = _BY_KIND.get(value)
    if descriptor is None:
        raise ValueError(f"unsupported science kind: {value}")
    return descriptor


def descriptor_for_plan(value: Any) -> ScienceDescriptor:
    if not isinstance(value, Mapping):
        raise ValueError("scientific plan must be an object")
    return descriptor_for_schema(value.get("schema"))


def descriptor_for_wave_plan(value: Any) -> ScienceDescriptor:
    if not isinstance(value, Mapping):
        raise ValueError("wave plan must be an object")
    scientific = value.get("full100_plan")
    if not isinstance(scientific, Mapping):
        raise ValueError("wave plan scientific plan is missing")
    return descriptor_for_plan(scientific)


def resolve_startup_sha256(
    wave_plan: Mapping[str, Any],
    supplied: str | None = None,
) -> str:
    descriptor = descriptor_for_wave_plan(wave_plan)
    expected = descriptor.startup_sha256
    if _SHA256.fullmatch(expected) is None:
        raise ValueError(
            f"startup SHA-256 changed for {descriptor.science_kind}"
        )
    if supplied is not None and supplied != expected:
        raise ValueError("startup SHA-256 does not match scientific plan")
    return expected


def validate_scientific_plan(
    value: Mapping[str, Any],
) -> tuple[ScienceDescriptor, dict[str, Any]]:
    descriptor = descriptor_for_plan(value)
    return descriptor, descriptor.validate_plan(value)


__all__ = [
    "DEVELOPMENT_EXECUTION_SCOPE",
    "DEVELOPMENT_PACKAGE_SCHEMA",
    "DEVELOPMENT_PACKAGE_SOURCE_NAME",
    "DEVELOPMENT_STARTUP_RELATIVE_PATH",
    "DEVELOPMENT_STARTUP_SHA256",
    "DEVELOPMENT_PLAN_SCOPE",
    "DEVELOPMENT_PLAN_SCHEMA",
    "DEVELOPMENT_SCIENCE_KIND",
    "DEVELOPMENT_WAVE_DECISION",
    "DEVELOPMENT_WAVE_STATUS",
    "PERFORMANCE_LOCK_V4_EXECUTION_SCOPE",
    "PERFORMANCE_LOCK_V4_PACKAGE_SCHEMA",
    "PERFORMANCE_LOCK_V4_PACKAGE_SOURCE_NAME",
    "PERFORMANCE_LOCK_V4_STARTUP_RELATIVE_PATH",
    "PERFORMANCE_LOCK_V4_STARTUP_SHA256",
    "PERFORMANCE_LOCK_V4_PLAN_SCOPE",
    "PERFORMANCE_LOCK_V4_PLAN_SCHEMA",
    "PERFORMANCE_LOCK_V4_SCIENCE_KIND",
    "PERFORMANCE_LOCK_V4_WAVE_DECISION",
    "PERFORMANCE_LOCK_V4_WAVE_STATUS",
    "STARTUP_CANARY_EXECUTION_SCOPE",
    "ScienceDescriptor",
    "descriptor_for_kind",
    "descriptor_for_plan",
    "descriptor_for_schema",
    "descriptor_for_wave_plan",
    "resolve_startup_sha256",
    "validate_scientific_plan",
]
