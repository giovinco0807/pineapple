"""Schema-dispatch loader for opt-in M4.3 artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import pickle
from pathlib import Path
from typing import Any, Mapping

from .hu_m43_attempt02_lifecycle import (
    M43_ATTEMPT02_FREEZE_SCHEMA,
    file_sha256,
    validate_attempt02_freeze,
)
from .hu_m43_joint_model_v4 import (
    HU_M43_V4_ARTIFACT_SCHEMA,
    HU_M43_V4_TRAINING_MANIFEST_SCHEMA,
    HuM43JointModelV4,
)
from .hu_m43_joint_model_v5 import HU_M43_V5_ARTIFACT_SCHEMA
from .hu_m43_joint_model_v6 import HU_M43_V6_ARTIFACT_SCHEMA
from .hu_m43_attempt08_distilled_model import (
    HU_M43_ATTEMPT08_DISTILLED_ARTIFACT_SCHEMA,
    HuM43Attempt08DistilledModel,
)
from .hu_m43_attempt10_distilled_model import (
    HU_M43_ATTEMPT10_DISTILLED_ARTIFACT_SCHEMA,
)
from .hu_m43_attempt11_distilled_model import (
    HU_M43_ATTEMPT11_DISTILLED_ARTIFACT_SCHEMA,
)
from .hu_m43_attempt12_distilled_model import (
    HU_M43_ATTEMPT12_DISTILLED_ARTIFACT_SCHEMA,
)
from .hu_m43_attempt13_distilled_model import (
    HU_M43_ATTEMPT13_DISTILLED_ARTIFACT_SCHEMA,
)
from .hu_m4_joint_model import (
    HU_M4_JOINT_ARTIFACT_SCHEMA,
    load_hu_m4_joint_action_model,
)


def load_hu_m43_joint_action_model(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    freeze_manifest: str | Path | Mapping[str, Any] | None = None,
    training_manifest_path: str | Path | None = None,
    threshold_lock_path: str | Path | None = None,
    runtime_source_manifest_path: str | Path | None = None,
    runtime_source_root: str | Path | None = None,
    runtime_dependency_root: str | Path | None = None,
) -> Any:
    """Dispatch on the versioned pickle envelope without changing v3 loader.

    A v4 runtime binding is deliberately all-or-nothing: expected artifact SHA,
    the Attempt02 freeze, and the exact training-manifest file are required
    together.  A raw v4 artifact may still be loaded explicitly for offline
    tests by omitting every binding argument.
    """

    source = Path(path)
    encoded = source.read_bytes()
    actual_sha = hashlib.sha256(encoded).hexdigest()
    if expected_sha256 is not None and actual_sha != _sha256_text(expected_sha256):
        raise ValueError("M4.3 joint artifact SHA-256 mismatch")
    payload = pickle.loads(encoded)
    if not isinstance(payload, dict):
        raise TypeError("M4.3 joint artifact envelope must be a mapping")
    artifact_schema = payload.get("artifact_schema")

    if artifact_schema == HU_M43_ATTEMPT13_DISTILLED_ARTIFACT_SCHEMA:
        if threshold_lock_path is not None:
            raise ValueError("Attempt13 distilled runtime has no threshold-lock input")
        binding_values = (
            expected_sha256,
            freeze_manifest,
            training_manifest_path,
            runtime_source_manifest_path,
            runtime_source_root,
            runtime_dependency_root,
        )
        binding_count = sum(value is not None for value in binding_values)
        if binding_count == 0:
            raise ValueError(
                "Attempt13 raw distilled artifact is not runtime-loadable; "
                "a complete verified runtime binding is required"
            )
        if binding_count != 6:
            raise ValueError(
                "Attempt13 distilled runtime requires expected SHA, runtime "
                "freeze, training manifest, frozen source manifest, verified "
                "extracted source root, and frozen runtime dependency root "
                "together"
            )
        from .validate_hu_m43_attempt13_acceptance import (
            load_bound_attempt13_distilled_model,
        )

        assert expected_sha256 is not None
        assert freeze_manifest is not None
        assert training_manifest_path is not None
        assert runtime_source_manifest_path is not None
        assert runtime_source_root is not None
        assert runtime_dependency_root is not None
        return load_bound_attempt13_distilled_model(
            source,
            expected_sha256=expected_sha256,
            runtime_freeze=freeze_manifest,
            training_manifest_path=training_manifest_path,
            runtime_source_manifest_path=runtime_source_manifest_path,
            runtime_source_root=runtime_source_root,
            runtime_dependency_root=runtime_dependency_root,
        )

    if artifact_schema == HU_M43_ATTEMPT12_DISTILLED_ARTIFACT_SCHEMA:
        if threshold_lock_path is not None:
            raise ValueError("Attempt12 distilled runtime has no threshold-lock input")
        binding_values = (
            expected_sha256,
            freeze_manifest,
            training_manifest_path,
            runtime_source_manifest_path,
            runtime_source_root,
            runtime_dependency_root,
        )
        binding_count = sum(value is not None for value in binding_values)
        if binding_count == 0:
            raise ValueError(
                "Attempt12 raw distilled artifact is not runtime-loadable; "
                "a complete verified runtime binding is required"
            )
        if binding_count != 6:
            raise ValueError(
                "Attempt12 distilled runtime requires expected SHA, runtime "
                "freeze, training manifest, frozen source manifest, verified "
                "extracted source root, and frozen runtime dependency root "
                "together"
            )
        from .validate_hu_m43_attempt12_acceptance import (
            load_bound_attempt12_distilled_model,
        )

        assert expected_sha256 is not None
        assert freeze_manifest is not None
        assert training_manifest_path is not None
        assert runtime_source_manifest_path is not None
        assert runtime_source_root is not None
        assert runtime_dependency_root is not None
        return load_bound_attempt12_distilled_model(
            source,
            expected_sha256=expected_sha256,
            runtime_freeze=freeze_manifest,
            training_manifest_path=training_manifest_path,
            runtime_source_manifest_path=runtime_source_manifest_path,
            runtime_source_root=runtime_source_root,
            runtime_dependency_root=runtime_dependency_root,
        )

    if artifact_schema == HU_M43_ATTEMPT11_DISTILLED_ARTIFACT_SCHEMA:
        if threshold_lock_path is not None:
            raise ValueError("Attempt11 distilled runtime has no threshold-lock input")
        binding_values = (
            expected_sha256,
            freeze_manifest,
            training_manifest_path,
            runtime_source_manifest_path,
            runtime_source_root,
            runtime_dependency_root,
        )
        binding_count = sum(value is not None for value in binding_values)
        if binding_count == 0:
            raise ValueError(
                "Attempt11 raw distilled artifact is not runtime-loadable; "
                "a complete verified runtime binding is required"
            )
        if binding_count != 6:
            raise ValueError(
                "Attempt11 distilled runtime requires expected SHA, runtime "
                "freeze, training manifest, frozen source manifest, verified "
                "extracted source root, and frozen runtime dependency root "
                "together"
            )
        from .validate_hu_m43_attempt11_acceptance import (
            load_bound_attempt11_distilled_model,
        )

        assert expected_sha256 is not None
        assert freeze_manifest is not None
        assert training_manifest_path is not None
        assert runtime_source_manifest_path is not None
        assert runtime_source_root is not None
        assert runtime_dependency_root is not None
        return load_bound_attempt11_distilled_model(
            source,
            expected_sha256=expected_sha256,
            runtime_freeze=freeze_manifest,
            training_manifest_path=training_manifest_path,
            runtime_source_manifest_path=runtime_source_manifest_path,
            runtime_source_root=runtime_source_root,
            runtime_dependency_root=runtime_dependency_root,
        )

    if artifact_schema == HU_M43_ATTEMPT10_DISTILLED_ARTIFACT_SCHEMA:
        if threshold_lock_path is not None:
            raise ValueError("Attempt10 distilled runtime has no threshold-lock input")
        binding_values = (
            expected_sha256,
            freeze_manifest,
            training_manifest_path,
            runtime_source_manifest_path,
            runtime_source_root,
            runtime_dependency_root,
        )
        binding_count = sum(value is not None for value in binding_values)
        if binding_count == 0:
            raise ValueError(
                "Attempt10 raw distilled artifact is not runtime-loadable; "
                "a complete verified runtime binding is required"
            )
        if binding_count != 6:
            raise ValueError(
                "Attempt10 distilled runtime requires expected SHA, runtime "
                "freeze, training manifest, frozen source manifest, verified "
                "extracted source root, and frozen runtime dependency root "
                "together"
            )
        from .validate_hu_m43_attempt10_acceptance import (
            load_bound_attempt10_distilled_model,
        )

        assert expected_sha256 is not None
        assert freeze_manifest is not None
        assert training_manifest_path is not None
        assert runtime_source_manifest_path is not None
        assert runtime_source_root is not None
        assert runtime_dependency_root is not None
        return load_bound_attempt10_distilled_model(
            source,
            expected_sha256=expected_sha256,
            runtime_freeze=freeze_manifest,
            training_manifest_path=training_manifest_path,
            runtime_source_manifest_path=runtime_source_manifest_path,
            runtime_source_root=runtime_source_root,
            runtime_dependency_root=runtime_dependency_root,
        )

    if artifact_schema == HU_M43_ATTEMPT08_DISTILLED_ARTIFACT_SCHEMA:
        if threshold_lock_path is not None:
            raise ValueError("Attempt08 distilled runtime has no threshold-lock input")
        binding_values = (
            expected_sha256,
            freeze_manifest,
            training_manifest_path,
            runtime_source_manifest_path,
            runtime_source_root,
            runtime_dependency_root,
        )
        binding_count = sum(value is not None for value in binding_values)
        if binding_count not in {0, 6}:
            raise ValueError(
                "Attempt08 distilled runtime requires expected SHA, runtime "
                "freeze, training manifest, frozen source manifest, and "
                "verified extracted source root, and frozen runtime dependency "
                "root together"
            )
        if binding_count == 0:
            return HuM43Attempt08DistilledModel.load(
                source, expected_sha256=actual_sha
            )
        from .validate_hu_m43_attempt08_acceptance import (
            load_bound_attempt08_distilled_model,
        )

        assert expected_sha256 is not None
        assert freeze_manifest is not None
        assert training_manifest_path is not None
        assert runtime_source_manifest_path is not None
        assert runtime_source_root is not None
        assert runtime_dependency_root is not None
        return load_bound_attempt08_distilled_model(
            source,
            expected_sha256=expected_sha256,
            runtime_freeze=freeze_manifest,
            training_manifest_path=training_manifest_path,
            runtime_source_manifest_path=runtime_source_manifest_path,
            runtime_source_root=runtime_source_root,
            runtime_dependency_root=runtime_dependency_root,
        )

    if (
        runtime_source_manifest_path is not None
        or runtime_source_root is not None
        or runtime_dependency_root is not None
    ):
        raise ValueError(
            "distilled runtime source bindings require a supported bound "
            "distilled artifact"
        )

    if artifact_schema == HU_M43_V6_ARTIFACT_SCHEMA:
        binding_values = (
            expected_sha256,
            freeze_manifest,
            training_manifest_path,
            threshold_lock_path,
        )
        binding_count = sum(value is not None for value in binding_values)
        if binding_count not in {0, 4}:
            raise ValueError(
                "Attempt04 v6 runtime requires expected SHA, runtime freeze, "
                "final training manifest, and threshold lock together"
            )
        if binding_count == 0:
            from .hu_m43_joint_model_v6 import HuM43JointModelV6

            return HuM43JointModelV6.load(source, expected_sha256=actual_sha)

        try:
            from .hu_m43_attempt04_runtime import load_bound_attempt04_v6_model
        except ImportError as error:
            raise RuntimeError(
                "Attempt04 v6 bound runtime is not installed; raw offline "
                "loading remains available only without binding arguments"
            ) from error

        assert expected_sha256 is not None
        assert freeze_manifest is not None
        assert training_manifest_path is not None
        assert threshold_lock_path is not None
        return load_bound_attempt04_v6_model(
            source,
            expected_sha256=expected_sha256,
            runtime_freeze=freeze_manifest,
            final_training_manifest_path=training_manifest_path,
            threshold_lock_path=threshold_lock_path,
        )

    if artifact_schema == HU_M43_V5_ARTIFACT_SCHEMA:
        if threshold_lock_path is not None:
            raise ValueError("Attempt03 v5 does not accept a separate threshold lock")
        binding_values = (
            expected_sha256,
            freeze_manifest,
            training_manifest_path,
        )
        binding_count = sum(value is not None for value in binding_values)
        if binding_count not in {0, 3}:
            raise ValueError(
                "Attempt03 v5 runtime requires expected SHA, runtime freeze, "
                "and final training manifest together"
            )
        if binding_count == 0:
            # Raw v5 loading remains available for offline diagnostics.  The
            # population evaluator and opt-in runtime use the all-three branch.
            from .hu_m43_joint_model_v5 import HuM43JointModelV5

            return HuM43JointModelV5.load(source, expected_sha256=actual_sha)

        from .hu_m43_attempt03_runtime import load_bound_attempt03_v5_model

        assert expected_sha256 is not None
        assert freeze_manifest is not None
        assert training_manifest_path is not None
        return load_bound_attempt03_v5_model(
            source,
            expected_sha256=expected_sha256,
            runtime_freeze=freeze_manifest,
            final_training_manifest_path=training_manifest_path,
        )

    if artifact_schema == HU_M43_V4_ARTIFACT_SCHEMA:
        if threshold_lock_path is not None:
            raise ValueError("Attempt02 v4 does not accept a separate threshold lock")
        binding_values = (
            expected_sha256,
            freeze_manifest,
            training_manifest_path,
        )
        binding_count = sum(value is not None for value in binding_values)
        if binding_count not in {0, 3}:
            raise ValueError(
                "Attempt02 v4 runtime requires expected SHA, freeze, and training manifest together"
            )
        if binding_count == 3:
            freeze = _load_mapping(freeze_manifest, "Attempt02 freeze")
            if freeze.get("schema") != M43_ATTEMPT02_FREEZE_SCHEMA:
                raise ValueError("v4 runtime requires an Attempt02 freeze")
            validate_attempt02_freeze(freeze)
            if freeze.get("model_sha256") != actual_sha:
                raise ValueError("v4 artifact SHA disagrees with Attempt02 freeze")
            assert training_manifest_path is not None
            if file_sha256(training_manifest_path) != freeze.get(
                "training_manifest_sha256"
            ):
                raise ValueError("v4 training manifest SHA disagrees with freeze")
            training = _load_mapping(
                training_manifest_path, "Attempt02 v4 training manifest"
            )
            if training.get("schema") != HU_M43_V4_TRAINING_MANIFEST_SCHEMA:
                raise ValueError("v4 training manifest schema mismatch")
            if training.get("model_sha256") != actual_sha:
                raise ValueError("v4 training manifest model SHA mismatch")
        else:
            freeze = None

        model = HuM43JointModelV4.load(source, expected_sha256=actual_sha)
        if freeze is not None:
            if model.model_id != freeze.get("model_id"):
                raise ValueError("v4 model_id disagrees with Attempt02 freeze")
            if not math.isclose(
                float(model.safety_threshold),
                float(freeze.get("frozen_threshold")),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ValueError("v4 safety threshold disagrees with Attempt02 freeze")
            if model.safety_enabled is not True or freeze.get("safety_enabled") is not True:
                raise ValueError("v4 frozen runtime safety is not enabled")
        return model

    # The existing implementation remains the sole authority for v3 bytes and
    # v3 freeze semantics.  Unknown schemas follow that same fail-closed path.
    if artifact_schema != HU_M4_JOINT_ARTIFACT_SCHEMA:
        if threshold_lock_path is not None:
            raise ValueError("legacy M4 artifacts do not accept a threshold lock")
        return load_hu_m4_joint_action_model(
            source,
            expected_sha256=expected_sha256,
            freeze_manifest=freeze_manifest,
            training_manifest_path=training_manifest_path,
        )
    if threshold_lock_path is not None:
        raise ValueError("legacy M4 artifacts do not accept a threshold lock")
    return load_hu_m4_joint_action_model(
        source,
        expected_sha256=expected_sha256,
        freeze_manifest=freeze_manifest,
        training_manifest_path=training_manifest_path,
    )


def _load_mapping(
    source: str | Path | Mapping[str, Any] | None,
    location: str,
) -> dict[str, Any]:
    if source is None:
        raise ValueError(f"{location} is required")
    if isinstance(source, Mapping):
        return dict(source)
    value = json.loads(Path(source).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be a mapping")
    return value


def _sha256_text(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("expected artifact SHA must be a string")
    normalized = value.lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError("expected artifact SHA must be a SHA-256 string")
    return normalized


__all__ = ["load_hu_m43_joint_action_model"]
