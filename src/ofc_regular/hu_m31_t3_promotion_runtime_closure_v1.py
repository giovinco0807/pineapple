"""Content-addressed runtime closure for the M3.1 locked T3 promotion.

The locked population/ABR evaluation must not silently load models from the
working tree.  This module packages the complete Python ``ofc_regular`` source
tree, every legacy model used by the five frozen population policies and the
``stage7_m5_r10`` baseline, the StreetPolicyNetV1 risk checkpoint bundle and
rich threshold lock, and the accepted exact-T4 native library.

The archive is deterministic, create-only, and self-describing.  Validation
replays every byte and, when a repository root is supplied, requires exact
source/model parity with the code that is about to execute.  It never resolves
``current``, registers a profile, starts cloud work, or authorizes promotion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import zipfile
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from . import ai_profiles
from . import hu_m31_t3_street_policy_training_v1 as training
from .action_key import ACTION_KEY_SCHEMA
from .hu_infoset import OBSERVATION_SCHEMA
from .hu_m3_t4_runtime import HU_M30_T4_ENGINE_VERSION
from .street_policy_net_v1 import FEATURE_SCHEMA_HASH, LOSS_SCHEMA_HASH


CLOSURE_SCHEMA = "hu_m31_t3_promotion_runtime_closure_v1"
CLOSURE_ARCHIVE_FORMAT = "deterministic_zip_stored_v1"
CLOSURE_MANIFEST_PATH = "closure_manifest.json"
CANDIDATE_BUNDLE_PREFIX = "candidate/checkpoint_bundle"
CANDIDATE_THRESHOLD_PATH = "candidate/training_threshold_lock.json"
BASELINE_PROFILE_ID = "stage7_m5_r10"
CANDIDATE_FACTORY_ID = "m31.street_policy_v1.evaluation_only"
NAMED_PROFILE_FACTORY_PREFIX = "ai_profiles.build_policy.exact_t4_v1"
OPENING_LOOKAHEAD_SAMPLES = 0
FROZEN_POLICY_REGISTRY_SHA256 = (
    "d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3"
)
ACCEPTED_EXACT_T4_LIBRARY_SHA256_BY_TARGET = {
    # M3.0 runtime-v3 accepted Windows release.
    "windows": "faf1eec9a86a4c2278b6e2d323d6df2dbf9fa1050911d6807894d9b4e6566261",
    # M3.1 candidate02 accepted Linux release; its compact scorer/exact-T4
    # semantics passed the full parity contract before the performance lock.
    "linux": "4050e04b22d7943da9e1691402a2b34cf3d3781041a44557f35e2dfcf366d55d",
}

POPULATION_PROFILE_IDS = (
    "stage19_p0",
    "stage9f_p2",
    "stage7_m5_r10",
    "stage3_baseline",
    "random_exact_final",
)
ABR_FACTORY_IDS = {
    "greedy_search_response": "m31.abr.greedy_search_response.v1",
    "foul_pressure_response": "m31.abr.foul_pressure_response.v1",
    "royalty_denial_response": "m31.abr.royalty_denial_response.v1",
}

_LEGACY_MODEL_PATHS = {
    "opening": ai_profiles.DEFAULT_OPENING_MODEL,
    "turn1": ai_profiles.DEFAULT_TURN1_MODEL,
    "turn2": ai_profiles.DEFAULT_TURN2_MODEL,
    "turn3": ai_profiles.DEFAULT_TURN3_MODEL,
    "hu_turn1_stage18_p1": ai_profiles.DEFAULT_HU_TURN1_STAGE18_P1_MODEL,
    "hu_turn1_stage18_p1_safe_selector": (
        ai_profiles.DEFAULT_HU_TURN1_STAGE18_P1_SAFE_SELECTOR_MODEL
    ),
    "hu_turn0_stage19_p0": ai_profiles.DEFAULT_HU_TURN0_STAGE19_P0_MODEL,
    "hu_turn0_stage19_p0_safe_selector": (
        ai_profiles.DEFAULT_HU_TURN0_STAGE19_P0_SAFE_SELECTOR_MODEL
    ),
    "hu_turn2_stage8b": ai_profiles.DEFAULT_HU_TURN2_STAGE8B_MODEL,
    "hu_turn3_stage7": ai_profiles.DEFAULT_HU_TURN3_STAGE7_MODEL,
    "hu_turn3_stage7_reference": (
        ai_profiles.DEFAULT_HU_TURN3_STAGE7_REFERENCE_MODEL
    ),
}
FROZEN_LEGACY_MODEL_SHA256_BY_FIELD = {
    "opening": "4cfec60e3035323d10348f4f28938c24428880edc557a5e824723d32838fe939",
    "turn1": "ecffa86f0ccbf0bea1d697ae98bf4b9de316006e5aa4539eda4476a29f3f8b4a",
    "turn2": "4be8e1c3647306a18b74676bf4835fcac9e03f3dde65fba391e9aaf5ac3929fe",
    "turn3": "5996204bf904b258451097042f38bbd87b3f570fbcbe9bf5f4a1d2bdaf376737",
    "hu_turn2_stage8b": "2a6a52ee09e329852d00197e686509b0747ff86ef627a7b6ed9a21311ec25b3b",
    "hu_turn3_stage7": "727fb766b940f17c6c6d00a373b47d68bd07aa1095fe86633ba0fcab6c6e8f20",
    "hu_turn3_stage7_reference": "36f6ac00ca308b92aa4a11a7d6b3c65c4b5f65f6fb5fa2e71585d7f87d8dd112",
    "hu_turn1_stage18_p1": "e910131715efbec7a116e411a8dc23dcfc9a904f2aef9e0eee9d4913a02dfee7",
    "hu_turn1_stage18_p1_safe_selector": "838ff3421c2393648aaabeb7316c9e6c3fe97197485b1363be4e8b5dd33ae086",
    "hu_turn0_stage19_p0": "8f360bc4c326f0d8d20710a4efc7c42c62a80cca11341b34a41a72a4c6aadf71",
    "hu_turn0_stage19_p0_safe_selector": "7cd85b3a824816223ceb0d84870feb62e43d827452031e6f51de0ee452514abe",
}
_REQUIRED_SOURCE_PATHS = frozenset(
    {
        "src/ofc_regular/__init__.py",
        "src/ofc_regular/action_key.py",
        "src/ofc_regular/action_space.py",
        "src/ofc_regular/ai_profiles.py",
        "src/ofc_regular/hu_infoset.py",
        "src/ofc_regular/hu_m3_t4_runtime.py",
        "src/ofc_regular/hu_m31_t3_locked_promotion_runner_v1.py",
        "src/ofc_regular/hu_m31_t3_locked_promotion_production_v1.py",
        "src/ofc_regular/hu_m31_t3_locked_promotion_cli_v1.py",
        "src/ofc_regular/hu_m31_t3_promotion_runtime_closure_v1.py",
        "src/ofc_regular/hu_m31_t3_step6d_locked_promotion_v1.py",
        "src/ofc_regular/hu_m31_t3_street_policy_runtime_v1.py",
        "src/ofc_regular/hu_m31_t3_street_policy_training_v1.py",
        "src/ofc_regular/play_ai.py",
        "src/ofc_regular/policy.py",
        "src/ofc_regular/street_policy_net_v1.py",
    }
)
_SHA_CHARS = frozenset("0123456789abcdef")
_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "status",
        "archive_format",
        "entries",
        "entry_count",
        "entry_aggregate_sha256",
        "source_entries",
        "legacy_model_bindings",
        "candidate_artifacts",
        "exact_t4_native",
        "schema_bindings",
        "factory_contract",
        "information_safety",
        "content_addressed",
        "cloud_execution_started",
        "promotion_authorized",
        "named_profile_added",
        "current_profile_changed",
        "runtime_activated",
        "manifest_identity_sha256",
    }
)
_ENTRY_KEYS = frozenset({"sha256", "bytes"})


class PromotionRuntimeClosureError(RuntimeError):
    """Raised when the immutable evaluation closure cannot be replayed."""


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


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= _SHA_CHARS
    )


def _safe_archive_name(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or value != path.as_posix()
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\\" in value
        or ":" in value
    ):
        raise PromotionRuntimeClosureError(
            f"unsafe runtime-closure archive path: {value!r}"
        )
    return path


def _safe_file(path: str | Path, label: str) -> Path:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise PromotionRuntimeClosureError(
            f"{label} must be a regular non-symlink file"
        )
    return source.resolve()


def _safe_directory(path: str | Path, label: str) -> Path:
    source = Path(path)
    if source.is_symlink() or not source.is_dir():
        raise PromotionRuntimeClosureError(
            f"{label} must be a regular non-symlink directory"
        )
    return source.resolve()


def _read_canonical(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PromotionRuntimeClosureError(
            f"{label} is not canonical JSON"
        ) from exc
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise PromotionRuntimeClosureError(
            f"{label} is not a canonical JSON object"
        )
    return value


def _relative_model_path(value: Path) -> str:
    raw = value.as_posix()
    _safe_archive_name(raw)
    if not raw.startswith("models/"):
        raise PromotionRuntimeClosureError(
            "legacy profile model path escaped models/"
        )
    return raw


def legacy_model_bindings() -> dict[str, str]:
    """Return the exact ``ModelPaths`` fields consumed by the frozen grid."""

    return {
        field: _relative_model_path(path)
        for field, path in sorted(_LEGACY_MODEL_PATHS.items())
    }


def _training_config_from_manifest(
    checkpoint_manifest: Mapping[str, Any],
) -> training.StreetPolicyTrainingConfig:
    raw = checkpoint_manifest.get("training_config")
    if not isinstance(raw, Mapping):
        raise PromotionRuntimeClosureError(
            "candidate checkpoint lacks its training config"
        )
    values = dict(raw)
    if values.pop("schema", None) != training.TRAINING_CONFIG_SCHEMA:
        raise PromotionRuntimeClosureError(
            "candidate checkpoint training config schema changed"
        )
    try:
        config = training.StreetPolicyTrainingConfig(**values)
    except (TypeError, ValueError) as exc:
        raise PromotionRuntimeClosureError(
            "candidate checkpoint training config is invalid"
        ) from exc
    if (
        config.to_dict() != dict(raw)
        or config.identity_sha256
        != checkpoint_manifest.get("training_config_sha256")
    ):
        raise PromotionRuntimeClosureError(
            "candidate checkpoint training config identity changed"
        )
    return config


def _candidate_payloads(
    *,
    checkpoint_bundle: Path,
    threshold_lock_path: Path,
) -> tuple[dict[str, bytes], dict[str, Any]]:
    manifest_path = _safe_file(
        checkpoint_bundle / "manifest.json", "candidate checkpoint manifest"
    )
    checkpoint_manifest = _read_canonical(
        manifest_path.read_bytes(), "candidate checkpoint manifest"
    )
    config = _training_config_from_manifest(checkpoint_manifest)
    if (
        checkpoint_manifest.get("schema") != training.CHECKPOINT_BUNDLE_SCHEMA
        or checkpoint_manifest.get("stage") != "risk"
        or checkpoint_manifest.get("completed_epoch") != config.risk_epochs
        or checkpoint_manifest.get("feature_schema_hash")
        != FEATURE_SCHEMA_HASH
        or checkpoint_manifest.get("loss_schema_hash") != LOSS_SCHEMA_HASH
        or checkpoint_manifest.get("teacher_values_are_realized_match_ev")
        is not False
        or checkpoint_manifest.get("current_profile_changed") is not False
    ):
        raise PromotionRuntimeClosureError(
            "candidate risk checkpoint boundary changed"
        )
    identity = dict(checkpoint_manifest)
    declared_bundle_identity = identity.pop("bundle_identity_sha256", None)
    if (
        not _is_sha256(declared_bundle_identity)
        or declared_bundle_identity != canonical_sha256(identity)
    ):
        raise PromotionRuntimeClosureError(
            "candidate checkpoint bundle identity changed"
        )
    models = checkpoint_manifest.get("models")
    if (
        not isinstance(models, list)
        or len(models) != config.ensemble_size
        or checkpoint_manifest.get("ensemble_size") != len(models)
    ):
        raise PromotionRuntimeClosureError(
            "candidate checkpoint ensemble grid changed"
        )
    payloads: dict[str, bytes] = {}
    expected_children = {"manifest.json"}
    model_hashes: list[str] = []
    for index, record in enumerate(models):
        if not isinstance(record, Mapping):
            raise PromotionRuntimeClosureError(
                "candidate checkpoint model record is missing"
            )
        filename = f"model_{index:02d}.zip"
        expected_children.add(filename)
        if (
            record.get("model_index") != index
            or record.get("path") != filename
            or not _is_sha256(record.get("sha256"))
            or not _is_sha256(record.get("model_state_sha256"))
        ):
            raise PromotionRuntimeClosureError(
                "candidate checkpoint model record changed"
            )
        model_path = _safe_file(
            checkpoint_bundle / filename,
            f"candidate checkpoint {filename}",
        )
        raw = model_path.read_bytes()
        if (
            len(raw) != record.get("bytes")
            or hashlib.sha256(raw).hexdigest() != record.get("sha256")
        ):
            raise PromotionRuntimeClosureError(
                "candidate checkpoint model bytes changed"
            )
        payloads[f"{CANDIDATE_BUNDLE_PREFIX}/{filename}"] = raw
        model_hashes.append(str(record["model_state_sha256"]))
    children = list(checkpoint_bundle.iterdir())
    if (
        any(child.is_symlink() or not child.is_file() for child in children)
        or {child.name for child in children} != expected_children
    ):
        raise PromotionRuntimeClosureError(
            "candidate checkpoint bundle has missing or unknown entries"
        )
    manifest_raw = manifest_path.read_bytes()
    payloads[f"{CANDIDATE_BUNDLE_PREFIX}/manifest.json"] = manifest_raw

    threshold_raw = threshold_lock_path.read_bytes()
    threshold_lock = _read_canonical(
        threshold_raw, "candidate rich threshold lock"
    )
    try:
        validated_lock = training._validate_threshold_lock(  # type: ignore[attr-defined]
            threshold_lock,
            training_config=config,
            expected_dataset_identity_sha256=str(
                checkpoint_manifest["training_view_identity_sha256"]
            ),
            expected_model_hashes=model_hashes,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PromotionRuntimeClosureError(
            "candidate rich threshold lock failed source replay"
        ) from exc
    if validated_lock != threshold_lock:
        raise PromotionRuntimeClosureError(
            "candidate rich threshold lock canonical value changed"
        )
    payloads[CANDIDATE_THRESHOLD_PATH] = threshold_raw
    return payloads, {
        "checkpoint_bundle_prefix": CANDIDATE_BUNDLE_PREFIX,
        "checkpoint_manifest_path": (
            f"{CANDIDATE_BUNDLE_PREFIX}/manifest.json"
        ),
        "checkpoint_manifest_sha256": hashlib.sha256(
            manifest_raw
        ).hexdigest(),
        "checkpoint_bundle_identity_sha256": declared_bundle_identity,
        "training_view_identity_sha256": checkpoint_manifest[
            "training_view_identity_sha256"
        ],
        "training_config_sha256": config.identity_sha256,
        "training_threshold_lock_path": CANDIDATE_THRESHOLD_PATH,
        "training_threshold_lock_sha256": hashlib.sha256(
            threshold_raw
        ).hexdigest(),
        "ensemble_size": config.ensemble_size,
        "model_state_sha256": model_hashes,
    }


def _target_os_for_library(path: Path) -> str:
    suffix = path.suffix.casefold()
    if suffix == ".so":
        return "linux"
    if suffix == ".dll":
        return "windows"
    if suffix == ".dylib":
        return "macos"
    raise PromotionRuntimeClosureError(
        "exact-T4 native library must end in .so, .dll, or .dylib"
    )


def _entry_records(payloads: Mapping[str, bytes]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
        }
        for name, raw in sorted(payloads.items())
    }


def _manifest(
    *,
    payloads: Mapping[str, bytes],
    source_entries: Sequence[str],
    candidate_artifacts: Mapping[str, Any],
    t4_archive_path: str,
) -> dict[str, Any]:
    entries = _entry_records(payloads)
    t4_record = entries[t4_archive_path]
    factory_contract = {
        "baseline": {
            "profile_id": BASELINE_PROFILE_ID,
            "factory_id": (
                f"{NAMED_PROFILE_FACTORY_PREFIX}:{BASELINE_PROFILE_ID}"
            ),
        },
        "candidate": {
            "candidate_id": "stage7_m31_street_policy_v1_locked_evaluation_only",
            "factory_id": CANDIDATE_FACTORY_ID,
            "baseline_profile_id": BASELINE_PROFILE_ID,
        },
        "population": [
            {
                "profile_id": profile_id,
                "factory_id": (
                    f"{NAMED_PROFILE_FACTORY_PREFIX}:{profile_id}"
                ),
            }
            for profile_id in POPULATION_PROFILE_IDS
        ],
        "abr": [
            {
                "response_id": response_id,
                "factory_id": factory_id,
                "artifact_binding": "external_post_plan_manifest_and_checkpoint",
            }
            for response_id, factory_id in ABR_FACTORY_IDS.items()
        ],
        "opening_lookahead_samples": OPENING_LOOKAHEAD_SAMPLES,
        "t4_policy": "hu_m3_t4_exact_both_seats_v1",
        "current_profile_allowed": False,
        "implicit_profile_resolution_allowed": False,
    }
    manifest: dict[str, Any] = {
        "schema": CLOSURE_SCHEMA,
        "status": "complete_dormant_locked_evaluation_runtime_closure",
        "archive_format": CLOSURE_ARCHIVE_FORMAT,
        "entries": entries,
        "entry_count": len(entries),
        "entry_aggregate_sha256": canonical_sha256(entries),
        "source_entries": list(sorted(source_entries)),
        "legacy_model_bindings": legacy_model_bindings(),
        "candidate_artifacts": dict(candidate_artifacts),
        "exact_t4_native": {
            "path": t4_archive_path,
            **t4_record,
            "engine_version": HU_M30_T4_ENGINE_VERSION,
            "target_os": _target_os_for_library(
                Path(PurePosixPath(t4_archive_path).name)
            ),
            "acceptance_contract": (
                "m30_runtime_v3_windows_or_m31_candidate02_linux"
            ),
        },
        "schema_bindings": {
            "action_key_schema": ACTION_KEY_SCHEMA,
            "observation_schema": OBSERVATION_SCHEMA,
            "street_policy_feature_schema_sha256": FEATURE_SCHEMA_HASH,
            "street_policy_loss_schema_sha256": LOSS_SCHEMA_HASH,
            "candidate_runtime_schema": "hu_m31_t3_street_policy_runtime_v1",
            "locked_promotion_plan_schema": (
                "hu_m31_t3_step6d_locked_promotion_plan_v1"
            ),
            "locked_promotion_runner_schema": (
                "hu_m31_t3_locked_promotion_runner_v1"
            ),
        },
        "factory_contract": factory_contract,
        "information_safety": {
            "actor_observation_only": True,
            "opponent_private_discards_used": False,
            "realized_deck_tail_exposed": False,
            "teacher_ev_lcb_is_runtime_gate": False,
            "nonfire_exact_baseline_action_required": True,
        },
        "content_addressed": True,
        "cloud_execution_started": False,
        "promotion_authorized": False,
        "named_profile_added": False,
        "current_profile_changed": False,
        "runtime_activated": False,
    }
    manifest["manifest_identity_sha256"] = canonical_sha256(manifest)
    return manifest


def _write_deterministic_zip(
    destination: Path, payloads: Mapping[str, bytes]
) -> None:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("runtime closure package is create-only")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}"
    )
    try:
        with temporary.open("xb") as raw:
            with zipfile.ZipFile(
                raw,
                "w",
                compression=zipfile.ZIP_STORED,
                strict_timestamps=True,
            ) as archive:
                for name in sorted(payloads):
                    _safe_archive_name(name)
                    info = zipfile.ZipInfo(
                        name, date_time=(1980, 1, 1, 0, 0, 0)
                    )
                    info.compress_type = zipfile.ZIP_STORED
                    info.create_system = 3
                    info.external_attr = (0o100644 & 0xFFFF) << 16
                    archive.writestr(info, payloads[name])
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def create_runtime_closure_package(
    *,
    repository_root: str | Path,
    candidate_checkpoint_bundle: str | Path,
    training_threshold_lock_path: str | Path,
    exact_t4_native_library_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Create the immutable package used as the plan's runtime closure."""

    repository = _safe_directory(repository_root, "repository root")
    source_root = _safe_directory(
        repository / "src/ofc_regular", "ofc_regular source"
    )
    checkpoint_bundle = _safe_directory(
        candidate_checkpoint_bundle, "candidate checkpoint bundle"
    )
    threshold_path = _safe_file(
        training_threshold_lock_path, "candidate rich threshold lock"
    )
    t4_library = _safe_file(
        exact_t4_native_library_path, "exact-T4 native library"
    )
    payloads: dict[str, bytes] = {}
    source_entries: list[str] = []
    for source in sorted(source_root.rglob("*.py")):
        if source.is_symlink() or not source.is_file():
            raise PromotionRuntimeClosureError(
                "ofc_regular source contains an unsafe path"
            )
        name = source.relative_to(repository).as_posix()
        _safe_archive_name(name)
        payloads[name] = source.read_bytes()
        source_entries.append(name)
    if not _REQUIRED_SOURCE_PATHS.issubset(payloads):
        missing = sorted(_REQUIRED_SOURCE_PATHS - set(payloads))
        raise PromotionRuntimeClosureError(
            f"runtime closure lacks required source files: {missing}"
        )
    if (
        hashlib.sha256(
            payloads["src/ofc_regular/ai_profiles.py"]
        ).hexdigest()
        != FROZEN_POLICY_REGISTRY_SHA256
    ):
        raise PromotionRuntimeClosureError(
            "frozen named-profile registry changed before closure creation"
        )
    for field, relative in legacy_model_bindings().items():
        model = _safe_file(
            repository / Path(relative), f"legacy model {field}"
        )
        raw = model.read_bytes()
        if (
            hashlib.sha256(raw).hexdigest()
            != FROZEN_LEGACY_MODEL_SHA256_BY_FIELD[field]
        ):
            raise PromotionRuntimeClosureError(
                f"frozen legacy model changed before closure creation: {field}"
            )
        payloads[relative] = raw
    candidate_payloads, candidate_artifacts = _candidate_payloads(
        checkpoint_bundle=checkpoint_bundle,
        threshold_lock_path=threshold_path,
    )
    if set(payloads) & set(candidate_payloads):
        raise PromotionRuntimeClosureError(
            "candidate artifacts collide with source/model paths"
        )
    payloads.update(candidate_payloads)
    target_os = _target_os_for_library(t4_library)
    accepted_t4_sha = ACCEPTED_EXACT_T4_LIBRARY_SHA256_BY_TARGET.get(
        target_os
    )
    if (
        accepted_t4_sha is None
        or sha256_file(t4_library) != accepted_t4_sha
    ):
        raise PromotionRuntimeClosureError(
            "exact-T4 native library is not an accepted platform artifact"
        )
    t4_archive_path = (
        f"native/{target_os}/release/"
        f"hu_m3_t4_exact{t4_library.suffix.casefold()}"
    )
    payloads[t4_archive_path] = t4_library.read_bytes()
    manifest = _manifest(
        payloads=payloads,
        source_entries=source_entries,
        candidate_artifacts=candidate_artifacts,
        t4_archive_path=t4_archive_path,
    )
    archive_payloads = dict(payloads)
    archive_payloads[CLOSURE_MANIFEST_PATH] = canonical_bytes(manifest)
    destination = Path(output_path).resolve()
    _write_deterministic_zip(destination, archive_payloads)
    validated = validate_runtime_closure_package(
        destination,
        expected_sha256=sha256_file(destination),
        source_replay_root=repository,
    )
    if validated != manifest:
        raise PromotionRuntimeClosureError(
            "stored runtime closure differs from source replay"
        )
    return {
        "schema": CLOSURE_SCHEMA,
        "status": "complete_dormant_locked_evaluation_runtime_closure",
        "package_path": str(destination),
        "package_sha256": sha256_file(destination),
        "package_bytes": destination.stat().st_size,
        "manifest_identity_sha256": manifest["manifest_identity_sha256"],
        "candidate_checkpoint_bundle_identity_sha256": candidate_artifacts[
            "checkpoint_bundle_identity_sha256"
        ],
        "cloud_execution_started": False,
        "promotion_authorized": False,
        "named_profile_added": False,
        "current_profile_changed": False,
        "runtime_activated": False,
    }


def _archive_payloads(path: Path) -> dict[str, bytes]:
    try:
        with zipfile.ZipFile(path, "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)) or not names:
                raise PromotionRuntimeClosureError(
                    "runtime closure has duplicate or no entries"
                )
            payloads: dict[str, bytes] = {}
            for info in infos:
                _safe_archive_name(info.filename)
                mode = (info.external_attr >> 16) & 0xFFFF
                if (
                    info.is_dir()
                    or info.compress_type != zipfile.ZIP_STORED
                    or (mode and (mode & 0o170000) != 0o100000)
                ):
                    raise PromotionRuntimeClosureError(
                        "runtime closure contains a non-regular entry"
                    )
                payloads[info.filename] = archive.read(info)
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        if isinstance(exc, PromotionRuntimeClosureError):
            raise
        raise PromotionRuntimeClosureError(
            "runtime closure is not a readable deterministic zip"
        ) from exc
    return payloads


def _validate_manifest(
    manifest: Mapping[str, Any], payloads: Mapping[str, bytes]
) -> dict[str, Any]:
    value = deepcopy(dict(manifest))
    if set(value) != _MANIFEST_KEYS:
        raise PromotionRuntimeClosureError(
            "runtime closure manifest field set changed"
        )
    declared_identity = value.pop("manifest_identity_sha256")
    if (
        not _is_sha256(declared_identity)
        or declared_identity != canonical_sha256(value)
    ):
        raise PromotionRuntimeClosureError(
            "runtime closure manifest identity changed"
        )
    value["manifest_identity_sha256"] = declared_identity
    content_payloads = {
        name: raw
        for name, raw in payloads.items()
        if name != CLOSURE_MANIFEST_PATH
    }
    entries = value.get("entries")
    if not isinstance(entries, Mapping):
        raise PromotionRuntimeClosureError(
            "runtime closure entries are missing"
        )
    actual_entries = _entry_records(content_payloads)
    if (
        value["schema"] != CLOSURE_SCHEMA
        or value["status"]
        != "complete_dormant_locked_evaluation_runtime_closure"
        or value["archive_format"] != CLOSURE_ARCHIVE_FORMAT
        or dict(entries) != actual_entries
        or value["entry_count"] != len(actual_entries)
        or value["entry_aggregate_sha256"]
        != canonical_sha256(actual_entries)
        or value["content_addressed"] is not True
        or any(
            value[field] is not False
            for field in (
                "cloud_execution_started",
                "promotion_authorized",
                "named_profile_added",
                "current_profile_changed",
                "runtime_activated",
            )
        )
    ):
        raise PromotionRuntimeClosureError(
            "runtime closure manifest boundary changed"
        )
    for record in entries.values():
        if (
            not isinstance(record, Mapping)
            or set(record) != _ENTRY_KEYS
            or not _is_sha256(record.get("sha256"))
            or isinstance(record.get("bytes"), bool)
            or not isinstance(record.get("bytes"), int)
            or record["bytes"] <= 0
        ):
            raise PromotionRuntimeClosureError(
                "runtime closure entry record changed"
            )
    sources = value.get("source_entries")
    if (
        not isinstance(sources, list)
        or sources != sorted(sources)
        or len(sources) != len(set(sources))
        or not _REQUIRED_SOURCE_PATHS.issubset(sources)
        or any(
            not isinstance(name, str)
            or not name.startswith("src/ofc_regular/")
            or not name.endswith(".py")
            or name not in entries
            for name in sources
        )
    ):
        raise PromotionRuntimeClosureError(
            "runtime closure source inventory changed"
        )
    if value.get("legacy_model_bindings") != legacy_model_bindings():
        raise PromotionRuntimeClosureError(
            "runtime closure legacy model binding changed"
        )
    if (
        entries["src/ofc_regular/ai_profiles.py"]["sha256"]
        != FROZEN_POLICY_REGISTRY_SHA256
    ):
        raise PromotionRuntimeClosureError(
            "runtime closure policy registry differs from the frozen baseline"
        )
    if any(path not in entries for path in legacy_model_bindings().values()):
        raise PromotionRuntimeClosureError(
            "runtime closure lacks a legacy model"
        )
    for field, relative in legacy_model_bindings().items():
        if (
            entries[relative]["sha256"]
            != FROZEN_LEGACY_MODEL_SHA256_BY_FIELD[field]
        ):
            raise PromotionRuntimeClosureError(
                f"runtime closure legacy model hash changed: {field}"
            )
    candidate = value.get("candidate_artifacts")
    if not isinstance(candidate, Mapping):
        raise PromotionRuntimeClosureError(
            "runtime closure candidate artifact record is missing"
        )
    required_candidate = {
        "checkpoint_bundle_prefix",
        "checkpoint_manifest_path",
        "checkpoint_manifest_sha256",
        "checkpoint_bundle_identity_sha256",
        "training_view_identity_sha256",
        "training_config_sha256",
        "training_threshold_lock_path",
        "training_threshold_lock_sha256",
        "ensemble_size",
        "model_state_sha256",
    }
    model_hashes = candidate.get("model_state_sha256")
    if (
        set(candidate) != required_candidate
        or candidate["checkpoint_bundle_prefix"] != CANDIDATE_BUNDLE_PREFIX
        or candidate["checkpoint_manifest_path"]
        != f"{CANDIDATE_BUNDLE_PREFIX}/manifest.json"
        or candidate["training_threshold_lock_path"]
        != CANDIDATE_THRESHOLD_PATH
        or any(
            not _is_sha256(candidate[field])
            for field in (
                "checkpoint_manifest_sha256",
                "checkpoint_bundle_identity_sha256",
                "training_view_identity_sha256",
                "training_config_sha256",
                "training_threshold_lock_sha256",
            )
        )
        or not isinstance(model_hashes, list)
        or not model_hashes
        or any(not _is_sha256(item) for item in model_hashes)
        or candidate["ensemble_size"] != len(model_hashes)
        or candidate["checkpoint_manifest_path"] not in entries
        or candidate["training_threshold_lock_path"] not in entries
        or entries[candidate["checkpoint_manifest_path"]]["sha256"]
        != candidate["checkpoint_manifest_sha256"]
        or entries[candidate["training_threshold_lock_path"]]["sha256"]
        != candidate["training_threshold_lock_sha256"]
    ):
        raise PromotionRuntimeClosureError(
            "runtime closure candidate artifact binding changed"
        )
    t4 = value.get("exact_t4_native")
    if (
        not isinstance(t4, Mapping)
        or set(t4)
        != {
            "path",
            "sha256",
            "bytes",
            "engine_version",
            "target_os",
            "acceptance_contract",
        }
        or t4["path"] not in entries
        or entries[t4["path"]]
        != {"sha256": t4["sha256"], "bytes": t4["bytes"]}
        or t4["engine_version"] != HU_M30_T4_ENGINE_VERSION
        or t4["target_os"] not in {"linux", "windows", "macos"}
        or t4["acceptance_contract"]
        != "m30_runtime_v3_windows_or_m31_candidate02_linux"
        or t4["sha256"]
        != ACCEPTED_EXACT_T4_LIBRARY_SHA256_BY_TARGET.get(t4["target_os"])
        or not str(t4["path"]).startswith(f"native/{t4['target_os']}/")
    ):
        raise PromotionRuntimeClosureError(
            "runtime closure exact-T4 binding changed"
        )
    expected_factory = _manifest(
        payloads=content_payloads,
        source_entries=sources,
        candidate_artifacts=candidate,
        t4_archive_path=str(t4["path"]),
    )
    if (
        value["factory_contract"] != expected_factory["factory_contract"]
        or value["schema_bindings"] != expected_factory["schema_bindings"]
        or value["information_safety"]
        != expected_factory["information_safety"]
    ):
        raise PromotionRuntimeClosureError(
            "runtime closure semantic/factory contract changed"
        )
    # Revalidate the candidate bundle and rich threshold from archived bytes.
    checkpoint_manifest = _read_canonical(
        content_payloads[candidate["checkpoint_manifest_path"]],
        "archived candidate checkpoint manifest",
    )
    config = _training_config_from_manifest(checkpoint_manifest)
    identity = dict(checkpoint_manifest)
    declared = identity.pop("bundle_identity_sha256", None)
    if (
        declared != candidate["checkpoint_bundle_identity_sha256"]
        or declared != canonical_sha256(identity)
        or checkpoint_manifest.get("training_view_identity_sha256")
        != candidate["training_view_identity_sha256"]
        or checkpoint_manifest.get("ensemble_size")
        != candidate["ensemble_size"]
        or checkpoint_manifest.get("stage") != "risk"
        or checkpoint_manifest.get("completed_epoch") != config.risk_epochs
    ):
        raise PromotionRuntimeClosureError(
            "archived candidate checkpoint contract changed"
        )
    records = checkpoint_manifest.get("models")
    if not isinstance(records, list) or len(records) != len(model_hashes):
        raise PromotionRuntimeClosureError(
            "archived candidate model grid changed"
        )
    for index, record in enumerate(records):
        archive_name = f"{CANDIDATE_BUNDLE_PREFIX}/model_{index:02d}.zip"
        raw = content_payloads.get(archive_name)
        if (
            not isinstance(record, Mapping)
            or raw is None
            or record.get("path") != f"model_{index:02d}.zip"
            or record.get("model_state_sha256") != model_hashes[index]
            or record.get("sha256") != hashlib.sha256(raw).hexdigest()
            or record.get("bytes") != len(raw)
        ):
            raise PromotionRuntimeClosureError(
                "archived candidate checkpoint bytes changed"
            )
    threshold = _read_canonical(
        content_payloads[CANDIDATE_THRESHOLD_PATH],
        "archived candidate rich threshold lock",
    )
    try:
        training._validate_threshold_lock(  # type: ignore[attr-defined]
            threshold,
            training_config=config,
            expected_dataset_identity_sha256=str(
                candidate["training_view_identity_sha256"]
            ),
            expected_model_hashes=[str(item) for item in model_hashes],
        )
    except (TypeError, ValueError) as exc:
        raise PromotionRuntimeClosureError(
            "archived rich threshold lock changed"
        ) from exc
    return value


def validate_runtime_closure_package(
    package_path: str | Path,
    *,
    expected_sha256: str | None = None,
    source_replay_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate package bytes and optionally replay live source/model files."""

    package = _safe_file(package_path, "runtime closure package")
    if expected_sha256 is not None:
        if not _is_sha256(expected_sha256):
            raise ValueError(
                "expected runtime closure SHA-256 must be lowercase hex"
            )
        if sha256_file(package) != expected_sha256:
            raise PromotionRuntimeClosureError(
                "runtime closure package SHA-256 changed"
            )
    payloads = _archive_payloads(package)
    if CLOSURE_MANIFEST_PATH not in payloads:
        raise PromotionRuntimeClosureError(
            "runtime closure package lacks its manifest"
        )
    manifest = _validate_manifest(
        _read_canonical(
            payloads[CLOSURE_MANIFEST_PATH],
            "runtime closure manifest",
        ),
        payloads,
    )
    if source_replay_root is not None:
        repository = _safe_directory(
            source_replay_root, "runtime closure source replay root"
        )
        live_sources = {
            path.relative_to(repository).as_posix(): path.read_bytes()
            for path in sorted(
                _safe_directory(
                    repository / "src/ofc_regular",
                    "source replay ofc_regular",
                ).rglob("*.py")
            )
            if path.is_file() and not path.is_symlink()
        }
        if set(live_sources) != set(manifest["source_entries"]):
            raise PromotionRuntimeClosureError(
                "live ofc_regular source inventory differs from closure"
            )
        for name, raw in live_sources.items():
            if payloads.get(name) != raw:
                raise PromotionRuntimeClosureError(
                    f"live source differs from closure: {name}"
                )
        for field, relative in manifest["legacy_model_bindings"].items():
            live = _safe_file(
                repository / Path(relative),
                f"live legacy model {field}",
            )
            if payloads.get(relative) != live.read_bytes():
                raise PromotionRuntimeClosureError(
                    f"live legacy model differs from closure: {field}"
                )
    return manifest


def extract_runtime_closure_package(
    package_path: str | Path,
    *,
    expected_sha256: str,
    output_directory: str | Path,
    source_replay_root: str | Path | None = None,
) -> dict[str, Any]:
    """Atomically extract a validated closure into a create-only directory."""

    package = _safe_file(package_path, "runtime closure package")
    manifest = validate_runtime_closure_package(
        package,
        expected_sha256=expected_sha256,
        source_replay_root=source_replay_root,
    )
    destination = Path(output_directory).resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            "runtime closure extraction directory is create-only"
        )
    temporary = destination.with_name(
        f".{destination.name}.tmp-{os.getpid()}"
    )
    if temporary.exists():
        raise FileExistsError(
            "runtime closure extraction temporary path already exists"
        )
    payloads = _archive_payloads(package)
    try:
        temporary.mkdir(parents=True)
        for name, raw in sorted(payloads.items()):
            relative = _safe_archive_name(name)
            target = temporary.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    for name, record in manifest["entries"].items():
        target = destination.joinpath(*PurePosixPath(name).parts)
        if (
            target.is_symlink()
            or not target.is_file()
            or target.stat().st_size != record["bytes"]
            or sha256_file(target) != record["sha256"]
        ):
            raise PromotionRuntimeClosureError(
                "extracted runtime closure failed byte replay"
            )
    return {
        "schema": CLOSURE_SCHEMA,
        "status": "validated_dormant_runtime_closure_extracted",
        "package_sha256": expected_sha256,
        "manifest_identity_sha256": manifest["manifest_identity_sha256"],
        "output_directory": str(destination),
        "target_os": manifest["exact_t4_native"]["target_os"],
        "cloud_execution_started": False,
        "promotion_authorized": False,
        "named_profile_added": False,
        "current_profile_changed": False,
        "runtime_activated": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="M3.1 locked-promotion runtime closure"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--repository-root", required=True)
    create.add_argument("--candidate-checkpoint-bundle", required=True)
    create.add_argument("--training-threshold-lock", required=True)
    create.add_argument("--exact-t4-native-library", required=True)
    create.add_argument("--output", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--package", required=True)
    validate.add_argument("--expected-sha256", required=True)
    validate.add_argument("--source-replay-root", required=True)
    extract = subparsers.add_parser("extract")
    extract.add_argument("--package", required=True)
    extract.add_argument("--expected-sha256", required=True)
    extract.add_argument("--output-directory", required=True)
    extract.add_argument("--source-replay-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "create":
            result = create_runtime_closure_package(
                repository_root=args.repository_root,
                candidate_checkpoint_bundle=args.candidate_checkpoint_bundle,
                training_threshold_lock_path=args.training_threshold_lock,
                exact_t4_native_library_path=args.exact_t4_native_library,
                output_path=args.output,
            )
        elif args.command == "validate":
            manifest = validate_runtime_closure_package(
                args.package,
                expected_sha256=args.expected_sha256,
                source_replay_root=args.source_replay_root,
            )
            result = {
                "schema": CLOSURE_SCHEMA,
                "status": "validated_dormant_runtime_closure",
                "package_sha256": args.expected_sha256,
                "manifest_identity_sha256": manifest[
                    "manifest_identity_sha256"
                ],
                "cloud_execution_started": False,
                "promotion_authorized": False,
                "named_profile_added": False,
                "current_profile_changed": False,
                "runtime_activated": False,
            }
        else:
            result = extract_runtime_closure_package(
                args.package,
                expected_sha256=args.expected_sha256,
                output_directory=args.output_directory,
                source_replay_root=args.source_replay_root,
            )
    except Exception as exc:  # pragma: no cover - subprocess boundary
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ACCEPTED_EXACT_T4_LIBRARY_SHA256_BY_TARGET",
    "ABR_FACTORY_IDS",
    "BASELINE_PROFILE_ID",
    "CANDIDATE_BUNDLE_PREFIX",
    "CANDIDATE_FACTORY_ID",
    "CANDIDATE_THRESHOLD_PATH",
    "CLOSURE_MANIFEST_PATH",
    "CLOSURE_SCHEMA",
    "FROZEN_POLICY_REGISTRY_SHA256",
    "FROZEN_LEGACY_MODEL_SHA256_BY_FIELD",
    "NAMED_PROFILE_FACTORY_PREFIX",
    "OPENING_LOOKAHEAD_SAMPLES",
    "POPULATION_PROFILE_IDS",
    "PromotionRuntimeClosureError",
    "canonical_bytes",
    "canonical_sha256",
    "create_runtime_closure_package",
    "extract_runtime_closure_package",
    "legacy_model_bindings",
    "main",
    "sha256_file",
    "validate_runtime_closure_package",
]
