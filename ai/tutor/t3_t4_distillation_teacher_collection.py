"""Streaming, resumable collection of verified T3/T4 teacher bundles.

This module does not generate labels.  It accepts only rows authenticated by
the existing v2 teacher verifier, groups at most ``MAX_IN_MEMORY_ROWS`` rows in
each content-addressed bundle, and maintains an append-only authenticated top
state.  Collection artifacts are candidate-only and never change serving.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import types
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import ai.tutor.t3_t4_distillation_teacher as _teacher_module
import ai.tutor.t3_t4_infoset_encoder as _encoder_module
from ai.tutor.runtime_semantic_anchor import (
    semantic_state,
    verify_module_function_anchor,
)
from ai.tutor.t3_t4_distillation_teacher import (
    MAX_IN_MEMORY_ROWS,
    MAX_ROWS_PER_SHARD,
    MCCFR_SOLVER_METHOD,
    SPLIT_NAMES,
    TEACHER_BUNDLE_SCHEMA,
    TeacherContractError,
    VerifiedTeacherBundle,
    canonical_json,
    canonical_sha256,
    self_hash,
    verify_teacher_bundle,
    verify_teacher_row,
    write_teacher_bundle,
)


COLLECTION_PLAN_SCHEMA = "ofc_t3_t4_teacher_collection_plan/v1"
COLLECTION_CHUNK_ENTRY_SCHEMA = "ofc_t3_t4_teacher_collection_chunk_entry/v1"
COLLECTION_TOP_SCHEMA = "ofc_t3_t4_teacher_collection_top/v1"
COLLECTION_RUNTIME_SCHEMA = "ofc_t3_t4_teacher_collection_runtime/v1"
LOCK_NAME = ".teacher-collection.lock"
CHUNKS_DIRECTORY = "chunks"
MANIFESTS_DIRECTORY = "manifests"
MAX_EXPECTED_ROWS = 1_000_000_000
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_PLAN_FILE = re.compile(r"generation-plan-([0-9a-f]{64})\.json\Z")
_TOP_FILE = re.compile(
    r"top-(?P<sequence>[0-9]{8})-(?P<digest>[0-9a-f]{64})\.json\Z"
)
_CHUNK_DIRECTORY = re.compile(
    r"chunk-(?P<index>[0-9]{6})-(?P<digest>[0-9a-f]{64})\Z"
)
_CHUNK_MARKER = re.compile(
    r"chunk-(?P<index>[0-9]{6})-(?P<digest>[0-9a-f]{64})\.complete\.json\Z"
)
_STAGE_DIRECTORY = re.compile(r"\.stage-[0-9]+-[0-9a-f]{32}\Z")

PHASE_ACTOR_JOKER_CELLS = tuple(
    f"{phase}_{actor}_joker_{joker}"
    for phase, actor in (
        ("t3_first", "bb"),
        ("t3_second", "btn"),
        ("t4_first", "bb"),
        ("t4_second", "btn"),
    )
    for joker in range(3)
)

_VERIFY_TEACHER_ROW = verify_teacher_row
_VERIFY_TEACHER_BUNDLE = verify_teacher_bundle
_WRITE_TEACHER_BUNDLE = write_teacher_bundle
_CANONICAL_JSON = canonical_json
_CANONICAL_SHA256 = canonical_sha256
_SELF_HASH = self_hash

_PINNED_SOURCE_PATHS = {
    "collection": Path(__file__).resolve(),
    "teacher": Path(_teacher_module.__file__).resolve(),
    "encoder": Path(_encoder_module.__file__).resolve(),
    "runtime_anchor": Path(semantic_state.__code__.co_filename).resolve(),
}
_PINNED_SOURCE_SHA256 = {
    label: hashlib.sha256(path.read_bytes()).hexdigest()
    for label, path in _PINNED_SOURCE_PATHS.items()
}
_EXTERNAL_BINDINGS = (
    ("teacher.verify_teacher_row", _teacher_module, "verify_teacher_row", _VERIFY_TEACHER_ROW),
    (
        "teacher.verify_teacher_bundle",
        _teacher_module,
        "verify_teacher_bundle",
        _VERIFY_TEACHER_BUNDLE,
    ),
    (
        "teacher.write_teacher_bundle",
        _teacher_module,
        "write_teacher_bundle",
        _WRITE_TEACHER_BUNDLE,
    ),
    ("teacher.canonical_json", _teacher_module, "canonical_json", _CANONICAL_JSON),
    (
        "teacher.canonical_sha256",
        _teacher_module,
        "canonical_sha256",
        _CANONICAL_SHA256,
    ),
    ("teacher.self_hash", _teacher_module, "self_hash", _SELF_HASH),
)
_EXTERNAL_SEMANTIC_STATES = tuple(
    (label, semantic_state(target))
    for label, _owner, _attribute, target in _EXTERNAL_BINDINGS
)
_COLLECTION_ALIAS_BINDINGS = (
    ("_VERIFY_TEACHER_ROW", _VERIFY_TEACHER_ROW),
    ("_VERIFY_TEACHER_BUNDLE", _VERIFY_TEACHER_BUNDLE),
    ("_WRITE_TEACHER_BUNDLE", _WRITE_TEACHER_BUNDLE),
    ("_CANONICAL_JSON", _CANONICAL_JSON),
    ("_CANONICAL_SHA256", _CANONICAL_SHA256),
    ("_SELF_HASH", _SELF_HASH),
)


class TeacherCollectionError(ValueError):
    """A collection plan, prefix, chunk, or finalization failed closed."""


class TeacherCollectionLockError(TeacherCollectionError):
    """Another writer or verifier owns the run-wide OS lock."""


@dataclass(frozen=True)
class VerifiedTeacherCollection:
    root: Path
    plan: dict[str, Any]
    top_manifest: dict[str, Any]

    @property
    def finalized(self) -> bool:
        return bool(self.top_manifest["finalized"])


@dataclass
class _AuditState:
    plan_sha256: str
    row_count: int = 0
    row_identities: set[str] = field(default_factory=set)
    row_contents: set[str] = field(default_factory=set)
    full_deals: set[str] = field(default_factory=set)
    root_families: set[str] = field(default_factory=set)
    information_digests: set[str] = field(default_factory=set)
    split_counts: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in SPLIT_NAMES}
    )
    stratum_counts: dict[str, int] = field(
        default_factory=lambda: {name: 0 for name in PHASE_ACTOR_JOKER_CELLS}
    )
    split_owners: dict[str, dict[str, str]] = field(
        default_factory=lambda: {
            "full_deal_commitment_sha256": {},
            "public_root_family_commitment_sha256": {},
            "information_digest": {},
        }
    )
    root_family_deal: dict[str, str] = field(default_factory=dict)
    solver_seeds: set[int] = field(default_factory=set)
    payoff_seeds: set[int] = field(default_factory=set)
    ordered_prefix_sha256: str = ""

    def __post_init__(self) -> None:
        if not self.ordered_prefix_sha256:
            self.ordered_prefix_sha256 = _CANONICAL_SHA256(
                {
                    "schema": "ofc_t3_t4_collection_ordered_prefix/v1",
                    "generation_plan_sha256": self.plan_sha256,
                    "initial": True,
                }
            )


def _sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise TeacherCollectionError(f"{label}: lowercase SHA-256 required")
    return value


def _integer(
    value: Any,
    *,
    label: str,
    minimum: int,
    maximum: int,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not minimum <= value <= maximum
    ):
        raise TeacherCollectionError(
            f"{label}: integer in [{minimum},{maximum}] required"
        )
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise TeacherCollectionError(
            f"{label}: exact fields required; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _source_hashes() -> dict[str, str]:
    current: dict[str, str] = {}
    for label, path in sorted(_PINNED_SOURCE_PATHS.items()):
        if not path.is_file() or path.is_symlink():
            raise TeacherCollectionError(f"source file {label} is not regular")
        current[label] = hashlib.sha256(path.read_bytes()).hexdigest()
    if current != _PINNED_SOURCE_SHA256:
        raise TeacherCollectionError("collection source files changed after import")
    return current


def _runtime_semantics() -> dict[str, Any]:
    for name, canonical in _COLLECTION_ALIAS_BINDINGS:
        if globals().get(name) is not canonical:
            raise TeacherCollectionError(f"collection runtime alias drift: {name}")
    for label, module, primary_name, mirror_name in (
        (
            "teacher",
            _teacher_module,
            "_TEACHER_RUNTIME_SEMANTIC_ANCHOR",
            "_TEACHER_RUNTIME_SEMANTIC_ANCHOR_MIRROR",
        ),
        (
            "encoder",
            _encoder_module,
            "_INFOSET_ENCODER_RUNTIME_SEMANTIC_ANCHOR",
            "_INFOSET_ENCODER_RUNTIME_SEMANTIC_ANCHOR_MIRROR",
        ),
    ):
        anchor = getattr(module, primary_name, None)
        if anchor is not getattr(module, mirror_name, None):
            raise TeacherCollectionError(f"{label} runtime anchor alias drift")
        try:
            verify_module_function_anchor(vars(module), anchor)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise TeacherCollectionError(
                f"{label} runtime semantic anchor verification failed: {exc}"
            ) from exc
    states = dict(_EXTERNAL_SEMANTIC_STATES)
    bindings: dict[str, str] = {}
    for label, owner, attribute, canonical in _EXTERNAL_BINDINGS:
        current = getattr(owner, attribute, None)
        if current is not canonical:
            raise TeacherCollectionError(f"runtime callable alias drift: {label}")
        current_state = semantic_state(current)
        if current_state != states[label]:
            raise TeacherCollectionError(f"runtime callable semantic drift: {label}")
        bindings[label] = hashlib.sha256(repr(current_state).encode("utf-8")).hexdigest()
    contract: dict[str, Any] = {
        "schema": COLLECTION_RUNTIME_SCHEMA,
        "callable_semantic_sha256": bindings,
        "teacher_runtime_anchor_identity_bound": True,
        "encoder_runtime_anchor_identity_bound": True,
    }
    contract["runtime_semantics_sha256"] = _SELF_HASH(
        contract, "runtime_semantics_sha256"
    )
    return contract


def _canonical_bytes(value: Any) -> bytes:
    return (_CANONICAL_JSON(value) + "\n").encode("utf-8")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TeacherCollectionError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_canonical(path: Path, *, label: str) -> Any:
    if path.is_symlink() or not path.is_file():
        raise TeacherCollectionError(f"{label}: regular non-symlink file required")
    try:
        payload = path.read_bytes()
        text = payload.decode("utf-8", errors="strict")
    except (OSError, UnicodeError) as exc:
        raise TeacherCollectionError(f"{label}: cannot read: {exc}") from exc
    if not text.endswith("\n") or text.count("\n") != 1:
        raise TeacherCollectionError(f"{label}: one canonical JSON line required")
    try:
        value = json.loads(text[:-1], object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise TeacherCollectionError(f"{label}: invalid JSON") from exc
    if _CANONICAL_JSON(value) != text[:-1]:
        raise TeacherCollectionError(f"{label}: noncanonical JSON")
    return value


def _publish_bytes_no_replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
                raise TeacherCollectionError(
                    f"immutable artifact collision: {path.name}"
                ) from exc
        except OSError as exc:
            raise TeacherCollectionError(
                f"no-replace publication failed: {path.name}: {exc}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    if path.is_symlink() or path.read_bytes() != payload:
        raise TeacherCollectionError(f"publication readback mismatch: {path.name}")


def _publish_json_no_replace(path: Path, value: Any) -> None:
    _publish_bytes_no_replace(path, _canonical_bytes(value))


def _prepare_root(value: str | Path, *, create: bool) -> Path:
    supplied = Path(value).absolute()
    if supplied.is_symlink():
        raise TeacherCollectionError("collection root symlink forbidden")
    if not supplied.exists():
        if not create:
            raise TeacherCollectionError("collection root does not exist")
        supplied.mkdir(parents=True)
    if not supplied.is_dir():
        raise TeacherCollectionError("collection root must be a directory")
    return supplied.resolve(strict=True)


@contextmanager
def _collection_lock(root: Path) -> Iterator[None]:
    lock_path = root / LOCK_NAME
    handle = lock_path.open("a+b")
    locked = False
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
            os.fsync(handle.fileno())
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError as exc:
            raise TeacherCollectionLockError(
                "teacher collection run-wide OS lock is busy"
            ) from exc
        yield
    finally:
        if locked:
            handle.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
        handle.close()


def _plan_name(plan_sha256: str) -> str:
    return f"generation-plan-{plan_sha256}.json"


def _top_name(sequence: int, digest: str) -> str:
    return f"top-{sequence:08d}-{digest}.json"


def _chunk_stem(index: int, bundle_manifest_sha256: str) -> str:
    return f"chunk-{index:06d}-{bundle_manifest_sha256}"


def _marker_name(stem: str) -> str:
    return f"{stem}.complete.json"


def _minimum_counts(value: Mapping[str, Any]) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise TeacherCollectionError("phase_actor_joker_minimum_counts must be mapping")
    if set(value) != set(PHASE_ACTOR_JOKER_CELLS):
        raise TeacherCollectionError("all phase x actor x Joker minima are required")
    return {
        cell: _integer(
            value[cell],
            label=f"minimum count {cell}",
            minimum=1,
            maximum=MAX_EXPECTED_ROWS,
        )
        for cell in PHASE_ACTOR_JOKER_CELLS
    }


def build_generation_plan(
    *,
    generation_source_sha256: str,
    expected_total_row_count: int,
    chunk_row_limit: int,
    bundle_shard_size: int,
    phase_actor_joker_minimum_counts: Mapping[str, int],
) -> dict[str, Any]:
    """Build the immutable plan that must exist before the first chunk."""

    expected_total = _integer(
        expected_total_row_count,
        label="expected_total_row_count",
        minimum=1,
        maximum=MAX_EXPECTED_ROWS,
    )
    limit = _integer(
        chunk_row_limit,
        label="chunk_row_limit",
        minimum=1,
        maximum=MAX_IN_MEMORY_ROWS,
    )
    shard_size = _integer(
        bundle_shard_size,
        label="bundle_shard_size",
        minimum=1,
        maximum=MAX_ROWS_PER_SHARD,
    )
    minima = _minimum_counts(phase_actor_joker_minimum_counts)
    if sum(minima.values()) > expected_total:
        raise TeacherCollectionError("coverage minima exceed expected total rows")
    plan: dict[str, Any] = {
        "schema": COLLECTION_PLAN_SCHEMA,
        "artifact_kind": "predeclared_streaming_t3_t4_teacher_generation_plan",
        "algorithm_teacher_only": True,
        "promotion_eligible": False,
        "runtime_allowed": False,
        "serving_changed": False,
        "plan_fields_locked_before_first_chunk": True,
        "generation_source_sha256": _sha(
            generation_source_sha256, label="generation_source_sha256"
        ),
        "expected_total_row_count": expected_total,
        "chunk_row_limit": limit,
        "bundle_shard_size": shard_size,
        "max_in_memory_rows": MAX_IN_MEMORY_ROWS,
        "teacher_bundle_schema": TEACHER_BUNDLE_SCHEMA,
        "phase_actor_joker_minimum_counts": minima,
        "source_file_sha256": _source_hashes(),
        "runtime_semantics": _runtime_semantics(),
    }
    plan["generation_plan_sha256"] = _SELF_HASH(
        plan, "generation_plan_sha256"
    )
    return plan


_PLAN_KEYS = {
    "schema",
    "artifact_kind",
    "algorithm_teacher_only",
    "promotion_eligible",
    "runtime_allowed",
    "serving_changed",
    "plan_fields_locked_before_first_chunk",
    "generation_source_sha256",
    "expected_total_row_count",
    "chunk_row_limit",
    "bundle_shard_size",
    "max_in_memory_rows",
    "teacher_bundle_schema",
    "phase_actor_joker_minimum_counts",
    "source_file_sha256",
    "runtime_semantics",
    "generation_plan_sha256",
}


def _verify_plan(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TeacherCollectionError("generation plan must be mapping")
    raw = dict(value)
    _exact_keys(raw, _PLAN_KEYS, label="generation plan")
    fixed = {
        "schema": COLLECTION_PLAN_SCHEMA,
        "artifact_kind": "predeclared_streaming_t3_t4_teacher_generation_plan",
        "algorithm_teacher_only": True,
        "promotion_eligible": False,
        "runtime_allowed": False,
        "serving_changed": False,
        "plan_fields_locked_before_first_chunk": True,
        "max_in_memory_rows": MAX_IN_MEMORY_ROWS,
        "teacher_bundle_schema": TEACHER_BUNDLE_SCHEMA,
        "source_file_sha256": _source_hashes(),
        "runtime_semantics": _runtime_semantics(),
    }
    for field_name, expected in fixed.items():
        if raw.get(field_name) != expected:
            raise TeacherCollectionError(f"generation plan {field_name} mismatch")
    rebuilt = build_generation_plan(
        generation_source_sha256=_sha(
            raw.get("generation_source_sha256"),
            label="plan generation_source_sha256",
        ),
        expected_total_row_count=_integer(
            raw.get("expected_total_row_count"),
            label="plan expected_total_row_count",
            minimum=1,
            maximum=MAX_EXPECTED_ROWS,
        ),
        chunk_row_limit=_integer(
            raw.get("chunk_row_limit"),
            label="plan chunk_row_limit",
            minimum=1,
            maximum=MAX_IN_MEMORY_ROWS,
        ),
        bundle_shard_size=_integer(
            raw.get("bundle_shard_size"),
            label="plan bundle_shard_size",
            minimum=1,
            maximum=MAX_ROWS_PER_SHARD,
        ),
        phase_actor_joker_minimum_counts=_minimum_counts(
            raw.get("phase_actor_joker_minimum_counts")
        ),
    )
    if raw != rebuilt:
        raise TeacherCollectionError("generation plan content or self-hash mismatch")
    return raw


def _read_plan(root: Path) -> dict[str, Any]:
    candidates = [path for path in root.iterdir() if _PLAN_FILE.fullmatch(path.name)]
    if len(candidates) != 1:
        raise TeacherCollectionError("exactly one immutable generation plan required")
    plan = _verify_plan(_read_canonical(candidates[0], label="generation plan"))
    if candidates[0].name != _plan_name(plan["generation_plan_sha256"]):
        raise TeacherCollectionError("generation plan content-addressed path mismatch")
    return plan


def _statistics(state: _AuditState) -> dict[str, Any]:
    return {
        "row_count": state.row_count,
        "split_row_counts": dict(state.split_counts),
        "phase_actor_joker_counts": dict(state.stratum_counts),
        "row_identity_count": len(state.row_identities),
        "row_content_count": len(state.row_contents),
        "full_deal_count": len(state.full_deals),
        "public_root_family_count": len(state.root_families),
        "information_digest_count": len(state.information_digests),
        "row_identity_set_sha256": _CANONICAL_SHA256(
            sorted(state.row_identities)
        ),
        "row_content_set_sha256": _CANONICAL_SHA256(sorted(state.row_contents)),
        "full_deal_set_sha256": _CANONICAL_SHA256(sorted(state.full_deals)),
        "public_root_family_set_sha256": _CANONICAL_SHA256(
            sorted(state.root_families)
        ),
        "information_digest_set_sha256": _CANONICAL_SHA256(
            sorted(state.information_digests)
        ),
        "solver_seed_set_sha256": _CANONICAL_SHA256(sorted(state.solver_seeds)),
        "payoff_seed_set_sha256": _CANONICAL_SHA256(sorted(state.payoff_seeds)),
        "cross_split_overlap_count": 0,
        "ordered_prefix_sha256": state.ordered_prefix_sha256,
    }


def _finalization_payload(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "generation_plan_sha256": plan["generation_plan_sha256"],
        "expected_total_row_count": plan["expected_total_row_count"],
        "fit_dev_test_nonempty_required": True,
        "phase_actor_joker_minima": plan[
            "phase_actor_joker_minimum_counts"
        ],
        "all_requirements_satisfied": True,
    }


def _build_top_manifest(
    *,
    plan: Mapping[str, Any],
    sequence: int,
    previous_sha256: str | None,
    chunks: Sequence[Mapping[str, Any]],
    state: _AuditState,
    finalized: bool,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema": COLLECTION_TOP_SCHEMA,
        "artifact_kind": "append_only_authenticated_teacher_collection_top",
        "algorithm_teacher_only": True,
        "promotion_eligible": False,
        "runtime_allowed": False,
        "serving_changed": False,
        "manifest_written_after_complete_chunks": True,
        "append_only": True,
        "generation_plan_sha256": plan["generation_plan_sha256"],
        "source_file_sha256": plan["source_file_sha256"],
        "runtime_semantics": plan["runtime_semantics"],
        "sequence": sequence,
        "previous_top_manifest_sha256": previous_sha256,
        "chunk_count": len(chunks),
        "chunks": [dict(entry) for entry in chunks],
        "statistics": _statistics(state),
        "finalized": finalized,
        "finalization": _finalization_payload(plan) if finalized else None,
    }
    manifest["top_manifest_sha256"] = _SELF_HASH(
        manifest, "top_manifest_sha256"
    )
    return manifest


_TOP_KEYS = {
    "schema",
    "artifact_kind",
    "algorithm_teacher_only",
    "promotion_eligible",
    "runtime_allowed",
    "serving_changed",
    "manifest_written_after_complete_chunks",
    "append_only",
    "generation_plan_sha256",
    "source_file_sha256",
    "runtime_semantics",
    "sequence",
    "previous_top_manifest_sha256",
    "chunk_count",
    "chunks",
    "statistics",
    "finalized",
    "finalization",
    "top_manifest_sha256",
}


def _read_top_manifests(root: Path, plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    directory = root / MANIFESTS_DIRECTORY
    if directory.is_symlink() or not directory.is_dir():
        raise TeacherCollectionError("manifests directory missing or unsafe")
    indexed: dict[int, tuple[Path, str]] = {}
    for path in directory.iterdir():
        match = _TOP_FILE.fullmatch(path.name)
        if match is None or path.is_symlink() or not path.is_file():
            raise TeacherCollectionError("invalid or non-regular top manifest artifact")
        sequence = int(match.group("sequence"))
        if sequence in indexed:
            raise TeacherCollectionError("multiple top manifests for one sequence")
        indexed[sequence] = (path, match.group("digest"))
    if not indexed or sorted(indexed) != list(range(len(indexed))):
        raise TeacherCollectionError("top manifest sequence gap or missing genesis")
    result: list[dict[str, Any]] = []
    previous: str | None = None
    prior_chunks: list[dict[str, Any]] = []
    prior_finalized = False
    for sequence in range(len(indexed)):
        path, filename_digest = indexed[sequence]
        raw = _read_canonical(path, label=f"top manifest {sequence}")
        if not isinstance(raw, Mapping):
            raise TeacherCollectionError("top manifest must be mapping")
        manifest = dict(raw)
        _exact_keys(manifest, _TOP_KEYS, label=f"top manifest {sequence}")
        fixed = {
            "schema": COLLECTION_TOP_SCHEMA,
            "artifact_kind": "append_only_authenticated_teacher_collection_top",
            "algorithm_teacher_only": True,
            "promotion_eligible": False,
            "runtime_allowed": False,
            "serving_changed": False,
            "manifest_written_after_complete_chunks": True,
            "append_only": True,
            "generation_plan_sha256": plan["generation_plan_sha256"],
            "source_file_sha256": plan["source_file_sha256"],
            "runtime_semantics": plan["runtime_semantics"],
            "sequence": sequence,
            "previous_top_manifest_sha256": previous,
        }
        for field_name, expected in fixed.items():
            if manifest.get(field_name) != expected:
                raise TeacherCollectionError(
                    f"top manifest {sequence} {field_name} mismatch"
                )
        digest = manifest.get("top_manifest_sha256")
        if (
            digest != filename_digest
            or digest != _SELF_HASH(manifest, "top_manifest_sha256")
            or path.name != _top_name(sequence, digest)
        ):
            raise TeacherCollectionError("top manifest hash/path mismatch")
        chunks = manifest.get("chunks")
        if not isinstance(chunks, list) or manifest.get("chunk_count") != len(chunks):
            raise TeacherCollectionError("top manifest chunk list/count mismatch")
        finalized = manifest.get("finalized")
        if not isinstance(finalized, bool):
            raise TeacherCollectionError("top manifest finalized flag invalid")
        if prior_finalized:
            raise TeacherCollectionError("state appended after finalization")
        if sequence == 0:
            if chunks or finalized:
                raise TeacherCollectionError("genesis top must be empty and unfinalized")
        elif finalized:
            if chunks != prior_chunks:
                raise TeacherCollectionError("finalization cannot alter chunk prefix")
        elif len(chunks) != len(prior_chunks) + 1 or chunks[:-1] != prior_chunks:
            raise TeacherCollectionError("top state is not a one-chunk prefix extension")
        prior_chunks = [dict(entry) for entry in chunks]
        prior_finalized = finalized
        previous = digest
        result.append(manifest)
    return result


_CHUNK_ENTRY_KEYS = {
    "schema",
    "artifact_kind",
    "algorithm_teacher_only",
    "promotion_eligible",
    "runtime_allowed",
    "serving_changed",
    "generation_plan_sha256",
    "index",
    "row_start",
    "row_stop_exclusive",
    "row_count",
    "bundle_relative_path",
    "completion_marker_relative_path",
    "bundle_manifest_sha256",
    "teacher_bundle_schema",
    "row_identity_order_sha256",
    "row_content_order_sha256",
    "split_row_counts",
    "phase_actor_joker_counts",
    "chunk_entry_sha256",
}


def _build_chunk_entry(
    *,
    plan: Mapping[str, Any],
    index: int,
    row_start: int,
    bundle: VerifiedTeacherBundle,
) -> dict[str, Any]:
    manifest_sha = _sha(
        bundle.manifest["manifest_sha256"], label="bundle manifest SHA"
    )
    stem = _chunk_stem(index, manifest_sha)
    split_counts = {name: 0 for name in SPLIT_NAMES}
    stratum_counts = {name: 0 for name in PHASE_ACTOR_JOKER_CELLS}
    identities: list[str] = []
    contents: list[str] = []
    for row in bundle.rows:
        split_counts[row["split_assignment"]["split"]] += 1
        stratum = row["stratum"]
        cell = (
            f"{stratum['phase']}_{stratum['actor']}_"
            f"joker_{stratum['visible_joker_count']}"
        )
        if cell not in stratum_counts:
            raise TeacherCollectionError("chunk row outside phase/actor/Joker contract")
        stratum_counts[cell] += 1
        identities.append(row["row_identity_sha256"])
        contents.append(row["row_sha256"])
    entry: dict[str, Any] = {
        "schema": COLLECTION_CHUNK_ENTRY_SCHEMA,
        "artifact_kind": "completed_content_addressed_v2_teacher_bundle_chunk",
        "algorithm_teacher_only": True,
        "promotion_eligible": False,
        "runtime_allowed": False,
        "serving_changed": False,
        "generation_plan_sha256": plan["generation_plan_sha256"],
        "index": index,
        "row_start": row_start,
        "row_stop_exclusive": row_start + len(bundle.rows),
        "row_count": len(bundle.rows),
        "bundle_relative_path": f"{CHUNKS_DIRECTORY}/{stem}",
        "completion_marker_relative_path": (
            f"{CHUNKS_DIRECTORY}/{_marker_name(stem)}"
        ),
        "bundle_manifest_sha256": manifest_sha,
        "teacher_bundle_schema": TEACHER_BUNDLE_SCHEMA,
        "row_identity_order_sha256": _CANONICAL_SHA256(identities),
        "row_content_order_sha256": _CANONICAL_SHA256(contents),
        "split_row_counts": split_counts,
        "phase_actor_joker_counts": stratum_counts,
    }
    entry["chunk_entry_sha256"] = _SELF_HASH(entry, "chunk_entry_sha256")
    return entry


def _consume_bundle(
    state: _AuditState,
    bundle: VerifiedTeacherBundle,
    entry: Mapping[str, Any],
) -> None:
    for row in bundle.rows:
        identity = _sha(row["row_identity_sha256"], label="row identity")
        content = _sha(row["row_sha256"], label="row content")
        if identity in state.row_identities:
            raise TeacherCollectionError("duplicate row identity across chunks")
        if content in state.row_contents:
            raise TeacherCollectionError("duplicate row content across chunks")
        assignment = row["split_assignment"]
        split = assignment["split"]
        full_deal = assignment["full_deal_commitment_sha256"]
        root_family = assignment["public_root_family_commitment_sha256"]
        information_digest = row["model_input"]["information_digest"]
        values = {
            "full_deal_commitment_sha256": full_deal,
            "public_root_family_commitment_sha256": root_family,
            "information_digest": information_digest,
        }
        prior_deal = state.root_family_deal.setdefault(root_family, full_deal)
        if prior_deal != full_deal:
            raise TeacherCollectionError(
                "public root family maps to multiple full deals across chunks"
            )
        for field_name, commitment in values.items():
            prior_split = state.split_owners[field_name].setdefault(
                commitment, split
            )
            if prior_split != split:
                raise TeacherCollectionError(
                    f"cross-split {field_name} overlap across chunks"
                )
        stratum = row["stratum"]
        cell = (
            f"{stratum['phase']}_{stratum['actor']}_"
            f"joker_{stratum['visible_joker_count']}"
        )
        if cell not in state.stratum_counts:
            raise TeacherCollectionError("row outside phase/actor/Joker contract")
        state.row_identities.add(identity)
        state.row_contents.add(content)
        state.full_deals.add(full_deal)
        state.root_families.add(root_family)
        state.information_digests.add(information_digest)
        state.split_counts[split] += 1
        state.stratum_counts[cell] += 1
        state.row_count += 1
        if row["solver"]["method"] == MCCFR_SOLVER_METHOD:
            seed_plan = row["solver"]["seed_plan"]
            state.solver_seeds.add(seed_plan["solver"]["seed"])
            state.payoff_seeds.add(seed_plan["payoff"]["seed"])
    if state.solver_seeds & state.payoff_seeds:
        raise TeacherCollectionError("solver/payoff seed overlap across chunks")
    state.ordered_prefix_sha256 = _CANONICAL_SHA256(
        {
            "schema": "ofc_t3_t4_collection_ordered_prefix/v1",
            "previous": state.ordered_prefix_sha256,
            "chunk_entry_sha256": entry["chunk_entry_sha256"],
            "bundle_manifest_sha256": entry["bundle_manifest_sha256"],
            "row_identity_order_sha256": entry["row_identity_order_sha256"],
        }
    )


def _safe_bundle_directory(root: Path, relative_path: str) -> Path:
    if not isinstance(relative_path, str) or "\\" in relative_path:
        raise TeacherCollectionError("chunk bundle path must be canonical POSIX")
    relative = Path(relative_path)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.as_posix() != relative_path
        or len(relative.parts) != 2
        or relative.parts[0] != CHUNKS_DIRECTORY
    ):
        raise TeacherCollectionError("unsafe chunk bundle relative path")
    target = root / relative
    if target.is_symlink() or not target.is_dir():
        raise TeacherCollectionError("chunk bundle directory missing or unsafe")
    resolved = target.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise TeacherCollectionError("chunk bundle escapes collection root") from exc
    return resolved


def _verify_chunk_entry(
    *,
    root: Path,
    plan: Mapping[str, Any],
    declared: Mapping[str, Any],
    expected_index: int,
    expected_start: int,
) -> tuple[dict[str, Any], VerifiedTeacherBundle]:
    if not isinstance(declared, Mapping):
        raise TeacherCollectionError("chunk entry must be mapping")
    entry = dict(declared)
    _exact_keys(entry, _CHUNK_ENTRY_KEYS, label="chunk entry")
    if (
        entry.get("schema") != COLLECTION_CHUNK_ENTRY_SCHEMA
        or entry.get("artifact_kind")
        != "completed_content_addressed_v2_teacher_bundle_chunk"
        or entry.get("algorithm_teacher_only") is not True
        or entry.get("promotion_eligible") is not False
        or entry.get("runtime_allowed") is not False
        or entry.get("serving_changed") is not False
        or entry.get("generation_plan_sha256")
        != plan["generation_plan_sha256"]
        or entry.get("teacher_bundle_schema") != TEACHER_BUNDLE_SCHEMA
        or entry.get("index") != expected_index
        or entry.get("row_start") != expected_start
        or entry.get("chunk_entry_sha256")
        != _SELF_HASH(entry, "chunk_entry_sha256")
    ):
        raise TeacherCollectionError("chunk entry fixed binding mismatch")
    manifest_sha = _sha(
        entry.get("bundle_manifest_sha256"), label="chunk bundle manifest SHA"
    )
    stem = _chunk_stem(expected_index, manifest_sha)
    expected_bundle_path = f"{CHUNKS_DIRECTORY}/{stem}"
    expected_marker_path = f"{CHUNKS_DIRECTORY}/{_marker_name(stem)}"
    if (
        entry.get("bundle_relative_path") != expected_bundle_path
        or entry.get("completion_marker_relative_path") != expected_marker_path
    ):
        raise TeacherCollectionError("chunk content-addressed paths mismatch")
    bundle_path = _safe_bundle_directory(root, expected_bundle_path)
    marker_path = root / expected_marker_path
    marker = _read_canonical(marker_path, label="chunk completion marker")
    if marker != entry:
        raise TeacherCollectionError("chunk completion marker/entry mismatch")
    try:
        bundle = _VERIFY_TEACHER_BUNDLE(bundle_path)
    except (TeacherContractError, OSError, TypeError, ValueError) as exc:
        raise TeacherCollectionError(f"v2 teacher chunk verification failed: {exc}") from exc
    if bundle.manifest["manifest_sha256"] != manifest_sha:
        raise TeacherCollectionError("chunk directory name/manifest mismatch")
    if not 1 <= len(bundle.rows) <= plan["chunk_row_limit"]:
        raise TeacherCollectionError("chunk exceeds predeclared bounded row limit")
    rebuilt = _build_chunk_entry(
        plan=plan,
        index=expected_index,
        row_start=expected_start,
        bundle=bundle,
    )
    if entry != rebuilt:
        raise TeacherCollectionError("chunk entry does not match verified bundle")
    return rebuilt, bundle


def _publish_top_manifest(root: Path, manifest: Mapping[str, Any]) -> None:
    sequence = manifest["sequence"]
    digest = manifest["top_manifest_sha256"]
    path = root / MANIFESTS_DIRECTORY / _top_name(sequence, digest)
    _publish_json_no_replace(path, manifest)


def _publish_chunk_locked(
    *,
    root: Path,
    plan: Mapping[str, Any],
    index: int,
    row_start: int,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise TypeError("rows must be a sequence")
    if not 1 <= len(rows) <= plan["chunk_row_limit"]:
        raise TeacherCollectionError("chunk row count outside predeclared bound")
    try:
        verified_rows = [_VERIFY_TEACHER_ROW(row) for row in rows]
    except (TeacherContractError, OSError, TypeError, ValueError) as exc:
        raise TeacherCollectionError(f"teacher row verification failed: {exc}") from exc
    chunks_root = root / CHUNKS_DIRECTORY
    stage = chunks_root / f".stage-{os.getpid()}-{uuid.uuid4().hex}"
    final_path: Path | None = None
    try:
        bundle = _WRITE_TEACHER_BUNDLE(
            stage,
            verified_rows,
            shard_size=plan["bundle_shard_size"],
        )
        manifest_sha = bundle.manifest["manifest_sha256"]
        final_path = chunks_root / _chunk_stem(index, manifest_sha)
        if final_path.exists() or final_path.is_symlink():
            raise TeacherCollectionError("chunk directory collision")
        try:
            os.rename(stage, final_path)
        except OSError as exc:
            raise TeacherCollectionError(
                f"same-filesystem no-replace chunk publication failed: {exc}"
            ) from exc
        published = _VERIFY_TEACHER_BUNDLE(final_path)
        entry = _build_chunk_entry(
            plan=plan,
            index=index,
            row_start=row_start,
            bundle=published,
        )
        marker = root / entry["completion_marker_relative_path"]
        _publish_json_no_replace(marker, entry)
        return entry
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def _cleanup_stages_locked(root: Path) -> None:
    directory = root / CHUNKS_DIRECTORY
    if not directory.exists():
        return
    for path in directory.iterdir():
        if _STAGE_DIRECTORY.fullmatch(path.name):
            if path.is_symlink() or not path.is_dir():
                raise TeacherCollectionError("unsafe interrupted stage artifact")
            resolved = path.resolve(strict=True)
            try:
                resolved.relative_to(directory.resolve(strict=True))
            except ValueError as exc:
                raise TeacherCollectionError("stage directory escapes chunk root") from exc
            shutil.rmtree(resolved)


def _scan_chunk_artifacts(root: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    directory = root / CHUNKS_DIRECTORY
    if directory.is_symlink() or not directory.is_dir():
        raise TeacherCollectionError("chunks directory missing or unsafe")
    bundles: dict[str, Path] = {}
    markers: dict[str, Path] = {}
    for path in directory.iterdir():
        if path.is_symlink():
            raise TeacherCollectionError("chunk symlink forbidden")
        directory_match = _CHUNK_DIRECTORY.fullmatch(path.name)
        marker_match = _CHUNK_MARKER.fullmatch(path.name)
        if directory_match is not None and path.is_dir():
            bundles[path.name] = path
        elif marker_match is not None and path.is_file():
            stem = path.name.removesuffix(".complete.json")
            markers[stem] = path
        else:
            raise TeacherCollectionError("invalid, incomplete, or orphan chunk artifact")
    return bundles, markers


def _verify_root_shape(root: Path, plan: Mapping[str, Any]) -> None:
    allowed = {
        LOCK_NAME,
        _plan_name(plan["generation_plan_sha256"]),
        CHUNKS_DIRECTORY,
        MANIFESTS_DIRECTORY,
    }
    actual = {path.name for path in root.iterdir()}
    if actual != allowed:
        raise TeacherCollectionError(
            f"collection root artifact mismatch: extra={sorted(actual-allowed)}, "
            f"missing={sorted(allowed-actual)}"
        )
    for path in root.iterdir():
        if path.is_symlink():
            raise TeacherCollectionError("collection root symlink artifact forbidden")


def _verify_locked(
    root: Path,
    *,
    expected_generation_plan_sha256: str,
    adopt_orphan: bool,
) -> VerifiedTeacherCollection:
    expected_plan = _sha(
        expected_generation_plan_sha256,
        label="expected_generation_plan_sha256",
    )
    plan = _read_plan(root)
    if plan["generation_plan_sha256"] != expected_plan:
        raise TeacherCollectionError("collection generation plan pin mismatch")
    tops = _read_top_manifests(root, plan)
    state = _AuditState(plan_sha256=expected_plan)
    genesis = _build_top_manifest(
        plan=plan,
        sequence=0,
        previous_sha256=None,
        chunks=(),
        state=state,
        finalized=False,
    )
    if tops[0] != genesis:
        raise TeacherCollectionError("genesis top manifest content mismatch")
    current_entries = tops[-1]["chunks"]
    if len(tops) not in {
        len(current_entries) + 1,
        len(current_entries) + 2,
    }:
        raise TeacherCollectionError("top log length does not match chunk prefix")
    entries: list[dict[str, Any]] = []
    for index, declared in enumerate(current_entries):
        rebuilt, bundle = _verify_chunk_entry(
            root=root,
            plan=plan,
            declared=declared,
            expected_index=index,
            expected_start=state.row_count,
        )
        _consume_bundle(state, bundle, rebuilt)
        entries.append(rebuilt)
        expected_top = _build_top_manifest(
            plan=plan,
            sequence=index + 1,
            previous_sha256=tops[index]["top_manifest_sha256"],
            chunks=entries,
            state=state,
            finalized=False,
        )
        if tops[index + 1] != expected_top:
            raise TeacherCollectionError("streamed chunk prefix/top state mismatch")
    if tops[-1]["finalized"]:
        if len(tops) != len(entries) + 2:
            raise TeacherCollectionError("finalization top sequence mismatch")
        expected_final = _build_top_manifest(
            plan=plan,
            sequence=len(entries) + 1,
            previous_sha256=tops[-2]["top_manifest_sha256"],
            chunks=entries,
            state=state,
            finalized=True,
        )
        if tops[-1] != expected_final:
            raise TeacherCollectionError("finalization top content mismatch")
    elif len(tops) != len(entries) + 1:
        raise TeacherCollectionError("unfinalized top sequence mismatch")
    if state.row_count > plan["expected_total_row_count"]:
        raise TeacherCollectionError("collection exceeds predeclared total rows")

    bundles, markers = _scan_chunk_artifacts(root)
    referenced_stems = {
        Path(entry["bundle_relative_path"]).name for entry in entries
    }
    extra_stems = (set(bundles) | set(markers)) - referenced_stems
    missing_bundles = referenced_stems - set(bundles)
    missing_markers = referenced_stems - set(markers)
    if missing_bundles or missing_markers:
        raise TeacherCollectionError("referenced chunk bundle or marker missing")
    if extra_stems:
        if not adopt_orphan:
            raise TeacherCollectionError("unreferenced complete/incomplete chunk orphan")
        if tops[-1]["finalized"]:
            raise TeacherCollectionError("orphan chunk after finalization")
        if len(extra_stems) != 1:
            raise TeacherCollectionError("multiple next chunk orphans")
        stem = next(iter(extra_stems))
        if stem not in bundles:
            raise TeacherCollectionError("orphan marker has no bundle")
        match = _CHUNK_DIRECTORY.fullmatch(stem)
        if match is None or int(match.group("index")) != len(entries):
            raise TeacherCollectionError("orphan chunk index gap or overlap")
        manifest_sha = match.group("digest")
        try:
            orphan_bundle = _VERIFY_TEACHER_BUNDLE(bundles[stem])
        except (TeacherContractError, OSError, TypeError, ValueError) as exc:
            raise TeacherCollectionError(f"invalid orphan teacher bundle: {exc}") from exc
        if orphan_bundle.manifest["manifest_sha256"] != manifest_sha:
            raise TeacherCollectionError("orphan content-addressed name mismatch")
        orphan_entry = _build_chunk_entry(
            plan=plan,
            index=len(entries),
            row_start=state.row_count,
            bundle=orphan_bundle,
        )
        marker_path = root / orphan_entry["completion_marker_relative_path"]
        if stem in markers:
            marker = _read_canonical(marker_path, label="orphan completion marker")
            if marker != orphan_entry:
                raise TeacherCollectionError("invalid orphan completion marker")
        else:
            _publish_json_no_replace(marker_path, orphan_entry)
        _consume_bundle(state, orphan_bundle, orphan_entry)
        entries.append(orphan_entry)
        adopted_top = _build_top_manifest(
            plan=plan,
            sequence=len(tops),
            previous_sha256=tops[-1]["top_manifest_sha256"],
            chunks=entries,
            state=state,
            finalized=False,
        )
        _publish_top_manifest(root, adopted_top)
        return _verify_locked(
            root,
            expected_generation_plan_sha256=expected_plan,
            adopt_orphan=False,
        )
    _verify_root_shape(root, plan)
    return VerifiedTeacherCollection(
        root=root,
        plan=plan,
        top_manifest=tops[-1],
    )


def initialize_teacher_collection(
    output_dir: str | Path,
    generation_plan: Mapping[str, Any],
) -> VerifiedTeacherCollection:
    """Publish the immutable plan and empty append-only genesis state."""

    root = _prepare_root(output_dir, create=True)
    with _collection_lock(root):
        plan = _verify_plan(generation_plan)
        (root / CHUNKS_DIRECTORY).mkdir(exist_ok=True)
        (root / MANIFESTS_DIRECTORY).mkdir(exist_ok=True)
        plan_path = root / _plan_name(plan["generation_plan_sha256"])
        existing_plans = [
            path for path in root.iterdir() if _PLAN_FILE.fullmatch(path.name)
        ]
        if existing_plans:
            if len(existing_plans) != 1 or _read_plan(root) != plan:
                raise TeacherCollectionError(
                    "generation plan is immutable after initialization"
                )
        else:
            _publish_json_no_replace(plan_path, plan)
        _cleanup_stages_locked(root)
        manifests = list((root / MANIFESTS_DIRECTORY).iterdir())
        if not manifests:
            state = _AuditState(plan_sha256=plan["generation_plan_sha256"])
            genesis = _build_top_manifest(
                plan=plan,
                sequence=0,
                previous_sha256=None,
                chunks=(),
                state=state,
                finalized=False,
            )
            _publish_top_manifest(root, genesis)
        return _verify_locked(
            root,
            expected_generation_plan_sha256=plan["generation_plan_sha256"],
            adopt_orphan=True,
        )


def append_teacher_chunk(
    collection_dir: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_generation_plan_sha256: str,
) -> VerifiedTeacherCollection:
    """Verify and append one bounded v2 bundle after adopting one next orphan."""

    root = _prepare_root(collection_dir, create=False)
    with _collection_lock(root):
        _cleanup_stages_locked(root)
        current = _verify_locked(
            root,
            expected_generation_plan_sha256=expected_generation_plan_sha256,
            adopt_orphan=True,
        )
        if current.finalized:
            raise TeacherCollectionError("cannot append after finalization")
        row_count = len(rows) if isinstance(rows, Sequence) else -1
        if current.top_manifest["statistics"]["row_count"] + row_count > current.plan[
            "expected_total_row_count"
        ]:
            raise TeacherCollectionError("chunk would exceed predeclared total rows")
        entry = _publish_chunk_locked(
            root=root,
            plan=current.plan,
            index=current.top_manifest["chunk_count"],
            row_start=current.top_manifest["statistics"]["row_count"],
            rows=rows,
        )
        rebuilt, bundle = _verify_chunk_entry(
            root=root,
            plan=current.plan,
            declared=entry,
            expected_index=current.top_manifest["chunk_count"],
            expected_start=current.top_manifest["statistics"]["row_count"],
        )
        state = _AuditState(plan_sha256=current.plan["generation_plan_sha256"])
        entries: list[dict[str, Any]] = []
        for index, existing in enumerate(current.top_manifest["chunks"]):
            verified_entry, existing_bundle = _verify_chunk_entry(
                root=root,
                plan=current.plan,
                declared=existing,
                expected_index=index,
                expected_start=state.row_count,
            )
            _consume_bundle(state, existing_bundle, verified_entry)
            entries.append(verified_entry)
        _consume_bundle(state, bundle, rebuilt)
        entries.append(rebuilt)
        next_top = _build_top_manifest(
            plan=current.plan,
            sequence=current.top_manifest["sequence"] + 1,
            previous_sha256=current.top_manifest["top_manifest_sha256"],
            chunks=entries,
            state=state,
            finalized=False,
        )
        _publish_top_manifest(root, next_top)
        return _verify_locked(
            root,
            expected_generation_plan_sha256=expected_generation_plan_sha256,
            adopt_orphan=False,
        )


def resume_teacher_collection(
    collection_dir: str | Path,
    *,
    expected_generation_plan_sha256: str,
) -> VerifiedTeacherCollection:
    """Resume and adopt exactly one verified next orphan, if present."""

    root = _prepare_root(collection_dir, create=False)
    with _collection_lock(root):
        _cleanup_stages_locked(root)
        return _verify_locked(
            root,
            expected_generation_plan_sha256=expected_generation_plan_sha256,
            adopt_orphan=True,
        )


def finalize_teacher_collection(
    collection_dir: str | Path,
    *,
    expected_generation_plan_sha256: str,
) -> VerifiedTeacherCollection:
    """Append the immutable finalization state after every plan gate passes."""

    root = _prepare_root(collection_dir, create=False)
    with _collection_lock(root):
        _cleanup_stages_locked(root)
        current = _verify_locked(
            root,
            expected_generation_plan_sha256=expected_generation_plan_sha256,
            adopt_orphan=True,
        )
        if current.finalized:
            return current
        stats = current.top_manifest["statistics"]
        if stats["row_count"] != current.plan["expected_total_row_count"]:
            raise TeacherCollectionError("final row count does not match plan")
        if any(stats["split_row_counts"][name] <= 0 for name in SPLIT_NAMES):
            raise TeacherCollectionError("fit/dev/test must all be nonempty")
        minima = current.plan["phase_actor_joker_minimum_counts"]
        deficits = {
            cell: minima[cell] - stats["phase_actor_joker_counts"][cell]
            for cell in PHASE_ACTOR_JOKER_CELLS
            if stats["phase_actor_joker_counts"][cell] < minima[cell]
        }
        if deficits:
            raise TeacherCollectionError(f"phase/actor/Joker coverage deficits: {deficits}")
        state = _AuditState(plan_sha256=current.plan["generation_plan_sha256"])
        entries: list[dict[str, Any]] = []
        for index, declared in enumerate(current.top_manifest["chunks"]):
            entry, bundle = _verify_chunk_entry(
                root=root,
                plan=current.plan,
                declared=declared,
                expected_index=index,
                expected_start=state.row_count,
            )
            _consume_bundle(state, bundle, entry)
            entries.append(entry)
        final_top = _build_top_manifest(
            plan=current.plan,
            sequence=current.top_manifest["sequence"] + 1,
            previous_sha256=current.top_manifest["top_manifest_sha256"],
            chunks=entries,
            state=state,
            finalized=True,
        )
        _publish_top_manifest(root, final_top)
        return _verify_locked(
            root,
            expected_generation_plan_sha256=expected_generation_plan_sha256,
            adopt_orphan=False,
        )


def verify_teacher_collection(
    collection_dir: str | Path,
    *,
    expected_generation_plan_sha256: str,
    require_finalized: bool = False,
) -> VerifiedTeacherCollection:
    """Freshly stream and authenticate every chunk without retaining all rows."""

    if not isinstance(require_finalized, bool):
        raise TypeError("require_finalized must be bool")
    root = _prepare_root(collection_dir, create=False)
    with _collection_lock(root):
        verified = _verify_locked(
            root,
            expected_generation_plan_sha256=expected_generation_plan_sha256,
            adopt_orphan=False,
        )
        if require_finalized and not verified.finalized:
            raise TeacherCollectionError("finalized collection required")
        return verified


__all__ = [
    "COLLECTION_CHUNK_ENTRY_SCHEMA",
    "COLLECTION_PLAN_SCHEMA",
    "COLLECTION_RUNTIME_SCHEMA",
    "COLLECTION_TOP_SCHEMA",
    "PHASE_ACTOR_JOKER_CELLS",
    "TeacherCollectionError",
    "TeacherCollectionLockError",
    "VerifiedTeacherCollection",
    "append_teacher_chunk",
    "build_generation_plan",
    "finalize_teacher_collection",
    "initialize_teacher_collection",
    "resume_teacher_collection",
    "verify_teacher_collection",
]
