"""Fail-closed local/cloud transport boundary for M3.1 fresh-quality jobs.

This module does not launch cloud resources and cannot change an AI profile.
It provides the immutable layer immediately below a future GCE adapter:

* replay and safely extract the deterministic fresh-quality input archive;
* bind one worker to exactly one of the fifteen sealed job manifests;
* execute the accepted T3 search semantics at 8/32/4/0, with a separate
  8/128/4/0 confirmation pass only for confirmation jobs;
* validate every decision with the existing ActionKey/counter-RNG oracle;
* retain completed per-root tasks across interruption and publish ``DONE``
  only after the complete gate-compatible result validates;
* build a content-addressed runtime source archive and a non-authorizing
  local launch manifest with quota-friendly 8+7 waves.

Teacher Q values remain diagnostic search estimates.  Nothing here treats
them as realized match EV, authorizes training, starts Spot VMs, or resolves
``current``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import time
import zipfile
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

from .hu_infoset import ActorObservation
from .hu_m31_t3_runtime import HuM31T3RuntimeConfig, HuM31T3SearchSolver
from .validate_hu_m31_t3_profile import process_memory_snapshot
from . import hu_m31_t3_step6d_fresh_quality_gate_v1 as gate
from . import hu_m31_t3_step6d_fresh_quality_v1 as quality
from . import run_hu_m31_t3_step6c_shard as step6c
from . import run_hu_m31_t3_step6d_performance as step6d_v1


TRANSPORT_PACKAGE_SCHEMA = "hu_m31_t3_step6d_fresh_quality_transport_package_v1"
TASK_SCHEMA = "hu_m31_t3_step6d_fresh_quality_transport_task_v1"
DONE_SCHEMA = "hu_m31_t3_step6d_fresh_quality_transport_done_v1"
SOURCE_MANIFEST_SCHEMA = "hu_m31_t3_step6d_fresh_quality_runtime_source_v1"
LAUNCH_MANIFEST_SCHEMA = "hu_m31_t3_step6d_fresh_quality_local_launch_v1"

SOURCE_ARCHIVE_NAME = "fresh_quality_runtime_source_v1.zip"
SOURCE_MANIFEST_NAME = "fresh_quality_runtime_source_v1.json"
DEFAULT_CANDIDATE_ARCHIVE_PATH = (
    "native/candidate/release/libofc_hu_m3_engine.so"
)
DEFAULT_FEATURE_ARCHIVE_PATH = (
    "target/release/libofc_stage3_feature_encoder.so"
)
WAVE_JOB_COUNTS = (8, 7)
WAVE_JOB_IDS = (
    tuple(f"primary-{index:02d}" for index in range(8)),
    (
        "primary-08",
        "primary-09",
        *(f"confirmation-{index:02d}" for index in range(5)),
    ),
)

_SHA = re.compile(r"^[0-9a-f]{64}$")
_SAFE_RUN = re.compile(r"^[a-z0-9][a-z0-9-]{2,62}[a-z0-9]$")
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_REQUIRED_SOURCE_PATHS = frozenset(
    {
        "pyproject.toml",
        "configs/hu_joint_policy_m31_t3_step6d_contract.json",
        "src/ofc_regular/__init__.py",
        "src/ofc_regular/hu_m31_t3_step6d_fresh_quality_v1.py",
        "src/ofc_regular/hu_m31_t3_step6d_fresh_quality_gate_v1.py",
        "src/ofc_regular/hu_m31_t3_step6d_fresh_quality_transport_v1.py",
        "src/ofc_regular/run_hu_m31_t3_step6c_shard.py",
        DEFAULT_CANDIDATE_ARCHIVE_PATH,
        DEFAULT_FEATURE_ARCHIVE_PATH,
    }
)

_TASK_KEYS = frozenset(
    {
        "schema",
        "job_id",
        "phase",
        "pair_index",
        "root_index",
        "root_path",
        "root_sha256",
        "row",
        "candidate_library_sha256",
        "teacher_value_status",
        "teacher_values_are_realized_match_ev",
        "opponent_private_discards_used",
        "training_eligible",
        "promotion_evidence",
        "current_profile_changed",
    }
)
_DONE_KEYS = frozenset(
    {
        "schema",
        "status",
        "job_id",
        "phase",
        "package_archive_sha256",
        "package_manifest_sha256",
        "plan_sha256",
        "root_seal_sha256",
        "job_manifest_sha256",
        "candidate_library_sha256",
        "result_path",
        "result_sha256",
        "result_bytes",
        "task_records",
        "task_record_aggregate_sha256",
        "completed_root_count",
        "done_published_last",
        "cloud_execution_started",
        "training_eligible",
        "promotion_evidence",
        "current_profile_changed",
    }
)
_FILE_RECORD_KEYS = frozenset({"path", "sha256", "bytes"})
_SOURCE_ENTRY_KEYS = frozenset({"sha256", "bytes"})
_SOURCE_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "status",
        "archive",
        "entries",
        "entry_count",
        "entry_aggregate_sha256",
        "candidate_library",
        "feature_encoder",
        "profile_registry_sha256",
        "content_addressed",
        "cloud_execution_started",
        "training_eligible",
        "current_profile_changed",
    }
)
_LAUNCH_JOB_KEYS = frozenset(
    {
        "job_id",
        "phase",
        "wave_index",
        "package_job_path",
        "job_manifest_sha256",
        "result_path",
        "output_prefix",
        "rayon_threads",
        "processes",
        "create_only",
    }
)
_LAUNCH_KEYS = frozenset(
    {
        "schema",
        "status",
        "run_name",
        "quality_package",
        "runtime_source",
        "runtime_source_manifest",
        "wheelhouse",
        "wheelhouse_manifest",
        "startup",
        "candidate_library",
        "feature_encoder",
        "allocation",
        "wave_job_counts",
        "waves",
        "jobs",
        "job_count",
        "package_plan_sha256",
        "package_root_seal_sha256",
        "profile_registry_sha256",
        "transport_ready",
        "cloud_launch_authorized",
        "cloud_execution_started",
        "training_eligible",
        "promotion_evidence",
        "current_profile_changed",
    }
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


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise ValueError(f"{label} fields changed: missing={missing}, extra={extra}")


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA.fullmatch(value) is None:
        raise ValueError(f"{label} is not a lowercase SHA-256")
    return value


def _safe_file(path: str | Path, label: str) -> Path:
    value = Path(path).resolve()
    if Path(path).is_symlink() or not value.is_file():
        raise ValueError(f"{label} is missing or unsafe")
    return value


def _safe_directory(path: str | Path, label: str) -> Path:
    value = Path(path).resolve()
    if Path(path).is_symlink() or not value.is_dir():
        raise ValueError(f"{label} is missing or unsafe")
    return value


def _read_canonical(path: str | Path, label: str) -> dict[str, Any]:
    source = _safe_file(path, label)
    raw = source.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not canonical JSON") from exc
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise ValueError(f"{label} is not a canonical object")
    return value


def _write_once(path: str | Path, value: Mapping[str, Any]) -> Path:
    destination = Path(path)
    raw = canonical_bytes(value)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        if destination.is_symlink() or not destination.is_file():
            raise ValueError(f"immutable output is unsafe: {destination}")
        if destination.read_bytes() != raw:
            raise FileExistsError(f"immutable output changed: {destination}") from None
    return destination


def _safe_archive_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != name
        or "\\" in name
    ):
        raise ValueError("archive member path is unsafe")
    return path


def _regular_zip_member(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0xFFFF
    # Archives created on Windows may omit the file type; directories remain
    # forbidden and the path inventory is exact.
    file_type = mode & 0o170000
    return not info.is_dir() and file_type in (0, 0o100000)


def _archive_payloads(
    archive_path: str | Path,
    *,
    expected_sha256: str | None = None,
    require_deterministic_timestamp: bool = True,
) -> dict[str, bytes]:
    source = _safe_file(archive_path, "transport archive")
    if expected_sha256 is not None and sha256_file(source) != _require_sha(
        expected_sha256, "transport archive digest"
    ):
        raise ValueError("transport archive digest changed")
    payloads: dict[str, bytes] = {}
    with zipfile.ZipFile(source, "r") as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if names != sorted(names) or len(names) != len(set(names)):
            raise ValueError("transport archive ordering/uniqueness changed")
        for info in infos:
            path = _safe_archive_name(info.filename)
            if not _regular_zip_member(info):
                raise ValueError("transport archive contains a non-regular member")
            if (
                require_deterministic_timestamp
                and info.date_time != (1980, 1, 1, 0, 0, 0)
            ):
                raise ValueError("transport archive timestamp changed")
            payloads[path.as_posix()] = archive.read(info)
    return payloads


def _atomic_extract_payloads(
    payloads: Mapping[str, bytes], destination: str | Path
) -> Path:
    target = Path(destination).resolve()
    if target.exists():
        root = _safe_directory(target, "existing extracted transport package")
        actual = {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*")
            if path.is_file() and not path.is_symlink()
        }
        if actual != dict(payloads):
            raise ValueError("resume extraction is not byte-identical")
        if any(path.is_symlink() for path in root.rglob("*")):
            raise ValueError("resume extraction contains a symlink")
        return root
    parent = target.parent.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    temporary = parent / f".{target.name}.extract-{os.getpid()}"
    if temporary.exists():
        raise FileExistsError("transport extraction temporary path already exists")
    temporary.mkdir()
    try:
        for relative, raw in sorted(payloads.items()):
            pure = _safe_archive_name(relative)
            output = temporary.joinpath(*pure.parts)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("xb") as stream:
                stream.write(raw)
        os.replace(temporary, target)
    except BaseException:
        # Preserve an unexpected partial tree for forensic inspection.
        raise
    return target


def _portable_package_validation(package_root: Path) -> dict[str, Any]:
    root = _safe_directory(package_root, "extracted fresh-quality package")
    manifest = _read_canonical(root / "PACKAGE_MANIFEST.json", "package manifest")
    ready = _read_canonical(root / "PACKAGE_READY.json", "package-ready receipt")
    _exact_keys(manifest, quality._PACKAGE_MANIFEST_KEYS, "package manifest")
    _exact_keys(ready, quality._PACKAGE_READY_KEYS, "package-ready receipt")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != 73:
        raise ValueError("fresh-quality package entry grid changed")
    expected_paths: set[str] = set()
    for record in entries:
        if not isinstance(record, Mapping):
            raise ValueError("fresh-quality package entry is not an object")
        _exact_keys(record, quality._PACKAGE_ENTRY_KEYS, "package entry")
        relative = _safe_archive_name(str(record["path"])).as_posix()
        if relative in expected_paths:
            raise ValueError("fresh-quality package entry is duplicated")
        path = _safe_file(root.joinpath(*PurePosixPath(relative).parts), "package entry")
        if (
            sha256_file(path) != _require_sha(record["sha256"], "package entry")
            or path.stat().st_size != record["bytes"]
        ):
            raise ValueError("fresh-quality package entry hash/size changed")
        expected_paths.add(relative)
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if actual_paths != expected_paths | {
        "PACKAGE_MANIFEST.json",
        "PACKAGE_READY.json",
    }:
        raise ValueError("fresh-quality package file inventory changed")

    plan = quality.validate_fresh_quality_plan(
        _read_canonical(root / "control/plan.json", "packaged quality plan")
    )
    materialization = _read_canonical(
        root / "control/materialization.json", "packaged materialization"
    )
    seal = _read_canonical(root / "control/root_seal.json", "packaged root seal")
    _exact_keys(
        materialization,
        quality._MATERIALIZATION_KEYS,
        "packaged materialization",
    )
    _exact_keys(seal, quality._SEAL_KEYS, "packaged root seal")

    root_records: list[dict[str, Any]] = []
    fingerprints: list[str] = []
    for row in quality.schedule_rows():
        relative = quality._root_relative_path(row["phase"], row["pair_index"])
        path = _safe_file(
            root.joinpath(*PurePosixPath(relative).parts),
            f"quality root {relative}",
        )
        value = quality.validate_root(
            _read_canonical(path, f"quality root {relative}"), plan=plan
        )
        fingerprints.extend(
            str(item["observation_fingerprint"]) for item in value["observations"]
        )
        root_records.append(
            {
                "phase": row["phase"],
                "pair_index": row["pair_index"],
                "path": relative,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "root_indices": row["root_indices"],
            }
        )
    if len(fingerprints) != 110 or len(set(fingerprints)) != 110:
        raise ValueError("fresh-quality observation fingerprints changed")
    expected_materialization = {
        "schema": quality.MATERIALIZATION_SCHEMA,
        "status": "exact_55_paired_110_roots_materialized",
        "plan_sha256": quality.canonical_sha256(plan),
        # This is source provenance, not an extraction target.  The package
        # hash and all sealed root bytes are independently replayed above.
        "root_directory": materialization.get("root_directory"),
        "root_count": 110,
        "paired_hand_count": 55,
        "phase_counts": {
            "primary_paired_hands": 50,
            "primary_roots": 100,
            "confirmation_paired_hands": 5,
            "confirmation_roots": 10,
        },
        "root_records": root_records,
        "root_record_aggregate_sha256": quality.canonical_sha256(root_records),
        "observation_count": 110,
        "unique_observation_fingerprint_count": 110,
        "observation_fingerprint_aggregate_sha256": quality.canonical_sha256(
            fingerprints
        ),
        "hidden_information_field_count": 0,
        "unknown_field_count": 0,
        "missing_root_count": 0,
        "current_profile_changed": False,
    }
    if (
        not isinstance(materialization.get("root_directory"), str)
        or not Path(materialization["root_directory"]).is_absolute()
        or materialization != expected_materialization
    ):
        raise ValueError("packaged materialization differs from portable replay")
    expected_seal = {
        "schema": quality.ROOT_SEAL_SCHEMA,
        "status": "sealed_create_only_fresh_quality_roots",
        "plan_sha256": quality.canonical_sha256(plan),
        "materialization_sha256": quality.canonical_sha256(materialization),
        "root_record_aggregate_sha256": materialization[
            "root_record_aggregate_sha256"
        ],
        "observation_fingerprint_aggregate_sha256": materialization[
            "observation_fingerprint_aggregate_sha256"
        ],
        "root_count": 110,
        "observation_count": 110,
        "root_records": deepcopy(root_records),
        "quality_execution_authorized": True,
        "cloud_execution_started": False,
        "training_eligible": False,
        "current_profile_changed": False,
    }
    if seal != expected_seal:
        raise ValueError("packaged root seal differs from portable replay")
    # `materialization["root_directory"]` is immutable source provenance.  It
    # deliberately names the machine on which the 110 roots were generated,
    # and therefore must not be dereferenced by a portable worker.  Every root
    # byte, observation fingerprint, materialization field, and seal field has
    # already been replayed from the extracted package above.  Validate the
    # frozen job descriptors directly against that replayed seal instead of
    # calling `build_job_descriptors`, whose source-side validator reopens the
    # original absolute directory.
    job_ids = [
        *(f"primary-{index:02d}" for index in range(10)),
        *(f"confirmation-{index:02d}" for index in range(5)),
    ]
    jobs: list[dict[str, Any]] = []
    for job_id in job_ids:
        stored = _read_canonical(
            root / "jobs" / f"{job_id}.json",
            f"quality job {job_id}",
        )
        validated_job = quality.validate_job_descriptor(
            stored,
            plan=plan,
            seal=seal,
        )
        if validated_job["job_id"] != job_id:
            raise ValueError("packaged fresh-quality job descriptor changed")
        jobs.append(validated_job)
    if (
        manifest["schema"] != quality.PACKAGE_SCHEMA
        or manifest["status"] != "immutable_fresh_quality_inputs_ready"
        or manifest["plan_sha256"] != quality.canonical_sha256(plan)
        or manifest["materialization_sha256"]
        != quality.canonical_sha256(materialization)
        or manifest["root_seal_sha256"] != quality.canonical_sha256(seal)
        or manifest["job_count"] != 15
        or manifest["primary_job_count"] != 10
        or manifest["confirmation_job_count"] != 5
        or manifest["root_file_count"] != 55
        or manifest["entry_aggregate_sha256"] != quality.canonical_sha256(entries)
        or any(
            manifest[field] is not False
            for field in (
                "cloud_execution_started",
                "training_eligible",
                "current_profile_changed",
            )
        )
    ):
        raise ValueError("fresh-quality package manifest boundary changed")
    expected_ready = {
        "schema": quality.PACKAGE_READY_SCHEMA,
        "status": "ready_for_separate_execution_authorization",
        "package_manifest_sha256": quality.canonical_sha256(manifest),
        "package_file_sha256": sha256_file(root / "PACKAGE_MANIFEST.json"),
        "job_count": 15,
        "cloud_execution_started": False,
        "training_eligible": False,
        "current_profile_changed": False,
    }
    if ready != expected_ready:
        raise ValueError("fresh-quality package-ready receipt changed")
    return {
        "manifest": manifest,
        "plan": plan,
        "materialization": materialization,
        "seal": seal,
        "jobs": jobs,
        "package_manifest_sha256": quality.canonical_sha256(manifest),
    }


def extract_and_validate_package(
    *,
    archive_path: str | Path,
    expected_archive_sha256: str,
    extraction_directory: str | Path,
) -> dict[str, Any]:
    payloads = _archive_payloads(
        archive_path, expected_sha256=expected_archive_sha256
    )
    root = _atomic_extract_payloads(payloads, extraction_directory)
    validated = _portable_package_validation(root)
    validated["package_root"] = str(root)
    validated["package_archive_sha256"] = expected_archive_sha256
    return validated


def validate_selected_job(
    *,
    package: Mapping[str, Any],
    job_id: str,
    expected_job_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    if not isinstance(job_id, str):
        raise ValueError("job_id must be a string")
    jobs = {str(job["job_id"]): job for job in package["jobs"]}
    if set(jobs) != {
        *(f"primary-{index:02d}" for index in range(10)),
        *(f"confirmation-{index:02d}" for index in range(5)),
    }:
        raise ValueError("fresh-quality package job coverage changed")
    selected = jobs.get(job_id)
    if selected is None:
        raise ValueError("selected job is outside the frozen 15-job grid")
    path = (
        Path(str(package["package_root"])).resolve()
        / "jobs"
        / f"{job_id}.json"
    )
    raw_sha = sha256_file(path)
    if (
        expected_job_manifest_sha256 is not None
        and raw_sha
        != _require_sha(expected_job_manifest_sha256, "selected job manifest")
    ):
        raise ValueError("selected job manifest digest changed")
    stored = _read_canonical(path, f"selected job {job_id}")
    if stored != selected:
        raise ValueError("selected job manifest bytes changed")
    return deepcopy(selected)


def _root_lookup(
    package_root: Path, job: Mapping[str, Any], plan: Mapping[str, Any]
) -> tuple[
    dict[int, tuple[dict[str, Any], ActorObservation]],
    list[dict[str, Any]],
]:
    lookup: dict[int, tuple[dict[str, Any], ActorObservation]] = {}
    artifacts: list[dict[str, Any]] = []
    for relative in job["root_paths"]:
        path = _safe_file(
            package_root.joinpath(*PurePosixPath(relative).parts),
            f"selected root {relative}",
        )
        root = quality.validate_root(
            _read_canonical(path, f"selected root {relative}"), plan=plan
        )
        if root["phase"] != job["phase"] or root["pair_index"] not in job["pair_indices"]:
            raise ValueError("selected root escaped the job")
        artifacts.append(
            {
                "phase": root["phase"],
                "pair_index": root["pair_index"],
                "path": relative,
                "sha256": sha256_file(path),
                "root_indices": root["root_indices"],
            }
        )
        for item in root["observations"]:
            observation = ActorObservation.from_dict(item["observation"])
            if (
                observation.fingerprint() != item["observation_fingerprint"]
                or quality.canonical_sha256(item["observation"])
                != item["observation_sha256"]
            ):
                raise ValueError("selected observation identity changed")
            lookup[int(item["root_index"])] = (root, observation)
    expected = [
        index
        for relative in job["root_paths"]
        for index in next(
            record["root_indices"]
            for record in package_root_validation_records(package_root)
            if record["path"] == relative
        )
    ]
    if list(lookup) != expected or len(lookup) != 2 * len(job["pair_indices"]):
        raise ValueError("selected job root ordering changed")
    return lookup, artifacts


def package_root_validation_records(package_root: Path) -> list[dict[str, Any]]:
    seal = _read_canonical(
        package_root / "control/root_seal.json", "packaged root seal"
    )
    records = seal.get("root_records")
    if not isinstance(records, list):
        raise ValueError("packaged root records are missing")
    return records


def _solve_row(
    *,
    root: Mapping[str, Any],
    observation_record: Mapping[str, Any],
    observation: ActorObservation,
    phase: str,
    library_path: Path,
    library_sha256: str,
) -> dict[str, Any]:
    seeds = dict(root["seeds"])
    primary_solver = HuM31T3SearchSolver(
        HuM31T3RuntimeConfig(
            expected_library_sha256=library_sha256,
            library_path=library_path,
            run_id=quality.ENGINE_RUN_ID,
            candidate_samples=8,
            evaluation_samples=32,
            downstream_t3_samples=4,
            seed=seeds["child"],
            candidate_seed=seeds["candidate"],
            evaluation_seed=seeds["evaluation"],
        )
    )
    primary_started = time.perf_counter()
    primary = primary_solver.solve(observation).to_dict()
    primary_seconds = time.perf_counter() - primary_started
    step6c._validate_decision_payload(
        primary,
        observation=observation,
        seeds=seeds,
        budget=step6c._PRIMARY_BUDGET,
        evaluation_seed_key="evaluation",
        native_library_sha256=library_sha256,
    )
    confirmation_payload = None
    confirmation_seconds = None
    if phase == quality.CONFIRMATION_PHASE:
        confirmation_solver = HuM31T3SearchSolver(
            HuM31T3RuntimeConfig(
                expected_library_sha256=library_sha256,
                library_path=library_path,
                run_id=quality.ENGINE_RUN_ID,
                candidate_samples=8,
                evaluation_samples=128,
                downstream_t3_samples=4,
                seed=seeds["child"],
                candidate_seed=seeds["candidate"],
                evaluation_seed=seeds["confirmation"],
            )
        )
        confirmation_started = time.perf_counter()
        confirmation = confirmation_solver.solve(observation).to_dict()
        confirmation_seconds = time.perf_counter() - confirmation_started
        step6c._validate_decision_payload(
            confirmation,
            observation=observation,
            seeds=seeds,
            budget=step6c._CONFIRMATION_BUDGET,
            evaluation_seed_key="confirmation",
            native_library_sha256=library_sha256,
        )
        confirmation_payload = step6c._confirmation_payload(primary, confirmation)
        step6c._validate_confirmation_pair(
            primary, confirmation, confirmation_payload
        )
    peak = int(process_memory_snapshot()["peak_rss_bytes"])
    if peak <= 0:
        raise ValueError("fresh-quality worker peak RSS is invalid")
    row = {
        "phase": root["phase"],
        "pair_index": root["pair_index"],
        "root_index": observation_record["root_index"],
        "seat": observation.seat,
        "observation_fingerprint": observation.fingerprint(),
        "observation_sha256": observation_record["observation_sha256"],
        "primary_wall_seconds": primary_seconds,
        "primary_decision": primary,
        "confirmation_wall_seconds": confirmation_seconds,
        "confirmation": confirmation_payload,
        "peak_rss_bytes": peak,
    }
    step6d_v1._reject_hidden(row, "fresh_quality_transport_row")
    return row


SolveRow = Callable[
    [Mapping[str, Any], Mapping[str, Any], ActorObservation, str, Path, str],
    dict[str, Any],
]


def _default_solve_adapter(
    root: Mapping[str, Any],
    observation_record: Mapping[str, Any],
    observation: ActorObservation,
    phase: str,
    library_path: Path,
    library_sha256: str,
) -> dict[str, Any]:
    return _solve_row(
        root=root,
        observation_record=observation_record,
        observation=observation,
        phase=phase,
        library_path=library_path,
        library_sha256=library_sha256,
    )


def _validate_task(
    value: Mapping[str, Any],
    *,
    job: Mapping[str, Any],
    lookup: Mapping[int, tuple[dict[str, Any], ActorObservation]],
    library_sha256: str,
) -> dict[str, Any]:
    task = deepcopy(dict(value))
    _exact_keys(task, _TASK_KEYS, "fresh-quality transport task")
    root_index = task.get("root_index")
    if isinstance(root_index, bool) or not isinstance(root_index, int):
        raise ValueError("fresh-quality transport task root index changed")
    source = lookup.get(root_index)
    if source is None:
        raise ValueError("fresh-quality transport task escaped selected roots")
    root, _observation = source
    expected_path = quality._root_relative_path(root["phase"], root["pair_index"])
    if (
        task["schema"] != TASK_SCHEMA
        or task["job_id"] != job["job_id"]
        or task["phase"] != job["phase"]
        or task["pair_index"] != root["pair_index"]
        or task["root_path"] != expected_path
        or task["root_sha256"]
        != hashlib.sha256(quality.canonical_bytes(root)).hexdigest()
        or task["candidate_library_sha256"] != library_sha256
        or task["teacher_value_status"] != "diagnostic_not_match_EV"
        or task["teacher_values_are_realized_match_ev"] is not False
        or task["opponent_private_discards_used"] is not False
        or any(
            task[field] is not False
            for field in (
                "training_eligible",
                "promotion_evidence",
                "current_profile_changed",
            )
        )
    ):
        raise ValueError("fresh-quality transport task provenance changed")
    gate._validate_result_row(task["row"], job=job, root_lookup=lookup)
    step6d_v1._reject_hidden(task, "fresh_quality_transport_task")
    return task


def _task_record(path: Path, root_index: int, output_root: Path) -> dict[str, Any]:
    return {
        "root_index": root_index,
        "path": path.relative_to(output_root).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _build_result(
    *,
    job: Mapping[str, Any],
    package: Mapping[str, Any],
    root_artifacts: Sequence[Mapping[str, Any]],
    tasks: Sequence[Mapping[str, Any]],
    library_sha256: str,
) -> dict[str, Any]:
    rows = [deepcopy(task["row"]) for task in tasks]
    return {
        "schema": gate.RESULT_SCHEMA,
        "status": "complete_create_only_quality_job",
        "job": deepcopy(dict(job)),
        "plan_sha256": quality.canonical_sha256(package["plan"]),
        "root_seal_sha256": quality.canonical_sha256(package["seal"]),
        "candidate_library_sha256": library_sha256,
        "root_artifacts": [deepcopy(dict(row)) for row in root_artifacts],
        "rows": rows,
        "peak_rss_bytes": max(int(row["peak_rss_bytes"]) for row in rows),
        "teacher_value_status": "diagnostic_not_match_EV",
        "teacher_values_are_realized_match_ev": False,
        "opponent_private_discards_used": False,
        "training_eligible": False,
        "promotion_evidence": False,
        "current_profile_changed": False,
    }


def _validate_done(
    value: Mapping[str, Any],
    *,
    output_root: Path,
    package: Mapping[str, Any],
    job: Mapping[str, Any],
    library_sha256: str,
    lookup: Mapping[int, tuple[dict[str, Any], ActorObservation]],
) -> dict[str, Any]:
    done = deepcopy(dict(value))
    _exact_keys(done, _DONE_KEYS, "fresh-quality transport DONE")
    result_path = output_root.joinpath(*PurePosixPath(done["result_path"]).parts)
    result = _read_canonical(result_path, "fresh-quality completed result")
    validated_result, _evidence = gate.validate_job_result(
        result,
        job=job,
        plan=package["plan"],
        seal=package["seal"],
        root_lookup=lookup,
    )
    records = done.get("task_records")
    if not isinstance(records, list):
        raise ValueError("fresh-quality DONE task records are missing")
    expected_records: list[dict[str, Any]] = []
    for root_index in lookup:
        path = output_root / "tasks" / f"root_{root_index:03d}.json"
        task = _validate_task(
            _read_canonical(path, f"completed task {root_index}"),
            job=job,
            lookup=lookup,
            library_sha256=library_sha256,
        )
        if int(task["root_index"]) != root_index:
            raise ValueError("fresh-quality completed task ordering changed")
        expected_records.append(_task_record(path, root_index, output_root))
    expected = {
        "schema": DONE_SCHEMA,
        "status": "complete_validated_quality_job",
        "job_id": job["job_id"],
        "phase": job["phase"],
        "package_archive_sha256": package["package_archive_sha256"],
        "package_manifest_sha256": package["package_manifest_sha256"],
        "plan_sha256": quality.canonical_sha256(package["plan"]),
        "root_seal_sha256": quality.canonical_sha256(package["seal"]),
        "job_manifest_sha256": sha256_file(
            Path(package["package_root"]) / "jobs" / f"{job['job_id']}.json"
        ),
        "candidate_library_sha256": library_sha256,
        "result_path": job["result_path"],
        "result_sha256": sha256_file(result_path),
        "result_bytes": result_path.stat().st_size,
        "task_records": expected_records,
        "task_record_aggregate_sha256": canonical_sha256(expected_records),
        "completed_root_count": len(lookup),
        "done_published_last": True,
        "cloud_execution_started": False,
        "training_eligible": False,
        "promotion_evidence": False,
        "current_profile_changed": False,
    }
    if result != validated_result or done != expected:
        raise ValueError("fresh-quality transport DONE or bound artifacts changed")
    done_mtime = (output_root / "DONE.json").stat().st_mtime_ns
    bound_paths = [result_path] + [
        output_root.joinpath(*PurePosixPath(row["path"]).parts)
        for row in expected_records
    ]
    if any(done_mtime < path.stat().st_mtime_ns for path in bound_paths):
        raise ValueError("fresh-quality DONE was not published last")
    return done


def run_job(
    *,
    package_archive: str | Path,
    package_archive_sha256: str,
    job_id: str,
    expected_job_manifest_sha256: str,
    library_path: str | Path,
    output_directory: str | Path,
    stop_after_roots: int | None = None,
    solve_row: SolveRow | None = None,
) -> dict[str, Any]:
    """Run or resume one immutable quality job and publish ``DONE`` last."""

    output_root = Path(output_directory).resolve()
    if output_root.is_symlink():
        raise ValueError("fresh-quality output directory is unsafe")
    output_root.mkdir(parents=True, exist_ok=True)
    package = extract_and_validate_package(
        archive_path=package_archive,
        expected_archive_sha256=package_archive_sha256,
        extraction_directory=output_root / "package",
    )
    job = validate_selected_job(
        package=package,
        job_id=job_id,
        expected_job_manifest_sha256=expected_job_manifest_sha256,
    )
    library = _safe_file(library_path, "accepted candidate library")
    library_sha = sha256_file(library)
    if (
        library_sha != quality.ACCEPTED_CANDIDATE_LIBRARY_SHA256
        or library_sha != job["candidate_library_sha256"]
    ):
        raise ValueError("fresh-quality accepted candidate library changed")
    os.environ["RAYON_NUM_THREADS"] = "16"
    os.environ["OFC_HU_M3_BATCH_THREADS"] = "1"
    lookup, root_artifacts = _root_lookup(
        Path(package["package_root"]), job, package["plan"]
    )
    done_path = output_root / "DONE.json"
    if done_path.exists():
        return _validate_done(
            _read_canonical(done_path, "fresh-quality DONE"),
            output_root=output_root,
            package=package,
            job=job,
            library_sha256=library_sha,
            lookup=lookup,
        )
    result_path = output_root.joinpath(*PurePosixPath(job["result_path"]).parts)
    if result_path.exists():
        raise ValueError("fresh-quality result exists without DONE")
    task_dir = output_root / "tasks"
    task_dir.mkdir(exist_ok=True)
    unexpected = {
        path.name
        for path in task_dir.iterdir()
        if path.is_symlink()
        or not path.is_file()
        or re.fullmatch(r"root_\d{3}\.json", path.name) is None
    }
    if unexpected:
        raise ValueError("fresh-quality task directory contains unknown files")

    solver = solve_row or _default_solve_adapter
    completed: dict[int, dict[str, Any]] = {}
    for root_index, (root, observation) in lookup.items():
        path = task_dir / f"root_{root_index:03d}.json"
        if path.exists():
            completed[root_index] = _validate_task(
                _read_canonical(path, f"resumed task {root_index}"),
                job=job,
                lookup=lookup,
                library_sha256=library_sha,
            )
            continue
        if stop_after_roots is not None and len(completed) >= stop_after_roots:
            break
        observation_record = next(
            item
            for item in root["observations"]
            if int(item["root_index"]) == root_index
        )
        row = solver(
            root,
            observation_record,
            observation,
            str(job["phase"]),
            library,
            library_sha,
        )
        task = {
            "schema": TASK_SCHEMA,
            "job_id": job["job_id"],
            "phase": job["phase"],
            "pair_index": root["pair_index"],
            "root_index": root_index,
            "root_path": quality._root_relative_path(
                root["phase"], root["pair_index"]
            ),
            "root_sha256": sha256_file(
                Path(package["package_root"])
                / quality._root_relative_path(root["phase"], root["pair_index"])
            ),
            "row": row,
            "candidate_library_sha256": library_sha,
            "teacher_value_status": "diagnostic_not_match_EV",
            "teacher_values_are_realized_match_ev": False,
            "opponent_private_discards_used": False,
            "training_eligible": False,
            "promotion_evidence": False,
            "current_profile_changed": False,
        }
        _validate_task(
            task, job=job, lookup=lookup, library_sha256=library_sha
        )
        _write_once(path, task)
        completed[root_index] = task
    if len(completed) != len(lookup):
        return {
            "schema": DONE_SCHEMA,
            "status": "interrupted_for_resume",
            "job_id": job["job_id"],
            "completed_root_count": len(completed),
            "pending_root_count": len(lookup) - len(completed),
            "done_published": False,
            "current_profile_changed": False,
        }
    tasks = [completed[index] for index in lookup]
    result = _build_result(
        job=job,
        package=package,
        root_artifacts=root_artifacts,
        tasks=tasks,
        library_sha256=library_sha,
    )
    gate.validate_job_result(
        result,
        job=job,
        plan=package["plan"],
        seal=package["seal"],
        root_lookup=lookup,
    )
    _write_once(result_path, result)
    task_records = [
        _task_record(
            task_dir / f"root_{index:03d}.json", index, output_root
        )
        for index in lookup
    ]
    done = {
        "schema": DONE_SCHEMA,
        "status": "complete_validated_quality_job",
        "job_id": job["job_id"],
        "phase": job["phase"],
        "package_archive_sha256": package_archive_sha256,
        "package_manifest_sha256": package["package_manifest_sha256"],
        "plan_sha256": quality.canonical_sha256(package["plan"]),
        "root_seal_sha256": quality.canonical_sha256(package["seal"]),
        "job_manifest_sha256": expected_job_manifest_sha256,
        "candidate_library_sha256": library_sha,
        "result_path": job["result_path"],
        "result_sha256": sha256_file(result_path),
        "result_bytes": result_path.stat().st_size,
        "task_records": task_records,
        "task_record_aggregate_sha256": canonical_sha256(task_records),
        "completed_root_count": len(lookup),
        "done_published_last": True,
        "cloud_execution_started": False,
        "training_eligible": False,
        "promotion_evidence": False,
        "current_profile_changed": False,
    }
    _write_once(done_path, done)
    return _validate_done(
        done,
        output_root=output_root,
        package=package,
        job=job,
        library_sha256=library_sha,
        lookup=lookup,
    )


def validate_job_only(
    *,
    package_archive: str | Path,
    package_archive_sha256: str,
    job_id: str,
    expected_job_manifest_sha256: str,
    extraction_directory: str | Path,
) -> dict[str, Any]:
    package = extract_and_validate_package(
        archive_path=package_archive,
        expected_archive_sha256=package_archive_sha256,
        extraction_directory=extraction_directory,
    )
    job = validate_selected_job(
        package=package,
        job_id=job_id,
        expected_job_manifest_sha256=expected_job_manifest_sha256,
    )
    lookup, _artifacts = _root_lookup(
        Path(package["package_root"]), job, package["plan"]
    )
    return {
        "schema": TRANSPORT_PACKAGE_SCHEMA,
        "status": "selected_job_validated_not_executed",
        "job_id": job_id,
        "phase": job["phase"],
        "root_count": len(lookup),
        "package_archive_sha256": package_archive_sha256,
        "job_manifest_sha256": expected_job_manifest_sha256,
        "cloud_execution_started": False,
        "training_eligible": False,
        "current_profile_changed": False,
    }


def _zip_payloads(payloads: Mapping[str, bytes], destination: Path) -> None:
    if destination.exists():
        raise FileExistsError("runtime source archive is create-only")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as raw:
        with zipfile.ZipFile(
            raw, "w", compression=zipfile.ZIP_STORED, strict_timestamps=True
        ) as archive:
            for name in sorted(payloads):
                _safe_archive_name(name)
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                info.create_system = 3
                info.external_attr = (0o100644 & 0xFFFF) << 16
                archive.writestr(info, payloads[name])
        raw.flush()
        os.fsync(raw.fileno())


def create_runtime_source_archive(
    *,
    repository_root: str | Path,
    candidate_library_path: str | Path,
    feature_encoder_path: str | Path,
    archive_path: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    repository = _safe_directory(repository_root, "repository root")
    candidate = _safe_file(candidate_library_path, "candidate library")
    feature = _safe_file(feature_encoder_path, "feature encoder")
    if sha256_file(candidate) != quality.ACCEPTED_CANDIDATE_LIBRARY_SHA256:
        raise ValueError("runtime source candidate library changed")
    if sha256_file(feature) != quality.ACCEPTED_FEATURE_ENCODER_SHA256:
        raise ValueError("runtime source feature encoder changed")
    profile = _safe_file(
        repository / "src/ofc_regular/ai_profiles.py", "profile registry"
    )
    if sha256_file(profile) != quality.CURRENT_PROFILE_REGISTRY_SHA256:
        raise ValueError("current profile registry changed before source packaging")
    payloads: dict[str, bytes] = {
        "pyproject.toml": _safe_file(repository / "pyproject.toml", "pyproject").read_bytes(),
        "configs/hu_joint_policy_m31_t3_step6d_contract.json": _safe_file(
            repository / "configs/hu_joint_policy_m31_t3_step6d_contract.json",
            "Step6d contract",
        ).read_bytes(),
    }
    source_root = _safe_directory(
        repository / "src/ofc_regular", "ofc_regular source"
    )
    for path in sorted(source_root.rglob("*.py")):
        if path.is_symlink() or not path.is_file():
            raise ValueError("runtime Python source contains an unsafe path")
        payloads[path.relative_to(repository).as_posix()] = path.read_bytes()
    payloads[DEFAULT_CANDIDATE_ARCHIVE_PATH] = candidate.read_bytes()
    payloads[DEFAULT_FEATURE_ARCHIVE_PATH] = feature.read_bytes()
    if not _REQUIRED_SOURCE_PATHS.issubset(payloads):
        raise ValueError("runtime source archive lacks a required file")
    archive = Path(archive_path).resolve()
    _zip_payloads(payloads, archive)
    entries = {
        name: {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        for name, raw in sorted(payloads.items())
    }
    manifest = {
        "schema": SOURCE_MANIFEST_SCHEMA,
        "status": "complete_hash_pinned_runtime_source",
        "archive": {
            "path": archive.name,
            "sha256": sha256_file(archive),
            "bytes": archive.stat().st_size,
        },
        "entries": entries,
        "entry_count": len(entries),
        "entry_aggregate_sha256": canonical_sha256(entries),
        "candidate_library": {
            "path": DEFAULT_CANDIDATE_ARCHIVE_PATH,
            **entries[DEFAULT_CANDIDATE_ARCHIVE_PATH],
        },
        "feature_encoder": {
            "path": DEFAULT_FEATURE_ARCHIVE_PATH,
            **entries[DEFAULT_FEATURE_ARCHIVE_PATH],
        },
        "profile_registry_sha256": quality.CURRENT_PROFILE_REGISTRY_SHA256,
        "content_addressed": True,
        "cloud_execution_started": False,
        "training_eligible": False,
        "current_profile_changed": False,
    }
    validate_runtime_source_manifest(
        manifest, archive_path=archive, require_all_runtime_sources=True
    )
    _write_once(manifest_path, manifest)
    return manifest


def validate_runtime_source_manifest(
    value: Mapping[str, Any],
    *,
    archive_path: str | Path,
    require_all_runtime_sources: bool = True,
) -> dict[str, Any]:
    manifest = deepcopy(dict(value))
    _exact_keys(manifest, _SOURCE_MANIFEST_KEYS, "runtime source manifest")
    archive_record = manifest.get("archive")
    entries = manifest.get("entries")
    candidate = manifest.get("candidate_library")
    feature = manifest.get("feature_encoder")
    if not all(
        isinstance(item, Mapping)
        for item in (archive_record, entries, candidate, feature)
    ):
        raise ValueError("runtime source manifest records are missing")
    _exact_keys(archive_record, _FILE_RECORD_KEYS, "runtime source archive")
    _exact_keys(candidate, _FILE_RECORD_KEYS, "candidate library record")
    _exact_keys(feature, _FILE_RECORD_KEYS, "feature encoder record")
    for name, record in entries.items():
        _safe_archive_name(str(name))
        if not isinstance(record, Mapping):
            raise ValueError("runtime source entry is not an object")
        _exact_keys(record, _SOURCE_ENTRY_KEYS, "runtime source entry")
        _require_sha(record["sha256"], "runtime source entry")
    archive = _safe_file(archive_path, "runtime source archive")
    payloads = _archive_payloads(
        archive, expected_sha256=str(archive_record["sha256"])
    )
    actual_entries = {
        name: {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        for name, raw in sorted(payloads.items())
    }
    if (
        manifest["schema"] != SOURCE_MANIFEST_SCHEMA
        or manifest["status"] != "complete_hash_pinned_runtime_source"
        or archive_record["path"] != archive.name
        or archive_record["bytes"] != archive.stat().st_size
        or entries != actual_entries
        or manifest["entry_count"] != len(entries)
        or manifest["entry_aggregate_sha256"] != canonical_sha256(entries)
        or candidate
        != {
            "path": DEFAULT_CANDIDATE_ARCHIVE_PATH,
            **entries.get(DEFAULT_CANDIDATE_ARCHIVE_PATH, {}),
        }
        or feature
        != {
            "path": DEFAULT_FEATURE_ARCHIVE_PATH,
            **entries.get(DEFAULT_FEATURE_ARCHIVE_PATH, {}),
        }
        or candidate["sha256"] != quality.ACCEPTED_CANDIDATE_LIBRARY_SHA256
        or feature["sha256"] != quality.ACCEPTED_FEATURE_ENCODER_SHA256
        or manifest["profile_registry_sha256"]
        != quality.CURRENT_PROFILE_REGISTRY_SHA256
        or entries.get("src/ofc_regular/ai_profiles.py", {}).get("sha256")
        != quality.CURRENT_PROFILE_REGISTRY_SHA256
        or manifest["content_addressed"] is not True
        or any(
            manifest[field] is not False
            for field in (
                "cloud_execution_started",
                "training_eligible",
                "current_profile_changed",
            )
        )
        or (
            require_all_runtime_sources
            and not _REQUIRED_SOURCE_PATHS.issubset(entries)
        )
    ):
        raise ValueError("runtime source manifest boundary changed")
    return manifest


def _relative_record(staging: Path, path: str | Path) -> dict[str, Any]:
    source = _safe_file(path, "staged transport artifact")
    try:
        relative = source.relative_to(staging).as_posix()
    except ValueError as exc:
        raise ValueError("transport artifact is outside staging root") from exc
    _safe_archive_name(relative)
    return {
        "path": relative,
        "sha256": sha256_file(source),
        "bytes": source.stat().st_size,
    }


def _validate_wheelhouse(
    *,
    archive_path: Path,
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    wheel_manifest_file = _safe_file(
        manifest_path, "offline wheelhouse manifest"
    )
    raw = wheel_manifest_file.read_bytes()
    try:
        manifest = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("offline wheelhouse manifest is not JSON") from exc
    if (
        not isinstance(manifest, dict)
        or raw not in (canonical_bytes(manifest), canonical_bytes(manifest) + b"\n")
    ):
        raise ValueError("offline wheelhouse manifest is not canonical")
    entries = manifest.get("entries")
    if (
        manifest.get("schema")
        != "hu_m31_t3_step6d_perfdev_v2_wheelhouse_v1"
        or manifest.get("status")
        != "complete_hash_pinned_offline_wheelhouse"
        or manifest.get("python_abi") != "cp311"
        or manifest.get("target_os") != "linux"
        or manifest.get("target_architecture") != "x86_64"
        or manifest.get("network_install_allowed") is not False
        or not isinstance(entries, list)
        or not entries
        or manifest.get("entry_count") != len(entries)
        or manifest.get("entries_sha256")
        != hashlib.sha256(canonical_bytes(entries) + b"\n").hexdigest()
    ):
        # Existing v4 wheel manifests hash newline-terminated canonical JSON.
        raise ValueError("offline wheelhouse manifest changed")
    payloads = _archive_payloads(
        archive_path, require_deterministic_timestamp=False
    )
    expected = {str(record.get("filename")): record for record in entries}
    if set(payloads) != set(expected):
        raise ValueError("offline wheelhouse inventory changed")
    for name, raw in payloads.items():
        record = expected[name]
        if (
            re.fullmatch(r"[A-Za-z0-9_.+!-]+\.whl", name) is None
            or hashlib.sha256(raw).hexdigest() != record.get("sha256")
            or len(raw) != record.get("bytes")
        ):
            raise ValueError("offline wheelhouse member changed")
    return manifest, {
        "entry_count": len(entries),
        "entries_sha256": manifest["entries_sha256"],
    }


def build_local_launch_manifest(
    *,
    run_name: str,
    staging_directory: str | Path,
    quality_package_archive: str | Path,
    runtime_source_archive: str | Path,
    runtime_source_manifest_path: str | Path,
    wheelhouse_archive: str | Path,
    wheelhouse_manifest_path: str | Path,
    startup_script: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    if _SAFE_RUN.fullmatch(run_name) is None:
        raise ValueError("fresh-quality run_name is unsafe")
    staging = _safe_directory(staging_directory, "transport staging directory")
    package_archive = _safe_file(
        quality_package_archive, "quality package archive"
    )
    source_archive = _safe_file(runtime_source_archive, "runtime source archive")
    source_manifest_path = _safe_file(
        runtime_source_manifest_path, "runtime source manifest"
    )
    source_manifest = validate_runtime_source_manifest(
        _read_canonical(source_manifest_path, "runtime source manifest"),
        archive_path=source_archive,
    )
    wheelhouse = _safe_file(wheelhouse_archive, "offline wheelhouse")
    wheel_manifest_path = _safe_file(
        wheelhouse_manifest_path, "offline wheelhouse manifest"
    )
    _validate_wheelhouse(
        archive_path=wheelhouse, manifest_path=wheel_manifest_path
    )
    startup = _safe_file(startup_script, "fresh-quality startup script")

    # One portable replay supplies the exact package plan, seal, and all job
    # hashes.  The temporary directory is next to the create-only output and is
    # deliberately retained if validation fails.
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError("local launch manifest is create-only")
    replay = output.parent / f".{output.name}.package-replay"
    if replay.exists():
        raise FileExistsError("local launch replay directory already exists")
    package = extract_and_validate_package(
        archive_path=package_archive,
        expected_archive_sha256=sha256_file(package_archive),
        extraction_directory=replay,
    )
    try:
        jobs_by_id = {job["job_id"]: job for job in package["jobs"]}
        launch_jobs = []
        for wave_index, ids in enumerate(WAVE_JOB_IDS):
            for job_id in ids:
                job = jobs_by_id[job_id]
                job_path = replay / "jobs" / f"{job_id}.json"
                launch_jobs.append(
                    {
                        "job_id": job_id,
                        "phase": job["phase"],
                        "wave_index": wave_index,
                        "package_job_path": f"jobs/{job_id}.json",
                        "job_manifest_sha256": sha256_file(job_path),
                        "result_path": job["result_path"],
                        "output_prefix": (
                            f"runs/{run_name}/jobs/{job_id}/attempt-a00"
                        ),
                        "rayon_threads": 16,
                        "processes": 1,
                        "create_only": True,
                    }
                )
        manifest = {
            "schema": LAUNCH_MANIFEST_SCHEMA,
            "status": "local_transport_ready_cloud_not_authorized",
            "run_name": run_name,
            "quality_package": _relative_record(staging, package_archive),
            "runtime_source": _relative_record(staging, source_archive),
            "runtime_source_manifest": _relative_record(
                staging, source_manifest_path
            ),
            "wheelhouse": _relative_record(staging, wheelhouse),
            "wheelhouse_manifest": _relative_record(
                staging, wheel_manifest_path
            ),
            "startup": _relative_record(staging, startup),
            "candidate_library": deepcopy(source_manifest["candidate_library"]),
            "feature_encoder": deepcopy(source_manifest["feature_encoder"]),
            "allocation": {"processes": 1, "rayon_threads": 16},
            "wave_job_counts": list(WAVE_JOB_COUNTS),
            "waves": [
                {"wave_index": index, "job_ids": list(ids), "job_count": len(ids)}
                for index, ids in enumerate(WAVE_JOB_IDS)
            ],
            "jobs": launch_jobs,
            "job_count": 15,
            "package_plan_sha256": quality.canonical_sha256(package["plan"]),
            "package_root_seal_sha256": quality.canonical_sha256(package["seal"]),
            "profile_registry_sha256": quality.CURRENT_PROFILE_REGISTRY_SHA256,
            "transport_ready": True,
            "cloud_launch_authorized": False,
            "cloud_execution_started": False,
            "training_eligible": False,
            "promotion_evidence": False,
            "current_profile_changed": False,
        }
        validate_local_launch_manifest(
            manifest, staging_directory=staging
        )
        _write_once(output, manifest)
        return manifest
    finally:
        # This is a local validation scratch tree derived entirely from the
        # immutable archive; unlike scientific evidence it is safe to remove.
        shutil.rmtree(replay, ignore_errors=False)


def validate_local_launch_manifest(
    value: Mapping[str, Any], *, staging_directory: str | Path
) -> dict[str, Any]:
    manifest = deepcopy(dict(value))
    _exact_keys(manifest, _LAUNCH_KEYS, "fresh-quality local launch manifest")
    staging = _safe_directory(staging_directory, "transport staging directory")
    artifact_fields = (
        "quality_package",
        "runtime_source",
        "runtime_source_manifest",
        "wheelhouse",
        "wheelhouse_manifest",
        "startup",
    )
    for field in artifact_fields:
        record = manifest.get(field)
        if not isinstance(record, Mapping):
            raise ValueError(f"launch artifact {field} is missing")
        _exact_keys(record, _FILE_RECORD_KEYS, f"launch artifact {field}")
        path = _safe_file(
            staging.joinpath(*_safe_archive_name(record["path"]).parts),
            f"launch artifact {field}",
        )
        if (
            sha256_file(path) != _require_sha(record["sha256"], field)
            or path.stat().st_size != record["bytes"]
        ):
            raise ValueError(f"launch artifact {field} changed")
    jobs = manifest.get("jobs")
    waves = manifest.get("waves")
    if not isinstance(jobs, list) or not isinstance(waves, list):
        raise ValueError("fresh-quality launch jobs/waves are missing")
    for job in jobs:
        if not isinstance(job, Mapping):
            raise ValueError("fresh-quality launch job is not an object")
        _exact_keys(job, _LAUNCH_JOB_KEYS, "fresh-quality launch job")
        _require_sha(job.get("job_manifest_sha256"), "launch job manifest")
    source_manifest_path = staging.joinpath(
        *_safe_archive_name(manifest["runtime_source_manifest"]["path"]).parts
    )
    source_archive_path = staging.joinpath(
        *_safe_archive_name(manifest["runtime_source"]["path"]).parts
    )
    source_manifest = validate_runtime_source_manifest(
        _read_canonical(source_manifest_path, "runtime source manifest"),
        archive_path=source_archive_path,
    )
    if (
        manifest["candidate_library"] != source_manifest["candidate_library"]
        or manifest["feature_encoder"] != source_manifest["feature_encoder"]
    ):
        raise ValueError("fresh-quality launch native source binding changed")
    _validate_wheelhouse(
        archive_path=staging.joinpath(
            *_safe_archive_name(manifest["wheelhouse"]["path"]).parts
        ),
        manifest_path=staging.joinpath(
            *_safe_archive_name(manifest["wheelhouse_manifest"]["path"]).parts
        ),
    )
    package_payloads = _archive_payloads(
        staging.joinpath(
            *_safe_archive_name(manifest["quality_package"]["path"]).parts
        ),
        expected_sha256=manifest["quality_package"]["sha256"],
    )
    try:
        packaged_plan = quality.validate_fresh_quality_plan(
            json.loads(package_payloads["control/plan.json"].decode("ascii"))
        )
        packaged_seal = json.loads(
            package_payloads["control/root_seal.json"].decode("ascii")
        )
    except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("fresh-quality launch package control is invalid") from exc
    if (
        package_payloads["control/plan.json"] != quality.canonical_bytes(packaged_plan)
        or not isinstance(packaged_seal, dict)
        or package_payloads["control/root_seal.json"]
        != quality.canonical_bytes(packaged_seal)
        or manifest["package_plan_sha256"]
        != quality.canonical_sha256(packaged_plan)
        or manifest["package_root_seal_sha256"]
        != quality.canonical_sha256(packaged_seal)
    ):
        raise ValueError("fresh-quality launch package science binding changed")
    launch_by_id = {job["job_id"]: job for job in jobs}
    for job_id, launch_job in launch_by_id.items():
        relative = f"jobs/{job_id}.json"
        raw = package_payloads.get(relative)
        if (
            raw is None
            or hashlib.sha256(raw).hexdigest()
            != launch_job["job_manifest_sha256"]
        ):
            raise ValueError("fresh-quality launch selected job digest changed")
    expected_ids = [job_id for ids in WAVE_JOB_IDS for job_id in ids]
    if (
        manifest["schema"] != LAUNCH_MANIFEST_SCHEMA
        or manifest["status"] != "local_transport_ready_cloud_not_authorized"
        or _SAFE_RUN.fullmatch(str(manifest["run_name"])) is None
        or manifest["candidate_library"]["sha256"]
        != quality.ACCEPTED_CANDIDATE_LIBRARY_SHA256
        or manifest["feature_encoder"]["sha256"]
        != quality.ACCEPTED_FEATURE_ENCODER_SHA256
        or manifest["allocation"] != {"processes": 1, "rayon_threads": 16}
        or manifest["wave_job_counts"] != [8, 7]
        or waves
        != [
            {"wave_index": index, "job_ids": list(ids), "job_count": len(ids)}
            for index, ids in enumerate(WAVE_JOB_IDS)
        ]
        or [job["job_id"] for job in jobs] != expected_ids
        or len({job["output_prefix"] for job in jobs}) != 15
        or any(
            job["wave_index"] != wave_index
            or job["phase"]
            != (
                quality.PRIMARY_PHASE
                if job["job_id"].startswith("primary-")
                else quality.CONFIRMATION_PHASE
            )
            or job["package_job_path"] != f"jobs/{job['job_id']}.json"
            or job["result_path"] != f"results/{job['job_id']}.json"
            or job["rayon_threads"] != 16
            or job["processes"] != 1
            or job["create_only"] is not True
            for wave_index, ids in enumerate(WAVE_JOB_IDS)
            for job in jobs
            if job["job_id"] in ids
        )
        or manifest["job_count"] != 15
        or manifest["profile_registry_sha256"]
        != quality.CURRENT_PROFILE_REGISTRY_SHA256
        or manifest["transport_ready"] is not True
        or any(
            manifest[field] is not False
            for field in (
                "cloud_launch_authorized",
                "cloud_execution_started",
                "training_eligible",
                "promotion_evidence",
                "current_profile_changed",
            )
        )
    ):
        raise ValueError("fresh-quality local launch boundary changed")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate-job")
    validate.add_argument("--package-archive", type=Path, required=True)
    validate.add_argument("--package-sha256", required=True)
    validate.add_argument("--job-id", required=True)
    validate.add_argument("--job-manifest-sha256", required=True)
    validate.add_argument("--extraction-dir", type=Path, required=True)
    run = sub.add_parser("run-job")
    run.add_argument("--package-archive", type=Path, required=True)
    run.add_argument("--package-sha256", required=True)
    run.add_argument("--job-id", required=True)
    run.add_argument("--job-manifest-sha256", required=True)
    run.add_argument("--library", type=Path, required=True)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--stop-after-roots", type=int)
    source = sub.add_parser("build-source")
    source.add_argument("--repository-root", type=Path, default=_REPOSITORY_ROOT)
    source.add_argument("--candidate-library", type=Path, required=True)
    source.add_argument("--feature-encoder", type=Path, required=True)
    source.add_argument("--archive", type=Path, required=True)
    source.add_argument("--manifest", type=Path, required=True)
    launch = sub.add_parser("build-launch")
    launch.add_argument("--run-name", required=True)
    launch.add_argument("--staging-dir", type=Path, required=True)
    launch.add_argument("--quality-package", type=Path, required=True)
    launch.add_argument("--runtime-source", type=Path, required=True)
    launch.add_argument("--runtime-source-manifest", type=Path, required=True)
    launch.add_argument("--wheelhouse", type=Path, required=True)
    launch.add_argument("--wheelhouse-manifest", type=Path, required=True)
    launch.add_argument("--startup", type=Path, required=True)
    launch.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate-job":
        result = validate_job_only(
            package_archive=args.package_archive,
            package_archive_sha256=args.package_sha256,
            job_id=args.job_id,
            expected_job_manifest_sha256=args.job_manifest_sha256,
            extraction_directory=args.extraction_dir,
        )
    elif args.command == "run-job":
        result = run_job(
            package_archive=args.package_archive,
            package_archive_sha256=args.package_sha256,
            job_id=args.job_id,
            expected_job_manifest_sha256=args.job_manifest_sha256,
            library_path=args.library,
            output_directory=args.output_dir,
            stop_after_roots=args.stop_after_roots,
        )
        if result["status"] == "interrupted_for_resume":
            print(json.dumps(result, sort_keys=True, separators=(",", ":")))
            return 75
    elif args.command == "build-source":
        result = create_runtime_source_archive(
            repository_root=args.repository_root,
            candidate_library_path=args.candidate_library,
            feature_encoder_path=args.feature_encoder,
            archive_path=args.archive,
            manifest_path=args.manifest,
        )
    else:
        result = build_local_launch_manifest(
            run_name=args.run_name,
            staging_directory=args.staging_dir,
            quality_package_archive=args.quality_package,
            runtime_source_archive=args.runtime_source,
            runtime_source_manifest_path=args.runtime_source_manifest,
            wheelhouse_archive=args.wheelhouse,
            wheelhouse_manifest_path=args.wheelhouse_manifest,
            startup_script=args.startup,
            output_path=args.output,
        )
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DEFAULT_CANDIDATE_ARCHIVE_PATH",
    "DEFAULT_FEATURE_ARCHIVE_PATH",
    "DONE_SCHEMA",
    "LAUNCH_MANIFEST_SCHEMA",
    "SOURCE_MANIFEST_SCHEMA",
    "TASK_SCHEMA",
    "TRANSPORT_PACKAGE_SCHEMA",
    "WAVE_JOB_COUNTS",
    "WAVE_JOB_IDS",
    "build_local_launch_manifest",
    "canonical_bytes",
    "canonical_sha256",
    "create_runtime_source_archive",
    "extract_and_validate_package",
    "main",
    "run_job",
    "sha256_file",
    "validate_job_only",
    "validate_local_launch_manifest",
    "validate_runtime_source_manifest",
    "validate_selected_job",
]
