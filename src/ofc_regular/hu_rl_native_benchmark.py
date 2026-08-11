"""Strictly local mechanics benchmark for the HU RL PyO3 batch boundary.

This module measures the current end-to-end Python/native boundary.  It is a
diagnostic harness, not a promotion or performance-gate artifact: the result
schema permanently records that no performance gate was evaluated.
"""

from __future__ import annotations

import ctypes
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import sys
import time
import urllib.parse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

from .cards import ALL_CARDS
from .hu_rl_contract import MAX_LEGAL_ACTIONS
from .hu_rl_native import (
    MAX_BATCH_LANES,
    MAX_BATCH_THREADS,
    NativeBatchHuRlEnvV1,
    PACKED_ACTION_BYTES,
    PACKED_DIGEST_BYTES,
    PACKED_OBSERVATION_RECORD_BYTES,
    PACKED_STEP_RECORD_BYTES,
    native_available,
)


HU_RL_NATIVE_BATCH_BENCHMARK_SCHEMA: Final = (
    "regular_ofc_hu_rl_native_batch_mechanics_benchmark_v3"
)
HU_RL_NATIVE_BATCH_LANE_RESULT_SCHEMA: Final = (
    "regular_ofc_hu_rl_native_batch_mechanics_lane_result_v3"
)
HU_RL_NATIVE_BATCH_PROVENANCE_SCHEMA: Final = (
    "regular_ofc_hu_rl_native_batch_benchmark_provenance_v1"
)
BENCHMARK_LANE_COUNTS: Final = (128, 512)
BENCHMARK_BOUNDARY_MODES: Final = ("separate_packed", "combined_packed")
DECISIONS_PER_HAND: Final = 10
DEFAULT_BENCHMARK_SEED: Final = 20260722
RL_READY_DIAGNOSTIC_TARGET_DECISIONS_PER_SECOND: Final = 20_000.0
_SOURCE_HASH_PATHS: Final = (
    "rust/hu_rl_engine/Cargo.toml",
    "rust/hu_rl_engine/src/batch.rs",
    "rust/hu_rl_engine/src/history.rs",
    "rust/hu_rl_engine/src/lib.rs",
    "rust/hu_rl_engine/src/observation.rs",
    "rust/hu_rl_engine/src/python.rs",
    "rust/hu_rl_engine/src/transition.rs",
    "rust/hu_rl_engine/src/world.rs",
    "scripts/benchmark_hu_rl_native_batch.py",
    "src/ofc_regular/ai_profiles.py",
    "src/ofc_regular/hu_rl_native.py",
    "src/ofc_regular/hu_rl_native_benchmark.py",
)

_TOP_LEVEL_FIELDS = {
    "schema",
    "status",
    "artifact_role",
    "scope",
    "engine",
    "cloud_executable",
    "launch_authorized",
    "performance_gate_evaluated",
    "performance_gate_pass",
    "diagnostic_target_actor_decisions_per_second",
    "diagnostic_disposition",
    "seed",
    "chunk_width",
    "thread_count",
    "decision_count_per_lane",
    "lane_counts",
    "separate_packed_byte_exact_by_lane",
    "results",
    "provenance",
    "runtime",
    "receipt_sha256",
}
_LANE_RESULT_FIELDS = {
    "schema",
    "boundary_mode",
    "lane_count",
    "actor_decisions",
    "decision_counts_digest",
    "all_done",
    "separate_packed_byte_exact",
    "timings_seconds",
    "actor_decisions_per_second",
    "end_to_end_actor_decisions_per_second",
    "diagnostic_disposition",
    "memory",
}
_TIMING_FIELDS = {
    "deck_generation",
    "construct",
    "reset",
    "actor_decision_total",
    "step_total",
    "decision_loop",
    "total",
}
_MEMORY_FIELDS = {
    "source",
    "sample_count",
    "baseline_rss_bytes",
    "baseline_peak_rss_bytes",
    "final_rss_bytes",
    "max_sampled_rss_bytes",
    "peak_rss_bytes",
    "peak_growth_bytes",
}
_RUNTIME_FIELDS = {
    "python_version",
    "python_implementation",
    "platform",
    "machine",
    "processor",
    "cpu_count",
    "process_id",
    "clock",
    "native_available",
    "memory_sources",
}
_PROVENANCE_FIELDS = {
    "schema",
    "package_name",
    "package_version",
    "wheel_filename",
    "wheel_sha256",
    "native_extension_filename",
    "native_extension_sha256",
    "source_sha256",
}
_EXPECTED_DECISION_IDENTITIES = tuple(
    (actor, f"T{street}") for street in range(5) for actor in range(2)
)
_EXPECTED_MEMORY_SAMPLE_COUNT = 4 + (DECISIONS_PER_HAND * 2)


class HuRlNativeBenchmarkError(ValueError):
    """A benchmark input, native result, timing, or receipt is malformed."""


@dataclass(frozen=True)
class ProcessMemorySample:
    """Current and process-lifetime peak resident memory in bytes."""

    rss_bytes: int
    peak_rss_bytes: int
    source: str

    def __post_init__(self) -> None:
        if not _strict_int(self.rss_bytes) or self.rss_bytes <= 0:
            raise HuRlNativeBenchmarkError("process RSS must be a positive integer")
        if not _strict_int(self.peak_rss_bytes) or self.peak_rss_bytes < self.rss_bytes:
            raise HuRlNativeBenchmarkError("process peak RSS is invalid")
        if not isinstance(self.source, str) or not self.source:
            raise HuRlNativeBenchmarkError("process memory source is invalid")


class _BenchmarkEnv(Protocol):
    @property
    def lane_count(self) -> int: ...

    @property
    def decision_counts(self) -> Sequence[int]: ...

    @property
    def all_done(self) -> bool: ...

    def reset_batch(self, explicit_decks: Sequence[Sequence[str]]) -> Sequence[object]: ...

    def observe_batch_packed(self) -> object: ...

    def legal_actions_batch_packed(self) -> object: ...

    def actor_decision_batch_packed(self) -> object: ...

    def step_batch_packed(self, selected_action_keys: bytes) -> object: ...


_EnvFactory = Callable[..., _BenchmarkEnv]
_MemoryReader = Callable[[], ProcessMemorySample]
_Clock = Callable[[], float]
_ProvenanceProvider = Callable[[], Mapping[str, Any]]


def read_process_memory() -> ProcessMemorySample:
    """Read RSS without psutil or any other external dependency."""

    if sys.platform == "win32":
        return _read_windows_process_memory()
    return _read_posix_process_memory()


def generate_benchmark_decks(*, lane_count: int, seed: int) -> list[list[str]]:
    """Build deterministic, lane-independent complete decks outside native code."""

    _validate_lane_count(lane_count)
    if not _strict_int(seed) or not 0 <= seed <= (1 << 63) - 1:
        raise HuRlNativeBenchmarkError("benchmark seed is outside the unsigned 63-bit domain")
    decks: list[list[str]] = []
    for lane in range(lane_count):
        lane_seed = _splitmix64(seed ^ lane)
        deck = list(ALL_CARDS)
        random.Random(lane_seed).shuffle(deck)
        decks.append(deck)
    return decks


def run_native_batch_mechanics_benchmark(
    *,
    seed: int = DEFAULT_BENCHMARK_SEED,
    chunk_width: int = 64,
    thread_count: int = 16,
    _env_factory: _EnvFactory | None = None,
    _memory_reader: _MemoryReader = read_process_memory,
    _clock: _Clock = time.perf_counter,
    _provenance_provider: _ProvenanceProvider | None = None,
) -> dict[str, Any]:
    """Run the fixed 128/512-lane local diagnostic.

    The underscored injection points exist only for focused contract tests.  A
    production invocation requires the optional native extension and measures
    both the raw PyO3 string boundary and the fully validated Python wrapper.
    """

    _validate_execution_config(chunk_width=chunk_width, thread_count=thread_count)
    if not _strict_int(seed) or not 0 <= seed <= (1 << 63) - 1:
        raise HuRlNativeBenchmarkError("benchmark seed is outside the unsigned 63-bit domain")

    if _env_factory is None:
        if not native_available():
            raise HuRlNativeBenchmarkError("HU RL PyO3 extension is unavailable")
        factories: tuple[tuple[str, _EnvFactory], ...] = (
            ("separate_packed", NativeBatchHuRlEnvV1),
            ("combined_packed", NativeBatchHuRlEnvV1),
        )
        engine = "pyo3_native_combined_vs_separate_packed"
        provenance_provider = collect_native_benchmark_provenance
    else:
        factories = tuple(
            (boundary_mode, _env_factory)
            for boundary_mode in BENCHMARK_BOUNDARY_MODES
        )
        engine = "injected_test_double"
        provenance_provider = _provenance_provider or _test_provenance

    if _provenance_provider is not None:
        provenance_provider = _provenance_provider
    provenance = dict(provenance_provider())
    _validate_provenance(provenance)

    byte_exact_by_lane = {
        str(lane_count): _verify_separate_combined_byte_exact(
            lane_count=lane_count,
            seed=seed,
            chunk_width=chunk_width,
            thread_count=thread_count,
            env_factory=factories[0][1],
        )
        for lane_count in BENCHMARK_LANE_COUNTS
    }

    results = [
        _run_lane_benchmark(
            boundary_mode=boundary_mode,
            lane_count=lane_count,
            seed=seed,
            chunk_width=chunk_width,
            thread_count=thread_count,
            env_factory=env_factory,
            memory_reader=_memory_reader,
            clock=_clock,
            separate_packed_byte_exact=byte_exact_by_lane[str(lane_count)],
        )
        for lane_count in BENCHMARK_LANE_COUNTS
        for boundary_mode, env_factory in factories
    ]
    validated_rates = [
        result["actor_decisions_per_second"]
        for result in results
        if result["boundary_mode"] == "combined_packed"
    ]
    diagnostic_disposition = (
        "rate_target_met_but_gate_not_evaluated"
        if all(
            rate >= RL_READY_DIAGNOSTIC_TARGET_DECISIONS_PER_SECOND
            for rate in validated_rates
        )
        else "below_rate_target_no_go"
    )
    document = {
        "schema": HU_RL_NATIVE_BATCH_BENCHMARK_SCHEMA,
        "status": "local_diagnostic_complete",
        "artifact_role": "local_batch_mechanics_diagnostic_only",
        "scope": "strictly_local_no_cloud",
        "engine": engine,
        "cloud_executable": False,
        "launch_authorized": False,
        "performance_gate_evaluated": False,
        "performance_gate_pass": None,
        "diagnostic_target_actor_decisions_per_second": (
            RL_READY_DIAGNOSTIC_TARGET_DECISIONS_PER_SECOND
        ),
        "diagnostic_disposition": diagnostic_disposition,
        "seed": seed,
        "chunk_width": chunk_width,
        "thread_count": thread_count,
        "decision_count_per_lane": DECISIONS_PER_HAND,
        "lane_counts": list(BENCHMARK_LANE_COUNTS),
        "separate_packed_byte_exact_by_lane": byte_exact_by_lane,
        "results": results,
        "provenance": provenance,
        "runtime": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "machine": platform.machine() or "unknown",
            "processor": (
                platform.processor()
                or os.environ.get("PROCESSOR_IDENTIFIER")
                or "unknown"
            ),
            "cpu_count": os.cpu_count() or 1,
            "process_id": os.getpid(),
            "clock": "time.perf_counter",
            "native_available": native_available(),
            "memory_sources": sorted(
                {result["memory"]["source"] for result in results}
            ),
        },
        "receipt_sha256": None,
    }
    final_provenance = dict(provenance_provider())
    if final_provenance != provenance:
        raise HuRlNativeBenchmarkError("benchmark provenance changed during execution")
    document["receipt_sha256"] = _receipt_digest(document)
    validate_native_batch_benchmark_document(
        document, require_native_engine=_env_factory is None
    )
    return document


def validate_native_batch_benchmark_document(
    document: Mapping[str, Any], *, require_native_engine: bool = True
) -> None:
    """Validate the closed diagnostic receipt and reject gate-like claims."""

    _require_mapping(document, "benchmark document")
    _require_exact_fields(document, _TOP_LEVEL_FIELDS, "benchmark document")
    if document["schema"] != HU_RL_NATIVE_BATCH_BENCHMARK_SCHEMA:
        raise HuRlNativeBenchmarkError("unsupported benchmark document schema")
    expected_literals = {
        "status": "local_diagnostic_complete",
        "artifact_role": "local_batch_mechanics_diagnostic_only",
        "scope": "strictly_local_no_cloud",
        "cloud_executable": False,
        "launch_authorized": False,
        "performance_gate_evaluated": False,
        "performance_gate_pass": None,
        "diagnostic_target_actor_decisions_per_second": (
            RL_READY_DIAGNOSTIC_TARGET_DECISIONS_PER_SECOND
        ),
        "decision_count_per_lane": DECISIONS_PER_HAND,
        "lane_counts": list(BENCHMARK_LANE_COUNTS),
    }
    for field, expected in expected_literals.items():
        if document[field] != expected or (
            isinstance(expected, bool) and type(document[field]) is not bool
        ):
            raise HuRlNativeBenchmarkError(f"benchmark document {field} is invalid")

    engine = document["engine"]
    if engine not in {
        "pyo3_native_combined_vs_separate_packed",
        "injected_test_double",
    }:
        raise HuRlNativeBenchmarkError("benchmark engine is invalid")
    if require_native_engine and engine != "pyo3_native_combined_vs_separate_packed":
        raise HuRlNativeBenchmarkError("persisted benchmark requires the native engine")
    seed = document["seed"]
    if not _strict_int(seed) or not 0 <= seed <= (1 << 63) - 1:
        raise HuRlNativeBenchmarkError("benchmark seed is invalid")
    _validate_execution_config(
        chunk_width=document["chunk_width"], thread_count=document["thread_count"]
    )
    byte_exact_by_lane = document["separate_packed_byte_exact_by_lane"]
    _require_mapping(byte_exact_by_lane, "benchmark byte-exact summary")
    expected_exact = {str(lane_count): True for lane_count in BENCHMARK_LANE_COUNTS}
    if dict(byte_exact_by_lane) != expected_exact:
        raise HuRlNativeBenchmarkError("separate/combined packed byte equivalence failed")

    results = document["results"]
    if isinstance(results, (str, bytes)) or not isinstance(results, Sequence):
        raise HuRlNativeBenchmarkError("benchmark results must be a sequence")
    if len(results) != len(BENCHMARK_LANE_COUNTS) * len(BENCHMARK_BOUNDARY_MODES):
        raise HuRlNativeBenchmarkError("benchmark result count is invalid")
    expected_pairs = tuple(
        (lane_count, boundary_mode)
        for lane_count in BENCHMARK_LANE_COUNTS
        for boundary_mode in BENCHMARK_BOUNDARY_MODES
    )
    for (expected_lane_count, expected_boundary_mode), result in zip(
        expected_pairs, results, strict=True
    ):
        _validate_lane_result(
            result,
            expected_lane_count=expected_lane_count,
            expected_boundary_mode=expected_boundary_mode,
        )
    validated_rates = [
        result["actor_decisions_per_second"]
        for result in results
        if result["boundary_mode"] == "combined_packed"
    ]
    expected_disposition = (
        "rate_target_met_but_gate_not_evaluated"
        if all(
            rate >= RL_READY_DIAGNOSTIC_TARGET_DECISIONS_PER_SECOND
            for rate in validated_rates
        )
        else "below_rate_target_no_go"
    )
    if document["diagnostic_disposition"] != expected_disposition:
        raise HuRlNativeBenchmarkError("benchmark diagnostic disposition disagrees")

    runtime = document["runtime"]
    _require_mapping(runtime, "benchmark runtime")
    _require_exact_fields(runtime, _RUNTIME_FIELDS, "benchmark runtime")
    for field in (
        "python_version",
        "python_implementation",
        "platform",
        "machine",
        "processor",
        "clock",
    ):
        if not isinstance(runtime[field], str) or not runtime[field]:
            raise HuRlNativeBenchmarkError(f"benchmark runtime {field} is invalid")
    if runtime["clock"] != "time.perf_counter":
        raise HuRlNativeBenchmarkError("benchmark clock is unsupported")
    if not _strict_int(runtime["process_id"]) or runtime["process_id"] <= 0:
        raise HuRlNativeBenchmarkError("benchmark process_id is invalid")
    if type(runtime["native_available"]) is not bool:
        raise HuRlNativeBenchmarkError("benchmark native_available is invalid")
    if not _strict_int(runtime["cpu_count"]) or runtime["cpu_count"] <= 0:
        raise HuRlNativeBenchmarkError("benchmark runtime cpu_count is invalid")
    if (
        engine == "pyo3_native_combined_vs_separate_packed"
        and runtime["native_available"] is not True
    ):
        raise HuRlNativeBenchmarkError("native benchmark contradicts runtime availability")
    expected_sources = sorted(
        {result["memory"]["source"] for result in results}
    )
    if runtime["memory_sources"] != expected_sources:
        raise HuRlNativeBenchmarkError("benchmark memory source summary disagrees")

    _validate_provenance(document["provenance"])
    receipt_sha256 = document["receipt_sha256"]
    if not _is_sha256(receipt_sha256) or receipt_sha256 != _receipt_digest(document):
        raise HuRlNativeBenchmarkError("benchmark receipt digest disagrees")


def canonical_benchmark_json(document: Mapping[str, Any]) -> str:
    """Return deterministic JSON only after the full receipt validates."""

    validate_native_batch_benchmark_document(document, require_native_engine=True)
    try:
        return json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise HuRlNativeBenchmarkError("benchmark document is not canonical JSON") from exc


def collect_native_benchmark_provenance() -> dict[str, Any]:
    """Bind a local receipt to the installed wheel and exact source snapshot."""

    try:
        distribution = importlib.metadata.distribution("ofc-hu-rl-engine-native")
        direct_url_text = distribution.read_text("direct_url.json")
        if direct_url_text is None:
            raise HuRlNativeBenchmarkError("native wheel direct_url metadata is absent")
        direct_url = json.loads(direct_url_text)
        url = direct_url["url"]
        declared_hash = direct_url["archive_info"]["hashes"]["sha256"]
    except (
        importlib.metadata.PackageNotFoundError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
    ) as exc:
        raise HuRlNativeBenchmarkError("native wheel provenance is unavailable") from exc
    if not isinstance(url, str) or not _is_sha256(declared_hash):
        raise HuRlNativeBenchmarkError("native wheel provenance metadata is invalid")
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
        raise HuRlNativeBenchmarkError("native wheel provenance is not a local file")
    wheel_path_text = urllib.parse.unquote(parsed.path)
    if os.name == "nt" and len(wheel_path_text) >= 3 and wheel_path_text[0] == "/":
        wheel_path_text = wheel_path_text[1:]
    wheel_path = Path(wheel_path_text)
    if not wheel_path.is_file() or wheel_path.suffix.lower() != ".whl":
        raise HuRlNativeBenchmarkError("native wheel provenance file is unavailable")
    wheel_sha256 = _sha256_file(wheel_path)
    if wheel_sha256 != declared_hash:
        raise HuRlNativeBenchmarkError("native wheel hash disagrees with install metadata")

    try:
        import _ofc_hu_rl_engine._ofc_hu_rl_engine as native_extension

        native_extension_path = Path(native_extension.__file__).resolve()
    except (AttributeError, ImportError, TypeError) as exc:
        raise HuRlNativeBenchmarkError("native extension provenance is unavailable") from exc
    if not native_extension_path.is_file():
        raise HuRlNativeBenchmarkError("native extension provenance file is unavailable")

    repository = Path(__file__).resolve().parents[2]
    source_sha256: dict[str, str] = {}
    for relative in _SOURCE_HASH_PATHS:
        source = repository / Path(relative)
        if not source.is_file():
            raise HuRlNativeBenchmarkError("benchmark source snapshot is incomplete")
        source_sha256[relative] = _sha256_file(source)

    return {
        "schema": HU_RL_NATIVE_BATCH_PROVENANCE_SCHEMA,
        "package_name": distribution.metadata["Name"],
        "package_version": distribution.version,
        "wheel_filename": wheel_path.name,
        "wheel_sha256": wheel_sha256,
        "native_extension_filename": native_extension_path.name,
        "native_extension_sha256": _sha256_file(native_extension_path),
        "source_sha256": source_sha256,
    }


def _test_provenance() -> dict[str, Any]:
    return {
        "schema": HU_RL_NATIVE_BATCH_PROVENANCE_SCHEMA,
        "package_name": "ofc-hu-rl-engine-native-test-double",
        "package_version": "0.0.0-test",
        "wheel_filename": "test-double.whl",
        "wheel_sha256": "a" * 64,
        "native_extension_filename": "test-double.pyd",
        "native_extension_sha256": "b" * 64,
        "source_sha256": {path: "c" * 64 for path in _SOURCE_HASH_PATHS},
    }


def _validate_provenance(value: object) -> None:
    _require_mapping(value, "benchmark provenance")
    _require_exact_fields(value, _PROVENANCE_FIELDS, "benchmark provenance")
    if value["schema"] != HU_RL_NATIVE_BATCH_PROVENANCE_SCHEMA:
        raise HuRlNativeBenchmarkError("benchmark provenance schema is invalid")
    for field in ("package_name", "package_version"):
        if not isinstance(value[field], str) or not value[field]:
            raise HuRlNativeBenchmarkError(f"benchmark provenance {field} is invalid")
    for field in ("wheel_filename", "native_extension_filename"):
        filename = value[field]
        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).name != filename
        ):
            raise HuRlNativeBenchmarkError(f"benchmark provenance {field} is invalid")
    for field in ("wheel_sha256", "native_extension_sha256"):
        if not _is_sha256(value[field]):
            raise HuRlNativeBenchmarkError(f"benchmark provenance {field} is invalid")
    source_sha256 = value["source_sha256"]
    _require_mapping(source_sha256, "benchmark source hashes")
    if set(source_sha256) != set(_SOURCE_HASH_PATHS) or any(
        not _is_sha256(digest) for digest in source_sha256.values()
    ):
        raise HuRlNativeBenchmarkError("benchmark source hashes are invalid")


def _receipt_digest(document: Mapping[str, Any]) -> str:
    payload = dict(document)
    payload["receipt_sha256"] = None
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise HuRlNativeBenchmarkError("benchmark provenance file could not be read") from exc
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _run_lane_benchmark(
    *,
    boundary_mode: str,
    lane_count: int,
    seed: int,
    chunk_width: int,
    thread_count: int,
    env_factory: _EnvFactory,
    memory_reader: _MemoryReader,
    clock: _Clock,
    separate_packed_byte_exact: bool,
) -> dict[str, Any]:
    if boundary_mode not in BENCHMARK_BOUNDARY_MODES:
        raise HuRlNativeBenchmarkError("benchmark boundary mode is invalid")
    _validate_lane_count(lane_count)
    samples: list[ProcessMemorySample] = []

    def sample_memory() -> None:
        sample = memory_reader()
        if type(sample) is not ProcessMemorySample:
            raise HuRlNativeBenchmarkError("memory reader returned an invalid sample")
        if samples and sample.source != samples[0].source:
            raise HuRlNativeBenchmarkError("memory reader source changed during the run")
        samples.append(sample)

    total_start = _read_clock(clock)
    sample_memory()

    decks_start = _read_clock(clock)
    decks = generate_benchmark_decks(lane_count=lane_count, seed=seed)
    deck_generation_seconds = _elapsed(decks_start, _read_clock(clock), "deck generation")
    sample_memory()

    construct_start = _read_clock(clock)
    env = env_factory(
        decks,
        chunk_width=chunk_width,
        thread_count=thread_count,
    )
    construct_seconds = _elapsed(construct_start, _read_clock(clock), "construction")
    if not _strict_int(env.lane_count) or env.lane_count != lane_count:
        raise HuRlNativeBenchmarkError("constructed batch lane count disagrees")
    sample_memory()

    reset_start = _read_clock(clock)
    reset_views = env.reset_batch(decks)
    reset_seconds = _elapsed(reset_start, _read_clock(clock), "reset")
    _require_lane_sequence(reset_views, lane_count=lane_count, operation="reset")
    sample_memory()

    actor_decision_seconds = 0.0
    step_seconds = 0.0
    decision_loop_start = _read_clock(clock)
    for decision, (expected_actor, expected_street) in enumerate(
        _EXPECTED_DECISION_IDENTITIES
    ):
        actor_decision_start = _read_clock(clock)
        if boundary_mode == "separate_packed":
            observed = env.observe_batch_packed()
            legal = env.legal_actions_batch_packed()
        else:
            actor_decision = env.actor_decision_batch_packed()
            observed, legal = _split_actor_decision(actor_decision)
        actor_decision_seconds += _elapsed(
            actor_decision_start,
            _read_clock(clock),
            "packed actor decision",
        )
        _validate_packed_observation(observed, lane_count=lane_count)
        _validate_packed_legal(legal, lane_count=lane_count)
        selected = _select_packed_actions(
            legal,
            lane_count=lane_count,
            decision_ordinal=decision,
        )
        sample_memory()

        step_start = _read_clock(clock)
        step_results = env.step_batch_packed(selected)
        step_seconds += _elapsed(step_start, _read_clock(clock), "batch step")
        _validate_packed_step_result(
            step_results,
            lane_count=lane_count,
            expected_actor=expected_actor,
            expected_street=expected_street,
            expected_done=decision == DECISIONS_PER_HAND - 1,
        )
        sample_memory()
    decision_loop_seconds = _elapsed(
        decision_loop_start, _read_clock(clock), "decision loop"
    )
    total_seconds = _elapsed(total_start, _read_clock(clock), "total benchmark")

    if type(env.all_done) is not bool or not env.all_done:
        raise HuRlNativeBenchmarkError("batch did not terminate after ten decisions")
    decision_counts = tuple(env.decision_counts)
    if len(decision_counts) != lane_count or any(
        not _strict_int(count) or count != DECISIONS_PER_HAND
        for count in decision_counts
    ):
        raise HuRlNativeBenchmarkError("batch decision counts disagree after terminal")
    if len(samples) != _EXPECTED_MEMORY_SAMPLE_COUNT:
        raise HuRlNativeBenchmarkError("benchmark memory sample count changed")

    timed_sum = (
        deck_generation_seconds
        + construct_seconds
        + reset_seconds
        + decision_loop_seconds
    )
    if total_seconds < timed_sum:
        raise HuRlNativeBenchmarkError("benchmark total timing is internally inconsistent")
    if decision_loop_seconds < actor_decision_seconds + step_seconds:
        raise HuRlNativeBenchmarkError("decision-loop timing is internally inconsistent")

    actor_decisions = lane_count * DECISIONS_PER_HAND
    boundary_call_seconds = actor_decision_seconds + step_seconds
    throughput = actor_decisions / boundary_call_seconds
    end_to_end_throughput = actor_decisions / decision_loop_seconds
    if (
        not math.isfinite(throughput)
        or throughput <= 0.0
        or not math.isfinite(end_to_end_throughput)
        or end_to_end_throughput <= 0.0
    ):
        raise HuRlNativeBenchmarkError("actor decision throughput is invalid")
    diagnostic_disposition = (
        "rate_target_met_but_gate_not_evaluated"
        if throughput >= RL_READY_DIAGNOSTIC_TARGET_DECISIONS_PER_SECOND
        else "below_rate_target_no_go"
    )

    baseline = samples[0]
    final = samples[-1]
    peak_rss = max(sample.peak_rss_bytes for sample in samples)
    max_sampled_rss = max(sample.rss_bytes for sample in samples)
    if peak_rss < baseline.peak_rss_bytes or peak_rss < max_sampled_rss:
        raise HuRlNativeBenchmarkError("benchmark RSS samples are inconsistent")

    return {
        "schema": HU_RL_NATIVE_BATCH_LANE_RESULT_SCHEMA,
        "boundary_mode": boundary_mode,
        "lane_count": lane_count,
        "actor_decisions": actor_decisions,
        "decision_counts_digest": _decision_counts_digest(lane_count),
        "all_done": True,
        "separate_packed_byte_exact": separate_packed_byte_exact,
        "timings_seconds": {
            "deck_generation": deck_generation_seconds,
            "construct": construct_seconds,
            "reset": reset_seconds,
            "actor_decision_total": actor_decision_seconds,
            "step_total": step_seconds,
            "decision_loop": decision_loop_seconds,
            "total": total_seconds,
        },
        "actor_decisions_per_second": throughput,
        "end_to_end_actor_decisions_per_second": end_to_end_throughput,
        "diagnostic_disposition": diagnostic_disposition,
        "memory": {
            "source": baseline.source,
            "sample_count": len(samples),
            "baseline_rss_bytes": baseline.rss_bytes,
            "baseline_peak_rss_bytes": baseline.peak_rss_bytes,
            "final_rss_bytes": final.rss_bytes,
            "max_sampled_rss_bytes": max_sampled_rss,
            "peak_rss_bytes": peak_rss,
            "peak_growth_bytes": peak_rss - baseline.peak_rss_bytes,
        },
    }


def _split_actor_decision(value: object) -> tuple[object, object]:
    try:
        observations = value.observations  # type: ignore[attr-defined]
        legal_actions = value.legal_actions  # type: ignore[attr-defined]
    except AttributeError as exc:
        raise HuRlNativeBenchmarkError(
            "combined actor decision is missing packed fields"
        ) from exc
    return observations, legal_actions


def _validate_packed_observation(value: object, *, lane_count: int) -> bytes:
    try:
        payload = value.payload  # type: ignore[attr-defined]
        reported_lane_count = value.lane_count  # type: ignore[attr-defined]
    except AttributeError as exc:
        raise HuRlNativeBenchmarkError("packed observation is malformed") from exc
    if (
        type(payload) is not bytes
        or reported_lane_count != lane_count
        or not _strict_int(reported_lane_count)
        or len(payload) != lane_count * PACKED_OBSERVATION_RECORD_BYTES
    ):
        raise HuRlNativeBenchmarkError("packed observation geometry changed")
    return payload


def _packed_legal_buffers(value: object, *, lane_count: int) -> tuple[bytes, ...]:
    try:
        reported_lane_count = value.lane_count  # type: ignore[attr-defined]
        fields = (
            value.action_keys,  # type: ignore[attr-defined]
            value.mask,  # type: ignore[attr-defined]
            value.action_counts,  # type: ignore[attr-defined]
            value.action_set_digests,  # type: ignore[attr-defined]
            value.action_order_digests,  # type: ignore[attr-defined]
        )
    except AttributeError as exc:
        raise HuRlNativeBenchmarkError("packed legal actions are malformed") from exc
    expected_lengths = (
        lane_count * MAX_LEGAL_ACTIONS * PACKED_ACTION_BYTES,
        lane_count * MAX_LEGAL_ACTIONS,
        lane_count,
        lane_count * PACKED_DIGEST_BYTES,
        lane_count * PACKED_DIGEST_BYTES,
    )
    if (
        reported_lane_count != lane_count
        or not _strict_int(reported_lane_count)
        or any(type(field) is not bytes for field in fields)
        or tuple(map(len, fields)) != expected_lengths
    ):
        raise HuRlNativeBenchmarkError("packed legal action geometry changed")
    return fields


def _validate_packed_legal(value: object, *, lane_count: int) -> None:
    _packed_legal_buffers(value, lane_count=lane_count)


def _select_packed_actions(
    legal: object, *, lane_count: int, decision_ordinal: int
) -> bytes:
    try:
        indices = tuple(
            (lane + decision_ordinal) % legal.action_count(lane)  # type: ignore[attr-defined]
            for lane in range(lane_count)
        )
        selected = legal.select(indices)  # type: ignore[attr-defined]
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        raise HuRlNativeBenchmarkError("packed action selection failed") from exc
    if type(selected) is not bytes or len(selected) != lane_count * PACKED_ACTION_BYTES:
        raise HuRlNativeBenchmarkError("packed action selection geometry changed")
    return selected


def _validate_packed_step_result(
    result: object,
    *,
    lane_count: int,
    expected_actor: int,
    expected_street: str,
    expected_done: bool,
) -> bytes:
    try:
        payload = result.payload  # type: ignore[attr-defined]
        reported_lane_count = result.lane_count  # type: ignore[attr-defined]
    except AttributeError as exc:
        raise HuRlNativeBenchmarkError("packed step result is malformed") from exc
    if (
        type(payload) is not bytes
        or reported_lane_count != lane_count
        or not _strict_int(reported_lane_count)
        or len(payload) != lane_count * PACKED_STEP_RECORD_BYTES
    ):
        raise HuRlNativeBenchmarkError("packed step geometry changed")
    for lane in range(lane_count):
        try:
            decoded = result.lane(lane)  # type: ignore[attr-defined]
            rewards = decoded.rewards
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            raise HuRlNativeBenchmarkError("packed step lane decode failed") from exc
        if (
            decoded.actor != expected_actor
            or decoded.street != expected_street
            or decoded.done is not expected_done
            or not isinstance(rewards, Sequence)
            or len(rewards) != 2
            or any(not math.isfinite(float(reward)) for reward in rewards)
            or (not expected_done and tuple(rewards) != (0.0, 0.0))
            or (expected_done and float(rewards[0]) != -float(rewards[1]))
        ):
            raise HuRlNativeBenchmarkError("packed step decision identity changed")
    return payload


def _verify_separate_combined_byte_exact(
    *,
    lane_count: int,
    seed: int,
    chunk_width: int,
    thread_count: int,
    env_factory: _EnvFactory,
) -> bool:
    decks = generate_benchmark_decks(lane_count=lane_count, seed=seed)
    separate = env_factory(
        decks,
        chunk_width=chunk_width,
        thread_count=thread_count,
    )
    combined = env_factory(
        decks,
        chunk_width=chunk_width,
        thread_count=thread_count,
    )
    for env in (separate, combined):
        reset = env.reset_batch(decks)
        _require_lane_sequence(reset, lane_count=lane_count, operation="reset")

    for decision, (expected_actor, expected_street) in enumerate(
        _EXPECTED_DECISION_IDENTITIES
    ):
        separate_observation = separate.observe_batch_packed()
        separate_legal = separate.legal_actions_batch_packed()
        combined_observation, combined_legal = _split_actor_decision(
            combined.actor_decision_batch_packed()
        )
        if _validate_packed_observation(
            separate_observation, lane_count=lane_count
        ) != _validate_packed_observation(combined_observation, lane_count=lane_count):
            raise HuRlNativeBenchmarkError("packed observation byte equivalence failed")
        if _packed_legal_buffers(
            separate_legal, lane_count=lane_count
        ) != _packed_legal_buffers(combined_legal, lane_count=lane_count):
            raise HuRlNativeBenchmarkError("packed legal byte equivalence failed")
        separate_selected = _select_packed_actions(
            separate_legal,
            lane_count=lane_count,
            decision_ordinal=decision,
        )
        combined_selected = _select_packed_actions(
            combined_legal,
            lane_count=lane_count,
            decision_ordinal=decision,
        )
        if separate_selected != combined_selected:
            raise HuRlNativeBenchmarkError("packed selection byte equivalence failed")
        separate_step = separate.step_batch_packed(separate_selected)
        combined_step = combined.step_batch_packed(combined_selected)
        expected_done = decision == DECISIONS_PER_HAND - 1
        if _validate_packed_step_result(
            separate_step,
            lane_count=lane_count,
            expected_actor=expected_actor,
            expected_street=expected_street,
            expected_done=expected_done,
        ) != _validate_packed_step_result(
            combined_step,
            lane_count=lane_count,
            expected_actor=expected_actor,
            expected_street=expected_street,
            expected_done=expected_done,
        ):
            raise HuRlNativeBenchmarkError("packed step byte equivalence failed")
    if (
        separate.all_done is not True
        or combined.all_done is not True
        or tuple(separate.decision_counts) != tuple(combined.decision_counts)
    ):
        raise HuRlNativeBenchmarkError("packed terminal equivalence failed")
    return True


def _validate_lane_result(
    result: object, *, expected_lane_count: int, expected_boundary_mode: str
) -> None:
    _require_mapping(result, "benchmark lane result")
    _require_exact_fields(result, _LANE_RESULT_FIELDS, "benchmark lane result")
    if result["schema"] != HU_RL_NATIVE_BATCH_LANE_RESULT_SCHEMA:
        raise HuRlNativeBenchmarkError("unsupported benchmark lane-result schema")
    if result["boundary_mode"] != expected_boundary_mode:
        raise HuRlNativeBenchmarkError("benchmark boundary mode is invalid")
    if not _strict_int(result["lane_count"]) or result["lane_count"] != expected_lane_count:
        raise HuRlNativeBenchmarkError("benchmark lane count is invalid")
    expected_decisions = expected_lane_count * DECISIONS_PER_HAND
    if not _strict_int(result["actor_decisions"]) or result["actor_decisions"] != expected_decisions:
        raise HuRlNativeBenchmarkError("benchmark actor decision count is invalid")
    if result["decision_counts_digest"] != _decision_counts_digest(expected_lane_count):
        raise HuRlNativeBenchmarkError("benchmark decision-count digest is invalid")
    if type(result["all_done"]) is not bool or result["all_done"] is not True:
        raise HuRlNativeBenchmarkError("benchmark terminal flag is invalid")
    if result["separate_packed_byte_exact"] is not True:
        raise HuRlNativeBenchmarkError("benchmark packed byte equivalence failed")

    timings = result["timings_seconds"]
    _require_mapping(timings, "benchmark timings")
    _require_exact_fields(timings, _TIMING_FIELDS, "benchmark timings")
    for field, value in timings.items():
        if type(value) not in (int, float) or not math.isfinite(float(value)) or value <= 0:
            raise HuRlNativeBenchmarkError(f"benchmark timing {field} is invalid")
    if timings["decision_loop"] < (
        timings["actor_decision_total"] + timings["step_total"]
    ):
        raise HuRlNativeBenchmarkError("benchmark decision timing disagrees")
    timed_sum = sum(
        timings[field]
        for field in ("deck_generation", "construct", "reset", "decision_loop")
    )
    if timings["total"] < timed_sum:
        raise HuRlNativeBenchmarkError("benchmark total timing disagrees")
    throughput = result["actor_decisions_per_second"]
    boundary_call_seconds = (
        timings["actor_decision_total"] + timings["step_total"]
    )
    if (
        type(throughput) not in (int, float)
        or not math.isfinite(float(throughput))
        or throughput <= 0
        or not math.isclose(
            float(throughput),
            expected_decisions / float(boundary_call_seconds),
            rel_tol=1e-12,
            abs_tol=0.0,
        )
    ):
        raise HuRlNativeBenchmarkError("benchmark actor throughput disagrees")
    end_to_end = result["end_to_end_actor_decisions_per_second"]
    if (
        type(end_to_end) not in (int, float)
        or not math.isfinite(float(end_to_end))
        or end_to_end <= 0
        or not math.isclose(
            float(end_to_end),
            expected_decisions / float(timings["decision_loop"]),
            rel_tol=1e-12,
            abs_tol=0.0,
        )
    ):
        raise HuRlNativeBenchmarkError("benchmark end-to-end throughput disagrees")
    expected_disposition = (
        "rate_target_met_but_gate_not_evaluated"
        if throughput >= RL_READY_DIAGNOSTIC_TARGET_DECISIONS_PER_SECOND
        else "below_rate_target_no_go"
    )
    if result["diagnostic_disposition"] != expected_disposition:
        raise HuRlNativeBenchmarkError("benchmark lane diagnostic disposition disagrees")

    memory = result["memory"]
    _require_mapping(memory, "benchmark memory")
    _require_exact_fields(memory, _MEMORY_FIELDS, "benchmark memory")
    if not isinstance(memory["source"], str) or not memory["source"]:
        raise HuRlNativeBenchmarkError("benchmark memory source is invalid")
    if (
        not _strict_int(memory["sample_count"])
        or memory["sample_count"] != _EXPECTED_MEMORY_SAMPLE_COUNT
    ):
        raise HuRlNativeBenchmarkError("benchmark memory sample count is invalid")
    byte_fields = _MEMORY_FIELDS - {"source", "sample_count"}
    if any(
        not _strict_int(memory[field]) or memory[field] < 0
        for field in byte_fields
    ):
        raise HuRlNativeBenchmarkError("benchmark memory byte count is invalid")
    if any(
        memory[field] <= 0
        for field in (
            "baseline_rss_bytes",
            "baseline_peak_rss_bytes",
            "final_rss_bytes",
            "max_sampled_rss_bytes",
            "peak_rss_bytes",
        )
    ):
        raise HuRlNativeBenchmarkError("benchmark RSS is not positive")
    if memory["peak_rss_bytes"] < max(
        memory["baseline_peak_rss_bytes"], memory["max_sampled_rss_bytes"]
    ):
        raise HuRlNativeBenchmarkError("benchmark peak RSS disagrees")
    if memory["max_sampled_rss_bytes"] < max(
        memory["baseline_rss_bytes"], memory["final_rss_bytes"]
    ):
        raise HuRlNativeBenchmarkError("benchmark sampled RSS disagrees")
    if memory["peak_growth_bytes"] != (
        memory["peak_rss_bytes"] - memory["baseline_peak_rss_bytes"]
    ):
        raise HuRlNativeBenchmarkError("benchmark peak RSS growth disagrees")


def _read_windows_process_memory() -> ProcessMemorySample:
    from ctypes import wintypes

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        ok = psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        )
    except (AttributeError, OSError) as exc:
        raise HuRlNativeBenchmarkError("Windows process memory query failed") from exc
    if not ok:
        raise HuRlNativeBenchmarkError("Windows process memory query failed")
    return ProcessMemorySample(
        rss_bytes=int(counters.WorkingSetSize),
        peak_rss_bytes=int(counters.PeakWorkingSetSize),
        source="windows_get_process_memory_info",
    )


def _read_posix_process_memory() -> ProcessMemorySample:
    try:
        import resource

        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, OSError, ValueError) as exc:
        raise HuRlNativeBenchmarkError("POSIX process memory query failed") from exc
    if sys.platform != "darwin":
        peak *= 1024
    current = peak
    if sys.platform.startswith("linux"):
        try:
            with open("/proc/self/statm", encoding="ascii") as handle:
                fields = handle.read().split()
            current = int(fields[1]) * int(os.sysconf("SC_PAGE_SIZE"))
        except (IndexError, OSError, ValueError) as exc:
            raise HuRlNativeBenchmarkError("Linux current RSS query failed") from exc
    return ProcessMemorySample(
        rss_bytes=current,
        peak_rss_bytes=max(current, peak),
        source="posix_rusage_statm",
    )


def _read_clock(clock: _Clock) -> float:
    value = clock()
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise HuRlNativeBenchmarkError("benchmark clock returned an invalid value")
    return float(value)


def _elapsed(start: float, end: float, operation: str) -> float:
    elapsed = end - start
    if not math.isfinite(elapsed) or elapsed <= 0.0:
        raise HuRlNativeBenchmarkError(f"{operation} timer did not advance")
    return elapsed


def _decision_counts_digest(lane_count: int) -> str:
    payload = json.dumps(
        [DECISIONS_PER_HAND] * lane_count,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _splitmix64(value: int) -> int:
    mask = (1 << 64) - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return value ^ (value >> 31)


def _validate_lane_count(lane_count: int) -> None:
    if not _strict_int(lane_count) or not 1 <= lane_count <= MAX_BATCH_LANES:
        raise HuRlNativeBenchmarkError("benchmark lane count is invalid")


def _validate_execution_config(*, chunk_width: object, thread_count: object) -> None:
    if not _strict_int(chunk_width) or chunk_width <= 0:
        raise HuRlNativeBenchmarkError("benchmark chunk width is invalid")
    if (
        not _strict_int(thread_count)
        or not 1 <= thread_count <= MAX_BATCH_THREADS
    ):
        raise HuRlNativeBenchmarkError("benchmark thread count is invalid")


def _require_lane_sequence(value: object, *, lane_count: int, operation: str) -> None:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != lane_count
    ):
        raise HuRlNativeBenchmarkError(f"{operation} returned an invalid lane sequence")


def _require_mapping(value: object, context: str) -> None:
    if not isinstance(value, Mapping):
        raise HuRlNativeBenchmarkError(f"{context} must be a mapping")


def _require_exact_fields(
    value: Mapping[str, Any], expected: set[str], context: str
) -> None:
    if set(value) != expected:
        raise HuRlNativeBenchmarkError(f"{context} fields are invalid")


def _strict_int(value: object) -> bool:
    return type(value) is int
