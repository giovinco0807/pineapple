"""Proof-only correctness preflight for the frozen Attempt08 teacher.

Only already-consumed Attempt06 source roots 0, 1, and 2 may be replayed.  The
runner emits hashes and correctness booleans, never teacher actions or values.
It cannot generate an Attempt08 development/audit root or authorize a run.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from .action_key import ActionKey
from .ai_profiles import ModelPaths, build_policy, load_model_bundle
from .hu_m43_attempt06_teacher import (
    ATTEMPT06_SHARD_ROW_SCHEMA,
    ATTEMPT06_T2_POLICY_ID,
    ATTEMPT06_TEACHER_SCHEMA,
    _require_stage9f_p2_policies,
)
from .hu_m43_attempt08_contract import (
    AI_PROFILES_SHA256,
    ATTEMPT08_LAMBDA_MODEL_SHA256,
    M43_ATTEMPT08_PLAN_SCHEMA,
    M43_ATTEMPT08_PLAN_SHA256,
    enumerate_attempt06_known_seed_schedules,
    enumerate_attempt07_known_seed_schedules,
    enumerate_attempt07_preflight_known_seed_schedules,
    enumerate_attempt08_seed_schedules,
    load_and_validate_attempt08_plan,
    validate_attempt08_artifact_bindings,
)
from .hu_m43_attempt08_teacher import (
    ATTEMPT08_SOLVER_ID,
    ATTEMPT08_TEACHER_SCHEMA,
    Attempt08TeacherConfig,
    FrozenAttempt08LambdaRanker,
    evaluate_attempt08_t1_second,
    validate_attempt08_teacher_output,
)
from .hu_m43_attempt08_runtime_anchor import (
    DEFAULT_MODEL_MANIFEST_PATH,
    DEFAULT_NATIVE_MANIFEST_PATH,
    DEFAULT_RUNTIME_ARTIFACT_ROOT,
    DEFAULT_REQUIREMENTS_PATH,
    ATTEMPT08_RUNTIME_REQUIREMENTS_SHA256,
    validate_runtime_semantic_anchor,
)
from .hu_m43_attempt08_runtime_anchor_contract import (
    ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SHA256,
    ATTEMPT08_RUNTIME_SOURCE_CLOSURE_SHA256,
    ATTEMPT08_RUNTIME_SOURCE_FILE_COUNT,
)
from .hu_m43_attempt08_runtime_identity import (
    ATTEMPT08_EXPECTED_RUNTIME_FINGERPRINT_SHA256,
    ATTEMPT08_GCP_IMAGE_ID,
    ATTEMPT08_GCP_IMAGE_NAME,
    validate_expected_runtime_fingerprint,
)
from .run_hu_m43_attempt07_preflight import (
    ATTEMPT06_SOURCE_PLAN_SHA256,
    ATTEMPT07_PREFLIGHT_SOURCE_SHA256,
    _atomic_write_once,
    _native_batch_threads,
    _sha256_file,
    _sha256_value,
    canonical_json_bytes,
    load_source_row,
)


ATTEMPT08_PREFLIGHT_PLAN_SCHEMA = "hu_m43_attempt08_preflight_plan_v1"
ATTEMPT08_PREFLIGHT_PROOF_SCHEMA = "hu_m43_attempt08_preflight_proof_v1"
ATTEMPT08_PREFLIGHT_PROOF_STATUS = "pass_proof_only_no_policy_science"
ATTEMPT08_PREFLIGHT_PLAN_SHA256 = (
    "1f0c7378f72a935ae1c27cf221cd1794627e443901e6f36a5cf7df1dd89716b1"
)
ATTEMPT08_PREFLIGHT_SOURCE_ROOTS = (0, 1, 2)
ATTEMPT08_PREFLIGHT_SEED_STRIDE = 1_000_003
ATTEMPT08_PREFLIGHT_SEED_BASES = {
    "hand": 90_108_071_901,
    "rerank": 91_108_071_901,
    "veto": 92_108_071_901,
    "stress": 93_108_071_901,
    "assessment": 94_108_071_901,
    "child": 95_108_071_901,
}
ATTEMPT08_PREFLIGHT_NATIVE_BATCH_THREADS = 4
ATTEMPT08_PREFLIGHT_SLOTS: dict[str, tuple[int, bool]] = {
    "root0_batch_a": (0, True),
    "root0_batch_b": (0, True),
    "root0_scalar": (0, False),
    "root1_batch": (1, True),
    "root2_batch": (2, True),
}

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREFLIGHT_PLAN_PATH = (
    _REPO_ROOT / "configs" / "hu_joint_policy_m43_attempt08_preflight.json"
)
DEFAULT_ATTEMPT08_PLAN_PATH = (
    _REPO_ROOT / "configs" / "hu_joint_policy_m43_attempt08.json"
)
DEFAULT_SOURCE_PATH = _REPO_ROOT / (
    "outputs/hu_joint_policy/m43_attempt06_search_quality/"
    "regular-hu-m43-attempt06-preflight-final-20260714-1154/"
    "merged/teacher.jsonl"
)
DEFAULT_MODEL_PATH = _REPO_ROOT / (
    "outputs/hu_joint_policy/m43_attempt05_old_dev900_architecture_once/"
    "lambda_rank_candidate.pkl"
)
DEFAULT_AI_PROFILES_PATH = _REPO_ROOT / "src" / "ofc_regular" / "ai_profiles.py"

_EXPECTED_TOP_KEYS = {
    "schema",
    "status",
    "source",
    "target",
    "seed_contract",
    "execution",
    "correctness_go_no_go",
    "output_contract",
    "finalization_contract",
}


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Attempt08 preflight {label} must be a mapping")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        raise ValueError(f"Attempt08 preflight {label} must be a sequence")
    return value


def load_preflight_plan(
    path: str | Path = DEFAULT_PREFLIGHT_PLAN_PATH,
) -> dict[str, Any]:
    """Load the byte-frozen preflight plan and verify its critical boundaries."""

    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != ATTEMPT08_PREFLIGHT_PLAN_SHA256:
        raise ValueError("Attempt08 preflight plan SHA-256 changed")
    payload = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(payload, dict) or set(payload) != _EXPECTED_TOP_KEYS:
        raise ValueError("Attempt08 preflight plan fields changed")
    source = _mapping(payload.get("source"), "source")
    target = _mapping(payload.get("target"), "target")
    execution = _mapping(payload.get("execution"), "execution")
    gates = _mapping(payload.get("correctness_go_no_go"), "gates")
    output = _mapping(payload.get("output_contract"), "output")
    frozen_slots = [
        {
            "name": name,
            "source_root_index": root,
            "batch_child_selectors": batch,
        }
        for name, (root, batch) in ATTEMPT08_PREFLIGHT_SLOTS.items()
    ]
    if (
        payload.get("schema") != ATTEMPT08_PREFLIGHT_PLAN_SCHEMA
        or payload.get("status")
        != "frozen_bounded_correctness_preflight_only"
        or source.get("sha256") != ATTEMPT07_PREFLIGHT_SOURCE_SHA256
        or source.get("plan_sha256") != ATTEMPT06_SOURCE_PLAN_SHA256
        or source.get("allowed_source_root_indices") != [0, 1, 2]
        or source.get("source_already_consumed_development_evidence") is not True
        or source.get("new_root_generation_allowed") is not False
        or target.get("plan_sha256") != M43_ATTEMPT08_PLAN_SHA256
        or target.get("teacher_schema") != ATTEMPT08_TEACHER_SCHEMA
        or target.get("model_sha256") != ATTEMPT08_LAMBDA_MODEL_SHA256
        or target.get("ai_profiles_sha256") != AI_PROFILES_SHA256
        or target.get("t2_policy_id") != ATTEMPT06_T2_POLICY_ID
        or target.get("t2_resolution") != "explicit_profile_never_current"
        or execution.get("machine_type") != "c4-highmem-4"
        or execution.get("native_batch_threads")
        != ATTEMPT08_PREFLIGHT_NATIVE_BATCH_THREADS
        or execution.get("slots") != frozen_slots
        or execution.get("fresh_seed_or_root_open_allowed") is not False
        or execution.get("development200_authorized") is not False
        or gates.get("teacher_elapsed_seconds_each_run_max") != 2400.0
        or gates.get("process_peak_rss_bytes_each_run_max") != 30_064_771_072
        or gates.get("operational_metrics_are_policy_science_input") is not False
        or gates.get("threshold_reselection_after_results_allowed") is not False
        or output.get("teacher_action_or_value_details_exported") is not False
        or output.get("aggregate_opaque_teacher_hashes_exported") is not False
        or output.get("aggregate_teacher_action_or_value_details_exported")
        is not False
        or output.get("runtime_activation_allowed") is not False
        or output.get("current_profile_mutation_allowed") is not False
        or output.get("development200_authorization_only_after_finalizer_go")
        is not True
    ):
        raise ValueError("Attempt08 preflight plan boundary changed")
    return payload


def attempt08_preflight_seeds(source_root_index: int) -> dict[str, int]:
    if (
        type(source_root_index) is not int
        or source_root_index not in ATTEMPT08_PREFLIGHT_SOURCE_ROOTS
    ):
        raise ValueError("Attempt08 preflight source root must be one of 0, 1, 2")
    return {
        domain: base + ATTEMPT08_PREFLIGHT_SEED_STRIDE * source_root_index
        for domain, base in ATTEMPT08_PREFLIGHT_SEED_BASES.items()
    }


def validate_preflight_seed_disjointness(
    preflight_plan: Mapping[str, Any], attempt08_plan: Mapping[str, Any]
) -> dict[str, tuple[int, ...]]:
    """Prove the six proof schedules avoid every declared/opened prior schedule."""

    if preflight_plan.get("schema") != ATTEMPT08_PREFLIGHT_PLAN_SCHEMA:
        raise ValueError("Attempt08 preflight plan changed before seed proof")
    schedules = {
        domain: tuple(
            base + ATTEMPT08_PREFLIGHT_SEED_STRIDE * root_index
            for root_index in ATTEMPT08_PREFLIGHT_SOURCE_ROOTS
        )
        for domain, base in ATTEMPT08_PREFLIGHT_SEED_BASES.items()
    }
    names = tuple(schedules)
    for index, left_name in enumerate(names):
        left = set(schedules[left_name])
        if len(left) != 3:
            raise ValueError(f"Attempt08 preflight {left_name} is not unique")
        for right_name in names[index + 1 :]:
            if left.intersection(schedules[right_name]):
                raise ValueError(
                    f"Attempt08 preflight {left_name}/{right_name} overlap"
                )

    prior: dict[str, tuple[int, ...]] = {}
    for prefix, group in (
        ("attempt06", enumerate_attempt06_known_seed_schedules()),
        ("attempt07", enumerate_attempt07_known_seed_schedules()),
        (
            "attempt07_preflight",
            enumerate_attempt07_preflight_known_seed_schedules(),
        ),
    ):
        prior.update({f"{prefix}.{name}": values for name, values in group.items()})
    for population in ("development", "future_audit"):
        prior.update(
            {
                f"attempt08.{population}.{name}": values
                for name, values in enumerate_attempt08_seed_schedules(
                    attempt08_plan, population=population
                ).items()
            }
        )
    for name, values in schedules.items():
        current = set(values)
        for prior_name, prior_values in prior.items():
            if current.intersection(prior_values):
                raise ValueError(
                    f"Attempt08 preflight {name} overlaps {prior_name}"
                )
    return schedules


def _semantic_parity_sha256(result: Mapping[str, Any]) -> str:
    """Hash teacher semantics after normalizing only scalar/batch execution mode."""

    normalized = copy.deepcopy(dict(result))
    search = normalized.get("search_config")
    if not isinstance(search, dict) or type(search.get("batch_child_selectors")) is not bool:
        raise ValueError("Attempt08 preflight parity metadata is incomplete")
    search["batch_child_selectors"] = "normalized_execution_mode"
    return _sha256_value(normalized)


def _process_peak_rss_bytes() -> int:
    """Return the OS process high-water RSS in bytes without a third-party package."""

    if os.name == "nt":
        import ctypes
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

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        process = ctypes.windll.kernel32.GetCurrentProcess()
        if not ctypes.windll.psapi.GetProcessMemoryInfo(
            process, ctypes.byref(counters), counters.cb
        ):
            raise OSError("GetProcessMemoryInfo failed")
        return int(counters.PeakWorkingSetSize)

    import resource

    high_water = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return high_water if sys.platform == "darwin" else high_water * 1024


def _validate_attempt08_result(
    result: Mapping[str, Any],
    *,
    observation: Any,
    baseline_action_key: str,
    config: Attempt08TeacherConfig,
) -> dict[str, bool]:
    """Run the exact teacher reference validator and proof-only safety checks."""

    normalized = validate_attempt08_teacher_output(
        observation,
        baseline_action_key=baseline_action_key,
        payload=result,
        config=config,
    )
    search = _mapping(result.get("search_config"), "teacher.search_config")
    frozen = _mapping(
        result.get("frozen_candidate_generator"), "teacher.frozen_candidate_generator"
    )
    continuation = _mapping(
        result.get("continuation_policy"), "teacher.continuation_policy"
    )
    expected_seeds = {
        "hand_seed": config.hand_seed,
        "rerank_seed": config.rerank_seed,
        "veto_seed": config.veto_seed,
        "stress_seed": config.stress_seed,
        "assessment_seed": config.assessment_seed,
        "child_policy_seed": config.child_policy_seed,
        "batch_child_selectors": config.batch_child_selectors,
    }
    if (
        result.get("status") != "ok"
        or result.get("schema") != ATTEMPT08_TEACHER_SCHEMA
        or result.get("solver_id") != ATTEMPT08_SOLVER_ID
        or result.get("policy_observation") != observation.to_dict()
        or any(search.get(name) != value for name, value in expected_seeds.items())
        or frozen.get("artifact_sha256") != ATTEMPT08_LAMBDA_MODEL_SHA256
        or frozen.get("runtime_authorized") is not False
        or continuation.get("t2_policy_id") != ATTEMPT06_T2_POLICY_ID
        or continuation.get("t2_resolution") != "explicit_profile_never_current"
        or result.get("teacher_value_status") != "diagnostic_not_match_EV"
        or result.get("runtime_gate_allowed") is not False
        or result.get("profile_activation_allowed") is not False
        or result.get("current_profile_resolved") is not False
        or result.get("development_only") is not True
    ):
        raise ValueError("Attempt08 preflight teacher boundary changed")
    encoded = json.dumps(result, sort_keys=True)
    if "opponent_private_discards" in encoded:
        raise ValueError("Attempt08 preflight teacher leaked hidden information")
    rng = _mapping(normalized.get("rng_key_digests"), "normalized rng")
    flat_rng = [str(value) for values in rng.values() for value in _sequence(values, "rng")]
    if not flat_rng or len(flat_rng) != len(set(flat_rng)):
        raise ValueError("Attempt08 preflight RNG separation changed")
    ActionKey.from_token(str(normalized["selected_action_key"]))
    return {
        "exact_actionkey_reference_parity_verified": True,
        "hidden_information_safety_verified": True,
        "rng_domain_separation_verified": True,
        "conditional_X_A_skip_contract_verified": True,
    }


def run_preflight_slot(
    *,
    slot: str,
    output: str | Path,
    source: str | Path = DEFAULT_SOURCE_PATH,
    preflight_plan: str | Path = DEFAULT_PREFLIGHT_PLAN_PATH,
    attempt08_plan: str | Path = DEFAULT_ATTEMPT08_PLAN_PATH,
    model: str | Path = DEFAULT_MODEL_PATH,
    ai_profiles: str | Path = DEFAULT_AI_PROFILES_PATH,
    runtime_model_manifest: str | Path = DEFAULT_MODEL_MANIFEST_PATH,
    runtime_native_manifest: str | Path = DEFAULT_NATIVE_MANIFEST_PATH,
    runtime_artifact_root: str | Path = DEFAULT_RUNTIME_ARTIFACT_ROOT,
    runtime_requirements: str | Path = DEFAULT_REQUIREMENTS_PATH,
    paths: ModelPaths | None = None,
) -> dict[str, Any]:
    """Replay one frozen slot and atomically publish a redacted proof row."""

    if slot not in ATTEMPT08_PREFLIGHT_SLOTS:
        raise ValueError("Attempt08 preflight slot is not frozen")
    source_root_index, batch_child_selectors = ATTEMPT08_PREFLIGHT_SLOTS[slot]
    output_path = Path(output)
    protected = tuple(
        Path(value)
        for value in (
            source,
            preflight_plan,
            attempt08_plan,
            model,
            ai_profiles,
            runtime_model_manifest,
            runtime_native_manifest,
            runtime_artifact_root,
            runtime_requirements,
        )
    )
    output_identity = os.path.normcase(str(output_path.resolve(strict=False)))
    if any(
        output_identity == os.path.normcase(str(path.resolve(strict=False)))
        for path in protected
    ):
        raise ValueError("Attempt08 preflight output aliases an immutable input")
    if output_path.exists():
        raise FileExistsError(f"Attempt08 preflight output exists: {output_path}")

    frozen_preflight = load_preflight_plan(preflight_plan)
    frozen_attempt08 = load_and_validate_attempt08_plan(attempt08_plan)
    if _sha256_file(attempt08_plan) != M43_ATTEMPT08_PLAN_SHA256:
        raise ValueError("Attempt08 target plan SHA-256 changed")
    validate_attempt08_artifact_bindings(
        frozen_attempt08, repository_root=Path(attempt08_plan).resolve().parents[1]
    )
    validate_preflight_seed_disjointness(frozen_preflight, frozen_attempt08)
    if _sha256_file(model) != ATTEMPT08_LAMBDA_MODEL_SHA256:
        raise ValueError("Attempt08 preflight Lambda artifact changed")
    if _sha256_file(ai_profiles) != AI_PROFILES_SHA256:
        raise ValueError("Attempt08 preflight ai_profiles.py changed")
    repository_root = Path(attempt08_plan).resolve().parents[1]
    validate_runtime_semantic_anchor(
        repository_root=repository_root,
        expected_source_closure_sha256=(
            ATTEMPT08_RUNTIME_SOURCE_CLOSURE_SHA256
        ),
        expected_anchor_sha256=ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SHA256,
        expected_source_file_count=ATTEMPT08_RUNTIME_SOURCE_FILE_COUNT,
        runtime_artifact_root=runtime_artifact_root,
        model_manifest_path=runtime_model_manifest,
        native_manifest_path=runtime_native_manifest,
        requirements_path=runtime_requirements,
    )
    runtime_fingerprint = validate_expected_runtime_fingerprint()
    if runtime_fingerprint != ATTEMPT08_EXPECTED_RUNTIME_FINGERPRINT_SHA256:
        raise ValueError("Attempt08 external runtime fingerprint changed")

    source_row, observation = load_source_row(source, source_root_index)
    baseline_action_key = str(source_row["baseline_action_key"])
    ActionKey.from_token(baseline_action_key)
    seeds = attempt08_preflight_seeds(source_root_index)
    bundle = load_model_bundle(
        paths or ModelPaths(), profiles={ATTEMPT06_T2_POLICY_ID}
    )
    continuation_seeds = {
        "first": seeds["child"],
        "second": seeds["child"] + 1,
    }
    t2_policies = {
        seat: build_policy(
            ATTEMPT06_T2_POLICY_ID,
            bundle,
            seed=continuation_seeds[seat],
            seat=seat,
            opening_lookahead_samples=0,
        )
        for seat in ("first", "second")
    }
    _require_stage9f_p2_policies(t2_policies)
    ranker = FrozenAttempt08LambdaRanker.load(
        model, expected_sha256=ATTEMPT08_LAMBDA_MODEL_SHA256
    )
    run_id = (
        f"attempt08-preflight:source-root={source_root_index}:"
        f"obs={observation.fingerprint()}"
    )
    config = Attempt08TeacherConfig(
        frozen_model_sha256=ATTEMPT08_LAMBDA_MODEL_SHA256,
        hand_seed=seeds["hand"],
        rerank_seed=seeds["rerank"],
        veto_seed=seeds["veto"],
        stress_seed=seeds["stress"],
        assessment_seed=seeds["assessment"],
        child_policy_seed=seeds["child"],
        run_id=run_id,
        batch_child_selectors=batch_child_selectors,
    )
    with _native_batch_threads(
        batch_child_selectors, ATTEMPT08_PREFLIGHT_NATIVE_BATCH_THREADS
    ):
        started = time.perf_counter()
        result = evaluate_attempt08_t1_second(
            observation,
            baseline_action_key=baseline_action_key,
            ranker=ranker,
            t2_policies=t2_policies,
            config=config,
        )
        elapsed_seconds = time.perf_counter() - started
    peak_rss_bytes = _process_peak_rss_bytes()
    if elapsed_seconds <= 0.0 or peak_rss_bytes <= 0:
        raise ValueError("Attempt08 preflight operational measurement is invalid")
    correctness = _validate_attempt08_result(
        result,
        observation=observation,
        baseline_action_key=baseline_action_key,
        config=config,
    )

    proof = {
        "schema": ATTEMPT08_PREFLIGHT_PROOF_SCHEMA,
        "status": ATTEMPT08_PREFLIGHT_PROOF_STATUS,
        "slot": slot,
        "source": {
            "merged_sha256": ATTEMPT07_PREFLIGHT_SOURCE_SHA256,
            "source_root_index": source_root_index,
            "source_row_sha256": _sha256_value(source_row),
            "wrapper_schema": ATTEMPT06_SHARD_ROW_SCHEMA,
            "teacher_schema": ATTEMPT06_TEACHER_SCHEMA,
            "observation_fingerprint": observation.fingerprint(),
            "source_already_consumed": True,
            "new_root_generated": False,
        },
        "contract": {
            "attempt08_plan_schema": M43_ATTEMPT08_PLAN_SCHEMA,
            "attempt08_plan_sha256": M43_ATTEMPT08_PLAN_SHA256,
            "preflight_plan_schema": ATTEMPT08_PREFLIGHT_PLAN_SCHEMA,
            "preflight_plan_sha256": ATTEMPT08_PREFLIGHT_PLAN_SHA256,
            "model_sha256": ATTEMPT08_LAMBDA_MODEL_SHA256,
            "ai_profiles_sha256": AI_PROFILES_SHA256,
            "runtime_semantic_anchor_sha256": (
                ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SHA256
            ),
            "runtime_source_closure_sha256": (
                ATTEMPT08_RUNTIME_SOURCE_CLOSURE_SHA256
            ),
            "runtime_requirements_sha256": ATTEMPT08_RUNTIME_REQUIREMENTS_SHA256,
            "gcp_image_name": ATTEMPT08_GCP_IMAGE_NAME,
            "gcp_image_id": ATTEMPT08_GCP_IMAGE_ID,
            "t2_policy_id": ATTEMPT06_T2_POLICY_ID,
            "t2_resolution": "explicit_profile_never_current",
            "seeds": seeds,
            "continuation_policy_seeds": continuation_seeds,
        },
        "execution": {
            "batch_child_selectors": batch_child_selectors,
            "native_batch_threads": ATTEMPT08_PREFLIGHT_NATIVE_BATCH_THREADS,
            "run_id": run_id,
            "teacher_elapsed_seconds": float(elapsed_seconds),
            "process_peak_rss_bytes": int(peak_rss_bytes),
            "runtime_fingerprint_sha256": runtime_fingerprint,
            "measurement_scope": (
                "elapsed_is_teacher_call_only_rss_is_one_root_process_high_water"
            ),
        },
        "result_proof": {
            "teacher_schema": ATTEMPT08_TEACHER_SCHEMA,
            "solver_id": ATTEMPT08_SOLVER_ID,
            "opaque_teacher_sha256": _sha256_value(result),
            "semantic_parity_sha256": _semantic_parity_sha256(result),
            **correctness,
            "teacher_action_or_value_details_exported": False,
        },
        "science_boundary": {
            "proof_only_not_policy_science": True,
            "development200_authorized": False,
            "future_audit_authorized": False,
            "fit_allowed": False,
            "threshold_selection_allowed": False,
            "runtime_activation_allowed": False,
            "current_profile_resolved": False,
            "current_profile_mutated": False,
            "fresh_seed_or_root_opened": False,
        },
    }
    _atomic_write_once(output_path, canonical_json_bytes(proof))
    return proof


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slot", required=True, choices=tuple(ATTEMPT08_PREFLIGHT_SLOTS))
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", default=str(DEFAULT_SOURCE_PATH))
    parser.add_argument("--preflight-plan", default=str(DEFAULT_PREFLIGHT_PLAN_PATH))
    parser.add_argument("--attempt08-plan", default=str(DEFAULT_ATTEMPT08_PLAN_PATH))
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--ai-profiles", default=str(DEFAULT_AI_PROFILES_PATH))
    parser.add_argument(
        "--runtime-model-manifest", default=str(DEFAULT_MODEL_MANIFEST_PATH)
    )
    parser.add_argument(
        "--runtime-native-manifest", default=str(DEFAULT_NATIVE_MANIFEST_PATH)
    )
    parser.add_argument(
        "--runtime-artifact-root", default=str(DEFAULT_RUNTIME_ARTIFACT_ROOT)
    )
    parser.add_argument(
        "--runtime-requirements", default=str(DEFAULT_REQUIREMENTS_PATH)
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_preflight_slot(
        slot=args.slot,
        output=args.output,
        source=args.source,
        preflight_plan=args.preflight_plan,
        attempt08_plan=args.attempt08_plan,
        model=args.model,
        ai_profiles=args.ai_profiles,
        runtime_model_manifest=args.runtime_model_manifest,
        runtime_native_manifest=args.runtime_native_manifest,
        runtime_artifact_root=args.runtime_artifact_root,
        runtime_requirements=args.runtime_requirements,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ATTEMPT08_PREFLIGHT_NATIVE_BATCH_THREADS",
    "ATTEMPT08_PREFLIGHT_PLAN_SCHEMA",
    "ATTEMPT08_PREFLIGHT_PLAN_SHA256",
    "ATTEMPT08_PREFLIGHT_PROOF_SCHEMA",
    "ATTEMPT08_PREFLIGHT_PROOF_STATUS",
    "ATTEMPT08_PREFLIGHT_SEED_BASES",
    "ATTEMPT08_PREFLIGHT_SEED_STRIDE",
    "ATTEMPT08_PREFLIGHT_SLOTS",
    "ATTEMPT08_RUNTIME_SEMANTIC_ANCHOR_SHA256",
    "ATTEMPT08_RUNTIME_SOURCE_CLOSURE_SHA256",
    "DEFAULT_ATTEMPT08_PLAN_PATH",
    "DEFAULT_PREFLIGHT_PLAN_PATH",
    "DEFAULT_SOURCE_PATH",
    "attempt08_preflight_seeds",
    "canonical_json_bytes",
    "load_preflight_plan",
    "run_preflight_slot",
    "validate_preflight_seed_disjointness",
]
