"""Bounded Attempt07 preflight over three already-opened Attempt06 roots.

This harness never deals a card or opens an Attempt07 development/audit root.
It accepts only source roots 0, 1, and 2 from the immutable Attempt06 merged
file, runs the Attempt07 evaluator, and emits an opaque determinism proof.  No
per-arm score, action, veto result, or assessment value is exported, so the
preflight output cannot be used to select an Attempt07 arm.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from .action_key import ActionKey
from .ai_profiles import ModelPaths, build_policy, load_model_bundle
from .audit_hu_m43_attempt06_search_quality import (
    validate_attempt06_teacher_row,
)
from .hu_infoset import ActorObservation
from .hu_m43_attempt06_teacher import (
    ATTEMPT06_FROZEN_MODEL_SHA256,
    ATTEMPT06_SHARD_ROW_SCHEMA,
    ATTEMPT06_T2_POLICY_ID,
    ATTEMPT06_TEACHER_SCHEMA,
    FrozenAttempt06LambdaRanker,
    _require_stage9f_p2_policies,
)
from .hu_m43_attempt07_contract import (
    AI_PROFILES_SHA256,
    M43_ATTEMPT07_PLAN_SHA256,
    enumerate_attempt06_known_seed_schedules,
    enumerate_attempt07_seed_schedules,
    load_and_validate_attempt07_plan,
)
from .hu_m43_attempt07_teacher import (
    ATTEMPT07_SOLVER_ID,
    ATTEMPT07_TEACHER_SCHEMA,
    Attempt07TeacherConfig,
    evaluate_attempt07_t1_second,
)


ATTEMPT07_PREFLIGHT_PLAN_SCHEMA = "hu_m43_attempt07_preflight_plan_v1"
ATTEMPT07_PREFLIGHT_ROW_SCHEMA = "hu_m43_attempt07_preflight_row_v1"
ATTEMPT07_PREFLIGHT_STATUS = "pass_preflight_only_no_arm_selection"
ATTEMPT07_PREFLIGHT_SOURCE_SHA256 = (
    "6b1063589aa2ee4f3e65d9489384176a85c69abe1dfefe30ffb4474f96964903"
)
ATTEMPT07_PREFLIGHT_PLAN_SHA256 = (
    "6eba7761c6a46f6c22a22f24f2065c5bc8ea6783cc6dbea235fde027ba67f72a"
)
ATTEMPT06_SOURCE_PLAN_SHA256 = (
    "4844fb970780c04ff093eb43b1672e403f006515c47b287e6abdbea17867f5b8"
)
ATTEMPT07_PREFLIGHT_SOURCE_ROOTS = (0, 1, 2)
ATTEMPT07_PREFLIGHT_SEED_STRIDE = 1_000_003
ATTEMPT07_PREFLIGHT_SEED_BASES = {
    "screen": 71_106_071_901,
    "rerank": 72_106_071_901,
    "veto": 73_106_071_901,
    "assessment": 74_106_071_901,
    "child": 75_106_071_901,
}
ATTEMPT07_PREFLIGHT_NATIVE_BATCH_THREADS = 4

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREFLIGHT_PLAN_PATH = (
    _REPO_ROOT / "configs" / "hu_joint_policy_m43_attempt07_preflight.json"
)
DEFAULT_ATTEMPT07_PLAN_PATH = (
    _REPO_ROOT / "configs" / "hu_joint_policy_m43_attempt07.json"
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

_EXPECTED_PREFLIGHT_PLAN: dict[str, Any] = {
    "schema": ATTEMPT07_PREFLIGHT_PLAN_SCHEMA,
    "status": "frozen_bounded_preflight_only",
    "source": {
        "path": (
            "outputs/hu_joint_policy/m43_attempt06_search_quality/"
            "regular-hu-m43-attempt06-preflight-final-20260714-1154/"
            "merged/teacher.jsonl"
        ),
        "sha256": ATTEMPT07_PREFLIGHT_SOURCE_SHA256,
        "wrapper_schema": ATTEMPT06_SHARD_ROW_SCHEMA,
        "teacher_schema": ATTEMPT06_TEACHER_SCHEMA,
        "plan_sha256": ATTEMPT06_SOURCE_PLAN_SHA256,
        "allowed_source_root_indices": list(ATTEMPT07_PREFLIGHT_SOURCE_ROOTS),
        "new_root_generation_allowed": False,
    },
    "target": {
        "plan_path": "configs/hu_joint_policy_m43_attempt07.json",
        "plan_sha256": M43_ATTEMPT07_PLAN_SHA256,
        "teacher_schema": ATTEMPT07_TEACHER_SCHEMA,
        "model_path": (
            "outputs/hu_joint_policy/m43_attempt05_old_dev900_architecture_once/"
            "lambda_rank_candidate.pkl"
        ),
        "model_sha256": ATTEMPT06_FROZEN_MODEL_SHA256,
        "ai_profiles_path": "src/ofc_regular/ai_profiles.py",
        "ai_profiles_sha256": AI_PROFILES_SHA256,
        "t2_policy_id": ATTEMPT06_T2_POLICY_ID,
        "t2_resolution": "explicit_profile_never_current",
    },
    "seed_contract": {
        "seed_stride": ATTEMPT07_PREFLIGHT_SEED_STRIDE,
        "source_root_index_formula": (
            "namespace_seed_base + seed_stride * source_root_index"
        ),
        "screen_seed_base": ATTEMPT07_PREFLIGHT_SEED_BASES["screen"],
        "rerank_seed_base": ATTEMPT07_PREFLIGHT_SEED_BASES["rerank"],
        "veto_seed_base": ATTEMPT07_PREFLIGHT_SEED_BASES["veto"],
        "assessment_seed_base": ATTEMPT07_PREFLIGHT_SEED_BASES["assessment"],
        "child_policy_seed_base": ATTEMPT07_PREFLIGHT_SEED_BASES["child"],
        "all_preflight_slices_pairwise_disjoint": True,
        "attempt06_schedules_disjoint": True,
        "attempt07_development_and_future_audit_schedules_disjoint": True,
    },
    "execution": {
        "cli_roots_per_invocation": 1,
        "native_batch_threads": ATTEMPT07_PREFLIGHT_NATIVE_BATCH_THREADS,
        "scalar_and_batch_modes_allowed": True,
        "current_profile_resolution_allowed": False,
        "fresh_seed_or_root_open_allowed": False,
    },
    "operational_go_no_go": {
        "all_five_done_metadata_required": True,
        "proof_aggregate_go_required": True,
        "batch_elapsed_seconds_per_root_max": 900.0,
        "scalar_elapsed_seconds_root0_max": 3600.0,
        "scalar_to_root0_batch_median_speedup_min": 1.25,
        "root0_batch_replicate_elapsed_ratio_max": 2.0,
        "peak_rss_bytes_per_job_max": 12_884_901_888,
        "operational_metrics_are_policy_science_input": False,
        "threshold_reselection_after_results_allowed": False,
    },
    "output_contract": {
        "schema": ATTEMPT07_PREFLIGHT_ROW_SCHEMA,
        "canonical_json": True,
        "trailing_newline": True,
        "atomic_create_no_clobber": True,
        "timestamps_allowed": False,
        "teacher_values_exported": False,
        "arm_details_exported": False,
        "opaque_teacher_sha256_purpose": (
            "determinism_only_not_arm_selection"
        ),
        "semantic_parity_sha256_purpose": (
            "scalar_batch_exact_parity_after_normalizing_only_"
            "batch_execution_metadata"
        ),
        "arm_selection_allowed": False,
        "fit_allowed": False,
        "threshold_selection_allowed": False,
        "runtime_activation_allowed": False,
        "current_profile_mutation_allowed": False,
    },
}


def canonical_json_bytes(value: Any) -> bytes:
    """Encode one canonical JSON row with the required trailing newline."""

    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _semantic_parity_sha256(result: Mapping[str, Any]) -> str:
    """Hash all teacher semantics while ignoring only the scalar/batch flag."""

    normalized = copy.deepcopy(dict(result))
    search = normalized.get("search_config")
    if not isinstance(search, dict) or not isinstance(
        search.get("batch_child_selectors"), bool
    ):
        raise ValueError("Attempt07 preflight parity metadata is incomplete")
    search["batch_child_selectors"] = "normalized_execution_mode"
    return _sha256_value(normalized)


def load_preflight_plan(path: str | Path = DEFAULT_PREFLIGHT_PLAN_PATH) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != ATTEMPT07_PREFLIGHT_PLAN_SHA256:
        raise ValueError("Attempt07 preflight plan changed")
    payload = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(payload, dict) or payload != _EXPECTED_PREFLIGHT_PLAN:
        raise ValueError("Attempt07 preflight plan changed")
    return payload


def attempt07_preflight_seeds(source_root_index: int) -> dict[str, int]:
    if (
        isinstance(source_root_index, bool)
        or not isinstance(source_root_index, int)
        or source_root_index not in ATTEMPT07_PREFLIGHT_SOURCE_ROOTS
    ):
        raise ValueError("Attempt07 preflight source root must be one of 0, 1, 2")
    return {
        domain: base + ATTEMPT07_PREFLIGHT_SEED_STRIDE * source_root_index
        for domain, base in ATTEMPT07_PREFLIGHT_SEED_BASES.items()
    }


def validate_preflight_seed_disjointness(
    preflight_plan: Mapping[str, Any],
    attempt07_plan: Mapping[str, Any],
) -> dict[str, tuple[int, ...]]:
    if dict(preflight_plan) != _EXPECTED_PREFLIGHT_PLAN:
        raise ValueError("Attempt07 preflight plan changed before seed validation")
    schedules = {
        domain: tuple(
            base + ATTEMPT07_PREFLIGHT_SEED_STRIDE * root_index
            for root_index in ATTEMPT07_PREFLIGHT_SOURCE_ROOTS
        )
        for domain, base in ATTEMPT07_PREFLIGHT_SEED_BASES.items()
    }
    named = list(schedules.items())
    for left_index, (left_name, left_values) in enumerate(named):
        if len(set(left_values)) != len(ATTEMPT07_PREFLIGHT_SOURCE_ROOTS):
            raise ValueError(f"Attempt07 preflight {left_name} seeds are not unique")
        for right_name, right_values in named[left_index + 1 :]:
            if set(left_values).intersection(right_values):
                raise ValueError(
                    f"Attempt07 preflight {left_name}/{right_name} seeds overlap"
                )

    prior: dict[str, tuple[int, ...]] = {
        f"attempt06.{name}": values
        for name, values in enumerate_attempt06_known_seed_schedules().items()
    }
    for population in ("development", "future_audit"):
        prior.update(
            {
                f"attempt07.{population}.{name}": values
                for name, values in enumerate_attempt07_seed_schedules(
                    attempt07_plan, population=population
                ).items()
            }
        )
    for name, values in schedules.items():
        value_set = set(values)
        for prior_name, prior_values in prior.items():
            if value_set.intersection(prior_values):
                raise ValueError(
                    f"Attempt07 preflight {name} overlaps {prior_name}"
                )
    return schedules


def validate_source_row(row: Mapping[str, Any], source_root_index: int) -> ActorObservation:
    if source_root_index not in ATTEMPT07_PREFLIGHT_SOURCE_ROOTS:
        raise ValueError("Attempt07 preflight source root must be one of 0, 1, 2")
    validate_attempt06_teacher_row(row, root_index=source_root_index)
    teacher = row.get("teacher")
    raw_observation = row.get("policy_observation")
    if not isinstance(teacher, Mapping) or not isinstance(raw_observation, Mapping):
        raise ValueError("Attempt07 preflight source wrapper is incomplete")
    observation = ActorObservation.from_dict(raw_observation)
    baseline = row.get("baseline_action_key")
    if (
        row.get("schema") != ATTEMPT06_SHARD_ROW_SCHEMA
        or row.get("root_index") != source_root_index
        or teacher.get("schema") != ATTEMPT06_TEACHER_SCHEMA
        or teacher.get("observation_fingerprint") != observation.fingerprint()
        or observation.to_dict() != raw_observation
        or not isinstance(baseline, str)
        or teacher.get("baseline_action_key") != baseline
    ):
        raise ValueError("Attempt07 preflight source identity changed")
    ActionKey.from_token(baseline)
    return observation


def load_source_row(
    source_path: str | Path,
    source_root_index: int,
) -> tuple[dict[str, Any], ActorObservation]:
    path = Path(source_path)
    if _sha256_file(path) != ATTEMPT07_PREFLIGHT_SOURCE_SHA256:
        raise ValueError("Attempt07 preflight Attempt06 merged source SHA-256 changed")
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not raw.strip():
            continue
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError(
                f"Attempt07 preflight source line {line_number} is not a mapping"
            )
        rows.append(payload)
    if len(rows) != 50 or [row.get("root_index") for row in rows] != list(range(50)):
        raise ValueError("Attempt07 preflight source must contain ordered roots 0..49")
    row = rows[source_root_index]
    return row, validate_source_row(row, source_root_index)


@contextmanager
def _native_batch_threads(enabled: bool, threads: int) -> Iterator[None]:
    previous = os.environ.get("OFC_HU_M3_BATCH_THREADS")
    if enabled:
        os.environ["OFC_HU_M3_BATCH_THREADS"] = str(threads)
    else:
        os.environ.pop("OFC_HU_M3_BATCH_THREADS", None)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("OFC_HU_M3_BATCH_THREADS", None)
        else:
            os.environ["OFC_HU_M3_BATCH_THREADS"] = previous


def _validate_attempt07_result(
    result: Mapping[str, Any],
    *,
    observation: ActorObservation,
    baseline_action_key: str,
    seeds: Mapping[str, int],
    batch_child_selectors: bool,
) -> None:
    search = result.get("search_config")
    continuation = result.get("continuation_policy")
    frozen = result.get("frozen_candidate_generator")
    if not all(isinstance(value, Mapping) for value in (search, continuation, frozen)):
        raise ValueError("Attempt07 preflight evaluator metadata is incomplete")
    expected_search = {
        "screen_seed": seeds["screen"],
        "rerank_seed": seeds["rerank"],
        "veto_seed": seeds["veto"],
        "assessment_seed": seeds["assessment"],
        "child_policy_seed": seeds["child"],
        "batch_child_selectors": batch_child_selectors,
    }
    if (
        result.get("status") != "ok"
        or result.get("schema") != ATTEMPT07_TEACHER_SCHEMA
        or result.get("solver_id") != ATTEMPT07_SOLVER_ID
        or result.get("street") != "T1"
        or result.get("seat") != "second"
        or result.get("to_act_order") != "second"
        or result.get("observation_fingerprint") != observation.fingerprint()
        or result.get("policy_observation") != observation.to_dict()
        or result.get("baseline_action_key") != baseline_action_key
        or any(search.get(key) != value for key, value in expected_search.items())
        or continuation.get("t2_policy_id") != ATTEMPT06_T2_POLICY_ID
        or continuation.get("t2_resolution") != "explicit_profile_never_current"
        or frozen.get("artifact_sha256") != ATTEMPT06_FROZEN_MODEL_SHA256
        or frozen.get("runtime_authorized") is not False
        or result.get("teacher_value_status") != "diagnostic_not_match_EV"
        or result.get("runtime_gate_allowed") is not False
        or result.get("profile_activation_allowed") is not False
        or result.get("current_profile_resolved") is not False
        or result.get("development_only") is not True
    ):
        raise ValueError("Attempt07 preflight evaluator boundary changed")


def _atomic_write_once(path: Path, payload: bytes) -> None:
    """Atomically publish a complete file and fail if the target exists."""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_preflight_root(
    *,
    source_root_index: int,
    output: str | Path,
    source: str | Path = DEFAULT_SOURCE_PATH,
    preflight_plan: str | Path = DEFAULT_PREFLIGHT_PLAN_PATH,
    attempt07_plan: str | Path = DEFAULT_ATTEMPT07_PLAN_PATH,
    model: str | Path = DEFAULT_MODEL_PATH,
    ai_profiles: str | Path = DEFAULT_AI_PROFILES_PATH,
    batch_child_selectors: bool = False,
    native_batch_threads: int = ATTEMPT07_PREFLIGHT_NATIVE_BATCH_THREADS,
    paths: ModelPaths | None = None,
) -> dict[str, Any]:
    """Run one source root and atomically emit a value-redacted proof row."""

    if not isinstance(batch_child_selectors, bool):
        raise TypeError("batch_child_selectors must be a bool")
    if native_batch_threads != ATTEMPT07_PREFLIGHT_NATIVE_BATCH_THREADS:
        raise ValueError("Attempt07 preflight native batch threads are frozen at 4")
    seeds = attempt07_preflight_seeds(source_root_index)
    output_path = Path(output)
    protected = [
        Path(source),
        Path(preflight_plan),
        Path(attempt07_plan),
        Path(model),
        Path(ai_profiles),
    ]
    output_identity = os.path.normcase(str(output_path.resolve(strict=False)))
    if any(
        output_identity == os.path.normcase(str(path.resolve(strict=False)))
        for path in protected
    ):
        raise ValueError("Attempt07 preflight output aliases an immutable input")
    if output_path.exists():
        raise FileExistsError(f"Attempt07 preflight output already exists: {output_path}")

    frozen_preflight = load_preflight_plan(preflight_plan)
    plan_payload = load_and_validate_attempt07_plan(attempt07_plan)
    if _sha256_file(attempt07_plan) != M43_ATTEMPT07_PLAN_SHA256:
        raise ValueError("Attempt07 authoritative plan SHA-256 changed")
    validate_preflight_seed_disjointness(frozen_preflight, plan_payload)
    if _sha256_file(model) != ATTEMPT06_FROZEN_MODEL_SHA256:
        raise ValueError("Attempt07 preflight frozen Lambda artifact changed")
    if _sha256_file(ai_profiles) != AI_PROFILES_SHA256:
        raise ValueError("Attempt07 preflight ai_profiles.py changed")

    source_row, observation = load_source_row(source, source_root_index)
    baseline_action_key = str(source_row["baseline_action_key"])
    explicit_profiles = {ATTEMPT06_T2_POLICY_ID}
    if "current" in explicit_profiles:
        raise AssertionError("Attempt07 preflight must never resolve current")
    bundle = load_model_bundle(paths or ModelPaths(), profiles=explicit_profiles)
    continuation_policy_seeds = {
        "first": seeds["child"],
        "second": seeds["child"] + 1,
    }
    t2_policies = {
        seat: build_policy(
            ATTEMPT06_T2_POLICY_ID,
            bundle,
            seed=continuation_policy_seeds[seat],
            seat=seat,
            opening_lookahead_samples=0,
        )
        for seat in ("first", "second")
    }
    _require_stage9f_p2_policies(t2_policies)
    ranker = FrozenAttempt06LambdaRanker.load(
        model, expected_sha256=ATTEMPT06_FROZEN_MODEL_SHA256
    )
    run_id = (
        f"attempt07-preflight:source-root={source_root_index}:"
        f"obs={observation.fingerprint()}"
    )
    config = Attempt07TeacherConfig(
        frozen_model_sha256=ATTEMPT06_FROZEN_MODEL_SHA256,
        screen_seed=seeds["screen"],
        rerank_seed=seeds["rerank"],
        veto_seed=seeds["veto"],
        assessment_seed=seeds["assessment"],
        child_policy_seed=seeds["child"],
        run_id=run_id,
        batch_child_selectors=batch_child_selectors,
    )
    with _native_batch_threads(batch_child_selectors, native_batch_threads):
        result = evaluate_attempt07_t1_second(
            observation,
            baseline_action_key=baseline_action_key,
            ranker=ranker,
            t2_policies=t2_policies,
            config=config,
        )
    if not isinstance(result, Mapping):
        raise ValueError("Attempt07 preflight evaluator returned a non-mapping")
    _validate_attempt07_result(
        result,
        observation=observation,
        baseline_action_key=baseline_action_key,
        seeds=seeds,
        batch_child_selectors=batch_child_selectors,
    )

    source_row_sha256 = _sha256_value(source_row)
    teacher_sha256 = _sha256_value(result)
    semantic_parity_sha256 = _semantic_parity_sha256(result)
    row = {
        "schema": ATTEMPT07_PREFLIGHT_ROW_SCHEMA,
        "status": ATTEMPT07_PREFLIGHT_STATUS,
        "source": {
            "merged_sha256": ATTEMPT07_PREFLIGHT_SOURCE_SHA256,
            "source_root_index": source_root_index,
            "source_row_sha256": source_row_sha256,
            "wrapper_schema": ATTEMPT06_SHARD_ROW_SCHEMA,
            "teacher_schema": ATTEMPT06_TEACHER_SCHEMA,
            "observation_fingerprint": observation.fingerprint(),
            "baseline_action_key": baseline_action_key,
            "new_root_generated": False,
        },
        "contract": {
            "plan_sha256": M43_ATTEMPT07_PLAN_SHA256,
            "preflight_plan_sha256": _sha256_file(preflight_plan),
            "model_sha256": ATTEMPT06_FROZEN_MODEL_SHA256,
            "ai_profiles_sha256": AI_PROFILES_SHA256,
            "t2_policy_id": ATTEMPT06_T2_POLICY_ID,
            "t2_resolution": "explicit_profile_never_current",
            "seeds": seeds,
            "continuation_policy_seeds": continuation_policy_seeds,
        },
        "execution": {
            "batch_child_selectors": batch_child_selectors,
            "native_batch_threads": native_batch_threads,
            "run_id": run_id,
        },
        "result_proof": {
            "teacher_schema": ATTEMPT07_TEACHER_SCHEMA,
            "solver_id": ATTEMPT07_SOLVER_ID,
            "opaque_teacher_sha256": teacher_sha256,
            "opaque_teacher_sha256_purpose": (
                "determinism_only_not_arm_selection"
            ),
            "semantic_parity_sha256": semantic_parity_sha256,
            "semantic_parity_sha256_purpose": (
                "scalar_batch_exact_parity_after_normalizing_only_"
                "batch_execution_metadata"
            ),
            "teacher_values_exported": False,
            "arm_details_exported": False,
        },
        "science_boundary": {
            "arm_selection_allowed": False,
            "fit_allowed": False,
            "threshold_selection_allowed": False,
            "runtime_activation_allowed": False,
            "current_profile_resolved": False,
            "current_profile_mutated": False,
            "fresh_seed_or_root_opened": False,
        },
    }
    _atomic_write_once(output_path, canonical_json_bytes(row))
    return row


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root-index",
        required=True,
        type=int,
        choices=ATTEMPT07_PREFLIGHT_SOURCE_ROOTS,
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", default=str(DEFAULT_SOURCE_PATH))
    parser.add_argument("--preflight-plan", default=str(DEFAULT_PREFLIGHT_PLAN_PATH))
    parser.add_argument("--attempt07-plan", default=str(DEFAULT_ATTEMPT07_PLAN_PATH))
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--ai-profiles", default=str(DEFAULT_AI_PROFILES_PATH))
    parser.add_argument(
        "--batch-child-selectors",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--native-batch-threads",
        type=int,
        default=ATTEMPT07_PREFLIGHT_NATIVE_BATCH_THREADS,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run_preflight_root(
        source_root_index=args.source_root_index,
        output=args.output,
        source=args.source,
        preflight_plan=args.preflight_plan,
        attempt07_plan=args.attempt07_plan,
        model=args.model,
        ai_profiles=args.ai_profiles,
        batch_child_selectors=args.batch_child_selectors,
        native_batch_threads=args.native_batch_threads,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "ATTEMPT07_PREFLIGHT_PLAN_SHA256",
    "ATTEMPT07_PREFLIGHT_PLAN_SCHEMA",
    "ATTEMPT07_PREFLIGHT_ROW_SCHEMA",
    "ATTEMPT07_PREFLIGHT_SEED_BASES",
    "ATTEMPT07_PREFLIGHT_SEED_STRIDE",
    "ATTEMPT07_PREFLIGHT_SOURCE_ROOTS",
    "attempt07_preflight_seeds",
    "canonical_json_bytes",
    "load_preflight_plan",
    "load_source_row",
    "main",
    "run_preflight_root",
    "validate_preflight_seed_disjointness",
    "validate_source_row",
]
