"""Run one restart-safe Attempt12 search root.

The persistence and root-generation mechanics are the already-tested
Attempt09 runner.  This module binds that generic machinery to the frozen
Attempt12 plan and teacher for the duration of one call, then restores every
binding.  The lock prevents mixed-contract calls in a shared Python process;
normal shard execution is one process per root.
"""

from __future__ import annotations

import argparse
import json
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

from .ai_profiles import ModelPaths
from . import run_hu_m43_attempt09 as _base
from .hu_m43_attempt12_contract import (
    AI_PROFILES_SHA256,
    ATTEMPT12_LAMBDA_MODEL_SHA256,
    M43_ATTEMPT12_PLAN_SHA256,
    M43_ATTEMPT12_PROFILES,
    enumerate_attempt12_seed_schedules,
    load_and_validate_attempt12_plan,
    validate_attempt12_artifact_bindings,
)
from .hu_m43_attempt12_teacher import (
    ATTEMPT12_TEACHER_SCHEMA,
    Attempt12TeacherConfig,
    FrozenAttempt12LambdaRanker,
    evaluate_attempt12_t1_second,
    validate_attempt12_teacher_output,
)


ATTEMPT12_ROW_SCHEMA = "hu_m43_attempt12_search_root_v1"
ATTEMPT12_CHECKPOINT_SCHEMA = "hu_m43_attempt12_search_checkpoint_v1"
ATTEMPT12_HEARTBEAT_SCHEMA = "hu_m43_attempt12_search_heartbeat_v1"
ATTEMPT12_SUMMARY_SCHEMA = "hu_m43_attempt12_search_summary_v1"
ATTEMPT12_AUTHORIZATION_SCHEMA = "hu_m43_attempt12_execution_authorization_v1"
ATTEMPT12_ROOT_CONTRACT_SCHEMA = "hu_m43_attempt12_root_contract_v1"
ATTEMPT12_BASELINE_PROFILE = "stage18_p1"
ATTEMPT12_CONTINUATION_PROFILE = "stage9f_p2"
ATTEMPT12_NATIVE_BATCH_THREADS = 4

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN = _REPO_ROOT / "configs" / "hu_joint_policy_m43_attempt12.json"
DEFAULT_AI_PROFILES = _REPO_ROOT / "src" / "ofc_regular" / "ai_profiles.py"
_MODES = ("preflight", "development", "future_audit")
_BIND_LOCK = threading.RLock()


_ATTEMPT12_BINDINGS: dict[str, Any] = {
    "AI_PROFILES_SHA256": AI_PROFILES_SHA256,
    "ATTEMPT09_LAMBDA_MODEL_SHA256": ATTEMPT12_LAMBDA_MODEL_SHA256,
    "M43_ATTEMPT09_PLAN_SHA256": M43_ATTEMPT12_PLAN_SHA256,
    "M43_ATTEMPT09_PROFILES": M43_ATTEMPT12_PROFILES,
    "enumerate_attempt09_seed_schedules": enumerate_attempt12_seed_schedules,
    "load_and_validate_attempt09_plan": load_and_validate_attempt12_plan,
    "validate_attempt09_artifact_bindings": validate_attempt12_artifact_bindings,
    "ATTEMPT09_TEACHER_SCHEMA": ATTEMPT12_TEACHER_SCHEMA,
    "Attempt09TeacherConfig": Attempt12TeacherConfig,
    "FrozenAttempt09LambdaRanker": FrozenAttempt12LambdaRanker,
    "evaluate_attempt09_t1_second": evaluate_attempt12_t1_second,
    "validate_attempt09_teacher_output": validate_attempt12_teacher_output,
    "ATTEMPT09_ROW_SCHEMA": ATTEMPT12_ROW_SCHEMA,
    "ATTEMPT09_CHECKPOINT_SCHEMA": ATTEMPT12_CHECKPOINT_SCHEMA,
    "ATTEMPT09_HEARTBEAT_SCHEMA": ATTEMPT12_HEARTBEAT_SCHEMA,
    "ATTEMPT09_SUMMARY_SCHEMA": ATTEMPT12_SUMMARY_SCHEMA,
    "ATTEMPT09_AUTHORIZATION_SCHEMA": ATTEMPT12_AUTHORIZATION_SCHEMA,
    "ATTEMPT09_ROOT_CONTRACT_SCHEMA": ATTEMPT12_ROOT_CONTRACT_SCHEMA,
    "ATTEMPT09_BASELINE_PROFILE": ATTEMPT12_BASELINE_PROFILE,
    "ATTEMPT09_CONTINUATION_PROFILE": ATTEMPT12_CONTINUATION_PROFILE,
    "ATTEMPT09_NATIVE_BATCH_THREADS": ATTEMPT12_NATIVE_BATCH_THREADS,
}


@contextmanager
def _attempt12_bindings() -> Iterator[None]:
    with _BIND_LOCK:
        prior = {name: getattr(_base, name) for name in _ATTEMPT12_BINDINGS}
        try:
            for name, value in _ATTEMPT12_BINDINGS.items():
                setattr(_base, name, value)
            yield
        finally:
            for name, value in prior.items():
                setattr(_base, name, value)


def run_attempt12_root(
    *,
    mode: str,
    root_index: int,
    output: str | Path,
    checkpoint: str | Path,
    heartbeat: str | Path,
    model: str | Path,
    model_sha256: str,
    run_id: str,
    plan: str | Path = DEFAULT_PLAN,
    ai_profiles: str | Path = DEFAULT_AI_PROFILES,
    authorization: str | Path | None = None,
    source_package_sha256: str | None = None,
    batch_child_selectors: bool = True,
    native_batch_threads: int = ATTEMPT12_NATIVE_BATCH_THREADS,
    paths: ModelPaths | None = None,
) -> dict[str, Any]:
    """Generate or validate exactly one deterministic Attempt12 row."""

    with _attempt12_bindings():
        return _base.run_attempt09_root(
            mode=mode,
            root_index=root_index,
            output=output,
            checkpoint=checkpoint,
            heartbeat=heartbeat,
            model=model,
            model_sha256=model_sha256,
            run_id=run_id,
            plan=plan,
            ai_profiles=ai_profiles,
            authorization=authorization,
            source_package_sha256=source_package_sha256,
            batch_child_selectors=batch_child_selectors,
            native_batch_threads=native_batch_threads,
            paths=paths,
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=_MODES, required=True)
    parser.add_argument("--root-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--heartbeat", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--ai-profiles", type=Path, default=DEFAULT_AI_PROFILES)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--source-package-sha256")
    parser.add_argument("--batch-child-selectors", action="store_true")
    parser.add_argument("--native-batch-threads", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    summary = run_attempt12_root(
        mode=args.mode,
        root_index=args.root_index,
        output=args.output,
        checkpoint=args.checkpoint,
        heartbeat=args.heartbeat,
        model=args.model,
        model_sha256=args.model_sha256,
        run_id=args.run_id,
        plan=args.plan,
        ai_profiles=args.ai_profiles,
        authorization=args.authorization,
        source_package_sha256=args.source_package_sha256,
        batch_child_selectors=args.batch_child_selectors,
        native_batch_threads=args.native_batch_threads,
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "ATTEMPT12_AUTHORIZATION_SCHEMA",
    "ATTEMPT12_CHECKPOINT_SCHEMA",
    "ATTEMPT12_HEARTBEAT_SCHEMA",
    "ATTEMPT12_ROOT_CONTRACT_SCHEMA",
    "ATTEMPT12_ROW_SCHEMA",
    "ATTEMPT12_SUMMARY_SCHEMA",
    "run_attempt12_root",
]

