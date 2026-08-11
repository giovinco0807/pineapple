"""Apply the frozen Attempt11 Development200 gates exactly once."""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from . import select_hu_m43_attempt09_development as _base
from .hu_m43_attempt11_contract import (
    AI_PROFILES_SHA256,
    ATTEMPT11_LAMBDA_MODEL_SHA256,
    M43_ATTEMPT11_PLAN_SHA256,
    M43_ATTEMPT11_PROFILES,
    enumerate_attempt11_seed_schedules,
    load_and_validate_attempt11_plan,
    validate_attempt11_plan,
)
from .hu_m43_attempt11_teacher import Attempt11TeacherConfig, validate_attempt11_teacher_output
from .run_hu_m43_attempt11 import (
    ATTEMPT11_AUTHORIZATION_SCHEMA,
    ATTEMPT11_BASELINE_PROFILE,
    ATTEMPT11_CONTINUATION_PROFILE,
    ATTEMPT11_ROOT_CONTRACT_SCHEMA,
    ATTEMPT11_ROW_SCHEMA,
    _attempt11_bindings as _runner_bindings,
)


ATTEMPT11_DEVELOPMENT_DECISION_SCHEMA = "hu_m43_attempt11_development_go_no_go_v1"
ATTEMPT11_DEVELOPMENT_RECEIPT_SCHEMA = (
    "hu_m43_attempt11_development_selector_receipt_v1"
)
ATTEMPT11_TEACHER_VALIDATION_SCHEMA = "hu_m43_attempt11_teacher_validation_v1"
_BIND_LOCK = threading.RLock()
_ATTEMPT11_GATE_RELATIVE = "artifacts/attempt11/preceding_gate.json"


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _attempt11_authorization(
    path: str | Path | None,
    *,
    mode: str,
    source_package_sha256: str | None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Validate authorization without resolving its gate against process CWD."""

    if mode == "preflight":
        if path is not None:
            raise ValueError("Attempt11 preflight must not consume later authorization")
        return None, None
    if path is None:
        raise ValueError(f"Attempt11 {mode} requires explicit authorization")
    if mode not in {"development", "future_audit"}:
        raise ValueError(f"Attempt11 authorization mode is unsupported: {mode}")
    target = Path(path).resolve()
    raw = target.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Attempt11 execution authorization is invalid JSON") from exc
    canonical = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    if not isinstance(payload, dict) or raw != canonical:
        raise ValueError("Attempt11 execution authorization is not canonical JSON")
    expected_keys = {
        "schema",
        "status",
        "mode",
        "plan_sha256",
        "source_package_sha256",
        "preceding_gate_artifact",
        "preceding_gate_sha256",
        "root_index_first",
        "root_index_last",
        "current_profile_mutated",
        "runtime_policy_activated",
    }
    expected_range = (0, 199) if mode == "development" else (200, 249)
    if (
        set(payload) != expected_keys
        or payload.get("schema")
        != ATTEMPT11_AUTHORIZATION_SCHEMA
        or payload.get("status") != "authorized"
        or payload.get("mode") != mode
        or payload.get("plan_sha256") != M43_ATTEMPT11_PLAN_SHA256
        or payload.get("root_index_first") != expected_range[0]
        or payload.get("root_index_last") != expected_range[1]
        or payload.get("current_profile_mutated") is not False
        or payload.get("runtime_policy_activated") is not False
    ):
        raise ValueError("Attempt11 authorization boundary changed")
    package_hash = payload.get("source_package_sha256")
    if not _is_sha256(package_hash) or source_package_sha256 != package_hash:
        raise ValueError("Attempt11 source package authorization changed")
    declared_gate = payload.get("preceding_gate_artifact")
    gate_hash = payload.get("preceding_gate_sha256")
    if declared_gate != _ATTEMPT11_GATE_RELATIVE or not _is_sha256(gate_hash):
        raise ValueError("Attempt11 preceding gate declaration changed")

    # On a VM, execution_authorization.json and artifacts/ are siblings.  From
    # a repository-root selector invocation, the authorization stays in the
    # run directory while the immutable gate is under run_dir/package_src/.
    # Resolve only those two package-owned locations; never process CWD.
    candidates = (
        target.parent / _ATTEMPT11_GATE_RELATIVE,
        target.parent / "package_src" / _ATTEMPT11_GATE_RELATIVE,
    )
    if not any(
        candidate.is_file()
        and not candidate.is_symlink()
        and hashlib.sha256(candidate.read_bytes()).hexdigest() == gate_hash
        for candidate in candidates
    ):
        raise ValueError("Attempt11 preceding gate artifact is missing or changed")
    return payload, hashlib.sha256(raw).hexdigest()


def _expected_search_config(
    config: Attempt11TeacherConfig, seeds: Mapping[str, int]
) -> dict[str, Any]:
    return {
        "learned_nonbaseline_max": config.candidate_max,
        "baseline_added_exactly_once": True,
        "rerank_samples": config.rerank_samples,
        "rerank_head_max": config.rerank_head_max,
        "shortlist_max": config.shortlist_max,
        "veto_samples": config.veto_samples,
        "stress_samples": config.stress_samples,
        "confirmation_samples": config.confirmation_samples,
        "evaluation_samples": config.evaluation_samples,
        "hand_seed": seeds["hand"],
        "rerank_seed": seeds["rerank"],
        "veto_seed": seeds["veto"],
        "stress_seed": seeds["stress"],
        "confirmation_seed": seeds["confirmation"],
        "evaluation_seed": seeds["evaluation"],
        "child_policy_seed": seeds["child"],
        "run_id": config.run_id,
        "batch_child_selectors": True,
    }


def _expected_seed_domain_provenance(seeds: Mapping[str, int]) -> dict[str, Any]:
    return {
        "domain_order": [
            "hand_external",
            "rerank_r128",
            "veto_v256",
            "stress_x1024",
            "confirmation_c512",
            "evaluation_e256",
            "child_policy",
        ],
        "hand_external": seeds["hand"],
        "rerank_r128": seeds["rerank"],
        "veto_v256": seeds["veto"],
        "stress_x1024": seeds["stress"],
        "confirmation_c512": seeds["confirmation"],
        "evaluation_e256": seeds["evaluation"],
        "child_policy": seeds["child"],
        "all_seven_base_seeds_pairwise_distinct": True,
        "hand_sampled_inside_teacher": False,
    }


_BINDINGS: dict[str, Any] = {
    "AI_PROFILES_SHA256": AI_PROFILES_SHA256,
    "ATTEMPT09_LAMBDA_MODEL_SHA256": ATTEMPT11_LAMBDA_MODEL_SHA256,
    "M43_ATTEMPT09_PLAN_SHA256": M43_ATTEMPT11_PLAN_SHA256,
    "M43_ATTEMPT09_PROFILES": M43_ATTEMPT11_PROFILES,
    "enumerate_attempt09_seed_schedules": enumerate_attempt11_seed_schedules,
    "load_and_validate_attempt09_plan": load_and_validate_attempt11_plan,
    "validate_attempt09_plan": validate_attempt11_plan,
    "Attempt09TeacherConfig": Attempt11TeacherConfig,
    "validate_attempt09_teacher_output": validate_attempt11_teacher_output,
    "ATTEMPT09_BASELINE_PROFILE": ATTEMPT11_BASELINE_PROFILE,
    "ATTEMPT09_CONTINUATION_PROFILE": ATTEMPT11_CONTINUATION_PROFILE,
    "ATTEMPT09_ROW_SCHEMA": ATTEMPT11_ROW_SCHEMA,
    "ATTEMPT09_DEVELOPMENT_DECISION_SCHEMA": ATTEMPT11_DEVELOPMENT_DECISION_SCHEMA,
    "ATTEMPT09_DEVELOPMENT_RECEIPT_SCHEMA": ATTEMPT11_DEVELOPMENT_RECEIPT_SCHEMA,
    "ROOT_CONTRACT_SCHEMA": ATTEMPT11_ROOT_CONTRACT_SCHEMA,
    "TEACHER_VALIDATION_SCHEMA": ATTEMPT11_TEACHER_VALIDATION_SCHEMA,
    "NO_GO_STATUS": "no_go_close_attempt11_development",
    "EVALUATION_SAMPLE_COUNT": 256,
    "SELECTOR_SOURCE_PATH": Path(__file__),
    "_expected_search_config": _expected_search_config,
    "_expected_seed_domain_provenance": _expected_seed_domain_provenance,
    "_authorization": _attempt11_authorization,
}


@contextmanager
def _attempt11_selector_bindings() -> Iterator[None]:
    with _BIND_LOCK, _runner_bindings():
        prior = {name: getattr(_base, name) for name in _BINDINGS}
        try:
            for name, value in _BINDINGS.items():
                setattr(_base, name, value)
            yield
        finally:
            for name, value in prior.items():
                setattr(_base, name, value)


def aggregate_attempt11_development_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    plan: Mapping[str, Any],
    source_input_sha256: str,
    source_plan_sha256: str,
    authorization_sha256: str,
    source_package_sha256: str,
    run_name: str,
) -> dict[str, Any]:
    with _attempt11_selector_bindings():
        return _base.aggregate_attempt09_development_rows(
            rows,
            plan=plan,
            source_input_sha256=source_input_sha256,
            source_plan_sha256=source_plan_sha256,
            authorization_sha256=authorization_sha256,
            source_package_sha256=source_package_sha256,
            run_name=run_name,
        )


def select_attempt11_development(
    *,
    input_path: str | Path,
    plan_path: str | Path,
    authorization_path: str | Path,
    source_package_sha256: str,
    run_name: str,
) -> dict[str, Any]:
    with _attempt11_selector_bindings():
        return _base.select_attempt09_development(
            input_path=input_path,
            plan_path=plan_path,
            authorization_path=authorization_path,
            source_package_sha256=source_package_sha256,
            run_name=run_name,
        )


def execute_attempt11_development_selector(
    *,
    input_path: str | Path,
    plan_path: str | Path,
    authorization_path: str | Path,
    source_package_sha256: str,
    run_name: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    with _attempt11_selector_bindings():
        return _base.execute_attempt09_development_selector(
            input_path=input_path,
            plan_path=plan_path,
            authorization_path=authorization_path,
            source_package_sha256=source_package_sha256,
            run_name=run_name,
            output_dir=output_dir,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--authorization", required=True, type=Path)
    parser.add_argument("--source-package-sha256", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = execute_attempt11_development_selector(
        input_path=args.input,
        plan_path=args.plan,
        authorization_path=args.authorization,
        source_package_sha256=args.source_package_sha256,
        run_name=args.run_name,
        output_dir=args.output_dir,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ATTEMPT11_DEVELOPMENT_DECISION_SCHEMA",
    "ATTEMPT11_DEVELOPMENT_RECEIPT_SCHEMA",
    "aggregate_attempt11_development_rows",
    "execute_attempt11_development_selector",
    "select_attempt11_development",
]
