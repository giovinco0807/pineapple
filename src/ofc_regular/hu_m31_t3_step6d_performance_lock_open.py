"""One-shot local open/materialize/seal lifecycle for M3.1 performance-lock.

The global claim is the irreversible boundary.  It is durably created before
this module performs *any* filesystem operation on the lock output tree.  A
crash after creation consumes the claim; only deterministic continuation of
the exact same identity is accepted.

This module creates roots only.  It cannot run search, inspect timings/Q/EV,
train, promote, resolve ``current``, or launch cloud work.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import merge_hu_m31_t3_step6d_full100_received_v1 as development_merge
from . import run_hu_m31_t3_step6d_performance_v2 as runner
from . import select_hu_m31_t3_step6d_candidate02_tail_v2 as development_roots
from .action_space import generate_turn_actions
from .hu_m31_t3_behavior_roots import M31_T3_BEHAVIOR_PROFILES


CLAIM_SCHEMA = "hu_m31_t3_step6d_performance_lock_global_claim_v1"
MATERIALIZATION_SCHEMA = "hu_m31_t3_step6d_performance_lock_root_materialization_v1"
SEAL_SCHEMA = "hu_m31_t3_step6d_performance_lock_root_seal_v1"

CANDIDATE_LIBRARY_SHA256 = (
    "4050e04b22d7943da9e1691402a2b34cf3d3781041a44557f35e2dfcf366d55d"
)
REFERENCE_LIBRARY_SHA256 = (
    "3f831615d9e751b8e1f266f14dd2009903b3ac2a4094f7e47616e0aa372a01b0"
)
FEATURE_ENCODER_SHA256 = (
    "82510e563ee290cd536e7aebe6bcf0efefac061af5488851f0a582bb4ef83411"
)
STEP6D_CONTRACT_BYTE_SHA256 = (
    "1924295b18070432cf3126159311102d9285dba37c498a3ea7666b0d5b777775"
)
AI_PROFILES_CURRENT_SHA256 = (
    "d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3"
)
DEVELOPMENT_SUMMARY_SHA256 = (
    "f587df6d037313e2111a5cbc3d474370106f6a341ef4ec130cee7516192e668f"
)
DEVELOPMENT_VALIDATION_SHA256 = (
    "da252b3cabfc361d89de2f2fe08931fe61302287b69b82bf9f812a4432baf3eb"
)
LOCK_RUN_CONTRACT_DIGEST = (
    "e73c2b06279c1f1e91c38b2887ee465acf85f8f1072afef636512283252c34a4"
)
PRECONTENT_PLAN_SHA256 = (
    "d2c9985b574839cf7e1515c12ebffc813ac6b4fb4e6494ed8d1288ef582a4b6b"
)

IMAGE = {
    "project": "debian-cloud",
    "name": "debian-12-bookworm-v20260609",
    "id": "1449487925682397051",
    "self_link": (
        "https://www.googleapis.com/compute/v1/projects/debian-cloud/"
        "global/images/debian-12-bookworm-v20260609"
    ),
}
ALLOCATION = {
    "machine_type": "c4-standard-16",
    "process_count": 1,
    "rayon_threads_per_process": 16,
    "omp_threads": 1,
    "m3_batch_threads": 1,
}

MODEL_INPUT_PATHS = (
    "models/opening_stage7_torch_wide.pt",
    "models/turn1_stage6_torch_wide.pt",
    "models/turn2_stage8.pkl",
    "models/turn3_stage6.pkl",
    "models/hu_turn1_stage18_first1801_mc32_hgb_abs_aug3.pkl",
    "models/hu_turn1_stage18_highmc_safe_selector_a_meta_only.pkl",
    "models/hu_turn0_stage3_top60_mc32_1000_hgb_baseline-delta_sew1_aug3.pkl",
    "models/hu_turn0_stage19_safe_selector_highmc_candidate_delta_plus_meta.pkl",
    "models/hu_turn2_stage8b_safe_lcb196_20k_reference_override_cached_rank_wide.pt",
    "models/hu_turn3_stage7_reference_override_cached_rank_wide.pt",
    "models/hu_turn3_stage3_mc32_500k_plus_m8_12_f128_100k_w2_cached_rank_wide.pt",
)
ROOT_GENERATOR_INPUT_PATHS = (
    "src/ofc_regular/run_hu_m31_t3_step6d_performance.py",
    "src/ofc_regular/hu_m31_t3_behavior_roots.py",
    "src/ofc_regular/ai_profiles.py",
    "src/ofc_regular/policy.py",
    "src/ofc_regular/hu_infoset.py",
    "src/ofc_regular/action_space.py",
    "src/ofc_regular/cards.py",
    "src/ofc_regular/state.py",
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPOSITORY_ROOT = _REPO_ROOT
DEFAULT_PRECONTENT_PLAN_PATH = (
    _REPO_ROOT / "outputs/hu_joint_policy/m31_t3_step6d/performance_lock/"
    "precontent_plan_v1.json"
)
DEFAULT_GLOBAL_CLAIM_PATH = (
    _REPO_ROOT / "outputs/hu_joint_policy/m31_t3_step6d/"
    "GLOBAL_PERFORMANCE_LOCK_CLAIM.json"
)
DEFAULT_DEVELOPMENT_MERGE_DIR = (
    _REPO_ROOT / "outputs/hu_joint_policy/m31_t3_step6d/full100_merge/"
    "regular-hu-m31-c02-full100-dev-20260717-002"
)
DEFAULT_DEVELOPMENT_ROOT_DIR = (
    _REPO_ROOT / "outputs/hu_joint_policy/m31_t3_step6d/candidate02_development/"
    "tail_reselection_v2/roots"
)

_CLAIM_KEYS = frozenset(
    {
        "schema",
        "status",
        "scope",
        "opened_unix_ns",
        "global_claim_path",
        "lock_output_directory",
        "precontent_plan",
        "development_go",
        "step6d_contract",
        "lock_run_contract",
        "lock_run_contract_digest",
        "runner_source",
        "ai_profiles_current",
        "accepted_binaries",
        "model_inputs",
        "root_generator_inputs",
        "image",
        "allocation",
        "seed_contract",
        "startup_source",
        "restrictions",
    }
)
_MATERIALIZATION_KEYS = frozenset(
    {
        "schema",
        "status",
        "global_claim_sha256",
        "plan_sha256",
        "run_contract_digest",
        "hand_indices",
        "root_count",
        "root_artifact_sha256",
        "aggregate_root_sha256",
        "same_identity_resume_only",
        "reseeded",
        "training_eligible",
        "current_profile_changed",
    }
)
_SEAL_KEYS = frozenset(
    {
        "schema",
        "status",
        "global_claim_sha256",
        "materialization_sha256",
        "plan_sha256",
        "run_contract_digest",
        "hand_indices",
        "root_count",
        "observation_count",
        "profile_counts",
        "seat_counts",
        "root_artifact_sha256",
        "aggregate_root_sha256",
        "root_topology_sha256",
        "observation_fingerprint_sha256",
        "root_artifact_unique",
        "observation_fingerprint_unique",
        "development_comparison",
        "visibility",
        "selection_inputs",
        "training_eligible",
        "quality_evidence",
        "promotion_evidence",
        "current_profile_changed",
        "named_profile_added",
        "runtime_policy_activated",
    }
)


@dataclass(frozen=True)
class PerformanceLockInputs:
    repository_root: Path
    plan_path: Path
    lock_output_directory: Path
    candidate_library: Path
    reference_library: Path
    feature_encoder: Path
    startup_source: Path
    development_summary_path: Path = DEFAULT_DEVELOPMENT_MERGE_DIR / "summary.json"
    development_validation_path: Path = (
        DEFAULT_DEVELOPMENT_MERGE_DIR / "validation.json"
    )
    development_root_directory: Path = DEFAULT_DEVELOPMENT_ROOT_DIR
    global_claim_path: Path = DEFAULT_GLOBAL_CLAIM_PATH


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _lexical_absolute(path: str | Path) -> str:
    # Deliberately does not stat/resolve the target.  This is used for the lock
    # output identity before the global claim exists.
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def _read_canonical(path: str | Path, label: str) -> dict[str, Any]:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise ValueError(f"{label} is missing or unsafe")
    raw = target.read_bytes()
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _write_once_durable(path: str | Path, value: Mapping[str, Any]) -> None:
    """Create the final path directly; even a partial crash consumes it."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        handle.write(canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    directory_fd: int | None = None
    try:
        if hasattr(os, "O_DIRECTORY"):
            directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
            os.fsync(directory_fd)
    finally:
        if directory_fd is not None:
            os.close(directory_fd)


def _write_or_validate_durable(
    path: str | Path, value: Mapping[str, Any], label: str
) -> None:
    target = Path(path)
    if target.exists():
        if _read_canonical(target, label) != dict(value):
            raise ValueError(f"existing {label} belongs to another identity")
        return
    _write_once_durable(target, value)


def _file_record(path: str | Path, *, root: Path | None = None) -> dict[str, Any]:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise ValueError(f"bound input is missing or unsafe: {target}")
    relative = (
        target.resolve().relative_to(root.resolve()).as_posix()
        if root is not None
        else str(target.resolve())
    )
    return {
        "path": relative,
        "sha256": sha256_file(target),
        "bytes": target.stat().st_size,
    }


def _input_records(
    root: Path, relative_paths: Sequence[str]
) -> dict[str, dict[str, Any]]:
    return {
        relative: _file_record(root / relative, root=root)
        for relative in relative_paths
    }


def _call_plan_validator(
    function: Callable[..., Any], raw: Mapping[str, Any], path: Path
) -> Any:
    parameters = inspect.signature(function).parameters
    if len(parameters) != 1:
        raise RuntimeError("performance-lock plan validator API is not unary")
    name = next(iter(parameters))
    return function(path if "path" in name else raw)


def _load_and_validate_plan(path: Path) -> dict[str, Any]:
    """Adapter kept narrow so the pre-content plan remains producer-owned."""

    raw = _read_canonical(path, "performance-lock pre-content plan")
    try:
        module = importlib.import_module(
            ".hu_m31_t3_step6d_candidate02_performance_lock_plan",
            package=__package__,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "performance-lock pre-content plan module is not available yet"
        ) from exc
    validator = None
    for name in (
        "validate_precontent_plan",
        "load_and_validate_performance_lock_plan",
        "validate_performance_lock_plan",
        "validate_plan",
    ):
        candidate = getattr(module, name, None)
        if callable(candidate):
            validator = candidate
            break
    if validator is None:
        raise RuntimeError("performance-lock plan validator API is unavailable")
    validated = _call_plan_validator(validator, raw, path)
    if not isinstance(validated, Mapping) or dict(validated) != raw:
        raise ValueError("performance-lock plan validator changed the plan")
    expected_sha = next(
        (
            getattr(module, name)
            for name in (
                "PRECONTENT_PLAN_SHA256",
                "PERFORMANCE_LOCK_PLAN_SHA256",
                "PLAN_SHA256",
            )
            if isinstance(getattr(module, name, None), str)
        ),
        PRECONTENT_PLAN_SHA256,
    )
    if (
        expected_sha != PRECONTENT_PLAN_SHA256
        or sha256_file(path) != PRECONTENT_PLAN_SHA256
    ):
        raise ValueError("performance-lock plan file SHA changed")
    return raw


def _validate_official_development_go(
    summary_path: Path, validation_path: Path
) -> dict[str, Any]:
    if (
        sha256_file(summary_path) != DEVELOPMENT_SUMMARY_SHA256
        or sha256_file(validation_path) != DEVELOPMENT_VALIDATION_SHA256
    ):
        raise ValueError("official development GO artifact SHA changed")
    report = development_merge.validate_received_full100_merge(
        summary_path=summary_path
    )
    stored = _read_canonical(validation_path, "official development GO validation")
    if stored != report:
        raise ValueError("official development GO replay differs from validation")
    if (
        stored.get("status") != "pass"
        or stored.get("decision")
        != "full100_receive_and_performance_go_open_performance_lock_only"
        or stored.get("all_gates_passed") is not True
        or stored.get("performance_candidate_frozen") is not True
        or stored.get("performance_lock_authorized") is not True
        or stored.get("run_contract_digest")
        != "92a87f1544a73104d5256db39b022094c8c7f7b11f77bd19dcf1ab1442581ebd"
        or any(
            stored.get(field) is not False
            for field in (
                "quality_pilot_authorized",
                "artifact_fanout_authorized",
                "training_authorized",
                "current_profile_changed",
                "named_profile_added",
                "runtime_policy_activated",
                "m31_complete",
            )
        )
    ):
        raise ValueError("official development result does not open lock")
    return {
        "summary": _file_record(summary_path),
        "validation": _file_record(validation_path),
        "validation_report_sha256": canonical_sha256(stored),
        "development_run_name": stored["run_name"],
        "all_gates_passed": True,
        "performance_lock_authorized": True,
    }


def _plan_contract(
    plan: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    contract = runner.build_run_contract(
        candidate_library_sha256=CANDIDATE_LIBRARY_SHA256,
        reference_library_sha256=REFERENCE_LIBRARY_SHA256,
        variant=runner.CANDIDATE02_PERFORMANCE_LOCK_VARIANT,
    )
    digest = runner.canonical_sha256(contract)
    if digest != LOCK_RUN_CONTRACT_DIGEST:
        raise ValueError("performance-lock runner contract digest changed")
    if plan.get("candidate_variant") != runner.CANDIDATE02_PERFORMANCE_LOCK_VARIANT:
        raise ValueError("performance-lock plan candidate variant changed")
    if plan.get("run_contract") != contract:
        raise ValueError("performance-lock plan runner contract changed")
    if plan.get("run_contract_digest") != digest:
        raise ValueError("performance-lock plan contract digest changed")
    if any(
        plan.get(field) is not False
        for field in (
            "cloud_started",
            "training_eligible",
            "quality_evidence",
            "promotion_evidence",
            "current_profile_changed",
            "named_profile_added",
            "runtime_policy_activated",
            "m31_complete",
        )
        if field in plan
    ):
        raise ValueError("performance-lock plan widened authorization")
    return contract, digest


def _claim_payload(
    inputs: PerformanceLockInputs, *, opened_unix_ns: int
) -> dict[str, Any]:
    root = Path(inputs.repository_root).resolve()
    plan_path = Path(inputs.plan_path).resolve()
    plan = _load_and_validate_plan(plan_path)
    contract, contract_digest = _plan_contract(plan)
    development_go = _validate_official_development_go(
        Path(inputs.development_summary_path).resolve(),
        Path(inputs.development_validation_path).resolve(),
    )

    step6d_contract_path = root / "configs/hu_joint_policy_m31_t3_step6d_contract.json"
    step6d_contract = json.loads(step6d_contract_path.read_text(encoding="utf-8"))
    if (
        sha256_file(step6d_contract_path) != STEP6D_CONTRACT_BYTE_SHA256
        or step6d_contract["anchors"]["policy_registry"]["byte_sha256"]
        != AI_PROFILES_CURRENT_SHA256
    ):
        raise ValueError("Step6d/current registry anchor changed")
    ai_profiles = _file_record(root / "src/ofc_regular/ai_profiles.py", root=root)
    if ai_profiles["sha256"] != AI_PROFILES_CURRENT_SHA256:
        raise ValueError("ai_profiles/current bytes changed")

    accepted = {
        "candidate": _file_record(inputs.candidate_library),
        "reference": _file_record(inputs.reference_library),
        "feature_encoder": _file_record(inputs.feature_encoder),
    }
    expected_binary_hashes = {
        "candidate": CANDIDATE_LIBRARY_SHA256,
        "reference": REFERENCE_LIBRARY_SHA256,
        "feature_encoder": FEATURE_ENCODER_SHA256,
    }
    if {
        name: record["sha256"] for name, record in accepted.items()
    } != expected_binary_hashes:
        raise ValueError("accepted performance-lock binary identity changed")

    seed_contract = runner.candidate02_performance_lock_seed_contract()
    if (
        seed_contract["seed_set_sha256"]
        != runner.CANDIDATE02_PERFORMANCE_LOCK_SEED_SET_SHA256
        or seed_contract["locked_before_content_read"] is not True
        or seed_contract["candidate02_development_overlap_count"] != 0
    ):
        raise ValueError("performance-lock seed contract changed")

    claim_path_identity = _lexical_absolute(inputs.global_claim_path)
    lock_identity = _lexical_absolute(inputs.lock_output_directory)
    if claim_path_identity != _lexical_absolute(DEFAULT_GLOBAL_CLAIM_PATH):
        raise ValueError("performance-lock claim path is not the fixed global path")
    try:
        common = os.path.commonpath((claim_path_identity, lock_identity))
    except ValueError:
        common = ""
    if common == lock_identity:
        raise ValueError("global claim must be outside the lock output tree")

    payload = {
        "schema": CLAIM_SCHEMA,
        "status": (
            "global_one_shot_claim_persisted_before_lock_root_touch_"
            "crash_consumes_claim"
        ),
        "scope": "candidate02_performance_lock_roots_only",
        "opened_unix_ns": opened_unix_ns,
        "global_claim_path": claim_path_identity,
        "lock_output_directory": lock_identity,
        "precontent_plan": {
            **_file_record(plan_path),
            "schema": plan.get("schema"),
            "canonical_sha256": canonical_sha256(plan),
        },
        "development_go": development_go,
        "step6d_contract": {
            **_file_record(step6d_contract_path, root=root),
            "canonical_sha256": canonical_sha256(step6d_contract),
        },
        "lock_run_contract": contract,
        "lock_run_contract_digest": contract_digest,
        "runner_source": _file_record(
            root / "src/ofc_regular/run_hu_m31_t3_step6d_performance_v2.py",
            root=root,
        ),
        "ai_profiles_current": ai_profiles,
        "accepted_binaries": accepted,
        "model_inputs": _input_records(root, MODEL_INPUT_PATHS),
        "root_generator_inputs": _input_records(root, ROOT_GENERATOR_INPUT_PATHS),
        "image": dict(IMAGE),
        "allocation": dict(ALLOCATION),
        "seed_contract": seed_contract,
        "startup_source": _file_record(inputs.startup_source),
        "restrictions": {
            "timing_used_for_root_selection": False,
            "q_used_for_root_selection": False,
            "ev_used_for_root_selection": False,
            "alternate_seed_allowed": False,
            "reseed_allowed": False,
            "cloud_authorized": False,
            "training_authorized": False,
            "promotion_authorized": False,
            "current_profile_resolution_allowed": False,
            "runtime_activation_allowed": False,
            "opponent_private_discards_allowed": False,
        },
    }
    if set(payload) != _CLAIM_KEYS:
        raise AssertionError("performance-lock claim schema implementation changed")
    return payload


def _after_claim_persisted(_claim_path: Path) -> None:
    """Test hook.  Raising here models a crash that consumes the claim."""


def open_performance_lock(inputs: PerformanceLockInputs) -> dict[str, Any]:
    """Persist the global claim without touching the lock output path."""

    claim = _claim_payload(inputs, opened_unix_ns=time.time_ns())
    claim_path = Path(inputs.global_claim_path)
    _write_once_durable(claim_path, claim)
    _after_claim_persisted(claim_path)
    return claim


def validate_global_claim(inputs: PerformanceLockInputs) -> dict[str, Any]:
    claim = _read_canonical(inputs.global_claim_path, "performance-lock claim")
    if set(claim) != _CLAIM_KEYS:
        raise ValueError("performance-lock global claim fields changed")
    opened = claim.get("opened_unix_ns")
    if isinstance(opened, bool) or not isinstance(opened, int) or opened <= 0:
        raise ValueError("performance-lock claim timestamp changed")
    expected = _claim_payload(inputs, opened_unix_ns=opened)
    if claim != expected:
        raise ValueError("performance-lock global claim identity changed")
    return claim


def validate_open_claim(inputs: PerformanceLockInputs) -> dict[str, Any]:
    """Merge-facing alias that fully replays the immutable open claim."""

    return validate_global_claim(inputs)


def _lock_output_after_claim(inputs: PerformanceLockInputs) -> Path:
    """The only adapter allowed to construct/touch the lock output Path."""

    return Path(inputs.lock_output_directory)


def _root_hashes(roots: Sequence[Mapping[str, Any]]) -> list[str]:
    return [canonical_sha256(root) for root in roots]


def _materialization_payload(
    claim: Mapping[str, Any], roots: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if len(roots) != 100:
        raise ValueError("performance-lock materialization is incomplete")
    hashes = _root_hashes(roots)
    value = {
        "schema": MATERIALIZATION_SCHEMA,
        "status": "all_100_lock_roots_materialized_same_identity",
        "global_claim_sha256": canonical_sha256(claim),
        "plan_sha256": claim["precontent_plan"]["sha256"],
        "run_contract_digest": claim["lock_run_contract_digest"],
        "hand_indices": list(runner.CONTRACT_HAND_INDICES),
        "root_count": 100,
        "root_artifact_sha256": hashes,
        "aggregate_root_sha256": canonical_sha256(hashes),
        "same_identity_resume_only": True,
        "reseeded": False,
        "training_eligible": False,
        "current_profile_changed": False,
    }
    if set(value) != _MATERIALIZATION_KEYS:
        raise AssertionError("performance-lock materialization schema changed")
    return value


def materialize_performance_lock(
    inputs: PerformanceLockInputs,
) -> dict[str, Any]:
    """Create/resume the same 100 deterministic roots after claim validation."""

    claim = validate_global_claim(inputs)
    # No lock-output Path object is constructed above this line.
    output = _lock_output_after_claim(inputs)
    output.mkdir(parents=True, exist_ok=True)
    contract = runner.validate_run_contract(claim["lock_run_contract"])
    roots = runner._materialize_roots(
        contract=contract,
        repository_root=Path(inputs.repository_root).resolve(),
        output_dir=output,
        indices=runner.CONTRACT_HAND_INDICES,
    )
    canonical_roots, _observations = _load_lock_roots(output, contract)
    if roots != canonical_roots:
        raise ValueError("performance-lock runner returned a non-canonical root set")
    value = _materialization_payload(claim, canonical_roots)
    _write_or_validate_durable(
        output / "materialization.json", value, "performance-lock materialization"
    )
    return value


def _load_lock_roots(
    output: Path, contract: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[tuple[Any, Any]]]:
    root_dir = output / "roots"
    expected = [f"hand_{index:03d}.json" for index in runner.CONTRACT_HAND_INDICES]
    observed = sorted(path.name for path in root_dir.glob("hand_*.json"))
    if observed != expected:
        raise ValueError("performance-lock root set must be exactly 000..099")
    roots: list[dict[str, Any]] = []
    observations: list[tuple[Any, Any]] = []
    for index, name in zip(runner.CONTRACT_HAND_INDICES, expected, strict=True):
        value = _read_canonical(root_dir / name, f"lock root {index}")
        parsed = runner._validate_root_artifact(contract, value, index=index)
        roots.append(value)
        observations.append(parsed)
    return roots, observations


def _topology_rows(
    roots: Sequence[Mapping[str, Any]],
    observations: Sequence[tuple[Any, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root, (first, second) in zip(roots, observations, strict=True):
        rows.append(
            {
                "hand_index": root["hand_index"],
                "profile": root["profile"],
                "first_fingerprint": first.fingerprint(),
                "second_fingerprint": second.fingerprint(),
                "first_hero_legal_actions": len(
                    generate_turn_actions(first.hero_board, first.dealt_cards)
                ),
                "first_opponent_response_legal_actions": len(
                    generate_turn_actions(
                        first.opponent_public_board, first.dealt_cards
                    )
                ),
                "second_hero_legal_actions": len(
                    generate_turn_actions(second.hero_board, second.dealt_cards)
                ),
            }
        )
    return rows


def _build_root_seal(inputs: PerformanceLockInputs) -> dict[str, Any]:
    claim = validate_global_claim(inputs)
    output = _lock_output_after_claim(inputs)
    materialization = _read_canonical(
        output / "materialization.json", "performance-lock materialization"
    )
    contract = runner.validate_run_contract(claim["lock_run_contract"])
    roots, observations = _load_lock_roots(output, contract)
    if materialization != _materialization_payload(claim, roots):
        raise ValueError("performance-lock materialization identity changed")
    hashes = _root_hashes(roots)

    profile_counts = Counter(str(root["profile"]) for root in roots)
    expected_profiles = {profile: 20 for profile in M31_T3_BEHAVIOR_PROFILES}
    if dict(profile_counts) != expected_profiles:
        raise ValueError("performance-lock profile balance changed")

    fingerprints = [
        observation.fingerprint() for pair in observations for observation in pair
    ]
    if len(fingerprints) != 200 or len(set(fingerprints)) != 200:
        raise ValueError("performance-lock observation fingerprints collide")
    if len(hashes) != 100 or len(set(hashes)) != 100:
        raise ValueError("performance-lock root artifact hashes collide")

    development = development_roots.load_frozen_roots(inputs.development_root_directory)
    development_hashes = _root_hashes(development)
    development_fingerprints = {
        raw["observation_fingerprint"]
        for root in development
        for raw in root["observations"]
    }
    lock_seeds = {value for root in roots for value in root["seeds"].values()}
    development_seeds = {
        value for root in development for value in root["seeds"].values()
    }
    fingerprint_overlap = set(fingerprints) & development_fingerprints
    hash_overlap = set(hashes) & set(development_hashes)
    seed_overlap = lock_seeds & development_seeds
    if fingerprint_overlap or hash_overlap or seed_overlap:
        raise ValueError("performance-lock roots overlap development evidence")

    topology = _topology_rows(roots, observations)
    value = {
        "schema": SEAL_SCHEMA,
        "status": "sealed_100_disjoint_hidden_safe_performance_lock_roots",
        "global_claim_sha256": canonical_sha256(claim),
        "materialization_sha256": canonical_sha256(materialization),
        "plan_sha256": claim["precontent_plan"]["sha256"],
        "run_contract_digest": claim["lock_run_contract_digest"],
        "hand_indices": list(runner.CONTRACT_HAND_INDICES),
        "root_count": 100,
        "observation_count": 200,
        "profile_counts": expected_profiles,
        "seat_counts": {"first": 100, "second": 100},
        "root_artifact_sha256": hashes,
        "aggregate_root_sha256": canonical_sha256(hashes),
        "root_topology_sha256": canonical_sha256(topology),
        "observation_fingerprint_sha256": canonical_sha256(fingerprints),
        "root_artifact_unique": True,
        "observation_fingerprint_unique": True,
        "development_comparison": {
            "development_all100_root_sha256": (development_roots.ALL100_ROOT_SHA256),
            "development_root_count": 100,
            "lock_fingerprint_overlap_count": 0,
            "lock_root_hash_overlap_count": 0,
            "lock_seed_overlap_count": 0,
        },
        "visibility": {
            "runner_validator_replayed_all_roots": True,
            "first_observation_count": 100,
            "second_observation_count": 100,
            "opponent_private_discards_used": False,
            "current_profile_resolved": False,
        },
        "selection_inputs": {
            "timing_used": False,
            "q_used": False,
            "ev_used": False,
            "all_100_preregistered_hands_used": True,
        },
        "training_eligible": False,
        "quality_evidence": False,
        "promotion_evidence": False,
        "current_profile_changed": False,
        "named_profile_added": False,
        "runtime_policy_activated": False,
    }
    if set(value) != _SEAL_KEYS:
        raise AssertionError("performance-lock seal schema implementation changed")
    return value


def seal_performance_lock(inputs: PerformanceLockInputs) -> dict[str, Any]:
    """Validate and immutably seal the claimed 100-root information set."""

    value = _build_root_seal(inputs)
    output = _lock_output_after_claim(inputs)
    _write_or_validate_durable(output / "seal.json", value, "performance-lock seal")
    return value


def validate_root_seal(inputs: PerformanceLockInputs) -> dict[str, Any]:
    """Replay a stored seal without creating or modifying any artifact."""

    validate_global_claim(inputs)
    output = _lock_output_after_claim(inputs)
    stored = _read_canonical(output / "seal.json", "performance-lock seal")
    if set(stored) != _SEAL_KEYS:
        raise ValueError("performance-lock seal fields changed")
    expected = _build_root_seal(inputs)
    if stored != expected:
        raise ValueError("performance-lock seal replay changed")
    return stored


def _inputs_from_args(args: argparse.Namespace) -> PerformanceLockInputs:
    return PerformanceLockInputs(
        repository_root=args.repository_root,
        plan_path=args.plan,
        lock_output_directory=args.lock_output,
        candidate_library=args.candidate_library,
        reference_library=args.reference_library,
        feature_encoder=args.feature_encoder,
        startup_source=args.startup_source,
        development_summary_path=args.development_summary,
        development_validation_path=args.development_validation,
        development_root_directory=args.development_roots,
        global_claim_path=args.global_claim,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("open", "materialize", "seal"))
    parser.add_argument("--repository-root", type=Path, default=DEFAULT_REPOSITORY_ROOT)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PRECONTENT_PLAN_PATH)
    parser.add_argument("--lock-output", type=Path, required=True)
    parser.add_argument("--candidate-library", type=Path, required=True)
    parser.add_argument("--reference-library", type=Path, required=True)
    parser.add_argument("--feature-encoder", type=Path, required=True)
    parser.add_argument("--startup-source", type=Path, required=True)
    parser.add_argument(
        "--development-summary",
        type=Path,
        default=DEFAULT_DEVELOPMENT_MERGE_DIR / "summary.json",
    )
    parser.add_argument(
        "--development-validation",
        type=Path,
        default=DEFAULT_DEVELOPMENT_MERGE_DIR / "validation.json",
    )
    parser.add_argument(
        "--development-roots",
        type=Path,
        default=DEFAULT_DEVELOPMENT_ROOT_DIR,
    )
    parser.add_argument("--global-claim", type=Path, default=DEFAULT_GLOBAL_CLAIM_PATH)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    inputs = _inputs_from_args(args)
    if args.command == "open":
        value = open_performance_lock(inputs)
    elif args.command == "materialize":
        value = materialize_performance_lock(inputs)
    else:
        value = seal_performance_lock(inputs)
    print(json.dumps(value, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CLAIM_SCHEMA",
    "DEFAULT_DEVELOPMENT_MERGE_DIR",
    "DEFAULT_DEVELOPMENT_ROOT_DIR",
    "DEFAULT_GLOBAL_CLAIM_PATH",
    "DEFAULT_PRECONTENT_PLAN_PATH",
    "DEFAULT_REPOSITORY_ROOT",
    "MATERIALIZATION_SCHEMA",
    "PRECONTENT_PLAN_SHA256",
    "PerformanceLockInputs",
    "SEAL_SCHEMA",
    "canonical_bytes",
    "canonical_sha256",
    "main",
    "materialize_performance_lock",
    "open_performance_lock",
    "seal_performance_lock",
    "validate_global_claim",
    "validate_open_claim",
    "validate_root_seal",
]
