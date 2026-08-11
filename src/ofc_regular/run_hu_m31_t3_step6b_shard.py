"""Run one restart-safe M3.1 Step 6b infrastructure-canary shard.

Step 6b is deliberately a separate executable boundary from Step 6a.  It may
execute only shards 1 through 9 and reuses the frozen Step 6a search task
format, seed schedule, budget, counter-RNG run id, and native/model identities.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import multiprocessing
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .action_key import (
    ACTION_KEY_SCHEMA,
    action_key,
    action_key_from_payload,
    legal_action_set_digest,
    ordered_action_mapping_digest,
)
from .action_space import generate_turn_actions
from .hu_belief import sample_hidden_card_particles
from .hu_infoset import ActorObservation
from .hu_m31_t3_behavior_roots import (
    M31_T3_BEHAVIOR_PROFILES,
    behavior_profile_for_index,
)
from .hu_m31_t3_runtime import HuM31T3RuntimeConfig, HuM31T3SearchSolver
from .run_hu_m31_t3_step6a_shard import (
    MAX_FIRST_P95_SECONDS,
    MAX_PEAK_RSS_BYTES,
    MAX_SECOND_P95_SECONDS,
    PAIRED_HANDS_PER_SHARD,
    RAYON_THREADS,
    ROOTS_PER_SHARD,
    STEP5_CONTRACT_CANONICAL_SHA256,
    STEP6A_RUN_ID,
    WORKERS,
    _latency,
    _load_json,
    _materialize_roots,
    _search_worker,
    _sha256,
    _validate_search_task as _validate_step6a_search_task,
    _verify_package_files,
    _write_heartbeat,
    _write_once,
    portable_decision_sha256,
    seed_values,
)
from .validate_hu_m31_t3_convergence import REFERENCE_BUDGET


STEP6B_PACKAGE_SCHEMA = "hu_m31_t3_step6b_spot_package_v1"
STEP6B_SHARD_SCHEMA = "hu_m31_t3_step6b_shard_v1"
STEP6B_PARITY_SCHEMA = "hu_m31_t3_step6b_linux_parity_v1"
STEP6B_SUMMARY_SCHEMA = "hu_m31_t3_step6b_summary_v1"
AUTHORIZED_SHARDS = tuple(range(1, 10))

EXPECTED_NATIVE_LIBRARY_SHA256 = (
    "3f831615d9e751b8e1f266f14dd2009903b3ac2a4094f7e47616e0aa372a01b0"
)
EXPECTED_FEATURE_ENCODER_SHA256 = (
    "82510e563ee290cd536e7aebe6bcf0efefac061af5488851f0a582bb4ef83411"
)
EXPECTED_MODELS = {
    "models/hu_turn0_stage19_safe_selector_highmc_candidate_delta_plus_meta.pkl": (
        "7cd85b3a824816223ceb0d84870feb62e43d827452031e6f51de0ee452514abe"
    ),
    "models/hu_turn0_stage3_top60_mc32_1000_hgb_baseline-delta_sew1_aug3.pkl": (
        "8f360bc4c326f0d8d20710a4efc7c42c62a80cca11341b34a41a72a4c6aadf71"
    ),
    "models/hu_turn1_stage18_first1801_mc32_hgb_abs_aug3.pkl": (
        "e910131715efbec7a116e411a8dc23dcfc9a904f2aef9e0eee9d4913a02dfee7"
    ),
    "models/hu_turn1_stage18_highmc_safe_selector_a_meta_only.pkl": (
        "838ff3421c2393648aaabeb7316c9e6c3fe97197485b1363be4e8b5dd33ae086"
    ),
    "models/hu_turn2_stage8b_safe_lcb196_20k_reference_override_cached_rank_wide.pt": (
        "2a6a52ee09e329852d00197e686509b0747ff86ef627a7b6ed9a21311ec25b3b"
    ),
    "models/hu_turn3_stage3_mc32_500k_plus_m8_12_f128_100k_w2_cached_rank_wide.pt": (
        "36f6ac00ca308b92aa4a11a7d6b3c65c4b5f65f6fb5fa2e71585d7f87d8dd112"
    ),
    "models/hu_turn3_stage7_reference_override_cached_rank_wide.pt": (
        "727fb766b940f17c6c6d00a373b47d68bd07aa1095fe86633ba0fcab6c6e8f20"
    ),
    "models/opening_stage7_torch_wide.pt": (
        "4cfec60e3035323d10348f4f28938c24428880edc557a5e824723d32838fe939"
    ),
    "models/turn1_stage6_torch_wide.pt": (
        "ecffa86f0ccbf0bea1d697ae98bf4b9de316006e5aa4539eda4476a29f3f8b4a"
    ),
    "models/turn2_stage8.pkl": (
        "4be8e1c3647306a18b74676bf4835fcac9e03f3dde65fba391e9aaf5ac3929fe"
    ),
    "models/turn3_stage6.pkl": (
        "5996204bf904b258451097042f38bbd87b3f570fbcbe9bf5f4a1d2bdaf376737"
    ),
}

_PARITY_GATE_KEYS = frozenset(
    {
        "both_seats",
        "portable_action_values_exact",
        "linux_native_hash_bound",
        "manifest_source_and_golden_bound",
        "no_profile_or_current_resolution",
    }
)
_INTEGRITY_KEYS = frozenset(
    {
        "budget",
        "candidate_evaluation_rng_domains_distinct",
        "candidate_seed",
        "continuation_seed",
        "downstream_t4_exact",
        "evaluation_seed",
        "exact_t4",
        "execution_mode_scalar",
        "finite_values",
        "legal_action_order_digest",
        "legal_action_set_digest",
        "native_library_sha256",
        "observation_fingerprint",
        "original_index_mapping",
        "run_id",
        "seat",
        "selected_action_mapping",
        "selected_regret_consistent",
        "teacher_value_diagnostic",
        "unique_action_keys",
    }
)


@dataclass(frozen=True)
class ShardSpec:
    run_name: str
    shard: int
    global_hand_start: int
    hand_count: int = PAIRED_HANDS_PER_SHARD

    def __post_init__(self) -> None:
        if not isinstance(self.run_name, str) or not self.run_name:
            raise ValueError("Step 6b run name must be non-empty")
        if (
            isinstance(self.shard, bool)
            or not isinstance(self.shard, int)
            or self.shard not in AUTHORIZED_SHARDS
        ):
            raise ValueError("Step 6b execution authorizes shards 1 through 9 only")
        if (
            self.global_hand_start != self.shard * PAIRED_HANDS_PER_SHARD
            or self.hand_count != PAIRED_HANDS_PER_SHARD
        ):
            raise ValueError("Step 6b shard schedule changed")

    @property
    def root_count(self) -> int:
        return self.hand_count * 2


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_frozen_runtime_manifest(manifest: Mapping[str, Any]) -> None:
    native = manifest.get("native_library")
    feature = manifest.get("feature_encoder_library")
    models = manifest.get("models")
    if (
        not isinstance(native, Mapping)
        or native.get("path") != "native/release/libofc_hu_m3_engine.so"
        or native.get("sha256") != EXPECTED_NATIVE_LIBRARY_SHA256
        or native.get("engine_version") != "ofc_hu_m3_engine/0.1.0"
        or not isinstance(feature, Mapping)
        or feature.get("path") != "target/release/libofc_stage3_feature_encoder.so"
        or feature.get("sha256") != EXPECTED_FEATURE_ENCODER_SHA256
        or models != EXPECTED_MODELS
    ):
        raise ValueError("Step 6b frozen native/model identity changed")


def _load_manifest_and_spec(
    *,
    manifest_path: Path,
    schedule_path: Path,
    source_package_sha256: str,
    shard: int,
) -> tuple[dict[str, Any], ShardSpec, str]:
    manifest = _load_json(manifest_path)
    manifest_sha = _sha256(manifest_path)
    if (
        manifest.get("schema") != STEP6B_PACKAGE_SCHEMA
        or manifest.get("status") != "packaged_local_no_gcloud"
        or manifest.get("source_sha256") != source_package_sha256
        or manifest.get("schedule_sha256") != _sha256(schedule_path)
        or manifest.get("total_shards") != 10
        or manifest.get("authorized_shards") != list(AUTHORIZED_SHARDS)
        or manifest.get("paired_hands_per_shard") != PAIRED_HANDS_PER_SHARD
        or manifest.get("roots_per_shard") != ROOTS_PER_SHARD
        or manifest.get("step5_contract_canonical_sha256")
        != STEP5_CONTRACT_CANONICAL_SHA256
        or manifest.get("canary_rows_training_eligible") is not False
        or manifest.get("remaining_canary_shards_authorized") is not True
        or manifest.get("current_profile_changed") is not False
        or manifest.get("production_fanout_authorized") is not False
        or not _is_sha256(manifest.get("parity_golden_sha256"))
    ):
        raise ValueError("Step 6b package manifest boundary changed")
    _validate_frozen_runtime_manifest(manifest)

    lines = schedule_path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 10 or shard not in AUTHORIZED_SHARDS:
        raise ValueError("Step 6b schedule changed")
    try:
        row = json.loads(lines[shard])
    except (IndexError, json.JSONDecodeError) as exc:
        raise ValueError("Step 6b schedule changed") from exc
    expected_keys = {
        "schema",
        "run_name",
        "shard",
        "global_hand_start",
        "hand_count",
        "root_count",
        "output_prefix",
    }
    if (
        not isinstance(row, dict)
        or set(row) != expected_keys
        or row.get("schema") != STEP6B_SHARD_SCHEMA
        or row.get("run_name") != manifest.get("run_name")
        or row.get("shard") != shard
        or row.get("global_hand_start") != shard * PAIRED_HANDS_PER_SHARD
        or row.get("hand_count") != PAIRED_HANDS_PER_SHARD
        or row.get("root_count") != ROOTS_PER_SHARD
        or row.get("output_prefix") != f"shard-{shard:03d}"
    ):
        raise ValueError("Step 6b shard row changed")
    spec = ShardSpec(
        run_name=str(row["run_name"]),
        shard=int(row["shard"]),
        global_hand_start=int(row["global_hand_start"]),
        hand_count=int(row["hand_count"]),
    )
    return manifest, spec, manifest_sha


def _validate_parity_report(
    value: Mapping[str, Any],
    *,
    source_package_sha256: str,
    manifest_sha256: str,
    parity_golden_sha256: str,
    native_library_sha256: str,
) -> None:
    rows = value.get("rows")
    gates = value.get("gates")
    if (
        value.get("schema") != STEP6B_PARITY_SCHEMA
        or value.get("status") != "pass"
        or value.get("source_package_sha256") != source_package_sha256
        or value.get("manifest_sha256") != manifest_sha256
        or value.get("parity_golden_sha256") != parity_golden_sha256
        or value.get("native_library_sha256") != native_library_sha256
        or value.get("all_gates_passed") is not True
        or value.get("current_profile_changed") is not False
        or value.get("teacher_generation_started") is not False
        or value.get("production_fanout_authorized") is not False
        or not isinstance(rows, list)
        or len(rows) != 2
        or not isinstance(gates, Mapping)
        or set(gates) != _PARITY_GATE_KEYS
        or any(result is not True for result in gates.values())
    ):
        raise ValueError("Step 6b Linux parity provenance changed")
    if [row.get("seat") for row in rows if isinstance(row, Mapping)] != [
        "first",
        "second",
    ]:
        raise ValueError("Step 6b Linux parity seats changed")
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Step 6b Linux parity row changed")
        expected = row.get("expected_portable_sha256")
        observed = row.get("observed_portable_sha256")
        if (
            not _is_sha256(row.get("observation_fingerprint"))
            or not _is_sha256(expected)
            or observed != expected
            or row.get("match") is not True
        ):
            raise ValueError("Step 6b Linux parity row changed")


def run_linux_parity(
    *,
    manifest: Mapping[str, Any],
    source_package_sha256: str,
    manifest_sha256: str,
    repository_root: Path,
    parity_golden_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    golden_sha = _sha256(parity_golden_path)
    native = manifest["native_library"]
    if golden_sha != manifest.get("parity_golden_sha256"):
        raise ValueError("Step 6b parity golden changed")
    expected = {
        "source_package_sha256": source_package_sha256,
        "manifest_sha256": manifest_sha256,
        "parity_golden_sha256": golden_sha,
        "native_library_sha256": str(native["sha256"]),
    }
    if output_path.exists():
        existing = _load_json(output_path)
        _validate_parity_report(existing, **expected)
        return existing

    golden = _load_json(parity_golden_path)
    config = golden.get("runtime_config")
    rows = golden.get("rows")
    if not isinstance(config, Mapping) or not isinstance(rows, list) or len(rows) != 2:
        raise ValueError("Step 6b parity golden changed")
    solver = HuM31T3SearchSolver(
        HuM31T3RuntimeConfig(
            expected_library_sha256=str(native["sha256"]),
            library_path=repository_root / str(native["path"]),
            run_id=str(config["run_id"]),
            candidate_samples=int(config["candidate_samples"]),
            evaluation_samples=int(config["evaluation_samples"]),
            downstream_t3_samples=int(config["downstream_t3_samples"]),
            seed=int(config["continuation_seed"]),
            candidate_seed=int(config["candidate_seed"]),
            evaluation_seed=int(config["evaluation_seed"]),
        )
    )
    observed = []
    for row in rows:
        observation = ActorObservation.from_dict(row["observation"])
        digest = portable_decision_sha256(solver.solve(observation).to_dict())
        observed.append(
            {
                "seat": observation.seat,
                "observation_fingerprint": observation.fingerprint(),
                "expected_portable_sha256": row["portable_decision_sha256"],
                "observed_portable_sha256": digest,
                "match": digest == row["portable_decision_sha256"],
            }
        )
    gates = {
        "both_seats": [row["seat"] for row in observed] == ["first", "second"],
        "portable_action_values_exact": all(row["match"] for row in observed),
        "linux_native_hash_bound": solver.library_sha256 == native["sha256"],
        "manifest_source_and_golden_bound": True,
        "no_profile_or_current_resolution": True,
    }
    report = {
        "schema": STEP6B_PARITY_SCHEMA,
        "status": "pass" if all(gates.values()) else "no_go",
        **expected,
        "rows": observed,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "current_profile_changed": False,
        "teacher_generation_started": False,
        "production_fanout_authorized": False,
    }
    _write_once(output_path, report)
    if not report["all_gates_passed"]:
        raise RuntimeError("Step 6b Linux portable parity failed")
    return report


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _validate_recovered_search_task(
    value: Mapping[str, Any],
    *,
    root_task: Mapping[str, Any],
    library_sha256: str,
) -> None:
    """Validate a recovered common-format task against its root observation."""

    _validate_step6a_search_task(
        value, root_task=root_task, library_sha256=library_sha256
    )
    rows = value.get("rows")
    root_rows = root_task.get("observations")
    global_index = root_task.get("global_hand_index")
    if (
        not isinstance(rows, list)
        or not isinstance(root_rows, list)
        or len(rows) != 2
        or len(root_rows) != 2
        or isinstance(global_index, bool)
        or not isinstance(global_index, int)
    ):
        raise ValueError("Step 6b recovered search task row envelope changed")
    memory = value.get("memory")
    if (
        not isinstance(memory, Mapping)
        or not isinstance(memory.get("peak_rss_bytes"), int)
        or isinstance(memory.get("peak_rss_bytes"), bool)
        or int(memory["peak_rss_bytes"]) <= 0
    ):
        raise ValueError("Step 6b recovered search task memory changed")

    seeds = root_task["seeds"]
    for offset, (raw_root, row) in enumerate(zip(root_rows, rows, strict=True)):
        if not isinstance(raw_root, Mapping) or not isinstance(row, Mapping):
            raise ValueError("Step 6b recovered search task row changed")
        observation = ActorObservation.from_dict(raw_root["observation"])
        fingerprint = observation.fingerprint()
        if (
            raw_root.get("seat") != observation.seat
            or raw_root.get("observation_fingerprint") != fingerprint
            or row.get("root_index") != global_index * 2 + offset
            or row.get("global_hand_index") != global_index
            or row.get("seat") != observation.seat
            or row.get("observation_fingerprint") != fingerprint
            or not _finite_number(row.get("wall_seconds"))
            or float(row["wall_seconds"]) < 0.0
        ):
            raise ValueError("Step 6b recovered search task identity changed")

        actions = generate_turn_actions(observation.hero_board, observation.dealt_cards)
        expected_keys = [action_key(action).to_token() for action in actions]
        legal_count = row.get("legal_action_count")
        decision = row.get("decision")
        integrity = row.get("integrity")
        if (
            isinstance(legal_count, bool)
            or legal_count != len(actions)
            or not isinstance(decision, Mapping)
            or not isinstance(integrity, Mapping)
            or set(integrity) != _INTEGRITY_KEYS
            or any(result is not True for result in integrity.values())
        ):
            raise ValueError("Step 6b recovered search task integrity changed")
        if (
            decision.get("seat") != observation.seat
            or decision.get("observation_fingerprint") != fingerprint
            or decision.get("run_id") != STEP6A_RUN_ID
            or decision.get("continuation_seed") != seeds["child"]
            or decision.get("candidate_seed") != seeds["candidate"]
            or decision.get("evaluation_seed") != seeds["evaluation"]
            or decision.get("candidate_samples") != REFERENCE_BUDGET.candidate_samples
            or decision.get("evaluation_samples") != REFERENCE_BUDGET.evaluation_samples
            or decision.get("downstream_t3_samples")
            != REFERENCE_BUDGET.downstream_t3_samples
            or decision.get("downstream_t4_samples") != 0
            or decision.get("native_library_sha256") != library_sha256
            or decision.get("execution_mode") != "scalar"
            or decision.get("batch_size") != 1
            or decision.get("teacher_value_status") != "diagnostic_not_match_EV"
            or decision.get("action_key_schema") != ACTION_KEY_SCHEMA
            or decision.get("legal_action_set_digest")
            != legal_action_set_digest(actions)
            or decision.get("legal_action_order_digest")
            != ordered_action_mapping_digest(actions)
        ):
            raise ValueError("Step 6b recovered search decision binding changed")

        action_values = decision.get("action_values")
        if not isinstance(action_values, list) or len(action_values) != len(actions):
            raise ValueError("Step 6b recovered action-value coverage changed")
        indices: list[int] = []
        ranks: list[int] = []
        by_key: dict[str, Mapping[str, Any]] = {}
        for action_value in action_values:
            if not isinstance(action_value, Mapping):
                raise ValueError("Step 6b recovered action value changed")
            index = action_value.get("original_index")
            rank = action_value.get("rank")
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < len(actions)
                or isinstance(rank, bool)
                or not isinstance(rank, int)
                or not 0 <= rank < len(actions)
                or action_value.get("action_key") != expected_keys[index]
                or action_key_from_payload(action_value).to_token()
                != expected_keys[index]
                or any(
                    not _finite_number(action_value.get(field))
                    for field in ("selection_ev", "evaluation_ev", "evaluation_regret")
                )
            ):
                raise ValueError("Step 6b recovered action mapping changed")
            key = str(action_value["action_key"])
            if key in by_key:
                raise ValueError("Step 6b recovered action keys are not unique")
            by_key[key] = action_value
            indices.append(index)
            ranks.append(rank)
        if sorted(indices) != list(range(len(actions))) or sorted(ranks) != list(
            range(len(actions))
        ):
            raise ValueError("Step 6b recovered action index coverage changed")

        selected_key = decision.get("selected_action_key")
        selected = by_key.get(str(selected_key))
        selected_payload = decision.get("selected_action")
        if (
            selected is None
            or not isinstance(selected_payload, Mapping)
            or action_key_from_payload(selected_payload).to_token() != selected_key
            or decision.get("selected_selection_ev") != selected.get("selection_ev")
            or decision.get("selected_evaluation_ev") != selected.get("evaluation_ev")
        ):
            raise ValueError("Step 6b recovered selected action changed")


def run_shard(
    *,
    manifest_path: Path,
    schedule_path: Path,
    source_package_sha256: str,
    repository_root: Path,
    shard: int,
    output_dir: Path,
    parity_report_path: Path,
    stop_after_tasks: int | None = None,
) -> dict[str, Any]:
    manifest, spec, manifest_sha = _load_manifest_and_spec(
        manifest_path=manifest_path,
        schedule_path=schedule_path,
        source_package_sha256=source_package_sha256,
        shard=shard,
    )
    _verify_package_files(manifest, repository_root)
    parity = _load_json(parity_report_path)
    _validate_parity_report(
        parity,
        source_package_sha256=source_package_sha256,
        manifest_sha256=manifest_sha,
        parity_golden_sha256=str(manifest["parity_golden_sha256"]),
        native_library_sha256=str(manifest["native_library"]["sha256"]),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    heartbeat_path = output_dir / "heartbeat.json"
    root_tasks = _materialize_roots(
        manifest=manifest,
        manifest_sha256=manifest_sha,
        spec=spec,
        root_dir=output_dir / "roots",
        heartbeat_path=heartbeat_path,
    )
    task_dir = output_dir / "tasks"
    task_dir.mkdir(parents=True, exist_ok=True)
    existing: dict[int, dict[str, Any]] = {}
    missing: list[dict[str, Any]] = []
    native = manifest["native_library"]
    for root_task in root_tasks:
        index = int(root_task["global_hand_index"])
        path = task_dir / f"hand_{index:03d}.json"
        if path.exists():
            value = _load_json(path)
            _validate_recovered_search_task(
                value, root_task=root_task, library_sha256=str(native["sha256"])
            )
            existing[index] = value
        else:
            missing.append(root_task)
    resumed_task_count = len(existing)
    selected = missing
    if stop_after_tasks is not None:
        if stop_after_tasks <= 0:
            raise ValueError("stop_after_tasks must be positive")
        selected = missing[:stop_after_tasks]
    _write_heartbeat(
        heartbeat_path,
        status="running_search",
        total_tasks=spec.hand_count,
        completed_tasks=len(existing),
        pending_tasks=len(missing),
        resumed_task_count=resumed_task_count,
    )
    if selected:
        os.environ["RAYON_NUM_THREADS"] = str(RAYON_THREADS)
        max_workers = 1 if stop_after_tasks is not None else WORKERS
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=multiprocessing.get_context("spawn"),
        ) as pool:
            futures = {
                pool.submit(
                    _search_worker,
                    {
                        "root_task": root_task,
                        "task_path": str(
                            task_dir
                            / f"hand_{int(root_task['global_hand_index']):03d}.json"
                        ),
                        "library_path": str(repository_root / str(native["path"])),
                        "library_sha256": str(native["sha256"]),
                    },
                ): root_task
                for root_task in selected
            }
            pending = set(futures)
            while pending:
                done, pending = concurrent.futures.wait(
                    pending,
                    timeout=5.0,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    future.result()
                completed = len(list(task_dir.glob("hand_*.json")))
                _write_heartbeat(
                    heartbeat_path,
                    status="running_search",
                    total_tasks=spec.hand_count,
                    completed_tasks=completed,
                    pending_tasks=spec.hand_count - completed,
                    resumed_task_count=resumed_task_count,
                )

    remaining = [
        root_task
        for root_task in root_tasks
        if not (
            task_dir / f"hand_{int(root_task['global_hand_index']):03d}.json"
        ).exists()
    ]
    if remaining:
        report = {
            "schema": STEP6B_SUMMARY_SCHEMA,
            "status": "interrupted_for_resume_drill",
            "run_name": spec.run_name,
            "shard": spec.shard,
            "completed_tasks": spec.hand_count - len(remaining),
            "pending_tasks": len(remaining),
            "resumed_task_count": resumed_task_count,
            "spot_vm_started": True,
            "remaining_canary_shards_authorized": True,
            "current_profile_changed": False,
            "production_fanout_authorized": False,
        }
        _write_heartbeat(
            heartbeat_path,
            status=report["status"],
            total_tasks=spec.hand_count,
            completed_tasks=report["completed_tasks"],
            pending_tasks=report["pending_tasks"],
            resumed_task_count=resumed_task_count,
        )
        return report

    reports = []
    for root_task in root_tasks:
        path = task_dir / f"hand_{int(root_task['global_hand_index']):03d}.json"
        value = _load_json(path)
        _validate_recovered_search_task(
            value, root_task=root_task, library_sha256=str(native["sha256"])
        )
        reports.append(value)
    rows = sorted(
        (row for report in reports for row in report["rows"]),
        key=lambda row: row["root_index"],
    )
    fingerprints = [row["observation_fingerprint"] for row in rows]
    profiles = [report["profile"] for report in reports]
    candidate_keys: set[str] = set()
    evaluation_keys: set[str] = set()
    for root_task in root_tasks:
        seeds = root_task["seeds"]
        for raw in root_task["observations"]:
            observation = ActorObservation.from_dict(raw["observation"])
            candidate_keys.update(
                particle.rng_key_digest
                for particle in sample_hidden_card_particles(
                    observation,
                    base_seed=seeds["candidate"],
                    run_id=STEP6A_RUN_ID,
                    sample_count=REFERENCE_BUDGET.candidate_samples,
                ).particles
            )
            evaluation_keys.update(
                particle.rng_key_digest
                for particle in sample_hidden_card_particles(
                    observation,
                    base_seed=seeds["evaluation"],
                    run_id=STEP6A_RUN_ID,
                    sample_count=REFERENCE_BUDGET.evaluation_samples,
                ).particles
            )
    latency = {
        seat: _latency(
            [float(row["wall_seconds"]) for row in rows if row["seat"] == seat]
        )
        for seat in ("first", "second")
    }
    peak_rss = max(int(report["memory"]["peak_rss_bytes"]) for report in reports)
    expected_hand_indices = list(
        range(spec.global_hand_start, spec.global_hand_start + spec.hand_count)
    )
    gates = {
        "exact_shard_hand_indices": sorted(
            int(report["global_hand_index"]) for report in reports
        )
        == expected_hand_indices,
        "exactly_50_roots": len(rows) == spec.root_count,
        "exactly_25_each_seat": (
            sum(row["seat"] == "first" for row in rows)
            == sum(row["seat"] == "second" for row in rows)
            == spec.hand_count
        ),
        "five_profiles_equal_quota": all(
            profiles.count(profile) == 5 for profile in M31_T3_BEHAVIOR_PROFILES
        ),
        "profiles_follow_frozen_cycle": all(
            report["profile"]
            == behavior_profile_for_index(int(report["global_hand_index"]))
            for report in reports
        ),
        "unique_observation_fingerprints": len(set(fingerprints)) == len(rows),
        "all_task_and_decision_integrity": all(
            all(row["integrity"].values()) for row in rows
        ),
        "candidate_evaluation_rng_disjoint": not (candidate_keys & evaluation_keys),
        "candidate_rng_unique": len(candidate_keys)
        == len(rows) * REFERENCE_BUDGET.candidate_samples,
        "evaluation_rng_unique": len(evaluation_keys)
        == len(rows) * REFERENCE_BUDGET.evaluation_samples,
        "linux_portable_parity_bound": True,
        "resume_drill_recovered_task": resumed_task_count >= 1,
        "first_p95_within_180_seconds": latency["first"]["p95_seconds"]
        <= MAX_FIRST_P95_SECONDS,
        "second_p95_within_6_seconds": latency["second"]["p95_seconds"]
        <= MAX_SECOND_P95_SECONDS,
        "peak_rss_within_1_gib": peak_rss <= MAX_PEAK_RSS_BYTES,
        "canary_rows_not_training_eligible": True,
        "no_current_profile_or_production_fanout": True,
    }
    summary = {
        "schema": STEP6B_SUMMARY_SCHEMA,
        "status": "pass" if all(gates.values()) else "no_go",
        "run_name": spec.run_name,
        "shard": spec.shard,
        "global_hand_start": spec.global_hand_start,
        "source_package_sha256": source_package_sha256,
        "manifest_sha256": manifest_sha,
        "parity_report_sha256": _sha256(parity_report_path),
        "native_library_sha256": native["sha256"],
        "contract": {
            "paired_hands": spec.hand_count,
            "roots": spec.root_count,
            "workers": WORKERS,
            "rayon_threads_per_worker": RAYON_THREADS,
            "budget": REFERENCE_BUDGET.to_dict(),
            "run_id": STEP6A_RUN_ID,
        },
        "integrity": {
            "fingerprints": len(fingerprints),
            "unique_fingerprints": len(set(fingerprints)),
            "candidate_rng_keys": len(candidate_keys),
            "evaluation_rng_keys": len(evaluation_keys),
            "candidate_evaluation_overlap": len(candidate_keys & evaluation_keys),
            "profile_counts": {
                profile: profiles.count(profile) for profile in M31_T3_BEHAVIOR_PROFILES
            },
        },
        "performance": {
            "latency_by_seat": latency,
            "peak_process_rss_bytes": peak_rss,
        },
        "resumed_task_count": resumed_task_count,
        "gates": gates,
        "all_gates_passed": all(gates.values()),
        "task_manifest": [
            {
                "global_hand_index": report["global_hand_index"],
                "sha256": _sha256(
                    task_dir / f"hand_{int(report['global_hand_index']):03d}.json"
                ),
            }
            for report in reports
        ],
        "teacher_value_status": "diagnostic_not_match_EV",
        "training_eligible": False,
        "remaining_canary_shards_authorized": True,
        "current_profile_changed": False,
        "spot_vm_started": True,
        "production_fanout_authorized": False,
        "m31_complete": False,
    }
    _write_once(output_dir / "summary.json", summary)
    _write_heartbeat(
        heartbeat_path,
        status=summary["status"],
        total_tasks=spec.hand_count,
        completed_tasks=spec.hand_count,
        pending_tasks=0,
        resumed_task_count=resumed_task_count,
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--source-package-sha256", required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parity-golden", type=Path, required=True)
    parser.add_argument("--parity-only", action="store_true")
    parser.add_argument("--stop-after-tasks", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest, _spec, manifest_sha = _load_manifest_and_spec(
        manifest_path=args.manifest,
        schedule_path=args.schedule,
        source_package_sha256=args.source_package_sha256,
        shard=args.shard,
    )
    _verify_package_files(manifest, args.repository_root)
    parity_path = args.output_dir / "parity.json"
    parity = run_linux_parity(
        manifest=manifest,
        source_package_sha256=args.source_package_sha256,
        manifest_sha256=manifest_sha,
        repository_root=args.repository_root,
        parity_golden_path=args.parity_golden,
        output_path=parity_path,
    )
    if args.parity_only:
        print(json.dumps(parity, sort_keys=True))
        return 0
    result = run_shard(
        manifest_path=args.manifest,
        schedule_path=args.schedule,
        source_package_sha256=args.source_package_sha256,
        repository_root=args.repository_root,
        shard=args.shard,
        output_dir=args.output_dir,
        parity_report_path=parity_path,
        stop_after_tasks=args.stop_after_tasks,
    )
    print(json.dumps(result, sort_keys=True))
    return 75 if result["status"] == "interrupted_for_resume_drill" else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTHORIZED_SHARDS",
    "EXPECTED_FEATURE_ENCODER_SHA256",
    "EXPECTED_MODELS",
    "EXPECTED_NATIVE_LIBRARY_SHA256",
    "STEP6B_PACKAGE_SCHEMA",
    "STEP6B_PARITY_SCHEMA",
    "STEP6B_SHARD_SCHEMA",
    "STEP6B_SUMMARY_SCHEMA",
    "ShardSpec",
    "run_linux_parity",
    "run_shard",
    "seed_values",
]
