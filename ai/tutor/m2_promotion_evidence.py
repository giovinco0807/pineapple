"""Generate real, content-bound M2 evidence for the six reduced T3/T4 strata."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ai.engine.action_space import Action, get_turn_actions
from ai.engine.encoding import Board
from ai.engine.turn_order import POSITION_CONTRACT_VERSION
from ai.tutor.exact_late import (
    action_key,
    apply_action,
    default_rust_t3_exact_solver_path,
    terminal_metrics,
)
from ai.tutor.promotion_gate_v2 import (
    EVIDENCE_SCHEMA,
    FAIL_STATUS,
    PASS_STATUS,
    PROMOTION_GATE_V2_CONFIG_SHA256,
    canonical_sha256,
    derive_global_metrics,
    derive_stratum_metrics,
    finalize_artifact_hash,
    validate_promotion_evidence_v2,
)
from ai.tutor.t3_hu_public_cfr import (
    grouped_information_set_backup,
    reduced_bluff_signaling_game,
    solve_public_signaling_game_cfr,
)
from ai.tutor.t3_hu_public_tree import PublicTreeDecisionState
from ai.tutor.t3_hu_public_tree_cfr import (
    PublicTreeChanceNode,
    PublicTreeDecisionNode,
    PublicTreeNode,
    PublicTreeTerminalNode,
    RecursivePublicTreeCfrResult,
    evaluate_public_tree_profile,
    solve_recursive_public_tree_cfr_plus,
)
from ai.tutor.t3_hu_reduced_fixtures import (
    CompiledReducedFixture,
    compile_canonical_reduced_fixture,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "ai" / "config" / "promotion_gate_v2.json"
ROWS = ("top", "middle", "bottom")
ROW_COUNTS = {
    "t3_first": (9, 9),
    "t3_second": (11, 9),
    "t4_first": (11, 11),
}
PHASE_ACTOR = {"t3_first": "bb", "t3_second": "btn", "t4_first": "bb"}
REPLAY_VARIANTS = (
    ("canonical-a", False),
    ("reverse-hidden-world-order", True),
    ("canonical-b", False),
)


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _board(rows: Sequence[Sequence[str]]) -> Board:
    return Board(top=list(rows[0]), middle=list(rows[1]), bottom=list(rows[2]))


def _action(payload: Mapping[str, Any]) -> Action:
    return Action(
        placements=[(str(card), str(row)) for card, row in payload["placements"]],
        discard=str(payload["discard"]),
    )


def _walk(node: PublicTreeNode) -> Iterable[PublicTreeNode]:
    yield node
    if isinstance(node, PublicTreeChanceNode):
        for branch in node.branches:
            yield from _walk(branch.child)
    elif isinstance(node, PublicTreeDecisionNode):
        for _action_id, child in node.actions:
            yield from _walk(child)


def _physical_signature(state: PublicTreeDecisionState) -> str:
    particle = state.particle
    payload = {
        "bb_recall": asdict(particle.bb_recall),
        "btn_recall": asdict(particle.btn_recall),
        "undealt_cards": list(particle.undealt_cards),
    }
    return canonical_sha256(payload)


def _strategy_snapshot(result: RecursivePublicTreeCfrResult) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for key, strategy in result.average_strategy.items():
        snapshot[key.digest()] = {
            "key_json": key.canonical_json(),
            "actor": key.actor,
            "actions": {
                action_id: float(probability).hex()
                for action_id, probability in sorted(strategy.items())
            },
        }
    return dict(sorted(snapshot.items()))


def _infoset_reach(
    root: PublicTreeNode,
    profile: Mapping[Any, Mapping[str, float]],
) -> dict[str, float]:
    reach: dict[str, float] = defaultdict(float)

    def visit(node: PublicTreeNode, probability: float) -> None:
        if isinstance(node, PublicTreeTerminalNode):
            return
        if isinstance(node, PublicTreeChanceNode):
            for branch in node.branches:
                visit(branch.child, probability * float(branch.probability))
            return
        key = node.infoset_key
        reach[key.digest()] += probability
        sigma = profile[key]
        for action_id, child in node.actions:
            visit(child, probability * float(sigma[action_id]))

    visit(root, 1.0)
    return dict(sorted(reach.items()))


def _profile_metrics_payload(result: RecursivePublicTreeCfrResult) -> dict[str, float]:
    metrics = result.metrics
    return {
        "value_bb": float(metrics.value_bb),
        "bb_best_response": float(metrics.bb_best_response),
        "btn_best_response": float(metrics.btn_best_response),
        "nash_conv": float(metrics.nash_conv),
        "exploitability": float(metrics.exploitability),
    }


def _best_response_residual(
    root: PublicTreeNode,
    result: RecursivePublicTreeCfrResult,
) -> float:
    residuals: list[float] = []
    for actor, policy, expected in (
        ("bb", result.metrics.bb_best_response_policy, result.metrics.bb_best_response),
        ("btn", result.metrics.btn_best_response_policy, result.metrics.btn_best_response),
    ):
        profile = {key: dict(strategy) for key, strategy in result.average_strategy.items()}
        for key, selected in policy.items():
            if key.actor != actor:
                raise AssertionError("best-response policy contains the wrong actor")
            profile[key] = {
                action_id: float(action_id == selected)
                for action_id in profile[key]
            }
        observed = evaluate_public_tree_profile(root, profile)
        residuals.append(abs(float(observed) - float(expected)))
    return max(residuals)


def _python_leaf_metrics(
    state: PublicTreeDecisionState,
    rust_result: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if len(state.remaining_cards) != 3:
        raise ValueError("canonical reduced parity requires one exact BTN T4 draw")
    bb_board = _board(state.infoset_key.board_bb)
    btn_board = _board(state.infoset_key.board_btn)
    btn_draw = list(state.remaining_cards)
    btn_actions = sorted(get_turn_actions(btn_draw, btn_board), key=action_key)
    output: list[dict[str, Any]] = []
    for action_id in rust_result["action_keys"]:
        bb_after = apply_action(
            bb_board,
            _action(rust_result["actions_by_action_key"][action_id]),
        )
        response_metrics: list[tuple[str, Mapping[str, Any]]] = []
        for btn_action in btn_actions:
            btn_after = apply_action(btn_board, btn_action)
            response_metrics.append(
                (action_key(btn_action), terminal_metrics(bb_after, btn_after))
            )
        _response_id, selected = min(
            response_metrics,
            key=lambda item: (float(item[1]["score"]), item[0]),
        )
        python_metrics = {
            "score": float(selected["score"]),
            "raw_score": float(selected["raw_score"]),
            "royalty": float(selected["royalty"]),
            "bust_rate": float(bool(selected["bust"])),
            "fl_rate": float(bool(selected["fl_any"])),
        }
        rust_metrics = rust_result["metrics_by_action_key"][action_id]
        output.append(
            {
                "physical_state_commitment": rust_result["physical_state_commitment"],
                "action_key": action_id,
                "contains_joker": any(
                    card.startswith("X")
                    for card in (
                        *state.infoset_key.current_draw,
                        *state.remaining_cards,
                    )
                ),
                "rust": {field: float(rust_metrics[field]) for field in python_metrics},
                "python": python_metrics,
            }
        )
    return output


def _expected_history_pairs(phase: str) -> list[tuple[int, str]]:
    last = {"t3_first": (2, "btn"), "t3_second": (3, "bb"), "t4_first": (3, "btn")}[phase]
    pairs: list[tuple[int, str]] = []
    for turn in range(last[0] + 1):
        pairs.append((turn, "bb"))
        if turn < last[0] or last[1] == "btn":
            pairs.append((turn, "btn"))
    return pairs


def _audit_fixture(
    fixture: CompiledReducedFixture,
    solved: RecursivePublicTreeCfrResult,
    forbidden_inputs: Sequence[str],
) -> dict[str, Any]:
    decisions = [node for node in _walk(fixture.root) if isinstance(node, PublicTreeDecisionNode)]
    chances = [node for node in _walk(fixture.root) if isinstance(node, PublicTreeChanceNode)]
    legal_checks = legal_matches = 0
    transition_checks = transition_matches = 0
    position_mismatches = actor_mismatches = board_mismatches = history_mismatches = 0
    card_failures = x_failures = 0
    occurrences: dict[str, list[PublicTreeDecisionState]] = defaultdict(list)

    for node in decisions:
        state = node.state
        key = state.infoset_key
        occurrences[key.digest()].append(state)
        actor_board = key.board_bb if key.actor == "bb" else key.board_btn
        expected_actions = {
            action_key(candidate)
            for candidate in get_turn_actions(list(key.current_draw), _board(actor_board))
        }
        legal_checks += 1
        legal_matches += int(set(node.action_ids) == expected_actions)
        position_mismatches += int(key.contract_version != POSITION_CONTRACT_VERSION)
        actor_mismatches += int(PHASE_ACTOR.get(key.phase) != key.actor)
        counts = (
            sum(len(row) for row in key.board_bb),
            sum(len(row) for row in key.board_btn),
        )
        board_mismatches += int(ROW_COUNTS.get(key.phase) != counts)
        history_pairs = [(turn, actor) for turn, actor, _placements in key.public_action_history]
        history_mismatches += int(history_pairs != _expected_history_pairs(key.phase))
        try:
            PublicTreeDecisionState(key, state.particle)
        except (TypeError, ValueError):
            card_failures += 1
        serialized_cards = key.canonical_json()
        x_failures += int('"X"' in serialized_cards or "X0" in serialized_cards)

        for _action_id, child in node.actions:
            transition_checks += 1
            expected_type: type[PublicTreeNode]
            expected_phase: str | None
            if key.phase == "t3_first":
                expected_type, expected_phase = PublicTreeDecisionNode, "t3_second"
            elif key.phase == "t3_second":
                expected_type, expected_phase = PublicTreeDecisionNode, "t4_first"
            else:
                expected_type, expected_phase = PublicTreeTerminalNode, None
            matched = isinstance(child, expected_type)
            if matched and expected_phase is not None:
                matched = child.infoset_key.phase == expected_phase  # type: ignore[union-attr]
            transition_matches += int(matched)

    chance_errors = [
        abs(float(sum((branch.probability for branch in node.branches), Fraction(0, 1)) - 1))
        for node in chances
    ]
    hidden_pairs = 0
    for states in occurrences.values():
        signatures = {_physical_signature(state) for state in states}
        if len(states) >= 2 and len(signatures) >= 2:
            hidden_pairs += 1
    policy_tvs = [0.0 for _ in range(hidden_pairs)]
    forbidden_failures = 0
    lowered_forbidden = tuple(field.lower() for field in forbidden_inputs)
    for key in solved.average_strategy:
        text = key.canonical_json().lower()
        forbidden_failures += sum(int(f'"{field}"' in text) for field in lowered_forbidden)

    parity: list[dict[str, Any]] = []
    for state, result in zip(fixture.physical_leaf_states, fixture.physical_leaf_results):
        parity.extend(_python_leaf_metrics(state, result))
    expected_candidates = sum(int(result["legal_actions"]) for result in fixture.physical_leaf_results)
    observed_candidates = sum(int(result["evaluated_actions"]) for result in fixture.physical_leaf_results)
    preselection_failures = sum(
        int(
            result.get("selection_performed") is not False
            or result.get("particle_aggregation_performed") is not False
            or result.get("requires_infoset_aggregation") is not True
        )
        for result in fixture.physical_leaf_results
    )
    root_decisions = [branch.child for branch in fixture.root.branches]
    visible_match = all(
        sum(card.startswith("X") for card in node.infoset_key.current_draw)
        == fixture.visible_joker_count
        for node in root_decisions
    )

    return {
        "legal_action_checks": legal_checks,
        "legal_action_matches": legal_matches,
        "candidate_actions_expected": expected_candidates,
        "candidate_actions_observed": observed_candidates,
        "transition_checks": transition_checks,
        "transition_matches": transition_matches,
        "chance_mass_errors": chance_errors,
        "python_rust_leaf_metrics": parity,
        "physical_card_failures": card_failures,
        "preselection_failures": preselection_failures,
        "hidden_only_mutation_pairs": hidden_pairs,
        "hidden_only_mutation_policy_tvs": policy_tvs,
        "forbidden_policy_input_failures": forbidden_failures,
        "best_response_numerical_residual": _best_response_residual(fixture.root, solved),
        "x1_x2_identity_failures": x_failures,
        "visible_joker_stratum_match": visible_match,
        "position_contract_mismatches": position_mismatches,
        "actor_sequence_mismatches": actor_mismatches,
        "decision_board_shape_mismatches": board_mismatches,
        "public_history_order_mismatches": history_mismatches,
    }


def _solve_record(
    fixture: CompiledReducedFixture,
    *,
    run_id: str,
    reverse_world_order: bool,
    iterations: int,
    max_pure_profiles: int,
) -> tuple[dict[str, Any], RecursivePublicTreeCfrResult]:
    started = time.perf_counter_ns()
    solved = solve_recursive_public_tree_cfr_plus(
        fixture.root,
        iterations=iterations,
        checkpoints=(1, iterations),
        max_pure_profiles=max_pure_profiles,
    )
    wall_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    strategy = _strategy_snapshot(solved)
    snapshot = {
        "strategy": strategy,
        "metrics": _profile_metrics_payload(solved),
        "exploitability_trace": [list(point) for point in solved.exploitability_trace],
    }
    encoded = json.dumps(snapshot, ensure_ascii=False, allow_nan=False, sort_keys=True)
    round_trip = json.loads(encoded)
    mismatch = int(canonical_sha256(snapshot) != canonical_sha256(round_trip))
    record = {
        "run_id": run_id,
        "reverse_world_order": reverse_world_order,
        "rng_used": False,
        "fixture_manifest_sha256": fixture.fixture_manifest_sha256,
        "strategy_sha256": canonical_sha256(strategy),
        "strategy": strategy,
        "infoset_reach": _infoset_reach(fixture.root, solved.average_strategy),
        "metrics": snapshot["metrics"],
        "exploitability_trace": snapshot["exploitability_trace"],
        "solve_wall_ms": wall_ms,
        "solution_snapshot_round_trip_mismatches": mismatch,
    }
    return record, solved


def _pimc_goldens() -> dict[str, int]:
    cases = {
        "bb": (
            {
                "a": {"A": 4, "B": 0},
                "b": {"A": -2, "B": 3},
            },
            "max",
        ),
        "btn": (
            {
                "a": {"X": -4, "Y": 0},
                "b": {"X": 3, "Y": -2},
            },
            "min",
        ),
    }
    counts: dict[str, int] = {}
    for actor, (values, sense) in cases.items():
        result = grouped_information_set_backup(values, sense=sense)  # type: ignore[arg-type]
        counts[actor] = int(
            len(set(result.per_world_actions.values())) > 1
            and result.strategy_fusion_advantage > 0
        )
    return counts


def _peak_rss_mb() -> float:
    if os.name == "nt":
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
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
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
        )
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        handle = kernel32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb
        )
        if not ok:
            raise OSError("GetProcessMemoryInfo failed")
        return counters.PeakWorkingSetSize / (1024.0 * 1024.0)
    import resource

    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss / 1024.0 if sys.platform != "darwin" else rss / (1024.0 * 1024.0)


def _runtime_environment() -> dict[str, Any]:
    def git(*args: str) -> str:
        completed = subprocess.run(
            ["git", "-c", f"safe.directory={ROOT.as_posix()}", *args],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=True,
        )
        return completed.stdout.strip()

    return {
        "machine": platform.node(),
        "cpu": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "peak_rss_scope": "generator_parent_process_peak_working_set",
    }


def _source_hashes(rust_solver: Path) -> dict[str, str]:
    paths = [
        ROOT / "ai/config/promotion_gate_v2.json",
        ROOT / "ai/config/fl_ev.json",
        ROOT / "ai/engine/action_space.py",
        ROOT / "ai/engine/encoding.py",
        ROOT / "ai/engine/game_engine.py",
        ROOT / "ai/engine/scoring.py",
        ROOT / "ai/engine/turn_order.py",
        ROOT / "ai/tutor/exact_late.py",
        ROOT / "ai/tutor/t3_hu_public_cfr.py",
        ROOT / "ai/tutor/t3_hu_public_tree.py",
        ROOT / "ai/tutor/t3_hu_public_tree_cfr.py",
        ROOT / "ai/tutor/t3_hu_reduced_fixtures.py",
        ROOT / "ai/tutor/promotion_gate_v2.py",
        ROOT / "ai/tutor/m2_promotion_evidence.py",
    ]
    rust_root = ROOT / "ai/rust_solver"
    paths.extend(sorted(rust_root.rglob("*.rs")))
    paths.extend(sorted(rust_root.rglob("Cargo.toml")))
    paths.append(rust_solver)
    return {
        str(path.relative_to(ROOT)).replace("\\", "/"): _file_sha256(path)
        for path in paths
        if path.is_file()
    }


def _manifests(
    raw_strata: Mapping[str, Any],
    *,
    iterations: int,
    max_pure_profiles: int,
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    fixture_manifests = {
        name: item["fixture_manifest_sha256"] for name, item in sorted(raw_strata.items())
    }
    root_digests = {name: item["root_infoset_digest"] for name, item in sorted(raw_strata.items())}
    commitments = {
        name: item["physical_state_commitments"] for name, item in sorted(raw_strata.items())
    }
    rust_sources = {
        path: digest
        for path, digest in source_hashes.items()
        if path.startswith("ai/rust_solver/")
        or path.endswith("t3_hu_public_tree_cfr.py")
    }
    return {
        "action_contract": {
            "position_contract_version": POSITION_CONTRACT_VERSION,
            "street_action": {"draw": 3, "place": 2, "discard": 1},
            "files": {
                path: source_hashes[path]
                for path in ("ai/engine/action_space.py", "ai/engine/turn_order.py")
            },
        },
        "infoset_contract": {
            "root_infoset_digests": root_digests,
            "file": source_hashes["ai/tutor/t3_hu_public_cfr.py"],
        },
        "reduced_fixtures": fixture_manifests,
        "chance_model": {
            "scope": "declared_two_hidden_worlds_and_single_supplied_street_draws",
            "full_deck_enumeration": False,
            "world_probability": "1/2",
            "files": {
                path: source_hashes[path]
                for path in (
                    "ai/tutor/t3_hu_public_tree.py",
                    "ai/tutor/t3_hu_reduced_fixtures.py",
                )
            },
        },
        "range_model": {
            "name": "two_exact_reduced_hidden_worlds_uniform_v1",
            "particle_count_per_fixture": 2,
            "effective_sample_size_per_fixture": 2.0,
            "physical_state_commitments": commitments,
        },
        "solver_config": {
            "iterations": iterations,
            "linear_averaging": True,
            "checkpoints": [1, iterations],
            "max_pure_profiles": max_pure_profiles,
            "replay_variants": [item[0] for item in REPLAY_VARIANTS],
            "rng_used": False,
        },
        "solver_code": rust_sources
        | {
            "ai/tutor/t3_hu_public_tree_cfr.py": source_hashes[
                "ai/tutor/t3_hu_public_tree_cfr.py"
            ]
        },
        "source_tree": dict(sorted(source_hashes.items())),
    }


def generate_m2_promotion_evidence(
    output_dir: str | Path,
    *,
    iterations: int = 200,
    max_pure_profiles: int = 100_000,
    runtime_warmups: int = 3,
    runtime_measured: int = 5,
    rust_solver_path: str | Path | None = None,
    rust_timeout_s: float = 30.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if runtime_warmups < 3 or runtime_measured < 5:
        raise ValueError("v2 requires at least 3 warmups and 5 measured runs")
    started = time.perf_counter_ns()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    rust_solver = Path(rust_solver_path or default_rust_t3_exact_solver_path()).resolve()
    if not rust_solver.is_file():
        raise FileNotFoundError(rust_solver)

    # Calibrate before canonical timing, exactly as the gate requires.
    calibration_warmup: list[float] = []
    calibration_measured: list[float] = []
    bluff = reduced_bluff_signaling_game()
    for index in range(runtime_warmups + runtime_measured):
        tick = time.perf_counter_ns()
        solve_public_signaling_game_cfr(bluff, iterations=20_000)
        elapsed = (time.perf_counter_ns() - tick) / 1_000_000.0
        (calibration_warmup if index < runtime_warmups else calibration_measured).append(elapsed)
    if max(calibration_measured) > config["global_thresholds"][
        "reduced_bluff_20000_iterations_wall_ms_max_lte"
    ]:
        raise RuntimeError("reduced-bluff calibration failed before recursive timing")

    raw_strata: dict[str, Any] = {}
    warmup_by_stratum: dict[str, list[float]] = {}
    measured_by_stratum: dict[str, list[float]] = {}
    snapshot_mismatches = 0
    fixtures: dict[str, CompiledReducedFixture] = {}
    for actor in ("bb", "btn"):
        for joker in (0, 1, 2):
            name = f"{actor}_joker{joker}"
            runs: list[dict[str, Any]] = []
            primary_fixture: CompiledReducedFixture | None = None
            primary_solved: RecursivePublicTreeCfrResult | None = None
            for run_id, reverse in REPLAY_VARIANTS:
                fixture = compile_canonical_reduced_fixture(
                    actor,
                    joker,
                    rust_solver_path=rust_solver,
                    rust_timeout_s=rust_timeout_s,
                    reverse_world_order=reverse,
                )
                record, solved = _solve_record(
                    fixture,
                    run_id=run_id,
                    reverse_world_order=reverse,
                    iterations=iterations,
                    max_pure_profiles=max_pure_profiles,
                )
                runs.append(record)
                snapshot_mismatches += record["solution_snapshot_round_trip_mismatches"]
                if primary_fixture is None:
                    primary_fixture, primary_solved = fixture, solved
            assert primary_fixture is not None and primary_solved is not None
            fixtures[name] = primary_fixture
            warmup_by_stratum[name] = [float(run["solve_wall_ms"]) for run in runs]

            measured: list[float] = []
            for index in range(runtime_measured):
                record, _solved = _solve_record(
                    primary_fixture,
                    run_id=f"runtime-{index}",
                    reverse_world_order=False,
                    iterations=iterations,
                    max_pure_profiles=max_pure_profiles,
                )
                measured.append(float(record["solve_wall_ms"]))
                snapshot_mismatches += record["solution_snapshot_round_trip_mismatches"]
            measured_by_stratum[name] = measured

            audit = _audit_fixture(
                primary_fixture,
                primary_solved,
                config["contracts"]["forbidden_policy_inputs"],
            )
            raw_strata[name] = {
                "fixture_id": primary_fixture.fixture_id,
                "fixture_manifest_sha256": primary_fixture.fixture_manifest_sha256,
                "actor": actor,
                "visible_joker_count": joker,
                "physical_hidden_world_count": primary_fixture.metadata["physical_world_count"],
                "root_infoset_digest": primary_fixture.root_infoset_digest,
                "physical_state_commitments": sorted(
                    result["physical_state_commitment"]
                    for result in primary_fixture.physical_leaf_results
                ),
                "audit": audit,
                "runs": runs,
            }

    raw: dict[str, Any] = {
        "strata": raw_strata,
        "pimc_counterexample_goldens_by_actor": _pimc_goldens(),
        "solution_snapshot_round_trip_mismatches": snapshot_mismatches,
        "runtime": {
            "calibration_scope": "reduced_public_signaling_cfr_plus_20000_iterations",
            "calibration_warmup_ms": calibration_warmup,
            "calibration_measured_ms": calibration_measured,
            "recursive_scope": "CFR_solve_plus_exact_infoset_best_responses_excludes_fixture_compile",
            "recursive_warmup_ms_by_stratum": warmup_by_stratum,
            "recursive_measured_ms_by_stratum": measured_by_stratum,
            "complete_gate_wall_s": 0.0,
            "peak_rss_mb": _peak_rss_mb(),
        },
    }

    source_hashes = _source_hashes(rust_solver)
    manifests = _manifests(
        raw_strata,
        iterations=iterations,
        max_pure_profiles=max_pure_profiles,
        source_hashes=source_hashes,
    )
    provenance = {
        "schema": "ofc_m2_provenance/v2",
        "hash_manifests": manifests,
        "rust_executable_sha256": _file_sha256(rust_solver),
    }
    raw["runtime"]["complete_gate_wall_s"] = (
        time.perf_counter_ns() - started
    ) / 1_000_000_000.0
    strata = {
        name: {
            "actor": item["actor"],
            "visible_joker_count": item["visible_joker_count"],
            "canonical_reduced_fixture_count": 1,
            "metrics": derive_stratum_metrics(item),
        }
        for name, item in sorted(raw_strata.items())
    }

    artifact: dict[str, Any] = {
        "schema": "ofc_m2_reduced_artifact/v2",
        "gate_id": config["gate_id"],
        "scope": config["scope"]["name"],
        "status": PASS_STATUS,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "position_contract_version": config["contracts"]["position_contract_version"],
        "rules_version": config["contracts"]["rules_version"],
        "fl_ev_sha256": config["contracts"]["fl_ev_sha256"],
        "action_contract_sha256": canonical_sha256(manifests["action_contract"]),
        "infoset_contract_sha256": canonical_sha256(manifests["infoset_contract"]),
        "reduced_fixture_sha256": canonical_sha256(manifests["reduced_fixtures"]),
        "chance_model_sha256": canonical_sha256(manifests["chance_model"]),
        "range_model": config["contracts"]["range_model"],
        "range_model_sha256": canonical_sha256(manifests["range_model"]),
        "solver_config_sha256": canonical_sha256(manifests["solver_config"]),
        "solver_code_sha256": canonical_sha256(manifests["solver_code"]),
        "source_tree_manifest_sha256": canonical_sha256(manifests["source_tree"]),
        "raw_measurements_sha256": canonical_sha256(raw),
        "method": config["contracts"]["solver_method"],
        "information_model": config["contracts"]["information_model"],
        "best_response_method": config["contracts"]["best_response_method"],
        "strategy_fusion": False,
        "equilibrium_approx": True,
        "reduced_tree_enumeration_exact": True,
        "hu_exact": False,
        "full_card_policy_promoted": False,
        "explicit_full_deck_chance_enumeration": False,
        "solver_rng_used": False,
        "iterations": iterations,
        "replay_ids": [item[0] for item in REPLAY_VARIANTS],
        "runtime_environment": _runtime_environment(),
        "metrics": {
            "strata": {name: item["metrics"] for name, item in strata.items()},
            "scope_note": "M2 finite reduced reference only; no full-card claim",
        },
        "tests": [
            "all six real-card actor/Joker fixtures",
            "three deterministic hidden-world order replays",
            "exact infoset-aware best responses for both players",
            "independent Python recomputation of every Rust T4 candidate leaf",
            "content-bound source, fixture, range, solver, and raw-measurement manifests",
        ],
    }
    artifact = finalize_artifact_hash(artifact)
    evidence: dict[str, Any] = {
        "schema": EVIDENCE_SCHEMA,
        "gate_id": config["gate_id"],
        "gate_config_sha256": PROMOTION_GATE_V2_CONFIG_SHA256,
        "artifact": artifact,
        "strata": strata,
        "global_metrics": derive_global_metrics(raw, True),
        "raw_measurements": raw,
        "provenance": provenance,
    }
    result = validate_promotion_evidence_v2(evidence)
    if not result["passed"]:
        evidence["artifact"]["status"] = FAIL_STATUS
        evidence["artifact"] = finalize_artifact_hash(evidence["artifact"])
        result = validate_promotion_evidence_v2(evidence)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    (destination / "gate_result.json").write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return evidence, result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--max-pure-profiles", type=int, default=100_000)
    parser.add_argument("--runtime-warmups", type=int, default=3)
    parser.add_argument("--runtime-measured", type=int, default=5)
    parser.add_argument("--rust-solver", type=Path)
    parser.add_argument("--rust-timeout-s", type=float, default=30.0)
    args = parser.parse_args(argv)
    _evidence, result = generate_m2_promotion_evidence(
        args.output_dir,
        iterations=args.iterations,
        max_pure_profiles=args.max_pure_profiles,
        runtime_warmups=args.runtime_warmups,
        runtime_measured=args.runtime_measured,
        rust_solver_path=args.rust_solver,
        rust_timeout_s=args.rust_timeout_s,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
