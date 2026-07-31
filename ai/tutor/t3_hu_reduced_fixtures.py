"""Canonical real-card reduced fixtures for the M2 public-tree gate.

The first fixture implemented here is ``bb_joker0``.  It is intentionally a
real, fully legal T3->T4 tree rather than a synthetic payoff table:

* chance chooses one of two physical hidden BTN-recall/range worlds;
* both worlds expose the same BB T3 information set;
* every legal BB T3 and BTN T3 action is retained;
* the supplied BB T4 draw is resolved for every public branch; and
* every legal BB T4 action receives its conditioned Rust terminal-response
  value before the recursive tree is constructed.

No physical world selects an action.  Shared ``InfoSetKey`` nodes are merged
only by the CFR strategy table, while terminal utilities remain attached to
their physical chance histories.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

from ai.engine.action_space import get_turn_actions
from ai.engine.encoding import Board
from ai.engine.turn_order import POSITION_CONTRACT_VERSION
from ai.tutor.exact_late import (
    action_key,
    evaluate_physical_bb_t4_action_vectors_rust_batch,
)
from ai.tutor.t3_hu_public_cfr import JointParticle, PrivateRecall
from ai.tutor.t3_hu_public_tree import (
    PublicTreeDecisionState,
    SuppliedChanceDraw,
    apply_public_tree_action,
    resolve_supplied_chance,
)
from ai.tutor.t3_hu_public_tree_cfr import (
    PublicTreeChanceBranch,
    PublicTreeChanceNode,
    PublicTreeDecisionNode,
    PublicTreeNode,
    public_tree_t4_decision_from_physical_result,
)


ROWS = ("top", "middle", "bottom")
FIXTURE_ID_BB_JOKER0 = "bb_joker0_canonical_reduced_v1"

BB_BOARD = (
    ("4h", "2c", "3d"),
    ("9s", "7c", "8h", "7d", "Tc"),
    ("Qc",),
)
BTN_BOARD = (
    ("8c", "5c", "6d"),
    ("Kc", "9c", "Jh", "9d", "Qs"),
    ("As",),
)
PUBLIC_HISTORY = (
    (0, "bb", (("4h", "top"), ("2c", "top"), ("3d", "top"), ("7c", "middle"), ("Qc", "bottom"))),
    (0, "btn", (("8c", "top"), ("5c", "top"), ("6d", "top"), ("9c", "middle"), ("As", "bottom"))),
    (1, "bb", (("7d", "middle"), ("8h", "middle"))),
    (1, "btn", (("9d", "middle"), ("Jh", "middle"))),
    (2, "bb", (("9s", "middle"), ("Tc", "middle"))),
    (2, "btn", (("Qs", "middle"), ("Kc", "middle"))),
)
BB_BOARD_AFTER_T3 = (
    BB_BOARD[0],
    BB_BOARD[1],
    ("Qc", "Ad", "Kd"),
)
PUBLIC_HISTORY_AFTER_BB_T3 = PUBLIC_HISTORY + (
    (3, "bb", (("Ad", "bottom"), ("Kd", "bottom"))),
)
BTN_T3_DRAW = ("2h", "3h", "5h")
BB_T4_DRAW = ("Ah", "Kh", "Jd")


@dataclass(frozen=True)
class CompiledReducedFixture:
    fixture_id: str
    actor: str
    visible_joker_count: int
    root: PublicTreeNode
    physical_leaf_results: tuple[Mapping[str, Any], ...]
    physical_leaf_states: tuple[PublicTreeDecisionState, ...]
    root_infoset_digest: str
    shared_t4_infoset_pairs: int
    fixture_manifest_sha256: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class _WorldSpec:
    world_id: str
    btn_t2_discard: str
    final_draw: tuple[str, str, str]


@dataclass(frozen=True)
class _BtnWorldSpec:
    world_id: str
    bb_t3_discard: str
    final_draw: tuple[str, str, str]


@dataclass(frozen=True)
class _LeafTask:
    world_id: str
    bb_action_key: str
    btn_action_key: str
    state: PublicTreeDecisionState
    payload: Mapping[str, Any]


def _bb_fixture_id(visible_joker_count: int) -> str:
    return f"bb_joker{visible_joker_count}_canonical_reduced_v1"


def _bb_t3_draw(visible_joker_count: int) -> tuple[str, str, str]:
    draws = {
        0: ("Ad", "Kd", "Qd"),
        1: ("Ad", "Kd", "X1"),
        2: ("Ad", "X1", "X2"),
    }
    try:
        return draws[visible_joker_count]
    except KeyError as exc:
        raise ValueError("visible_joker_count must be 0, 1, or 2") from exc


def _bb_world_specs(visible_joker_count: int) -> tuple[_WorldSpec, _WorldSpec]:
    alternate_final = {0: "X1", 1: "X2", 2: "5s"}[visible_joker_count]
    return (
        _WorldSpec("world-a", "7h", ("2d", "3c", "4d")),
        _WorldSpec("world-b", "5d", ("2d", "3c", alternate_final)),
    )


def _btn_fixture_id(visible_joker_count: int) -> str:
    return f"btn_joker{visible_joker_count}_canonical_reduced_v1"


def _btn_t3_draw(visible_joker_count: int) -> tuple[str, str, str]:
    draws = {
        0: ("2h", "3h", "5h"),
        1: ("2h", "3h", "X1"),
        2: ("2h", "X1", "X2"),
    }
    try:
        return draws[visible_joker_count]
    except KeyError as exc:
        raise ValueError("visible_joker_count must be 0, 1, or 2") from exc


def _btn_world_specs(visible_joker_count: int) -> tuple[_BtnWorldSpec, _BtnWorldSpec]:
    alternate_final = {0: "X1", 1: "X2", 2: "5s"}[visible_joker_count]
    return (
        _BtnWorldSpec("world-a", "Qd", ("2d", "3c", "4d")),
        _BtnWorldSpec("world-b", "Jc", ("2d", "3c", alternate_final)),
    )


def _recall(*, actor: str, t2_discard: str) -> PrivateRecall:
    placements = {
        "bb": {1: ("7d", "8h"), 2: ("9s", "Tc")},
        "btn": {1: ("9d", "Jh"), 2: ("Qs", "Kc")},
    }[actor]
    t1_discard = "6c" if actor == "bb" else "4s"
    return PrivateRecall(
        dealt_by_turn=(
            (1, (*placements[1], t1_discard)),
            (2, (*placements[2], t2_discard)),
        ),
        discards_by_turn=((1, t1_discard), (2, t2_discard)),
    )


def _initial_state(
    spec: _WorldSpec,
    *,
    current_draw: tuple[str, str, str],
) -> PublicTreeDecisionState:
    future = (*BTN_T3_DRAW, *BB_T4_DRAW, *spec.final_draw)
    particle = JointParticle(
        bb_recall=_recall(actor="bb", t2_discard="6s"),
        btn_recall=_recall(actor="btn", t2_discard=spec.btn_t2_discard),
        undealt_cards=future,
        weight=Fraction(1, 2),
    )
    return PublicTreeDecisionState.from_particle(
        particle,
        phase="t3_first",
        board_bb=BB_BOARD,
        board_btn=BTN_BOARD,
        public_action_history=PUBLIC_HISTORY,
        current_draw=current_draw,
    )


def _btn_initial_state(
    spec: _BtnWorldSpec,
    *,
    current_draw: tuple[str, str, str],
) -> PublicTreeDecisionState:
    base_bb = _recall(actor="bb", t2_discard="6s")
    bb_recall = PrivateRecall(
        dealt_by_turn=base_bb.dealt_by_turn
        + ((3, ("Ad", "Kd", spec.bb_t3_discard)),),
        discards_by_turn=base_bb.discards_by_turn + ((3, spec.bb_t3_discard),),
    )
    particle = JointParticle(
        bb_recall=bb_recall,
        btn_recall=_recall(actor="btn", t2_discard="7h"),
        undealt_cards=(*BB_T4_DRAW, *spec.final_draw),
        weight=Fraction(1, 2),
    )
    return PublicTreeDecisionState.from_particle(
        particle,
        phase="t3_second",
        board_bb=BB_BOARD_AFTER_T3,
        board_btn=BTN_BOARD,
        public_action_history=PUBLIC_HISTORY_AFTER_BB_T3,
        current_draw=current_draw,
    )


def _t4_payload(state: PublicTreeDecisionState, particle_id: str) -> dict[str, Any]:
    key = state.infoset_key
    return {
        "turn": 4,
        "actor": "bb",
        "is_btn": False,
        "first_actor": "bb",
        "position_contract_version": POSITION_CONTRACT_VERSION,
        "particle_id": particle_id,
        "board_self": {
            row: list(cards) for row, cards in zip(ROWS, key.board_bb)
        },
        "board_opponent": {
            row: list(cards) for row, cards in zip(ROWS, key.board_btn)
        },
        "dealt_cards": list(key.current_draw),
        "remaining_cards": list(state.particle.undealt_cards),
    }


def _manifest(
    *,
    fixture_id: str,
    actor: str,
    visible_joker_count: int,
    world_ids: Sequence[str],
    root_digest: str,
    tasks: Sequence[_LeafTask],
    results: Sequence[Mapping[str, Any]],
) -> str:
    leaves = []
    for task, result in zip(tasks, results):
        stable_metrics = {}
        for action_id in result["action_keys"]:
            metrics = result["metrics_by_action_key"][action_id]
            stable_metrics[action_id] = {
                field: metrics.get(field)
                for field in (
                    "score",
                    "raw_score",
                    "royalty",
                    "bust_rate",
                    "fl_rate",
                    "samples",
                    "remaining_deck_size",
                    "enumerated_draws",
                )
            }
        leaves.append(
            {
                "path": [task.world_id, task.bb_action_key, task.btn_action_key],
                "physical_state_commitment": result["physical_state_commitment"],
                "action_keys": list(result["action_keys"]),
                "metrics_by_action_key": stable_metrics,
            }
        )
    leaves.sort(key=lambda leaf: leaf["path"])
    canonical = {
        "fixture_id": fixture_id,
        "actor": actor,
        "visible_joker_count": visible_joker_count,
        "position_contract_version": POSITION_CONTRACT_VERSION,
        "root_infoset_digest": root_digest,
        "worlds": sorted(world_ids),
        "leaves": leaves,
    }
    raw = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def compile_bb_joker0_reduced_fixture(
    *,
    rust_solver_path: str | Path | None = None,
    rust_timeout_s: float = 120.0,
    reverse_world_order: bool = False,
) -> CompiledReducedFixture:
    """Compile the first real-card, all-action M2 fixture."""
    return compile_bb_reduced_fixture(
        0,
        rust_solver_path=rust_solver_path,
        rust_timeout_s=rust_timeout_s,
        reverse_world_order=reverse_world_order,
    )


def compile_bb_reduced_fixture(
    visible_joker_count: int,
    *,
    rust_solver_path: str | Path | None = None,
    rust_timeout_s: float = 120.0,
    reverse_world_order: bool = False,
) -> CompiledReducedFixture:
    """Compile one BB-root canonical fixture for the requested visible Joker stratum."""
    current_draw = _bb_t3_draw(visible_joker_count)
    fixture_id = _bb_fixture_id(visible_joker_count)
    canonical_specs = _bb_world_specs(visible_joker_count)
    specs = tuple(reversed(canonical_specs)) if reverse_world_order else canonical_specs
    world_states = {
        spec.world_id: _initial_state(spec, current_draw=current_draw)
        for spec in specs
    }
    root_keys = {state.infoset_key for state in world_states.values()}
    if len(root_keys) != 1:
        raise AssertionError("BB hidden-only worlds must share one root InfoSetKey")
    root_key = next(iter(root_keys))
    if sum(card.startswith("X") for card in root_key.current_draw) != visible_joker_count:
        raise AssertionError("BB fixture root visible Joker stratum mismatch")

    tasks: list[_LeafTask] = []
    btn_states: dict[tuple[str, str], PublicTreeDecisionState] = {}
    btn_action_keys: dict[tuple[str, str], tuple[str, ...]] = {}
    t4_states: dict[tuple[str, str, str], PublicTreeDecisionState] = {}
    for spec in specs:
        root_state = world_states[spec.world_id]
        root_actions = sorted(
            get_turn_actions(
                list(root_state.infoset_key.current_draw),
                Board(
                    top=list(root_state.infoset_key.board_bb[0]),
                    middle=list(root_state.infoset_key.board_bb[1]),
                    bottom=list(root_state.infoset_key.board_bb[2]),
                ),
            ),
            key=action_key,
        )
        for bb_action in root_actions:
            bb_key = action_key(bb_action)
            btn_state = resolve_supplied_chance(
                apply_public_tree_action(root_state, bb_action),
                (SuppliedChanceDraw(BTN_T3_DRAW, 1),),
            )[0].state
            btn_states[(spec.world_id, bb_key)] = btn_state
            actions = sorted(
                get_turn_actions(
                    list(btn_state.infoset_key.current_draw),
                    Board(
                        top=list(btn_state.infoset_key.board_btn[0]),
                        middle=list(btn_state.infoset_key.board_btn[1]),
                        bottom=list(btn_state.infoset_key.board_btn[2]),
                    ),
                ),
                key=action_key,
            )
            btn_action_keys[(spec.world_id, bb_key)] = tuple(action_key(action) for action in actions)
            for btn_action in actions:
                response_key = action_key(btn_action)
                t4_state = resolve_supplied_chance(
                    apply_public_tree_action(btn_state, btn_action),
                    (SuppliedChanceDraw(BB_T4_DRAW, 1),),
                )[0].state
                path = (spec.world_id, bb_key, response_key)
                t4_states[path] = t4_state
                tasks.append(
                    _LeafTask(
                        world_id=spec.world_id,
                        bb_action_key=bb_key,
                        btn_action_key=response_key,
                        state=t4_state,
                        payload=_t4_payload(
                            t4_state,
                            f"{fixture_id}:{spec.world_id}:{len(tasks)}",
                        ),
                    )
                )

    results = evaluate_physical_bb_t4_action_vectors_rust_batch(
        [dict(task.payload) for task in tasks],
        rust_solver_path=rust_solver_path,
        timeout_s=rust_timeout_s,
        position_parallel=True,
    )
    if len(results) != len(tasks):
        raise RuntimeError(f"fixture Rust leaf batch returned {len(results)}/{len(tasks)}")
    leaves = {
        (task.world_id, task.bb_action_key, task.btn_action_key):
        public_tree_t4_decision_from_physical_result(result, task.state)
        for task, result in zip(tasks, results)
    }

    root_nodes: dict[str, PublicTreeDecisionNode] = {}
    for spec in specs:
        root_state = world_states[spec.world_id]
        by_root_action: dict[str, PublicTreeDecisionNode] = {}
        root_action_ids = sorted(
            {
                task.bb_action_key
                for task in tasks
                if task.world_id == spec.world_id
            }
        )
        for bb_key in root_action_ids:
            btn_state = btn_states[(spec.world_id, bb_key)]
            by_root_action[bb_key] = PublicTreeDecisionNode(
                btn_state,
                {
                    btn_key: leaves[(spec.world_id, bb_key, btn_key)]
                    for btn_key in btn_action_keys[(spec.world_id, bb_key)]
                },
            )
        root_nodes[spec.world_id] = PublicTreeDecisionNode(root_state, by_root_action)

    root = PublicTreeChanceNode(
        PublicTreeChanceBranch(
            spec.world_id,
            Fraction(1, len(specs)),
            root_nodes[spec.world_id],
        )
        for spec in specs
    )

    paths_without_world = sorted(
        {(task.bb_action_key, task.btn_action_key) for task in tasks}
    )
    shared_pairs = 0
    for bb_key, btn_key in paths_without_world:
        keys = {
            t4_states[(spec.world_id, bb_key, btn_key)].infoset_key
            for spec in specs
        }
        if len(keys) != 1:
            raise AssertionError("hidden BTN worlds must share each BB T4 InfoSetKey")
        shared_pairs += 1

    result_commitments = {
        task.world_id: {
            str(result["physical_state_commitment"])
            for current_task, result in zip(tasks, results)
            if current_task.world_id == task.world_id
        }
        for task in tasks
    }
    if result_commitments["world-a"] == result_commitments["world-b"]:
        raise AssertionError("fixture hidden worlds must condition distinct physical ranges")

    manifest = _manifest(
        fixture_id=fixture_id,
        actor="bb",
        visible_joker_count=visible_joker_count,
        world_ids=[spec.world_id for spec in canonical_specs],
        root_digest=root_key.digest(),
        tasks=tasks,
        results=results,
    )
    metadata = {
        "method": "canonical_reduced_real_card_fixture",
        "fixture_id": fixture_id,
        "actor": "bb",
        "visible_joker_count": visible_joker_count,
        "position_contract_version": POSITION_CONTRACT_VERSION,
        "physical_world_count": len(specs),
        "root_legal_actions": len(root_nodes[specs[0].world_id].action_ids),
        "btn_decision_nodes": len(btn_states),
        "physical_t4_leaf_positions": len(tasks),
        "shared_t4_infoset_pairs": shared_pairs,
        "selection_performed_in_physical_world": False,
        "strategy_fusion": False,
        "rust_leaf_integrated": True,
        "full_card": False,
        "hu_exact": False,
    }
    return CompiledReducedFixture(
        fixture_id=fixture_id,
        actor="bb",
        visible_joker_count=visible_joker_count,
        root=root,
        physical_leaf_results=tuple(results),
        physical_leaf_states=tuple(task.state for task in tasks),
        root_infoset_digest=root_key.digest(),
        shared_t4_infoset_pairs=shared_pairs,
        fixture_manifest_sha256=manifest,
        metadata=metadata,
    )


def compile_btn_reduced_fixture(
    visible_joker_count: int,
    *,
    rust_solver_path: str | Path | None = None,
    rust_timeout_s: float = 120.0,
    reverse_world_order: bool = False,
) -> CompiledReducedFixture:
    """Compile one BTN-root canonical fixture with hidden BB T3 recall worlds."""
    current_draw = _btn_t3_draw(visible_joker_count)
    fixture_id = _btn_fixture_id(visible_joker_count)
    canonical_specs = _btn_world_specs(visible_joker_count)
    specs = tuple(reversed(canonical_specs)) if reverse_world_order else canonical_specs
    world_states = {
        spec.world_id: _btn_initial_state(spec, current_draw=current_draw)
        for spec in specs
    }
    root_keys = {state.infoset_key for state in world_states.values()}
    if len(root_keys) != 1:
        raise AssertionError("BTN hidden-only worlds must share one root InfoSetKey")
    root_key = next(iter(root_keys))
    if sum(card.startswith("X") for card in root_key.current_draw) != visible_joker_count:
        raise AssertionError("BTN fixture root visible Joker stratum mismatch")

    tasks: list[_LeafTask] = []
    action_keys_by_world: dict[str, tuple[str, ...]] = {}
    t4_states: dict[tuple[str, str], PublicTreeDecisionState] = {}
    for spec in specs:
        state = world_states[spec.world_id]
        actions = sorted(
            get_turn_actions(
                list(state.infoset_key.current_draw),
                Board(
                    top=list(state.infoset_key.board_btn[0]),
                    middle=list(state.infoset_key.board_btn[1]),
                    bottom=list(state.infoset_key.board_btn[2]),
                ),
            ),
            key=action_key,
        )
        action_keys_by_world[spec.world_id] = tuple(action_key(action) for action in actions)
        for btn_action in actions:
            btn_key = action_key(btn_action)
            t4_state = resolve_supplied_chance(
                apply_public_tree_action(state, btn_action),
                (SuppliedChanceDraw(BB_T4_DRAW, 1),),
            )[0].state
            t4_states[(spec.world_id, btn_key)] = t4_state
            tasks.append(
                _LeafTask(
                    world_id=spec.world_id,
                    bb_action_key="fixed-bb-t3-public-action",
                    btn_action_key=btn_key,
                    state=t4_state,
                    payload=_t4_payload(
                        t4_state,
                        f"{fixture_id}:{spec.world_id}:{len(tasks)}",
                    ),
                )
            )

    results = evaluate_physical_bb_t4_action_vectors_rust_batch(
        [dict(task.payload) for task in tasks],
        rust_solver_path=rust_solver_path,
        timeout_s=rust_timeout_s,
        position_parallel=True,
    )
    if len(results) != len(tasks):
        raise RuntimeError(f"fixture Rust leaf batch returned {len(results)}/{len(tasks)}")
    leaves = {
        (task.world_id, task.btn_action_key):
        public_tree_t4_decision_from_physical_result(result, task.state)
        for task, result in zip(tasks, results)
    }
    root_nodes = {
        spec.world_id: PublicTreeDecisionNode(
            world_states[spec.world_id],
            {
                btn_key: leaves[(spec.world_id, btn_key)]
                for btn_key in action_keys_by_world[spec.world_id]
            },
        )
        for spec in specs
    }
    root = PublicTreeChanceNode(
        PublicTreeChanceBranch(
            spec.world_id,
            Fraction(1, len(specs)),
            root_nodes[spec.world_id],
        )
        for spec in specs
    )

    shared_t4_pairs = 0
    for btn_key in action_keys_by_world[specs[0].world_id]:
        keys = {t4_states[(spec.world_id, btn_key)].infoset_key for spec in specs}
        if len(keys) != len(specs):
            raise AssertionError(
                "BB must retain its own hidden T3 recall at each BTN-root T4 branch"
            )

    commitments_by_world = {
        spec.world_id: {
            str(result["physical_state_commitment"])
            for task, result in zip(tasks, results)
            if task.world_id == spec.world_id
        }
        for spec in specs
    }
    if commitments_by_world["world-a"] == commitments_by_world["world-b"]:
        raise AssertionError("fixture hidden worlds must condition distinct physical ranges")

    manifest = _manifest(
        fixture_id=fixture_id,
        actor="btn",
        visible_joker_count=visible_joker_count,
        world_ids=[spec.world_id for spec in canonical_specs],
        root_digest=root_key.digest(),
        tasks=tasks,
        results=results,
    )
    metadata = {
        "method": "canonical_reduced_real_card_fixture",
        "fixture_id": fixture_id,
        "actor": "btn",
        "visible_joker_count": visible_joker_count,
        "position_contract_version": POSITION_CONTRACT_VERSION,
        "physical_world_count": len(specs),
        "root_legal_actions": len(root_nodes[specs[0].world_id].action_ids),
        "btn_decision_nodes": len(root_nodes),
        "physical_t4_leaf_positions": len(tasks),
        "shared_t4_infoset_pairs": shared_t4_pairs,
        "selection_performed_in_physical_world": False,
        "strategy_fusion": False,
        "rust_leaf_integrated": True,
        "full_card": False,
        "hu_exact": False,
    }
    return CompiledReducedFixture(
        fixture_id=fixture_id,
        actor="btn",
        visible_joker_count=visible_joker_count,
        root=root,
        physical_leaf_results=tuple(results),
        physical_leaf_states=tuple(task.state for task in tasks),
        root_infoset_digest=root_key.digest(),
        shared_t4_infoset_pairs=shared_t4_pairs,
        fixture_manifest_sha256=manifest,
        metadata=metadata,
    )


def compile_canonical_reduced_fixture(
    actor: str,
    visible_joker_count: int,
    *,
    rust_solver_path: str | Path | None = None,
    rust_timeout_s: float = 120.0,
    reverse_world_order: bool = False,
) -> CompiledReducedFixture:
    """Compile one of the six required M2 actor/Joker strata."""
    if actor == "bb":
        return compile_bb_reduced_fixture(
            visible_joker_count,
            rust_solver_path=rust_solver_path,
            rust_timeout_s=rust_timeout_s,
            reverse_world_order=reverse_world_order,
        )
    if actor == "btn":
        return compile_btn_reduced_fixture(
            visible_joker_count,
            rust_solver_path=rust_solver_path,
            rust_timeout_s=rust_timeout_s,
            reverse_world_order=reverse_world_order,
        )
    raise ValueError("actor must be 'bb' or 'btn'")
