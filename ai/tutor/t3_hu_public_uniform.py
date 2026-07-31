"""BTN-root T3 evaluator with a uniform public BB range at exact T4 leaves.

The BB T4 policy pass is intentionally separated from the physical value pass.
BB chooses from every exact T4 candidate using only its legal PlayerView.  The
chosen action is then valued in the sampled physical world, where BTN's hidden
discard identities are finally removed from the deck.  This prevents strategy
fusion through dead-card leakage while remaining an explicitly non-equilibrium
range approximation.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from ai.engine.action_space import Action, get_turn_actions
from ai.engine.encoding import ALL_CARDS, Board
from ai.engine.turn_order import (
    POSITION_CONTRACT_VERSION,
    normalize_position,
    validate_decision_board_counts,
)
from ai.tutor.exact_late import (
    CardNormalizer,
    action_key,
    action_to_dict,
    apply_action,
    board_card_count,
    board_to_dict,
    evaluate_late_positions_rust_batch,
    evaluate_public_cfr_bb_t4_leaves_rust_batch,
    normalize_board,
)


METHOD = "btn_t3_uniform_public_range_exact_t4"
RANGE_MODEL = "uniform_hidden_discards_no_history_v1"


@dataclass(frozen=True)
class _PhysicalScenario:
    bb_private_discards: tuple[str, str, str]
    bb_t4_draw: tuple[str, str, str]


@dataclass(frozen=True)
class _LeafTask:
    root_index: int
    scenario_index: int


def _cards(raw: Any, *, label: str, normalizer: CardNormalizer) -> list[str]:
    value = raw or []
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} must be a card list")
    return normalizer.cards(value)


def _reject_private_or_ambiguous_fields(payload: dict[str, Any]) -> None:
    for field in ("exclude", "candidate_actions", "candidates", "top_n"):
        if field in payload:
            raise ValueError(f"BTN public-range T3 input forbids ambiguous field {field!r}")

    def walk(value: Any, path: str = "") -> None:
        if isinstance(value, dict):
            for raw_key, child in value.items():
                key = str(raw_key).strip().lower().replace("-", "_")
                child_path = f"{path}.{raw_key}" if path else str(raw_key)
                if not path and key == "known_discards_self":
                    continue
                if "discard" in key:
                    raise ValueError(
                        "opponent-private or ambiguous discard field is forbidden: "
                        f"{child_path}"
                    )
                walk(child, child_path)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")

    walk(payload)


def _one_alias(payload: dict[str, Any], names: Sequence[str], label: str) -> Any:
    present = [name for name in names if name in payload and payload[name] is not None]
    if not present:
        raise ValueError(f"BTN public-range T3 input is missing {label}")
    if len(present) > 1:
        raise ValueError(f"BTN public-range T3 input has ambiguous {label} aliases: {present}")
    return payload[present[0]]


def _normalize_input(payload: dict[str, Any]) -> tuple[Board, Board, list[str], list[str], list[str]]:
    if not isinstance(payload, dict):
        raise ValueError("BTN public-range T3 input must be an object")
    _reject_private_or_ambiguous_fields(payload)
    if "turn" not in payload or int(payload["turn"]) != 3:
        raise ValueError("BTN public-range evaluator requires turn=3")
    supplied_contract = payload.get("position_contract_version")
    if supplied_contract != POSITION_CONTRACT_VERSION:
        raise ValueError(
            f"BTN public-range evaluator requires explicit position contract "
            f"{supplied_contract!r}; "
            f"expected {POSITION_CONTRACT_VERSION!r}"
        )

    positions: list[str] = []
    for field in ("actor", "position", "player_position"):
        if field in payload and payload[field] not in (None, ""):
            positions.append(normalize_position(payload[field]))
    if "is_btn" in payload and payload["is_btn"] is not None:
        positions.append(normalize_position(None, is_btn=payload["is_btn"]))
    if not positions:
        raise ValueError("BTN public-range evaluator requires an explicit actor/position")
    if len(set(positions)) != 1:
        raise ValueError(f"contradictory BTN public-range position fields: {positions}")
    if positions[0] != "btn":
        raise ValueError("BTN public-range T3 evaluator only accepts actor='btn'")
    if "first_actor" not in payload:
        raise ValueError("BTN public-range T3 evaluator requires explicit first_actor='bb'")
    if normalize_position(payload["first_actor"]) != "bb":
        raise ValueError("BTN public-range T3 evaluator requires first_actor='bb'")

    normalizer = CardNormalizer()
    raw_board = _one_alias(payload, ("board_self", "board"), "BTN board")
    raw_opponent = _one_alias(payload, ("board_opponent", "opponent_board"), "BB board")
    raw_dealt = _one_alias(payload, ("dealt_cards", "dealt"), "BTN dealt cards")
    raw_known = _one_alias(payload, ("known_discards_self",), "BTN known discards")
    if not isinstance(raw_board, dict) or not isinstance(raw_opponent, dict):
        raise ValueError("BTN and BB boards must be row objects")
    board = normalize_board(raw_board, normalizer)
    opponent = normalize_board(raw_opponent, normalizer)
    dealt = _cards(raw_dealt, label="BTN dealt cards", normalizer=normalizer)
    known_self = _cards(raw_known, label="known_discards_self", normalizer=normalizer)
    public_exclude = _cards(
        payload.get("public_exclude") or [],
        label="public_exclude",
        normalizer=normalizer,
    )

    for label, candidate_board in (("BTN", board), ("BB", opponent)):
        lengths = (len(candidate_board.top), len(candidate_board.middle), len(candidate_board.bottom))
        if any(actual > limit for actual, limit in zip(lengths, (3, 5, 5))):
            raise ValueError(f"{label} board exceeds row capacity: {lengths}")
    validate_decision_board_counts(3, "btn", board_card_count(board), board_card_count(opponent))
    if len(dealt) != 3:
        raise ValueError(f"BTN T3 requires exactly 3 dealt cards, got {len(dealt)}")
    if len(known_self) != 2:
        raise ValueError(f"BTN T3 requires exactly 2 prior BTN discards, got {len(known_self)}")

    valid_cards = set(ALL_CARDS)
    seen: dict[str, str] = {}
    zones: list[tuple[str, Iterable[str]]] = [
        ("board_self", board.all_cards()),
        ("board_opponent", opponent.all_cards()),
        ("dealt_cards", dealt),
        ("known_discards_self", known_self),
        ("public_exclude", public_exclude),
    ]
    for zone, zone_cards in zones:
        for card in zone_cards:
            if card not in valid_cards:
                raise ValueError(f"unknown physical card {card!r} in {zone}")
            if card in seen:
                raise ValueError(f"duplicate physical card {card!r} in {seen[card]} and {zone}")
            seen[card] = zone
    if len(valid_cards - set(seen)) < 6:
        raise ValueError("BTN T3 public range leaves fewer than 6 cards for the BB scenario")
    return board, opponent, dealt, known_self, public_exclude


def _decision_id(
    board: Board,
    opponent: Board,
    dealt: Sequence[str],
    known_self: Sequence[str],
    public_exclude: Sequence[str],
) -> str:
    canonical = {
        "turn": 3,
        "actor": "btn",
        "board": {key: sorted(value) for key, value in board_to_dict(board).items()},
        "opponent_board": {key: sorted(value) for key, value in board_to_dict(opponent).items()},
        "dealt": sorted(dealt),
        "known_discards_self": sorted(known_self),
        "public_exclude": sorted(public_exclude),
        "range_model": RANGE_MODEL,
    }
    raw = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _sample_cards(
    live: Sequence[str],
    count: int,
    *,
    seed: int,
    decision_id: str,
    scenario_index: int,
) -> tuple[str, ...]:
    ordered = sorted(
        live,
        key=lambda card: hashlib.sha256(
            f"{int(seed)}|{decision_id}|{scenario_index}|bb_private_and_t4|{card}".encode("utf-8")
        ).digest(),
    )
    return tuple(ordered[:count])


def _normalize_scenarios(
    overrides: Sequence[dict[str, Any]] | None,
    *,
    live: Sequence[str],
    outer_samples: int,
    seed: int,
    decision_id: str,
) -> tuple[list[_PhysicalScenario], str]:
    live_set = set(live)
    if overrides is None:
        scenarios: list[_PhysicalScenario] = []
        for scenario_index in range(max(1, int(outer_samples))):
            sampled = _sample_cards(
                live,
                6,
                seed=seed,
                decision_id=decision_id,
                scenario_index=scenario_index,
            )
            scenarios.append(
                _PhysicalScenario(
                    bb_private_discards=tuple(sampled[:3]),  # type: ignore[arg-type]
                    bb_t4_draw=tuple(sampled[3:]),  # type: ignore[arg-type]
                )
            )
        return scenarios, "deterministic_priority"

    if not overrides:
        raise ValueError("scenario_overrides must contain at least one scenario")
    scenarios = []
    for index, raw in enumerate(overrides):
        if not isinstance(raw, dict):
            raise ValueError(f"scenario override {index} must be an object")
        if set(raw) != {"bb_private_discards", "bb_t4_draw"}:
            raise ValueError(
                f"scenario override {index} requires only bb_private_discards and bb_t4_draw"
            )
        private = tuple(str(card) for card in raw["bb_private_discards"])
        draw = tuple(str(card) for card in raw["bb_t4_draw"])
        if len(private) != 3 or len(draw) != 3:
            raise ValueError(f"scenario override {index} requires 3 BB discards and 3 draw cards")
        combined = (*private, *draw)
        if len(set(combined)) != 6 or any(card not in live_set for card in combined):
            raise ValueError(f"scenario override {index} contains duplicate or unavailable cards")
        scenarios.append(
            _PhysicalScenario(
                bb_private_discards=private,  # type: ignore[arg-type]
                bb_t4_draw=draw,  # type: ignore[arg-type]
            )
        )
    return scenarios, "provided_physical_scenarios"


def _policy_action_key(leaf: dict[str, Any]) -> str:
    keys = list(leaf.get("action_keys") or [])
    metrics_by_key = leaf.get("metrics_by_action_key") or {}
    if not keys or set(keys) != set(metrics_by_key):
        raise RuntimeError("public BB T4 policy leaf has incomplete candidate metrics")

    def rank(key: str) -> tuple[float, float, float, float, float, str]:
        metrics = metrics_by_key[key]
        return (
            -float(metrics.get("score", metrics.get("ev", 0.0)) or 0.0),
            float(metrics.get("bust_rate", 0.0) or 0.0),
            -float(metrics.get("fl_rate", 0.0) or 0.0),
            -float(metrics.get("raw_score", 0.0) or 0.0),
            -float(metrics.get("royalty", 0.0) or 0.0),
            key,
        )

    return min(keys, key=rank)


def _summary_stats(values: Sequence[float]) -> dict[str, Any]:
    n = len(values)
    if n == 0:
        raise ValueError("cannot summarize zero T3 scenarios")
    mean = sum(values) / n
    variance = sum((value - mean) ** 2 for value in values) / (n - 1) if n > 1 else 0.0
    standard_error = math.sqrt(variance / n) if n > 1 else 0.0
    half_width = 1.96 * standard_error
    return {
        "score": float(mean),
        "ev": float(mean),
        "samples": n,
        "variance": float(variance),
        "standard_error": float(standard_error),
        "ci95": [float(mean - half_width), float(mean + half_width)],
    }


def evaluate_btn_t3_public_uniform(
    payload: dict[str, Any],
    *,
    seed: int = 20260712,
    outer_samples: int = 8,
    scenario_overrides: Sequence[dict[str, Any]] | None = None,
    rust_solver_path: str | Path | None = None,
    rust_timeout_s: float = 300.0,
    include_scenario_values: bool = False,
    allow_synthetic_public_exclude: bool = False,
) -> dict[str, Any]:
    """Evaluate every BTN T3 action without leaking BTN discards to BB policy."""
    started = time.perf_counter()
    if isinstance(outer_samples, bool) or int(outer_samples) <= 0:
        raise ValueError("outer_samples must be a positive integer")
    board, opponent, dealt, known_self, public_exclude = _normalize_input(payload)
    if public_exclude and not allow_synthetic_public_exclude:
        raise ValueError(
            "public_exclude is disabled for production PlayerView inputs; "
            "enable it only for an explicit synthetic public deck variant"
        )
    root_actions = get_turn_actions(dealt, board)
    if not root_actions:
        raise ValueError("BTN T3 public-range position has no legal root actions")
    root_keys = [action_key(action) for action in root_actions]
    decision_id = _decision_id(board, opponent, dealt, known_self, public_exclude)
    public_known = {
        *board.all_cards(),
        *opponent.all_cards(),
        *dealt,
        *known_self,
        *public_exclude,
    }
    live = tuple(card for card in ALL_CARDS if card not in public_known)
    scenarios, chance_source = _normalize_scenarios(
        scenario_overrides,
        live=live,
        outer_samples=outer_samples,
        seed=seed,
        decision_id=decision_id,
    )

    policy_payloads: list[dict[str, Any]] = []
    tasks: list[_LeafTask] = []
    btn11_by_root: list[Board] = []
    for root_index, root_action in enumerate(root_actions):
        btn11 = apply_action(board, root_action)
        btn11_by_root.append(btn11)
        for scenario_index, scenario in enumerate(scenarios):
            policy_payloads.append(
                {
                    "turn": 4,
                    "actor": "bb",
                    "is_btn": False,
                    "first_actor": "bb",
                    "position_contract_version": POSITION_CONTRACT_VERSION,
                    "board_self": board_to_dict(opponent),
                    "board_opponent": board_to_dict(btn11),
                    "dealt_cards": list(scenario.bb_t4_draw),
                    "known_discards_self": list(scenario.bb_private_discards),
                    "public_exclude": list(public_exclude),
                }
            )
            tasks.append(_LeafTask(root_index, scenario_index))

    policy_started = time.perf_counter()
    policy_leaves = evaluate_public_cfr_bb_t4_leaves_rust_batch(
        policy_payloads,
        rust_solver_path=rust_solver_path,
        timeout_s=rust_timeout_s,
        position_parallel=True,
        allow_synthetic_public_exclude=allow_synthetic_public_exclude,
    )
    policy_wall_ms = (time.perf_counter() - policy_started) * 1000.0
    if len(policy_leaves) != len(tasks):
        raise RuntimeError(f"BB public policy returned {len(policy_leaves)}/{len(tasks)} leaves")

    physical_payloads: list[dict[str, Any]] = []
    selected_policy_keys: list[str] = []
    for task, policy_leaf in zip(tasks, policy_leaves):
        if policy_leaf.get("selection_performed") is not False:
            raise RuntimeError("BB public policy adapter unexpectedly selected an action")
        selected_key = _policy_action_key(policy_leaf)
        selected_action = (policy_leaf.get("actions_by_action_key") or {}).get(selected_key)
        if selected_action is None:
            raise RuntimeError(f"BB public policy leaf is missing selected action {selected_key}")
        selected_policy_keys.append(selected_key)
        root_action = root_actions[task.root_index]
        scenario = scenarios[task.scenario_index]
        physical_payloads.append(
            {
                "turn": 4,
                "board": board_to_dict(opponent),
                "opponent_board": board_to_dict(btn11_by_root[task.root_index]),
                "dealt": list(scenario.bb_t4_draw),
                "exclude": [
                    *scenario.bb_private_discards,
                    *known_self,
                    root_action.discard,
                    *public_exclude,
                ],
                "candidate_actions": [selected_action],
            }
        )

    physical_started = time.perf_counter()
    physical_leaves = evaluate_late_positions_rust_batch(
        physical_payloads,
        top_n=1,
        rust_solver_path=rust_solver_path,
        timeout_s=rust_timeout_s,
        position_parallel=True,
    )
    physical_wall_ms = (time.perf_counter() - physical_started) * 1000.0
    if len(physical_leaves) != len(tasks):
        raise RuntimeError(f"physical BB value pass returned {len(physical_leaves)}/{len(tasks)} leaves")

    scenario_values: dict[int, list[float]] = {index: [] for index in range(len(root_actions))}
    scenario_policy_keys: dict[int, list[str]] = {index: [] for index in range(len(root_actions))}
    response_draws_enumerated = 0
    for task, selected_key, physical_leaf in zip(tasks, selected_policy_keys, physical_leaves):
        candidates = list(physical_leaf.get("candidates") or [])
        if int(physical_leaf.get("evaluated_actions") or 0) != 1 or len(candidates) != 1:
            raise RuntimeError("physical BB value pass must evaluate exactly one policy-selected action")
        candidate = candidates[0]
        if action_key(candidate.get("action") or {}) != selected_key:
            raise RuntimeError("physical BB value pass evaluated a different action from BB policy")
        metrics = candidate.get("metrics") or {}
        if metrics.get("source") != "exact_hu_response" or metrics.get("opponent_response") is not True:
            raise RuntimeError("physical BB value pass is not an exact T4 opponent-response leaf")
        scenario_values[task.root_index].append(-float(metrics["score"]))
        scenario_policy_keys[task.root_index].append(selected_key)
        response_draws_enumerated += int(metrics.get("samples") or 0)

    candidates: list[dict[str, Any]] = []
    for root_index, root_action in enumerate(root_actions):
        metrics = {
            **_summary_stats(scenario_values[root_index]),
            "source": METHOD,
            "range_model": RANGE_MODEL,
            "strategy_fusion": False,
            "equilibrium_approx": False,
            "hu_exact": False,
            "inner_t4_exact": True,
            "seed": int(seed),
        }
        candidate = {
            "action": action_to_dict(root_action),
            "action_key": root_keys[root_index],
            "board": board_to_dict(btn11_by_root[root_index]),
            "metrics": metrics,
            "scenario_policy_action_keys": list(scenario_policy_keys[root_index]),
        }
        if include_scenario_values:
            candidate["scenario_values"] = list(scenario_values[root_index])
        candidates.append(candidate)
    candidates.sort(key=lambda candidate: (-float(candidate["metrics"]["score"]), candidate["action_key"]))
    best = candidates[0]

    return {
        "schema": "ofc_hu_gate/v1",
        "turn": 3,
        "actor": "btn",
        "position": "btn",
        "is_btn": True,
        "first_actor": "bb",
        "position_contract_version": POSITION_CONTRACT_VERSION,
        "method": METHOD,
        "range_model": RANGE_MODEL,
        "candidate_scope": "all_legal",
        "strategy_fusion": False,
        "equilibrium_approx": False,
        "hu_exact": False,
        "inner_t4_exact": True,
        "bb_policy_information": "public_boards_plus_bb_private_discards_and_draw",
        "btn_hidden_discards_in_policy_payload": False,
        "physical_hidden_discards_applied_after_policy": True,
        "synthetic_public_exclude_used": bool(public_exclude),
        "decision_id": decision_id,
        "seed": int(seed),
        "outer_samples": len(scenarios),
        "chance_source": chance_source,
        "chance_tape_shared_across_root_actions": True,
        "legal_actions": len(root_actions),
        "evaluated_actions": len(candidates),
        "policy_leaf_positions": len(policy_leaves),
        "physical_leaf_positions": len(physical_leaves),
        "response_draws_enumerated": response_draws_enumerated,
        "chosen_action": best["action"],
        "best": best,
        "candidates": candidates,
        "policy_wall_ms": policy_wall_ms,
        "physical_wall_ms": physical_wall_ms,
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
    }
