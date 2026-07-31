"""Fail-closed public-root mixture compiler for T3/T4 HU search.

The multi-root MCCFR solver consumes a chance distribution over actor-private
information sets.  This module supplies the first repository-owned compiler
for that boundary.  A :class:`PublicRootContext` contains public state and
content bindings only.  Private cards enter only as a *declared ex-ante
hypothetical support*, never as an ``actual_actor_private_cards`` argument.

This first slice intentionally accepts a deterministic finite support rather
than claiming full-deck private-type enumeration.  Its exact ``Fraction``
prior is exact only conditional on that declared support.  Every result is
therefore non-promoting and not production-sampling-ready.  The audit manifest
states those limits explicitly and serializes commitments, not raw private
support payloads.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import ai.tutor.t3_hu_full_card_range as _full_card_range_module
import ai.tutor.t3_hu_multi_root_mccfr as _multi_root_module
from ai.engine.encoding import ALL_CARDS
from ai.engine.turn_order import POSITION_CONTRACT_VERSION
from ai.tutor.t3_hu_full_card_mccfr import FullCardGenerativeAdapter
from ai.tutor.t3_hu_full_card_range import FullCardRange, verify_full_card_range
from ai.tutor.t3_hu_multi_root_mccfr import MultiRootChanceEntry
from ai.tutor.t3_hu_public_cfr import (
    CardRows,
    InfoSetKey,
    PHASE_SPECS,
    PrivateRecall,
    PublicHistoryEntry,
    ROWS,
)


PUBLIC_ROOT_MIXTURE_SCHEMA = "ofc_t3_t4_public_root_mixture/v1"
PUBLIC_ROOT_CONTEXT_SCHEMA = "ofc_t3_t4_public_root_context/v1"
PUBLIC_ROOT_BINDINGS_SCHEMA = "ofc_t3_t4_public_root_bindings/v1"
EXPLICIT_SUPPORT_SCOPE = "declared_finite_hypothetical_private_type_support_v1"
MIXTURE_EXACTNESS = "exact_fraction_prior_on_declared_finite_support_only"
EXPLICIT_SUPPORT_AUTHORIZATION_CLASS = (
    "algorithm_only_explicit_finite_support"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_VALID_CARDS = frozenset(ALL_CARDS)
_PHYSICAL_JOKERS = ("X1", "X2")
_CANONICAL_VERIFY_FULL_CARD_RANGE = verify_full_card_range
_CANONICAL_MULTI_ROOT_SOURCE_BINDING_ATTESTOR = (
    _multi_root_module._multi_root_source_binding
)
_CANONICAL_ADAPTER_CHECKPOINT_BINDING_ATTESTOR = (
    _multi_root_module._adapter_checkpoint_binding
)

_FORBIDDEN_CONTEXT_FIELDS = frozenset(
    {
        "actual_actor_private_cards",
        "actual_private_cards",
        "actor_private_cards",
        "current_draw",
        "dealt_cards",
        "determinization_id",
        "live_cards",
        "opponent_discards",
        "opponent_private",
        "own_recall",
        "particle_id",
        "remaining_deck",
        "rng_seed",
        "seed",
        "undealt_cards",
        "world_id",
    }
)
_PUBLIC_CONTEXT_INPUT_FIELDS = frozenset(
    {
        "contract_version",
        "actor",
        "turn",
        "phase",
        "board_bb",
        "board_btn",
        "public_action_history",
        "fantasy_state",
    }
)

_SOURCE_PATHS = MappingProxyType(
    {
        "public_root_mixture_compiler": Path(__file__),
        "full_card_range": Path(__file__).with_name("t3_hu_full_card_range.py"),
        "full_card_mccfr": Path(__file__).with_name("t3_hu_full_card_mccfr.py"),
        "multi_root_mccfr": Path(__file__).with_name("t3_hu_multi_root_mccfr.py"),
        "public_infoset": Path(__file__).with_name("t3_hu_public_cfr.py"),
        "public_tree": Path(__file__).with_name("t3_hu_public_tree.py"),
        "terminal_scoring": Path(__file__).with_name("exact_late.py"),
        "action_space": Path(__file__).parents[1] / "engine" / "action_space.py",
        "deck_encoding": Path(__file__).parents[1] / "engine" / "encoding.py",
        "game_engine": Path(__file__).parents[1] / "engine" / "game_engine.py",
        "scoring": Path(__file__).parents[1] / "engine" / "scoring.py",
        "turn_order": Path(__file__).parents[1] / "engine" / "turn_order.py",
        "rollout_evaluator": (
            Path(__file__).parents[1] / "mcts" / "rollout_evaluator.py"
        ),
        "fantasyland_ev": Path(__file__).parents[1] / "config" / "fl_ev.json",
    }
)

PRODUCTION_SAMPLING_REMAINING_REQUIREMENTS = (
    "enumerate_or_sample_the_complete_publicly_compatible_actor_private_type_space",
    "bind_audited_type_generation_probabilities_and_coverage_without_actual_hand_conditioning",
    "replace_finite_declared_support_with_a_scalable_unbiased_super_root_sampler",
    "require_promoted_calibrated_behavior_posteriors_for_every_conditional_range",
    "run_independent_online_solver_strength_and_uncertainty_gates_on_unseen_public_roots",
)


class PublicRootMixtureError(ValueError):
    """The public mixture or one of its content bindings failed validation."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    try:
        raw = path.resolve(strict=True).read_bytes()
    except OSError as exc:  # pragma: no cover - deployment packaging defense
        raise PublicRootMixtureError(f"required source file is unavailable: {path}") from exc
    return hashlib.sha256(raw).hexdigest()


def _live_source_sha256s() -> dict[str, str]:
    return {
        name: _file_sha256(path)
        for name, path in sorted(_SOURCE_PATHS.items())
    }


def _rules_manifest() -> dict[str, Any]:
    return {
        "schema": "ofc_t3_t4_joker_hu_rules/v1",
        "position_contract_version": POSITION_CONTRACT_VERSION,
        "deck": list(ALL_CARDS),
        "deck_size": len(ALL_CARDS),
        "physical_joker_ids": list(_PHYSICAL_JOKERS),
        "joker_physical_identity_collapsed": False,
        "fantasy_state_contract": "standard_normal_hand_none_only",
        "row_names": list(ROWS),
        "row_capacities": [3, 5, 5],
        "phase_specs": {
            phase: {
                "actor": spec["actor"],
                "turn": 3 if phase.startswith("t3") else 4,
                "board_counts": list(spec["board_counts"]),
                "last_action": list(spec["last_action"]),
            }
            for phase, spec in sorted(PHASE_SPECS.items())
        },
    }


def _rules_sha256() -> str:
    return _canonical_sha256(_rules_manifest())


def _require_live_attestor_identities() -> None:
    if (
        _multi_root_module._multi_root_source_binding
        is not _CANONICAL_MULTI_ROOT_SOURCE_BINDING_ATTESTOR
    ):
        raise PublicRootMixtureError(
            "canonical multi-root source-binding attestor was overridden"
        )
    if (
        _multi_root_module._adapter_checkpoint_binding
        is not _CANONICAL_ADAPTER_CHECKPOINT_BINDING_ATTESTOR
    ):
        raise PublicRootMixtureError(
            "canonical FullCard adapter-binding attestor was overridden"
        )


def _independently_verify_self_hash(
    value: Any,
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PublicRootMixtureError(f"{label} must be a mapping")
    snapshot = json.loads(_canonical_json(dict(value)))
    declared = snapshot.get("binding_sha256")
    _require_sha256(declared, label=f"{label}.binding_sha256")
    content = dict(snapshot)
    del content["binding_sha256"]
    if _canonical_sha256(content) != declared:
        raise PublicRootMixtureError(
            f"{label} binding SHA256 does not match canonical content"
        )
    return snapshot


def _canonical_solver_source_binding() -> dict[str, Any]:
    """Fresh canonical solver/runtime binding, including native rule assets."""

    _require_live_attestor_identities()
    snapshot = _independently_verify_self_hash(
        _CANONICAL_MULTI_ROOT_SOURCE_BINDING_ATTESTOR(),
        label="canonical solver source binding",
    )
    runtime_binding = snapshot.get("live_runtime_semantic_binding")
    _independently_verify_self_hash(
        runtime_binding,
        label="canonical solver live runtime semantic binding",
    )
    return snapshot


def _require_live_range_verifier() -> None:
    _require_live_attestor_identities()
    if (
        verify_full_card_range is not _CANONICAL_VERIFY_FULL_CARD_RANGE
        or _full_card_range_module.verify_full_card_range
        is not _CANONICAL_VERIFY_FULL_CARD_RANGE
    ):
        raise PublicRootMixtureError(
            "canonical verify_full_card_range binding was overridden"
        )


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise PublicRootMixtureError(f"{label} must be a lowercase SHA256 digest")
    return value


def _canonical_cards(cards: Sequence[str], *, label: str) -> tuple[str, ...]:
    if isinstance(cards, (str, bytes)):
        raise TypeError(f"{label} must be a card sequence")
    result = tuple(sorted(str(card) for card in cards))
    invalid = sorted(card for card in result if card not in _VALID_CARDS)
    if invalid:
        raise PublicRootMixtureError(f"{label} contains invalid physical cards: {invalid}")
    if len(result) != len(set(result)):
        raise PublicRootMixtureError(f"{label} contains duplicate physical cards")
    return result


def _canonical_board(board: Sequence[Sequence[str]], *, label: str) -> CardRows:
    if isinstance(board, (str, bytes)) or len(board) != 3:
        raise PublicRootMixtureError(f"{label} must contain top/middle/bottom rows")
    result = tuple(
        _canonical_cards(cards, label=f"{label}.{row}")
        for row, cards in zip(ROWS, board)
    )
    return result  # type: ignore[return-value]


def _canonical_history(
    history: Sequence[PublicHistoryEntry],
) -> tuple[PublicHistoryEntry, ...]:
    if isinstance(history, (str, bytes)):
        raise TypeError("public_action_history must be a sequence")
    result: list[PublicHistoryEntry] = []
    prior_order: tuple[int, int] | None = None
    seen_public: set[str] = set()
    for raw_turn, raw_actor, raw_placements in history:
        turn = int(raw_turn)
        actor = str(raw_actor)
        if actor not in ("bb", "btn"):
            raise PublicRootMixtureError("public history actor must be bb or btn")
        order = (turn, 0 if actor == "bb" else 1)
        if prior_order is not None and order <= prior_order:
            raise PublicRootMixtureError(
                "public history must be strict turn/BB-before-BTN order"
            )
        prior_order = order
        placements: list[tuple[str, str]] = []
        local_cards: set[str] = set()
        for raw_card, raw_row in raw_placements:
            card = str(raw_card)
            row = str(raw_row)
            if card not in _VALID_CARDS:
                raise PublicRootMixtureError(
                    f"public history contains invalid physical card {card!r}"
                )
            if row not in ROWS:
                raise PublicRootMixtureError(f"invalid public row {row!r}")
            if card in local_cards or card in seen_public:
                raise PublicRootMixtureError(
                    f"public history places physical card {card!r} more than once"
                )
            local_cards.add(card)
            seen_public.add(card)
            placements.append((card, row))
        result.append(
            (turn, actor, tuple(sorted(placements, key=lambda item: (item[1], item[0]))))
        )
    return tuple(result)


def _public_history_dict(history: Sequence[PublicHistoryEntry]) -> list[dict[str, Any]]:
    return [
        {
            "turn": turn,
            "actor": actor,
            "placements": [[card, row] for card, row in placements],
        }
        for turn, actor, placements in history
    ]


def _recall_dict(recall: PrivateRecall) -> dict[str, Any]:
    return recall.to_canonical_dict()


@dataclass(frozen=True, slots=True)
class PublicRootBindings:
    """Live source/rule identity plus the expected frozen behavior identity."""

    behavior_model_id: str
    behavior_model_sha256: str
    rules_sha256: str
    canonical_solver_source_binding_sha256: str
    canonical_solver_runtime_semantic_binding_sha256: str
    source_sha256s: Mapping[str, str] = field(repr=False)

    def __post_init__(self) -> None:
        model_id = str(self.behavior_model_id)
        if not model_id or model_id != model_id.strip():
            raise PublicRootMixtureError("behavior_model_id must be non-empty and trimmed")
        object.__setattr__(self, "behavior_model_id", model_id)
        _require_sha256(self.behavior_model_sha256, label="behavior_model_sha256")
        _require_sha256(self.rules_sha256, label="rules_sha256")
        _require_sha256(
            self.canonical_solver_source_binding_sha256,
            label="canonical_solver_source_binding_sha256",
        )
        _require_sha256(
            self.canonical_solver_runtime_semantic_binding_sha256,
            label="canonical_solver_runtime_semantic_binding_sha256",
        )
        if not isinstance(self.source_sha256s, Mapping):
            raise TypeError("source_sha256s must be a mapping")
        expected_names = set(_SOURCE_PATHS)
        if set(self.source_sha256s) != expected_names:
            raise PublicRootMixtureError(
                "source hash set mismatch: "
                f"missing={sorted(expected_names - set(self.source_sha256s))}, "
                f"extra={sorted(set(self.source_sha256s) - expected_names)}"
            )
        frozen: dict[str, str] = {}
        for name in sorted(expected_names):
            frozen[name] = _require_sha256(
                self.source_sha256s[name], label=f"source_sha256s[{name!r}]"
            )
        object.__setattr__(self, "source_sha256s", MappingProxyType(frozen))

    @classmethod
    def capture(
        cls,
        *,
        behavior_model_id: str,
        behavior_model_sha256: str,
    ) -> "PublicRootBindings":
        """Capture the live repository boundary used by a later compilation."""

        _require_live_range_verifier()
        solver_binding = _canonical_solver_source_binding()
        runtime_binding = solver_binding.get("live_runtime_semantic_binding")
        if not isinstance(runtime_binding, Mapping):
            raise PublicRootMixtureError(
                "canonical solver binding has no live runtime semantic binding"
            )
        runtime_sha256 = runtime_binding.get("binding_sha256")
        _require_sha256(
            runtime_sha256,
            label="canonical solver runtime semantic binding SHA256",
        )
        return cls(
            behavior_model_id=behavior_model_id,
            behavior_model_sha256=behavior_model_sha256,
            rules_sha256=_rules_sha256(),
            canonical_solver_source_binding_sha256=solver_binding[
                "binding_sha256"
            ],
            canonical_solver_runtime_semantic_binding_sha256=runtime_sha256,
            source_sha256s=_live_source_sha256s(),
        )

    def to_canonical_dict(self) -> dict[str, Any]:
        source_rows = dict(self.source_sha256s)
        return {
            "schema": PUBLIC_ROOT_BINDINGS_SCHEMA,
            "behavior_model_id": self.behavior_model_id,
            "behavior_model_sha256": self.behavior_model_sha256,
            "rules_sha256": self.rules_sha256,
            "canonical_solver_source_binding_sha256": (
                self.canonical_solver_source_binding_sha256
            ),
            "canonical_solver_runtime_semantic_binding_sha256": (
                self.canonical_solver_runtime_semantic_binding_sha256
            ),
            "source_sha256s": source_rows,
            "source_set_sha256": _canonical_sha256(source_rows),
        }

    def verify_live(self) -> None:
        _require_live_range_verifier()
        if self.rules_sha256 != _rules_sha256():
            raise PublicRootMixtureError("rules hash does not match live canonical rules")
        live = _live_source_sha256s()
        if dict(self.source_sha256s) != live:
            changed = sorted(
                name
                for name in set(live) | set(self.source_sha256s)
                if live.get(name) != self.source_sha256s.get(name)
            )
            raise PublicRootMixtureError(
                f"source hash binding does not match live files: {changed}"
            )
        solver_binding = _canonical_solver_source_binding()
        if (
            solver_binding["binding_sha256"]
            != self.canonical_solver_source_binding_sha256
        ):
            raise PublicRootMixtureError(
                "canonical multi-root solver source/runtime binding drifted"
            )
        runtime_binding = solver_binding.get("live_runtime_semantic_binding")
        runtime_sha256 = (
            runtime_binding.get("binding_sha256")
            if isinstance(runtime_binding, Mapping)
            else None
        )
        if runtime_sha256 != self.canonical_solver_runtime_semantic_binding_sha256:
            raise PublicRootMixtureError(
                "canonical multi-root solver runtime semantic binding drifted"
            )


@dataclass(frozen=True, slots=True)
class PublicRootContext:
    """Public state and immutable content bindings; no actor-private fields."""

    contract_version: str
    actor: str
    turn: int
    phase: str
    board_bb: CardRows
    board_btn: CardRows
    public_action_history: tuple[PublicHistoryEntry, ...]
    fantasy_state: str | None
    bindings: PublicRootBindings

    def __post_init__(self) -> None:
        if self.contract_version != POSITION_CONTRACT_VERSION:
            raise PublicRootMixtureError("public root context requires bb_first_v1")
        phase = str(self.phase)
        if phase not in PHASE_SPECS:
            raise PublicRootMixtureError(f"unsupported public root phase {phase!r}")
        spec = PHASE_SPECS[phase]
        actor = str(self.actor)
        turn = int(self.turn)
        expected_turn = 3 if phase.startswith("t3") else 4
        if actor != spec["actor"] or turn != expected_turn:
            raise PublicRootMixtureError(
                f"phase {phase!r} requires actor={spec['actor']!r}, turn={expected_turn}"
            )
        if not isinstance(self.bindings, PublicRootBindings):
            raise TypeError("bindings must be PublicRootBindings")
        board_bb = _canonical_board(self.board_bb, label="board_bb")
        board_btn = _canonical_board(self.board_btn, label="board_btn")
        history = _canonical_history(self.public_action_history)
        object.__setattr__(self, "actor", actor)
        object.__setattr__(self, "turn", turn)
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "board_bb", board_bb)
        object.__setattr__(self, "board_btn", board_btn)
        object.__setattr__(self, "public_action_history", history)
        if self.fantasy_state is not None:
            raise PublicRootMixtureError(
                "standard T3/T4 public-root slice requires fantasy_state=None"
            )
        self._validate_public_cutoff()

    @classmethod
    def from_public_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        bindings: PublicRootBindings,
    ) -> "PublicRootContext":
        """Parse an untrusted public payload and reject private/leakage keys."""

        if not isinstance(payload, Mapping):
            raise TypeError("public root payload must be a mapping")
        raw_keys = {str(key) for key in payload}
        lowered = {key.lower() for key in raw_keys}
        leaked = sorted(lowered & _FORBIDDEN_CONTEXT_FIELDS)
        if leaked:
            raise PublicRootMixtureError(
                f"public root context leaked forbidden private fields: {leaked}"
            )
        extra = raw_keys - _PUBLIC_CONTEXT_INPUT_FIELDS
        missing = _PUBLIC_CONTEXT_INPUT_FIELDS - raw_keys - {"fantasy_state"}
        if extra or missing:
            raise PublicRootMixtureError(
                f"public root context schema mismatch: missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )
        return cls(
            contract_version=payload["contract_version"],
            actor=payload["actor"],
            turn=payload["turn"],
            phase=payload["phase"],
            board_bb=payload["board_bb"],
            board_btn=payload["board_btn"],
            public_action_history=payload["public_action_history"],
            fantasy_state=payload.get("fantasy_state"),
            bindings=bindings,
        )

    def _validate_public_cutoff(self) -> None:
        spec = PHASE_SPECS[self.phase]
        actual_counts = (
            sum(len(row) for row in self.board_bb),
            sum(len(row) for row in self.board_btn),
        )
        expected_counts = tuple(spec["board_counts"])
        if actual_counts != expected_counts:
            raise PublicRootMixtureError(
                f"phase {self.phase!r} requires BB/BTN board counts "
                f"{expected_counts}, got {actual_counts}"
            )
        for label, board in (("bb", self.board_bb), ("btn", self.board_btn)):
            if any(len(cards) > capacity for cards, capacity in zip(board, (3, 5, 5))):
                raise PublicRootMixtureError(f"{label} board exceeds row capacity")

        last_turn, last_actor = spec["last_action"]
        expected_pairs: list[tuple[int, str]] = []
        for turn in range(int(last_turn) + 1):
            expected_pairs.append((turn, "bb"))
            if turn < int(last_turn) or last_actor == "btn":
                expected_pairs.append((turn, "btn"))
        actual_pairs = [
            (turn, actor) for turn, actor, _placements in self.public_action_history
        ]
        if actual_pairs != expected_pairs:
            raise PublicRootMixtureError(
                f"phase {self.phase!r} public history cutoff mismatch"
            )
        reconstructed = {
            "bb": {row: [] for row in ROWS},
            "btn": {row: [] for row in ROWS},
        }
        for turn, actor, placements in self.public_action_history:
            expected_placements = 5 if turn == 0 else 2
            if len(placements) != expected_placements:
                raise PublicRootMixtureError(
                    f"T{turn} {actor} requires {expected_placements} public placements"
                )
            for card, row in placements:
                reconstructed[actor][row].append(card)
        for actor, board in (("bb", self.board_bb), ("btn", self.board_btn)):
            rebuilt = tuple(
                tuple(sorted(reconstructed[actor][row])) for row in ROWS
            )
            if rebuilt != board:
                raise PublicRootMixtureError(
                    f"{actor} board does not match public action history"
                )

    def to_canonical_dict(self) -> dict[str, Any]:
        payload = {
            "schema": PUBLIC_ROOT_CONTEXT_SCHEMA,
            "contract_version": self.contract_version,
            "actor": self.actor,
            "turn": self.turn,
            "phase": self.phase,
            "board_bb": {
                row: list(cards) for row, cards in zip(ROWS, self.board_bb)
            },
            "board_btn": {
                row: list(cards) for row, cards in zip(ROWS, self.board_btn)
            },
            "public_action_history": _public_history_dict(self.public_action_history),
            "fantasy_state": self.fantasy_state,
            "bindings": self.bindings.to_canonical_dict(),
        }
        lowered_keys = {
            str(key).lower()
            for key in payload
        }
        leaked = lowered_keys & _FORBIDDEN_CONTEXT_FIELDS
        if leaked:  # pragma: no cover - declared schema defense
            raise AssertionError(f"canonical public context leaked private fields: {leaked}")
        return payload

    def digest(self) -> str:
        return _canonical_sha256(self.to_canonical_dict())


@dataclass(frozen=True, slots=True)
class ExplicitHypotheticalPrivateType:
    """One mutually exclusive ex-ante type in a declared finite support.

    The cards are hypotheses used to build solver roots.  They are not a
    runtime actor observation and the compiler has no actual-type selector.
    """

    type_id: str
    hypothetical_own_recall: PrivateRecall
    hypothetical_current_draw: tuple[str, ...]
    prior_mass: Fraction
    root_range: FullCardRange = field(repr=False)

    def __post_init__(self) -> None:
        type_id = str(self.type_id)
        if not type_id or type_id != type_id.strip():
            raise PublicRootMixtureError("type_id must be non-empty and trimmed")
        if not isinstance(self.hypothetical_own_recall, PrivateRecall):
            raise TypeError("hypothetical_own_recall must be PrivateRecall")
        draw = _canonical_cards(
            self.hypothetical_current_draw,
            label="hypothetical_current_draw",
        )
        if len(draw) != 3:
            raise PublicRootMixtureError(
                "each hypothetical private type requires exactly three current cards"
            )
        if not isinstance(self.prior_mass, Fraction):
            raise TypeError("private type prior_mass must be fractions.Fraction")
        if self.prior_mass <= 0:
            raise PublicRootMixtureError("private type prior_mass must be positive")
        if not isinstance(self.root_range, FullCardRange):
            raise TypeError("root_range must be FullCardRange")
        object.__setattr__(self, "type_id", type_id)
        object.__setattr__(self, "hypothetical_current_draw", draw)


@dataclass(frozen=True, slots=True)
class CompiledPublicRootMixture:
    """Opt-in multi-root entries and a private-payload-free audit envelope."""

    context_digest: str
    entries: tuple[MultiRootChanceEntry, ...]
    type_id_sha256s: tuple[str, ...]
    private_type_commitments: tuple[str, ...]
    support_sha256: str
    audit_manifest_json: str = field(repr=False)
    audit_manifest_sha256: str

    @property
    def audit_manifest(self) -> Mapping[str, Any]:
        # Return a fresh snapshot so callers cannot mutate the committed JSON.
        return MappingProxyType(json.loads(self.audit_manifest_json))

    @property
    def production_sampling_ready(self) -> bool:
        return False

    @property
    def support_authorization_class(self) -> str:
        return EXPLICIT_SUPPORT_AUTHORIZATION_CLASS

    @property
    def authorized_as_production_mccfr_prior(self) -> bool:
        return False


def _context_public_tuple(context: PublicRootContext) -> tuple[Any, ...]:
    return (
        context.contract_version,
        context.actor,
        context.turn,
        context.phase,
        context.board_bb,
        context.board_btn,
        context.public_action_history,
        context.fantasy_state,
    )


def _observation_public_tuple(observation: InfoSetKey) -> tuple[Any, ...]:
    return (
        observation.contract_version,
        observation.actor,
        observation.turn,
        observation.phase,
        observation.board_bb,
        observation.board_btn,
        observation.public_action_history,
        observation.fantasy_state,
    )


def _observation_for_type(
    context: PublicRootContext,
    private_type: ExplicitHypotheticalPrivateType,
) -> InfoSetKey:
    return InfoSetKey(
        contract_version=context.contract_version,
        actor=context.actor,  # type: ignore[arg-type]
        turn=context.turn,
        phase=context.phase,
        board_bb=context.board_bb,
        board_btn=context.board_btn,
        public_action_history=context.public_action_history,
        own_recall=private_type.hypothetical_own_recall,
        current_draw=private_type.hypothetical_current_draw,
        fantasy_state=context.fantasy_state,
    )


def _private_type_commitment(
    context_digest: str,
    observation: InfoSetKey,
) -> str:
    return _canonical_sha256(
        {
            "schema": "ofc_hypothetical_actor_private_type_commitment/v1",
            "public_context_digest": context_digest,
            "actor": observation.actor,
            "phase": observation.phase,
            "own_recall": _recall_dict(observation.own_recall),
            "current_draw": list(observation.current_draw),
            "physical_joker_ids": list(_PHYSICAL_JOKERS),
            "joker_physical_identity_collapsed": False,
        }
    )


_AUDIT_SUPPORT_ROW_FIELDS = frozenset(
    {
        "root_id_sha256",
        "type_id_sha256",
        "hypothetical_private_type_commitment",
        "prior_mass_exact",
        "observation_digest",
        "range_content_sha256",
        "range_build_sha256",
        "behavior_model_sha256",
        "canonical_full_card_adapter_binding_sha256",
        "producer_claimed_hidden_assignment_enumeration_exhaustive",
        "independently_verified_hidden_assignment_enumeration_exhaustive",
    }
)


def _support_row(
    *,
    entry: MultiRootChanceEntry,
    type_id_sha256: str,
    private_type_commitment: str,
) -> dict[str, Any]:
    adapter = entry.adapter
    if type(adapter) is not FullCardGenerativeAdapter:
        raise PublicRootMixtureError(
            "compiled public mixture requires exact FullCardGenerativeAdapter instances"
        )
    observation = adapter.observation
    root_range = adapter.root_range
    producer_claim = root_range.metadata.get(
        "exhaustive_hidden_discard_enumeration"
    )
    if not isinstance(producer_claim, bool):
        raise PublicRootMixtureError(
            "range metadata must bind hidden-discard enumeration exactness"
        )
    _require_live_attestor_identities()
    adapter_binding = _independently_verify_self_hash(
        _CANONICAL_ADAPTER_CHECKPOINT_BINDING_ATTESTOR(entry),
        label="canonical FullCard adapter binding",
    )
    _independently_verify_self_hash(
        adapter_binding.get("live_runtime_semantic_binding"),
        label="canonical FullCard adapter live runtime semantic binding",
    )
    adapter_binding_sha256 = adapter_binding.get("binding_sha256")
    _require_sha256(
        adapter_binding_sha256,
        label="canonical FullCard adapter binding SHA256",
    )
    row = {
        "root_id_sha256": entry.root_id_sha256,
        "type_id_sha256": type_id_sha256,
        "hypothetical_private_type_commitment": private_type_commitment,
        "prior_mass_exact": (
            f"{entry.prior_mass.numerator}/{entry.prior_mass.denominator}"
        ),
        "observation_digest": observation.digest(),
        "range_content_sha256": root_range.range_content_sha256,
        "range_build_sha256": root_range.range_build_sha256,
        "behavior_model_sha256": root_range.behavior_model_sha256,
        "canonical_full_card_adapter_binding_sha256": adapter_binding_sha256,
        "producer_claimed_hidden_assignment_enumeration_exhaustive": (
            producer_claim
        ),
        # The explicit-support compiler does not receive the raw enumerated
        # assignment universe and therefore cannot independently rederive this
        # producer claim.  Keep the independent evidence fail-closed.
        "independently_verified_hidden_assignment_enumeration_exhaustive": False,
    }
    if set(row) != _AUDIT_SUPPORT_ROW_FIELDS:  # pragma: no cover - schema defense
        raise AssertionError("audit support row schema drifted")
    return row


def _audit_manifest(
    context: PublicRootContext,
    rows: Sequence[Mapping[str, Any]],
    *,
    support_sha256: str,
) -> dict[str, Any]:
    all_producers_claim_exhaustive = all(
        row["producer_claimed_hidden_assignment_enumeration_exhaustive"]
        for row in rows
    )
    return {
        "schema": PUBLIC_ROOT_MIXTURE_SCHEMA,
        "compiler_mode": "opt_in_explicit_finite_support",
        "public_context": context.to_canonical_dict(),
        "public_context_digest": context.digest(),
        "support_scope": EXPLICIT_SUPPORT_SCOPE,
        "support_sha256": support_sha256,
        "support_size": len(rows),
        "support_rows": [dict(row) for row in rows],
        "prior_probability_mass_exact": "1/1",
        "mixture_exactness": MIXTURE_EXACTNESS,
        "support_authorization": {
            "authorization_class": EXPLICIT_SUPPORT_AUTHORIZATION_CLASS,
            "proposal_origin_commitments_verified": False,
            "promoted_behavior_likelihood_verified": False,
            "conditional_range_posterior_verified": False,
            "posterior_join_normalization_verified": False,
            "authorized_as_production_mccfr_prior": False,
            "required_production_api": (
                "compile_posterior_joined_public_root_mixture"
            ),
        },
        "full_deck_actor_private_type_enumeration_exhaustive": False,
        "producer_claimed_conditional_hidden_assignment_enumeration_exhaustive_for_all_types": (
            all_producers_claim_exhaustive
        ),
        "independently_verified_conditional_hidden_assignment_enumeration_exhaustive_for_all_types": False,
        "physical_joker_ids": list(_PHYSICAL_JOKERS),
        "joker_physical_identity_collapsed": False,
        "policy_identity": "InfoSetKey",
        "hidden_information_audit": {
            "audit_proof_scope": "public_context_schema_and_committed_serialization_only",
            "public_context_allowed_fields": sorted(_PUBLIC_CONTEXT_INPUT_FIELDS),
            "public_context_forbidden_fields": sorted(_FORBIDDEN_CONTEXT_FIELDS),
            "public_context_contains_actor_private_fields": False,
            "actual_actor_private_cards_parameter_exists": False,
            "actual_actor_type_selector_consumed": False,
            "declared_support_origin_independently_verified": False,
            "hypothetical_support_raw_cards_serialized_in_audit": False,
            "hypothetical_support_payloads_committed_by_sha256": True,
            "singleton_actual_hand_conditioning_rejected": True,
        },
        "observation_range_behavior_source_rules_and_live_semantics_bound": True,
        "promotion_eligible": False,
        "serving_default_changed": False,
        "production_sampling_ready": False,
        "remaining_before_production_sampling": list(
            PRODUCTION_SAMPLING_REMAINING_REQUIREMENTS
        ),
    }


def compile_explicit_public_root_mixture(
    context: PublicRootContext,
    hypothetical_support: Sequence[ExplicitHypotheticalPrivateType],
) -> CompiledPublicRootMixture:
    """Compile a finite ex-ante type support into real full-card root adapters.

    There is deliberately no parameter for the actor's actual recall/current
    draw or for selecting which support element is real.  A singleton support
    is rejected.  This API does not establish that the supplied finite support
    covers the full publicly compatible private-type universe, so its result
    is always non-promoting.
    """

    if not isinstance(context, PublicRootContext):
        raise TypeError("context must be PublicRootContext")
    context.bindings.verify_live()
    supplied = tuple(hypothetical_support)
    if len(supplied) < 2:
        raise PublicRootMixtureError(
            "public mixture requires at least two hypothetical private types; "
            "singleton actual-hand conditioning is forbidden"
        )
    if any(not isinstance(item, ExplicitHypotheticalPrivateType) for item in supplied):
        raise TypeError(
            "hypothetical_support must contain ExplicitHypotheticalPrivateType values"
        )
    if len({item.type_id for item in supplied}) != len(supplied):
        raise PublicRootMixtureError("hypothetical private type IDs must be unique")
    if sum((item.prior_mass for item in supplied), Fraction(0, 1)) != 1:
        raise PublicRootMixtureError(
            "hypothetical private type prior masses must sum exactly to one"
        )

    context_digest = context.digest()
    prepared: list[
        tuple[
            str,
            str,
            str,
            Fraction,
            FullCardGenerativeAdapter,
        ]
    ] = []
    for item in supplied:
        observation = _observation_for_type(context, item)
        if _observation_public_tuple(observation) != _context_public_tuple(context):
            raise AssertionError("hypothetical observation changed public context")
        verify_full_card_range(observation, item.root_range)
        if item.root_range.behavior_model_id != context.bindings.behavior_model_id:
            raise PublicRootMixtureError(
                "range behavior model ID does not match public context binding"
            )
        if (
            item.root_range.behavior_model_sha256
            != context.bindings.behavior_model_sha256
        ):
            raise PublicRootMixtureError(
                "range behavior model hash does not match public context binding"
            )
        commitment = _private_type_commitment(context_digest, observation)
        type_id_sha256 = hashlib.sha256(item.type_id.encode("utf-8")).hexdigest()
        adapter = FullCardGenerativeAdapter(observation, item.root_range)
        prepared.append(
            (
                commitment,
                type_id_sha256,
                item.type_id,
                item.prior_mass,
                adapter,
            )
        )
    commitments = [row[0] for row in prepared]
    if len(commitments) != len(set(commitments)):
        raise PublicRootMixtureError(
            "hypothetical support contains duplicate actor-private types"
        )

    ordered = sorted(prepared, key=lambda row: row[0])
    entries: list[MultiRootChanceEntry] = []
    type_id_sha256s: list[str] = []
    ordered_commitments: list[str] = []
    for commitment, type_id_sha256, _type_id, prior_mass, adapter in ordered:
        entries.append(
            MultiRootChanceEntry(
                root_id=f"explicit-private-type-{commitment}",
                adapter=adapter,
                prior_mass=prior_mass,
            )
        )
        type_id_sha256s.append(type_id_sha256)
        ordered_commitments.append(commitment)

    rows = [
        _support_row(
            entry=entry,
            type_id_sha256=type_id_sha256,
            private_type_commitment=commitment,
        )
        for entry, type_id_sha256, commitment in zip(
            entries, type_id_sha256s, ordered_commitments
        )
    ]
    support_sha256 = _canonical_sha256(rows)
    manifest = _audit_manifest(context, rows, support_sha256=support_sha256)
    manifest_json = _canonical_json(manifest)
    result = CompiledPublicRootMixture(
        context_digest=context_digest,
        entries=tuple(entries),
        type_id_sha256s=tuple(type_id_sha256s),
        private_type_commitments=tuple(ordered_commitments),
        support_sha256=support_sha256,
        audit_manifest_json=manifest_json,
        audit_manifest_sha256=hashlib.sha256(manifest_json.encode("utf-8")).hexdigest(),
    )
    verify_compiled_public_root_mixture(context, result)
    return result


def verify_compiled_public_root_mixture(
    context: PublicRootContext,
    result: CompiledPublicRootMixture,
) -> Mapping[str, Any]:
    """Freshly verify bindings, adapters, exact masses, and the audit envelope."""

    if not isinstance(context, PublicRootContext):
        raise TypeError("context must be PublicRootContext")
    if not isinstance(result, CompiledPublicRootMixture):
        raise TypeError("result must be CompiledPublicRootMixture")
    context.bindings.verify_live()
    if result.context_digest != context.digest():
        raise PublicRootMixtureError("compiled mixture public context hash mismatch")
    count = len(result.entries)
    if count < 2:
        raise PublicRootMixtureError("compiled mixture cannot contain a singleton root")
    if not (
        len(result.type_id_sha256s)
        == len(result.private_type_commitments)
        == count
    ):
        raise PublicRootMixtureError("compiled support identity counts do not match")
    if any(not isinstance(entry, MultiRootChanceEntry) for entry in result.entries):
        raise TypeError("compiled entries must be MultiRootChanceEntry values")
    if len(set(result.type_id_sha256s)) != count:
        raise PublicRootMixtureError("compiled type ID commitments are not unique")
    if len({entry.root_id for entry in result.entries}) != count:
        raise PublicRootMixtureError("compiled root IDs are not unique")
    if len({id(entry.adapter) for entry in result.entries}) != count:
        raise PublicRootMixtureError("compiled roots must own distinct adapter instances")
    if sum((entry.prior_mass for entry in result.entries), Fraction(0, 1)) != 1:
        raise PublicRootMixtureError("compiled root prior mass is not exactly one")
    if tuple(sorted(result.private_type_commitments)) != result.private_type_commitments:
        raise PublicRootMixtureError("compiled private support is not canonically ordered")

    rows: list[dict[str, Any]] = []
    derived_commitments: list[str] = []
    for entry, type_id_sha256, declared_commitment in zip(
        result.entries,
        result.type_id_sha256s,
        result.private_type_commitments,
    ):
        _require_sha256(type_id_sha256, label="type_id_sha256")
        _require_sha256(declared_commitment, label="private_type_commitment")
        if type(entry.adapter) is not FullCardGenerativeAdapter:
            raise PublicRootMixtureError(
                "compiled root adapter is not the canonical FullCard adapter"
            )
        observation = entry.adapter.observation
        if _observation_public_tuple(observation) != _context_public_tuple(context):
            raise PublicRootMixtureError("compiled root changed public root context")
        verify_full_card_range(observation, entry.adapter.root_range)
        root_range = entry.adapter.root_range
        if (
            root_range.behavior_model_id != context.bindings.behavior_model_id
            or root_range.behavior_model_sha256
            != context.bindings.behavior_model_sha256
        ):
            raise PublicRootMixtureError(
                "compiled range behavior identity drifted from context binding"
            )
        commitment = _private_type_commitment(context.digest(), observation)
        if commitment != declared_commitment:
            raise PublicRootMixtureError("compiled private type commitment mismatch")
        if entry.root_id != f"explicit-private-type-{commitment}":
            raise PublicRootMixtureError("compiled root ID is not commitment-derived")
        derived_commitments.append(commitment)
        rows.append(
            _support_row(
                entry=entry,
                type_id_sha256=type_id_sha256,
                private_type_commitment=commitment,
            )
        )
    if len(derived_commitments) != len(set(derived_commitments)):
        raise PublicRootMixtureError("compiled mixture contains duplicate private types")
    support_sha256 = _canonical_sha256(rows)
    if support_sha256 != result.support_sha256:
        raise PublicRootMixtureError("compiled support hash mismatch")
    expected_manifest = _audit_manifest(
        context,
        rows,
        support_sha256=support_sha256,
    )
    expected_json = _canonical_json(expected_manifest)
    if result.audit_manifest_json != expected_json:
        raise PublicRootMixtureError("compiled audit manifest content mismatch")
    expected_manifest_sha256 = hashlib.sha256(expected_json.encode("utf-8")).hexdigest()
    if result.audit_manifest_sha256 != expected_manifest_sha256:
        raise PublicRootMixtureError("compiled audit manifest hash mismatch")
    return MappingProxyType(
        {
            "verified": True,
            "public_context_digest": context.digest(),
            "support_size": count,
            "support_sha256": support_sha256,
            "audit_manifest_sha256": expected_manifest_sha256,
            "prior_probability_mass_exact": "1/1",
            "support_authorization_class": (
                EXPLICIT_SUPPORT_AUTHORIZATION_CLASS
            ),
            "authorized_as_production_mccfr_prior": False,
            "production_sampling_ready": False,
            "promotion_eligible": False,
        }
    )


__all__ = [
    "CompiledPublicRootMixture",
    "EXPLICIT_SUPPORT_SCOPE",
    "EXPLICIT_SUPPORT_AUTHORIZATION_CLASS",
    "ExplicitHypotheticalPrivateType",
    "MIXTURE_EXACTNESS",
    "PRODUCTION_SAMPLING_REMAINING_REQUIREMENTS",
    "PUBLIC_ROOT_BINDINGS_SCHEMA",
    "PUBLIC_ROOT_CONTEXT_SCHEMA",
    "PUBLIC_ROOT_MIXTURE_SCHEMA",
    "PublicRootBindings",
    "PublicRootContext",
    "PublicRootMixtureError",
    "compile_explicit_public_root_mixture",
    "verify_compiled_public_root_mixture",
]
