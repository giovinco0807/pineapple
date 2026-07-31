"""Public-information primitives for the T3 heads-up solver.

This module is deliberately limited to the first public-belief milestone.  It
contains information-set identities, exact finite-belief diagnostics, and a
small tabular CFR+ reference game.  It does *not* construct full-card ranges,
call the Rust T4 kernel, or claim that the production T3 policy is complete.

The central invariant is that a policy key contains everything remembered by
the acting player and nothing known only to the physical world.  In
particular, opponent discards, the undealt deck, particle identifiers, and RNG
seeds never enter :class:`InfoSetKey`.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Hashable, Iterable, Literal, Mapping, Sequence

from ai.engine.encoding import ALL_CARDS
from ai.engine.turn_order import POSITION_CONTRACT_VERSION


Actor = Literal["bb", "btn"]
BackupSense = Literal["max", "min"]
CardRows = tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]
PublicPlacement = tuple[str, str]
PublicHistoryEntry = tuple[int, str, tuple[PublicPlacement, ...]]
FractionInput = Fraction | int | str

ROWS = ("top", "middle", "bottom")
VALID_PHYSICAL_CARDS = frozenset(ALL_CARDS)
PHASES = ("t3_first", "t3_second", "t4_first", "t4_second")
PHASE_SPECS = {
    "t3_first": {"actor": "bb", "turn": 3, "board_counts": (9, 9), "last_action": (2, "btn")},
    "t3_second": {"actor": "btn", "turn": 3, "board_counts": (11, 9), "last_action": (3, "bb")},
    "t4_first": {"actor": "bb", "turn": 4, "board_counts": (11, 11), "last_action": (3, "btn")},
    "t4_second": {"actor": "btn", "turn": 4, "board_counts": (13, 11), "last_action": (4, "bb")},
}
FORBIDDEN_INFOSET_FIELDS = frozenset(
    {
        "opponent_private",
        "opponent_discards",
        "opponent_private_discards",
        "undealt",
        "undealt_cards",
        "remaining_deck",
        "live_cards",
        "particle_id",
        "world_id",
        "determinization_id",
        "seed",
        "rng_seed",
    }
)


def _fraction(value: FractionInput, *, label: str) -> Fraction:
    """Convert an exact input without silently accepting binary floats."""
    if isinstance(value, bool) or isinstance(value, float):
        raise TypeError(f"{label} must be an exact Fraction/int/str, not {type(value).__name__}")
    try:
        return value if isinstance(value, Fraction) else Fraction(value)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise TypeError(f"{label} is not an exact rational value: {value!r}") from exc


def _canonical_cards(cards: Iterable[str], *, label: str) -> tuple[str, ...]:
    out = tuple(sorted(str(card) for card in cards))
    if any(not card for card in out):
        raise ValueError(f"{label} contains an empty card")
    invalid = sorted(card for card in out if card not in VALID_PHYSICAL_CARDS)
    if invalid:
        raise ValueError(f"{label} contains invalid physical cards: {invalid}")
    if len(out) != len(set(out)):
        raise ValueError(f"{label} contains duplicate cards")
    return out


def _canonical_board(board: Sequence[Sequence[str]], *, label: str) -> CardRows:
    if len(board) != 3:
        raise ValueError(f"{label} must contain top/middle/bottom rows")
    rows = tuple(
        _canonical_cards(cards, label=f"{label}.{row}")
        for row, cards in zip(ROWS, board)
    )
    return rows  # type: ignore[return-value]


def _canonical_public_history(
    history: Iterable[PublicHistoryEntry],
) -> tuple[PublicHistoryEntry, ...]:
    canonical: list[PublicHistoryEntry] = []
    previous_order: tuple[int, int] | None = None
    actor_order = {"bb": 0, "btn": 1}
    for raw_turn, raw_actor, raw_placements in history:
        turn = int(raw_turn)
        actor = str(raw_actor)
        if actor not in actor_order:
            raise ValueError(f"public history actor must be 'bb' or 'btn', got {actor!r}")
        placements: list[PublicPlacement] = []
        seen_cards: set[str] = set()
        for raw_card, raw_row in raw_placements:
            card = str(raw_card)
            row = str(raw_row)
            if not card:
                raise ValueError("public history contains an empty card")
            if card not in VALID_PHYSICAL_CARDS:
                raise ValueError(f"public history contains invalid physical card {card!r}")
            if row not in ROWS:
                raise ValueError(f"invalid public placement row: {row!r}")
            if card in seen_cards:
                raise ValueError(f"public action places {card!r} more than once")
            seen_cards.add(card)
            placements.append((card, row))
        order = (turn, actor_order[actor])
        if previous_order is not None and order <= previous_order:
            raise ValueError("public history must be in strict turn/BB-before-BTN order")
        previous_order = order
        canonical.append((turn, actor, tuple(sorted(placements, key=lambda item: (item[1], item[0])))))
    return tuple(canonical)


@dataclass(frozen=True)
class PrivateRecall:
    """The acting player's perfect recall of private cards.

    Turns are retained explicitly.  Card order inside one draw is irrelevant,
    while turn order is significant.  This object must only describe the
    player whose information set is being constructed.
    """

    dealt_by_turn: tuple[tuple[int, tuple[str, ...]], ...] = ()
    discards_by_turn: tuple[tuple[int, str], ...] = ()

    def __post_init__(self) -> None:
        dealt: list[tuple[int, tuple[str, ...]]] = []
        previous_turn = -1
        all_dealt: set[str] = set()
        for raw_turn, raw_cards in self.dealt_by_turn:
            turn = int(raw_turn)
            if turn <= previous_turn:
                raise ValueError("private dealt turns must be strictly increasing")
            cards = _canonical_cards(raw_cards, label=f"private draw T{turn}")
            if all_dealt.intersection(cards):
                raise ValueError("private recall deals the same card on multiple turns")
            all_dealt.update(cards)
            dealt.append((turn, cards))
            previous_turn = turn

        discards: list[tuple[int, str]] = []
        previous_turn = -1
        seen_discards: set[str] = set()
        dealt_by_turn = dict(dealt)
        for raw_turn, raw_card in self.discards_by_turn:
            turn = int(raw_turn)
            card = str(raw_card)
            if turn <= previous_turn:
                raise ValueError("private discard turns must be strictly increasing")
            if not card:
                raise ValueError("private recall contains an empty discard")
            if card not in VALID_PHYSICAL_CARDS:
                raise ValueError(f"private recall contains invalid physical card {card!r}")
            if card in seen_discards:
                raise ValueError("private recall discards the same card more than once")
            if turn not in dealt_by_turn:
                raise ValueError(f"discard T{turn} has no matching remembered private draw")
            if card not in dealt_by_turn[turn]:
                raise ValueError(f"discard {card!r} was not in remembered T{turn} draw")
            seen_discards.add(card)
            discards.append((turn, card))
            previous_turn = turn

        object.__setattr__(self, "dealt_by_turn", tuple(dealt))
        object.__setattr__(self, "discards_by_turn", tuple(discards))

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "dealt_by_turn": [
                {"turn": turn, "cards": list(cards)} for turn, cards in self.dealt_by_turn
            ],
            "discards_by_turn": [
                {"turn": turn, "card": card} for turn, card in self.discards_by_turn
            ],
        }


@dataclass(frozen=True)
class JointParticle:
    """One physical hypothesis used by a range, never a policy identity.

    ``undealt_cards`` is the represented remainder after the current acting
    player's draw.  It may be a reduced-deck subset, but it must remain
    physically disjoint from both players' completed private recall and, when
    attached through :meth:`InfoSetKey.for_particle`, from public/current cards.
    """

    bb_recall: PrivateRecall
    btn_recall: PrivateRecall
    undealt_cards: tuple[str, ...]
    weight: Fraction = Fraction(1, 1)

    def __post_init__(self) -> None:
        undealt_cards = _canonical_cards(
            self.undealt_cards,
            label="particle undealt cards",
        )
        object.__setattr__(
            self,
            "undealt_cards",
            undealt_cards,
        )
        physical_zones: list[tuple[str, Iterable[str]]] = []
        for actor, recall in (("bb", self.bb_recall), ("btn", self.btn_recall)):
            physical_zones.extend(
                (f"{actor} private draw T{turn}", cards)
                for turn, cards in recall.dealt_by_turn
            )
        physical_zones.append(("particle undealt cards", undealt_cards))
        seen: dict[str, str] = {}
        for zone, cards in physical_zones:
            for card in cards:
                prior = seen.get(card)
                if prior is not None:
                    raise ValueError(
                        f"particle physical card {card!r} appears in both {prior} and {zone}"
                    )
                seen[card] = zone
        weight = _fraction(self.weight, label="particle weight")
        if weight < 0:
            raise ValueError("particle weight must be non-negative")
        object.__setattr__(self, "weight", weight)

    def own_recall(self, actor: Actor) -> PrivateRecall:
        if actor == "bb":
            return self.bb_recall
        if actor == "btn":
            return self.btn_recall
        raise ValueError(f"actor must be 'bb' or 'btn', got {actor!r}")


@dataclass(frozen=True)
class InfoSetKey:
    """Canonical, information-safe identity for one acting-player decision."""

    contract_version: str
    actor: Actor
    turn: int
    phase: str
    board_bb: CardRows
    board_btn: CardRows
    public_action_history: tuple[PublicHistoryEntry, ...]
    own_recall: PrivateRecall
    current_draw: tuple[str, ...]
    fantasy_state: str | None = None

    def __post_init__(self) -> None:
        if self.contract_version != POSITION_CONTRACT_VERSION:
            raise ValueError(
                "unsupported position contract: "
                f"{self.contract_version!r}; expected {POSITION_CONTRACT_VERSION!r}"
            )
        if self.actor not in ("bb", "btn"):
            raise ValueError(f"actor must be 'bb' or 'btn', got {self.actor!r}")
        if int(self.turn) not in (3, 4):
            raise ValueError(f"public T3 reference supports turn 3/4, got {self.turn!r}")
        if self.phase not in PHASES:
            raise ValueError(f"unsupported public-tree phase: {self.phase!r}")
        phase_turn = 3 if self.phase.startswith("t3") else 4
        if int(self.turn) != phase_turn:
            raise ValueError(f"phase {self.phase!r} does not belong to turn {self.turn}")
        spec = PHASE_SPECS[self.phase]
        expected_actor = str(spec["actor"])
        if self.actor != expected_actor:
            raise ValueError(f"phase {self.phase!r} requires actor={expected_actor!r}")

        object.__setattr__(self, "turn", int(self.turn))
        object.__setattr__(self, "board_bb", _canonical_board(self.board_bb, label="board_bb"))
        object.__setattr__(self, "board_btn", _canonical_board(self.board_btn, label="board_btn"))
        object.__setattr__(
            self,
            "public_action_history",
            _canonical_public_history(self.public_action_history),
        )
        object.__setattr__(
            self,
            "current_draw",
            _canonical_cards(self.current_draw, label="current private draw"),
        )
        if self.fantasy_state is not None:
            object.__setattr__(self, "fantasy_state", str(self.fantasy_state))
        self._validate_observation_cutoff()

    def _validate_observation_cutoff(self) -> None:
        """Reject incomplete or future-contaminated information-set states."""
        spec = PHASE_SPECS[self.phase]
        bb_count = sum(len(row) for row in self.board_bb)
        btn_count = sum(len(row) for row in self.board_btn)
        expected_counts = tuple(spec["board_counts"])
        if (bb_count, btn_count) != expected_counts:
            raise ValueError(
                f"phase {self.phase!r} requires BB/BTN board counts "
                f"{expected_counts[0]}/{expected_counts[1]}, got {bb_count}/{btn_count}"
            )
        for label, board in (("BB", self.board_bb), ("BTN", self.board_btn)):
            lengths = tuple(len(row) for row in board)
            if any(actual > limit for actual, limit in zip(lengths, (3, 5, 5))):
                raise ValueError(f"{label} board exceeds row capacity: {lengths}")
        if len(self.current_draw) != 3:
            raise ValueError(f"phase {self.phase!r} requires a 3-card current draw")

        expected_pairs: list[tuple[int, str]] = []
        last_turn, last_actor = spec["last_action"]
        for turn in range(int(last_turn) + 1):
            expected_pairs.append((turn, "bb"))
            if turn < int(last_turn) or last_actor == "btn":
                expected_pairs.append((turn, "btn"))
        actual_pairs = [(turn, actor) for turn, actor, _placements in self.public_action_history]
        if actual_pairs != expected_pairs:
            raise ValueError(
                f"phase {self.phase!r} requires public history through "
                f"T{last_turn} {last_actor}, got {actual_pairs}"
            )

        reconstructed = {
            "bb": {row: [] for row in ROWS},
            "btn": {row: [] for row in ROWS},
        }
        seen_public: set[str] = set()
        for turn, actor, placements in self.public_action_history:
            expected_placements = 5 if turn == 0 else 2
            if len(placements) != expected_placements:
                raise ValueError(
                    f"T{turn} {actor} history requires {expected_placements} placements, "
                    f"got {len(placements)}"
                )
            for card, row in placements:
                if card in seen_public:
                    raise ValueError(f"public history places {card!r} more than once")
                seen_public.add(card)
                reconstructed[actor][row].append(card)
        for actor, board in (("bb", self.board_bb), ("btn", self.board_btn)):
            expected_board = tuple(tuple(sorted(reconstructed[actor][row])) for row in ROWS)
            if expected_board != board:
                raise ValueError(
                    f"phase {self.phase!r} {actor.upper()} board does not match public history"
                )

        expected_recall_turns = tuple(range(1, self.turn))
        dealt_turns = tuple(turn for turn, _cards in self.own_recall.dealt_by_turn)
        discard_turns = tuple(turn for turn, _card in self.own_recall.discards_by_turn)
        if dealt_turns != expected_recall_turns or discard_turns != expected_recall_turns:
            raise ValueError(
                f"phase {self.phase!r} private recall must contain draws/discards for "
                f"turns {expected_recall_turns}, got draws={dealt_turns}, discards={discard_turns}"
            )
        if any(len(cards) != 3 for _turn, cards in self.own_recall.dealt_by_turn):
            raise ValueError("private recall requires exactly 3 dealt cards on each prior turn")

        own_public_by_turn = {
            turn: {card for card, _row in placements}
            for turn, actor, placements in self.public_action_history
            if actor == self.actor and turn > 0
        }
        own_discards = dict(self.own_recall.discards_by_turn)
        discard_public_overlap = set(own_discards.values()) & seen_public
        if discard_public_overlap:
            raise ValueError(
                "own private discard overlaps a public card: "
                f"{sorted(discard_public_overlap)}"
            )
        for turn, cards in self.own_recall.dealt_by_turn:
            expected_deal = own_public_by_turn.get(turn, set()) | {own_discards[turn]}
            if set(cards) != expected_deal:
                raise ValueError(
                    f"T{turn} {self.actor} private recall must equal its 2 public placements "
                    f"plus own discard; got {sorted(cards)}, expected {sorted(expected_deal)}"
                )

        remembered_cards = {
            card
            for _turn, cards in self.own_recall.dealt_by_turn
            for card in cards
        }
        current_overlap = set(self.current_draw) & (seen_public | remembered_cards)
        if current_overlap:
            raise ValueError(
                "current private draw overlaps public or previously remembered cards: "
                f"{sorted(current_overlap)}"
            )

    @classmethod
    def for_particle(
        cls,
        particle: JointParticle,
        *,
        contract_version: str,
        actor: Actor,
        turn: int,
        phase: str,
        board_bb: CardRows,
        board_btn: CardRows,
        public_action_history: tuple[PublicHistoryEntry, ...],
        current_draw: Sequence[str],
        fantasy_state: str | None = None,
    ) -> "InfoSetKey":
        """Validate a physical world, then key only on the actor's information."""
        key = cls(
            contract_version=contract_version,
            actor=actor,
            turn=turn,
            phase=phase,
            board_bb=board_bb,
            board_btn=board_btn,
            public_action_history=public_action_history,
            own_recall=particle.own_recall(actor),
            current_draw=tuple(current_draw),
            fantasy_state=fantasy_state,
        )
        key._validate_physical_particle(particle)
        return key

    def _validate_physical_particle(self, particle: JointParticle) -> None:
        """Ensure a hidden world is physically compatible without keying on it."""
        public_by_actor_turn: dict[str, dict[int, set[str]]] = {"bb": {}, "btn": {}}
        seen_public: set[str] = set()
        for turn, actor, placements in self.public_action_history:
            public_by_actor_turn[actor][turn] = {card for card, _row in placements}
            seen_public.update(card for card, _row in placements)

        all_recalled: set[str] = set()
        for actor, recall in (("bb", particle.bb_recall), ("btn", particle.btn_recall)):
            expected_turns = tuple(
                sorted(turn for turn in public_by_actor_turn[actor] if turn > 0)
            )
            dealt_turns = tuple(turn for turn, _cards in recall.dealt_by_turn)
            discard_turns = tuple(turn for turn, _card in recall.discards_by_turn)
            if dealt_turns != expected_turns or discard_turns != expected_turns:
                raise ValueError(
                    f"physical {actor} recall must contain completed public turns "
                    f"{expected_turns}, got draws={dealt_turns}, discards={discard_turns}"
                )
            discards = dict(recall.discards_by_turn)
            for turn, cards in recall.dealt_by_turn:
                if len(cards) != 3:
                    raise ValueError(
                        f"physical {actor} recall T{turn} requires exactly 3 dealt cards"
                    )
                expected_deal = public_by_actor_turn[actor][turn] | {discards[turn]}
                if set(cards) != expected_deal:
                    raise ValueError(
                        f"physical {actor} recall T{turn} must equal public placements plus "
                        f"own discard; got {sorted(cards)}, expected {sorted(expected_deal)}"
                    )
            discard_overlap = set(discards.values()) & seen_public
            if discard_overlap:
                raise ValueError(
                    f"physical {actor} discard overlaps public cards: "
                    f"{sorted(discard_overlap)}"
                )
            all_recalled.update(
                card for _turn, cards in recall.dealt_by_turn for card in cards
            )

        current_overlap = set(self.current_draw) & all_recalled
        if current_overlap:
            raise ValueError(
                "current private draw overlaps a physical particle's prior recalled cards: "
                f"{sorted(current_overlap)}"
            )
        undealt_overlap = set(particle.undealt_cards) & (
            seen_public | set(self.current_draw)
        )
        if undealt_overlap:
            raise ValueError(
                "particle undealt cards overlap public or current dealt cards: "
                f"{sorted(undealt_overlap)}"
            )

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "actor": self.actor,
            "turn": self.turn,
            "phase": self.phase,
            "board_bb": {row: list(cards) for row, cards in zip(ROWS, self.board_bb)},
            "board_btn": {row: list(cards) for row, cards in zip(ROWS, self.board_btn)},
            "public_action_history": [
                {
                    "turn": turn,
                    "actor": actor,
                    "placements": [[card, row] for card, row in placements],
                }
                for turn, actor, placements in self.public_action_history
            ],
            "own_recall": self.own_recall.to_canonical_dict(),
            "current_draw": list(self.current_draw),
            "fantasy_state": self.fantasy_state,
        }

    def canonical_json(self) -> str:
        encoded = json.dumps(
            self.to_canonical_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        lowered = encoded.lower()
        leaked = sorted(field for field in FORBIDDEN_INFOSET_FIELDS if f'"{field}"' in lowered)
        if leaked:
            raise AssertionError(f"information-set serialization leaked forbidden fields: {leaked}")
        return encoded

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def epsilon_smoothed_likelihood(
    likelihood: FractionInput,
    *,
    epsilon: FractionInput,
    action_count: int,
) -> Fraction:
    """Apply ``(1-epsilon)*p + epsilon/|A|`` using exact rationals."""
    probability = _fraction(likelihood, label="behavior likelihood")
    smoothing = _fraction(epsilon, label="epsilon")
    if not 0 <= probability <= 1:
        raise ValueError("behavior likelihood must be in [0, 1]")
    if not 0 <= smoothing <= 1:
        raise ValueError("epsilon must be in [0, 1]")
    if isinstance(action_count, bool) or int(action_count) <= 0:
        raise ValueError("action_count must be a positive integer")
    return (1 - smoothing) * probability + smoothing / int(action_count)


def bayes_posterior(
    prior: Mapping[Hashable, FractionInput],
    behavior_likelihood: Mapping[Hashable, FractionInput],
    *,
    epsilon: FractionInput = Fraction(0, 1),
    action_count: int = 1,
    physically_possible: Mapping[Hashable, bool] | None = None,
) -> dict[Hashable, Fraction]:
    """Return an exact posterior over hidden histories after a public action.

    Epsilon is a behavioral tremble for legal off-policy actions.  It must not
    revive a world that is physically incompatible with the observed cards;
    callers mark those worlds false in ``physically_possible``.
    """
    if not prior:
        raise ValueError("prior must contain at least one hypothesis")
    if set(prior) != set(behavior_likelihood):
        raise ValueError("prior and behavior likelihood must have identical hypotheses")
    if physically_possible is not None and set(physically_possible) != set(prior):
        raise ValueError("physically_possible must have identical hypotheses")
    exact_prior = {key: _fraction(value, label=f"prior[{key!r}]") for key, value in prior.items()}
    if any(value < 0 for value in exact_prior.values()) or sum(exact_prior.values()) <= 0:
        raise ValueError("prior weights must be non-negative and sum to a positive value")

    unnormalized: dict[Hashable, Fraction] = {}
    for key in prior:
        possible = True if physically_possible is None else physically_possible[key]
        if not isinstance(possible, bool):
            raise ValueError(f"physically_possible[{key!r}] must be boolean")
        smoothed = (
            epsilon_smoothed_likelihood(
                behavior_likelihood[key],
                epsilon=epsilon,
                action_count=action_count,
            )
            if possible
            else Fraction(0, 1)
        )
        unnormalized[key] = exact_prior[key] * smoothed
    evidence = sum(unnormalized.values(), Fraction(0, 1))
    if evidence <= 0:
        raise ValueError("observed public action has zero likelihood under every hypothesis")
    return {key: value / evidence for key, value in unnormalized.items()}


@dataclass(frozen=True)
class GroupedBackupResult:
    sense: BackupSense
    expected_action_values: Mapping[str, Fraction]
    selected_action: str
    selected_value: Fraction
    per_world_actions: Mapping[Hashable, str]
    pimc_value: Fraction
    strategy_fusion_advantage: Fraction
    metadata: Mapping[str, Any]


def _stable_extreme(values: Mapping[str, Fraction], sense: BackupSense) -> tuple[str, Fraction]:
    if not values:
        raise ValueError("at least one action is required")
    extreme = max(values.values()) if sense == "max" else min(values.values())
    action = min(action_id for action_id, value in values.items() if value == extreme)
    return action, extreme


def grouped_information_set_backup(
    world_action_values: Mapping[Hashable, Mapping[str, FractionInput]],
    *,
    weights: Mapping[Hashable, FractionInput] | None = None,
    sense: BackupSense,
) -> GroupedBackupResult:
    """Choose one action after averaging every world in an information set.

    ``pimc_value`` is returned only as a diagnostic counterexample.  It is the
    illegal value obtained by choosing a separate action inside each world.
    """
    if sense not in ("max", "min"):
        raise ValueError("sense must be 'max' or 'min'")
    if not world_action_values:
        raise ValueError("at least one world is required")
    worlds = list(world_action_values)
    action_ids = set(next(iter(world_action_values.values())))
    if not action_ids:
        raise ValueError("each world must contain at least one action")
    if any(set(values) != action_ids for values in world_action_values.values()):
        raise ValueError("all worlds in one information set must expose identical action IDs")

    if weights is None:
        exact_weights = {world: Fraction(1, len(worlds)) for world in worlds}
    else:
        if set(weights) != set(worlds):
            raise ValueError("weights must have exactly one entry per world")
        raw = {world: _fraction(weights[world], label=f"weight[{world!r}]") for world in worlds}
        if any(value < 0 for value in raw.values()):
            raise ValueError("world weights must be non-negative")
        total = sum(raw.values(), Fraction(0, 1))
        if total <= 0:
            raise ValueError("world weights must sum to a positive value")
        exact_weights = {world: value / total for world, value in raw.items()}

    exact_values: dict[Hashable, dict[str, Fraction]] = {}
    for world, values in world_action_values.items():
        exact_values[world] = {
            str(action_id): _fraction(value, label=f"value[{world!r}][{action_id!r}]")
            for action_id, value in values.items()
        }

    expected = {
        action_id: sum(
            exact_weights[world] * exact_values[world][action_id] for world in worlds
        )
        for action_id in sorted(action_ids)
    }
    selected_action, selected_value = _stable_extreme(expected, sense)
    per_world: dict[Hashable, str] = {}
    pimc_value = Fraction(0, 1)
    for world in worlds:
        action, value = _stable_extreme(exact_values[world], sense)
        per_world[world] = action
        pimc_value += exact_weights[world] * value
    advantage = (
        pimc_value - selected_value if sense == "max" else selected_value - pimc_value
    )
    if advantage < 0:
        raise AssertionError("PIMC extreme cannot be worse than a shared information-set action")
    return GroupedBackupResult(
        sense=sense,
        expected_action_values=expected,
        selected_action=selected_action,
        selected_value=selected_value,
        per_world_actions=per_world,
        pimc_value=pimc_value,
        strategy_fusion_advantage=advantage,
        metadata={
            "method": "grouped_information_set_backup",
            "strategy_fusion": False,
            "equilibrium_approx": False,
            "hu_exact": False,
            "position_contract_version": POSITION_CONTRACT_VERSION,
        },
    )


@dataclass(frozen=True)
class PublicSignalingGame:
    """A finite sender/receiver zero-sum game used as the CFR reference.

    Chance chooses ``private_type`` and reveals it only to the maximizing
    sender.  The sender chooses a public signal.  The minimizing receiver sees
    the signal, but not the type, and chooses a response.
    """

    private_types: tuple[str, ...]
    prior: Mapping[str, FractionInput]
    signals: tuple[str, ...]
    responses: tuple[str, ...]
    payoff: Mapping[str, Mapping[str, Mapping[str, FractionInput]]]

    def __post_init__(self) -> None:
        for label, values in (
            ("private_types", self.private_types),
            ("signals", self.signals),
            ("responses", self.responses),
        ):
            if not values or any(not value for value in values) or len(values) != len(set(values)):
                raise ValueError(f"{label} must contain unique, non-empty stable IDs")
        if set(self.prior) != set(self.private_types):
            raise ValueError("prior keys must match private_types")
        exact_prior = {
            private_type: _fraction(self.prior[private_type], label=f"prior[{private_type!r}]")
            for private_type in self.private_types
        }
        if any(value < 0 for value in exact_prior.values()):
            raise ValueError("type prior must be non-negative")
        total = sum(exact_prior.values(), Fraction(0, 1))
        if total <= 0:
            raise ValueError("type prior must sum to a positive value")
        exact_prior = {key: value / total for key, value in exact_prior.items()}

        exact_payoff: dict[str, dict[str, dict[str, Fraction]]] = {}
        if set(self.payoff) != set(self.private_types):
            raise ValueError("payoff type keys must match private_types")
        for private_type in self.private_types:
            by_signal = self.payoff[private_type]
            if set(by_signal) != set(self.signals):
                raise ValueError(f"payoff signals for {private_type!r} do not match signals")
            exact_payoff[private_type] = {}
            for signal in self.signals:
                by_response = by_signal[signal]
                if set(by_response) != set(self.responses):
                    raise ValueError(
                        f"payoff responses for {private_type!r}/{signal!r} do not match responses"
                    )
                exact_payoff[private_type][signal] = {
                    response: _fraction(
                        by_response[response],
                        label=f"payoff[{private_type!r}][{signal!r}][{response!r}]",
                    )
                    for response in self.responses
                }
        object.__setattr__(self, "prior", exact_prior)
        object.__setattr__(self, "payoff", exact_payoff)


@dataclass(frozen=True)
class SignalingProfileMetrics:
    value: float
    maximizing_best_response: float
    minimizing_best_response: float
    nash_conv: float
    exploitability: float


@dataclass(frozen=True)
class PublicCfrResult:
    iterations: int
    sender_average_strategy: Mapping[str, Mapping[str, float]]
    receiver_average_strategy: Mapping[str, Mapping[str, float]]
    sender_current_strategy: Mapping[str, Mapping[str, float]]
    receiver_current_strategy: Mapping[str, Mapping[str, float]]
    metrics: SignalingProfileMetrics
    exploitability_trace: tuple[tuple[int, float], ...]
    metadata: Mapping[str, Any]


def _regret_matching_plus_strategy(
    regrets: Mapping[str, float],
    action_ids: Sequence[str],
) -> dict[str, float]:
    positive = [max(0.0, float(regrets[action])) for action in action_ids]
    total = sum(positive)
    if total <= 0.0:
        uniform = 1.0 / len(action_ids)
        return {action: uniform for action in action_ids}
    return {action: value / total for action, value in zip(action_ids, positive)}


def _normalized_average(
    accumulated: Mapping[str, float],
    action_ids: Sequence[str],
) -> dict[str, float]:
    total = sum(float(accumulated[action]) for action in action_ids)
    if total <= 0.0:
        uniform = 1.0 / len(action_ids)
        return {action: uniform for action in action_ids}
    return {action: float(accumulated[action]) / total for action in action_ids}


def signaling_profile_metrics(
    game: PublicSignalingGame,
    sender_strategy: Mapping[str, Mapping[str, float]],
    receiver_strategy: Mapping[str, Mapping[str, float]],
) -> SignalingProfileMetrics:
    """Return profile value and exact best-response gaps for the finite game."""
    value = 0.0
    max_best = 0.0
    for private_type in game.private_types:
        probability = float(game.prior[private_type])
        type_value = 0.0
        action_values: list[float] = []
        for signal in game.signals:
            signal_value = sum(
                float(receiver_strategy[signal][response])
                * float(game.payoff[private_type][signal][response])
                for response in game.responses
            )
            action_values.append(signal_value)
            type_value += float(sender_strategy[private_type][signal]) * signal_value
        value += probability * type_value
        max_best += probability * max(action_values)

    min_best = 0.0
    for signal in game.signals:
        response_values = []
        for response in game.responses:
            response_values.append(
                sum(
                    float(game.prior[private_type])
                    * float(sender_strategy[private_type][signal])
                    * float(game.payoff[private_type][signal][response])
                    for private_type in game.private_types
                )
            )
        min_best += min(response_values)
    nash_conv = max(0.0, max_best - min_best)
    return SignalingProfileMetrics(
        value=value,
        maximizing_best_response=max_best,
        minimizing_best_response=min_best,
        nash_conv=nash_conv,
        exploitability=nash_conv / 2.0,
    )


def solve_public_signaling_game_cfr(
    game: PublicSignalingGame,
    *,
    iterations: int,
    linear_averaging: bool = True,
    checkpoints: Sequence[int] = (),
) -> PublicCfrResult:
    """Solve an explicit reduced public signaling game with deterministic CFR+.

    This is a correctness reference, not the full-card T3 runtime.  All chance
    outcomes and information sets are enumerated exactly; RM+ regrets and the
    average strategy use deterministic stable action order.
    """
    if isinstance(iterations, bool) or int(iterations) <= 0:
        raise ValueError("iterations must be a positive integer")
    iterations = int(iterations)
    checkpoint_set = {int(point) for point in checkpoints if 0 < int(point) <= iterations}
    checkpoint_set.add(iterations)

    sender_regret = {
        private_type: {signal: 0.0 for signal in game.signals}
        for private_type in game.private_types
    }
    receiver_regret = {
        signal: {response: 0.0 for response in game.responses}
        for signal in game.signals
    }
    sender_sum = {
        private_type: {signal: 0.0 for signal in game.signals}
        for private_type in game.private_types
    }
    receiver_sum = {
        signal: {response: 0.0 for response in game.responses}
        for signal in game.signals
    }
    trace: list[tuple[int, float]] = []

    for iteration in range(1, iterations + 1):
        sender_strategy = {
            private_type: _regret_matching_plus_strategy(
                sender_regret[private_type], game.signals
            )
            for private_type in game.private_types
        }
        receiver_strategy = {
            signal: _regret_matching_plus_strategy(receiver_regret[signal], game.responses)
            for signal in game.signals
        }
        average_weight = float(iteration if linear_averaging else 1)
        for private_type in game.private_types:
            for signal in game.signals:
                sender_sum[private_type][signal] += (
                    average_weight * sender_strategy[private_type][signal]
                )
        for signal in game.signals:
            for response in game.responses:
                receiver_sum[signal][response] += average_weight * receiver_strategy[signal][response]

        sender_delta = {
            private_type: {signal: 0.0 for signal in game.signals}
            for private_type in game.private_types
        }
        for private_type in game.private_types:
            signal_values = {
                signal: sum(
                    receiver_strategy[signal][response]
                    * float(game.payoff[private_type][signal][response])
                    for response in game.responses
                )
                for signal in game.signals
            }
            on_policy = sum(
                sender_strategy[private_type][signal] * signal_values[signal]
                for signal in game.signals
            )
            chance_reach = float(game.prior[private_type])
            for signal in game.signals:
                sender_delta[private_type][signal] = chance_reach * (
                    signal_values[signal] - on_policy
                )

        receiver_delta = {
            signal: {response: 0.0 for response in game.responses}
            for signal in game.signals
        }
        for signal in game.signals:
            # Counterfactual receiver utility is the negation of sender utility.
            response_values = {
                response: -sum(
                    float(game.prior[private_type])
                    * sender_strategy[private_type][signal]
                    * float(game.payoff[private_type][signal][response])
                    for private_type in game.private_types
                )
                for response in game.responses
            }
            on_policy = sum(
                receiver_strategy[signal][response] * response_values[response]
                for response in game.responses
            )
            for response in game.responses:
                receiver_delta[signal][response] = response_values[response] - on_policy

        for private_type in game.private_types:
            for signal in game.signals:
                sender_regret[private_type][signal] = max(
                    0.0,
                    sender_regret[private_type][signal] + sender_delta[private_type][signal],
                )
        for signal in game.signals:
            for response in game.responses:
                receiver_regret[signal][response] = max(
                    0.0,
                    receiver_regret[signal][response] + receiver_delta[signal][response],
                )

        if iteration in checkpoint_set:
            average_sender = {
                private_type: _normalized_average(sender_sum[private_type], game.signals)
                for private_type in game.private_types
            }
            average_receiver = {
                signal: _normalized_average(receiver_sum[signal], game.responses)
                for signal in game.signals
            }
            trace.append(
                (
                    iteration,
                    signaling_profile_metrics(game, average_sender, average_receiver).exploitability,
                )
            )

    average_sender = {
        private_type: _normalized_average(sender_sum[private_type], game.signals)
        for private_type in game.private_types
    }
    average_receiver = {
        signal: _normalized_average(receiver_sum[signal], game.responses)
        for signal in game.signals
    }
    current_sender = {
        private_type: _regret_matching_plus_strategy(sender_regret[private_type], game.signals)
        for private_type in game.private_types
    }
    current_receiver = {
        signal: _regret_matching_plus_strategy(receiver_regret[signal], game.responses)
        for signal in game.signals
    }
    metrics = signaling_profile_metrics(game, average_sender, average_receiver)
    if not all(math.isfinite(value) for _iteration, value in trace):
        raise RuntimeError("CFR produced a non-finite exploitability trace")
    return PublicCfrResult(
        iterations=iterations,
        sender_average_strategy=average_sender,
        receiver_average_strategy=average_receiver,
        sender_current_strategy=current_sender,
        receiver_current_strategy=current_receiver,
        metrics=metrics,
        exploitability_trace=tuple(trace),
        metadata={
            "method": "reduced_public_signaling_cfr_plus",
            "strategy_fusion": False,
            "equilibrium_approx": True,
            "hu_exact": False,
            "runtime_integrated": False,
            "rust_leaf_integrated": False,
            "position_contract_version": POSITION_CONTRACT_VERSION,
        },
    )


def reduced_bluff_signaling_game() -> PublicSignalingGame:
    """Return a two-type public bluff game with a non-pure equilibrium.

    Equilibrium behavior is: strong always bets, weak bets with probability
    1/3, and the receiver calls a bet with probability 2/3.  The zero-sum
    sender value is 1/3.
    """
    return PublicSignalingGame(
        private_types=("strong", "weak"),
        prior={"strong": Fraction(1, 2), "weak": Fraction(1, 2)},
        signals=("bet", "check"),
        responses=("call", "fold"),
        payoff={
            "strong": {
                "bet": {"call": 2, "fold": 1},
                "check": {"call": 1, "fold": 1},
            },
            "weak": {
                "bet": {"call": -2, "fold": 1},
                "check": {"call": -1, "fold": -1},
            },
        },
    )
