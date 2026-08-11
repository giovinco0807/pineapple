"""Information-set-safe hidden-card particles for heads-up regular OFC.

The sampler deliberately accepts only :class:`ActorObservation`.  It never
accepts a simulator ``WorldState``, a replay record, a realized deck tail, or
an opponent's realized private discards.  At T1-T4 the cards unknown to the
actor are partitioned into:

* the opponent's inferred number of private discards; and
* an ordered unseen deck for rollout chance events.

The initial prior is exchangeable over those unknown cards.  Later search
code may reweight complete particles using an opponent policy, but must not
inject replay truth into this sampler.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .cards import ALL_CARDS, validate_cards
from .counter_rng import COUNTER_RNG_SCHEMA, CounterRngKey
from .hu_infoset import ActorObservation, ScoringContext
from .state import Board


HIDDEN_CARD_BELIEF_SCHEMA = "regular_ofc_hidden_card_belief_v1"
HIDDEN_CARD_PARTICLE_SCHEMA = "regular_ofc_hidden_card_particle_v1"
HIDDEN_CARD_BATCH_SCHEMA = "regular_ofc_hidden_card_particle_batch_v1"
HIDDEN_CARD_PRIOR = "uniform_exchangeable_v1"

_RNG_PHASE = "hidden_card_belief"
_RNG_STREAM = "unknown_card_permutation"
_RNG_DOMAIN = 1 << 63
_ATTEMPT_BITS = 32
_HEX_DIGITS = frozenset("0123456789abcdef")

# (street, action order) ->
# (hero board cards, opponent public cards, hero private discards,
#  opponent private discard count)
_DECISION_GEOMETRY: dict[
    tuple[str, str], tuple[int, int, int, int]
] = {
    ("T1", "first"): (5, 5, 0, 0),
    ("T1", "second"): (5, 7, 0, 1),
    ("T2", "first"): (7, 7, 1, 1),
    ("T2", "second"): (7, 9, 1, 2),
    ("T3", "first"): (9, 9, 2, 2),
    ("T3", "second"): (9, 11, 2, 3),
    ("T4", "first"): (11, 11, 3, 3),
    ("T4", "second"): (11, 13, 3, 4),
}


class HiddenCardBeliefError(ValueError):
    """Raised when an observation or generated particle is inconsistent."""


def turn2_actor_observation(
    *,
    hero_board: Board,
    opponent_public_board: Board,
    dealt_cards: Iterable[str],
    hero_seat: str,
    hero_private_discards: Iterable[str] = (),
    visible_dead_cards: Iterable[str] | None = None,
    scoring: ScoringContext | None = None,
) -> ActorObservation:
    """Build a T2 policy root from actor-visible cards only.

    ``dead_cards`` and opponent-private discards are intentionally absent from
    this API.  Historical replay artifacts may carry those fields as offline
    truth, but they must never affect a hidden-card belief.  A caller may pass
    the hero's explicit private discard directly or a legacy visible-dead list
    containing opponent public cards plus that hero-private discard.
    """

    dealt = tuple(dealt_cards)
    explicit_hero_private = tuple(hero_private_discards)
    visible = None if visible_dead_cards is None else tuple(visible_dead_cards)
    if visible is not None:
        try:
            validate_cards(visible)
        except ValueError as exc:
            raise HiddenCardBeliefError(str(exc)) from exc
        opponent_public = set(opponent_public_board.all_cards())
        visible_hero_private = tuple(
            card for card in visible if card not in opponent_public
        )
        if explicit_hero_private and set(visible_hero_private) != set(
            explicit_hero_private
        ):
            raise HiddenCardBeliefError(
                "visible dead cards disagree with explicit hero private discards"
            )
    else:
        visible_hero_private = ()

    hero_private = explicit_hero_private or visible_hero_private
    if len(hero_private) != 1:
        raise HiddenCardBeliefError(
            "T2 belief root requires exactly one explicit hero-visible discard"
        )
    if hero_seat not in {"first", "second"}:
        raise HiddenCardBeliefError(f"invalid hero seat: {hero_seat!r}")
    observation = ActorObservation(
        hero_board=hero_board,
        opponent_public_board=opponent_public_board,
        dealt_cards=dealt,
        hero_private_discards=hero_private,
        seat=hero_seat,  # type: ignore[arg-type]
        street="T2",
        to_act_order=(
            "second"
            if opponent_public_board.card_count() > hero_board.card_count()
            else "first"
        ),
        scoring=scoring or ScoringContext(),
    )
    _validate_decision_observation(observation)
    return observation


@dataclass(frozen=True)
class HiddenCardParticle:
    """One complete assignment of cards hidden from the acting player."""

    observation_fingerprint: str
    street: str
    sample_index: int
    opponent_private_discards: tuple[str, ...]
    unseen_deck: tuple[str, ...]
    rng_key_digest: str
    prior: str = HIDDEN_CARD_PRIOR

    def __post_init__(self) -> None:
        opponent_discards = tuple(self.opponent_private_discards)
        unseen_deck = tuple(self.unseen_deck)
        if self.street not in {"T1", "T2", "T3", "T4"}:
            raise HiddenCardBeliefError("hidden-card particles support only T1-T4")
        if self.sample_index < 0:
            raise HiddenCardBeliefError("sample_index must be non-negative")
        if not _is_sha256(self.observation_fingerprint):
            raise HiddenCardBeliefError("invalid observation fingerprint")
        if not _is_sha256(self.rng_key_digest):
            raise HiddenCardBeliefError("invalid RNG key digest")
        if self.prior != HIDDEN_CARD_PRIOR:
            raise HiddenCardBeliefError(f"unsupported hidden-card prior: {self.prior!r}")
        try:
            validate_cards((*opponent_discards, *unseen_deck))
        except ValueError as exc:
            raise HiddenCardBeliefError(str(exc)) from exc
        object.__setattr__(self, "opponent_private_discards", opponent_discards)
        object.__setattr__(self, "unseen_deck", unseen_deck)

    @property
    def hidden_cards(self) -> tuple[str, ...]:
        return (*self.opponent_private_discards, *self.unseen_deck)

    def draw(self, count: int, *, offset: int = 0) -> tuple[str, ...]:
        """Return a deterministic deck slice without mutating the particle."""
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("count must be a non-negative integer")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        end = offset + count
        if end > len(self.unseen_deck):
            raise ValueError("draw exceeds unseen deck")
        return self.unseen_deck[offset:end]

    def validate_against(self, observation: ActorObservation) -> None:
        """Prove this particle is a full partition of an observation's unknowns."""
        observation = _require_observation(observation)
        _validate_decision_observation(observation)
        if self.observation_fingerprint != observation.fingerprint():
            raise HiddenCardBeliefError("particle belongs to a different observation")
        if self.street != observation.street:
            raise HiddenCardBeliefError("particle street disagrees with observation")
        if len(self.opponent_private_discards) != observation.opponent_discard_count:
            raise HiddenCardBeliefError("opponent private discard count is inconsistent")

        known = set(observation.known_unavailable_cards())
        hidden = set(self.hidden_cards)
        if known & hidden:
            raise HiddenCardBeliefError("particle overlaps actor-visible cards")
        if len(hidden) != len(self.hidden_cards):
            raise HiddenCardBeliefError("particle contains duplicate hidden cards")
        if known | hidden != set(ALL_CARDS):
            raise HiddenCardBeliefError("particle does not cover the regular 52-card deck")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": HIDDEN_CARD_PARTICLE_SCHEMA,
            "prior": self.prior,
            "observation_fingerprint": self.observation_fingerprint,
            "street": self.street,
            "sample_index": self.sample_index,
            "opponent_private_discards": list(self.opponent_private_discards),
            "unseen_deck": list(self.unseen_deck),
            "rng_key_digest": self.rng_key_digest,
        }

    def digest(self) -> str:
        return _canonical_digest(self.to_dict())


@dataclass(frozen=True)
class HiddenCardParticleBatch:
    """A deterministic, shard-addressable batch of hidden-card particles."""

    observation_fingerprint: str
    street: str
    base_seed: int
    run_id: str
    start_index: int
    particles: tuple[HiddenCardParticle, ...]
    prior: str = HIDDEN_CARD_PRIOR

    def __post_init__(self) -> None:
        particles = tuple(self.particles)
        if isinstance(self.base_seed, bool) or not isinstance(self.base_seed, int):
            raise TypeError("base_seed must be an integer")
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("run_id must not be empty")
        if self.start_index < 0:
            raise ValueError("start_index must be non-negative")
        if not particles:
            raise ValueError("a belief batch requires at least one particle")
        expected_indices = tuple(range(self.start_index, self.start_index + len(particles)))
        actual_indices = tuple(particle.sample_index for particle in particles)
        if actual_indices != expected_indices:
            raise HiddenCardBeliefError("particle sample indices are not contiguous")
        if any(
            particle.observation_fingerprint != self.observation_fingerprint
            or particle.street != self.street
            or particle.prior != self.prior
            for particle in particles
        ):
            raise HiddenCardBeliefError("batch contains a particle from another root")
        object.__setattr__(self, "particles", particles)

    @property
    def particle_digests(self) -> tuple[str, ...]:
        return tuple(particle.digest() for particle in self.particles)

    def validate_against(self, observation: ActorObservation) -> None:
        observation = _require_observation(observation)
        if observation.fingerprint() != self.observation_fingerprint:
            raise HiddenCardBeliefError("batch belongs to a different observation")
        for particle in self.particles:
            particle.validate_against(observation)

    def to_dict(self, *, include_particles: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": HIDDEN_CARD_BATCH_SCHEMA,
            "belief_schema": HIDDEN_CARD_BELIEF_SCHEMA,
            "prior": self.prior,
            "counter_rng_schema": COUNTER_RNG_SCHEMA,
            "observation_fingerprint": self.observation_fingerprint,
            "street": self.street,
            "base_seed": self.base_seed,
            "run_id": self.run_id,
            "start_index": self.start_index,
            "sample_count": len(self.particles),
            "particle_digests": list(self.particle_digests),
        }
        if include_particles:
            payload["particles"] = [particle.to_dict() for particle in self.particles]
        return payload

    def digest(self) -> str:
        return _canonical_digest(self.to_dict())


def sample_hidden_card_particle(
    observation: ActorObservation,
    *,
    base_seed: int,
    run_id: str,
    sample_index: int,
) -> HiddenCardParticle:
    """Sample one T1-T4 hidden-card assignment from actor-visible state only."""
    observation = _require_observation(observation)
    _validate_decision_observation(observation)
    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise TypeError("base_seed must be an integer")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id must not be empty")
    if isinstance(sample_index, bool) or not isinstance(sample_index, int):
        raise TypeError("sample_index must be an integer")
    if sample_index < 0:
        raise ValueError("sample_index must be non-negative")

    fingerprint = observation.fingerprint()
    known = set(observation.known_unavailable_cards())
    unknown = [card for card in ALL_CARDS if card not in known]
    shuffled = _counter_shuffle(
        unknown,
        base_seed=base_seed,
        run_id=run_id,
        sample_index=sample_index,
        street=observation.street,
        root_fingerprint=fingerprint,
    )
    opponent_discard_count = observation.opponent_discard_count
    # Opponent discard order is not observable and has no semantic effect at
    # this root, so canonicalize that subset.  Deck order remains meaningful.
    opponent_discards = tuple(
        sorted(shuffled[:opponent_discard_count], key=ALL_CARDS.index)
    )
    unseen_deck = tuple(shuffled[opponent_discard_count:])
    particle = HiddenCardParticle(
        observation_fingerprint=fingerprint,
        street=observation.street,
        sample_index=sample_index,
        opponent_private_discards=opponent_discards,
        unseen_deck=unseen_deck,
        rng_key_digest=_rng_key_digest(
            base_seed=base_seed,
            run_id=run_id,
            sample_index=sample_index,
            street=observation.street,
            root_fingerprint=fingerprint,
        ),
    )
    particle.validate_against(observation)
    return particle


def sample_hidden_card_particles(
    observation: ActorObservation,
    *,
    base_seed: int,
    run_id: str,
    sample_count: int,
    start_index: int = 0,
) -> HiddenCardParticleBatch:
    """Generate a deterministic T1-T4 particle batch suitable for sharding."""
    observation = _require_observation(observation)
    _validate_decision_observation(observation)
    if isinstance(sample_count, bool) or not isinstance(sample_count, int):
        raise TypeError("sample_count must be an integer")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if isinstance(start_index, bool) or not isinstance(start_index, int):
        raise TypeError("start_index must be an integer")
    if start_index < 0:
        raise ValueError("start_index must be non-negative")

    particles = tuple(
        sample_hidden_card_particle(
            observation,
            base_seed=base_seed,
            run_id=run_id,
            sample_index=sample_index,
        )
        for sample_index in range(start_index, start_index + sample_count)
    )
    batch = HiddenCardParticleBatch(
        observation_fingerprint=observation.fingerprint(),
        street=observation.street,
        base_seed=base_seed,
        run_id=run_id,
        start_index=start_index,
        particles=particles,
    )
    batch.validate_against(observation)
    return batch


def _require_observation(value: object) -> ActorObservation:
    if not isinstance(value, ActorObservation):
        raise TypeError(
            "hidden-card belief requires ActorObservation; WorldState and replay truth are forbidden"
        )
    return value


def _validate_decision_observation(observation: ActorObservation) -> None:
    geometry = _DECISION_GEOMETRY.get((observation.street, observation.to_act_order))
    if geometry is None:
        raise HiddenCardBeliefError("hidden-card belief supports only T1-T4 decisions")
    expected_hero, expected_opponent, expected_hero_discards, expected_opponent_discards = (
        geometry
    )
    actual = (
        observation.hero_board.card_count(),
        observation.opponent_public_board.card_count(),
        len(observation.hero_private_discards),
        observation.opponent_discard_count,
    )
    expected = (
        expected_hero,
        expected_opponent,
        expected_hero_discards,
        expected_opponent_discards,
    )
    if len(observation.dealt_cards) != 3:
        raise HiddenCardBeliefError("T1-T4 observation must contain exactly three dealt cards")
    if actual != expected:
        raise HiddenCardBeliefError(
            "inconsistent T1-T4 decision geometry: "
            f"expected hero/opponent/hero-discards/opponent-discards={expected}, got {actual}"
        )


def _counter_shuffle(
    cards: list[str],
    *,
    base_seed: int,
    run_id: str,
    sample_index: int,
    street: str,
    root_fingerprint: str,
) -> list[str]:
    shuffled = list(cards)
    # Explicit Fisher-Yates plus counter-derived rejection sampling is stable
    # across Python processes and directly reproducible by the future Rust engine.
    for step, final_index in enumerate(range(len(shuffled) - 1, 0, -1)):
        swap_index = _counter_randbelow(
            final_index + 1,
            base_seed=base_seed,
            run_id=run_id,
            sample_index=sample_index,
            street=street,
            root_fingerprint=root_fingerprint,
            step=step,
        )
        shuffled[final_index], shuffled[swap_index] = (
            shuffled[swap_index],
            shuffled[final_index],
        )
    return shuffled


def _counter_randbelow(
    upper_bound: int,
    *,
    base_seed: int,
    run_id: str,
    sample_index: int,
    street: str,
    root_fingerprint: str,
    step: int,
) -> int:
    if upper_bound <= 0:
        raise ValueError("upper_bound must be positive")
    limit = _RNG_DOMAIN - (_RNG_DOMAIN % upper_bound)
    attempt = 0
    while True:
        counter = (step << _ATTEMPT_BITS) | attempt
        value = CounterRngKey(
            base_seed=base_seed,
            run_id=run_id,
            phase=_RNG_PHASE,
            sample_index=sample_index,
            actor="chance",
            street=street,
            stream=_RNG_STREAM,
            counter=counter,
            root_fingerprint=root_fingerprint,
        ).seed()
        if value < limit:
            return value % upper_bound
        attempt += 1
        if attempt >= (1 << _ATTEMPT_BITS):  # pragma: no cover - unreachable guard
            raise RuntimeError("counter RNG rejection sampling exhausted its domain")


def _rng_key_digest(
    *,
    base_seed: int,
    run_id: str,
    sample_index: int,
    street: str,
    root_fingerprint: str,
) -> str:
    root_key = CounterRngKey(
        base_seed=base_seed,
        run_id=run_id,
        phase=_RNG_PHASE,
        sample_index=sample_index,
        actor="chance",
        street=street,
        stream=_RNG_STREAM,
        counter=0,
        root_fingerprint=root_fingerprint,
    )
    return _canonical_digest(root_key.payload())


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX_DIGITS for character in value)
    )
