"""Public-only counterfactual actor-private proposals for T3/T4.

The public-root compiler needs an ex-ante distribution over every actor
private type compatible with one public cut.  Its first implementation accepts
an arbitrary finite support and therefore cannot establish where that support
came from or how much of the physical 54-card universe it covers.

This module closes only the *proposal-generation* part of that gap.  Starting
from :class:`~ai.tutor.t3_t4_public_root_mixture.PublicRootContext` (which has
no actor-private fields), it defines the complete compatible actor-private
type universe in closed form and returns a bounded, deterministic,
content-addressed sample without scanning that universe.  A uniformly drawn
affine full-cycle permutation gives every universe member the same marginal
inclusion probability and never emits a duplicate within one traversal.

The output is deliberately an **unweighted proposal**, not a posterior and not
an MCCFR chance prior.  Actor-history behavior likelihoods, the opposing
conditional range, and their promoted calibration artifacts must be applied
and freshly verified downstream.  Consequently every artifact here remains
``promotion_eligible=false`` and ``production_sampling_ready=false``.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
from collections import Counter
from dataclasses import InitVar, dataclass, field, fields as dataclass_fields
from fractions import Fraction
from functools import wraps
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterator, Mapping, Sequence

import ai.engine.encoding as _encoding_module
import ai.tutor.t3_hu_multi_root_mccfr as _multi_root_module
import ai.tutor.t3_hu_public_cfr as _public_cfr_module
import ai.tutor.t3_t4_public_root_mixture as _mixture_module
from ai.tutor.t3_hu_public_cfr import InfoSetKey, JointParticle, PrivateRecall, ROWS
from ai.tutor.t3_t4_public_root_mixture import PublicRootContext


COUNTERFACTUAL_PRIVATE_TYPE_SCHEMA = (
    "ofc_t3_t4_counterfactual_private_type_proposal/v1"
)
COUNTERFACTUAL_PRIVATE_TYPE_BATCH_SCHEMA = (
    "ofc_t3_t4_counterfactual_private_type_proposal_batch/v1"
)
UNIVERSE_SCHEMA = "ofc_t3_t4_public_compatible_actor_private_type_universe/v1"
PUBLIC_SEED_PLAN_SCHEMA = "ofc_t3_t4_public_seed_plan/v1"
PUBLIC_SEED_REVEAL_SCHEMA = "ofc_t3_t4_public_seed_reveal/v1"
PRIVACY_SAFE_BATCH_SCHEMA = (
    "ofc_t3_t4_privacy_safe_counterfactual_private_type_batch/v1"
)
PRIVACY_SAFE_TRAVERSAL_DOMAIN = (
    "ofc-repository-owned-public-context-private-type-traversal"
)
PRIVACY_SAFE_TRAVERSAL_VERSION = "v1"
PRIVACY_SAFE_TRAVERSAL_ALGORITHM = (
    "sha256_public_context_domain_schedule_coprime_affine_full_cycle_v1"
)
TRAVERSAL_ALGORITHM = (
    "sha256_exact_randbelow_uniform_coprime_affine_full_cycle_v1"
)
RANKING_ALGORITHM = (
    "lexicographic_ordered_prior_discards_then_unordered_current_draw_v1"
)
PROPOSAL_SEMANTICS = (
    "unweighted_marginal_uniform_public_compatible_actor_private_type_proposal"
)
_PHYSICAL_JOKERS = ("X1", "X2")
_EXPECTED_UNDEALT_BY_PHASE = MappingProxyType(
    {
        "t3_first": 29,
        "t3_second": 26,
        "t4_first": 23,
        "t4_second": 20,
    }
)
_VALID_CARDS = frozenset(_encoding_module.ALL_CARDS)
_CANONICAL_DECK = tuple(sorted(_encoding_module.ALL_CARDS))
_SOURCE_PATHS = MappingProxyType(
    {
        "generator": str(Path(__file__).resolve()),
        "public_root_mixture_compiler": str(
            Path(__file__).with_name("t3_t4_public_root_mixture.py").resolve()
        ),
        "public_infoset": str(
            Path(__file__).with_name("t3_hu_public_cfr.py").resolve()
        ),
        "action_space": str(
            (Path(__file__).parents[1] / "engine" / "action_space.py").resolve()
        ),
        "deck_encoding": str(
            (Path(__file__).parents[1] / "engine" / "encoding.py").resolve()
        ),
        "turn_order": str(
            (Path(__file__).parents[1] / "engine" / "turn_order.py").resolve()
        ),
        "action_semantics": str(
            Path(__file__).with_name("exact_late.py").resolve()
        ),
        "runtime_semantic_graph": str(
            Path(__file__).with_name("t3_hu_multi_root_mccfr.py").resolve()
        ),
    }
)
_SEED_PLAN_CONSTRUCTION_TOKEN = object()
_PRIVACY_SAFE_BATCH_CONSTRUCTION_TOKEN = object()
PRIVACY_SAFE_PHASE_SCHEDULE = MappingProxyType(
    {
        "t3_first": ("t3_first_public_context_bootstrap_v1", 0, 2),
        "t3_second": ("t3_second_public_context_bootstrap_v1", 0, 2),
        "t4_first": ("t4_first_public_context_bootstrap_v1", 0, 2),
        "t4_second": ("t4_second_public_context_bootstrap_v1", 0, 2),
    }
)


class CounterfactualPrivateTypeError(ValueError):
    """A public proposal, its physical completion, or its binding is invalid."""


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


def _file_sha256(path: str | Path) -> str:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _live_source_sha256s() -> dict[str, str]:
    return {
        name: _file_sha256(path)
        for name, path in sorted(_SOURCE_PATHS.items())
    }


def _require_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CounterfactualPrivateTypeError(
            f"{label} must be a lowercase SHA256 digest"
        )
    return value


def _require_public_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise CounterfactualPrivateTypeError(
            f"{label} must be non-empty trimmed text"
        )
    if len(value) > 200 or any(ord(character) < 0x20 for character in value):
        raise CounterfactualPrivateTypeError(f"{label} is not canonical public text")
    return value


def _seed_reveal_payload(
    context_digest: str,
    *,
    seed_namespace: str,
    seed: int,
    seed_nonce: str,
) -> dict[str, Any]:
    _require_sha256(context_digest, label="seed reveal context_digest")
    namespace = _require_public_text(
        seed_namespace, label="seed reveal namespace"
    )
    nonce = _require_public_text(seed_nonce, label="seed reveal nonce")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    return {
        "schema": PUBLIC_SEED_REVEAL_SCHEMA,
        "public_context_digest": context_digest,
        "seed_namespace": namespace,
        "raw_seed_decimal": str(seed),
        "seed_nonce": nonce,
        "traversal_algorithm": TRAVERSAL_ALGORITHM,
        "ranking_algorithm": RANKING_ALGORITHM,
    }


def build_public_seed_reveal_commitment(
    context: PublicRootContext,
    *,
    seed_namespace: str,
    seed: int,
    seed_nonce: str,
) -> str:
    """Commit to the public namespace and hidden seed material before reveal."""

    if type(context) is not PublicRootContext:
        raise TypeError("context must be exact PublicRootContext")
    context.bindings.verify_live()
    return _canonical_sha256(
        _seed_reveal_payload(
            context.digest(),
            seed_namespace=seed_namespace,
            seed=seed,
            seed_nonce=seed_nonce,
        )
    )


@dataclass(frozen=True, slots=True)
class PreregisteredPublicSeedPlan:
    """Public pre-registration record; raw seed/reveal material is absent."""

    manifest_json: str = field(repr=False)
    plan_sha256: str

    def __post_init__(self) -> None:
        if type(self) is not PreregisteredPublicSeedPlan:
            raise TypeError("seed plan subclasses are forbidden")
        if not isinstance(self.manifest_json, str) or not self.manifest_json:
            raise TypeError("seed plan manifest_json must be non-empty text")
        _require_sha256(self.plan_sha256, label="seed plan SHA256")
        try:
            parsed = json.loads(self.manifest_json)
        except json.JSONDecodeError as exc:
            raise CounterfactualPrivateTypeError(
                "seed plan manifest is not JSON"
            ) from exc
        if not isinstance(parsed, dict):
            raise CounterfactualPrivateTypeError(
                "seed plan manifest must encode an object"
            )
        if _canonical_json(parsed) != self.manifest_json:
            raise CounterfactualPrivateTypeError(
                "seed plan manifest must use canonical JSON"
            )
        if hashlib.sha256(self.manifest_json.encode("utf-8")).hexdigest() != (
            self.plan_sha256
        ):
            raise CounterfactualPrivateTypeError("seed plan SHA256 mismatch")

    @property
    def manifest(self) -> Mapping[str, Any]:
        return MappingProxyType(json.loads(self.manifest_json))


def build_preregistered_public_seed_plan(
    context: PublicRootContext,
    *,
    seed_namespace: str,
    seed_reveal_commitment_sha256: str,
    sample_count: int,
    start_ordinal: int = 0,
    trusted_registry_id: str,
    registration_record_id: str,
) -> PreregisteredPublicSeedPlan:
    """Build the public plan that an external registry must approve.

    This function never accepts the raw seed or nonce.  The returned plan hash
    must be registered outside this process before proposal generation.  The
    generator later requires that external approved hash as an independent
    input and verifies the raw reveal against this plan.
    """

    if type(context) is not PublicRootContext:
        raise TypeError("context must be exact PublicRootContext")
    context.bindings.verify_live()
    namespace = _require_public_text(seed_namespace, label="seed_namespace")
    registry_id = _require_public_text(
        trusted_registry_id, label="trusted_registry_id"
    )
    record_id = _require_public_text(
        registration_record_id, label="registration_record_id"
    )
    reveal_commitment = _require_sha256(
        seed_reveal_commitment_sha256,
        label="seed_reveal_commitment_sha256",
    )
    universe = _universe_manifest(context)
    universe_size = int(universe["universe_size"])
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count <= 0
    ):
        raise ValueError("seed plan sample_count must be a positive integer")
    if (
        isinstance(start_ordinal, bool)
        or not isinstance(start_ordinal, int)
        or start_ordinal < 0
    ):
        raise ValueError("seed plan start_ordinal must be a non-negative integer")
    if start_ordinal + sample_count > universe_size:
        raise ValueError("seed plan slice exceeds one complete universe traversal")
    manifest = {
        "schema": PUBLIC_SEED_PLAN_SCHEMA,
        "public_context_digest": context.digest(),
        "public_universe_sha256": _canonical_sha256(universe),
        "seed_namespace": namespace,
        "namespace_source": (
            "external_public_literal_preregistered_before_private_observation"
        ),
        "seed_reveal_commitment_sha256": reveal_commitment,
        "start_ordinal": start_ordinal,
        "sample_count": sample_count,
        "slice_preregistered_before_seed_reveal": True,
        "traversal_algorithm": TRAVERSAL_ALGORITHM,
        "ranking_algorithm": RANKING_ALGORITHM,
        "trusted_registry_id": registry_id,
        "registration_record_id": record_id,
        "external_registry_approval_required": True,
        "hidden_state_inputs_forbidden": True,
        "actual_hand_inputs_forbidden": True,
        "raw_seed_serialized": False,
        "seed_nonce_serialized": False,
        "promotion_eligible": False,
    }
    manifest_json = _canonical_json(manifest)
    return PreregisteredPublicSeedPlan(
        manifest_json=manifest_json,
        plan_sha256=hashlib.sha256(manifest_json.encode("utf-8")).hexdigest(),
    )


def verify_preregistered_public_seed_plan(
    context: PublicRootContext,
    seed_plan: PreregisteredPublicSeedPlan,
    *,
    approved_seed_plan_sha256: str,
) -> Mapping[str, Any]:
    """Require exact plan semantics and an external trusted approved hash."""

    if type(context) is not PublicRootContext:
        raise TypeError("context must be exact PublicRootContext")
    if type(seed_plan) is not PreregisteredPublicSeedPlan:
        raise TypeError("seed_plan must be exact PreregisteredPublicSeedPlan")
    context.bindings.verify_live()
    approved = _require_sha256(
        approved_seed_plan_sha256,
        label="external approved seed plan SHA256",
    )
    if approved != seed_plan.plan_sha256:
        raise CounterfactualPrivateTypeError(
            "seed plan is not the externally approved pre-registration"
        )
    raw = dict(seed_plan.manifest)
    expected_keys = {
        "schema",
        "public_context_digest",
        "public_universe_sha256",
        "seed_namespace",
        "namespace_source",
        "seed_reveal_commitment_sha256",
        "start_ordinal",
        "sample_count",
        "slice_preregistered_before_seed_reveal",
        "traversal_algorithm",
        "ranking_algorithm",
        "trusted_registry_id",
        "registration_record_id",
        "external_registry_approval_required",
        "hidden_state_inputs_forbidden",
        "actual_hand_inputs_forbidden",
        "raw_seed_serialized",
        "seed_nonce_serialized",
        "promotion_eligible",
    }
    if set(raw) != expected_keys:
        raise CounterfactualPrivateTypeError("seed plan schema fields mismatch")
    if raw["schema"] != PUBLIC_SEED_PLAN_SCHEMA:
        raise CounterfactualPrivateTypeError("unsupported public seed plan schema")
    if raw["public_context_digest"] != context.digest():
        raise CounterfactualPrivateTypeError("seed plan context binding mismatch")
    universe_sha256 = _canonical_sha256(_universe_manifest(context))
    if raw["public_universe_sha256"] != universe_sha256:
        raise CounterfactualPrivateTypeError("seed plan universe binding mismatch")
    _require_public_text(raw["seed_namespace"], label="seed plan namespace")
    _require_public_text(
        raw["trusted_registry_id"], label="seed plan trusted_registry_id"
    )
    _require_public_text(
        raw["registration_record_id"],
        label="seed plan registration_record_id",
    )
    reveal_commitment = _require_sha256(
        raw["seed_reveal_commitment_sha256"],
        label="seed plan reveal commitment",
    )
    start_ordinal = raw["start_ordinal"]
    sample_count = raw["sample_count"]
    if (
        isinstance(start_ordinal, bool)
        or not isinstance(start_ordinal, int)
        or start_ordinal < 0
    ):
        raise CounterfactualPrivateTypeError(
            "seed plan start_ordinal must be a non-negative integer"
        )
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count <= 0
    ):
        raise CounterfactualPrivateTypeError(
            "seed plan sample_count must be a positive integer"
        )
    if start_ordinal + sample_count > int(
        _universe_manifest(context)["universe_size"]
    ):
        raise CounterfactualPrivateTypeError(
            "seed plan slice exceeds one complete universe traversal"
        )
    fixed = {
        "namespace_source": (
            "external_public_literal_preregistered_before_private_observation"
        ),
        "traversal_algorithm": TRAVERSAL_ALGORITHM,
        "ranking_algorithm": RANKING_ALGORITHM,
        "external_registry_approval_required": True,
        "hidden_state_inputs_forbidden": True,
        "actual_hand_inputs_forbidden": True,
        "slice_preregistered_before_seed_reveal": True,
        "raw_seed_serialized": False,
        "seed_nonce_serialized": False,
        "promotion_eligible": False,
    }
    for name, expected in fixed.items():
        if raw.get(name) != expected:
            raise CounterfactualPrivateTypeError(
                f"seed plan fixed field drifted: {name}"
            )
    return MappingProxyType(
        {
            "verified": True,
            "seed_plan_sha256": seed_plan.plan_sha256,
            "approved_seed_plan_sha256": approved,
            "seed_namespace": raw["seed_namespace"],
            "seed_reveal_commitment_sha256": reveal_commitment,
            "start_ordinal": start_ordinal,
            "sample_count": sample_count,
            "trusted_registry_id": raw["trusted_registry_id"],
            "registration_record_id": raw["registration_record_id"],
            "public_universe_sha256": universe_sha256,
            "promotion_eligible": False,
        }
    )


def _public_cards(context: PublicRootContext) -> tuple[str, ...]:
    cards = tuple(
        card
        for _turn, _actor, placements in context.public_action_history
        for card, _row in placements
    )
    if len(cards) != len(set(cards)):
        raise CounterfactualPrivateTypeError(
            "public context repeats a physical card"
        )
    if any(card not in _VALID_CARDS for card in cards):
        raise CounterfactualPrivateTypeError(
            "public context contains a non-canonical physical card"
        )
    return tuple(sorted(cards))


def _available_actor_private_pool(context: PublicRootContext) -> tuple[str, ...]:
    public = frozenset(_public_cards(context))
    pool = tuple(card for card in _CANONICAL_DECK if card not in public)
    if len(pool) + len(public) != len(_CANONICAL_DECK):
        raise AssertionError("public/private deck partition cardinality drifted")
    return pool


def _actor_prior_turns(context: PublicRootContext) -> tuple[int, ...]:
    turns = tuple(range(1, context.turn))
    actor_history_turns = tuple(
        turn
        for turn, actor, _placements in context.public_action_history
        if actor == context.actor and turn > 0
    )
    if actor_history_turns != turns:
        raise CounterfactualPrivateTypeError(
            "public cut does not contain every required prior actor turn"
        )
    return turns


def _universe_size(pool_size: int, discard_count: int) -> int:
    if pool_size - discard_count < 3:
        raise CounterfactualPrivateTypeError(
            "public cut leaves too few cards for a current draw"
        )
    return math.perm(pool_size, discard_count) * math.comb(
        pool_size - discard_count, 3
    )


def _unrank_permutation(
    values: Sequence[str], count: int, rank: int
) -> tuple[str, ...]:
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise TypeError("permutation count must be a non-negative integer")
    total = math.perm(len(values), count)
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise TypeError("permutation rank must be an integer")
    if not 0 <= rank < total:
        raise CounterfactualPrivateTypeError(
            f"permutation rank must be in [0,{total})"
        )
    available = list(values)
    selected: list[str] = []
    remainder = rank
    for position in range(count):
        suffix_count = count - position - 1
        block = math.perm(len(available) - 1, suffix_count)
        index, remainder = divmod(remainder, block)
        selected.append(available.pop(index))
    if remainder != 0:  # pragma: no cover - arithmetic defense
        raise AssertionError("permutation unranking left a non-zero remainder")
    return tuple(selected)


def _unrank_combination(
    values: Sequence[str], count: int, rank: int
) -> tuple[str, ...]:
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise TypeError("combination count must be a non-negative integer")
    total = math.comb(len(values), count)
    if isinstance(rank, bool) or not isinstance(rank, int):
        raise TypeError("combination rank must be an integer")
    if not 0 <= rank < total:
        raise CounterfactualPrivateTypeError(
            f"combination rank must be in [0,{total})"
        )
    selected: list[str] = []
    start = 0
    remainder = rank
    for position in range(count):
        needed_after = count - position - 1
        last_candidate = len(values) - needed_after
        for index in range(start, last_candidate):
            block = math.comb(len(values) - index - 1, needed_after)
            if remainder < block:
                selected.append(values[index])
                start = index + 1
                break
            remainder -= block
        else:  # pragma: no cover - arithmetic defense
            raise AssertionError("combination unranking exhausted its candidates")
    if remainder != 0:  # pragma: no cover - arithmetic defense
        raise AssertionError("combination unranking left a non-zero remainder")
    return tuple(selected)


def _decode_universe_rank(
    pool: tuple[str, ...], discard_count: int, universe_rank: int
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    total = _universe_size(len(pool), discard_count)
    if (
        isinstance(universe_rank, bool)
        or not isinstance(universe_rank, int)
        or not 0 <= universe_rank < total
    ):
        raise CounterfactualPrivateTypeError(
            f"private-type rank must be in [0,{total})"
        )
    draw_count = math.comb(len(pool) - discard_count, 3)
    discard_rank, draw_rank = divmod(universe_rank, draw_count)
    discards = _unrank_permutation(pool, discard_count, discard_rank)
    discard_set = frozenset(discards)
    draw_pool = tuple(card for card in pool if card not in discard_set)
    current_draw = _unrank_combination(draw_pool, 3, draw_rank)
    return discards, current_draw


def _hash_randbelow(upper: int, *, domain: bytes, counter: int) -> tuple[int, int]:
    """Exact deterministic randbelow using 256-bit rejection, never modulo bias."""

    if isinstance(upper, bool) or not isinstance(upper, int) or upper <= 0:
        raise ValueError("randbelow upper bound must be a positive integer")
    if isinstance(counter, bool) or not isinstance(counter, int) or counter < 0:
        raise ValueError("randbelow counter must be a non-negative integer")
    space = 1 << 256
    limit = space - (space % upper)
    next_counter = counter
    while True:
        payload = (
            b"ofc-counterfactual-private-type-randbelow-v1\x00"
            + domain
            + b"\x00"
            + next_counter.to_bytes(16, "big", signed=False)
        )
        value = int.from_bytes(hashlib.sha256(payload).digest(), "big")
        next_counter += 1
        if value < limit:
            return value % upper, next_counter


def _traversal_parameters(
    context_digest: str,
    universe_size: int,
    *,
    seed_namespace: str,
    seed: int,
    seed_nonce: str,
) -> tuple[int, int, str]:
    seed_payload = _seed_reveal_payload(
        context_digest,
        seed_namespace=seed_namespace,
        seed=seed,
        seed_nonce=seed_nonce,
    )
    seed_commitment = _canonical_sha256(seed_payload)
    domain = _canonical_json(seed_payload).encode("utf-8")
    offset, counter = _hash_randbelow(
        universe_size,
        domain=b"offset\x00" + domain,
        counter=0,
    )
    if universe_size == 1:  # impossible for the supported T3/T4 cuts
        return offset, 1, seed_commitment
    step_counter = counter
    while True:
        candidate, step_counter = _hash_randbelow(
            universe_size - 1,
            domain=b"step\x00" + domain,
            counter=step_counter,
        )
        step = candidate + 1
        if math.gcd(step, universe_size) == 1:
            return offset, step, seed_commitment


def _private_recall(
    context: PublicRootContext, discards: tuple[str, ...]
) -> PrivateRecall:
    prior_turns = _actor_prior_turns(context)
    if len(discards) != len(prior_turns):
        raise CounterfactualPrivateTypeError(
            "discard assignment does not match prior actor turn count"
        )
    public_by_turn = {
        turn: tuple(card for card, _row in placements)
        for turn, actor, placements in context.public_action_history
        if actor == context.actor and turn > 0
    }
    return PrivateRecall(
        dealt_by_turn=tuple(
            (turn, (*public_by_turn[turn], discard))
            for turn, discard in zip(prior_turns, discards)
        ),
        discards_by_turn=tuple(zip(prior_turns, discards)),
    )


def _opponent_completion_recall(
    context: PublicRootContext,
    available_hidden_cards: Sequence[str],
) -> tuple[PrivateRecall, tuple[str, ...]]:
    opponent = "btn" if context.actor == "bb" else "bb"
    public_turns = tuple(
        (turn, tuple(card for card, _row in placements))
        for turn, actor, placements in context.public_action_history
        if actor == opponent and turn > 0
    )
    if len(available_hidden_cards) < len(public_turns):
        raise CounterfactualPrivateTypeError(
            "private proposal leaves no physical opponent completion"
        )
    opponent_discards = tuple(available_hidden_cards[: len(public_turns)])
    recall = PrivateRecall(
        dealt_by_turn=tuple(
            (turn, (*placements, discard))
            for (turn, placements), discard in zip(
                public_turns, opponent_discards
            )
        ),
        discards_by_turn=tuple(
            (turn, discard)
            for (turn, _placements), discard in zip(
                public_turns, opponent_discards
            )
        ),
    )
    return recall, opponent_discards


def _observation_and_physical_witness(
    context: PublicRootContext,
    discards: tuple[str, ...],
    current_draw: tuple[str, ...],
) -> tuple[InfoSetKey, JointParticle]:
    own_recall = _private_recall(context, discards)
    public = frozenset(_public_cards(context))
    actor_hidden = frozenset((*discards, *current_draw))
    if len(actor_hidden) != len(discards) + 3:
        raise CounterfactualPrivateTypeError(
            "counterfactual actor type repeats a physical card"
        )
    if actor_hidden & public:
        raise CounterfactualPrivateTypeError(
            "counterfactual actor type overlaps a public card"
        )
    remainder = tuple(
        card
        for card in _CANONICAL_DECK
        if card not in public and card not in actor_hidden
    )
    opponent_recall, opponent_discards = _opponent_completion_recall(
        context, remainder
    )
    opponent_discard_set = frozenset(opponent_discards)
    undealt = tuple(card for card in remainder if card not in opponent_discard_set)
    if context.actor == "bb":
        particle = JointParticle(
            bb_recall=own_recall,
            btn_recall=opponent_recall,
            undealt_cards=undealt,
            weight=Fraction(1, 1),
        )
    else:
        particle = JointParticle(
            bb_recall=opponent_recall,
            btn_recall=own_recall,
            undealt_cards=undealt,
            weight=Fraction(1, 1),
        )
    observation = InfoSetKey.for_particle(
        particle,
        contract_version=context.contract_version,
        actor=context.actor,  # type: ignore[arg-type]
        turn=context.turn,
        phase=context.phase,
        board_bb=context.board_bb,
        board_btn=context.board_btn,
        public_action_history=context.public_action_history,
        current_draw=current_draw,
        fantasy_state=context.fantasy_state,
    )
    partition = Counter(
        (
            *public,
            *discards,
            *opponent_discards,
            *current_draw,
            *undealt,
        )
    )
    if partition != Counter(_CANONICAL_DECK):
        raise AssertionError("counterfactual physical witness is not a 54-card partition")
    return observation, particle


def _private_type_commitment(
    context_digest: str, observation: InfoSetKey
) -> str:
    # This intentionally matches the existing explicit public-root compiler's
    # commitment semantics so a later posterior/range join can bind the same
    # actor type without exposing its cards in either audit manifest.
    return _canonical_sha256(
        {
            "schema": "ofc_hypothetical_actor_private_type_commitment/v1",
            "public_context_digest": context_digest,
            "actor": observation.actor,
            "phase": observation.phase,
            "own_recall": observation.own_recall.to_canonical_dict(),
            "current_draw": list(observation.current_draw),
            "physical_joker_ids": list(_PHYSICAL_JOKERS),
            "joker_physical_identity_collapsed": False,
        }
    )


@dataclass(frozen=True, slots=True)
class CounterfactualPrivateTypeProposal:
    """One generated hypothetical type; raw private payload stays out of audit."""

    sample_ordinal: int
    private_type_commitment: str
    observation: InfoSetKey = field(repr=False)
    universe_rank: int = field(repr=False)
    physical_completion_witness: JointParticle = field(repr=False)

    def __post_init__(self) -> None:
        if type(self) is not CounterfactualPrivateTypeProposal:
            raise TypeError("proposal subclasses are forbidden")
        if (
            isinstance(self.sample_ordinal, bool)
            or not isinstance(self.sample_ordinal, int)
            or self.sample_ordinal < 0
        ):
            raise ValueError("sample_ordinal must be a non-negative integer")
        if (
            isinstance(self.universe_rank, bool)
            or not isinstance(self.universe_rank, int)
            or self.universe_rank < 0
        ):
            raise ValueError("universe_rank must be a non-negative integer")
        _require_sha256(
            self.private_type_commitment,
            label="private_type_commitment",
        )
        if type(self.observation) is not InfoSetKey:
            raise TypeError("proposal observation must be exact InfoSetKey")
        if type(self.physical_completion_witness) is not JointParticle:
            raise TypeError("physical completion witness must be exact JointParticle")


@dataclass(frozen=True, slots=True)
class CounterfactualPrivateTypeProposalBatch:
    """Bounded unweighted sample plus a private-payload-free audit envelope."""

    context_digest: str
    proposals: tuple[CounterfactualPrivateTypeProposal, ...] = field(repr=False)
    universe_size: int
    start_ordinal: int
    sample_count: int
    seed_plan: PreregisteredPublicSeedPlan = field(repr=False)
    approved_seed_plan_sha256: str = field(repr=False)
    seed: int = field(repr=False)
    seed_nonce: str = field(repr=False)
    audit_manifest_json: str = field(repr=False)
    audit_manifest_sha256: str

    def __post_init__(self) -> None:
        if type(self) is not CounterfactualPrivateTypeProposalBatch:
            raise TypeError("proposal batch subclasses are forbidden")
        _require_sha256(self.context_digest, label="batch context_digest")
        if not isinstance(self.proposals, tuple) or any(
            type(item) is not CounterfactualPrivateTypeProposal
            for item in self.proposals
        ):
            raise TypeError("batch proposals must be an exact proposal tuple")
        for name, value, positive in (
            ("universe_size", self.universe_size, True),
            ("start_ordinal", self.start_ordinal, False),
            ("sample_count", self.sample_count, True),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or (value <= 0 if positive else value < 0)
            ):
                qualifier = "positive" if positive else "non-negative"
                raise ValueError(f"batch {name} must be a {qualifier} integer")
        if len(self.proposals) != self.sample_count:
            raise ValueError("batch proposal count must equal sample_count")
        if self.start_ordinal + self.sample_count > self.universe_size:
            raise ValueError("batch slice exceeds its universe")
        if type(self.seed_plan) is not PreregisteredPublicSeedPlan:
            raise TypeError("batch seed_plan must be exact PreregisteredPublicSeedPlan")
        _require_sha256(
            self.approved_seed_plan_sha256,
            label="batch approved_seed_plan_sha256",
        )
        if self.approved_seed_plan_sha256 != self.seed_plan.plan_sha256:
            raise ValueError("batch seed plan is not externally approved")
        seed_manifest = self.seed_plan.manifest
        if (
            seed_manifest.get("start_ordinal") != self.start_ordinal
            or seed_manifest.get("sample_count") != self.sample_count
        ):
            raise ValueError(
                "batch slice does not match the externally approved seed plan"
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("batch seed must be an integer")
        _require_public_text(self.seed_nonce, label="batch seed_nonce")
        if not isinstance(self.audit_manifest_json, str) or not self.audit_manifest_json:
            raise TypeError("batch audit_manifest_json must be non-empty text")
        _require_sha256(
            self.audit_manifest_sha256, label="batch audit_manifest_sha256"
        )
        try:
            parsed = json.loads(self.audit_manifest_json)
        except json.JSONDecodeError as exc:
            raise CounterfactualPrivateTypeError(
                "batch audit manifest is not JSON"
            ) from exc
        if not isinstance(parsed, dict) or _canonical_json(parsed) != (
            self.audit_manifest_json
        ):
            raise CounterfactualPrivateTypeError(
                "batch audit manifest must be a canonical JSON object"
            )
        if hashlib.sha256(self.audit_manifest_json.encode("utf-8")).hexdigest() != (
            self.audit_manifest_sha256
        ):
            raise CounterfactualPrivateTypeError(
                "batch audit manifest SHA256 mismatch"
            )

    @property
    def audit_manifest(self) -> Mapping[str, Any]:
        return MappingProxyType(json.loads(self.audit_manifest_json))

    @property
    def promotion_eligible(self) -> bool:
        return False

    @property
    def production_sampling_ready(self) -> bool:
        return False

    def __iter__(self) -> Iterator[CounterfactualPrivateTypeProposal]:
        return iter(self.proposals)


def _universe_manifest(context: PublicRootContext) -> dict[str, Any]:
    public_cards = _public_cards(context)
    pool = _available_actor_private_pool(context)
    prior_turns = _actor_prior_turns(context)
    total = _universe_size(len(pool), len(prior_turns))
    opponent = "btn" if context.actor == "bb" else "bb"
    opponent_completed_turns = tuple(
        turn
        for turn, actor, _placements in context.public_action_history
        if actor == opponent and turn > 0
    )
    guaranteed_undealt_count = (
        len(pool) - len(prior_turns) - 3 - len(opponent_completed_turns)
    )
    expected_undealt_count = _EXPECTED_UNDEALT_BY_PHASE[context.phase]
    if guaranteed_undealt_count != expected_undealt_count:
        raise CounterfactualPrivateTypeError(
            "closed-form physical completion count does not match phase"
        )
    deck_manifest = {
        "schema": "ofc_physical_deck_manifest/v1",
        "cards": list(_CANONICAL_DECK),
        "deck_size": len(_CANONICAL_DECK),
        "physical_joker_ids": list(_PHYSICAL_JOKERS),
        "joker_physical_identity_collapsed": False,
    }
    return {
        "schema": UNIVERSE_SCHEMA,
        "public_context_digest": context.digest(),
        "actor": context.actor,
        "turn": context.turn,
        "phase": context.phase,
        "public_card_count": len(public_cards),
        "available_actor_private_pool_size": len(pool),
        "ordered_prior_discard_turns": list(prior_turns),
        "ordered_prior_discard_count": len(prior_turns),
        "unordered_current_draw_count": 3,
        "universe_size": total,
        "universe_size_formula": (
            f"P({len(pool)},{len(prior_turns)})*"
            f"C({len(pool) - len(prior_turns)},3)"
        ),
        "ranking_algorithm": RANKING_ALGORITHM,
        "complete_publicly_compatible_universe_defined": True,
        "every_rank_has_one_actor_private_type": True,
        "every_actor_private_type_has_one_rank": True,
        "physical_completion_exists_for_every_rank": True,
        "closed_form_physical_completion_proof": {
            "actor_prior_discard_count": len(prior_turns),
            "actor_current_draw_count": 3,
            "opponent_completed_discard_turns": list(
                opponent_completed_turns
            ),
            "opponent_completed_discard_count": len(
                opponent_completed_turns
            ),
            "guaranteed_undealt_count": guaranteed_undealt_count,
            "phase_expected_undealt_count": expected_undealt_count,
            "partition_card_count_equation": (
                f"{len(public_cards)}+{len(prior_turns)}+3+"
                f"{len(opponent_completed_turns)}+{guaranteed_undealt_count}=54"
            ),
            "completion_algorithm": (
                "lexicographically_assign_opponent_discards_then_remainder_undealt"
            ),
        },
        "deck_manifest_sha256": _canonical_sha256(deck_manifest),
        "deck_manifest": deck_manifest,
    }


def _audit_manifest(
    context: PublicRootContext,
    proposals: Sequence[CounterfactualPrivateTypeProposal],
    *,
    universe: Mapping[str, Any],
    start_ordinal: int,
    sample_count: int,
    seed_commitment: str,
    seed_plan_verification: Mapping[str, Any],
    runtime_semantic_binding_sha256: str,
) -> dict[str, Any]:
    commitments = [item.private_type_commitment for item in proposals]
    universe_size = int(universe["universe_size"])
    coverage = Fraction(sample_count, universe_size)
    source_sha256s = _live_source_sha256s()
    context_sources = dict(context.bindings.source_sha256s)
    expected_context_hashes = {
        "public_root_mixture_compiler": source_sha256s[
            "public_root_mixture_compiler"
        ],
        "public_infoset": source_sha256s["public_infoset"],
        "action_space": source_sha256s["action_space"],
        "deck_encoding": source_sha256s["deck_encoding"],
        "turn_order": source_sha256s["turn_order"],
        "terminal_scoring": source_sha256s["action_semantics"],
    }
    for name, source_hash in expected_context_hashes.items():
        if context_sources.get(name) != source_hash:
            raise CounterfactualPrivateTypeError(
                f"public context source binding does not match generator source {name}"
            )
    return {
        "schema": COUNTERFACTUAL_PRIVATE_TYPE_BATCH_SCHEMA,
        "artifact_kind": "unweighted_actor_private_type_proposal_support_only",
        "proposal_schema": COUNTERFACTUAL_PRIVATE_TYPE_SCHEMA,
        "public_context": context.to_canonical_dict(),
        "public_context_digest": context.digest(),
        "universe": dict(universe),
        "universe_sha256": _canonical_sha256(dict(universe)),
        "traversal": {
            "algorithm": TRAVERSAL_ALGORITHM,
            "seed_reveal_commitment_sha256": seed_commitment,
            "seed_plan_sha256": seed_plan_verification["seed_plan_sha256"],
            "external_approved_seed_plan_sha256": seed_plan_verification[
                "approved_seed_plan_sha256"
            ],
            "trusted_seed_registry_id": seed_plan_verification[
                "trusted_registry_id"
            ],
            "seed_registration_record_id": seed_plan_verification[
                "registration_record_id"
            ],
            "seed_namespace": seed_plan_verification["seed_namespace"],
            "seed_namespace_preregistered_publicly": True,
            "slice_preregistered_before_seed_reveal": True,
            "planned_start_ordinal": seed_plan_verification["start_ordinal"],
            "planned_sample_count": seed_plan_verification["sample_count"],
            "seed_plan_verified_before_traversal": True,
            "seed_material_serialized": False,
            "external_seed_material_required_for_replay": True,
            "start_ordinal": start_ordinal,
            "sample_count": sample_count,
            "without_replacement": True,
            "full_cycle_bijective": True,
            "sampled_support_exhaustive": (
                start_ordinal == 0 and sample_count == universe_size
            ),
            "sampled_support_size": sample_count,
            "universe_size": universe_size,
            "coverage_fraction_exact": (
                f"{coverage.numerator}/{coverage.denominator}"
            ),
            "marginal_inclusion_probability_exact": (
                f"{coverage.numerator}/{coverage.denominator}"
            ),
            "proposal_semantics": PROPOSAL_SEMANTICS,
            "sample_commitments": commitments,
            "sample_commitment_order_sha256": _canonical_sha256(commitments),
            "sample_commitments_unique": len(set(commitments)) == sample_count,
            "raw_universe_ranks_serialized": False,
            "raw_private_cards_serialized": False,
        },
        "content_bindings": {
            "generator_source_sha256": source_sha256s["generator"],
            "public_root_mixture_source_sha256": source_sha256s[
                "public_root_mixture_compiler"
            ],
            "public_infoset_source_sha256": source_sha256s["public_infoset"],
            "action_space_source_sha256": source_sha256s["action_space"],
            "deck_encoding_source_sha256": source_sha256s["deck_encoding"],
            "turn_order_source_sha256": source_sha256s["turn_order"],
            "action_semantics_source_sha256": source_sha256s[
                "action_semantics"
            ],
            "runtime_semantic_graph_source_sha256": source_sha256s[
                "runtime_semantic_graph"
            ],
            "generator_runtime_semantic_binding_sha256": (
                runtime_semantic_binding_sha256
            ),
            "rules_sha256": context.bindings.rules_sha256,
            "public_context_source_set_sha256": _canonical_sha256(
                context_sources
            ),
            "expected_behavior_model_id": (
                context.bindings.behavior_model_id
            ),
            "expected_behavior_model_sha256": (
                context.bindings.behavior_model_sha256
            ),
        },
        "physical_validation": {
            "canonical_deck_size": len(_CANONICAL_DECK),
            "physical_joker_ids": list(_PHYSICAL_JOKERS),
            "joker_physical_identity_collapsed": False,
            "every_emitted_type_has_full_54_card_completion_witness": True,
            "every_emitted_observation_validated_by_InfoSetKey_for_particle": True,
        },
        "information_safety": {
            "actual_actor_private_cards_parameter_exists": False,
            "actual_actor_private_cards_consumed": False,
            "public_context_contains_actor_private_fields": False,
            "private_payloads_committed_not_serialized": True,
        },
        "posterior_boundary": {
            "uniform_proposal_is_posterior": False,
            "actor_history_behavior_likelihood_applied": False,
            "opponent_conditional_range_applied": False,
            "calibrated_behavior_artifact_bound": False,
            "conditional_range_artifacts_bound": False,
            "posterior_normalization_applied": False,
            "authorized_as_mccfr_prior": False,
            "required_downstream_join": (
                "fresh_verified_promoted_behavior_likelihood_and_conditional_range"
            ),
        },
        "promotion_eligible": False,
        "production_sampling_ready": False,
        "serving_default_changed": False,
        "remaining_before_production_sampling": [
            "join_each_commitment_to_fresh_calibrated_actor_history_likelihood",
            "join_each_commitment_to_fresh_verified_opponent_conditional_range",
            "derive_and_verify_posterior_super_root_mass_without_actual_hand_conditioning",
            "pass_independent_online_solver_strength_and_uncertainty_gates",
        ],
    }


def build_counterfactual_private_type_proposals(
    context: PublicRootContext,
    *,
    sample_count: int,
    seed_plan: PreregisteredPublicSeedPlan,
    approved_seed_plan_sha256: str,
    seed: int,
    seed_nonce: str,
    start_ordinal: int = 0,
) -> CounterfactualPrivateTypeProposalBatch:
    """Generate a bounded public-only proposal support.

    Deliberately absent from the signature: the actor's actual recall, actual
    current draw, actual private cards, opponent private cards, and any
    actual-type selector.  ``sample_count`` is bounded by one full traversal so
    the emitted support cannot contain duplicates.
    """

    if type(context) is not PublicRootContext:
        raise TypeError("context must be exact PublicRootContext")
    context.bindings.verify_live()
    seed_plan_verification = verify_preregistered_public_seed_plan(
        context,
        seed_plan,
        approved_seed_plan_sha256=approved_seed_plan_sha256,
    )
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count <= 0
    ):
        raise ValueError("sample_count must be a positive integer")
    if (
        isinstance(start_ordinal, bool)
        or not isinstance(start_ordinal, int)
        or start_ordinal < 0
    ):
        raise ValueError("start_ordinal must be a non-negative integer")
    universe = _universe_manifest(context)
    universe_size = int(universe["universe_size"])
    if start_ordinal + sample_count > universe_size:
        raise ValueError(
            "requested proposal slice exceeds one duplicate-free universe traversal"
        )
    if (
        seed_plan_verification["start_ordinal"] != start_ordinal
        or seed_plan_verification["sample_count"] != sample_count
    ):
        raise CounterfactualPrivateTypeError(
            "requested proposal slice was not externally pre-registered"
        )
    offset, step, seed_commitment = _traversal_parameters(
        context.digest(),
        universe_size,
        seed_namespace=str(seed_plan_verification["seed_namespace"]),
        seed=seed,
        seed_nonce=seed_nonce,
    )
    if seed_commitment != seed_plan_verification[
        "seed_reveal_commitment_sha256"
    ]:
        raise CounterfactualPrivateTypeError(
            "raw seed reveal does not match the pre-registered public seed plan"
        )
    pool = _available_actor_private_pool(context)
    discard_count = len(_actor_prior_turns(context))
    proposals: list[CounterfactualPrivateTypeProposal] = []
    for sample_ordinal in range(
        start_ordinal, start_ordinal + sample_count
    ):
        universe_rank = (offset + step * sample_ordinal) % universe_size
        discards, current_draw = _decode_universe_rank(
            pool, discard_count, universe_rank
        )
        observation, witness = _observation_and_physical_witness(
            context, discards, current_draw
        )
        commitment = _private_type_commitment(context.digest(), observation)
        proposals.append(
            CounterfactualPrivateTypeProposal(
                sample_ordinal=sample_ordinal,
                private_type_commitment=commitment,
                observation=observation,
                universe_rank=universe_rank,
                physical_completion_witness=witness,
            )
        )
    if len({item.universe_rank for item in proposals}) != sample_count:
        raise AssertionError("affine traversal emitted duplicate universe ranks")
    if len({item.private_type_commitment for item in proposals}) != sample_count:
        raise AssertionError("distinct universe ranks emitted duplicate private types")
    manifest = _audit_manifest(
        context,
        proposals,
        universe=universe,
        start_ordinal=start_ordinal,
        sample_count=sample_count,
        seed_commitment=seed_commitment,
        seed_plan_verification=seed_plan_verification,
        runtime_semantic_binding_sha256=_require_runtime_integrity(),
    )
    manifest_json = _canonical_json(manifest)
    result = CounterfactualPrivateTypeProposalBatch(
        context_digest=context.digest(),
        proposals=tuple(proposals),
        universe_size=universe_size,
        start_ordinal=start_ordinal,
        sample_count=sample_count,
        seed_plan=seed_plan,
        approved_seed_plan_sha256=approved_seed_plan_sha256,
        seed=seed,
        seed_nonce=seed_nonce,
        audit_manifest_json=manifest_json,
        audit_manifest_sha256=hashlib.sha256(
            manifest_json.encode("utf-8")
        ).hexdigest(),
    )
    verify_counterfactual_private_type_proposals(context, result)
    return result


def verify_counterfactual_private_type_proposals(
    context: PublicRootContext,
    result: CounterfactualPrivateTypeProposalBatch,
) -> Mapping[str, Any]:
    """Freshly replay generation and require byte-semantic artifact equality."""

    if type(context) is not PublicRootContext:
        raise TypeError("context must be exact PublicRootContext")
    if type(result) is not CounterfactualPrivateTypeProposalBatch:
        raise TypeError("result must be exact CounterfactualPrivateTypeProposalBatch")
    context.bindings.verify_live()
    if (
        isinstance(result.sample_count, bool)
        or not isinstance(result.sample_count, int)
        or result.sample_count <= 0
    ):
        raise CounterfactualPrivateTypeError(
            "proposal batch sample_count must be positive"
        )
    if (
        isinstance(result.start_ordinal, bool)
        or not isinstance(result.start_ordinal, int)
        or result.start_ordinal < 0
    ):
        raise CounterfactualPrivateTypeError(
            "proposal batch start_ordinal must be non-negative"
        )
    if any(
        type(item) is not CounterfactualPrivateTypeProposal
        for item in result.proposals
    ):
        raise TypeError("proposal batch contains a non-exact proposal type")
    seed_plan_verification = verify_preregistered_public_seed_plan(
        context,
        result.seed_plan,
        approved_seed_plan_sha256=result.approved_seed_plan_sha256,
    )
    if result.context_digest != context.digest():
        raise CounterfactualPrivateTypeError(
            "proposal batch public context digest mismatch"
        )
    if len(result.proposals) != result.sample_count:
        raise CounterfactualPrivateTypeError(
            "proposal tuple length does not match sample_count"
        )
    # Replay directly instead of calling the public builder recursively.
    universe = _universe_manifest(context)
    universe_size = int(universe["universe_size"])
    if result.universe_size != universe_size:
        raise CounterfactualPrivateTypeError("proposal universe size mismatch")
    if result.start_ordinal + result.sample_count > universe_size:
        raise CounterfactualPrivateTypeError(
            "proposal slice exceeds one universe traversal"
        )
    if (
        seed_plan_verification["start_ordinal"] != result.start_ordinal
        or seed_plan_verification["sample_count"] != result.sample_count
    ):
        raise CounterfactualPrivateTypeError(
            "proposal batch slice does not match its externally approved seed plan"
        )
    offset, step, seed_commitment = _traversal_parameters(
        context.digest(),
        universe_size,
        seed_namespace=str(seed_plan_verification["seed_namespace"]),
        seed=result.seed,
        seed_nonce=result.seed_nonce,
    )
    if seed_commitment != seed_plan_verification[
        "seed_reveal_commitment_sha256"
    ]:
        raise CounterfactualPrivateTypeError(
            "batch seed reveal does not match its pre-registered seed plan"
        )
    pool = _available_actor_private_pool(context)
    discard_count = len(_actor_prior_turns(context))
    expected: list[CounterfactualPrivateTypeProposal] = []
    for sample_ordinal in range(
        result.start_ordinal, result.start_ordinal + result.sample_count
    ):
        universe_rank = (offset + step * sample_ordinal) % universe_size
        discards, current_draw = _decode_universe_rank(
            pool, discard_count, universe_rank
        )
        observation, witness = _observation_and_physical_witness(
            context, discards, current_draw
        )
        expected.append(
            CounterfactualPrivateTypeProposal(
                sample_ordinal=sample_ordinal,
                private_type_commitment=_private_type_commitment(
                    context.digest(), observation
                ),
                observation=observation,
                universe_rank=universe_rank,
                physical_completion_witness=witness,
            )
        )
    if tuple(expected) != result.proposals:
        raise CounterfactualPrivateTypeError(
            "proposal contents do not match deterministic public-only replay"
        )
    expected_manifest = _audit_manifest(
        context,
        expected,
        universe=universe,
        start_ordinal=result.start_ordinal,
        sample_count=result.sample_count,
        seed_commitment=seed_commitment,
        seed_plan_verification=seed_plan_verification,
        runtime_semantic_binding_sha256=_require_runtime_integrity(),
    )
    expected_json = _canonical_json(expected_manifest)
    if result.audit_manifest_json != expected_json:
        raise CounterfactualPrivateTypeError(
            "proposal audit manifest content mismatch"
        )
    expected_sha256 = hashlib.sha256(expected_json.encode("utf-8")).hexdigest()
    if result.audit_manifest_sha256 != expected_sha256:
        raise CounterfactualPrivateTypeError(
            "proposal audit manifest SHA256 mismatch"
        )
    signatures = {
        "build": tuple(
            inspect.signature(build_counterfactual_private_type_proposals).parameters
        ),
        "verify": tuple(
            inspect.signature(verify_counterfactual_private_type_proposals).parameters
        ),
    }
    forbidden = {
        "actual_actor_private_cards",
        "actual_private_cards",
        "actor_private_cards",
        "actual_hand",
        "own_recall",
        "current_draw",
    }
    if any(forbidden.intersection(parameters) for parameters in signatures.values()):
        raise AssertionError("public proposal API acquired an actor-private input")
    return MappingProxyType(
        {
            "verified": True,
            "public_context_digest": context.digest(),
            "universe_size": universe_size,
            "sample_count": result.sample_count,
            "coverage_fraction_exact": expected_manifest["traversal"][
                "coverage_fraction_exact"
            ],
            "sampled_support_exhaustive": expected_manifest["traversal"][
                "sampled_support_exhaustive"
            ],
            "audit_manifest_sha256": expected_sha256,
            "posterior_weighted": False,
            "authorized_as_mccfr_prior": False,
            "promotion_eligible": False,
            "production_sampling_ready": False,
        }
    )


def _privacy_safe_traversal_parameters(
    context: PublicRootContext,
    universe: Mapping[str, Any],
) -> tuple[int, int, str, str, int, int]:
    try:
        schedule_id, start_ordinal, sample_count = PRIVACY_SAFE_PHASE_SCHEDULE[
            context.phase
        ]
    except KeyError as exc:  # pragma: no cover - PublicRootContext defense
        raise CounterfactualPrivateTypeError(
            f"no repository-owned privacy-safe schedule for {context.phase}"
        ) from exc
    universe_size = int(universe["universe_size"])
    if start_ordinal + sample_count > universe_size:
        raise AssertionError("privacy-safe phase schedule exceeds its universe")
    payload = {
        "schema": "ofc_t3_t4_privacy_safe_traversal_derivation/v1",
        "domain": PRIVACY_SAFE_TRAVERSAL_DOMAIN,
        "version": PRIVACY_SAFE_TRAVERSAL_VERSION,
        "algorithm": PRIVACY_SAFE_TRAVERSAL_ALGORITHM,
        "ranking_algorithm": RANKING_ALGORITHM,
        "public_context_digest": context.digest(),
        "public_universe_sha256": _canonical_sha256(dict(universe)),
        "phase": context.phase,
        "schedule_id": schedule_id,
        "start_ordinal": start_ordinal,
        "sample_count": sample_count,
    }
    derivation_commitment = _canonical_sha256(payload)
    domain = _canonical_json(payload).encode("utf-8")
    offset, counter = _hash_randbelow(
        universe_size,
        domain=b"privacy-safe-offset\x00" + domain,
        counter=0,
    )
    if universe_size == 1:  # pragma: no cover - supported universes are larger
        return offset, 1, derivation_commitment, schedule_id, start_ordinal, sample_count
    while True:
        step, counter = _hash_randbelow(
            universe_size - 1,
            domain=b"privacy-safe-step\x00" + domain,
            counter=counter,
        )
        step += 1
        if math.gcd(step, universe_size) == 1:
            return (
                offset,
                step,
                derivation_commitment,
                schedule_id,
                start_ordinal,
                sample_count,
            )


@dataclass(frozen=True, slots=True)
class PrivacySafeCounterfactualPrivateTypeProposalBatch:
    """Repository-derived proposal origin with no caller entropy or slice."""

    context_digest: str
    proposals: tuple[CounterfactualPrivateTypeProposal, ...] = field(repr=False)
    universe_size: int
    schedule_id: str
    start_ordinal: int
    sample_count: int
    traversal_derivation_commitment_sha256: str
    audit_manifest_json: str = field(repr=False)
    audit_manifest_sha256: str
    _construction_token: InitVar[object | None] = None

    def __post_init__(self, _construction_token: object | None) -> None:
        if type(self) is not PrivacySafeCounterfactualPrivateTypeProposalBatch:
            raise TypeError("privacy-safe batch subclasses are forbidden")
        if _construction_token is not _PRIVACY_SAFE_BATCH_CONSTRUCTION_TOKEN:
            raise TypeError(
                "use build_privacy_safe_counterfactual_private_type_proposals"
            )
        _require_sha256(self.context_digest, label="privacy-safe context_digest")
        if not isinstance(self.proposals, tuple) or any(
            type(item) is not CounterfactualPrivateTypeProposal
            for item in self.proposals
        ):
            raise TypeError("privacy-safe proposals must be an exact proposal tuple")
        if (
            isinstance(self.universe_size, bool)
            or not isinstance(self.universe_size, int)
            or self.universe_size <= 0
        ):
            raise ValueError("privacy-safe universe_size must be positive")
        _require_public_text(self.schedule_id, label="privacy-safe schedule_id")
        for name, value, positive in (
            ("start_ordinal", self.start_ordinal, False),
            ("sample_count", self.sample_count, True),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or (value <= 0 if positive else value < 0)
            ):
                raise ValueError(f"privacy-safe {name} is invalid")
        if len(self.proposals) != self.sample_count:
            raise ValueError("privacy-safe proposal count mismatch")
        if self.start_ordinal + self.sample_count > self.universe_size:
            raise ValueError("privacy-safe slice exceeds its universe")
        _require_sha256(
            self.traversal_derivation_commitment_sha256,
            label="privacy-safe traversal derivation commitment",
        )
        if not isinstance(self.audit_manifest_json, str):
            raise TypeError("privacy-safe audit_manifest_json must be text")
        try:
            manifest = json.loads(self.audit_manifest_json)
        except json.JSONDecodeError as exc:
            raise CounterfactualPrivateTypeError(
                "privacy-safe audit manifest is not JSON"
            ) from exc
        if not isinstance(manifest, dict) or _canonical_json(manifest) != (
            self.audit_manifest_json
        ):
            raise CounterfactualPrivateTypeError(
                "privacy-safe audit manifest must be canonical JSON"
            )
        _require_sha256(
            self.audit_manifest_sha256,
            label="privacy-safe audit manifest SHA256",
        )
        if hashlib.sha256(self.audit_manifest_json.encode("utf-8")).hexdigest() != (
            self.audit_manifest_sha256
        ):
            raise CounterfactualPrivateTypeError(
                "privacy-safe audit manifest SHA256 mismatch"
            )

    @property
    def audit_manifest(self) -> Mapping[str, Any]:
        return MappingProxyType(json.loads(self.audit_manifest_json))

    @property
    def authorized_as_mccfr_prior(self) -> bool:
        return False

    @property
    def eligible_as_posterior_join_origin(self) -> bool:
        return True

    @property
    def production_sampling_ready(self) -> bool:
        return False

    @property
    def promotion_eligible(self) -> bool:
        return False


def _privacy_safe_audit_manifest(
    context: PublicRootContext,
    proposals: Sequence[CounterfactualPrivateTypeProposal],
    *,
    universe: Mapping[str, Any],
    schedule_id: str,
    start_ordinal: int,
    sample_count: int,
    traversal_derivation_commitment_sha256: str,
    runtime_semantic_binding_sha256: str,
) -> dict[str, Any]:
    universe_size = int(universe["universe_size"])
    coverage = Fraction(sample_count, universe_size)
    commitments = [proposal.private_type_commitment for proposal in proposals]
    source_sha256s = _live_source_sha256s()
    context_sources = dict(context.bindings.source_sha256s)
    expected_context_hashes = {
        "public_root_mixture_compiler": source_sha256s[
            "public_root_mixture_compiler"
        ],
        "public_infoset": source_sha256s["public_infoset"],
        "action_space": source_sha256s["action_space"],
        "deck_encoding": source_sha256s["deck_encoding"],
        "turn_order": source_sha256s["turn_order"],
        "terminal_scoring": source_sha256s["action_semantics"],
    }
    for name, expected in expected_context_hashes.items():
        if context_sources.get(name) != expected:
            raise CounterfactualPrivateTypeError(
                f"privacy-safe context source binding mismatch: {name}"
            )
    return {
        "schema": PRIVACY_SAFE_BATCH_SCHEMA,
        "artifact_kind": (
            "repository_owned_public_context_deterministic_private_type_proposal"
        ),
        "proposal_schema": COUNTERFACTUAL_PRIVATE_TYPE_SCHEMA,
        "public_context": context.to_canonical_dict(),
        "public_context_digest": context.digest(),
        "universe": dict(universe),
        "universe_sha256": _canonical_sha256(dict(universe)),
        "privacy_safe_traversal": {
            "domain": PRIVACY_SAFE_TRAVERSAL_DOMAIN,
            "version": PRIVACY_SAFE_TRAVERSAL_VERSION,
            "algorithm": PRIVACY_SAFE_TRAVERSAL_ALGORITHM,
            "ranking_algorithm": RANKING_ALGORITHM,
            "schedule_id": schedule_id,
            "phase": context.phase,
            "start_ordinal": start_ordinal,
            "sample_count": sample_count,
            "universe_size": universe_size,
            "coverage_fraction_exact": (
                f"{coverage.numerator}/{coverage.denominator}"
            ),
            "traversal_derivation_commitment_sha256": (
                traversal_derivation_commitment_sha256
            ),
            "sample_commitments": commitments,
            "sample_commitment_order_sha256": _canonical_sha256(commitments),
            "sample_commitments_unique": len(set(commitments)) == sample_count,
            "caller_supplied_entropy_parameter_exists": False,
            "caller_supplied_namespace_parameter_exists": False,
            "caller_supplied_slice_parameter_exists": False,
            "caller_supplied_registry_parameter_exists": False,
            "generic_seed_plan_consumed": False,
            "generic_batch_conversion_consumed": False,
            "actual_hand_input_exists": False,
            "raw_private_cards_serialized": False,
            "raw_universe_ranks_serialized": False,
        },
        "content_bindings": {
            "source_sha256s": source_sha256s,
            "source_set_sha256": _canonical_sha256(source_sha256s),
            "generator_runtime_semantic_binding_sha256": (
                runtime_semantic_binding_sha256
            ),
            "rules_sha256": context.bindings.rules_sha256,
            "canonical_deck_sha256": _canonical_sha256(list(_CANONICAL_DECK)),
        },
        "posterior_boundary": {
            "uniform_proposal_is_posterior": False,
            "actor_history_behavior_likelihood_applied": False,
            "opponent_public_evidence_marginal_likelihood_applied": False,
            "opponent_conditional_range_applied": False,
            "posterior_normalization_applied": False,
            "authorized_as_mccfr_prior": False,
            "eligible_as_posterior_join_origin": True,
        },
        "privacy_safe_origin_verified": True,
        "promotion_eligible": False,
        "production_sampling_ready": False,
        "serving_default_changed": False,
    }


def _build_privacy_safe_batch(
    context: PublicRootContext,
    *,
    runtime_semantic_binding_sha256: str,
) -> PrivacySafeCounterfactualPrivateTypeProposalBatch:
    if type(context) is not PublicRootContext:
        raise TypeError("context must be exact PublicRootContext")
    context.bindings.verify_live()
    universe = _universe_manifest(context)
    universe_size = int(universe["universe_size"])
    (
        offset,
        step,
        derivation_commitment,
        schedule_id,
        start_ordinal,
        sample_count,
    ) = _privacy_safe_traversal_parameters(context, universe)
    pool = _available_actor_private_pool(context)
    discard_count = len(_actor_prior_turns(context))
    proposals: list[CounterfactualPrivateTypeProposal] = []
    for sample_ordinal in range(
        start_ordinal, start_ordinal + sample_count
    ):
        universe_rank = (offset + step * sample_ordinal) % universe_size
        discards, current_draw = _decode_universe_rank(
            pool, discard_count, universe_rank
        )
        observation, witness = _observation_and_physical_witness(
            context, discards, current_draw
        )
        proposals.append(
            CounterfactualPrivateTypeProposal(
                sample_ordinal=sample_ordinal,
                private_type_commitment=_private_type_commitment(
                    context.digest(), observation
                ),
                observation=observation,
                universe_rank=universe_rank,
                physical_completion_witness=witness,
            )
        )
    manifest = _privacy_safe_audit_manifest(
        context,
        proposals,
        universe=universe,
        schedule_id=schedule_id,
        start_ordinal=start_ordinal,
        sample_count=sample_count,
        traversal_derivation_commitment_sha256=derivation_commitment,
        runtime_semantic_binding_sha256=runtime_semantic_binding_sha256,
    )
    manifest_json = _canonical_json(manifest)
    return PrivacySafeCounterfactualPrivateTypeProposalBatch(
        context_digest=context.digest(),
        proposals=tuple(proposals),
        universe_size=universe_size,
        schedule_id=schedule_id,
        start_ordinal=start_ordinal,
        sample_count=sample_count,
        traversal_derivation_commitment_sha256=derivation_commitment,
        audit_manifest_json=manifest_json,
        audit_manifest_sha256=hashlib.sha256(
            manifest_json.encode("utf-8")
        ).hexdigest(),
        _construction_token=_PRIVACY_SAFE_BATCH_CONSTRUCTION_TOKEN,
    )


def build_privacy_safe_counterfactual_private_type_proposals(
    context: PublicRootContext,
) -> PrivacySafeCounterfactualPrivateTypeProposalBatch:
    """Build the repository schedule from public context alone."""

    runtime_sha256 = _require_runtime_integrity()
    result = _build_privacy_safe_batch(
        context,
        runtime_semantic_binding_sha256=runtime_sha256,
    )
    verify_privacy_safe_counterfactual_private_type_proposals(context, result)
    return result


def verify_privacy_safe_counterfactual_private_type_proposals(
    context: PublicRootContext,
    result: PrivacySafeCounterfactualPrivateTypeProposalBatch,
) -> Mapping[str, Any]:
    """Freshly replay the context-only schedule and require exact equality."""

    runtime_sha256 = _require_runtime_integrity()
    if type(context) is not PublicRootContext:
        raise TypeError("context must be exact PublicRootContext")
    if type(result) is not PrivacySafeCounterfactualPrivateTypeProposalBatch:
        raise TypeError(
            "result must be exact PrivacySafeCounterfactualPrivateTypeProposalBatch"
        )
    expected = _build_privacy_safe_batch(
        context,
        runtime_semantic_binding_sha256=runtime_sha256,
    )
    if result != expected:
        raise CounterfactualPrivateTypeError(
            "privacy-safe batch does not match deterministic public replay"
        )
    build_signature = set(
        inspect.signature(
            build_privacy_safe_counterfactual_private_type_proposals
        ).parameters
    )
    verify_signature = set(
        inspect.signature(
            verify_privacy_safe_counterfactual_private_type_proposals
        ).parameters
    )
    if build_signature != {"context"} or verify_signature != {"context", "result"}:
        raise AssertionError("privacy-safe public API signature drifted")
    fields_present = {item.name for item in dataclass_fields(type(result))}
    forbidden_state = {
        "seed",
        "seed_nonce",
        "seed_plan",
        "approved_seed_plan_sha256",
        "seed_namespace",
        "trusted_registry_id",
        "registration_record_id",
        "actual_hand",
        "actual_actor_private_cards",
    }
    if fields_present & forbidden_state:
        raise AssertionError("privacy-safe batch acquired caller-controlled state")
    return MappingProxyType(
        {
            "verified": True,
            "privacy_safe_origin_verified": True,
            "public_context_digest": context.digest(),
            "schedule_id": result.schedule_id,
            "sample_count": result.sample_count,
            "traversal_derivation_commitment_sha256": (
                result.traversal_derivation_commitment_sha256
            ),
            "audit_manifest_sha256": result.audit_manifest_sha256,
            "authorized_as_mccfr_prior": False,
            "eligible_as_posterior_join_origin": True,
            "promotion_eligible": False,
            "production_sampling_ready": False,
        }
    )


_BUILD_SEED_REVEAL_IMPL = build_public_seed_reveal_commitment
_BUILD_SEED_PLAN_IMPL = build_preregistered_public_seed_plan
_VERIFY_SEED_PLAN_IMPL = verify_preregistered_public_seed_plan
_BUILD_PROPOSALS_IMPL = build_counterfactual_private_type_proposals
_VERIFY_PROPOSALS_IMPL = verify_counterfactual_private_type_proposals
_BUILD_PRIVACY_SAFE_IMPL = (
    build_privacy_safe_counterfactual_private_type_proposals
)
_VERIFY_PRIVACY_SAFE_IMPL = (
    verify_privacy_safe_counterfactual_private_type_proposals
)

_CANONICAL_MODULE_ALIASES = (
    ("_encoding_module", _encoding_module),
    ("_multi_root_module", _multi_root_module),
    ("_public_cfr_module", _public_cfr_module),
    ("_mixture_module", _mixture_module),
)
_CANONICAL_EXTERNALS = (
    (
        "public_cfr.InfoSetKey",
        _public_cfr_module,
        "InfoSetKey",
        _public_cfr_module.InfoSetKey,
    ),
    (
        "public_cfr.JointParticle",
        _public_cfr_module,
        "JointParticle",
        _public_cfr_module.JointParticle,
    ),
    (
        "public_cfr.PrivateRecall",
        _public_cfr_module,
        "PrivateRecall",
        _public_cfr_module.PrivateRecall,
    ),
    (
        "mixture.PublicRootContext",
        _mixture_module,
        "PublicRootContext",
        _mixture_module.PublicRootContext,
    ),
    (
        "multi_root._python_function_binding",
        _multi_root_module,
        "_python_function_binding",
        _multi_root_module._python_function_binding,
    ),
    (
        "multi_root._runtime_data_binding",
        _multi_root_module,
        "_runtime_data_binding",
        _multi_root_module._runtime_data_binding,
    ),
    (
        "multi_root._runtime_semantic_graph",
        _multi_root_module,
        "_runtime_semantic_graph",
        _multi_root_module._runtime_semantic_graph,
    ),
)
_CANONICAL_EXTERNAL_DATA = (
    (
        "encoding.ALL_CARDS",
        _encoding_module,
        "ALL_CARDS",
        _encoding_module.ALL_CARDS,
        tuple(_encoding_module.ALL_CARDS),
    ),
)
_CANONICAL_RUNTIME_DATA_ALIASES = (
    ("COUNTERFACTUAL_PRIVATE_TYPE_SCHEMA", COUNTERFACTUAL_PRIVATE_TYPE_SCHEMA),
    (
        "COUNTERFACTUAL_PRIVATE_TYPE_BATCH_SCHEMA",
        COUNTERFACTUAL_PRIVATE_TYPE_BATCH_SCHEMA,
    ),
    ("UNIVERSE_SCHEMA", UNIVERSE_SCHEMA),
    ("PUBLIC_SEED_PLAN_SCHEMA", PUBLIC_SEED_PLAN_SCHEMA),
    ("PUBLIC_SEED_REVEAL_SCHEMA", PUBLIC_SEED_REVEAL_SCHEMA),
    ("PRIVACY_SAFE_BATCH_SCHEMA", PRIVACY_SAFE_BATCH_SCHEMA),
    ("PRIVACY_SAFE_TRAVERSAL_DOMAIN", PRIVACY_SAFE_TRAVERSAL_DOMAIN),
    ("PRIVACY_SAFE_TRAVERSAL_VERSION", PRIVACY_SAFE_TRAVERSAL_VERSION),
    ("PRIVACY_SAFE_TRAVERSAL_ALGORITHM", PRIVACY_SAFE_TRAVERSAL_ALGORITHM),
    ("PRIVACY_SAFE_PHASE_SCHEDULE", PRIVACY_SAFE_PHASE_SCHEDULE),
    ("TRAVERSAL_ALGORITHM", TRAVERSAL_ALGORITHM),
    ("RANKING_ALGORITHM", RANKING_ALGORITHM),
    ("PROPOSAL_SEMANTICS", PROPOSAL_SEMANTICS),
    ("_PHYSICAL_JOKERS", _PHYSICAL_JOKERS),
    ("_EXPECTED_UNDEALT_BY_PHASE", _EXPECTED_UNDEALT_BY_PHASE),
    ("_VALID_CARDS", _VALID_CARDS),
    ("_CANONICAL_DECK", _CANONICAL_DECK),
    ("_SOURCE_PATHS", _SOURCE_PATHS),
)


def _live_runtime_semantic_binding() -> dict[str, Any]:
    critical_roots = {
        "_seed_reveal_payload",
        "_public_cards",
        "_available_actor_private_pool",
        "_actor_prior_turns",
        "_universe_size",
        "_unrank_permutation",
        "_unrank_combination",
        "_decode_universe_rank",
        "_hash_randbelow",
        "_traversal_parameters",
        "_private_recall",
        "_opponent_completion_recall",
        "_observation_and_physical_witness",
        "_private_type_commitment",
        "_universe_manifest",
        "_audit_manifest",
        "_privacy_safe_traversal_parameters",
        "_privacy_safe_audit_manifest",
    }
    roots = {
        name: function
        for name, function in _CANONICAL_LOCAL_HELPERS
        if name in critical_roots
    }
    data_roots = {
        name: value for name, value in _CANONICAL_RUNTIME_DATA_ALIASES
    }
    try:
        return _multi_root_module._runtime_semantic_graph(
            roots,
            data_roots=data_roots,
        )
    except Exception as exc:
        for name, function in roots.items():
            try:
                _multi_root_module._runtime_semantic_graph(
                    {name: function}, data_roots=data_roots
                )
            except Exception as root_exc:
                raise CounterfactualPrivateTypeError(
                    f"runtime semantic graph cannot bind root {name}: {root_exc}"
                ) from root_exc
        raise CounterfactualPrivateTypeError(
            f"runtime semantic graph construction failed: {exc}"
        ) from exc


def _require_runtime_identities(
    *,
    expected_module_aliases: tuple[tuple[str, object], ...],
    expected_externals: tuple[tuple[str, object, str, object], ...],
    expected_external_data: tuple[
        tuple[str, object, str, object, tuple[Any, ...]], ...
    ],
    expected_local_helpers: tuple[tuple[str, object], ...],
    expected_data_aliases: tuple[tuple[str, object], ...],
    expected_runtime_binding_sha256: str,
    expected_source_sha256s: Mapping[str, str],
) -> str:
    for name, expected in expected_module_aliases:
        if globals().get(name) is not expected:
            raise CounterfactualPrivateTypeError(
                f"canonical module alias drifted: {name}"
            )
    for label, module, attribute, expected in expected_externals:
        if getattr(module, attribute) is not expected:
            raise CounterfactualPrivateTypeError(
                f"canonical runtime callable drifted: {label}"
            )
    for label, module, attribute, expected_object, expected_value in (
        expected_external_data
    ):
        actual = getattr(module, attribute)
        if actual is not expected_object or tuple(actual) != expected_value:
            raise CounterfactualPrivateTypeError(
                f"canonical runtime data drifted: {label}"
            )
    for name, expected in expected_local_helpers:
        if globals().get(name) is not expected:
            raise CounterfactualPrivateTypeError(
                f"generator runtime helper drifted: {name}"
            )
    for name, expected in expected_data_aliases:
        if globals().get(name) is not expected:
            raise CounterfactualPrivateTypeError(
                f"generator runtime data alias drifted: {name}"
            )
    live_runtime = _live_runtime_semantic_binding()
    if live_runtime.get("binding_sha256") != expected_runtime_binding_sha256:
        raise CounterfactualPrivateTypeError(
            "generator transitive runtime semantic binding drifted"
        )
    live_sources = _live_source_sha256s()
    if live_sources != dict(expected_source_sha256s):
        changed = sorted(
            name
            for name in set(live_sources) | set(expected_source_sha256s)
            if live_sources.get(name) != expected_source_sha256s.get(name)
        )
        raise CounterfactualPrivateTypeError(
            f"generator source binding drifted: {changed}"
        )
    return expected_runtime_binding_sha256


_CANONICAL_LOCAL_HELPERS = (
    ("_canonical_json", _canonical_json),
    ("_canonical_sha256", _canonical_sha256),
    ("_file_sha256", _file_sha256),
    ("_live_source_sha256s", _live_source_sha256s),
    ("_require_sha256", _require_sha256),
    ("_require_public_text", _require_public_text),
    ("_seed_reveal_payload", _seed_reveal_payload),
    ("_public_cards", _public_cards),
    ("_available_actor_private_pool", _available_actor_private_pool),
    ("_actor_prior_turns", _actor_prior_turns),
    ("_universe_size", _universe_size),
    ("_unrank_permutation", _unrank_permutation),
    ("_unrank_combination", _unrank_combination),
    ("_decode_universe_rank", _decode_universe_rank),
    ("_hash_randbelow", _hash_randbelow),
    ("_traversal_parameters", _traversal_parameters),
    ("_private_recall", _private_recall),
    ("_opponent_completion_recall", _opponent_completion_recall),
    ("_observation_and_physical_witness", _observation_and_physical_witness),
    ("_private_type_commitment", _private_type_commitment),
    ("_universe_manifest", _universe_manifest),
    ("_audit_manifest", _audit_manifest),
    ("_privacy_safe_traversal_parameters", _privacy_safe_traversal_parameters),
    ("_privacy_safe_audit_manifest", _privacy_safe_audit_manifest),
    ("_build_privacy_safe_batch", _build_privacy_safe_batch),
    ("_BUILD_SEED_REVEAL_IMPL", _BUILD_SEED_REVEAL_IMPL),
    ("_BUILD_SEED_PLAN_IMPL", _BUILD_SEED_PLAN_IMPL),
    ("_VERIFY_SEED_PLAN_IMPL", _VERIFY_SEED_PLAN_IMPL),
    ("_BUILD_PROPOSALS_IMPL", _BUILD_PROPOSALS_IMPL),
    ("_VERIFY_PROPOSALS_IMPL", _VERIFY_PROPOSALS_IMPL),
    ("_BUILD_PRIVACY_SAFE_IMPL", _BUILD_PRIVACY_SAFE_IMPL),
    ("_VERIFY_PRIVACY_SAFE_IMPL", _VERIFY_PRIVACY_SAFE_IMPL),
    ("_live_runtime_semantic_binding", _live_runtime_semantic_binding),
)
_CANONICAL_SOURCE_SHA256S = MappingProxyType(_live_source_sha256s())
_CANONICAL_RUNTIME_SEMANTIC_BINDING_SHA256 = _live_runtime_semantic_binding()[
    "binding_sha256"
]
_CANONICAL_RUNTIME_GUARD = _require_runtime_identities


def _require_runtime_integrity() -> str:
    return _CANONICAL_RUNTIME_GUARD(
        expected_module_aliases=_CANONICAL_MODULE_ALIASES,
        expected_externals=_CANONICAL_EXTERNALS,
        expected_external_data=_CANONICAL_EXTERNAL_DATA,
        expected_local_helpers=_CANONICAL_LOCAL_HELPERS,
        expected_data_aliases=_CANONICAL_RUNTIME_DATA_ALIASES,
        expected_runtime_binding_sha256=(
            _CANONICAL_RUNTIME_SEMANTIC_BINDING_SHA256
        ),
        expected_source_sha256s=_CANONICAL_SOURCE_SHA256S,
    )


def _guard_entrypoint(function: Callable[..., Any]) -> Callable[..., Any]:
    """Close over exact import-time identities so rebinding guard aliases fails."""

    canonical_guard = _require_runtime_identities
    canonical_module_aliases = _CANONICAL_MODULE_ALIASES
    canonical_externals = _CANONICAL_EXTERNALS
    canonical_external_data = _CANONICAL_EXTERNAL_DATA
    canonical_helpers = _CANONICAL_LOCAL_HELPERS
    canonical_data = _CANONICAL_RUNTIME_DATA_ALIASES
    canonical_runtime_sha256 = _CANONICAL_RUNTIME_SEMANTIC_BINDING_SHA256
    canonical_sources = _CANONICAL_SOURCE_SHA256S

    @wraps(function)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        if (
            globals().get("_CANONICAL_RUNTIME_GUARD") is not canonical_guard
            or globals().get("_require_runtime_identities") is not canonical_guard
        ):
            raise CounterfactualPrivateTypeError(
                "canonical generator runtime guard alias drifted"
            )
        identity_sets = (
            ("_CANONICAL_MODULE_ALIASES", canonical_module_aliases),
            ("_CANONICAL_EXTERNALS", canonical_externals),
            ("_CANONICAL_EXTERNAL_DATA", canonical_external_data),
            ("_CANONICAL_LOCAL_HELPERS", canonical_helpers),
            ("_CANONICAL_RUNTIME_DATA_ALIASES", canonical_data),
            ("_CANONICAL_SOURCE_SHA256S", canonical_sources),
        )
        for name, expected in identity_sets:
            if globals().get(name) is not expected:
                raise CounterfactualPrivateTypeError(
                    f"canonical generator identity set drifted: {name}"
                )
        if globals().get(
            "_CANONICAL_RUNTIME_SEMANTIC_BINDING_SHA256"
        ) != canonical_runtime_sha256:
            raise CounterfactualPrivateTypeError(
                "canonical generator runtime snapshot alias drifted"
            )
        canonical_guard(
            expected_module_aliases=canonical_module_aliases,
            expected_externals=canonical_externals,
            expected_external_data=canonical_external_data,
            expected_local_helpers=canonical_helpers,
            expected_data_aliases=canonical_data,
            expected_runtime_binding_sha256=canonical_runtime_sha256,
            expected_source_sha256s=canonical_sources,
        )
        return function(*args, **kwargs)

    return guarded


build_public_seed_reveal_commitment = _guard_entrypoint(_BUILD_SEED_REVEAL_IMPL)
build_preregistered_public_seed_plan = _guard_entrypoint(_BUILD_SEED_PLAN_IMPL)
verify_preregistered_public_seed_plan = _guard_entrypoint(_VERIFY_SEED_PLAN_IMPL)
build_counterfactual_private_type_proposals = _guard_entrypoint(
    _BUILD_PROPOSALS_IMPL
)
verify_counterfactual_private_type_proposals = _guard_entrypoint(
    _VERIFY_PROPOSALS_IMPL
)
build_privacy_safe_counterfactual_private_type_proposals = _guard_entrypoint(
    _BUILD_PRIVACY_SAFE_IMPL
)
verify_privacy_safe_counterfactual_private_type_proposals = _guard_entrypoint(
    _VERIFY_PRIVACY_SAFE_IMPL
)


__all__ = [
    "COUNTERFACTUAL_PRIVATE_TYPE_BATCH_SCHEMA",
    "COUNTERFACTUAL_PRIVATE_TYPE_SCHEMA",
    "CounterfactualPrivateTypeError",
    "CounterfactualPrivateTypeProposal",
    "CounterfactualPrivateTypeProposalBatch",
    "PUBLIC_SEED_PLAN_SCHEMA",
    "PUBLIC_SEED_REVEAL_SCHEMA",
    "PRIVACY_SAFE_BATCH_SCHEMA",
    "PRIVACY_SAFE_PHASE_SCHEDULE",
    "PRIVACY_SAFE_TRAVERSAL_ALGORITHM",
    "PRIVACY_SAFE_TRAVERSAL_DOMAIN",
    "PRIVACY_SAFE_TRAVERSAL_VERSION",
    "PROPOSAL_SEMANTICS",
    "PreregisteredPublicSeedPlan",
    "PrivacySafeCounterfactualPrivateTypeProposalBatch",
    "RANKING_ALGORITHM",
    "TRAVERSAL_ALGORITHM",
    "UNIVERSE_SCHEMA",
    "build_counterfactual_private_type_proposals",
    "build_privacy_safe_counterfactual_private_type_proposals",
    "build_preregistered_public_seed_plan",
    "build_public_seed_reveal_commitment",
    "verify_counterfactual_private_type_proposals",
    "verify_privacy_safe_counterfactual_private_type_proposals",
    "verify_preregistered_public_seed_plan",
]
