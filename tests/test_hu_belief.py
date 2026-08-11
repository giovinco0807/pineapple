from dataclasses import replace

import pytest

from ofc_regular.cards import ALL_CARDS
from ofc_regular.hu_belief import (
    HIDDEN_CARD_BATCH_SCHEMA,
    HIDDEN_CARD_BELIEF_SCHEMA,
    HIDDEN_CARD_PARTICLE_SCHEMA,
    HIDDEN_CARD_PRIOR,
    HiddenCardBeliefError,
    HiddenCardParticle,
    sample_hidden_card_particle,
    sample_hidden_card_particles,
    turn2_actor_observation,
)
from ofc_regular.hu_infoset import (
    ActorObservation,
    InformationSetError,
    ScoringContext,
    WorldState,
)
from ofc_regular.state import Board


# The FL EV the cross-language parity vector below was pinned under.  It is the
# superseded June constant deliberately: the vector guards the counter-based
# shuffle, which must survive an FL EV re-measurement unchanged.
_PARITY_VECTOR_FL_EV_14 = 10.227020614683454


def _board(cards: tuple[str, ...]) -> Board:
    top_count = min(3, len(cards))
    middle_count = min(5, len(cards) - top_count)
    return Board.from_rows(
        top=cards[:top_count],
        middle=cards[top_count : top_count + middle_count],
        bottom=cards[top_count + middle_count :],
    )


def _observation(street: str, to_act_order: str) -> ActorObservation:
    geometry = {
        ("T1", "first"): (5, 5, 0),
        ("T1", "second"): (5, 7, 0),
        ("T2", "first"): (7, 7, 1),
        ("T2", "second"): (7, 9, 1),
        ("T3", "first"): (9, 9, 2),
        ("T3", "second"): (9, 11, 2),
        ("T4", "first"): (11, 11, 3),
        ("T4", "second"): (11, 13, 3),
    }
    hero_count, opponent_count, hero_discard_count = geometry[(street, to_act_order)]
    cursor = 0
    hero_cards = ALL_CARDS[cursor : cursor + hero_count]
    cursor += hero_count
    opponent_cards = ALL_CARDS[cursor : cursor + opponent_count]
    cursor += opponent_count
    dealt = ALL_CARDS[cursor : cursor + 3]
    cursor += 3
    hero_discards = ALL_CARDS[cursor : cursor + hero_discard_count]
    return ActorObservation(
        hero_board=_board(hero_cards),
        opponent_public_board=_board(opponent_cards),
        dealt_cards=dealt,
        hero_private_discards=hero_discards,
        seat=to_act_order,  # type: ignore[arg-type]
        street=street,  # type: ignore[arg-type]
        to_act_order=to_act_order,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    ("street", "to_act_order", "opponent_discard_count"),
    (
        ("T1", "first", 0),
        ("T1", "second", 1),
        ("T2", "first", 1),
        ("T2", "second", 2),
        ("T3", "first", 2),
        ("T3", "second", 3),
        ("T4", "first", 3),
        ("T4", "second", 4),
    ),
)
def test_t1_t4_particles_partition_every_unknown_card_without_duplicates(
    street, to_act_order, opponent_discard_count
):
    observation = _observation(street, to_act_order)

    batch = sample_hidden_card_particles(
        observation,
        base_seed=2026071201,
        run_id="m1-belief-smoke",
        sample_count=4,
    )

    known = set(observation.known_unavailable_cards())
    for particle in batch.particles:
        particle.validate_against(observation)
        assert len(particle.opponent_private_discards) == opponent_discard_count
        assert len(particle.unseen_deck) == 52 - len(known) - opponent_discard_count
        assert len(set(particle.hidden_cards)) == len(particle.hidden_cards)
        assert known.isdisjoint(particle.hidden_cards)
        assert known | set(particle.hidden_cards) == set(ALL_CARDS)


def test_sampling_is_repeatable_shard_addressable_and_digestible():
    observation = _observation("T2", "second")
    kwargs = {
        "base_seed": 918273,
        "run_id": "belief-determinism",
        "sample_count": 6,
    }

    first = sample_hidden_card_particles(observation, **kwargs)
    second = sample_hidden_card_particles(observation, **kwargs)
    different_seed = sample_hidden_card_particles(
        observation,
        base_seed=kwargs["base_seed"] + 1,
        run_id=kwargs["run_id"],
        sample_count=kwargs["sample_count"],
    )
    left_shard = sample_hidden_card_particles(
        observation,
        base_seed=kwargs["base_seed"],
        run_id=kwargs["run_id"],
        sample_count=2,
    )
    right_shard = sample_hidden_card_particles(
        observation,
        base_seed=kwargs["base_seed"],
        run_id=kwargs["run_id"],
        sample_count=4,
        start_index=2,
    )

    assert first == second
    assert first.digest() == second.digest()
    assert first.particle_digests == second.particle_digests
    assert first.digest() != different_seed.digest()
    assert first.particle_digests != different_seed.particle_digests
    assert first.particle_digests == (
        *left_shard.particle_digests,
        *right_shard.particle_digests,
    )
    assert len(set(first.particle_digests)) == len(first.particles)

    # Python/Rust parity vector for the counter-based Fisher-Yates contract.
    # The FL EV is inside the observation fingerprint that keys the counter
    # stream, so the vector is re-derived here under the constant it was
    # verified against rather than under today's production default: this pins
    # the shuffle contract, and a re-measured FL EV must not disturb it.
    golden_batch = sample_hidden_card_particles(
        replace(
            observation,
            scoring=ScoringContext(fl_ev=((14, _PARITY_VECTOR_FL_EV_14),)),
        ),
        **kwargs,
    )
    golden = golden_batch.particles[0]
    assert golden.digest() == (
        "2828abbe316c82ad99df2a431899fb7f6bb26055857a3216ef1133518bc0ffc3"
    )
    assert golden.rng_key_digest == (
        "9ebfccec564550201568f4f5b9b1ce7ada5eb8a5aff18fa61370557e746e02ce"
    )
    assert golden.opponent_private_discards == ("Td", "8c")
    assert golden.unseen_deck[:8] == (
        "7s",
        "2s",
        "Ts",
        "8s",
        "6s",
        "Kc",
        "As",
        "Ad",
    )

    manifest = first.to_dict()
    assert manifest["schema"] == HIDDEN_CARD_BATCH_SCHEMA
    assert manifest["belief_schema"] == HIDDEN_CARD_BELIEF_SCHEMA
    assert manifest["prior"] == HIDDEN_CARD_PRIOR
    assert manifest["particle_digests"] == list(first.particle_digests)
    assert "particles" not in manifest
    expanded = first.to_dict(include_particles=True)
    assert expanded["particles"][0]["schema"] == HIDDEN_CARD_PARTICLE_SCHEMA


@pytest.mark.parametrize(
    ("street", "to_act_order"),
    (
        ("T3", "first"),
        ("T3", "second"),
        ("T4", "first"),
        ("T4", "second"),
    ),
)
def test_t3_t4_sampling_is_repeatable_shard_addressable_and_order_invariant(
    street, to_act_order
):
    observation = _observation(street, to_act_order)
    reordered = ActorObservation(
        hero_board=Board.from_rows(
            top=reversed(observation.hero_board.top),
            middle=reversed(observation.hero_board.middle),
            bottom=reversed(observation.hero_board.bottom),
        ),
        opponent_public_board=Board.from_rows(
            top=reversed(observation.opponent_public_board.top),
            middle=reversed(observation.opponent_public_board.middle),
            bottom=reversed(observation.opponent_public_board.bottom),
        ),
        dealt_cards=tuple(reversed(observation.dealt_cards)),
        hero_private_discards=tuple(reversed(observation.hero_private_discards)),
        seat=observation.seat,
        street=observation.street,
        to_act_order=observation.to_act_order,
        scoring=observation.scoring,
    )
    kwargs = {
        "base_seed": 2026071301,
        "run_id": f"m2-late-belief-{street}-{to_act_order}",
        "sample_count": 5,
    }

    first = sample_hidden_card_particles(observation, **kwargs)
    repeated = sample_hidden_card_particles(observation, **kwargs)
    reordered_batch = sample_hidden_card_particles(reordered, **kwargs)
    different_seed = sample_hidden_card_particles(
        observation,
        base_seed=kwargs["base_seed"] + 1,
        run_id=kwargs["run_id"],
        sample_count=kwargs["sample_count"],
    )
    left_shard = sample_hidden_card_particles(
        observation,
        base_seed=kwargs["base_seed"],
        run_id=kwargs["run_id"],
        sample_count=2,
    )
    right_shard = sample_hidden_card_particles(
        observation,
        base_seed=kwargs["base_seed"],
        run_id=kwargs["run_id"],
        sample_count=3,
        start_index=2,
    )

    assert observation.fingerprint() == reordered.fingerprint()
    assert first == repeated == reordered_batch
    assert first.digest() == repeated.digest() == reordered_batch.digest()
    assert first.particle_digests != different_seed.particle_digests
    assert first.particle_digests == (
        *left_shard.particle_digests,
        *right_shard.particle_digests,
    )
    assert len(set(first.particle_digests)) == len(first.particles)


@pytest.mark.parametrize(
    ("street", "to_act_order"),
    (
        ("T3", "first"),
        ("T3", "second"),
        ("T4", "first"),
        ("T4", "second"),
    ),
)
def test_t3_t4_sampling_cannot_depend_on_realized_opponent_discard_identity(
    street, to_act_order
):
    observation = _observation(street, to_act_order)
    unknown = [
        card
        for card in ALL_CARDS
        if card not in set(observation.known_unavailable_cards())
    ]
    discard_count = observation.opponent_discard_count
    first_opponent_discards = tuple(unknown[:discard_count])
    second_opponent_discards = tuple(unknown[discard_count : 2 * discard_count])

    if to_act_order == "first":
        boards = (observation.hero_board, observation.opponent_public_board)
        first_discards = (
            observation.hero_private_discards,
            first_opponent_discards,
        )
        second_discards = (
            observation.hero_private_discards,
            second_opponent_discards,
        )
        actor = 0
    else:
        boards = (observation.opponent_public_board, observation.hero_board)
        first_discards = (
            first_opponent_discards,
            observation.hero_private_discards,
        )
        second_discards = (
            second_opponent_discards,
            observation.hero_private_discards,
        )
        actor = 1

    first_truth = WorldState(
        boards=boards,
        private_discards=first_discards,
        street=street,
        next_player=actor,
    )
    second_truth = WorldState(
        boards=boards,
        private_discards=second_discards,
        street=street,
        next_player=actor,
    )
    first_observation = first_truth.observe(actor, observation.dealt_cards)
    second_observation = second_truth.observe(actor, observation.dealt_cards)

    assert first_observation == observation
    assert second_observation == observation
    first = sample_hidden_card_particles(
        first_observation,
        base_seed=2026071302,
        run_id=f"m2-late-truth-invariance-{street}-{to_act_order}",
        sample_count=4,
    )
    second = sample_hidden_card_particles(
        second_observation,
        base_seed=2026071302,
        run_id=f"m2-late-truth-invariance-{street}-{to_act_order}",
        sample_count=4,
    )

    assert first == second
    assert first.digest() == second.digest()


def test_sampling_cannot_depend_on_realized_opponent_discard_identity():
    hero = _board(ALL_CARDS[0:7])
    opponent = _board(ALL_CARDS[7:14])
    hero_discard = ALL_CARDS[14]
    dealt = ALL_CARDS[15:18]
    first_truth = WorldState(
        boards=(hero, opponent),
        private_discards=((hero_discard,), (ALL_CARDS[18],)),
        street="T2",
        next_player=0,
    )
    second_truth = WorldState(
        boards=(hero, opponent),
        private_discards=((hero_discard,), (ALL_CARDS[19],)),
        street="T2",
        next_player=0,
    )
    first_observation = first_truth.observe(0, dealt)
    second_observation = second_truth.observe(0, dealt)

    assert first_observation == second_observation
    first = sample_hidden_card_particles(
        first_observation,
        base_seed=44,
        run_id="truth-invariance",
        sample_count=3,
    )
    second = sample_hidden_card_particles(
        second_observation,
        base_seed=44,
        run_id="truth-invariance",
        sample_count=3,
    )
    assert first == second
    assert first.digest() == second.digest()


def test_turn2_observation_uses_only_explicit_hero_visible_discard():
    hero = _board(ALL_CARDS[0:7])
    opponent = _board(ALL_CARDS[7:14])
    hero_discard = ALL_CARDS[14]
    dealt = ALL_CARDS[15:18]
    visible_dead = (*opponent.all_cards(), hero_discard)

    explicit = turn2_actor_observation(
        hero_board=hero,
        opponent_public_board=opponent,
        dealt_cards=dealt,
        hero_seat="first",
        hero_private_discards=(hero_discard,),
        visible_dead_cards=visible_dead,
    )
    derived = turn2_actor_observation(
        hero_board=hero,
        opponent_public_board=opponent,
        dealt_cards=dealt,
        hero_seat="first",
        visible_dead_cards=visible_dead,
    )

    assert explicit == derived
    assert derived.hero_private_discards == (hero_discard,)
    assert derived.to_act_order == "first"


def test_turn2_observation_fails_closed_on_truth_like_or_conflicting_dead_cards():
    hero = _board(ALL_CARDS[0:7])
    opponent = _board(ALL_CARDS[7:14])
    hero_discard = ALL_CARDS[14]
    opponent_discard = ALL_CARDS[15]
    dealt = ALL_CARDS[16:19]

    with pytest.raises(HiddenCardBeliefError, match="exactly one"):
        turn2_actor_observation(
            hero_board=hero,
            opponent_public_board=opponent,
            dealt_cards=dealt,
            hero_seat="first",
            visible_dead_cards=(
                *opponent.all_cards(),
                hero_discard,
                opponent_discard,
            ),
        )
    with pytest.raises(HiddenCardBeliefError, match="disagree"):
        turn2_actor_observation(
            hero_board=hero,
            opponent_public_board=opponent,
            dealt_cards=dealt,
            hero_seat="first",
            hero_private_discards=(hero_discard,),
            visible_dead_cards=(*opponent.all_cards(), opponent_discard),
        )


def test_replay_belief_is_invariant_to_offline_opponent_truth_field():
    hero = _board(ALL_CARDS[0:7])
    opponent = _board(ALL_CARDS[7:14])
    safe_record = {
        "hero_board": hero,
        "opponent_board": opponent,
        "dealt": ALL_CARDS[15:18],
        "hero_private_discards": (ALL_CARDS[14],),
        "visible_dead_cards": (*opponent.all_cards(), ALL_CARDS[14]),
        "seat": "first",
    }
    first_record = {
        **safe_record,
        "opponent_private_discards": (ALL_CARDS[18],),
    }
    second_record = {
        **safe_record,
        "opponent_private_discards": (ALL_CARDS[19],),
    }

    def belief_for(record):
        observation = turn2_actor_observation(
            hero_board=record["hero_board"],
            opponent_public_board=record["opponent_board"],
            dealt_cards=record["dealt"],
            hero_seat=record["seat"],
            hero_private_discards=record["hero_private_discards"],
            visible_dead_cards=record["visible_dead_cards"],
        )
        return sample_hidden_card_particles(
            observation,
            base_seed=901,
            run_id="offline-truth-invariance",
            sample_count=4,
        )

    assert belief_for(first_record) == belief_for(second_record)


def test_sampler_rejects_world_state_and_replay_mapping_at_api_boundary():
    observation = _observation("T1", "first")
    world = WorldState(
        boards=(observation.hero_board, observation.opponent_public_board),
        private_discards=((), ()),
        street="T1",
        next_player=0,
    )

    with pytest.raises(TypeError, match="ActorObservation.*WorldState"):
        sample_hidden_card_particle(  # type: ignore[arg-type]
            world, base_seed=1, run_id="unsafe", sample_index=0
        )
    with pytest.raises(TypeError, match="ActorObservation.*replay truth"):
        sample_hidden_card_particles(  # type: ignore[arg-type]
            {"policy_observation": observation.to_dict(), "deck": list(ALL_CARDS)},
            base_seed=1,
            run_id="unsafe",
            sample_count=1,
        )


def test_particle_validation_detects_visible_overlap_and_duplicate_hidden_cards():
    observation = _observation("T2", "first")
    particle = sample_hidden_card_particle(
        observation,
        base_seed=31,
        run_id="validation",
        sample_index=0,
    )
    visible_card = observation.hero_board.all_cards()[0]
    overlapping = replace(
        particle,
        unseen_deck=(visible_card, *particle.unseen_deck[1:]),
    )
    with pytest.raises(HiddenCardBeliefError, match="overlaps actor-visible"):
        overlapping.validate_against(observation)

    with pytest.raises(HiddenCardBeliefError, match="duplicate card"):
        HiddenCardParticle(
            observation_fingerprint=observation.fingerprint(),
            street="T2",
            sample_index=0,
            opponent_private_discards=("As",),
            unseen_deck=("As",),
            rng_key_digest="0" * 64,
        )


def test_observation_geometry_is_checked_before_sampling():
    valid = _observation("T1", "first")
    with pytest.raises(InformationSetError, match="decision geometry"):
        replace(valid, hero_private_discards=("As",))

    with pytest.raises(InformationSetError, match="decision geometry"):
        replace(valid, street="T3")


def test_particle_draw_is_non_mutating_and_bounds_checked():
    particle = sample_hidden_card_particle(
        _observation("T1", "second"),
        base_seed=71,
        run_id="draw",
        sample_index=4,
    )

    assert particle.draw(3) == particle.unseen_deck[:3]
    assert particle.draw(3, offset=3) == particle.unseen_deck[3:6]
    assert particle.draw(0) == ()
    with pytest.raises(ValueError, match="exceeds"):
        particle.draw(len(particle.unseen_deck) + 1)
