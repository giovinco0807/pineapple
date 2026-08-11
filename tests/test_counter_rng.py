from dataclasses import fields

import pytest

from ofc_regular.counter_rng import (
    COUNTER_RNG_SCHEMA,
    CounterRngKey,
    common_future_seed,
    policy_decision_seed,
)


def test_counter_seed_is_repeatable_and_uses_stable_schema():
    key = CounterRngKey(
        base_seed=2026071201,
        run_id="m1-smoke",
        phase="common_future",
        sample_index=7,
        actor="chance",
        street="T2",
        stream="future_cards",
        counter=3,
        root_fingerprint="abc123",
    )
    clone = CounterRngKey(
        base_seed=key.base_seed,
        run_id=key.run_id,
        phase=key.phase,
        sample_index=key.sample_index,
        actor=key.actor,
        street=key.street,
        stream=key.stream,
        counter=key.counter,
        root_fingerprint=key.root_fingerprint,
    )

    assert key.payload()["schema"] == COUNTER_RNG_SCHEMA
    assert key.seed() == 1867823025197256593  # Python/Rust golden vector.
    assert key.seed() == clone.seed()
    assert key.random().getrandbits(64) == key.random().getrandbits(64)


def test_common_future_seeds_do_not_depend_on_candidate_order():
    candidates = ["action-c", "action-a", "action-b"]

    def by_candidate(order):
        return {
            candidate: [
                common_future_seed(
                    base_seed=17,
                    run_id="teacher",
                    root_fingerprint="root-state",
                    sample_index=index,
                    street="T1",
                )
                for index in range(4)
            ]
            for candidate in order
        }

    assert by_candidate(candidates) == by_candidate(list(reversed(candidates)))


def test_policy_decision_seed_has_no_action_index_coordinate():
    field_names = {field.name for field in fields(CounterRngKey)}
    assert "action_index" not in field_names
    assert "action_key" not in field_names

    first = policy_decision_seed(
        base_seed=23,
        run_id="teacher",
        root_fingerprint="root-state",
        future_index=2,
        actor=1,
        street="T3",
        decision_ordinal=7,
    )
    second = policy_decision_seed(
        base_seed=23,
        run_id="teacher",
        root_fingerprint="root-state",
        future_index=2,
        actor=1,
        street="T3",
        decision_ordinal=7,
    )
    assert first == second


def test_counter_coordinates_are_domain_separated():
    base = dict(
        base_seed=5,
        run_id="run",
        phase="phase",
        sample_index=1,
        actor=0,
        street="T2",
    )
    seeds = {
        CounterRngKey(**base).seed(),
        CounterRngKey(**{**base, "sample_index": 2}).seed(),
        CounterRngKey(**{**base, "actor": 1}).seed(),
        CounterRngKey(**{**base, "street": "T3"}).seed(),
        CounterRngKey(**{**base, "phase": "other"}).seed(),
    }
    assert len(seeds) == 5


@pytest.mark.parametrize(("name", "value"), (("sample_index", -1), ("counter", -1)))
def test_counter_key_rejects_negative_coordinates(name, value):
    kwargs = dict(
        base_seed=1,
        run_id="run",
        phase="phase",
        sample_index=0,
        actor="chance",
        street="T0",
        counter=0,
    )
    kwargs[name] = value
    with pytest.raises(ValueError, match="non-negative"):
        CounterRngKey(**kwargs)
