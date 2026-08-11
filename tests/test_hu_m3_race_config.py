"""The racing schedule's Python surface: validation, and what reaches the wire.

The engine refuses every malformed schedule on its own, but a plan that carries
one should be rejected when the plan is read rather than on the first position
of the first shard -- a fleet that has already spun up its VMs and copied its
weights is an expensive place to discover a typo in a list of integers. So the
dataclass repeats the engine's rules, and these tests are what keep the two
statements of the same rule from drifting apart.

The other half is the wire format. Every optional field in this config is
emitted only when it is asked for, because a request that asked for nothing has
to be byte-identical to one built before the field existed -- that is what makes
every label already generated reproducible from a rebuilt engine. Racing is held
to the same promise.
"""

from __future__ import annotations

import dataclasses

import pytest

from ofc_regular.hu_m3_rust import _joint_config_payload
from ofc_regular.hu_turn3_joint_exact_teacher import JointExactConfig


def base(**overrides) -> JointExactConfig:
    """A config that would run, with the racing fields left at their defaults."""

    fields = dict(
        candidate_samples=8,
        evaluation_samples=256,
        downstream_t3_samples=4,
        downstream_t4_samples=0,
        seed=11,
        candidate_seed=11,
        evaluation_seed=11,
        run_id="race-config-test",
        prefilter_samples=32,
        prefilter_keep=48,
        prefilter_margin=2.4,
    )
    fields.update(overrides)
    return JointExactConfig(**fields)


def test_the_default_config_does_not_race_and_says_nothing_about_it():
    config = base()
    assert config.race_schedule == ()
    assert config.race_lcb_z == 0.0
    payload = _joint_config_payload(config)
    assert not [key for key in payload if key.startswith("race")], (
        "a config that did not ask for a race must not put one on the wire"
    )


def test_a_schedule_reaches_the_wire_with_its_threshold():
    config = base(race_schedule=(32, 64, 128, 256), race_lcb_z=2.0)
    payload = _joint_config_payload(config)
    assert payload["race_schedule"] == [32, 64, 128, 256]
    assert payload["race_lcb_z"] == 2.0


def test_a_zero_threshold_is_still_emitted_alongside_a_schedule():
    """Zero is a legal, aggressive z -- eliminate on any positive mean deficit.

    It has to be emitted, because the engine requires the pair and would
    otherwise read a schedule with no threshold at all.
    """

    payload = _joint_config_payload(base(race_schedule=(128, 256)))
    assert payload["race_schedule"] == [128, 256]
    assert payload["race_lcb_z"] == 0.0


def test_a_list_is_normalised_to_a_tuple():
    """So the config stays hashable and a caller's list cannot be mutated out
    from under a running shard."""

    config = base(race_schedule=[32, 256])
    assert config.race_schedule == (32, 256)


def test_a_schedule_that_stops_short_of_the_batch_is_refused():
    with pytest.raises(ValueError, match="evaluation_samples"):
        base(race_schedule=(32, 64))


def test_a_schedule_that_does_not_strictly_increase_is_refused():
    for schedule in [(32, 32, 256), (64, 32, 256)]:
        with pytest.raises(ValueError, match="strictly increasing"):
            base(race_schedule=schedule)


def test_a_non_positive_checkpoint_is_refused():
    with pytest.raises(ValueError, match="positive"):
        base(race_schedule=(0, 256))


def test_a_threshold_without_a_schedule_is_refused():
    """The same reasoning as a margin with no boundary to widen: it reads as a
    tuned elimination rule and does nothing."""

    with pytest.raises(ValueError, match="race_lcb_z"):
        base(race_lcb_z=2.0)


def test_a_negative_or_non_finite_threshold_is_refused():
    for z in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="race_lcb_z"):
            base(race_schedule=(32, 256), race_lcb_z=z)


def test_a_boolean_is_not_a_number_here():
    with pytest.raises(TypeError, match="race_lcb_z"):
        base(race_schedule=(32, 256), race_lcb_z=True)
    with pytest.raises(TypeError, match="race_schedule"):
        base(race_schedule=True)


def test_the_schedule_survives_a_dataclasses_replace():
    """How the gate builds its racing arm from its uniform one, so the
    validation has to run on the replaced copy rather than only at first
    construction."""

    raced = dataclasses.replace(base(), race_schedule=(32, 64, 128, 256),
                                race_lcb_z=2.0)
    assert raced.race_schedule == (32, 64, 128, 256)
    with pytest.raises(ValueError, match="evaluation_samples"):
        dataclasses.replace(base(), race_schedule=(32, 64))
