from __future__ import annotations

import pickle
import struct

import pytest

from ofc_regular.cards import ALL_CARDS
from ofc_regular.hu_rl_contract import HU_RL_ACTOR_VIEW_SCHEMA, MAX_LEGAL_ACTIONS
from ofc_regular.hu_rl_native import (
    BATCH_STEP_OUTCOME_SCHEMA,
    LEGAL_ACTION_BATCH_SCHEMA,
    MAX_BATCH_LANES,
    PACKED_ACTION_BYTES,
    PACKED_HISTORY_RECORD_BYTES,
    PACKED_OBSERVATION_PREFIX_BYTES,
    PACKED_OBSERVATION_RECORD_BYTES,
    PACKED_SCORING_IDENTITY,
    PACKED_STEP_RECORD_BYTES,
    HuRlNativeContractError,
    LegalActionBatchV1,
    NativeBatchHuRlEnvV1,
    PackedActorDecisionBatchV1,
    PackedActorObservationBatchV1,
    PackedLegalActionBatchV1,
    PackedStepBatchV1,
    _validate_public_step_result,
    native_available,
)
from ofc_regular.hu_rl_native_benchmark import generate_benchmark_decks


pytestmark = pytest.mark.skipif(not native_available(), reason="optional PyO3 extension not built")


def _decks() -> list[list[str]]:
    base = list(ALL_CARDS)
    return [base, base[7:] + base[:7], list(reversed(base))]


def _paired_python_oracle(seed_base: int, global_pair_start: int, pair_count: int, seed_stride: int) -> list[list[str]]:
    decks: list[list[str]] = []
    windows = ((0, 5, 5, 10), (10, 13, 13, 16), (16, 19, 19, 22), (22, 25, 25, 28), (28, 31, 31, 34))
    for local_pair in range(pair_count):
        pair_seed = seed_base + (global_pair_start + local_pair) * seed_stride
        ab = generate_benchmark_decks(lane_count=1, seed=pair_seed)[0]
        ba = list(ab)
        for a0, a1, b0, b1 in windows:
            ba[a0:a1], ba[b0:b1] = ab[b0:b1], ab[a0:a1]
        assert ba[34:] == ab[34:]
        decks.extend((ab, ba))
    return decks


def _card_mask(cards: list[str] | tuple[str, ...]) -> int:
    return sum(1 << ALL_CARDS.index(card) for card in cards)


def _json_mask(value: str) -> int:
    return int(value, 16)


def _assert_no_cross_actor_private_fields(value: object) -> None:
    forbidden = {
        "action_key",
        "audit_truth",
        "deck",
        "deck_tail",
        "discard_card",
        "discard_cards",
        "discard_mask",
        "opponent_private_discard",
        "opponent_private_discards",
        "world_state",
    }
    if isinstance(value, dict):
        assert forbidden.isdisjoint(value)
        for nested in value.values():
            _assert_no_cross_actor_private_fields(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            _assert_no_cross_actor_private_fields(nested)


def test_native_batch_lane_order_fixed_action_geometry_and_safe_step() -> None:
    env = NativeBatchHuRlEnvV1(_decks(), chunk_width=2, thread_count=7)
    assert env.lane_count == 3
    views = env.observe_batch()
    assert tuple(view["schema"] for view in views) == (HU_RL_ACTOR_VIEW_SCHEMA,) * 3
    assert tuple(tuple(view["observation"]["dealt_cards"]) for view in views) == (
        tuple(sorted(_decks()[0][:5], key=ALL_CARDS.index)),
        tuple(sorted(_decks()[1][:5], key=ALL_CARDS.index)),
        tuple(sorted(_decks()[2][:5], key=ALL_CARDS.index)),
    )
    for view in views:
        _assert_no_cross_actor_private_fields(view)

    legal = env.legal_actions_batch()
    assert legal.schema == LEGAL_ACTION_BATCH_SCHEMA
    assert legal.lane_count == 3
    assert all(len(row) == MAX_LEGAL_ACTIONS for row in legal.action_keys)
    assert all(len(row) == MAX_LEGAL_ACTIONS for row in legal.mask)
    assert all(count == 232 for count in legal.action_counts)
    for lane, view in enumerate(views):
        mapping = view["legal_action_mapping"]
        assert tuple(mapping["action_keys"]) == legal.action_keys[lane][: legal.action_counts[lane]]
        assert mapping["action_set_digest"] == legal.action_set_digests[lane]
        assert mapping["action_order_digest"] == legal.action_order_digests[lane]

    selected = [row[0] for row in legal.action_keys]
    assert all(isinstance(token, str) for token in selected)
    results = env.step_batch(selected)  # type: ignore[arg-type]
    assert env.decision_counts == (1, 1, 1)
    for result in results:
        assert result["schema"] == BATCH_STEP_OUTCOME_SCHEMA
        assert set(result) == {
            "schema",
            "actor",
            "street",
            "public_placement",
            "done",
            "rewards",
        }
        _assert_no_cross_actor_private_fields(result)


def test_snapshot_restore_is_opaque_and_replays_exact_lane_order() -> None:
    env = NativeBatchHuRlEnvV1(_decks())
    for _ in range(3):
        legal = env.legal_actions_batch()
        env.step_batch([row[0] for row in legal.action_keys])  # type: ignore[list-item]
    checkpoint = env.snapshot_batch()
    checkpoint_views = env.observe_batch()
    assert "<redacted>" in repr(checkpoint)
    assert not hasattr(checkpoint, "__dict__")
    with pytest.raises((TypeError, pickle.PicklingError)):
        pickle.dumps(checkpoint)

    for _ in range(2):
        legal = env.legal_actions_batch()
        env.step_batch([row[-1 if count == MAX_LEGAL_ACTIONS else count - 1] for row, count in zip(
            legal.action_keys, legal.action_counts, strict=True
        )])  # type: ignore[list-item]
    env.restore_batch(checkpoint)
    assert env.decision_counts == (3, 3, 3)
    assert env.observe_batch() == checkpoint_views


def test_full_hand_terminal_rewards_are_zero_sum_and_reset_reopens_all_lanes() -> None:
    env = NativeBatchHuRlEnvV1(_decks())
    terminal_results = ()
    for decision in range(10):
        legal = env.legal_actions_batch()
        selected = [
            row[(lane + decision) % count]
            for lane, (row, count) in enumerate(
                zip(legal.action_keys, legal.action_counts, strict=True)
            )
        ]
        terminal_results = env.step_batch(selected)  # type: ignore[arg-type]
    assert env.all_done
    assert env.decision_counts == (10, 10, 10)
    assert all(result["done"] for result in terminal_results)
    for result in terminal_results:
        first, second = result["rewards"]
        assert first == -second
        _assert_no_cross_actor_private_fields(result)
    with pytest.raises(RuntimeError, match="observe_batch failed closed"):
        env.observe_batch()

    reset_views = env.reset_batch(_decks())
    assert not env.all_done
    assert env.decision_counts == (0, 0, 0)
    assert tuple(view["schema"] for view in reset_views) == (HU_RL_ACTOR_VIEW_SCHEMA,) * 3


def test_reset_and_step_fail_closed_without_leaking_rejected_values() -> None:
    env = NativeBatchHuRlEnvV1(_decks())
    before = env.observe_batch()
    duplicate = _decks()
    duplicate[1][-1] = duplicate[1][0]
    with pytest.raises(ValueError) as reset_error:
        env.reset_batch(duplicate)
    assert "duplicate" not in str(reset_error.value).lower()
    assert duplicate[1][0] not in str(reset_error.value)
    assert env.observe_batch() == before

    legal = env.legal_actions_batch()
    invalid = [row[0] for row in legal.action_keys]
    invalid[1] = "rak1:0000000000000:0000000000000:0000000000000:0000000000000"
    with pytest.raises(ValueError) as step_error:
        env.step_batch(invalid)  # type: ignore[arg-type]
    assert invalid[1] not in str(step_error.value)
    assert env.observe_batch() == before


def test_native_boundary_rejects_more_than_4096_lanes_before_deck_parsing() -> None:
    import _ofc_hu_rl_engine as native

    malformed_lane = [object()] * 52
    oversized = [malformed_lane] * (MAX_BATCH_LANES + 1)
    with pytest.raises(ValueError, match="construction rejected input"):
        native.BatchHuRlEnv(oversized)


def test_reset_requires_fixed_batch_width() -> None:
    env = NativeBatchHuRlEnvV1(_decks())
    with pytest.raises(HuRlNativeContractError):
        env.reset_batch(_decks()[:2])


def test_snapshot_lineage_rejects_cross_batch_and_pre_reset_restore() -> None:
    first = NativeBatchHuRlEnvV1(_decks())
    second = NativeBatchHuRlEnvV1(_decks())
    snapshot = first.snapshot_batch()
    second_before = second.observe_batch()
    with pytest.raises(ValueError, match="restore_batch rejected input"):
        second.restore_batch(snapshot)
    assert second.observe_batch() == second_before

    first.reset_batch(_decks())
    reset_views = first.observe_batch()
    with pytest.raises(ValueError, match="restore_batch rejected input"):
        first.restore_batch(snapshot)
    assert first.observe_batch() == reset_views


def test_native_boundary_rejects_thread_count_above_frozen_cap() -> None:
    import _ofc_hu_rl_engine as native

    with pytest.raises(ValueError, match="construction rejected input"):
        native.BatchHuRlEnv(_decks(), thread_count=native.MAX_BATCH_THREADS + 1)


def test_python_boundary_rejects_mapping_digest_and_public_result_tampering() -> None:
    env = NativeBatchHuRlEnvV1(_decks()[:1])
    legal = env.legal_actions_batch()
    with pytest.raises(HuRlNativeContractError, match="mapping digest disagrees"):
        LegalActionBatchV1(
            action_keys=legal.action_keys,
            mask=legal.mask,
            action_counts=legal.action_counts,
            action_set_digests=("0" * 64,),
            action_order_digests=legal.action_order_digests,
        )

    selected = [legal.action_keys[0][0]]
    result = dict(env.step_batch(selected)[0])  # type: ignore[arg-type]
    result["actor"] = 1
    with pytest.raises(HuRlNativeContractError, match="identity disagrees"):
        _validate_public_step_result(result)


@pytest.mark.parametrize(
    ("lane_count", "chunk_width", "thread_count"),
    ((1, 1, 1), (3, 2, 7), (3, 8, 3)),
)
def test_packed_boundary_matches_every_v1_field_action_digest_and_step(
    lane_count: int, chunk_width: int, thread_count: int
) -> None:
    decks = _decks()[:lane_count]
    json_env = NativeBatchHuRlEnvV1(
        decks, chunk_width=chunk_width, thread_count=thread_count
    )
    packed_env = NativeBatchHuRlEnvV1(
        decks, chunk_width=chunk_width, thread_count=thread_count
    )

    for decision in range(10):
        json_views = json_env.observe_batch()
        packed_views = packed_env.observe_batch_packed()
        assert packed_views.buffer.readonly
        assert packed_views.buffer.nbytes == lane_count * PACKED_OBSERVATION_RECORD_BYTES
        for lane, json_view in enumerate(json_views):
            packed = packed_views.lane(lane)
            observation = json_view["observation"]
            assert packed.hero_board_masks == tuple(
                _card_mask(tuple(observation["hero_board"][row]))
                for row in ("top", "middle", "bottom")
            )
            assert packed.opponent_public_board_masks == tuple(
                _card_mask(tuple(observation["opponent_public_board"][row]))
                for row in ("top", "middle", "bottom")
            )
            assert packed.hero_private_discards_mask == _card_mask(
                tuple(observation["hero_private_discards"])
            )
            assert packed.dealt_cards_mask == _card_mask(
                tuple(observation["dealt_cards"])
            )
            assert packed.seat == observation["seat"]
            assert packed.street == observation["street"]
            assert packed.to_act_order == observation["to_act_order"]
            assert packed.hero_in_fantasyland == observation["hero_in_fantasyland"]
            assert (
                packed.opponent_in_fantasyland
                == observation["opponent_in_fantasyland"]
            )
            assert packed.opponent_discard_count == observation["opponent_discard_count"]
            assert packed.scoring_identity == PACKED_SCORING_IDENTITY
            assert len(packed.public_history) == len(json_view["public_history"])
            for packed_event, json_event in zip(
                packed.public_history, json_view["public_history"], strict=True
            ):
                assert packed_event.street == json_event["street"]
                assert packed_event.acting_seat == json_event["acting_seat"]
                assert packed_event.placement_masks == (
                    _json_mask(json_event["top_placement_mask"]),
                    _json_mask(json_event["middle_placement_mask"]),
                    _json_mask(json_event["bottom_placement_mask"]),
                )
                assert packed_event.discard_count == json_event["discard_count"]
                assert packed_event.to_public_placement().to_dict() == json_event
            assert set(vars(packed)) == {
                "hero_board_masks",
                "opponent_public_board_masks",
                "hero_private_discards_mask",
                "dealt_cards_mask",
                "seat",
                "street",
                "to_act_order",
                "hero_in_fantasyland",
                "opponent_in_fantasyland",
                "opponent_discard_count",
                "scoring_identity",
                "public_history",
            }

        json_legal = json_env.legal_actions_batch()
        packed_legal = packed_env.legal_actions_batch_packed()
        assert packed_legal.action_keys_buffer.readonly
        assert packed_legal.mask_buffer.readonly
        for lane in range(lane_count):
            count = json_legal.action_counts[lane]
            assert packed_legal.action_count(lane) == count
            assert packed_legal.action_set_digest(lane) == json_legal.action_set_digests[lane]
            assert (
                packed_legal.action_order_digest(lane)
                == json_legal.action_order_digests[lane]
            )
            for action_index in range(count):
                assert packed_legal.action_key_at(lane, action_index).to_token() == (
                    json_legal.action_keys[lane][action_index]
                )

        indices = tuple(
            (lane + decision * 7) % packed_legal.action_count(lane)
            for lane in range(lane_count)
        )
        selected_tokens = [
            json_legal.action_keys[lane][index]
            for lane, index in enumerate(indices)
        ]
        json_results = json_env.step_batch(selected_tokens)  # type: ignore[arg-type]
        packed_results = packed_env.step_batch_packed(packed_legal.select(indices))
        assert packed_results.buffer.readonly
        assert packed_results.buffer.nbytes == lane_count * PACKED_STEP_RECORD_BYTES
        for lane, json_result in enumerate(json_results):
            packed_result = packed_results.lane(lane)
            placement = json_result["public_placement"]
            assert packed_result.actor == json_result["actor"]
            assert packed_result.street == json_result["street"]
            assert packed_result.done == json_result["done"]
            assert packed_result.placement_masks == (
                _json_mask(placement["top_placement_mask"]),
                _json_mask(placement["middle_placement_mask"]),
                _json_mask(placement["bottom_placement_mask"]),
            )
            assert packed_result.discard_count == placement["discard_count"]
            assert packed_result.rewards == tuple(json_result["rewards"])

    assert json_env.all_done and packed_env.all_done
    assert json_env.decision_counts == packed_env.decision_counts


def test_packed_boundary_rejects_length_padding_tampering_and_is_atomic() -> None:
    env = NativeBatchHuRlEnvV1(_decks()[:1])
    observation = env.observe_batch_packed()
    with pytest.raises(HuRlNativeContractError, match="geometry mismatch"):
        PackedActorObservationBatchV1(observation.payload[:-1], 1)
    tampered_observation = bytearray(observation.payload)
    tampered_observation[PACKED_OBSERVATION_PREFIX_BYTES] = 1
    with pytest.raises(HuRlNativeContractError, match="padding is nonzero"):
        PackedActorObservationBatchV1(bytes(tampered_observation), 1)
    tampered_observation = bytearray(observation.payload)
    tampered_observation[71] = 1
    with pytest.raises(HuRlNativeContractError, match="reserved byte"):
        PackedActorObservationBatchV1(bytes(tampered_observation), 1)

    legal = env.legal_actions_batch_packed()
    with pytest.raises(HuRlNativeContractError, match="geometry mismatch"):
        PackedLegalActionBatchV1(
            legal.action_keys[:-1],
            legal.mask,
            legal.action_counts,
            legal.action_set_digests,
            legal.action_order_digests,
            1,
        )
    tampered_mask = bytearray(legal.mask)
    tampered_mask[0] = 0
    with pytest.raises(HuRlNativeContractError, match="mask/count disagree"):
        PackedLegalActionBatchV1(
            legal.action_keys,
            bytes(tampered_mask),
            legal.action_counts,
            legal.action_set_digests,
            legal.action_order_digests,
            1,
        )

    before = env.observe_batch_packed().payload
    valid = legal.select((0,))
    with pytest.raises(HuRlNativeContractError, match="geometry mismatch"):
        env.step_batch_packed(valid[:-1])
    outside_card_domain = bytearray(valid)
    struct.pack_into("<Q", outside_card_domain, 0, 1 << 63)
    with pytest.raises(ValueError, match="step_batch_packed rejected input"):
        env.step_batch_packed(outside_card_domain)
    overlapping = bytearray(valid)
    struct.pack_into("<4Q", overlapping, 0, 1, 1, 0, 0)
    with pytest.raises(ValueError, match="step_batch_packed rejected input"):
        env.step_batch_packed(memoryview(overlapping))
    assert env.observe_batch_packed().payload == before

    result = env.step_batch_packed(valid)
    with pytest.raises(HuRlNativeContractError, match="geometry mismatch"):
        PackedStepBatchV1(result.payload[:-1], 1)
    tampered_step = bytearray(result.payload)
    tampered_step[4] = 1
    with pytest.raises(HuRlNativeContractError, match="metadata is invalid"):
        PackedStepBatchV1(bytes(tampered_step), 1)


def test_packed_legal_padding_is_zero_and_tampering_is_rejected() -> None:
    env = NativeBatchHuRlEnvV1(_decks()[:1])
    for _ in range(2):
        legal = env.legal_actions_batch_packed()
        env.step_batch_packed(legal.select((0,)))
    legal = env.legal_actions_batch_packed()
    count = legal.action_count(0)
    assert count < MAX_LEGAL_ACTIONS
    padding_start = count * PACKED_ACTION_BYTES
    assert legal.action_keys[padding_start:] == bytes(
        (MAX_LEGAL_ACTIONS - count) * PACKED_ACTION_BYTES
    )
    tampered = bytearray(legal.action_keys)
    tampered[padding_start] = 1
    with pytest.raises(HuRlNativeContractError, match="padding is nonzero"):
        PackedLegalActionBatchV1(
            bytes(tampered),
            legal.mask,
            legal.action_counts,
            legal.action_set_digests,
            legal.action_order_digests,
            1,
        )

    invalid_used = bytearray(legal.action_keys)
    struct.pack_into("<Q", invalid_used, 0, 1 << 63)
    invalid_batch = PackedLegalActionBatchV1(
        bytes(invalid_used),
        legal.mask,
        legal.action_counts,
        legal.action_set_digests,
        legal.action_order_digests,
        1,
    )
    with pytest.raises(HuRlNativeContractError, match="ActionKey is invalid"):
        invalid_batch.select((0,))


@pytest.mark.parametrize(
    ("lane_count", "chunk_width", "thread_count"),
    ((1, 1, 1), (3, 2, 7), (3, 8, 3)),
)
def test_combined_actor_decision_is_byte_exact_with_separate_packed_calls(
    lane_count: int, chunk_width: int, thread_count: int
) -> None:
    env = NativeBatchHuRlEnvV1(
        _decks()[:lane_count],
        chunk_width=chunk_width,
        thread_count=thread_count,
    )
    for decision in range(10):
        combined = env.actor_decision_batch_packed()
        observations = env.observe_batch_packed()
        legal = env.legal_actions_batch_packed()
        assert combined.lane_count == lane_count
        assert combined.observations.payload == observations.payload
        assert combined.legal_actions.action_keys == legal.action_keys
        assert combined.legal_actions.mask == legal.mask
        assert combined.legal_actions.action_counts == legal.action_counts
        assert combined.legal_actions.action_set_digests == legal.action_set_digests
        assert combined.legal_actions.action_order_digests == legal.action_order_digests
        indices = tuple(
            (decision * 11 + lane) % combined.legal_actions.action_count(lane)
            for lane in range(lane_count)
        )
        env.step_batch_packed(combined.legal_actions.select(indices))
    assert env.all_done


def test_combined_actor_decision_raw_geometry_is_six_immutable_buffers() -> None:
    import _ofc_hu_rl_engine as native

    raw_env = native.BatchHuRlEnv(_decks(), chunk_width=2, thread_count=3)
    raw = raw_env.actor_decision_batch_packed()
    assert type(raw) is tuple
    assert len(raw) == 6
    assert all(type(buffer) is bytes for buffer in raw)
    assert tuple(map(len, raw)) == (
        3 * PACKED_OBSERVATION_RECORD_BYTES,
        3 * MAX_LEGAL_ACTIONS * PACKED_ACTION_BYTES,
        3 * MAX_LEGAL_ACTIONS,
        3,
        3 * 32,
        3 * 32,
    )


def test_combined_actor_decision_rejects_container_geometry_and_tampering() -> None:
    one = NativeBatchHuRlEnvV1(_decks()[:1]).actor_decision_batch_packed()
    two = NativeBatchHuRlEnvV1(_decks()[:2]).actor_decision_batch_packed()
    with pytest.raises(HuRlNativeContractError, match="lane geometry disagrees"):
        PackedActorDecisionBatchV1(one.observations, two.legal_actions)
    with pytest.raises(HuRlNativeContractError, match="unsupported.*schema"):
        PackedActorDecisionBatchV1(
            one.observations,
            one.legal_actions,
            schema="regular_ofc_hu_rl_packed_boundary_v0",
        )

    tampered = bytearray(one.observations.payload)
    tampered[71] = 1
    with pytest.raises(HuRlNativeContractError, match="reserved byte"):
        PackedActorDecisionBatchV1(
            PackedActorObservationBatchV1(bytes(tampered), 1),
            one.legal_actions,
        )


@pytest.mark.parametrize(
    "seed_base",
    (0, (1 << 32) - 1, 1 << 32, (1 << 63) - 1),
)
def test_rust_paired_seed_path_is_full_episode_exact_with_python_explicit_oracle(
    seed_base: int,
) -> None:
    explicit_decks = _paired_python_oracle(seed_base, 0, 1, 1)
    seeded = NativeBatchHuRlEnvV1.from_paired_seed_range(
        seed_base=seed_base,
        global_pair_start=0,
        pair_count=1,
        seed_stride=1,
        chunk_width=1,
        thread_count=2,
    )
    explicit = NativeBatchHuRlEnvV1(explicit_decks, chunk_width=2, thread_count=1)
    assert seeded.lane_count == explicit.lane_count == 2
    assert "hidden_state='<redacted>'" in repr(seeded._native)
    for forbidden in ("seed_base", "deck_tail", "explicit_deck", explicit_decks[0][34]):
        assert forbidden not in repr(seeded._native)

    for decision in range(10):
        seeded_boundary = seeded.actor_decision_batch_packed()
        explicit_boundary = explicit.actor_decision_batch_packed()
        assert seeded_boundary.observations.payload == explicit_boundary.observations.payload
        assert seeded_boundary.legal_actions == explicit_boundary.legal_actions
        indices = tuple(
            (decision * 13 + lane) % seeded_boundary.legal_actions.action_count(lane)
            for lane in range(2)
        )
        selected = seeded_boundary.legal_actions.select(indices)
        assert selected == explicit_boundary.legal_actions.select(indices)
        assert seeded.step_batch_packed(selected) == explicit.step_batch_packed(selected)
    assert seeded.all_done and explicit.all_done


def test_paired_seed_reset_rejects_invalid_ranges_atomically_and_without_leakage() -> None:
    env = NativeBatchHuRlEnvV1.from_paired_seed_range(
        seed_base=123,
        global_pair_start=4,
        pair_count=2,
        seed_stride=7,
    )
    first = env.actor_decision_batch_packed()
    env.step_batch_packed(first.legal_actions.select((0, 0, 0, 0)))
    before = env.observe_batch_packed()
    before_counts = env.decision_counts
    checkpoint = env.snapshot_batch()
    bad_ranges = (
        {"seed_base": True, "global_pair_start": 0, "pair_count": 2, "seed_stride": 1},
        {"seed_base": -1, "global_pair_start": 0, "pair_count": 2, "seed_stride": 1},
        {"seed_base": 0, "global_pair_start": 0, "pair_count": 1, "seed_stride": 1},
        {"seed_base": (1 << 63) - 1, "global_pair_start": 1, "pair_count": 2, "seed_stride": 1},
    )
    for values in bad_ranges:
        with pytest.raises((TypeError, ValueError)) as caught:
            env.reset_from_paired_seed_range(**values)
        error = str(caught.value)
        for forbidden in ("9223372036854775807", "seed_base=True", "deck", "tail", "As"):
            assert forbidden not in error
        assert env.observe_batch_packed() == before
        assert env.decision_counts == before_counts
        env.restore_batch(checkpoint)

    replacement = _paired_python_oracle(999, 8, 2, 5)
    reset_views = env.reset_from_paired_seed_range(
        seed_base=999,
        global_pair_start=8,
        pair_count=2,
        seed_stride=5,
    )
    explicit = NativeBatchHuRlEnvV1(replacement)
    assert reset_views == explicit.observe_batch()
    with pytest.raises(ValueError, match="restore_batch rejected input"):
        env.restore_batch(checkpoint)
    assert env.observe_batch() == explicit.observe_batch()


def test_raw_pyo3_paired_seed_api_rejects_non_exact_integers_before_mutation() -> None:
    import _ofc_hu_rl_engine as native

    for invalid in (True, 1.0, "1", None):
        with pytest.raises(TypeError, match="rejected input type"):
            native.BatchHuRlEnv.from_paired_seed_range(invalid, 0, 1, 1)
    with pytest.raises(ValueError, match="rejected input"):
        native.BatchHuRlEnv.from_paired_seed_range(-1, 0, 1, 1)

    env = native.BatchHuRlEnv.from_paired_seed_range(55, 3, 2, 7)
    before = env.observe_batch_packed()
    before_counts = env.decision_counts
    checkpoint = env.snapshot_batch()
    invalid_resets = (
        (True, 0, 2, 1),
        (-1, 0, 2, 1),
        (0, 0, 1, 1),
        ((1 << 63) - 1, 1, 2, 1),
    )
    for values in invalid_resets:
        with pytest.raises((TypeError, ValueError)) as caught:
            env.reset_from_paired_seed_range(*values)
        assert "9223372036854775807" not in str(caught.value)
        assert "deck" not in str(caught.value).lower()
        assert env.observe_batch_packed() == before
        assert env.decision_counts == before_counts
        env.restore_batch(checkpoint)
