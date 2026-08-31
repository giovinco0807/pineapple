import copy
import json
from pathlib import Path

import pytest

from ai.engine.encoding import ALL_CARDS
from ai.tutor.behavior_calibration_contract import (
    canonical_json,
    commit_hidden_root_trace,
    root_split,
)
from ai.tutor.collect_hu_behavior_traces import (
    CALLER_SUPPLIED_DECKS,
    DETERMINISTIC_JOKER_CYCLE,
    NATURAL_UNIFORM_SHUFFLE,
    TARGETED_JOKER_CHALLENGE,
    BehaviorTraceCollectionConfig,
    build_root_deal_plan,
    canonical_t0_action,
    collect_hu_behavior_traces,
    read_behavior_trace_dataset,
    resume_behavior_trace_dataset,
    t0_baseline_manifest,
    t0_baseline_sha256,
    targeted_joker_cell,
    verify_collected_behavior_traces,
    write_behavior_trace_dataset,
)
from ai.tutor.exact_late import action_key
from ai.tutor.t3_hu_full_card_range import UniformLegalBehaviorModel


def _forced_deck(constraints):
    constraints = dict(constraints)
    if len(set(constraints.values())) != len(constraints):
        raise AssertionError("test deck constraints duplicate a card")
    remaining = iter(card for card in ALL_CARDS if card not in set(constraints.values()))
    deck = [constraints[index] if index in constraints else next(remaining) for index in range(54)]
    assert len(deck) == len(set(deck)) == 54
    assert set(deck) == set(ALL_CARDS)
    return tuple(deck)


def _challenge_config(namespace="collector-tests/challenge-v1"):
    return BehaviorTraceCollectionConfig(
        seed_namespace=namespace,
        root_sampling_mode=TARGETED_JOKER_CHALLENGE,
        challenge_id="test-forced-joker-cells-v1",
        challenge_deck_source=CALLER_SUPPLIED_DECKS,
    )


def _two_forced_decks():
    return {
        # Root 0: T1 BB receives both distinct physical Jokers.
        0: _forced_deck({10: "X1", 11: "X2"}),
        # Root 1: T2 BTN receives X1 while X2 remains outside logged draws.
        1: _forced_deck({19: "X1", 53: "X2"}),
    }


def _walk_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


def test_two_root_core_collects_all_roles_jokers_and_separate_hidden_preimages():
    config = _challenge_config()
    dataset = collect_hu_behavior_traces(
        config,
        UniformLegalBehaviorModel(),
        root_count=2,
        forced_decks=_two_forced_decks(),
    )

    assert len(dataset.records) == 8
    assert len(dataset.hidden_roots) == 2
    assert [(record["turn"], record["actor"]) for record in dataset.records] == [
        (1, "bb"),
        (1, "btn"),
        (2, "bb"),
        (2, "btn"),
    ] * 2
    assert dataset.records[0]["visible_joker_count"] == 2
    assert dataset.records[7]["visible_joker_count"] == 1
    assert dataset.manifest["policy_query_count"] == 8
    assert dataset.manifest["model_evaluation_count"] == 8
    assert dataset.manifest["elapsed_runtime_ns"] >= 0
    assert dataset.manifest["collection_config"]["root_sampling_mode"] == TARGETED_JOKER_CHALLENGE
    assert dataset.manifest["hidden_root_artifact"]["preimage_in_decision_records"] is False

    for trace in dataset.hidden_roots:
        root_records = [record for record in dataset.records if record["root_id"] == trace["root_id"]]
        commitment = commit_hidden_root_trace(trace)
        assert len(root_records) == 4
        assert {record["root_commitment"] for record in root_records} == {commitment}
        assert trace["physical_deck_order"].count("X1") == 1
        assert trace["physical_deck_order"].count("X2") == 1
        expected_btn_seat = trace["root_index"] % 2
        assert trace["seat_assignment"] == {"bb": 1 - expected_btn_seat, "btn": expected_btn_seat}
        assert {(item["turn"], item["actor"]) for item in trace["deal_sequence"]} == {
            (0, "bb"), (0, "btn"), (1, "bb"), (1, "btn"), (2, "bb"), (2, "btn")
        }

    forbidden = {
        "hidden_full_trace",
        "physical_deck_order",
        "opponent_private",
        "opponent_discards",
        "remaining_deck",
        "undealt_cards",
        "root_commitment",
        "root_id",
        "seed_namespace",
    }
    for record in dataset.records:
        assert forbidden.isdisjoint(set(_walk_keys(record["information"])))
        history_pairs = [
            (entry["turn"], entry["actor"])
            for entry in record["information"]["public_action_history"]
        ]
        expected = [(0, "bb"), (0, "btn")]
        if record["turn"] == 2:
            expected += [(1, "bb"), (1, "btn")]
        if record["actor"] == "btn":
            expected += [(record["turn"], "bb")]
        assert history_pairs == expected
    assert verify_collected_behavior_traces(dataset)["record_count"] == 8


def test_collection_is_reproducible_except_observed_wall_runtime():
    config = _challenge_config("collector-tests/replay-v1")
    forced = _two_forced_decks()
    first = collect_hu_behavior_traces(
        config, UniformLegalBehaviorModel(), root_count=2, forced_decks=forced
    )
    second = collect_hu_behavior_traces(
        config, UniformLegalBehaviorModel(), root_count=2, forced_decks=forced
    )

    assert first.records == second.records
    assert first.hidden_roots == second.hidden_roots
    assert (
        first.manifest["collection_content_sha256"]
        == second.manifest["collection_content_sha256"]
    )
    assert first.manifest["record_order_sha256"] == second.manifest["record_order_sha256"]
    assert first.manifest["root_id_order_sha256"] == second.manifest["root_id_order_sha256"]
    # Every decision of one root is immutable in one preregistered split.
    for trace in first.hidden_roots:
        splits = {
            record["split"]
            for record in first.records
            if record["root_id"] == trace["root_id"]
        }
        assert splits == {root_split(trace["root_id"])}


def test_t0_baseline_is_content_addressed_and_uses_only_own_draw():
    own = ("2h", "3h", "4h", "5h", "X1")
    first = canonical_t0_action(own)
    second = canonical_t0_action(tuple(reversed(own)))
    assert action_key(first) == action_key(second)
    manifest = t0_baseline_manifest()
    assert manifest["opponent_private_input"] is False
    assert manifest["opponent_public_input"] is False
    assert manifest["hidden_deck_input"] is False
    assert len(t0_baseline_sha256()) == 64


def test_natural_and_targeted_populations_cannot_mix_or_accept_wrong_deck_source():
    natural = BehaviorTraceCollectionConfig(
        seed_namespace="collector-tests/natural-v1",
        root_sampling_mode=NATURAL_UNIFORM_SHUFFLE,
    )
    natural_plan = build_root_deal_plan(natural, 0)
    assert len(natural_plan.deck) == len(set(natural_plan.deck)) == 54
    with pytest.raises(ValueError, match="rejects forced decks"):
        build_root_deal_plan(natural, 0, forced_deck=_forced_deck({10: "X1", 11: "X2"}))
    with pytest.raises(ValueError, match="natural collection"):
        BehaviorTraceCollectionConfig(
            seed_namespace="bad",
            root_sampling_mode=NATURAL_UNIFORM_SHUFFLE,
            challenge_id="must-not-mix",
        )
    with pytest.raises(ValueError, match="normalized lowercase"):
        BehaviorTraceCollectionConfig(
            seed_namespace="bad-target",
            root_sampling_mode=TARGETED_JOKER_CHALLENGE,
            challenge_id="m3-joker-challenge-v1/",
            challenge_deck_source=DETERMINISTIC_JOKER_CYCLE,
        )

    targeted = _challenge_config("collector-tests/no-mix-v1")
    existing = collect_hu_behavior_traces(
        targeted,
        UniformLegalBehaviorModel(),
        root_count=1,
        forced_decks={0: _two_forced_decks()[0]},
    )
    with pytest.raises(ValueError, match="resume config"):
        collect_hu_behavior_traces(
            natural,
            UniformLegalBehaviorModel(),
            root_count=1,
            root_index_start=1,
            existing_dataset=existing,
        )


def test_deterministic_challenge_has_prefixed_ids_and_all_twelve_target_cells():
    config = BehaviorTraceCollectionConfig(
        seed_namespace="collector-tests/m3-joker-grid-v1",
        root_sampling_mode=TARGETED_JOKER_CHALLENGE,
        challenge_id="m3-joker-challenge-v1",
        challenge_deck_source=DETERMINISTIC_JOKER_CYCLE,
    )
    dataset = collect_hu_behavior_traces(
        config,
        UniformLegalBehaviorModel(),
        root_count=12,
    )

    expected_cells = {
        (turn, actor, joker_count)
        for turn in (1, 2)
        for actor in ("bb", "btn")
        for joker_count in (0, 1, 2)
    }
    observed_targets = {
        (
            trace["challenge_target"]["turn"],
            trace["challenge_target"]["actor"],
            trace["challenge_target"]["visible_joker_count"],
        )
        for trace in dataset.hidden_roots
    }
    assert observed_targets == expected_cells
    assert {targeted_joker_cell(index) for index in range(12)} == expected_cells
    for trace in dataset.hidden_roots:
        assert trace["root_id"].startswith("m3-joker-challenge-v1/")
        target = trace["challenge_target"]
        record = next(
            record
            for record in dataset.records
            if record["root_id"] == trace["root_id"]
            and record["turn"] == target["turn"]
            and record["actor"] == target["actor"]
        )
        assert record["visible_joker_count"] == target["visible_joker_count"]


def test_atomic_write_readback_and_decision_root_manifest_tamper_fail_closed(tmp_path):
    config = _challenge_config("collector-tests/io-v1")
    dataset = collect_hu_behavior_traces(
        config,
        UniformLegalBehaviorModel(),
        root_count=1,
        forced_decks={0: _two_forced_decks()[0]},
    )
    decisions_path = tmp_path / "decisions.jsonl"
    readback = write_behavior_trace_dataset(dataset, decisions_path)
    assert readback.records == dataset.records
    assert readback.hidden_roots == dataset.hidden_roots
    assert readback.manifest == dataset.manifest
    assert (tmp_path / "roots.jsonl").is_file()
    assert (tmp_path / "manifest.json").is_file()

    # Decision tamper, even canonicalized, is rejected by the bound record hash.
    rows = [json.loads(line) for line in decisions_path.read_text(encoding="utf-8").splitlines()]
    rows[0]["visible_joker_count"] = (rows[0]["visible_joker_count"] + 1) % 3
    decisions_path.write_text(
        "\n".join(canonical_json(row) for row in rows) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="record SHA-256 mismatch"):
        read_behavior_trace_dataset(decisions_path)

    write_behavior_trace_dataset(dataset, decisions_path)
    root_path = tmp_path / "roots.jsonl"
    roots = [json.loads(line) for line in root_path.read_text(encoding="utf-8").splitlines()]
    roots[0]["physical_deck_order"][30], roots[0]["physical_deck_order"][31] = (
        roots[0]["physical_deck_order"][31], roots[0]["physical_deck_order"][30]
    )
    root_path.write_text(
        "\n".join(canonical_json(root) for root in roots) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="preimage hash"):
        read_behavior_trace_dataset(decisions_path)

    write_behavior_trace_dataset(dataset, decisions_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["elapsed_runtime_ns"] += 1
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="manifest SHA-256 mismatch"):
        read_behavior_trace_dataset(decisions_path)


def test_resume_matches_one_shot_content_and_rejects_duplicate_range(tmp_path):
    config = _challenge_config("collector-tests/resume-v1")
    decks = {
        0: _forced_deck({10: "X1", 11: "X2"}),
        1: _forced_deck({13: "X1", 53: "X2"}),
        2: _forced_deck({16: "X1", 17: "X2"}),
    }
    model = UniformLegalBehaviorModel()
    one_shot = collect_hu_behavior_traces(
        config, model, root_count=3, forced_decks=decks
    )
    first = collect_hu_behavior_traces(
        config, model, root_count=1, forced_decks={0: decks[0]}
    )
    path = tmp_path / "decisions.jsonl"
    write_behavior_trace_dataset(first, path)
    resumed = resume_behavior_trace_dataset(
        path,
        model,
        additional_root_count=2,
        forced_decks={1: decks[1], 2: decks[2]},
    )

    assert resumed.records == one_shot.records
    assert resumed.hidden_roots == one_shot.hidden_roots
    assert (
        resumed.manifest["collection_content_sha256"]
        == one_shot.manifest["collection_content_sha256"]
    )
    with pytest.raises(ValueError, match="resume range"):
        collect_hu_behavior_traces(
            config,
            model,
            root_count=1,
            root_index_start=0,
            forced_decks={0: decks[0]},
            existing_dataset=first,
        )


def test_caller_supplied_challenge_requires_exact_new_root_deck_keys():
    config = _challenge_config("collector-tests/forced-key-audit-v1")
    with pytest.raises(ValueError, match="exactly one deck"):
        collect_hu_behavior_traces(
            config,
            UniformLegalBehaviorModel(),
            root_count=2,
            forced_decks={0: _two_forced_decks()[0]},
        )
    with pytest.raises(ValueError, match="exactly one deck"):
        collect_hu_behavior_traces(
            config,
            UniformLegalBehaviorModel(),
            root_count=1,
            forced_decks={0: _two_forced_decks()[0], 1: _two_forced_decks()[1]},
        )
