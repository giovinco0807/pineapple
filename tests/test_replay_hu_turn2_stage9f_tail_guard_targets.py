from ofc_regular.replay_hu_turn2_stage9f_tail_guard_targets import (
    canonical_action_from_payload,
    parse_args,
    prepare_target,
)


def target_row(**overrides):
    row = {
        "target_id": "t1",
        "target_group": "tail_loss",
        "config_id": "stage9f_cse2_csemax2_firstseat",
        "seed": 100,
        "hand_id": 100,
        "seat": "first",
        "hero_board": {"top": ["Qh"], "middle": ["Kh", "Ks"], "bottom": ["5h", "5d", "6c", "6h"]},
        "opponent_board": {"top": ["Ac"], "middle": ["6s", "8c", "8d"], "bottom": ["Jd", "Th", "Tc"]},
        "cards_to_place": ["Ah", "Jh", "Qd"],
        "dead_cards": ["4s", "3d"],
        "visible_dead_cards": ["Ac", "6s", "8c", "8d", "Jd", "Th", "Tc", "4s"],
        "hero_private_discards": ["4s"],
        "opponent_private_discards": ["3d"],
        "baseline_action_index": 19,
        "candidate_action_index": 11,
        "baseline_action": {
            "placements": [["Jh", "middle"], ["Qd", "top"]],
            "discards": ["Ah"],
        },
        "candidate_action": {
            "placements": [["Ah", "middle"], ["Qd", "top"]],
            "discards": ["Jh"],
        },
        "realized_delta": -31.227,
    }
    row.update(overrides)
    return row


def test_canonical_action_from_payload_is_order_insensitive():
    left = canonical_action_from_payload(
        {"placements": [["Jh", "middle"], ["Qd", "top"]], "discards": ["Ah"]}
    )
    right = canonical_action_from_payload(
        {"placements": [["Qd", "top"], ["Jh", "middle"]], "discards": ["Ah"]}
    )

    assert left == right


def test_prepare_target_resolves_runtime_action_indices():
    prepared = prepare_target(target_row())

    assert prepared.replay_ready is True
    assert prepared.failure_reason == "ready"
    assert prepared.baseline_index == 19
    assert prepared.candidate_index == 11
    assert len(prepared.actions) > 0


def test_prepare_target_keeps_offline_truth_optional_for_belief_replay():
    prepared = prepare_target(
        target_row(dead_cards=[], opponent_private_discards=[])
    )

    assert prepared.replay_ready is True
    assert prepared.failure_reason == "ready"


def test_prepare_target_blocks_missing_hero_visible_discard():
    prepared = prepare_target(
        target_row(hero_private_discards=[], visible_dead_cards=[])
    )

    assert prepared.replay_ready is False
    assert "hero_visible_discard" in prepared.failure_reason


def test_prepare_target_searches_when_hint_is_wrong():
    prepared = prepare_target(target_row(candidate_action_index=0))

    assert prepared.replay_ready is True
    assert prepared.candidate_index == 11


def test_cli_accepts_target_group_filter(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--targets",
            "targets.jsonl",
            "--target-group",
            "tail_loss",
            "--target-group",
            "confirm_z_boundary",
        ],
    )

    args = parse_args()

    assert args.target_group == ["tail_loss", "confirm_z_boundary"]


def test_cli_accepts_offset_and_limit_for_shards(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--targets",
            "targets.jsonl",
            "--offset",
            "20",
            "--limit",
            "10",
        ],
    )

    args = parse_args()

    assert args.offset == 20
    assert args.limit == 10
