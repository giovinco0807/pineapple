from ofc_regular.action_space import generate_turn_actions
from ofc_regular.replay_hu_turn2_stage8b_topk_hard_negatives import (
    ReplayModelParts,
    action_from_json,
    action_signature,
    belief_replay_ready,
    board_from_json,
    deterministic_replay_seed,
    label_from_delta,
    make_rollout_policy,
    replay_source_metadata,
    resolve_action_index,
    select_replay_rows,
)


def test_board_and_action_json_roundtrip_helpers():
    board = board_from_json(
        {
            "top": ["Kh", "Qh"],
            "middle": ["As", "Ac"],
            "bottom": ["2d", "3d", "4d"],
        }
    )
    action = action_from_json(
        {
            "placements": [["5h", "middle"], ["7c", "top"]],
            "discards": ["2c"],
        }
    )

    assert board.card_count() == 7
    assert action.placements == (("5h", "middle"), ("7c", "top"))
    assert action.discards == ("2c",)


def test_resolve_action_index_uses_signature_not_logged_index():
    board = board_from_json(
        {
            "top": ["Kh", "Qh"],
            "middle": ["As", "Ac"],
            "bottom": ["2d", "3d", "4d"],
        }
    )
    dealt = ("5h", "2c", "7c")
    actions = generate_turn_actions(board, dealt)
    target_index = 4
    target_action = actions[target_index]
    logged_action = {
        "placements": [list(placement) for placement in reversed(target_action.placements)],
        "discards": list(reversed(target_action.discards)),
    }

    assert resolve_action_index(actions, logged_action) == target_index
    assert action_signature(target_action) == action_signature(actions[target_index])


def test_resolve_action_index_returns_none_for_missing_action():
    board = board_from_json(
        {
            "top": ["Kh", "Qh"],
            "middle": ["As", "Ac"],
            "bottom": ["2d", "3d", "4d"],
        }
    )
    actions = generate_turn_actions(board, ("5h", "2c", "7c"))

    assert (
        resolve_action_index(
            actions,
            {
                "placements": [["5h", "top"], ["2c", "top"]],
                "discards": ["7c"],
            },
        )
        is None
    )


def test_deterministic_replay_seed_depends_on_state_and_action():
    row = {
        "state_signature": "state-a",
        "action_signature": "action-a",
        "hand_seed": 123,
        "config_id": "cfg",
    }

    seed_a = deterministic_replay_seed(base_seed=10, row=row, row_index=0, future_samples=512)
    seed_b = deterministic_replay_seed(base_seed=10, row=row, row_index=0, future_samples=512)
    seed_c = deterministic_replay_seed(
        base_seed=10,
        row={**row, "action_signature": "action-b"},
        row_index=0,
        future_samples=512,
    )

    assert seed_a == seed_b
    assert seed_a != seed_c


def test_select_replay_rows_preserves_global_row_index_for_chunks():
    rows = [{"id": index} for index in range(6)]

    assert select_replay_rows(rows, offset=2, limit=3) == [
        (2, {"id": 2}),
        (3, {"id": 3}),
        (4, {"id": 4}),
    ]
    assert select_replay_rows(rows, offset=4, limit=0) == [(4, {"id": 4}), (5, {"id": 5})]


def test_label_from_delta_uses_lcb_and_negative_delta():
    positive = label_from_delta(3.0, 0.5)
    gray = label_from_delta(0.5, 0.5)
    negative = label_from_delta(-0.1, 0.5)

    assert positive["safe_lcb196_label"] == "positive"
    assert positive["hard_negative_label"] == 0
    assert gray["safe_lcb196_label"] == "gray"
    assert gray["hard_negative_label"] == 0
    assert negative["safe_lcb196_label"] == "negative"
    assert negative["hard_negative_label"] == 1


def test_replay_source_metadata_preserves_tracking_fields():
    row = {"hand_seed": 2026062601000007, "config_id": "cfg", "hand_id": "h1"}

    metadata = replay_source_metadata(row, row_index=3, candidate_index=14)

    assert metadata["source"] == "hu_turn2_stage8b_topk_hard_negative_replay"
    assert metadata["source_bucket"] == "topk_hard_negative_replay"
    assert metadata["hand_seed"] == 2026062601000007
    assert metadata["hand_id"] == "h1"
    assert metadata["game_id"] == 2026062601000007
    assert metadata["sample_id"] == "topk_hard_negative_replay:2026062601000007:14"
    assert metadata["state_id"] == "topk_hard_negative_replay:2026062601000007"
    assert metadata["source_config_id"] == "cfg"


def test_belief_replay_readiness_ignores_offline_opponent_truth():
    row = {
        "replay_ready": False,
        "hero_board": {"top": ["Kh"], "middle": [], "bottom": []},
        "opponent_board": {"top": ["Qh"], "middle": [], "bottom": []},
        "cards_to_place": ["5h", "6h", "7h"],
        "hero_private_discards": ["2c"],
        "baseline_action": {"placements": [], "discards": []},
        "candidate_action": {"placements": [], "discards": []},
        "dead_cards": [],
        "opponent_private_discards": [],
    }

    assert belief_replay_ready(row)
    assert not belief_replay_ready(
        {**row, "hero_private_discards": [], "visible_dead_cards": []}
    )


def test_topk_hard_negative_replay_policy_uses_explicit_t3_continuation():
    parts = ReplayModelParts(
        opening="opening",
        turn1="turn1",
        turn2_baseline="turn2",
        turn3="turn3",
        hu_turn3_stage7="stage7",
        hu_turn3_reference="stage3-reference",
    )

    stage3_policy = make_rollout_policy(
        parts,
        seed=1,
        seat="first",
        opening_lookahead_samples=2,
        t3_continuation="stage3_reference_default",
    )
    stage7_policy = make_rollout_policy(
        parts,
        seed=1,
        seat="first",
        opening_lookahead_samples=2,
        t3_continuation="stage7_m5_r10",
    )

    assert stage3_policy.hu_turn3_stage7_enabled is False
    assert stage3_policy.hu_turn3_model is None
    assert stage3_policy.hu_turn3_reference_model == "stage3-reference"
    assert stage3_policy.hu_turn3_reference_min_margin == 0.0

    assert stage7_policy.hu_turn3_stage7_enabled is True
    assert stage7_policy.hu_turn3_model == "stage7"
    assert stage7_policy.hu_turn3_reference_model == "stage3-reference"
    assert stage7_policy.hu_turn3_min_margin == 5.0
    assert stage7_policy.hu_turn3_reference_min_margin == 10.0
