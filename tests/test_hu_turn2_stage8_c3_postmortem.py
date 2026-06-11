import json

from ofc_regular.analyze_hu_turn2_stage8_c3_postmortem import (
    action_signature,
    fired_overlap_rows,
    no_override_reason_rows,
    state_key,
    top_loss_audit_rows,
)


def _decision(config_id: str, *, fired: bool, reason: str = "", score: float = 1.0, seat: str = "first"):
    return {
        "config_id": config_id,
        "hand_id": 100,
        "paired_index": 1,
        "seat_swap": "ab",
        "seat": seat,
        "hero_board": {"top": ["Ah"], "middle": ["2c"], "bottom": ["3d"]},
        "opponent_board": {"top": [], "middle": ["4c"], "bottom": ["5d"]},
        "cards_to_place": ["Ks", "Qh", "Jd"],
        "dead_cards": ["2c", "3d", "4c"],
        "baseline_action": {"placements": [["Ks", "top"], ["Qh", "middle"]], "discards": ["Jd"]},
        "stage8_action": {"placements": [["Qh", "middle"], ["Ks", "top"]], "discards": ["Jd"]},
        "final_action": {"placements": [["Ks", "top"], ["Qh", "middle"]], "discards": ["Jd"]},
        "override_fired": fired,
        "no_override_reason": reason,
        "predicted_delta": 3.0,
        "gate_probability": 0.95,
        "reference_margin_raw": 0.1,
        "candidate_seat_score": score,
        "_state_key": "",
        "_stage8_action_signature": "",
        "_final_action_signature": "",
    }


def _prepare(row):
    row["_state_key"] = state_key(row)
    row["_stage8_action_signature"] = action_signature(row["stage8_action"])
    row["_final_action_signature"] = action_signature(row["final_action"])
    return row


def test_state_and_action_keys_ignore_card_order():
    left = _prepare(_decision("a", fired=True))
    right = _prepare(_decision("b", fired=True))
    right["hero_board"]["top"] = list(reversed(left["hero_board"]["top"]))
    right["cards_to_place"] = list(reversed(left["cards_to_place"]))
    right["stage8_action"] = {"placements": [["Ks", "top"], ["Qh", "middle"]], "discards": ["Jd"]}

    assert state_key(left) == state_key(right)
    assert action_signature(left["stage8_action"]) == action_signature(right["stage8_action"])


def test_fired_overlap_and_reason_rows():
    rows = [
        _prepare(_decision("m2.5_r0_g0.9", fired=True)),
        _prepare(_decision("m2.75_r0_g0.9", fired=True)),
        _prepare(_decision("m2.75_r0_g0.9", fired=False, reason="below_stage8_margin", seat="second")),
    ]

    overlap = fired_overlap_rows(rows, {})
    pair = next(row for row in overlap if row.get("row_type") == "pairwise_overlap")
    assert pair["intersection_states"] == 1
    assert pair["same_action_overlap"] == 1

    reasons = no_override_reason_rows(rows)
    margin_reason = next(row for row in reasons if row["no_override_reason"] == "below_stage8_margin")
    assert margin_reason["reason_group"] == "below_predicted_delta"


def test_top_loss_audit_dedupes_same_state_action():
    loss_a = _prepare(_decision("m2.5_r0_g0.9", fired=True, score=-2.0))
    loss_b = json.loads(json.dumps(loss_a))
    loss_b["config_id"] = "m2.75_r0_g0.9"
    loss_b = _prepare(loss_b)

    rows = top_loss_audit_rows([loss_a, loss_b], limit=10)
    assert len(rows) == 1
    assert rows[0]["configs"] == ["m2.5_r0_g0.9", "m2.75_r0_g0.9"]
    assert rows[0]["replay_ready"] is True
