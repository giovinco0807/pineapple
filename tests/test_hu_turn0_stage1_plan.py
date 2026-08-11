import json
from pathlib import Path


def test_turn0_stage1_plan_locks_chain_and_all_action_pilot():
    plan = json.loads(
        Path("configs/hu_turn0_stage1_stage18_p1_plan.json").read_text(
            encoding="utf-8"
        )
    )

    assert plan["status"] == "pilot_ready_not_p0"
    assert plan["t0_p0_fixed"] is False
    assert plan["production_default"] is False
    assert plan["fixed_continuations"]["t1"]["profile"] == "stage18_p1"
    assert plan["fixed_continuations"]["t2"]["profile"] == "stage9f_p2"
    assert plan["fixed_continuations"]["t3"]["profile"] == "stage7_m5_r10"
    assert plan["fixed_continuations"]["fl_ev"] > 10.0
    assert plan["state_definition"]["legal_actions"] == 232
    assert plan["state_definition"]["common_random_futures_required"] is True
    assert plan["state_definition"]["action_independent_continuation_seed_required"] is True

    pilot = plan["initial_gcp_pilot"]
    assert pilot["total_states"] == 100
    assert pilot["first_states"] == 50
    assert pilot["second_states"] == 50
    assert pilot["all_legal_actions"] is True
    assert pilot["future_samples"] == 4
    assert pilot["total_shards"] == 50
    assert pilot["vm_count"] == 20
    assert pilot["machine_type"] == "c4-highmem-2"
    assert pilot["provisioning_model"] == "SPOT"
    assert plan["local_speed_evidence"]["c4_benchmark_records"] == 2

    hard_pass = plan["pilot_hard_pass"]
    assert hard_pass["records"] == 100
    assert hard_pass["legal_actions_per_state"] == 232
    assert hard_pass["evaluated_actions_per_state"] == 232
    assert hard_pass["non_finite_actions"] == 0
    assert hard_pass["common_random_future_failures"] == 0
    assert hard_pass["replay_not_ready"] == 0
    assert plan["final_acceptance_principles"]["primary_metric"] == (
        "realized_fired_whole_game_seat_swap_counterfactual_delta"
    )
