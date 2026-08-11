from __future__ import annotations

import hashlib
import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from ofc_regular import hu_m31_t3_dataset_contract_v1 as subject
from ofc_regular.action_key import (
    ACTION_KEY_SCHEMA,
    action_key,
    canonicalize_actions,
    legal_action_set_digest,
    ordered_action_mapping_digest,
)
from ofc_regular.action_space import generate_turn_actions
from ofc_regular.cards import ALL_CARDS
from ofc_regular.hu_infoset import ActorObservation
from ofc_regular.hu_m31_t3_behavior_roots import M31_T3_BEHAVIOR_PROFILES
from ofc_regular.state import Board


def _observation(global_pair_index: int, seat: str) -> ActorObservation:
    cards = list(ALL_CARDS)
    random.Random(9_100_000 + global_pair_index * 2 + (seat == "second")).shuffle(
        cards
    )
    cursor = 0

    def take(count: int) -> tuple[str, ...]:
        nonlocal cursor
        result = tuple(cards[cursor : cursor + count])
        cursor += count
        return result

    hero = Board.from_rows(take(2), take(3), take(4))
    opponent = (
        Board.from_rows(take(2), take(3), take(4))
        if seat == "first"
        else Board.from_rows(take(2), take(4), take(5))
    )
    return ActorObservation(
        hero_board=hero,
        opponent_public_board=opponent,
        dealt_cards=take(3),
        hero_private_discards=take(2),
        seat=seat,  # type: ignore[arg-type]
        street="T3",
        to_act_order=seat,  # type: ignore[arg-type]
    )


def _seat_row(
    observation: ActorObservation,
    *,
    root_index: int,
    confirmation_required: bool,
) -> dict[str, Any]:
    actions = canonicalize_actions(
        generate_turn_actions(observation.hero_board, observation.dealt_cards)
    )
    keys = [action_key(action).to_token() for action in actions]
    primary = [float(len(keys) - index) for index in range(len(keys))]
    confirmation = [value + 0.25 for value in primary]
    baseline_index = len(keys) - 1
    targets = []
    for index, key in enumerate(keys):
        targets.append(
            {
                "action_index": index,
                "action_key": key,
                "primary_q": primary[index],
                "primary_delta": primary[index] - primary[baseline_index],
                "primary_rank": index,
                "confirmation_q": (
                    confirmation[index] if confirmation_required else None
                ),
                "confirmation_delta": (
                    confirmation[index] - confirmation[baseline_index]
                    if confirmation_required
                    else None
                ),
                "confirmation_rank": index if confirmation_required else None,
            }
        )
    return {
        "schema": subject.SEAT_ROW_SCHEMA,
        "root_index": root_index,
        "seat": observation.seat,
        "observation_fingerprint": observation.fingerprint(),
        "observation_sha256": subject.canonical_sha256(observation.to_dict()),
        "observation": observation.to_dict(),
        "action_key_schema": ACTION_KEY_SCHEMA,
        "legal_action_keys": keys,
        "legal_action_set_digest": legal_action_set_digest(actions),
        "legal_action_order_digest": ordered_action_mapping_digest(actions),
        "teacher": {
            "schema": subject.TEACHER_LABEL_SCHEMA,
            "primary_budget": dict(subject.PRIMARY_BUDGET),
            "confirmation_budget": (
                dict(subject.CONFIRMATION_BUDGET)
                if confirmation_required
                else None
            ),
            "baseline_action_key": keys[baseline_index],
            "selected_action_key": keys[0],
            "state_value": primary[0],
            "confirmation_state_value": (
                confirmation[0] if confirmation_required else None
            ),
            "teacher_value_status": "search_estimate_not_realized_match_ev",
            "teacher_values_are_realized_match_ev": False,
            "action_targets": targets,
        },
    }


def _pair_result(
    plan: dict[str, Any], split: str, local_pair_index: int
) -> dict[str, Any]:
    contract = subject.pair_contract(plan, split, local_pair_index)
    rows = [
        _seat_row(
            _observation(contract["global_pair_index"], seat),
            root_index=contract["root_indices"][offset],
            confirmation_required=contract["confirmation_required"],
        )
        for offset, seat in enumerate(("first", "second"))
    ]
    return {
        "schema": subject.PAIR_RESULT_SCHEMA,
        "plan_sha256": subject.canonical_sha256(plan),
        "pair_contract_sha256": subject.canonical_sha256(contract),
        "shard_id": contract["shard_id"],
        "split": split,
        "local_pair_index": local_pair_index,
        "global_pair_index": contract["global_pair_index"],
        "root_indices": contract["root_indices"],
        "profile": contract["profile"],
        "seeds": contract["seeds"],
        "confirmation_required": contract["confirmation_required"],
        "rows": rows,
        "opponent_private_discards_used": False,
        "realized_deck_tail_used": False,
        "teacher_values_are_realized_match_ev": False,
        "current_profile_changed": False,
    }


def _write_complete_smoke(
    plan: dict[str, Any], directory: Path
) -> dict[str, Any]:
    for index in range(subject.SHARD_PAIR_COUNT):
        subject.write_pair_result(
            plan=plan,
            shard_id=subject.SMOKE_SHARD_ID,
            shard_directory=directory,
            local_pair_index=index,
            value=_pair_result(plan, "train", index),
        )
    return subject.finalize_shard(
        plan=plan,
        shard_id=subject.SMOKE_SHARD_ID,
        shard_directory=directory,
    )


def _pair_records(plan: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for shard in plan["shards"]:
        for local_index, global_index in zip(
            shard["local_pair_indices"],
            shard["global_pair_indices"],
            strict=True,
        ):
            records.append(
                {
                    "split": shard["split"],
                    "local_pair_index": local_index,
                    "global_pair_index": global_index,
                    "shard_id": shard["shard_id"],
                    "path": f"pairs/pair_{local_index:06d}.json",
                    "sha256": hashlib.sha256(
                        f"pair:{global_index}".encode("ascii")
                    ).hexdigest(),
                    "bytes": 1 + global_index,
                }
            )
    records.sort(key=lambda row: row["global_pair_index"])
    return records


def test_fixed_plan_has_exact_splits_confirmations_shards_and_hash() -> None:
    plan = subject.build_dataset_plan()
    assert subject.canonical_sha256(plan) == subject.EXPECTED_DATASET_PLAN_SHA256
    assert plan["paired_hand_count"] == 9_000
    assert plan["root_count"] == 18_000
    assert [(row["split"], row["paired_hand_count"]) for row in plan["splits"]] == [
        ("train", 6_000),
        ("safety-fit", 1_000),
        ("threshold-lock", 1_000),
        ("diagnostic-holdout", 1_000),
    ]
    assert [
        row["confirmation_selection"]["selected_pair_count"]
        for row in plan["splits"]
    ] == [600, 100, 100, 100]
    assert all(
        row["confirmation_selection"]["locked_before_label_read"] is True
        for row in plan["splits"]
    )
    assert len(plan["shards"]) == 360
    assert plan["shards"][0]["shard_id"] == subject.SMOKE_SHARD_ID
    assert plan["shards"][0]["local_pair_indices"] == list(range(25))
    assert plan["shards"][0]["confirmation_pair_count"] == 3
    assert plan["shards"][0]["requires_smoke_gate_receipt"] is False
    assert all(
        row["requires_smoke_gate_receipt"] is True for row in plan["shards"][1:]
    )
    assert plan["scientific_boundaries"]["plan_alone_authorizes_compute"] is False


def test_seed_stride_is_deterministic_and_disjoint_from_all_prior_namespaces() -> None:
    plan = subject.build_dataset_plan()
    audit = plan["seed_contract"]
    assert audit["dataset_seed_count"] == 54_000
    assert audit["all_dataset_seeds_unique"] is True
    assert audit["split_seed_sets_disjoint"] is True
    assert audit["all_external_overlap_counts_zero"] is True
    assert all(
        row["dataset_overlap_count"] == 0
        for row in audit["external_namespace_audit"].values()
    )
    row2 = subject.pair_contract(plan, "train", 2)
    row3 = subject.pair_contract(plan, "train", 3)
    assert {
        key: row3["seeds"][key] - row2["seeds"][key] for key in row2["seeds"]
    } == {key: 1_000_003 for key in row2["seeds"]}
    assert row2["root_indices"] == [4, 5]
    assert subject.pair_contract(plan, "safety-fit", 0)["root_indices"] == [
        12_000,
        12_001,
    ]
    assert {
        subject.pair_contract(plan, "train", index)["profile"]
        for index in range(5)
    } == set(M31_T3_BEHAVIOR_PROFILES)
    assert all(
        subject.pair_contract(plan, split, index)["confirmation_required"]
        == (index % 10 == 0)
        for split, count in (
            ("train", 6_000),
            ("safety-fit", 1_000),
            ("threshold-lock", 1_000),
            ("diagnostic-holdout", 1_000),
        )
        for index in (0, 1, 9, 10, count - 1)
    )


def test_plan_is_source_replayed_canonical_and_write_once(tmp_path: Path) -> None:
    plan = subject.build_dataset_plan()
    output = tmp_path / "dataset_plan.json"
    assert subject.write_dataset_plan(output) == plan
    assert output.read_bytes() == subject.canonical_bytes(plan)
    with pytest.raises(FileExistsError, match="overwrite"):
        subject.write_dataset_plan(output)
    changed = deepcopy(plan)
    changed["splits"][0]["paired_hand_count"] = 5_999
    with pytest.raises(ValueError, match="frozen replay"):
        subject.validate_dataset_plan(changed)


def test_pair_row_recomputes_canonical_action_mapping_and_rejects_leakage() -> None:
    plan = subject.build_dataset_plan()
    pair = _pair_result(plan, "train", 1)
    assert subject.validate_pair_result(pair, plan=plan) == pair
    first_keys = pair["rows"][0]["legal_action_keys"]

    observation = ActorObservation.from_dict(pair["rows"][0]["observation"])
    permuted = ActorObservation(
        hero_board=observation.hero_board,
        opponent_public_board=observation.opponent_public_board,
        dealt_cards=tuple(reversed(observation.dealt_cards)),
        hero_private_discards=observation.hero_private_discards,
        seat=observation.seat,
        street=observation.street,
        to_act_order=observation.to_act_order,
        scoring=observation.scoring,
    )
    assert _seat_row(
        permuted, root_index=pair["root_indices"][0], confirmation_required=False
    )["legal_action_keys"] == first_keys

    mapping_drift = deepcopy(pair)
    mapping_drift["rows"][0]["legal_action_keys"][0:2] = reversed(
        mapping_drift["rows"][0]["legal_action_keys"][0:2]
    )
    with pytest.raises(ValueError, match="ActionKey mapping"):
        subject.validate_pair_result(mapping_drift, plan=plan)

    hidden = deepcopy(pair)
    hidden["rows"][0]["observation"]["opponent_private_discards"] = ["As"]
    with pytest.raises(ValueError):
        subject.validate_pair_result(hidden, plan=plan)

    realized_tail = deepcopy(pair)
    realized_tail["realized_deck_tail"] = ["As"]
    with pytest.raises(ValueError):
        subject.validate_pair_result(realized_tail, plan=plan)


def test_confirmation_targets_are_required_exactly_on_preregistered_pairs() -> None:
    plan = subject.build_dataset_plan()
    confirmed = _pair_result(plan, "threshold-lock", 10)
    assert confirmed["confirmation_required"] is True
    assert subject.validate_pair_result(confirmed, plan=plan) == confirmed

    missing = deepcopy(confirmed)
    missing["rows"][0]["teacher"]["action_targets"][0]["confirmation_q"] = None
    with pytest.raises(ValueError, match="confirmation_q"):
        subject.validate_pair_result(missing, plan=plan)

    unconfirmed = _pair_result(plan, "threshold-lock", 11)
    injected = deepcopy(unconfirmed)
    injected["rows"][0]["teacher"]["action_targets"][0][
        "confirmation_q"
    ] = 1.0
    with pytest.raises(ValueError, match="nonconfirmation"):
        subject.validate_pair_result(injected, plan=plan)


def test_shard_resume_write_once_finalize_gap_and_tamper_checks(
    tmp_path: Path,
) -> None:
    plan = subject.build_dataset_plan()
    shard = tmp_path / "smoke"
    for index in (0, 1):
        pair = _pair_result(plan, "train", index)
        subject.write_pair_result(
            plan=plan,
            shard_id=subject.SMOKE_SHARD_ID,
            shard_directory=shard,
            local_pair_index=index,
            value=pair,
        )
    resume = subject.inspect_shard_resume(
        plan=plan,
        shard_id=subject.SMOKE_SHARD_ID,
        shard_directory=shard,
    )
    assert resume["completed_pair_indices"] == [0, 1]
    assert resume["pending_pair_indices"] == list(range(2, 25))
    assert resume["safe_to_resume"] is True
    with pytest.raises(FileExistsError, match="overwrite"):
        subject.write_pair_result(
            plan=plan,
            shard_id=subject.SMOKE_SHARD_ID,
            shard_directory=shard,
            local_pair_index=1,
            value=_pair_result(plan, "train", 1),
        )
    with pytest.raises(ValueError, match="pair gaps"):
        subject.finalize_shard(
            plan=plan,
            shard_id=subject.SMOKE_SHARD_ID,
            shard_directory=shard,
        )

    for index in range(2, 25):
        subject.write_pair_result(
            plan=plan,
            shard_id=subject.SMOKE_SHARD_ID,
            shard_directory=shard,
            local_pair_index=index,
            value=_pair_result(plan, "train", index),
        )
    done = subject.finalize_shard(
        plan=plan,
        shard_id=subject.SMOKE_SHARD_ID,
        shard_directory=shard,
    )
    assert done["pair_count"] == 25
    assert done["root_count"] == 50
    assert subject.finalize_shard(
        plan=plan,
        shard_id=subject.SMOKE_SHARD_ID,
        shard_directory=shard,
    ) == done
    assert subject.inspect_shard_resume(
        plan=plan,
        shard_id=subject.SMOKE_SHARD_ID,
        shard_directory=shard,
    )["already_complete"] is True

    tampered_path = subject.pair_artifact_path(shard, 7)
    tampered = json.loads(tampered_path.read_text(encoding="ascii"))
    tampered["rows"][0]["teacher"]["action_targets"][0]["primary_delta"] += 1.0
    tampered_path.write_bytes(subject.canonical_bytes(tampered))
    with pytest.raises(ValueError, match="arithmetic"):
        subject.validate_completed_shard(
            plan=plan,
            shard_id=subject.SMOKE_SHARD_ID,
            shard_directory=shard,
        )


def test_unknown_or_extra_pair_file_fails_resume_closed(tmp_path: Path) -> None:
    plan = subject.build_dataset_plan()
    shard = tmp_path / "smoke"
    (shard / "pairs").mkdir(parents=True)
    (shard / "pairs" / "copy.json").write_text("{}", encoding="ascii")
    with pytest.raises(ValueError, match="unknown"):
        subject.inspect_shard_resume(
            plan=plan,
            shard_id=subject.SMOKE_SHARD_ID,
            shard_directory=shard,
        )


def test_smoke_receipt_opens_only_post_smoke_shards_and_is_write_once(
    tmp_path: Path,
) -> None:
    plan = subject.build_dataset_plan()
    smoke = tmp_path / "smoke"
    _write_complete_smoke(plan, smoke)
    receipt = subject.build_smoke_gate_receipt(
        plan=plan, smoke_shard_directory=smoke
    )
    assert receipt["all_gates_passed"] is True
    assert receipt["full_9000_paired_fanout_authorized"] is True
    assert receipt["metrics"]["paired_hand_count"] == 25
    assert receipt["metrics"]["seat_counts"] == {"first": 25, "second": 25}
    assert receipt["metrics"]["confirmation_pair_count"] == 3

    scale_shard = plan["shards"][1]["shard_id"]
    with pytest.raises(ValueError, match="requires"):
        subject.validate_shard_start_authorization(
            plan=plan, shard_id=scale_shard
        )
    authorization = subject.validate_shard_start_authorization(
        plan=plan,
        shard_id=scale_shard,
        smoke_gate_receipt=receipt,
        smoke_shard_directory=smoke,
    )
    assert authorization["authorized_by_dataset_smoke_gate"] is True
    assert subject.validate_shard_start_authorization(
        plan=plan, shard_id=subject.SMOKE_SHARD_ID
    )["scope"] == "first_25_paired_smoke_only"

    output = tmp_path / "SMOKE_GATE.json"
    assert subject.write_smoke_gate_receipt(
        plan=plan, smoke_shard_directory=smoke, output_path=output
    ) == receipt
    with pytest.raises(FileExistsError, match="overwrite"):
        subject.write_smoke_gate_receipt(
            plan=plan, smoke_shard_directory=smoke, output_path=output
        )
    changed = deepcopy(receipt)
    changed["metrics"]["missing_pair_count"] = 1
    with pytest.raises(ValueError, match="source replay"):
        subject.validate_smoke_gate_receipt(
            changed, plan=plan, smoke_shard_directory=smoke
        )


def test_pure_merge_index_rejects_duplicate_gap_order_and_mapping_drift() -> None:
    plan = subject.build_dataset_plan()
    records = _pair_records(plan)
    audit = subject.validate_pair_record_index(records)
    assert audit["paired_hand_count"] == 9_000
    assert audit["split_counts"] == {
        "train": 6_000,
        "safety-fit": 1_000,
        "threshold-lock": 1_000,
        "diagnostic-holdout": 1_000,
    }

    duplicate = deepcopy(records)
    duplicate[-1] = deepcopy(duplicate[-2])
    with pytest.raises(ValueError, match="duplicate"):
        subject.validate_pair_record_index(duplicate)

    gap = deepcopy(records[:-1])
    with pytest.raises(ValueError, match="gaps"):
        subject.validate_pair_record_index(gap)

    reordered = deepcopy(records)
    reordered[100], reordered[101] = reordered[101], reordered[100]
    with pytest.raises(ValueError, match="canonical global order"):
        subject.validate_pair_record_index(reordered)

    mapping = deepcopy(records)
    mapping[6_000]["shard_id"] = subject.SMOKE_SHARD_ID
    with pytest.raises(ValueError, match="frozen grid"):
        subject.validate_pair_record_index(mapping)


def test_merge_manifest_structural_replay_rejects_shard_and_pair_tamper() -> None:
    plan = subject.build_dataset_plan()
    pair_records = _pair_records(plan)
    index = subject.validate_pair_record_index(pair_records)
    shard_records = [
        {
            "shard_id": shard["shard_id"],
            "split": shard["split"],
            "done_sha256": hashlib.sha256(
                f"done:{shard['shard_id']}".encode("ascii")
            ).hexdigest(),
            "pair_record_aggregate_sha256": hashlib.sha256(
                f"pairs:{shard['shard_id']}".encode("ascii")
            ).hexdigest(),
            "pair_count": 25,
        }
        for shard in plan["shards"]
    ]
    merge = {
        "schema": subject.MERGE_SCHEMA,
        "status": "complete_immutable_9000_paired_dataset_index",
        "plan_sha256": subject.canonical_sha256(plan),
        "shard_records": shard_records,
        "shard_record_aggregate_sha256": subject.canonical_sha256(shard_records),
        "pair_records": pair_records,
        "pair_record_aggregate_sha256": index["pair_record_aggregate_sha256"],
        "paired_hand_count": 9_000,
        "root_count": 18_000,
        "split_counts": index["split_counts"],
        "seat_counts": {"first": 9_000, "second": 9_000},
        "confirmation_pair_count": 900,
        "global_pair_index_digest": index["global_pair_index_digest"],
        "hidden_information_field_count": 0,
        "unknown_field_count": 0,
        "action_key_mapping_mismatch_count": 0,
        "missing_pair_count": 0,
        "duplicate_pair_count": 0,
        "dataset_ready_for_training": True,
        "teacher_values_are_realized_match_ev": False,
        "current_profile_changed": False,
    }
    assert subject.validate_merge_manifest(merge, plan=plan) == merge

    shard_tamper = deepcopy(merge)
    shard_tamper["shard_records"][0]["split"] = "safety-fit"
    shard_tamper["shard_record_aggregate_sha256"] = subject.canonical_sha256(
        shard_tamper["shard_records"]
    )
    with pytest.raises(ValueError, match="shard record"):
        subject.validate_merge_manifest(shard_tamper, plan=plan)

    pair_tamper = deepcopy(merge)
    pair_tamper["pair_records"][0]["path"] = "pairs/copied.json"
    pair_tamper["pair_record_aggregate_sha256"] = subject.canonical_sha256(
        pair_tamper["pair_records"]
    )
    with pytest.raises(ValueError, match="path changed"):
        subject.validate_merge_manifest(pair_tamper, plan=plan)
