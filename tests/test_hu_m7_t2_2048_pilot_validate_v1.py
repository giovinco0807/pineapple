from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
from dataclasses import replace
from typing import Any, Callable

import pytest

from ofc_regular.action_key import action_key
from ofc_regular.action_space import generate_actions
from ofc_regular.hu_infoset import ActorObservation, ScoringContext
from ofc_regular.hu_m31_label_gen_resume_v1 import build_complete_checkpoint
from ofc_regular.hu_m7_t2_2048_pilot_validate_v1 import (
    main,
    validate_pilot_shard,
)
from ofc_regular.hu_m7_t2_2048_validate_v1 import (
    DEFAULT_CONTRACT,
    T22048Contract,
    T22048ValidationError,
)
from ofc_regular.state import Board


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: pathlib.Path, value: object, *, canonical: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if canonical:
        path.write_bytes(_canonical(value))
    else:
        path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _shards(positions: int) -> list[dict[str, Any]]:
    small, extra = divmod(positions, 2)
    counts = [small] * (2 - extra) + [small + 1] * extra
    start = 0
    rows = []
    for index, count in enumerate(counts):
        rows.append({"shard_id": f"{index:02d}", "start": start, "count": count})
        start += count
    return rows


def _plan(seat: str, contract: T22048Contract) -> dict[str, Any]:
    plan = {
        "schema": "hu_m31_label_gen_plan_v1",
        "job_id": f"m7v5-t2{seat}-25k-2048p-pilot-test",
        "street": "T2",
        "seat": seat,
        "samples": contract.samples,
        "seeds_per_position": contract.seeds_per_position,
        "hand_seed_base": contract.hand_seed_base,
        "behavior_seed_offset": contract.behavior_seed_offset,
        "eval_seed_base": contract.eval_seed_base,
        "engine_library": "native/libofc_hu_m3_engine.so",
        "engine_library_sha256": contract.engine_library_sha256,
        "feature_encoder_library": "native/libofc_stage3_feature_encoder.so",
        "feature_encoder_library_sha256": contract.feature_encoder_library_sha256,
        "t4_model": "weights/t4_model_v6.bin",
        "t4_model_sha256": contract.t4_model_sha256,
        "t3_first_model": "weights/t3first_model_v2.bin",
        "t3_first_model_sha256": contract.t3_first_model_sha256,
        "t3_second_model": "weights/t3_model_v3.bin",
        "t3_second_model_sha256": contract.t3_second_model_sha256,
        "fl_ev_cards": contract.fl_ev_cards,
        "fl_ev_value": contract.fl_ev_value,
        "shards": _shards(contract.positions_per_seat),
    }
    if seat == "first":
        plan.update(
            {
                "t2_second_model": "weights/t2_model_v1.bin",
                "t2_second_model_sha256": contract.t2_second_model_sha256,
            }
        )
    return plan


def _observation(seat: str, contract: T22048Contract) -> ActorObservation:
    hero = Board.from_rows(
        top=("As",),
        middle=("2s", "3s", "4s"),
        bottom=("5s", "6s", "7s"),
    )
    if seat == "first":
        opponent = Board.from_rows(
            top=("8s",),
            middle=("9s", "Ts", "Js"),
            bottom=("Qs", "Ks", "Ah"),
        )
        dealt = ("2h", "3h", "4h")
        discard = ("5h",)
    else:
        opponent = Board.from_rows(
            top=("8s", "9s"),
            middle=("Ts", "Js", "Qs"),
            bottom=("Ks", "Ah", "2h", "3h"),
        )
        dealt = ("4h", "5h", "6h")
        discard = ("7h",)
    return ActorObservation(
        hero_board=hero,
        opponent_public_board=opponent,
        dealt_cards=dealt,
        hero_private_discards=discard,
        seat=seat,
        street="T2",
        to_act_order=seat,
        scoring=ScoringContext(
            fl_ev=((contract.fl_ev_cards, contract.fl_ev_value),)
        ),
    )


def _skeleton(observation: ActorObservation) -> str:
    text = (
        "".join(sorted(observation.hero_board.all_cards()))
        + "|"
        + "".join(sorted(observation.opponent_public_board.all_cards()))
    )
    return hashlib.sha256(text.encode()).hexdigest()[:2]


def _rebuild_checkpoint(case: dict[str, Any]) -> None:
    shard_dir = case["shard_directory"]
    prefix = case["object_prefix"]
    inventory = []
    for path in sorted(shard_dir.iterdir(), key=lambda item: item.name):
        raw = path.read_bytes()
        inventory.append(
            {
                "relative_path": path.name,
                "object_name": f"{prefix}/files/{path.name}",
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
            }
        )
    checkpoint = build_complete_checkpoint(
        plan_sha256=case["plan_sha"],
        shard_id=case["shard"]["shard_id"],
        attempt_id="pilot-attempt-1",
        shard_start=case["shard"]["start"],
        shard_count=case["shard"]["count"],
        files=inventory,
    )
    _write_json(case["checkpoint"], checkpoint, canonical=True)
    generation_rows = [
        {
            "relative_path": row["relative_path"],
            "object_name": row["object_name"],
            "generation": str(2_000 + index),
            "bytes": row["bytes"],
            "sha256": row["sha256"],
        }
        for index, row in enumerate(inventory)
        if row["relative_path"].startswith("position_")
    ]
    manifest_core = {
        "schema": "hu_m31_label_gen_position_generation_manifest_v1",
        "run_name": "pilot-test",
        "bucket": "pilot-bucket",
        "worker_plan_sha256": case["plan_sha"],
        "shard_id": case["shard"]["shard_id"],
        "positions": case["shard"]["count"],
        "files": generation_rows,
        "exact_prefix_inventory": True,
        "checkpoint_sha_and_bytes_matched": True,
        "all_position_gets_generation_pinned": True,
    }
    manifest = {
        **manifest_core,
        "manifest_sha256": hashlib.sha256(_canonical(manifest_core)).hexdigest(),
    }
    _write_json(case["generation_manifest"], manifest, canonical=True)
    position_inventory = [
        {
            "relative_path": row["relative_path"],
            "sha256": row["sha256"],
            "bytes": row["bytes"],
        }
        for row in generation_rows
    ]
    done = shard_dir / "SHARD_DONE.json"
    checkpoint_raw = case["checkpoint"].read_bytes()
    evidence = {
        "schema": "hu_m31_label_gen_shard_receive_evidence_v2",
        "run_name": "pilot-test",
        "bucket": "pilot-bucket",
        "worker_plan_sha256": case["plan_sha"],
        "shard_id": case["shard"]["shard_id"],
        "positions": case["shard"]["count"],
        "done_object": {
            "object_name": f"{prefix}/files/SHARD_DONE.json",
            "generation": "1001",
            "sha256": _sha(done),
            "bytes": done.stat().st_size,
        },
        "complete_checkpoint_object": {
            "object_name": f"{prefix}/checkpoints/complete.json",
            "generation": "1002",
            "sha256": hashlib.sha256(checkpoint_raw).hexdigest(),
            "bytes": len(checkpoint_raw),
        },
        "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "position_inventory_sha256": hashlib.sha256(
            _canonical(position_inventory)
        ).hexdigest(),
        "position_generation_manifest": {
            "relative_path": (
                f"_audit/shard_{case['shard']['shard_id']}/"
                "position_generation_manifest.json"
            ),
            "sha256": _sha(case["generation_manifest"]),
            "positions": case["shard"]["count"],
        },
        "generation_pinned_done_complete_and_all_positions": True,
    }
    _write_json(case["receive_evidence"], evidence, canonical=True)


def _case(
    tmp_path: pathlib.Path,
    *,
    seat: str = "first",
    shard_id: str = "00",
    positions_per_seat: int = 4,
) -> dict[str, Any]:
    contract = replace(
        DEFAULT_CONTRACT,
        positions_per_seat=positions_per_seat,
        shards_per_seat=2,
    )
    plan_path = tmp_path / f"{seat}_plan.json"
    plan = _plan(seat, contract)
    _write_json(plan_path, plan, canonical=False)
    plan_sha = _sha(plan_path)
    contract = replace(
        contract,
        first_plan_sha256=(
            plan_sha if seat == "first" else contract.first_plan_sha256
        ),
        second_plan_sha256=(
            plan_sha if seat == "second" else contract.second_plan_sha256
        ),
    )
    shard = next(row for row in plan["shards"] if row["shard_id"] == shard_id)
    received = tmp_path / "received"
    shard_directory = received / f"shard_{shard_id}"
    shard_directory.mkdir(parents=True)
    checkpoint = received / "_audit" / f"shard_{shard_id}" / "complete.json"
    observation = _observation(seat, contract)
    scores = {
        action_key(action).to_token(): float(index) / 10.0
        for index, action in enumerate(
            generate_actions(observation.hero_board, observation.dealt_cards)
        )
    }
    for offset in range(shard["start"], shard["start"] + shard["count"]):
        _write_json(
            shard_directory / f"position_{offset:08d}.json",
            {
                "schema": "hu_m31_label_gen_position_v1",
                "plan_sha256": plan_sha,
                "offset": offset,
                "skeleton": _skeleton(observation),
                "observation": observation.to_dict(),
                "samples": contract.samples,
                "runs": [{"seed_trial": 0, "scores": scores}],
            },
            canonical=True,
        )
    _write_json(
        shard_directory / "SHARD_DONE.json",
        {
            "schema": "hu_m31_label_gen_shard_done_v1",
            "plan_sha256": plan_sha,
            "shard_id": shard_id,
            "positions": shard["count"],
        },
        canonical=True,
    )
    case = {
        "contract": contract,
        "plan": plan_path,
        "plan_sha": plan_sha,
        "shard": shard,
        "shard_directory": shard_directory,
        "checkpoint": checkpoint,
        "generation_manifest": (
            received
            / "_audit"
            / f"shard_{shard_id}"
            / "position_generation_manifest.json"
        ),
        "receive_evidence": (
            received / "_audit" / f"shard_{shard_id}" / "receive_evidence.json"
        ),
        "object_prefix": f"labelgen/pilot-test/shards/{shard_id}",
    }
    _rebuild_checkpoint(case)
    return case


def _validate(case: dict[str, Any]) -> dict[str, Any]:
    return validate_pilot_shard(
        worker_plan_path=case["plan"],
        shard_id=case["shard"]["shard_id"],
        shard_directory=case["shard_directory"],
        checkpoint_path=case["checkpoint"],
        contract=case["contract"],
    )


@pytest.mark.parametrize(
    ("shard_id", "expected"),
    [("00", 143), ("01", 144)],
)
def test_pilot_accepts_the_real_143_and_144_position_shard_sizes(
    tmp_path: pathlib.Path, shard_id: str, expected: int
) -> None:
    case = _case(tmp_path, shard_id=shard_id, positions_per_seat=287)
    report = _validate(case)
    assert report["status"] == "PASS"
    assert report["scope"] == "single_shard_pilot_only"
    assert report["positions"] == expected
    assert report["all_positions_fully_validated"] is True
    assert report["full_25000_postflight_not_satisfied_by_this_report"] is True


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda record: record["observation"].__setitem__(
                "opponent_private_discards", ["8h"]
            ),
            "forbidden private-truth",
        ),
        (lambda record: record.__setitem__("custom_field", 1), "fields differ"),
    ],
)
def test_pilot_rejects_hidden_truth_and_custom_position_fields_even_when_inventoried(
    tmp_path: pathlib.Path,
    mutation: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    case = _case(tmp_path)
    path = case["shard_directory"] / "position_00000000.json"
    record = json.loads(path.read_bytes())
    mutation(record)
    _write_json(path, record, canonical=True)
    _rebuild_checkpoint(case)
    with pytest.raises(T22048ValidationError, match=message):
        _validate(case)


def test_pilot_rejects_missing_position_and_checkpoint_digest_drift(
    tmp_path: pathlib.Path,
) -> None:
    missing = _case(tmp_path / "missing")
    (missing["shard_directory"] / "position_00000000.json").unlink()
    with pytest.raises(T22048ValidationError, match="contents differ"):
        _validate(missing)

    drift = _case(tmp_path / "drift")
    checkpoint = json.loads(drift["checkpoint"].read_bytes())
    checkpoint["checkpoint_sha256"] = "0" * 64
    _write_json(drift["checkpoint"], checkpoint, canonical=True)
    with pytest.raises(T22048ValidationError, match="digest mismatch"):
        _validate(drift)


def test_pilot_rejects_checkpoint_inventory_object_path_drift(
    tmp_path: pathlib.Path,
) -> None:
    case = _case(tmp_path)
    checkpoint = json.loads(case["checkpoint"].read_bytes())
    checkpoint["files"][0]["object_name"] = checkpoint["files"][0][
        "object_name"
    ].replace("/shards/00/", "/shards/01/")
    unsigned = dict(checkpoint)
    unsigned.pop("checkpoint_sha256")
    checkpoint["checkpoint_sha256"] = hashlib.sha256(
        _canonical(unsigned)
    ).hexdigest()
    _write_json(case["checkpoint"], checkpoint, canonical=True)
    with pytest.raises(T22048ValidationError, match="wrong shard prefix"):
        _validate(case)


def test_pilot_rejects_noncanonical_done_even_when_checkpoint_hash_matches(
    tmp_path: pathlib.Path,
) -> None:
    case = _case(tmp_path)
    done = case["shard_directory"] / "SHARD_DONE.json"
    done.write_text(json.dumps(json.loads(done.read_bytes()), indent=2), encoding="utf-8")
    _rebuild_checkpoint(case)
    with pytest.raises(T22048ValidationError, match="not canonical"):
        _validate(case)


def test_pilot_rejects_byte_identical_checkpoint_from_another_receive_root(
    tmp_path: pathlib.Path,
) -> None:
    case = _case(tmp_path / "case")
    foreign = tmp_path / "foreign" / "_audit" / "shard_00" / "complete.json"
    foreign.parent.mkdir(parents=True)
    shutil.copyfile(case["checkpoint"], foreign)
    with pytest.raises(T22048ValidationError, match="same receive root"):
        validate_pilot_shard(
            worker_plan_path=case["plan"],
            shard_id=case["shard"]["shard_id"],
            shard_directory=case["shard_directory"],
            checkpoint_path=foreign,
            contract=case["contract"],
        )


def test_pilot_rejects_stale_same_root_receive_evidence(
    tmp_path: pathlib.Path,
) -> None:
    case = _case(tmp_path)
    evidence = json.loads(case["receive_evidence"].read_bytes())
    evidence["checkpoint_sha256"] = "0" * 64
    _write_json(case["receive_evidence"], evidence, canonical=True)
    with pytest.raises(T22048ValidationError, match="stale or foreign"):
        _validate(case)


@pytest.mark.parametrize("linked_component", ["shard", "checkpoint"])
def test_pilot_rejects_symlinked_shard_or_checkpoint_path_chain(
    tmp_path: pathlib.Path, linked_component: str
) -> None:
    case = _case(tmp_path)
    try:
        if linked_component == "shard":
            link = case["shard_directory"]
            target = tmp_path / "real-shard"
            link.rename(target)
            link.symlink_to(target, target_is_directory=True)
        else:
            link = case["checkpoint"]
            target = tmp_path / "real-complete.json"
            link.rename(target)
            link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")
    with pytest.raises(T22048ValidationError, match="symlink or junction"):
        _validate(case)


def test_pilot_cli_emits_pass_json_and_optional_report(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    case = _case(tmp_path)
    output = tmp_path / "pilot_report.json"
    result = main(
        [
            "--worker-plan",
            str(case["plan"]),
            "--shard-id",
            case["shard"]["shard_id"],
            "--shard-directory",
            str(case["shard_directory"]),
            "--checkpoint",
            str(case["checkpoint"]),
            "--output",
            str(output),
        ],
        contract=case["contract"],
    )
    assert result == 0
    stdout = json.loads(capsys.readouterr().out)
    assert stdout["status"] == "PASS"
    assert json.loads(output.read_text(encoding="utf-8")) == stdout
