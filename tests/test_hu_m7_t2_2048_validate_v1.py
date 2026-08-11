from __future__ import annotations

import hashlib
import io
import json
import pathlib
import tarfile
import zipfile
from dataclasses import replace

import pytest

from ofc_regular.action_key import action_key
from ofc_regular.action_space import generate_actions
from ofc_regular.hu_infoset import ActorObservation, ScoringContext
from ofc_regular.hu_m7_t2_2048_validate_v1 import (
    DEFAULT_CONTRACT,
    T22048Contract,
    T22048ValidationError,
    validate_postflight,
    validate_preflight,
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


def _write_json(path: pathlib.Path, value: object, *, canonical: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if canonical:
        path.write_bytes(_canonical(value))
    else:
        path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _build_package(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path, T22048Contract]:
    package = tmp_path / "package"
    package.mkdir()
    startup = tmp_path / "startup.sh"
    startup.write_bytes(b"#!/bin/sh\nexit 0\n")

    member_bytes = {
        "native/libofc_hu_m3_engine.so": b"engine-fixture",
        "native/libofc_stage3_feature_encoder.so": b"encoder-fixture",
        "weights/t4_model_v6.bin": b"t4-fixture",
        "weights/t3first_model_v2.bin": b"t3first-fixture",
        "weights/t3_model_v3.bin": b"t3second-fixture",
        "weights/t2_model_v1.bin": b"t2second-fixture",
        "configs/fl_ev_regular_v4_selfplay.json": _canonical(
            {"schema": "fixture", "fl_ev": {"14": 9.6}}
        ),
        "src/ofc_regular/hu_m31_label_gen_worker_v1.py": b"# worker fixture\n",
    }
    runtime = package / "runtime.tar.gz"
    with tarfile.open(runtime, "w:gz") as archive:
        for relative, payload in sorted(member_bytes.items()):
            info = tarfile.TarInfo(f"runtime/{relative}")
            info.size = len(payload)
            info.mtime = 0
            archive.addfile(info, io.BytesIO(payload))

    wheelhouse = package / "wheelhouse.zip"
    with zipfile.ZipFile(wheelhouse, "w") as archive:
        archive.writestr("wheelhouse/fixture.whl", b"wheel fixture")

    digest = lambda relative: hashlib.sha256(member_bytes[relative]).hexdigest()
    partial = replace(
        DEFAULT_CONTRACT,
        positions_per_seat=4,
        shards_per_seat=2,
        hand_seed_base=1_000,
        behavior_seed_offset=500,
        eval_seed_base=2_000,
        runtime_archive_sha256=_sha(runtime),
        wheelhouse_archive_sha256=_sha(wheelhouse),
        startup_script_sha256=_sha(startup),
        engine_library_sha256=digest("native/libofc_hu_m3_engine.so"),
        feature_encoder_library_sha256=digest(
            "native/libofc_stage3_feature_encoder.so"
        ),
        t4_model_sha256=digest("weights/t4_model_v6.bin"),
        t3_first_model_sha256=digest("weights/t3first_model_v2.bin"),
        t3_second_model_sha256=digest("weights/t3_model_v3.bin"),
        t2_second_model_sha256=digest("weights/t2_model_v1.bin"),
        fl_ev_config_sha256=digest("configs/fl_ev_regular_v4_selfplay.json"),
        worker_sha256=digest("src/ofc_regular/hu_m31_label_gen_worker_v1.py"),
    )
    ledger = {
        "schema": "hu_m31_label_gen_package_ledger_v1",
        "engine_library_sha256": partial.engine_library_sha256,
        "feature_encoder_library_sha256": partial.feature_encoder_library_sha256,
        "t4_model_v6_sha256": partial.t4_model_sha256,
        "t3_first_model_v2_sha256": partial.t3_first_model_sha256,
        "t3_second_model_v3_sha256": partial.t3_second_model_sha256,
        "t2_second_model_sha256": partial.t2_second_model_sha256,
        "runtime_archive": {"sha256": partial.runtime_archive_sha256},
        "wheelhouse_archive": {"sha256": partial.wheelhouse_archive_sha256},
        "fl_ev": {
            "cards": partial.fl_ev_cards,
            "value": partial.fl_ev_value,
            "config_sha256": partial.fl_ev_config_sha256,
        },
    }
    ledger_path = package / "ledger.json"
    _write_json(ledger_path, ledger)
    contract = replace(partial, ledger_sha256=_sha(ledger_path))
    return package, startup, contract


def _plan(seat: str, contract: T22048Contract) -> dict[str, object]:
    plan: dict[str, object] = {
        "schema": "hu_m31_label_gen_plan_v1",
        "job_id": f"m7v4-t2{seat}-2048p-25k-test",
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
        "shards": [
            {"shard_id": "00", "start": 0, "count": 2},
            {"shard_id": "01", "start": 2, "count": 2},
        ],
    }
    if seat == "first":
        plan.update(
            {
                "t2_second_model": "weights/t2_model_v1.bin",
                "t2_second_model_sha256": contract.t2_second_model_sha256,
            }
        )
    return plan


@pytest.fixture
def contract_fixture(tmp_path: pathlib.Path):
    package, startup, contract = _build_package(tmp_path)
    first = tmp_path / "first_plan.json"
    second = tmp_path / "second_plan.json"
    _write_json(first, _plan("first", contract))
    _write_json(second, _plan("second", contract))
    contract = replace(
        contract,
        first_plan_sha256=_sha(first),
        second_plan_sha256=_sha(second),
    )
    return package, startup, contract, first, second


def test_preflight_accepts_exact_paired_contract(contract_fixture) -> None:
    package, startup, contract, first, second = contract_fixture
    report = validate_preflight(
        first_plan_path=first,
        second_plan_path=second,
        package_dir=package,
        startup_script=startup,
        contract=contract,
    )
    assert report["status"] == "PASS"
    assert report["pair"]["total_shards"] == 4
    assert report["pair"]["paired_hand_seed_range"] == [1000, 1003]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda plan: plan.__setitem__("samples", 512), "samples"),
        (lambda plan: plan.__setitem__("probe", {"rungs": [512]}), "unknown"),
        (
            lambda plan: plan["shards"][1].__setitem__("start", 1),
            "start=2",
        ),
    ],
)
def test_preflight_rejects_non_2048_probe_and_bad_partition(
    contract_fixture, mutation, message: str
) -> None:
    package, startup, contract, first, second = contract_fixture
    payload = json.loads(first.read_text(encoding="utf-8"))
    mutation(payload)
    _write_json(first, payload)
    with pytest.raises(T22048ValidationError, match=message):
        validate_preflight(
            first_plan_path=first,
            second_plan_path=second,
            package_dir=package,
            startup_script=startup,
            contract=contract,
        )


def test_preflight_enforces_first_seat_t2_second_asymmetry(contract_fixture) -> None:
    package, startup, contract, first, second = contract_fixture
    first_payload = json.loads(first.read_text(encoding="utf-8"))
    del first_payload["t2_second_model"]
    del first_payload["t2_second_model_sha256"]
    _write_json(first, first_payload)
    with pytest.raises(T22048ValidationError, match="missing"):
        validate_preflight(
            first_plan_path=first,
            second_plan_path=second,
            package_dir=package,
            startup_script=startup,
            contract=contract,
        )


def test_preflight_rejects_package_byte_drift(contract_fixture) -> None:
    package, startup, contract, first, second = contract_fixture
    with (package / "runtime.tar.gz").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(T22048ValidationError, match="runtime archive digest mismatch"):
        validate_preflight(
            first_plan_path=first,
            second_plan_path=second,
            package_dir=package,
            startup_script=startup,
            contract=contract,
        )


def _observation(seat: str, contract: T22048Contract) -> ActorObservation:
    scoring = ScoringContext(fl_ev=((contract.fl_ev_cards, contract.fl_ev_value),))
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
        scoring=scoring,
    )


def _skeleton(observation: ActorObservation) -> str:
    text = (
        "".join(sorted(observation.hero_board.all_cards()))
        + "|"
        + "".join(sorted(observation.opponent_public_board.all_cards()))
    )
    return hashlib.sha256(text.encode()).hexdigest()[:2]


def _write_postflight_seat(
    root: pathlib.Path,
    plan_path: pathlib.Path,
    seat: str,
    contract: T22048Contract,
) -> None:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan_sha = _sha(plan_path)
    observation = _observation(seat, contract)
    actions = generate_actions(observation.hero_board, observation.dealt_cards)
    scores = {
        action_key(action).to_token(): float(index) / 10.0
        for index, action in enumerate(actions)
    }
    for shard in plan["shards"]:
        directory = root / f"shard_{shard['shard_id']}"
        directory.mkdir(parents=True, exist_ok=True)
        for offset in range(shard["start"], shard["start"] + shard["count"]):
            record = {
                "schema": "hu_m31_label_gen_position_v1",
                "plan_sha256": plan_sha,
                "offset": offset,
                "skeleton": _skeleton(observation),
                "observation": observation.to_dict(),
                "samples": contract.samples,
                "runs": [{"seed_trial": 0, "scores": scores}],
            }
            _write_json(
                directory / f"position_{offset:08d}.json", record, canonical=True
            )
        marker = {
            "schema": "hu_m31_label_gen_shard_done_v1",
            "plan_sha256": plan_sha,
            "shard_id": shard["shard_id"],
            "positions": shard["count"],
        }
        _write_json(directory / "SHARD_DONE.json", marker, canonical=True)


def _build_postflight(contract_fixture, tmp_path: pathlib.Path):
    package, startup, contract, first, second = contract_fixture
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    _write_postflight_seat(first_root, first, "first", contract)
    _write_postflight_seat(second_root, second, "second", contract)
    return contract, first, second, first_root, second_root


def test_postflight_accepts_complete_all_action_corpus(
    contract_fixture, tmp_path: pathlib.Path
) -> None:
    contract, first, second, first_root, second_root = _build_postflight(
        contract_fixture, tmp_path
    )
    report = validate_postflight(
        first_plan_path=first,
        second_plan_path=second,
        first_root=first_root,
        second_root=second_root,
        contract=contract,
    )
    assert report["status"] == "PASS"
    assert report["first"]["positions"] == 4
    assert report["second"]["positions"] == 4
    assert report["first"]["total_action_scores"] > 0


def test_postflight_rejects_hidden_opponent_discard_field(
    contract_fixture, tmp_path: pathlib.Path
) -> None:
    contract, first, second, first_root, second_root = _build_postflight(
        contract_fixture, tmp_path
    )
    path = first_root / "shard_00" / "position_00000000.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["observation"]["opponent_private_discards"] = ["8h"]
    _write_json(path, record, canonical=True)
    with pytest.raises(T22048ValidationError, match="forbidden private-truth"):
        validate_postflight(
            first_plan_path=first,
            second_plan_path=second,
            first_root=first_root,
            second_root=second_root,
            contract=contract,
        )


def test_postflight_rejects_missing_legal_action_and_marker(
    contract_fixture, tmp_path: pathlib.Path
) -> None:
    contract, first, second, first_root, second_root = _build_postflight(
        contract_fixture, tmp_path
    )
    path = second_root / "shard_00" / "position_00000000.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["runs"][0]["scores"].pop(next(iter(record["runs"][0]["scores"])))
    _write_json(path, record, canonical=True)
    with pytest.raises(T22048ValidationError, match="action set differs"):
        validate_postflight(
            first_plan_path=first,
            second_plan_path=second,
            first_root=first_root,
            second_root=second_root,
            contract=contract,
        )

    _write_postflight_seat(second_root, second, "second", contract)
    (second_root / "shard_01" / "SHARD_DONE.json").unlink()
    with pytest.raises(T22048ValidationError, match="SHARD_DONE"):
        validate_postflight(
            first_plan_path=first,
            second_plan_path=second,
            first_root=first_root,
            second_root=second_root,
            contract=contract,
        )
