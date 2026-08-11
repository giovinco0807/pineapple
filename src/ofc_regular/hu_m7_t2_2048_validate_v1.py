"""Strict preflight and postflight validation for the M7 T2 2048p corpus.

The generic label worker deliberately accepts several historical plan shapes.
This module is narrower: it validates the one paired 25,000-position T2 run
approved for the M7 v5 (FL EV 9.6) cascade and fails closed on every semantic
field.  It never launches instances or writes cloud state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import re
import sys
import tarfile
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .action_key import ActionKey, action_key
from .action_space import generate_actions
from .hu_infoset import ActorObservation


PLAN_SCHEMA = "hu_m31_label_gen_plan_v1"
POSITION_SCHEMA = "hu_m31_label_gen_position_v1"
DONE_SCHEMA = "hu_m31_label_gen_shard_done_v1"
LEDGER_SCHEMA = "hu_m31_label_gen_package_ledger_v1"

_POSITION_RE = re.compile(r"^position_([0-9]{8})\.json$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class T22048ValidationError(ValueError):
    """The paired plan or its produced corpus violates the frozen contract."""


@dataclass(frozen=True)
class T22048Contract:
    positions_per_seat: int = 25_000
    shards_per_seat: int = 174
    samples: int = 2_048
    seeds_per_position: int = 1
    hand_seed_base: int = 947_000_000
    behavior_seed_offset: int = 500_000
    eval_seed_base: int = 10_000_000
    fl_ev_cards: int = 14
    fl_ev_value: float = 9.6

    runtime_archive_sha256: str = (
        "ae8be6cd2aa333654141368c1ad713a1c6ab3ff1dedea752edcdeab2551d1988"
    )
    wheelhouse_archive_sha256: str = (
        "338ca072984775dc886e7c9c88d1d7e10b5363ae4aff5a7ba936eebf57d25053"
    )
    ledger_sha256: str = (
        "d124e98617a34d7767f0da9e04b5ceea489c97a52009b7779ffd150b20ee5dcf"
    )
    startup_script_sha256: str = (
        "5d4658ea56fc09853acc56bf4a25cec6ddbf61c413a10dc2c6a71ba2d797e472"
    )
    engine_library_sha256: str = (
        "e17aa39f797e01e487d0dbcc726073da420ef6fc2c61fb0aa7a47cb5add3670f"
    )
    feature_encoder_library_sha256: str = (
        "82510e563ee290cd536e7aebe6bcf0efefac061af5488851f0a582bb4ef83411"
    )
    t4_model_sha256: str = (
        "763b77a025bd3ce3a0f003498f7600e6d082cba321dfda38ff5f25ed1306668c"
    )
    t3_first_model_sha256: str = (
        "2d94b3e3776bc54f34a95b56b5c5e335143f72f577b2c8761be0aecca8a54a19"
    )
    t3_second_model_sha256: str = (
        "75119ebeae1b9250fd10d420806c527015aac33c19d8d04b4e73734ed0085df4"
    )
    t2_second_model_sha256: str = (
        "996c430e88e0f8a27600e91d45d0c5440bfb741341a9acc816309841d06e0301"
    )
    fl_ev_config_sha256: str = (
        "0cfe78f71e03b71503b18a960e319e638403e2849d80f78a54c7494fc49a3fea"
    )
    worker_sha256: str = (
        "1f505605d1cd87e9fdfa3d8309c75d5972f759bd0495cba06c402b94bc41846e"
    )
    first_plan_sha256: str = (
        "358445e90543819977479365650b49b144458731f4a224f56330696171adfa91"
    )
    second_plan_sha256: str = (
        "8800ab83b976e117c6a5652a7a8d58fa4dd93b5ed104da05c5b1e59bb7f5573b"
    )


DEFAULT_CONTRACT = T22048Contract()

_ARTIFACT_PATHS = {
    "engine_library": "native/libofc_hu_m3_engine.so",
    "feature_encoder_library": "native/libofc_stage3_feature_encoder.so",
    "t4_model": "weights/t4_model_v6.bin",
    "t3_first_model": "weights/t3first_model_v2.bin",
    "t3_second_model": "weights/t3_model_v3.bin",
    "t2_second_model": "weights/t2_model_v1.bin",
}

_ARCHIVE_DIGEST_FIELDS = {
    "native/libofc_hu_m3_engine.so": "engine_library_sha256",
    "native/libofc_stage3_feature_encoder.so": "feature_encoder_library_sha256",
    "weights/t4_model_v6.bin": "t4_model_sha256",
    "weights/t3first_model_v2.bin": "t3_first_model_sha256",
    "weights/t3_model_v3.bin": "t3_second_model_sha256",
    "weights/t2_model_v1.bin": "t2_second_model_sha256",
    "configs/fl_ev_regular_v4_selfplay.json": "fl_ev_config_sha256",
    "src/ofc_regular/hu_m31_label_gen_worker_v1.py": "worker_sha256",
}

_PLAN_BASE_FIELDS = {
    "schema",
    "job_id",
    "street",
    "seat",
    "samples",
    "seeds_per_position",
    "hand_seed_base",
    "behavior_seed_offset",
    "eval_seed_base",
    "engine_library",
    "engine_library_sha256",
    "feature_encoder_library",
    "feature_encoder_library_sha256",
    "t4_model",
    "t4_model_sha256",
    "t3_first_model",
    "t3_first_model_sha256",
    "t3_second_model",
    "t3_second_model_sha256",
    "fl_ev_cards",
    "fl_ev_value",
    "shards",
}
_PLAN_METADATA_FIELDS = {"notes", "provenance"}
_FIRST_ONLY_FIELDS = {"t2_second_model", "t2_second_model_sha256"}

_POSITION_FIELDS = {
    "schema",
    "plan_sha256",
    "offset",
    "skeleton",
    "observation",
    "samples",
    "runs",
}
_RUN_FIELDS = {"seed_trial", "scores"}

_FORBIDDEN_PRIVATE_KEYS = {
    "opponent_private_discard",
    "opponent_private_discards",
    "opponent_discard_card",
    "opponent_discard_cards",
    "opponent_hidden_discard",
    "opponent_hidden_discards",
    "opponent_private_card",
    "opponent_private_cards",
    "realized_deck_tail",
    "realised_deck_tail",
    "deck_tail",
    "future_deck",
    "hidden_truth",
    "world_state",
}


def _fail(message: str) -> None:
    raise T22048ValidationError(message)


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        _fail(f"value is not canonical JSON: {exc}")


def _object_no_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(token: str) -> None:
    _fail(f"non-finite JSON number is forbidden: {token}")


def _read_json(path: pathlib.Path, *, canonical: bool = False) -> dict[str, Any]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        _fail(f"cannot read {path}: {exc}")
    try:
        value = json.loads(
            data,
            object_pairs_hook=_object_no_duplicates,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        _fail(f"invalid JSON at {path}: {exc}")
    if not isinstance(value, dict):
        _fail(f"{path} must contain a JSON object")
    if canonical and data != _canonical_bytes(value):
        _fail(f"{path} is not canonical create-only JSON")
    return value


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
    except OSError as exc:
        _fail(f"cannot hash {path}: {exc}")
    return digest.hexdigest()


def _require_sha(value: Any, expected: str, label: str) -> None:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        _fail(f"{label} is not a lowercase sha256: {value!r}")
    if value != expected:
        _fail(f"{label} digest mismatch: {value}, expected {expected}")


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], label: str
) -> None:
    actual = set(value)
    if actual != expected:
        _fail(
            f"{label} fields differ: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _require_int(value: Any, expected: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        _fail(f"{label} must be integer {expected}, got {value!r}")


def _contract_digest(contract: T22048Contract) -> str:
    return hashlib.sha256(_canonical_bytes(asdict(contract))).hexdigest()


def _validate_shards(
    plan: Mapping[str, Any], contract: T22048Contract, label: str
) -> list[dict[str, int | str]]:
    rows = plan.get("shards")
    if not isinstance(rows, list):
        _fail(f"{label}.shards must be a list")
    if len(rows) != contract.shards_per_seat:
        _fail(
            f"{label} must have {contract.shards_per_seat} shards, got {len(rows)}"
        )
    normalized: list[dict[str, int | str]] = []
    seen_ids: set[str] = set()
    shard_id_width = max(2, len(str(contract.shards_per_seat - 1)))
    base_count, larger_shards = divmod(
        contract.positions_per_seat, contract.shards_per_seat
    )
    smaller_shards = contract.shards_per_seat - larger_shards
    expected_start = 0
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            _fail(f"{label}.shards[{index}] must be an object")
        _require_exact_keys(row, {"shard_id", "start", "count"}, f"{label}.shards[{index}]")
        shard_id = row.get("shard_id")
        expected_id = f"{index:0{shard_id_width}d}"
        if shard_id != expected_id:
            _fail(
                f"{label}.shards[{index}].shard_id must be {expected_id!r}, "
                f"got {shard_id!r}"
            )
        if shard_id in seen_ids:
            _fail(f"{label} repeats shard_id {shard_id!r}")
        seen_ids.add(str(shard_id))
        start = row.get("start")
        count = row.get("count")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or start < 0
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
        ):
            _fail(f"{label}.shards[{index}] has invalid start/count")
        expected_count = base_count if index < smaller_shards else base_count + 1
        if start != expected_start or count != expected_count:
            _fail(
                f"{label}.shards[{index}] must be start={expected_start}, "
                f"count={expected_count}; got start={start}, count={count}"
            )
        expected_start += expected_count
        normalized.append({"shard_id": str(shard_id), "start": start, "count": count})

    cursor = 0
    for row in sorted(normalized, key=lambda item: int(item["start"])):
        start = int(row["start"])
        count = int(row["count"])
        if start != cursor:
            kind = "overlap" if start < cursor else "gap"
            _fail(f"{label} shard partition has a {kind} at offset {cursor}: next={start}")
        cursor += count
    if cursor != contract.positions_per_seat:
        _fail(
            f"{label} shard partition ends at {cursor}, expected "
            f"{contract.positions_per_seat}"
        )
    return normalized


def _validate_plan(
    plan: Mapping[str, Any], seat: str, contract: T22048Contract, label: str
) -> dict[str, Any]:
    expected_fields = _PLAN_BASE_FIELDS | _PLAN_METADATA_FIELDS
    if seat == "first":
        expected_fields |= _FIRST_ONLY_FIELDS
    # Notes and provenance are optional, but no other ignored field is allowed.
    allowed = expected_fields
    required = expected_fields - _PLAN_METADATA_FIELDS
    actual = set(plan)
    if not required <= actual or not actual <= allowed:
        _fail(
            f"{label} fields differ: missing={sorted(required - actual)}, "
            f"unknown={sorted(actual - allowed)}"
        )
    if "probe" in plan or "plan_kind" in plan:
        _fail(f"{label} is a probe/non-joint plan, not a production label plan")
    if plan.get("schema") != PLAN_SCHEMA:
        _fail(f"{label}.schema must be {PLAN_SCHEMA!r}")
    if plan.get("street") != "T2" or plan.get("seat") != seat:
        _fail(f"{label} must be T2/{seat}")
    if plan.get("job_id") is None or not isinstance(plan.get("job_id"), str):
        _fail(f"{label}.job_id must be a string")
    job_id = str(plan["job_id"])
    expected_token = "t2first" if seat == "first" else "t2second"
    if expected_token not in job_id or "2048p" not in job_id or "25k" not in job_id:
        _fail(
            f"{label}.job_id must identify {expected_token}, 2048p and 25k; got {job_id!r}"
        )

    for field in (
        "samples",
        "seeds_per_position",
        "hand_seed_base",
        "behavior_seed_offset",
        "eval_seed_base",
        "fl_ev_cards",
    ):
        _require_int(plan.get(field), int(getattr(contract, field)), f"{label}.{field}")
    value = plan.get("fl_ev_value")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{label}.fl_ev_value must be a finite real number")
    if not math.isfinite(float(value)) or float(value) != contract.fl_ev_value:
        _fail(
            f"{label}.fl_ev_value must be {contract.fl_ev_value}, got {value!r}"
        )

    pins = {
        "engine_library": contract.engine_library_sha256,
        "feature_encoder_library": contract.feature_encoder_library_sha256,
        "t4_model": contract.t4_model_sha256,
        "t3_first_model": contract.t3_first_model_sha256,
        "t3_second_model": contract.t3_second_model_sha256,
    }
    if seat == "first":
        pins["t2_second_model"] = contract.t2_second_model_sha256
    for field, digest in pins.items():
        path_value = plan.get(field)
        if path_value != _ARTIFACT_PATHS[field]:
            _fail(
                f"{label}.{field} must be {_ARTIFACT_PATHS[field]!r}, "
                f"got {path_value!r}"
            )
        _require_sha(plan.get(f"{field}_sha256"), digest, f"{label}.{field}")

    if seat == "second" and any(field in plan for field in _FIRST_ONLY_FIELDS):
        _fail(f"{label} must not pin T2-second weights on the second-seat root")
    shards = _validate_shards(plan, contract, label)
    return {"job_id": job_id, "shards": shards}


def _validate_plan_pair_payloads(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    contract: T22048Contract,
) -> dict[str, Any]:
    first_result = _validate_plan(first, "first", contract, "first plan")
    second_result = _validate_plan(second, "second", contract, "second plan")
    if first_result["job_id"] == second_result["job_id"]:
        _fail("first and second plans must have different write-once job IDs")
    for field in ("hand_seed_base", "behavior_seed_offset", "eval_seed_base"):
        if first.get(field) != second.get(field):
            _fail(f"paired plans disagree on {field}")
    combined = len(first_result["shards"]) + len(second_result["shards"])
    expected = 2 * contract.shards_per_seat
    if combined != expected:
        _fail(f"paired plan must have {expected} total shard entries, got {combined}")
    return {
        "total_shards": combined,
        "positions_per_seat": contract.positions_per_seat,
        "paired_hand_seed_range": [
            contract.hand_seed_base,
            contract.hand_seed_base + contract.positions_per_seat - 1,
        ],
        "paired_behavior_seed_range": [
            contract.hand_seed_base + contract.behavior_seed_offset,
            contract.hand_seed_base
            + contract.behavior_seed_offset
            + contract.positions_per_seat
            - 1,
        ],
        "paired_eval_seed_range": [
            contract.eval_seed_base,
            contract.eval_seed_base + (contract.positions_per_seat - 1) * 7,
        ],
    }


def _validate_package(
    package_dir: pathlib.Path,
    startup_script: pathlib.Path,
    contract: T22048Contract,
) -> dict[str, Any]:
    runtime = package_dir / "runtime.tar.gz"
    wheelhouse = package_dir / "wheelhouse.zip"
    ledger_path = package_dir / "ledger.json"
    for path in (runtime, wheelhouse, ledger_path, startup_script):
        if not path.is_file():
            _fail(f"required package artifact is missing: {path}")
    _require_sha(_sha256(runtime), contract.runtime_archive_sha256, "runtime archive")
    _require_sha(
        _sha256(wheelhouse), contract.wheelhouse_archive_sha256, "wheelhouse archive"
    )
    _require_sha(_sha256(ledger_path), contract.ledger_sha256, "package ledger")
    _require_sha(_sha256(startup_script), contract.startup_script_sha256, "label startup")

    ledger = _read_json(ledger_path)
    if ledger.get("schema") != LEDGER_SCHEMA:
        _fail(f"unsupported package ledger schema: {ledger.get('schema')!r}")
    ledger_checks = {
        "engine_library_sha256": contract.engine_library_sha256,
        "feature_encoder_library_sha256": contract.feature_encoder_library_sha256,
        "t4_model_v6_sha256": contract.t4_model_sha256,
        "t3_first_model_v2_sha256": contract.t3_first_model_sha256,
        "t3_second_model_v3_sha256": contract.t3_second_model_sha256,
        "t2_second_model_sha256": contract.t2_second_model_sha256,
    }
    for field, expected in ledger_checks.items():
        _require_sha(ledger.get(field), expected, f"ledger.{field}")
    runtime_row = ledger.get("runtime_archive")
    wheel_row = ledger.get("wheelhouse_archive")
    if not isinstance(runtime_row, Mapping) or not isinstance(wheel_row, Mapping):
        _fail("ledger archive rows must be objects")
    _require_sha(
        runtime_row.get("sha256"), contract.runtime_archive_sha256, "ledger.runtime_archive"
    )
    _require_sha(
        wheel_row.get("sha256"),
        contract.wheelhouse_archive_sha256,
        "ledger.wheelhouse_archive",
    )
    fl = ledger.get("fl_ev")
    if not isinstance(fl, Mapping):
        _fail("ledger.fl_ev must be an object")
    _require_int(fl.get("cards"), contract.fl_ev_cards, "ledger.fl_ev.cards")
    if fl.get("value") != contract.fl_ev_value:
        _fail(f"ledger.fl_ev.value must be {contract.fl_ev_value}")
    _require_sha(
        fl.get("config_sha256"), contract.fl_ev_config_sha256, "ledger.fl_ev.config"
    )

    archive_hashes: dict[str, str] = {}
    try:
        with tarfile.open(runtime, "r:gz") as archive:
            for relative, contract_field in _ARCHIVE_DIGEST_FIELDS.items():
                name = f"runtime/{relative}"
                try:
                    member = archive.getmember(name)
                except KeyError:
                    _fail(f"runtime archive is missing {name}")
                if not member.isfile() or member.issym() or member.islnk():
                    _fail(f"runtime archive member is not a regular file: {name}")
                stream = archive.extractfile(member)
                if stream is None:
                    _fail(f"runtime archive member cannot be read: {name}")
                actual = hashlib.sha256(stream.read()).hexdigest()
                expected = str(getattr(contract, contract_field))
                _require_sha(actual, expected, f"runtime member {relative}")
                archive_hashes[relative] = actual
    except (tarfile.TarError, OSError) as exc:
        _fail(f"runtime archive cannot be verified: {exc}")

    return {
        "runtime_archive_sha256": contract.runtime_archive_sha256,
        "wheelhouse_archive_sha256": contract.wheelhouse_archive_sha256,
        "ledger_sha256": contract.ledger_sha256,
        "startup_script_sha256": contract.startup_script_sha256,
        "runtime_members": archive_hashes,
    }


def validate_preflight(
    *,
    first_plan_path: pathlib.Path,
    second_plan_path: pathlib.Path,
    package_dir: pathlib.Path,
    startup_script: pathlib.Path,
    contract: T22048Contract = DEFAULT_CONTRACT,
) -> dict[str, Any]:
    """Validate paired plans and the immutable package before staging."""

    first = _read_json(first_plan_path)
    second = _read_json(second_plan_path)
    pair = _validate_plan_pair_payloads(first, second, contract)
    first_plan_sha = _sha256(first_plan_path)
    second_plan_sha = _sha256(second_plan_path)
    _require_sha(first_plan_sha, contract.first_plan_sha256, "first plan")
    _require_sha(second_plan_sha, contract.second_plan_sha256, "second plan")
    package = _validate_package(package_dir, startup_script, contract)
    return {
        "schema": "hu_m7_t2_2048_preflight_report_v1",
        "status": "PASS",
        "contract_sha256": _contract_digest(contract),
        "first_plan_sha256": first_plan_sha,
        "second_plan_sha256": second_plan_sha,
        "pair": pair,
        "package": package,
    }


def _walk_forbidden(value: Any, location: str = "record") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            if normalized in _FORBIDDEN_PRIVATE_KEYS:
                _fail(f"forbidden private-truth field {key!r} at {location}")
            _walk_forbidden(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk_forbidden(child, f"{location}[{index}]")


def _expected_scoring(contract: T22048Contract) -> dict[str, Any]:
    return {
        "schema": "regular_ofc_scoring_context_v1",
        "fl_ev": {str(contract.fl_ev_cards): contract.fl_ev_value},
        "middle_trips_royalty": 2,
        "hu_line_points": True,
        "scoop_bonus": 3,
        "foul_enabled": True,
        "fantasyland_cards": 14,
    }


def _skeleton(observation: ActorObservation) -> str:
    payload = (
        "".join(sorted(observation.hero_board.all_cards()))
        + "|"
        + "".join(sorted(observation.opponent_public_board.all_cards()))
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:2]


def _validate_position(
    path: pathlib.Path,
    expected_offset: int,
    expected_plan_sha: str,
    seat: str,
    contract: T22048Contract,
) -> int:
    record = _read_json(path, canonical=True)
    _require_exact_keys(record, _POSITION_FIELDS, f"position {path}")
    _walk_forbidden(record, str(path))
    if record.get("schema") != POSITION_SCHEMA:
        _fail(f"{path} has unsupported position schema")
    _require_sha(record.get("plan_sha256"), expected_plan_sha, f"{path}.plan_sha256")
    _require_int(record.get("offset"), expected_offset, f"{path}.offset")
    _require_int(record.get("samples"), contract.samples, f"{path}.samples")
    observation_payload = record.get("observation")
    if not isinstance(observation_payload, Mapping):
        _fail(f"{path}.observation must be an object")
    if observation_payload.get("seat") != seat:
        _fail(f"{path} has observation seat {observation_payload.get('seat')!r}")
    if observation_payload.get("street") != "T2":
        _fail(f"{path} is not a T2 observation")
    if observation_payload.get("to_act_order") != seat:
        _fail(f"{path} action order does not match seat")
    if observation_payload.get("scoring") != _expected_scoring(contract):
        _fail(f"{path} scoring context differs from the frozen FL 9.6 contract")
    if observation_payload.get("hero_in_fantasyland") is not False:
        _fail(f"{path} unexpectedly places the hero in Fantasyland")
    if observation_payload.get("opponent_in_fantasyland") is not False:
        _fail(f"{path} unexpectedly places the opponent in Fantasyland")
    try:
        observation = ActorObservation.from_dict(observation_payload)
    except (TypeError, ValueError) as exc:
        _fail(f"{path} has an invalid information-set observation: {exc}")
    if record.get("skeleton") != _skeleton(observation):
        _fail(f"{path} skeleton digest disagrees with its public boards")

    runs = record.get("runs")
    if not isinstance(runs, list) or len(runs) != 1:
        _fail(f"{path}.runs must contain exactly one trial")
    run = runs[0]
    if not isinstance(run, Mapping):
        _fail(f"{path}.runs[0] must be an object")
    _require_exact_keys(run, _RUN_FIELDS, f"{path}.runs[0]")
    _require_int(run.get("seed_trial"), 0, f"{path}.runs[0].seed_trial")
    scores = run.get("scores")
    if not isinstance(scores, Mapping):
        _fail(f"{path}.runs[0].scores must be an object")

    try:
        legal = generate_actions(observation.hero_board, observation.dealt_cards)
        expected_actions = {action_key(action).to_token() for action in legal}
    except (TypeError, ValueError) as exc:
        _fail(f"{path} legal actions cannot be reconstructed: {exc}")
    actual_actions = set(scores)
    if actual_actions != expected_actions:
        _fail(
            f"{path} action set differs from all legal actions: "
            f"missing={len(expected_actions - actual_actions)}, "
            f"extra={len(actual_actions - expected_actions)}"
        )
    for token, score in scores.items():
        try:
            parsed = ActionKey.from_token(str(token))
        except (TypeError, ValueError) as exc:
            _fail(f"{path} has invalid ActionKey {token!r}: {exc}")
        if parsed.to_token() != token:
            _fail(f"{path} has a non-canonical ActionKey {token!r}")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            _fail(f"{path} action {token} has a non-numeric score")
        if not math.isfinite(float(score)):
            _fail(f"{path} action {token} has a non-finite score")
    return len(scores)


def _validate_done_markers(
    root: pathlib.Path,
    plan: Mapping[str, Any],
    plan_sha: str,
    contract: T22048Contract,
    position_parents: set[pathlib.Path],
) -> None:
    markers = sorted(root.rglob("SHARD_DONE.json"))
    if len(markers) != contract.shards_per_seat:
        _fail(
            f"{root} must contain {contract.shards_per_seat} SHARD_DONE markers, "
            f"got {len(markers)}"
        )
    expected_rows = {str(row["shard_id"]): row for row in plan["shards"]}
    seen: set[str] = set()
    marker_parents: set[pathlib.Path] = set()
    for marker in markers:
        payload = _read_json(marker, canonical=True)
        _require_exact_keys(
            payload, {"schema", "plan_sha256", "shard_id", "positions"}, str(marker)
        )
        if payload.get("schema") != DONE_SCHEMA:
            _fail(f"{marker} has unsupported SHARD_DONE schema")
        _require_sha(payload.get("plan_sha256"), plan_sha, f"{marker}.plan_sha256")
        shard_id = payload.get("shard_id")
        if shard_id not in expected_rows:
            _fail(f"{marker} names unknown shard {shard_id!r}")
        if shard_id in seen:
            _fail(f"duplicate SHARD_DONE marker for shard {shard_id}")
        seen.add(str(shard_id))
        row = expected_rows[str(shard_id)]
        _require_int(payload.get("positions"), int(row["count"]), f"{marker}.positions")
        local_positions = list(marker.parent.glob("position_*.json"))
        if len(local_positions) != int(row["count"]):
            _fail(
                f"{marker.parent} holds {len(local_positions)} positions for shard "
                f"{shard_id}, expected {row['count']}"
            )
        marker_parents.add(marker.parent.resolve())
    if seen != set(expected_rows):
        _fail(f"{root} SHARD_DONE shard IDs are incomplete")
    if marker_parents != position_parents:
        _fail(f"{root} has position directories without exactly one SHARD_DONE marker")


def _validate_postflight_seat(
    *,
    root: pathlib.Path,
    plan_path: pathlib.Path,
    plan: Mapping[str, Any],
    seat: str,
    contract: T22048Contract,
) -> dict[str, Any]:
    if not root.is_dir():
        _fail(f"postflight directory is missing: {root}")
    plan_sha = _sha256(plan_path)
    position_paths = sorted(root.rglob("position_*.json"))
    if len(position_paths) != contract.positions_per_seat:
        _fail(
            f"{root} must contain exactly {contract.positions_per_seat} position files, "
            f"got {len(position_paths)}"
        )
    by_offset: dict[int, pathlib.Path] = {}
    position_parents: set[pathlib.Path] = set()
    action_counts: list[int] = []
    for path in position_paths:
        match = _POSITION_RE.fullmatch(path.name)
        if match is None:
            _fail(f"malformed position filename: {path}")
        offset = int(match.group(1))
        if offset in by_offset:
            _fail(f"duplicate position offset {offset}: {by_offset[offset]} and {path}")
        by_offset[offset] = path
        position_parents.add(path.parent.resolve())
    expected_offsets = set(range(contract.positions_per_seat))
    if set(by_offset) != expected_offsets:
        missing = sorted(expected_offsets - set(by_offset))
        extra = sorted(set(by_offset) - expected_offsets)
        _fail(f"{root} offset set differs: missing={missing[:10]}, extra={extra[:10]}")
    for offset in range(contract.positions_per_seat):
        action_counts.append(
            _validate_position(by_offset[offset], offset, plan_sha, seat, contract)
        )
    _validate_done_markers(root, plan, plan_sha, contract, position_parents)
    return {
        "seat": seat,
        "positions": len(position_paths),
        "shards": contract.shards_per_seat,
        "plan_sha256": plan_sha,
        "action_count_min": min(action_counts),
        "action_count_max": max(action_counts),
        "total_action_scores": sum(action_counts),
    }


def validate_postflight(
    *,
    first_plan_path: pathlib.Path,
    second_plan_path: pathlib.Path,
    first_root: pathlib.Path,
    second_root: pathlib.Path,
    contract: T22048Contract = DEFAULT_CONTRACT,
) -> dict[str, Any]:
    """Validate the fully received paired corpus before feature extraction."""

    first = _read_json(first_plan_path)
    second = _read_json(second_plan_path)
    pair = _validate_plan_pair_payloads(first, second, contract)
    _require_sha(_sha256(first_plan_path), contract.first_plan_sha256, "first plan")
    _require_sha(_sha256(second_plan_path), contract.second_plan_sha256, "second plan")
    first_result = _validate_postflight_seat(
        root=first_root,
        plan_path=first_plan_path,
        plan=first,
        seat="first",
        contract=contract,
    )
    second_result = _validate_postflight_seat(
        root=second_root,
        plan_path=second_plan_path,
        plan=second,
        seat="second",
        contract=contract,
    )
    return {
        "schema": "hu_m7_t2_2048_postflight_report_v1",
        "status": "PASS",
        "contract_sha256": _contract_digest(contract),
        "pair": pair,
        "first": first_result,
        "second": second_result,
    }


def _repository_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[2]


def _emit(report: Mapping[str, Any], output: pathlib.Path | None) -> None:
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--first-plan", type=pathlib.Path, required=True)
    preflight.add_argument("--second-plan", type=pathlib.Path, required=True)
    preflight.add_argument("--package-dir", type=pathlib.Path, required=True)
    preflight.add_argument(
        "--startup-script",
        type=pathlib.Path,
        default=_repository_root() / "scripts/startup_hu_m31_label_gen_v1.sh",
    )
    preflight.add_argument("--output", type=pathlib.Path)

    postflight = subparsers.add_parser("postflight")
    postflight.add_argument("--first-plan", type=pathlib.Path, required=True)
    postflight.add_argument("--second-plan", type=pathlib.Path, required=True)
    postflight.add_argument("--first-root", type=pathlib.Path, required=True)
    postflight.add_argument("--second-root", type=pathlib.Path, required=True)
    postflight.add_argument("--output", type=pathlib.Path)

    args = parser.parse_args(argv)
    try:
        if args.command == "preflight":
            report = validate_preflight(
                first_plan_path=args.first_plan,
                second_plan_path=args.second_plan,
                package_dir=args.package_dir,
                startup_script=args.startup_script,
            )
        else:
            report = validate_postflight(
                first_plan_path=args.first_plan,
                second_plan_path=args.second_plan,
                first_root=args.first_root,
                second_root=args.second_root,
            )
    except T22048ValidationError as exc:
        print(f"T2 2048 validation failed: {exc}", file=sys.stderr)
        return 1
    _emit(report, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
