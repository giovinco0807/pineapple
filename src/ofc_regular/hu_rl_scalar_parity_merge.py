"""Fail-closed merger for frozen RLB v3 scalar-parity shards."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .hu_rl_scalar_parity import (
    DECISION_COUNT,
    MAX_HANDS,
    PRIVILEGED_AUDIT_ROLE,
    SCALAR_PARITY_SUMMARY_SCHEMA,
    HuRlScalarBinaryError,
    HuRlScalarParityError,
    _atomic_write_no_clobber,
    build_scalar_parity_run_contract,
    load_scalar_parity_summary,
    scalar_parity_run_contract_digest,
    validate_runtime_source_identity,
)


SCALAR_PARITY_MERGE_SCHEMA = "regular_ofc_hu_rl_scalar_parity_merge_v3"
_MERGER_SOURCE_RELPATHS = (
    "scripts/merge_hu_rl_scalar_parity.py",
    "src/ofc_regular/hu_rl_scalar_parity_merge.py",
)


class HuRlScalarParityMergeError(ValueError):
    """A shard set or its frozen provenance failed the RLB v3 contract."""


def merge_scalar_parity_shards(
    shard_paths: Sequence[str | Path],
    *,
    binary_path: str | Path,
    profile_path: str | Path,
    expected_seed: int,
    expected_hands: int,
    expected_run_contract_sha256: str,
    source_root: str | Path,
) -> dict[str, Any]:
    """Validate complete coverage under one immutable RLB v3 producer contract."""

    if not _strict_int(expected_seed):
        raise HuRlScalarParityMergeError("expected_seed must be an integer")
    if not _strict_int(expected_hands) or not 1 <= expected_hands <= MAX_HANDS:
        raise HuRlScalarParityMergeError(
            f"expected_hands must be in [1, {MAX_HANDS}]"
        )
    _require_sha256(expected_run_contract_sha256, "expected_run_contract_sha256")
    if not shard_paths:
        raise HuRlScalarParityMergeError("at least one shard is required")
    merger_sources = _merger_source_manifest(source_root)

    loaded = [_load_v3_shard(path) for path in shard_paths]
    loaded.sort(key=lambda row: row[1]["hand_start"])
    if len({row[0].resolve() for row in loaded}) != len(loaded):
        raise HuRlScalarParityMergeError("the same shard path was supplied twice")

    next_start = 0
    request_digests: set[str] = set()
    result_digests: set[str] = set()
    ordered_proofs: list[dict[str, Any]] = []
    shard_records: list[dict[str, Any]] = []
    frozen_contract: dict[str, Any] | None = None
    for path, payload, encoded in loaded:
        if (
            payload["schema"] != SCALAR_PARITY_SUMMARY_SCHEMA
            or payload["seed"] != expected_seed
            or payload["global_hands"] != expected_hands
            or payload["hand_start"] != next_start
            or payload["run_contract_sha256"]
            != expected_run_contract_sha256
        ):
            raise HuRlScalarParityMergeError("parity shard identity changed")
        contract = dict(payload["run_contract"])
        if frozen_contract is None:
            frozen_contract = contract
        elif _canonical_bytes(contract) != _canonical_bytes(frozen_contract):
            raise HuRlScalarParityMergeError(
                "parity shards used different frozen run contracts"
            )
        for proof in payload["proofs"]:
            request_digest = proof["request_sha256"]
            result_digest = proof["result_sha256"]
            if request_digest in request_digests:
                raise HuRlScalarParityMergeError(
                    "request digest overlaps between parity shards"
                )
            request_digests.add(request_digest)
            result_digests.add(result_digest)
            ordered_proofs.append(dict(proof))
        shard_records.append(
            {
                "name": path.name,
                "sha256": _sha256_bytes(encoded),
                "size_bytes": len(encoded),
                "hand_start": payload["hand_start"],
                "hand_end_exclusive": payload["hand_end_exclusive"],
                "hands": payload["hands"],
                "workers": payload["workers"],
            }
        )
        next_start = payload["hand_end_exclusive"]

    if (
        frozen_contract is None
        or next_start != expected_hands
        or len(ordered_proofs) != expected_hands
    ):
        raise HuRlScalarParityMergeError(
            "parity shards do not cover the exact expected hand range"
        )
    if [proof["hand_ordinal"] for proof in ordered_proofs] != list(
        range(expected_hands)
    ):
        raise HuRlScalarParityMergeError("merged proof ordinals are not contiguous")

    current_contract = _rehash_current_contract(
        binary_path,
        profile_path=profile_path,
        source_root=source_root,
        seed=expected_seed,
        global_hands=expected_hands,
    )
    if (
        scalar_parity_run_contract_digest(current_contract)
        != expected_run_contract_sha256
        or _canonical_bytes(current_contract) != _canonical_bytes(frozen_contract)
    ):
        raise HuRlScalarParityMergeError("current pinned parity inputs drifted")

    receipt: dict[str, Any] = {
        "schema": SCALAR_PARITY_MERGE_SCHEMA,
        "status": "exact_python_rust_scalar_parity_merged",
        "artifact_role": PRIVILEGED_AUDIT_ROLE,
        "hands": expected_hands,
        "hand_start": 0,
        "hand_end_exclusive": expected_hands,
        "global_hands": expected_hands,
        "total_decisions": expected_hands * DECISION_COUNT,
        "seed": expected_seed,
        "shard_count": len(shard_records),
        "shards": shard_records,
        "unique_request_count": len(request_digests),
        "unique_result_count": len(result_digests),
        "ordered_proof_sha256": _sha256_bytes(_canonical_bytes(ordered_proofs)),
        "request_digest_set_sha256": _digest_set(request_digests),
        "result_digest_set_sha256": _digest_set(result_digests),
        "run_contract": frozen_contract,
        "run_contract_sha256": expected_run_contract_sha256,
        "native_binary": frozen_contract["native_binary"],
        "validator_sources": frozen_contract["validator_sources"],
        "profile_registry": frozen_contract["profile_registry"],
        "python_runtime": frozen_contract["python_runtime"],
        "merger_sources": merger_sources,
        "merger_source_manifest_sha256": _sha256_bytes(
            _canonical_bytes(merger_sources)
        ),
        "full_trace_persisted": False,
        "contains_raw_cross_actor_private_information": False,
        "contains_reconstructable_hidden_oracle_state": True,
        "policy_input_eligible": False,
        "replay_eligible": False,
        "training_eligible": False,
        "current_profile_changed": False,
    }
    receipt["self_sha256"] = _sha256_bytes(_canonical_bytes(receipt))

    end_contract = _rehash_current_contract(
        binary_path,
        profile_path=profile_path,
        source_root=source_root,
        seed=expected_seed,
        global_hands=expected_hands,
    )
    if _canonical_bytes(end_contract) != _canonical_bytes(frozen_contract):
        raise HuRlScalarParityMergeError("pinned parity inputs changed during merge")
    if _merger_source_manifest(source_root) != merger_sources:
        raise HuRlScalarParityMergeError("merger sources changed during merge")
    return receipt


def write_merge_receipt(path: str | Path, receipt: Mapping[str, Any]) -> None:
    """Atomically publish or byte-validate one immutable merge receipt."""

    if receipt.get("schema") != SCALAR_PARITY_MERGE_SCHEMA:
        raise HuRlScalarParityMergeError("merge receipt schema changed")
    expected_self = receipt.get("self_sha256")
    if not isinstance(expected_self, str):
        raise HuRlScalarParityMergeError("merge receipt self digest is missing")
    without_self = dict(receipt)
    without_self.pop("self_sha256", None)
    if expected_self != _sha256_bytes(_canonical_bytes(without_self)):
        raise HuRlScalarParityMergeError("merge receipt self digest changed")
    encoded = _canonical_bytes(receipt) + b"\n"
    _atomic_write_no_clobber(Path(path), encoded, HuRlScalarParityMergeError)


def _load_v3_shard(path: str | Path) -> tuple[Path, dict[str, Any], bytes]:
    try:
        payload, encoded = load_scalar_parity_summary(path)
    except (HuRlScalarParityError, HuRlScalarBinaryError) as exc:
        raise HuRlScalarParityMergeError("parity shard failed v3 validation") from exc
    return Path(path), payload, encoded


def _rehash_current_contract(
    binary_path: str | Path,
    *,
    profile_path: str | Path,
    source_root: str | Path,
    seed: int,
    global_hands: int,
) -> dict[str, Any]:
    try:
        return build_scalar_parity_run_contract(
            binary_path,
            profile_path=profile_path,
            source_root=source_root,
            seed=seed,
            global_hands=global_hands,
        )
    except (HuRlScalarParityError, HuRlScalarBinaryError) as exc:
        raise HuRlScalarParityMergeError(
            "current pinned parity inputs could not be verified"
        ) from exc


def _merger_source_manifest(source_root: str | Path) -> dict[str, Any]:
    root = Path(source_root)
    if root.is_symlink() or not root.is_dir():
        raise HuRlScalarParityMergeError("merger source root is missing or unsafe")
    try:
        root = root.resolve(strict=True)
        validate_runtime_source_identity(
            root,
            __file__,
            "src/ofc_regular/hu_rl_scalar_parity_merge.py",
        )
    except (OSError, HuRlScalarParityError) as exc:
        raise HuRlScalarParityMergeError(
            "executing merger source does not match frozen source_root"
        ) from exc
    result: dict[str, Any] = {}
    for relpath in _MERGER_SOURCE_RELPATHS:
        candidate = root / relpath
        if candidate.is_symlink() or not candidate.is_file():
            raise HuRlScalarParityMergeError("merger source is missing or unsafe")
        try:
            encoded = candidate.read_bytes()
        except OSError as exc:
            raise HuRlScalarParityMergeError("merger source could not be read") from exc
        if not encoded:
            raise HuRlScalarParityMergeError("merger source is empty")
        result[relpath] = {
            "sha256": _sha256_bytes(encoded),
            "size_bytes": len(encoded),
        }
    return result


def validate_merge_runtime_entrypoint(
    source_root: str | Path, entrypoint_path: str | Path
) -> None:
    """Bind the executing merge CLI to the source manifest root."""

    try:
        validate_runtime_source_identity(
            source_root,
            entrypoint_path,
            "scripts/merge_hu_rl_scalar_parity.py",
        )
    except HuRlScalarParityError as exc:
        raise HuRlScalarParityMergeError(
            "executing merge CLI does not match frozen source_root"
        ) from exc


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _digest_set(values: set[str]) -> str:
    return _sha256_bytes(("\n".join(sorted(values)) + "\n").encode("ascii"))


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise HuRlScalarParityMergeError(f"{label} must be lowercase SHA-256")


def _strict_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)
