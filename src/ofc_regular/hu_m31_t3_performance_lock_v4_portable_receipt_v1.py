"""No-secret portability bridge for one fully replayed M3.1 v4 receipt.

The production performance receipt normally remains path-bound and is
recomputed from its complete Windows evidence tree.  That is always the
preferred validation mode when those native paths are available.

The single receipt pinned below was already validated with that full replay.
Linux/WSL cannot interpret its embedded ``C:\\`` and ``D:\\`` paths.  For that
case only, this module accepts the exact canonical receipt bytes and replays
all path-free scientific structures.  It never resolves or opens an embedded
source path.  No secret, signature key, new seed, or replacement native binary
is introduced.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from . import hu_m31_t3_step6d_candidate02_performance_lock_v4_plan as lock_plan
from . import hu_m31_t3_step6d_full100_wave_plan_v2 as wave_v2
from . import hu_m31_t3_step6d_full100_wave_scientific_bridge_v2 as wave_bridge
from . import hu_m31_t3_step6d_performance_lock_v4_production_bridge as production
from . import merge_hu_m31_t3_step6d_candidate02_performance_lock_v4 as pure_v4
from . import run_hu_m31_t3_step6d_performance_v2 as runner


PINNED_RECEIPT_FILE_SHA256 = (
    "13e267de6a8150fa58bdd8629f068440b1570a475c6ab67154f7cbd20579ce2d"
)
PINNED_RECEIPT_SHA256 = (
    "035f14d05200782aea39f90ed6e418085a15e5b92a8b61e56bfc7448ef012275"
)
PINNED_ACCEPTED_SNAPSHOT_SHA256 = (
    "221a799a89c2d45579ceae7075950817f1eb8d49edc321b7c1cfdf7da5268d75"
)
PINNED_LIFECYCLE_CHAIN_SHA256 = (
    "92468600aeb1808b6a1bc55207ab3d8abee4368c06fee9b8408659c9417a8ee2"
)
PINNED_MERGE_VIEW_MANIFEST_SHA256 = (
    "907ee98cfb6f44acc85bfd152f81a7a726cc35052530bb2c2bc3560f44df3a9c"
)
PINNED_PURE_V4_MERGE_SHA256 = (
    "c357b06da0576fa9b6ecce32eaf9d6a64a3fd60ab413140caa3d5f3355d72965"
)
PINNED_TRANSPORT_AUDIT_SHA256 = (
    "6a6385668ca39ba2c56277cc88056e3fc7f6fd187e4598b2047666460736358c"
)
PINNED_CANDIDATE_LIBRARY_SHA256 = (
    "4050e04b22d7943da9e1691402a2b34cf3d3781041a44557f35e2dfcf366d55d"
)
PINNED_REFERENCE_LIBRARY_SHA256 = (
    "3f831615d9e751b8e1f266f14dd2009903b3ac2a4094f7e47616e0aa372a01b0"
)
PINNED_PROFILE_REGISTRY_SHA256 = (
    "d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3"
)
PINNED_RUN_CONTRACT_DIGEST = (
    "669c1efa1afeebe41fcc531c6458c9d72fffdd5df2cca751c99988a872f3e2b6"
)
PINNED_SEED_SET_SHA256 = (
    "f63b2f0cb9212e9f16d9a05c79cdd6946fec54daccfd4a212f0db08c498a9847"
)

AUDIT_SCHEMA = "hu_m31_t3_performance_lock_v4_portable_receipt_audit_v1"


def canonical_bytes(value: Any) -> bytes:
    return production.canonical_bytes(value)


def canonical_sha256(value: Any) -> str:
    return production.canonical_sha256(value)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return deepcopy(dict(value))


def _read_canonical(path: str | Path) -> tuple[dict[str, Any], bytes]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("pinned performance receipt is missing or unsafe")
    raw = source.read_bytes()
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("pinned performance receipt is not canonical JSON") from exc
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise ValueError("pinned performance receipt is not canonical JSON")
    return value, raw


def source_replay_paths_natively_available(value: Mapping[str, Any]) -> bool:
    """Check availability without touching non-native path spellings.

    On Linux/WSL a Windows drive path is not a native absolute ``Path``.  The
    function returns immediately in that case and performs no filesystem
    operation against any embedded source path.
    """

    receipt = _mapping(value, "performance receipt")
    source_paths = receipt.get("source_paths")
    accepted = receipt.get("accepted_results_snapshot")
    if (
        not isinstance(source_paths, Mapping)
        or set(source_paths) != production._SOURCE_PATH_KEYS  # type: ignore[attr-defined]
        or not isinstance(accepted, Mapping)
        or not isinstance(accepted.get("accepted_root"), str)
    ):
        return False
    raw_paths = [str(source_paths[key]) for key in sorted(source_paths)]
    raw_paths.append(str(accepted["accepted_root"]))
    native_paths: list[Path] = []
    for raw in raw_paths:
        path = Path(raw)
        if not path.is_absolute():
            return False
        native_paths.append(path)
    return all(path.exists() for path in native_paths)


def _validate_path_free_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    receipt = _mapping(value, "pinned performance receipt")
    if set(receipt) != production._RECEIPT_KEYS:  # type: ignore[attr-defined]
        raise ValueError("pinned performance receipt fields changed")
    declared = receipt.get("receipt_sha256")
    body = deepcopy(receipt)
    body.pop("receipt_sha256", None)
    if (
        declared != PINNED_RECEIPT_SHA256
        or declared != canonical_sha256(body)
        or receipt.get("schema") != production.RECEIPT_SCHEMA
        or receipt.get("status") != "qualified"
        or receipt.get("decision") != production.QUALIFIED_DECISION
    ):
        raise PermissionError("receipt is not the exact pinned qualified v4 result")

    source_paths = _mapping(receipt.get("source_paths"), "source paths")
    if (
        set(source_paths)
        != production._SOURCE_PATH_KEYS  # type: ignore[attr-defined]
        or any(not isinstance(value, str) or not value for value in source_paths.values())
    ):
        raise ValueError("pinned performance source path declarations changed")

    plan = lock_plan.validate_performance_lock_v4_plan(
        _mapping(receipt.get("performance_lock_plan"), "performance plan")
    )
    materialization = lock_plan.validate_materialization_receipt(
        _mapping(receipt.get("materialization_receipt"), "materialization")
    )
    seal = lock_plan.validate_root_seal(
        _mapping(receipt.get("root_seal"), "root seal")
    )
    pure = pure_v4.validate_candidate02_performance_lock_v4_value(
        _mapping(receipt.get("pure_v4_merge"), "pure v4 merge"),
        replay_sources=False,
    )
    wave = wave_v2.validate_wave_plan(
        _mapping(receipt.get("wave_plan"), "wave plan")
    )
    ledger = wave_v2.validate_attempt_ledger(
        wave, _mapping(receipt.get("attempt_ledger"), "attempt ledger")
    )
    lifecycle = wave_bridge.validate_validated_lifecycle_chain(
        wave_plan=wave,
        attempt_ledger=ledger,
        value=_mapping(
            receipt.get("validated_lifecycle_chain"), "lifecycle chain"
        ),
    )
    accepted = _mapping(
        receipt.get("accepted_results_snapshot"), "accepted snapshot"
    )
    observed_inventory = _mapping(
        accepted.get("observed_inventory"), "accepted observed inventory"
    )
    merge_view = _mapping(
        receipt.get("merge_view_manifest"), "merge view manifest"
    )
    transport = _mapping(receipt.get("transport_audit"), "transport audit")
    source_hashes = _mapping(
        receipt.get("source_file_sha256"), "source file hashes"
    )

    contract = _mapping(plan.get("run_contract"), "run contract")
    seed_contract = _mapping(plan.get("seed_contract"), "seed contract")
    boundary_true = (
        "all_gates_passed",
        "performance_lock_finalized",
        "one_shot_lock_consumed",
        "transport_lineage_validated",
        "performance_lock_qualified",
        "quality_pilot_authorized",
    )
    boundary_false = (
        "candidate_finalized_no_go",
        "rerun_authorized",
        "reseed_authorized",
        "artifact_fanout_authorized",
        "training_eligible",
        "training_authorized",
        "promotion_evidence",
        "promotion_authorized",
        "current_profile_changed",
        "named_profile_added",
        "runtime_policy_activated",
        "m31_complete",
        "bridge_cloud_network_invoked",
    )
    if (
        receipt.get("performance_lock_plan") != plan
        or receipt.get("materialization_receipt") != materialization
        or receipt.get("root_seal") != seal
        or receipt.get("pure_v4_merge") != pure
        or receipt.get("wave_plan") != wave
        or receipt.get("attempt_ledger") != ledger
        or receipt.get("validated_lifecycle_chain") != lifecycle
        or any(receipt.get(field) is not True for field in boundary_true)
        or any(receipt.get(field) is not False for field in boundary_false)
        or contract.get("candidate_library_sha256")
        != PINNED_CANDIDATE_LIBRARY_SHA256
        or contract.get("reference_library_sha256")
        != PINNED_REFERENCE_LIBRARY_SHA256
        or plan.get("run_contract_digest") != PINNED_RUN_CONTRACT_DIGEST
        or seed_contract.get("seed_set_sha256") != PINNED_SEED_SET_SHA256
        or source_hashes.get("current_profile_registry")
        != PINNED_PROFILE_REGISTRY_SHA256
        or plan.get("current_profile_registry", {}).get("sha256")
        != PINNED_PROFILE_REGISTRY_SHA256
        or pure.get("summary_sha256") != PINNED_PURE_V4_MERGE_SHA256
        or accepted.get("snapshot_sha256")
        != PINNED_ACCEPTED_SNAPSHOT_SHA256
        or accepted.get("accepted_job_count") != 20
        or accepted.get("accepted_object_count") != 440
        or observed_inventory.get("object_count") != 440
        or len(observed_inventory.get("objects", [])) != 440
        or lifecycle.get("chain_sha256") != PINNED_LIFECYCLE_CHAIN_SHA256
        or merge_view.get("manifest_sha256")
        != PINNED_MERGE_VIEW_MANIFEST_SHA256
        or transport.get("audit_sha256") != PINNED_TRANSPORT_AUDIT_SHA256
        or transport.get("accepted_snapshot_sha256")
        != PINNED_ACCEPTED_SNAPSHOT_SHA256
        or transport.get("validated_lifecycle_chain_sha256")
        != PINNED_LIFECYCLE_CHAIN_SHA256
        or transport.get("merge_view_manifest_sha256")
        != PINNED_MERGE_VIEW_MANIFEST_SHA256
        or transport.get("pure_v4_merge_sha256")
        != PINNED_PURE_V4_MERGE_SHA256
        or transport.get("run_contract_digest")
        != PINNED_RUN_CONTRACT_DIGEST
        or transport.get("accepted_job_count") != 20
        or transport.get("accepted_object_count") != 440
        or transport.get("paired_hand_count") != 100
        or transport.get("root_count") != 200
        or transport.get("transport_lineage_validated") is not True
        or transport.get("current_profile_changed") is not False
    ):
        raise PermissionError("pinned performance scientific identity changed")
    return receipt


def validate_pinned_portable_receipt(
    receipt_path: str | Path,
    *,
    expected_profile_sha256: str,
) -> dict[str, Any]:
    """Validate the exact receipt without dereferencing embedded source paths."""

    if expected_profile_sha256 != PINNED_PROFILE_REGISTRY_SHA256:
        raise PermissionError("portable receipt profile pin changed")
    value, raw = _read_canonical(receipt_path)
    if hashlib.sha256(raw).hexdigest() != PINNED_RECEIPT_FILE_SHA256:
        raise PermissionError("performance receipt is not the exact portable pin")
    return _validate_path_free_receipt(value)


def load_preferred_or_pinned_receipt(
    receipt_path: str | Path,
    *,
    expected_profile_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Prefer full source replay; use the exact pin only when paths are absent."""

    value, raw = _read_canonical(receipt_path)
    file_sha = hashlib.sha256(raw).hexdigest()
    if source_replay_paths_natively_available(value):
        receipt = production.validate_performance_lock_v4_production_receipt(
            receipt_path,
            expected_profile_sha256=expected_profile_sha256,
            replay_sources=True,
        )
        mode = "full_native_source_replay"
        source_paths_dereferenced = True
    else:
        receipt = validate_pinned_portable_receipt(
            receipt_path,
            expected_profile_sha256=expected_profile_sha256,
        )
        mode = "exact_pinned_portable_no_source_path_dereference"
        source_paths_dereferenced = False
    audit = {
        "schema": AUDIT_SCHEMA,
        "mode": mode,
        "receipt_file_sha256": file_sha,
        "receipt_sha256": receipt["receipt_sha256"],
        "expected_profile_sha256": expected_profile_sha256,
        "source_paths_dereferenced": source_paths_dereferenced,
        "new_seed_authorized": False,
        "replacement_binary_authorized": False,
        "current_profile_changed": False,
    }
    return receipt, {
        **audit,
        "audit_sha256": canonical_sha256(audit),
    }


__all__ = [
    "AUDIT_SCHEMA",
    "PINNED_RECEIPT_FILE_SHA256",
    "PINNED_RECEIPT_SHA256",
    "canonical_bytes",
    "canonical_sha256",
    "load_preferred_or_pinned_receipt",
    "source_replay_paths_natively_available",
    "validate_pinned_portable_receipt",
]
