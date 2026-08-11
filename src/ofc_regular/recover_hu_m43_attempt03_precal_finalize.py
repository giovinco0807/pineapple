"""Fail-closed recovery for an Attempt03 pre-cal receiver crash.

This recovery is deliberately row-value blind: it never JSON-decodes a teacher
row.  Exact file hashes re-establish the already-sealed content and identity
bindings in the canonical data contract before any receipt or directory is
created/moved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .validate_hu_m43_attempt03_teacher_receive import (
    validate_closure,
    validate_precal_open_authorization,
)


CLAIM_SCHEMA = "hu_m43_attempt03_precal_open_claim_v1"
CONTRACT_SCHEMA = "hu_m43_attempt03_data_contract_v1"
RECEIPT_SCHEMA = "hu_m43_attempt03_teacher_precal_receive_receipt_v1"
_HEX = frozenset("0123456789abcdef")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _mapping(path: Path, location: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be a JSON mapping")
    return value


def _nested(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be a mapping")
    return value


def _sha(value: Any, location: str) -> str:
    text = str(value)
    if len(text) != 64 or any(char not in _HEX for char in text):
        raise ValueError(f"{location} must be a lowercase SHA-256")
    return text


def _self_digest(value: Mapping[str, Any], field: str, location: str) -> str:
    claimed = _sha(value.get(field), f"{location}.{field}")
    actual = _digest({key: item for key, item in value.items() if key != field})
    if claimed != actual:
        raise ValueError(f"{location} canonical self-hash changed")
    return claimed


def _under(root: Path, path: Path, location: str) -> Path:
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{location} escapes repository root")
    return resolved


def _bound_path(root: Path, value: Any, location: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{location} path is missing")
    source = Path(value)
    return _under(root, source if source.is_absolute() else root / source, location)


def _line_count(path: Path) -> int:
    count = 0
    last = b""
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            count += block.count(b"\n")
            last = block[-1:]
    if path.stat().st_size and last != b"\n":
        raise ValueError(f"JSONL lacks final newline: {path}")
    return count


def _merge_digest(paths: Sequence[Path]) -> tuple[str, int]:
    """Mirror PowerShell/.NET ReadLines + UTF-8 StreamWriter.WriteLine on Windows."""

    digest = hashlib.sha256()
    rows = 0
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline=None) as handle:
            for line in handle:
                line = line.removesuffix("\n").removesuffix("\r")
                digest.update(line.encode("utf-8"))
                digest.update(b"\r\n")
                rows += 1
    return digest.hexdigest(), rows


def _validate_role(
    *, root: Path, role: Mapping[str, Any], expected_shards: int
) -> tuple[list[Path], list[dict[str, Any]]]:
    shards = role.get("ordered_shards")
    if not isinstance(shards, list) or len(shards) != expected_shards:
        raise ValueError("Attempt03 ordered shard count changed")
    paths: list[Path] = []
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(shards):
        row = dict(_nested(raw, f"ordered_shards[{index}]"))
        path = _bound_path(root, row.get("path"), f"ordered_shards[{index}]")
        if row.get("index") != index or row.get("records") != 10:
            raise ValueError(f"Attempt03 ordered shard geometry changed: {index}")
        if not path.is_file() or path.stat().st_size != row.get("bytes"):
            raise ValueError(f"Attempt03 ordered shard bytes changed: {index}")
        if _sha256(path) != _sha(row.get("file_sha256"), "file_sha256"):
            raise ValueError(f"Attempt03 ordered shard file hash changed: {index}")
        if _line_count(path) != 10:
            raise ValueError(f"Attempt03 ordered shard line count changed: {index}")
        _sha(row.get("canonical_rows_sha256"), "canonical_rows_sha256")
        _sha(row.get("identity_sha256"), "identity_sha256")
        paths.append(path)
        normalized.append(row)
    if role.get("records") != expected_shards * 10:
        raise ValueError("Attempt03 ordered role record count changed")
    _sha(role.get("canonical_rows_sha256"), "role canonical_rows_sha256")
    declared = _sha(role.get("ordered_shards_sha256"), "ordered_shards_sha256")
    unsigned = {
        "ordered_shards": normalized,
        "records": role["records"],
        "canonical_rows_sha256": role["canonical_rows_sha256"],
    }
    if _digest(unsigned) != declared:
        raise ValueError("Attempt03 ordered shard binding digest changed")
    return paths, normalized


def _validate_preflight_binding(
    *, preflight_path: Path, plan_path: Path, manifest_path: Path
) -> dict[str, Any]:
    preflight = _mapping(preflight_path, "Attempt03 preflight receipt")
    receipt_sha = _self_digest(
        preflight, "receipt_sha256", "Attempt03 preflight receipt"
    )
    manifest = _mapping(manifest_path, "Attempt03 cloud manifest")
    if (
        preflight.get("schema") != "hu_m43_attempt03_preflight_receipt_v1"
        or preflight.get("status") != "pass_frozen_before_fresh_generation"
        or manifest.get("preflight_file_sha256") != _sha256(preflight_path)
        or manifest.get("plan_file_sha256") != _sha256(plan_path)
    ):
        raise ValueError("Attempt03 recovery preflight/manifest binding changed")
    return {
        "receipt_sha256": receipt_sha,
        "file_sha256": _sha256(preflight_path),
    }


def _validate_fit_receipt_shards(
    *,
    root: Path,
    fit_receipt: Mapping[str, Any],
    contract_paths: Sequence[Path],
    contract_rows: Sequence[Mapping[str, Any]],
) -> None:
    received = _nested(
        fit_receipt.get("fresh_train_fit"), "fit receipt fresh_train_fit"
    ).get("shards")
    if not isinstance(received, list) or len(received) != len(contract_paths):
        raise ValueError("Attempt03 recovery fit shard receipt count changed")
    ordered = sorted(
        (dict(_nested(row, "fit receipt shard")) for row in received),
        key=lambda row: int(row.get("shard", -1)),
    )
    for index, row in enumerate(ordered):
        path = _bound_path(root, row.get("path"), f"fit receipt shard {index}")
        expected_hash = contract_rows[index]["file_sha256"]
        if (
            row.get("shard") != index
            or row.get("logical_split") != "train.fit"
            or row.get("split_shard") != index
            or row.get("rows") != 10
            or row.get("sha256") != expected_hash
            or path != contract_paths[index]
            or _sha256(path) != expected_hash
            or _line_count(path) != 10
        ):
            raise ValueError(
                f"Attempt03 recovery fit receipt/contract shard mismatch: {index}"
            )


def _receipt_payload(
    *, audit: Mapping[str, Any], received_at: str
) -> dict[str, Any]:
    claim = _nested(audit["claim"], "claim")
    contract = _nested(audit["contract"], "contract")
    return {
        "schema": RECEIPT_SCHEMA,
        "status": "verified_structural_precal_open_after_frozen_claim",
        "run_name": audit["run_name"],
        "manifest_sha256": audit["manifest_sha256"],
        "schedule_sha256": audit["schedule_sha256"],
        "model_freeze_path": claim["model_freeze_path"],
        "model_freeze_file_sha256": claim["model_freeze_file_sha256"],
        "precal_open_authorization_path": claim[
            "precal_open_authorization_path"
        ],
        "precal_open_authorization_file_sha256": claim[
            "precal_open_authorization_file_sha256"
        ],
        "precal_open_authorization_sha256": claim[
            "precal_open_authorization_sha256"
        ],
        "candidate_model_sha256": claim["candidate_model_sha256"],
        "fit_bundle_sha256": claim["fit_bundle_sha256"],
        "fit_manifest_file_sha256": claim["fit_manifest_file_sha256"],
        "fold_cloud_contract_file_sha256": claim[
            "fold_cloud_contract_file_sha256"
        ],
        "training_freeze_file_sha256": claim[
            "training_freeze_file_sha256"
        ],
        "open_claim_path": audit["claim_path"],
        "open_claim_file_sha256": audit["claim_file_sha256"],
        "verified_shards": 20,
        "verified_roots": 200,
        "fresh_precal_holdout": {
            "path": str(Path(audit["final_output_dir"]) / "fresh_precal_holdout.jsonl"),
            "rows": 200,
            "sha256": audit["merged_sha256"],
            "shards": audit["precal_receipt_shards"],
        },
        "data_contract": str(Path(audit["final_output_dir"]) / "data_contract.json"),
        "data_contract_sha256": contract["contract_sha256"],
        "downstream_model_evaluation": {
            "model_freeze_bound": True,
            "model_evaluation_count": 0,
            "global_consumption_marker": contract["precal_holdout"][
                "consumption_marker"
            ],
            "global_consumption_marker_must_be_created_exclusively_before_parse": True,
            "ready_for_exactly_one_bound_model_evaluation": True,
        },
        "sealed_calibration_opened": False,
        "inherited_locked_opened": False,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
        "received_at": received_at,
    }


def _publish_stage_no_clobber(stage: Path, final: Path) -> None:
    if final.exists():
        raise FileExistsError("Attempt03 final pre-cal output appeared before move")
    # ``rename`` is intentionally no-clobber on Windows.  The receiver's final
    # output is immutable and must never replace an independently appeared dir.
    os.rename(stage, final)


def audit_recovery_state(
    *,
    repo_root: str | Path,
    run_name: str,
    project_id: str,
    bucket: str,
    plan_path: str | Path,
    preflight_receipt_path: str | Path,
    claim_path: str | Path,
    rp_dir: str | Path,
    stage_dir: str | Path,
    final_output_dir: str | Path,
    fit_receipt_path: str | Path,
    model_freeze_path: str | Path,
    original_freeze_path: str | Path,
    freeze_lineage_path: str | Path,
    expected_claim_sha256: str,
    expected_contract_file_sha256: str,
    expected_merged_sha256: str,
    receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    plan = _under(root, Path(plan_path), "plan")
    preflight = _under(root, Path(preflight_receipt_path), "preflight receipt")
    claim_source = _under(root, Path(claim_path), "claim")
    rp = _under(root, Path(rp_dir), "rp")
    stage = _under(root, Path(stage_dir), "stage")
    final = _under(root, Path(final_output_dir), "final output")
    fit_receipt_source = _under(root, Path(fit_receipt_path), "fit receipt")
    model_freeze = _under(root, Path(model_freeze_path), "model freeze")
    original_freeze = _under(root, Path(original_freeze_path), "original freeze")
    freeze_lineage = _under(root, Path(freeze_lineage_path), "freeze lineage")
    for expected, location in (
        (expected_claim_sha256, "expected claim"),
        (expected_contract_file_sha256, "expected contract"),
        (expected_merged_sha256, "expected merged pre-cal"),
    ):
        _sha(expected, location)
    if final.exists():
        raise ValueError("Attempt03 final pre-cal output already exists")
    if not rp.is_dir() or not stage.is_dir():
        raise ValueError("Attempt03 recovery rp/stage directory is missing")

    contract_path = stage / "data_contract.json"
    merged_path = stage / "fresh_precal_holdout.jsonl"
    actual_receipt = stage / "one_shot_receipt.json"
    expected_names = {contract_path.name, merged_path.name}
    if receipt_path is not None:
        supplied = _under(root, Path(receipt_path), "one-shot receipt")
        if supplied != actual_receipt:
            raise ValueError("Attempt03 recovery receipt is not in the stage")
        expected_names.add(actual_receipt.name)
    elif actual_receipt.exists():
        raise ValueError("Attempt03 recovery receipt appeared before CreateNew")
    if {path.name for path in stage.iterdir()} != expected_names:
        raise ValueError("Attempt03 recovery stage file set changed")

    claim_file_sha = _sha256(claim_source)
    if claim_file_sha != expected_claim_sha256:
        raise ValueError("Attempt03 recovery claim file hash changed")
    claim = _mapping(claim_source, "Attempt03 pre-cal open claim")
    expected_claim_keys = {
        "schema", "status", "run_name", "model_freeze_path",
        "model_freeze_file_sha256", "original_model_freeze_file_sha256",
        "model_freeze_lineage_file_sha256", "precal_open_authorization_path",
        "precal_open_authorization_file_sha256", "precal_open_authorization_sha256",
        "candidate_model_sha256", "fit_bundle_sha256", "fit_manifest_file_sha256",
        "fold_cloud_contract_file_sha256", "training_freeze_file_sha256",
        "teacher_manifest_sha256", "schedule_sha256", "fit_receipt_file_sha256",
        "intended_shards", "global_model_evaluation_consumption_marker_created",
        "current_profile_mutated", "runtime_policy_activated", "claimed_at",
    }
    if set(claim) != expected_claim_keys or claim.get("schema") != CLAIM_SCHEMA:
        raise ValueError("Attempt03 recovery claim schema/key set changed")
    if (
        claim.get("status") != "claimed_before_any_precal_result_download_or_parse"
        or claim.get("run_name") != run_name
        or claim.get("intended_shards") != list(range(50, 70))
        or any(
            claim.get(key) is not False
            for key in (
                "global_model_evaluation_consumption_marker_created",
                "current_profile_mutated", "runtime_policy_activated",
            )
        )
    ):
        raise ValueError("Attempt03 recovery claim lifecycle changed")

    closure_paths = {
        "manifest": rp / "manifest.json",
        "schedule": rp / "source" / "shards_manifest.jsonl",
        "source": rp / "source" / "ofc_regular_hu_m43_attempt03_teacher_source.zip",
        "startup": rp / "source" / "startup_hu_m43_attempt03_teacher.sh",
        "model_manifest": rp / "source" / "source_model_manifest.json",
        "native_manifest": rp / "source" / "source_native_manifest.json",
    }
    closure = validate_closure(
        plan_path=plan,
        manifest_path=closure_paths["manifest"],
        schedule_path=closure_paths["schedule"],
        source_path=closure_paths["source"],
        startup_path=closure_paths["startup"],
        model_manifest_path=closure_paths["model_manifest"],
        native_manifest_path=closure_paths["native_manifest"],
        run_name=run_name,
        project_id=project_id,
        bucket=bucket,
    )
    preflight_binding = _validate_preflight_binding(
        preflight_path=preflight,
        plan_path=plan,
        manifest_path=closure_paths["manifest"],
    )
    authorization_source = _bound_path(
        root, claim["precal_open_authorization_path"], "authorization"
    )
    authorization = validate_precal_open_authorization(
        repo_root=root,
        authorization_path=authorization_source,
        model_freeze_path=model_freeze,
        original_freeze_path=original_freeze,
        freeze_lineage_path=freeze_lineage,
    )
    comparisons = {
        "model_freeze_file_sha256": authorization["freeze_lineage"]["amended_freeze_file_sha256"],
        "original_model_freeze_file_sha256": authorization["freeze_lineage"]["original_freeze_file_sha256"],
        "model_freeze_lineage_file_sha256": authorization["freeze_lineage"]["freeze_lineage_file_sha256"],
        "precal_open_authorization_file_sha256": authorization["authorization_file_sha256"],
        "precal_open_authorization_sha256": authorization["authorization_sha256"],
        "candidate_model_sha256": authorization["candidate_model_sha256"],
        "fit_bundle_sha256": authorization["fit_bundle_sha256"],
        "fit_manifest_file_sha256": authorization["fit_manifest_file_sha256"],
        "fold_cloud_contract_file_sha256": authorization["fold_cloud_contract_file_sha256"],
        "training_freeze_file_sha256": authorization["training_freeze_file_sha256"],
        "teacher_manifest_sha256": closure["manifest_sha256"],
        "schedule_sha256": closure["schedule_sha256"],
    }
    for key, actual in comparisons.items():
        if claim.get(key) != actual:
            raise ValueError(f"Attempt03 recovery claim binding changed: {key}")
    if _bound_path(root, claim["model_freeze_path"], "claim model freeze") != model_freeze:
        raise ValueError("Attempt03 recovery claim names a different model freeze")

    if _sha256(fit_receipt_source) != claim.get("fit_receipt_file_sha256"):
        raise ValueError("Attempt03 recovery fit receipt file hash changed")
    fit_receipt = _mapping(fit_receipt_source, "Attempt03 fit receipt")
    if (
        fit_receipt.get("schema") != "hu_m43_attempt03_teacher_fit_receive_receipt_v1"
        or fit_receipt.get("status") != "verified_fresh_train_fit_only_precal_unopened"
        or fit_receipt.get("run_name") != run_name
        or fit_receipt.get("verified_shards") != 50
        or fit_receipt.get("verified_roots") != 500
        or fit_receipt.get("precal_result_opened") is not False
        or fit_receipt.get("manifest_sha256") != closure["manifest_sha256"]
        or fit_receipt.get("schedule_sha256") != closure["schedule_sha256"]
    ):
        raise ValueError("Attempt03 recovery fit receipt lifecycle changed")

    if _sha256(contract_path) != expected_contract_file_sha256:
        raise ValueError("Attempt03 recovery data-contract file hash changed")
    contract = _mapping(contract_path, "Attempt03 data contract")
    _self_digest(contract, "contract_sha256", "Attempt03 data contract")
    if (
        contract.get("schema") != CONTRACT_SCHEMA
        or contract.get("status")
        != "pass_fresh_fit_and_one_shot_precal_sealed_calibration_locked_unopened"
        or _nested(contract.get("fit"), "fit").get("records") != 700
        or _nested(contract.get("precal_holdout"), "precal").get("records") != 200
        or contract["precal_holdout"].get("model_evaluation_count") != 0
        or contract["precal_holdout"].get("consumption_status") != "unconsumed_sealed"
        or _nested(contract.get("sealed_calibration"), "sealed").get("jsonl_content_parse_count") != 0
        or _nested(contract.get("inherited_locked"), "locked").get("jsonl_content_parse_count") != 0
    ):
        raise ValueError("Attempt03 recovery data-contract boundary changed")
    contract_preflight = _nested(contract.get("preflight"), "contract preflight")
    if dict(contract_preflight) != preflight_binding:
        raise ValueError("Attempt03 recovery contract/preflight binding changed")
    marker = _bound_path(root, contract["precal_holdout"]["consumption_marker"], "precal marker")
    if marker.exists():
        raise ValueError("Attempt03 pre-cal identity was already consumed")
    roles = _nested(_nested(contract.get("teacher_shards"), "teacher_shards").get("roles"), "roles")
    fit_paths, fit_contract_rows = _validate_role(root=root, role=_nested(roles.get("train.fit"), "fit role"), expected_shards=50)
    precal_paths, precal_rows = _validate_role(root=root, role=_nested(roles.get("train.precal_holdout"), "precal role"), expected_shards=20)
    fit_declared = _nested(fit_receipt.get("fresh_train_fit"), "fit receipt fresh_train_fit")
    if fit_declared.get("rows") != 500 or _sha256(_bound_path(root, fit_declared.get("path"), "merged fit")) != fit_declared.get("sha256"):
        raise ValueError("Attempt03 recovery merged fit binding changed")
    _validate_fit_receipt_shards(
        root=root,
        fit_receipt=fit_receipt,
        contract_paths=fit_paths,
        contract_rows=fit_contract_rows,
    )

    merged_sha = _sha256(merged_path)
    if merged_sha != expected_merged_sha256 or _line_count(merged_path) != 200:
        raise ValueError("Attempt03 recovery merged pre-cal changed")
    reconstructed_sha, reconstructed_rows = _merge_digest(precal_paths)
    if reconstructed_sha != merged_sha or reconstructed_rows != 200:
        raise ValueError("Attempt03 recovery merged pre-cal is not the ordered shard merge")
    receipt_shards = [
        {
            "shard": 50 + index,
            "logical_split": "train.precal_holdout",
            "split_shard": index,
            "rows": 10,
            "sha256": row["file_sha256"],
            "path": str(precal_paths[index]),
        }
        for index, row in enumerate(precal_rows)
    ]
    audit: dict[str, Any] = {
        "schema": "hu_m43_attempt03_claimed_precal_finalize_audit_v1",
        "status": "pass_without_teacher_row_json_decode_or_model_evaluation",
        "run_name": run_name,
        "repo_root": str(root),
        "claim_path": str(claim_source),
        "claim_file_sha256": claim_file_sha,
        "claim": claim,
        "contract": contract,
        "contract_file_sha256": expected_contract_file_sha256,
        "manifest_sha256": closure["manifest_sha256"],
        "schedule_sha256": closure["schedule_sha256"],
        "merged_sha256": merged_sha,
        "precal_receipt_shards": receipt_shards,
        "stage_dir": str(stage),
        "final_output_dir": str(final),
        "teacher_row_json_decode_count": 0,
        "model_evaluation_count": 0,
    }
    if receipt_path is not None:
        receipt = _mapping(actual_receipt, "Attempt03 one-shot receipt")
        _self_digest(receipt, "receipt_sha256", "Attempt03 one-shot receipt")
        received_at = receipt.get("received_at")
        if not isinstance(received_at, str):
            raise ValueError("Attempt03 recovery receipt timestamp missing")
        datetime.fromisoformat(received_at.replace("Z", "+00:00"))
        expected = _receipt_payload(audit=audit, received_at=received_at)
        if {key: value for key, value in receipt.items() if key != "receipt_sha256"} != expected:
            raise ValueError("Attempt03 recovery one-shot receipt binding changed")
        audit["receipt_file_sha256"] = _sha256(actual_receipt)
        audit["receipt_sha256"] = receipt["receipt_sha256"]
    return audit


def finalize_recovery(**kwargs: Any) -> dict[str, Any]:
    audit = audit_recovery_state(**kwargs)
    stage = Path(audit["stage_dir"])
    final = Path(audit["final_output_dir"])
    receipt_path = stage / "one_shot_receipt.json"
    payload = _receipt_payload(
        audit=audit, received_at=datetime.now(timezone.utc).isoformat()
    )
    payload["receipt_sha256"] = _digest(payload)
    with receipt_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    verified = audit_recovery_state(**kwargs, receipt_path=receipt_path)
    _publish_stage_no_clobber(stage, final)
    return {
        "schema": "hu_m43_attempt03_claimed_precal_finalize_result_v1",
        "status": "recovered_after_claim_with_full_hash_revalidation",
        "run_name": audit["run_name"],
        "output_dir": str(final),
        "data_contract": str(final / "data_contract.json"),
        "one_shot_receipt": str(final / "one_shot_receipt.json"),
        "claim_file_sha256": audit["claim_file_sha256"],
        "contract_file_sha256": audit["contract_file_sha256"],
        "merged_sha256": audit["merged_sha256"],
        "receipt_file_sha256": verified["receipt_file_sha256"],
        "teacher_row_json_decode_count": 0,
        "model_evaluation_count": 0,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit", "finalize"))
    for name in (
        "repo-root", "run-name", "project-id", "bucket", "plan",
        "preflight-receipt", "claim", "rp-dir", "stage-dir", "final-output-dir",
        "fit-receipt", "model-freeze", "original-freeze", "freeze-lineage",
        "expected-claim-sha256", "expected-contract-file-sha256",
        "expected-merged-sha256",
    ):
        parser.add_argument(f"--{name}", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    kwargs = {
        "repo_root": args.repo_root,
        "run_name": args.run_name,
        "project_id": args.project_id,
        "bucket": args.bucket,
        "plan_path": args.plan,
        "preflight_receipt_path": args.preflight_receipt,
        "claim_path": args.claim,
        "rp_dir": args.rp_dir,
        "stage_dir": args.stage_dir,
        "final_output_dir": args.final_output_dir,
        "fit_receipt_path": args.fit_receipt,
        "model_freeze_path": args.model_freeze,
        "original_freeze_path": args.original_freeze,
        "freeze_lineage_path": args.freeze_lineage,
        "expected_claim_sha256": args.expected_claim_sha256,
        "expected_contract_file_sha256": args.expected_contract_file_sha256,
        "expected_merged_sha256": args.expected_merged_sha256,
    }
    result = audit_recovery_state(**kwargs) if args.command == "audit" else finalize_recovery(**kwargs)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":  # pragma: no cover
    main()
