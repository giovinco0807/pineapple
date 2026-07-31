"""Build and verify the T1/T2 behavior bootstrap from a promoted calibration.

M3a closeout step: with gate v3 the production sharded calibration is
``promotion_eligible=true``, so the fixed-point bootstrap runtime can finally
be constructed with ``require_promoted_source=True``.  That flag is the only
path that refuses uniform, legacy, or gate-failing behavior sources.

The build freshly re-verifies all four immutable shard trees, the four
approved policy-value checkpoints, and the calibration artifact before any
route is created.  This script adds no new trust: it records what the existing
fail-closed builder accepted, as a content-bound receipt.

What this does NOT claim: the bootstrap runtime remains
``promotion_eligible=false`` / ``fixed_point_bootstrap_only=true`` by its own
contract.  A promoted *calibration source* is not a promoted policy; this
runtime only supplies no-fallback T1/T2 likelihood routes for solving the
endogenous T3 fixed point.

Usage:
    python -m ai.tutor.verify_promoted_behavior_bootstrap --out <receipt.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from ai.tutor.behavior_calibration_contract import canonical_snapshot
from ai.tutor.calibrated_behavior_sharded_bootstrap import (
    build_fixed_point_bootstrap_calibrated_hu_t1_t2_behavior_dispatch_from_shards,
)

RECEIPT_SCHEMA = "ofc_promoted_behavior_bootstrap_receipt/v1"
DEFAULT_RUN_DIR = Path("ai/data/m3_behavior_calibration_production_20260713")
DEFAULT_ARTIFACT_NAME = "calibration_v3.json"
EXPECTED_ROUTES = ((1, "bb"), (1, "btn"), (2, "bb"), (2, "btn"))


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_receipt(
    *,
    workspace_root: Path,
    run_dir: Path,
    artifact_name: str,
    scratch_dir: Path | None,
) -> dict[str, Any]:
    artifact_path = (run_dir / artifact_name).resolve(strict=True)
    started = time.time()
    dispatch = (
        build_fixed_point_bootstrap_calibrated_hu_t1_t2_behavior_dispatch_from_shards(
            run_dir / "natural",
            run_dir / "natural_evaluation",
            run_dir / "challenge",
            run_dir / "challenge_evaluation",
            artifact_path,
            workspace_root=workspace_root,
            scratch_dir=scratch_dir,
            require_promoted_source=True,
        )
    )
    elapsed = time.time() - started

    manifest = canonical_snapshot(dispatch.model_manifest)
    routes = sorted(
        (
            {
                "turn": int(entry["turn"]),
                "actor": str(entry["actor"]),
                "role": entry.get("role"),
                "temperature": str(entry.get("temperature")),
                "child_model_id": entry.get("child_model_id"),
                "child_model_sha256": entry.get("child_model_sha256"),
                "checkpoint_sha256": entry.get("checkpoint_sha256"),
            }
            for entry in manifest["routes"]
        ),
        key=lambda row: (row["turn"], row["actor"]),
    )
    if tuple((row["turn"], row["actor"]) for row in routes) != EXPECTED_ROUTES:
        raise ValueError("bootstrap dispatch does not expose the four T1/T2 routes")

    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "require_promoted_source": True,
        "source_calibration": {
            "path_relative": str(artifact_path.relative_to(workspace_root)).replace(
                "\\", "/"
            ),
            "file_sha256": _file_sha256(artifact_path),
        },
        "build_seconds": elapsed,
        "routes": routes,
        "dispatch_model_id": dispatch.model_id,
        "dispatch_model_sha256": dispatch.model_sha256,
        "dispatch_manifest": manifest,
        # The bootstrap runtime's own non-promotion contract is unchanged.
        "runtime_promotion_eligible": False,
        "fixed_point_bootstrap_only": True,
        "strategic_strength_claim": False,
        "serving_changed": False,
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workspace-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--artifact-name", default=DEFAULT_ARTIFACT_NAME)
    parser.add_argument("--scratch-dir", type=Path, default=None)
    args = parser.parse_args()

    workspace_root = args.workspace_root.resolve(strict=True)
    run_dir = args.run_dir
    if not run_dir.is_absolute():
        run_dir = workspace_root / run_dir
    receipt = build_receipt(
        workspace_root=workspace_root,
        run_dir=run_dir.resolve(strict=True),
        artifact_name=args.artifact_name,
        scratch_dir=args.scratch_dir,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "routes": receipt["routes"],
                "build_seconds": round(receipt["build_seconds"], 1),
                "receipt_sha256": receipt["receipt_sha256"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
