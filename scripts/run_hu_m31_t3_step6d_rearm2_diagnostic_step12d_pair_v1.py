#!/usr/bin/env python3
"""Run one fresh-identity exact-pair canary over the Step12b v2 protocol.

Step12d is a new authorization/identity boundary, not a retry of the
terminal Step12b or Step12c executions.  Both old runners and audit trees
remain unchanged.  Cloud execution additionally requires fresh explicit
user authorization outside this program.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from ofc_regular import (
    hu_m31_t3_step6d_rearm2_diagnostic_step12d_fresh_identity_v1
    as fresh_identity,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
STEP12B_RUNNER_PATH = (
    REPO_ROOT
    / "scripts"
    / "run_hu_m31_t3_step6d_rearm2_diagnostic_step12b_pair_v2.py"
)
EXECUTION_CONFIRMATION = "EXECUTE_STEP12D_DIRECT_V2_EXACT_PAIR_ATTEMPT0"
RUNNER_MODULE_NAME = "_hu_m31_t3_step12b_protocol_v2_runner"
FINAL_SCHEMA = "hu_m31_t3_step6d_step12d_pair_v1_final"
FAILURE_SCHEMA = "hu_m31_t3_step6d_step12d_pair_v1_failure"


def _load_step12b_protocol_runner() -> Any:
    existing = sys.modules.get(RUNNER_MODULE_NAME)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(
        RUNNER_MODULE_NAME, STEP12B_RUNNER_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError("Step12b v2 protocol runner cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[RUNNER_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(RUNNER_MODULE_NAME, None)
        raise
    return module


step12b = _load_step12b_protocol_runner()


def _fresh_receipt(prepared: Any, output_root: str | Path) -> dict[str, Any]:
    return fresh_identity.validate_fresh_identity(
        deployment_contract=prepared.deployment_contract,
        controller_public_key_record=prepared.controller_public_key_record,
        source_plan=prepared.source_plan,
        output_root=output_root,
    )


def dry_run_receipt(
    prepared: Any, *, output_root: str | Path
) -> dict[str, Any]:
    fresh = _fresh_receipt(prepared, output_root)
    protocol = step12b.dry_run_receipt(prepared)
    return fresh_identity.seal(
        {
            "schema": "hu_m31_t3_step6d_step12d_pair_v1_dry_run",
            "status": "step12d_offline_fresh_identity_no_cloud_adapter",
            "fresh_identity_receipt_sha256": fresh["receipt_sha256"],
            "protocol_dry_run_receipt_sha256": protocol["receipt_sha256"],
            "deployment_contract_sha256": fresh[
                "deployment_contract_sha256"
            ],
            "run_identity_sha256": fresh["run_identity_sha256"],
            "direct_stage_identity_sha256": fresh[
                "direct_stage_identity_sha256"
            ],
            "source_prefix": fresh["source_prefix"],
            "output_root": fresh["output_root"],
            "instance_names": fresh["instance_names"],
            "attempt_index": 0,
            "attempt1_authorized": False,
            "third_vm_authorized": False,
            "automatic_retry_authorized": False,
            "cloud_adapter_constructed": False,
            "cloud_mutation_performed": False,
            "current_profile_changed": False,
        }
    )


def _terminal_trees_unchanged(before: dict[str, Any]) -> dict[str, Any]:
    after = fresh_identity.terminal_tree_snapshot()
    if after != before:
        raise RuntimeError("terminal Step12b/Step12c artifact tree changed")
    return after


def _write_wrapper_failure(
    *,
    output_root: Path,
    fresh: dict[str, Any],
    failure_stage: str,
    exception_type: str,
    terminal_trees: dict[str, Any],
) -> None:
    if not output_root.is_dir() or output_root.is_symlink():
        return
    path = output_root / "STEP12D_FAILURE.json"
    if path.exists() or path.is_symlink():
        return
    body = fresh_identity.seal(
        {
            "schema": FAILURE_SCHEMA,
            "status": "step12d_stopped_without_retry",
            "failure_stage": failure_stage,
            "exception_type": exception_type,
            "fresh_identity_receipt_sha256": fresh["receipt_sha256"],
            "terminal_trees": terminal_trees,
            "attempt_index": 0,
            "attempt1_authorized": False,
            "third_vm_authorized": False,
            "automatic_retry_performed": False,
            "exception_message_stored": False,
            "private_key_stored": False,
            "access_token_stored": False,
            "current_profile_changed": False,
        }
    )
    step12b._exclusive_write_json(path, body)


def execute_once(
    *, prepared: Any, output_root: str | Path
) -> dict[str, Any]:
    root = fresh_identity.exact_output_root(output_root)
    fresh = _fresh_receipt(prepared, root)
    terminal_before = fresh_identity.terminal_tree_snapshot()
    step12b._IMMUTABLE_ROOTS.add(
        fresh_identity.TERMINAL_STEP12B_OUTPUT_ROOT.resolve()
    )
    step12b._IMMUTABLE_ROOTS.add(
        fresh_identity.TERMINAL_STEP12C_OUTPUT_ROOT.resolve()
    )
    backend_constructed = False
    try:
        backend = step12b.LiveExecutionBackend(prepared)
        backend_constructed = True
        protocol_final = step12b.execute_prepared(
            prepared=prepared,
            backend=backend,
            output_root=root,
        )
        terminal_after = _terminal_trees_unchanged(terminal_before)
        final = fresh_identity.seal(
            {
                "schema": FINAL_SCHEMA,
                "status": "step12d_pair_received_cleaned_and_disjoint",
                "fresh_identity_receipt_sha256": fresh["receipt_sha256"],
                "protocol_final_receipt_sha256": protocol_final[
                    "receipt_sha256"
                ],
                "deployment_contract_sha256": fresh[
                    "deployment_contract_sha256"
                ],
                "instance_names": fresh["instance_names"],
                "result_object_count": protocol_final[
                    "result_object_count"
                ],
                "terminal_trees_before": terminal_before,
                "terminal_trees_after": terminal_after,
                "terminal_trees_unchanged": True,
                "attempt_index": 0,
                "attempt1_authorized": False,
                "third_vm_authorized": False,
                "automatic_retry_performed": False,
                "private_key_stored": False,
                "access_token_stored": False,
                "current_profile_changed": False,
            }
        )
        step12b._exclusive_write_json(root / "STEP12D_FINAL.json", final)
        return final
    except BaseException as error:
        terminal_after = _terminal_trees_unchanged(terminal_before)
        try:
            _write_wrapper_failure(
                output_root=root,
                fresh=fresh,
                failure_stage=(
                    "step12b_protocol_execution"
                    if backend_constructed
                    else "live_backend_construction"
                ),
                exception_type=type(error).__name__,
                terminal_trees=terminal_after,
            )
        except BaseException:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step12d fresh-identity exact pair canary"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="enable the exact one-shot cloud mutation path",
    )
    parser.add_argument(
        "--confirm",
        default="",
        help="exact Step12d execution confirmation",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=fresh_identity.EXPECTED_OUTPUT_ROOT,
        help="must be the exact fresh Step12d output root",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.execute and args.confirm != EXECUTION_CONFIRMATION:
        raise PermissionError(
            "exact Step12d execution confirmation is missing"
        )
    root = fresh_identity.exact_output_root(args.output_root)
    prepared = step12b.prepare_run()
    if not args.execute:
        receipt = dry_run_receipt(prepared, output_root=root)
    else:
        receipt = execute_once(prepared=prepared, output_root=root)
    sys.stdout.write(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
