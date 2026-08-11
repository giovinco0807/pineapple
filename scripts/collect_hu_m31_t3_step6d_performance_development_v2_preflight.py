"""Collect GET-only live observations for the T3 perfdev-v2 dry run.

This command cannot package or launch anything.  It reads a short-lived OAuth
token from an environment variable, executes only the fixed HTTPS GET surface,
and writes immutable local JSON artifacts.  ``--dry-run-output`` is optional;
even a passing receipt keeps launch authorization false.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

from ofc_regular import (
    hu_m31_t3_step6d_performance_development_v2_contract as contract_v1,
)
from ofc_regular import (
    hu_m31_t3_step6d_performance_development_v2_live_readonly as live,
)
from ofc_regular import (
    hu_m31_t3_step6d_performance_development_v2_preflight as preflight,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=live.PROJECT)
    parser.add_argument("--bucket", default=live.BUCKET)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--identity-namespace", required=True)
    parser.add_argument("--result-prefix", required=True)
    parser.add_argument(
        "--token-environment-variable", default="GOOGLE_OAUTH_ACCESS_TOKEN"
    )
    parser.add_argument("--image-output", type=Path, required=True)
    parser.add_argument("--runtime-output", type=Path, required=True)
    parser.add_argument("--collection-output", type=Path, required=True)
    parser.add_argument("--dry-run-output", type=Path)
    return parser


def _outputs(args: argparse.Namespace) -> list[Path]:
    values = [args.image_output, args.runtime_output, args.collection_output]
    if args.dry_run_output is not None:
        values.append(args.dry_run_output)
    resolved = [path.resolve() for path in values]
    if len(set(resolved)) != len(resolved):
        raise ValueError("live preflight output paths must be distinct")
    if any(path.exists() or path.is_symlink() for path in resolved):
        raise FileExistsError("refusing to overwrite a live preflight artifact")
    return resolved


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _outputs(args)
    observed_at = int(time.time())
    collection = live.collect_read_only_observations(
        project=args.project,
        bucket=args.bucket,
        run_name=args.run_name,
        identity_namespace=args.identity_namespace,
        result_prefix=args.result_prefix,
        transport=live.StdlibGoogleJsonReadOnlyTransport(),
        token_source=live.EnvironmentAccessTokenSource(
            args.token_environment_variable
        ),
        observed_at_unix_seconds=observed_at,
        require_live_transport=True,
    )
    receipt = None
    if args.dry_run_output is not None:
        receipt = preflight.build_dry_run_receipt(
            image_observation=collection["image_observation"],
            runtime_observation=collection["runtime_observation"],
        )

    # All validation and optional receipt assembly complete before the first
    # local write.  Every individual artifact is then created exactly once.
    preflight.write_once(args.image_output, collection["image_observation"])
    preflight.write_once(args.runtime_output, collection["runtime_observation"])
    preflight.write_once(args.collection_output, collection)
    if receipt is not None:
        preflight.write_once(args.dry_run_output, receipt)

    print(
        json.dumps(
            {
                "schema": collection["schema"],
                "status": collection["status"],
                "collection_sha256": collection["collection_sha256"],
                "query_count": collection["query_count"],
                "all_query_methods": ["GET"],
                "image_observation_sha256": contract_v1.canonical_sha256(
                    collection["image_observation"]
                ),
                "runtime_observation_sha256": contract_v1.canonical_sha256(
                    collection["runtime_observation"]
                ),
                "dry_run_receipt_written": receipt is not None,
                "dry_run_receipt_sha256": (
                    contract_v1.canonical_sha256(receipt)
                    if receipt is not None
                    else None
                ),
                "gcloud_invoked": False,
                "cloud_mutated": False,
                "package_built": False,
                "cloud_executable": False,
                "launch_authorized": False,
                "instances_created": False,
                "current_profile_changed": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
