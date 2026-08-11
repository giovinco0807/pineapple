#!/usr/bin/env python3
"""Stage 2 pair adapter for the frozen Step 11 trust bootstrap.

The proven Step 11 bootstrap is supplied in a second immutable metadata value.
This adapter verifies that source byte-for-byte, accepts only the frozen Stage
2 candidate/reference attempt-0 jobs, patches the two job constants in an
isolated module namespace, and delegates every cryptographic/package/identity
check to the frozen bootstrap implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import types
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence


BASE_METADATA_KEY = "ofc-step12-prebootstrap-base"
BASE_SHA256 = "9e58d367de175f3fef111f6f89a52f5f0dbf57b37a47e57d6e1c8af6324f0b93"
BASE_MAX_BYTES = 1_000_000
METADATA_ROOT = "http://metadata.google.internal/computeMetadata/v1"
STAGE_ID = "stage2_candidate_reference_pair"
ALLOWED_JOBS = frozenset({"candidate-shard-01", "reference-shard-01"})
ORDERED_JOBS = ("candidate-shard-01", "reference-shard-01")
EXPECTED_JOB_ROLES = {
    "candidate-shard-01": "candidate",
    "reference-shard-01": "reference",
}
EXPECTED_HAND_INDICES = (5, 6, 35, 39, 47, 53, 76, 83, 87, 89)
ATTEMPT_INDEX = 0


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Any,
        fp: Any,
        code: int,
        msg: str,
        headers: Mapping[str, str],
        newurl: str,
    ) -> None:
        return None


def _metadata_get(key: str) -> bytes:
    if key != BASE_METADATA_KEY:
        raise ValueError("Step12 base metadata key changed")
    url = (
        METADATA_ROOT
        + "/instance/attributes/"
        + urllib.parse.quote(key, safe="")
    )
    request = urllib.request.Request(
        url,
        headers={"Metadata-Flavor": "Google"},
        method="GET",
    )
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), _NoRedirectHandler()
    )
    try:
        with opener.open(request, timeout=10) as response:
            raw = response.read(BASE_MAX_BYTES + 1)
            if (
                response.status != 200
                or response.headers.get("Metadata-Flavor") != "Google"
                or not raw
                or len(raw) > BASE_MAX_BYTES
            ):
                raise RuntimeError("Step12 base metadata response changed")
            return raw
    except urllib.error.HTTPError as error:
        raise RuntimeError(
            f"Step12 base metadata returned HTTP {int(error.code)}"
        ) from None


def _read_contract(path: Path) -> dict[str, Any]:
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size <= 1
        or path.stat().st_size > 1_000_000
    ):
        raise ValueError("Step12 transport contract file identity changed")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Step12 transport contract is not JSON") from error
    canonical = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if not isinstance(value, dict) or raw != canonical:
        raise ValueError("Step12 transport contract is not canonical JSON")
    return value


def _job_from_contract(contract: Mapping[str, Any]) -> str:
    binding = contract.get("metadata_binding")
    preview = contract.get("adapter_preview")
    remote = contract.get("remote_layout")
    if not all(
        isinstance(value, Mapping) for value in (binding, preview, remote)
    ):
        raise ValueError("Step12 transport anchor is incomplete")
    job_id = binding.get("job_id")
    preview_jobs = preview.get("jobs")
    remote_jobs = remote.get("jobs")
    expected_role = EXPECTED_JOB_ROLES.get(job_id)
    direct_marker = "/hu-m31-r2diag-direct-v1/stages/"
    if (
        binding.get("stage_id") != STAGE_ID
        or binding.get("attempt_index") != ATTEMPT_INDEX
        or expected_role is None
        or binding.get("source_role") != expected_role
        or preview.get("stage_id") != STAGE_ID
        or preview.get("attempt_index") != ATTEMPT_INDEX
        or preview.get("selected_job_ids") != list(ORDERED_JOBS)
        or preview.get("vm_count") != 2
        or not isinstance(preview_jobs, list)
        or [row.get("job_id") for row in preview_jobs] != list(ORDERED_JOBS)
        or [row.get("source_role") for row in preview_jobs]
        != ["candidate", "reference"]
        or any(
            row.get("attempt_index") != ATTEMPT_INDEX
            or row.get("work_hand_indices") != list(EXPECTED_HAND_INDICES)
            for row in preview_jobs
        )
        or not isinstance(remote_jobs, list)
        or [row.get("job_id") for row in remote_jobs] != list(ORDERED_JOBS)
        or any(
            not isinstance(row.get("done_uri"), str)
            or direct_marker not in row["done_uri"]
            or "/hu-m31-r2diag-worker-v1/" in row["done_uri"]
            for row in remote_jobs
        )
    ):
        raise ValueError("Step12 accepts only the frozen Stage2 attempt-0 pair")
    return str(job_id)


def _load_frozen_base() -> types.ModuleType:
    raw = _metadata_get(BASE_METADATA_KEY)
    if hashlib.sha256(raw).hexdigest() != BASE_SHA256:
        raise ValueError("frozen Step11 prebootstrap source changed")
    module = types.ModuleType("_ofc_step12_frozen_prebootstrap")
    module.__file__ = "<metadata:ofc-step12-prebootstrap-base>"
    exec(compile(raw, module.__file__, "exec"), module.__dict__)
    if (
        getattr(module, "STAGE_ID", None)
        != "stage1_lifecycle_one_candidate_vm"
        or getattr(module, "JOB_ID", None) != "candidate-shard-00"
        or getattr(module, "ATTEMPT_INDEX", None) != ATTEMPT_INDEX
        or not callable(getattr(module, "main", None))
    ):
        raise ValueError("frozen Step11 prebootstrap semantic anchor changed")
    return module


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--contract", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = list(argv) if argv is not None else None
    known, _unknown = _parser().parse_known_args(raw_args)
    contract = _read_contract(known.contract)
    job_id = _job_from_contract(contract)
    module = _load_frozen_base()
    module.STAGE_ID = STAGE_ID
    module.JOB_ID = job_id
    module.ATTEMPT_INDEX = ATTEMPT_INDEX
    return int(module.main(raw_args))


if __name__ == "__main__":
    raise SystemExit(main())
