"""Emit the exact 30-job argv schedule for local or Spot workers.

The output contains argument arrays rather than shell-quoted strings.  A VM
dispatcher can therefore execute each entry without reinterpreting paths or
digests.  Four jobs are assigned per shard, yielding eight resumable shards;
shard zero is the intended canary.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .hu_m43_attempt03_training import load_attempt03_fold_cloud_contract


M43_ATTEMPT03_MODEL_JOB_SCHEDULE_SCHEMA = (
    "hu_m43_attempt03_v5_model_job_schedule_v1"
)


def build_attempt03_model_job_schedule(
    *,
    fold_cloud_contract_path: str | Path,
    run_name: str,
    source_sha256: str,
    run_manifest_sha256: str,
    inherited_train_worker_path: str,
    fresh_train_fit_worker_path: str,
    fold_cloud_contract_worker_path: str,
    output_root_worker_path: str,
    python_executable: str = "python",
    jobs_per_shard: int = 4,
) -> dict[str, Any]:
    contract = load_attempt03_fold_cloud_contract(fold_cloud_contract_path)
    if jobs_per_shard != 4:
        raise ValueError("Attempt03 Spot schedule freezes four jobs per shard")
    if not run_name or any(
        character
        not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
        for character in run_name
    ):
        raise ValueError("Attempt03 schedule run_name is unsafe")
    source_sha = _sha(source_sha256, "source_sha256")
    manifest_sha = _sha(run_manifest_sha256, "run_manifest_sha256")
    jobs: list[dict[str, Any]] = []
    specs = contract["fold_plan"]["jobs"]
    for index, spec in enumerate(specs):
        shard = index // jobs_per_shard
        output = f"{output_root_worker_path.rstrip('/')}/job-{index:02d}"
        argv = [
            python_executable,
            "-B",
            "-m",
            "ofc_regular.train_hu_m43_attempt03_fold_job",
            "--job-index",
            str(index),
            "--inherited-train",
            inherited_train_worker_path,
            "--fresh-train-fit",
            fresh_train_fit_worker_path,
            "--fold-cloud-contract",
            fold_cloud_contract_worker_path,
            "--output-dir",
            output,
            "--run-name",
            run_name,
            "--source-sha256",
            source_sha,
            "--run-manifest-sha256",
            manifest_sha,
            "--input-bundle-sha256",
            contract["input_bundle_sha256"],
            "--job-spec-sha256",
            spec["job_spec_sha256"],
        ]
        jobs.append(
            {
                "job_index": index,
                "shard_index": shard,
                "canary": shard == 0,
                "job_kind": spec["kind"],
                "outer_fold": spec["outer_fold"],
                "inner_fold": spec["inner_fold"],
                "job_spec_sha256": spec["job_spec_sha256"],
                "output_dir": output,
                "argv": argv,
            }
        )
    if len(jobs) != 30 or [row["job_index"] for row in jobs] != list(range(30)):
        raise AssertionError("Attempt03 job schedule lost exact 30-job coverage")
    shards = [
        {
            "shard_index": shard,
            "canary": shard == 0,
            "job_indices": [row["job_index"] for row in jobs if row["shard_index"] == shard],
        }
        for shard in range(8)
    ]
    if [len(row["job_indices"]) for row in shards] != [4, 4, 4, 4, 4, 4, 4, 2]:
        raise AssertionError("Attempt03 30-job shard partition changed")
    return {
        "schema": M43_ATTEMPT03_MODEL_JOB_SCHEDULE_SCHEMA,
        "status": "frozen_commands_not_started",
        "run_name": run_name,
        "fold_cloud_contract_file_sha256": _file_sha256(
            Path(fold_cloud_contract_path)
        ),
        "fold_cloud_contract_sha256": contract["contract_sha256"],
        "input_bundle_sha256": contract["input_bundle_sha256"],
        "training_config_sha256": contract["training_config_sha256"],
        "model_freeze_file_sha256": contract["model_freeze"]["file_sha256"],
        "training_freeze_file_sha256": contract["training_freeze"][
            "file_sha256"
        ],
        "source_sha256": source_sha,
        "run_manifest_sha256": manifest_sha,
        "process_environment": dict(contract["process_environment"]),
        "jobs_per_shard": jobs_per_shard,
        "shard_count": 8,
        "job_count": 30,
        "canary_shard": 0,
        "fanout_before_canary_done_allowed": False,
        "shards": shards,
        "jobs": jobs,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }


def write_attempt03_model_job_schedule(
    path: str | Path, **kwargs: Any
) -> dict[str, Any]:
    payload = build_attempt03_model_job_schedule(**kwargs)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return payload


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _file_sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-cloud-contract", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--run-manifest-sha256", required=True)
    parser.add_argument("--inherited-train-worker-path", required=True)
    parser.add_argument("--fresh-train-fit-worker-path", required=True)
    parser.add_argument("--fold-cloud-contract-worker-path", required=True)
    parser.add_argument("--output-root-worker-path", required=True)
    parser.add_argument("--python-executable", default="python")
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = write_attempt03_model_job_schedule(
        args.output,
        fold_cloud_contract_path=args.fold_cloud_contract,
        run_name=args.run_name,
        source_sha256=args.source_sha256,
        run_manifest_sha256=args.run_manifest_sha256,
        inherited_train_worker_path=args.inherited_train_worker_path,
        fresh_train_fit_worker_path=args.fresh_train_fit_worker_path,
        fold_cloud_contract_worker_path=args.fold_cloud_contract_worker_path,
        output_root_worker_path=args.output_root_worker_path,
        python_executable=args.python_executable,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "M43_ATTEMPT03_MODEL_JOB_SCHEDULE_SCHEMA",
    "build_attempt03_model_job_schedule",
    "main",
    "parse_args",
    "write_attempt03_model_job_schedule",
]
