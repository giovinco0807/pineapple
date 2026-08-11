"""Fast, fail-closed parallel Spot launcher for Attempt13 search shards.

The ordinary Attempt13 launcher intentionally caps one wave at 25 sequential
creates.  Development200 and Audit50 need the same immutable inputs and VM
contract, but benefit from concurrent creates.  This module validates the
frozen Attempt13 package once, publishes or byte-verifies every immutable GCS
object once, performs one all-zone instance precheck and one GCS DONE precheck,
then creates only missing shards with at most 25 worker threads.

It never packages, authorizes, selects, fits, or activates a policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import hu_m43_attempt09_spot as _spot_engine
from . import hu_m43_attempt13_spot as spot


PARALLEL_LAUNCH_SCHEMA = "hu_m43_attempt13_parallel_spot_launch_v1"
MAX_PARALLEL_CREATES = 25
SUPPORTED_MODES = frozenset(("development", "future_audit"))
EXPECTED_IMAGE_NAME = _spot_engine.EXPECTED_IMAGE_NAME
EXPECTED_IMAGE_ID = _spot_engine.EXPECTED_IMAGE_ID
EXPECTED_IMAGE_SELF_LINK = _spot_engine.EXPECTED_IMAGE_SELF_LINK
EXPECTED_MACHINE_TYPE = _spot_engine.EXPECTED_MACHINE_TYPE

_SAFE_PROJECT = re.compile(r"^[a-z][a-z0-9-]{4,61}[a-z0-9]$")
_SAFE_BUCKET = re.compile(r"^[a-z0-9][a-z0-9._-]{1,220}[a-z0-9]$")
_SAFE_ZONE = re.compile(r"^[a-z0-9][a-z0-9-]{2,62}[a-z0-9]$")
_ACTIVE_INSTANCE_STATES = frozenset(("PROVISIONING", "STAGING", "RUNNING"))


class ParallelLaunchError(RuntimeError):
    """Raised after concurrent creates when one or more shards failed."""

    def __init__(self, result: Mapping[str, Any]) -> None:
        self.result = dict(result)
        failures = self.result.get("failures", ())
        super().__init__(f"Attempt13 parallel create failed for {len(failures)} shard(s)")


def _invoke(
    command: Sequence[str], *, timeout: int
) -> Any:
    """Use the audited Windows-aware gcloud subprocess boundary."""

    return _spot_engine._subprocess_run(
        list(command),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def _checked(command: Sequence[str], *, timeout: int) -> Any:
    completed = _invoke(command, timeout=timeout)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    return completed


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def _write_once(path: str | Path, payload: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def parse_shards(values: Sequence[str], *, total_shards: int) -> tuple[int, ...]:
    """Parse exact shard indices/ranges without imposing a wave-size cap."""

    if isinstance(total_shards, bool) or not isinstance(total_shards, int):
        raise TypeError("Attempt13 total_shards must be an integer")
    if total_shards <= 0:
        raise ValueError("Attempt13 total_shards must be positive")
    if not values:
        raise ValueError("Attempt13 requires at least one shard selector")
    if any(value.strip().lower() == "all" for value in values):
        if len(values) != 1 or values[0].strip().lower() != "all":
            raise ValueError("Attempt13 'all' shard selector must be used alone")
        return tuple(range(total_shards))
    selected: set[int] = set()
    for value in values:
        for token in value.split(","):
            token = token.strip()
            if not token:
                continue
            if "-" in token:
                pieces = token.split("-", 1)
                if len(pieces) != 2 or not all(piece.isdigit() for piece in pieces):
                    raise ValueError(f"Attempt13 invalid shard range: {token}")
                first, last = (int(piece) for piece in pieces)
                if first > last:
                    raise ValueError("Attempt13 shard range is reversed")
                expanded = range(first, last + 1)
            else:
                if not token.isdigit():
                    raise ValueError(f"Attempt13 invalid shard selector: {token}")
                expanded = (int(token),)
            for shard in expanded:
                if not 0 <= shard < total_shards:
                    raise ValueError(
                        f"Attempt13 shard {shard} is outside 0..{total_shards - 1}"
                    )
                if shard in selected:
                    raise ValueError(f"Attempt13 duplicate shard selection: {shard}")
                selected.add(shard)
    if not selected:
        raise ValueError("Attempt13 shard selection is empty")
    return tuple(sorted(selected))


def parse_zones(values: Sequence[str]) -> tuple[str, ...]:
    zones: list[str] = []
    for value in values:
        for token in value.split(","):
            zone = token.strip()
            if not zone:
                continue
            if not _SAFE_ZONE.fullmatch(zone):
                raise ValueError(f"Attempt13 unsafe GCE zone: {zone}")
            if zone in zones:
                raise ValueError(f"Attempt13 duplicate GCE zone: {zone}")
            zones.append(zone)
    if not zones:
        raise ValueError("Attempt13 requires at least one GCE zone")
    return tuple(zones)


def _validate_cloud_identity(*, project: str, bucket: str) -> None:
    if not _SAFE_PROJECT.fullmatch(project):
        raise ValueError("Attempt13 unsafe GCP project identity")
    if not _SAFE_BUCKET.fullmatch(bucket):
        raise ValueError("Attempt13 unsafe GCS bucket identity")


def _load_schedule(
    run_dir: Path, manifest: Mapping[str, Any]
) -> tuple[dict[str, Any], ...]:
    raw = (run_dir / spot.SCHEDULE_NAME).read_bytes()
    if not raw or not raw.endswith(b"\n") or b"\r" in raw:
        raise ValueError("Attempt13 schedule must be LF-terminated JSONL")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(raw.splitlines()):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("Attempt13 schedule row must be a mapping")
        if (
            value.get("schema") != spot.SHARD_SCHEMA
            or value.get("shard") != index
            or value.get("run_name") != manifest.get("run_name")
            or value.get("mode") != manifest.get("mode")
        ):
            raise ValueError(f"Attempt13 schedule identity changed at shard {index}")
        rows.append(value)
    if len(rows) != manifest.get("total_shards"):
        raise ValueError("Attempt13 schedule length changed")
    return tuple(rows)


def _verify_pinned_image(*, manifest: Mapping[str, Any]) -> None:
    if manifest.get("image") != {
        "name": EXPECTED_IMAGE_NAME,
        "id": EXPECTED_IMAGE_ID,
        "project": "debian-cloud",
        "self_link": EXPECTED_IMAGE_SELF_LINK,
    }:
        raise ValueError("Attempt13 package image contract changed")
    completed = _checked(
        [
            "gcloud",
            "compute",
            "images",
            "describe",
            EXPECTED_IMAGE_NAME,
            "--project",
            "debian-cloud",
            "--format=json",
        ],
        timeout=120,
    )
    image = json.loads(completed.stdout or "{}")
    if (
        str(image.get("id")) != EXPECTED_IMAGE_ID
        or image.get("selfLink") != EXPECTED_IMAGE_SELF_LINK
    ):
        raise ValueError("Attempt13 immutable Debian image identity changed")


def _immutable_input_specs(
    *, run_dir: Path, manifest: Mapping[str, Any], bucket: str
) -> tuple[tuple[Path, str], ...]:
    prefix = f"gs://{bucket}/runs/{manifest['run_name']}"
    specs: list[tuple[Path, str]] = [
        (run_dir / "manifest.json", f"{prefix}/manifest.json"),
        (run_dir / spot.SOURCE_NAME, f"{prefix}/source/{spot.SOURCE_NAME}"),
        (run_dir / spot.SCHEDULE_NAME, f"{prefix}/source/{spot.SCHEDULE_NAME}"),
        (
            run_dir / "launch_authorization.json",
            f"{prefix}/source/launch_authorization.json",
        ),
        (
            run_dir / "execution_authorization.json",
            f"{prefix}/source/execution_authorization.json",
        ),
    ]
    if any(not path.is_file() or path.is_symlink() for path, _uri in specs):
        raise ValueError("Attempt13 immutable launch inputs are incomplete or unsafe")
    return tuple(specs)


def publish_immutable_inputs_once(
    *, run_dir: Path, manifest: Mapping[str, Any], project: str, bucket: str
) -> tuple[dict[str, Any], ...]:
    """Conditionally publish each input once, or download and byte-verify it once."""

    rows: list[dict[str, Any]] = []
    for index, (source, uri) in enumerate(
        _immutable_input_specs(run_dir=run_dir, manifest=manifest, bucket=bucket)
    ):
        source_sha = _sha256_file(source)
        upload = _invoke(
            [
                "gcloud",
                "storage",
                "cp",
                str(source),
                uri,
                "--project",
                project,
                "--if-generation-match=0",
            ],
            timeout=900,
        )
        disposition = "published"
        if upload.returncode != 0:
            disposition = "verified_existing"
            with tempfile.TemporaryDirectory() as directory:
                downloaded = Path(directory) / f"{index:02d}-{source.name}"
                _checked(
                    [
                        "gcloud",
                        "storage",
                        "cp",
                        uri,
                        str(downloaded),
                        "--project",
                        project,
                    ],
                    timeout=900,
                )
                if _sha256_file(downloaded) != source_sha:
                    raise RuntimeError(
                        f"Attempt13 immutable GCS object differs from local input: {uri}"
                    )
        rows.append(
            {
                "path": source.name,
                "uri": uri,
                "sha256": source_sha,
                "bytes": source.stat().st_size,
                "disposition": disposition,
            }
        )
    return tuple(rows)


def _instance_prefix(run_name: str) -> str:
    prefix = re.sub(r"[^a-z0-9-]", "-", run_name)[-45:].strip("-")
    if not prefix:
        raise ValueError("Attempt13 run name has no safe instance prefix")
    return prefix


def _instance_name(run_name: str, shard: int) -> str:
    return f"{_instance_prefix(run_name)}-s{shard:03d}"[-63:]


def _authorization_sha256(run_dir: Path) -> str:
    path = run_dir / "execution_authorization.json"
    if not path.is_file() or path.is_symlink():
        raise ValueError("Attempt13 development/audit execution authorization is missing")
    return _sha256_file(path)


def _metadata_for(
    *,
    manifest: Mapping[str, Any],
    authorization: Mapping[str, Any],
    project: str,
    bucket: str,
    shard: int,
    authorization_sha256: str,
    no_self_delete: bool,
) -> dict[str, str]:
    prefix = f"gs://{bucket}/runs/{manifest['run_name']}"
    return {
        "PROJECT_ID": project,
        "BUCKET": bucket,
        "RUN_NAME": str(manifest["run_name"]),
        "SHARD": str(shard),
        "SOURCE_URI": f"{prefix}/source/{spot.SOURCE_NAME}",
        "SOURCE_SHA256": str(manifest["source_sha256"]),
        "MANIFEST_SHA256": str(authorization["manifest_sha256"]),
        "SCHEDULE_SHA256": str(manifest["schedule_sha256"]),
        "AUTHORIZATION_SHA256": authorization_sha256,
        "SELF_DELETE": "0" if no_self_delete else "1",
    }


def _metadata_items(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError("Attempt13 existing instance metadata is missing")
    items = value.get("items")
    if not isinstance(items, list):
        raise ValueError("Attempt13 existing instance metadata items are missing")
    result: dict[str, str] = {}
    for raw in items:
        if not isinstance(raw, Mapping):
            raise ValueError("Attempt13 existing instance metadata row changed")
        key, value = raw.get("key"), raw.get("value")
        if not isinstance(key, str) or not isinstance(value, str) or key in result:
            raise ValueError("Attempt13 existing instance metadata is invalid")
        result[key] = value
    return result


def _list_existing_instances(
    *, project: str, run_name: str
) -> tuple[dict[str, Any], ...]:
    prefix = _instance_prefix(run_name)
    completed = _checked(
        [
            "gcloud",
            "compute",
            "instances",
            "list",
            "--project",
            project,
            "--filter",
            f"name~'{prefix}-s'",
            "--format=json",
        ],
        timeout=180,
    )
    value = json.loads(completed.stdout or "[]")
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError("Attempt13 all-zone instance listing changed")
    return tuple(value)


def _list_done_uris(*, project: str, bucket: str, run_name: str) -> frozenset[str]:
    prefix = f"gs://{bucket}/runs/{run_name}"
    completed = _invoke(
        [
            "gcloud",
            "storage",
            "ls",
            "--recursive",
            f"{prefix}/results",
            "--project",
            project,
        ],
        timeout=300,
    )
    if completed.returncode != 0:
        # Distinguish an empty prefix from an auth/network failure before
        # treating the run as having no completed shards.
        _checked(
            [
                "gcloud",
                "storage",
                "buckets",
                "describe",
                f"gs://{bucket}",
                "--project",
                project,
            ],
            timeout=120,
        )
        return frozenset()
    return frozenset(
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip().endswith("/DONE.json")
    )


def _validate_existing_instance(
    instance: Mapping[str, Any], *, expected_metadata: Mapping[str, str]
) -> dict[str, Any]:
    name = instance.get("name")
    status = instance.get("status")
    if not isinstance(name, str) or status not in _ACTIVE_INSTANCE_STATES:
        raise ValueError(f"Attempt13 existing instance is not active: {name} status={status}")
    machine = str(instance.get("machineType", "")).rsplit("/", 1)[-1]
    if machine != EXPECTED_MACHINE_TYPE:
        raise ValueError(f"Attempt13 existing instance machine changed: {name}")
    scheduling = instance.get("scheduling")
    if not isinstance(scheduling, Mapping) or scheduling.get("provisioningModel") != "SPOT":
        raise ValueError(f"Attempt13 existing instance is not pinned Spot: {name}")
    metadata = _metadata_items(instance.get("metadata"))
    changed = {
        key: (metadata.get(key), expected)
        for key, expected in expected_metadata.items()
        if metadata.get(key) != expected
    }
    if changed:
        raise ValueError(
            f"Attempt13 existing instance metadata differs for {name}: {changed}"
        )
    zone = str(instance.get("zone", "")).rsplit("/", 1)[-1]
    if not _SAFE_ZONE.fullmatch(zone):
        raise ValueError(f"Attempt13 existing instance zone changed: {name}")
    return {"name": name, "zone": zone, "status": status}


def _precheck(
    *,
    schedule: Sequence[Mapping[str, Any]],
    selected: Sequence[int],
    manifest: Mapping[str, Any],
    authorization: Mapping[str, Any],
    project: str,
    bucket: str,
    authorization_sha256: str,
    no_self_delete: bool,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[dict[str, Any], ...]]:
    """Return completed shards, active shards, and validated active instances."""

    requested_by_name = {
        _instance_name(str(manifest["run_name"]), shard): shard for shard in selected
    }
    listed = _list_existing_instances(project=project, run_name=str(manifest["run_name"]))
    by_name: dict[str, list[Mapping[str, Any]]] = {}
    for instance in listed:
        name = instance.get("name")
        if isinstance(name, str) and name in requested_by_name:
            by_name.setdefault(name, []).append(instance)
    if any(len(rows) != 1 for rows in by_name.values()):
        duplicates = sorted(name for name, rows in by_name.items() if len(rows) != 1)
        raise ValueError(f"Attempt13 duplicate all-zone instances found: {duplicates}")

    active: list[int] = []
    active_rows: list[dict[str, Any]] = []
    for name, rows in sorted(by_name.items()):
        shard = requested_by_name[name]
        expected = _metadata_for(
            manifest=manifest,
            authorization=authorization,
            project=project,
            bucket=bucket,
            shard=shard,
            authorization_sha256=authorization_sha256,
            no_self_delete=no_self_delete,
        )
        active_rows.append(_validate_existing_instance(rows[0], expected_metadata=expected))
        active.append(shard)

    done_uris = _list_done_uris(
        project=project, bucket=bucket, run_name=str(manifest["run_name"])
    )
    prefix = f"gs://{bucket}/runs/{manifest['run_name']}"
    completed = tuple(
        shard
        for shard in selected
        if f"{prefix}/results/{schedule[shard]['output_prefix']}/DONE.json"
        in done_uris
    )
    return completed, tuple(sorted(active)), tuple(active_rows)


def _create_instance(
    *,
    run_dir: Path,
    manifest: Mapping[str, Any],
    authorization: Mapping[str, Any],
    project: str,
    bucket: str,
    zone: str,
    shard: int,
    root_index: int,
    authorization_sha256: str,
    no_self_delete: bool,
) -> dict[str, Any]:
    instance = _instance_name(str(manifest["run_name"]), shard)
    metadata = _metadata_for(
        manifest=manifest,
        authorization=authorization,
        project=project,
        bucket=bucket,
        shard=shard,
        authorization_sha256=authorization_sha256,
        no_self_delete=no_self_delete,
    )
    encoded_metadata = ",".join(f"{key}={value}" for key, value in metadata.items())
    _checked(
        [
            "gcloud",
            "compute",
            "instances",
            "create",
            instance,
            "--project",
            project,
            "--zone",
            zone,
            "--machine-type",
            EXPECTED_MACHINE_TYPE,
            "--provisioning-model=SPOT",
            "--instance-termination-action=DELETE",
            "--image-project=debian-cloud",
            f"--image={EXPECTED_IMAGE_NAME}",
            "--boot-disk-size=50GB",
            "--boot-disk-type=hyperdisk-balanced",
            "--scopes=https://www.googleapis.com/auth/cloud-platform",
            f"--metadata={encoded_metadata}",
            f"--metadata-from-file=startup-script={run_dir / spot.STARTUP_NAME}",
            "--quiet",
        ],
        timeout=420,
    )
    return {
        "shard": shard,
        "root_index": root_index,
        "instance": instance,
        "zone": zone,
        "source_sha256": str(manifest["source_sha256"]),
        "manifest_sha256": str(authorization["manifest_sha256"]),
        "schedule_sha256": str(manifest["schedule_sha256"]),
        "authorization_sha256": authorization_sha256,
    }


def _create_missing_parallel(
    *,
    shards: Sequence[int],
    max_workers: int,
    zones: Sequence[str],
    schedule: Sequence[Mapping[str, Any]],
    create_kwargs: Mapping[str, Any],
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    if not shards:
        return (), ()
    worker_count = min(max_workers, len(shards))
    created: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(
        max_workers=worker_count, thread_name_prefix="attempt13-spot-create"
    ) as pool:
        futures = {
            pool.submit(
                _create_instance,
                shard=shard,
                root_index=int(schedule[shard]["root_index"]),
                zone=zones[shard % len(zones)],
                **create_kwargs,
            ): shard
            for shard in shards
        }
        for future in as_completed(futures):
            shard = futures[future]
            try:
                created.append(future.result())
            except Exception as error:
                failures.append(
                    {
                        "shard": shard,
                        "zone": zones[shard % len(zones)],
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )
    return (
        tuple(sorted(created, key=lambda row: int(row["shard"]))),
        tuple(sorted(failures, key=lambda row: int(row["shard"]))),
    )


def launch_attempt13_parallel(
    *,
    run_dir: str | Path,
    project: str,
    bucket: str,
    zones: Sequence[str],
    shards: Sequence[str],
    max_workers: int = MAX_PARALLEL_CREATES,
    resume_missing: bool = False,
    no_self_delete: bool = False,
    output: str | Path | None = None,
) -> dict[str, Any]:
    """Publish once, precheck once, and launch only frozen missing shards."""

    _validate_cloud_identity(project=project, bucket=bucket)
    normalized_zones = parse_zones(zones)
    if (
        isinstance(max_workers, bool)
        or not isinstance(max_workers, int)
        or not 1 <= max_workers <= MAX_PARALLEL_CREATES
    ):
        raise ValueError(f"Attempt13 max_workers must be in 1..{MAX_PARALLEL_CREATES}")

    target = Path(run_dir).resolve(strict=True)
    manifest, authorization = spot.validate_launch(target)
    if manifest.get("mode") not in SUPPORTED_MODES:
        raise ValueError("Attempt13 parallel launcher allows Development200/Audit50 only")
    if (
        manifest.get("current_profile_mutated") is not False
        or manifest.get("runtime_policy_activated") is not False
        or authorization.get("current_profile_mutated") is not False
        or authorization.get("runtime_policy_activated") is not False
    ):
        raise ValueError("Attempt13 launch package mutates current/runtime policy")
    schedule = _load_schedule(target, manifest)
    selected = parse_shards(shards, total_shards=len(schedule))
    authorization_sha = _authorization_sha256(target)

    _verify_pinned_image(manifest=manifest)
    published = publish_immutable_inputs_once(
        run_dir=target, manifest=manifest, project=project, bucket=bucket
    )
    completed, active, active_rows = _precheck(
        schedule=schedule,
        selected=selected,
        manifest=manifest,
        authorization=authorization,
        project=project,
        bucket=bucket,
        authorization_sha256=authorization_sha,
        no_self_delete=no_self_delete,
    )
    occupied = tuple(sorted(set(completed) | set(active)))
    if occupied and not resume_missing:
        raise FileExistsError(
            "Attempt13 requested shards already have DONE/active instances: "
            + ",".join(str(shard) for shard in occupied)
        )
    missing = tuple(shard for shard in selected if shard not in set(occupied))
    created, failures = _create_missing_parallel(
        shards=missing,
        max_workers=max_workers,
        zones=normalized_zones,
        schedule=schedule,
        create_kwargs={
            "run_dir": target,
            "manifest": manifest,
            "authorization": authorization,
            "project": project,
            "bucket": bucket,
            "authorization_sha256": authorization_sha,
            "no_self_delete": no_self_delete,
        },
    )
    base_result = {
        "schema": PARALLEL_LAUNCH_SCHEMA,
        "run_name": manifest["run_name"],
        "mode": manifest["mode"],
        "project": project,
        "bucket": bucket,
        "zones": list(normalized_zones),
        "max_parallel_creates": max_workers,
        "resume_missing": resume_missing,
        "requested_shards": list(selected),
        "skipped_done_shards": list(completed),
        "skipped_active_shards": [
            shard for shard in active if shard not in set(completed)
        ],
        "active_instances": list(active_rows),
        "missing_precreate_shards": list(missing),
        "created": list(created),
        "immutable_inputs": list(published),
        "source_sha256": manifest["source_sha256"],
        "manifest_sha256": authorization["manifest_sha256"],
        "schedule_sha256": manifest["schedule_sha256"],
        "authorization_sha256": authorization_sha,
        "current_profile_mutated": False,
        "runtime_policy_activated": False,
    }
    if failures:
        result = {**base_result, "status": "partial_failure", "failures": list(failures)}
        raise ParallelLaunchError(result)
    result = {
        **base_result,
        "status": "created" if created else "nothing_missing",
        "failures": [],
    }
    if output is not None:
        _write_once(output, _canonical_json_bytes(result))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--project", default="ofc-solver-485418")
    parser.add_argument("--bucket", default="pokerhu-ofc-solver-485418-training")
    parser.add_argument(
        "--zones",
        action="append",
        default=None,
        help="Comma-separated zones; shard modulo zone count is deterministic.",
    )
    parser.add_argument(
        "--shards",
        action="append",
        required=True,
        help="Shard index, inclusive range, comma list, or 'all'.",
    )
    parser.add_argument("--max-workers", type=int, default=MAX_PARALLEL_CREATES)
    parser.add_argument("--resume-missing", action="store_true")
    parser.add_argument("--no-self-delete", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = launch_attempt13_parallel(
            run_dir=args.run_dir,
            project=args.project,
            bucket=args.bucket,
            zones=args.zones or ["asia-northeast1-b"],
            shards=args.shards,
            max_workers=args.max_workers,
            resume_missing=args.resume_missing,
            no_self_delete=args.no_self_delete,
            output=args.output,
        )
    except ParallelLaunchError as error:
        print(json.dumps(error.result, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_IMAGE_ID",
    "EXPECTED_IMAGE_NAME",
    "EXPECTED_IMAGE_SELF_LINK",
    "EXPECTED_MACHINE_TYPE",
    "MAX_PARALLEL_CREATES",
    "PARALLEL_LAUNCH_SCHEMA",
    "ParallelLaunchError",
    "launch_attempt13_parallel",
    "parse_shards",
    "parse_zones",
    "publish_immutable_inputs_once",
]
