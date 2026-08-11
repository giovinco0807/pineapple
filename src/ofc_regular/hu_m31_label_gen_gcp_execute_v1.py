"""Execute phases for label-generation cloud runs: stage, launch, poll,
receive, cleanup.

Every write is gated three ways: the phase must be named explicitly, the
`--allow-writes` flag must be present, and GOOGLE_OAUTH_ACCESS_TOKEN must be
in the environment -- absent any of these the module renders what it would do
and exits. All REST traffic goes through the production adapter chain
(`GcpQualityRestAdapter` -> `GcpRestTransport`), instance bodies mirror the
lifecycle module's `build_instance_spec` shape field for field, and its
constants (zone, disk, NIC, provisioning model, storage scope) are imported
rather than copied so drift is impossible. Receipts are write-once.

Long receives may opt into an in-memory gcloud token refresher.  That path is
receive-only, inherits ``CLOUDSDK_CONFIG``, never prints or serializes a token,
and leaves every fixed-token mutating phase unchanged.

Spot replacement starts with an empty disk, but startup restores each existing
create-only position from GCS only after generation, byte-count, custom-SHA and
position-provenance validation, then resumes the remaining offsets. IAM is not
touched here at all. The designated worker service account must be granted read
on the staging prefix and write on the run prefix by an administrator at
approval time, because the alternative -- programmatic IAM mutation -- is where
the constraint surface is sharpest.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
import urllib.parse
import uuid
from typing import Any, Callable, Mapping

from . import hu_rl_c4_gcp_lifecycle as c4
from .hu_m31_label_gen_gcp_plan_v1 import CARRY_MANIFEST_RELATIVE
from .hu_m31_t3_step6d_fresh_quality_gcp_provider_v1 import GcpQualityRestAdapter
from .hu_m31_label_gen_resume_v1 import (
    COMPLETE_CHECKPOINT_SCHEMA,
    complete_checkpoint_object_name,
)
from .hu_m31_label_gen_worker_v1 import (
    DONE_SCHEMA,
    POSITION_SCHEMA,
    T0_VS_FL_KIND,
    T0_VS_FL_POSITION_SCHEMA,
    T1_VS_FL_KIND,
    T1_VS_FL_POSITION_SCHEMA,
    T2_VS_FL_KIND,
    T2_VS_FL_POSITION_SCHEMA,
    T3_VS_FL_KIND,
    T3_VS_FL_POSITION_SCHEMA,
    canonical_bytes,
)

RECEIPT_ROOT = pathlib.Path.home() / "ofc-labelgen/runs"
DEFAULT_IMAGE = (
    "https://www.googleapis.com/compute/v1/projects/debian-cloud/global/images/"
    "family/debian-12"
)
DEFAULT_MAX_RUN_SECONDS = 8 * 3600
RECEIVE_RECEIPT_SCHEMA = "hu_m31_label_gen_receive_receipt_v2"
SHARD_RECEIVE_EVIDENCE_SCHEMA = "hu_m31_label_gen_shard_receive_evidence_v2"
POSITION_GENERATION_MANIFEST_SCHEMA = (
    "hu_m31_label_gen_position_generation_manifest_v1"
)
GCLOUD_TOKEN_REFRESH_SECONDS = 45 * 60


def _gcloud_access_token(
    *,
    runner: Callable[..., Any] = subprocess.run,
    executable_resolver: Callable[[str], str | None] = shutil.which,
) -> str:
    """Read one token from the inherited gcloud configuration, in memory only."""

    executable = executable_resolver("gcloud")
    if not isinstance(executable, str) or not executable.strip():
        raise RuntimeError("gcloud executable is unavailable")
    completed = runner(
        [executable, "auth", "print-access-token", "--quiet"],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
        stdin=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        # Neither captured stream is included: stdout can contain a credential,
        # and a provider error must never turn it into log or receipt data.
        raise RuntimeError("gcloud access-token refresh failed")
    token = completed.stdout.strip()
    if not token or any(character.isspace() for character in token):
        raise RuntimeError("gcloud returned an invalid access token")
    return token


class RefreshableGcpQualityRestAdapter(GcpQualityRestAdapter):
    """Receive-only adapter with proactive and one-shot 401 token refresh.

    The token provider is called only into process memory.  Each REST call may
    recover from one 401; a second 401 is fatal.  The fixed-token adapter used
    by stage/execute/poll/cleanup remains untouched.
    """

    def __init__(
        self,
        *,
        access_token: str,
        zone: str,
        token_provider: Callable[[], str],
        monotonic: Callable[[], float] = time.monotonic,
        refresh_interval_seconds: int = GCLOUD_TOKEN_REFRESH_SECONDS,
        request: Callable[..., c4.HttpResponse] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if refresh_interval_seconds <= 0:
            raise ValueError("token refresh interval must be positive")
        super().__init__(
            access_token=access_token,
            zone=zone,
            request=request,
            sleep=sleep,
            token_provider=None,
        )
        self._receive_token_provider = token_provider
        self._receive_monotonic = monotonic
        self._receive_refresh_interval = float(refresh_interval_seconds)
        self._receive_last_refresh = float(monotonic())
        self._receive_401_refresh_used = False

    def _replace_receive_token(self) -> bool:
        try:
            token = self._receive_token_provider()
        except Exception:
            raise RuntimeError("GCP access-token refresh failed") from None
        if not isinstance(token, str) or not token or any(ch.isspace() for ch in token):
            raise RuntimeError("GCP access-token refresh returned invalid data")
        self._token = token
        self._receive_last_refresh = float(self._receive_monotonic())
        return True

    def _refresh_token(self) -> bool:
        if self._receive_401_refresh_used:
            return False
        self._receive_401_refresh_used = True
        return self._replace_receive_token()

    def _call(self, *args, **kwargs):
        now = float(self._receive_monotonic())
        if now - self._receive_last_refresh >= self._receive_refresh_interval:
            self._replace_receive_token()
        self._receive_401_refresh_used = False
        return super()._call(*args, **kwargs)


def _write_once(path: pathlib.Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True).encode("utf-8"))


def _placement(plan: Mapping[str, Any]) -> tuple[str, str, str]:
    """Zone, region and subnetwork for a run.

    Receipts written before runs could leave asia-northeast1 carry none of
    these fields, and the fallbacks are exactly the constants those runs were
    launched with -- so an old receipt still cleans up the instances it made.
    """

    zone = plan.get("zone", c4.ZONE)
    return zone, plan.get("region", zone.rsplit("-", 1)[0]), \
        plan.get("subnetwork", c4.SUBNETWORK_NAME)


def _adapter(
    plan: Mapping[str, Any],
    *,
    refresh_gcloud_token: bool = False,
    token_provider: Callable[[], str] | None = None,
):
    token = os.environ.get("GOOGLE_OAUTH_ACCESS_TOKEN", "")
    if not token and not refresh_gcloud_token:
        raise SystemExit(
            "GOOGLE_OAUTH_ACCESS_TOKEN is required for this phase; it stays in "
            "process memory and is never written anywhere"
        )
    if not refresh_gcloud_token:
        return GcpQualityRestAdapter(access_token=token, zone=_placement(plan)[0])
    provider = token_provider or _gcloud_access_token
    if not token:
        try:
            token = provider()
        except Exception:
            raise SystemExit("could not obtain an in-memory gcloud access token") from None
    return RefreshableGcpQualityRestAdapter(
        access_token=token,
        zone=_placement(plan)[0],
        token_provider=provider,
    )


def _load_receipt(run_dir: pathlib.Path, name: str) -> dict[str, Any]:
    path = run_dir / name
    if not path.is_file():
        raise SystemExit(f"required receipt missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _instance_name(run_name: str, shard_id: str) -> str:
    return f"ofc-lg-{run_name}-{shard_id}"


def _instance_spec(
    plan: Mapping[str, Any],
    shard: Mapping[str, Any],
    *,
    bindings_b64: str,
    attempt_id: str,
    startup_script: str,
    service_account: str,
    image: str,
    max_run_seconds: int,
) -> dict[str, Any]:
    metadata = dict(shard["metadata_values"])
    metadata["lg-attempt-id"] = attempt_id
    metadata["lg-content-bindings-b64"] = bindings_b64
    metadata["startup-script"] = startup_script
    zone, region, subnetwork = _placement(plan)
    return {
        "name": _instance_name(plan["run_name"], shard["shard_id"]),
        "machineType": f"zones/{zone}/machineTypes/{plan['machine_type']}",
        "labels": {"ofc-owner": "labelgen", "ofc-plan": plan["run_name"],
                   "ofc-role": "labelgen-worker"},
        "deletionProtection": False,
        "disks": [{
            "boot": True,
            "autoDelete": True,
            "interface": c4.BOOT_DISK_INTERFACE,
            "initializeParams": {
                "sourceImage": image,
                "diskType": f"zones/{zone}/diskTypes/{c4.BOOT_DISK_TYPE}",
                "diskSizeGb": str(c4.BOOT_DISK_SIZE_GB),
            },
        }],
        "networkInterfaces": [{
            "network": f"global/networks/{c4.NETWORK_NAME}",
            "subnetwork": f"regions/{region}/subnetworks/{subnetwork}",
            "nicType": c4.NIC_TYPE,
        }],
        "scheduling": {
            "provisioningModel": c4.PROVISIONING_MODEL,
            "preemptible": True,
            "automaticRestart": False,
            "onHostMaintenance": "TERMINATE",
            "instanceTerminationAction": "DELETE",
            "maxRunDuration": {"seconds": str(max_run_seconds)},
        },
        "serviceAccounts": [{"email": service_account,
                             "scopes": [c4.STORAGE_SCOPE]}],
        "metadata": {"items": [{"key": key, "value": value}
                               for key, value in sorted(metadata.items())]},
    }


def phase_stage(plan, run_dir, package_dir, worker_plan, adapter,
                carry_manifest: pathlib.Path | None = None) -> None:
    sources = {
        "static/runtime_archive/runtime.tar.gz": package_dir / "runtime.tar.gz",
        "static/wheelhouse_archive/wheelhouse.zip": package_dir / "wheelhouse.zip",
        "static/plan/plan.json": worker_plan,
    }
    if carry_manifest is not None:
        sources[CARRY_MANIFEST_RELATIVE] = carry_manifest
    bindings = []
    for binding in plan["content_bindings_without_generations"]:
        if binding["relative"] not in sources:
            raise SystemExit(
                f"the plan binds {binding['relative']} but this invocation did "
                "not supply it; pass --carry-manifest with the same file the "
                "planner digested"
            )
        source = sources[binding["relative"]]
        data = source.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != binding["sha256"]:
            raise SystemExit(
                f"{source} digest changed since planning; re-run the planner"
            )
        existing = adapter.get_object_metadata(
            bucket=plan["bucket"], object_name=binding["object"]
        )
        if existing is not None:
            raise SystemExit(
                f"staged object already exists: {binding['object']}; runs are "
                "write-once, pick a fresh run name"
            )
        result = adapter.put_object_new(
            bucket=plan["bucket"], object_name=binding["object"], payload=data,
            content_type="application/octet-stream",
        )
        bound = dict(binding)
        bound["generation"] = str(result["generation"])
        bindings.append(bound)
        print(f"  staged {binding['object']} generation {bound['generation']}")
    _write_once(run_dir / "stage_receipt.json", {
        "schema": "hu_m31_label_gen_stage_receipt_v1",
        "run_name": plan["run_name"],
        "bucket": plan["bucket"],
        "content_bindings": bindings,
        "staged_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })


def phase_execute(plan, run_dir, args, adapter) -> None:
    stage = _load_receipt(run_dir, "stage_receipt.json")
    bindings_b64 = base64.b64encode(
        canonical_bytes(stage["content_bindings"])
    ).decode("ascii")
    startup = (
        pathlib.Path(__file__).resolve().parents[2]
        / plan["startup_script_relative"]
    ).read_text(encoding="utf-8")
    actual = hashlib.sha256(startup.encode("utf-8")).hexdigest()
    if actual != plan["startup_script_sha256"]:
        raise SystemExit(
            "startup script changed since planning; re-run the planner so the "
            "receipt matches what launches"
        )

    launched = []
    skipped = 0
    for shard in plan["shards"]:
        # Spot instances that are preempted are DELETED, not stopped, so a
        # fleet erodes shard by shard. --only-missing refills just the gaps:
        # it never touches an instance that still exists, which is what makes
        # a top-up safe to run against a fleet that is otherwise working.
        if args.only_missing and not args.render_only:
            existing = adapter.get_instance(
                instance_name=_instance_name(plan["run_name"], shard["shard_id"])
            )
            if existing is not None:
                skipped += 1
                continue
        attempt_id = str(uuid.uuid4())
        request_id = str(uuid.uuid4())
        spec = _instance_spec(
            plan, shard,
            bindings_b64=bindings_b64,
            attempt_id=attempt_id,
            startup_script=startup,
            service_account=args.service_account,
            image=args.image,
            max_run_seconds=args.max_run_seconds,
        )
        if args.render_only:
            launched.append({"shard_id": shard["shard_id"], "spec": spec})
            continue
        operation = adapter.create_instance(
            instance_spec=spec, request_id=request_id
        )
        launched.append({
            "shard_id": shard["shard_id"],
            "instance_name": spec["name"],
            "attempt_id": attempt_id,
            "request_id": request_id,
            "operation": operation.get("name"),
        })
        print(f"  launched {spec['name']} operation {operation.get('name')}")
    if args.only_missing:
        print(f"  only-missing: {skipped} instances already present, "
              f"{len(launched)} created")
    if args.render_only:
        rendered = run_dir / "rendered_instance_specs.json"
        rendered.write_text(json.dumps(launched, indent=2, sort_keys=True),
                            encoding="utf-8")
        print(f"render-only: wrote {rendered}; no instance was created")
        return
    # Named by launch time, like the cleanup receipt. A run that outlives the
    # 8h instance cap is executed again after a cleanup, and a single fixed
    # name made that second launch die on its own write-once guard *after* it
    # had already created every instance -- leaving a live fleet and a receipt
    # that never recorded it. One file per attempt keeps both.
    _write_once(run_dir / f"execute_receipt_{int(time.time())}.json", {
        "schema": "hu_m31_label_gen_execute_receipt_v1",
        "run_name": plan["run_name"],
        "zone": _placement(plan)[0],
        "region": _placement(plan)[1],
        "service_account": args.service_account,
        "image": args.image,
        "max_run_seconds": args.max_run_seconds,
        "instances": launched,
        "launched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })


def phase_poll(plan, run_dir, adapter) -> None:
    complete = 0
    for shard in plan["shards"]:
        prefix = shard["object_prefix"]
        done = adapter.get_object_metadata(
            bucket=plan["bucket"], object_name=f"{prefix}/files/SHARD_DONE.json"
        )
        instance = adapter.get_instance(
            instance_name=_instance_name(plan["run_name"], shard["shard_id"])
        )
        state = instance.get("status") if instance else "GONE"
        marker = "DONE" if done is not None else "...."
        if done is not None:
            complete += 1
        print(f"  shard {shard['shard_id']}  {marker}  instance {state}")
    print(f"{complete}/{len(plan['shards'])} shards complete")


def _write_bytes_once_or_same(path: pathlib.Path, raw: bytes, *, label: str) -> bool:
    """Create *path*, or reuse it only when its complete bytes are identical.

    Receive is intentionally restartable, but restartable does not mean local
    state is authoritative.  A pre-existing file is evidence from an earlier
    pass and is accepted only when the generation-pinned cloud witness says the
    bytes are exactly the same.  ``True`` means a local file was reused.
    """

    if path.exists():
        if not path.is_file() or path.read_bytes() != raw:
            raise SystemExit(f"existing local {label} disagrees: {path}")
        return True
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        # A concurrent receiver is not part of the contract.  It may have
        # written the same bytes, but accepting that race would leave neither
        # process able to say which evidence it validated before creation.
        raise SystemExit(f"concurrent local writer appeared for {label}: {path}") from error
    return False


def _generation_pinned_object(
    adapter,
    *,
    bucket: str,
    object_name: str,
    label: str,
) -> tuple[bytes, dict[str, Any]] | None:
    """Read one immutable GCS generation and return its compact evidence."""

    metadata = adapter.get_object_metadata(bucket=bucket, object_name=object_name)
    if metadata is None:
        return None
    if metadata.get("name") != object_name:
        raise SystemExit(f"{label} metadata names a different object")
    generation = str(metadata.get("generation", ""))
    if not generation.isdigit() or int(generation) <= 0:
        raise SystemExit(f"{label} has invalid GCS generation {generation!r}")
    raw = adapter.get_object_bytes(
        bucket=bucket, object_name=object_name, generation=generation
    )
    if raw is None:
        raise SystemExit(
            f"{label} generation {generation} vanished after metadata was read"
        )
    stated_size = metadata.get("size")
    if stated_size is not None:
        try:
            size = int(stated_size)
        except (TypeError, ValueError) as error:
            raise SystemExit(f"{label} metadata has invalid size {stated_size!r}") from error
        if size != len(raw):
            raise SystemExit(f"{label} metadata size {size} != payload {len(raw)}")
    return raw, {
        "object_name": object_name,
        "generation": generation,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
    }


def _list_prefix_metadata(
    adapter, *, bucket: str, prefix: str
) -> list[Mapping[str, Any]]:
    """List exact GCS object metadata under one narrow prefix, with paging."""

    direct = getattr(adapter, "list_prefix", None)
    if direct is not None:
        rows = direct(bucket=bucket, prefix=prefix)
        if not isinstance(rows, list):
            raise SystemExit("GCS position prefix listing is not a list")
        return rows
    call = getattr(adapter, "_call", None)
    decode = getattr(adapter, "_json", None)
    if call is None or decode is None:
        raise SystemExit("adapter cannot list GCS position metadata")
    rows: list[Mapping[str, Any]] = []
    page_token = ""
    seen_page_tokens: set[str] = set()
    while True:
        query = {
            "prefix": prefix,
            "fields": "items(name,generation,size,metadata),nextPageToken",
            "maxResults": "1000",
        }
        if page_token:
            query["pageToken"] = page_token
        url = (
            "https://storage.googleapis.com/storage/v1/b/"
            + urllib.parse.quote(bucket, safe="")
            + "/o?"
            + urllib.parse.urlencode(query)
        )
        payload = decode(call("GET", url), "GCS position prefix listing")
        page_rows = payload.get("items", [])
        if not isinstance(page_rows, list):
            raise SystemExit("GCS position prefix listing has non-list items")
        if any(not isinstance(row, Mapping) for row in page_rows):
            raise SystemExit("GCS position prefix listing has a non-object row")
        rows.extend(page_rows)
        next_token = payload.get("nextPageToken", "")
        if not next_token:
            return rows
        if not isinstance(next_token, str):
            raise SystemExit("GCS position prefix listing has invalid page token")
        if next_token in seen_page_tokens:
            raise SystemExit("GCS position prefix listing repeats a page token")
        seen_page_tokens.add(next_token)
        page_token = next_token


def _listed_position_generation_inventory(
    adapter,
    *,
    plan: Mapping[str, Any],
    shard: Mapping[str, Any],
    checkpoint_inventory: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Bind every expected position to one listed generation/size/custom SHA."""

    object_prefix = shard["object_prefix"]
    listing_prefix = f"{object_prefix}/files/position_"
    rows = _list_prefix_metadata(
        adapter, bucket=plan["bucket"], prefix=listing_prefix
    )
    count = int(shard["count"])
    expected = {
        f"position_{offset:08d}.json"
        for offset in range(int(shard["start"]), int(shard["start"]) + count)
    }
    observed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise SystemExit(
                f"shard {shard['shard_id']} position listing has a non-object row"
            )
        name = row.get("name")
        if not isinstance(name, str) or not name.startswith(listing_prefix):
            raise SystemExit(
                f"shard {shard['shard_id']} position listing name drifted"
            )
        relative = name.rsplit("/", 1)[-1]
        expected_name = f"{object_prefix}/files/{relative}"
        if relative not in expected or name != expected_name:
            raise SystemExit(
                f"shard {shard['shard_id']} position listing has extra object {name!r}"
            )
        if relative in observed:
            raise SystemExit(
                f"shard {shard['shard_id']} position listing duplicates {relative}"
            )
        generation = row.get("generation")
        if (
            not isinstance(generation, str)
            or not generation.isdigit()
            or int(generation) <= 0
        ):
            raise SystemExit(
                f"shard {shard['shard_id']} {relative} generation is invalid"
            )
        stated_size = row.get("size")
        if not isinstance(stated_size, str) or not stated_size.isdigit():
            raise SystemExit(
                f"shard {shard['shard_id']} {relative} listed size is invalid"
            )
        byte_count = int(stated_size)
        metadata = row.get("metadata")
        if not isinstance(metadata, Mapping) or set(metadata) != {"sha256"}:
            raise SystemExit(
                f"shard {shard['shard_id']} {relative} custom SHA metadata drifted"
            )
        digest = metadata.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise SystemExit(
                f"shard {shard['shard_id']} {relative} custom SHA is invalid"
            )
        checkpoint_row = checkpoint_inventory[relative]
        if (
            checkpoint_row["bytes"] != byte_count
            or checkpoint_row["sha256"] != digest
            or checkpoint_row["object_name"] != name
        ):
            raise SystemExit(
                f"shard {shard['shard_id']} {relative} listing disagrees with "
                "complete checkpoint"
            )
        observed[relative] = {
            "relative_path": relative,
            "object_name": name,
            "generation": generation,
            "bytes": byte_count,
            "sha256": digest,
        }
    if set(observed) != expected:
        raise SystemExit(
            f"shard {shard['shard_id']} position listing inventory mismatch; "
            f"missing={sorted(expected - set(observed))[:3]}, "
            f"extra={sorted(set(observed) - expected)[:3]}"
        )
    return [observed[relative] for relative in sorted(observed)]


def _validate_done_bytes(
    raw: bytes, *, plan: Mapping[str, Any], shard: Mapping[str, Any]
) -> dict[str, Any]:
    expected = {
        "schema": DONE_SCHEMA,
        "plan_sha256": plan["worker_plan_sha256"],
        "shard_id": shard["shard_id"],
        "positions": shard["count"],
    }
    if raw != canonical_bytes(expected):
        raise SystemExit(
            f"shard {shard['shard_id']} SHARD_DONE is non-canonical or foreign"
        )
    return expected


def _validate_complete_checkpoint_bytes(
    raw: bytes, *, plan: Mapping[str, Any], shard: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(
            f"shard {shard['shard_id']} complete checkpoint is not JSON"
        ) from error
    if not isinstance(payload, Mapping) or raw != canonical_bytes(payload):
        raise SystemExit(
            f"shard {shard['shard_id']} complete checkpoint is non-canonical"
        )
    stated_digest = payload.get("checkpoint_sha256")
    unsigned = dict(payload)
    unsigned.pop("checkpoint_sha256", None)
    actual_digest = hashlib.sha256(canonical_bytes(unsigned)).hexdigest()
    if stated_digest != actual_digest:
        raise SystemExit(
            f"shard {shard['shard_id']} complete checkpoint digest mismatch"
        )
    expected_keys = {
        "schema", "checkpoint_kind", "plan_sha256", "shard_id", "attempt_id",
        "completed_position_count", "complete", "files",
        "checkpoint_published_after_files", "create_only", "checkpoint_sha256",
    }
    if set(payload) != expected_keys:
        raise SystemExit(
            f"shard {shard['shard_id']} complete checkpoint fields drifted"
        )
    count = int(shard["count"])
    if (
        payload["schema"] != COMPLETE_CHECKPOINT_SCHEMA
        or payload["checkpoint_kind"] != "complete"
        or payload["plan_sha256"] != plan["worker_plan_sha256"]
        or payload["shard_id"] != shard["shard_id"]
        or not isinstance(payload["attempt_id"], str)
        or not payload["attempt_id"]
        or payload["completed_position_count"] != count
        or payload["complete"] is not True
        or payload["checkpoint_published_after_files"] is not True
        or payload["create_only"] is not True
    ):
        raise SystemExit(
            f"shard {shard['shard_id']} complete checkpoint provenance drifted"
        )
    files = payload["files"]
    if not isinstance(files, list):
        raise SystemExit(f"shard {shard['shard_id']} checkpoint files are not a list")
    expected_relatives = {"SHARD_DONE.json"} | {
        f"position_{offset:08d}.json"
        for offset in range(int(shard["start"]), int(shard["start"]) + count)
    }
    inventory: dict[str, Mapping[str, Any]] = {}
    for row in files:
        if not isinstance(row, Mapping) or set(row) != {
            "relative_path", "object_name", "sha256", "bytes"
        }:
            raise SystemExit(
                f"shard {shard['shard_id']} checkpoint file row drifted"
            )
        relative = row["relative_path"]
        if not isinstance(relative, str) or relative in inventory:
            raise SystemExit(
                f"shard {shard['shard_id']} checkpoint duplicates a relative path"
            )
        if row["object_name"] != f"{shard['object_prefix']}/files/{relative}":
            raise SystemExit(
                f"shard {shard['shard_id']} checkpoint object path drifted"
            )
        digest = row["sha256"]
        byte_count = row["bytes"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(ch not in "0123456789abcdef" for ch in digest)
            or type(byte_count) is not int
            or byte_count <= 0
        ):
            raise SystemExit(
                f"shard {shard['shard_id']} checkpoint file metadata is invalid"
            )
        inventory[relative] = row
    if set(inventory) != expected_relatives:
        missing = sorted(expected_relatives - set(inventory))[:3]
        extra = sorted(set(inventory) - expected_relatives)[:3]
        raise SystemExit(
            f"shard {shard['shard_id']} checkpoint inventory mismatch; "
            f"missing={missing}, extra={extra}"
        )
    return dict(payload), inventory


def _validate_position_bytes(
    raw: bytes,
    *,
    inventory_row: Mapping[str, Any],
    plan: Mapping[str, Any],
    offset: int,
    object_name: str,
) -> None:
    if len(raw) != inventory_row["bytes"]:
        raise SystemExit(f"{object_name} byte count disagrees with complete checkpoint")
    actual = hashlib.sha256(raw).hexdigest()
    if actual != inventory_row["sha256"]:
        raise SystemExit(f"{object_name} digest disagrees with complete checkpoint")
    try:
        record = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SystemExit(f"{object_name} is not JSON") from error
    if raw != canonical_bytes(record):
        raise SystemExit(f"{object_name} is not canonical JSON")
    position_schemas = {
        None: POSITION_SCHEMA,
        T3_VS_FL_KIND: T3_VS_FL_POSITION_SCHEMA,
        T2_VS_FL_KIND: T2_VS_FL_POSITION_SCHEMA,
        T1_VS_FL_KIND: T1_VS_FL_POSITION_SCHEMA,
        T0_VS_FL_KIND: T0_VS_FL_POSITION_SCHEMA,
    }
    plan_kind = plan.get("plan_kind")
    if plan_kind not in position_schemas:
        raise SystemExit(f"unsupported receive plan_kind {plan_kind!r}")
    if (
        not isinstance(record, Mapping)
        or record.get("schema") != position_schemas[plan_kind]
        or record.get("plan_sha256") != plan["worker_plan_sha256"]
        or type(record.get("offset")) is not int
        or record.get("offset") != offset
    ):
        raise SystemExit(f"{object_name} position provenance drifted")


def phase_receive(plan, run_dir, adapter, out_root: pathlib.Path) -> None:
    """Receive a complete, checkpoint-bound label corpus.

    A shard is locally complete only when all expected positions, its canonical
    ``SHARD_DONE.json`` and the supervisor's after-all-files checkpoint agree.
    DONE, checkpoint and every position read are pinned to the GCS generations
    observed just before download.  A per-shard prefix listing must exactly
    match the checkpoint's name/bytes/SHA inventory before any position is
    accepted, including local files left by an interrupted receive.
    """

    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    receipt_path = run_dir / "receive_receipt.json"
    if receipt_path.exists():
        try:
            prior = json.loads(receipt_path.read_bytes())
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SystemExit("existing receive receipt is unreadable") from error
        if (
            prior.get("schema") != RECEIVE_RECEIPT_SCHEMA
            or prior.get("run_name") != plan["run_name"]
            or prior.get("worker_plan_sha256") != plan["worker_plan_sha256"]
            or prior.get("output_root") != str(out_root)
        ):
            raise SystemExit(
                "existing receive receipt predates or disagrees with strict receive"
            )

    expected_positions = sum(int(shard["count"]) for shard in plan["shards"])
    received = 0
    missing_positions = 0
    done_count = 0
    missing_done = 0
    checkpoint_count = 0
    missing_checkpoints = 0
    generation_manifest_count = 0
    shard_evidence: list[dict[str, Any]] = []

    for shard in plan["shards"]:
        prefix = shard["object_prefix"]
        shard_id = shard["shard_id"]
        count = int(shard["count"])
        target = out_root / f"shard_{shard_id}"
        target.mkdir(parents=True, exist_ok=True)

        done_name = f"{prefix}/files/SHARD_DONE.json"
        done_download = _generation_pinned_object(
            adapter,
            bucket=plan["bucket"],
            object_name=done_name,
            label=f"shard {shard_id} SHARD_DONE",
        )
        if done_download is None:
            missing_done += 1
            missing_positions += count
            print(f"  shard {shard_id}: SHARD_DONE not uploaded", flush=True)
            continue
        done_raw, done_evidence = done_download
        _validate_done_bytes(done_raw, plan=plan, shard=shard)
        _write_bytes_once_or_same(
            target / "SHARD_DONE.json", done_raw, label=f"shard {shard_id} DONE"
        )
        done_count += 1

        checkpoint_name = complete_checkpoint_object_name(prefix)
        checkpoint_download = _generation_pinned_object(
            adapter,
            bucket=plan["bucket"],
            object_name=checkpoint_name,
            label=f"shard {shard_id} complete checkpoint",
        )
        if checkpoint_download is None:
            missing_checkpoints += 1
            missing_positions += count
            print(f"  shard {shard_id}: complete checkpoint not uploaded", flush=True)
            continue
        checkpoint_raw, checkpoint_evidence = checkpoint_download
        checkpoint, inventory = _validate_complete_checkpoint_bytes(
            checkpoint_raw, plan=plan, shard=shard
        )
        done_row = inventory["SHARD_DONE.json"]
        if (
            done_row["sha256"] != hashlib.sha256(done_raw).hexdigest()
            or done_row["bytes"] != len(done_raw)
        ):
            raise SystemExit(
                f"shard {shard_id} DONE disagrees with complete checkpoint"
            )
        audit_dir = out_root / "_audit" / f"shard_{shard_id}"
        _write_bytes_once_or_same(
            audit_dir / "complete.json",
            checkpoint_raw,
            label=f"shard {shard_id} complete checkpoint",
        )
        checkpoint_count += 1

        generation_rows = _listed_position_generation_inventory(
            adapter,
            plan=plan,
            shard=shard,
            checkpoint_inventory=inventory,
        )
        shard_received = 0
        for generation_row in generation_rows:
            relative = generation_row["relative_path"]
            offset = int(relative[9:-5])
            name = generation_row["object_name"]
            local = target / relative
            payload = adapter.get_object_bytes(
                bucket=plan["bucket"],
                object_name=name,
                generation=generation_row["generation"],
            )
            if payload is None:
                raise SystemExit(
                    f"shard {shard_id} {relative} listed generation "
                    f"{generation_row['generation']} drifted or vanished"
                )
            if (
                len(payload) != generation_row["bytes"]
                or hashlib.sha256(payload).hexdigest() != generation_row["sha256"]
            ):
                raise SystemExit(
                    f"shard {shard_id} {relative} generation-pinned bytes "
                    "disagree with listing metadata"
                )
            _validate_position_bytes(
                payload,
                inventory_row=inventory[relative],
                plan=plan,
                offset=offset,
                object_name=name,
            )
            _write_bytes_once_or_same(local, payload, label=name)
            shard_received += 1
            received += 1

        manifest_core = {
            "schema": POSITION_GENERATION_MANIFEST_SCHEMA,
            "run_name": plan["run_name"],
            "bucket": plan["bucket"],
            "worker_plan_sha256": plan["worker_plan_sha256"],
            "shard_id": shard_id,
            "positions": count,
            "files": generation_rows,
            "exact_prefix_inventory": True,
            "checkpoint_sha_and_bytes_matched": True,
            "all_position_gets_generation_pinned": True,
        }
        generation_manifest = {
            **manifest_core,
            "manifest_sha256": hashlib.sha256(
                canonical_bytes(manifest_core)
            ).hexdigest(),
        }
        generation_manifest_raw = canonical_bytes(generation_manifest)
        generation_manifest_path = audit_dir / "position_generation_manifest.json"
        _write_bytes_once_or_same(
            generation_manifest_path,
            generation_manifest_raw,
            label=f"shard {shard_id} position generation manifest",
        )
        generation_manifest_count += 1

        position_inventory = [
            {
                "relative_path": row["relative_path"],
                "sha256": row["sha256"],
                "bytes": row["bytes"],
            }
            for row in generation_rows
        ]
        evidence = {
            "schema": SHARD_RECEIVE_EVIDENCE_SCHEMA,
            "run_name": plan["run_name"],
            "bucket": plan["bucket"],
            "worker_plan_sha256": plan["worker_plan_sha256"],
            "shard_id": shard_id,
            "positions": count,
            "done_object": done_evidence,
            "complete_checkpoint_object": checkpoint_evidence,
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "position_inventory_sha256": hashlib.sha256(
                canonical_bytes(position_inventory)
            ).hexdigest(),
            "position_generation_manifest": {
                "relative_path": generation_manifest_path.relative_to(
                    out_root
                ).as_posix(),
                "sha256": hashlib.sha256(generation_manifest_raw).hexdigest(),
                "positions": count,
            },
            "generation_pinned_done_complete_and_all_positions": True,
        }
        evidence_raw = canonical_bytes(evidence)
        evidence_path = audit_dir / "receive_evidence.json"
        _write_bytes_once_or_same(
            evidence_path, evidence_raw, label=f"shard {shard_id} receive evidence"
        )
        shard_evidence.append({
            "shard_id": shard_id,
            "relative_path": evidence_path.relative_to(out_root).as_posix(),
            "sha256": hashlib.sha256(evidence_raw).hexdigest(),
            "position_generation_manifest_sha256": hashlib.sha256(
                generation_manifest_raw
            ).hexdigest(),
        })
        print(
            f"  shard {shard_id}: {shard_received}/{count} positions verified, "
            "0 outstanding",
            flush=True,
        )

    print(
        f"receive: {received}/{expected_positions} positions, "
        f"DONE {done_count}/{len(plan['shards'])}, checkpoints "
        f"{checkpoint_count}/{len(plan['shards'])}, "
        f"outstanding positions={missing_positions}",
        flush=True,
    )
    complete = (
        received == expected_positions
        and missing_positions == 0
        and missing_done == 0
        and missing_checkpoints == 0
        and done_count == len(plan["shards"])
        and checkpoint_count == len(plan["shards"])
        and generation_manifest_count == len(plan["shards"])
        and len(shard_evidence) == len(plan["shards"])
    )
    if not complete:
        if receipt_path.exists():
            raise SystemExit(
                "existing final receive receipt contradicts incomplete cloud/local state"
            )
        return

    receipt = {
        "schema": RECEIVE_RECEIPT_SCHEMA,
        "run_name": plan["run_name"],
        "bucket": plan["bucket"],
        "worker_plan_sha256": plan["worker_plan_sha256"],
        "positions": received,
        "shards": len(plan["shards"]),
        "done_markers": done_count,
        "complete_checkpoints": checkpoint_count,
        "position_generation_manifests": generation_manifest_count,
        "output_root": str(out_root),
        "audit_subdirectory": "_audit",
        "shard_evidence": shard_evidence,
        "all_expected_artifacts_verified": True,
        "all_done_and_checkpoints_generation_pinned": True,
        "all_positions_generation_pinned": True,
    }
    _write_bytes_once_or_same(
        receipt_path, canonical_bytes(receipt), label="receive receipt"
    )


def phase_cleanup(plan, run_dir, adapter) -> None:
    deleted = []
    for shard in plan["shards"]:
        name = _instance_name(plan["run_name"], shard["shard_id"])
        request_id = str(uuid.uuid4())
        result = adapter.delete_instance(instance_name=name, request_id=request_id)
        state = "absent" if result is None else "deleting"
        deleted.append({"instance_name": name, "request_id": request_id,
                        "state": state})
        print(f"  {name}: {state}")
    _write_once(run_dir / f"cleanup_receipt_{int(time.time())}.json", {
        "schema": "hu_m31_label_gen_cleanup_receipt_v1",
        "run_name": plan["run_name"],
        "instances": deleted,
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True,
                        choices=("stage", "execute", "poll", "receive", "cleanup"))
    parser.add_argument("--receive-root", default="",
                        help="receive phase: local directory for position files")
    parser.add_argument("--plan-receipt", required=True)
    parser.add_argument("--package-dir", default=str(
        pathlib.Path.home() / "ofc-labelgen/package"))
    parser.add_argument("--worker-plan", required=True)
    parser.add_argument("--service-account", default="")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--max-run-seconds", type=int,
                        default=DEFAULT_MAX_RUN_SECONDS)
    parser.add_argument("--carry-manifest", default="",
                        help="stage phase: the same already-published position "
                             "list the planner digested, when the plan binds one")
    parser.add_argument("--allow-writes", action="store_true")
    parser.add_argument("--only-missing", action="store_true",
                        help="execute phase: create instances only for shards "
                             "that have none, leaving a working fleet alone; "
                             "refills what Spot preemption deleted")
    parser.add_argument("--render-only", action="store_true",
                        help="execute phase: write the instance specs to disk "
                             "for review instead of creating anything")
    parser.add_argument(
        "--refresh-gcloud-token",
        action="store_true",
        help=(
            "receive phase only: refresh the in-memory OAuth token through "
            "inherited gcloud auth before 45 minutes and once after a 401"
        ),
    )
    args = parser.parse_args()

    plan = json.loads(pathlib.Path(args.plan_receipt).read_text(encoding="utf-8"))
    run_dir = RECEIPT_ROOT / plan["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.refresh_gcloud_token and args.phase != "receive":
        raise SystemExit(
            "--refresh-gcloud-token is receive-only; fixed-token "
            "stage/execute/poll/cleanup behavior is unchanged"
        )

    writes = args.phase in ("stage", "execute", "cleanup") and not args.render_only
    if writes and not args.allow_writes:
        raise SystemExit(
            f"phase {args.phase!r} writes to the cloud; pass --allow-writes "
            "after the spend has been approved"
        )
    if args.phase == "execute" and not args.render_only and not args.service_account:
        raise SystemExit(
            "--service-account is required to launch; the f100 accounts are "
            "not used outside their production lifecycle, so the account must "
            "be designated at approval time"
        )

    if args.phase == "execute" and args.render_only:
        class _NoAdapter:
            def __getattr__(self, name):
                raise SystemExit("render-only must not touch the network")
        adapter = _NoAdapter()
    else:
        adapter = _adapter(
            plan, refresh_gcloud_token=args.refresh_gcloud_token
        )

    if args.phase == "stage":
        phase_stage(plan, run_dir, pathlib.Path(args.package_dir),
                    pathlib.Path(args.worker_plan), adapter,
                    carry_manifest=(pathlib.Path(args.carry_manifest)
                                    if args.carry_manifest else None))
    elif args.phase == "execute":
        phase_execute(plan, run_dir, args, adapter)
    elif args.phase == "poll":
        phase_poll(plan, run_dir, adapter)
    elif args.phase == "receive":
        if not args.receive_root:
            raise SystemExit("--receive-root is required for the receive phase")
        phase_receive(plan, run_dir, adapter, pathlib.Path(args.receive_root))
    elif args.phase == "cleanup":
        phase_cleanup(plan, run_dir, adapter)
    return 0


if __name__ == "__main__":
    sys.exit(main())
