"""`--only-missing` must read the receipt, not just the instance.

The flag exists to refill what Spot preemption deleted, and it used to decide
purely on whether an instance record existed.  A finished shard can present
either way, and on 2026-08-16 both ways broke the T1 second-seat run:

  * its instance kept  -> terminated instances still count against the region's
    INSTANCES quota (72 in us-west1), so the fleet filled up with shards that
    were already done and the remaining ones could not launch;
  * its instance deleted -> the shard was re-issued, the worker found no
    positions to generate, collided on the create-only complete.json, exited 1
    and therefore never reached its own shutdown line, and 31 VMs idled for 90
    minutes.

A published SHARD_DONE settles it in both directions, so that is what these
pin.  The instance-side skip must survive too: it is what makes a top-up safe
to run against a fleet that is still working.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
from types import SimpleNamespace
from typing import Any

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
for _path in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ofc_regular import hu_m31_label_gen_gcp_execute_v1 as subject  # noqa: E402

STARTUP_RELATIVE = "scripts/startup_hu_m31_label_gen_v1.sh"


class RecordingAdapter:
    """Just the three calls the execute loop makes."""

    def __init__(self, *, published: set[str], instances: set[str]) -> None:
        self.published = published
        self.instances = instances
        self.created: list[str] = []
        self.metadata_lookups: list[str] = []

    def get_object_metadata(self, *, bucket: str, object_name: str):
        self.metadata_lookups.append(object_name)
        shard = object_name.split("/shards/")[1].split("/")[0]
        return {"name": object_name} if shard in self.published else None

    def get_instance(self, *, instance_name: str):
        shard = instance_name.rsplit("-", 1)[1]
        # Status is deliberately TERMINATED: a finished-and-shut-down worker is
        # the case the old code could not tell from a live one.
        return {"status": "TERMINATED"} if shard in self.instances else None

    def create_instance(self, *, instance_spec, request_id: str):
        self.created.append(instance_spec["name"])
        return {"name": f"operation-{request_id}"}


def _plan(shard_ids: list[str]) -> dict[str, Any]:
    startup = (_REPO_ROOT / STARTUP_RELATIVE).read_text(encoding="utf-8")
    return {
        "run_name": "onlymissing-test",
        "bucket": "bucket",
        "machine_type": "c4-standard-8",
        "worker_plan_sha256": "a" * 64,
        "startup_script_relative": STARTUP_RELATIVE,
        "startup_script_sha256": hashlib.sha256(
            startup.encode("utf-8")
        ).hexdigest(),
        "shards": [
            {
                "shard_id": shard_id,
                "start": index * 10,
                "count": 10,
                "object_prefix": f"labelgen/onlymissing-test/shards/{shard_id}",
                "metadata_values": {"lg-shard-id": shard_id},
            }
            for index, shard_id in enumerate(shard_ids)
        ],
    }


def _run_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "stage_receipt.json").write_text(
        json.dumps({"content_bindings": {"weights": {}}}), encoding="utf-8"
    )
    return run_dir


def _args(**overrides: Any) -> SimpleNamespace:
    base = dict(only_missing=True, render_only=False, service_account="sa@x",
                image=subject.DEFAULT_IMAGE, max_run_seconds=28_800)
    base.update(overrides)
    return SimpleNamespace(**base)


def _execute(tmp_path, *, published, instances, shard_ids, **arg_overrides):
    adapter = RecordingAdapter(published=published, instances=instances)
    subject.phase_execute(_plan(shard_ids), _run_dir(tmp_path),
                          _args(**arg_overrides), adapter)
    return adapter


def test_a_published_shard_is_skipped_even_with_no_instance(tmp_path):
    """The quota fix: its instance may be deleted to free the INSTANCES cap."""
    adapter = _execute(tmp_path, shard_ids=["000", "001", "002"],
                       published={"000", "001"}, instances=set())
    assert adapter.created == ["ofc-lg-onlymissing-test-002"]


def test_a_published_shard_is_skipped_with_a_terminated_instance(tmp_path):
    adapter = _execute(tmp_path, shard_ids=["000", "001"],
                       published={"000"}, instances={"000", "001"})
    assert adapter.created == []


def test_an_unpublished_shard_with_an_instance_is_still_left_alone(tmp_path):
    """A working fleet must survive a top-up; only the gaps get refilled."""
    adapter = _execute(tmp_path, shard_ids=["000", "001"],
                       published=set(), instances={"000"})
    assert adapter.created == ["ofc-lg-onlymissing-test-001"]


def test_an_unpublished_shard_with_no_instance_is_launched(tmp_path):
    adapter = _execute(tmp_path, shard_ids=["000"],
                       published=set(), instances=set())
    assert adapter.created == ["ofc-lg-onlymissing-test-000"]


def test_the_receipt_is_read_from_the_shard_prefix(tmp_path):
    adapter = _execute(tmp_path, shard_ids=["007"],
                       published=set(), instances=set())
    assert adapter.metadata_lookups == [
        "labelgen/onlymissing-test/shards/007/files/SHARD_DONE.json"
    ]


def test_without_only_missing_everything_launches(tmp_path):
    """A first launch must not consult receipts or instances at all."""
    adapter = _execute(tmp_path, shard_ids=["000", "001"],
                       published={"000", "001"}, instances={"000", "001"},
                       only_missing=False)
    assert adapter.created == ["ofc-lg-onlymissing-test-000",
                               "ofc-lg-onlymissing-test-001"]
    assert adapter.metadata_lookups == []
