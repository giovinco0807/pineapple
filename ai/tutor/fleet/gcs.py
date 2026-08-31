"""The object-store idioms a label worker needs, and nothing else.

These four functions started life inside `run_t1_shard`, whose module-level
imports reach that fleet's root sampler and through it `ai.engine` -- so a
worker that wanted only `upload()` had to ship the vs-FL label generator's
entire world, and discovered it did not at boot, in the cloud, after paying
for the instance.  They live here so a runner can take the idioms without the
dependencies.

The idioms themselves are the expensive part and are unchanged: create-only
uploads, so a re-run of a shard whose labels are already published is a no-op
rather than a conflict; a listing that cannot be read is fatal, because
labelling a shard whose state is unknown is how a run silently doubles its
bill; and a short retry on upload, because a blip that loses a root loses
work that has already been paid for.
"""
from __future__ import annotations

import shutil
import subprocess
import time

# On Windows the entry point is gcloud.cmd, which subprocess cannot resolve
# from the bare name without a shell.
GCLOUD = shutil.which("gcloud") or "gcloud"
NO_OBJECTS = "matched no objects"


def say(message: str) -> None:
    print(f"[shard] {message}", flush=True)


def listing(url: str) -> list[str]:
    """Object names under `url`; empty is a legitimate answer, unreadable is not."""
    done = subprocess.run(
        [GCLOUD, "storage", "ls", url], capture_output=True, text=True
    )
    if done.returncode != 0:
        if NO_OBJECTS in (done.stderr or ""):
            return []
        raise SystemExit(f"FATAL: cannot read {url}: {done.stderr.strip()[:400]}")
    return [line.rsplit("/", 1)[-1].strip() for line in done.stdout.splitlines()]


def upload_once(text: str, url: str, *, create_only: bool = True) -> bool:
    """Write `text` to `url`.  False means "already there", which is benign."""
    command = [GCLOUD, "storage", "cp"]
    if create_only:
        command += ["--if-generation-match=0"]
    command += ["-", url]
    done = subprocess.run(command, input=text, capture_output=True, text=True)
    if done.returncode == 0:
        return True
    blurb = (done.stderr or "") + (done.stdout or "")
    if "recondition" in blurb or "412" in blurb:
        return False
    raise RuntimeError(f"upload {url} failed: {blurb.strip()[:300]}")


def upload(text: str, url: str, *, attempts: int = 3) -> bool:
    """`upload_once` with a short retry."""
    for attempt in range(attempts):
        try:
            return upload_once(text, url)
        except RuntimeError as error:
            if attempt == attempts - 1:
                raise
            say(f"retrying upload of {url}: {error}")
            time.sleep(5 * (attempt + 1))
    raise AssertionError("unreachable")
