"""Refill T0 shards that Spot took, and nothing else.

The judgement is made on PUBLISHED ROOTS, never on how many VMs are alive --
a finished shard and a preempted shard both show zero instances, and treating
them alike is how wave 4 of lap 1 resurrected completed work every ten minutes.

For each of the twenty shards: it needs refilling only if no instance carries
its start offset AND some of the request ids in its slice are still missing
from the labels prefix.  A job whose labels are complete is dropped from the
watch entirely, so the loop cannot wake it back up.

Relaunching is safe and cheap: a street shard resumes root by root, skipping
what it already published, so a shard that died at 122/656 pays only for 534.
"""
from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys
import time

BUCKET = "gs://pokerhu-ofc-solver-485418-training/hu-street"
RUN = "t0l2"
ROOTS = 13103
SHARDS = 20
REQ = pathlib.Path("D:/ofc_data/hu/onpol2_requests")
import sys
if hasattr(sys.stdout, "reconfigure"):      # cp932 kills a print
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

GCLOUD = shutil.which("gcloud") or "gcloud"


def run(args: list[str]) -> str:
    done = subprocess.run(args, capture_output=True, text=True)
    return done.stdout


def published(job: str) -> set[str]:
    out = run([GCLOUD, "storage", "ls", f"{BUCKET}/runs/{RUN}/labels/{job}/"])
    return {p.rsplit("/", 1)[-1][:-6] for p in out.split() if p.endswith(".jsonl")}


def encoded(job: str) -> set[int]:
    """Shard start offsets whose three npz files are all published."""
    out = run([GCLOUD, "storage", "ls", f"{BUCKET}/runs/{RUN}/enc/{job}/"])
    seen: dict[int, int] = {}
    for path in out.split():
        if path.endswith(".npz"):
            stem = path.rsplit("/", 1)[-1]
            seen[int(stem.split("_")[0])] = seen.get(int(stem.split("_")[0]), 0) + 1
    return {start for start, count in seen.items() if count >= 3}


def alive(job: str) -> set[int]:
    out = run([GCLOUD, "compute", "instances", "list",
               f"--filter=name~^hu-{RUN}-{job.replace('_', '-')}",
               "--format=value(name)"])
    return {int(n.rsplit("-", 1)[-1]) for n in out.split() if n}


def relaunch(job: str, seat: str, labels_done: bool) -> None:
    args = [
        sys.executable, "-m", "ai.tutor.fleet.launch_hu_street_fleet",
        "--run-id", RUN, "--job", job, "--seat", seat,
        "--roots", str(ROOTS), "--shards", str(SHARDS),
        "--placement-offset", "41", "--skip-done",
        "--requests-object", f"{job}_lap2.jsonl",
        "--value-object", "v1s_lap2.npz", "--encode",
        "--src", "hu_src_20260820b.tar.gz", "--binstamp", "20260820b",
        "--watchdog-seconds", "14400", "--batch-roots", "8",
    ]
    # Every root is already labelled; re-running the teacher would pay for it
    # twice.  Feeding the labels back in makes the shard encode only -- but
    # only if that object actually exists, since a missing one would fail the
    # shard rather than fall back.
    obj = f"{job}_lap2_labels.jsonl"
    if labels_done and run([GCLOUD, "storage", "ls",
                            f"{BUCKET}/artifacts/{obj}"]).strip():
        args += ["--labels-object", obj]
    subprocess.run(args, cwd="C:/Users/Owner/.gemini/antigravity/scratch/ofc-pineapple",
                   capture_output=True, text=True)


def main() -> None:
    jobs = {"t0_bb": "bb", "t0_btn": "btn"}
    ids = {job: [json.loads(l)["id"]
                 for l in open(REQ / f"{job}_lap2.jsonl", encoding="utf-8")
                 if l.strip()] for job in jobs}
    per = -(-ROOTS // SHARDS)
    while jobs:
        for job in list(jobs):
            have = published(job)
            enc = encoded(job)
            # The job is finished when its LAST artefact is in, not its first.
            # Watching labels alone let three shards stall at 45/60 encodes in
            # lap 1 -- labels complete, job dropped, nobody looking.
            if len(enc) >= SHARDS:
                print(f"{time.strftime('%H:%M')} {job}: エンコード完了 "
                      f"{len(enc)}/{SHARDS} / 監視終了", flush=True)
                jobs.pop(job)
                continue
            labels_done = len(have) >= ROOTS
            live = alive(job)
            gaps = []
            for index in range(SHARDS):
                start = index * per
                chunk = ids[job][start:start + per]
                got = sum(1 for r in chunk if r in have)
                if start in live or start in enc:
                    continue
                if got < len(chunk) or start not in enc:
                    gaps.append((start, got, len(chunk)))
            if gaps:
                detail = ", ".join(f"{s}({g}/{w})" for s, g, w in gaps)
                phase = "エンコードのみ" if labels_done else "ラベル+エンコード"
                print(f"{time.strftime('%H:%M')} {job}: 補充 {len(gaps)}シャード "
                      f"[{phase}] [{detail}] labels {len(have):,}/{ROOTS:,} "
                      f"enc {len(enc)}/{SHARDS}", flush=True)
                relaunch(job, jobs[job], labels_done)
        if jobs:
            time.sleep(420)
    print("両ジョブ完了 / 補充ループ終了", flush=True)


if __name__ == "__main__":
    main()
