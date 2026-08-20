# Measure one root on the fleet machine before launching a fleet

2026-08-20. Two runs, about $90, zero labels. Both failed the same way and the
check that would have caught either costs about a dollar.

## What happened

| run | machines | wall | produced | spend |
|---|---|---|---|---|
| `t0first-ladder20-1024p-x12` | 30 x c4-standard-4 | 5h47m | 0 of 240 positions | ~$22 |
| `t0first-akq500-1024p` | 54 x c4-standard-8 | ~5h20m | 0 of 500 positions | ~$68 |

Both uploaded heartbeats the whole time. Every heartbeat said
`completed_position_count: 0`. The upload path was healthy, the workers were
alive, the plans validated — and not one root ever finished.

## The single cause

The cost model came from one local timing and was never checked on the machine
that would run it.

```
local measurement       8,101 core-seconds for a 32-candidate root at 1,024 particles
                        = 2.25 hours on one core
fleet, measured         > 5.8 hours on one core, i.e. >= 2.6x
shard watchdog          6 hours
```

At the predicted rate a worker holding two roots needed 4.5 hours against a
6-hour watchdog — 25% margin, already thin. At the real rate it needed 11.6.
**The configuration could not finish a shard however long it ran.** The ladder
run was preempted thirteen minutes before its watchdog would have killed it with
the identical result.

Nothing about this is subtle in hindsight. It was never measured.

## The gate

**Before any fleet launch: one VM, one worker, one root, run to completion, on
the production package.** Record the wall time. Then:

```
if root_seconds * roots_per_worker > watchdog_seconds:
    the fleet cannot finish. Re-plan. Do not launch.
```

About thirty minutes and a dollar. It would have saved $90 twice.

The existing smoke test does not do this. It checks that the worker validates
the plan and *starts* solving — the failure that wasted $8 in the run before
these two — and both of these runs passed it. **"Started" and "can finish" are
different questions and the second one is the expensive one.**

## Watch the whole project, not your own instances

The akq500 fleet was not launched from this session. It ran for five hours
before anyone noticed, and was only found because the owner asked whether GCP
was running at all.

The mistake was assuming that not having launched something meant not having to
look. **List every instance in the project at the start of a session and
whenever spend is in question**, not just the ones you started:

```
gcloud compute instances list --format="value(name,zone,status,creationTimestamp)"
```

A fleet producing only heartbeats is invisible to anything that counts VMs or
checks liveness. The signal is `completed_position_count` staying at zero while
the sequence number climbs — a worker that is alive, uploading, and doing
nothing that will ever be kept.

## Related

* `docs/t0_particle_noise_20260820.md` — why the roots cost what they cost
* `docs/t0_sequential_elimination_20260820.md` — how to need fewer of them
* `docs/gpu_playout_investigation_20260820.md` — whether they can be made cheaper
* the reachability trap in the same family: of the five approved zones,
  `europe-west4-a` has neither Private Google Access nor Cloud NAT, so its VMs
  cannot reach GCS at all and sit with a live downloader and an empty staging
  directory. The first symptom is the guest agent logging an ACS `i/o timeout`,
  which is easy to dismiss as telemetry noise and is not.
