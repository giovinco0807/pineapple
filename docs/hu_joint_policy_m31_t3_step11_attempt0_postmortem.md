# M3.1 T3 Step 11 one-VM lifecycle attempt-0 postmortem

Date: 2026-07-19 JST  
Branch: `codex/regular-ofc-pineapple`  
Scope: diagnostic candidate VM only

## Result

The original v10 attempt described below remains **No-Go / incomplete**. The
later fix3 one-VM follow-up completed the worker lifecycle and was independently
recovered and validated; see the final section.

Exactly one candidate VM was created. The package, launch, provider identity,
signed claim, claim CAS, and immediate post-claim launch-authority revocation
all succeeded. The controller then failed while polling for an object that did
not exist yet, and the fail-closed path deleted the VM before a heartbeat or
`DONE` object was published.

This run is lifecycle debugging evidence only. It is not performance-lock,
quality, training, promotion, or scientific result evidence.

## Exact attempt

- Stage: `stage1_lifecycle_one_candidate_vm`
- Job: `candidate-shard-00`
- Source role: `candidate`
- Attempt index: `0`
- Instance:
  `r2d-10c2-s1-candidate-00-a0-0ccd956a`
- Zone: `asia-northeast1-b`
- Provider instance ID: `4778063103071228158`
- VM count created: `1`
- Additional VM or attempt-1 created: `0`

The v10 output is:

`outputs/hu_joint_policy/m31_t3_step6d/step11_one_vm_v10_actual`

The final lifecycle receipt is intentionally absent because the worker result
was not received. The failure traceback is retained in:

`outputs/hu_joint_policy/m31_t3_step6d/step11_one_vm_v10_actual.runner.stderr.log`

## What passed

1. The immutable 16-object package was reused without another upload.
2. The old generation receipt was validated against its original contract.
3. Outer identity, direct-stage identity, package prefix, and package-record
   digest matched the new ephemeral-controller contract.
4. Fresh live package readback passed:
   `eb03a208f6667394349c6d36c2834e013997ee43663286fceab9065e7387d8bf`.
5. Live IAM evidence passed the bounded shared-project exception validator:
   `d85ac972f04758552211f610901bdbdf91f06c296499d049a125f2cdcae011be`.
6. `strict_iam_gate_passed` remained `false`; no strict-gate claim was made.
7. GCE insert, provider GET, signed claim, and metadata CAS succeeded.
8. After claim confirmation, controller launch and worker `actAs` permissions
   were removed. Receipt:
   `1d7194a8bb9cc22850544deded2f2dadcbad576e4bbe47d3d0df8262954ca698`.
9. No second VM could be launched after that revocation.
10. The failure path deleted the exact VM and disk and removed every Step 11
    project, bucket, worker-SA, and controller-SA binding. Cleanup receipt:
    `16478bf9fbc2cf7bdc9804d02f632fa394a234e002005b43de46d7e17a6afc66`.

## Root cause

The controller used its condition-scoped object-reader credential to poll the
future `DONE.envelope.json` object.

For an existing object, GCS can evaluate the condition against
`resource.name`. For an object that does not exist yet, that resource attribute
is unavailable. The conditioned GET therefore returned HTTP `403`, not the
expected missing-object `404`.

`wait_for_done()` treated the `403` as fatal. The exact-instance cleanup path
then deleted the VM. Cloud Audit Logs show the controller service account
issuing the delete; this was not a worker self-delete.

## Pre-insert issues found and fixed

Two earlier controller starts never created a VM:

1. v8 stopped before package upload because Windows Python could not execute
   the PowerShell-only `gcloud` command name. The runner now resolves
   `gcloud.cmd` explicitly on Windows.
2. v9 stopped before insert because GCP removed redundant parentheses from the
   single-instance CEL condition on readback. The gate now emits the canonical
   singleton expression while retaining the exact type, name, and expiry.

Both paths remained fail-closed. v9 also produced a complete cleanup receipt.

## Implemented correction

The `DONE` polling edge now uses the separately injected GET-only collector:

`wait_for_done(client=collector_client or client, ...)`

The controller credential still handles GCE lifecycle operations and
generation-pinned reads of result objects that already exist. The independent
collector is used only where a missing future object must be observed.

A regression test injects distinct controller and collector clients and
requires the `DONE` poll to use the collector. The singleton-CEL regression
test also requires the GCP-normalized exact condition.

Focused validation after the fix:

- Step 11 controller plus IAM-gate tests: `46 passed`
- VM inventory: `0`
- disk inventory: `0`
- Step 11 project bindings: `0`
- Step 11 bucket bindings: `0`
- Step 11 worker-SA bindings: `0`
- Step 11 controller-SA bindings: `0`
- retained package objects: `16`
- result objects: `0`

## Baseline and profile safety

- `current` was not changed.
- No named P0/P1/P2/T3 baseline was replaced.
- `src/ofc_regular/ai_profiles.py` SHA-256 remains
  `d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3`.

## Historical next gate after v10

Do not mark Step 11 complete and do not start Step 12.

The existing one-VM exception explicitly does not authorize a retry. A new
one-VM lifecycle attempt requires a fresh explicit authorization, fresh output
root, fresh short-lived IAM window, and the corrected collector-poll contract.
Only a received and independently validated `DONE` tree plus provider GET 404
may close Step 11.

## v11 follow-up: guest shutdown before first result

The separately authorized v11 attempt also remains **No-Go / incomplete**.
It used exactly one candidate attempt-0 VM and did not create a second VM:

- output:
  `outputs/hu_joint_policy/m31_t3_step6d/step11_one_vm_v11_actual`;
- instance:
  `r2d-10c2-s1-candidate-00-a0-0ccd956a`;
- provider instance ID: `342551519229490733`;
- post-claim revoke receipt:
  `7aadce6eb46f95586dad3f5a1bea9cc00d2fde678979277bada465f22d80d815`;
- guest termination audit time: `2026-07-18T23:15:10.105798Z`;
- provider message: `Instance terminated by guest OS shutdown.`;
- heartbeat, `DONE`, and result objects: `0`.

The v10 missing-object `403` did not recur. The corrected GET-only collector
continued polling, while the worker itself exited and the stage-0 failure path
requested shutdown. The controller launch permission and worker `actAs`
permission had already been removed after the signed claim was confirmed.

The final v11 cleanup receipt is:

`16478bf9fbc2cf7bdc9804d02f632fa394a234e002005b43de46d7e17a6afc66`

It proves that the exact VM and disk are absent and all Step 11 project,
bucket, worker-SA, and controller-SA target bindings are empty. The v11
exception has `exception_authorizes_retry=false`; its IAM window is expired
and removed.

### Reproduced direct cause

The v11 bootstrap closure contained only six Python/shell support files.
Running the retained v11 closure with repository and `site-packages` imports
removed reproduces the worker failure:

`validate_job_contract -> _lazy_modules -> cloud_package -> ImportError`

The first missing module is
`hu_m31_t3_step6d_rearm2_diagnostic_canary_local`. The VM therefore passed the
signed claim and prebootstrap checks, entered the inner transport, failed
before worker execution, and followed the bounded guest-shutdown path. This
matches the claim-success, roughly 50-second lifetime, and zero-result
timeline.

An additional latent mismatch was also found: prebootstrap accepted the GCE
scope response as exactly one line, while the inner transport compared the raw
response to the scope URL. A valid trailing newline could therefore have
caused the next immediate failure.

### Local v12 correction frozen

The existing v9/v10/v11 content-addressed package is immutable and was not
modified. Local source now prepares a new package identity with these changes:

1. Runtime contract validation no longer imports the development package/plan
   graph before the worker archive is downloaded and extracted.
2. The result-envelope adapter defers development-only imports and keeps its
   canonical serialization and hashing locally self-contained.
3. The GCE scope response must contain exactly one line equal to
   `cloud-platform`; a trailing newline is accepted, extra scopes are rejected.
4. A subprocess regression validates the declared bootstrap closure with both
   the repository and `site-packages` removed from `sys.path`.
5. Stage-0 and prebootstrap emit bounded, secret-free phase markers.
6. The controller checks `DONE` first, then the exact VM state; a
   `STOPPING`, `SUSPENDING`, `SUSPENDED`, `TERMINATED`, or `404` state now
   fails immediately instead of waiting the full 4,200 seconds.

The final local-only artifact is:

`outputs/hu_joint_policy/m31_t3_step6d/step11_v12_local_preflight_fix3`

- `LOCAL_V12_READY.json` receipt:
  `c6dec3db3be51f93120536b93a577261812eff785e8ada8f9d1ea5b91312df5e`;
- outer identity:
  `593a1f1caaa8710811fc3044c32c66d681ccfe484095b9c94b3c6ba886aa6548`;
- direct-stage identity:
  `8e27e55dab68ae9697c1c40c80353e8905ad1339423c52d51070f559fbceb44a`;
- package inventory: `16` objects;
- focused regression: `136 passed`;
- complete rearm2 diagnostic regression: `293 passed`;
- receipt, transport, launch, package-plan, local-preflight, and profile hashes:
  all revalidated;
- `package_uploaded`, `authorization_created`, `claim_created`, `iam_changed`,
  `vm_created`, and `cloud_mutation_performed`: all `false`.

Two earlier local-only v12 preparation snapshots were superseded by deeper
isolation checks; neither performed a cloud operation and neither is a launch
artifact. The final fix also validates runner output inside the pinned worker
venv, accepts the normal Debian `venv/bin/python` symlink shape, binds
`pyvenv.cfg`, and verifies `sys.prefix` inside the subprocess. The bootstrap
parent does not require NumPy.

At the time this local correction was frozen, no replacement package had been
uploaded and no additional VM had been launched. The following cloud follow-up
records the separately authorized execution.

## fix3 cloud follow-up

The exact fix3 identity received a fresh one-VM authorization and ran on
2026-07-19 UTC:

- output:
  `outputs/hu_joint_policy/m31_t3_step6d/step11_one_vm_v12_fix3_actual`;
- instance:
  `r2d-10c2-s1-candidate-00-a0-0ccd956a`;
- provider instance ID: `7124870466676841813`;
- zone and machine: `asia-northeast1-b`, `c4-standard-16` Spot;
- provider creation time: `2026-07-19T05:42:19.383Z`;
- VM count: exactly `1`;
- attempt-1 or second VM count: `0`.

The signed claim CAS succeeded. Controller launch permission and worker
`actAs` permission were removed immediately afterward. The worker completed
hands `[0, 10, 13, 43, 49, 62, 66, 81, 82, 99]`, published 10 heartbeats,
10 upload envelopes, 10 roots, 10 hand outputs, two tree-control objects,
`tree/DONE.json`, and `DONE.envelope.json`, and then self-deleted.

The original controller process nevertheless reported:

```text
RuntimeError: worker instance entered STOPPING before publishing DONE
```

The primary cause was result-namespace drift, not a missing worker result.
`adapter_preview` intentionally retained the legacy
`hu-m31-r2diag-worker-v1` provenance URIs, while the authorized worker wrote to
the content-addressed direct-v1 layout in `remote_layout`. The controller
incorrectly polled the legacy `done_uri` and used the legacy tree prefix for
materialization. Observing `STOPPING` exposed that mismatch. A final re-read
alone would not fix it.

The controller now uses `remote_layout.jobs[*].done_uri` and
`remote_layout.jobs[*].tree_prefix` for real cloud reads, while retaining the
legacy preview only for identity/content validation. It also re-reads `DONE`
once after observing terminal or absent instance state to cover a genuine
publication/self-delete race.

No additional VM was needed for recovery. The direct-v1 `DONE` envelope and
all 23 tree objects were read with generation pins, remapped through the
existing strict legacy-to-direct receiver contract, freshly materialized, and
passed `runner.validate_completed_output`:

- `DONE.envelope.json` generation: `1784440287298553`;
- `tree/DONE.json` generation: `1784440286914997`;
- completed hands: `10/10`;
- heartbeat/upload counts: `10/10`;
- recovery receipt SHA:
  `e92e5e8af28e7977870e51443108c4a17ebac18f5e6692228c7265d6844051c7`;
- materialization SHA:
  `cd0225c01203e9ccabf8deef583fd619d8f36c260fe851950b355f5f023dbf15`.

Final readback proves the exact instance and disk return HTTP `404`, and all
targeted project, bucket, worker-SA, and controller-SA bindings are empty.
Cleanup receipt SHA:
`579ebb0d4a22fca2a42b68db8186b18eb6c6460e6d3825c3b668f1a0a6a0faa3`.
`current` and every named baseline remain unchanged.

Step 11 is complete as lifecycle/correctness evidence. This run is still
diagnostic-only: it is not performance-lock, quality, training, promotion, or
M3.1 completion evidence. Step 12 remains separately unauthorized and blocked
by the currently recorded 24-vCPU Tokyo C4 quota for two simultaneous
`c4-standard-16` VMs.
