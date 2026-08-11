# M3.1 T3 Step 12c Phase2 schema failure closeout

Date: 2026-07-21

## Outcome

The freshly authorized Step12c exact-pair canary crossed the corrected
controller-token barrier, then stopped fail-closed at the first Phase2 install
validation. No Phase2 binding was installed and no Compute insert was issued.

This is not a completed two-VM canary and provides no performance, quality,
training, model-strength, or promotion evidence.

## Bound execution

- deployment contract SHA-256:
  `37c23ffb1dbeca76b026ab580bb51702f952dbecc7959fcf7ef16fd1b65488be`
- run identity SHA-256:
  `cad71d845ca36ba5c764f6e2ca144058580565938cb81ab82ea0f2fc8d0ea7c3`
- direct-stage identity SHA-256:
  `4e43c974eb68827e294011f456731f893ac0dc0d711df9f94b6b339f8463c110`
- run-scoped controller service account:
  `ofc-m31-s2b-8a4739e86b50@ofc-solver-485418.iam.gserviceaccount.com`
- candidate instance/disk: `r2d-s2b-c-a0-4e43c974eb68`
- reference instance/disk: `r2d-s2b-r-a0-4e43c974eb68`
- authorized topology: two `c4-standard-8` Spot VMs, attempt 0 only
- attempt 1, automatic retry, and a third VM: not authorized

The Step12c entrypoint first proved that signer, nonce, deployment/run/direct
identity, source prefix, output root, service account, and VM/disk names were
all disjoint from terminal Step12b.

## Pre-launch correction

The live read-only preflight found that Tokyo `c4-standard-8` exposes
`memoryMb=30720`, while the old mocked validator expected `32768`. The
contract now binds the provider's exact 30 GiB shape and rejects the old test
fixture. This correction passed before the Step12c launch.

## What happened

1. Three fresh bootstrap-source objects were created and generation-pinned.
2. The run-scoped controller service account was created.
3. Token Creator propagation produced six bounded 403 responses followed by a
   successful `GenerateAccessToken`.
4. Token Creator was removed; the first readback proved exact zero. No second
   binding was added and no token was reminted.
5. The success receipt included the newly added
   `token_creator_revoke_zero_readback` and
   `token_creator_revoke_zero_readback_evidence_sha256` fields.
6. The Phase2 validator still used the older exact-field allowlist, rejected
   those two fields, and stopped before its first IAM mutation.
7. Failure cleanup deleted the controller service account and independently
   obtained four VM/disk 404s. Its Phase2-zero subcheck was initially marked
   unverified because it called the same stale receipt validator.

Root-cause classification:

```text
token_success_receipt_zero_readback_fields_missing_from_phase2_validator_allowlist
```

## Independent closeout

The GET-only closeout receipt is:

`outputs/hu_joint_policy/m31_t3_step6d/step12c_pair_v1_actual/phase2_schema_failure_closeout_receipt.json`

Receipt SHA-256:

`31e675593ca168fc899a6dc34083a6f54c8258aabbb62dcd71906d56d77161ef`

It independently proves:

- candidate/reference VM GET statuses: `[404, 404]`;
- candidate/reference disk GET statuses: `[404, 404]`;
- run-scoped controller service account: absent;
- all eight exact Phase2 condition-title bindings: zero;
- Token Creator zero was observed, with no second add and no remint;
- Phase2 install receipt: absent;
- direct-v2 result object count: zero;
- bootstrap source object count: exactly three, retained and
  generation-pinned;
- terminal Step12b tree: unchanged;
- `src/ofc_regular/ai_profiles.py` SHA-256:
  `d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3`;
- credential/private-key persistence: zero.

There is no residual cloud-resource, IAM, or credential incident.

## Correction completed locally

The Phase2 lifecycle now accepts exactly the two added success fields and
fully validates the nested zero-readback evidence: exact fields, observation
digests/counts, bounded timing, zero status, no second add, no remint, and its
content digest. The actual Step12c token receipt is a regression fixture.

The complete Step12b/Step12c regression set passed after the correction:
`275 passed in 166.97s`.

Current corrected source SHA-256:

- Phase2 lifecycle:
  `8602713c8afdb8e004eace0659ec6ac8c2271c7b689ef95335d353be9a2fa1ed`
- external preflight gate with 30 GiB binding:
  `89b9caeec80bca48344d72258cd8e6dd8663aa9573144e9839f03fc4f756ec7e`
- terminal Step12c runner:
  `4c1e5cf9d1faef62176789df3c8ec3aac67f71cebbbf81d10f048e5a8b423c68`

## Next gate

The Step12c signer, nonce, contract, source prefix, output root, service
account, and VM/disk names are terminal. They must not be resumed or retried.

A future canary requires a new versioned entrypoint, fresh explicit
authorization, and a wholly new identity. Attempt 1 and a third VM remain
forbidden.

The later 20-VM topology also cannot be authorized as currently written:
Tokyo C4-family quota is 24 vCPU, while twenty `c4-standard-16` VMs require
320 vCPU. That fanout must be repartitioned or receive a separate quota change
after a two-VM lifecycle canary finally passes.
