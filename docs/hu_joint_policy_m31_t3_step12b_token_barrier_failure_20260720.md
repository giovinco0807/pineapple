# M3.1 T3 Step 12b token-barrier failure closeout

Date: 2026-07-20

## Outcome

The explicitly authorized Step 12b direct-v2 command stopped fail-closed at
the controller-token barrier. The candidate/reference pair controller was not
entered and no Compute insert was issued.

This execution is not a completed two-VM canary. It provides no performance,
quality, training, model-strength, or promotion evidence.

## Bound execution

- deployment contract SHA-256:
  `eba36e0f5510dcf51443dfa8219fe0fdff0b4d7429dc755e402cc799f13d9293`
- run identity SHA-256:
  `cd4281852f6bcc138739ab9875f47b642a3bdcf09460d8c8068df79ef0890ac6`
- direct-stage identity SHA-256:
  `36983f8327d8f857a074d41c65088c119195eddbb5919de8b3f759a0c9589f52`
- run-scoped controller service account:
  `ofc-m31-s2b-f8cf17b20ce6@ofc-solver-485418.iam.gserviceaccount.com`
- candidate instance name: `r2d-s2b-c-a0-36983f8327d8`
- reference instance name: `r2d-s2b-r-a0-36983f8327d8`
- authorized topology: two `c4-standard-8` Spot VMs, attempt 0 only
- attempt 1, automatic retry, and a third VM: not authorized

Three fresh direct-v2 bootstrap objects were created once and read back by
generation, size, and SHA-256. The immutable direct-v1 package was only read.

## What happened

The cloud audit trail proves this sequence:

1. The run-scoped service account and the time-bounded Token Creator binding
   were created.
2. `GenerateAccessToken` succeeded at
   `2026-07-20T09:43:49.190306825Z`.
3. Token Creator removal through `SetIAMPolicy` succeeded at
   `2026-07-20T09:43:56.190854856Z`.
4. The returned empty IAM policy contained `etag` and `version: 1` but omitted
   the optional `bindings` field.
5. The shared IAM validator treated a missing `bindings` field as an empty
   list, but the token barrier subsequently required the field itself to be a
   list. That normalization mismatch raised `TokenBarrierFailure`.
6. The outer cleanup deleted the run-scoped service account at
   `2026-07-20T09:44:12.574156296Z` and verified that no VM or disk existed.

The root-cause classification is:

```text
empty_iam_policy_response_bindings_omission_normalization_mismatch
```

This was not a permission-propagation timeout and not a failed Token Creator
removal.

## Independent closeout

The read-only closeout receipt is:

`outputs/hu_joint_policy/m31_t3_step6d/step12b_pair_v2_actual/token_barrier_failure_closeout_receipt.json`

Receipt SHA-256:

`c66e75f8dfbc877e7602467231d361054bc382161e724ecd3e5acda88696b2b0`

It proves:

- candidate/reference VM GET statuses: `[404, 404]`;
- candidate/reference disk GET statuses: `[404, 404]`;
- run-scoped controller service account: absent;
- all eight exact Phase2 condition-title bindings: zero;
- controller and worker principal hits on the bound policy surfaces: zero;
- Phase2 install and VM insert count: zero;
- direct-v2 result object count: zero;
- bootstrap source object count: exactly three, retained and generation-pinned;
- old Step 11, old Step 12, and immutable package local tree digests:
  unchanged;
- `src/ofc_regular/ai_profiles.py` SHA-256:
  `d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3`;
- access token, Authorization header, and private key persistence: zero.

There is no residual cloud-resource or credential incident.

## Correction completed locally

The IAM boundary now normalizes an omitted `bindings` field to an exact empty
list while continuing to reject `null` and non-list values. Token-barrier,
outer Token Creator cleanup, and Phase2 zero verification use the same
normalization contract.

The revoke-zero readback remains bounded. It cannot mint a second token, add a
second binding, or enter the launch path. Future token-barrier failures also
persist their sealed, secret-checked failure receipt.

Validation:

- focused token-barrier/runner/closeout tests: `47 passed`;
- complete Step 12b regression: `263 passed`;
- relevant `py_compile`: passed.

Current corrected source SHA-256 values:

- Step 12b runner:
  `158628e796de5dc413b74d64d5b029d83e3698e5e14d6d127c759638b01b1c1f`
- token barrier:
  `1e2394e02fc2a74da62db981fe64feb5db4ea146aa6b8d136a9e22f45a709440`
- closeout verifier:
  `afe6f8b5886e817b28d14580a6a90169ac6ca85c1f3f88d7a50539efa8888213`

## Next gate

The consumed identity, signer, source prefix, service account, instance names,
and output root are terminal. They must not be resumed or retried.

A future canary requires fresh explicit authorization and a new signer,
deployment/run/direct-stage identity, source prefix, output root, run-scoped
service account, and exact VM/disk names. Attempt 1 and a third VM remain
forbidden.

