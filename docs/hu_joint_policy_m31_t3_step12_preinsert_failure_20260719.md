# M3.1 T3 Step 12 pre-insert failure closeout

Date: 2026-07-19

## Outcome

The explicitly authorized Step 12 command failed closed before the
candidate/reference pair controller was entered. No VM insert was issued.

The immediate exception was:

```text
RestIamAdminError: controller_access_token_generation_failed
```

This execution is not a completed two-VM canary and provides no performance,
quality, training, or promotion evidence.

## Bound execution

- runner SHA-256:
  `2699d53f0fbb27549f83c7fa505336837190d5249129caca365351f3d2f7382b`
- gate source SHA-256:
  `816da64b3f52e1a45c757c2978635d0ccb6be762dc902d72e86b312c55090915`
- runner-plan SHA-256:
  `4b3ae98ae77388e027a852a37686076cfcb3786cba4552c74a19c5da6a8a599a`
- gate-plan SHA-256:
  `fb175da38a8f474444f640927ec81f16030b340d65c045433c6e5461a536f6f1`
- candidate instance name:
  `r2d-10c2-s2-candidate-01-a0-29e3c6f8`
- reference instance name:
  `r2d-10c2-s2-reference-01-a0-29e3c6f8`
- actual topology: two `c4-standard-8` Spot VMs, attempt 0 only
- frozen inner topology: `c4-standard-16`, Rayon 16

The oversubscribed inner topology remains diagnostic-only and could not have
served as performance evidence even if the canary had completed.

## What completed before the stop

- all GET-only package, prefix, capacity, reservation, NAT, service, role, and
  exact VM/disk absence checks passed;
- the bounded Step 12 exception was sealed before the first mutation;
- ten temporary IAM bindings were created and read back exactly;
- controller token generation was attempted through the bounded retry helper;
- the helper exhausted without obtaining a controller credential;
- the pair controller and Compute insert path were never entered;
- final cleanup removed all ten temporary IAM bindings.

Authorization receipt:
`241a3e3fce314694da746bfa7bc5f05649596b0efc1df5253804f7268d8d4646`.

Final cleanup receipt:
`31a8e77c75d33f077af0d0cec2630911f80507626bf16bf6489b52c1061f860d`.

Post-execution closeout receipt:
`717c89023fddc7185aa65448a7706f58e86329e6c28e25d03023a0113c44cb29`.

## Independent safety audit

The live and retained evidence independently proves:

- project VM inventory: zero;
- project disk inventory: zero;
- exact Step 12 Compute operations: zero;
- exact Step 12 `instances.insert` audit records: zero;
- direct Stage 2 result objects: zero;
- exact temporary IAM bindings: zero;
- both exact instance and disk names: HTTP 404;
- all 40 canonical and cross-reference checks: passed;
- `src/ofc_regular/ai_profiles.py` SHA-256:
  `d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3`.

No profile or baseline changed.

## Root-cause assessment

The HTTP status was not retained by the runner, so `403` is an inference rather
than a directly preserved fact. It is nevertheless the high-confidence
explanation:

- the wait helper retries only HTTP 403;
- its eight-attempt schedule sleeps for 39 seconds in total;
- the observed interval matches exhaustion of that loop;
- the Token Creator binding was added at `10:32:24.570Z` and removed at
  `10:33:27.177Z`;
- no successful `GenerateAccessToken` record exists in that interval.

Google documents IAM policy changes as eventually consistent: policy edits
typically propagate in about two minutes and can take seven minutes or longer.
The current 39-second barrier is therefore not sufficient for a cold grant:

<https://docs.cloud.google.com/iam/docs/access-change-propagation?hl=en>

Earlier Step 11 timing does not disprove this diagnosis. Its first successful
controller tokens followed cold grants by roughly 171-191 seconds. The one
near-immediate fix3 success followed a recent removal of the same principal and
can be explained by revocation propagation lag.

## Required correction before any new launch

1. Create a new run identity, two new exact instance names, a fresh output
   root, gate, expiry window, signer, and bounded exception.
2. Add only the Token Creator binding first and verify its exact readback.
3. Retry only HTTP 403 for a bounded period of about eight minutes.
4. Persist a secret-free token-barrier receipt containing HTTP status, attempt
   count, first/last timestamps, elapsed time, approved Google error metadata,
   and response-body SHA-256. Never persist a token or Authorization header.
5. After the controller token exists, add and exactly read back the remaining
   nine temporary bindings.
6. Use the controller credential for read-only permission and immutable-package
   propagation barriers.
7. Recollect capacity, prefix-empty, exact VM/disk 404, and authorization
   remaining-time evidence immediately before insert.
8. On a non-403 response, timeout, or any failed barrier, issue no insert and
   converge all cleanup independently.

The consumed exception has `retry=false` and `resume=false`. A corrected run
therefore requires a new explicit user authorization. Attempt 1, a third VM,
and reuse of this output root remain forbidden.
