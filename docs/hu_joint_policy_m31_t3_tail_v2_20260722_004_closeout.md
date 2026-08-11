# M3.1 T3 tail-v2 run `20260722-004` closeout

## Disposition

Candidate02 tail-v2 qualification **passed**. This opens only the repeatable
full-100 performance-development run. It does not authorize performance-lock,
fresh quality, training, promotion, runtime activation, or a named/current
profile change.

Decision:
`candidate02_tail_v2_qualification_pass_open_full_performance_development_only`

## Frozen identity

- run: `regular-hu-m31-c02-perfdev-v2-tailv2-20260722-004`
- tail-v2 hands: `[0, 4, 5, 12, 14, 16, 17, 23, 41, 43]`
- heavy 21x21 hands: `[0, 5, 12, 16, 17, 23, 41, 43]`
- random calibration hands: `[4, 14]`
- run-contract digest:
  `b5d3114d0857723809ec85cef957921acafa67e22756aef050fa1c8ef8f79bf6`
- selection manifest SHA-256:
  `62cfbe2d95477ed7686a1b583997ee2e35ab23291fd338cfeac597e9a216fed1`
- source archive SHA-256:
  `624d7e57998fe0c8a7a425db09df0688f4031c20933ca33ad9b5d744e3c5e8de`
- execution plan SHA-256:
  `84dc6505ad953e185b7199c60550352787de0a877db7fc76690d1c13f9967468`

## Gate evidence

| Metric | Result | Gate |
|---|---:|---:|
| paired hand parity | 10/10 | 10/10 |
| portable root parity | 20/20 | 20/20 |
| heavy candidate first median | 69.140 s | <= 135 s |
| heavy candidate first max | 74.443 s | <= 145 s |
| heavy geometric-mean speedup | 2.915x | >= 1.55x |
| all candidate first median | 68.750 s | diagnostic |
| all candidate first max | 74.443 s | diagnostic |
| candidate second max | 0.877 s | <= 5 s |
| peak source-process RSS | 509,657,088 B | <= 858,993,459 B |

Missing, duplicate, out-of-contract, censored, and partial artifacts were all
zero. All eight heavy first-seat roots were exactly 21x21. The independent
validator recomputed integrity from the received source artifacts and passed.

- merge summary SHA-256:
  `3bf1da55d4f800800c8fbf7cd12a771d29f86eaaf5f1fcac6a0cb2c96a1bb0f6`
- gate summary:
  `D:\ofc-gcp-runs\perfdev-v2-control-20260722-v9\tail_gate_summary.json`
- independent validation:
  `D:\ofc-gcp-runs\perfdev-v2-control-20260722-v9\tail_gate_validation.json`
- immutable receive tree:
  `D:\ofc-gcp-runs\perfdev-v2-received-20260722-v6\regular-hu-m31-c02-perfdev-v2-tailv2-20260722-004`

## Cleanup and profile safety

- exact two owned instances were deleted; wildcard deletion was not used
- exact two temporary worker IAM bindings were removed
- live post-cleanup targeted IAM binding count was zero
- unrelated instances were not touched
- `src/ofc_regular/ai_profiles.py` SHA-256 remains
  `d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3`
- `current` and all named profiles remain unchanged

## Authorized next action

Implement and independently audit the quota-bounded full-100 cloud transport,
then run the frozen 100 paired hands as `8 + 8 + 4` candidate/reference VMs.
The transport must remain fail-closed until persistent atomic launch claim,
fresh live quota/headroom readback, one-VM/one-job/one-role launch receipts, and
owned-root plus parent-directory fsync evidence are connected and tested.

Only after that run passes may the one-shot performance-lock stage be prepared.
