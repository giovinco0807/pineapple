# M3.1 T3 perfdev-v2 run `20260722-002` closeout

## Disposition

This run is retained as a valid transport and mixed-geometry performance
diagnostic. It is **not** a Candidate02 tail qualification result and does not
authorize full100, performance-lock, quality, training, promotion, or a runtime
profile change.

The frozen run used hands
`[2, 6, 7, 9, 13, 20, 21, 29, 33, 50]`. Those hands matched the explicit
profile-tail list in the revised roadmap, but the inherited gate also required
all eight non-random first-seat roots to be 21x21. The received root geometries
show that only hand 20 is 21x21:

| hand | first-seat geometry |
|---:|---:|
| 2 | 21x12 |
| 6 | 9x9 |
| 7 | 12x12 |
| 9 | 9x12 |
| 13 | 9x21 |
| 20 | 21x21 |
| 21 | 21x12 |
| 29 | 12x12 |
| 33 | 12x12 |
| 50 | 12x21 |

The local gate therefore failed closed with
`Step 6d heavy tail hand 2 is not 21x21`. The validator is not weakened and
the result is not relabelled after observation.

## Valid evidence from the run

- exact candidate/reference VM pair launched and completed
- immutable candidate/reference receive succeeded
- 10/10 hands and 20/20 seat roots have exact portable-decision parity
- missing, duplicate, partial, and censored artifacts: 0
- candidate first-seat median: 29.016 seconds
- candidate first-seat maximum: 78.104 seconds
- candidate/reference first-seat geometric-mean speedup: 3.108
- candidate second-seat maximum: 1.018 seconds
- peak source-process RSS: 487,563,264 bytes
- exact two owned VMs deleted
- exact two temporary IAM bindings removed; post-cleanup targeted binding count 0
- `current` and all named profiles unchanged

These timings are diagnostic only because the geometry qualification contract
was inconsistent.

## Corrected next probe

The correction uses the already frozen Candidate02 tail-v2 selection:

- heavy 21x21 hands: `[0, 5, 12, 16, 17, 23, 41, 43]`
- random calibration hands: `[4, 14]`
- exact tail-v2 set: `[0, 4, 5, 12, 14, 16, 17, 23, 41, 43]`
- selection manifest SHA-256:
  `62cfbe2d95477ed7686a1b583997ee2e35ab23291fd338cfeac597e9a216fed1`

This is a repeatable performance-development probe. All development roots were
already exposed by the historical full100 run, so it must not be represented as
fresh quality or holdout evidence. It will use a new package, run identity,
result prefix, launch nonce, receive tree, and explicit tail-v2 schemas.

## Corrected probe result

The corrected run `20260722-004` passed all tail-v2 qualification gates. Its
authoritative closeout is
`docs/hu_joint_policy_m31_t3_tail_v2_20260722_004_closeout.md`. The authorization
is limited to full-100 performance-development; performance-lock, quality,
training, promotion, and profile activation remain closed.
