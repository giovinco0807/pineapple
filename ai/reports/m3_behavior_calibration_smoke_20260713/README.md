# M3 behavior-calibration wiring smoke (2026-07-13)

This directory is a reproducible, deliberately non-promoting bootstrap for
the T1/T2 behavior-likelihood pipeline.  It uses the four exact-hash HU
PolicyValueNet priors to sample full-private 54-card traces, reads the same
checkpoints' legal logits before temperature scaling or Q32 quantization, and
rebuilds the production temperature gate from disk.

It proves the following wiring only:

- BB-first T0/T1/T2 physical progression with distinct `X1` and `X2`;
- public decision logs separated from restricted hidden-root preimages;
- decision-local exact categorical sampling and deterministic replay;
- root-disjoint fit/dev/test assignment;
- direct pre-temperature checkpoint-logit extraction;
- a separate `m3-joker-challenge-v1/` population covering all 12
  T1/T2 x BB/BTN x visible-Joker(0/1/2) target cells;
- canonical artifact write/readback and full metric regeneration.

The run is not behavior-strategy or T3/T4 strength evidence.  Its labels are
self-sampled from the same frozen ranking prior, and its 12 natural plus 12
challenge roots are far below the production count thresholds.  Therefore
`production_gate_passed`, `promotion_eligible`,
`strategic_strength_evaluated`, and `strategic_strength_claimed` are all
`false` by design.

Stable readback commitments:

- report SHA-256: `de43673ad93a85fac32865748058a54ca4317388eaa8b0bc929fc395fea994c1`
- calibration artifact SHA-256:
  `0bd23f10fd1c8bae6b01a38b185302f87342b5b0a56bf7c245e00ec0df83fcc7`
- temperature gate v2 config SHA-256:
  `4f90642b47740785ea767aa80ce88055b429769756b657441299449c1e708569`
- behavior dispatch SHA-256:
  `7624a62afb98efce0961dba145afb3e8497b86edb55600cd69f75efe9b63b6b4`
- natural collection content SHA-256:
  `48b7137abf36e8d5cbf7016b404fe3801de9f28ece838ca85a8b1fb092c38ad2`
- Joker challenge collection content SHA-256:
  `3d12eab4c4f0b4cffd66ed7e9a62368ace39c636945eea485272da29aba0d678`

Reproduce from the repository root:

```powershell
python -m ai.tutor.run_behavior_calibration_smoke `
  --workspace-root . `
  --output-dir ai/reports/m3_behavior_calibration_smoke_20260713
```
