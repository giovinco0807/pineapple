# M2 reduced public-tree promotion gate v2

Date: 2026-07-13

## Result

`promotion_gate_v2` passes all six required reduced strata:

- BB x visible Joker 0/1/2;
- BTN x visible Joker 0/1/2.

The result is `m2_reduced_reference_ready`. It explicitly leaves
`full_card_policy_promoted=false`, `hu_exact=false`, and
`explicit_full_deck_chance_enumeration=false`.

## Measured summary

| Stratum | Exploitability | NashConv | Python/Rust max leaf error | 3-replay max policy TV | BR residual |
|---|---:|---:|---:|---:|---:|
| BB Joker 0 | 0.00003731343283597255 | 0.0000746268656719451 | 0 | 0 | 0 |
| BB Joker 1 | 0.000152017689331152 | 0.000304035378662304 | 0 | 0 | 0 |
| BB Joker 2 | 0.00005804311774459947 | 0.00011608623548919894 | 0 | 0 | 0 |
| BTN Joker 0 | 0.00003731343283597255 | 0.0000746268656719451 | 0 | 0 | 0 |
| BTN Joker 1 | 0.00007048092868977562 | 0.00014096185737955125 | 0 | 0 | 0 |
| BTN Joker 2 | 0 | 0 | 0 | 0 | 0 |

Runtime evidence:

- reduced-bluff 20,000-iteration calibration max: 645.4155 ms;
- canonical recursive solve p95: 5743.0552 ms;
- canonical recursive solve max: 6221.8679 ms;
- complete gate: 127.658032 s;
- generator parent-process peak working set: 390.20703125 MB.

## Evidence integrity

The evidence embeds the raw per-stratum run records and manifests for the
action, infoset, fixture, chance, range, solver configuration, solver code,
and complete source tree. The validator re-derives every thresholded metric
from the raw records and recomputes each manifest binding.

- gate config SHA-256:
  `ad1f657dc098f5d35ebd593550e250c2f27f16e2315cf74721465e8c03e92fb2`
- evidence SHA-256:
  `72e8c80dcb7ecbb9ff97ca82db70f05845c1814251a5da9a817b42fe441fd3dc`
- artifact SHA-256:
  `c13e335b3f1f5e98c09c7f22d06f9748cf7fdec5a8debe1afeec903f88b15beb`
- raw measurement SHA-256:
  `de7865a759f04e2b5e58160f8211d552c8579afce6e9f77eadbf71aca2dc2a23`

## Reproduction

```powershell
python -m ai.tutor.m2_promotion_evidence `
  --output-dir ai/reports/m2_promotion_gate_v2_20260713 `
  --iterations 200 `
  --runtime-warmups 3 `
  --runtime-measured 5
```

Saved files:

- `evidence.json`: raw measurements, strategies, metrics, and provenance;
- `gate_result.json`: fail-closed readback result.

This report completes M2 only. M3 still needs a history-weighted full-card
range and a full-card external-sampling MCCFR policy with root-disjoint gates.
