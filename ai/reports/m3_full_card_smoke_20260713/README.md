# M3 full-card six-stratum smoke (non-promoting)

This artifact verifies one real-card dynamic MCCFR iteration for each
`BB/BTN x visible Joker 0/1/2` stratum. It is an execution and wiring check,
not a strength or promotion result.

Reproduce from the workspace root:

```powershell
python -m ai.tutor.run_m3_full_card_smoke `
  --output-dir ai/reports/m3_full_card_smoke_20260713 `
  --iterations 1 `
  --base-seed 20260713 `
  --max-particles 4 `
  --max-infosets 100000 `
  --epsilon 0/1 `
  --temperature 1/1
```

Expected identities:

- behavior dispatch SHA-256:
  `fbd245d873aab3b79a80e25a37c378433a6803faed69506e08596c49bf431b46`
- evidence artifact SHA-256:
  `5f4375a67fa21d7bd55107c888ff589895b17f9309bef61cb01cc780a8b1e22f`
- gate result SHA-256:
  `5b9ba7c4a54d8d9c7a835e9ea47f749e24462339f1d8ad989a198b13272fec54`

Expected status:

```text
passed=true
execution_smoke_passed=true
behavior_prior_promotion_eligible=false
m3_promotion_passed=false
full_card_policy_promoted=false
```

The behavior dispatch uses the four exact-hash T1/T2 BB/BTN ranking priors and
the exact-hash T3 BB discard-sensitive ranking prior needed by BTN-root range
conditioning. All remain uncalibrated and non-promotable. A passing smoke must
not be reused as M3 strength or final-policy evidence.
