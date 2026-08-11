# M4.3 Attempt07 No-Go closeout

Attempt07 is closed at the development100 selector boundary. The frozen run
`regular-hu-m43-attempt07-development100-20260714-150046` completed all 100
roots, with 20 roots for each of the five locked opponent profiles. The one
canonical selector artifact has SHA-256
`53b797756747cbb71f89251020e7fdc86dabce75cd298507dc59c0764ed6373a`.
Its decision is `no_go`, `selected_arm` is null, and `winner` is null.

All four frozen arms are ineligible. The development selector verified zero
action-mapping, RNG-domain, and hidden-information violations, and verified
exact baseline-action fallback for every non-fire. It did not claim complete
runtime trajectory cancellation; that acceptance item remains deferred. The
teacher A128 deltas are development diagnostics, not realized match EV.

The immutable evidence chain is recorded in
`configs/hu_joint_policy_m43_attempt07_closeout.json`. It binds the exact
selection, merged input, receive receipt, merge receipt, irreversible
consumption claim, frozen plan, selector source, package manifest, Spot
authorization, schedule, source closure ZIP, startup script, and unchanged
policy registry.

The closeout is intentionally fail-closed:

- audit50 is neither authorized nor open and must not run;
- no model fit or threshold selection is allowed from Attempt07;
- no runtime policy activation or full replacement is allowed;
- `current` and `src/ofc_regular/ai_profiles.py` remain unchanged;
- Attempt07 data is development/postmortem evidence only;
- any continuation requires a new attempt and new disjoint evidence.

Validate the closeout without writing or replacing any artifact:

```powershell
python -B -m ofc_regular.validate_hu_m43_attempt07_closeout --repo-root .
```

The validator fails on any changed byte identity, any altered run-chain link,
any winner or Go decision, any unbalanced development population, any nonzero
integrity count, or any opened science boundary.
