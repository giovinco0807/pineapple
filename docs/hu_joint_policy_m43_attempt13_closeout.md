# M4.3 Attempt13 immutable closeout

Date: 2026-07-16

Attempt13 is closed `complete_no_go_development`. The fixed Development200
selector was executed exactly once for
`regular-hu-m43-attempt13-development200-20260716-010306`. The immutable
closeout receipt is `configs/hu_joint_policy_m43_attempt13_closeout.json`,
SHA-256
`00e4189f879f9e5da6f020f7a91e22c2f5cdd2a9628a41a6a36be513213509c1`.
It binds the frozen search, distillation, and population contracts, the passed
correctness preflight, unchanged policy registry, Development200 package and
authorization chain, merged teacher rows and receive receipt, and the one-shot
selector decision and receipt by SHA-256.

## Frozen decision

Sixteen of eighteen gates passed. The result contained 44 fires against the
required 40, with per-profile fires of `stage19_p0=6`, `stage9f_p2=5`,
`stage7_m5_r10=5`, `stage3_baseline=11`, and `random_exact_final=17`; every
profile therefore exceeded the required minimum of three. Mean E512 delta was
`0.15868339330946415` per state and `0.7212881514066553` per fire. The
false-positive rate was `0.22727272727272727`, below the frozen maximum of
`0.4`.

The frozen fired-root loss p95 was `29.227020614683454`, above its limit of
`25`, and the maximum loss was `57.45404122936691`, above its limit of `50`.
Those are the two failed gates. Fired-root p99 loss was
`35.787020614683456`, below its limit of `40`. Action mapping, RNG domain,
hidden information, risk reserve, retained order, phase filtering, pooled
phase, locked-action, and extreme-tail-statistic violation counts were all
zero. Every non-fire returned the exact baseline action.

## Closed scope

The Audit50 namespace was not opened. No fit, threshold selection, runtime
freeze, runtime activation, population evaluation, profile promotion, or full
replacement was performed. `current` and the existing baseline profiles are
unchanged. `stage20_m4_attempt13` is not promotion eligible and has no explicit
or automatic activation authorization.

Development E512 values are search-quality evidence, not realized match EV.
They cannot be relabelled as a population result, used to reselect a threshold,
or retried on alternate seeds. This closeout does not claim unseen-population
robustness, an exploitability bound, Nash proximity, or mathematical complete
optimality. A continuation would require a separately frozen new attempt and
new disjoint evidence.

Validate the complete evidence chain without writing or activating anything:

```powershell
python -B -m ofc_regular.validate_hu_m43_attempt13_closeout --repo-root .
```
