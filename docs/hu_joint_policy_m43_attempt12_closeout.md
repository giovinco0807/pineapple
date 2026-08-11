# M4.3 Attempt12 immutable closeout

Date: 2026-07-15

Attempt12 is closed `No-Go`. The immutable closeout receipt is
`configs/hu_joint_policy_m43_attempt12_closeout.json`, SHA-256
`f8b46789698e799233d0902e9925d11d62caa054084a3c255354fa0b44f167a3`.
It binds the frozen plan, selector decision and receipt, merged Development200
teacher data, and receive receipt by SHA-256.

Development200 produced 35 fires against a required 40. Per-profile fires
were `stage19_p0=3`, `stage9f_p2=4`, `stage7_m5_r10=2`,
`stage3_baseline=2`, and `random_exact_final=24`, so two profiles also missed
the required minimum of three. Mean delta per state was
`0.12179967622634498`, mean delta per fire was `0.6959981498648284`, and the
false-positive rate was `0.14285714285714285`. Fired-root p95/p99 losses
passed at `24.677020614683453/35.227020614683454`, but maximum loss was
`53.45404122936691`, above the frozen limit of 50. Every integrity violation
count was zero and every non-fire returned the exact baseline action.

These rows are now consumed architecture/training diagnostics only. They may
inform Attempt13 design, training, and root-grouped OOF threshold selection,
but they are not fresh generalization evidence and cannot be Attempt13
development or audit rows. Attempt12 Audit50, fitting, runtime activation,
population evaluation, threshold reselection, seed reuse, `current` changes,
and full replacement remain unauthorized.

