# M4.3 Attempt05 architecture one-shot No-Go

Date: 2026-07-14

## Decision

Attempt05 is complete with status `complete_no_go_architecture_one_shot`.
Both pre-registered model families failed the frozen raw architecture gate, so
`selected_family` is `null`. No winner was frozen and no fresh audit, Spot run,
acceptance holdout, population evaluation, model registration, or runtime
activation was authorized.

The comparison used the already-consumed old dev900 set. It is a development
architecture diagnostic, not fresh generalization evidence. Its teacher
deltas are not realized HU match EV and cannot be used as a runtime gate.

## One-shot result

The immutable report is
`outputs/hu_joint_policy/m43_attempt05_old_dev900_architecture_once/architecture_comparison.json`
with SHA-256
`c8c8f364632ec85c600572239a387bedcc0ccbf82d3ef1f87147233ac765553f`.
It records 900 states, five identity-grouped folds, five balanced opponent
profiles, status `development_no_go_no_new_audit_authorized`, and no threshold
sweep.

The frozen raw gate required an overall selected positive rate of at least
0.40 and a selected positive rate of at least 0.30 for every profile.

- `lambda_rank`: raw gate failed. Overall positive rate was
  `0.35333333333333333`; the minimum profile rate was
  `0.31666666666666665`; selected mean teacher delta was
  `-0.9728898070556743`; mean regret to the best non-baseline action was
  `1.4570509226893777`.
- `deepsets`: raw gate failed. Overall positive rate was `0.21`; the minimum
  profile rate was `0.17777777777777778`; selected mean teacher delta was
  `-2.769567713271671`; mean regret was `3.253728828905375`.

The lower regret of LambdaRank does not rescue it: its predeclared overall
positive-rate gate still failed and its selected mean delta was negative.
Choosing it after seeing these results would be post-hoc family selection.

## Post-run trust audit

The trust audit is
`outputs/hu_joint_policy/m43_attempt05_old_dev900_architecture_once/postrun_trust_audit.json`
with SHA-256
`0414b3bbc7b8e678911c66cd038d73528f2a03c015e53b120017e116eb35512f`.
Its status is `pass_current_trust_boundary_no_reselection`.
The recorded one-shot wall time was `447.158` seconds.

The current hardened loader revalidated all 900 unique observation identities,
zero identity leakage, T1-second-only observations, canonical public
observations, complete legal ActionKey sets, exact baseline ActionKey/index
binding, and equality with the one-shot source manifest. Both candidate
artifacts loaded identically twice and remain `runtime_enabled: false` and
`winner_frozen: false`:

- LambdaRank SHA-256:
  `e7fa728308c8973e2e202baf79309e30b9b6f58c37431d8ea55850390abc28c3`
- DeepSets SHA-256:
  `c05b7469abe8d9ac8b4ccaf9def9f24bed4d6fc4670aab95d096baa702696b1b`

The chronology is intentionally explicit. The comparison process began before
the final trust hardening landed. The comparison was not reexecuted, the model
weights were not recomputed, and architecture selection was not recomputed.
The post-run audit instead revalidated the identical infoset-safe source data
and candidate artifacts through the hardened loader. That closes the current
integrity boundary; it does not create a winner, fresh evidence, or permission
to rerun selection on the consumed dev900 set.

## Exact-all and MC1 timing boundary

A local feasibility probe measured `400.711` seconds for exact evaluation of
all legal actions and `2.509` seconds for MC1, an observed wall-time ratio of
`159.709446`. These are timing diagnostics, not comparable policy-quality or
realized-EV results.

The exact-all result rules out an unbounded dev900 exact sweep under the
current implementation. The MC1 result only demonstrates a cheap sampling
path; it does not establish candidate recall, safe override quality, tail
control, or acceptance. Neither timing authorizes scaled local generation or
a Spot run.

## Fail-closed state

- Fresh `pilot_train`, `pilot_audit`, `expand_train`, and `final_audit` were
  never opened.
- All inherited Attempt04 acceptance roles remain unopened.
- Population acceptance remains unopened.
- No Spot package, instance, or run was started.
- `src/ofc_regular/ai_profiles.py` remains bound to SHA-256
  `d2eb02669e27426623c70af21f1af4d7002db2a7eab77efd3319f8fb65b868d3`;
  `current` was not changed.
- Runtime activation, full replacement, teacher-LCB gating, threshold
  reselection, and large-scale authorization all remain false.

The two candidate pickle files are reproducibility artifacts only. Their
existence is not model freeze or runtime authorization.

## Attempt06 handoff boundary

After this No-Go was closed, the strict-OOF top-4 and candidate-set diagnostics
completed as design evidence only. A corrected Attempt06 pre-fresh plan is now
separately frozen at `configs/hu_joint_policy_m43_attempt06.json`, selecting
the LambdaRank rank8/c8/e128 search design. This handoff does not change any
Attempt05 result or promote an Attempt05 artifact.

Attempt06 remains before local preflight. Fresh generation, Spot, runtime, and
acceptance are still unauthorized. Its diagnostic evidence cannot convert the
consumed dev900 metrics into an Attempt05 acceptance gate.
