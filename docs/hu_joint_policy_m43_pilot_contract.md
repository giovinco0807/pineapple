# M4.3 bounded pilot contract

Status: frozen pre-generation contract. No cloud run, model promotion, runtime
activation, or `current` profile change is implied by this document.

M4.2 demonstrated that fitting a high-dimensional ranker on 20 independent
training identities and five threshold-lock candidates did not generalize. The
M4.3 pilot therefore tests an explicit candidate-minus-baseline risk ensemble
on a larger but still bounded fresh sample before any large-scale run.

## Frozen allocation

- 200 T1-second roots total, in 20 restart-safe shards of 10 roots;
- 100 train roots;
- 60 calibration roots, deterministically divided into 30 safety-fit and 30
  threshold-lock identities;
- 40 locked-holdout roots, evaluated once only after the model, calibrator, and
  threshold are frozen;
- five root profiles, with 20/12/8 roots per profile across
  train/calibration/locked holdout;
- teacher search `c2/e16`, with common evaluation futures and disjoint
  candidate-selection/evaluation RNG domains.

The frozen plan is
`configs/hu_joint_policy_m43_pilot.json`. Its validator is
`src/ofc_regular/hu_m43_pilot_contract.py`.

## Freshness

Every available M4, M4.1, and M4.2 teacher shard is bound by path, record
count, and SHA-256. The current exclusion union contains 252 unique hand seeds,
252 unique observation fingerprints, and 252 unique joint identities. M4.3
fails closed if either a hand seed or an observation fingerprint overlaps this
union. New train, calibration, and locked identities must also be mutually
disjoint.

The calibration role assignment is deterministic and profile-stratified. For
each of the five profiles, six identities enter safety-fit and six enter
threshold-lock. Reordering shard rows does not change the assignment.

## Paired label and risk targets

All legal actions retain the M4.2 common-future paired summary. M4.3 consumes:

- `paired_delta_mean = paired_delta_vs_baseline.mean`;
- `paired_delta_se = paired_delta_vs_baseline.standard_error`;
- `downside_loss_p95 = max(0, -paired_delta_vs_baseline.p05)`;
- `downside_loss_p99 = max(0, -paired_delta_vs_baseline.p01)`;
- `downside_loss_max = max(0, -paired_delta_vs_baseline.min)`.

The explicit baseline action must have every one of these targets exactly
zero. Teacher scores and teacher LCBs remain training/diagnostic inputs only;
they are not realized match EV and are not legal runtime gate inputs.

## Freeze and one-shot boundary

Ranker fitting and nested OOF prediction use train identities only. The
low-capacity safety calibrator may use train OOF plus calibration safety-fit.
Only calibration threshold-lock may select a threshold. A valid
`hu_m43_model_threshold_freeze_v1` manifest must record zero locked-label
access by model or threshold selection before the one-shot evaluator can run.

The locked receipt must bind the frozen model/threshold manifest and exact
locked identity digest. It rejects a second pass, threshold/model/feature
selection, policy promotion, runtime activation, and `current` resolution.
Changing the model or threshold after observing the locked result consumes the
holdout and requires a new fresh locked split.

The sealed data contract also binds every teacher shard in caller-supplied
order by normalized path, byte length, file SHA-256, canonical-row SHA-256,
record count, and identity SHA-256. It embeds the complete base data audit and
binds its digest, the frozen plan digest, and the prior-freshness audit digest.
Changing a score, action, paired SE, or downside-tail value while retaining the
same seed/fingerprint is therefore rejected.

Ten of the 30 threshold-lock identities must fire for a positive bounded-pilot
signal. This is deliberately a pre-promotion sensitivity gate: requiring all
30 identities to override would reject a correct baseline-relative model that
keeps the baseline on some states. Fewer than ten fires produce
`no_go_insufficient_pilot_signal`. The independent final acceptance requirement
remains at least 300 fresh valid realized overrides and is not lowered.

## Promotion boundary

This 200-root teacher pilot cannot promote a policy. A passing pilot only
authorizes a separately seeded population evaluation with at least 300 valid
overrides, paired seat swap, first/second breakdowns, multiple opponent
families, false-positive and p95/p99 tail gates, and exact non-fire
counterfactual cancellation. Existing M4 acceptance thresholds remain fixed.

## Local preflight

Run before generating any teacher shard:

```powershell
python -m pytest tests/test_hu_m43_pilot_contract.py -q
python -c "from pathlib import Path; from ofc_regular.hu_m43_pilot_contract import load_and_validate_plan, audit_frozen_exclusions; r=Path.cwd(); p=load_and_validate_plan(r/'configs/hu_joint_policy_m43_pilot.json'); print(audit_frozen_exclusions(p, repo_root=r)['identity_sha256'])"
```

After generation, audit and freeze the complete split contract before the
trainer receives any locked path:

```powershell
python -m ofc_regular.hu_m43_pilot_contract `
  --plan configs/hu_joint_policy_m43_pilot.json `
  --repo-root . `
  --train <train-shard> `
  --calibration <calibration-shard> `
  --locked-holdout <locked-shard> `
  --output <data-contract.json>
```

Repeat `--train`, `--calibration`, and `--locked-holdout` for multiple shards.
The command performs the independent structural audit; it does not train,
evaluate, activate, or modify any profile.

After training has saved the model and a manifest containing only train and
calibration inputs, create the immutable pre-holdout freeze:

```powershell
python -m ofc_regular.freeze_hu_m43_model `
  --model <model.pkl> `
  --training-manifest <training_manifest.json> `
  --data-contract <data_contract.json> `
  --plan configs/hu_joint_policy_m43_pilot.json `
  --repo-root . `
  --train <train-shard> `
  --calibration <calibration-shard> `
  --output <model_threshold_freeze.json>
```

Only then may the separate evaluator receive the locked path:

```powershell
python -m ofc_regular.evaluate_hu_m43_locked_holdout `
  --model <model.pkl> `
  --freeze-manifest <model_threshold_freeze.json> `
  --data-contract <data_contract.json> `
  --plan configs/hu_joint_policy_m43_pilot.json `
  --repo-root . `
  --locked-holdout <locked-shard> `
  --consumption-marker <data-contract-dir>/M43_LOCKED_CONSUMED.json `
  --receipt <locked_receipt.json>
```

The evaluator creates the consumption marker with an exclusive filesystem
claim before reading locked content. A mismatch or crash after the claim still
consumes the holdout; deleting the marker to rerun is outside the contract.
