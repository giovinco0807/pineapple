# M4.3 Attempt06 search-quality No-Go

Date: 2026-07-14

## Decision

Attempt06 is closed as `complete_no_go_search_quality`. The one-shot fresh
audit failed six strength and tail gates, so no fit, threshold retry, Spot
expansion, runtime activation, or profile promotion is allowed. The existing
P0/P1/P2/T3 baselines and `current` remain unchanged.

The immutable audit is
`outputs/hu_joint_policy/m43_attempt06_search_quality/regular-hu-m43-attempt06-preflight-final-20260714-1154-search_quality_audit.json`
with SHA-256
`7ee02666f86d93f905183e45ecff08dfde9a8120567ea061d9690149d75997f2`.
The explicit machine-readable closeout is
`configs/hu_joint_policy_m43_attempt06_closeout.json`.

## What passed

The 50-root population was balanced across five opponent policies. It retained
the complete legal ActionKey mapping, disjoint candidate/evaluation RNG
domains, infoset-safe observations, and exact non-fire counterfactual
cancellation. Action-mapping, RNG-domain, and hidden-information violation
counts were all zero. The audit lifecycle also remained one-shot: content was
opened only after the consumption claim and no fit or threshold selection ran.

## Why search quality failed

The rank-8 LambdaRank output was used as a candidate generator, then common-
random MC8 selected one of eight proposals plus the explicit baseline. The
independent E128 block could diagnose that locked choice but could not rerank
or cancel it.

MC8 fired on 36 of 50 states. On those fires its selected-minus-baseline
estimate averaged `+5.2984`, while independent E128 averaged `-0.7022`.
Mean optimism was `+6.0006`, RMSE was `7.2772`, Pearson correlation was
`0.0266`, and Spearman correlation was `-0.0099`. The MC8 top-versus-runner
gap also had no useful relationship to E128 (`-0.0634` Pearson), and 28 of 36
gaps were smaller than the selected action's own MC8 standard error.

That selection noise produced 22 false-positive fires out of 36 (`61.11%`).
Only 17 fires passed all three per-root tail limits, and only 8 of the 14
positive-mean fires also passed every tail limit. Overall mean delta per state
was `-0.5056`; the maximum fired-root P95/P99/max losses were
`31.8270 / 42.9140 / 65.4540`.

The correct conclusion is narrow: MC8 cannot be the final selector. The fresh
audit evaluated only the locked action and baseline, so it neither proves nor
disproves rank-8 candidate recall.

## Attempt07 boundary

Attempt07 must preserve LambdaRank as candidate-generation-only and add
independent stages:

1. rank-8 plus the explicit baseline, fixed before sampling;
2. a cheap common-random screen over all nine actions;
3. an independent higher-budget rerank over a fixed small shortlist plus the
   baseline;
4. an independent safety-confirmation block that may fall back to baseline by
   a predeclared mean-and-tail rule;
5. a final disjoint audit block that can diagnose but never rerank.

A finite pilot matrix must be selected once on new balanced development roots.
Only one frozen configuration may proceed to a new disjoint one-shot audit.
Attempt06 thresholds or seeds cannot be retried, and teacher EV/LCB must not
become a direct runtime gate.
