# HU T2 Stage8 High-MC Audit Metrics

- total audited states: `50`
- by selection type: `{'fired_teacher': 20, 'suspected_false_positive': 20, 'missed_positive': 10}`
- diagnosis counts: `{'needs_more_samples': 8, 'model_ranking_error': 9, 'reference_margin_bad_gate': 21, 'false_positive_gate': 1, 'low_margin_noise': 1, 'underfire': 10}`

## MC512 vs MC4096

- sign agreement: `0.9600`
- delta Pearson: `0.9942`
- delta Spearman: `0.9915`
- teacher best == high-MC best: `0.8200`
- model top1 == high-MC best: `0.6000`

## Stage8 Candidate Rank

- average rank: `2.6400`
- p50 / p90 rank: `1.00` / `5.10`
- rank <= 3 rate: `0.7600`
- rank > 5 rate: `0.1000`

## Baseline Rank

- average rank: `8.3600`
- p50 / p90 rank: `6.00` / `20.10`
- baseline best rate: `0.0600`

## Candidate Delta

- mean / p50 / p10: `6.2149` / `8.2136` / `-0.1383`
- negative rate: `0.1200`
- positive >= 0.5 rate: `0.8200`
