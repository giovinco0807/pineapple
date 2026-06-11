# HU T2 Stage8b Training Audit

This is a post-training audit for the deployable safe-override head. It is not production approval.

## Safe Head

- test p=0.90 precision / recall / F1 / PR-AUC: `0.8047` / `0.8460` / `0.8248` / `0.9262`
- holdout p=0.90 precision / recall / F1 / PR-AUC: `0.8307` / `0.8429` / `0.8368` / `0.9343`
- hard negatives caught at p=0.90: `6/15`

## Old Proxy Comparison

- old Stage8 m2.5/g0.9 holdout fires / avg gain / FP rate: `223` / `4.8549` / `0.0359`

## Runtime Gate Candidates

- `m2.75_p0.95_k1`: fires `187`, avg gain `5.1977`, FP `0.0374`, oracle precision/recall `0.8503` / `0.1951`
- `m2.75_p0.9_k1`: fires `192`, avg gain `5.1159`, FP `0.0469`, oracle precision/recall `0.8333` / `0.1963`
- `m2.75_p0.85_k1`: fires `196`, avg gain `5.0517`, FP `0.0510`, oracle precision/recall `0.8265` / `0.1988`
- `m2.75_p0.8_k1`: fires `199`, avg gain `4.9937`, FP `0.0553`, oracle precision/recall `0.8241` / `0.2012`
- `m2.75_p0.95_k2`: fires `216`, avg gain `4.9014`, FP `0.0463`, oracle precision/recall `0.8472` / `0.2245`

## Decision

Stage8b C2-small: `Go` for validation-only runtime proxy testing.

- 50k teacher: `No-Go`
- T1: `No-Go`
- production / P2 fixed: `No-Go`
