# T2 T3-union sample sweep

This sweep checks whether increasing T3 continuation samples improves the T2
runtime decision quality.  It uses the first 20 T2 positions from:

```text
ai/data/hybrid_t1t2_active_20260531/local_t1t2_500_mc300.jsonl
```

The teacher comparison is against the available MC300 full-action labels, so it
is a noisy regression signal rather than final exact truth.

## Results

| Mode | Sync candidates | T3 samples per candidate | Hit / 20 | Mean EV loss | Max EV loss | Loss >= 1.0 | Mean latency | Max latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fast baseline | 3 | 1 | 8 | 1.4511 | 7.8673 | 8 | 754 ms | 1046 ms |
| More samples | 3 | 4 | 9 | 1.1528 | 7.8673 | 5 | 2626 ms | 3995 ms |
| More samples | 3 | 8 | 10 | 1.0233 | 7.8673 | 4 | 5485 ms | 8310 ms |
| More samples | 3 | 16 | 8 | 1.1637 | 7.8673 | 5 | 10743 ms | 16438 ms |
| Wider sync | 10 | 4 | 10 | 0.8129 | 4.4063 | 6 | 8314 ms | 12774 ms |
| Wider sync | 10 | 8 | 12 | 0.5095 | 2.6823 | 4 | 16920 ms | 26301 ms |
| Shape selector | 3 | 1 | 10 | 0.8635 | 7.0687 | 5 | 824 ms | 1417 ms |
| Shape selector | 3 | 4 | 10 | 1.0658 | 10.8500 | 5 | 2770 ms | 4051 ms |
| Shape selector | 5 | 1 | 7 | 1.5551 | 7.0687 | 8 | 1149 ms | 1663 ms |
| Shape selector | 5 | 4 | 9 | 1.4743 | 10.8500 | 7 | 4140 ms | 4962 ms |

## 100-position check

| Mode | Sync candidates | T3 samples per candidate | Hit / 100 | Mean EV loss | Max EV loss | Loss >= 1.0 | Mean latency | Max latency |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Fast baseline | 3 | 1 | 31 | 1.9652 | 16.6777 | 36 | 745 ms | 1193 ms |
| Shape selector | 3 | 1 | 36 | 1.7202 | 15.4937 | 32 | 741 ms | 1272 ms |
| MC300 model top1 reference | n/a | n/a | n/a | 1.6285 | 20.1687 | n/a | n/a | n/a |

Output directories:

```text
D:/ofc-pineapple-data/t3_evloss_fresh_20260607/t2_t3_union_sample_sweep/local500_limit20_s4
D:/ofc-pineapple-data/t3_evloss_fresh_20260607/t2_t3_union_sample_sweep/local500_limit20_s8
D:/ofc-pineapple-data/t3_evloss_fresh_20260607/t2_t3_union_sample_sweep/local500_limit20_s16
D:/ofc-pineapple-data/t3_evloss_fresh_20260607/t2_t3_union_sample_sweep/local500_limit20_sync10_s4
D:/ofc-pineapple-data/t3_evloss_fresh_20260607/t2_t3_union_sample_sweep/local500_limit20_sync10_s8
D:/ofc-pineapple-data/t3_evloss_fresh_20260607/t2_sync_selector/t2_sync_selector_local500_mc300_shape_20260607.json
D:/ofc-pineapple-data/t3_evloss_fresh_20260607/t2_selector_benchmark/local500_limit20_selector_s1
D:/ofc-pineapple-data/t3_evloss_fresh_20260607/t2_selector_benchmark/local500_limit20_selector_s4
D:/ofc-pineapple-data/t3_evloss_fresh_20260607/t2_selector_benchmark/local500_limit20_selector_top5_s1
D:/ofc-pineapple-data/t3_evloss_fresh_20260607/t2_selector_benchmark/local500_limit20_selector_top5_s4
D:/ofc-pineapple-data/t3_evloss_fresh_20260607/t2_selector_benchmark/local500_limit100_selector_s1
```

## Findings

- Increasing samples from 1 to 8 helps, but not enough when the teacher-best
  action is outside the sync Top3.
- Going from 8 to 16 samples with only sync Top3 did not improve this slice.
  That points to candidate selection as the larger bottleneck, not only sample
  noise.
- Sync Top10 plus 8 samples gave the best quality in this sweep, reducing mean
  EV loss from 1.4511 to 0.5095 and max loss from 7.8673 to 2.6823.
- Sync Top10 plus 8 samples is too slow for live play: mean latency was about
  16.9 seconds on local CPU.
- The shape/blocker selector improved the 5-second candidate selection on this
  20-position slice at sync Top3, reducing mean EV loss from 1.4511 to 0.8635
  while staying under 1.5 seconds max.
- On 100 positions, the same selector still improves over the fast baseline
  but only modestly: hit count 31 to 36 and mean EV loss 1.9652 to 1.7202.
  This is not strong enough to call it a stable accuracy solution yet.
- Raising the selector path to 4 samples or Top5 did not improve this slice.
  The extra candidates/samples introduced enough sampled-continuation noise that
  worse actions sometimes won the refinement.
- Remaining large misses are mostly actions that are in the pool and refined,
  but the sampled T3 continuation estimate still disagrees with MC300, plus one
  case where the MC300 teacher-best action is not in the candidate pool.

## Practical conclusion

The 5-second path should not simply raise samples a lot.  A better next step is:

1. Keep the live path at selector sync Top3 with 1 T3 sample for now.
2. Mine cases where selector Top3 loses EV but Top10/Top20 contains a much better
   action, then train on those hard negatives.
3. Keep high-sample and wider-sync runs as offline labels, not the live answer
   path.
4. Do not promote selector Top5 or 4-sample mode until a larger holdout shows
   stable improvement.
