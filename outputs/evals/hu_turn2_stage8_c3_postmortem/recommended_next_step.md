# C3 Postmortem Recommended Next Step

## Decision

- C3 larger seat-swap: `No-Go`
- production / P2 fixed: `No-Go`
- 50k teacher: `No-Go`
- T1 training: `No-Go`
- next: `Stage8b safe_override/confidence head design and selected high-MC diagnostic labels`

## Main Finding

- main candidate `m2.5_r0_g0.9` fired `45` unique states with override rate `0.0045`.
- dominant no-override reason: `below_stage8_margin`.
- selected top-loss audit states written: `81`.

The C1f oracle LCB signal is not deployable directly. The current runtime proxy underfires and does not reproduce the oracle advantage strongly enough in C3 seat-swap.

## Next Implementation Command

```powershell
.\scripts\Run-HuTurn2Stage8C3Postmortem.ps1
```

Equivalent direct command:

```powershell
python -m ofc_regular.analyze_hu_turn2_stage8_c3_postmortem `
  --c3-dir outputs/evals/hu_turn2_stage8_c3_larger_seat_swap `
  --output-dir outputs/evals/hu_turn2_stage8_c3_postmortem
```

After review, build Stage8b labels on the existing 20k teacher EV cache and selected MC4096/8192 replay set. Do not start 50k teacher or T1.
