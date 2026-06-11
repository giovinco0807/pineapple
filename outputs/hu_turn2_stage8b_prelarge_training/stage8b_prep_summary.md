# HU Turn2 Stage8b Pre-Large-Training Prep

This prepares labels and selected replay inputs. It does not start large training, 50k teacher generation, T1, or production.

- rows: `20000`
- replay-ready rows: `20000`
- safe_lcb196 positive / gray / negative: `2917` / `15218` / `1865`
- hard negatives: `15`
- high-MC label overrides: `50`
- current proxy m2.5/g0.9 fired: `961`
- selected teacher high-MC states: `200`
- C3 runtime top-loss replay states: `81`
- elapsed seconds: `6.58`

## Large Training Command

```powershell
python -m ofc_regular.train_hu_turn2_pilot_model --cache-dir D:\ofc-pineapple-storage\regular-ofc-pineapple\feature_cache\hu_t2_stage8_20k_mc512 --stage8b-labels-csv outputs/hu_turn2_stage8b_prelarge_training/stage8b_safe_override_labels.csv --gate-label-column safe_lcb196_gate_label_id --gate-weight-column stage8b_gate_weight --model-output models/hu_turn2_stage8b_safe_lcb196_20k_reference_override_cached_rank_wide.pt --output-dir outputs/training/hu_turn2_stage8b_safe_lcb196_20k
```

## Decision

- Stage8b large training: `Ready to launch with current MC4096 diagnostics included`
- 50k teacher: `No-Go`
- T1: `No-Go`
- production / P2 fixed: `No-Go`
