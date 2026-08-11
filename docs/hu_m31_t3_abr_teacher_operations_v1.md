# M3.1 ABR real-teacher operations

This pipeline labels balanced T3 response roots with real terminal outcomes.
It does not launch a VM, train from locked evaluation seeds, register a
profile, resolve `current`, or report teacher Q as realized match EV.

## Scientific boundary

The accepted Candidate02 engine remains authoritative for the regular HU
action Q. Its existing T3 API does not expose foul, scoop, royalty, or
Fantasyland decomposition. Therefore a diagnostic-only native endpoint,
`t3_abr_components`, accumulates those terminal components without changing
the legacy `t3` request or output.

Every root is evaluated by both binaries with identical public
`ActorObservation`, candidate-selection seed, independent evaluation seed,
continuation seed, and budget. The generator rejects the root unless every
ActionKey's selection Q, evaluation Q, and selected ActionKey are bit-exact.
Only aggregate component means are serialized; sampled opponent discards and
future cards stay inside the native world state.

The response utilities are frozen in
`hu_m31_t3_abr_teacher_v1.FAMILY_REWARD_DEFINITIONS`:

- `greedy_search_response`: regular HU terminal score. This is the only
  direct HU-score approximate-best-response family.
- `foul_pressure_response`: HU score plus weight 3 for opponent foul and
  weight 3 for hero scoop. This is an exploitative stress family, not a
  direct best response.
- `royalty_denial_response`: HU score minus half of opponent royalty and
  opponent 14-card FL value. This is also an exploitative stress family, not
  a direct best response.

They are linear expectations of terminal rewards, not hand-written synthetic
rank targets. None of the three is a NashConv upper bound or a proof of low
exploitability.

## Coverage

`abr_development` reserves 250 paired seeds. Each seed produces one first-seat
and one second-seat T3 root, and every root carries all three response
families.

- Pilot: 50–249 pairs (100–498 states). It can test correctness and speed but
  cannot enter production normalization.
- Production: exactly 250 pairs, 500 states, 250 per seat, all three families,
  and the exact reserved hand-seed grid. Because every root carries all three
  targets, each response family has exactly 250 pairs / 500 seat roots.

The exact production coverage is rechecked by:

1. teacher finalization;
2. raw-to-normalized dataset freeze;
3. three-policy training/resume; and
4. locked-promotion provider preflight through
   `production_build_receipt.json`.

The direct numerical library APIs remain available for small unit fixtures,
but the production CLIs reject those synthetic fixtures.

## Isolated diagnostic build

Build in the same Linux/WSL environment used to load the accepted Linux
Candidate02 `.so`. Use a separate target directory so the accepted artifact
cannot be overwritten:

```bash
cd /path/to/regular-ofc-pineapple
CARGO_TARGET_DIR=target/abr_diag_v1 \
  cargo build -p ofc_hu_m3_engine --release
sha256sum target/abr_diag_v1/release/libofc_hu_m3_engine.so
```

Do not replace or rename the accepted Candidate02 binary. The generator takes
both paths and pins both hashes.

## 100-state pilot entry

The following only prepares and runs a bounded local/WSL pilot. Replace the
paths and diagnostic hash with the actual immutable artifacts:

```bash
python scripts/run_hu_m31_t3_abr_teacher_v1.py prepare \
  --run-directory outputs/hu_joint_policy/m31_t3/abr_teacher/pilot100 \
  --pair-count 50 \
  --accepted-library /immutable/libofc_hu_m3_engine_candidate02_4050e04b22d7943d.so \
  --diagnostic-library target/abr_diag_v1/release/libofc_hu_m3_engine.so \
  --expected-diagnostic-library-sha256 <diagnostic_sha256>

python scripts/run_hu_m31_t3_abr_teacher_v1.py run \
  --run-directory outputs/hu_joint_policy/m31_t3/abr_teacher/pilot100 \
  --accepted-library /immutable/libofc_hu_m3_engine_candidate02_4050e04b22d7943d.so \
  --diagnostic-library target/abr_diag_v1/release/libofc_hu_m3_engine.so \
  --max-new-pairs 1

python scripts/run_hu_m31_t3_abr_teacher_v1.py status \
  --run-directory outputs/hu_joint_policy/m31_t3/abr_teacher/pilot100
```

Increase `--max-new-pairs` only after the first pair has passed and its timing
is acceptable. `run` replays every retained immutable pair before doing new
work. Unknown files, gaps, duplicate roots, altered hashes, RNG overlap,
ActionKey drift, hidden fields, or Q parity failure stop the run.

For independent workers, prepare the same plan in separate run directories,
pass one or more repeatable `--pair-index` values to each worker, and merge
only after the shard runs have stopped writing:

```bash
python scripts/run_hu_m31_t3_abr_teacher_v1.py merge \
  --run-directory outputs/hu_joint_policy/m31_t3/abr_teacher/pilot100 \
  --source-run-directory /immutable/shards/shard-000 \
  --source-run-directory /immutable/shards/shard-001
```

Merge replays each source plan and pair certificate, rejects a mismatched plan
or conflicting duplicate, and publishes `DONE` only when the full frozen pair
grid is present.

After all planned pairs:

```bash
python scripts/run_hu_m31_t3_abr_teacher_v1.py finalize \
  --run-directory outputs/hu_joint_policy/m31_t3/abr_teacher/pilot100
```

A pilot receipt remains explicitly `pilot_only`; the production
`freeze-dataset` command refuses it. No cloud fanout is performed by any
command in this module.
