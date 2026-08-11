# OFC Pineapple Web App — Acceptance Report

Date: 2026-07-29  
Ruleset: Regular OFC Pineapple, heads-up, no Joker

## Clarified Fantasy Land rule

The user's final clarification is authoritative: this app uses the repository's
regular rules. QQ, KK, AA, and front-row trips all enter Fantasy Land with
14 cards, and a qualifying stay also receives 14 cards. The earlier
14/15/16/17 acceptance matrix is therefore replaced by four 14-card entry
cases.

## M-A — State machine, API, and persistence

- Implemented the hand and match state machines, alternating first position,
  200-point stacks, loss-side capping, forced match end at zero, and
  continue/end handling.
- Human actions are checked against
  `src/ofc_regular/action_space.py::generate_turn_actions` on the server.
- The deck and all private AI information remain server-side in public API
  responses.
- SQLite records decisions and hand summaries; the authenticated export route
  produces one JSON object per line.
- Result: backend/domain/API/repository tests pass.

## M-B — Native scoring and AI assembly

- Final score, foul, royalties, and Fantasy Land status use
  `score_final` from the supplied Rust engine as the point authority.
- A Rust inspector obtains display-only canonical score details and fails
  closed if its HU score differs from `score_final`.
- T0/T1 use the `stage19_p0` production policy; later streets dispatch to the
  pinned learned/native evaluators declared in `assembly.json`.
- Every AI decision records evaluator, weight SHA-256 values, elapsed thinking
  time, and three evaluated candidates.
- AI Fantasy Land uses `regular_fl_solver`; its rank-one answer is checked
  against an independent canonical Rust enumeration, which also supplies
  ranks two and three.
- Result: Rust inspector 4 tests passed.

## M-C — Mobile PWA

- Implemented the seven specified React/TypeScript screens: lobby, normal
  placement, Fantasy Land placement, hand result, match summary, history
  replay, and export.
- Verified a complete flow at a 375 x 812 viewport.
- Result: 5 frontend test files / 12 tests passed; production build passed;
  dependency audit reported zero vulnerabilities.

Screenshots:

- `artifacts/screenshots/mobile-375-match-result.png`
- `artifacts/screenshots/mobile-375-history-replay.png`
- `artifacts/screenshots/foul-fix-production.png`

## M-D — Acceptance tests

- Full backend/native suite: 70 passed in 19.78 seconds.
- Golden seed: 1717.
- Golden match: at least five hands, with an actual Fantasy Land entry.
- Every completed hand was re-scored directly through `score_final`.
- Stack changes and loss-side capping matched the settlement rule.
- Decision count matched JSONL decision rows; each hand added exactly one
  summary row.
- Invalid overflow, duplicate-card, and wrong-discard actions returned HTTP
  400.
- All four regular entry categories (QQ, KK, AA, trips) were exercised as
  14-card Fantasy Land cases through UI/API/recording tests.
- AI Fantasy Land rank one matched the solver output.
- The local replay export `artifacts/exports/match-53f650af.jsonl` contains
  10 decision rows and one hand-summary row. Reconstructing both 13-card
  boards from those decisions reproduced the stored final boards and score.

## M-E — Cloud Run production verification

- Project: `ofc-solver-485418`
- Region: `asia-northeast1`
- Service:
  `https://ofc-pineapple-webapp-zylrq47xqq-an.a.run.app`
- Revision: `ofc-pineapple-webapp-00004-k6c` (Ready, 100% traffic)
- Cloud Build: `aed00524-cdb9-493c-ad7a-e0668ea69922` (SUCCESS)
- Image digest:
  `sha256:8b723b2e4eb477ae99688e425b4da17c9d416fad6b2d0fcb4813d5c71ad818f7`
- Assembly SHA:
  `914abdd0667f514fe5e5e900548d200421f8650c1d99c4ba858031d9ed25434d`
- Container health: 19.72 seconds; revision ready: 49.31 seconds.
- `/` and `/api/healthz` returned HTTP 200.
- Authenticated `/api/meta` returned HTTP 200 and the expected regular
  assembly/SHA configuration; the same request without Bearer authentication
  returned HTTP 401.
- Production seed-1717 smoke match
  `daf762ad-103c-484b-99d6-09ac44967a8a` completed with 10 decisions,
  a final score of -10, and stacks 190/210.
- Its point components satisfied
  `-6 + 0 + 0 - 4 = -10 = hu_score = raw_score`.
- Production export contained 10 decision rows and one summary row. All five
  AI rows contained three ranked evaluations and the production assembly SHA.
- The final SQLite backup reached
  `gs://ofc-webapp-records-ofc-solver-485418/sqlite/ofc-webapp.sqlite3`
  at generation `1785312846266694`; `last_backup_error` was null.
- The offline demonstration fixture was re-verified after deployment with its
  fixed AI board: middle flush over bottom two pair is shown as an AI foul,
  settled as +6 foul points plus +8 valid-side royalties, for +14 and
  214/186 stacks.

The Bearer token is stored in Secret Manager as
`ofc-webapp-shared-token`; its value is intentionally not included here.

## Isolation

All implementation and evidence files are under `webapp/`. Existing repository
files and pre-existing dirty worktree changes were not modified. No git commit
was created.
