# HU M9 — realtime solver API schema, v0

Status: prototype, local only. Nothing here is deployed; no ports are open
beyond `127.0.0.1`. The running reference implementation is `~/ofc-api`
(FastAPI, one warm process, `serve.sh` binds `127.0.0.1:8099`).

This document is the contract. Where the pinned artifacts cannot honour a
part of it, the endpoint says so in the response rather than returning a
number it cannot stand behind — the "what is real" table at the bottom is
the authoritative list of those places.

---

## 1. Design constraints this shape answers to

Three things were required of v0 up front, and each one shows up as a field
rather than as a branch in the code:

**Rule variants are parameters, not endpoints.** A `ruleset` block carries
the deck size, the joker flag, and `fl_entry_cards` — a map from entry
qualifier to the number of cards the Fantasyland hand is dealt. Regular play
maps every qualifier to 14. The joker variant's 14/15/16/17 chain is the same
field with more values, so a client written against v0 keeps compiling when
that variant lands. The server already reads its own value from the FL EV
config (`configs/fl_ev_regular_v3_direct2.json` carries
`fl_entry_cards: {qq:14, kk:14, aa:14, trips:14}` and `fl_stay_cards: 14`),
so the config and the wire format agree by construction.

**Table size is a parameter, not a rewrite.** Requests carry a `players`
array and a `hero_index`. v0 refuses anything other than two players — the
models are heads-up — but the refusal is a validation error against a shape
that already admits three. A three-way table needs new models, not a new
request type.

**Provenance travels with the answer.** Every response carries a
`provenance` block naming the evaluator, whether it is exact or a learned
proxy, the model digest where one was used, and the fallback tag when the
answer did not come from the machinery the endpoint is nominally for. A
client can tell a solved line from a proxied one without asking.

---

## 2. Common types

### `ruleset`

| field | type | default | meaning |
|---|---|---|---|
| `id` | string | `"regular_hu_v1"` | variant identity, echoed back as `ruleset_id` |
| `deck_cards` | int | `52` | |
| `include_jokers` | bool | `false` | |
| `fl_entry_cards` | map<string,int> | server default | qualifier → FL deal size |
| `fl_stay_cards` | int | server default | cards dealt on a Fantasyland stay |

### `rows`

```json
{"top": ["As"], "middle": ["Kd", "Qc"], "bottom": ["2h", "3h"]}
```

Cards are two-character tokens, rank in `23456789TJQKA`, suit in `hdcs` —
byte-identical to `ofc_regular.cards.ALL_CARDS` and to the Rust engine's
`u8` index layout (`suit * 13 + rank_index`).

### `player`

| field | type | meaning |
|---|---|---|
| `seat` | `"first"` \| `"second"` | within-street act order |
| `board` | `rows` | public board; empty for a Fantasyland player |
| `private_discards` | string[] | that player's own discards, hero only |
| `in_fantasyland` | bool | hidden-FL flag |

### `options`

| field | default | range | meaning |
|---|---|---|---|
| `max_actions` | 10 | 1–512 | how many ranked actions to return |
| `include_scores` | true | | skip the ranking pass when false |
| `opponent_samples` (N) | 200 | 1–4000 | vs-FL: sampled Fantasyland hands. The dominant cost. |
| `hero_deals` (M) | 8 | 0–512 | vs-FL T3: hero's own draw expectation; `0` means exhaustive |
| `t3_draws` / `t4_draws` | 8 / 4 | 1–128 | vs-FL T2: the two nested hero draws |
| `threads` | 1 | 1–6 | FL solver worker threads |

### Response envelope

```json
{
  "schema": "ofc_solver_api_v0",
  "request_id": "uuid4",
  "ruleset_id": "regular_hu_v1",
  "street": "T3",
  "seat": "first",
  "served": true,
  "best_action": {"action_key": "rak1:…", "top": [], "middle": ["2s","Ah"],
                  "bottom": [], "discards": ["3s"]},
  "actions": [{"rank": 1, "action_key": "rak1:…", "score": -0.793,
               "top": [], "middle": ["2s","Ah"], "bottom": [], "discards": ["3s"]}],
  "candidates_total": 12,
  "provenance": {"decide_evaluator": "learned", "ranking_source": "t4m1_labelgen_v1",
                 "ranking_units": "model_score_teacher_ev_proxy",
                 "model_slot": "t3_first", "model_sha256": "36334b…",
                 "exact": false, "fallback": null},
  "warnings": [],
  "timing": {"solve_ms": 22.4}
}
```

`served: false` means the position is legal but the pinned artifacts have no
path to it. `best_action` is then `null`, `provenance.reason` carries the
engine's own refusal string, and `warnings` names what is missing. This is
the honest-failure mode; it is not an error status, because the request was
fine.

`score` units are named by `ranking_units`, never assumed:

| value | meaning |
|---|---|
| `exact_hand_ev_points` | exact enumeration, hand EV in points |
| `hand_ev_points_vs_sampled_fl` | exact against N sampled Fantasyland boards, with `standard_error` |
| `model_score_teacher_ev_proxy` | learned model output; a monotone proxy for teacher EV, not a calibrated EV |
| `royalty_plus_fl_stay_ev` | FL placement objective: royalties plus the stay term |

---

## 3. `POST /v0/solve/normal`

A decision on a normal table, either seat, streets T0–T4.

Request:

```json
{
  "ruleset": {"id": "regular_hu_v1"},
  "street": "T1",
  "hero_index": 0,
  "players": [
    {"seat": "first",  "board": {"top": ["As"], "middle": ["Kd","Qc"],
                                 "bottom": ["2h","3h"]}, "private_discards": []},
    {"seat": "second", "board": {"top": ["7s"], "middle": ["8d","9c"],
                                 "bottom": ["Th","Jh"]}}
  ],
  "dealt": ["4c","5c","6c"],
  "options": {"max_actions": 10}
}
```

Card counts are validated against the regular decision geometry — hero /
opponent / dealt / hero-discards per (street, seat):

| street | first seat | second seat |
|---|---|---|
| T0 | 0 / 0 / 5 / 0 | 0 / 5 / 5 / 0 |
| T1 | 5 / 5 / 3 / 0 | 5 / 7 / 3 / 0 |
| T2 | 7 / 7 / 3 / 1 | 7 / 9 / 3 / 1 |
| T3 | 9 / 9 / 3 / 2 | 9 / 11 / 3 / 2 |
| T4 | 11 / 11 / 3 / 3 | 11 / 13 / 3 / 3 |

`422` on a geometry mismatch, an unknown card token, a duplicate card, three
players, or an opponent flagged `in_fantasyland` (that is the other
endpoint's job).

`best_action` comes from the engine's `decide`. `actions` comes from a
second pass that scores the whole legal fan; where no parity-checked encoder
exists for the (street, seat) pair the ranking is omitted and a
`ranking_unavailable` warning is attached — `best_action` is still served.

## 4. `POST /v0/solve/vs-fl`

Same request shape; the opponent carries `in_fantasyland: true` and an empty
board.

T4, T3 and T2 are solved exactly by the pinned `fl_solver_regular`: the
Fantasyland side is enumerated against `opponent_samples` sampled hands and
the hero's fan is scored against it, with `standard_error` per candidate and
a `decision_margin` on the root. `provenance.latency_class` distinguishes
`interactive` (T4) from `batch` (T3, T2) — see the latency table.

T1 and T0 have no vs-FL model. The endpoint attempts the intended
`fallback_normal_chain` and reports the result truthfully: on the currently
pinned engine image the fallback is refused, so the response carries
`served: false`, `fallback_status: "unavailable_on_pinned_engine"` and the
engine's verbatim reason. When an engine build that accepts the Fantasyland
observation schema lands, the same code path returns `served: true` with
`fallback: "fallback_normal_chain"` and a `model_pending` warning, with no
wire-format change.

## 5. `POST /v0/solve/fl-placement`

```json
{
  "ruleset": {"id": "regular_hu_v1"},
  "hero_index": 0,
  "hand": ["As","Ks","Qs","Js","Ts","9s","8h","8d","7c","6c","5c","4h","3h","2h"],
  "opponent_final_board": null,
  "options": {"threads": 1}
}
```

With `opponent_final_board: null` the answer is the exact static optimum over
the whole 14-card arrangement space — `mode: "static_optimum_v1"`, and the
response carries the per-row royalties, the stay flag and the objective
identity.

With an opponent board supplied, the correct answer is the frontier best
response. That function is compiled into the pinned `fl_solver_regular`
(`frontier::build_frontier` / `frontier::best_response`, exercised by the
`frontier-stats` command) but has no single-position CLI entry point, and the
crate is read-only in this workstream. The endpoint therefore returns the
exact static optimum with `mode: "static_optimum_v1_fallback"` and a
`best_response_unavailable` warning rather than a silently different answer.
Closing this needs one subcommand on the crate — hand plus opponent rows in,
`best_response` index out.

## 6. `GET /v0/health`

`{"schema", "status", "warm", "warm_seconds"}`. Cheap; suitable as a
container liveness probe.

## 7. `GET /v0/version` — the provenance surface

Everything a client or an auditor needs to pin an answer to the artifacts
that produced it:

```json
{
  "api_version": "v0",
  "engine_version": "ofc_hu_m3_engine/0.1.0",
  "engine_library_sha256": "b510bb38…",
  "feature_encoder_sha256": "82510e56…",
  "fl_solver_sha256": "e26eb514…",
  "models": {"t0_first": "07301034…", "t0_second": "e0ba6519…", "…": "…"},
  "fl_ev_config_path": "…/fl_ev_regular_v3_direct2.json",
  "fl_ev_config_sha256": "c7279f3f…",
  "fl_ev": {"14": 9.109},
  "fl_entry_cards": {"qq": 14, "kk": 14, "aa": 14, "trips": 14},
  "fl_stay_cards": 14,
  "ruleset": {"id": "regular_hu_v1", "deck_cards": 52, "include_jokers": false},
  "uptime_seconds": 14.32
}
```

The eleven model digests are the `weights_sha256` values the engine itself
verifies at load; they match the label-generation package ledger exactly, so
an answer served here and a label generated on the fleet are traceable to the
same weights.

---

## 8. What is real, and what is not

| decision point | best action | ranked fan | exact? |
|---|---|---|---|
| normal T0 first | T4M1 `t0first` model (numpy) | 232 candidates | no — learned |
| normal T0 second | engine `decide`, learned | 232 candidates | no — learned |
| normal T1 both | engine `decide`, learned | 27 candidates | no — learned |
| normal T2 both | engine `decide`, learned | 24 candidates | no — learned |
| normal T3 first | engine `decide`, learned | 12 candidates | no — learned |
| normal T3 second | engine `decide`, learned | **none** — no parity-checked encoder | no — learned |
| normal T4 both | engine exact enumeration | 6 candidates, exact EV | **yes** |
| vs-FL T4 | `fl_solver_regular label` | full fan + stderr | **yes**, to sampling error in N |
| vs-FL T3 | `fl_solver_regular label-t3` | full fan + stderr | **yes**, to N and M |
| vs-FL T2 | `fl_solver_regular label-t2` | full fan + stderr | **yes**, to N and the two draws |
| vs-FL T1 / T0 | **unserved** | — | — |
| FL placement, no opponent | `fl_solver_regular solve` | single | **yes** |
| FL placement, opponent known | static optimum, tagged | single | no — best response not exposed |

Two facts about the pinned engine image `b510bb38…` shape this table, and
both are build-staleness rather than design:

1. It rejects the config field `learned_t0_first_model_path`, which the
   Python dataclass and the engine source both have. T0 first seat is
   therefore served through the numpy T4M1 path, which is what the
   scratchpad probes already do. Cross-checked against every other street:
   the numpy path's rank-1 action is identical to the engine's `decide`
   argmax on all eight pairs where both run, so the two implementations
   agree where they overlap.
2. It rejects observations carrying Fantasyland flags ("fantasyland flags
   require the FL observation schema"). That is what blocks the vs-FL
   fallback at T1/T0, not a missing model alone.

Rebuilding the engine from the current source would likely close both. That
rebuild is not this workstream's to make.

---

## 9. Measured latency

Method: the server warm, one uvicorn worker, one request in flight at a
time, a fresh randomly-generated position per request so nothing caches.
`OMP_NUM_THREADS=1`, `RAYON_NUM_THREADS=1`, FL solver `--threads 1`. What
this measures is one request on one core, which is the number capacity
planning needs; concurrency would measure the box instead.

**Caveat, and it is not small:** the host was carrying an unrelated
eight-shard ladder match throughout (load average ~18 on 16 cores). Treat
every figure as a loaded-box upper bound. Spot checks on the same code
earlier in the session, at lower load, came in 30–40 % faster (T0 first:
242 ms vs the 383 ms below). The *ratios* between endpoints are the durable
part; the absolute numbers are pessimistic.

| endpoint | n | p50 ms | p95 ms | req/s/core | notes |
|---|---:|---:|---:|---:|---|
| `GET /v0/health` | 50 | 2.2 | 5.8 | 364 | |
| `GET /v0/version` | 50 | 2.3 | 7.9 | 305 | |
| `normal` T4 second | 50 | 4.1 | 10.1 | 198 | exact |
| `normal` T3 second | 50 | 9.1 | 14.3 | 95 | no ranking pass — that is why it is the fastest solve |
| `normal` T4 first | 50 | 22.1 | 47.3 | 40 | exact, 6 candidates |
| `fl-placement` | 50 | 22.9 | 50.9 | 39 | exact static optimum |
| `normal` T3 first | 50 | 31.2 | 59.8 | 31 | 12 candidates |
| `normal` T2 second | 50 | 72.4 | 143.4 | 12.2 | 24 candidates |
| `normal` T2 first | 50 | 82.5 | 123.5 | 11.4 | 24 candidates |
| `normal` T1 second | 50 | 256.4 | 337.1 | 3.8 | 27 candidates |
| `normal` T1 first | 50 | 315.3 | 400.5 | 3.1 | 27 candidates |
| `normal` T0 first | 50 | 382.8 | 469.4 | 2.5 | 232 candidates, numpy T4M1 |
| `normal` T0 second | 6 | 8515 | 9974 | 0.11 | 232 candidates × an 8-open-slot outlook |
| `vs-fl` T4 N=100 | 10 | 688 | 918 | 1.39 | exact |
| `vs-fl` T4 N=200 | 10 | 1379 | 1657 | 0.71 | exact |
| `vs-fl` T4 N=400 | 10 | 2707 | 2869 | 0.37 | exact |
| `vs-fl` T3 N=100 M=8 | 3 | 5281 | 5494 | 0.19 | exact |
| `vs-fl` T3 N=200 M=8 | 3 | 11077 | 11880 | 0.089 | exact |
| `vs-fl` T3 N=400 M=8 | 3 | 21572 | 21950 | 0.046 | exact |
| `vs-fl` T1 (unserved) | 10 | 6.0 | 31.5 | 99 | `served: false` |
| `vs-fl` T0 (unserved) | 10 | 31.0 | 83.9 | 31 | `served: false` |

### The N knob

vs-FL cost is linear in `opponent_samples`, and the linearity is clean
enough to price against:

| N | T4 p50 | ×N=100 | T3 (M=8) p50 | ×N=100 |
|---:|---:|---:|---:|---:|
| 100 | 0.69 s | 1.00 | 5.28 s | 1.00 |
| 200 | 1.38 s | 2.00 | 11.08 s | 2.10 |
| 400 | 2.71 s | 3.93 | 21.57 s | 4.08 |

N buys precision, and the response says how much: the T4 root's best
candidate reported `standard_error` 0.697 / 0.529 / 0.370 at N = 100 / 200 /
400 — the expected √N shape. A client that needs a decision rather than a
number can often stop at N=100 and read `decision_margin` against the
standard error.

M (`hero_deals`, the T3 hero-draw expectation) is nearly free by comparison:
M = 8, 16 and 32 all landed within noise of 22 s at N=400, because the cost
is dominated by building the N-hand Fantasyland pool once per root, not by
re-scanning it per deal.

### Where the time actually goes

Two structural facts, both worth designing around:

* **The ranking pass costs more than the decision.** At T3 first seat,
  `decide` alone is ~7.6 ms and the ranked fan is ~23 ms on top. At T0
  second seat, `decide` is ~1.3 s and the fan is ~7.5 s. `include_scores:
  false` is therefore a real 3×–7× lever, not a cosmetic one.
* **T0 second seat is the outlier by an order of magnitude.** 232 candidates
  each needing a free-slot outlook over an eight-open-slot board is simply
  a big computation. It is a job, not a request, unless the fan is pruned.
