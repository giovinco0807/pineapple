# HU M9 — solver API product notes, v0

Scope: the owner's stated commercial intents, written down so the
engineering choices in `hu_m9_api_schema_v0.md` can be checked against them.
**Every number that touches money is marked TBD-owner.** Nothing in this
document has been decided, priced, agreed, or published, and nothing is
deployed.

---

## 1. Platform intent

The service runs on the owner's own platform — **Cloud Run** — rather than
inside a third-party marketplace. What that buys, and what the prototype
already does to suit it:

* **Scale-to-zero with a fast cold start.** Everything the service needs is
  baked into the image (8.4 MB of native artifacts and model weights, no
  download at boot). Warm-up measured locally is ~10 ms for the engine and
  model images; the bulk of a cold start is the Python interpreter and
  `ofc_regular` import, not the solver.
* **One worker per container.** The native engine and the pinned feature
  encoder are process-global, so a request holds a process lock. Capacity is
  a replica count, which is exactly the dial Cloud Run exposes. Do not
  configure concurrency > 1 per instance without re-measuring; the latency
  table is a single-core table.
* **One env var of configuration.** `OFC_FL_EV_CONFIG` selects the FL EV
  assumption. Everything else is baked and digest-recorded.
* **Non-root, read-only tree.** The image runs as uid 10001 and the served
  tree is chmod'd unwritable.

Open, owner-level: **region**, **min-instances** (the cold-start/idle-cost
trade), **max-instances** (the spend ceiling), **CPU allocation**
(request-scoped vs always-on — matters because the vs-FL endpoints are
CPU-bound for seconds). All TBD-owner.

## 2. Usage metering

Metering is implemented as a record shape, not a promise. `ofc_api/metering.py`
writes one JSON line per request:

```json
{"schema": "ofc_api_usage_v0", "timestamp": 1780000000.123,
 "key_id": "local-dev", "endpoint": "solve_vs_fl",
 "request_id": "uuid4", "latency_ms": 2668.1, "served": true}
```

Six fields. There is no field a board, a card, a solved line, or a client
identity beyond the key could be written into — **hand data is not retained
by default because the record has nowhere to put it**, which is a stronger
guarantee than a retention policy. `latency_ms` is included because the tier
axis is latency (§4), so billing and SLO share one source.

The sink is `OFC_API_METERING_LOG`; unset means metering is off, the right
default for a local prototype. On Cloud Run the natural sink is stdout into
Cloud Logging, with a log-based metric per `(key_id, endpoint)`.

`key_id` is populated from a context variable that the auth layer sets. That
layer does not exist yet — see §5.

Open, owner-level: **billing unit** (per request? per solved candidate? per
CPU-second, which is what vs-FL actually costs?), **whether unserved
responses are billable** (they cost ~1 ms and return no answer; the
prototype records `served: false` so either policy is implementable),
**aggregation window**, **invoice cadence**. All TBD-owner.

## 3. Terms of service — clauses to draft

Placeholders, not drafted text. A lawyer writes these; this list exists so
none is forgotten.

**RTA prohibition (placeholder).** The single most important clause. The
service returns solver-quality decisions in real time, which is exactly what
"real-time assistance" means in every online poker room's rules. Intended
shape:

> The Service must not be used to obtain assistance during play on any
> third-party poker platform where such assistance is prohibited by that
> platform's terms. The Customer is solely responsible for compliance with
> the rules of any venue in which it plays. [TBD-owner: enforcement posture
> — contractual only, or technical measures such as rate shaping, session
> pattern detection, or a training-mode/live-mode distinction.]

Supporting clauses to draft: permitted uses (study, review, training tools,
bot-detection research); prohibited uses (RTA as above, resale of raw solver
output, scraping to reconstruct the model); **no warranty of EV accuracy**
(the learned streets are proxies, not exact — the API already labels them,
and the terms should match that labelling); liability cap; data processing
terms consistent with §2's no-retention stance; acceptable-use suspension
rights; governing law and venue.

Open, owner-level: **jurisdiction**, **enforcement posture**, **whether a
training-only tier exists with different terms**. All TBD-owner.

## 4. Tier sketch — latency and rate as the axis

Price is not the axis; **what the customer can ask for, and how fast** is.
The measured latency spread across endpoints is roughly three orders of
magnitude (4 ms at T4 second seat, 8.5 s at T0 second seat, ~23 s for an
exact T3-vs-FL at N=400), so a flat "requests per month" plan would either
bankrupt the seller or starve the buyer.

A shape that follows the cost structure:

| tier | endpoints | vs-FL sample budget (N) | rate limit | latency target | price |
|---|---|---|---|---|---|
| Study | normal T2–T4, fl-placement | — | TBD-owner | best-effort | TBD-owner |
| Table | + normal T0/T1, vs-FL T4 | ≤ 200 | TBD-owner | p95 < 1 s on the fast streets | TBD-owner |
| Deep | + vs-FL T3/T2 (async) | ≤ 400 | TBD-owner | async job, minutes | TBD-owner |
| Enterprise | all, dedicated replicas | negotiated | negotiated | negotiated | TBD-owner |

Two structural points worth the owner's attention before any pricing:

1. **`opponent_samples` is the customer-visible cost dial.** It is already a
   request option, and vs-FL cost is close to linear in it (measured: N=100
   → 0.75 s, N=200 → 1.27 s, N=400 → 2.67 s at T4, single-threaded). Tiering
   on N is honest — the customer is buying precision, and `standard_error`
   comes back in the response so they can see what they bought.
2. **T0 second seat and vs-FL T3/T2 do not belong on a synchronous
   endpoint.** At 8.5 s and ~23 s of single-core time they are jobs, not
   requests. Either they get an async submit/poll pair, or they are priced
   as such, or the ranking pass is made optional for them (the fast path,
   `include_scores: false`, returns the chosen action in ~1.3 s at T0
   second).

Open, owner-level: **all prices**, **free-tier existence and size**,
**overage behaviour** (throttle vs bill), **annual vs monthly**. All
TBD-owner.

## 5. Decisions that need the owner

Listed, not decided:

1. **Pricing tiers** — the numbers in every cell of §4.
2. **Auth provider** — API keys issued in-house, or an identity platform
   (Firebase Auth / Auth0 / Cloud Endpoints + API Gateway)? This decides
   what fills `key_id` and where quota state lives. The prototype has no
   auth at all; the hook is a context variable.
3. **Domain and TLS** — the public hostname, and whether Cloud Run's managed
   domain mapping is enough.
4. **Payment** — processor (Stripe is the obvious default), whether metering
   drives usage-based invoicing or plans are prepaid, tax handling.
5. **RTA enforcement posture** — contractual only, or technical measures
   (§3). This one has product consequences, not just legal ones: technical
   enforcement would change the request shape (session identity, pacing).
6. **Async job surface** — whether the Deep tier gets submit/poll endpoints,
   which is a schema change and should be decided before v0 is published to
   any customer.
7. **Support and SLA** — whether any tier carries an availability
   commitment, and what credit applies.

## 6. Engineering prerequisites before any of this is sellable

Not owner decisions — flagged so the commercial plan is not built on sand:

* The pinned engine image predates two features its own source has: the
  T0-first learned model config field, and the Fantasyland observation
  schema. That is why vs-FL T1/T0 return `served: false`. A rebuild of the
  engine is the single highest-value unblock.
* vs-FL T3-second has no parity-checked candidate encoder, so that street
  serves a best action but no ranked fan.
* The FL frontier best response is compiled into the FL solver but has no
  CLI entry point, so `fl-placement` against a known opponent board falls
  back to the static optimum. One subcommand closes it.
* No auth, no quota, no rate limiting exists. The prototype binds
  `127.0.0.1` only.
