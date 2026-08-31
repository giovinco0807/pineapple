# Fantasyland for the joker rules, built the way the regular track's was

2026-08-11. A plan, not a result. Companion to the regular track's
[`fantasyland_opponent_method.md`](../../regular-ofc-pineapple/docs/fantasyland_opponent_method.md),
which records the method this proposes to port.

The regular track's Fantasyland work turned a T0 label from 130.5 seconds into
3.77 — 34.6x — and the speed was the smaller half of it. The larger half was
that the Fantasyland side stopped being sampled and became **exact**, which is
why its labels are quiet enough to have a yardstick at all while the normal
table's teacher still gives five different best openings across five seeds.

This is about porting that to the joker rules, starting at FL 14 cards.

---

## 1. What the method actually is, and which parts are about rules

Three pieces. They are independent, and they transfer independently.

**(a) The Fantasyland side is a best response, not a static solve.** The FL
player sets 13 of their cards *after* seeing the opponent's finished board. So
their arrangement is a function of that board. Scoring a hero candidate against
a statically-solved FL board — one that maximised its own royalty in advance —
is optimistic for hero: it lets hero beat an opponent who was not allowed to
react.

**(b) The frontier makes that best response exact and cheap.** Write the FL
player's score against a fixed hero board `H`:

```text
f(A, H) = g(s_top, s_mid, s_bot) + static(A)
  where s_row     = sign(row_key(A) - row_key(H))
        g         = the 1-6 line term including the scoop
        static(A) = royalty(A) + fl_ev * stays(A)
```

`g` is monotone non-decreasing in each of its three arguments, and each `s_row`
is monotone in `row_key(A)`. So `f(., H)` is monotone in all four of
`(top_key, mid_key, bot_key, static)` **for every `H` at once**, and a
dominated arrangement can never be the unique best response. The maximum is
always on the non-dominated frontier; scanning it is exact. In the regular deck
that turns 1,009,008 arrangements into ~57 rows.

**(c) A pre-solved pool is exact by conditioning.** A uniform 14-card hand from
the deck, conditioned on sharing no card with hero's seventeen, is distributed
exactly as a uniform 14-card hand from what hero left. That is the definition of
conditioning, not an approximation, so hands can be solved once and drawn from
by mask.

**Which of these depend on the rules?** (a) is a rule and holds in both games.
(b) depends only on the *shape* of the scoring function — three row comparisons
plus a hero-independent term — which the joker rules share. (c) depends only on
uniform dealing. **All three transfer. None of them is about jokers.**

---

## 2. The gap

`ai/rust_solver/fl_solver` solves Fantasyland, and solves it well: three
generations of `solve_fantasyland`, parallel over the C(n,5) bottom choices,
with a joker-aware evaluator behind a 4M-entry cache.

It is a **static** solve. `solve_fantasyland(cards: &[Card])` takes the FL
player's cards and nothing else — no hero board, no best response. Grepping the
crate for `opponent`, `hero`, `best_response` or `frontier` finds only incidental
matches.

So the joker track today is where the regular track was before the frontier:
every number computed against a Fantasyland opponent is optimistic for hero by
an unmeasured amount, and every one of them is a sampled quantity where an exact
one is available.

**This is the thing to build.** Not a faster solver — a different question asked
of it.

---

## 3. What the joker rules actually change

Verified against the code rather than assumed.

**The deck is 54.** `create_deck(true)` pushes 52 cards plus two jokers, both
`Card { rank: 0, suit: 4 }`. This changes the pool arithmetic and nothing about
the argument: a hand fits hero's seen cards with probability C(54-k, 14)/C(54, 14)
for whatever k hero holds, and the mask test is still a bitwise AND — though a
54-card deck no longer fits a `u64` mask the way 52 did if two jokers must be
distinguishable. **They are identical cards**, so the mask needs a joker *count*
beside it rather than two more bits.

**Fantasyland is progressive.** `check_fl_entry` returns a card count, not a
bool: QQ enters at 14, KK at 15, AA at 16, trips at 17. The regular track has a
single entry width and therefore a single `fl_ev` constant (9.6). The joker
track's `ai/config/fl_ev.json` carries four — `{14: 0, 15: 10.7, 16: 29.9,
17: 63.5}` in `direct` mode — plus a separate `fl_ev_first_hand` table.

*(Note: the project memory records `reward_mode: "chain"` with a different
table. The file on disk says `direct`. The memory is stale; the file wins.)*

**The stay rule has the same shape.** `check_fl_stay` is trips on top, or
quads/straight-flush/royal-flush on the bottom. The middle is not a stay
condition, exactly as in the regular track.

**Jokers multiply the arrangements, and the existing code already collapses
that.** A joker is substituted by the best available card — `available_subs`
excludes ranks/suits already present, `evaluate_hand_value_with_jokers` tries
every substitution (every pair, for two jokers) and keeps the best. Crucially,
`evaluate_board_with_joker_constraint` then walks *bottom to top*: the bottom
takes its joker at full strength, the middle is constrained to `<= bottom` by
`constrain_5_vs_5`, and the top to `<= middle` by `constrain_3_vs_5`.

That greedy looks exact, and the argument is short enough to state: raising a
row's key raises both its royalty (royalty is monotone in category, category is
monotone in key) and its chance of beating hero's row, so within a row the
maximum key dominates; and lowering a row can only tighten the constraint on the
rows above it, never loosen it, so taking the maximum from the bottom up is
weakly optimal at every step.

**That is an argument, and this project does not ship arguments.** It ships them
pinned against brute force — the regular frontier is checked against all
1,009,008 arrangements on 344 (deal, hero board) pairs. The joker version needs
the same fixture, enumerating joker substitutions explicitly, before anything is
built on top of it.

---

## 4. Order of work

Each step says what would falsify it. A step that cannot fail is not a step.

### Step 1 — pin the joker evaluator against brute force

Enumerate every joker substitution explicitly for random 14-card hands with 0, 1
and 2 jokers, take the true maximum over arrangements, and compare with what
`evaluate_board_with_joker_constraint` returns.

*Falsified if:* the bottom-up greedy ever misses a legal arrangement that beats
the one it found. Most likely place for it to break is two jokers in different
rows, where the constraint chain binds twice.

Cheap, and everything below depends on it.

### Step 2 — the best-response frontier for joker hands

Port `build_frontier` and `best_response`: reduce a 14-card FL hand to its
non-dominated `(top_key, mid_key, bot_key, static)` rows, where `static` uses the
joker track's `fl_ev` for the width the hand would re-enter at.

*Falsified if:* the frontier's argmax score ever differs from the brute-force
argmax over all arrangements, for any hero board. Same fixture discipline as the
regular track: bit-identical or it does not ship.

*The number to watch:* the frontier's **row count**. The regular deck averages
57.2 rows. Jokers can only add reachable key tuples, so this will be larger, and
how much larger decides whether the pool is affordable. **Measure it; do not
project it.** Every projection made on the regular side was wrong, in both
directions, by factors of 2 to 8.

### Step 3 — measure the two costs that decide everything else

Per 14-card hand: the solve, and the frontier build. The regular numbers are
5.18 ms and 38.3 ms. The joker versions will be worse — by how much is the whole
question, because the pool's size and the label cost follow from it directly.

*Decision point:* if the frontier build is under ~200 ms, a 200,000-hand pool
costs a few hours on twelve threads and the regular plan ports as written. If it
is seconds, the pool has to shrink or the frontier has to be cheaper before
anything else is worth doing.

### Step 4 — the pool

Same format discipline as `FLL1`: flat little-endian, magic, version, and the
`fl_ev` constant checked on load, because a pool built under a different
constant prices its stay term differently and is a different pool. Two things
the regular version learned the hard way and should be built in from the start:

* **The stream index is the entry's position in the pool**, not its position in
  the call, so the pool does not change with the thread count.
* **A short draw is an error, not a shrug.** A leaf that quietly averages over
  three opponents when the plan said sixteen is a silent failure.

*Falsified if:* the pool's answers differ from solving at every leaf. The
regular check was four runs — both shapes under two seeds — comparing the chosen
action and each shape's own seed-to-seed noise. Reuse it.

### Step 5 — only then, the teachers

FL14 first, as asked. What the regular track learned about *which axis to buy*
should be re-measured rather than assumed, because the joker deck is a different
distribution — but the shape of the finding is likely to repeat: the Fantasyland
side contributes no variance once it is solved exactly, so opponent sample count
stops mattering and only hero's own draws do.

---

## 5. Rules of engagement, carried over

These are not style preferences. Each one cost hours or dollars on the regular
track.

**Measure, do not extrapolate.** Predictions that were wrong there: narrowing
11.6x → 1.42x; the pool 3.1x → 2.3x; T3's curve exponent −0.53 → −0.27; T0's
−0.56 → −0.18; a "saturated" curve that was an artefact of index-ordered data.
The one time a projection held (11.6x for a model-based fan cut) it was
projected from three measured points of the same system.

**Pin every model and every pool by digest**, and carry a parity fixture that
proves the consumer reproduces the producer. The regular `.vfl1` images ship
with 256 real rows and their scores so the Rust loader can be checked against
the Python fit before anything is labelled.

**Borrow, do not clone.** The regular T0 teacher materialises its whole draw
tree before scoring any action; cloning frontiers out of the pool cost 32 GB and
the kernel killed the process with no message at all. `Cow::Borrowed` fixed it:
714 MB, and 6% faster. A worker VM has 30 GB, so this is a live failure mode,
not a tidiness point.

**Narrowing changes what a label means** — from "the best action" to "the best
of the twenty the model liked" — so any narrowed run carries an audit slice at
full width. The cost is a few percent and it is the only way to notice the next
model inheriting this one's blind spots.

**Do not read partial results.** A mid-run average over positions that have not
finished their repeats flipped a mean from +0.837 to −0.078 today, purely by
admitting eleven positions nobody had measured yet.

---

## 6. The two rules that were open, and what they decide

Both settled by the owner, 2026-08-11.

**A stay inherits the width you entered at.** A 17-card Fantasyland that stays
continues at 17, not at 14. So the stay term is not one constant but a function
of the hand being solved:

```text
static(A) = royalty(A) + fl_ev[width] * stays(A)
```

where `width` is the width of *this* Fantasyland, fixed for the whole hand. It
is a constant within a solve, which is what the frontier needs — `static` must
not depend on hero's board, and it does not.

**The table is `fl_ev_direct`**: `{14: 0, 15: 10.7, 16: 29.9, 17: 63.5}`, and
it is provisional — the values are expected to move.

### Two consequences worth stating before anything is built

**`fl_ev` is a surplus, and at 14 it may be negative.** The number is what
Fantasyland is worth *over playing a normal hand*, and with jokers a normal hand
is strong — jokers pay large royalties on an ordinary board. So the narrowest
Fantasyland is worth about what a normal hand is worth, and the owner's reading
is that it could come out below it.

At exactly 0 the stay term simply vanishes and `static(A) = royalty(A)`. Below
0 something more interesting happens, and it breaks a piece of the existing
implementation.

### A negative surplus breaks the joker substitution, and only that

The frontier's exactness argument survives untouched. It needs `static(A)` to be
a number that does not depend on hero's board, and dominance is taken on its
*value*; nothing in the proof cares whether a stay adds or subtracts. Every
arrangement still carries a well-defined `(top_key, mid_key, bot_key, static)`
and the maximum is still on the non-dominated frontier.

What breaks is the collapse. `evaluate_board_with_joker_constraint` picks **one**
substitution per row — the one maximising that row's hand value — and that is
justified by an argument that a negative `fl_ev` destroys:

> raising a row's key raises both its royalty and its chance of beating hero's
> row, so within a row the maximum key dominates

With `fl_ev < 0` a higher key can *lower* `static`. Making the top row trips
raises `top_key` and its royalty but triggers `stays` and subtracts `|fl_ev|`.
The maximum-key substitution can therefore be the wrong one, and worse, it is
the **only** one the current code considers — so the better arrangement is never
built and the frontier never sees it.

Note this is a joker-only defect. Without jokers, "a weaker top that avoids
trips" is a different arrangement, enumerated on its own, and the sweep keeps
whichever of the two is not dominated. The collapse is what loses it.

**The fix is small and falls out of a useful observation:** `stays` is a
function of the category, the category is a function of the key, and each row's
royalty is a function of its key. So `static` is entirely determined by the three
row keys. For a row containing jokers, then, the substitution choice is a choice
among *achievable row keys*, and what must be kept is the Pareto set over
`(key, static contribution)` rather than the single maximum. That set is small —
typically the maximum key, plus the best key that does not trip the stay
condition.

There is a corollary worth recording. Because `static` is a function of the three
keys, the regular track could nearly have dropped its fourth dominance component:
with `fl_ev > 0`, `static` is monotone in the keys and dominance on the three
keys implies dominance on `static`. **With `fl_ev < 0` it is not monotone, and
the fourth component becomes load-bearing.** Anyone porting the sweep and
"simplifying" it to three components would produce something that is exact on the
regular deck and quietly wrong here.

### The table has no derivation, and does not agree with its own file

`ai/config/fl_ev.json` records its own provenance: `"source":
"user_direct_20260613"`. The four numbers were entered by hand. The owner
confirms there is no derivation behind them and that measuring them properly
needs a strong model first.

That is true, and it is not the end of what can be said today, because the same
file carries the statistics the numbers should have come from. If a stay
inherits its width, the chain value of a width-`N` Fantasyland is
`V(N) = (value of this hand) / (1 - stay_rate(N))`. Putting the file's own
`fl_ev_first_hand` and `fl_stats` through that:

| width | stay rate | `V(N)` | surplus over 14 | **the table** | ratio |
| --- | --- | --- | --- | --- | --- |
| 14 | 0.351 | 21.57 | 0 | **0** | — |
| 15 | 0.502 | 43.57 | 22.0 | **10.7** | 0.49 |
| 16 | 0.645 | 67.89 | 46.3 | **29.9** | 0.65 |
| 17 | 0.781 | 121.92 | 100.4 | **63.5** | 0.63 |

Using `fl_stats[N].R` in place of `fl_ev_first_hand` gives the same shape with
ratios of 0.62-0.71. Either way **the table is around two thirds of what the
rest of the file implies, and the shortfall is not a constant factor.** Which
side is wrong is not determinable from here; that the two do not describe the
same world is.

### The circularity is solvable, and the regular track solved it

"The model needs a constant, and the constant needs a model" is exactly how the
regular track's 9.6 was obtained. Its provenance records the method:

1. Run self-play with the **current** constant `x0` in force — Fantasyland
   actually played out on both sides, not assumed.
2. Measure the realised entry-episode value `V0` (there: 9.669, n = 52,303).
3. `V(x)` is measured-linear with a slope `s` taken from a separate anchor
   sweep, so the self-consistent constant is the crossing, in closed form:
   `V* = (V0 - s*x0) / (1 - s)`.
4. **Do not iterate.** The map's slope is negative, so feeding the answer back
   in oscillates rather than converges — recorded there as "the June failure".

One self-play run and one closed-form solve. The joker version has five
unknowns instead of one — four widths plus normal play — but it stays linear,
and because a stay inherits its width each width's chain is self-contained, so
the system is nearly diagonal. `fl_stats` already holds the per-width inputs;
what a stronger model buys is the right to trust them.

### None of this blocks the work

**The frontier and the pool do not need the constant to be right. They need it
pinned.**

Correctness here — exactness by dominance, distributional exactness by
conditioning — does not depend on the value of `fl_ev`. Only the *labels* built
on top of it do. So:

* Build under `{14: 0, 15: 10.7, 16: 29.9, 17: 63.5}` now.
* Stamp **all four values** into the pool header and refuse to load under any
  other, so that a re-derived table later makes the pool a **re-run** rather
  than a rebuild.
* And note that **FL14 is the width least exposed to the unknown.** At
  `fl_ev[14] = 0` the stay term leaves the expression entirely and the 14-card
  frontier does not depend on the constant at all.

Which is a second reason the owner's instinct to start at 14 is the right one:
it is the width furthest from the thing nobody has measured yet. If the constant
later resolves to a small negative number, the 14-card work does not have to be
redone — only re-run.

**A pool is per width, and per table.** Two things follow:

* A 14-card pool and a 17-card pool are different objects — different hand
  sizes, different arrangement counts, different frontiers. Four pools, not one.
* Because the values are provisional, the pool must pin **the whole table it was
  built under**, not just one constant. The regular `FLL1` header carries a
  single `fl_ev` and refuses to load under a different one; the joker version
  carries four and refuses on any mismatch. A pool built under a superseded
  table is not a stale pool, it is a wrong one, and nothing downstream would
  notice.

### Why FL14 first is not just an arbitrary starting point

The arrangement count is where the widths part company. Choosing 5 bottom,
5 middle and 3 top out of `n`:

| width | arrangements | vs 14 cards |
| --- | --- | --- |
| 14 | 1,009,008 | 1x |
| 15 | 7,567,560 | 7.5x |
| 16 | 40,360,320 | 40x |
| 17 | 171,530,160 | **170x** |

The regular deck's 14-card frontier build is 38.3 ms over 1,009,008
arrangements. At the same rate 17 cards is about 6.5 seconds *before* jokers
multiply it — and a pool of any useful size at that rate is days, not hours.

So 14 is the width where the method can be proved cheaply, and 15-17 are where
it will need the enumeration itself to get smarter — pruning bottom rows that
cannot appear on any frontier, rather than building every arrangement and
sweeping afterwards. **Do not design that until 14 is measured.** The regular
track's costs came in at 1/8 to 8x of every prediction made about them.
