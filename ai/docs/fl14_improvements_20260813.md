# FL14: what is wrong, what it costs, and what was measured

2026-08-13 (JST). Everything below has a number behind it. Items marked
**open** are not done; the rest record what was changed and how it was checked.

The one-lap result this list came out of:

| street | served by | regret | note |
|---|---|---|---|
| T4 | solver, 0.9 ms | 0.0018 | opponents sampled, nothing else |
| T3 | solver, 0.55 s | 0.0017 | the T4 draw is fully enumerated |
| T2 | sampled solver, 4.78 s | 0.064 | T3 draws, T4 draws and opponents all sampled |
| T1 | 96-dim net | 0.132 | bootstrap: the T2 net's value, no pool |
| T0 | 96-dim net | 0.068 | bootstrap: the T1 net's value, no pool |

"Exact" is only honest at T3 and T4. A fully enumerated T2 decision is 612
billion terminals — 45 hours a hand at 60 opponents, 137 at 240 — because the
T3 draw is *observed before the T3 choice*, so the expectation does not factor
and the pair table that makes T3 affordable has nothing to amortise over.

---

## A. Correctness

### A1. Half of every placement was missing — FIXED

`open_patterns` returned only the row pairs with `first <= second`, and its
callers put the first kept card in `pattern[0]`. So two cards going to two
different rows had two outcomes and one was built. A (2,4,3) board offers 21
T3 actions; the labeler enumerated 12.

Every FL14 value was a maximum over 57% of the legal moves. On the same twenty
deals with only the action set corrected, the best action is worth **+0.696**
and a root's spread goes 1.42 → 2.11.

Found because a key lookup missed on 37% of positions while measuring something
else. The number that lookup was about to produce would have been a mean over
the biased 63% that matched.

### A2. The T3 teacher had one board shape — REBUILT

`teach` dealt `cards[0..2]` to the top, `[2..6]` to the middle, `[6..9]` to the
bottom, so all 30,000 roots were (2,4,3). Playing the same fourteen cards
reaches fourteen shapes, of which (2,4,3) is 32%.

The shapes were the visible half. A dealt root's actions were worth 1.53 apart;
a played root's are worth 9.49. The old teacher spent 30,000 roots on decisions
that barely mattered, which is also why its gate read well.

Same test split, both models: trained on played roots **0.0905** regret
(p99 2.26), trained on dealt roots **0.5753** (p99 10.82). 6.4x, and the loss
lives in the tail.

### A3. The T2 teacher has one board shape — IN PROGRESS

`cards[0..2]/[2..5]/[5..7]`, so (2,3,2) for every root — **14.4%** of what play
reaches. The top two played shapes, (1,3,3) at 32.4% and (2,2,3) at 23.3%,
never appear.

### A4. The T1 teacher draws its shape uniformly — **open**

`sample_t1_root` randomises the row counts, so all 18 shapes appear equally.
Better than one shape, still wrong: the T0 model reaches **(1,2,2) 59.3%** of
the time and the teacher gives that shape 5.5% of its labels. The cards within
a row are also the first k of a shuffled deck, not a choice.

### A5. T0 and T1 labels are bootstrap — by design, worth restating

At `truncate_depth: 0` the playout stops at the chooser below and no eleven-card
board is ever priced: every label carries `mean_fl_samples: 0.0`. The
best-responding opponent reaches them only through what the street below
learned, and their regret measures fidelity to that street, not correctness.
Their floors are near zero for the same reason and mean nothing.

The alternative was measured: a full playout to an exactly-priced terminal cost
91 s a root against 3.5, at a floor of 2.03 — thirty times the noise of the
street below it. On four positions the two disagreed on every one, the
truncated shape preferring to build the top row where the exact one spread into
the middle and bottom. The exact reference at that budget does not reproduce
its own argmax between seeds, so that sizes the disagreement and does not
settle it.

---

## B. Speed

### B1. The row memo — DONE

`completion_value` cloned the base board and re-evaluated all three rows for
every one of ~33,000 completions a root, though only two cards move. Split by
`--own-only`: the Fantasyland frontier scan is **6-8%** of a T3 solve and
terminal evaluation is **92-94%**.

Thread-seconds a root: dealt 1.10 → 0.74, **played 1.63 → 0.23**. Labels
byte-identical across six configurations covering 0/1/2 jokers, all fourteen
played board shapes, `--own-only`, 240 opponents and `--stream-offset`.

Two things the port had to change from `t4_first_exact`'s version: reading
`has_joker` off the cached row evaluation made it **70% slower** than no memo
(it evaluates the top and middle before knowing it may use them), and the
second cache level cannot hit here because each (pair, placement) is visited
once.

### B2. `constrain_5_vs_5` is the remaining wall — **open, best lever**

Per root, one thread: no joker **0.27 s**, one joker in the middle **2.3 s**,
two jokers in the middle **42-71 s**. The memo cannot touch the joint path, and
one such root sets the wall time of any batch containing it.

The lever, named but not taken: `constrain_5_vs_5` and `constrain_3_vs_5` read
their reference row **only through `evaluate_hand_value`** — the value, never
the cards. So `mid_final = f(mid cards, bot value)` and
`top_final = g(top cards, mid_final value)` are exact decompositions, and for
actions that leave the bottom fixed they collapse 780 constrained solves into
40. That is a correctness claim beyond anything currently asserted and needs
the same treatment `available_subs`'s row-locality got: a test that pins it.

### B3. The card representation — **open, measure after B2**

`Card { rank: u8, suit: u8 }` in `Vec`s, converted between two identical types
per call: `hero_terminal` allocates four `Vec`s, and
`evaluate_natural_hand_value` allocates and sorts another per row. Roughly
230,000 heap allocations a root.

Rank nibbles in one u64 and per-suit 13-bit masks in another make flushes a
popcount, straights a shift-and-mask, and adding a card an add and an or —
allocation-free and incremental from the base board's mask. But `ofc_core` is
the foundation of both tracks and its joker substitution rules are pinned by
golden tests, so this needs byte-equality against all of them, and B2 may
remove most of what it would speed up.

### B4. `teach` writes everything at the end — **open**

It collects `Vec<(String, String)>` over all roots and writes once. `teach-t2`
was fixed to publish per root with a flush every ten; `teach` was not. This
shape has cost this project an hour of Spot work twice.

### B5. `--play-roots` exhausts 64 GB after 32 chunks — **open**

Any single chunk runs fine, so something accumulates across them. The smaller
chunk size used to get past it only moves the wall.

### B6. `fl_solver`'s `lib.rs` and `main.rs` are duplicate copies — **open**

Same `Card`, same helpers, and the modules compile into both. Low priority; it
is why the library export was done by declaring the modules twice rather than
by untangling them.

---

## C. How things are measured

### C1. Labels do not carry a standard error — **open**

The regular track's labels do, so its noise floor is computable without
relabelling anything: resample each candidate from `N(ev, se)` twice and charge
one draw's argmax on the other's means. Measuring the T1 floor here cost 35
minutes of relabelling instead.

### C2. Nothing measures the chain's absolute strength — **open**

No head-to-head, no exploitability, no benchmark against another bot. Every
number in this document is internal: regret against a teacher, or a teacher
against itself.

The one absolute-ish reading: playing the chain to T3 and then optimally,
hero is worth **-11.9** a hand against a Fantasyland-14 opponent, against
**-20.7** for the mechanical arrangement the old teacher used. That beats a
baseline nobody would play.

### C3. Estimates have been wrong repeatedly, in both directions

0.335 s a deal measured 0.44-0.82; 1,624 core-seconds a root measured 1,016;
2.11 s a root measured 9.63 under load; a ten-root probe read 0.98 s where a
hundred-root run read 2.11. Ten roots is not a measurement.

---

## D. Not yet ported from the regular track

Its `docs/fantasyland_opponent_method.md` gets a T0 label in **3.77 s** without
truncating anything, by two savings that multiply:

* **share the draw tree across actions** — the per-root fixed cost does not
  depend on the opening, and until it is gone, narrowing the fan scales with
  nothing (measured 1.42x, predicted 11.6x);
* **`--narrow-keep`** — 232 openings to 20 is 13.9x once the fixed cost is gone.

Plus C1 above.

---

## E. Out of scope so far

* Pools for widths 15, 16 and 17. Arrangement counts are 7.5x, 40x and 170x of
  14's, so the enumeration has to get smarter first.
* `fl_ev` is provisional (`source: user_direct_20260613`). The plan's
  closed-form fix for the circularity — `V* = (V0 - s*x0)/(1-s)`, and **do not
  iterate** — has not been applied to this track.
* A `t2_labels.rs` module doc indents prose that rustdoc reads as Rust, so
  `cargo test -p fl_solver` exits 101 on a doctest. Predates this work.

---

## Recommended order

**B2 → C1 → A4 → B4.**

B2 is 19x and needs no representation change, and one two-joker root currently
decides how long a batch takes. C1 is cheap and makes every measurement after
it cheaper. A4 is the last teacher still built on positions nobody reaches.
B4 is an hour of work that has already been lost twice.
