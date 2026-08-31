//! Exact T4-first (the seat acting first on the last street), both boards
//! visible: the ground floor of the HU ladder.
//!
//! From the first actor's information set everything relevant is known -- own
//! eleven cards, own three discards, the opponent's eleven visible cards --
//! and the two unknowns are draws.  Exchangeability makes both exact: the
//! opponent's three dead cards are a uniform three-subset of hero's 29 unseen,
//! which collapses their draw to a uniform three-subset of whatever hero has
//! not seen (the standard restart-belief result, here it is not an
//! approximation because hero's own hidden cards are all accounted).
//!
//! The value is
//!
//! ```text
//!   V = E[own draw T] max[own placement of 2 of T] E[opp draw S]
//!         min[opp placement]  score(own 13, opp 13)
//! ```
//!
//! where score is the symmetric zero-sum settlement: lines with scoop,
//! royalty difference, plus own Fantasyland entry credit minus theirs.  The
//! opponent minimises hero's score because the game is zero-sum in that
//! definition.
//!
//! # Why this is tens of milliseconds and not tens of seconds
//!
//! Enumerated naively the tree is C(29,3) x ~9 x C(26,3) x ~9 terminals.  Three
//! amortisations collapse it:
//!
//! * hero's distinct final boards are pairs-times-patterns, about eight
//!   hundred, not draws-times-placements (~33,000): a draw only chooses among
//!   boards the pair table already priced;
//! * the opponent's completions are priced once against the shared 29-card
//!   pool; per hero candidate they are only *composed* (three u32 compares
//!   and adds), never re-evaluated;
//! * hero's discard shrinks the opponent pool by one card, which would make
//!   every (board, discard) pair its own sweep; instead each candidate's
//!   sweep records, per card, the summed value of the draws containing it,
//!   and the discard-conditional mean is one subtraction.

use anyhow::{anyhow, bail, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::discard_model::DiscardModel;
use super::row_memo::TerminalMemo;
use super::{all_cards, to_core_card, BoardStr, Card, CoreBoard, FlEv, Terminal};

/// Whose hidden discards the pool contains, i.e. which visible board the
/// discard model should condition on.
#[derive(Clone, Copy)]
pub enum DeadSide {
    /// The `board` owner's discards are hidden (the swapped T3-second call).
    Board,
    /// The `opp_board` owner's discards are hidden (the direct call).
    OppBoard,
}

#[derive(Deserialize)]
pub struct V4FirstRequest {
    pub id: String,
    /// Hero's eleven placed cards.
    pub board: BoardStr,
    /// Hero's three prior discards (private, known to hero).
    pub dead: Vec<String>,
    /// Hero's T4 draw.  Present: per-action values for these three cards.
    /// Absent: the street-start state value, the mean over all draws.
    #[serde(default)]
    pub draw: Option<Vec<String>>,
    /// Opponent's eleven visible cards.
    pub opp_board: BoardStr,
    /// The opponent's three discards, which no seat can actually see.  For
    /// the omniscient audit only: with them the pool is the physical 26-card
    /// deck and the value is the true-world one, and comparing it against the
    /// belief value on traced hands measures what the exchangeable treatment
    /// costs.  Labels must not use it.
    #[serde(default)]
    pub opp_dead: Vec<String>,
}

#[derive(Serialize)]
pub struct V4Action {
    /// "c1@row,c2@row|discard"
    pub key: String,
    pub value: f64,
}

#[derive(Serialize)]
pub struct V4FirstResponse {
    pub id: String,
    pub schema: &'static str,
    /// Mean over draws (state mode) or over the given draw's best action
    /// (decision mode, `best` of `actions`).
    pub value: f64,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub actions: Vec<V4Action>,
    pub pool: usize,
    pub candidates: usize,
}

/// Hero's score against one concrete pair of finished boards.
#[inline(always)]
pub(crate) fn compose(mine: &Terminal, theirs: &Terminal, fl_ev: &[f64; 4]) -> f64 {
    let entry = |t: &Terminal| -> f64 {
        if !t.busted && t.fl_card_count >= 14 {
            fl_ev[(t.fl_card_count - 14) as usize]
        } else {
            0.0
        }
    };
    match (mine.busted, theirs.busted) {
        (true, true) => 0.0,
        (true, false) => -(6.0 + theirs.royalty as f64) - entry(theirs),
        (false, true) => 6.0 + mine.royalty as f64 + entry(mine),
        (false, false) => {
            let sign = |a: u32, b: u32| -> i32 {
                match a.cmp(&b) {
                    std::cmp::Ordering::Greater => 1,
                    std::cmp::Ordering::Less => -1,
                    std::cmp::Ordering::Equal => 0,
                }
            };
            let lines = fl_solver::frontier::scoop_aware_line(
                sign(mine.values[0], theirs.values[0]),
                sign(mine.values[1], theirs.values[1]),
                sign(mine.values[2], theirs.values[2]),
            );
            lines as f64 + (mine.royalty - theirs.royalty) as f64 + entry(mine) - entry(theirs)
        }
    }
}

/// Two-card placements of a board with exactly two open slots, as **ordered**
/// row pairs: `pattern[0]` takes the first card of the pair, `pattern[1]` the
/// second.  Both orders of a two-row split are distinct legal actions and both
/// are enumerated; a same-row pattern `[r, r]` comes out once because the two
/// loops meet there only once.
///
/// It used to read `for b in a..3`, which is the unordered pair set.  Every
/// caller here feeds it `(pool[i], pool[j])` with `i < j` fixed, so the
/// unordered form built only one of the two assignments whenever the open
/// slots sat in different rows -- half the legal actions, silently.  That is
/// the same defect class as d3cca6c ("Half of every placement was missing"),
/// and it made hero's max a max over half the moves and the opponent's min a
/// min over half of theirs.  Matches `ai/engine/action_space.get_turn_actions`,
/// which loops `for pos0 in POSITIONS { for pos1 in POSITIONS }`.
pub(crate) fn placement_patterns(board: &CoreBoard) -> Vec<[usize; 2]> {
    let open = board.open_slots();
    let mut out = Vec::new();
    for a in 0..3usize {
        for b in 0..3usize {
            let mut need = [0usize; 3];
            need[a] += 1;
            need[b] += 1;
            if (0..3).all(|r| need[r] <= open[r]) {
                out.push([a, b]);
            }
        }
    }
    out
}

pub fn solve(request: &V4FirstRequest, fl_ev_table: &[f64; 4]) -> Result<V4FirstResponse> {
    solve_weighted(request, fl_ev_table, None)
}

/// The 29-card unseen pool of a V4 request (26 in the omniscient audit),
/// with the same validation and the same ordering `solve_weighted` builds
/// inline.  Written twice because `solve_weighted` is frozen byte-for-byte
/// while the collapse needs the pool before deciding whether to call it;
/// `rank_collapse::tests::collapse_pool_matches_the_solver_pool` pins the two
/// together, and the real-root gate would show any drift as a value gap.
pub(crate) fn unseen_pool(request: &V4FirstRequest) -> Result<Vec<Card>> {
    let my_board = CoreBoard::from_str_board(&request.board)?;
    let opp_board = CoreBoard::from_str_board(&request.opp_board)?;
    if my_board.card_count() != 11 || opp_board.card_count() != 11 {
        bail!("{}: V4 wants two eleven-card boards", request.id);
    }
    if request.dead.len() != 3 {
        bail!("{}: hero has three discards at T4", request.id);
    }
    if !(request.opp_dead.is_empty() || request.opp_dead.len() == 3) {
        bail!("{}: opp_dead is empty (belief) or three cards (audit)", request.id);
    }
    let mut seen_naturals: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let mut seen_jokers = 0usize;
    for name in request
        .dead
        .iter()
        .chain(&request.opp_dead)
        .chain(&request.board.top)
        .chain(&request.board.middle)
        .chain(&request.board.bottom)
        .chain(&request.opp_board.top)
        .chain(&request.opp_board.middle)
        .chain(&request.opp_board.bottom)
    {
        if name.starts_with('X') {
            seen_jokers += 1;
        } else if !seen_naturals.insert(name.clone()) {
            bail!("{}: card {name} appears twice across the boards", request.id);
        }
    }
    if seen_jokers > 2 {
        bail!("{}: {} jokers visible, the deck has two", request.id, seen_jokers);
    }
    let mut pool_names: Vec<String> = all_cards()
        .into_iter()
        .filter(|n| !n.starts_with('X') && !seen_naturals.contains(n))
        .collect();
    for slot in 0..(2 - seen_jokers) {
        pool_names.push(format!("X{}", slot + 1));
    }
    let pool: Vec<Card> = pool_names
        .iter()
        .map(|n| to_core_card(n))
        .collect::<Result<Vec<_>>>()?;
    let wanted = 29 - request.opp_dead.len();
    if pool.len() != wanted {
        bail!("{}: pool is {} cards, wanted {wanted}", request.id, pool.len());
    }
    Ok(pool)
}

/// The V4 entry the HU teachers call.  When `enable` is set **and** the
/// request is one the collapse is licensed for, the state sweep runs over
/// `(rank, live-suit bucket)` classes; otherwise this is `solve_weighted`
/// verbatim.
///
/// The licence is narrow on purpose:
///
/// * **state mode only** -- a present `draw` is the decision mode, whose
///   output is per-card action keys the collapse has no way to name;
/// * **uniform pool only** -- a discard model weights each card separately,
///   and per-card weights are exactly what a class throws away;
/// * **at most one live flush suit across both boards** -- hero's candidates
///   and the opponent's completions share one pool, so the alphabet has to
///   distinguish every suit either side could still use, and past one suit
///   what is left to merge no longer pays for the merging.  The bound is
///   `rank_collapse::FIRE_MAX_LIVE` and it is a speed judgement, not a
///   correctness one.
///
/// Everything else, including every serve path and every encoder, keeps the
/// old route by construction: this function is only reachable from the three
/// modes that pass the flag.
pub fn solve_maybe_collapsed(
    request: &V4FirstRequest,
    fl_ev_table: &[f64; 4],
    discard: Option<(&DiscardModel, DeadSide)>,
    enable: bool,
) -> Result<V4FirstResponse> {
    super::rank_collapse::note_call();
    if !enable || request.draw.is_some() || discard.is_some() {
        return solve_weighted(request, fl_ev_table, discard);
    }
    // A request that does not parse falls through so that `solve_weighted`
    // raises the canonical error rather than this path inventing one.
    let eligible = (|| {
        let my_board = CoreBoard::from_str_board(&request.board).ok()?;
        let opp_board = CoreBoard::from_str_board(&request.opp_board).ok()?;
        // The union of the two boards' live flush suits: hero and the
        // opponent eat the same pool, so a suit either of them can still use
        // has to stand as its own class for both.  Counted before the
        // threshold decides, so the histogram describes all the traffic.
        let live_union = super::rank_collapse::board_live_mask(&my_board)
            | super::rank_collapse::board_live_mask(&opp_board);
        super::rank_collapse::note_live(live_union);
        // Past the threshold the collapse would still be exact and would
        // still be slower: at two live suits the class alphabet shrinks the
        // triple sweep by 1.58x, less than a collapsed term costs over a card
        // term, and the wall clock comes out at 0.49x.  See FIRE_MAX_LIVE.
        if live_union.count_ones() > super::rank_collapse::FIRE_MAX_LIVE {
            return None;
        }
        let pool = unseen_pool(request).ok()?;
        Some((my_board, opp_board, pool, live_union))
    })();
    let Some((my_board, opp_board, pool, live_union)) = eligible else {
        return solve_weighted(request, fl_ev_table, discard);
    };
    super::rank_collapse::note_fired();
    let collapsed = super::rank_collapse::collapse_state(
        fl_ev_table,
        &my_board,
        &opp_board,
        &pool,
        live_union,
        false,
    );
    Ok(V4FirstResponse {
        id: request.id.clone(),
        schema: "ofc_hu_v4_first/v1",
        value: collapsed.value,
        actions: Vec::new(),
        pool: pool.len(),
        candidates: collapsed.candidates,
    })
}

/// `discard`: weight the unseen pool by a trained model's per-card discard
/// probability instead of treating it as exchangeable.  A drawn card must
/// have survived the hidden discards, so draw triples are weighted by the
/// product of their survival probabilities (independence across the three,
/// the model's own conditioning within each).  `None` is the uniform pool,
/// bit-identical to the unweighted implementation.
pub fn solve_weighted(
    request: &V4FirstRequest,
    fl_ev_table: &[f64; 4],
    discard: Option<(&DiscardModel, DeadSide)>,
) -> Result<V4FirstResponse> {
    let my_board = CoreBoard::from_str_board(&request.board)?;
    let opp_board = CoreBoard::from_str_board(&request.opp_board)?;
    if my_board.card_count() != 11 || opp_board.card_count() != 11 {
        bail!("{}: V4 wants two eleven-card boards", request.id);
    }
    if request.dead.len() != 3 {
        bail!("{}: hero has three discards at T4", request.id);
    }
    if !(request.opp_dead.is_empty() || request.opp_dead.len() == 3) {
        bail!("{}: opp_dead is empty (belief) or three cards (audit)", request.id);
    }

    // The 29-card pool: everything hero has not seen.  The opponent's dead
    // cards are inside it, and exchangeability makes that exact.
    //
    // Jokers are counted, not name-matched: each seat names its own first
    // joker X1, so a hand where both seats hold one produces two distinct
    // physical jokers under one name, and a name set would collapse them and
    // leave a phantom joker in the pool.
    let mut seen_naturals: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let mut seen_jokers = 0usize;
    for name in request
        .dead
        .iter()
        .chain(&request.opp_dead)
        .chain(&request.board.top)
        .chain(&request.board.middle)
        .chain(&request.board.bottom)
        .chain(&request.opp_board.top)
        .chain(&request.opp_board.middle)
        .chain(&request.opp_board.bottom)
    {
        if name.starts_with('X') {
            seen_jokers += 1;
        } else if !seen_naturals.insert(name.clone()) {
            bail!("{}: card {name} appears twice across the boards", request.id);
        }
    }
    if seen_jokers > 2 {
        bail!("{}: {} jokers visible, the deck has two", request.id, seen_jokers);
    }
    let mut pool_names: Vec<String> = all_cards()
        .into_iter()
        .filter(|n| !n.starts_with('X') && !seen_naturals.contains(n))
        .collect();
    for slot in 0..(2 - seen_jokers) {
        pool_names.push(format!("X{}", slot + 1));
    }
    let pool: Vec<Card> = pool_names
        .iter()
        .map(|n| to_core_card(n))
        .collect::<Result<Vec<_>>>()?;
    let n = pool.len();
    let wanted = 29 - request.opp_dead.len();
    if n != wanted {
        bail!("{}: pool is {} cards, wanted {wanted}", request.id, n);
    }

    // Per-card survival weights: 1 everywhere for the uniform pool.
    let survive: Vec<f64> = match discard {
        None => vec![1.0; n],
        Some((model, side)) => {
            let rows = match side {
                DeadSide::Board => &my_board.rows,
                DeadSide::OppBoard => &opp_board.rows,
            };
            pool.iter()
                .map(|card| 1.0 - model.predict(card, rows) as f64)
                .collect()
        }
    };

    // --- opponent completions, priced once against the whole pool ---------
    let opp_patterns = placement_patterns(&opp_board);
    let mut opp_memo = TerminalMemo::new(&opp_board);
    // (i, j) -> per-pattern terminals; i < j index the pool.
    let mut opp_terms: Vec<Vec<Terminal>> = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let mut per = Vec::with_capacity(opp_patterns.len());
            for pattern in &opp_patterns {
                per.push(opp_memo.terminal(&[(pattern[0], pool[i]), (pattern[1], pool[j])]));
            }
            opp_terms.push(per);
        }
    }
    let pair_index = |i: usize, j: usize| -> usize {
        let (a, b) = if i < j { (i, j) } else { (j, i) };
        a * n - a * (a + 1) / 2 + (b - a - 1)
    };

    // --- hero candidates: every (pair, pattern) final board ---------------
    let my_patterns = placement_patterns(&my_board);
    let mut my_memo = TerminalMemo::new(&my_board);
    struct Candidate {
        i: usize,
        j: usize,
        pattern: usize,
        term: Terminal,
    }
    let mut candidates: Vec<Candidate> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            for (slot, pattern) in my_patterns.iter().enumerate() {
                candidates.push(Candidate {
                    i,
                    j,
                    pattern: slot,
                    term: my_memo.terminal(&[(pattern[0], pool[i]), (pattern[1], pool[j])]),
                });
            }
        }
    }

    // --- per candidate: opponent sweep with per-card containing sums ------
    // value_of[c][d] = hero's expected score for candidate c when hero's
    // discard is pool card d, i.e. the mean over the opponent's C(26,3)
    // draws from pool minus {c.i, c.j, d} of their best (hero-minimising)
    // placement.
    let c26_3 = ((n - 3) * (n - 4) * (n - 5) / 6) as f64;
    let values: Vec<Vec<f64>> = candidates
        .par_iter()
        .map(|candidate| {
            // pairval[(i,j)] = hero's score if the opponent holds i and j and
            // places them at its own best pattern.
            let mut pairval = vec![f64::INFINITY; n * n];
            for i in 0..n {
                if i == candidate.i || i == candidate.j {
                    continue;
                }
                for j in (i + 1)..n {
                    if j == candidate.i || j == candidate.j {
                        continue;
                    }
                    let mut best = f64::INFINITY;
                    for term in &opp_terms[pair_index(i, j)] {
                        let score = compose(&candidate.term, term, fl_ev_table);
                        if score < best {
                            best = score;
                        }
                    }
                    pairval[i * n + j] = best;
                }
            }
            // Sweep every three-subset of the remaining 27 cards once,
            // accumulating the weighted total, the weight mass, and both
            // per-card so one discard's exclusion is a subtraction.
            let mut total = 0.0f64;
            let mut mass = 0.0f64;
            let mut containing = vec![0.0f64; n];
            let mut containing_mass = vec![0.0f64; n];
            let uniform = discard.is_none();
            let members: Vec<usize> = (0..n)
                .filter(|k| *k != candidate.i && *k != candidate.j)
                .collect();
            for a in 0..members.len() {
                for b in (a + 1)..members.len() {
                    let pv_ab = pairval[members[a] * n + members[b]];
                    let w_ab = survive[members[a]] * survive[members[b]];
                    for c in (b + 1)..members.len() {
                        let (i, j, k) = (members[a], members[b], members[c]);
                        let mut draw_min = pv_ab;
                        let v = pairval[i * n + k];
                        if v < draw_min {
                            draw_min = v;
                        }
                        let v = pairval[j * n + k];
                        if v < draw_min {
                            draw_min = v;
                        }
                        let weight = w_ab * survive[k];
                        let weighted = weight * draw_min;
                        total += weighted;
                        mass += weight;
                        containing[i] += weighted;
                        containing[j] += weighted;
                        containing[k] += weighted;
                        containing_mass[i] += weight;
                        containing_mass[j] += weight;
                        containing_mass[k] += weight;
                    }
                }
            }
            // Mean over draws avoiding one discard card each.
            (0..n)
                .map(|d| {
                    if d == candidate.i || d == candidate.j {
                        f64::NAN
                    } else if uniform {
                        (total - containing[d]) / c26_3
                    } else {
                        (total - containing[d]) / (mass - containing_mass[d])
                    }
                })
                .collect()
        })
        .collect();

    // candidate lookup: (i, j, pattern) -> index
    let candidate_at = |i: usize, j: usize, pattern: usize| -> usize {
        pair_index(i, j) * my_patterns.len() + pattern
    };

    let name_at = |k: usize| pool_names[k].clone();
    let action_key = |i: usize, j: usize, pattern: usize, d: usize| -> String {
        let rows = ["top", "middle", "bottom"];
        format!(
            "{}@{},{}@{}|{}",
            name_at(i),
            rows[my_patterns[pattern][0]],
            name_at(j),
            rows[my_patterns[pattern][1]],
            name_at(d)
        )
    };

    match &request.draw {
        Some(draw) => {
            if draw.len() != 3 {
                bail!("{}: a T4 draw is three cards", request.id);
            }
            // Jokers in the draw match by kind, not by the seat-local name.
            let mut picks: Vec<usize> = Vec::with_capacity(3);
            for name in draw {
                let position = if name.starts_with('X') {
                    pool_names
                        .iter()
                        .enumerate()
                        .position(|(k, p)| p.starts_with('X') && !picks.contains(&k))
                        .map(|k| k)
                } else {
                    pool_names.iter().position(|p| p == name)
                };
                picks.push(position.ok_or_else(|| {
                    anyhow!("{}: draw card {name} is not unseen", request.id)
                })?);
            }
            let mut actions: Vec<V4Action> = Vec::new();
            let mut best = f64::NEG_INFINITY;
            for discard in 0..3usize {
                let kept: Vec<usize> = (0..3).filter(|k| *k != discard).collect();
                let (i, j) = (picks[kept[0]].min(picks[kept[1]]), picks[kept[0]].max(picks[kept[1]]));
                for pattern in 0..my_patterns.len() {
                    let value = values[candidate_at(i, j, pattern)][picks[discard]];
                    if value > best {
                        best = value;
                    }
                    actions.push(V4Action {
                        key: action_key(i, j, pattern, picks[discard]),
                        value,
                    });
                }
            }
            actions.sort_by(|a, b| a.key.cmp(&b.key));
            Ok(V4FirstResponse {
                id: request.id.clone(),
                schema: "ofc_hu_v4_first/v1",
                value: best,
                actions,
                pool: n,
                candidates: candidates.len(),
            })
        }
        None => {
            // State value: weighted mean over every C(29,3) draw of the best
            // action -- hero's own draw also had to survive the hidden dead.
            let mut total = 0.0f64;
            let mut mass = 0.0f64;
            for x in 0..n {
                for y in (x + 1)..n {
                    let w_xy = survive[x] * survive[y];
                    for z in (y + 1)..n {
                        let mut best = f64::NEG_INFINITY;
                        for (i, j, d) in [(x, y, z), (x, z, y), (y, z, x)] {
                            for pattern in 0..my_patterns.len() {
                                let value = values[candidate_at(i, j, pattern)][d];
                                if value > best {
                                    best = value;
                                }
                            }
                        }
                        let weight = w_xy * survive[z];
                        total += weight * best;
                        mass += weight;
                    }
                }
            }
            Ok(V4FirstResponse {
                id: request.id.clone(),
                schema: "ofc_hu_v4_first/v1",
                value: total / mass,
                actions: Vec::new(),
                pool: n,
                candidates: candidates.len(),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn board(top: &[&str], middle: &[&str], bottom: &[&str]) -> BoardStr {
        BoardStr {
            top: top.iter().map(|s| s.to_string()).collect(),
            middle: middle.iter().map(|s| s.to_string()).collect(),
            bottom: bottom.iter().map(|s| s.to_string()).collect(),
        }
    }

    fn fixture() -> V4FirstRequest {
        V4FirstRequest {
            id: "v4-fixture".into(),
            board: board(
                &["Ah", "Kd"],
                &["7c", "7d", "2h", "3h"],
                &["Ts", "Js", "Qs", "9s", "8s"],
            ),
            dead: vec!["2c".into(), "2d".into(), "2s".into()],
            draw: None,
            opp_dead: Vec::new(),
            opp_board: board(
                &["Qh", "Qd"],
                &["5c", "5d", "5h", "Kc"],
                &["Ac", "Ad", "As", "Kh", "9h"],
            ),
        }
    }

    /// The §2.1 probe: two open slots in different rows, on both boards.
    /// Hero always busts here, so the three values coincide -- what this
    /// fixture is for is the *size* of the action set.
    fn probe_fixture() -> V4FirstRequest {
        V4FirstRequest {
            id: "v4-probe".into(),
            board: board(
                &["Ah", "Kd", "Qc"],
                &["7c", "7d", "2h", "3s"],
                &["Ts", "Jh", "4d", "9c"],
            ),
            dead: vec!["2c".into(), "2d".into(), "2s".into()],
            draw: Some(vec!["5h".into(), "6h".into(), "8d".into()]),
            opp_dead: Vec::new(),
            opp_board: board(
                &["Qh", "Qd", "Jc"],
                &["5c", "5d", "Kc", "8h"],
                &["Ac", "Ad", "As", "Kh"],
            ),
        }
    }

    /// Row pairs enumerated **without** `placement_patterns`, so the test that
    /// certifies the enumerator is not written with the enumerator (d3cca6c's
    /// lesson: a test that shares the generator agrees with its defects).  The
    /// nine ordered assignments are decoded from a counter and each is checked
    /// by building the resulting row lengths against the capacities, not by
    /// consulting `open_slots`.
    fn independent_row_pairs(row_lens: [usize; 3]) -> Vec<[usize; 2]> {
        const CAPACITY: [usize; 3] = [3, 5, 5];
        let mut out = Vec::new();
        for code in 0..9usize {
            let (a, b) = (code / 3, code % 3);
            let mut lens = row_lens;
            lens[a] += 1;
            lens[b] += 1;
            if (0..3).all(|r| lens[r] <= CAPACITY[r]) {
                out.push([a, b]);
            }
        }
        out
    }

    /// The final boards a pattern list reaches, canonically keyed so that two
    /// patterns landing on the same board collapse and two patterns landing on
    /// different boards do not.
    fn final_boards(base: &CoreBoard, patterns: &[[usize; 2]], pair: [Card; 2]) -> Vec<String> {
        let mut keys: Vec<String> = patterns
            .iter()
            .map(|pattern| {
                let mut rows: [Vec<(u8, u8)>; 3] = [Vec::new(), Vec::new(), Vec::new()];
                for row in 0..3 {
                    rows[row] = base.rows[row].iter().map(|c| (c.rank, c.suit)).collect();
                }
                rows[pattern[0]].push((pair[0].rank, pair[0].suit));
                rows[pattern[1]].push((pair[1].rank, pair[1].suit));
                for row in rows.iter_mut() {
                    row.sort_unstable();
                }
                format!("{rows:?}")
            })
            .collect();
        keys.sort();
        keys
    }

    /// **Completeness.**  Every legal final board of a two-card placement is
    /// reachable through `placement_patterns`, on every open-slot shape, and
    /// no board is produced twice.  Counts are pinned as literals so that a
    /// quietly smaller search is caught as a number, not as a value drift.
    #[test]
    fn placement_patterns_reach_every_legal_final_board() {
        // (top, middle, bottom, open-slot shape, expected pattern count)
        let shapes: [(&[&str], &[&str], &[&str], [usize; 3], usize); 6] = [
            // [0,1,1]: the §2.1 probe shape -- two different rows, both orders.
            (
                &["Ah", "Kd", "Qc"],
                &["7c", "7d", "2h", "3s"],
                &["Ts", "Jh", "4d", "9c"],
                [0, 1, 1],
                2,
            ),
            // [0,2,0]: both cards into one row, one pattern.
            (
                &["Ah", "Kd", "Qc"],
                &["7c", "7d", "2h"],
                &["Ts", "Jh", "4d", "9c", "8c"],
                [0, 2, 0],
                1,
            ),
            // [1,1,0]
            (
                &["Ah", "Kd"],
                &["7c", "7d", "2h", "3s"],
                &["Ts", "Jh", "4d", "9c", "8c"],
                [1, 1, 0],
                2,
            ),
            // [2,0,0]
            (
                &["Ah"],
                &["7c", "7d", "2h", "3s", "6s"],
                &["Ts", "Jh", "4d", "9c", "8c"],
                [2, 0, 0],
                1,
            ),
            // [1,0,1]
            (
                &["Ah", "Kd"],
                &["7c", "7d", "2h", "3s", "6s"],
                &["Ts", "Jh", "4d", "9c"],
                [1, 0, 1],
                2,
            ),
            // [0,0,2]
            (
                &["Ah", "Kd", "Qc"],
                &["7c", "7d", "2h", "3s", "6s"],
                &["Ts", "Jh", "4d"],
                [0, 0, 2],
                1,
            ),
        ];
        let pair = [
            to_core_card("5h").unwrap(),
            to_core_card("6d").unwrap(),
        ];
        for (top, middle, bottom, open, expected) in shapes {
            let core = CoreBoard::from_str_board(&board(top, middle, bottom)).unwrap();
            let lens = [core.rows[0].len(), core.rows[1].len(), core.rows[2].len()];
            assert_eq!(
                [3 - lens[0], 5 - lens[1], 5 - lens[2]],
                open,
                "fixture {open:?} does not have the open slots it claims"
            );
            let mine = placement_patterns(&core);
            let theirs = independent_row_pairs(lens);
            assert_eq!(
                mine.len(),
                expected,
                "open {open:?}: {} patterns, pinned at {expected}",
                mine.len()
            );
            assert_eq!(
                final_boards(&core, &mine, pair),
                final_boards(&core, &theirs, pair),
                "open {open:?}: the reachable final boards disagree with the \
                 independent enumeration"
            );
            let mut sorted = mine.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), mine.len(), "open {open:?}: duplicate pattern");
        }
    }

    /// **The count pin.**  The §2.1 probe has two open slots in different rows
    /// and three drawn cards: three discards times two orderings is six legal
    /// actions, and `ai/engine/action_space.get_turn_actions` agrees.  Before
    /// the ordered-pair fix this returned three.
    #[test]
    fn probe_fixture_has_six_actions() {
        const PROBE_ACTIONS: usize = 6;
        let table = [3.17f64, 16.61, 38.76, 70.07];
        let solved = solve(&probe_fixture(), &table).expect("solve");
        assert_eq!(
            solved.actions.len(),
            PROBE_ACTIONS,
            "probe actions: {:?}",
            solved.actions.iter().map(|a| &a.key).collect::<Vec<_>>()
        );
        let keys: Vec<&str> = solved.actions.iter().map(|a| a.key.as_str()).collect();
        // Both assignments of one kept pair, which is exactly what the
        // unordered enumeration dropped.
        assert!(keys.contains(&"5h@middle,6h@bottom|8d"), "{keys:?}");
        assert!(keys.contains(&"5h@bottom,6h@middle|8d"), "{keys:?}");
    }

    /// **The brute force agrees.**  The amortised sweep must equal the naive
    /// enumeration -- draw by draw, placement by placement, opponent draw by
    /// opponent draw -- on a small fixture, to the bit.  This is the whole
    /// module's correctness in one assertion.
    ///
    /// The brute side enumerates placements itself (`independent_row_pairs`)
    /// rather than borrowing `placement_patterns`: sharing the generator would
    /// have made this test agree with the half-enumeration it is meant to
    /// certify against.
    #[test]
    fn amortised_state_value_matches_brute_force() {
        let table = [3.17f64, 16.61, 38.76, 70.07];
        let request = fixture();
        let fast = solve(&request, &table).expect("solve");

        // Brute force.
        let my_board = CoreBoard::from_str_board(&request.board).unwrap();
        let opp_board = CoreBoard::from_str_board(&request.opp_board).unwrap();
        let mut seen: std::collections::BTreeSet<String> = request.dead.iter().cloned().collect();
        for name in request
            .board
            .top
            .iter()
            .chain(&request.board.middle)
            .chain(&request.board.bottom)
            .chain(&request.opp_board.top)
            .chain(&request.opp_board.middle)
            .chain(&request.opp_board.bottom)
        {
            seen.insert(name.clone());
        }
        let pool_names: Vec<String> = all_cards()
            .into_iter()
            .filter(|n| !seen.contains(n))
            .collect();
        let pool: Vec<Card> = pool_names.iter().map(|n| to_core_card(n).unwrap()).collect();
        let n = pool.len();
        let my_patterns = independent_row_pairs([
            my_board.rows[0].len(),
            my_board.rows[1].len(),
            my_board.rows[2].len(),
        ]);
        let opp_patterns = independent_row_pairs([
            opp_board.rows[0].len(),
            opp_board.rows[1].len(),
            opp_board.rows[2].len(),
        ]);
        let mut my_memo = TerminalMemo::new(&my_board);
        let mut opp_memo = TerminalMemo::new(&opp_board);
        let mut total = 0.0f64;
        let mut draws = 0usize;
        for x in 0..n {
            for y in (x + 1)..n {
                for z in (y + 1)..n {
                    let mut best = f64::NEG_INFINITY;
                    for (i, j, d) in [(x, y, z), (x, z, y), (y, z, x)] {
                        for pattern in &my_patterns {
                            let mine =
                                my_memo.terminal(&[(pattern[0], pool[i]), (pattern[1], pool[j])]);
                            // opponent: mean over their draws from pool minus
                            // {i, j, d} of their best (hero-minimising) move.
                            let mut sum = 0.0f64;
                            let mut count = 0usize;
                            for a in 0..n {
                                if a == i || a == j || a == d {
                                    continue;
                                }
                                for b in (a + 1)..n {
                                    if b == i || b == j || b == d {
                                        continue;
                                    }
                                    for c in (b + 1)..n {
                                        if c == i || c == j || c == d {
                                            continue;
                                        }
                                        let mut low = f64::INFINITY;
                                        for (p, q) in [(a, b), (a, c), (b, c)] {
                                            for op in &opp_patterns {
                                                let theirs = opp_memo.terminal(&[
                                                    (op[0], pool[p]),
                                                    (op[1], pool[q]),
                                                ]);
                                                let s = compose(&mine, &theirs, &table);
                                                if s < low {
                                                    low = s;
                                                }
                                            }
                                        }
                                        sum += low;
                                        count += 1;
                                    }
                                }
                            }
                            let value = sum / count as f64;
                            if value > best {
                                best = value;
                            }
                        }
                    }
                    total += best;
                    draws += 1;
                }
            }
        }
        let brute = total / draws as f64;
        assert!(
            (fast.value - brute).abs() < 1e-9,
            "amortised {} vs brute {}",
            fast.value,
            brute
        );
    }
}
