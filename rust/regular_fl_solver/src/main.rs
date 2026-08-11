use std::cmp::Ordering;
use std::env;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Card {
    rank: u8,
    suit: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct HandValue {
    cat: u8,
    ranks: [u8; 5],
}

#[derive(Clone, Copy, Debug)]
struct Combo3 {
    mask: u16,
    cards: [Card; 3],
    value: HandValue,
    top_royalty: i32,
    top_stay: bool,
}

#[derive(Clone, Copy, Debug)]
struct Combo5 {
    mask: u16,
    cards: [Card; 5],
    value: HandValue,
    middle_royalty: i32,
    bottom_royalty: i32,
    bottom_stay: bool,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct Placement {
    top: [Card; 3],
    middle: [Card; 5],
    bottom: [Card; 5],
    discard: Card,
    top_royalty: i32,
    middle_royalty: i32,
    bottom_royalty: i32,
    total_royalty: i32,
    can_stay: bool,
    score: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct TrialStats {
    trials: usize,
    solved: usize,
    total_royalty: f64,
    stays: usize,
}

impl TrialStats {
    fn avg_royalty(self) -> f64 {
        self.total_royalty / self.solved.max(1) as f64
    }

    fn stay_rate(self) -> f64 {
        self.stays as f64 / self.solved.max(1) as f64
    }

    fn net_chain_ev(self, opponent_avg: f64, line_scoop_advantage: f64) -> f64 {
        let denom = (1.0 - self.stay_rate()).max(1e-9);
        (self.avg_royalty() - opponent_avg + line_scoop_advantage) / denom
    }
}

#[derive(Clone, Debug)]
struct Config {
    trials: usize,
    seed: u64,
    iterations: usize,
    tolerance: f64,
    opponent_avg: f64,
    line_scoop_advantage: f64,
    initial_ev: f64,
    fixed_stay_bonus: Option<f64>,
    solve_hand: Option<String>,
    output: Option<PathBuf>,
    teacher_output: Option<PathBuf>,
    teacher_samples: usize,
    teacher_future_samples: usize,
    teacher_fl_ev: f64,
    teacher_min_score_gap: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            trials: 1_000,
            seed: 42,
            iterations: 8,
            tolerance: 0.001,
            opponent_avg: 5.0,
            line_scoop_advantage: 4.0,
            initial_ev: 10.0,
            fixed_stay_bonus: None,
            solve_hand: None,
            output: None,
            teacher_output: None,
            teacher_samples: 1_000,
            teacher_future_samples: 128,
            // configs/fl_ev_regular_v4_selfplay.json (M6 run B, 2026-08-06);
            // supersedes 9.109 and, before it, 10.227020614683454.
            teacher_fl_ev: 9.6,
            teacher_min_score_gap: 0.0,
        }
    }
}

struct Rng64 {
    state: u64,
}

impl Rng64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn gen_range(&mut self, upper: usize) -> usize {
        if upper <= 1 {
            return 0;
        }
        (self.next_u64() as usize) % upper
    }
}

fn create_deck() -> Vec<Card> {
    let mut deck = Vec::with_capacity(52);
    for suit in 0..4u8 {
        for rank in 2..=14u8 {
            deck.push(Card { rank, suit });
        }
    }
    deck
}

fn shuffle(deck: &mut [Card], rng: &mut Rng64) {
    for i in (1..deck.len()).rev() {
        let j = rng.gen_range(i + 1);
        deck.swap(i, j);
    }
}

fn card_to_string(card: Card) -> String {
    const RANKS: &[u8; 13] = b"23456789TJQKA";
    const SUITS: &[u8; 4] = b"hdcs";
    let r = RANKS[(card.rank - 2) as usize] as char;
    let s = SUITS[card.suit as usize] as char;
    format!("{}{}", r, s)
}

fn cards_to_string<const N: usize>(cards: &[Card; N]) -> String {
    cards
        .iter()
        .map(|card| card_to_string(*card))
        .collect::<Vec<_>>()
        .join(" ")
}

fn parse_card(input: &str) -> Result<Card, String> {
    let s = input.trim();
    if s.len() != 2 {
        return Err(format!("invalid card {:?}: expected rank+suit", input));
    }
    let bytes = s.as_bytes();
    let rank = match bytes[0] as char {
        '2' => 2,
        '3' => 3,
        '4' => 4,
        '5' => 5,
        '6' => 6,
        '7' => 7,
        '8' => 8,
        '9' => 9,
        'T' | 't' => 10,
        'J' | 'j' => 11,
        'Q' | 'q' => 12,
        'K' | 'k' => 13,
        'A' | 'a' => 14,
        _ => return Err(format!("invalid card rank in {:?}", input)),
    };
    let suit = match bytes[1] as char {
        'h' | 'H' => 0,
        'd' | 'D' => 1,
        'c' | 'C' => 2,
        's' | 'S' => 3,
        _ => return Err(format!("invalid card suit in {:?}", input)),
    };
    Ok(Card { rank, suit })
}

fn parse_hand(input: &str) -> Result<[Card; 14], String> {
    let parts: Vec<&str> = input
        .split(|ch: char| ch == ',' || ch.is_whitespace())
        .filter(|part| !part.is_empty())
        .collect();
    if parts.len() != 14 {
        return Err(format!("--solve expects 14 cards, got {}", parts.len()));
    }
    let mut cards = Vec::with_capacity(14);
    for part in parts {
        let card = parse_card(part)?;
        if cards.contains(&card) {
            return Err(format!("duplicate card in --solve: {}", part));
        }
        cards.push(card);
    }
    Ok(cards.try_into().expect("validated hand has 14 cards"))
}

fn count_ranks(cards: &[Card]) -> [u8; 15] {
    let mut counts = [0u8; 15];
    for c in cards {
        counts[c.rank as usize] += 1;
    }
    counts
}

fn sorted_ranks(cards: &[Card]) -> Vec<u8> {
    let mut ranks: Vec<u8> = cards.iter().map(|c| c.rank).collect();
    ranks.sort_unstable_by(|a, b| b.cmp(a));
    ranks
}

fn straight_high(counts: &[u8; 15]) -> u8 {
    if counts[14] > 0 && counts[5] > 0 && counts[4] > 0 && counts[3] > 0 && counts[2] > 0 {
        return 5;
    }
    for high in (6..=14u8).rev() {
        let mut ok = true;
        for r in (high - 4)..=high {
            if counts[r as usize] == 0 {
                ok = false;
                break;
            }
        }
        if ok {
            return high;
        }
    }
    0
}

fn hand_value(cat: u8, values: &[u8]) -> HandValue {
    let mut ranks = [0u8; 5];
    for (idx, value) in values.iter().enumerate().take(5) {
        ranks[idx] = *value;
    }
    HandValue { cat, ranks }
}

fn evaluate_5(cards: &[Card; 5]) -> HandValue {
    let counts = count_ranks(cards);
    let ranks = sorted_ranks(cards);
    let flush = cards.iter().all(|c| c.suit == cards[0].suit);
    let straight = straight_high(&counts);

    if flush && straight > 0 {
        return hand_value(8, &[straight]);
    }

    let mut groups: Vec<(u8, u8)> = (2..=14u8)
        .filter_map(|rank| {
            let count = counts[rank as usize];
            if count > 0 {
                Some((count, rank))
            } else {
                None
            }
        })
        .collect();
    groups.sort_unstable_by(|a, b| b.cmp(a));

    if groups[0].0 == 4 {
        let quad = groups[0].1;
        let kicker = ranks.iter().copied().find(|r| *r != quad).unwrap_or(0);
        return hand_value(7, &[quad, kicker]);
    }

    if groups[0].0 == 3 && groups.len() > 1 && groups[1].0 == 2 {
        return hand_value(6, &[groups[0].1, groups[1].1]);
    }

    if flush {
        return hand_value(5, &ranks);
    }

    if straight > 0 {
        return hand_value(4, &[straight]);
    }

    if groups[0].0 == 3 {
        let trips = groups[0].1;
        let kickers: Vec<u8> = ranks.iter().copied().filter(|r| *r != trips).collect();
        return hand_value(3, &[trips, kickers[0], kickers[1]]);
    }

    let pairs: Vec<u8> = groups.iter().filter(|g| g.0 == 2).map(|g| g.1).collect();
    if pairs.len() == 2 {
        let kicker = ranks
            .iter()
            .copied()
            .find(|r| *r != pairs[0] && *r != pairs[1])
            .unwrap_or(0);
        return hand_value(2, &[pairs[0], pairs[1], kicker]);
    }

    if pairs.len() == 1 {
        let pair = pairs[0];
        let kickers: Vec<u8> = ranks.iter().copied().filter(|r| *r != pair).collect();
        return hand_value(1, &[pair, kickers[0], kickers[1], kickers[2]]);
    }

    hand_value(0, &ranks)
}

fn evaluate_3(cards: &[Card; 3]) -> HandValue {
    let counts = count_ranks(cards);
    let ranks = sorted_ranks(cards);
    let mut groups: Vec<(u8, u8)> = (2..=14u8)
        .filter_map(|rank| {
            let count = counts[rank as usize];
            if count > 0 {
                Some((count, rank))
            } else {
                None
            }
        })
        .collect();
    groups.sort_unstable_by(|a, b| b.cmp(a));

    if groups[0].0 == 3 {
        return hand_value(3, &[groups[0].1]);
    }
    if groups[0].0 == 2 {
        let pair = groups[0].1;
        let kicker = ranks.iter().copied().find(|r| *r != pair).unwrap_or(0);
        return hand_value(1, &[pair, kicker]);
    }
    hand_value(0, &ranks)
}

fn top_royalty(cards: &[Card; 3]) -> i32 {
    let value = evaluate_3(cards);
    let rank = value.ranks[0] as i32;
    match value.cat {
        3 => 10 + (rank - 2),
        1 if rank >= 6 => rank - 5,
        _ => 0,
    }
}

fn middle_royalty(cards: &[Card; 5]) -> i32 {
    let value = evaluate_5(cards);
    match value.cat {
        8 if value.ranks[0] == 14 => 50,
        8 => 30,
        7 => 20,
        6 => 12,
        5 => 8,
        4 => 4,
        3 => 2,
        _ => 0,
    }
}

fn bottom_royalty(cards: &[Card; 5]) -> i32 {
    let value = evaluate_5(cards);
    match value.cat {
        8 if value.ranks[0] == 14 => 25,
        8 => 15,
        7 => 10,
        6 => 6,
        5 => 4,
        4 => 2,
        _ => 0,
    }
}

#[allow(dead_code)]
fn fl_entry_type(top: &[Card; 3]) -> Option<&'static str> {
    let counts = count_ranks(top);
    if counts.iter().any(|c| *c >= 3) {
        return Some("trips");
    }
    if counts[14] >= 2 {
        return Some("aa");
    }
    if counts[13] >= 2 {
        return Some("kk");
    }
    if counts[12] >= 2 {
        return Some("qq");
    }
    None
}

fn mask_for(indices: &[usize]) -> u16 {
    indices.iter().fold(0u16, |mask, idx| mask | (1u16 << idx))
}

fn build_combo3(cards: &[Card; 14]) -> Vec<Combo3> {
    let mut out = Vec::with_capacity(364);
    for a in 0..12 {
        for b in (a + 1)..13 {
            for c in (b + 1)..14 {
                let hand = [cards[a], cards[b], cards[c]];
                let value = evaluate_3(&hand);
                out.push(Combo3 {
                    mask: mask_for(&[a, b, c]),
                    cards: hand,
                    value,
                    top_royalty: top_royalty(&hand),
                    top_stay: value.cat == 3,
                });
            }
        }
    }
    out
}

fn build_combo5(cards: &[Card; 14]) -> Vec<Combo5> {
    let mut out = Vec::with_capacity(2002);
    for a in 0..10 {
        for b in (a + 1)..11 {
            for c in (b + 1)..12 {
                for d in (c + 1)..13 {
                    for e in (d + 1)..14 {
                        let hand = [cards[a], cards[b], cards[c], cards[d], cards[e]];
                        let value = evaluate_5(&hand);
                        out.push(Combo5 {
                            mask: mask_for(&[a, b, c, d, e]),
                            cards: hand,
                            value,
                            middle_royalty: middle_royalty(&hand),
                            bottom_royalty: bottom_royalty(&hand),
                            bottom_stay: value.cat >= 7,
                        });
                    }
                }
            }
        }
    }
    out
}

fn single_card_from_mask(cards: &[Card; 14], mask: u16) -> Card {
    let idx = mask.trailing_zeros() as usize;
    cards[idx]
}

fn solve_fantasyland(cards: &[Card; 14], stay_bonus: f64) -> Option<Placement> {
    let top_combos = build_combo3(cards);
    let five_combos = build_combo5(cards);
    let all_mask = (1u16 << 14) - 1;

    let mut tops_by_remaining: Vec<Vec<usize>> = vec![Vec::new(); 1 << 14];
    for mask in 0u16..(1u16 << 14) {
        if mask.count_ones() == 4 {
            for (idx, top) in top_combos.iter().enumerate() {
                if top.mask & mask == top.mask {
                    tops_by_remaining[mask as usize].push(idx);
                }
            }
        }
    }

    let mut best: Option<Placement> = None;
    let mut best_score = f64::NEG_INFINITY;
    let mut best_royalty = i32::MIN;

    for bottom in &five_combos {
        let remaining_after_bottom = all_mask ^ bottom.mask;
        for middle in &five_combos {
            if middle.mask & bottom.mask != 0 {
                continue;
            }
            if middle.value > bottom.value {
                continue;
            }
            let remaining_after_middle = remaining_after_bottom ^ middle.mask;
            for top_idx in &tops_by_remaining[remaining_after_middle as usize] {
                let top = top_combos[*top_idx];
                if top.value > middle.value {
                    continue;
                }

                let discard_mask = remaining_after_middle ^ top.mask;
                let total = top.top_royalty + middle.middle_royalty + bottom.bottom_royalty;
                let stay = top.top_stay || bottom.bottom_stay;
                let score = total as f64 + if stay { stay_bonus } else { 0.0 };

                let better = match score.partial_cmp(&best_score).unwrap_or(Ordering::Equal) {
                    Ordering::Greater => true,
                    Ordering::Equal => total > best_royalty,
                    Ordering::Less => false,
                };

                if better {
                    best_score = score;
                    best_royalty = total;
                    best = Some(Placement {
                        top: top.cards,
                        middle: middle.cards,
                        bottom: bottom.cards,
                        discard: single_card_from_mask(cards, discard_mask),
                        top_royalty: top.top_royalty,
                        middle_royalty: middle.middle_royalty,
                        bottom_royalty: bottom.bottom_royalty,
                        total_royalty: total,
                        can_stay: stay,
                        score,
                    });
                }
            }
        }
    }
    best
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Row {
    Top,
    Middle,
    Bottom,
}

impl Row {
    fn name(self) -> &'static str {
        match self {
            Row::Top => "top",
            Row::Middle => "middle",
            Row::Bottom => "bottom",
        }
    }

    fn capacity(self) -> usize {
        match self {
            Row::Top => 3,
            Row::Middle => 5,
            Row::Bottom => 5,
        }
    }
}

const ROWS: [Row; 3] = [Row::Top, Row::Middle, Row::Bottom];

#[derive(Clone, Debug)]
struct Board {
    top: Vec<Card>,
    middle: Vec<Card>,
    bottom: Vec<Card>,
}

impl Board {
    fn empty() -> Self {
        Self {
            top: Vec::new(),
            middle: Vec::new(),
            bottom: Vec::new(),
        }
    }

    fn row(&self, row: Row) -> &[Card] {
        match row {
            Row::Top => &self.top,
            Row::Middle => &self.middle,
            Row::Bottom => &self.bottom,
        }
    }

    fn row_mut(&mut self, row: Row) -> &mut Vec<Card> {
        match row {
            Row::Top => &mut self.top,
            Row::Middle => &mut self.middle,
            Row::Bottom => &mut self.bottom,
        }
    }

    fn all_cards(&self) -> Vec<Card> {
        let mut cards = Vec::with_capacity(self.card_count());
        cards.extend_from_slice(&self.top);
        cards.extend_from_slice(&self.middle);
        cards.extend_from_slice(&self.bottom);
        cards
    }

    fn card_count(&self) -> usize {
        self.top.len() + self.middle.len() + self.bottom.len()
    }

    fn open_slots(&self, row: Row) -> usize {
        row.capacity() - self.row(row).len()
    }

    fn is_complete(&self) -> bool {
        self.top.len() == 3 && self.middle.len() == 5 && self.bottom.len() == 5
    }

    fn place(&self, placements: &[(Card, Row)]) -> Option<Board> {
        let mut next = self.clone();
        for (card, row) in placements {
            if next.row(*row).len() >= row.capacity() {
                return None;
            }
            if next.all_cards().contains(card) {
                return None;
            }
            next.row_mut(*row).push(*card);
        }
        Some(next)
    }
}

#[derive(Clone, Debug)]
struct Action {
    placements: Vec<(Card, Row)>,
    discards: Vec<Card>,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct BoardScore {
    busted: bool,
    top_value: HandValue,
    middle_value: HandValue,
    bottom_value: HandValue,
    top_royalty: i32,
    middle_royalty: i32,
    bottom_royalty: i32,
    total_royalty: i32,
    fl_entry: Option<&'static str>,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct EvaluatedAction {
    action: Action,
    board: Board,
    score: f64,
    board_score: BoardScore,
}

fn generate_turn_actions(board: &Board, dealt: &[Card]) -> Vec<Action> {
    let open_total = 13usize.saturating_sub(board.card_count());
    if open_total == 0 || dealt.is_empty() {
        return Vec::new();
    }
    let place_count = 2usize.min(open_total).min(dealt.len());
    let mut actions = Vec::new();

    if place_count == 1 {
        for i in 0..dealt.len() {
            for row in ROWS {
                if board.open_slots(row) == 0 {
                    continue;
                }
                let placements = vec![(dealt[i], row)];
                if board.place(&placements).is_some() {
                    let discards = dealt
                        .iter()
                        .enumerate()
                        .filter_map(|(idx, card)| if idx == i { None } else { Some(*card) })
                        .collect();
                    actions.push(Action {
                        placements,
                        discards,
                    });
                }
            }
        }
        return actions;
    }

    for i in 0..dealt.len() - 1 {
        for j in (i + 1)..dealt.len() {
            let place_cards = [dealt[i], dealt[j]];
            let discards: Vec<Card> = dealt
                .iter()
                .enumerate()
                .filter_map(|(idx, card)| {
                    if idx == i || idx == j {
                        None
                    } else {
                        Some(*card)
                    }
                })
                .collect();
            for row_a in ROWS {
                for row_b in ROWS {
                    let rows = [row_a, row_b];
                    if rows.iter().filter(|row| **row == row_a).count() > board.open_slots(row_a) {
                        continue;
                    }
                    if row_a != row_b && board.open_slots(row_b) == 0 {
                        continue;
                    }
                    let placements = vec![(place_cards[0], row_a), (place_cards[1], row_b)];
                    if board.place(&placements).is_some() {
                        actions.push(Action {
                            placements,
                            discards: discards.clone(),
                        });
                    }
                }
            }
        }
    }
    actions
}

fn score_board(board: &Board) -> Option<BoardScore> {
    if !board.is_complete() {
        return None;
    }
    let top: [Card; 3] = board.top.clone().try_into().ok()?;
    let middle: [Card; 5] = board.middle.clone().try_into().ok()?;
    let bottom: [Card; 5] = board.bottom.clone().try_into().ok()?;
    let top_value = evaluate_3(&top);
    let middle_value = evaluate_5(&middle);
    let bottom_value = evaluate_5(&bottom);
    let busted = top_value > middle_value || middle_value > bottom_value;
    let (top_roy, middle_roy, bottom_roy, fl_entry) = if busted {
        (0, 0, 0, None)
    } else {
        (
            top_royalty(&top),
            middle_royalty(&middle),
            bottom_royalty(&bottom),
            fl_entry_type(&top),
        )
    };
    Some(BoardScore {
        busted,
        top_value,
        middle_value,
        bottom_value,
        top_royalty: top_roy,
        middle_royalty: middle_roy,
        bottom_royalty: bottom_roy,
        total_royalty: top_roy + middle_roy + bottom_roy,
        fl_entry,
    })
}

fn terminal_score(board: &Board, fl_ev: f64) -> Option<(f64, BoardScore)> {
    let board_score = score_board(board)?;
    if board_score.busted {
        return Some((0.0, board_score));
    }
    let fl_bonus = if board_score.fl_entry.is_some() {
        fl_ev
    } else {
        0.0
    };
    Some((board_score.total_royalty as f64 + fl_bonus, board_score))
}

fn evaluate_final_turn_actions(board: &Board, dealt: &[Card], fl_ev: f64) -> Vec<EvaluatedAction> {
    let mut out = Vec::new();
    for action in generate_turn_actions(board, dealt) {
        if let Some(next_board) = board.place(&action.placements) {
            if !next_board.is_complete() {
                continue;
            }
            if let Some((score, board_score)) = terminal_score(&next_board, fl_ev) {
                out.push(EvaluatedAction {
                    action,
                    board: next_board,
                    score,
                    board_score,
                });
            }
        }
    }
    out.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.board_score.busted.cmp(&b.board_score.busted))
            .then_with(|| {
                b.board_score
                    .total_royalty
                    .cmp(&a.board_score.total_royalty)
            })
    });
    out
}

#[derive(Clone, Debug)]
struct ExpectedAction {
    action: Action,
    board: Board,
    score: f64,
    future_count: usize,
    non_bust_future_count: usize,
}

fn evaluate_turn3_actions(
    board: &Board,
    dealt: &[Card; 3],
    future_deals: &[[Card; 3]],
    fl_ev: f64,
) -> Vec<ExpectedAction> {
    let mut out = Vec::new();
    for action in generate_turn_actions(board, dealt) {
        let Some(next_board) = board.place(&action.placements) else {
            continue;
        };
        let mut score_sum = 0.0;
        let mut future_count = 0;
        let mut non_bust_future_count = 0;
        for future in future_deals {
            let ranked = evaluate_final_turn_actions(&next_board, future, fl_ev);
            if let Some(best) = ranked.first() {
                score_sum += best.score;
                future_count += 1;
                if ranked.iter().any(|item| !item.board_score.busted) {
                    non_bust_future_count += 1;
                }
            }
        }
        if future_count > 0 {
            out.push(ExpectedAction {
                action,
                board: next_board,
                score: score_sum / future_count as f64,
                future_count,
                non_bust_future_count,
            });
        }
    }
    out.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
    out
}

fn sample_turn3_state(rng: &mut Rng64) -> (Board, [Card; 3], Vec<Card>) {
    let mut deck = create_deck();
    shuffle(&mut deck, rng);

    let mut slot_indices: Vec<usize> = (0..13).collect();
    for i in (1..slot_indices.len()).rev() {
        let j = rng.gen_range(i + 1);
        slot_indices.swap(i, j);
    }
    let occupied: Vec<usize> = slot_indices.into_iter().take(9).collect();
    let mut board = Board::empty();
    let mut card_idx = 0;
    for slot in 0..13 {
        if !occupied.contains(&slot) {
            continue;
        }
        let row = if slot < 3 {
            Row::Top
        } else if slot < 8 {
            Row::Middle
        } else {
            Row::Bottom
        };
        board.row_mut(row).push(deck[card_idx]);
        card_idx += 1;
    }

    let dealt = [deck[9], deck[10], deck[11]];
    let remaining = deck[12..].to_vec();
    (board, dealt, remaining)
}

fn sample_future_deals(remaining: &[Card], samples: usize, rng: &mut Rng64) -> Vec<[Card; 3]> {
    if samples == 0 {
        let mut out = Vec::new();
        for a in 0..remaining.len() - 2 {
            for b in (a + 1)..remaining.len() - 1 {
                for c in (b + 1)..remaining.len() {
                    out.push([remaining[a], remaining[b], remaining[c]]);
                }
            }
        }
        return out;
    }

    let mut out = Vec::with_capacity(samples);
    for _ in 0..samples {
        let mut indices: Vec<usize> = (0..remaining.len()).collect();
        for i in 0..3 {
            let j = i + rng.gen_range(remaining.len() - i);
            indices.swap(i, j);
        }
        out.push([
            remaining[indices[0]],
            remaining[indices[1]],
            remaining[indices[2]],
        ]);
    }
    out
}

fn write_turn3_teacher_data(
    path: &PathBuf,
    samples: usize,
    future_samples: usize,
    seed: u64,
    fl_ev: f64,
    min_score_gap: f64,
) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("failed to create {}: {}", parent.display(), e))?;
        }
    }
    let file = fs::File::create(path)
        .map_err(|e| format!("failed to create {}: {}", path.display(), e))?;
    let mut writer = BufWriter::new(file);
    let mut rng = Rng64::new(seed);
    let mut written = 0usize;
    let mut attempts = 0usize;
    let max_attempts = samples.saturating_mul(1000).max(1000);

    while written < samples {
        attempts += 1;
        if attempts > max_attempts {
            return Err(format!(
                "only wrote {} samples after {} attempts; lower --teacher-min-score-gap",
                written, max_attempts
            ));
        }
        let (board, dealt, remaining) = sample_turn3_state(&mut rng);
        let futures = sample_future_deals(&remaining, future_samples, &mut rng);
        let ranked = evaluate_turn3_actions(&board, &dealt, &futures, fl_ev);
        if ranked.is_empty() {
            continue;
        }
        if ranked.iter().all(|item| item.non_bust_future_count == 0) {
            continue;
        }
        let score_gap = if ranked.len() > 1 {
            ranked[0].score - ranked[1].score
        } else {
            0.0
        };
        if score_gap < min_score_gap {
            continue;
        }
        let line = turn3_sample_json(written, &board, &dealt, fl_ev, score_gap, &ranked);
        writer
            .write_all(line.as_bytes())
            .map_err(|e| format!("failed to write {}: {}", path.display(), e))?;
        writer
            .write_all(b"\n")
            .map_err(|e| format!("failed to write {}: {}", path.display(), e))?;
        written += 1;
    }
    writer
        .flush()
        .map_err(|e| format!("failed to flush {}: {}", path.display(), e))?;
    Ok(())
}

fn turn3_sample_json(
    sample_id: usize,
    board: &Board,
    dealt: &[Card; 3],
    fl_ev: f64,
    score_gap: f64,
    ranked: &[ExpectedAction],
) -> String {
    let actions = ranked
        .iter()
        .map(expected_action_json)
        .collect::<Vec<_>>()
        .join(",");
    format!(
        "{{\"sample_id\":{},\"rule_set\":\"regular\",\"phase\":\"turn3_9card\",\"fl_ev\":{:.6},\"board\":{},\"dealt\":{},\"best_action\":0,\"score_gap\":{:.6},\"actions\":[{}]}}",
        sample_id,
        fl_ev,
        board_json(board),
        cards_json(dealt),
        score_gap,
        actions
    )
}

fn expected_action_json(item: &ExpectedAction) -> String {
    format!(
        "{{\"placements\":{},\"discards\":{},\"score\":{:.6},\"future_count\":{},\"non_bust_future_count\":{},\"next_board\":{}}}",
        placements_json(&item.action.placements),
        cards_json(&item.action.discards),
        item.score,
        item.future_count,
        item.non_bust_future_count,
        board_json(&item.board)
    )
}

fn board_json(board: &Board) -> String {
    format!(
        "{{\"top\":{},\"middle\":{},\"bottom\":{}}}",
        cards_json(&board.top),
        cards_json(&board.middle),
        cards_json(&board.bottom)
    )
}

fn placements_json(placements: &[(Card, Row)]) -> String {
    let body = placements
        .iter()
        .map(|(card, row)| format!("[\"{}\",\"{}\"]", card_to_string(*card), row.name()))
        .collect::<Vec<_>>()
        .join(",");
    format!("[{}]", body)
}

fn cards_json(cards: &[Card]) -> String {
    let body = cards
        .iter()
        .map(|card| format!("\"{}\"", card_to_string(*card)))
        .collect::<Vec<_>>()
        .join(",");
    format!("[{}]", body)
}

fn run_trials(trials: usize, seed: u64, stay_bonus: f64) -> TrialStats {
    let mut rng = Rng64::new(seed);
    let mut stats = TrialStats {
        trials,
        ..TrialStats::default()
    };
    for _ in 0..trials {
        let mut deck = create_deck();
        shuffle(&mut deck, &mut rng);
        let hand: [Card; 14] = deck[0..14].try_into().expect("deck slice must be 14 cards");
        if let Some(placement) = solve_fantasyland(&hand, stay_bonus) {
            stats.solved += 1;
            stats.total_royalty += placement.total_royalty as f64;
            if placement.can_stay {
                stats.stays += 1;
            }
        }
    }
    stats
}

fn write_config(
    path: &PathBuf,
    stats: TrialStats,
    opponent_avg: f64,
    line_scoop_advantage: f64,
) -> std::io::Result<()> {
    let body = format!(
        concat!(
            "{{\n",
            "  \"rule_set\": \"regular\",\n",
            "  \"include_jokers\": false,\n",
            "  \"deck_cards\": 52,\n",
            "  \"fl_entry_cards\": {{\"qq\": 14, \"kk\": 14, \"aa\": 14, \"trips\": 14}},\n",
            "  \"fl_stay_cards\": 14,\n",
            "  \"opponent_avg_royalty\": {:.6},\n",
            "  \"line_scoop_advantage\": {:.6},\n",
            "  \"reward_mode\": \"chain\",\n",
            "  \"fl_stats\": {{\n",
            "    \"14\": {{\"R\": {:.6}, \"stay_rate\": {:.6}, \"count\": {}}}\n",
            "  }},\n",
            "  \"fl_ev\": {{\"14\": {:.6}}}\n",
            "}}\n"
        ),
        opponent_avg,
        line_scoop_advantage,
        stats.avg_royalty(),
        stats.stay_rate(),
        stats.solved,
        stats.net_chain_ev(opponent_avg, line_scoop_advantage)
    );
    fs::write(path, body)
}

fn parse_config() -> Result<Config, String> {
    let mut config = Config::default();
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--trials" => config.trials = parse_next(&mut args, "--trials")?,
            "--seed" => config.seed = parse_next(&mut args, "--seed")?,
            "--iterations" => config.iterations = parse_next(&mut args, "--iterations")?,
            "--tolerance" => config.tolerance = parse_next(&mut args, "--tolerance")?,
            "--opponent-avg" => config.opponent_avg = parse_next(&mut args, "--opponent-avg")?,
            "--line-scoop-advantage" => {
                config.line_scoop_advantage = parse_next(&mut args, "--line-scoop-advantage")?
            }
            "--initial-ev" => config.initial_ev = parse_next(&mut args, "--initial-ev")?,
            "--stay-bonus" => {
                config.fixed_stay_bonus = Some(parse_next(&mut args, "--stay-bonus")?)
            }
            "--solve" => config.solve_hand = Some(next_value(&mut args, "--solve")?),
            "--output" => config.output = Some(PathBuf::from(next_value(&mut args, "--output")?)),
            "--teacher-output" => {
                config.teacher_output =
                    Some(PathBuf::from(next_value(&mut args, "--teacher-output")?))
            }
            "--teacher-samples" => {
                config.teacher_samples = parse_next(&mut args, "--teacher-samples")?
            }
            "--future-samples" => {
                config.teacher_future_samples = parse_next(&mut args, "--future-samples")?
            }
            "--teacher-fl-ev" => config.teacher_fl_ev = parse_next(&mut args, "--teacher-fl-ev")?,
            "--teacher-min-score-gap" => {
                config.teacher_min_score_gap = parse_next(&mut args, "--teacher-min-score-gap")?
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            _ => return Err(format!("unknown argument: {}", arg)),
        }
    }
    Ok(config)
}

fn next_value(args: &mut impl Iterator<Item = String>, name: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("missing value for {}", name))
}

fn parse_next<T: std::str::FromStr>(
    args: &mut impl Iterator<Item = String>,
    name: &str,
) -> Result<T, String> {
    let value = next_value(args, name)?;
    value
        .parse::<T>()
        .map_err(|_| format!("invalid value for {}: {}", name, value))
}

fn print_help() {
    println!("regular_fl_solver");
    println!("  --trials N          random 14-card FL hands per iteration (default 1000)");
    println!("  --seed N            RNG seed (default 42)");
    println!("  --iterations N      fixed-point iterations (default 8)");
    println!("  --initial-ev X      starting stay bonus / EV (default 10.0)");
    println!("  --opponent-avg X    opponent average royalty deduction (default 5.0)");
    println!("  --line-scoop-advantage X");
    println!("                      FL-side line/scoop edge per hand (default 4.0)");
    println!("  --tolerance X       fixed-point stop tolerance (default 0.001)");
    println!("  --stay-bonus X      run one fixed stay-bonus pass instead of fixed point");
    println!("  --solve CARDS       solve one 14-card hand, comma or space separated");
    println!("  --output PATH       write fl_ev-style JSON config");
    println!("  --teacher-output PATH");
    println!("                      write turn3 9-card teacher JSONL instead of FL EV");
    println!("  --teacher-samples N teacher rows to write (default 1000)");
    println!("  --future-samples N  final-turn deals sampled per row (0 = exact all futures)");
    println!("  --teacher-fl-ev X   FL entry value used by teacher scoring (default 9.6)");
    println!("  --teacher-min-score-gap X");
    println!("                      skip rows where best-second score gap is smaller");
}

fn main() {
    let config = match parse_config() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {}", e);
            print_help();
            std::process::exit(2);
        }
    };

    println!("Regular no-joker 14-card FL EV");
    println!("  trials/iter:   {}", config.trials);
    println!("  seed:          {}", config.seed);
    println!("  opponent avg:  {:.3}", config.opponent_avg);
    println!("  line/scoop:    {:.3}", config.line_scoop_advantage);

    if let Some(path) = &config.teacher_output {
        println!("Turn3 9-card teacher data");
        println!("  samples:       {}", config.teacher_samples);
        println!("  future samples: {}", config.teacher_future_samples);
        println!("  teacher FL EV: {:.6}", config.teacher_fl_ev);
        println!("  min gap:       {:.3}", config.teacher_min_score_gap);
        if let Err(e) = write_turn3_teacher_data(
            path,
            config.teacher_samples,
            config.teacher_future_samples,
            config.seed,
            config.teacher_fl_ev,
            config.teacher_min_score_gap,
        ) {
            eprintln!("error: {}", e);
            std::process::exit(1);
        }
        println!("  wrote:         {}", path.display());
        return;
    }

    if let Some(hand_text) = &config.solve_hand {
        let cards = match parse_hand(hand_text) {
            Ok(cards) => cards,
            Err(e) => {
                eprintln!("error: {}", e);
                std::process::exit(2);
            }
        };
        let stay_bonus = config.fixed_stay_bonus.unwrap_or(config.initial_ev);
        let placement =
            solve_fantasyland(&cards, stay_bonus).expect("regular FL hand should solve");
        println!("Solution");
        println!("  hand:         {}", cards_to_string(&cards));
        println!("  stay bonus:   {:.3}", stay_bonus);
        println!("  top:          {}", cards_to_string(&placement.top));
        println!("  middle:       {}", cards_to_string(&placement.middle));
        println!("  bottom:       {}", cards_to_string(&placement.bottom));
        println!("  discard:      {}", card_to_string(placement.discard));
        println!(
            "  royalties:    top={} middle={} bottom={} total={}",
            placement.top_royalty,
            placement.middle_royalty,
            placement.bottom_royalty,
            placement.total_royalty,
        );
        println!("  can stay:     {}", placement.can_stay);
        println!("  score:        {:.3}", placement.score);
        return;
    }

    let final_stats = if let Some(stay_bonus) = config.fixed_stay_bonus {
        let stats = run_trials(config.trials, config.seed, stay_bonus);
        println!(
            "  stay_bonus={:.3} solved={} R={:.3} stay={:.2}% netEV={:.3}",
            stay_bonus,
            stats.solved,
            stats.avg_royalty(),
            stats.stay_rate() * 100.0,
            stats.net_chain_ev(config.opponent_avg, config.line_scoop_advantage),
        );
        stats
    } else {
        let mut ev = config.initial_ev;
        let mut final_stats = TrialStats::default();
        for iter in 1..=config.iterations {
            let stats = run_trials(config.trials, config.seed, ev);
            let new_ev = stats.net_chain_ev(config.opponent_avg, config.line_scoop_advantage);
            println!(
                "  iter {:>2}: stay_bonus={:>8.3} solved={:>6} R={:>7.3} stay={:>6.2}% netEV={:>8.3}",
                iter,
                ev,
                stats.solved,
                stats.avg_royalty(),
                stats.stay_rate() * 100.0,
                new_ev,
            );
            let delta = (new_ev - ev).abs();
            ev = new_ev;
            final_stats = stats;
            if delta <= config.tolerance {
                break;
            }
        }
        final_stats
    };

    println!("Summary");
    println!("  trials:       {}", final_stats.trials);
    println!("  solved:       {}", final_stats.solved);
    println!("  avg royalty:  {:.3}", final_stats.avg_royalty());
    println!("  stay rate:    {:.2}%", final_stats.stay_rate() * 100.0);
    println!(
        "  net chain EV: {:.3}",
        final_stats.net_chain_ev(config.opponent_avg, config.line_scoop_advantage)
    );

    if let Some(path) = &config.output {
        if let Err(e) = write_config(
            path,
            final_stats,
            config.opponent_avg,
            config.line_scoop_advantage,
        ) {
            eprintln!("error writing {}: {}", path.display(), e);
            std::process::exit(1);
        }
        println!("  wrote:        {}", path.display());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(rank: u8, suit: u8) -> Card {
        Card { rank, suit }
    }

    #[test]
    fn deck_is_52_cards_without_jokers() {
        let deck = create_deck();
        assert_eq!(deck.len(), 52);
        assert!(deck.iter().all(|card| card.rank >= 2 && card.rank <= 14));
        assert!(deck.iter().all(|card| card.suit < 4));
    }

    #[test]
    fn royalties_match_regular_tables() {
        assert_eq!(top_royalty(&[c(6, 0), c(6, 1), c(4, 2)]), 1);
        assert_eq!(top_royalty(&[c(14, 0), c(14, 1), c(4, 2)]), 9);
        assert_eq!(top_royalty(&[c(2, 0), c(2, 1), c(2, 2)]), 10);
        assert_eq!(top_royalty(&[c(14, 0), c(14, 1), c(14, 2)]), 22);

        assert_eq!(
            middle_royalty(&[c(14, 0), c(14, 1), c(14, 2), c(13, 0), c(13, 1)]),
            12
        );
        assert_eq!(
            middle_royalty(&[c(14, 0), c(14, 1), c(14, 2), c(14, 3), c(13, 1)]),
            20
        );
        assert_eq!(
            middle_royalty(&[c(14, 0), c(13, 0), c(12, 0), c(11, 0), c(10, 0)]),
            50
        );

        assert_eq!(
            bottom_royalty(&[c(14, 0), c(14, 1), c(14, 2), c(13, 0), c(13, 1)]),
            6
        );
        assert_eq!(
            bottom_royalty(&[c(14, 0), c(14, 1), c(14, 2), c(14, 3), c(13, 1)]),
            10
        );
        assert_eq!(
            bottom_royalty(&[c(14, 0), c(13, 0), c(12, 0), c(11, 0), c(10, 0)]),
            25
        );
    }

    #[test]
    fn fl_entry_types_keep_type_even_when_all_entries_deal_14() {
        assert_eq!(fl_entry_type(&[c(12, 0), c(12, 1), c(4, 2)]), Some("qq"));
        assert_eq!(fl_entry_type(&[c(13, 0), c(13, 1), c(4, 2)]), Some("kk"));
        assert_eq!(fl_entry_type(&[c(14, 0), c(14, 1), c(4, 2)]), Some("aa"));
        assert_eq!(fl_entry_type(&[c(2, 0), c(2, 1), c(2, 2)]), Some("trips"));
        assert_eq!(fl_entry_type(&[c(11, 0), c(11, 1), c(4, 2)]), None);
    }

    #[test]
    fn fl_solver_returns_valid_13_plus_discard() {
        let cards = [
            c(14, 0),
            c(14, 1),
            c(13, 0),
            c(13, 1),
            c(12, 0),
            c(12, 1),
            c(11, 0),
            c(10, 0),
            c(9, 0),
            c(8, 0),
            c(7, 0),
            c(6, 0),
            c(5, 0),
            c(4, 0),
        ];
        let placement = solve_fantasyland(&cards, 100.0).expect("solver should find a placement");
        assert_eq!(
            placement.top.len() + placement.middle.len() + placement.bottom.len(),
            13
        );
        assert!(placement.total_royalty >= 0);
        assert!(placement.score >= placement.total_royalty as f64);
        assert!(cards.contains(&placement.discard));
    }

    #[test]
    fn parse_solve_hand_accepts_commas_or_spaces() {
        let hand = parse_hand("Ah,Kh,Qh,Jh,Th,9h,8h,7h,6h,5h,4h,3h,2h,As")
            .expect("comma-separated hand should parse");
        assert_eq!(hand.len(), 14);
        assert!(parse_hand("Ah Kh Qh Jh Th 9h 8h 7h 6h 5h 4h 3h 2h As").is_ok());
        assert!(parse_hand("Ah Ah Qh Jh Th 9h 8h 7h 6h 5h 4h 3h 2h As").is_err());
    }

    #[test]
    fn terminal_score_adds_regular_fl_ev() {
        let board = Board {
            top: vec![c(12, 2), c(12, 3), c(4, 1)],
            middle: vec![c(2, 0), c(3, 0), c(4, 0), c(5, 0), c(7, 0)],
            bottom: vec![c(14, 0), c(13, 0), c(12, 0), c(11, 0), c(10, 0)],
        };
        let (score, board_score) = terminal_score(&board, 8.0).expect("complete board scores");
        assert!(!board_score.busted);
        assert_eq!(board_score.total_royalty, 40);
        assert_eq!(board_score.fl_entry, Some("qq"));
        assert_eq!(score, 48.0);
    }

    #[test]
    fn turn3_teacher_actions_are_sorted() {
        let board = Board {
            top: vec![c(12, 0)],
            middle: vec![c(13, 0), c(13, 1), c(6, 2), c(8, 3)],
            bottom: vec![c(9, 2), c(9, 1), c(9, 3), c(13, 2)],
        };
        let dealt = [c(12, 3), c(14, 0), c(7, 1)];
        let futures = vec![[c(12, 2), c(2, 2), c(3, 2)], [c(14, 1), c(2, 2), c(3, 2)]];
        let ranked = evaluate_turn3_actions(&board, &dealt, &futures, 8.0);
        assert!(!ranked.is_empty());
        for pair in ranked.windows(2) {
            assert!(pair[0].score >= pair[1].score);
        }
        assert_eq!(ranked[0].future_count, 2);
    }

    #[test]
    fn turn3_teacher_marks_all_bust_actions() {
        let board = Board {
            top: vec![c(14, 0), c(14, 1)],
            middle: vec![c(2, 2), c(5, 1), c(9, 0), c(11, 2)],
            bottom: vec![c(3, 3), c(7, 1), c(8, 2)],
        };
        let dealt = [c(13, 2), c(12, 1), c(10, 3)];
        let futures = vec![[c(4, 0), c(6, 3), c(8, 1)]];
        let ranked = evaluate_turn3_actions(&board, &dealt, &futures, 8.0);
        assert!(!ranked.is_empty());
        assert!(ranked.iter().all(|item| item.non_bust_future_count == 0));
    }
}
