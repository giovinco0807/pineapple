use ofc_core::Card;


pub const STATE_DIM: usize = 522;
pub const ACTION_DIM: usize = 27;

// Python CARD_TO_IDX mapping
// Python: RANKS = "23456789TJQKA" (0-12)
// Python: SUITS = "hdcs" (h=0, d=1, c=2, s=3)
// ALL_CARDS = [f"{r}{s}" for s in SUITS for r in RANKS] + ["X1", "X2"]
// Rust: s=0, h=1, d=2, c=3

pub fn card_to_idx(card: &Card) -> usize {
    if card.rank == 0 {
        // Joker
        if card.suit == 4 { 52 } else { 53 }
    } else {
        let r = card.rank - 2; // 0-12
        let s_idx = match card.suit {
            0 => 3, // s -> 3
            1 => 0, // h -> 0
            2 => 1, // d -> 1
            3 => 2, // c -> 2
            _ => panic!("Invalid suit"),
        };
        (s_idx * 13 + r) as usize
    }
}

// Location constants
pub const LOC_MY_TOP: usize = 0;
pub const LOC_MY_MID: usize = 1;
pub const LOC_MY_BOT: usize = 2;
pub const LOC_OPP_TOP: usize = 3;
pub const LOC_OPP_MID: usize = 4;
pub const LOC_OPP_BOT: usize = 5;
pub const LOC_IN_HAND: usize = 6;
pub const LOC_MY_DISCARD: usize = 7;
pub const LOC_UNSEEN: usize = 8;

#[derive(Clone, Default)]
pub struct Board {
    pub top: Vec<Card>,
    pub middle: Vec<Card>,
    pub bottom: Vec<Card>,
}

impl Board {
    pub fn all_cards(&self) -> Vec<Card> {
        let mut all = Vec::new();
        all.extend(self.top.iter());
        all.extend(self.middle.iter());
        all.extend(self.bottom.iter());
        all
    }
}

#[derive(Clone)]
pub struct Observation {
    pub board_self: Board,
    pub board_opponent: Board,
    pub dealt_cards: Vec<Card>,
    pub known_discards_self: Vec<Card>,
    pub turn: u8,
    pub is_btn: bool,
    pub is_fl: bool,
    pub opp_is_fl: bool,
    pub chips_self: i32,
    pub chips_opponent: i32,
}

impl Observation {
    pub fn unseen_cards(&self) -> Vec<Card> {
        let mut seen = vec![false; 54];
        for c in self.board_self.all_cards() { seen[card_to_idx(&c)] = true; }
        for c in self.board_opponent.all_cards() { seen[card_to_idx(&c)] = true; }
        for c in &self.dealt_cards { seen[card_to_idx(c)] = true; }
        for c in &self.known_discards_self { seen[card_to_idx(c)] = true; }
        
        let mut unseen = Vec::new();
        // Construct all 54 cards
        for suit in 0..4 {
            for rank in 2..=14 {
                let c = Card { rank, suit };
                if !seen[card_to_idx(&c)] { unseen.push(c); }
            }
        }
        if !seen[52] { unseen.push(Card { rank: 0, suit: 4 }); }
        if !seen[53] { unseen.push(Card { rank: 0, suit: 5 }); } // Assuming suit 5 is X2
        unseen
    }
}

fn row_rank_numeric(cards: &[Card], expected_size: usize) -> f32 {
    if cards.is_empty() { return 0.0; }
    
    let mut real_cards = Vec::new();
    for c in cards { if c.rank > 0 { real_cards.push(c); } }
    
    if real_cards.is_empty() { return 0.0; }
    
    let mut ranks: Vec<u8> = real_cards.iter().map(|c| c.rank - 2).collect();
    ranks.sort_unstable_by(|a, b| b.cmp(a));
    
    let mut rank_counts = [0u8; 13];
    for &r in &ranks { rank_counts[r as usize] += 1; }
    
    let mut counts: Vec<u8> = rank_counts.iter().filter(|&&c| c > 0).cloned().collect();
    counts.sort_unstable_by(|a, b| b.cmp(a));
    
    let score = if expected_size == 3 {
        if counts[0] >= 3 { 8.0 }
        else if counts[0] >= 2 { 4.0 + (ranks[0] as f32 / 12.0) }
        else { ranks[0] as f32 / 12.0 }
    } else {
        let mut suit_counts = [0u8; 4];
        for c in &real_cards { suit_counts[c.suit as usize] += 1; }
        let max_suited = *suit_counts.iter().max().unwrap_or(&0);
        
        if counts[0] >= 4 { 7.0 }
        else if counts[0] >= 3 && counts.len() > 1 && counts[1] >= 2 { 6.0 }
        else if real_cards.len() >= 4 && max_suited >= 4 { 5.5 }
        else if counts[0] >= 3 { 3.5 }
        else if counts[0] >= 2 && counts.len() > 1 && counts[1] >= 2 { 2.5 }
        else if counts[0] >= 2 { 1.5 }
        else { ranks[0] as f32 / 12.0 }
    };
    
    (score / 9.0).min(1.0)
}

fn fl_features(top_cards: &[Card]) -> [f32; 6] {
    if top_cards.is_empty() { return [0.0; 6]; }
    
    let mut ranks = Vec::new();
    let mut n_jokers = 0;
    
    for c in top_cards {
        if c.rank > 0 { ranks.push(c.rank - 2); } else { n_jokers += 1; }
    }
    
    if ranks.is_empty() && n_jokers == 0 { return [0.0; 6]; }
    
    let has_q = if ranks.contains(&10) { 1.0 } else { 0.0 };
    let has_k = if ranks.contains(&11) { 1.0 } else { 0.0 };
    let mut has_a = if ranks.contains(&12) { 1.0 } else { 0.0 };
    
    let mut effective_counts = [0u8; 13];
    for &r in &ranks { effective_counts[r as usize] += 1; }
    
    let mut jokers_left = n_jokers;
    
    for r in (0..13).rev() {
        if jokers_left == 0 { break; }
        if effective_counts[r] == 1 {
            effective_counts[r] = 2;
            jokers_left -= 1;
        }
    }
    
    if jokers_left > 0 && ranks.is_empty() {
        effective_counts[12] = jokers_left.min(2);
        has_a = 1.0;
    } else if jokers_left > 0 && !ranks.is_empty() {
        let best_r = ranks.iter().max().unwrap();
        effective_counts[*best_r as usize] += jokers_left;
    }
    
    let mut pairs = Vec::new();
    let mut trips = Vec::new();
    for (r, &c) in effective_counts.iter().enumerate() {
        if c >= 2 { pairs.push(r as u8); }
        if c >= 3 { trips.push(r as u8); }
    }
    
    let has_pair = if !pairs.is_empty() { 1.0 } else { 0.0 };
    let pair_rank = if !pairs.is_empty() { *pairs.iter().max().unwrap() as f32 / 14.0 } else { 0.0 };
    
    let mut fl_ready = 0.0;
    if !pairs.is_empty() && *pairs.iter().max().unwrap() >= 10 { fl_ready = 1.0; }
    if !trips.is_empty() && *trips.iter().max().unwrap() >= 10 { fl_ready = 1.0; }
    
    [has_q, has_k, has_a, has_pair, pair_rank, fl_ready]
}

fn draw_features(mid_cards: &[Card], bot_cards: &[Card]) -> Vec<f32> {
    let mut features = Vec::new();
    for cards in [mid_cards, bot_cards].iter() {
        let mut real_cards = Vec::new();
        for c in *cards { if c.rank > 0 { real_cards.push(c); } }
        
        if real_cards.is_empty() {
            features.extend_from_slice(&[0.0, 0.0, 0.0]);
            continue;
        }
        
        let mut suit_counts = [0u8; 4];
        for c in &real_cards { suit_counts[c.suit as usize] += 1; }
        let flush_draw = *suit_counts.iter().max().unwrap_or(&0) as f32 / 5.0;
        
        let mut ranks: Vec<u8> = real_cards.iter().map(|c| c.rank - 2).collect();
        ranks.sort_unstable();
        ranks.dedup();
        
        let mut straight_pot = 0.0;
        if ranks.len() >= 2 {
            let mut max_consecutive = 1;
            let mut current = 1;
            for i in 1..ranks.len() {
                if ranks[i] == ranks[i-1] + 1 {
                    current += 1;
                    max_consecutive = max_consecutive.max(current);
                } else {
                    current = 1;
                }
            }
            straight_pot = max_consecutive as f32 / 5.0;
        }
        
        let mut rank_counts = [0u8; 13];
        for c in &real_cards { rank_counts[(c.rank - 2) as usize] += 1; }
        let pair_count = rank_counts.iter().filter(|&&c| c >= 2).count() as f32 / 3.0;
        
        features.extend_from_slice(&[flush_draw, straight_pot, pair_count]);
    }
    features
}

pub fn encode_state(obs: &Observation) -> Vec<f32> {
    let mut vec = vec![0.0; STATE_DIM];
    
    // 1. Card matrix (486 dims)
    for c in &obs.board_self.top { vec[card_to_idx(c) * 9 + LOC_MY_TOP] = 1.0; }
    for c in &obs.board_self.middle { vec[card_to_idx(c) * 9 + LOC_MY_MID] = 1.0; }
    for c in &obs.board_self.bottom { vec[card_to_idx(c) * 9 + LOC_MY_BOT] = 1.0; }
    
    for c in &obs.board_opponent.top { vec[card_to_idx(c) * 9 + LOC_OPP_TOP] = 1.0; }
    for c in &obs.board_opponent.middle { vec[card_to_idx(c) * 9 + LOC_OPP_MID] = 1.0; }
    for c in &obs.board_opponent.bottom { vec[card_to_idx(c) * 9 + LOC_OPP_BOT] = 1.0; }
    
    for c in &obs.dealt_cards { vec[card_to_idx(c) * 9 + LOC_IN_HAND] = 1.0; }
    for c in &obs.known_discards_self { vec[card_to_idx(c) * 9 + LOC_MY_DISCARD] = 1.0; }
    for c in &obs.unseen_cards() { vec[card_to_idx(c) * 9 + LOC_UNSEEN] = 1.0; }
    
    // 2. Meta features (6 dims)
    let meta_offset = 486;
    vec[meta_offset + 0] = obs.turn as f32 / 4.0;
    vec[meta_offset + 1] = if obs.is_btn { 1.0 } else { 0.0 };
    vec[meta_offset + 2] = if obs.is_fl { 1.0 } else { 0.0 };
    vec[meta_offset + 3] = if obs.opp_is_fl { 1.0 } else { 0.0 };
    vec[meta_offset + 4] = obs.chips_self as f32 / 200.0;
    vec[meta_offset + 5] = obs.chips_opponent as f32 / 200.0;
    
    // 3. Game-aware features (30 dims)
    let game_offset = meta_offset + 6;
    
    // Row slots
    vec[game_offset + 0] = (3.0 - obs.board_self.top.len() as f32) / 3.0;
    vec[game_offset + 1] = (5.0 - obs.board_self.middle.len() as f32) / 5.0;
    vec[game_offset + 2] = (5.0 - obs.board_self.bottom.len() as f32) / 5.0;
    vec[game_offset + 3] = (3.0 - obs.board_opponent.top.len() as f32) / 3.0;
    vec[game_offset + 4] = (5.0 - obs.board_opponent.middle.len() as f32) / 5.0;
    vec[game_offset + 5] = (5.0 - obs.board_opponent.bottom.len() as f32) / 5.0;
    
    // FL features
    let fl_self = fl_features(&obs.board_self.top);
    for i in 0..6 { vec[game_offset + 6 + i] = fl_self[i]; }
    
    let fl_opp = fl_features(&obs.board_opponent.top);
    vec[game_offset + 12] = fl_opp[0];
    vec[game_offset + 13] = fl_opp[1];
    vec[game_offset + 14] = fl_opp[2];
    vec[game_offset + 15] = fl_opp[5];
    
    // Hand ranks
    let hr_s_t = row_rank_numeric(&obs.board_self.top, 3);
    let hr_s_m = row_rank_numeric(&obs.board_self.middle, 5);
    let hr_s_b = row_rank_numeric(&obs.board_self.bottom, 5);
    let hr_o_t = row_rank_numeric(&obs.board_opponent.top, 3);
    let hr_o_m = row_rank_numeric(&obs.board_opponent.middle, 5);
    let hr_o_b = row_rank_numeric(&obs.board_opponent.bottom, 5);
    
    vec[game_offset + 16] = hr_s_t;
    vec[game_offset + 17] = hr_s_m;
    vec[game_offset + 18] = hr_s_b;
    vec[game_offset + 19] = hr_o_t;
    vec[game_offset + 20] = hr_o_m;
    vec[game_offset + 21] = hr_o_b;
    
    // Bust risk
    vec[game_offset + 22] = if hr_s_t > hr_s_m && !obs.board_self.top.is_empty() && !obs.board_self.middle.is_empty() { 1.0 } else { 0.0 };
    vec[game_offset + 23] = if hr_o_t > hr_o_m && !obs.board_opponent.top.is_empty() && !obs.board_opponent.middle.is_empty() { 1.0 } else { 0.0 };
    
    // Draws
    let draws = draw_features(&obs.board_self.middle, &obs.board_self.bottom);
    for i in 0..6 { vec[game_offset + 24 + i] = draws[i]; }
    
    vec
}
