use ofc_core::{Card, is_valid_placement, get_top_royalty, get_middle_royalty, get_bottom_royalty, compare_5_hands, evaluate_3_card, HandRank3, get_pair_rank, get_trips_rank, check_fl_entry};
use crate::bitboard::BitBoard;

pub fn bitboard_to_array(bb: BitBoard, out: &mut [Card]) -> usize {
    let mut i = 0;
    for c in bb.cards() {
        if i >= out.len() { break; }
        if c == 52 || c == 53 {
            out[i] = Card { rank: 0, suit: 4 };
        } else {
            out[i] = Card {
                rank: (c % 13) + 2,
                suit: c / 13,
            };
        }
        i += 1;
    }
    i
}

pub fn evaluate_board(top: BitBoard, mid: BitBoard, bot: BitBoard) -> (bool, i32) {
    let mut top_buf = [Card { rank: 0, suit: 0 }; 3];
    let mut mid_buf = [Card { rank: 0, suit: 0 }; 5];
    let mut bot_buf = [Card { rank: 0, suit: 0 }; 5];
    
    let t_len = bitboard_to_array(top, &mut top_buf);
    let m_len = bitboard_to_array(mid, &mut mid_buf);
    let b_len = bitboard_to_array(bot, &mut bot_buf);

    let top_cards = &top_buf[..t_len];
    let mid_cards = &mid_buf[..m_len];
    let bot_cards = &bot_buf[..b_len];

    let is_bust = !is_valid_placement(top_cards, mid_cards, bot_cards);
    
    if is_bust {
        return (true, 0);
    }
    
    let royalties = get_top_royalty(top_cards)
        + get_middle_royalty(mid_cards)
        + get_bottom_royalty(bot_cards);
        
    (false, royalties)
}

fn compare_3(a: &[Card], b: &[Card]) -> i32 {
    let (ra, _) = evaluate_3_card(a);
    let (rb, _) = evaluate_3_card(b);
    if (ra as u8) != (rb as u8) {
        return if (ra as u8) > (rb as u8) { 1 } else { -1 };
    }
    match ra {
        HandRank3::HighCard => {
            let high_a = a.iter().filter(|c| !c.is_joker()).map(|c| c.rank).max().unwrap_or(0);
            let high_b = b.iter().filter(|c| !c.is_joker()).map(|c| c.rank).max().unwrap_or(0);
            if high_a > high_b { 1 } else if high_a < high_b { -1 } else { 0 }
        }
        HandRank3::OnePair => {
            let pair_a = get_pair_rank(a);
            let pair_b = get_pair_rank(b);
            if pair_a != pair_b {
                return if pair_a > pair_b { 1 } else { -1 };
            }
            let kicker_a = a.iter().filter(|c| !c.is_joker() && c.rank != pair_a).map(|c| c.rank).max().unwrap_or(0);
            let kicker_b = b.iter().filter(|c| !c.is_joker() && c.rank != pair_b).map(|c| c.rank).max().unwrap_or(0);
            if kicker_a > kicker_b { 1 } else if kicker_a < kicker_b { -1 } else { 0 }
        }
        HandRank3::Trips => {
            let trips_a = get_trips_rank(a);
            let trips_b = get_trips_rank(b);
            if trips_a > trips_b { 1 } else if trips_a < trips_b { -1 } else { 0 }
        }
    }
}

pub fn compute_score(
    p1_top: BitBoard, p1_mid: BitBoard, p1_bot: BitBoard,
    p2_top: BitBoard, p2_mid: BitBoard, p2_bot: BitBoard
) -> f64 {
    let (p1_bust, p1_royalties) = evaluate_board(p1_top, p1_mid, p1_bot);
    let (p2_bust, p2_royalties) = evaluate_board(p2_top, p2_mid, p2_bot);

    if p1_bust && p2_bust {
        return 0.0;
    } else if p1_bust {
        return (-6 - p2_royalties) as f64;
    } else if p2_bust {
        return (6 + p1_royalties) as f64;
    }

    let mut p1_t_buf = [Card { rank: 0, suit: 0 }; 3];
    let mut p1_m_buf = [Card { rank: 0, suit: 0 }; 5];
    let mut p1_b_buf = [Card { rank: 0, suit: 0 }; 5];
    let mut p2_t_buf = [Card { rank: 0, suit: 0 }; 3];
    let mut p2_m_buf = [Card { rank: 0, suit: 0 }; 5];
    let mut p2_b_buf = [Card { rank: 0, suit: 0 }; 5];

    let p1_t_len = bitboard_to_array(p1_top, &mut p1_t_buf);
    let p1_m_len = bitboard_to_array(p1_mid, &mut p1_m_buf);
    let p1_b_len = bitboard_to_array(p1_bot, &mut p1_b_buf);
    let p2_t_len = bitboard_to_array(p2_top, &mut p2_t_buf);
    let p2_m_len = bitboard_to_array(p2_mid, &mut p2_m_buf);
    let p2_b_len = bitboard_to_array(p2_bot, &mut p2_b_buf);

    let p1_top_cards = &p1_t_buf[..p1_t_len];
    let p1_mid_cards = &p1_m_buf[..p1_m_len];
    let p1_bot_cards = &p1_b_buf[..p1_b_len];
    let p2_top_cards = &p2_t_buf[..p2_t_len];
    let p2_mid_cards = &p2_m_buf[..p2_m_len];
    let p2_bot_cards = &p2_b_buf[..p2_b_len];

    let mut line_total = 0;
    line_total += compare_3(p1_top_cards, p2_top_cards);
    line_total += compare_5_hands(p1_mid_cards, p2_mid_cards);
    line_total += compare_5_hands(p1_bot_cards, p2_bot_cards);

    let scoop_bonus = if line_total == 3 { 3 } else if line_total == -3 { -3 } else { 0 };

    // FL entry bonus based on exact pre-calculated EV
    let get_fl_ev = |fc: u8| -> f64 {
        match fc {
            14 => 15.8, // QQ
            15 => 22.7, // KK
            16 => 28.6, // AA
            17 => 35.1, // Trips
            _ => 0.0,
        }
    };

    let (_, p1_fc) = check_fl_entry(p1_top_cards);
    let (_, p2_fc) = check_fl_entry(p2_top_cards);
    
    let p1_fl_ev = get_fl_ev(p1_fc);
    let p2_fl_ev = get_fl_ev(p2_fc);
    let fl_diff = p1_fl_ev - p2_fl_ev;

    (line_total + scoop_bonus + p1_royalties - p2_royalties) as f64 + fl_diff
}
