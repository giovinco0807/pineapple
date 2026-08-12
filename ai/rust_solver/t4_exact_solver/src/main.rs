use ofc_core::{
    Card, check_fl_entry, compare_3_hands, compare_5_hands,
    evaluate_board_with_joker_constraint, get_bottom_royalty, get_middle_royalty,
    get_top_royalty,
};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader, Write, BufWriter};
use std::time::Instant;
use rayon::prelude::*;
use itertools::Itertools;

const RANK_CHARS: [char; 13] = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'];
const SUIT_CHARS: [char; 4] = ['s', 'h', 'd', 'c'];

fn char_to_rank(c: char) -> u8 {
    if c == 'X' || c == 'x' { return 0; }
    let c = c.to_ascii_uppercase();
    match c {
        'T' => 10,
        'J' => 11,
        'Q' => 12,
        'K' => 13,
        'A' => 14,
        _ => c.to_digit(10).unwrap() as u8,
    }
}

fn char_to_suit(c: char) -> u8 {
    match c.to_ascii_lowercase() {
        's' => 0,
        'h' => 1,
        'd' => 2,
        'c' => 3,
        'j' => 4,
        _ => panic!("Invalid suit char: {}", c),
    }
}

fn string_to_card(s: &str) -> Card {
    if s.eq_ignore_ascii_case("JK") || s.starts_with('X') || s.starts_with('x') {
        return Card { rank: 0, suit: 4 };
    }
    let chars: Vec<char> = s.chars().collect();
    Card {
        rank: char_to_rank(chars[0]),
        suit: char_to_suit(chars[1]),
    }
}

fn strings_to_cards(v: &[String]) -> Vec<Card> {
    v.iter().map(|s| string_to_card(s)).collect()
}

#[derive(Deserialize, Debug)]
struct PlayerStateStr {
    top: Vec<String>,
    middle: Vec<String>,
    bottom: Vec<String>,
    #[serde(rename = "discards")]
    _discards: Vec<String>,
}

#[derive(Deserialize, Debug)]
struct T4GameStateStr {
    sample_id: usize,
    #[serde(default)]
    is_btn: bool,
    bb: PlayerStateStr,
    btn: PlayerStateStr,
    deck: Vec<String>,
}

#[derive(Serialize, Debug)]
struct PlacementEV {
    cards_placed: Vec<String>,
    slots: Vec<String>,
    ev: f64,
}

#[derive(Serialize, Debug)]
struct T4Solution {
    sample_id: usize,
    bb_top: Vec<String>,
    bb_middle: Vec<String>,
    bb_bottom: Vec<String>,
    bb_drawn: Vec<String>,
    placements: Vec<PlacementEV>,
    best_ev: f64,
}

#[derive(Clone, Debug)]
struct PlayerBoard {
    top: Vec<Card>,
    middle: Vec<Card>,
    bottom: Vec<Card>,
}

fn card_to_string(c: &Card) -> String {
    if c.suit == 4 { return "JK".to_string(); }
    format!("{}{}", RANK_CHARS[(c.rank - 2) as usize], SUIT_CHARS[c.suit as usize])
}

struct PlayerStats {
    board: PlayerBoard,
    bust: bool,
    roy: i32,
    fl_ev: i32,
}

fn compute_stats(board: &PlayerBoard) -> PlayerStats {
    let eval = evaluate_board_with_joker_constraint(&board.top, &board.middle, &board.bottom);
    let constrained_board = PlayerBoard {
        top: eval.top,
        middle: eval.mid,
        bottom: eval.bot,
    };
    let bust = eval.busted;
    let roy = if bust {
        0
    } else {
        get_top_royalty(&constrained_board.top)
            + get_middle_royalty(&constrained_board.middle)
            + get_bottom_royalty(&constrained_board.bottom)
    };
    let fl_ev = if bust {
        0
    } else {
        let (qualified, cards) = check_fl_entry(&constrained_board.top);
        if qualified { cards as i32 } else { 0 }
    };
    PlayerStats { bust, roy, fl_ev, board: constrained_board }
}

fn calculate_score_fast(bb: &PlayerStats, btn: &PlayerStats) -> i32 {
    if bb.bust && btn.bust {
        return 0;
    } else if bb.bust {
        return -6 - btn.roy;
    } else if btn.bust {
        return 6 + bb.roy;
    }
    
    let top_win = compare_3_hands(&bb.board.top, &btn.board.top);
    let mid_win = compare_5_hands(&bb.board.middle, &btn.board.middle);
    let bot_win = compare_5_hands(&bb.board.bottom, &btn.board.bottom);
    
    let line_score = top_win + mid_win + bot_win;
    let scoop = if line_score == 3 { 3 } else if line_score == -3 { -3 } else { 0 };
    let royalty_diff = bb.roy - btn.roy;
    
    line_score + scoop + royalty_diff + bb.fl_ev - btn.fl_ev
}

// Generate all possible valid placements for 2 cards into the available slots
fn get_placements(board: &PlayerBoard, c1: Card, c2: Card) -> Vec<(PlayerBoard, [Card; 2], [&'static str; 2])> {
    let mut results = Vec::new();
    let slots = ["top", "mid", "bot"];
    
    for i in 0..3 {
        for j in 0..3 {
            let mut p = board.clone();
            let mut valid = true;
            
            // Try place c1 in slots[i]
            if slots[i] == "top" && p.top.len() < 3 { p.top.push(c1); }
            else if slots[i] == "mid" && p.middle.len() < 5 { p.middle.push(c1); }
            else if slots[i] == "bot" && p.bottom.len() < 5 { p.bottom.push(c1); }
            else { valid = false; }
            
            if !valid { continue; }
            
            // Try place c2 in slots[j]
            if slots[j] == "top" && p.top.len() < 3 { p.top.push(c2); }
            else if slots[j] == "mid" && p.middle.len() < 5 { p.middle.push(c2); }
            else if slots[j] == "bot" && p.bottom.len() < 5 { p.bottom.push(c2); }
            else { valid = false; }
            
            if valid {
                results.push((p, [c1, c2], [slots[i], slots[j]]));
            }
        }
    }
    
    results
}

fn place_card(board: &mut PlayerBoard, slot: &str, card: Card) {
    match slot {
        "top" => board.top.push(card),
        "mid" => board.middle.push(card),
        "bot" => board.bottom.push(card),
        _ => {}
    }
}

fn solve_t4(state: &T4GameStateStr) -> T4Solution {
    let bb_base = PlayerBoard {
        top: strings_to_cards(&state.bb.top),
        middle: strings_to_cards(&state.bb.middle),
        bottom: strings_to_cards(&state.bb.bottom),
    };
    let btn_base = PlayerBoard {
        top: strings_to_cards(&state.btn.top),
        middle: strings_to_cards(&state.btn.middle),
        bottom: strings_to_cards(&state.btn.bottom),
    };
    let deck = strings_to_cards(&state.deck);
    
    let mut placements_ev = Vec::new();
    let best_ev;
    let drawn_cards_str: Vec<String>;
    
    if state.is_btn {
        // BTN is acting (BB already finished 13 cards)
        // 1-step solver
        let btn_drawn = vec![deck[0], deck[1], deck[2]];
        drawn_cards_str = btn_drawn.iter().map(|c| card_to_string(c)).collect();
        let bb_stat = compute_stats(&bb_base);
        
        for combos in btn_drawn.iter().copied().combinations(2) {
            let p = get_placements(&btn_base, combos[0], combos[1]);
            for (btn_board, cards, slots) in p {
                let btn_stat = compute_stats(&btn_board);
                // score_for_bb is positive if BB wins. For BTN, we want negative of that.
                let score_for_btn = -calculate_score_fast(&bb_stat, &btn_stat);
                
                placements_ev.push(PlacementEV {
                    cards_placed: vec![card_to_string(&cards[0]), card_to_string(&cards[1])],
                    slots: vec![slots[0].to_string(), slots[1].to_string()],
                    ev: score_for_btn as f64,
                });
            }
        }
        
        best_ev = placements_ev.iter().map(|p| p.ev).fold(f64::NEG_INFINITY, f64::max);
        
    } else {
        // BB is acting, BTN acts after
        // 2-step solver
        let bb_drawn = vec![deck[0], deck[1], deck[2]];
        drawn_cards_str = bb_drawn.iter().map(|c| card_to_string(c)).collect();
        let remaining_deck = deck[3..].to_vec();
        
        let mut btn_all_stats = Vec::with_capacity(1330);
        for btn_drawn in remaining_deck.iter().copied().combinations(3) {
            let mut btn_draw_stats = Vec::with_capacity(6);
            for btn_combos in btn_drawn.iter().copied().combinations(2) {
                let btn_placements = get_placements(&btn_base, btn_combos[0], btn_combos[1]);
                for (btn_board, _, _) in btn_placements {
                    btn_draw_stats.push(compute_stats(&btn_board));
                }
            }
            btn_all_stats.push(btn_draw_stats);
        }
        
        for combos in bb_drawn.iter().copied().combinations(2) {
            let p = get_placements(&bb_base, combos[0], combos[1]);
            for (bb_board, cards, slots) in p {
                let mut total_score = 0;
                let mut count = 0;
                
                let bb_stat = compute_stats(&bb_board);
                
                for btn_draw_stats in &btn_all_stats {
                    let mut best_btn_score_for_btn = -9999;
                    
                    for btn_stat in btn_draw_stats {
                        let score_for_bb = calculate_score_fast(&bb_stat, btn_stat);
                        let score_for_btn = -score_for_bb;
                        if score_for_btn > best_btn_score_for_btn {
                            best_btn_score_for_btn = score_for_btn;
                        }
                    }
                    
                    let best_btn_score_for_bb = -best_btn_score_for_btn;
                    total_score += best_btn_score_for_bb;
                    count += 1;
                }
                
                let ev = total_score as f64 / count as f64;
                placements_ev.push(PlacementEV {
                    cards_placed: vec![card_to_string(&cards[0]), card_to_string(&cards[1])],
                    slots: vec![slots[0].to_string(), slots[1].to_string()],
                    ev,
                });
            }
        }
        
        best_ev = placements_ev.iter().map(|p| p.ev).fold(f64::NEG_INFINITY, f64::max);
    }
    
    T4Solution {
        sample_id: state.sample_id,
        bb_top: state.bb.top.clone(),
        bb_middle: state.bb.middle.clone(),
        bb_bottom: state.bb.bottom.clone(),
        bb_drawn: drawn_cards_str,
        placements: placements_ev,
        best_ev,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let input_path = if args.len() > 1 { &args[1] } else { "../t4_generator/t4_game_states.jsonl" };
    let output_path = if args.len() > 2 { &args[2] } else { "t4_exact_solutions.jsonl" };
    
    let file = File::open(input_path).unwrap_or_else(|_| panic!("Failed to open input file: {}", input_path));
    let reader = BufReader::new(file);
    
    let mut states = Vec::new();
    for line in reader.lines() {
        if let Ok(l) = line {
            let state: T4GameStateStr = serde_json::from_str(&l).unwrap();
            states.push(state);
        }
    }
    
    println!("Loaded {} T4 game states from {}.", states.len(), input_path);
    let start = Instant::now();
    
    let solutions: Vec<T4Solution> = states.into_par_iter().map(|state| {
        solve_t4(&state)
    }).collect();
    
    let elapsed = start.elapsed().as_secs_f64();
    println!("Solved {} states in {:.2}s", solutions.len(), elapsed);
    
    let out_file = File::create(output_path).expect("Failed to create output file");
    let mut writer = BufWriter::new(out_file);
    
    for sol in solutions {
        let j = serde_json::to_string(&sol).unwrap();
        writeln!(writer, "{}", j).unwrap();
    }
    println!("Saved solutions to {}", output_path);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn board(top: &[&str], middle: &[&str], bottom: &[&str]) -> PlayerBoard {
        PlayerBoard {
            top: top.iter().map(|card| string_to_card(card)).collect(),
            middle: middle.iter().map(|card| string_to_card(card)).collect(),
            bottom: bottom.iter().map(|card| string_to_card(card)).collect(),
        }
    }

    #[test]
    fn x1_and_x2_parse_as_jokers() {
        assert!(string_to_card("X1").is_joker());
        assert!(string_to_card("X2").is_joker());
    }

    #[test]
    fn compute_stats_uses_canonical_joker_downgrade() {
        let stats = compute_stats(&board(
            &["Qh", "Qs", "X1"],
            &["Kh", "Ks", "9d", "8c", "7h"],
            &["Ah", "Ad", "Ac", "5s", "4d"],
        ));
        assert!(!stats.bust);
        assert_eq!(stats.roy, 7);
        assert_eq!(stats.fl_ev, 14);
    }

    #[test]
    fn top_trips_above_middle_trips_is_bust() {
        let stats = compute_stats(&board(
            &["Ah", "Ad", "Ac"],
            &["2h", "2d", "2c", "Kh", "Qh"],
            &["3h", "3d", "3c", "3s", "4h"],
        ));
        assert!(stats.bust);
    }
}
