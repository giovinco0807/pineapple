use fl_solver::{Card, solve_fantasyland_v2_fast, create_deck};
use rand::seq::SliceRandom;
use rand::thread_rng;
use rand::Rng;
use serde::Serialize;
use std::fs::File;
use std::io::{BufWriter, Write};
use rayon::prelude::*;

const RANK_CHARS: [char; 13] = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'];
const SUIT_CHARS: [char; 4] = ['s', 'h', 'd', 'c'];

fn rank_to_char(rank: u8) -> char {
    if rank >= 2 && rank <= 14 {
        RANK_CHARS[(rank - 2) as usize]
    } else if rank == 0 {
        'X'
    } else {
        '?'
    }
}

fn suit_to_char(suit: u8) -> char {
    if suit < 4 {
        SUIT_CHARS[suit as usize]
    } else if suit == 4 {
        'J'
    } else {
        '?'
    }
}

fn card_to_string(card: &Card) -> String {
    if card.is_joker() {
        "JK".to_string()
    } else {
        format!("{}{}", rank_to_char(card.rank), suit_to_char(card.suit))
    }
}

#[derive(Serialize)]
struct PlayerState {
    top: Vec<String>,
    middle: Vec<String>,
    bottom: Vec<String>,
    discards: Vec<String>, // 3 cards total
}

#[derive(Serialize)]
struct T4GameState {
    sample_id: usize,
    is_btn: bool,
    bb: PlayerState,
    btn: PlayerState,
    deck: Vec<String>, // Remaining 24 cards
}

fn peel_card(row: &mut Vec<Card>, discards: &mut Vec<Card>, rng: &mut impl Rng) -> bool {
    let valid_indices: Vec<usize> = row.iter()
        .enumerate()
        .filter(|(_, c)| !c.is_joker())
        .map(|(i, _)| i)
        .collect();
        
    if !valid_indices.is_empty() {
        let &idx = valid_indices.choose(rng).unwrap();
        discards.push(row.remove(idx));
        true
    } else {
        false
    }
}

fn create_player_state(cards: &[Card], rng: &mut impl Rng, do_peel: bool) -> Option<PlayerState> {
    let placement = solve_fantasyland_v2_fast(cards)?;
    
    let mut top = placement.top.clone();
    let mut middle = placement.middle.clone();
    let mut bottom = placement.bottom.clone();
    
    // Original discard from solver
    let mut discards = vec![placement.discards[0].clone()];
    
    if do_peel {
        let is_top_biased = rng.gen_bool(0.5);
        
        if is_top_biased {
            // 50%: Peel 1 from Top, 1 from Middle/Bottom
            peel_card(&mut top, &mut discards, rng);
            let peel_middle = rng.gen_bool(0.5);
            if peel_middle {
                if !peel_card(&mut middle, &mut discards, rng) {
                    peel_card(&mut bottom, &mut discards, rng);
                }
            } else {
                if !peel_card(&mut bottom, &mut discards, rng) {
                    peel_card(&mut middle, &mut discards, rng);
                }
            }
        } else {
            // 50%: Peel 2 completely randomly from anywhere (Top, Middle, Bottom)
            let mut all_valid = Vec::new();
            for (i, c) in top.iter().enumerate() { if !c.is_joker() { all_valid.push((0, i)); } }
            for (i, c) in middle.iter().enumerate() { if !c.is_joker() { all_valid.push((1, i)); } }
            for (i, c) in bottom.iter().enumerate() { if !c.is_joker() { all_valid.push((2, i)); } }
            
            all_valid.shuffle(rng);
            let mut removes = all_valid[0..2].to_vec();
            
            // Sort by row descending, then by index descending to avoid index shifting during removal
            removes.sort_by(|a, b| {
                if a.0 != b.0 {
                    b.0.cmp(&a.0)
                } else {
                    b.1.cmp(&a.1)
                }
            });
            
            for (row, idx) in removes {
                let card = match row {
                    0 => top.remove(idx),
                    1 => middle.remove(idx),
                    2 => bottom.remove(idx),
                    _ => unreachable!(),
                };
                discards.push(card);
            }
        }
    }
    
    Some(PlayerState {
        top: top.iter().map(card_to_string).collect(),
        middle: middle.iter().map(card_to_string).collect(),
        bottom: bottom.iter().map(card_to_string).collect(),
        discards: discards.iter().map(card_to_string).collect(),
    })
}

fn generate_t4_game_state(sample_id: usize) -> Option<T4GameState> {
    let mut rng = thread_rng();
    
    // Deal 28 cards from 52-card deck
    // Note: If jokers were enabled, we would change this to create_deck(true)
    let mut full_deck = create_deck(false);
    full_deck.shuffle(&mut rng);
    
    let bb_cards: Vec<Card> = full_deck[0..14].to_vec();
    let btn_cards: Vec<Card> = full_deck[14..28].to_vec();
    let deck_cards: Vec<Card> = full_deck[28..52].to_vec();
    
    let is_btn = rng.gen_bool(0.5);
    
    // If BB acts, both have 11 cards (both peel 2 from 13).
    // If BTN acts, BB has 13 cards (BB does not peel), BTN has 11 cards (peel 2).
    let bb_state = create_player_state(&bb_cards, &mut rng, !is_btn)?;
    let btn_state = create_player_state(&btn_cards, &mut rng, true)?;
    
    Some(T4GameState {
        sample_id,
        is_btn,
        bb: bb_state,
        btn: btn_state,
        deck: deck_cards.iter().map(card_to_string).collect(),
    })
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let samples: usize = if args.len() > 1 {
        args[1].parse().unwrap_or(50000)
    } else {
        50000
    };
    let output_path = "t4_game_states.jsonl";
    
    println!("Generating {} T4 game states (BB and BTN)...", samples);
    let start = std::time::Instant::now();
    
    let results: Vec<T4GameState> = (0..samples)
        .into_par_iter()
        .filter_map(|i| generate_t4_game_state(i))
        .collect();
        
    let elapsed = start.elapsed().as_secs_f64();
    println!("Generated {} valid states in {:.2}s", results.len(), elapsed);
    
    let file = File::create(output_path).expect("Failed to create file");
    let mut writer = BufWriter::new(file);
    
    for state in &results {
        let json = serde_json::to_string(state).unwrap();
        writeln!(writer, "{}", json).unwrap();
    }
    
    println!("Saved to {}", output_path);
}
