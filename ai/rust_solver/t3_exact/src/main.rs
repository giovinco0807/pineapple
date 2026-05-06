use itertools::Itertools;
use ofc_core::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{self, BufRead, Write};

// ============================================================
//  Board representation for exact solver
// ============================================================

#[derive(Clone, Debug)]
struct Board {
    top: Vec<Card>,
    mid: Vec<Card>,
    bot: Vec<Card>,
}

impl Board {
    fn new(top: Vec<Card>, mid: Vec<Card>, bot: Vec<Card>) -> Self {
        Board { top, mid, bot }
    }

    fn place(&self, card: Card, row: u8) -> Board {
        let mut b = self.clone();
        match row {
            0 => b.top.push(card),
            1 => b.mid.push(card),
            2 => b.bot.push(card),
            _ => unreachable!(),
        }
        b
    }

    fn is_complete(&self) -> bool {
        self.top.len() == 3 && self.mid.len() == 5 && self.bot.len() == 5
    }
}

// ============================================================
//  Action Generation
// ============================================================

#[derive(Clone, Debug, Serialize)]
struct TurnAction {
    discard: Card,
    placements: [(Card, u8); 2],
}

fn gen_turn_actions(cards: &[Card; 3], board: &Board) -> Vec<TurnAction> {
    let top_cap = 3 - board.top.len();
    let mid_cap = 5 - board.mid.len();
    let bot_cap = 5 - board.bot.len();
    let rows = [0u8, 1, 2];

    let mut actions = Vec::new();

    for disc in 0..3 {
        // Joker cannot be discarded
        if cards[disc].is_joker() {
            continue;
        }

        let remaining: Vec<Card> = (0..3).filter(|&i| i != disc).map(|i| cards[i]).collect();

        for &r0 in &rows {
            for &r1 in &rows {
                let mut cap = [top_cap, mid_cap, bot_cap];
                if cap[r0 as usize] == 0 {
                    continue;
                }
                cap[r0 as usize] -= 1;
                if cap[r1 as usize] == 0 {
                    continue;
                }

                // If identical placements (same cards to same rows), deduplicate?
                // Actually, if remaining[0] == remaining[1], swapping r0 and r1 is identical.
                if remaining[0] == remaining[1] && r0 > r1 {
                    continue;
                }

                actions.push(TurnAction {
                    discard: cards[disc],
                    placements: [(remaining[0], r0), (remaining[1], r1)],
                });
            }
        }
    }
    
    // Simple deduplication based on exact placements
    let mut unique_actions: Vec<TurnAction> = Vec::new();
    for action in actions {
        let mut exists = false;
        for u in &unique_actions {
            if u.discard == action.discard {
                let p1 = action.placements;
                let p2 = u.placements;
                if (p1[0] == p2[0] && p1[1] == p2[1]) || (p1[0] == p2[1] && p1[1] == p2[0]) {
                    exists = true;
                    break;
                }
            }
        }
        if !exists {
            unique_actions.push(action);
        }
    }
    
    unique_actions
}

// ============================================================
//  Terminal Evaluation
// ============================================================

struct FlEvConfig {
    bust_penalty: f64,
    fl_ev: HashMap<u8, f64>,
}

fn evaluate_terminal(board: &Board, config: &FlEvConfig) -> f64 {
    let eval = evaluate_board_with_joker_constraint(&board.top, &board.mid, &board.bot);
    
    if eval.busted {
        return config.bust_penalty;
    }

    let top_r = get_top_royalty(&eval.top);
    let mid_r = get_middle_royalty(&eval.mid);
    let bot_r = get_bottom_royalty(&eval.bot);
    let total = (top_r + mid_r + bot_r) as f64;

    let (fl_qualified, fl_cards) = check_fl_entry(&eval.top);
    let fl_bonus = if fl_qualified {
        config.fl_ev.get(&fl_cards).copied().unwrap_or(0.0)
    } else {
        0.0
    };

    total + fl_bonus
}

// ============================================================
//  Expectimax Exact Solver
// ============================================================

#[derive(Serialize)]
struct ActionResult {
    action_idx: usize,
    action_desc: String,
    ev: f64,
}

#[derive(Serialize)]
struct SolveResult {
    board_state: String,
    dealt: Vec<String>,
    best_action_idx: usize,
    best_ev: f64,
    actions: Vec<ActionResult>,
    elapsed_ms: u64,
}

fn format_turn_action(action: &TurnAction) -> String {
    let row_names = ["T", "M", "B"];
    format!(
        "d:{} {}->{} {}->{}",
        card_to_string(&action.discard),
        card_to_string(&action.placements[0].0),
        row_names[action.placements[0].1 as usize],
        card_to_string(&action.placements[1].0),
        row_names[action.placements[1].1 as usize],
    )
}

fn expectimax_t3(
    board: &Board,
    dealt: &[Card; 3],
    remaining_deck: &[Card],
    config: &FlEvConfig,
) -> SolveResult {
    let start = std::time::Instant::now();

    let t3_actions = gen_turn_actions(dealt, board);

    // Pre-calculate all T4 combinations
    // remaining_deck should have 40 cards. C(40, 3) = 9880.
    let t4_draws: Vec<Vec<Card>> = remaining_deck
        .iter()
        .copied()
        .combinations(3)
        .collect();
    
    let n_draws = t4_draws.len() as f64;

    // Evaluate each T3 action in parallel
    let mut results: Vec<ActionResult> = t3_actions
        .par_iter()
        .enumerate()
        .map(|(i, t3_action)| {
            let mut t4_board = board.clone();
            for &(card, row) in &t3_action.placements {
                t4_board = t4_board.place(card, row);
            }

            // Sum up EV over all possible T4 draws
            let mut total_ev = 0.0;
            
            for draw in &t4_draws {
                let draw_arr = [draw[0], draw[1], draw[2]];
                let t4_actions = gen_turn_actions(&draw_arr, &t4_board);
                
                let mut best_t4_val = f64::NEG_INFINITY;
                
                for t4_action in &t4_actions {
                    let mut final_board = t4_board.clone();
                    for &(card, row) in &t4_action.placements {
                        final_board = final_board.place(card, row);
                    }
                    
                    let val = evaluate_terminal(&final_board, config);
                    if val > best_t4_val {
                        best_t4_val = val;
                    }
                }
                
                total_ev += best_t4_val;
            }

            let avg_ev = total_ev / n_draws;

            ActionResult {
                action_idx: i,
                action_desc: format_turn_action(t3_action),
                ev: avg_ev,
            }
        })
        .collect();

    results.sort_by(|a, b| b.ev.partial_cmp(&a.ev).unwrap());

    let best_ev = results.first().map(|r| r.ev).unwrap_or(0.0);
    let best_action_idx = results.first().map(|r| r.action_idx).unwrap_or(0);

    SolveResult {
        board_state: format!(
            "T:{} M:{} B:{}",
            board.top.len(),
            board.mid.len(),
            board.bot.len()
        ),
        dealt: dealt.iter().map(|c| card_to_string(c)).collect(),
        best_action_idx,
        best_ev,
        actions: results,
        elapsed_ms: start.elapsed().as_millis() as u64,
    }
}

// ============================================================
//  JSON Request Parsing
// ============================================================

#[derive(Deserialize)]
struct SolveRequest {
    board_top: Vec<String>,
    board_mid: Vec<String>,
    board_bot: Vec<String>,
    discards: Vec<String>,
    dealt: Vec<String>,
    #[serde(default = "default_bust_penalty")]
    bust_penalty: f64,
    #[serde(default)]
    fl_ev: HashMap<String, f64>,
}

fn default_bust_penalty() -> f64 {
    -4.0
}

fn parse_card_str(s: &str) -> Option<Card> {
    let s = s.trim();
    if s == "X1" || s == "X2" {
        return Some(Card { rank: 0, suit: 4 });
    }
    if s.len() != 2 {
        return None;
    }
    let chars: Vec<char> = s.chars().collect();
    let rank = match chars[0] {
        '2' => 2, '3' => 3, '4' => 4, '5' => 5, '6' => 6, '7' => 7,
        '8' => 8, '9' => 9, 'T' => 10, 'J' => 11, 'Q' => 12, 'K' => 13, 'A' => 14,
        _ => return None,
    };
    let suit = match chars[1] {
        's' => 0, 'h' => 1, 'd' => 2, 'c' => 3,
        _ => return None,
    };
    Some(Card { rank, suit })
}

fn main() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut stdout = stdout.lock();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        if line.trim().is_empty() {
            continue;
        }

        let req: SolveRequest = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                writeln!(stdout, "{{\"error\":\"JSON parse error: {}\"}}", e).unwrap();
                stdout.flush().unwrap();
                continue;
            }
        };

        if req.dealt.len() != 3 {
            writeln!(stdout, "{{\"error\":\"dealt must be exactly 3 cards\"}}").unwrap();
            stdout.flush().unwrap();
            continue;
        }

        let parse_cards = |strs: &[String]| -> Vec<Card> {
            strs.iter().filter_map(|s| parse_card_str(s)).collect()
        };

        let board = Board::new(
            parse_cards(&req.board_top),
            parse_cards(&req.board_mid),
            parse_cards(&req.board_bot),
        );

        let dealt_vec = parse_cards(&req.dealt);
        let mut dealt = [Card { rank: 0, suit: 0 }; 3];
        dealt.copy_from_slice(&dealt_vec[..3]);

        let discards = parse_cards(&req.discards);

        // Determine remaining deck
        let full_deck = create_deck(true);
        let mut used = Vec::new();
        used.extend_from_slice(&board.top);
        used.extend_from_slice(&board.mid);
        used.extend_from_slice(&board.bot);
        used.extend_from_slice(&dealt);
        used.extend_from_slice(&discards);

        let mut remaining = Vec::new();
        for c in full_deck {
            // Check if card is in used (handle identical jokers carefully)
            let mut found = false;
            for i in 0..used.len() {
                if used[i] == c {
                    used.remove(i);
                    found = true;
                    break;
                }
            }
            if !found {
                remaining.push(c);
            }
        }

        let mut fl_ev_map = HashMap::new();
        for (k, v) in &req.fl_ev {
            if let Ok(cards) = k.parse::<u8>() {
                fl_ev_map.insert(cards, *v);
            }
        }
        
        // Default FL EV if not provided
        if fl_ev_map.is_empty() {
            fl_ev_map.insert(14, 13.0);   // QQ
            fl_ev_map.insert(15, 40.0);   // KK
            fl_ev_map.insert(16, 55.1);   // AA
            fl_ev_map.insert(17, 90.7);   // Trips
        }

        let config = FlEvConfig {
            bust_penalty: req.bust_penalty,
            fl_ev: fl_ev_map,
        };

        let result = expectimax_t3(&board, &dealt, &remaining, &config);

        let json = serde_json::to_string(&result).unwrap();
        writeln!(stdout, "{}", json).unwrap();
        stdout.flush().unwrap();
    }
}
