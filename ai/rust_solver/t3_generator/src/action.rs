use ofc_core::Card;


#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Action {
    pub placements: Vec<(Card, usize)>, // usize: 0=top, 1=middle, 2=bottom
    pub discard: Option<Card>,
}

pub fn get_turn_actions(dealt_cards: &[Card], top_len: usize, mid_len: usize, bot_len: usize) -> Vec<Action> {
    assert_eq!(dealt_cards.len(), 3);
    let mut cards = dealt_cards.to_vec();
    // Sort logic should exactly match Python's card string sorting.
    // Python string format: "2h", "Ts", "Ad". Rank first, suit second.
    // Python ranks: 2, 3, 4, 5, 6, 7, 8, 9, T, J, Q, K, A
    // Python strings sort alphabetically: "2h" < "3c" ... "Ah" < "As" < "Tc"
    // Wait, Python sorts card strings algebraically!
    // "2" < "3" < "4" < "5" < "6" < "7" < "8" < "9" < "A" < "J" < "K" < "Q" < "T"
    // This is a common pitfall. Let's look at `get_turn_actions` in Python again.
    // Python string sort: 'A' < 'J' < 'K' < 'Q' < 'T' (Wait! A=65, J=74, K=75, Q=81, T=84).
    cards.sort_by(|a, b| {
        let rank_char_a = rank_to_char(a.rank);
        let rank_char_b = rank_to_char(b.rank);
        if rank_char_a != rank_char_b {
            rank_char_a.cmp(&rank_char_b)
        } else {
            let suit_char_a = suit_to_char(a.suit);
            let suit_char_b = suit_to_char(b.suit);
            suit_char_a.cmp(&suit_char_b)
        }
    });

    let mut actions: Vec<Action> = Vec::new();
    let limits = [3, 5, 5];

    for discard_idx in 0..3 {
        let discard = cards[discard_idx];
        let mut remaining = Vec::new();
        for i in 0..3 {
            if i != discard_idx { remaining.push(cards[i]); }
        }

        for pos0 in 0..3 {
            for pos1 in 0..3 {
                let mut counts = [top_len, mid_len, bot_len];
                counts[pos0] += 1;
                if counts[pos0] > limits[pos0] { continue; }
                counts[pos1] += 1;
                if counts[pos1] > limits[pos1] { continue; }

                let action = Action {
                    placements: vec![(remaining[0], pos0), (remaining[1], pos1)],
                    discard: Some(discard),
                };
                
                // deduplicate
                let mut is_dup = false;
                for a in &actions {
                    if a.discard == action.discard {
                        // compare placements as sets
                        let mut p1 = a.placements.clone();
                        let mut p2 = action.placements.clone();
                        p1.sort_by_key(|p| p.1);
                        p2.sort_by_key(|p| p.1);
                        if p1 == p2 {
                            is_dup = true;
                            break;
                        }
                    }
                }
                if !is_dup {
                    actions.push(action);
                }
            }
        }
    }
    actions
}

pub fn get_semantic_action_index(action: &Action, dealt_cards: &[Card]) -> usize {
    let mut cards = dealt_cards.to_vec();
    cards.sort_by(|a, b| {
        let rank_char_a = rank_to_char(a.rank);
        let rank_char_b = rank_to_char(b.rank);
        if rank_char_a != rank_char_b {
            rank_char_a.cmp(&rank_char_b)
        } else {
            let suit_char_a = suit_to_char(a.suit);
            let suit_char_b = suit_to_char(b.suit);
            suit_char_a.cmp(&suit_char_b)
        }
    });

    let discard_idx = cards.iter().position(|c| Some(*c) == action.discard).unwrap();
    let mut remaining = Vec::new();
    for i in 0..3 {
        if i != discard_idx { remaining.push(cards[i]); }
    }

    let pos0 = action.placements.iter().find(|p| p.0 == remaining[0]).unwrap().1;
    let pos1 = action.placements.iter().find(|p| p.0 == remaining[1]).unwrap().1;

    discard_idx * 9 + pos0 * 3 + pos1
}

fn rank_to_char(rank: u8) -> char {
    if rank == 0 { return 'X'; }
    let chars = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'];
    chars[(rank - 2) as usize]
}

fn suit_to_char(suit: u8) -> char {
    if suit == 4 { return '1'; }
    if suit == 5 { return '2'; }
    let _chars = ['s', 'h', 'd', 'c']; // Wait, Python uses 'h','d','c','s'. Let's match python.
    let python_suits = ['s', 'h', 'd', 'c']; // Rust uses 0=s, 1=h, 2=d, 3=c
    python_suits[suit as usize]
}
