use crate::bitboard::BitBoard;
use crate::mcts::Action;
use crate::state::PlayerBoard;

pub fn get_initial_actions(dealt_cards: BitBoard, _board: &PlayerBoard) -> Vec<Action> {
    // dealt_cards has 5 cards
    let cards: Vec<u8> = dealt_cards.cards().collect();
    assert_eq!(cards.len(), 5);

    let mut actions = Vec::new();
    
    // We have 5 cards, each can go to Top (0), Middle (1), Bottom (2).
    // 3^5 = 243 possibilities.
    let mut pow3 = [1; 6];
    for i in 1..=5 {
        pow3[i] = pow3[i - 1] * 3;
    }

    for i in 0..pow3[5] {
        let mut t_count = 0;
        let mut m_count = 0;
        let mut b_count = 0;

        let mut to_top = BitBoard::EMPTY;
        let mut to_middle = BitBoard::EMPTY;
        let mut to_bottom = BitBoard::EMPTY;

        let mut valid = true;
        for c_idx in 0..5 {
            let row = (i / pow3[c_idx]) % 3;
            let card = cards[c_idx];
            match row {
                0 => {
                    t_count += 1;
                    if t_count > 3 { valid = false; break; }
                    to_top.add(card);
                }
                1 => {
                    m_count += 1;
                    if m_count > 5 { valid = false; break; }
                    to_middle.add(card);
                }
                2 => {
                    b_count += 1;
                    if b_count > 5 { valid = false; break; }
                    to_bottom.add(card);
                }
                _ => unreachable!(),
            }
        }

        if valid {
            actions.push(Action {
                to_top,
                to_middle,
                to_bottom,
                discards: BitBoard::EMPTY,
            });
        }
    }

    actions
}

pub fn get_turn_actions(dealt_cards: BitBoard, board: &PlayerBoard) -> Vec<Action> {
    let cards: Vec<u8> = dealt_cards.cards().collect();
    assert_eq!(cards.len(), 3);

    let mut actions = Vec::new();
    let top_space = 3 - board.top_count;
    let mid_space = 5 - board.middle_count;
    let bot_space = 5 - board.bottom_count;

    // Pick 1 discard out of 3
    for discard_idx in 0..3 {
        let discard_card = cards[discard_idx];
        let mut discards = BitBoard::EMPTY;
        discards.add(discard_card);

        let p1_card = cards[(discard_idx + 1) % 3];
        let p2_card = cards[(discard_idx + 2) % 3];

        // 2 cards left to place, 9 possibilities (3x3)
        for row1 in 0..3 {
            for row2 in 0..3 {
                let mut t_count = 0;
                let mut m_count = 0;
                let mut b_count = 0;

                let mut to_top = BitBoard::EMPTY;
                let mut to_middle = BitBoard::EMPTY;
                let mut to_bottom = BitBoard::EMPTY;

                let mut assign = |row: u8, card: u8| {
                    match row {
                        0 => { t_count += 1; to_top.add(card); }
                        1 => { m_count += 1; to_middle.add(card); }
                        2 => { b_count += 1; to_bottom.add(card); }
                        _ => unreachable!(),
                    }
                };

                assign(row1, p1_card);
                assign(row2, p2_card);

                if t_count <= top_space && m_count <= mid_space && b_count <= bot_space {
                    // Only add if we haven't added this exact action already
                    // (Since we iterate over ordered cards, action is unique because cards are distinct.
                    // Wait, if 2 jokers are present, they are distinct u8s (52, 53) so we might generate duplicate Action structs.
                    // But that's fine, or we can deduplicate them).
                    let action = Action {
                        to_top,
                        to_middle,
                        to_bottom,
                        discards,
                    };
                    if !actions.contains(&action) {
                        actions.push(action);
                    }
                }
            }
        }
    }

    actions
}
