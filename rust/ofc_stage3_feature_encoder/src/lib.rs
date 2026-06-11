use std::cmp::Ordering;
use std::slice;
use std::time::Instant;

const ROWS: usize = 3;
const RANKS: usize = 13;
const SUITS: usize = 4;
const CARDS: usize = 52;
const FEATURES: usize = 1076;

const CURRENT_OFFSET: usize = 0;
const DEALT_OFFSET: usize = CURRENT_OFFSET + 3 * 52;
const PLACEMENT_OFFSET: usize = DEALT_OFFSET + 52;
const DISCARD_OFFSET: usize = PLACEMENT_OFFSET + 3 * 52;
const NEXT_OFFSET: usize = DISCARD_OFFSET + 52;
const ROW_LEN_OFFSET: usize = NEXT_OFFSET + 3 * 52;
const RANK_COUNT_OFFSET: usize = ROW_LEN_OFFSET + 3;
const SUIT_COUNT_OFFSET: usize = RANK_COUNT_OFFSET + 3 * 13;
const BASE_FEATURE_DIM: usize = SUIT_COUNT_OFFSET + 3 * 4;
const ROW_EXTRA_OFFSET: usize = BASE_FEATURE_DIM;
const ROW_EXTRA_DIM: usize = 24;
const GLOBAL_EXTRA_OFFSET: usize = ROW_EXTRA_OFFSET + 3 * ROW_EXTRA_DIM;
const SELF_FEATURE_DIM: usize = GLOBAL_EXTRA_OFFSET + 4;

const OPPONENT_OFFSET: usize = SELF_FEATURE_DIM;
const DEAD_OFFSET: usize = OPPONENT_OFFSET + 3 * 52;
const SEAT_OFFSET: usize = DEAD_OFFSET + 52;
const ORDER_OFFSET: usize = SEAT_OFFSET + 2;
const OPP_ROW_LEN_OFFSET: usize = ORDER_OFFSET + 2;
const OPP_RANK_COUNT_OFFSET: usize = OPP_ROW_LEN_OFFSET + 3;
const OPP_SUIT_COUNT_OFFSET: usize = OPP_RANK_COUNT_OFFSET + 3 * 13;
const GLOBAL_OFFSET: usize = OPP_SUIT_COUNT_OFFSET + 3 * 4;
const HU_DERIVED_OFFSET: usize = GLOBAL_OFFSET + 12;
const HU_MATCHUP_OFFSET: usize = HU_DERIVED_OFFSET + 48;

const HAND_HIGH: i32 = 0;
const HAND_PAIR: i32 = 1;
const HAND_TWO_PAIR: i32 = 2;
const HAND_TRIPS: i32 = 3;
const HAND_STRAIGHT: i32 = 4;
const HAND_FLUSH: i32 = 5;
const HAND_FULL_HOUSE: i32 = 6;
const HAND_QUADS: i32 = 7;
const HAND_STRAIGHT_FLUSH: i32 = 8;

#[derive(Clone, Copy)]
struct HandValue {
    category: i32,
    ranks: [i32; 5],
    len: usize,
}

#[derive(Clone, Copy)]
struct CompleteSummary {
    top_royalty: f32,
    middle_royalty: f32,
    bottom_royalty: f32,
    total_royalty: f32,
    top_middle_order: f32,
    middle_bottom_order: f32,
}

#[derive(Clone, Copy)]
struct TopSummary {
    values: [f32; 15],
}

#[derive(Clone, Copy)]
struct MatchupSummary {
    count: f32,
    slots: f32,
    category: f32,
    royalty: f32,
    total_royalty: f32,
    premium_potential: f32,
}

#[repr(C)]
pub struct OfcStage3EncoderProfile {
    pub total_seconds: f64,
    pub after_board_seconds: f64,
    pub row_summary_seconds: f64,
    pub global_summary_seconds: f64,
    pub action_delta_seconds: f64,
    pub rows: u64,
}

#[no_mangle]
pub extern "C" fn ofc_stage3_feature_dim() -> usize {
    FEATURES
}

#[no_mangle]
pub unsafe extern "C" fn ofc_stage3_encode(
    state_count: usize,
    max_actions: usize,
    max_placements: usize,
    max_discards: usize,
    hero_board_masks_ptr: *const u64,
    opponent_board_masks_ptr: *const u64,
    dead_card_masks_ptr: *const u64,
    dealt_card_ids_ptr: *const i16,
    seat_ids_ptr: *const i8,
    order_ids_ptr: *const i8,
    action_counts_ptr: *const i16,
    action_placement_card_ids_ptr: *const i16,
    action_placement_row_ids_ptr: *const i8,
    action_discard_card_ids_ptr: *const i16,
    out_features_ptr: *mut f32,
    out_row_to_state_ptr: *mut i32,
    out_row_to_action_ptr: *mut i16,
    profile_ptr: *mut OfcStage3EncoderProfile,
) -> i32 {
    if hero_board_masks_ptr.is_null()
        || opponent_board_masks_ptr.is_null()
        || dead_card_masks_ptr.is_null()
        || dealt_card_ids_ptr.is_null()
        || seat_ids_ptr.is_null()
        || order_ids_ptr.is_null()
        || action_counts_ptr.is_null()
        || action_placement_card_ids_ptr.is_null()
        || action_placement_row_ids_ptr.is_null()
        || action_discard_card_ids_ptr.is_null()
        || out_features_ptr.is_null()
        || out_row_to_state_ptr.is_null()
        || out_row_to_action_ptr.is_null()
    {
        return -1;
    }

    let started = Instant::now();
    let hero_masks = slice::from_raw_parts(hero_board_masks_ptr, state_count * ROWS);
    let opponent_masks = slice::from_raw_parts(opponent_board_masks_ptr, state_count * ROWS);
    let dead_masks = slice::from_raw_parts(dead_card_masks_ptr, state_count);
    let dealt_ids = slice::from_raw_parts(dealt_card_ids_ptr, state_count * 3);
    let seat_ids = slice::from_raw_parts(seat_ids_ptr, state_count);
    let order_ids = slice::from_raw_parts(order_ids_ptr, state_count);
    let action_counts = slice::from_raw_parts(action_counts_ptr, state_count);
    let placement_cards =
        slice::from_raw_parts(action_placement_card_ids_ptr, state_count * max_actions * max_placements);
    let placement_rows =
        slice::from_raw_parts(action_placement_row_ids_ptr, state_count * max_actions * max_placements);
    let discard_cards =
        slice::from_raw_parts(action_discard_card_ids_ptr, state_count * max_actions * max_discards);
    let row_count = action_counts
        .iter()
        .map(|value| usize::try_from(*value.max(&0)).unwrap_or(0))
        .sum::<usize>();
    let out = slice::from_raw_parts_mut(out_features_ptr, row_count * FEATURES);
    let row_to_state = slice::from_raw_parts_mut(out_row_to_state_ptr, row_count);
    let row_to_action = slice::from_raw_parts_mut(out_row_to_action_ptr, row_count);

    let mut after_board_seconds = 0.0;
    let mut row_summary_seconds = 0.0;
    let mut global_summary_seconds = 0.0;
    let mut action_delta_seconds = 0.0;
    let mut output_row = 0usize;

    for state_index in 0..state_count {
        let hero_rows = [
            cards_from_mask(hero_masks[state_index * 3]),
            cards_from_mask(hero_masks[state_index * 3 + 1]),
            cards_from_mask(hero_masks[state_index * 3 + 2]),
        ];
        let opponent_rows = [
            cards_from_mask(opponent_masks[state_index * 3]),
            cards_from_mask(opponent_masks[state_index * 3 + 1]),
            cards_from_mask(opponent_masks[state_index * 3 + 2]),
        ];
        let dead_cards = cards_from_mask(dead_masks[state_index]);
        let dealt = [
            dealt_ids[state_index * 3],
            dealt_ids[state_index * 3 + 1],
            dealt_ids[state_index * 3 + 2],
        ];
        let mut common = [0f32; FEATURES];
        encode_board_rows(&mut common, &hero_rows, CURRENT_OFFSET);
        encode_card_ids(&mut common, &dealt, DEALT_OFFSET);
        encode_board_rows(&mut common, &opponent_rows, OPPONENT_OFFSET);
        encode_cards(&mut common, &dead_cards, DEAD_OFFSET);
        common[SEAT_OFFSET + normalize_binary_id(seat_ids[state_index])] = 1.0;
        common[ORDER_OFFSET + normalize_binary_id(order_ids[state_index])] = 1.0;
        encode_opponent_row_stats(&mut common, &opponent_rows);

        let mut visible = [false; CARDS];
        mark_rows(&mut visible, &hero_rows);
        mark_rows(&mut visible, &opponent_rows);
        mark_cards(&mut visible, &dead_cards);
        for card in dealt.iter().copied().filter(|card| *card >= 0) {
            visible[card as usize] = true;
        }
        let available = available_by_rank(&visible);
        let opponent_top = top_summary(&opponent_rows[0], &available);
        let opponent_complete = complete_summary(&opponent_rows);
        let opponent_matchups = [
            row_matchup_summary(0, &opponent_rows[0], &available),
            row_matchup_summary(1, &opponent_rows[1], &available),
            row_matchup_summary(2, &opponent_rows[2], &available),
        ];
        let opponent_needed = opponent_needed_top(&opponent_rows[0], &available);
        let opponent_count = row_len(&opponent_rows[0]) + row_len(&opponent_rows[1]) + row_len(&opponent_rows[2]);
        let opponent_terminal = if opponent_count == 13 {
            Some(board_score_terminal(&opponent_rows))
        } else {
            None
        };

        let action_count = usize::try_from(action_counts[state_index].max(0)).unwrap_or(0);
        for action_index in 0..action_count {
            let row = &mut out[output_row * FEATURES..(output_row + 1) * FEATURES];
            row.copy_from_slice(&common);
            row_to_state[output_row] = state_index as i32;
            row_to_action[output_row] = action_index as i16;

            let action_base = (state_index * max_actions + action_index) * max_placements;
            let discard_base = (state_index * max_actions + action_index) * max_discards;

            let action_started = Instant::now();
            for placement_index in 0..max_placements {
                let card = placement_cards[action_base + placement_index];
                let target_row = placement_rows[action_base + placement_index];
                if card >= 0 && target_row >= 0 {
                    set_card_row(row, card as usize, target_row as usize, PLACEMENT_OFFSET);
                }
            }
            for discard_index in 0..max_discards {
                let card = discard_cards[discard_base + discard_index];
                if card >= 0 {
                    row[DISCARD_OFFSET + card as usize] = 1.0;
                }
            }
            action_delta_seconds += action_started.elapsed().as_secs_f64();

            let after_started = Instant::now();
            let mut next_rows = hero_rows.clone();
            for placement_index in 0..max_placements {
                let card = placement_cards[action_base + placement_index];
                let target_row = placement_rows[action_base + placement_index];
                if card >= 0 && target_row >= 0 {
                    push_card(&mut next_rows[target_row as usize], card);
                }
            }
            encode_board_rows(row, &next_rows, NEXT_OFFSET);
            encode_self_row_stats(row, &next_rows);
            after_board_seconds += after_started.elapsed().as_secs_f64();

            let row_started = Instant::now();
            encode_self_extra_stats(row, &next_rows);
            let hero_top = top_summary(&next_rows[0], &available);
            let hero_complete = complete_summary(&next_rows);
            let hero_matchups = [
                row_matchup_summary(0, &next_rows[0], &available),
                row_matchup_summary(1, &next_rows[1], &available),
                row_matchup_summary(2, &next_rows[2], &available),
            ];
            row_summary_seconds += row_started.elapsed().as_secs_f64();

            let global_started = Instant::now();
            encode_global(row, &next_rows, &opponent_rows, opponent_count, opponent_terminal);
            encode_derived(
                row,
                &hero_top,
                &opponent_top,
                &hero_complete,
                &opponent_complete,
                &available,
                &opponent_needed,
                discard_cards,
                discard_base,
                max_discards,
            );
            encode_matchup(
                row,
                &next_rows,
                &opponent_rows,
                &hero_matchups,
                &opponent_matchups,
                &hero_complete,
                &opponent_complete,
            );
            global_summary_seconds += global_started.elapsed().as_secs_f64();
            output_row += 1;
        }
    }

    if !profile_ptr.is_null() {
        *profile_ptr = OfcStage3EncoderProfile {
            total_seconds: started.elapsed().as_secs_f64(),
            after_board_seconds,
            row_summary_seconds,
            global_summary_seconds,
            action_delta_seconds,
            rows: output_row as u64,
        };
    }
    0
}

fn normalize_binary_id(value: i8) -> usize {
    if value == 1 {
        1
    } else {
        0
    }
}

fn cards_from_mask(mask: u64) -> Vec<i16> {
    let mut cards = Vec::new();
    for card in 0..52 {
        if (mask & (1u64 << card)) != 0 {
            cards.push(card as i16);
        }
    }
    cards
}

fn mark_rows(visible: &mut [bool; CARDS], rows: &[Vec<i16>; ROWS]) {
    for row in rows {
        mark_cards(visible, row);
    }
}

fn mark_cards(visible: &mut [bool; CARDS], cards: &[i16]) {
    for card in cards.iter().copied().filter(|card| *card >= 0) {
        visible[card as usize] = true;
    }
}

fn available_by_rank(visible: &[bool; CARDS]) -> [i32; RANKS] {
    let mut used = [0i32; RANKS];
    for card in 0..52 {
        if visible[card] {
            used[rank_index(card)] += 1;
        }
    }
    let mut available = [0i32; RANKS];
    for rank in 0..RANKS {
        available[rank] = (4 - used[rank]).max(0);
    }
    available
}

fn row_len(cards: &[i16]) -> usize {
    cards.iter().filter(|card| **card >= 0).count()
}

fn push_card(row: &mut Vec<i16>, card: i16) {
    if !row.contains(&card) {
        row.push(card);
        row.sort_unstable();
    }
}

fn encode_board_rows(vector: &mut [f32], rows: &[Vec<i16>; ROWS], offset: usize) {
    for (row_index, row) in rows.iter().enumerate() {
        for card in row.iter().copied().filter(|card| *card >= 0) {
            vector[offset + row_index * 52 + card as usize] = 1.0;
        }
    }
}

fn encode_cards(vector: &mut [f32], cards: &[i16], offset: usize) {
    for card in cards.iter().copied().filter(|card| *card >= 0) {
        vector[offset + card as usize] = 1.0;
    }
}

fn encode_card_ids(vector: &mut [f32], cards: &[i16; 3], offset: usize) {
    for card in cards.iter().copied().filter(|card| *card >= 0) {
        vector[offset + card as usize] = 1.0;
    }
}

fn set_card_row(vector: &mut [f32], card: usize, row: usize, offset: usize) {
    if row < ROWS && card < CARDS {
        vector[offset + row * 52 + card] = 1.0;
    }
}

fn encode_self_row_stats(vector: &mut [f32], rows: &[Vec<i16>; ROWS]) {
    for (row_index, row) in rows.iter().enumerate() {
        vector[ROW_LEN_OFFSET + row_index] = row_len(row) as f32 / 5.0;
        for card in row.iter().copied().filter(|card| *card >= 0) {
            vector[RANK_COUNT_OFFSET + row_index * 13 + rank_index(card as usize)] += 0.25;
            vector[SUIT_COUNT_OFFSET + row_index * 4 + suit_index(card as usize)] += 0.2;
        }
    }
}

fn encode_opponent_row_stats(vector: &mut [f32], rows: &[Vec<i16>; ROWS]) {
    for (row_index, row) in rows.iter().enumerate() {
        let capacity = row_capacity(row_index) as f32;
        vector[OPP_ROW_LEN_OFFSET + row_index] = row_len(row) as f32 / capacity;
        for card in row.iter().copied().filter(|card| *card >= 0) {
            vector[OPP_RANK_COUNT_OFFSET + row_index * 13 + rank_index(card as usize)] += 0.25;
            vector[OPP_SUIT_COUNT_OFFSET + row_index * 4 + suit_index(card as usize)] += 0.2;
        }
    }
}

fn encode_self_extra_stats(vector: &mut [f32], rows: &[Vec<i16>; ROWS]) {
    for row_index in 0..ROWS {
        let values = row_extra_values(row_index, &rows[row_index]);
        let offset = ROW_EXTRA_OFFSET + row_index * ROW_EXTRA_DIM;
        vector[offset..offset + ROW_EXTRA_DIM].copy_from_slice(&values);
    }
    let values = global_extra_values(rows);
    vector[GLOBAL_EXTRA_OFFSET..GLOBAL_EXTRA_OFFSET + 4].copy_from_slice(&values);
}

fn encode_global(
    vector: &mut [f32],
    hero_rows: &[Vec<i16>; ROWS],
    opponent_rows: &[Vec<i16>; ROWS],
    opponent_count: usize,
    opponent_terminal: Option<(f32, f32, f32)>,
) {
    let hero_count = row_len(&hero_rows[0]) + row_len(&hero_rows[1]) + row_len(&hero_rows[2]);
    vector[GLOBAL_OFFSET] = hero_count as f32 / 13.0;
    vector[GLOBAL_OFFSET + 1] = opponent_count as f32 / 13.0;
    for row in 0..ROWS {
        let cap = row_capacity(row) as f32;
        vector[GLOBAL_OFFSET + 2 + row] = (row_capacity(row) - row_len(&hero_rows[row])) as f32 / cap;
        vector[GLOBAL_OFFSET + 5 + row] = (row_capacity(row) - row_len(&opponent_rows[row])) as f32 / cap;
    }
    if let Some((busted, royalty, fl_entry)) = opponent_terminal {
        vector[GLOBAL_OFFSET + 8] = 1.0;
        vector[GLOBAL_OFFSET + 9] = busted;
        vector[GLOBAL_OFFSET + 10] = royalty;
        vector[GLOBAL_OFFSET + 11] = fl_entry;
    }
}

fn encode_derived(
    vector: &mut [f32],
    hero_top: &TopSummary,
    opponent_top: &TopSummary,
    hero_rows: &CompleteSummary,
    opponent_rows: &CompleteSummary,
    available: &[i32; RANKS],
    opponent_needed: &[bool; RANKS],
    discard_cards: &[i16],
    discard_base: usize,
    max_discards: usize,
) {
    write_top_summary(vector, HU_DERIVED_OFFSET, hero_top);
    write_top_summary(vector, HU_DERIVED_OFFSET + 12, opponent_top);
    vector[HU_DERIVED_OFFSET + 24] = hero_rows.top_royalty / 22.0;
    vector[HU_DERIVED_OFFSET + 25] = hero_rows.middle_royalty / 50.0;
    vector[HU_DERIVED_OFFSET + 26] = hero_rows.bottom_royalty / 25.0;
    vector[HU_DERIVED_OFFSET + 27] = hero_rows.top_middle_order;
    vector[HU_DERIVED_OFFSET + 28] = hero_rows.middle_bottom_order;
    vector[HU_DERIVED_OFFSET + 29] = opponent_rows.top_royalty / 22.0;
    vector[HU_DERIVED_OFFSET + 30] = opponent_rows.middle_royalty / 50.0;
    vector[HU_DERIVED_OFFSET + 31] = opponent_rows.bottom_royalty / 25.0;
    vector[HU_DERIVED_OFFSET + 32] = opponent_rows.top_middle_order;
    vector[HU_DERIVED_OFFSET + 33] = opponent_rows.middle_bottom_order;
    vector[HU_DERIVED_OFFSET + 34] = hero_top.values[3] - opponent_top.values[3];
    vector[HU_DERIVED_OFFSET + 35] = hero_top.values[5] - opponent_top.values[5];
    vector[HU_DERIVED_OFFSET + 36] = hero_rows.total_royalty / 100.0 - opponent_rows.total_royalty / 100.0;

    let mut discard_count = 0usize;
    let mut qka_count = 0usize;
    let mut opponent_needed_count = 0usize;
    for discard_index in 0..max_discards {
        let card = discard_cards[discard_base + discard_index];
        if card >= 0 {
            discard_count += 1;
            let rank = rank_index(card as usize);
            if rank >= 10 {
                qka_count += 1;
            }
            if opponent_needed[rank] {
                opponent_needed_count += 1;
            }
        }
    }
    let denom = discard_count.max(1) as f32;
    vector[HU_DERIVED_OFFSET + 37] = qka_count as f32 / denom;
    vector[HU_DERIVED_OFFSET + 38] = opponent_needed_count as f32 / denom;
    for (idx, rank) in [10usize, 11, 12].iter().copied().enumerate() {
        vector[HU_DERIVED_OFFSET + 39 + idx] = available[rank] as f32 / 4.0;
        vector[HU_DERIVED_OFFSET + 42 + idx] = opponent_top.values[12 + idx];
        vector[HU_DERIVED_OFFSET + 45 + idx] = hero_top.values[12 + idx];
    }
}

fn encode_matchup(
    vector: &mut [f32],
    hero_rows_cards: &[Vec<i16>; ROWS],
    opponent_rows_cards: &[Vec<i16>; ROWS],
    hero_summaries: &[MatchupSummary; ROWS],
    opponent_summaries: &[MatchupSummary; ROWS],
    hero_rows: &CompleteSummary,
    opponent_rows: &CompleteSummary,
) {
    for row in 0..ROWS {
        let hero = hero_summaries[row];
        let opponent = opponent_summaries[row];
        let offset = HU_MATCHUP_OFFSET + row * 12;
        vector[offset] = hero.count;
        vector[offset + 1] = opponent.count;
        vector[offset + 2] = hero.count - opponent.count;
        vector[offset + 3] = hero.slots;
        vector[offset + 4] = opponent.slots;
        vector[offset + 5] = hero.category;
        vector[offset + 6] = opponent.category;
        vector[offset + 7] = hero.category - opponent.category;
        vector[offset + 8] = hero.royalty;
        vector[offset + 9] = opponent.royalty;
        vector[offset + 10] = hero.total_royalty / 100.0 - opponent.total_royalty / 100.0;
        vector[offset + 11] = hero.premium_potential - opponent.premium_potential;
    }
    let completed_hero = (0..ROWS)
        .filter(|row| row_len(&hero_rows_cards[*row]) == row_capacity(*row))
        .count();
    let completed_opponent = (0..ROWS)
        .filter(|row| row_len(&opponent_rows_cards[*row]) == row_capacity(*row))
        .count();
    let offset = HU_MATCHUP_OFFSET + 36;
    vector[offset] = hero_rows.total_royalty / 100.0 - opponent_rows.total_royalty / 100.0;
    vector[offset + 1] = (completed_hero as f32 - completed_opponent as f32) / 3.0;
    vector[offset + 2] = hero_rows.top_middle_order;
    vector[offset + 3] = hero_rows.middle_bottom_order;
    vector[offset + 4] = opponent_rows.top_middle_order;
    vector[offset + 5] = opponent_rows.middle_bottom_order;
    vector[offset + 6] = hero_rows.top_middle_order - opponent_rows.top_middle_order;
    vector[offset + 7] = hero_rows.middle_bottom_order - opponent_rows.middle_bottom_order;
    vector[offset + 8] = hero_summaries[0].premium_potential - opponent_summaries[0].premium_potential;
    vector[offset + 9] = hero_summaries[1].premium_potential - opponent_summaries[1].premium_potential;
    vector[offset + 10] = hero_summaries[2].premium_potential - opponent_summaries[2].premium_potential;
    vector[offset + 11] = (0..ROWS)
        .map(|row| hero_summaries[row].premium_potential - opponent_summaries[row].premium_potential)
        .sum::<f32>()
        / 3.0;
}

fn write_top_summary(vector: &mut [f32], offset: usize, summary: &TopSummary) {
    vector[offset..offset + 12].copy_from_slice(&summary.values[..12]);
}

fn row_extra_values(row: usize, cards: &[i16]) -> [f32; ROW_EXTRA_DIM] {
    let mut values = [0f32; ROW_EXTRA_DIM];
    let capacity = row_capacity(row);
    let slots = capacity as i32 - row_len(cards) as i32;
    let mut rank_counts = [0i32; RANKS];
    let mut suit_counts = [0i32; SUITS];
    let mut ranks = Vec::new();
    for card in cards.iter().copied().filter(|card| *card >= 0) {
        let rank = rank_value(card as usize);
        ranks.push(rank);
        rank_counts[rank_index(card as usize)] += 1;
        suit_counts[suit_index(card as usize)] += 1;
    }
    values[0] = row_len(cards) as f32 / capacity as f32;
    values[1] = slots.max(0) as f32 / capacity as f32;
    if !ranks.is_empty() {
        values[2] = ranks.iter().sum::<i32>() as f32 / (14.0 * capacity as f32);
        values[3] = *ranks.iter().max().unwrap() as f32 / 14.0;
        values[4] = *ranks.iter().min().unwrap() as f32 / 14.0;
    }
    values[5] = rank_counts.iter().filter(|count| **count > 0).count() as f32 / capacity as f32;
    let max_multiplicity = *rank_counts.iter().max().unwrap_or(&0);
    let pairs: Vec<i32> = (0..RANKS)
        .filter(|rank| rank_counts[*rank] >= 2)
        .map(|rank| rank_value_from_index(rank))
        .collect();
    let trips: Vec<i32> = (0..RANKS)
        .filter(|rank| rank_counts[*rank] >= 3)
        .map(|rank| rank_value_from_index(rank))
        .collect();
    let quads: Vec<i32> = (0..RANKS)
        .filter(|rank| rank_counts[*rank] >= 4)
        .map(|rank| rank_value_from_index(rank))
        .collect();
    values[6] = max_multiplicity as f32 / capacity as f32;
    values[7] = pairs.len().min(2) as f32 / 2.0;
    values[8] = if trips.is_empty() { 0.0 } else { 1.0 };
    values[9] = if quads.is_empty() { 0.0 } else { 1.0 };
    values[10] = pairs.iter().max().copied().unwrap_or(0) as f32 / 14.0;
    values[11] = trips.iter().max().copied().unwrap_or(0) as f32 / 14.0;
    let suit_max = *suit_counts.iter().max().unwrap_or(&0);
    values[12] = suit_max as f32 / capacity as f32;
    if row != 0 && suit_max + slots.max(0) >= 5 {
        values[13] = suit_max as f32 / 5.0;
    }
    let (straight_score, straight_high) = if row != 0 {
        straight_potential(&ranks, slots)
    } else {
        (0.0, 0.0)
    };
    values[14] = straight_score;
    values[15] = straight_high;
    let complete = row_len(cards) == capacity;
    let mut category = 0;
    let mut made_ranks = [0i32; 5];
    let mut made_len = 0usize;
    let mut royalty = 0;
    if complete {
        let hand = if row == 0 {
            evaluate_3(cards)
        } else {
            evaluate_5(cards)
        };
        category = hand.category;
        made_ranks = hand.ranks;
        made_len = hand.len;
        royalty = row_royalty(row, hand);
    }
    values[16] = category as f32 / 8.0;
    values[17] = if made_len > 0 { made_ranks[0] as f32 / 14.0 } else { 0.0 };
    values[18] = if made_len > 1 { made_ranks[1] as f32 / 14.0 } else { 0.0 };
    values[19] = royalty as f32 / royalty_scale(row);
    if row == 0 {
        values[20] = if complete && fl_entry(cards) { 1.0 } else { 0.0 };
        values[21] = top_fl_potential(cards, slots);
    }
    values[22] = if complete { 1.0 } else { 0.0 };
    if row != 0 && complete {
        values[23] = if category >= HAND_FLUSH { 1.0 } else { 0.0 };
    }
    values
}

fn global_extra_values(rows: &[Vec<i16>; ROWS]) -> [f32; 4] {
    let mut values = [0f32; 4];
    values[0] = row_order_feature(0, &rows[0], 1, &rows[1]);
    values[1] = row_order_feature(1, &rows[1], 2, &rows[2]);
    let completed = (0..ROWS)
        .filter(|row| row_len(&rows[*row]) == row_capacity(*row))
        .count();
    values[2] = completed as f32 / 3.0;
    let mut royalty = 0;
    for row in 0..ROWS {
        if row_len(&rows[row]) == row_capacity(row) {
            let hand = if row == 0 { evaluate_3(&rows[row]) } else { evaluate_5(&rows[row]) };
            royalty += row_royalty(row, hand);
        }
    }
    values[3] = royalty as f32 / 100.0;
    values
}

fn top_summary(cards: &[i16], available: &[i32; RANKS]) -> TopSummary {
    let mut values = [0f32; 15];
    let slots = (3i32 - row_len(cards) as i32).max(0);
    let mut counts = [0i32; RANKS];
    for card in cards.iter().copied().filter(|card| *card >= 0) {
        counts[rank_index(card as usize)] += 1;
    }
    let high_pair_ranks: Vec<usize> = [10usize, 11, 12]
        .iter()
        .copied()
        .filter(|rank| counts[*rank] >= 2)
        .collect();
    let max_multiplicity = *counts.iter().max().unwrap_or(&0);
    let complete_fl = if row_len(cards) == 3 && fl_entry(cards) { 1.0 } else { 0.0 };
    let high_pair_potential = [10usize, 11, 12]
        .iter()
        .any(|rank| counts[*rank] + slots.min(available[*rank]) >= 2);
    let trip_potential = (0..RANKS).any(|rank| counts[rank] + slots.min(available[rank]) >= 3);
    let fl_potential = if complete_fl > 0.0 || trip_potential {
        1.0
    } else if high_pair_potential {
        0.75
    } else {
        0.0
    };
    let royalty = if row_len(cards) == 3 { get_top_royalty(cards) } else { 0 };
    values[0] = row_len(cards) as f32 / 3.0;
    values[1] = slots as f32 / 3.0;
    values[2] = complete_fl;
    values[3] = fl_potential;
    values[4] = if high_pair_ranks.is_empty() { 0.0 } else { 1.0 };
    values[5] = high_pair_ranks
        .iter()
        .map(|rank| rank_value_from_index(*rank))
        .max()
        .unwrap_or(0) as f32
        / 14.0;
    values[6] = if counts.iter().any(|count| *count >= 2) { 1.0 } else { 0.0 };
    values[7] = if max_multiplicity >= 3 { 1.0 } else { 0.0 };
    values[8] = if trip_potential { 1.0 } else { 0.0 };
    values[9] = if high_pair_potential || trip_potential { 1.0 } else { 0.0 };
    values[10] = max_multiplicity as f32 / 3.0;
    values[11] = royalty as f32 / 22.0;
    for (idx, rank) in [10usize, 11, 12].iter().copied().enumerate() {
        values[12 + idx] = needs_rank_for_top_pair(rank, &counts, slots, available);
    }
    TopSummary { values }
}

fn complete_summary(rows: &[Vec<i16>; ROWS]) -> CompleteSummary {
    let mut top_royalty = 0;
    let mut middle_royalty = 0;
    let mut bottom_royalty = 0;
    let top = if row_len(&rows[0]) == 3 {
        let hand = evaluate_3(&rows[0]);
        top_royalty = get_top_royalty(&rows[0]);
        Some(hand)
    } else {
        None
    };
    let middle = if row_len(&rows[1]) == 5 {
        let hand = evaluate_5(&rows[1]);
        middle_royalty = get_middle_royalty(hand);
        Some(hand)
    } else {
        None
    };
    let bottom = if row_len(&rows[2]) == 5 {
        let hand = evaluate_5(&rows[2]);
        bottom_royalty = get_bottom_royalty(hand);
        Some(hand)
    } else {
        None
    };
    CompleteSummary {
        top_royalty: top_royalty as f32,
        middle_royalty: middle_royalty as f32,
        bottom_royalty: bottom_royalty as f32,
        total_royalty: (top_royalty + middle_royalty + bottom_royalty) as f32,
        top_middle_order: order_value(top, middle),
        middle_bottom_order: order_value(middle, bottom),
    }
}

fn row_matchup_summary(row: usize, cards: &[i16], available: &[i32; RANKS]) -> MatchupSummary {
    let capacity = row_capacity(row);
    let slots = capacity.saturating_sub(row_len(cards));
    let complete = row_len(cards) == capacity;
    let mut category = 0;
    let mut royalty = 0;
    if complete {
        let hand = if row == 0 { evaluate_3(cards) } else { evaluate_5(cards) };
        category = hand.category;
        royalty = row_royalty(row, hand);
    }
    MatchupSummary {
        count: row_len(cards) as f32 / capacity as f32,
        slots: slots as f32 / capacity as f32,
        category: category as f32 / 8.0,
        royalty: royalty as f32 / royalty_scale(row),
        total_royalty: royalty as f32,
        premium_potential: row_premium_potential(row, cards, slots as i32, category, available),
    }
}

fn row_premium_potential(
    row: usize,
    cards: &[i16],
    slots: i32,
    category: i32,
    available: &[i32; RANKS],
) -> f32 {
    if row == 0 {
        return top_summary(cards, available).values[3];
    }
    let mut ranks = Vec::new();
    let mut rank_counts = [0i32; RANKS];
    let mut suit_counts = [0i32; SUITS];
    for card in cards.iter().copied().filter(|card| *card >= 0) {
        ranks.push(rank_value(card as usize));
        rank_counts[rank_index(card as usize)] += 1;
        suit_counts[suit_index(card as usize)] += 1;
    }
    let made_score = if row_len(cards) == row_capacity(row) {
        category as f32 / 8.0
    } else {
        0.0
    };
    let max_multiplicity = *rank_counts.iter().max().unwrap_or(&0);
    let pair_count = rank_counts.iter().filter(|count| **count >= 2).count();
    let multiplicity_score = if max_multiplicity + slots >= 4 {
        1.0
    } else if max_multiplicity >= 3 && pair_count >= 1 {
        0.85
    } else if max_multiplicity + slots >= 3 {
        0.65
    } else if pair_count >= 2 {
        0.5
    } else if pair_count >= 1 {
        0.35
    } else {
        0.0
    };
    let suit_max = *suit_counts.iter().max().unwrap_or(&0);
    let flush_score = if suit_max + slots >= 5 {
        suit_max as f32 / 5.0
    } else {
        0.0
    };
    let (straight_score, _) = straight_potential(&ranks, slots);
    made_score.max(multiplicity_score).max(flush_score).max(straight_score)
}

fn opponent_needed_top(top: &[i16], available: &[i32; RANKS]) -> [bool; RANKS] {
    let mut result = [false; RANKS];
    let slots = (3i32 - row_len(top) as i32).max(0);
    let mut counts = [0i32; RANKS];
    for card in top.iter().copied().filter(|card| *card >= 0) {
        counts[rank_index(card as usize)] += 1;
    }
    for rank in [10usize, 11, 12] {
        result[rank] = needs_rank_for_top_pair(rank, &counts, slots, available) > 0.0;
    }
    result
}

fn needs_rank_for_top_pair(rank: usize, counts: &[i32; RANKS], slots: i32, available: &[i32; RANKS]) -> f32 {
    if slots <= 0 || counts[rank] >= 2 {
        return 0.0;
    }
    let missing = 2 - counts[rank];
    if missing <= slots.min(available[rank]) {
        1.0
    } else {
        0.0
    }
}

fn board_score_terminal(rows: &[Vec<i16>; ROWS]) -> (f32, f32, f32) {
    let top = evaluate_3(&rows[0]);
    let middle = evaluate_5(&rows[1]);
    let bottom = evaluate_5(&rows[2]);
    let busted = compare_hand(top, middle) == Ordering::Greater || compare_hand(middle, bottom) == Ordering::Greater;
    if busted {
        return (1.0, 0.0, 0.0);
    }
    let royalty = get_top_royalty(&rows[0]) + get_middle_royalty(middle) + get_bottom_royalty(bottom);
    let fl = if fl_entry(&rows[0]) { 1.0 } else { 0.0 };
    (0.0, royalty as f32 / 100.0, fl)
}

fn row_order_feature(left_row: usize, left_cards: &[i16], right_row: usize, right_cards: &[i16]) -> f32 {
    if row_len(left_cards) != row_capacity(left_row) || row_len(right_cards) != row_capacity(right_row) {
        return 0.0;
    }
    let left = if left_row == 0 { evaluate_3(left_cards) } else { evaluate_5(left_cards) };
    let right = if right_row == 0 { evaluate_3(right_cards) } else { evaluate_5(right_cards) };
    if compare_hand(left, right) != Ordering::Greater {
        1.0
    } else {
        -1.0
    }
}

fn order_value(left: Option<HandValue>, right: Option<HandValue>) -> f32 {
    match (left, right) {
        (Some(left), Some(right)) => {
            if compare_hand(left, right) != Ordering::Greater {
                1.0
            } else {
                -1.0
            }
        }
        _ => 0.0,
    }
}

fn evaluate_3(cards: &[i16]) -> HandValue {
    let mut ranks: Vec<i32> = cards.iter().copied().filter(|card| *card >= 0).map(|card| rank_value(card as usize)).collect();
    ranks.sort_by(|a, b| b.cmp(a));
    let counts = rank_counts_from_values(&ranks);
    let groups = sorted_groups(&counts);
    if groups[0].0 == 3 {
        return hand(HAND_TRIPS, &[groups[0].1]);
    }
    if groups[0].0 == 2 {
        let pair = groups[0].1;
        let kicker = ranks.iter().copied().filter(|rank| *rank != pair).max().unwrap_or(0);
        return hand(HAND_PAIR, &[pair, kicker]);
    }
    hand(HAND_HIGH, &ranks)
}

fn evaluate_5(cards: &[i16]) -> HandValue {
    let mut ranks: Vec<i32> = cards.iter().copied().filter(|card| *card >= 0).map(|card| rank_value(card as usize)).collect();
    ranks.sort_by(|a, b| b.cmp(a));
    let counts = rank_counts_from_values(&ranks);
    let groups = sorted_groups(&counts);
    let flush = cards
        .iter()
        .copied()
        .filter(|card| *card >= 0)
        .map(|card| suit_index(card as usize))
        .collect::<Vec<_>>();
    let is_flush = flush.first().map(|first| flush.iter().all(|suit| suit == first)).unwrap_or(false);
    let straight = straight_high(&ranks);
    if is_flush && straight.is_some() {
        return hand(HAND_STRAIGHT_FLUSH, &[straight.unwrap()]);
    }
    if groups[0].0 == 4 {
        let quad = groups[0].1;
        let kicker = ranks.iter().copied().filter(|rank| *rank != quad).max().unwrap_or(0);
        return hand(HAND_QUADS, &[quad, kicker]);
    }
    if groups[0].0 == 3 && groups.get(1).map(|g| g.0).unwrap_or(0) == 2 {
        return hand(HAND_FULL_HOUSE, &[groups[0].1, groups[1].1]);
    }
    if is_flush {
        return hand(HAND_FLUSH, &ranks);
    }
    if let Some(high) = straight {
        return hand(HAND_STRAIGHT, &[high]);
    }
    if groups[0].0 == 3 {
        let trips = groups[0].1;
        let mut values = vec![trips];
        values.extend(ranks.iter().copied().filter(|rank| *rank != trips));
        return hand(HAND_TRIPS, &values);
    }
    let mut pairs: Vec<i32> = counts
        .iter()
        .enumerate()
        .filter(|(_, count)| **count == 2)
        .map(|(rank, _)| rank_value_from_index(rank))
        .collect();
    pairs.sort_by(|a, b| b.cmp(a));
    if pairs.len() == 2 {
        let kicker = ranks
            .iter()
            .copied()
            .filter(|rank| *rank != pairs[0] && *rank != pairs[1])
            .max()
            .unwrap_or(0);
        return hand(HAND_TWO_PAIR, &[pairs[0], pairs[1], kicker]);
    }
    if pairs.len() == 1 {
        let pair = pairs[0];
        let mut values = vec![pair];
        values.extend(ranks.iter().copied().filter(|rank| *rank != pair));
        return hand(HAND_PAIR, &values);
    }
    hand(HAND_HIGH, &ranks)
}

fn hand(category: i32, values: &[i32]) -> HandValue {
    let mut ranks = [0i32; 5];
    for (index, value) in values.iter().copied().take(5).enumerate() {
        ranks[index] = value;
    }
    HandValue {
        category,
        ranks,
        len: values.len().min(5),
    }
}

fn compare_hand(left: HandValue, right: HandValue) -> Ordering {
    if left.category != right.category {
        return left.category.cmp(&right.category);
    }
    let max_len = left.len.max(right.len);
    for index in 0..max_len {
        if left.ranks[index] != right.ranks[index] {
            return left.ranks[index].cmp(&right.ranks[index]);
        }
    }
    Ordering::Equal
}

fn rank_counts_from_values(ranks: &[i32]) -> [i32; RANKS] {
    let mut counts = [0i32; RANKS];
    for rank in ranks {
        counts[(*rank as usize).saturating_sub(2)] += 1;
    }
    counts
}

fn sorted_groups(counts: &[i32; RANKS]) -> Vec<(i32, i32)> {
    let mut groups: Vec<(i32, i32)> = counts
        .iter()
        .enumerate()
        .filter(|(_, count)| **count > 0)
        .map(|(rank, count)| (*count, rank_value_from_index(rank)))
        .collect();
    groups.sort_by(|a, b| b.cmp(a));
    groups
}

fn straight_high(ranks: &[i32]) -> Option<i32> {
    let mut present = [false; 15];
    for rank in ranks {
        present[*rank as usize] = true;
    }
    if present[14] && present[5] && present[4] && present[3] && present[2] {
        return Some(5);
    }
    for high in (6..=14).rev() {
        if (high - 4..=high).all(|rank| present[rank as usize]) {
            return Some(high);
        }
    }
    None
}

fn straight_potential(ranks: &[i32], slots: i32) -> (f32, f32) {
    if slots < 0 {
        return (0.0, 0.0);
    }
    let mut present = [false; 15];
    for rank in ranks {
        present[*rank as usize] = true;
    }
    let mut best_score = 0.0;
    let mut best_high = 0;
    for high in [5].into_iter().chain(6..=14) {
        let sequence: Vec<i32> = if high == 5 {
            vec![14, 5, 4, 3, 2]
        } else {
            (high - 4..=high).collect()
        };
        let missing = sequence.iter().filter(|rank| !present[**rank as usize]).count() as i32;
        if missing <= slots {
            let score = (5 - missing) as f32 / 5.0;
            if score > best_score || (score == best_score && high > best_high) {
                best_score = score;
                best_high = high;
            }
        }
    }
    (best_score, if best_high > 0 { best_high as f32 / 14.0 } else { 0.0 })
}

fn fl_entry(top: &[i16]) -> bool {
    if row_len(top) != 3 {
        return false;
    }
    let mut counts = [0i32; RANKS];
    for card in top.iter().copied().filter(|card| *card >= 0) {
        counts[rank_index(card as usize)] += 1;
    }
    counts.iter().any(|count| *count >= 3) || counts[12] >= 2 || counts[11] >= 2 || counts[10] >= 2
}

fn top_fl_potential(cards: &[i16], slots: i32) -> f32 {
    if slots < 0 {
        return 0.0;
    }
    let mut counts = [0i32; RANKS];
    for card in cards.iter().copied().filter(|card| *card >= 0) {
        counts[rank_index(card as usize)] += 1;
    }
    if counts.iter().any(|count| *count + slots >= 3) {
        return 1.0;
    }
    if slots >= 2 && row_len(cards) <= 1 {
        return 0.75;
    }
    if [12usize, 11, 10].iter().any(|rank| counts[*rank] + slots >= 2) {
        return 0.75;
    }
    0.0
}

fn get_top_royalty(cards: &[i16]) -> i32 {
    let value = evaluate_3(cards);
    if value.category == HAND_TRIPS {
        return 10 + (value.ranks[0] - 2);
    }
    if value.category == HAND_PAIR && value.ranks[0] >= 6 {
        return value.ranks[0] - 5;
    }
    0
}

fn get_middle_royalty(value: HandValue) -> i32 {
    match value.category {
        HAND_STRAIGHT_FLUSH if value.ranks[0] == 14 => 50,
        HAND_STRAIGHT_FLUSH => 30,
        HAND_QUADS => 20,
        HAND_FULL_HOUSE => 12,
        HAND_FLUSH => 8,
        HAND_STRAIGHT => 4,
        HAND_TRIPS => 2,
        _ => 0,
    }
}

fn get_bottom_royalty(value: HandValue) -> i32 {
    match value.category {
        HAND_STRAIGHT_FLUSH if value.ranks[0] == 14 => 25,
        HAND_STRAIGHT_FLUSH => 15,
        HAND_QUADS => 10,
        HAND_FULL_HOUSE => 6,
        HAND_FLUSH => 4,
        HAND_STRAIGHT => 2,
        _ => 0,
    }
}

fn row_royalty(row: usize, value: HandValue) -> i32 {
    match row {
        0 => {
            if value.category == HAND_TRIPS {
                10 + (value.ranks[0] - 2)
            } else if value.category == HAND_PAIR && value.ranks[0] >= 6 {
                value.ranks[0] - 5
            } else {
                0
            }
        }
        1 => get_middle_royalty(value),
        _ => get_bottom_royalty(value),
    }
}

fn row_capacity(row: usize) -> usize {
    if row == 0 {
        3
    } else {
        5
    }
}

fn royalty_scale(row: usize) -> f32 {
    match row {
        0 => 22.0,
        1 => 50.0,
        _ => 25.0,
    }
}

fn rank_index(card: usize) -> usize {
    card % 13
}

fn suit_index(card: usize) -> usize {
    card / 13
}

fn rank_value(card: usize) -> i32 {
    rank_value_from_index(rank_index(card))
}

fn rank_value_from_index(rank: usize) -> i32 {
    rank as i32 + 2
}
