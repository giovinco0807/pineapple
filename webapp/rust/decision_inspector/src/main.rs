use ofc_hu_m3_engine::action::{generate_turn_actions, Action};
use ofc_hu_m3_engine::action_key::{
    action_key, canonical_descending_indices, legal_action_set_digest,
};
use ofc_hu_m3_engine::cards::{validate_cards, Card};
use ofc_hu_m3_engine::infoset::{ActorObservation, ScoringContext};
use ofc_hu_m3_engine::scoring::{
    evaluate_3_card, evaluate_5_card, get_bottom_royalty, get_middle_royalty, get_top_royalty,
    score_board, BoardScore, HandValue, HAND_QUADS, HAND_TRIPS,
};
use ofc_hu_m3_engine::search::score_completed_boards;
use ofc_hu_m3_engine::state::Board;
use ofc_hu_m3_engine::t3_features;
use ofc_hu_m3_engine::t3first_features;
use ofc_hu_m3_engine::t4_model::Model;
use serde::Deserialize;
use serde_json::{json, Value};
use std::cmp::Ordering;
use std::io::{self, Read};

const DETAILED_SCORE_SCHEMA: &str = "ofc_webapp_score_final_detailed_v1";
const FL_TOPK_SCHEMA: &str = "ofc_webapp_fantasyland_topk_v1";
const CANONICAL_SOURCE: &str = "canonical_rust_hu_m3_engine";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    mode: String,
    observation: Option<ActorObservation>,
    model_path: Option<String>,
    model_sha256: Option<String>,
    #[serde(default = "default_top_k")]
    top_k: usize,
    first_board: Option<Board>,
    second_board: Option<Board>,
    scoring: Option<ScoringContext>,
    #[serde(default)]
    first_in_fantasyland: bool,
    #[serde(default)]
    second_in_fantasyland: bool,
    cards: Option<Vec<Card>>,
    stay_bonus: Option<f64>,
}

fn default_top_k() -> usize {
    3
}

fn score_actions(
    mode: &str,
    observation: &ActorObservation,
    actions: &[Action],
    model: &Model,
) -> Result<Vec<f64>, String> {
    let unknown = t3_features::unknown_cards(observation);
    let mut scratch = model.scratch();
    match mode {
        "t3_second" => {
            if observation.hero_board.card_count() != 9
                || observation.opponent_public_board.card_count() != 11
            {
                return Err(
                    "t3_second requires 9-card hero and 11-card opponent boards".to_string()
                );
            }
            let (opponent_outlook, opponent_finishes) =
                t3_features::side_outlook(&observation.opponent_public_board, &unknown);
            actions
                .iter()
                .map(|action| {
                    let board = action.apply(&observation.hero_board)?;
                    let features = t3_features::encode(
                        observation,
                        &board,
                        &unknown,
                        &opponent_outlook,
                        &opponent_finishes,
                    );
                    Ok(f64::from(model.predict_with(&features, &mut scratch)?))
                })
                .collect()
        }
        "t3_first" => {
            if observation.hero_board.card_count() != 9
                || observation.opponent_public_board.card_count() != 9
            {
                return Err("t3_first requires 9-card hero and 9-card opponent boards".to_string());
            }
            let (opponent_block, opponent_finishes) = t3first_features::opponent_outlook_first(
                &observation.opponent_public_board,
                &unknown,
            )?;
            actions
                .iter()
                .map(|action| {
                    let board = action.apply(&observation.hero_board)?;
                    let features = t3first_features::encode_first(
                        observation,
                        &board,
                        &unknown,
                        &opponent_block,
                        &opponent_finishes,
                    );
                    Ok(f64::from(model.predict_with(&features, &mut scratch)?))
                })
                .collect()
        }
        "t2_second" => {
            if observation.hero_board.card_count() != 7
                || observation.opponent_public_board.card_count() != 9
            {
                return Err("t2_second requires 7-card hero and 9-card opponent boards".to_string());
            }
            let (opponent_block, opponent_finishes) = t3first_features::opponent_outlook_first(
                &observation.opponent_public_board,
                &unknown,
            )?;
            actions
                .iter()
                .map(|action| {
                    let board = action.apply(&observation.hero_board)?;
                    let mut features = [0.0f32; t3first_features::FEATURE_SIZE];
                    features[..86]
                        .copy_from_slice(&t3_features::encode_structural(observation, &board));
                    let (hero_block, hero_finishes) =
                        t3first_features::opponent_outlook_first(&board, &unknown)?;
                    features[86..122].copy_from_slice(&hero_block);
                    features[122..158].copy_from_slice(&opponent_block);
                    features[158..].copy_from_slice(&t3_features::head_to_head(
                        &hero_finishes,
                        &opponent_finishes,
                    ));
                    Ok(f64::from(model.predict_with(&features, &mut scratch)?))
                })
                .collect()
        }
        _ => Err(format!("unsupported inspector mode: {mode}")),
    }
}

fn inspect_model(request: &Request) -> Result<Value, String> {
    if request.top_k == 0 {
        return Err("top_k must be positive".to_string());
    }
    let observation = request
        .observation
        .as_ref()
        .ok_or_else(|| format!("{} requires observation", request.mode))?;
    observation.validate()?;
    let actions = generate_turn_actions(&observation.hero_board, &observation.dealt_cards)?;
    if actions.is_empty() {
        return Err("observation has no legal actions".to_string());
    }
    let model_path = request
        .model_path
        .as_deref()
        .ok_or_else(|| format!("{} requires model_path", request.mode))?;
    let model_sha256 = request
        .model_sha256
        .as_deref()
        .ok_or_else(|| format!("{} requires model_sha256", request.mode))?;
    let bytes = std::fs::read(model_path)
        .map_err(|error| format!("cannot read model {model_path}: {error}"))?;
    let model = Model::load_pinned(&bytes, model_sha256)?;
    let values = score_actions(&request.mode, observation, &actions, &model)?;
    let ranked = canonical_descending_indices(&values, &actions)?;
    let rows = ranked
        .iter()
        .take(request.top_k)
        .map(|&index| {
            let action = &actions[index];
            Ok(json!({
                "rank": rows_rank(&ranked, index),
                "action_key": action_key(action)?.to_token(),
                "placements": action.placements,
                "discards": action.discards,
                "score": values[index],
            }))
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(json!({
        "status": "ok",
        "schema": "ofc_webapp_decision_inspector_v1",
        "mode": request.mode,
        "model_sha256": model_sha256,
        "legal_action_count": actions.len(),
        "legal_action_set_digest": legal_action_set_digest(&actions)?,
        "scores_topk": rows,
    }))
}

fn inspect_score_final_detailed(request: &Request) -> Result<Value, String> {
    let first = request
        .first_board
        .as_ref()
        .ok_or_else(|| "score_final_detailed requires first_board".to_string())?;
    let second = request
        .second_board
        .as_ref()
        .ok_or_else(|| "score_final_detailed requires second_board".to_string())?;
    let scoring = request
        .scoring
        .as_ref()
        .ok_or_else(|| "score_final_detailed requires scoring".to_string())?;
    let fl_ev = scoring
        .fl_ev
        .get(&14)
        .copied()
        .ok_or_else(|| "scoring must define 14-card FL EV".to_string())?;
    if fl_ev.abs() > 1e-12 {
        return Err(
            "score_final_detailed requires zero 14-card FL EV for table settlement".to_string(),
        );
    }

    // This is the same canonical function reached by the score_final FFI path.
    let official = score_completed_boards(first, second, scoring)?;
    let hu_score = official
        .get("hu_score")
        .and_then(Value::as_f64)
        .ok_or_else(|| "canonical score_final response lacks numeric hu_score".to_string())?;
    let first_score = score_board(first)?;
    let second_score = score_board(second)?;

    let (row_results, line_total) = if first_score.busted || second_score.busted {
        (
            json!({
                "top": "not_scored_foul",
                "middle": "not_scored_foul",
                "bottom": "not_scored_foul",
            }),
            0,
        )
    } else {
        let top = compare_row(&first_score.top_value, &second_score.top_value);
        let middle = compare_row(&first_score.middle_value, &second_score.middle_value);
        let bottom = compare_row(&first_score.bottom_value, &second_score.bottom_value);
        (
            json!({
                "top": top.0,
                "middle": middle.0,
                "bottom": bottom.0,
            }),
            top.1 + middle.1 + bottom.1,
        )
    };
    let foul_base = match (first_score.busted, second_score.busted) {
        (true, false) => -6,
        (false, true) => 6,
        _ => 0,
    };
    let scoop_bonus = if !first_score.busted && !second_score.busted {
        match line_total {
            3 => 3,
            -3 => -3,
            _ => 0,
        }
    } else {
        0
    };
    let royalty_delta = first_score.total_royalty - second_score.total_royalty;
    let total = foul_base + line_total + scoop_bonus + royalty_delta;
    if (hu_score - f64::from(total)).abs() > 1e-9 {
        return Err(format!(
            "canonical point components disagree with score_final: components={total}, hu_score={hu_score}"
        ));
    }

    let first_fl = fantasyland_result(&first_score, request.first_in_fantasyland);
    let second_fl = fantasyland_result(&second_score, request.second_in_fantasyland);
    Ok(json!({
        "status": "ok",
        "schema": DETAILED_SCORE_SCHEMA,
        "mode": "score_final_detailed",
        "source": CANONICAL_SOURCE,
        "engine_version": official.get("engine_version").cloned().unwrap_or(Value::Null),
        "hu_score": hu_score,
        "fouls": {
            "first": first_score.busted,
            "second": second_score.busted,
        },
        "hand_values": {
            "first": hand_values_json(&first_score),
            "second": hand_values_json(&second_score),
        },
        "row_results": row_results,
        "scoop": {
            "first": scoop_bonus > 0,
            "second": scoop_bonus < 0,
        },
        "royalties": {
            "first": royalties_json(&first_score),
            "second": royalties_json(&second_score),
        },
        "point_components": {
            "perspective": "first",
            "foul_base": foul_base,
            "line_total": line_total,
            "scoop_bonus": scoop_bonus,
            "royalty_delta": royalty_delta,
            "total": total,
        },
        "fantasyland": {
            "first": first_fl,
            "second": second_fl,
            "first_next": first_fl["qualifies"],
            "second_next": second_fl["qualifies"],
            "cards": 14,
        },
    }))
}

fn compare_row(first: &HandValue, second: &HandValue) -> (&'static str, i32) {
    match first.cmp(second) {
        Ordering::Greater => ("first", 1),
        Ordering::Less => ("second", -1),
        Ordering::Equal => ("tie", 0),
    }
}

fn hand_value_json(value: &HandValue) -> Value {
    json!({
        "category": value.0,
        "tie_breakers": value.1,
    })
}

fn hand_values_json(score: &BoardScore) -> Value {
    json!({
        "top": hand_value_json(&score.top_value),
        "middle": hand_value_json(&score.middle_value),
        "bottom": hand_value_json(&score.bottom_value),
    })
}

fn royalties_json(score: &BoardScore) -> Value {
    json!({
        "top": score.top_royalty,
        "middle": score.middle_royalty,
        "bottom": score.bottom_royalty,
        "total": score.total_royalty,
    })
}

fn fantasyland_result(score: &BoardScore, already_in: bool) -> Value {
    if score.busted {
        return json!({
            "qualifies": false,
            "card_count": 0,
            "entry_type": Value::Null,
        });
    }
    if already_in {
        let entry_type = if score.top_value.0 == HAND_TRIPS {
            Some("stay_top_trips")
        } else if score.bottom_value.0 >= HAND_QUADS {
            Some("stay_bottom_quads_plus")
        } else {
            None
        };
        return match entry_type {
            Some(entry_type) => json!({
                "qualifies": true,
                "card_count": 14,
                "entry_type": entry_type,
            }),
            None => json!({
                "qualifies": false,
                "card_count": 0,
                "entry_type": Value::Null,
            }),
        };
    }
    json!({
        "qualifies": score.fl_entry.qualifies,
        "card_count": score.fl_entry.card_count,
        "entry_type": score.fl_entry.entry_type,
    })
}

#[derive(Clone)]
struct Combo3 {
    mask: u16,
    cards: [Card; 3],
    value: HandValue,
    top_royalty: i32,
    top_stay: bool,
}

#[derive(Clone)]
struct Combo5 {
    mask: u16,
    cards: [Card; 5],
    value: HandValue,
    middle_royalty: i32,
    bottom_royalty: i32,
    bottom_stay: bool,
}

#[derive(Clone)]
struct FlCandidate {
    encounter_index: usize,
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

fn inspect_fantasyland_topk(request: &Request) -> Result<Value, String> {
    if request.top_k != 3 {
        return Err("fantasyland_topk requires top_k=3".to_string());
    }
    let cards = request
        .cards
        .as_ref()
        .ok_or_else(|| "fantasyland_topk requires cards".to_string())?;
    if cards.len() != 14 {
        return Err(format!(
            "fantasyland_topk requires exactly 14 cards, got {}",
            cards.len()
        ));
    }
    validate_cards(cards)?;
    let stay_bonus = request
        .stay_bonus
        .ok_or_else(|| "fantasyland_topk requires stay_bonus".to_string())?;
    if !stay_bonus.is_finite() {
        return Err("fantasyland_topk stay_bonus must be finite".to_string());
    }
    let hand: [Card; 14] = cards
        .clone()
        .try_into()
        .map_err(|_| "fantasyland_topk requires exactly 14 cards".to_string())?;
    let (ranked, legal_count) = solve_fantasyland_topk(&hand, stay_bonus, 3)?;
    if ranked.len() != 3 {
        return Err(format!(
            "fantasyland_topk found only {} legal placements",
            ranked.len()
        ));
    }
    let rows = ranked
        .iter()
        .enumerate()
        .map(|(index, candidate)| candidate_json(candidate, index + 1))
        .collect::<Vec<_>>();
    Ok(json!({
        "status": "ok",
        "schema": FL_TOPK_SCHEMA,
        "mode": "fantasyland_topk",
        "source": CANONICAL_SOURCE,
        "stay_bonus": stay_bonus,
        "legal_placement_count": legal_count,
        "scores_topk": rows,
    }))
}

fn solve_fantasyland_topk(
    cards: &[Card; 14],
    stay_bonus: f64,
    top_k: usize,
) -> Result<(Vec<FlCandidate>, usize), String> {
    let top_combos = build_combo3(cards)?;
    let five_combos = build_combo5(cards)?;
    let all_mask = (1_u16 << 14) - 1;
    let mut tops_by_remaining = vec![Vec::<usize>::new(); 1 << 14];
    for (top_index, top) in top_combos.iter().enumerate() {
        for extra_index in 0..14 {
            let extra_bit = 1_u16 << extra_index;
            if top.mask & extra_bit == 0 {
                tops_by_remaining[(top.mask | extra_bit) as usize].push(top_index);
            }
        }
    }

    // This traversal is intentionally identical to regular_fl_solver:
    // bottom combo, middle combo, then the four possible top combos in their
    // original input order. The mask index avoids scanning all 364 top combos.
    let mut best: Vec<FlCandidate> = Vec::with_capacity(top_k + 1);
    let mut legal_count = 0usize;
    for bottom in &five_combos {
        let remaining_after_bottom = all_mask ^ bottom.mask;
        for middle in &five_combos {
            if middle.mask & bottom.mask != 0 || middle.value > bottom.value {
                continue;
            }
            let remaining_after_middle = remaining_after_bottom ^ middle.mask;
            if remaining_after_middle.count_ones() != 4 {
                continue;
            }
            for &top_index in &tops_by_remaining[remaining_after_middle as usize] {
                let top = &top_combos[top_index];
                if top.value > middle.value {
                    continue;
                }
                let discard_mask = remaining_after_middle ^ top.mask;
                let total = top.top_royalty + middle.middle_royalty + bottom.bottom_royalty;
                let can_stay = top.top_stay || bottom.bottom_stay;
                let score = f64::from(total) + if can_stay { stay_bonus } else { 0.0 };
                let candidate = FlCandidate {
                    encounter_index: legal_count,
                    top: top.cards,
                    middle: middle.cards,
                    bottom: bottom.cards,
                    discard: cards[discard_mask.trailing_zeros() as usize],
                    top_royalty: top.top_royalty,
                    middle_royalty: middle.middle_royalty,
                    bottom_royalty: bottom.bottom_royalty,
                    total_royalty: total,
                    can_stay,
                    score,
                };
                legal_count += 1;
                if best.len() < top_k
                    || candidate_is_better(&candidate, best.last().expect("non-empty top-k"))
                {
                    best.push(candidate);
                    best.sort_by(compare_fl_candidates);
                    best.truncate(top_k);
                }
            }
        }
    }
    Ok((best, legal_count))
}

fn candidate_is_better(left: &FlCandidate, right: &FlCandidate) -> bool {
    match left
        .score
        .partial_cmp(&right.score)
        .unwrap_or(Ordering::Equal)
    {
        Ordering::Greater => true,
        Ordering::Less => false,
        Ordering::Equal => {
            left.total_royalty > right.total_royalty
                || (left.total_royalty == right.total_royalty
                    && left.encounter_index < right.encounter_index)
        }
    }
}

fn compare_fl_candidates(left: &FlCandidate, right: &FlCandidate) -> Ordering {
    if candidate_is_better(left, right) {
        Ordering::Less
    } else if candidate_is_better(right, left) {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}

fn build_combo3(cards: &[Card; 14]) -> Result<Vec<Combo3>, String> {
    let mut combinations = Vec::with_capacity(364);
    for a in 0..12 {
        for b in (a + 1)..13 {
            for c in (b + 1)..14 {
                let hand = [cards[a], cards[b], cards[c]];
                let value = evaluate_3_card(&hand)?;
                combinations.push(Combo3 {
                    mask: mask_for(&[a, b, c]),
                    cards: hand,
                    top_royalty: get_top_royalty(&hand)?,
                    top_stay: value.0 == HAND_TRIPS,
                    value,
                });
            }
        }
    }
    Ok(combinations)
}

fn build_combo5(cards: &[Card; 14]) -> Result<Vec<Combo5>, String> {
    let mut combinations = Vec::with_capacity(2002);
    for a in 0..10 {
        for b in (a + 1)..11 {
            for c in (b + 1)..12 {
                for d in (c + 1)..13 {
                    for e in (d + 1)..14 {
                        let hand = [cards[a], cards[b], cards[c], cards[d], cards[e]];
                        let value = evaluate_5_card(&hand)?;
                        combinations.push(Combo5 {
                            mask: mask_for(&[a, b, c, d, e]),
                            cards: hand,
                            middle_royalty: get_middle_royalty(&hand)?,
                            bottom_royalty: get_bottom_royalty(&hand)?,
                            bottom_stay: value.0 >= HAND_QUADS,
                            value,
                        });
                    }
                }
            }
        }
    }
    Ok(combinations)
}

fn mask_for(indices: &[usize]) -> u16 {
    indices
        .iter()
        .fold(0_u16, |mask, index| mask | (1_u16 << index))
}

fn candidate_json(candidate: &FlCandidate, rank: usize) -> Value {
    let placements = candidate
        .top
        .iter()
        .map(|card| json!([card, "top"]))
        .chain(candidate.middle.iter().map(|card| json!([card, "middle"])))
        .chain(candidate.bottom.iter().map(|card| json!([card, "bottom"])))
        .collect::<Vec<_>>();
    json!({
        "rank": rank,
        "action_key": format!(
            "fl:{}|{}|{}|{}",
            card_tokens(&candidate.top),
            card_tokens(&candidate.middle),
            card_tokens(&candidate.bottom),
            candidate.discard,
        ),
        "placements": placements,
        "discards": [candidate.discard],
        "score": candidate.score,
        "royalties": {
            "top": candidate.top_royalty,
            "middle": candidate.middle_royalty,
            "bottom": candidate.bottom_royalty,
            "total": candidate.total_royalty,
        },
        "can_stay": candidate.can_stay,
        "encounter_index": candidate.encounter_index,
    })
}

fn card_tokens<const N: usize>(cards: &[Card; N]) -> String {
    cards
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

fn rows_rank(ranked: &[usize], index: usize) -> usize {
    ranked
        .iter()
        .position(|candidate| *candidate == index)
        .expect("ranked contains every action")
        + 1
}

fn inspect(request: Request) -> Result<Value, String> {
    match request.mode.as_str() {
        "score_final_detailed" => inspect_score_final_detailed(&request),
        "fantasyland_topk" => inspect_fantasyland_topk(&request),
        _ => inspect_model(&request),
    }
}

fn real_main() -> Result<(), String> {
    let mut raw = String::new();
    io::stdin()
        .read_to_string(&mut raw)
        .map_err(|error| format!("cannot read request: {error}"))?;
    let request: Request =
        serde_json::from_str(&raw).map_err(|error| format!("invalid request JSON: {error}"))?;
    let response = inspect(request)?;
    println!(
        "{}",
        serde_json::to_string(&response)
            .map_err(|error| format!("cannot serialize response: {error}"))?
    );
    Ok(())
}

fn main() {
    if let Err(error) = real_main() {
        eprintln!("ofc_webapp_decision_inspector: {error}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn board(top: &[&str], middle: &[&str], bottom: &[&str]) -> Board {
        Board::new(
            top.iter().map(|card| card.parse().unwrap()).collect(),
            middle.iter().map(|card| card.parse().unwrap()).collect(),
            bottom.iter().map(|card| card.parse().unwrap()).collect(),
        )
        .unwrap()
    }

    fn detailed_request(first: Board, second: Board, first_in_fantasyland: bool) -> Request {
        let mut scoring = ScoringContext::default();
        scoring.fl_ev.insert(14, 0.0);
        Request {
            mode: "score_final_detailed".to_string(),
            observation: None,
            model_path: None,
            model_sha256: None,
            top_k: 3,
            first_board: Some(first),
            second_board: Some(second),
            scoring: Some(scoring),
            first_in_fantasyland,
            second_in_fantasyland: false,
            cards: None,
            stay_bonus: None,
        }
    }

    #[test]
    fn detailed_score_components_sum_to_canonical_hu_score() {
        let first = board(
            &["Qh", "Qd", "2c"],
            &["3h", "4d", "5c", "6s", "7h"],
            &["8h", "9h", "Th", "Jh", "Kh"],
        );
        let second = board(
            &["2h", "2d", "3c"],
            &["4h", "4c", "5d", "6c", "7s"],
            &["8d", "8c", "9d", "9c", "Ts"],
        );
        let response = inspect_score_final_detailed(&detailed_request(first, second, false))
            .expect("valid boards score");
        assert_eq!(response["source"], CANONICAL_SOURCE);
        assert_eq!(response["hu_score"], 21.0);
        assert_eq!(response["point_components"]["line_total"], 3);
        assert_eq!(response["point_components"]["scoop_bonus"], 3);
        assert_eq!(response["point_components"]["royalty_delta"], 15);
        assert_eq!(response["point_components"]["total"], 21);
        assert_eq!(response["fantasyland"]["first_next"], true);
    }

    #[test]
    fn foul_uses_base_six_without_fake_rows_or_scoop() {
        let first = board(
            &["Ah", "Ad", "Kc"],
            &["2h", "3d", "4c", "5s", "7h"],
            &["8h", "8d", "9c", "Th", "Js"],
        );
        let second = board(
            &["2d", "2c", "3h"],
            &["4h", "4d", "5c", "6h", "7d"],
            &["9h", "9d", "Tc", "Jc", "Qs"],
        );
        let response = inspect_score_final_detailed(&detailed_request(first, second, false))
            .expect("foul scores");
        assert_eq!(response["hu_score"], -6.0);
        assert_eq!(response["point_components"]["foul_base"], -6);
        assert_eq!(response["point_components"]["line_total"], 0);
        assert_eq!(response["point_components"]["scoop_bonus"], 0);
        assert_eq!(response["row_results"]["top"], "not_scored_foul");
        assert_eq!(response["scoop"]["second"], false);
    }

    #[test]
    fn current_fl_uses_stay_rule_not_qq_entry_rule() {
        let first = board(
            &["Qh", "Qd", "2c"],
            &["3h", "4d", "5c", "6s", "7h"],
            &["8h", "9h", "Th", "Jh", "Kh"],
        );
        let second = board(
            &["2h", "2d", "3c"],
            &["4h", "4c", "5d", "6c", "7s"],
            &["8d", "8c", "9d", "9c", "Ts"],
        );
        let response = inspect_score_final_detailed(&detailed_request(first, second, true))
            .expect("valid boards score");
        assert_eq!(response["fantasyland"]["first_next"], false);
    }

    #[test]
    fn fantasyland_topk_has_three_real_sorted_candidates() {
        let cards = [
            "Ah", "Ad", "Kh", "Kd", "Qh", "Qd", "Jh", "Th", "9h", "8h", "7h", "6h", "5h", "4h",
        ]
        .map(|card| card.parse::<Card>().unwrap());
        let (ranked, legal_count) =
            solve_fantasyland_topk(&cards, 10.227_020_614_683_454, 3).expect("FL hand solves");
        assert_eq!(ranked.len(), 3);
        assert!(legal_count > 3);
        assert!(candidate_is_better(&ranked[0], &ranked[1]));
        assert!(candidate_is_better(&ranked[1], &ranked[2]));
        assert!(ranked.iter().all(|candidate| candidate.score.is_finite()));
    }
}
