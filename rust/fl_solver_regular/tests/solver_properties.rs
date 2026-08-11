//! Properties the Fantasyland solver must hold for every deal, plus
//! hand-constructed stay cases.

use fl_solver_regular::cards::{mask_of, parse_cards};
use fl_solver_regular::distribution::sample_deal;
use fl_solver_regular::eval::{eval3, eval5, fl_stay, is_foul};
use fl_solver_regular::objective::{ObjectiveConfig, ObjectiveKind, ReferenceSet};
use fl_solver_regular::solver::{solve_brute_force, FlSolver};

fn pure(fl_ev_stay: f64) -> ObjectiveConfig {
    ObjectiveConfig {
        kind: ObjectiveKind::PureV1,
        fl_ev_stay,
        fl_ev_config_path: "test-only".to_owned(),
        fl_ev_cards: 14,
        hero_foul_prob: 0.2482,
        hero_foul_prob_provenance: "test-only".to_owned(),
        reference: None,
    }
}

fn line_equity(fl_ev_stay: f64) -> ObjectiveConfig {
    ObjectiveConfig {
        kind: ObjectiveKind::LineEquityV1,
        fl_ev_stay,
        fl_ev_config_path: "test-only".to_owned(),
        fl_ev_cards: 14,
        hero_foul_prob: 0.2482,
        hero_foul_prob_provenance: "test-only".to_owned(),
        reference: Some(ReferenceSet::uniform_random_legal(64, 997_000_900)),
    }
}

#[test]
fn a_solved_placement_never_fouls_and_uses_every_card_once() {
    let config = pure(9.109);
    let mut solver = FlSolver::new(&config).unwrap();
    for index in 0..3_000_u64 {
        let hand = sample_deal(0, 997_000_400, index).unwrap();
        let solution = solver.solve(&hand).expect("14 cards always place legally");

        assert!(
            !is_foul(
                solution.top_key,
                solution.middle_key,
                solution.bottom_key
            ),
            "solver fouled on deal {index}"
        );

        let mut used: Vec<u8> = solution
            .top
            .iter()
            .chain(solution.middle.iter())
            .chain(solution.bottom.iter())
            .copied()
            .collect();
        used.push(solution.discard);
        used.sort_unstable();
        let mut expected = hand.to_vec();
        expected.sort_unstable();
        assert_eq!(used, expected, "card conservation failed on deal {index}");
        assert_eq!(mask_of(&used).count_ones(), 14);
    }
}

#[test]
fn reported_row_keys_and_royalties_match_a_fresh_evaluation() {
    let config = pure(9.109);
    let mut solver = FlSolver::new(&config).unwrap();
    for index in 0..2_000_u64 {
        let hand = sample_deal(0, 997_000_401, index).unwrap();
        let solution = solver.solve(&hand).unwrap();
        assert_eq!(solution.top_key, eval3(&solution.top));
        assert_eq!(solution.middle_key, eval5(&solution.middle));
        assert_eq!(solution.bottom_key, eval5(&solution.bottom));
        assert_eq!(
            solution.total_royalty,
            solution.top_royalty + solution.middle_royalty + solution.bottom_royalty
        );
        assert_eq!(
            solution.stay.as_deref(),
            fl_stay(solution.top_key, solution.bottom_key)
        );
        // The reported score must be reproducible from the reported parts.
        let expected = solution.total_royalty as f64
            + if solution.stay.is_some() { 9.109 } else { 0.0 };
        assert!((solution.score - expected).abs() < 1e-12);
    }
}

#[test]
fn stay_detection_is_the_repo_rule_on_hand_constructed_cases() {
    // ofc_regular.rules.check_fl_stay: trips on top, or quads-or-better on the
    // bottom. Nothing else stays -- notably not a QQ/KK/AA top, which is what
    // *entry* needs.
    let three = |text: &str| -> [u8; 3] { parse_cards(text).unwrap().try_into().unwrap() };
    let five = |text: &str| -> [u8; 5] { parse_cards(text).unwrap().try_into().unwrap() };

    let cases: [(&str, &str, Option<&str>); 8] = [
        ("2h 2d 2c", "3h 4d 5c 6s 8h", Some("stay_top_trips")),
        ("Ah Ad Ac", "3h 4d 5c 6s 8h", Some("stay_top_trips")),
        ("Ah Ad Kc", "3h 4d 5c 6s 8h", None),
        ("Kh Kd 2c", "3h 4d 5c 6s 8h", None),
        ("Qh Qd 2c", "3h 4d 5c 6s 8h", None),
        ("2h 3d 4c", "5h 5d 5c 5s 8h", Some("stay_bottom_quads_plus")),
        ("2h 3d 4c", "5h 6h 7h 8h 9h", Some("stay_bottom_quads_plus")),
        // A full house on the bottom is a big royalty but not a stay.
        ("2h 3d 4c", "5h 5d 5c 8s 8h", None),
    ];
    for (top, bottom, expected) in cases {
        assert_eq!(
            fl_stay(eval3(&three(top)), eval5(&five(bottom))),
            expected,
            "stay disagreed for top {top:?} bottom {bottom:?}"
        );
    }
}

#[test]
fn a_bigger_stay_bonus_never_lowers_the_stay_rate() {
    // Monotonicity: paying more for a stay cannot make the solver stay less.
    let mut cheap = FlSolver::new(&pure(0.0)).unwrap();
    let mut dear = FlSolver::new(&pure(40.0)).unwrap();
    let mut cheap_stays = 0_usize;
    let mut dear_stays = 0_usize;
    for index in 0..2_000_u64 {
        let hand = sample_deal(0, 997_000_402, index).unwrap();
        if cheap.solve(&hand).unwrap().stays() {
            cheap_stays += 1;
        }
        if dear.solve(&hand).unwrap().stays() {
            dear_stays += 1;
        }
    }
    assert!(
        dear_stays >= cheap_stays,
        "stay rate fell when the bonus rose: {cheap_stays} -> {dear_stays}"
    );
}

#[test]
fn pruned_search_is_exact_against_brute_force_pure_objective() {
    // The headline exactness claim, on the objective that ships by default.
    let config = pure(9.109);
    let mut solver = FlSolver::new(&config).unwrap();
    for index in 0..120_u64 {
        let hand = sample_deal(0, 997_000_403, index).unwrap();
        let fast = solver.solve(&hand).unwrap();
        let slow = solve_brute_force(&hand, &config).unwrap();
        assert_eq!(
            fast, slow,
            "pruned search disagreed with brute force on deal {index}"
        );
        assert_eq!(
            fast.score.to_bits(),
            slow.score.to_bits(),
            "scores differed in the last bit on deal {index}"
        );
    }
}

#[test]
fn pruned_search_is_exact_against_brute_force_line_equity_objective() {
    let config = line_equity(9.109);
    let mut solver = FlSolver::new(&config).unwrap();
    for index in 0..40_u64 {
        let hand = sample_deal(0, 997_000_404, index).unwrap();
        let fast = solver.solve(&hand).unwrap();
        let slow = solve_brute_force(&hand, &config).unwrap();
        assert_eq!(
            fast, slow,
            "pruned search disagreed with brute force on deal {index}"
        );
        assert_eq!(fast.score.to_bits(), slow.score.to_bits());
    }
}

#[test]
fn the_two_objectives_disagree_at_least_sometimes() {
    // If the line term never changed a decision it would not be worth having;
    // this pins that it is actually wired into the argmax.
    let pure_config = pure(9.109);
    let line_config = line_equity(9.109);
    let mut pure_solver = FlSolver::new(&pure_config).unwrap();
    let mut line_solver = FlSolver::new(&line_config).unwrap();
    let mut differences = 0_usize;
    for index in 0..500_u64 {
        let hand = sample_deal(0, 997_000_405, index).unwrap();
        let left = pure_solver.solve(&hand).unwrap();
        let right = line_solver.solve(&hand).unwrap();
        if left.arrangement_key() != right.arrangement_key() {
            differences += 1;
        }
        // Neither objective may foul.
        assert!(!is_foul(right.top_key, right.middle_key, right.bottom_key));
    }
    assert!(
        differences > 0,
        "line equity never changed a placement in 500 deals"
    );
}
