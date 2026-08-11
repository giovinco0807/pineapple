//! Heads-up terminal scoring, including the asymmetric Fantasyland case.
//!
//! `ofc_hu_m3_engine::scoring::heads_up_terminal_score` awards its Fantasyland
//! bonus to whichever side shows QQ+ on top. That is right when both players
//! arrived normally, and wrong when one of them is already in Fantasyland: a
//! player in Fantasyland continues only under the *stay* rule, and a QQ top
//! buys them nothing. `ofc_regular.estimate_hu_fl_ev_direct.score_fl_vs_normal`
//! is the repo's statement of the asymmetric case, and [`hero_vs_fl_score`]
//! reproduces it from the hero's side.
//!
//! [`symmetric_hu_score`] keeps the ordinary both-normal rule and exists so
//! `tests/engine_parity.rs` can pin this module against the engine.

use crate::eval::{
    bottom_royalty, eval3, eval5, fl_entry_from_top, fl_stay, is_foul, middle_royalty,
    top_royalty, HandKey,
};

/// A scored 13-card board.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoardScore {
    pub fouled: bool,
    pub top_key: HandKey,
    pub middle_key: HandKey,
    pub bottom_key: HandKey,
    pub total_royalty: i32,
}

impl BoardScore {
    pub fn evaluate(top: &[u8; 3], middle: &[u8; 5], bottom: &[u8; 5]) -> Self {
        let top_key = eval3(top);
        let middle_key = eval5(middle);
        let bottom_key = eval5(bottom);
        Self::from_keys(top_key, middle_key, bottom_key)
    }

    pub fn from_keys(top_key: HandKey, middle_key: HandKey, bottom_key: HandKey) -> Self {
        let fouled = is_foul(top_key, middle_key, bottom_key);
        let total_royalty = if fouled {
            0
        } else {
            top_royalty(top_key) + middle_royalty(middle_key) + bottom_royalty(bottom_key)
        };
        Self {
            fouled,
            top_key,
            middle_key,
            bottom_key,
            total_royalty,
        }
    }

    /// Ordinary QQ+ Fantasyland entry; zero on a fouled board.
    pub fn enters_fantasyland(&self) -> bool {
        !self.fouled && fl_entry_from_top(self.top_key).is_some()
    }

    /// Fantasyland stay for a board played *from* Fantasyland.
    pub fn stays_in_fantasyland(&self) -> bool {
        !self.fouled && fl_stay(self.top_key, self.bottom_key).is_some()
    }
}

#[inline(always)]
fn line_total(own: &BoardScore, other: &BoardScore) -> i32 {
    let mut total = 0_i32;
    for (own_key, other_key) in [
        (own.top_key, other.top_key),
        (own.middle_key, other.middle_key),
        (own.bottom_key, other.bottom_key),
    ] {
        total += match own_key.cmp(&other_key) {
            std::cmp::Ordering::Greater => 1,
            std::cmp::Ordering::Less => -1,
            std::cmp::Ordering::Equal => 0,
        };
    }
    total
}

#[inline(always)]
fn combine(own: &BoardScore, other: &BoardScore, own_fl: f64, other_fl: f64) -> f64 {
    let own_royalty = if own.fouled { 0 } else { own.total_royalty };
    let other_royalty = if other.fouled { 0 } else { other.total_royalty };
    let own_fl = if own.fouled { 0.0 } else { own_fl };
    let other_fl = if other.fouled { 0.0 } else { other_fl };

    if own.fouled && other.fouled {
        return 0.0;
    }
    if own.fouled {
        return -6.0 - other_royalty as f64 - other_fl;
    }
    if other.fouled {
        return 6.0 + own_royalty as f64 + own_fl;
    }
    let lines = line_total(own, other);
    let scoop = if lines.abs() == 3 { 3 * lines.signum() } else { 0 };
    (lines + scoop + own_royalty - other_royalty) as f64 + own_fl - other_fl
}

/// Both players arrived normally: each side's Fantasyland value is ordinary
/// QQ+ entry. Mirrors `heads_up_terminal_score` in the engine.
pub fn symmetric_hu_score(own: &BoardScore, other: &BoardScore, fl_ev: f64) -> f64 {
    let own_fl = if own.enters_fantasyland() { fl_ev } else { 0.0 };
    let other_fl = if other.enters_fantasyland() { fl_ev } else { 0.0 };
    combine(own, other, own_fl, other_fl)
}

/// Score a normal hero against an opponent who is already in Fantasyland,
/// from the hero's side.
///
/// The hero's next-Fantasyland value comes from ordinary QQ+ entry; the
/// Fantasyland opponent's comes from the stay rule. Both are priced at the
/// same `fl_ev`, which is what `score_fl_vs_normal` does.
pub fn hero_vs_fl_score(hero: &BoardScore, fantasyland: &BoardScore, fl_ev: f64) -> f64 {
    let hero_fl = if hero.enters_fantasyland() { fl_ev } else { 0.0 };
    let opponent_fl = if fantasyland.stays_in_fantasyland() {
        fl_ev
    } else {
        0.0
    };
    combine(hero, fantasyland, hero_fl, opponent_fl)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cards::parse_cards;

    fn board(top: &str, middle: &str, bottom: &str) -> BoardScore {
        BoardScore::evaluate(
            &parse_cards(top).unwrap().try_into().unwrap(),
            &parse_cards(middle).unwrap().try_into().unwrap(),
            &parse_cards(bottom).unwrap().try_into().unwrap(),
        )
    }

    #[test]
    fn symmetric_score_reproduces_the_engine_twenty_one_fixture() {
        // The fixture pinned in hu_m3_engine::scoring tests.
        let hero = board("Qh Qs 2d", "Kh Kd 6c 8s Th", "9c 9d 9s Kc Ah");
        let opponent = board("Jh Js 3d", "2h 3h 4c 5d 7s", "Ac Ad 4h 4s 8c");
        assert_eq!(symmetric_hu_score(&hero, &opponent, 8.0), 21.0);
        assert_eq!(symmetric_hu_score(&opponent, &hero, 8.0), -21.0);
    }

    #[test]
    fn bust_fixture_subtracts_opponent_royalty_and_fl() {
        let hero = board("Ah As Kd", "2h 3h 4c 5d 7s", "4d 4s 8c 9c Td");
        let opponent = board("Qh Qs 2d", "Kh Kc 6c 8s Jh", "9h 9s 9d Tc Jc");
        assert!(hero.fouled);
        assert_eq!(symmetric_hu_score(&hero, &opponent, 8.0), -21.0);
    }

    #[test]
    fn a_fantasyland_opponent_is_paid_on_stay_not_on_a_queens_top() {
        // Opponent top is QQ (ordinary entry) but neither trips-top nor
        // quads-plus-bottom, so from Fantasyland they do not continue.
        let hero = board("2h 3d 4c", "5h 6d 7c 8s 9h", "Ah Kd Qc Js Th");
        let fantasyland = board("Qh Qs 2d", "Kh Kd 6c 8s Tc", "9c 9d 9s Kc Ad");
        assert!(fantasyland.enters_fantasyland());
        assert!(!fantasyland.stays_in_fantasyland());

        let symmetric = symmetric_hu_score(&hero, &fantasyland, 9.109);
        let asymmetric = hero_vs_fl_score(&hero, &fantasyland, 9.109);
        // The hero stops paying for an entry the FL player never collects.
        assert!((asymmetric - (symmetric + 9.109)).abs() < 1e-12);
    }

    #[test]
    fn a_fantasyland_opponent_with_trips_top_is_paid_the_stay_value() {
        let hero = board("2h 3d 4c", "5h 6d 7c 8s 9h", "Ah Kd Qc Js Th");
        let fantasyland = board("9c 9d 9s", "Kh Kd Kc 8s Tc", "Ac Ad As 2c 3c");
        assert!(fantasyland.stays_in_fantasyland());
        let asymmetric = hero_vs_fl_score(&hero, &fantasyland, 9.109);
        let without = combine(&hero, &fantasyland, 0.0, 0.0);
        assert!((asymmetric - (without - 9.109)).abs() < 1e-12);
    }
}
