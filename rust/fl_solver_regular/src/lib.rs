//! Regular-rule (52 cards, no jokers) Fantasyland machinery for the
//! "opponent is in Fantasyland, hero plays normally" model family.
//!
//! Three layers, each built on the one before:
//!
//! * [`solver`] -- exact 14-card Fantasyland placement search.
//! * [`distribution`] -- sample and solve opponent Fantasyland deals from a
//!   hero's unseen remainder, so a hero decision can be priced against the
//!   board distribution it actually faces.
//! * [`teacher`] -- the T4-vs-FL label: score every hero candidate against
//!   sampled, solved opponent Fantasyland boards under common random numbers.
//!
//! # Rules, as confirmed against this repository
//!
//! * Fantasyland entry is QQ+ or trips on top of a non-fouling board, and the
//!   regular ruleset deals 14 cards for every entry type
//!   (`src/ofc_regular/rules.py::REGULAR_RULES`, and `fl_entry_cards` in
//!   `configs/fl_ev_regular_v4_selfplay.json`). There is no 15/16/17 chain here.
//! * Fantasyland stay is trips on top or quads-or-better on the bottom, again
//!   for 14 cards (`src/ofc_regular/rules.py::check_fl_stay`, and
//!   `rust/regular_fl_solver` agrees: `top_stay = cat == TRIPS`,
//!   `bottom_stay = cat >= QUADS`).
//! * Scoring is 1-6 line scoring plus royalties
//!   (`rust/hu_m3_engine/src/scoring.rs::heads_up_terminal_score`), except that
//!   a player already *in* Fantasyland continues on stay rather than on
//!   ordinary entry -- see [`scoring`] and
//!   `src/ofc_regular/estimate_hu_fl_ev_direct.py::score_fl_vs_normal`.
//!
//! # The FL EV is never a literal
//!
//! Every entry point reads `fl_ev[14]` from a config file and records the path
//! and value it read in the artifact it produces. See
//! [`objective::load_fl_ev`].

pub mod behavior;
pub mod cards;
pub mod distribution;
pub mod engine_features;
pub mod eval;
pub mod fl_library;
pub mod frontier;
pub mod objective;
pub mod rng;
pub mod scoring;
pub mod solver;
pub mod t0_teacher;
pub mod t1_teacher;
pub mod t2_teacher;
pub mod teacher;
pub mod vfl_model;

/// Bumped whenever a change can move a solved placement or a label.
/// 0.2.0 adds the adaptive (best-response) Fantasyland opponent. Labels
/// from 0.1.x are static-opponent labels and are not interchangeable.
/// 0.3.0 adds the T1 teacher, whose continuations are the distilled T2/T3
/// rankers -- so the engine is now linked in, and a T1 label's identity
/// includes the two model digests and `engine_features_rev`.
pub const SOLVER_VERSION: &str = "fl_solver_regular/0.3.0";
