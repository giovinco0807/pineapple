//! A deliberately coarse twin of the free-slot outlook encoder, for the T1
//! replies inside a rollout.
//!
//! The T0 evaluator spends nearly all of its time below the root. Every T0
//! rollout plays a T1 reply, every T1 reply is a learned decision, and every
//! learned decision runs the free-slot outlook once per candidate board -- some
//! twenty-seven of them, each striding four thousand row completions and five
//! hundred and twelve joint draws. The root's own arithmetic is a rounding
//! error beside it.
//!
//! What is being bought with that precision is a *reply*, not a value. The
//! rollout does not read the outlook; it reads the action the outlook ranked
//! first, plays it, and scores the finish. A cheaper encoder that ranks the
//! same action first is worth exactly as much and costs a fraction, and one
//! that ranks a near-tie first instead costs the rollout the difference between
//! two actions the teacher itself could barely separate.
//!
//! So this module encodes the same four blocks in the same 168-dimension
//! layout, from the same machinery, at [`FAST_HISTOGRAM_CAP`] row completions
//! and [`FAST_JOINT_DRAWS`] joint draws. It is a *different function* of the
//! board, not an approximation of [`crate::t3first_features`] that could be
//! made to agree by trying harder: the histogram strides a tenth as many
//! completions and the joint block an eighth as many draws, so the numbers
//! differ in the third decimal and sometimes in the first. Nothing here carries
//! a parity claim against the full encoder, and a model trained on these
//! features is not loadable against those.
//!
//! What it does carry is determinism. There is no RNG on either path -- both
//! stride by index -- so the same board and unknown set produce the same vector
//! on every run and every machine, which `fast_encode_is_pinned` holds to six
//! hardcoded values.

use crate::cards::Card;
use crate::infoset::ActorObservation;
use crate::state::Board;
use crate::t3_features::{
    encode_structural, head_to_head, Finish, HEAD_TO_HEAD_SIZE, SIDE_OUTLOOK_SIZE, STRUCTURAL_SIZE,
};
use crate::t3first_features::{FreeOutlookCache, FEATURE_SIZE};

/// Row completions the per-row histogram stride keeps, against 4,000 on the
/// full path.
///
/// The histograms are the block the coarsening costs least: a row's category
/// distribution is smooth in the number of samples, and four hundred evenly
/// strided completions place it to within a percent or so of where four
/// thousand do.
pub const FAST_HISTOGRAM_CAP: usize = 400;

/// Draws the joint block visits, against 512 on the full path.
///
/// This is the expensive one and the one the coarsening costs most. Each draw
/// runs the whole arrangement loop, so the block is linear in this number, and
/// each draw is also one sample of the foul rate and one finish in the
/// head-to-head. Sixty-four is a coarse estimate of a foul rate and a coarse
/// finish distribution; whether it is coarse enough to change which action
/// ranks first is the question the validation gate answers, not one this
/// constant can assert.
pub const FAST_JOINT_DRAWS: usize = 64;

/// The shared work one fast decision's candidate boards reuse.
///
/// A thin wrapper rather than a type alias, so a cache built for the fast caps
/// cannot be passed to the full encoder or the other way round. The sharing it
/// provides is the same sharing [`FreeOutlookCache`] always provided: draws
/// depend only on the unknown set and the open-slot count, and per-row work
/// depends only on that row, so the rows a candidate action did not touch are
/// answered once for the whole decision.
pub struct FastOutlookCache {
    inner: FreeOutlookCache,
}

impl FastOutlookCache {
    pub fn new() -> Self {
        Self {
            inner: FreeOutlookCache::with_caps(FAST_HISTOGRAM_CAP, FAST_JOINT_DRAWS),
        }
    }
}

impl Default for FastOutlookCache {
    fn default() -> Self {
        Self::new()
    }
}

/// The coarse outlook of one board: the same 36-wide block layout the full
/// encoder emits, and the finishes the head-to-head block reads.
///
/// Accepts the same four-, six- and eight-open-slot boards the full outlook
/// does, and refuses the same widths by the same message, because the geometry
/// check belongs to the method rather than to the caps.
pub fn fast_outlook(
    board: &Board,
    unknown: &[Card],
    cache: &mut FastOutlookCache,
) -> Result<([f32; SIDE_OUTLOOK_SIZE], Vec<Finish>), String> {
    cache.inner.outlook(board, unknown)
}

/// The coarse outlook block without the finishes, for the hidden-opponent
/// encoder below: its head-to-head columns stay zero, so finishes built here
/// would be dropped unread. Same block bytes as [`fast_outlook`].
pub fn fast_outlook_block_only(
    board: &Board,
    unknown: &[Card],
    cache: &mut FastOutlookCache,
) -> Result<[f32; SIDE_OUTLOOK_SIZE], String> {
    cache.inner.outlook_block_only(board, unknown)
}

/// Where the hero's half of the vector ends and the opponent's begins.
///
/// Named because the T0 first-seat encoder below writes everything up to it and
/// nothing after it, and a reader of that function should be able to see which
/// blocks are real without counting block widths.
pub const HERO_BLOCK_END: usize = STRUCTURAL_SIZE + SIDE_OUTLOOK_SIZE;

/// One candidate board's 168 features against an opponent whose board cannot be
/// read: structural, coarse hero outlook, and zeros where the opponent would be.
///
/// Two situations reach this, and they are different situations with the same
/// consequence.
///
/// The first is the opening street acting first. The opponent has not acted, so
/// its board is empty -- thirteen open slots -- and the free-slot outlook
/// enumerates four, six and eight. That board is going to fill in; it is simply
/// not filled in yet.
///
/// The second is any street against a **Fantasyland** opponent. There the board
/// is empty because it will always be empty: the opponent took fourteen cards
/// face down and never places one where the hero can see it. No later street
/// improves the situation, which is why the hidden-opponent flag on the
/// observation is a property of the hand rather than of the street.
///
/// Either way the two blocks that read the opponent are left at zero:
///
/// ```text
/// [0, 86)    structural            real
/// [86, 122)  hero outlook (coarse) real
/// [122, 158) opponent outlook      zero
/// [158, 168) head-to-head          zero
/// ```
///
/// The zeros are not a stand-in for numbers this function could have produced
/// more cheaply. They are the encoding a model was fitted to, and the fit is
/// what makes them load-bearing: `t0first_model_v1` was trained on rows from a
/// standalone extractor that wrote exactly this layout, so an engine that
/// filled those columns in would be handing that image a vector it has never
/// seen. Widening the geometry is a change of feature version and a retrain,
/// not a change to this function. Any vs-Fantasyland model distilled on these
/// rows inherits that contract whole.
///
/// What the zeroed blocks cost is worth stating plainly rather than leaving to
/// be discovered, and the cost is not the same in the two situations.
///
/// The 36 opponent-outlook columns cost the RANKING nothing in either.
/// `unknown_cards` reads the observation and the opponent's board does not move
/// when the hero places, so that block is constant across a position's whole
/// candidate fan and cannot reorder it.
///
/// The 10 head-to-head columns are a real loss in both, and a heavier one
/// against Fantasyland. They compare a hero finish distribution that does vary
/// per candidate against an opponent one this geometry cannot produce. At T0
/// first seat the model is blind to an opponent who has not played yet, which
/// is a small thing to be blind to. Against a Fantasyland opponent it is blind
/// to a hand that is already set and is, on average, a strong one -- so a model
/// reading these rows cannot learn to play differently against a made hand than
/// against an ordinary one. That is the debt, and it is the same debt the T0
/// first-seat arm carries: a future architecture wanting those columns needs a
/// belief over the hidden hand, not a wider guard here.
pub fn fast_encode_hidden_opponent(
    observation: &ActorObservation,
    candidate_board: &Board,
    unknown: &[Card],
    cache: &mut FastOutlookCache,
) -> Result<[f32; FEATURE_SIZE], String> {
    let mut out = [0.0f32; FEATURE_SIZE];
    out[..STRUCTURAL_SIZE].copy_from_slice(&encode_structural(observation, candidate_board));
    let hero_block = fast_outlook_block_only(candidate_board, unknown, cache)?;
    out[STRUCTURAL_SIZE..HERO_BLOCK_END].copy_from_slice(&hero_block);
    // [HERO_BLOCK_END, FEATURE_SIZE) stays zero; see above.
    Ok(out)
}

/// One candidate board's 168 features: structural, coarse hero outlook, coarse
/// opponent outlook, head-to-head over the coarse finishes.
///
/// The opponent block does not depend on the hero's action, so it is computed
/// once per decision and passed in rather than recomputed per candidate --
/// which is also why it arrives already coarse: mixing a full opponent block
/// with a coarse hero block would compare two finish distributions drawn at
/// different resolutions, and the head-to-head block would read the difference
/// in resolution as a difference in strength.
pub fn fast_encode(
    observation: &ActorObservation,
    candidate_board: &Board,
    unknown: &[Card],
    opponent_block: &[f32; SIDE_OUTLOOK_SIZE],
    opponent_finishes: &[Finish],
    cache: &mut FastOutlookCache,
) -> Result<[f32; FEATURE_SIZE], String> {
    let mut out = [0.0f32; FEATURE_SIZE];
    out[..STRUCTURAL_SIZE].copy_from_slice(&encode_structural(observation, candidate_board));
    let (hero_block, hero_finishes) = fast_outlook(candidate_board, unknown, cache)?;
    let mut base = STRUCTURAL_SIZE;
    out[base..base + SIDE_OUTLOOK_SIZE].copy_from_slice(&hero_block);
    base += SIDE_OUTLOOK_SIZE;
    out[base..base + SIDE_OUTLOOK_SIZE].copy_from_slice(opponent_block);
    base += SIDE_OUTLOOK_SIZE;
    out[base..].copy_from_slice(&head_to_head(&hero_finishes, opponent_finishes));
    debug_assert_eq!(base + HEAD_TO_HEAD_SIZE, FEATURE_SIZE);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::t3_features::unknown_cards;

    /// A T1 first-seat position: five cards on each board, three dealt. Written
    /// out rather than loaded from a fixture so the golden values below are
    /// pinned to something that cannot be edited from another directory.
    const PINNED: &str = r#"{
        "schema": "regular_ofc_actor_observation_v1",
        "hero_board": {"top": ["Kd"], "middle": ["3h", "Td"], "bottom": ["7s", "8s"]},
        "opponent_public_board": {"top": ["9h", "6c"], "middle": ["Jh", "2h"], "bottom": ["2s"]},
        "dealt_cards": ["Kh", "Ks", "9d"],
        "hero_private_discards": [],
        "opponent_discard_count": 0,
        "seat": "first",
        "street": "T1",
        "to_act_order": "first",
        "hero_in_fantasyland": false,
        "opponent_in_fantasyland": false,
        "scoring": {
            "schema": "regular_ofc_scoring_context_v1",
            "fantasyland_cards": 14,
            "fl_ev": {"14": 10.227020614683454},
            "foul_enabled": true,
            "hu_line_points": true,
            "middle_trips_royalty": 2,
            "scoop_bonus": 3
        }
    }"#;

    fn pinned() -> ActorObservation {
        serde_json::from_str(PINNED).expect("pinned observation parses")
    }

    /// The vector is the width the model loader expects and holds no NaN.
    ///
    /// The foul rate and the average royalty are both divided by counts that
    /// the coarsening can drive to zero -- sixty-four draws that all foul leave
    /// no finishes to average over -- so "no NaN" is a live claim about the
    /// guards rather than a formality.
    #[test]
    fn every_feature_is_finite_and_the_vector_is_the_expected_width() {
        let observation = pinned();
        let unknown = unknown_cards(&observation);
        let mut cache = FastOutlookCache::new();
        let (opponent_block, opponent_finishes) =
            fast_outlook(&observation.opponent_public_board, &unknown, &mut cache)
                .expect("opponent outlook");

        let actions = crate::action::generate_turn_actions(
            &observation.hero_board,
            &observation.dealt_cards,
        )
        .expect("legal actions");
        assert!(!actions.is_empty());

        for (index, action) in actions.iter().enumerate() {
            let board = observation
                .hero_board
                .place(&action.placements)
                .expect("legal placement");
            let features = fast_encode(
                &observation,
                &board,
                &unknown,
                &opponent_block,
                &opponent_finishes,
                &mut cache,
            )
            .expect("fast encode");
            assert_eq!(features.len(), FEATURE_SIZE);
            for (slot, value) in features.iter().enumerate() {
                assert!(
                    value.is_finite(),
                    "action {index} feature {slot} is {value}"
                );
            }
        }
    }

    /// Six values, captured from the first run of this encoder and hardcoded.
    ///
    /// The point is not that these numbers are right -- there is nothing to be
    /// right against, the function is new -- but that they do not move. Both
    /// strides are by index and neither block consults an RNG, so a change here
    /// means the encoder changed, and a model trained against the old one is
    /// reading a different vector than it was fitted to. That includes changing
    /// either cap: at a different resolution these are different numbers, which
    /// is the intended reading rather than a fragility.
    ///
    /// The six span all four blocks and both branches of the joint block. On
    /// this position the first legal action fouls on every one of the
    /// sixty-four draws, so its outlook divides royalty and Fantasy Land by a
    /// count of zero and its head-to-head compares against an empty finish set;
    /// action four survives most draws and exercises the ordinary path. Pinning
    /// one of each is what makes the guard on the degenerate case a tested
    /// branch rather than an argument.
    #[test]
    fn fast_encode_is_pinned() {
        let observation = pinned();
        let unknown = unknown_cards(&observation);
        let mut cache = FastOutlookCache::new();
        let (opponent_block, opponent_finishes) =
            fast_outlook(&observation.opponent_public_board, &unknown, &mut cache)
                .expect("opponent outlook");
        let actions = crate::action::generate_turn_actions(
            &observation.hero_board,
            &observation.dealt_cards,
        )
        .expect("legal actions");
        assert_eq!(actions.len(), 27, "the pinned position's legal set moved");

        let encoded: Vec<[f32; FEATURE_SIZE]> = actions
            .iter()
            .map(|action| {
                let board = observation
                    .hero_board
                    .place(&action.placements)
                    .expect("legal placement");
                fast_encode(
                    &observation,
                    &board,
                    &unknown,
                    &opponent_block,
                    &opponent_finishes,
                    &mut cache,
                )
                .expect("fast encode")
            })
            .collect();

        for (action, slot, expected) in GOLDEN {
            let actual = encoded[*action][*slot];
            assert!(
                (actual - *expected).abs() < 1e-6,
                "action {action} feature {slot} is {actual}, pinned at {expected}"
            );
        }
    }

    /// `(action, feature, value)`; see [`fast_encode_is_pinned`].
    const GOLDEN: &[(usize, usize, f32)] = &[
        // Structural: the hero's filled fraction, which the outlook never sees.
        (0, 1, 0.928_571_403),
        // Opponent outlook, a bottom-row category share. Computed once for the
        // decision, so every action carries the same value.
        (0, 141, 0.402_500_004),
        // Hero outlook, foul rate: the all-foul candidate.
        (0, 116, 1.0),
        // Head-to-head against that candidate's empty finish set.
        (0, 164, 1.0),
        // The same foul-rate slot on a candidate that mostly survives.
        (4, 116, 0.296_875),
        // Head-to-head with finishes on both sides.
        (4, 159, 0.967_013_896),
    ];

    /// A T0 first-seat position: two empty boards and five dealt cards, which
    /// is the whole of the opening street acting first. Written out for the
    /// same reason [`PINNED`] is.
    const PINNED_T0_FIRST: &str = r#"{
        "schema": "regular_ofc_actor_observation_v1",
        "hero_board": {"top": [], "middle": [], "bottom": []},
        "opponent_public_board": {"top": [], "middle": [], "bottom": []},
        "dealt_cards": ["8h", "Qd", "Qs", "7c", "Qh"],
        "hero_private_discards": [],
        "opponent_discard_count": 0,
        "seat": "first",
        "street": "T0",
        "to_act_order": "first",
        "hero_in_fantasyland": false,
        "opponent_in_fantasyland": false,
        "scoring": {
            "schema": "regular_ofc_scoring_context_v1",
            "fantasyland_cards": 14,
            "fl_ev": {"14": 9.109},
            "foul_enabled": true,
            "hu_line_points": true,
            "middle_trips_royalty": 2,
            "scoop_bonus": 3
        }
    }"#;

    fn pinned_t0_first() -> ActorObservation {
        serde_json::from_str(PINNED_T0_FIRST).expect("pinned T0 first observation parses")
    }

    /// The opponent's half of a T0 first-seat row is zero, and the hero's half
    /// is real.
    ///
    /// Both halves matter. The zeros are the contract the trained image was fit
    /// under, so a future encoder that learns to fill them in has to fail this
    /// test rather than silently hand that image a vector it has never seen.
    /// The finiteness of the hero half is the same live claim it is on the
    /// four-block path: sixty-four draws that all foul leave no finishes to
    /// average over, and the guards rather than the arithmetic are what keep
    /// the row free of NaN.
    #[test]
    fn a_t0_first_row_is_real_in_the_hero_half_and_zero_in_the_opponent_half() {
        let observation = pinned_t0_first();
        let unknown = unknown_cards(&observation);
        let mut cache = FastOutlookCache::new();
        let actions = crate::action::generate_initial_actions(
            &observation.hero_board,
            &observation.dealt_cards,
        )
        .expect("legal actions");
        assert_eq!(actions.len(), 232, "the opening legal set moved");

        for (index, action) in actions.iter().enumerate() {
            let board = observation
                .hero_board
                .place(&action.placements)
                .expect("legal placement");
            let features = fast_encode_hidden_opponent(&observation, &board, &unknown, &mut cache)
                .expect("T0 first encode");
            assert_eq!(features.len(), FEATURE_SIZE);
            for (slot, value) in features.iter().enumerate().take(HERO_BLOCK_END) {
                assert!(value.is_finite(), "action {index} feature {slot} is {value}");
            }
            for (slot, value) in features.iter().enumerate().skip(HERO_BLOCK_END) {
                assert_eq!(
                    *value, 0.0,
                    "action {index} feature {slot} is {value}, and the T0 \
                     first-seat image expects the opponent half at zero"
                );
            }
        }
    }

    /// The engine's T0 first-seat row is the composition the standalone
    /// extractor wrote, block for block.
    ///
    /// The extractor that produced `t0first_model_v1`'s training rows lives
    /// outside this crate and calls these same two pinned functions in this
    /// same order. Restating the composition here means a change to
    /// [`fast_encode_hidden_opponent`] that reorders or rescales a block fails inside
    /// the crate, without waiting for the cross-binary parity harness to notice
    /// from outside it.
    #[test]
    fn a_t0_first_row_is_structural_then_hero_outlook_and_nothing_else() {
        let observation = pinned_t0_first();
        let unknown = unknown_cards(&observation);
        let actions = crate::action::generate_initial_actions(
            &observation.hero_board,
            &observation.dealt_cards,
        )
        .expect("legal actions");

        let mut shared = FastOutlookCache::new();
        for action in actions.iter().take(24) {
            let board = observation
                .hero_board
                .place(&action.placements)
                .expect("legal placement");
            let features = fast_encode_hidden_opponent(&observation, &board, &unknown, &mut shared)
                .expect("T0 first encode");
            let structural = encode_structural(&observation, &board);
            let (hero_block, _) = fast_outlook(&board, &unknown, &mut FastOutlookCache::new())
                .expect("fresh hero outlook");
            assert_eq!(&features[..STRUCTURAL_SIZE], &structural[..]);
            assert_eq!(&features[STRUCTURAL_SIZE..HERO_BLOCK_END], &hero_block[..]);
        }
    }

    /// Sharing a cache across a decision's candidate boards must not change the
    /// vectors, the same way it does not on the full path.
    ///
    /// The caps are the only thing this module varies, and they are fixed at
    /// construction, so a fresh cache and a shared one are answering the same
    /// question. If they ever disagree the sharing has become a bug rather than
    /// an optimisation, and the decision function below runs entirely on the
    /// shared path.
    #[test]
    fn a_shared_cache_answers_what_a_fresh_one_does() {
        let observation = pinned();
        let unknown = unknown_cards(&observation);
        let mut shared = FastOutlookCache::new();
        let (shared_opponent, _) =
            fast_outlook(&observation.opponent_public_board, &unknown, &mut shared)
                .expect("opponent outlook");

        let actions = crate::action::generate_turn_actions(
            &observation.hero_board,
            &observation.dealt_cards,
        )
        .expect("legal actions");
        for (index, action) in actions.iter().enumerate() {
            let board = observation
                .hero_board
                .place(&action.placements)
                .expect("legal placement");
            let (from_shared, _) =
                fast_outlook(&board, &unknown, &mut shared).expect("shared hero outlook");
            let (from_fresh, _) = fast_outlook(&board, &unknown, &mut FastOutlookCache::new())
                .expect("fresh hero outlook");
            assert_eq!(from_shared, from_fresh, "board {index}: outlook moved");
        }

        let (fresh_opponent, _) = fast_outlook(
            &observation.opponent_public_board,
            &unknown,
            &mut FastOutlookCache::new(),
        )
        .expect("fresh opponent outlook");
        assert_eq!(shared_opponent, fresh_opponent, "opponent block moved");
    }
}
