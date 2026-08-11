//! Simulator-only full-hand state.

use crate::{
    history::PublicPlacement,
    transition::{decision_spec, DECISION_COUNT},
    HuRlError, HuRlResult,
};
use ofc_hu_m3_engine::{
    action::Action,
    cards::{validate_cards, Card},
    infoset::{ActOrder, ActorObservation, ScoringContext, Seat, Street},
    scoring::terminal_score,
    state::{Board, ALL_ROWS},
};

pub(crate) const CARDS_USED_PER_NORMAL_HAND: usize = 34;

/// Hidden simulator state. Fields are crate-private so no policy-facing type
/// can accidentally serialize the deck or either player's private discards.
#[derive(Clone, PartialEq)]
pub(crate) struct WorldState {
    pub(crate) explicit_deck: [Card; 52],
    pub(crate) boards: [Board; 2],
    pub(crate) private_discards: [Vec<Card>; 2],
    pub(crate) decision_count: usize,
    pub(crate) public_history: Vec<PublicPlacement>,
    pub(crate) scoring: ScoringContext,
}

impl WorldState {
    pub(crate) fn new(explicit_deck: &[Card], scoring: ScoringContext) -> HuRlResult<Self> {
        let explicit_deck = validated_explicit_deck(explicit_deck)?;
        scoring.validate().map_err(HuRlError::from)?;
        // V1 is a fixed Regular OFC rules contract.  Supporting arbitrary FL
        // maps would also require a cross-language float-canonicalization
        // contract; accepting them here would make some otherwise valid
        // environments fail only on the terminal transition.  A future schema
        // can widen this deliberately.
        if scoring != ScoringContext::default() {
            return Err(HuRlError::new(
                "scalar RL V1 requires the pinned Regular OFC scoring context",
            ));
        }
        let world = Self {
            explicit_deck,
            boards: [Board::empty(), Board::empty()],
            private_discards: [Vec::new(), Vec::new()],
            decision_count: 0,
            public_history: Vec::new(),
            scoring,
        };
        world.validate()?;
        Ok(world)
    }

    pub(crate) fn done(&self) -> bool {
        self.decision_count == DECISION_COUNT
    }

    pub(crate) fn current_deal(&self) -> HuRlResult<&[Card]> {
        let spec = decision_spec(self.decision_count)?;
        Ok(&self.explicit_deck[spec.deal_start..spec.deal_start + spec.deal_size])
    }

    pub(crate) fn observe(&self) -> HuRlResult<ActorObservation> {
        let spec = decision_spec(self.decision_count)?;
        let (seat, order) = actor_identity(spec.actor)?;
        ActorObservation::new(
            self.boards[spec.actor].clone(),
            self.boards[1 - spec.actor].clone(),
            self.current_deal()?.to_vec(),
            self.private_discards[spec.actor].clone(),
            seat,
            spec.street,
            order,
            self.scoring.clone(),
        )
        .map_err(HuRlError::from)
    }

    pub(crate) fn transitioned(&self, action: &Action, event: PublicPlacement) -> HuRlResult<Self> {
        let spec = decision_spec(self.decision_count)?;
        if event.street() != spec.street || seat_index(event.acting_seat()) != spec.actor {
            return Err(HuRlError::new(
                "public placement does not match the active decision",
            ));
        }
        let mut next = self.clone();
        next.boards[spec.actor] = action
            .apply(&self.boards[spec.actor])
            .map_err(HuRlError::from)?;
        next.private_discards[spec.actor].extend(action.discards.iter().copied());
        next.public_history.push(event);
        next.decision_count += 1;
        next.validate()?;
        Ok(next)
    }

    pub(crate) fn terminal_rewards(&self) -> HuRlResult<[f64; 2]> {
        if !self.done() {
            return Err(HuRlError::new("terminal rewards require a completed hand"));
        }
        let fl_ev_14 = self
            .scoring
            .fl_ev
            .get(&14)
            .copied()
            .ok_or_else(|| HuRlError::new("scoring context must define FL EV for 14 cards"))?;
        let score = terminal_score(&self.boards[0], Some(&self.boards[1]), fl_ev_14)
            .map_err(HuRlError::from)?
            .0;
        Ok([score, -score])
    }

    pub(crate) fn validate(&self) -> HuRlResult<()> {
        validate_in_domain(&self.explicit_deck)?;
        validate_cards(&self.explicit_deck).map_err(HuRlError::from)?;
        self.scoring.validate().map_err(HuRlError::from)?;
        if self.decision_count > DECISION_COUNT {
            return Err(HuRlError::new(
                "decision_count is outside the 10-decision hand",
            ));
        }
        if self.public_history.len() != self.decision_count {
            return Err(HuRlError::new(
                "public history length disagrees with decision_count",
            ));
        }
        for board in &self.boards {
            board.validate().map_err(HuRlError::from)?;
        }

        let mut expected_board_counts = [0_usize; 2];
        let mut expected_discard_counts = [0_usize; 2];
        let mut expected_private_discards = [Vec::new(), Vec::new()];
        let mut history_masks = [0_u64; 2];
        let mut history_row_masks = [[0_u64; 3]; 2];
        let mut all_history_cards = 0_u64;
        for (index, event) in self.public_history.iter().enumerate() {
            let spec = decision_spec(index)?;
            event.validate()?;
            if event.street() != spec.street || seat_index(event.acting_seat()) != spec.actor {
                return Err(HuRlError::new("public history event order/seat changed"));
            }
            let event_mask = event.placement_mask();
            if all_history_cards & event_mask != 0 {
                return Err(HuRlError::new(
                    "public history places the same card more than once",
                ));
            }
            all_history_cards |= event_mask;
            history_masks[spec.actor] |= event_mask;
            for (row_index, mask) in event.placement_masks().into_iter().enumerate() {
                history_row_masks[spec.actor][row_index] |= mask;
            }
            let dealt_cards =
                &self.explicit_deck[spec.deal_start..spec.deal_start + spec.deal_size];
            if event_mask & !cards_mask(dealt_cards) != 0 {
                return Err(HuRlError::new(
                    "public history placement violates its per-decision deal",
                ));
            }
            let discarded = dealt_cards
                .iter()
                .copied()
                .filter(|card| event_mask & card.bit() == 0)
                .collect::<Vec<_>>();
            let expected_discard_count = if spec.street == Street::T0 { 0 } else { 1 };
            if discarded.len() != expected_discard_count {
                return Err(HuRlError::new(
                    "public history does not consume its exact per-decision deal",
                ));
            }
            expected_private_discards[spec.actor].extend(discarded);
            expected_board_counts[spec.actor] += if spec.street == Street::T0 { 5 } else { 2 };
            expected_discard_counts[spec.actor] += if spec.street == Street::T0 { 0 } else { 1 };
        }

        for actor in 0..2 {
            if self.boards[actor].card_count() != expected_board_counts[actor] {
                return Err(HuRlError::new(
                    "board geometry disagrees with decision_count",
                ));
            }
            if self.private_discards[actor].len() != expected_discard_counts[actor] {
                return Err(HuRlError::new(
                    "private-discard geometry disagrees with decision_count",
                ));
            }
            if self.private_discards[actor] != expected_private_discards[actor] {
                return Err(HuRlError::new(
                    "private discards disagree with the per-decision deals",
                ));
            }
            if board_mask(&self.boards[actor]) != history_masks[actor] {
                return Err(HuRlError::new(
                    "public history disagrees with the public boards",
                ));
            }
            for (row_index, row) in ALL_ROWS.iter().copied().enumerate() {
                if cards_mask(self.boards[actor].cards(row)) != history_row_masks[actor][row_index]
                {
                    return Err(HuRlError::new(
                        "public history does not exactly reconstruct every board row",
                    ));
                }
            }
        }

        let consumed = self
            .boards
            .iter()
            .flat_map(Board::all_cards)
            .chain(self.private_discards[0].iter().copied())
            .chain(self.private_discards[1].iter().copied())
            .collect::<Vec<_>>();
        validate_in_domain(&consumed)?;
        validate_cards(&consumed).map_err(HuRlError::from)?;
        let consumed_end = if self.done() {
            CARDS_USED_PER_NORMAL_HAND
        } else {
            decision_spec(self.decision_count)?.deal_start
        };
        if consumed.len() != consumed_end
            || cards_mask(&consumed) != cards_mask(&self.explicit_deck[..consumed_end])
        {
            return Err(HuRlError::new(
                "completed actions violate deck-prefix conservation",
            ));
        }
        if self.done() && !self.boards.iter().all(Board::is_complete) {
            return Err(HuRlError::new("terminal boards are incomplete"));
        }
        Ok(())
    }
}

fn validated_explicit_deck(explicit_deck: &[Card]) -> HuRlResult<[Card; 52]> {
    if explicit_deck.len() != 52 {
        return Err(HuRlError::new(
            "explicit_deck must contain exactly 52 cards",
        ));
    }
    validate_in_domain(explicit_deck)?;
    validate_cards(explicit_deck).map_err(HuRlError::from)?;
    explicit_deck
        .try_into()
        .map_err(|_| HuRlError::new("explicit_deck must contain exactly 52 cards"))
}

fn validate_in_domain(cards: &[Card]) -> HuRlResult<()> {
    if cards.iter().any(|card| card.index() >= 52) {
        return Err(HuRlError::new("card index is outside the 52-card domain"));
    }
    Ok(())
}

fn actor_identity(actor: usize) -> HuRlResult<(Seat, ActOrder)> {
    match actor {
        0 => Ok((Seat::First, ActOrder::First)),
        1 => Ok((Seat::Second, ActOrder::Second)),
        _ => Err(HuRlError::new("actor must be 0 or 1")),
    }
}

pub(crate) const fn seat_index(seat: Seat) -> usize {
    match seat {
        Seat::First => 0,
        Seat::Second => 1,
    }
}

fn board_mask(board: &Board) -> u64 {
    board
        .top
        .iter()
        .chain(&board.middle)
        .chain(&board.bottom)
        .fold(0_u64, |mask, card| mask | card.bit())
}

fn cards_mask(cards: &[Card]) -> u64 {
    cards.iter().fold(0_u64, |mask, card| mask | card.bit())
}
