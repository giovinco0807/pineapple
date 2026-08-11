//! Fixed normal-hand deal schedule and atomic scalar transitions.

use crate::{
    history::PublicPlacement,
    observation::{HuRlActorViewV1, LegalActionMappingV1},
    snapshot::WorldSnapshot,
    world::WorldState,
    HuRlError, HuRlResult,
};
use ofc_hu_m3_engine::{
    action_key::{generate_canonical_actions, resolve_action_key, ActionKey},
    cards::Card,
    infoset::{ScoringContext, Seat, Street},
    state::Board,
};
use std::fmt;

pub const DECISION_COUNT: usize = 10;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct DecisionSpec {
    pub street: Street,
    pub actor: usize,
    pub deal_start: usize,
    pub deal_size: usize,
}

pub const DECISION_SCHEDULE: [DecisionSpec; DECISION_COUNT] = [
    DecisionSpec {
        street: Street::T0,
        actor: 0,
        deal_start: 0,
        deal_size: 5,
    },
    DecisionSpec {
        street: Street::T0,
        actor: 1,
        deal_start: 5,
        deal_size: 5,
    },
    DecisionSpec {
        street: Street::T1,
        actor: 0,
        deal_start: 10,
        deal_size: 3,
    },
    DecisionSpec {
        street: Street::T1,
        actor: 1,
        deal_start: 13,
        deal_size: 3,
    },
    DecisionSpec {
        street: Street::T2,
        actor: 0,
        deal_start: 16,
        deal_size: 3,
    },
    DecisionSpec {
        street: Street::T2,
        actor: 1,
        deal_start: 19,
        deal_size: 3,
    },
    DecisionSpec {
        street: Street::T3,
        actor: 0,
        deal_start: 22,
        deal_size: 3,
    },
    DecisionSpec {
        street: Street::T3,
        actor: 1,
        deal_start: 25,
        deal_size: 3,
    },
    DecisionSpec {
        street: Street::T4,
        actor: 0,
        deal_start: 28,
        deal_size: 3,
    },
    DecisionSpec {
        street: Street::T4,
        actor: 1,
        deal_start: 31,
        deal_size: 3,
    },
];

pub fn decision_spec(index: usize) -> HuRlResult<DecisionSpec> {
    DECISION_SCHEDULE
        .get(index)
        .copied()
        .ok_or_else(|| HuRlError::new("the hand is already terminal"))
}

/// Raw transition result for privileged simulator correctness/audit code.
///
/// `action_key` contains the acting player's private discard identity after
/// T0, so this type must never cross the actor/replay/training boundary.  The
/// batch API converts it to an actor-safe outcome before returning.
#[derive(Clone, PartialEq)]
pub struct StepResult {
    pub actor: usize,
    pub street: Street,
    pub action_key: ActionKey,
    pub public_placement: PublicPlacement,
    pub done: bool,
    pub rewards: [f64; 2],
}

pub const RAW_STEP_RESULT_ARTIFACT_ROLE: &str = "privileged_simulator_audit_only";

impl StepResult {
    pub const fn policy_input_eligible(&self) -> bool {
        false
    }

    pub const fn replay_eligible(&self) -> bool {
        false
    }

    pub const fn training_eligible(&self) -> bool {
        false
    }
}

impl fmt::Debug for StepResult {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StepResult")
            .field("artifact_role", &RAW_STEP_RESULT_ARTIFACT_ROLE)
            .field("actor", &self.actor)
            .field("street", &self.street)
            .field("action_key", &"<redacted privileged ActionKey>")
            .field("public_placement", &self.public_placement)
            .field("done", &self.done)
            .field("rewards", &self.rewards)
            .finish()
    }
}

/// One deterministic normal-hand environment.
#[derive(Clone)]
pub struct ScalarHuRlEnv {
    world: WorldState,
}

impl fmt::Debug for ScalarHuRlEnv {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ScalarHuRlEnv")
            .field("decision_count", &self.world.decision_count)
            .field("hidden_state", &"<redacted>")
            .finish()
    }
}

impl ScalarHuRlEnv {
    pub fn new(explicit_deck: &[Card]) -> HuRlResult<Self> {
        Self::with_scoring(explicit_deck, ScoringContext::default())
    }

    pub fn with_scoring(explicit_deck: &[Card], scoring: ScoringContext) -> HuRlResult<Self> {
        Ok(Self {
            world: WorldState::new(explicit_deck, scoring)?,
        })
    }

    pub fn reset(&mut self, explicit_deck: &[Card]) -> HuRlResult<HuRlActorViewV1> {
        let next = WorldState::new(explicit_deck, self.world.scoring.clone())?;
        let view = actor_view(&next)?;
        self.world = next;
        Ok(view)
    }

    pub fn decision_count(&self) -> usize {
        self.world.decision_count
    }

    pub fn done(&self) -> bool {
        self.world.done()
    }

    pub fn boards(&self) -> &[Board; 2] {
        &self.world.boards
    }

    pub fn public_history(&self) -> &[PublicPlacement] {
        &self.world.public_history
    }

    pub fn scoring(&self) -> &ScoringContext {
        &self.world.scoring
    }

    pub fn observe(&self) -> HuRlResult<HuRlActorViewV1> {
        actor_view(&self.world)
    }

    pub fn legal_mapping(&self) -> HuRlResult<LegalActionMappingV1> {
        Ok(self.observe()?.legal_action_mapping().clone())
    }

    pub fn legal_actions(&self) -> HuRlResult<Vec<ActionKey>> {
        Ok(self.legal_mapping()?.action_keys().to_vec())
    }

    pub fn step(&mut self, selected: ActionKey) -> HuRlResult<StepResult> {
        let (next, result) = self.preview_step(selected)?;
        *self = next;
        Ok(result)
    }

    /// Build one validated successor without mutating this environment.
    ///
    /// The batch coordinator uses this to evaluate every lane in parallel and
    /// commit the complete successor vector only after all lanes succeed.  It
    /// avoids cloning the current world once before validation and again while
    /// constructing the successor.
    pub(crate) fn preview_step(&self, selected: ActionKey) -> HuRlResult<(Self, StepResult)> {
        if self.done() {
            return Err(HuRlError::new("the hand is already terminal"));
        }
        selected.validate().map_err(HuRlError::from)?;
        let observation = self.world.observe()?;
        let actions = generate_canonical_actions(&observation.hero_board, &observation.dealt_cards)
            .map_err(HuRlError::from)?;
        let selected_index = resolve_action_key(&actions, selected).map_err(|_| {
            HuRlError::new(format!(
                "ActionKey is not legal at this decision: {}",
                selected.to_token()
            ))
        })?;
        let spec = decision_spec(self.world.decision_count)?;
        let seat = if spec.actor == 0 {
            Seat::First
        } else {
            Seat::Second
        };
        let public_placement = PublicPlacement::from_action_key(spec.street, seat, selected)?;
        let next = self
            .world
            .transitioned(&actions[selected_index], public_placement.clone())?;
        let rewards = if next.done() {
            next.terminal_rewards()?
        } else {
            [0.0, 0.0]
        };
        let result = StepResult {
            actor: spec.actor,
            street: spec.street,
            action_key: selected,
            public_placement,
            done: next.done(),
            rewards,
        };
        Ok((Self { world: next }, result))
    }

    pub fn terminal_rewards(&self) -> HuRlResult<[f64; 2]> {
        self.world.terminal_rewards()
    }

    pub fn snapshot(&self) -> WorldSnapshot {
        WorldSnapshot(self.world.clone())
    }

    pub fn restore(&mut self, snapshot: &WorldSnapshot) -> HuRlResult<()> {
        let next = snapshot.0.clone();
        next.validate()?;
        self.world = next;
        Ok(())
    }
}

fn actor_view(world: &WorldState) -> HuRlResult<HuRlActorViewV1> {
    if world.done() {
        return Err(HuRlError::new("the hand is already terminal"));
    }
    HuRlActorViewV1::from_observation(world.observe()?, world.public_history.clone())
}
