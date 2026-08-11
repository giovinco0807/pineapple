//! Actor-safe public action history.
//!
//! A history event records only face-up placements and the number of discarded
//! cards. It never stores the discarded card identity.

use crate::{HuRlError, HuRlResult};
use ofc_hu_m3_engine::{
    action_key::ActionKey,
    cards::{Card, ALL_CARDS},
    infoset::{Seat, Street},
    state::Row,
};
use serde_json::{json, Value};

pub const PUBLIC_PLACEMENT_SCHEMA: &str = "regular_ofc_hu_rl_public_placement_v1";
const MASK_HEX_WIDTH: usize = 13;
const MAX_MASK: u64 = (1_u64 << 52) - 1;

/// One publicly observable placement with private discard identity erased.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PublicPlacement {
    pub(crate) street: Street,
    pub(crate) acting_seat: Seat,
    pub(crate) top_placement_mask: u64,
    pub(crate) middle_placement_mask: u64,
    pub(crate) bottom_placement_mask: u64,
    pub(crate) discard_count: u8,
}

impl PublicPlacement {
    pub fn new(
        street: Street,
        acting_seat: Seat,
        top_placement_mask: u64,
        middle_placement_mask: u64,
        bottom_placement_mask: u64,
        discard_count: u8,
    ) -> HuRlResult<Self> {
        let event = Self {
            street,
            acting_seat,
            top_placement_mask,
            middle_placement_mask,
            bottom_placement_mask,
            discard_count,
        };
        event.validate()?;
        Ok(event)
    }

    pub fn from_action_key(street: Street, acting_seat: Seat, key: ActionKey) -> HuRlResult<Self> {
        key.validate().map_err(HuRlError::from)?;
        Self::new(
            street,
            acting_seat,
            key.top_mask,
            key.middle_mask,
            key.bottom_mask,
            key.discard_mask.count_ones() as u8,
        )
    }

    pub fn validate(&self) -> HuRlResult<()> {
        let masks = self.placement_masks();
        if masks.iter().any(|mask| *mask > MAX_MASK) {
            return Err(HuRlError::new(
                "public placement mask is outside the 52-card domain",
            ));
        }
        let mut union = 0_u64;
        for mask in masks {
            if union & mask != 0 {
                return Err(HuRlError::new(
                    "public placement masks must be pairwise disjoint",
                ));
            }
            union |= mask;
        }
        let (expected_placements, expected_discards) = match self.street {
            Street::T0 => (5, 0),
            Street::T1 | Street::T2 | Street::T3 | Street::T4 => (2, 1),
        };
        if union.count_ones() != expected_placements {
            return Err(HuRlError::new(format!(
                "{} public action requires {expected_placements} placed cards",
                self.street.as_str()
            )));
        }
        if self.discard_count != expected_discards {
            return Err(HuRlError::new(format!(
                "{} public action requires discard_count={expected_discards}",
                self.street.as_str()
            )));
        }
        Ok(())
    }

    pub const fn placement_masks(&self) -> [u64; 3] {
        [
            self.top_placement_mask,
            self.middle_placement_mask,
            self.bottom_placement_mask,
        ]
    }

    pub const fn street(&self) -> Street {
        self.street
    }

    pub const fn acting_seat(&self) -> Seat {
        self.acting_seat
    }

    pub const fn discard_count(&self) -> u8 {
        self.discard_count
    }

    pub const fn placement_mask(&self) -> u64 {
        self.top_placement_mask | self.middle_placement_mask | self.bottom_placement_mask
    }

    pub fn cards(&self, row: Row) -> Vec<Card> {
        let mask = match row {
            Row::Top => self.top_placement_mask,
            Row::Middle => self.middle_placement_mask,
            Row::Bottom => self.bottom_placement_mask,
        };
        ALL_CARDS
            .iter()
            .copied()
            .filter(|card| mask & card.bit() != 0)
            .collect()
    }

    pub fn to_json(&self) -> Value {
        self.validate()
            .expect("invalid PublicPlacement cannot be serialized");
        json!({
            "schema": PUBLIC_PLACEMENT_SCHEMA,
            "street": self.street,
            "acting_seat": self.acting_seat,
            "top_placement_mask": format!("{:0MASK_HEX_WIDTH$x}", self.top_placement_mask),
            "middle_placement_mask": format!("{:0MASK_HEX_WIDTH$x}", self.middle_placement_mask),
            "bottom_placement_mask": format!("{:0MASK_HEX_WIDTH$x}", self.bottom_placement_mask),
            "discard_count": self.discard_count,
        })
    }
}
