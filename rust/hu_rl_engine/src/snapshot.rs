//! Opaque simulator checkpoint.

use crate::world::WorldState;
use std::fmt;

/// Exact restore point whose hidden fields cannot be serialized or logged.
#[derive(Clone, PartialEq)]
pub struct WorldSnapshot(pub(crate) WorldState);

impl WorldSnapshot {
    pub fn decision_count(&self) -> usize {
        self.0.decision_count
    }

    pub fn done(&self) -> bool {
        self.0.done()
    }
}

impl fmt::Debug for WorldSnapshot {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WorldSnapshot")
            .field("decision_count", &self.0.decision_count)
            .field("hidden_state", &"<redacted>")
            .finish()
    }
}
