"""Counter-derived deterministic RNG keys for common-random OFC search.

Seeds are derived from semantic coordinates, never Python's process-randomized
``hash()`` and never a legal-action list index.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Literal


COUNTER_RNG_SCHEMA = "regular_ofc_counter_rng_v1"
_PERSONALIZATION = b"OFC-RNG-v1"
_SEED_MASK = (1 << 63) - 1


@dataclass(frozen=True)
class CounterRngKey:
    base_seed: int
    run_id: str
    phase: str
    sample_index: int
    actor: int | Literal["hero", "opponent", "chance"]
    street: str
    stream: str = "default"
    counter: int = 0
    root_fingerprint: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.base_seed, int) or isinstance(self.base_seed, bool):
            raise TypeError("base_seed must be an integer")
        if self.sample_index < 0:
            raise ValueError("sample_index must be non-negative")
        if self.counter < 0:
            raise ValueError("counter must be non-negative")
        for name in ("run_id", "phase", "street", "stream"):
            if not str(getattr(self, name)):
                raise ValueError(f"{name} must not be empty")
        if isinstance(self.actor, int):
            if isinstance(self.actor, bool) or self.actor not in (0, 1):
                raise ValueError("integer actor must be 0 or 1")
        elif self.actor not in {"hero", "opponent", "chance"}:
            raise ValueError(f"invalid actor: {self.actor!r}")

    def payload(self) -> dict[str, object]:
        return {
            "schema": COUNTER_RNG_SCHEMA,
            "base_seed": self.base_seed,
            "run_id": self.run_id,
            "phase": self.phase,
            "sample_index": self.sample_index,
            "actor": self.actor,
            "street": self.street,
            "stream": self.stream,
            "counter": self.counter,
            "root_fingerprint": self.root_fingerprint,
        }

    def seed(self) -> int:
        encoded = json.dumps(
            self.payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        digest = hashlib.blake2b(
            encoded, digest_size=16, person=_PERSONALIZATION
        ).digest()
        return int.from_bytes(digest[:8], "big") & _SEED_MASK

    def random(self) -> random.Random:
        return random.Random(self.seed())


def common_future_seed(
    *,
    base_seed: int,
    run_id: str,
    root_fingerprint: str,
    sample_index: int,
    street: str,
    stream: str = "future_cards",
) -> int:
    """Seed a chance sample shared by every candidate action at a search root."""
    return CounterRngKey(
        base_seed=base_seed,
        run_id=run_id,
        phase="common_future",
        sample_index=sample_index,
        actor="chance",
        street=street,
        stream=stream,
        root_fingerprint=root_fingerprint,
    ).seed()


def policy_decision_seed(
    *,
    base_seed: int,
    run_id: str,
    root_fingerprint: str,
    future_index: int,
    actor: int | Literal["hero", "opponent"],
    street: str,
    decision_ordinal: int,
    stream: str = "policy_decision",
) -> int:
    """Seed a rollout decision without depending on candidate enumeration."""
    return CounterRngKey(
        base_seed=base_seed,
        run_id=run_id,
        phase="rollout_policy",
        sample_index=future_index,
        actor=actor,
        street=street,
        stream=stream,
        counter=decision_ordinal,
        root_fingerprint=root_fingerprint,
    ).seed()
