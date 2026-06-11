"""Named policy profiles for regular OFC evaluation and data collection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .hu_turn3_model import load_hu_action_value_model
from .policy import RegularAiPolicy
from .turn3_model import load_action_value_model

ProfileName = Literal[
    "current",
    "stage3_baseline",
    "stage7_off",
    "stage7_m5_r10",
    "stage7_m4_r10",
    "stage7_m3_r10_experiment",
    "stage7_m3_r12_experiment",
    "old_opening",
    "random_exact_final",
    "late_t2t3",
    "hu_t3_candidate",
]

DEFAULT_OPENING_MODEL = Path("models/opening_stage7_torch_wide.pt")
DEFAULT_OLD_OPENING_MODEL = Path("models/opening_stage6.pkl")
DEFAULT_TURN1_MODEL = Path("models/turn1_stage6_torch_wide.pt")
DEFAULT_TURN2_MODEL = Path("models/turn2_stage8.pkl")
DEFAULT_TURN3_MODEL = Path("models/turn3_stage6.pkl")
DEFAULT_HU_TURN3_CANDIDATE_MODEL = Path(
    "models/hu_turn3_stage2_mc32_500k_cached_rank_wide.pt"
)
DEFAULT_HU_TURN3_CANDIDATE_MIN_MARGIN = 10.0
DEFAULT_HU_TURN3_STAGE7_MODEL = Path(
    "models/hu_turn3_stage7_reference_override_cached_rank_wide.pt"
)
DEFAULT_HU_TURN3_STAGE7_REFERENCE_MODEL = Path(
    "models/hu_turn3_stage3_mc32_500k_plus_m8_12_f128_100k_w2_cached_rank_wide.pt"
)
DEFAULT_HU_TURN3_STAGE7_MIN_MARGIN = 5.0
DEFAULT_HU_TURN3_STAGE7_REFERENCE_MIN_MARGIN = 10.0
DEFAULT_HU_TURN3_STAGE7_CANARY_MIN_MARGIN = 4.0
DEFAULT_HU_TURN3_STAGE7_EXPERIMENT_MIN_MARGIN = 3.0


@dataclass(frozen=True)
class ModelPaths:
    opening: Path = DEFAULT_OPENING_MODEL
    old_opening: Path = DEFAULT_OLD_OPENING_MODEL
    turn1: Path = DEFAULT_TURN1_MODEL
    turn2: Path = DEFAULT_TURN2_MODEL
    turn3: Path = DEFAULT_TURN3_MODEL
    hu_turn3_candidate: Path = DEFAULT_HU_TURN3_CANDIDATE_MODEL
    hu_turn3_stage7: Path = DEFAULT_HU_TURN3_STAGE7_MODEL
    hu_turn3_stage7_reference: Path = DEFAULT_HU_TURN3_STAGE7_REFERENCE_MODEL


@dataclass
class ModelBundle:
    opening: Any | None = None
    old_opening: Any | None = None
    turn1: Any | None = None
    turn2: Any | None = None
    turn3: Any | None = None
    hu_turn3_candidate: Any | None = None
    hu_turn3_stage7: Any | None = None
    hu_turn3_stage7_reference: Any | None = None


def load_model_bundle(paths: ModelPaths, profiles: set[str] | None = None) -> ModelBundle:
    """Load only the models required by the requested profile names."""
    profiles = profiles or {
        "current",
        "stage3_baseline",
        "stage7_off",
        "stage7_m5_r10",
        "stage7_m4_r10",
        "stage7_m3_r10_experiment",
        "stage7_m3_r12_experiment",
        "old_opening",
        "late_t2t3",
        "hu_t3_candidate",
    }
    bundle = ModelBundle()
    base_model_profiles = {
        "current",
        "stage3_baseline",
        "stage7_off",
        "stage7_m5_r10",
        "stage7_m4_r10",
        "stage7_m3_r10_experiment",
        "stage7_m3_r12_experiment",
        "hu_t3_candidate",
    }
    stage7_profiles = {
        "current",
        "stage7_m5_r10",
        "stage7_m4_r10",
        "stage7_m3_r10_experiment",
        "stage7_m3_r12_experiment",
    }
    stage3_hu_profiles = {"stage3_baseline", "stage7_off"}
    if base_model_profiles & profiles:
        bundle.opening = load_action_value_model(paths.opening)
        bundle.turn1 = load_action_value_model(paths.turn1)
        bundle.turn2 = load_action_value_model(paths.turn2)
        bundle.turn3 = load_action_value_model(paths.turn3)
    if stage7_profiles & profiles:
        bundle.hu_turn3_stage7 = safe_load_hu_action_value_model(paths.hu_turn3_stage7)
    if (stage7_profiles | stage3_hu_profiles) & profiles:
        bundle.hu_turn3_stage7_reference = safe_load_hu_action_value_model(
            paths.hu_turn3_stage7_reference
        )
    if "hu_t3_candidate" in profiles:
        bundle.hu_turn3_candidate = load_hu_action_value_model(paths.hu_turn3_candidate)
    if "old_opening" in profiles:
        bundle.old_opening = load_action_value_model(paths.old_opening)
        if bundle.turn1 is None:
            bundle.turn1 = load_action_value_model(paths.turn1)
        if bundle.turn2 is None:
            bundle.turn2 = load_action_value_model(paths.turn2)
        if bundle.turn3 is None:
            bundle.turn3 = load_action_value_model(paths.turn3)
    if "late_t2t3" in profiles:
        if bundle.turn2 is None:
            bundle.turn2 = load_action_value_model(paths.turn2)
        if bundle.turn3 is None:
            bundle.turn3 = load_action_value_model(paths.turn3)
    return bundle


def safe_load_hu_action_value_model(path: Path) -> Any | None:
    """Load an optional HU model, falling back to Stage3 if loading fails."""
    try:
        return load_hu_action_value_model(path)
    except Exception:
        return None


def build_policy(
    profile: str,
    bundle: ModelBundle,
    *,
    seed: int,
    opening_lookahead_samples: int,
    seat: str = "first",
) -> RegularAiPolicy:
    """Create a playable policy from a named profile."""
    if profile in {"current", "stage7_m5_r10"}:
        return _build_stage7_policy(
            bundle,
            seed=seed,
            seat=seat,
            opening_lookahead_samples=opening_lookahead_samples,
            enabled=True,
            hu_min_margin=DEFAULT_HU_TURN3_STAGE7_MIN_MARGIN,
            reference_min_margin=DEFAULT_HU_TURN3_STAGE7_REFERENCE_MIN_MARGIN,
        )
    if profile in {"stage3_baseline", "stage7_off"}:
        return _build_stage3_hu_policy(
            bundle,
            seed=seed,
            seat=seat,
            opening_lookahead_samples=opening_lookahead_samples,
            hu_min_margin=DEFAULT_HU_TURN3_STAGE7_REFERENCE_MIN_MARGIN,
        )
    if profile == "stage7_m4_r10":
        return _build_stage7_policy(
            bundle,
            seed=seed,
            seat=seat,
            opening_lookahead_samples=opening_lookahead_samples,
            enabled=True,
            hu_min_margin=DEFAULT_HU_TURN3_STAGE7_CANARY_MIN_MARGIN,
            reference_min_margin=DEFAULT_HU_TURN3_STAGE7_REFERENCE_MIN_MARGIN,
        )
    if profile in {"stage7_m3_r10_experiment", "stage7_m3_r12_experiment"}:
        reference_min_margin = (
            12.0 if profile == "stage7_m3_r12_experiment" else 10.0
        )
        return _build_stage7_policy(
            bundle,
            seed=seed,
            seat=seat,
            opening_lookahead_samples=opening_lookahead_samples,
            enabled=True,
            hu_min_margin=DEFAULT_HU_TURN3_STAGE7_EXPERIMENT_MIN_MARGIN,
            reference_min_margin=reference_min_margin,
        )
    if profile == "old_opening":
        return RegularAiPolicy(
            opening_model=bundle.old_opening,
            turn1_model=bundle.turn1,
            turn2_model=bundle.turn2,
            turn3_model=bundle.turn3,
            seed=seed,
            seat=seat,
            opening_lookahead_samples=opening_lookahead_samples,
        )
    if profile == "random_exact_final":
        return RegularAiPolicy(seed=seed, seat=seat)
    if profile == "late_t2t3":
        return RegularAiPolicy(
            turn2_model=bundle.turn2,
            turn3_model=bundle.turn3,
            seed=seed,
            seat=seat,
            opening_lookahead_samples=opening_lookahead_samples,
        )
    if profile == "hu_t3_candidate":
        return RegularAiPolicy(
            opening_model=bundle.opening,
            turn1_model=bundle.turn1,
            turn2_model=bundle.turn2,
            turn3_model=bundle.turn3,
            hu_turn3_model=bundle.hu_turn3_candidate,
            hu_turn3_min_margin=DEFAULT_HU_TURN3_CANDIDATE_MIN_MARGIN,
            seed=seed,
            seat=seat,
            opening_lookahead_samples=opening_lookahead_samples,
        )
    raise ValueError(f"unknown policy profile: {profile}")


def _build_stage7_policy(
    bundle: ModelBundle,
    *,
    seed: int,
    seat: str,
    opening_lookahead_samples: int,
    enabled: bool,
    hu_min_margin: float,
    reference_min_margin: float,
) -> RegularAiPolicy:
    stage7_model = getattr(bundle, "hu_turn3_stage7", None)
    reference_model = getattr(bundle, "hu_turn3_stage7_reference", None)
    if not enabled or stage7_model is None:
        return _build_stage3_hu_policy(
            bundle,
            seed=seed,
            seat=seat,
            opening_lookahead_samples=opening_lookahead_samples,
            hu_min_margin=reference_min_margin,
        )
    return RegularAiPolicy(
        opening_model=bundle.opening,
        turn1_model=bundle.turn1,
        turn2_model=bundle.turn2,
        turn3_model=bundle.turn3,
        hu_turn3_model=stage7_model,
        hu_turn3_reference_model=reference_model,
        hu_turn3_min_margin=hu_min_margin,
        hu_turn3_reference_min_margin=reference_min_margin,
        hu_turn3_stage7_enabled=enabled,
        seed=seed,
        seat=seat,
        opening_lookahead_samples=opening_lookahead_samples,
    )


def _build_stage3_hu_policy(
    bundle: ModelBundle,
    *,
    seed: int,
    seat: str,
    opening_lookahead_samples: int,
    hu_min_margin: float,
) -> RegularAiPolicy:
    return RegularAiPolicy(
        opening_model=bundle.opening,
        turn1_model=bundle.turn1,
        turn2_model=bundle.turn2,
        turn3_model=bundle.turn3,
        hu_turn3_model=getattr(bundle, "hu_turn3_stage7_reference", None),
        hu_turn3_min_margin=hu_min_margin,
        hu_turn3_stage7_enabled=True,
        seed=seed,
        seat=seat,
        opening_lookahead_samples=opening_lookahead_samples,
    )


def required_profiles(profile_a: str, profile_b: str) -> set[str]:
    profiles = {profile_a, profile_b}
    return {profile for profile in profiles if profile != "random_exact_final"}
