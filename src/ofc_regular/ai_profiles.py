"""Named policy profiles for regular OFC evaluation and data collection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .hu_turn0_candidate import load_turn0_candidate_model
from .hu_turn0_safe_selector import load_hu_turn0_safe_selector_model
from .hu_turn2_stage8_runtime import (
    HuTurn2Stage8RuntimeConfig,
    HuTurn2Stage8SelectiveOverridePolicy,
    load_hu_turn2_stage8_model,
)
from .hu_turn1_safe_selector import load_hu_turn1_safe_selector_model
from .hu_turn3_gate_model import load_hu_turn3_gate_model
from .hu_turn3_model import load_hu_action_value_model
from .hu_infoset import card_free_metadata
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
    "stage9f_cse1p5_firstseat",
    "stage9f_cse1p5_csemax2p5_firstseat",
    "stage9f_cse2_csemax2p5_firstseat",
    "stage9f_cse2_csemax2_firstseat",
    "stage9f_cse2_csemax2_bothseat",
    "stage9f_cse2_csemax2_rank1_firstseat",
    "stage9f_p2",
    "stage9f_p2_hu_t1_stage1",
    "stage9f_p2_hu_t1_topk_confirm",
    "stage18_p1",
    "stage19_p0",
    "stage9f_fast_t2_t1_teacher",
    "stage9d_p07_relaxed_both",
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
DEFAULT_HU_TURN2_STAGE8B_MODEL = Path(
    "models/hu_turn2_stage8b_safe_lcb196_20k_reference_override_cached_rank_wide.pt"
)
DEFAULT_HU_TURN1_STAGE1_MODEL = Path(
    "models/hu_turn1_stage1_stage9f_p2_full2k_hgb_leaf3_lr01_l2_1.pkl"
)
DEFAULT_HU_TURN1_STAGE2_6MODEL_POOL: tuple[Path, ...] = (
    Path("models/hu_turn1_stage2_candidate1000_union_b_e_broad1k_top20_cap25_mc32_hgb_regressor.pkl"),
    Path("models/hu_turn1_stage2_candidate1200_union_top20_mc16_mc32_hgb_regressor.pkl"),
    Path("models/hu_turn1_stage2_top20cap25_mc32_1k_hgb_regressor.pkl"),
    Path("models/hu_turn1_stage2_top20cap25_mc32_1k_listwise_torch.pt"),
    Path("models/hu_turn1_stage2_union4_maxz_top20cap25_mc32_balanced2k_hgb_regressor.pkl"),
    Path("models/hu_turn1_stage2_union4_maxz_top20cap25_mc32_balanced2k_listwise_torch.pt"),
)
DEFAULT_HU_TURN1_SAFE_SELECTOR_MODEL = Path(
    "models/hu_turn1_safe_override_selector_teacher_regret_combined296_accept0p25_gray1_delta_plus_meta.pkl"
)
DEFAULT_HU_TURN1_STAGE18_P1_MODEL = Path(
    "models/hu_turn1_stage18_first1801_mc32_hgb_abs_aug3.pkl"
)
DEFAULT_HU_TURN1_STAGE18_P1_SAFE_SELECTOR_MODEL = Path(
    "models/hu_turn1_stage18_highmc_safe_selector_a_meta_only.pkl"
)
DEFAULT_HU_TURN0_STAGE19_P0_MODEL = Path(
    "models/hu_turn0_stage3_top60_mc32_1000_hgb_baseline-delta_sew1_aug3.pkl"
)
DEFAULT_HU_TURN0_STAGE19_P0_SAFE_SELECTOR_MODEL = Path(
    "models/hu_turn0_stage19_safe_selector_highmc_candidate_delta_plus_meta.pkl"
)
DEFAULT_HU_TURN0_STAGE19_P0_TOPK = 60
DEFAULT_HU_TURN0_STAGE19_P0_MIN_MARGIN_BY_SEAT = {
    "first": 0.5,
    "second": 999.0,
}
DEFAULT_HU_TURN0_STAGE19_P0_SAFE_SELECTOR_THRESHOLD_BY_SEAT = {
    "first": 0.6,
    "second": 1.0,
}
DEFAULT_HU_TURN0_STAGE19_P0_ALLOWED_SEATS = ("first",)
DEFAULT_HU_TURN1_STAGE1_MIN_MARGIN = 1.0
DEFAULT_HU_TURN1_TOPK_CONFIRM_CONFIG = (
    "k3/mc8/d0/confirm16/cse1/pd0/seat=first+second"
)
DEFAULT_HU_TURN1_STAGE18_P1_CONFIG = (
    "k5/mc8/d3/confirm32/cse1.5/pd1.5/seat=first/safe0.7"
)
DEFAULT_HU_TURN2_STAGE9F_CSE1P5_CONFIG = (
    "k3/mc16/d0/se0/confirm32/cse1.5/pd0/seat=first/bygate_delta"
)
DEFAULT_HU_TURN2_STAGE9F_CSE1P5_CSEMAX2P5_CONFIG = (
    "k3/mc16/d0/se0/confirm32/cse1.5/csemax2.5/pd0/seat=first/bygate_delta"
)
DEFAULT_HU_TURN2_STAGE9F_CSE2_CSEMAX2P5_CONFIG = (
    "k3/mc16/d0/se0/confirm32/cse2/csemax2.5/pd0/seat=first/bygate_delta"
)
DEFAULT_HU_TURN2_STAGE9F_CSE2_CSEMAX2_CONFIG = (
    "k3/mc16/d0/se0/confirm32/cse2/csemax2/pd0/seat=first/bygate_delta"
)
DEFAULT_HU_TURN2_STAGE9F_CSE2_CSEMAX2_BOTHSEAT_CONFIG = (
    "k3/mc16/d0/se0/confirm32/cse2/csemax2/fcd0/scd0/pd0/seat=first+second/bygate_delta"
)
DEFAULT_HU_TURN2_STAGE9F_CSE2_CSEMAX2_RANK1_CONFIG = (
    "k3/mc16/d0/se0/confirm32/cse2/csemax2/pd0/rank1/seat=first/bygate_delta"
)
DEFAULT_HU_TURN2_STAGE9F_FAST_T1_TEACHER_MIN_MARGIN = 2.5
DEFAULT_HU_TURN2_STAGE9F_FAST_T1_TEACHER_GATE = 0.9
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
DEFAULT_HU_TURN3_STAGE9D_MODEL = Path(
    "models/hu_turn3_joint_exact_stage9_mixed1200_mc128_extra_trees.pkl"
)
DEFAULT_HU_TURN3_STAGE9D_SUPPORT_MODEL = Path(
    "models/hu_turn3_stage9d_mixed_runtime_hn245_mc128_extra_trees.pkl"
)
DEFAULT_HU_TURN3_STAGE9D_GATE_MODEL = Path(
    "models/hu_turn3_stage9_accept_gate_rf_265fired_runtimepred_second3seed.pkl"
)
DEFAULT_HU_TURN3_STAGE9D_MIN_MARGIN = 0.5
DEFAULT_HU_TURN3_STAGE9D_REFERENCE_MIN_MARGIN = 0.0
DEFAULT_HU_TURN3_STAGE9D_MIN_SUPPORT_MARGIN = 0.5
DEFAULT_HU_TURN3_STAGE9D_MIN_MODEL_SCORE = 5.0
DEFAULT_HU_TURN3_STAGE9D_MIN_GATE_PROBABILITY = 0.7


@dataclass(frozen=True)
class ModelPaths:
    opening: Path = DEFAULT_OPENING_MODEL
    old_opening: Path = DEFAULT_OLD_OPENING_MODEL
    turn1: Path = DEFAULT_TURN1_MODEL
    turn2: Path = DEFAULT_TURN2_MODEL
    turn3: Path = DEFAULT_TURN3_MODEL
    hu_turn1_stage1: Path = DEFAULT_HU_TURN1_STAGE1_MODEL
    hu_turn1_stage1_models: tuple[Path, ...] = ()
    hu_turn1_safe_selector: Path = DEFAULT_HU_TURN1_SAFE_SELECTOR_MODEL
    hu_turn1_stage18_p1: Path = DEFAULT_HU_TURN1_STAGE18_P1_MODEL
    hu_turn1_stage18_p1_safe_selector: Path = DEFAULT_HU_TURN1_STAGE18_P1_SAFE_SELECTOR_MODEL
    hu_turn0_stage19_p0: Path = DEFAULT_HU_TURN0_STAGE19_P0_MODEL
    hu_turn0_stage19_p0_safe_selector: Path = DEFAULT_HU_TURN0_STAGE19_P0_SAFE_SELECTOR_MODEL
    hu_turn2_stage8b: Path = DEFAULT_HU_TURN2_STAGE8B_MODEL
    hu_turn3_candidate: Path = DEFAULT_HU_TURN3_CANDIDATE_MODEL
    hu_turn3_stage7: Path = DEFAULT_HU_TURN3_STAGE7_MODEL
    hu_turn3_stage7_reference: Path = DEFAULT_HU_TURN3_STAGE7_REFERENCE_MODEL
    hu_turn3_stage9d: Path = DEFAULT_HU_TURN3_STAGE9D_MODEL
    hu_turn3_stage9d_support: Path = DEFAULT_HU_TURN3_STAGE9D_SUPPORT_MODEL
    hu_turn3_stage9d_gate: Path = DEFAULT_HU_TURN3_STAGE9D_GATE_MODEL


@dataclass
class ModelBundle:
    opening: Any | None = None
    old_opening: Any | None = None
    turn1: Any | None = None
    turn2: Any | None = None
    turn3: Any | None = None
    hu_turn1_stage1: Any | None = None
    hu_turn1_stage1_models: tuple[Any, ...] = ()
    hu_turn1_safe_selector: Any | None = None
    hu_turn1_stage18_p1: Any | None = None
    hu_turn1_stage18_p1_safe_selector: Any | None = None
    hu_turn0_stage19_p0: Any | None = None
    hu_turn0_stage19_p0_safe_selector: Any | None = None
    hu_turn2_stage8b: Any | None = None
    hu_turn3_candidate: Any | None = None
    hu_turn3_stage7: Any | None = None
    hu_turn3_stage7_reference: Any | None = None
    hu_turn3_stage9d: Any | None = None
    hu_turn3_stage9d_support: Any | None = None
    hu_turn3_stage9d_gate: Any | None = None


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
        "stage9f_cse1p5_firstseat",
        "stage9f_cse1p5_csemax2p5_firstseat",
        "stage9f_cse2_csemax2p5_firstseat",
        "stage9f_cse2_csemax2_firstseat",
        "stage9f_cse2_csemax2_bothseat",
        "stage9f_cse2_csemax2_rank1_firstseat",
        "stage9f_p2",
        "stage9f_p2_hu_t1_stage1",
        "stage9f_p2_hu_t1_topk_confirm",
        "stage18_p1",
        "stage19_p0",
        "stage9f_fast_t2_t1_teacher",
        "stage9d_p07_relaxed_both",
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
        "stage9f_cse1p5_firstseat",
        "stage9f_cse1p5_csemax2p5_firstseat",
        "stage9f_cse2_csemax2p5_firstseat",
        "stage9f_cse2_csemax2_firstseat",
        "stage9f_cse2_csemax2_bothseat",
        "stage9f_cse2_csemax2_rank1_firstseat",
        "stage9f_p2",
        "stage9f_p2_hu_t1_stage1",
        "stage9f_p2_hu_t1_topk_confirm",
        "stage18_p1",
        "stage19_p0",
        "stage9f_fast_t2_t1_teacher",
        "stage9d_p07_relaxed_both",
        "hu_t3_candidate",
    }
    stage7_profiles = {
        "current",
        "stage7_m5_r10",
        "stage7_m4_r10",
        "stage7_m3_r10_experiment",
        "stage7_m3_r12_experiment",
        "stage9f_cse1p5_firstseat",
        "stage9f_cse1p5_csemax2p5_firstseat",
        "stage9f_cse2_csemax2p5_firstseat",
        "stage9f_cse2_csemax2_firstseat",
        "stage9f_cse2_csemax2_bothseat",
        "stage9f_cse2_csemax2_rank1_firstseat",
        "stage9f_p2",
        "stage9f_p2_hu_t1_stage1",
        "stage9f_p2_hu_t1_topk_confirm",
        "stage18_p1",
        "stage19_p0",
        "stage9f_fast_t2_t1_teacher",
        "stage9d_p07_relaxed_both",
    }
    stage3_hu_profiles = {"stage3_baseline", "stage7_off"}
    stage9d_profiles = {"current", "stage9d_p07_relaxed_both"}
    stage9f_profiles = {
        "stage9f_cse1p5_firstseat",
        "stage9f_cse1p5_csemax2p5_firstseat",
        "stage9f_cse2_csemax2p5_firstseat",
        "stage9f_cse2_csemax2_firstseat",
        "stage9f_cse2_csemax2_bothseat",
        "stage9f_cse2_csemax2_rank1_firstseat",
        "stage9f_p2",
        "stage9f_p2_hu_t1_stage1",
        "stage9f_p2_hu_t1_topk_confirm",
        "stage18_p1",
        "stage19_p0",
        "stage9f_fast_t2_t1_teacher",
    }
    hu_turn1_stage1_profiles = {"stage9f_p2_hu_t1_stage1", "stage9f_p2_hu_t1_topk_confirm"}
    hu_turn1_safe_selector_profiles = {"stage9f_p2_hu_t1_topk_confirm"}
    if base_model_profiles & profiles:
        bundle.opening = load_action_value_model(paths.opening)
        bundle.turn1 = load_action_value_model(paths.turn1)
        bundle.turn2 = load_action_value_model(paths.turn2)
        bundle.turn3 = load_action_value_model(paths.turn3)
    if hu_turn1_stage1_profiles & profiles:
        if paths.hu_turn1_stage1_models:
            bundle.hu_turn1_stage1_models = tuple(
                load_hu_action_value_model(path) for path in paths.hu_turn1_stage1_models
            )
            bundle.hu_turn1_stage1 = bundle.hu_turn1_stage1_models[0]
        else:
            bundle.hu_turn1_stage1 = load_hu_action_value_model(paths.hu_turn1_stage1)
    if hu_turn1_safe_selector_profiles & profiles:
        bundle.hu_turn1_safe_selector = load_hu_turn1_safe_selector_model(
            paths.hu_turn1_safe_selector
        )
    if {"stage18_p1", "stage19_p0"} & profiles:
        bundle.hu_turn1_stage18_p1 = safe_load_hu_action_value_model(
            paths.hu_turn1_stage18_p1
        )
        bundle.hu_turn1_stage18_p1_safe_selector = safe_load_hu_turn1_safe_selector_model(
            paths.hu_turn1_stage18_p1_safe_selector
        )
    if "stage19_p0" in profiles:
        bundle.hu_turn0_stage19_p0 = safe_load_hu_turn0_candidate_model(
            paths.hu_turn0_stage19_p0
        )
        bundle.hu_turn0_stage19_p0_safe_selector = safe_load_hu_turn0_safe_selector_model(
            paths.hu_turn0_stage19_p0_safe_selector
        )
    if stage9f_profiles & profiles:
        bundle.hu_turn2_stage8b = load_hu_turn2_stage8_model(paths.hu_turn2_stage8b)
    if stage7_profiles & profiles:
        bundle.hu_turn3_stage7 = safe_load_hu_action_value_model(paths.hu_turn3_stage7)
    if (stage7_profiles | stage3_hu_profiles) & profiles:
        bundle.hu_turn3_stage7_reference = safe_load_hu_action_value_model(
            paths.hu_turn3_stage7_reference
        )
    if stage9d_profiles & profiles:
        bundle.hu_turn3_stage9d = safe_load_hu_action_value_model(paths.hu_turn3_stage9d)
        bundle.hu_turn3_stage9d_support = safe_load_hu_action_value_model(
            paths.hu_turn3_stage9d_support
        )
        bundle.hu_turn3_stage9d_gate = safe_load_hu_turn3_gate_model(
            paths.hu_turn3_stage9d_gate
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


def safe_load_hu_turn3_gate_model(path: Path) -> Any | None:
    """Load an optional HU Turn3 gate model, falling back to Stage7 if loading fails."""
    try:
        return load_hu_turn3_gate_model(path)
    except Exception:
        return None


def safe_load_hu_turn1_safe_selector_model(path: Path) -> Any | None:
    """Load the optional T1 P1 selector, falling back to Stage9f P2 on failure."""
    try:
        return load_hu_turn1_safe_selector_model(path)
    except Exception:
        return None


def safe_load_hu_turn0_candidate_model(path: Path) -> Any | None:
    """Load the optional P0 candidate, falling back to Stage18 P1 on failure."""
    try:
        return load_turn0_candidate_model(path)
    except Exception:
        return None


def safe_load_hu_turn0_safe_selector_model(path: Path) -> Any | None:
    """Load the optional P0 selector, falling back to Stage18 P1 on failure."""
    try:
        return load_hu_turn0_safe_selector_model(path)
    except Exception:
        return None


def build_policy(
    profile: str,
    bundle: ModelBundle,
    *,
    seed: int,
    opening_lookahead_samples: int,
    seat: str = "first",
    hu_turn1_min_margin: float | None = None,
    hu_turn1_topk_config: str | None = None,
) -> RegularAiPolicy:
    """Create a playable policy from a named profile."""
    if profile == "current":
        return _build_stage9d_policy(
            bundle,
            seed=seed,
            seat=seat,
            opening_lookahead_samples=opening_lookahead_samples,
        )
    if profile == "stage7_m5_r10":
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
    if profile == "stage9d_p07_relaxed_both":
        return _build_stage9d_policy(
            bundle,
            seed=seed,
            seat=seat,
            opening_lookahead_samples=opening_lookahead_samples,
        )
    if profile == "stage9f_cse1p5_firstseat":
        return _build_stage9f_cse1p5_policy(
            bundle,
            seed=seed,
            seat=seat,
            opening_lookahead_samples=opening_lookahead_samples,
            config_string=DEFAULT_HU_TURN2_STAGE9F_CSE1P5_CONFIG,
            runtime_profile="stage9f_cse1p5_firstseat",
        )
    if profile == "stage9f_cse1p5_csemax2p5_firstseat":
        return _build_stage9f_cse1p5_policy(
            bundle,
            seed=seed,
            seat=seat,
            opening_lookahead_samples=opening_lookahead_samples,
            config_string=DEFAULT_HU_TURN2_STAGE9F_CSE1P5_CSEMAX2P5_CONFIG,
            runtime_profile="stage9f_cse1p5_csemax2p5_firstseat",
        )
    if profile == "stage9f_cse2_csemax2p5_firstseat":
        return _build_stage9f_cse1p5_policy(
            bundle,
            seed=seed,
            seat=seat,
            opening_lookahead_samples=opening_lookahead_samples,
            config_string=DEFAULT_HU_TURN2_STAGE9F_CSE2_CSEMAX2P5_CONFIG,
            runtime_profile="stage9f_cse2_csemax2p5_firstseat",
        )
    if profile == "stage9f_cse2_csemax2_firstseat":
        return _build_stage9f_cse1p5_policy(
            bundle,
            seed=seed,
            seat=seat,
            opening_lookahead_samples=opening_lookahead_samples,
            config_string=DEFAULT_HU_TURN2_STAGE9F_CSE2_CSEMAX2_CONFIG,
            runtime_profile="stage9f_cse2_csemax2_firstseat",
        )
    if profile == "stage9f_cse2_csemax2_bothseat":
        return _build_stage9f_cse1p5_policy(
            bundle,
            seed=seed,
            seat=seat,
            opening_lookahead_samples=opening_lookahead_samples,
            config_string=DEFAULT_HU_TURN2_STAGE9F_CSE2_CSEMAX2_BOTHSEAT_CONFIG,
            runtime_profile="stage9f_cse2_csemax2_bothseat",
        )
    if profile == "stage9f_cse2_csemax2_rank1_firstseat":
        return _build_stage9f_cse1p5_policy(
            bundle,
            seed=seed,
            seat=seat,
            opening_lookahead_samples=opening_lookahead_samples,
            config_string=DEFAULT_HU_TURN2_STAGE9F_CSE2_CSEMAX2_RANK1_CONFIG,
            runtime_profile="stage9f_cse2_csemax2_rank1_firstseat",
        )
    if profile == "stage9f_p2":
        return _build_stage9f_cse1p5_policy(
            bundle,
            seed=seed,
            seat=seat,
            opening_lookahead_samples=opening_lookahead_samples,
            config_string=DEFAULT_HU_TURN2_STAGE9F_CSE2_CSEMAX2_BOTHSEAT_CONFIG,
            runtime_profile="stage9f_p2",
            runtime_status="p2_fixed",
        )
    if profile == "stage9f_p2_hu_t1_stage1":
        return _build_stage9f_cse1p5_policy(
            bundle,
            seed=seed,
            seat=seat,
            opening_lookahead_samples=opening_lookahead_samples,
            config_string=DEFAULT_HU_TURN2_STAGE9F_CSE2_CSEMAX2_BOTHSEAT_CONFIG,
            runtime_profile="stage9f_p2_hu_t1_stage1",
            runtime_status="t1_stage1_validation",
            hu_turn1_model=bundle.hu_turn1_stage1,
            hu_turn1_min_margin=(
                DEFAULT_HU_TURN1_STAGE1_MIN_MARGIN
                if hu_turn1_min_margin is None
                else hu_turn1_min_margin
            ),
        )
    if profile == "stage9f_p2_hu_t1_topk_confirm":
        return _build_stage9f_p2_hu_t1_topk_confirm_policy(
            bundle,
            seed=seed,
            seat=seat,
            opening_lookahead_samples=opening_lookahead_samples,
            config_string=hu_turn1_topk_config or DEFAULT_HU_TURN1_TOPK_CONFIRM_CONFIG,
            hu_turn1_candidate_model=bundle.hu_turn1_stage1,
            hu_turn1_candidate_models=bundle.hu_turn1_stage1_models,
            hu_turn1_safe_selector_model=bundle.hu_turn1_safe_selector,
            runtime_profile="stage9f_p2_hu_t1_topk_confirm",
            runtime_status="t1_topk_confirm_validation",
        )
    if profile == "stage18_p1":
        return _build_stage9f_p2_hu_t1_topk_confirm_policy(
            bundle,
            seed=seed,
            seat=seat,
            opening_lookahead_samples=opening_lookahead_samples,
            config_string=DEFAULT_HU_TURN1_STAGE18_P1_CONFIG,
            hu_turn1_candidate_model=bundle.hu_turn1_stage18_p1,
            hu_turn1_candidate_models=(),
            hu_turn1_safe_selector_model=bundle.hu_turn1_stage18_p1_safe_selector,
            runtime_profile="stage18_p1",
            runtime_status="p1_fixed",
        )
    if profile == "stage19_p0":
        return _build_stage19_p0_policy(
            bundle,
            seed=seed,
            seat=seat,
            opening_lookahead_samples=opening_lookahead_samples,
        )
    if profile == "stage9f_fast_t2_t1_teacher":
        return _build_stage9f_fast_t2_t1_teacher_policy(
            bundle,
            seed=seed,
            seat=seat,
            opening_lookahead_samples=opening_lookahead_samples,
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


def _build_stage9f_cse1p5_policy(
    bundle: ModelBundle,
    *,
    seed: int,
    seat: str,
    opening_lookahead_samples: int,
    config_string: str,
    runtime_profile: str,
    runtime_status: str = "validation_only",
    hu_turn1_model: Any | None = None,
    hu_turn1_min_margin: float | None = None,
) -> RegularAiPolicy:
    from .evaluate_hu_turn2_stage8b_topk_mc_rerank import (
        HuTurn2Stage8bTopKMcRerankPolicy,
        parse_topk_configs,
    )

    topk_config = parse_topk_configs(config_string)[0]
    return HuTurn2Stage8bTopKMcRerankPolicy(
        opening_model=bundle.opening,
        turn1_model=bundle.turn1,
        hu_turn1_model=hu_turn1_model,
        hu_turn1_min_margin=hu_turn1_min_margin,
        turn2_model=bundle.turn2,
        turn3_model=bundle.turn3,
        hu_turn3_model=bundle.hu_turn3_stage7,
        hu_turn3_reference_model=bundle.hu_turn3_stage7_reference,
        hu_turn3_min_margin=DEFAULT_HU_TURN3_STAGE7_MIN_MARGIN,
        hu_turn3_reference_min_margin=DEFAULT_HU_TURN3_STAGE7_REFERENCE_MIN_MARGIN,
        hu_turn3_stage7_enabled=True,
        hu_turn2_stage8b_model=bundle.hu_turn2_stage8b,
        topk_rerank_config=topk_config,
        t3_continuation="stage7_m5_r10",
        topk_context={
            "runtime_profile": runtime_profile,
            "runtime_status": runtime_status,
        },
        seed=seed,
        seat=seat,
        opening_lookahead_samples=opening_lookahead_samples,
    )


def _build_stage9f_p2_hu_t1_topk_confirm_policy(
    bundle: ModelBundle,
    *,
    seed: int,
    seat: str,
    opening_lookahead_samples: int,
    config_string: str,
    hu_turn1_candidate_model: Any | None,
    hu_turn1_candidate_models: tuple[Any, ...],
    hu_turn1_safe_selector_model: Any | None,
    runtime_profile: str,
    runtime_status: str,
) -> RegularAiPolicy:
    from .hu_turn1_topk_confirm import (
        HuTurn1TopKConfirmPolicy,
        parse_hu_turn1_topk_confirm_configs,
    )
    from .evaluate_hu_turn2_stage8b_topk_mc_rerank import parse_topk_configs

    t1_topk_config = parse_hu_turn1_topk_confirm_configs(config_string)[0]
    t2_topk_config = parse_topk_configs(DEFAULT_HU_TURN2_STAGE9F_CSE2_CSEMAX2_BOTHSEAT_CONFIG)[0]

    def rollout_factory(*, seed: int, seat: str) -> RegularAiPolicy:
        return _build_stage9f_cse1p5_policy(
            bundle,
            seed=seed,
            seat=seat,
            opening_lookahead_samples=opening_lookahead_samples,
            config_string=DEFAULT_HU_TURN2_STAGE9F_CSE2_CSEMAX2_BOTHSEAT_CONFIG,
            runtime_profile=f"{runtime_profile}_rollout",
            runtime_status="t2_stage9f_p2_continuation",
        )

    return HuTurn1TopKConfirmPolicy(
        opening_model=bundle.opening,
        turn1_model=bundle.turn1,
        turn2_model=bundle.turn2,
        turn3_model=bundle.turn3,
        hu_turn3_model=bundle.hu_turn3_stage7,
        hu_turn3_reference_model=bundle.hu_turn3_stage7_reference,
        hu_turn3_min_margin=DEFAULT_HU_TURN3_STAGE7_MIN_MARGIN,
        hu_turn3_reference_min_margin=DEFAULT_HU_TURN3_STAGE7_REFERENCE_MIN_MARGIN,
        hu_turn3_stage7_enabled=True,
        hu_turn1_candidate_model=hu_turn1_candidate_model,
        hu_turn1_candidate_models=hu_turn1_candidate_models or None,
        hu_turn1_safe_selector_model=hu_turn1_safe_selector_model,
        topk_confirm_config=t1_topk_config,
        hu_turn2_stage8b_model=bundle.hu_turn2_stage8b,
        topk_rerank_config=t2_topk_config,
        t3_continuation="stage7_m5_r10",
        topk_context={
            "runtime_profile": f"{runtime_profile}_t2",
            "runtime_status": "t2_stage9f_p2_continuation",
        },
        rollout_policy_factory=rollout_factory,
        decision_context={
            "runtime_profile": runtime_profile,
            "runtime_status": runtime_status,
            "selective_override_only": True,
            "full_replacement_enabled": False,
            "fallback_policy": "stage9f_p2",
            "t2_continuation": "stage9f_p2",
            "t3_continuation": "stage7_m5_r10",
        },
        seed=seed,
        seat=seat,
        opening_lookahead_samples=opening_lookahead_samples,
    )


def _build_stage19_p0_policy(
    bundle: ModelBundle,
    *,
    seed: int,
    seat: str,
    opening_lookahead_samples: int,
) -> RegularAiPolicy:
    policy = _build_stage9f_p2_hu_t1_topk_confirm_policy(
        bundle,
        seed=seed,
        seat=seat,
        opening_lookahead_samples=opening_lookahead_samples,
        config_string=DEFAULT_HU_TURN1_STAGE18_P1_CONFIG,
        hu_turn1_candidate_model=bundle.hu_turn1_stage18_p1,
        hu_turn1_candidate_models=(),
        hu_turn1_safe_selector_model=bundle.hu_turn1_stage18_p1_safe_selector,
        runtime_profile="stage18_p1",
        runtime_status="p1_fixed",
    )
    candidate_model = bundle.hu_turn0_stage19_p0
    selector_model = bundle.hu_turn0_stage19_p0_safe_selector
    if candidate_model is None or selector_model is None:
        return policy

    policy.hu_turn0_model = candidate_model
    policy.hu_turn0_candidate_topk = DEFAULT_HU_TURN0_STAGE19_P0_TOPK
    policy.hu_turn0_min_margin = DEFAULT_HU_TURN0_STAGE19_P0_MIN_MARGIN_BY_SEAT["first"]
    policy.hu_turn0_min_margin_by_seat = dict(
        DEFAULT_HU_TURN0_STAGE19_P0_MIN_MARGIN_BY_SEAT
    )
    policy.hu_turn0_allowed_seats = DEFAULT_HU_TURN0_STAGE19_P0_ALLOWED_SEATS
    policy.hu_turn0_safe_selector_model = selector_model
    policy.hu_turn0_safe_selector_enabled = True
    policy.hu_turn0_safe_selector_threshold = (
        DEFAULT_HU_TURN0_STAGE19_P0_SAFE_SELECTOR_THRESHOLD_BY_SEAT["first"]
    )
    policy.hu_turn0_safe_selector_threshold_by_seat = dict(
        DEFAULT_HU_TURN0_STAGE19_P0_SAFE_SELECTOR_THRESHOLD_BY_SEAT
    )
    policy.decision_context = card_free_metadata({
        **policy.decision_context,
        "runtime_profile": "stage19_p0",
        "runtime_status": "p0_fixed",
        "selective_override_only": True,
        "full_replacement_enabled": False,
        "fallback_policy": "stage18_p1",
        "t1_continuation": "stage18_p1",
        "t2_continuation": "stage9f_p2",
        "t3_continuation": "stage7_m5_r10",
    })
    return policy


def _build_stage9f_fast_t2_t1_teacher_policy(
    bundle: ModelBundle,
    *,
    seed: int,
    seat: str,
    opening_lookahead_samples: int,
) -> RegularAiPolicy:
    return HuTurn2Stage8SelectiveOverridePolicy(
        opening_model=bundle.opening,
        turn1_model=bundle.turn1,
        turn2_model=bundle.turn2,
        turn3_model=bundle.turn3,
        hu_turn3_model=bundle.hu_turn3_stage7,
        hu_turn3_reference_model=bundle.hu_turn3_stage7_reference,
        hu_turn3_min_margin=DEFAULT_HU_TURN3_STAGE7_MIN_MARGIN,
        hu_turn3_reference_min_margin=DEFAULT_HU_TURN3_STAGE7_REFERENCE_MIN_MARGIN,
        hu_turn3_stage7_enabled=True,
        hu_turn2_stage8_model=bundle.hu_turn2_stage8b,
        hu_turn2_stage8_config=HuTurn2Stage8RuntimeConfig(
            min_margin=DEFAULT_HU_TURN2_STAGE9F_FAST_T1_TEACHER_MIN_MARGIN,
            reference_min_margin=0.0,
            gate_threshold=DEFAULT_HU_TURN2_STAGE9F_FAST_T1_TEACHER_GATE,
            enabled=True,
            allowed_seats=("first", "second"),
        ),
        hu_turn2_context={
            "runtime_profile": "stage9f_fast_t2_t1_teacher",
            "runtime_status": "t1_teacher_fast_continuation",
            "t3_continuation": "stage7_m5_r10",
        },
        seed=seed,
        seat=seat,
        opening_lookahead_samples=opening_lookahead_samples,
    )


def _build_stage9d_policy(
    bundle: ModelBundle,
    *,
    seed: int,
    seat: str,
    opening_lookahead_samples: int,
) -> RegularAiPolicy:
    stage9d_model = getattr(bundle, "hu_turn3_stage9d", None)
    support_model = getattr(bundle, "hu_turn3_stage9d_support", None)
    gate_model = getattr(bundle, "hu_turn3_stage9d_gate", None)
    reference_model = getattr(bundle, "hu_turn3_stage7_reference", None)
    if (
        stage9d_model is None
        or support_model is None
        or gate_model is None
        or reference_model is None
    ):
        return _build_stage7_policy(
            bundle,
            seed=seed,
            seat=seat,
            opening_lookahead_samples=opening_lookahead_samples,
            enabled=True,
            hu_min_margin=DEFAULT_HU_TURN3_STAGE7_MIN_MARGIN,
            reference_min_margin=DEFAULT_HU_TURN3_STAGE7_REFERENCE_MIN_MARGIN,
        )
    return RegularAiPolicy(
        opening_model=bundle.opening,
        turn1_model=bundle.turn1,
        turn2_model=bundle.turn2,
        turn3_model=bundle.turn3,
        hu_turn3_model=stage9d_model,
        hu_turn3_reference_model=reference_model,
        hu_turn3_support_model=support_model,
        hu_turn3_gate_model=gate_model,
        hu_turn3_min_margin=DEFAULT_HU_TURN3_STAGE9D_MIN_MARGIN,
        hu_turn3_reference_min_margin=DEFAULT_HU_TURN3_STAGE9D_REFERENCE_MIN_MARGIN,
        hu_turn3_min_support_margin=DEFAULT_HU_TURN3_STAGE9D_MIN_SUPPORT_MARGIN,
        hu_turn3_min_model_score=DEFAULT_HU_TURN3_STAGE9D_MIN_MODEL_SCORE,
        hu_turn3_min_gate_probability=DEFAULT_HU_TURN3_STAGE9D_MIN_GATE_PROBABILITY,
        hu_turn3_stage7_enabled=True,
        seed=seed,
        seat=seat,
        opening_lookahead_samples=opening_lookahead_samples,
    )


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
