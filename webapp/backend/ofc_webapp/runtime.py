"""Production adapters for the immutable regular-OFC engines and models.

This module deliberately contains no poker scoring implementation. Final
table points come from the Rust ``score_final`` request, and the display
breakdown/regular Fantasy Land flags come from a second fail-closed call into
the same canonical Rust scoring crate.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from typing import Any, Mapping, Sequence

from ofc_regular.action_key import action_key
from ofc_regular.action_space import Action, generate_actions, generate_turn_actions
from ofc_regular.ai_profiles import ModelPaths, build_policy, load_model_bundle
from ofc_regular.hu_infoset import ActorObservation, ScoringContext
from ofc_regular.hu_late_street_teacher import T4SearchConfig
from ofc_regular.hu_m3_rust import (
    HU_M3_REQUEST_SCHEMA,
    _joint_config_payload,
    evaluate_request,
    load_native_engine,
    t4_request,
)
from ofc_regular.hu_turn2_stage8_runtime import hu_turn2_policy_sample
from ofc_regular.hu_turn3_joint_exact_teacher import JointExactConfig
from ofc_regular.hu_turn3_model import hu_policy_sample
from ofc_regular.hu_turn3_stage3_feature_rust import (
    pinned_feature_encoder_library,
)
from ofc_regular.policy import policy_sample
from ofc_regular.state import Board
from ofc_regular.teacher import evaluate_turn_actions

from .domain import (
    ActionSubmission,
    DecisionObservation,
    FinalScore,
    HiddenFantasylandObservation,
)
from .ports import AIDecision, AIMetadata


_SHA256_REPLACEMENT = "BUILD_TIME"
_ROW_NAMES = ("top", "middle", "bottom")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _action_payload(action: Action, *, score: float, rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "action_key": action_key(action).to_token(),
        "placements": [[card, row] for card, row in action.placements],
        "discards": list(action.discards),
        "score": float(score),
    }


def _normalize_topk(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    """Return exactly three audit rows.

    Older policy surfaces can expose only their selected action. Missing rows
    remain explicit unavailable sentinels instead of fabricated values. The FL
    path is stricter and supplies three real canonical Rust candidates.
    """

    normalized = [dict(row) for row in rows[:3]]
    while len(normalized) < 3:
        normalized.append(
            {
                "rank": len(normalized) + 1,
                "action_key": None,
                "placements": [],
                "discards": [],
                "score": None,
                "available": False,
                "reason": "upstream_evaluator_returns_best_only",
            }
        )
    return tuple(normalized)


def _run_inspector(
    path: Path,
    payload: Mapping[str, Any],
    *,
    timeout: int = 30,
) -> Mapping[str, Any]:
    completed = subprocess.run(
        [str(path)],
        input=json.dumps(
            dict(payload),
            ensure_ascii=True,
            separators=(",", ":"),
        ),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "decision inspector failed: " + completed.stderr[-2000:]
        )
    try:
        parsed = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("decision inspector returned invalid JSON") from exc
    if not isinstance(parsed, dict) or parsed.get("status") != "ok":
        raise RuntimeError("decision inspector returned a non-ok response")
    return parsed


@dataclass(frozen=True)
class Assembly:
    path: Path
    root: Path
    payload: Mapping[str, Any]
    sha256: str

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Assembly":
        configured = path or os.environ.get("OFC_ASSEMBLY_PATH")
        assembly_path = (
            Path(configured).resolve()
            if configured
            else (_repo_root() / "webapp" / "assembly.json").resolve()
        )
        payload = json.loads(assembly_path.read_text(encoding="utf-8"))
        declared = str(payload.get("assembly_sha256", ""))
        if len(declared) == 64 and all(ch in "0123456789abcdef" for ch in declared):
            assembly_sha = declared
        else:
            material = dict(payload)
            material["assembly_sha256"] = _SHA256_REPLACEMENT
            assembly_sha = _canonical_sha(material)
        root = Path(
            os.environ.get(
                "OFC_ASSEMBLY_ROOT",
                "/app" if assembly_path == Path("/app/assembly.json") else _repo_root(),
            )
        ).resolve()
        return cls(assembly_path, root, payload, assembly_sha)

    def weight(self, name: str) -> tuple[Path, str]:
        entry = self.payload["weights"][name]
        path = self._resolve_artifact(
            str(entry["path"]),
            local_fallback=(
                _repo_root()
                / "rust"
                / "hu_m3_engine"
                / "tests"
                / "fixtures"
                / Path(str(entry["path"])).name
            ),
        )
        return path, str(entry["sha256"])

    def policy_artifact(self, name: str) -> tuple[Path, str]:
        entry = self.payload["policies"]["early_streets"]["artifacts"][name]
        path = self._resolve_artifact(
            str(entry["path"]),
            local_fallback=_repo_root() / str(entry["path"]),
        )
        return path, str(entry["sha256"])

    def native_artifact(
        self,
        name: str,
        *,
        env_name: str,
        local_names: Sequence[str],
    ) -> Path:
        configured = os.environ.get(env_name)
        if configured:
            return Path(configured).resolve()
        entry = self.payload["artifacts"][name]
        candidate = self.root / str(entry["path"])
        if candidate.is_file():
            return candidate.resolve()
        search_roots = (
            _repo_root() / "webapp" / ".windows-target" / "release",
            _repo_root() / "webapp" / ".native-target" / "release",
            _repo_root() / "webapp" / ".inspector-target" / "release",
            _repo_root() / "target" / "release",
        )
        for directory in search_roots:
            for local_name in local_names:
                local = directory / local_name
                if local.is_file():
                    return local.resolve()
        raise FileNotFoundError(f"required native artifact is missing: {name}")

    def verify_pinned(self, path: Path, expected_sha: str, *, label: str) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing: {path}")
        actual = _sha256(path)
        if actual != expected_sha:
            raise RuntimeError(
                f"{label} SHA-256 mismatch: expected {expected_sha}, got {actual}"
            )

    def public_meta(self) -> dict[str, Any]:
        result = json.loads(json.dumps(self.payload))
        result["assembly_sha256"] = self.sha256
        return result

    def _resolve_artifact(self, relative: str, *, local_fallback: Path) -> Path:
        candidate = self.root / relative
        if candidate.is_file():
            return candidate.resolve()
        return local_fallback.resolve()


class NativeFinalScoreAdapter:
    """Authoritative final score adapter backed by Rust ``score_final``."""

    def __init__(self, assembly: Assembly) -> None:
        self.assembly = assembly
        self.library_path = assembly.native_artifact(
            "engine_library",
            env_name="OFC_ENGINE_LIBRARY",
            local_names=(
                "ofc_hu_m3_engine.dll",
                "libofc_hu_m3_engine.so",
                "libofc_hu_m3_engine.dylib",
            ),
        )
        self.library = load_native_engine(path=self.library_path)
        self.inspector_path = assembly.native_artifact(
            "decision_inspector",
            env_name="OFC_DECISION_INSPECTOR_PATH",
            local_names=(
                "ofc_webapp_decision_inspector.exe",
                "ofc_webapp_decision_inspector",
            ),
        )

    def score_final(
        self,
        *,
        first_board: Board,
        second_board: Board,
        scoring: ScoringContext,
        first_in_fantasyland: bool,
        second_in_fantasyland: bool,
    ) -> FinalScore:
        response = evaluate_request(
            {
                "schema": HU_M3_REQUEST_SCHEMA,
                "kind": "score_final",
                "hero_board": _board_payload(first_board),
                "opponent_board": _board_payload(second_board),
                "scoring": scoring.to_dict(),
            },
            library=self.library,
        )
        raw_score = float(response["hu_score"])
        rounded = round(raw_score)
        if abs(raw_score - rounded) > 1e-9:
            raise RuntimeError(
                "table settlement requires an integral score_final result; "
                "use the zero-FL-EV settlement context"
            )
        detailed = _run_inspector(
            self.inspector_path,
            {
                "mode": "score_final_detailed",
                "first_board": _board_payload(first_board),
                "second_board": _board_payload(second_board),
                "scoring": scoring.to_dict(),
                "first_in_fantasyland": first_in_fantasyland,
                "second_in_fantasyland": second_in_fantasyland,
            },
        )
        if detailed.get("schema") != "ofc_webapp_score_final_detailed_v1":
            raise RuntimeError("unexpected detailed score schema")
        if detailed.get("source") != "canonical_rust_hu_m3_engine":
            raise RuntimeError("detailed score did not identify canonical Rust")
        try:
            detailed_score = float(detailed["hu_score"])
            components = detailed["point_components"]
            component_total = float(components["total"])
            fantasyland = detailed["fantasyland"]
            first_next = fantasyland["first_next"]
            second_next = fantasyland["second_next"]
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("detailed score response is incomplete") from exc
        if (
            abs(detailed_score - raw_score) > 1e-9
            or abs(component_total - raw_score) > 1e-9
        ):
            raise RuntimeError(
                "detailed canonical Rust score disagrees with score_final"
            )
        if components.get("perspective") != "first":
            raise RuntimeError("detailed score components have wrong perspective")
        if not isinstance(first_next, bool) or not isinstance(second_next, bool):
            raise RuntimeError("detailed Fantasy Land flags must be booleans")
        return FinalScore(
            hu_score=int(rounded),
            breakdown=dict(detailed),
            first_next_fantasyland=first_next,
            second_next_fantasyland=second_next,
        )


class RegularAIRuntime:
    """Street-dispatched production AI with complete decision audit metadata."""

    def __init__(self, assembly: Assembly) -> None:
        self.assembly = assembly
        self._lock = threading.RLock()
        self.library_path = assembly.native_artifact(
            "engine_library",
            env_name="OFC_ENGINE_LIBRARY",
            local_names=(
                "ofc_hu_m3_engine.dll",
                "libofc_hu_m3_engine.so",
                "libofc_hu_m3_engine.dylib",
            ),
        )
        self.library = load_native_engine(path=self.library_path)
        self.fl_solver_path = assembly.native_artifact(
            "fl_solver",
            env_name="OFC_FL_SOLVER_PATH",
            local_names=("regular_fl_solver.exe", "regular_fl_solver"),
        )
        self.inspector_path = assembly.native_artifact(
            "decision_inspector",
            env_name="OFC_DECISION_INSPECTOR_PATH",
            local_names=(
                "ofc_webapp_decision_inspector.exe",
                "ofc_webapp_decision_inspector",
            ),
        )
        self.feature_encoder_path = assembly.native_artifact(
            "feature_encoder",
            env_name="OFC_FEATURE_ENCODER_LIBRARY",
            local_names=(
                "ofc_stage3_feature_encoder.dll",
                "libofc_stage3_feature_encoder.so",
                "libofc_stage3_feature_encoder.dylib",
            ),
        )
        declared_encoder_sha = str(
            assembly.payload["artifacts"]["feature_encoder"]["sha256"]
        )
        self.feature_encoder_sha = (
            _sha256(self.feature_encoder_path)
            if declared_encoder_sha == _SHA256_REPLACEMENT
            else declared_encoder_sha
        )
        self.weights = {
            name: assembly.weight(name)
            for name in ("t4", "t3_second", "t3_first", "t2_second")
        }
        for name, (path, digest) in self.weights.items():
            assembly.verify_pinned(path, digest, label=f"{name} learned weights")
        self.policy_paths, self.policy_shas = self._policy_paths()
        self.bundle = load_model_bundle(self.policy_paths, profiles={"stage19_p0"})
        self.policies = {
            seat: build_policy(
                "stage19_p0",
                self.bundle,
                seed=20260729,
                opening_lookahead_samples=64,
                seat=seat,
            )
            for seat in ("first", "second")
        }

    def decide(self, observation: DecisionObservation) -> AIDecision:
        started = time.perf_counter()
        with self._lock:
            if observation.street == "FL":
                action, evaluator, weights, topk = self._decide_fl(observation)
            elif isinstance(observation, HiddenFantasylandObservation):
                action, evaluator, weights, topk = self._decide_hidden_fl_fallback(
                    observation
                )
            elif self._uses_native_decide(observation):
                action, evaluator, weights, topk = self._decide_native(observation)
            else:
                action, evaluator, weights, topk = self._decide_policy(observation)
        think_ms = max(0, round((time.perf_counter() - started) * 1000))
        return AIDecision(
            action=action,
            think_ms=think_ms,
            meta=AIMetadata(
                evaluator=evaluator,
                weights_sha=tuple(weights),
                assembly_sha=self.assembly.sha256,
                scores_topk=_normalize_topk(topk),
            ),
        )

    def _uses_native_decide(self, observation: ActorObservation) -> bool:
        return (
            observation.street in {"T3", "T4"}
            or (
                observation.street == "T2"
                and observation.to_act_order == "second"
            )
        )

    def _decide_native(
        self, observation: ActorObservation
    ) -> tuple[ActionSubmission, str, Sequence[str], Sequence[Mapping[str, Any]]]:
        seed = int(observation.fingerprint()[:16], 16) & ((1 << 63) - 1)
        config = JointExactConfig(
            candidate_samples=1,
            evaluation_samples=1,
            downstream_t3_samples=1,
            downstream_t4_samples=0,
            seed=seed,
            run_id=f"ofc-webapp:{observation.fingerprint()[:16]}",
            seat=observation.seat,
            to_act_order=observation.to_act_order,
            learned_t4_model_path=str(self.weights["t4"][0]),
            learned_t4_model_sha256=self.weights["t4"][1],
            learned_t3_second_model_path=str(self.weights["t3_second"][0]),
            learned_t3_second_model_sha256=self.weights["t3_second"][1],
            learned_t3_first_model_path=str(self.weights["t3_first"][0]),
            learned_t3_first_model_sha256=self.weights["t3_first"][1],
            learned_t2_second_model_path=str(self.weights["t2_second"][0]),
            learned_t2_second_model_sha256=self.weights["t2_second"][1],
        )
        response = evaluate_request(
            {
                "schema": HU_M3_REQUEST_SCHEMA,
                "kind": "decide",
                "observation": observation.to_dict(),
                "observation_fingerprint": observation.fingerprint(),
                "config": _joint_config_payload(config),
            },
            library=self.library,
        )
        action = ActionSubmission.from_parts(
            response["placements"], response.get("discards", ())
        )
        if observation.street == "T4":
            ranked_response = evaluate_request(
                t4_request(
                    observation,
                    config=T4SearchConfig(
                        candidate_samples=0,
                        evaluation_samples=0,
                        seed=seed,
                        run_id=f"ofc-webapp-topk:{observation.fingerprint()[:16]}",
                    ),
                ),
                library=self.library,
            )
            rows = sorted(
                ranked_response["actions"],
                key=lambda row: (int(row["sorted_index"]), str(row["action_key"])),
            )[:3]
            topk = [
                {
                    "rank": rank,
                    "action_key": row["action_key"],
                    "placements": row["placements"],
                    "discards": row["discards"],
                    "score": row["score"],
                }
                for rank, row in enumerate(rows, 1)
            ]
            return action, "rust_decide:exact_enumeration", (
                self.weights["t4"][1],
            ), topk

        mode = (
            "t2_second"
            if observation.street == "T2"
            else f"t3_{observation.to_act_order}"
        )
        weight_name = {
            "t2_second": "t2_second",
            "t3_first": "t3_first",
            "t3_second": "t3_second",
        }[mode]
        inspected = self._inspect_native(
            mode=mode,
            observation=observation,
            model_path=self.weights[weight_name][0],
            model_sha=self.weights[weight_name][1],
        )
        return (
            action,
            f"rust_decide:learned:{mode}",
            (self.weights[weight_name][1],),
            inspected["scores_topk"],
        )

    def _inspect_native(
        self,
        *,
        mode: str,
        observation: ActorObservation,
        model_path: Path,
        model_sha: str,
    ) -> Mapping[str, Any]:
        return _run_inspector(
            self.inspector_path,
            {
                "mode": mode,
                "observation": observation.to_dict(),
                "model_path": str(model_path),
                "model_sha256": model_sha,
                "top_k": 3,
            },
        )

    def _decide_policy(
        self, observation: ActorObservation
    ) -> tuple[ActionSubmission, str, Sequence[str], Sequence[Mapping[str, Any]]]:
        policy = self.policies[observation.seat]
        seed = int(observation.fingerprint()[:16], 16) & ((1 << 63) - 1)
        context = pinned_feature_encoder_library(
            self.feature_encoder_path,
            expected_sha256=self.feature_encoder_sha,
        )
        with context:
            chosen = policy.choose_action_observation(
                observation,
                hand_id=observation.fingerprint()[:16],
                game_id="ofc-webapp",
                decision_seed=seed,
            )
            topk = self._policy_topk(observation, chosen)
        return (
            ActionSubmission(
                tuple(chosen.placements), tuple(chosen.discards)
            ),
            f"python_policy:stage19_p0:{observation.street}",
            self.policy_shas,
            topk,
        )

    def _decide_hidden_fl_fallback(
        self, observation: HiddenFantasylandObservation
    ) -> tuple[ActionSubmission, str, Sequence[str], Sequence[Mapping[str, Any]]]:
        """Run the production policy without exposing an FL opponent's cards."""

        policy = self.policies[observation.seat]
        seed_material = json.dumps(
            observation.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("ascii")
        seed = int(hashlib.sha256(seed_material).hexdigest()[:16], 16)
        with pinned_feature_encoder_library(
            self.feature_encoder_path,
            expected_sha256=self.feature_encoder_sha,
        ):
            chosen = policy.choose_action(
                observation.hero_board,
                observation.dealt_cards,
                dead_cards=observation.hero_private_discards,
                opponent_board=None,
                hand_id=hashlib.sha256(seed_material).hexdigest()[:16],
                game_id="ofc-webapp-hidden-fl",
                decision_seed=seed,
                street=observation.street,
            )
            actions = _legal_for_observation(observation)
            if observation.street == "T4":
                exact = evaluate_turn_actions(
                    observation.hero_board,
                    observation.dealt_cards,
                    opponent_board=None,
                    fl_ev=dict(observation.scoring.fl_ev),
                )
                topk = [
                    _action_payload(item.action, score=item.score, rank=rank)
                    for rank, item in enumerate(exact[:3], 1)
                ]
            else:
                model = _baseline_model(self.bundle, observation.street)
                topk = _model_topk(
                    model,
                    policy_sample(
                        observation.hero_board,
                        observation.dealt_cards,
                        actions,
                    ),
                    actions,
                    chosen,
                )
        return (
            ActionSubmission(tuple(chosen.placements), tuple(chosen.discards)),
            f"python_policy:hidden_fl_safe_fallback:{observation.street}",
            self.policy_shas,
            topk,
        )

    def _policy_topk(
        self, observation: ActorObservation, chosen: Action
    ) -> Sequence[Mapping[str, Any]]:
        actions = _legal_for_observation(observation)
        if observation.street == "T0":
            model = self.bundle.opening
            sample = policy_sample(
                observation.hero_board, observation.dealt_cards, actions
            )
        elif observation.street == "T1":
            model = self.bundle.hu_turn1_stage18_p1 or self.bundle.turn1
            sample = hu_policy_sample(
                observation.hero_board,
                observation.dealt_cards,
                actions,
                opponent_board=observation.opponent_public_board,
                dead_cards=observation.legacy_dead_cards(),
                seat=observation.seat,
                to_act_order=observation.to_act_order,
            )
        elif observation.street == "T2":
            model = self.bundle.hu_turn2_stage8b or self.bundle.turn2
            sample = hu_turn2_policy_sample(
                observation.hero_board,
                observation.dealt_cards,
                actions,
                opponent_board=observation.opponent_public_board,
                dead_cards=observation.legacy_dead_cards(),
                seat=observation.seat,
                to_act_order=observation.to_act_order,
            )
        else:
            model = self.bundle.turn3
            sample = policy_sample(
                observation.hero_board, observation.dealt_cards, actions
            )
        return _model_topk(model, sample, actions, chosen)

    def _decide_fl(
        self, observation: ActorObservation
    ) -> tuple[ActionSubmission, str, Sequence[str], Sequence[Mapping[str, Any]]]:
        stay_bonus = dict(observation.scoring.fl_ev).get(14, 0.0)
        completed = subprocess.run(
            [
                str(self.fl_solver_path),
                "--solve",
                ",".join(observation.dealt_cards),
                "--stay-bonus",
                str(stay_bonus),
            ],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
            timeout=30,
        )
        if completed.returncode != 0:
            raise RuntimeError("regular_fl_solver failed: " + completed.stderr[-2000:])
        parsed = _parse_fl_solution(completed.stdout)
        placements = tuple(
            (card, row)
            for row in _ROW_NAMES
            for card in parsed[row]
        )
        action = ActionSubmission(placements, (parsed["discard"][0],))
        inspected = _run_inspector(
            self.inspector_path,
            {
                "mode": "fantasyland_topk",
                "cards": list(observation.dealt_cards),
                "stay_bonus": stay_bonus,
                "top_k": 3,
            },
        )
        if inspected.get("schema") != "ofc_webapp_fantasyland_topk_v1":
            raise RuntimeError("unexpected Fantasy Land top-k schema")
        if inspected.get("source") != "canonical_rust_hu_m3_engine":
            raise RuntimeError("Fantasy Land top-k is not canonical Rust")
        rows = inspected.get("scores_topk")
        if not isinstance(rows, list) or len(rows) != 3:
            raise RuntimeError("Fantasy Land inspector must return three candidates")
        for rank, row in enumerate(rows, 1):
            if not isinstance(row, dict) or row.get("rank") != rank:
                raise RuntimeError("Fantasy Land inspector returned invalid ranks")
            score = row.get("score")
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
            ):
                raise RuntimeError(
                    "Fantasy Land inspector returned a non-real candidate"
                )
        inspected_best = ActionSubmission.from_parts(
            rows[0].get("placements", ()),
            rows[0].get("discards", ()),
        )
        solver_exact_score = float(parsed["total_royalty"]) + (
            float(stay_bonus) if parsed["can_stay"] else 0.0
        )
        if (
            inspected_best != action
            or abs(float(rows[0]["score"]) - solver_exact_score) > 1e-9
            or rows[0].get("can_stay") is not parsed["can_stay"]
        ):
            raise RuntimeError(
                "canonical Rust Fantasy Land rank 1 disagrees with "
                "regular_fl_solver: "
                f"solver={action!r}/{solver_exact_score}/{parsed['can_stay']}, "
                f"inspector={inspected_best!r}/{rows[0]['score']}/"
                f"{rows[0].get('can_stay')}"
            )
        solver_sha = _sha256(self.fl_solver_path)
        inspector_sha = _sha256(self.inspector_path)
        return (
            action,
            "regular_fl_solver+canonical_rust_top3",
            (solver_sha, inspector_sha),
            rows,
        )

    def _policy_paths(self) -> tuple[ModelPaths, tuple[str, ...]]:
        names = (
            "opening",
            "turn1",
            "turn2",
            "turn3",
            "t0_candidate",
            "t0_safe_selector",
            "t1_candidate",
            "t1_safe_selector",
            "t2_reference",
            "t3_candidate",
            "t3_reference",
        )
        resolved = {name: self.assembly.policy_artifact(name) for name in names}
        for name, (path, digest) in resolved.items():
            self.assembly.verify_pinned(path, digest, label=f"{name} policy weights")
        paths = ModelPaths(
            opening=resolved["opening"][0],
            turn1=resolved["turn1"][0],
            turn2=resolved["turn2"][0],
            turn3=resolved["turn3"][0],
            hu_turn0_stage19_p0=resolved["t0_candidate"][0],
            hu_turn0_stage19_p0_safe_selector=resolved["t0_safe_selector"][0],
            hu_turn1_stage18_p1=resolved["t1_candidate"][0],
            hu_turn1_stage18_p1_safe_selector=resolved["t1_safe_selector"][0],
            hu_turn2_stage8b=resolved["t2_reference"][0],
            hu_turn3_stage7=resolved["t3_candidate"][0],
            hu_turn3_stage7_reference=resolved["t3_reference"][0],
        )
        return paths, tuple(resolved[name][1] for name in names)


def settlement_scoring_context() -> ScoringContext:
    """Standard table points: FL is a next-hand state, not bought as EV now."""

    return ScoringContext(fl_ev=((14, 0.0),), fantasyland_cards=14)


def evaluation_scoring_context() -> ScoringContext:
    """Assembly-consistent AI evaluation context."""

    return ScoringContext(fantasyland_cards=14)


def _legal_for_observation(observation: DecisionObservation) -> list[Action]:
    if observation.street == "T0":
        return generate_actions(
            observation.hero_board, observation.dealt_cards
        )
    return generate_turn_actions(
        observation.hero_board, observation.dealt_cards
    )


def _baseline_model(bundle: Any, street: str) -> Any:
    return {
        "T0": bundle.opening,
        "T1": bundle.turn1,
        "T2": bundle.turn2,
        "T3": bundle.turn3,
    }.get(street, bundle.turn3)


def _model_topk(
    model: Any,
    sample: Mapping[str, Any],
    actions: Sequence[Action],
    chosen: Action,
) -> Sequence[Mapping[str, Any]]:
    if model is None:
        chosen_row = _action_payload(chosen, score=0.0, rank=1)
        chosen_row["score_role"] = "selected_without_model_score"
        return (chosen_row,)
    predicted = model.predict_sample(dict(sample))
    # The Stage8 T2 network is multi-head; column zero is its absolute-EV head.
    if getattr(predicted, "ndim", 1) == 2:
        predicted = predicted[:, 0]
    values = [float(value) for value in predicted]
    ranked = sorted(
        range(len(actions)),
        key=lambda index: (-values[index], action_key(actions[index]).sort_key()),
    )
    rows = [
        _action_payload(actions[index], score=values[index], rank=rank)
        for rank, index in enumerate(ranked[:3], 1)
    ]
    selected_key = action_key(chosen).to_token()
    for row in rows:
        row["selected"] = row["action_key"] == selected_key
        row["score_role"] = "diagnostic_model_value"
    return rows


def _parse_fl_solution(output: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        if key in _ROW_NAMES:
            parsed[key] = value.split()
        elif key == "discard":
            parsed[key] = value.split()
        elif key == "can stay":
            parsed["can_stay"] = value.lower() == "true"
        elif key == "royalties":
            fields = dict(
                item.split("=", 1)
                for item in value.split()
                if "=" in item
            )
            if "total" in fields:
                parsed["total_royalty"] = int(fields["total"])
        elif key == "score":
            parsed["score"] = float(value)
    required = {
        "top",
        "middle",
        "bottom",
        "discard",
        "can_stay",
        "total_royalty",
        "score",
    }
    missing = required - parsed.keys()
    if missing:
        raise RuntimeError(
            "regular_fl_solver output is missing: " + ", ".join(sorted(missing))
        )
    return parsed


def _board_payload(board: Board) -> dict[str, list[str]]:
    return {
        "top": list(board.top),
        "middle": list(board.middle),
        "bottom": list(board.bottom),
    }


__all__ = [
    "Assembly",
    "NativeFinalScoreAdapter",
    "RegularAIRuntime",
    "evaluation_scoring_context",
    "settlement_scoring_context",
]
