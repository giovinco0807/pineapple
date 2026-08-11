"""FastAPI application for a private human-vs-AI regular OFC match."""

from __future__ import annotations

from collections import defaultdict
from contextlib import asynccontextmanager
import logging
import mimetypes
import os
from pathlib import Path
import secrets
import threading
from typing import Any, Callable, Iterator, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Response, status
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from ofc_regular.hu_m3_rust import engine_version
from ofc_regular.state import Board

from .domain import (
    ActionSubmission,
    DomainError,
    HandState,
    HandStatus,
    MatchState,
    Player,
    apply_action,
    build_observation,
    continue_match,
    create_match,
    player_index,
    settle_hand,
    start_hand,
)
from .ports import (
    AIDecisionPort,
    AIMetadata,
    DatabaseBackupPort,
    FinalScorePort,
)
from .repository import DecisionRecord, SQLiteRepository
from .runtime import (
    Assembly,
    NativeFinalScoreAdapter,
    RegularAIRuntime,
    evaluation_scoring_context,
    settlement_scoring_context,
)
from .serialization import serialize_hand, serialize_match
from .storage import GCSDatabaseBackup


LOGGER = logging.getLogger("ofc_webapp")
APP_VERSION = "0.1.0"

# Windows' registry-backed mimetypes table can map module scripts to
# ``text/plain``.  Module browsers then refuse to execute the production
# bundle, so pin the web types before Starlette creates FileResponse objects.
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("text/css", ".css")
mimetypes.add_type("application/manifest+json", ".webmanifest")


class CreateMatchRequest(BaseModel):
    seed: int | None = None


class ActionRequest(BaseModel):
    placements: list[tuple[str, Literal["top", "middle", "bottom"]]]
    discards: list[str] = Field(default_factory=list)

    def submission(self) -> ActionSubmission:
        return ActionSubmission.from_parts(self.placements, self.discards)


class ContinueRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    should_continue: bool = Field(alias="continue")


class AppServices:
    def __init__(
        self,
        *,
        assembly: Assembly,
        repository: SQLiteRepository,
        ai: AIDecisionPort,
        scorer: FinalScorePort,
        backup: DatabaseBackupPort,
        shared_token: str,
    ) -> None:
        self.assembly = assembly
        self.repository = repository
        self.ai = ai
        self.scorer = scorer
        self.backup = backup
        self.shared_token = shared_token
        self.locks: defaultdict[str, threading.RLock] = defaultdict(
            threading.RLock
        )
        self.last_backup_error: str | None = None

    def lock(self, match_id: str) -> threading.RLock:
        return self.locks[match_id]


def create_app(
    *,
    assembly: Assembly | None = None,
    repository: SQLiteRepository | None = None,
    ai_factory: Callable[[Assembly], AIDecisionPort] = RegularAIRuntime,
    scorer_factory: Callable[[Assembly], FinalScorePort] = NativeFinalScoreAdapter,
    backup: DatabaseBackupPort | None = None,
    shared_token: str | None = None,
    static_dir: str | Path | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Iterator[None]:
        active_assembly = assembly or Assembly.load()
        db_path = Path(
            os.environ.get("OFC_DB_PATH", str(_default_db_path()))
        ).resolve()
        active_backup = backup or GCSDatabaseBackup()
        if repository is None:
            active_backup.restore(db_path)
            active_repository = SQLiteRepository(db_path)
        else:
            active_repository = repository
        active_repository.initialize()
        token = (
            shared_token
            if shared_token is not None
            else os.environ.get("SHARED_TOKEN", "")
        )
        if not token:
            raise RuntimeError("SHARED_TOKEN must not be empty")
        app.state.services = AppServices(
            assembly=active_assembly,
            repository=active_repository,
            ai=ai_factory(active_assembly),
            scorer=scorer_factory(active_assembly),
            backup=active_backup,
            shared_token=token,
        )
        yield

    app = FastAPI(
        title="Regular OFC Pineapple",
        version=APP_VERSION,
        lifespan=lifespan,
    )

    def services() -> AppServices:
        value = getattr(app.state, "services", None)
        if value is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="service is starting",
            )
        return value

    async def require_bearer(
        authorization: str | None = Header(default=None),
        service: AppServices = Depends(services),
    ) -> None:
        expected = f"Bearer {service.shared_token}"
        if authorization is None or not secrets.compare_digest(
            authorization, expected
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    auth = [Depends(require_bearer)]

    @app.get("/api/healthz")
    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/meta", dependencies=auth)
    def meta(service: AppServices = Depends(services)) -> dict[str, Any]:
        payload = service.assembly.public_meta()
        engine = getattr(service.scorer, "library", None)
        return {
            "app_version": APP_VERSION,
            "assembly_sha": service.assembly.sha256,
            "assembly": payload.get("dispatch", {}),
            "weights": {
                name: value["sha256"]
                for name, value in payload.get("weights", {}).items()
            },
            "rules": {
                **payload.get("rules", {}),
                "settlement_fl_ev": {"14": 0.0},
                "ai_evaluation_fl_ev": evaluation_scoring_context().to_dict()[
                    "fl_ev"
                ],
                "scoring_authority": "rust_score_final",
            },
            "artifacts": payload.get("artifacts", {}),
            "engine_version": (
                engine_version(library=engine) if engine is not None else None
            ),
            "storage": {
                "gcs_enabled": bool(
                    getattr(service.backup, "enabled", False)
                ),
                "last_backup_error": service.last_backup_error,
            },
        }

    @app.post("/api/match", dependencies=auth)
    def new_match(
        request: CreateMatchRequest,
        service: AppServices = Depends(services),
    ) -> dict[str, Any]:
        match = create_match(
            seed=request.seed,
            assembly_sha=service.assembly.sha256,
            app_version=APP_VERSION,
        )
        service.repository.save_match(match)
        return serialize_match(match)

    @app.get("/api/match/{match_id}", dependencies=auth)
    def get_match(
        match_id: str,
        service: AppServices = Depends(services),
    ) -> dict[str, Any]:
        match = _require_match(service.repository, match_id)
        hands = service.repository.list_hands(match_id)
        payload = serialize_match(match, hands=hands)
        payload["assembly_sha"] = match.assembly_sha
        return payload

    @app.post("/api/match/{match_id}/hand", dependencies=auth)
    def new_hand(
        match_id: str,
        service: AppServices = Depends(services),
    ) -> dict[str, Any]:
        with service.lock(match_id):
            match = _require_match(service.repository, match_id)
            match, hand = start_hand(match)
            service.repository.save_transition(match=match, hand=hand)
            match, hand = _advance_ai(service, match, hand)
            return _hand_payload(service.repository, hand)

    @app.get("/api/hand/{hand_id}", dependencies=auth)
    def get_hand(
        hand_id: str,
        service: AppServices = Depends(services),
    ) -> dict[str, Any]:
        hand = _require_hand(service.repository, hand_id)
        return _hand_payload(service.repository, hand)

    @app.post("/api/hand/{hand_id}/action", dependencies=auth)
    def normal_action(
        hand_id: str,
        request: ActionRequest,
        service: AppServices = Depends(services),
    ) -> dict[str, Any]:
        hand = _require_hand(service.repository, hand_id)
        with service.lock(hand.match_id):
            hand = _require_hand(service.repository, hand_id)
            match = _require_match(service.repository, hand.match_id)
            if hand.current_turn is None or hand.current_turn.actor != "human":
                raise DomainError("it is not the human player's turn")
            if hand.current_turn.street == "FL":
                raise DomainError("use the fantasyland placement endpoint")
            transition = apply_action(
                hand, request.submission(), actor="human"
            )
            hand = transition.hand
            decision = DecisionRecord.from_applied(
                transition.decision,
                hand_id=hand.id,
                think_ms=0,
            )
            service.repository.save_transition(
                match=match, hand=hand, decision=decision
            )
            match, hand = _advance_ai(service, match, hand)
            return _hand_payload(service.repository, hand)

    @app.post("/api/hand/{hand_id}/fl-placement", dependencies=auth)
    def fantasyland_action(
        hand_id: str,
        request: ActionRequest,
        service: AppServices = Depends(services),
    ) -> dict[str, Any]:
        hand = _require_hand(service.repository, hand_id)
        with service.lock(hand.match_id):
            hand = _require_hand(service.repository, hand_id)
            match = _require_match(service.repository, hand.match_id)
            if hand.current_turn is None or hand.current_turn.actor != "human":
                raise DomainError("it is not the human player's turn")
            if hand.current_turn.street != "FL":
                raise DomainError("current turn is not Fantasy Land")
            transition = apply_action(
                hand, request.submission(), actor="human"
            )
            hand = transition.hand
            decision = DecisionRecord.from_applied(
                transition.decision,
                hand_id=hand.id,
                think_ms=0,
            )
            service.repository.save_transition(
                match=match, hand=hand, decision=decision
            )
            match, hand = _advance_ai(service, match, hand)
            return _hand_payload(service.repository, hand)

    @app.post("/api/match/{match_id}/continue", dependencies=auth)
    def continue_or_finish(
        match_id: str,
        request: ContinueRequest,
        service: AppServices = Depends(services),
    ) -> dict[str, Any]:
        with service.lock(match_id):
            match = _require_match(service.repository, match_id)
            match = continue_match(
                match, should_continue=request.should_continue
            )
            service.repository.save_match(match)
            _backup_database(service)
            return serialize_match(
                match, hands=service.repository.list_hands(match_id)
            )

    @app.get("/api/match/{match_id}/export", dependencies=auth)
    def export_match(
        match_id: str,
        service: AppServices = Depends(services),
    ) -> PlainTextResponse:
        _require_match(service.repository, match_id)
        content = service.repository.export_jsonl(match_id)
        return PlainTextResponse(
            content,
            media_type="application/x-ndjson",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="ofc-match-{match_id}.jsonl"'
                )
            },
        )

    @app.exception_handler(DomainError)
    async def domain_error_handler(_request: Any, exc: DomainError) -> Response:
        from fastapi.responses import JSONResponse

        return JSONResponse(status_code=400, content={"detail": str(exc)})

    resolved_static = Path(
        static_dir
        or os.environ.get(
            "OFC_STATIC_DIR", str(_repo_root() / "webapp" / "frontend" / "dist")
        )
    )
    if resolved_static.is_dir():
        app.mount(
            "/",
            StaticFiles(directory=str(resolved_static), html=True),
            name="frontend",
        )
    return app


def _default_db_path() -> Path:
    return _repo_root() / "webapp" / "data" / "ofc-webapp.sqlite3"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _require_match(repository: SQLiteRepository, match_id: str) -> MatchState:
    match = repository.get_match(match_id)
    if match is None:
        raise HTTPException(status_code=404, detail="match not found")
    return match


def _require_hand(repository: SQLiteRepository, hand_id: str) -> HandState:
    hand = repository.get_hand(hand_id)
    if hand is None:
        raise HTTPException(status_code=404, detail="hand not found")
    return hand


def _backup_database(service: AppServices) -> None:
    try:
        service.backup.backup(service.repository.path)
        service.last_backup_error = None
    except Exception as exc:  # pragma: no cover - cloud-only failure path
        LOGGER.exception("GCS SQLite backup failed")
        service.last_backup_error = str(exc)


def _advance_ai(
    service: AppServices,
    match: MatchState,
    hand: HandState,
) -> tuple[MatchState, HandState]:
    while (
        hand.status == HandStatus.PLAYING
        and hand.current_turn is not None
        and hand.current_turn.actor == "ai"
    ):
        observation = build_observation(
            hand, actor="ai", scoring=evaluation_scoring_context()
        )
        ai_decision = service.ai.decide(observation)
        transition = apply_action(
            hand, ai_decision.action, actor="ai"
        )
        hand = transition.hand
        record = DecisionRecord.from_applied(
            transition.decision,
            hand_id=hand.id,
            think_ms=ai_decision.think_ms,
            ai_meta=ai_decision.meta,
        )
        service.repository.save_transition(
            match=match, hand=hand, decision=record
        )

    if hand.status == HandStatus.AWAITING_SCORE:
        first_player: Player = (
            "human" if hand.positions[0] == "first" else "ai"
        )
        second_player: Player = "ai" if first_player == "human" else "human"
        score = service.scorer.score_final(
            first_board=hand.board_for(first_player),
            second_board=hand.board_for(second_player),
            scoring=settlement_scoring_context(),
            first_in_fantasyland=hand.in_fantasyland(first_player),
            second_in_fantasyland=hand.in_fantasyland(second_player),
        )
        match, hand = settle_hand(match, hand, score)
        service.repository.save_transition(match=match, hand=hand)
        _backup_database(service)
    return match, hand


def _hand_payload(
    repository: SQLiteRepository, hand: HandState
) -> dict[str, Any]:
    payload = serialize_hand(hand)
    payload["replay"] = _replay_payload(
        hand, repository.list_decisions(hand.id)
    )
    return payload


def _replay_payload(
    hand: HandState, decisions: list[DecisionRecord]
) -> list[dict[str, Any]]:
    boards = [Board(), Board()]
    rows: list[dict[str, Any]] = []
    for index, decision in enumerate(decisions):
        actor_index = player_index(decision.actor)
        boards[actor_index] = boards[actor_index].place(
            decision.placements
        )
        visible_ai_board = (
            Board()
            if hand.status != HandStatus.COMPLETE
            and hand.in_fantasyland("ai")
            else boards[1]
        )
        rows.append(
            {
                "id": decision.id,
                "index": index,
                "actor": decision.actor,
                "street": decision.street,
                "label": f"{decision.street} · "
                + ("あなた" if decision.actor == "human" else "AI"),
                "boards": {
                    "human": _board_dict(boards[0]),
                    "ai": _board_dict(visible_ai_board),
                },
                "dealt_cards": (
                    list(decision.dealt_cards)
                    if decision.actor == "human"
                    else []
                ),
                "discards": (
                    list(decision.discards)
                    if decision.actor == "human"
                    else []
                ),
                "think_ms": decision.think_ms,
                "ai_meta": (
                    _public_ai_meta(decision.ai_meta)
                    if decision.ai_meta is not None
                    and hand.status == HandStatus.COMPLETE
                    else None
                ),
            }
        )
    return rows


def _public_ai_meta(meta: AIMetadata) -> dict[str, Any]:
    """Expose evaluation values without candidate cards or action keys."""

    safe_candidate_fields = {
        "rank",
        "score",
        "available",
        "selected",
        "score_role",
        "reason",
        "can_stay",
    }
    return {
        "evaluator": meta.evaluator,
        "weights_sha": list(meta.weights_sha),
        "assembly_sha": meta.assembly_sha,
        "scores_topk": [
            {
                key: value
                for key, value in candidate.items()
                if key in safe_candidate_fields
            }
            for candidate in meta.scores_topk
        ],
    }


def _board_dict(board: Board) -> dict[str, list[str]]:
    return {
        "top": list(board.top),
        "middle": list(board.middle),
        "bottom": list(board.bottom),
    }


app = create_app()


__all__ = ["APP_VERSION", "AppServices", "app", "create_app"]
