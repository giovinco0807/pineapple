"""FastAPI app for the regular OFC Pineapple trainer (olegsolvers-style).

Local, single-user tool: no auth, SQLite persistence for mistakes.

Run from the regular-ofc-pineapple directory:
    python -m uvicorn trainer.app:app --host 127.0.0.1 --port 8093
"""

from __future__ import annotations

import logging
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent  # regular-ofc-pineapple/
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from trainer import evaluator as trainer_evaluator  # noqa: E402
from trainer.game import TrainingSession  # noqa: E402
from trainer.store import TrainerStore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("trainer.app")

app = FastAPI(title="OFC Regular Trainer")

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

store = TrainerStore(Path(__file__).resolve().parent / "data" / "trainer.sqlite3")

MAX_SESSIONS = 40
_sessions: "OrderedDict[str, TrainingSession]" = OrderedDict()

# session id -> account that started it, so a session keeps writing to the
# account it began under even if the browser switches accounts mid-hand.
_session_accounts: Dict[str, int] = {}


def current_account(x_account_id: Optional[str] = Header(default=None)) -> int:
    """Resolve the acting account from the X-Account-Id header.

    Local tool, so this identifies rather than authenticates: an unknown or
    missing id falls back to the default account instead of erroring, which
    keeps old bookmarks and curl calls working.
    """
    if x_account_id:
        try:
            account_id = int(x_account_id)
        except ValueError:
            return store.default_account_id
        if store.account_exists(account_id):
            return account_id
    return store.default_account_id


def _get_session(session_id: str) -> TrainingSession:
    session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="セッションが見つかりません（サーバー再起動後は新しいハンドを開始してください）")
    return session


def _account_for_session(session_id: str) -> int:
    return _session_accounts.get(session_id, store.default_account_id)


def _persist_mistake(payload: Dict[str, Any]) -> None:
    session_id = payload.get("session_id", "")
    store.add_mistake(account_id=_account_for_session(session_id), **payload)


def _persist_hand(session: TrainingSession) -> None:
    result = session.result or {}
    store.add_hand(
        account_id=_account_for_session(session.id),
        session_id=session.id,
        position=session.position,
        score=result.get("score"),
        total_ev_loss=result.get("total_ev_loss"),
        decisions=len(session.history),
        mistakes=result.get("mistakes", 0),
        summary=result,
    )


# ----------------------------------------------------------------------
# pages
# ----------------------------------------------------------------------


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/health")
async def health():
    return {"ok": True, "evaluator": trainer_evaluator.describe()}


@app.get("/api/stats")
async def stats(account_id: int = Depends(current_account)):
    return store.stats(account_id)


# ----------------------------------------------------------------------
# accounts
# ----------------------------------------------------------------------


@app.get("/api/accounts")
async def accounts_list(account_id: int = Depends(current_account)):
    return {"accounts": store.list_accounts(), "current": account_id}


class AccountRequest(BaseModel):
    name: str


@app.post("/api/accounts")
async def accounts_create(req: AccountRequest):
    try:
        return store.get_or_create_account(req.name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.delete("/api/accounts/{target_id}")
async def accounts_delete(target_id: int):
    if not store.account_exists(target_id):
        raise HTTPException(status_code=404, detail="アカウントが見つかりません")
    try:
        store.delete_account(target_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"ok": True}


# ----------------------------------------------------------------------
# history
# ----------------------------------------------------------------------


@app.get("/api/history")
async def history(limit: int = 100, account_id: int = Depends(current_account)):
    return {
        "hands": store.list_hands(account_id, limit=max(1, min(limit, 500))),
        "stats": store.stats(account_id),
    }


# ----------------------------------------------------------------------
# training
# ----------------------------------------------------------------------


class NewTrainingRequest(BaseModel):
    position: str = Field(default="random", pattern="^(random|first|second)$")
    precision: str = Field(default="standard", pattern="^(fast|standard|high)$")
    mistake_threshold: float = 1.0


@app.post("/api/training/new")
async def training_new(req: NewTrainingRequest, account_id: int = Depends(current_account)):
    session = TrainingSession(
        trainer_evaluator.evaluate_position,
        position=req.position,
        precision=req.precision,
        mistake_threshold=req.mistake_threshold,
        on_mistake=_persist_mistake,
        on_hand_done=_persist_hand,
    )
    _sessions[session.id] = session
    _session_accounts[session.id] = account_id
    while len(_sessions) > MAX_SESSIONS:
        evicted, _ = _sessions.popitem(last=False)
        _session_accounts.pop(evicted, None)
    return {"session_id": session.id, "state": session.state()}


@app.get("/api/training/{session_id}")
async def training_state(session_id: str):
    return _get_session(session_id).state()


class ActRequest(BaseModel):
    placements: Optional[list] = None
    discard: Optional[str] = None
    auto_best: bool = False


@app.post("/api/training/{session_id}/act")
async def training_act(session_id: str, req: ActRequest):
    session = _get_session(session_id)
    try:
        session.act(placements=req.placements, discard=req.discard, auto_best=req.auto_best)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return session.state()


class ContinueRequest(BaseModel):
    use: str = Field(default="user", pattern="^(user|best)$")


@app.post("/api/training/{session_id}/continue")
async def training_continue(session_id: str, req: ContinueRequest):
    session = _get_session(session_id)
    try:
        session.continue_hand(use=req.use)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return session.state()


# ----------------------------------------------------------------------
# editor
# ----------------------------------------------------------------------


class EditorRequest(BaseModel):
    hero_board: Dict[str, list]
    opp_board: Dict[str, list]
    dealt: list
    dead: list = []
    turn: int
    position: str = Field(default="first", pattern="^(first|second)$")
    precision: str = Field(default="standard", pattern="^(fast|standard|high)$")


@app.post("/api/editor/evaluate")
async def editor_evaluate(req: EditorRequest):
    started = time.time()
    try:
        result = trainer_evaluator.evaluate_position(
            hero_board=req.hero_board,
            opp_board=req.opp_board,
            dealt=req.dealt,
            dead=req.dead,
            turn=req.turn,
            position=req.position,
            precision=req.precision,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("editor evaluation failed")
        raise HTTPException(status_code=503, detail=f"評価に失敗しました: {exc}")
    result["turn"] = req.turn
    result["candidate_count"] = len(result.get("candidates") or [])
    result["elapsed"] = time.time() - started
    return result


# ----------------------------------------------------------------------
# mistakes
# ----------------------------------------------------------------------


@app.get("/api/mistakes")
async def mistakes_list(account_id: int = Depends(current_account)):
    return {"mistakes": store.list_mistakes(account_id), "stats": store.stats(account_id)}


@app.get("/api/mistakes/{mistake_id}")
async def mistake_detail(mistake_id: int, account_id: int = Depends(current_account)):
    item = store.get_mistake(mistake_id, account_id)
    if item is None:
        raise HTTPException(status_code=404, detail="ミスが見つかりません")
    return item


class RetryRequest(BaseModel):
    solved: bool = False


@app.post("/api/mistakes/{mistake_id}/retry")
async def mistake_retry(
    mistake_id: int, req: RetryRequest, account_id: int = Depends(current_account)
):
    store.mark_retry(mistake_id, account_id, req.solved)
    return {"ok": True}


@app.delete("/api/mistakes/{mistake_id}")
async def mistake_delete(mistake_id: int, account_id: int = Depends(current_account)):
    store.delete_mistake(mistake_id, account_id)
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8093)
