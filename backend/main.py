"""
OFC Pineapple Web App - FastAPI Backend
"""
import uuid
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.room import GameRoom, rooms, cleanup_stale_rooms
from backend.ai_player import AI_PLAYER_ID, init_ai
from backend.handlers import handle_message

app = FastAPI(title="OFC Pineapple", version="1.0.0")


@app.on_event("startup")
async def startup():
    import asyncio
    asyncio.create_task(cleanup_stale_rooms())

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check for Render
@app.get("/health")
async def health():
    return {"status": "ok"}


# Serve frontend static files (production)
_frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if _frontend_dist.exists():
    from starlette.staticfiles import StaticFiles
    from starlette.responses import FileResponse

    @app.get("/")
    async def serve_root():
        return FileResponse(str(_frontend_dist / "index.html"))

    # Mount static assets
    if (_frontend_dist / "assets").exists():
        app.mount("/assets", StaticFiles(directory=str(_frontend_dist / "assets")), name="assets")

    # SPA fallback - serve index.html for unmatched routes
    @app.get("/{path:path}")
    async def spa_fallback(path: str):
        file_path = _frontend_dist / path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(_frontend_dist / "index.html"))


# ========== API ROUTES ==========
@app.get("/")
async def root():
    return {"message": "OFC Pineapple Server", "status": "running"}


@app.post("/api/rooms")
async def create_room(vs_ai: bool = False):
    room_id = str(uuid.uuid4())[:8]
    room = GameRoom(room_id, is_ai_game=vs_ai)
    if vs_ai:
        if not init_ai():
            return {"error": "AI model not available"}
        room.players.append(AI_PLAYER_ID)
    rooms[room_id] = room
    return {"room_id": room_id, "vs_ai": vs_ai}


@app.get("/api/rooms")
async def list_rooms():
    return {
        "rooms": [
            {"room_id": r.room_id, "players": len(r.players),
             "status": "waiting" if len(r.players) < 2 else "full"}
            for r in rooms.values()
        ]
    }


@app.get("/api/logs/{room_id}")
async def get_logs(room_id: str):
    room = rooms.get(room_id)
    if not room:
        return {"error": "Room not found"}
    return {"logs": room.logs}


@app.get("/api/export")
async def export_logs():
    all_logs = []
    for room in rooms.values():
        all_logs.extend(room.logs)
    return {"logs": all_logs, "count": len(all_logs)}


# ========== WEBSOCKET ==========
@app.websocket("/ws/{room_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str, player_id: Optional[str] = None):
    await websocket.accept()

    if room_id not in rooms:
        rooms[room_id] = GameRoom(room_id)
    room = rooms[room_id]

    if not player_id:
        player_id = str(uuid.uuid4())[:8]

    if len(room.players) >= 2 and player_id not in room.players:
        await websocket.send_json({"type": "error", "message": "Room is full"})
        await websocket.close()
        return

    if player_id not in room.players:
        room.players.append(player_id)
    room.websockets[player_id] = websocket
    seat = room.players.index(player_id)

    await websocket.send_json({
        "type": "connected",
        "room_id": room_id,
        "player_id": player_id,
        "seat": seat,
        "players_in_room": len(room.players)
    })

    await room.broadcast({
        "type": "player_joined",
        "player_id": player_id,
        "seat": seat,
        "players_in_room": len(room.players)
    }, exclude=player_id)

    if len(room.players) == 2:
        await room.broadcast({"type": "ready_to_start", "players": room.players})

    try:
        while True:
            data = await websocket.receive_json()
            await handle_message(room, player_id, seat, data)
    except WebSocketDisconnect:
        if player_id in room.websockets:
            del room.websockets[player_id]
        await room.broadcast({"type": "player_disconnected", "player_id": player_id})


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
