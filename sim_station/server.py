"""HTTP + WebSocket front for the live twin. Binary JPEG frames at ~30 fps and a JSON snapshot at 10 Hz share one socket."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from sim_station import audio as A
from pydantic import BaseModel

from sim_station.station import Station

STATIC = Path(__file__).resolve().parent / "static"
app = FastAPI(title="Everest dual-YAM twin")
station: Optional[Station] = None


def st() -> Station:
    global station
    if station is None:
        station = Station()
    return station


class ResetReq(BaseModel):
    level: Optional[int] = None
    material: Optional[str] = None
    seed: Optional[int] = None
    randomize: Optional[bool] = None


class SpeedReq(BaseModel):
    speed: float


class ScrubReq(BaseModel):
    id: str


class CamReq(BaseModel):
    az: Optional[float] = None
    el: Optional[float] = None
    dist: Optional[float] = None
    lookat: Optional[List[float]] = None
    ortho: Optional[bool] = None
    follow: Optional[bool] = None
    overlay: Optional[bool] = None


class PickReq(BaseModel):
    x: float
    y: float


class MCReq(BaseModel):
    levels: List[int] = [1, 2, 3, 4, 5]
    trials: int = 50


class DanceReq(BaseModel):
    song: str = "rick"
    t: float = 0.0
    playing: bool = False
    sync_ms: float = 0.0


class GridReq(BaseModel):
    song: str
    bpm: float
    downbeat_s: float


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})


@app.get("/audio/{sid}")
def audio_file(sid: str):
    meta = A.SONGS.get(sid)
    path = A.ASSETS / meta["file"] if meta else None
    if path is None or not path.exists():
        return JSONResponse({"error": "no such song"}, status_code=404)
    return FileResponse(path, media_type="audio/mpeg")


@app.get("/api/state")
def state() -> dict:
    return st().snapshot()


@app.post("/api/reset")
def reset(r: ResetReq) -> dict:
    st().reset(r.level, r.material, r.seed, r.randomize)
    return {"ok": True}


@app.post("/api/run")
def run() -> dict:
    st().run()
    return {"ok": True}


@app.post("/api/step")
def step() -> dict:
    st().step_once()
    return {"ok": True}


@app.post("/api/pause")
def pause() -> dict:
    st().pause()
    return {"ok": True}


@app.post("/api/abort")
def abort() -> dict:
    st().abort()
    return {"ok": True}


@app.post("/api/speed")
def speed(r: SpeedReq) -> dict:
    st().set_speed(r.speed)
    return {"ok": True}


@app.post("/api/scrub")
def scrub(r: ScrubReq) -> dict:
    return {"ok": st().scrub(r.id)}


@app.post("/api/camera")
def camera(r: CamReq) -> dict:
    st().set_camera(**{k: v for k, v in r.model_dump().items() if v is not None})
    return {"ok": True}


@app.post("/api/pick")
def pick(r: PickReq) -> dict:
    return {"point": st().pick(r.x, r.y)}


@app.post("/api/montecarlo")
def montecarlo(r: MCReq) -> dict:
    st().start_montecarlo(r.levels, max(2, min(r.trials, 200)))
    return {"ok": True}


@app.post("/api/proof/{kind}")
def proof(kind: str) -> dict:
    if kind not in ("replay", "equivalence"):
        return {"ok": False}
    st().start_proof(kind)
    return {"ok": True}


@app.post("/api/dance/start")
def dance_start(r: DanceReq) -> dict:
    st().dance_start(r.song)
    return {"ok": True}


@app.post("/api/dance/clock")
def dance_clock(r: DanceReq) -> dict:
    st().dance_clock(r.song, r.t, r.playing, r.sync_ms)
    return {"ok": True}


@app.post("/api/dance/stop")
def dance_stop() -> dict:
    st().dance_stop()
    return {"ok": True}


@app.post("/api/dance/grid")
def dance_grid(r: GridReq) -> dict:
    st().dance_grid(r.song, r.bpm, r.downbeat_s)
    return {"ok": True}


@app.websocket("/ws")
async def ws(sock: WebSocket) -> None:
    await sock.accept()
    S = st()
    last_id = -1

    async def sender() -> None:
        nonlocal last_id
        n = 0
        while True:
            if S.frame is not None and S.frame_id != last_id:
                last_id = S.frame_id
                await sock.send_bytes(S.frame)
            n += 1
            if n % 8 == 0:
                snap = await asyncio.get_running_loop().run_in_executor(None, S.snapshot)
                await sock.send_text(json.dumps(snap, default=float))
            await asyncio.sleep(0.012)

    async def receiver() -> None:
        while True:
            msg = json.loads(await sock.receive_text())
            kind = msg.get("type")
            if kind == "view":
                S.set_view(int(msg["w"] * msg.get("dpr", 1.0)), int(msg["h"] * msg.get("dpr", 1.0)))
            elif kind == "cam":
                S.set_camera(**{k: msg[k] for k in ("az", "el", "dist", "lookat") if k in msg})
            elif kind == "dance":
                S.dance_clock(msg.get("song", ""), float(msg.get("t", 0.0)), bool(msg.get("playing", False)), float(msg.get("sync_ms", 0.0)))

    tasks = [asyncio.create_task(sender()), asyncio.create_task(receiver())]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
    except WebSocketDisconnect:
        pass
    finally:
        for t in tasks:
            t.cancel()


def main() -> None:
    port = int(os.environ.get("PORT", "8090"))
    st()
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
