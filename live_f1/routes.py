import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException

from state import StateManager
from replay import FastF1Replayer
from models import ReplayConfig

log = logging.getLogger(__name__)


def make_router(state: StateManager, replayer: FastF1Replayer) -> APIRouter:
    r = APIRouter(prefix="/api")

    # ---- snapshot reads ----

    @r.get("/session")
    def get_session():
        return state.get("session")

    @r.get("/drivers")
    def get_drivers():
        return state.get("drivers")

    @r.get("/timing")
    def get_timing():
        return state.get("timing")

    @r.get("/telemetry")
    def get_all_telemetry():
        return state.get("telemetry")

    @r.get("/telemetry/{driver_number}")
    def get_telemetry(driver_number: int):
        t = state.get("telemetry") or {}
        if driver_number not in t:
            raise HTTPException(404, f"no telemetry for driver {driver_number}")
        return t[driver_number]

    @r.get("/positions")
    def get_positions():
        return state.get("positions")

    @r.get("/weather")
    def get_weather():
        return state.get("weather")

    @r.get("/race-control")
    def get_race_control(limit: int = 50):
        msgs = state.get("race_control") or []
        return msgs[-limit:]

    @r.get("/snapshot")
    def get_snapshot():
        return state.snapshot()

    # ---- replay control ----

    @r.post("/replay/start")
    async def replay_start(config: ReplayConfig):
        try:
            await replayer.start(config)
        except Exception as e:
            log.exception("replay start failed")
            raise HTTPException(500, f"replay start failed: {e}")
        return {"status": "started", "config": config.model_dump()}

    @r.post("/replay/stop")
    async def replay_stop():
        await replayer.stop()
        return {"status": "stopped"}

    @r.get("/replay/status")
    def replay_status():
        return {
            "is_running": replayer.running,
            "config": replayer.config.model_dump() if replayer.config else None,
            "elapsed_seconds": replayer.virtual_seconds,
            "total_seconds": replayer.total_seconds,
        }

    # ---- WebSocket stream ----

    @r.websocket("/stream")
    async def stream(ws: WebSocket):
        await ws.accept()
        q = state.subscribe()
        try:
            # send full state on connect so late joiners are caught up
            await ws.send_json({"topic": "_snapshot", "data": state.snapshot()})
            while True:
                msg = await q.get()
                await ws.send_json(msg)
        except WebSocketDisconnect:
            pass
        except Exception as e:
            log.warning(f"ws stream closed: {e}")
        finally:
            state.unsubscribe(q)

    return r