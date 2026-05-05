import logging
import os
from pathlib import Path
import fastf1
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes_historic import router as historic_router
from state import StateManager
from replay import FastF1Replayer
from routes import make_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# enable FastF1 cache so repeated loads of the same session are instant
CACHE_DIR = Path(os.getenv("FASTF1_CACHE", "./fastf1_cache")).resolve()
CACHE_DIR.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

app = FastAPI(title="F1 Live Timing API (Simulated)")

# CORS - permissive while building, lock down later
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

state = StateManager()
replayer = FastF1Replayer(state)
app.include_router(make_router(state, replayer))
app.include_router(historic_router)

@app.get("/")
def root():
    return {
        "name": "F1 Live Timing API (Simulated)",
        "endpoints": [
            "GET  /api/session",
            "GET  /api/drivers",
            "GET  /api/timing",
            "GET  /api/telemetry",
            "GET  /api/telemetry/{driver_number}",
            "GET  /api/positions",
            "GET  /api/weather",
            "GET  /api/race-control?limit=50",
            "GET  /api/snapshot",
            "POST /api/replay/start  body: {year, race, session, speed, start_at}",
            "POST /api/replay/stop",
            "GET  /api/replay/status",
            "WS   /api/stream",
        ],
    }