# live_f1

Simulated F1 live timing dashboard and REST/WebSocket API. Replays historical race sessions via FastF1 to simulate a live data feed. Built as a drop-in for a real SignalR connection to F1's live timing stream when a live session is active.

---

## What it does

- Loads any past F1 session (Race, Qualifying, Sprint, Practice) via FastF1
- Replays it on a configurable speed multiplier (1x to 100x real time)
- Exposes live state via REST endpoints and a WebSocket stream
- Provides historical analytics: tyre degradation, strategy timelines, pace consistency, season standings, track category performance
- Pre-drawn circuit maps with sector overlays derived from fastest lap position data
- Connects to F1's live SignalR timing stream when a session is live (requires F1 TV subscription for telemetry and position data)

---

## Project structure

```
live_f1/
├── main.py               # FastAPI app entry point, CORS, cache config
├── state.py              # In-memory state store with pub/sub broadcast
├── replay.py             # FastF1 session loader and tick-based replayer
├── models.py             # Pydantic request models
├── routes.py             # Live state REST and WebSocket endpoints
├── routes_historic.py    # Historic data endpoints (FastF1 backed)
├── requirements.txt
├── F1_API.md             # Full API reference
└── fastf1_cache/         # Auto-created on first run (gitignore this)
```

---

## Requirements

- Python 3.11+
- Internet connection on first run per session (FastF1 downloads from F1's static archive)
- F1 TV subscription for live telemetry and GPS position data

---

## Setup

```bash
cd Formula1/live_f1
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Running

```bash
uvicorn main:app --reload --port 8000
```

Server starts at `http://localhost:8000`. Visit `/` for a full list of available endpoints.

The FastF1 cache is created automatically at `./fastf1_cache`. First load for any session downloads data from F1's archive (30–60 seconds). Every subsequent load for the same session is served from disk and is instant.

---

## Quick start

**Start a replay**

```bash
curl -X POST http://localhost:8000/api/replay/start \
  -H 'Content-Type: application/json' \
  -d '{"year": 2024, "race": "Bahrain", "session": "R", "speed": 20}'
```

**Poll live state**

```bash
curl http://localhost:8000/api/timing
curl http://localhost:8000/api/telemetry/1
curl http://localhost:8000/api/weather
curl http://localhost:8000/api/race-control?limit=10
```

**Connect to the WebSocket stream**

```python
import asyncio, json, websockets

async def stream():
    async with websockets.connect("ws://localhost:8000/api/stream") as ws:
        while True:
            msg = json.loads(await ws.recv())
            print(msg["topic"], str(msg["data"])[:120])

asyncio.run(stream())
```

On connect the server immediately sends a full `_snapshot` so late-joining clients are caught up. All subsequent messages are topic-level updates.

**Stop replay**

```bash
curl -X POST http://localhost:8000/api/replay/stop
```

---

## API overview

Full documentation is in [`F1_API.md`](./F1_API.md).

### Live endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/session` | Session status, lap count, elapsed time |
| GET | `/api/drivers` | Driver list with team colors |
| GET | `/api/timing` | Positions, gaps, sector times, tyre info, pit status |
| GET | `/api/telemetry` | Speed, RPM, gear, throttle, brake, DRS for all drivers |
| GET | `/api/telemetry/{driver_number}` | Telemetry for a single driver |
| GET | `/api/positions` | X/Y/Z track coordinates per driver |
| GET | `/api/weather` | Air temp, track temp, wind, rainfall |
| GET | `/api/race-control?limit=50` | Flag and penalty message log |
| GET | `/api/snapshot` | Full state snapshot in one response |
| WS | `/api/stream` | Live push stream for all state changes |

### Replay control

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/replay/start` | Load a session and begin replay |
| POST | `/api/replay/stop` | Stop active replay |
| GET | `/api/replay/status` | Check if replay is running and progress |

**Replay request body:**

```json
{
  "year": 2024,
  "race": "Bahrain",
  "session": "R",
  "speed": 10.0,
  "start_at": 0.0
}
```

`session` accepts: `R`, `Q`, `S`, `SS`, `SQ`, `FP1`, `FP2`, `FP3`

`speed` is the replay multiplier. `1.0` = real time, `10.0` = 10x, `50.0` = ~2 minute race replay.

`start_at` is an offset in seconds into the session. Use this to skip to a specific point, for example lap 30 of a race is roughly `5400.0`.

### Historic endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/historic/seasons/{year}/races` | Race calendar for a season |
| GET | `/api/historic/{year}/{race}/overview/{session_type}` | Full session analytics |
| GET | `/api/historic/{year}/{race}/driver/{driver}/laps` | Per-lap data for a driver |
| GET | `/api/historic/{year}/{race}/circuit` | Circuit outline and sector boundaries |
| GET | `/api/historic/{year}/{race}/tyre-degradation` | Per-stint degradation analysis |
| GET | `/api/historic/{year}/season-performance` | Season standings and track category breakdown |

---

## WebSocket message format

Every message has the same envelope:

```json
{
  "topic": "timing",
  "data": { ... },
  "ts": 1714905600.123
}
```

`topic` is one of: `_snapshot`, `session`, `drivers`, `timing`, `telemetry`, `positions`, `weather`, `track_status`, `race_control_msg`

`_snapshot` is only sent once on connect and contains the full state. `race_control_msg` carries a single new message. All other topics carry the full current state for that topic, replacing whatever the client had previously.

---

## How the replayer works

The replayer loads a session from FastF1 and pre-indexes all data into numpy arrays sorted by session time. On each tick (every 250ms real time), it advances a virtual clock by `tick_interval * speed_multiplier` and does a binary search (`numpy.searchsorted`) to find the current value for each topic. This means no iteration over DataFrames on the hot path — each tick is O(log n) per topic.

Topics updated each tick:

- **Timing** — latest completed lap per driver, position ordering, gaps, sector times, tyre info, pit status
- **Telemetry** — closest car data sample (speed, RPM, gear, throttle, brake, DRS)
- **Positions** — closest position sample (X/Y/Z coordinates)
- **Weather** — closest weather reading
- **Track status** — current flag state from `session.track_status`
- **Race control** — any new messages whose timestamp falls within the current tick window

Pit detection is done by pre-computing pit windows `(PitInTime, PitOutTime)` per driver during cache setup. The `in_pit` flag on timing is a simple range check.

---

## Tyre degradation model

The `/api/historic/{year}/{race}/tyre-degradation` endpoint fits a linear regression to each driver's cleaned stint data:

```
lap_time = base_pace + (deg_rate * tyre_age)
```

Laps are excluded from the regression if:
- `IsAccurate` is false (FastF1 quality flag)
- Track status was SC, VSC, or red flag during the lap
- The lap is an in lap or out lap
- Lap time is more than 2 standard deviations from the stint mean

`deg_rate` is the slope in seconds per lap. A value of `0.05` means each additional lap on the tyre costs 50ms. `r2` is the coefficient of determination — values above `0.7` indicate a clean, reliable fit. Low `r2` typically means the stint was disrupted by traffic, safety cars, or deliberate tyre management.

---

## Track categories

Used in the season performance track analysis view.

| Category | Circuits |
|----------|----------|
| Power | Monza, Spa-Francorchamps, Baku, Jeddah, Las Vegas, Silverstone, Spielberg |
| Technical | Monaco, Budapest, Zandvoort, Barcelona, Suzuka, Losail |
| Street | Marina Bay, Melbourne, Miami, Montreal |
| Mixed | Sakhir, Shanghai, Austin, Mexico City, São Paulo, Yas Marina, Imola, and others |

Circuits not in the mapping default to `Mixed`.

---

## Live F1 connection

The replayer is designed to be swapped out for a direct connection to F1's live SignalR timing stream. The `StateManager` and all downstream routes are unaware of where data comes from — they only consume calls to `state.set_topic()` and `state.append_race_control()`.

To connect live:

1. Negotiate with `https://livetiming.formula1.com/signalrcore/negotiate?negotiateVersion=1`
2. Open a WebSocket to `wss://livetiming.formula1.com/signalrcore`
3. Send the SignalR Core handshake: `{"protocol":"json","version":1}` followed by record separator `\x1e`
4. Subscribe to topics by sending an invocation message
5. Parse incoming frames, decompress `.z` topics (base64 + zlib deflate), and pipe decoded data into `state.set_topic()`

Pass the `login-session` cookie from formula1.com in all negotiate and WebSocket headers to unlock `CarData.z` (telemetry) and `Position.z` (GPS coordinates). Without auth these topics are silently dropped by the server.

---

## Known issues

- **Miami 2026 telemetry** — FastF1 3.6.1 fails to parse the car data file for this session with an `IndexError`. Telemetry and position data will be empty. Timing, weather, race control, and strategy data are unaffected. Update FastF1 when a fix is released.
- **`t0_date` unavailable** — when telemetry fails to load, `session.t0_date` raises `DataNotLoadedError`. The replayer falls back to the earliest timestamp in race control or lap data to establish a time reference.
- **Historic data latency** — FastF1 typically publishes session data 30–120 minutes after a session ends. Very recent sessions may not be available yet.
- **Season performance cache** — `_get_season_performance` is cached with `lru_cache`. To force a refresh after new races complete, restart the server.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` | Web framework |
| `uvicorn[standard]` | ASGI server with WebSocket support |
| `fastf1` | F1 historical session data |
| `pandas` | DataFrame operations on session data |
| `numpy` | Binary search on time arrays (hot path) |
| `pydantic` | Request validation |