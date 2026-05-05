import asyncio
import logging
from typing import Optional

import fastf1
import numpy as np
import pandas as pd

from state import StateManager
from models import ReplayConfig

log = logging.getLogger(__name__)


# 2026 grid - 11 teams. Strings here are normalized (lowercased, stripped).
# Actual team strings come from FastF1, which mirrors F1's API. Multiple
# variants included to handle title-sponsor prefixes / legacy names.
TEAM_COLORS = {
    "mclaren": "#FF8000",
    "mclaren mastercard": "#FF8000",
    "mclaren mastercard formula 1 team": "#FF8000",
    "mercedes": "#27F4D2",
    "mercedes-amg petronas f1 team": "#27F4D2",
    "red bull racing": "#3671C6",
    "oracle red bull racing": "#3671C6",
    "ferrari": "#E80020",
    "scuderia ferrari": "#E80020",
    "williams": "#64C4FF",
    "williams racing": "#64C4FF",
    "racing bulls": "#6692FF",
    "visa cash app racing bulls": "#6692FF",
    "rb": "#6692FF",
    "aston martin": "#229971",
    "aston martin aramco f1 team": "#229971",
    "haas": "#9C9FA2",
    "haas f1 team": "#9C9FA2",
    "tgr haas f1 team": "#9C9FA2",
    "audi": "#00404F",
    "audi f1 team": "#00404F",
    "alpine": "#0093CC",
    "bwt alpine f1 team": "#0093CC",
    "cadillac": "#C9B36C",
    "cadillac f1 team": "#C9B36C",
    # legacy (pre-2026), useful if testing past sessions
    "kick sauber": "#52E252",
    "stake f1 team kick sauber": "#52E252",
    "alfa romeo": "#900000",
    "alphatauri": "#2B4562",
    "scuderia alphatauri": "#2B4562",
}


def team_color(team_name: str | None) -> str:
    if not team_name:
        return "#888888"
    return TEAM_COLORS.get(team_name.strip().lower(), "#888888")


def fmt_time(td) -> Optional[str]:
    if td is None or pd.isna(td):
        return None
    total = td.total_seconds() if hasattr(td, "total_seconds") else float(td)
    if total >= 60:
        m = int(total // 60)
        s = total - 60 * m
        return f"{m}:{s:06.3f}"
    return f"{total:.3f}"


def fmt_gap(td) -> Optional[str]:
    if td is None or pd.isna(td):
        return None
    if isinstance(td, str):
        return td
    return f"+{td.total_seconds():.3f}"


def safe_int(x, default=0):
    try:
        if pd.isna(x):
            return default
        return int(x)
    except (TypeError, ValueError):
        return default


def safe_float(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def safe_str(x):
    if x is None or pd.isna(x):
        return None
    return str(x)


class FastF1Replayer:
    """Loads a past F1 session via FastF1 and replays it through StateManager
    on a virtual clock so it looks like a live stream."""

    TICK_INTERVAL = 0.25  # seconds of real time per tick

    def __init__(self, state: StateManager):
        self.state = state
        self.task: Optional[asyncio.Task] = None
        self.config: Optional[ReplayConfig] = None
        self.session = None
        self.running = False
        self.virtual_seconds = 0.0
        self.total_seconds = 0.0

        # caches
        self._laps: Optional[pd.DataFrame] = None
        self._laps_time_s: Optional[np.ndarray] = None  # Time col in seconds
        self._weather: Optional[pd.DataFrame] = None
        self._weather_time_s: Optional[np.ndarray] = None
        self._race_control: Optional[pd.DataFrame] = None
        self._rc_time_s: Optional[np.ndarray] = None
        self._rc_idx = 0
        self._car_data: dict[str, pd.DataFrame] = {}
        self._car_times: dict[str, np.ndarray] = {}
        self._pos_data: dict[str, pd.DataFrame] = {}
        self._pos_times: dict[str, np.ndarray] = {}

    async def start(self, config: ReplayConfig):
        if self.running:
            await self.stop()
        self.config = config
        await self.state.set_topic("session", {
            "status": "loading",
            "year": config.year,
            "name": config.race,
            "type": config.session,
            "speed": config.speed,
        })
        await self._load()
        self.virtual_seconds = config.start_at
        self.running = True
        self.task = asyncio.create_task(self._run())

    async def stop(self):
        self.running = False
        if self.task:
            try:
                await self.task
            except asyncio.CancelledError:
                pass
            self.task = None

    async def _load(self):
        loop = asyncio.get_running_loop()
        self.session = await loop.run_in_executor(None, self._load_sync)
        self._prepare_caches()
        await self._publish_drivers()
        
        # --- JUMP TO RACE START LOGIC ---
        if self.config.start_at == 0.0 and self._laps is not None and not self._laps.empty:
            lap_1 = self._laps[self._laps["LapNumber"] == 1]
            if not lap_1.empty:
                # Time (Lap 1 End) - LapTime (Lap 1 Duration) = Lights Out
                lights_out = (lap_1["Time"] - lap_1["LapTime"]).min()
                if pd.notna(lights_out):
                    # Add a 1.0 second offset to sync with grid launch
                    self.virtual_seconds = float(lights_out.total_seconds()) + 1.0

        await self.state.set_topic("session", {
            **(self.state.get("session") or {}),
            "status": "ready",
            "name": self._event_name(),
            "total_laps": self._total_laps(),
            "total_seconds": self.total_seconds,
        })

    def _load_sync(self):
        s = fastf1.get_session(self.config.year, self.config.race, self.config.session)
        s.load(telemetry=True, weather=True, messages=True)
        return s

    def _event_name(self) -> str:
        try:
            return str(self.session.event["EventName"])
        except Exception:
            return self.config.race

    def _total_laps(self) -> int:
        try:
            if self._laps is not None and not self._laps.empty:
                return int(self._laps["LapNumber"].max())
        except Exception:
            pass
        return 0

    def _prepare_caches(self):
        self._laps = self.session.laps.copy()
        if not self._laps.empty:
            self._laps_time_s = self._series_to_seconds(self._laps, "Time")
            self.total_seconds = float(np.nanmax(self._laps_time_s)) if len(self._laps_time_s) else 0.0
        else:
            self.total_seconds = 0.0

        if hasattr(self.session, "weather_data") and self.session.weather_data is not None:
            self._weather = self.session.weather_data.copy()
            if not self._weather.empty:
                self._weather_time_s = self._series_to_seconds(self._weather, "Time")

        if hasattr(self.session, "race_control_messages") and self.session.race_control_messages is not None:
            self._race_control = self.session.race_control_messages.copy()
            if not self._race_control.empty:
                self._rc_time_s = self._series_to_seconds(self._race_control, "Time")
        self._rc_idx = 0

        for drv in self.session.drivers:
            try:
                car = self.session.car_data[drv].copy()
                if not car.empty:
                    self._car_data[drv] = car
                    self._car_times[drv] = self._series_to_seconds(car, "Time")
            except Exception as e:
                log.warning(f"no car_data for {drv}: {e}")
            try:
                pos = self.session.pos_data[drv].copy()
                if not pos.empty:
                    self._pos_data[drv] = pos
                    self._pos_times[drv] = self._series_to_seconds(pos, "Time")
            except Exception as e:
                log.warning(f"no pos_data for {drv}: {e}")

    def _series_to_seconds(self, df: pd.DataFrame, time_col: str = "Time") -> np.ndarray:
        """Return time column as seconds-from-session-start regardless of dtype."""
        if time_col not in df.columns:
            # fall back to Date if Time is missing
            if "Date" in df.columns:
                s = df["Date"]
            else:
                return np.array([])
        else:
            s = df[time_col]

        if pd.api.types.is_timedelta64_dtype(s):
            return s.dt.total_seconds().to_numpy()
        if pd.api.types.is_datetime64_any_dtype(s):
            ref = self._reference_datetime()
            return (s - ref).dt.total_seconds().to_numpy()
        return np.array([])

    def _reference_datetime(self) -> pd.Timestamp:
        """Absolute datetime corresponding to virtual_seconds=0."""
        try:
            t0 = self.session.t0_date
            if t0 is not None:
                return pd.Timestamp(t0)
        except Exception:
            pass

        # fallbacks when telemetry didn't load (no t0_date)
        if self._race_control is not None and not self._race_control.empty and "Date" in self._race_control.columns:
            return pd.Timestamp(self._race_control["Date"].min())
        if self._laps is not None and not self._laps.empty and "Date" in self._laps.columns:
            return pd.Timestamp(self._laps["Date"].min())
        if self._weather is not None and not self._weather.empty and "Date" in self._weather.columns:
            return pd.Timestamp(self._weather["Date"].min())
        return pd.Timestamp(0)

    async def _publish_drivers(self):
        drivers = {}
        for drv in self.session.drivers:
            try:
                info = self.session.get_driver(drv)
            except Exception:
                continue
            num = int(drv)
            team_name = str(info.get("TeamName", ""))
            drivers[num] = {
                "number": num,
                "abbreviation": str(info.get("Abbreviation", "")),
                "full_name": str(info.get("FullName", "")),
                "team_name": team_name,
                "team_color": team_color(team_name),
            }
        await self.state.set_topic("drivers", drivers)

    async def _run(self):
        await self.state.set_topic("session", {
            **(self.state.get("session") or {}),
            "status": "running",
        })
        try:
            loop = asyncio.get_running_loop()
            while self.running and self.virtual_seconds < self.total_seconds:
                t0 = loop.time()
                
                # Prevent silent task death on NaN processing errors
                try:
                    await self._tick(self.virtual_seconds)
                except Exception as e:
                    log.error(f"Tick extraction failed at {self.virtual_seconds}: {e}")
                
                self.virtual_seconds += self.TICK_INTERVAL * self.config.speed
                elapsed = loop.time() - t0
                await asyncio.sleep(max(0.0, self.TICK_INTERVAL - elapsed))
        finally:
            self.running = False
            await self.state.set_topic("session", {
                **(self.state.get("session") or {}),
                "status": "finished",
            })

    async def _tick(self, vs: float):
        timing = self._compute_timing(vs)
        if timing:
            await self.state.set_topic("timing", timing)

        telemetry = self._compute_telemetry(vs)
        if telemetry:
            await self.state.set_topic("telemetry", telemetry)

        positions = self._compute_positions(vs)
        if positions:
            await self.state.set_topic("positions", positions)

        weather = self._compute_weather(vs)
        if weather:
            await self.state.set_topic("weather", weather)

        for msg in self._consume_race_control(vs):
            await self.state.append_race_control(msg)

        sess = dict(self.state.get("session") or {})
        sess["elapsed_seconds"] = vs
        sess["current_lap"] = self._current_lap(vs)
        await self.state.set_topic("session", sess)

    def _current_lap(self, vs: float) -> int:
        if self._laps is None or self._laps.empty:
            return 0
        mask = self._laps["LapStartTime"].dt.total_seconds() <= vs
        if not mask.any():
            return 0
        return int(self._laps.loc[mask, "LapNumber"].max())

    def _compute_timing(self, vs: float) -> dict:
        if self._laps is None or self._laps.empty:
            return {}
        td = pd.Timedelta(seconds=vs)
        completed = self._laps[self._laps["Time"] <= td]
        if completed.empty:
            return {}
        # latest lap per driver up to current time
        latest = (
            completed.sort_values("Time")
            .groupby("DriverNumber", as_index=False)
            .tail(1)
        )
        # rough race position: more laps = ahead, then earlier Time = ahead
        latest = latest.sort_values(by=["LapNumber", "Time"], ascending=[False, True]).reset_index(drop=True)

        result: dict[int, dict] = {}
        leader_time: Optional[pd.Timedelta] = None
        prev_time: Optional[pd.Timedelta] = None

        for i, row in latest.iterrows():
            num = safe_int(row["DriverNumber"])
            this_time = row["Time"]
            if leader_time is None:
                leader_time = this_time
                gap = "Leader"
                interval = "Leader"
            else:
                gap = fmt_gap(this_time - leader_time)
                interval = fmt_gap(this_time - prev_time)
            prev_time = this_time

            drv_laps = self._laps[self._laps["DriverNumber"] == row["DriverNumber"]]
            best_lap = drv_laps["LapTime"].min()
            pit_stops = int(drv_laps["PitInTime"].notna().sum())

            result[num] = {
                "driver_number": num,
                "position": i + 1,
                "lap_number": safe_int(row.get("LapNumber")),
                "gap_to_leader": gap,
                "interval": interval,
                "last_lap_time": fmt_time(row.get("LapTime")),
                "best_lap_time": fmt_time(best_lap),
                "sector_1": fmt_time(row.get("Sector1Time")),
                "sector_2": fmt_time(row.get("Sector2Time")),
                "sector_3": fmt_time(row.get("Sector3Time")),
                "compound": safe_str(row.get("Compound")),
                "tyre_age": safe_int(row.get("TyreLife")) if pd.notna(row.get("TyreLife")) else None,
                "pit_stops": pit_stops,
            }
        return result

    def _compute_telemetry(self, vs: float) -> dict:
        out = {}
        for drv, df in self._car_data.items():
            times = self._car_times[drv]
            idx = int(np.searchsorted(times, vs, side="right")) - 1
            if idx < 0:
                continue
            row = df.iloc[idx]
            num = int(drv)
            out[num] = {
                "driver_number": num,
                "speed": safe_int(row.get("Speed")),
                "rpm": safe_int(row.get("RPM")),
                "gear": safe_int(row.get("nGear")),
                "throttle": safe_int(row.get("Throttle")),
                "brake": 100 if bool(row.get("Brake", False)) else 0,
                "drs": safe_int(row.get("DRS")),
                "timestamp": float(times[idx]),
            }
        return out

    def _compute_positions(self, vs: float) -> dict:
        out = {}
        for drv, df in self._pos_data.items():
            times = self._pos_times[drv]
            idx = int(np.searchsorted(times, vs, side="right")) - 1
            if idx < 0:
                continue
            row = df.iloc[idx]
            num = int(drv)
            out[num] = {
                "driver_number": num,
                "x": safe_float(row.get("X")),
                "y": safe_float(row.get("Y")),
                "z": safe_float(row.get("Z")),
                "status": str(row.get("Status", "OnTrack")) if pd.notna(row.get("Status")) else "OnTrack",
            }
        return out

    def _compute_weather(self, vs: float) -> Optional[dict]:
        if self._weather is None or self._weather_time_s is None:
            return None
        idx = int(np.searchsorted(self._weather_time_s, vs, side="right")) - 1
        if idx < 0:
            return None
        row = self._weather.iloc[idx]
        return {
            "air_temp": safe_float(row.get("AirTemp")),
            "track_temp": safe_float(row.get("TrackTemp")),
            "humidity": safe_float(row.get("Humidity")),
            "pressure": safe_float(row.get("Pressure")),
            "wind_speed": safe_float(row.get("WindSpeed")),
            "wind_direction": safe_int(row.get("WindDirection")),
            "rainfall": bool(row.get("Rainfall", False)),
        }

    def _consume_race_control(self, vs: float) -> list[dict]:
        if self._race_control is None or self._rc_time_s is None:
            return []
        end_idx = int(np.searchsorted(self._rc_time_s, vs, side="right"))
        msgs = []
        for i in range(self._rc_idx, end_idx):
            row = self._race_control.iloc[i]
            msgs.append({
                "timestamp": float(self._rc_time_s[i]),
                "category": safe_str(row.get("Category")),
                "message": str(row.get("Message", "")),
                "flag": safe_str(row.get("Flag")),
                "scope": safe_str(row.get("Scope")),
                "driver_number": safe_int(row.get("RacingNumber")) if pd.notna(row.get("RacingNumber")) else None,
            })
        self._rc_idx = end_idx
        return msgs