import asyncio
import math
import datetime
import fastf1
import pandas as pd
import numpy as np
import threading
from fastapi import APIRouter, HTTPException
from functools import lru_cache
from typing import Optional

# primary characteristic per circuit (by FastF1 Location field)
TRACK_CATEGORIES = {
    # Power — engine/straight limited, low-downforce
    'Monza': 'Power', 'Baku': 'Power', 'Jeddah': 'Power',
    'Las Vegas': 'Power', 'Spielberg': 'Power', 'Silverstone': 'Power',
    'Spa-Francorchamps': 'Power',

    # Technical — mechanical grip, slow corners, high-downforce
    'Monaco': 'Technical', 'Budapest': 'Technical', 'Zandvoort': 'Technical',
    'Barcelona': 'Technical', 'Suzuka': 'Technical', 'Losail': 'Technical',

    # Street — temporary circuit characteristics, walls, bumps
    'Marina Bay': 'Street', 'Melbourne': 'Street',
    'Miami': 'Street', 'Montreal': 'Street',

    # Mixed — balanced characteristics
    'Sakhir': 'Mixed', 'Shanghai': 'Mixed', 'Austin': 'Mixed',
    'Mexico City': 'Mixed', 'São Paulo': 'Mixed', 'Yas Marina': 'Mixed',
    'Imola': 'Mixed', 'Portimão': 'Mixed', 'Mugello': 'Mixed',
    'Nürburgring': 'Mixed', 'Istanbul': 'Mixed', 'Bahrain': 'Mixed',
    'Lusail': 'Mixed', 'Kyalami': 'Mixed',
}

CATEGORIES = ['Power', 'Technical', 'Street', 'Mixed']


router = APIRouter(prefix="/api/historic")

# Global lock to prevent FastF1 cache corruption when React makes concurrent requests
_F1_LOCK = threading.Lock()


#helper functions 


def _compute_push_scores(session) -> dict[str, list[float]]:
    """
    Vectorised version — reads car_data once per driver, 
    then assigns throttle mean per lap using LapNumber index.
    Avoids ~1000 individual get_telemetry() calls.
    """
    scores = {}
    try:
        laps = session.laps
        for drv in session.drivers:
            try:
                # read full car data for driver in one shot
                car = session.car_data[drv].copy()
                if car.empty or 'Throttle' not in car.columns:
                    scores[drv] = []
                    continue

                # add LapNumber to car data by merging on SessionTime
                drv_laps = laps.pick_driver(drv)[['LapNumber', 'LapStartTime', 'Time']].copy()
                drv_laps = drv_laps.dropna(subset=['LapStartTime', 'Time'])

                # assign lap number to each telemetry sample via searchsorted
                lap_start_times = drv_laps['LapStartTime'].dt.total_seconds().to_numpy()
                car_times       = car['SessionTime'].dt.total_seconds().to_numpy()
                lap_indices     = np.searchsorted(lap_start_times, car_times, side='right') - 1

                car = car.copy()
                car['_lap_idx'] = lap_indices

                # per lap: mean throttle excluding brake-on samples
                lap_scores = []
                drv_lap_list = laps.pick_driver(drv)['LapNumber'].tolist()

                for i, _ in enumerate(drv_lap_list):
                    lap_tel = car[
                        (car['_lap_idx'] == i) &
                        (car['Brake'] == False)
                    ]
                    if len(lap_tel) < 10:
                        lap_scores.append(None)
                    else:
                        lap_scores.append(float(lap_tel['Throttle'].mean()))

                scores[drv] = lap_scores

            except Exception:
                scores[drv] = []
    except Exception:
        pass
    return scores


def _compute_dirty_air_laps(session) -> dict[str, set[int]]:
    """
    Returns {driver_number: {lap_numbers_in_dirty_air}}
    A lap is dirty air if the driver was within 1.2s of the car ahead
    for most of that lap. Uses lap time position ordering as proxy.
    """
    laps = session.laps
    dirty = {}
    try:
        results = session.results
        drv_pos = {str(r['DriverNumber']): int(r['Position']) for _, r in results.iterrows()
                   if pd.notna(r.get('Position'))}

        for lap_num in laps['LapNumber'].dropna().unique():
            lap_slice = laps[laps['LapNumber'] == lap_num].copy()
            if lap_slice.empty:
                continue
            # sort by elapsed time to find who was where on track
            lap_slice = lap_slice.sort_values('Time')
            drv_list = lap_slice['DriverNumber'].tolist()
            times = lap_slice['Time'].dt.total_seconds().tolist()

            for i, drv in enumerate(drv_list):
                if i == 0:
                    continue  # race leader, no car ahead
                car_ahead_idx = i - 1
                gap = times[i] - times[car_ahead_idx]
                if 0 <= gap <= 1.2:
                    dirty.setdefault(str(drv), set()).add(int(lap_num))
    except Exception:
        pass
    return dirty


REF_AIR_TEMP   = 40.0   # °C
REF_TRACK_TEMP = 55.0   # °C
# Pirelli estimate: roughly 0.003s per lap per °C above reference track temp
# for soft/medium. Hard is slightly less sensitive.
TEMP_SENSITIVITY = {"SOFT": 0.004, "MEDIUM": 0.003, "HARD": 0.002,
                    "INTERMEDIATE": 0.001, "WET": 0.001}

def _temp_correction(mean_track_temp: float, compound: str) -> float:
    """
    Returns seconds/lap to subtract from observed deg rate to normalise
    to reference track temperature. Positive = track was hotter than ref.
    """
    delta = mean_track_temp - REF_TRACK_TEMP
    sensitivity = TEMP_SENSITIVITY.get(compound.upper(), 0.003)
    return delta * sensitivity



def _compute_downforce_index(session) -> dict[str, float]:
    """
    Returns {driver_number: downforce_index}
    Index > 1.0 = running more downforce than field average (slower in straights)
    Index < 1.0 = running less downforce than field average (faster in straights)
    Uses SpeedST (speed trap) relative to field median.
    """
    laps = session.laps
    if 'SpeedST' not in laps.columns:
        return {}
    try:
        field_median = laps['SpeedST'].median()
        if pd.isna(field_median) or field_median == 0:
            return {}
        index = {}
        for drv in session.drivers:
            drv_laps = laps.pick_drivers(drv)
            drv_median = drv_laps['SpeedST'].median()
            if pd.notna(drv_median) and drv_median > 0:
                # lower speed = more downforce, so invert
                index[str(drv)] = float(field_median / drv_median)
        return index
    except Exception:
        return {}
    
    
def _compute_preservation_ranking(stints: list[dict]) -> list[dict]:
    """
    Ranks drivers by how well they preserved tyres relative to expectation.
    
    Method:
      1. For each compound, compute the field median normalised deg rate
      2. Each driver's preservation score = median - their deg rate
         (positive = better than field, negative = worse)
      3. Weight by r2 and stint length
      4. Average across all stints per driver
    
    Returns sorted list, best preserver first.
    """
    from collections import defaultdict

    # field median per compound (normalised, r2-weighted)
    compound_rates: dict[str, list] = defaultdict(list)
    for s in stints:
        if s['r2'] >= 0.5:   # only reliable stints in field reference
            compound_rates[s['compound']].append(
                (s['deg_rate_normalised'], s['r2'], s['n_laps'])
            )

    compound_median: dict[str, float] = {}
    for comp, vals in compound_rates.items():
        weights = [r2 * n for _, r2, n in vals]
        total_w = sum(weights) or 1
        compound_median[comp] = sum(r * w for (r, _, _), w in zip(vals, weights)) / total_w

    # per-driver weighted preservation score
    driver_scores: dict[str, dict] = {}
    for s in stints:
        if s['r2'] < 0.4:
            continue   # too noisy to count
        ref = compound_median.get(s['compound'])
        if ref is None:
            continue

        score    = ref - s['deg_rate_normalised']   # higher = better
        weight   = s['r2'] * s['n_laps']
        drv      = s['abbreviation']
        compound = s['compound']

        if drv not in driver_scores:
            driver_scores[drv] = {
                "abbreviation":   drv,
                "team":           s['team'],
                "weighted_sum":   0.0,
                "total_weight":   0.0,
                "by_compound":    {},
                "avg_push_score": [],
                "stints_counted": 0,
            }

        ds = driver_scores[drv]
        ds["weighted_sum"]  += score * weight
        ds["total_weight"]  += weight
        ds["stints_counted"] += 1
        if s['avg_push_score'] is not None:
            ds["avg_push_score"].append(s['avg_push_score'])

        if compound not in ds["by_compound"]:
            ds["by_compound"][compound] = {"weighted_sum": 0.0, "total_weight": 0.0}
        ds["by_compound"][compound]["weighted_sum"]  += score * weight
        ds["by_compound"][compound]["total_weight"]  += weight

    ranking = []
    for drv, ds in driver_scores.items():
        if ds["total_weight"] == 0:
            continue
        overall = ds["weighted_sum"] / ds["total_weight"]
        by_comp = {
            comp: round(v["weighted_sum"] / v["total_weight"], 4)
            for comp, v in ds["by_compound"].items()
            if v["total_weight"] > 0
        }
        push_scores_list = ds["avg_push_score"]
        ranking.append({
            "abbreviation":        drv,
            "team":                ds["team"],
            "preservation_score":  round(overall, 4),
            # positive = saved X seconds/lap vs field on same compound
            "avg_push_score":      round(float(np.mean(push_scores_list)), 1) if push_scores_list else None,
            "stints_counted":      ds["stints_counted"],
            "by_compound":         by_comp,
        })

    return sorted(ranking, key=lambda x: x["preservation_score"], reverse=True)
    

def to_seconds(val, t0=None):
    if val is None or pd.isna(val):
        return None
    try:
        if hasattr(val, 'total_seconds'):
            sec = float(val.total_seconds())
            return None if math.isnan(sec) else sec
        if hasattr(val, 'timestamp'):
            if t0 is not None and pd.notna(t0):
                sec = float((val - t0).total_seconds())
                return None if math.isnan(sec) else sec
            return float(val.hour * 3600 + val.minute * 60 + val.second + val.microsecond / 1e6)
        sec = float(val)
        return None if math.isnan(sec) else sec
    except (ValueError, TypeError):
        return None
def safe_float(val, default=None):
    if val is None or pd.isna(val):
        return default
    try:
        if math.isnan(float(val)):
            return default
        return float(val)
    except (TypeError, ValueError):
        return default
    

def _get_safe_session(year: int, race: str, session_type: str, telemetry: bool = False):
    """Thread-safe session loader that prevents cache corruption and catches future races."""
    with _F1_LOCK:
        session = fastf1.get_session(year, race, session_type)
        
        if session.date > pd.Timestamp.now():
            raise ValueError(f"The {year} {race} {session_type} session has not occurred yet.")

        # Pass the telemetry flag dynamically to save RAM/Time
        session.load(laps=True, telemetry=telemetry, weather=True, messages=True)
        
        if getattr(session, 'laps', None) is None or session.laps.empty:
            raise ValueError("FastF1 loaded the session but no lap data was found.")
            
        return session


#main functions

@lru_cache(maxsize=32)
def _get_race_results(year: int, race: str) -> list[dict]:
    session = fastf1.get_session(year, race, 'R')
    session.load(telemetry=False, weather=False, messages=False)
    res = session.results
    processed = []
    for _, row in res.iterrows():
        processed.append({
            "driver_number": str(row["DriverNumber"]),
            "position": row["Position"],
            "grid": row["GridPosition"],
            "points": row["Points"],
            "status": str(row["Status"]),
            "team": str(row["TeamName"]),
            "abbreviation": str(row.get("Abbreviation", ""))
        })
    return processed

@lru_cache(maxsize=64)
def _get_driver_laps(year: int, race: str, driver: str) -> list[dict]:
    try:
        session = _get_safe_session(year, race, 'R', telemetry=False)
    except Exception as e:
        return []
        
    laps = session.laps.pick_drivers(driver)
    processed = []
    
    for _, row in laps.iterrows():
        processed.append({
            "lap": safe_float(row.get("LapNumber")),
            "lap_time": to_seconds(row.get("LapTime")),
            "position": safe_float(row.get("Position")),           # NEW: Lap Position
            "pit_in": pd.notna(row.get("PitInTime")),              # NEW: Pit Stop Flag
            "sector_1": to_seconds(row.get("Sector1Time")),
            "sector_2": to_seconds(row.get("Sector2Time")),
            "sector_3": to_seconds(row.get("Sector3Time")),
            "compound": str(row.get("Compound", "UNKNOWN")),
            "tyre_life": safe_float(row.get("TyreLife"))
        })
    return processed





def _safely_load_session(session):
    """Attempts fast load. Falls back to deep load if FastF1 cache throws DataNotLoadedError."""
    try:
        session.load(telemetry=False, weather=False, messages=True)
        _ = session.laps
        _ = session.results
    except Exception:
        try:
            session.load() # Failsafe
        except Exception:
            pass
        
        
@lru_cache(maxsize=32)
def _get_qualifying_analytics(year: int, race: str, session_type: str) -> dict:
    try:
        session = _get_safe_session(year, race, session_type)
        laps = session.laps
        results = session.results
        
        # Explicitly catch the FastF1 DataNotLoadedError
        try:
            t0 = session.t0_date
        except Exception:
            t0 = None
            
    except Exception as e:
        return {"error": str(e)}

    def _sec(val):
        return to_seconds(val, t0)

    fastest = laps.pick_fastest() if not laps.empty else pd.Series(dtype=float)
    drv_val = fastest.get('Driver')
    overall_fastest = {
        "driver": str(drv_val) if pd.notna(drv_val) else "N/A", 
        "time": _sec(fastest.get('LapTime'))
    }

    sectors = {}
    for i in range(1, 4):
        col = f'Sector{i}Time'
        if not laps.empty and col in laps.columns:
            valid = laps.dropna(subset=[col])
            if not valid.empty:
                idx = valid[col].idxmin()
                sectors[f"s{i}"] = {
                    "driver": str(valid.loc[idx, 'Driver']),
                    "time": _sec(valid.loc[idx, col])
                }

    try:
        rc = session.race_control_messages
    except Exception:
        rc = None

    incidents = []
    if rc is not None and not rc.empty:
        for _, row in rc.iterrows():
            msg = str(row['Message']).lower()
            if any(k in msg for k in ['flag', 'incident', 'track limit', 'deleted']):
                incidents.append({
                    "time": _sec(row.get('Time')), 
                    "message": str(row['Message'])
                })

    driver_results = []
    driver_attempts = {}

    for _, row in results.iterrows():
        drv = str(row['DriverNumber'])
        abbrev = str(row.get('Abbreviation', drv))

        driver_results.append({
            "abbreviation": abbrev,
            "position": safe_float(row.get('Position')),
            "team": str(row.get('TeamName', '')),
            "q1": _sec(row.get('Q1')),
            "q2": _sec(row.get('Q2')),
            "q3": _sec(row.get('Q3')),
        })

        drv_laps = laps.pick_drivers(drv).pick_quicklaps() if not laps.empty else pd.DataFrame()
        attempts = []
        if not drv_laps.empty:
            for _, lap in drv_laps.iterrows():
                if pd.notna(lap.get('LapTime')):
                    attempts.append({
                        "lap": safe_float(lap.get('LapNumber')),
                        "time": _sec(lap.get('LapTime')),
                        "s1": _sec(lap.get('Sector1Time')),
                        "s2": _sec(lap.get('Sector2Time')),
                        "s3": _sec(lap.get('Sector3Time')),
                        "compound": str(lap.get('Compound', ''))
                    })
        driver_attempts[abbrev] = attempts

    return {
        "session_type": session_type,
        "overall_fastest": overall_fastest,
        "overall_sectors": sectors,
        "incidents": incidents,
        "results": sorted(driver_results, key=lambda x: x['position'] or 999),
        "attempts": driver_attempts
    }

@lru_cache(maxsize=32)
def _get_session_overview(year: int, race: str, session_type: str = 'R') -> dict:
    if session_type in ['Q', 'SQ']:
        return _get_qualifying_analytics(year, race, session_type)

    try:
        session = _get_safe_session(year, race, session_type)
        laps = session.laps
        results = session.results
        total_laps = int(laps['LapNumber'].max()) if not laps.empty and 'LapNumber' in laps.columns else 0
        try:
            t0 = session.t0_date
        except Exception:
            t0 = None
    except Exception as e:
        return {"error": str(e)}

    strategies = {}
    consistency = []
    positions_delta = []
    start_performance = []
    speeds = []
    dnfs = []

    fastest_overall = laps.pick_fastest() if not laps.empty else pd.Series(dtype=float)
    drv_val = fastest_overall.get('Driver')
    overall_fastest_lap = {
        "driver": str(drv_val) if pd.notna(drv_val) else "N/A",
        "time": to_seconds(fastest_overall.get('LapTime'), t0)
    }



    try:
        rc = session.race_control_messages
    except Exception:
        rc = None

    incidents = []
    if rc is not None and not rc.empty:
        for _, row in rc.iterrows():
            msg = str(row['Message']).lower()
            if any(k in msg for k in ['flag', 'incident', 'investigation', 'penalty', 'safety car']):
                incidents.append({
                    "time": to_seconds(row.get('Time'), t0), 
                    "message": str(row['Message'])
                })

    laps_1 = laps[laps['LapNumber'] == 1.0] if not laps.empty and 'LapNumber' in laps.columns else pd.DataFrame()

    for _, row in results.iterrows():
        drv = str(row['DriverNumber'])
        abbrev = str(row.get('Abbreviation', drv))
        team = str(row.get('TeamName', ''))
        grid = safe_float(row.get('GridPosition'))
        finish = safe_float(row.get('Position'))
        pts = safe_float(row.get('Points', 0.0))
        status = str(row.get('Status', ''))

        if not status.startswith("Finished") and not status.startswith("+"):
            dnfs.append({
                "driver": drv, "abbreviation": abbrev, "team": team,
                "reason": status, "laps_completed": int(row['Laps']) if pd.notna(row.get('Laps')) else 0
            })

        if grid and finish and grid > 0:
            positions_delta.append({
                "abbreviation": abbrev, "gained": int(grid - finish), "team": team, "finish": int(finish), "points": pts
            })

        if grid and grid > 0 and not laps_1.empty:
            l1 = laps_1[laps_1['DriverNumber'] == drv]
            if not l1.empty:
                pos_l1 = safe_float(l1.iloc[0]['Position'])
                if pos_l1:
                    start_performance.append({
                        "abbreviation": abbrev, "grid": int(grid), "lap1_pos": int(pos_l1), "delta": int(grid - pos_l1)
                    })

        drv_laps = laps.pick_drivers(drv)
        if drv_laps.empty:
            continue

        stints = []
        for stint_num, stint_laps in drv_laps.groupby("Stint"):
            stints.append({
                "stint": int(stint_num),
                "compound": str(stint_laps["Compound"].iloc[0]),
                "laps": len(stint_laps)
            })
        strategies[abbrev] = stints

        clean_laps = drv_laps.pick_quicklaps()
        if not clean_laps.empty and len(clean_laps) > 3:
            std_dev = clean_laps['LapTime'].dt.total_seconds().std()
            consistency.append({
                "driver": drv, "abbreviation": abbrev, "variance": float(std_dev) if pd.notna(std_dev) else 0
            })

        if 'SpeedST' in drv_laps.columns:
            max_speed = drv_laps['SpeedST'].max()
            if pd.notna(max_speed):
                speeds.append({
                    "driver": drv, "abbreviation": abbrev, "speed": float(max_speed)
                })

    return {
        "session_type": session_type,
        "total_laps": total_laps,
        "overall_fastest": overall_fastest_lap,
        "incidents": incidents,
        "results": sorted(positions_delta, key=lambda x: x['finish']), 
        "strategies": strategies,
        "dnfs": dnfs,
        "speeds": sorted(speeds, key=lambda x: x['speed'], reverse=True),
        "start_performance": sorted(start_performance, key=lambda x: x['delta'], reverse=True),
        "consistency": sorted([c for c in consistency if c['variance'] > 0], key=lambda x: x['variance'])
    }
    
def _stint_degradation(
    stint_laps: pd.DataFrame,
    push_scores: list[Optional[float]],   # per lap in this stint
    dirty_air_laps: set[int],             # lap numbers in dirty air
    mean_track_temp: float,
    compound: str,
) -> Optional[dict]:
    """
    Linear regression of lap time vs tyre age with:
    - push score weighting (de-weight managed laps)
    - dirty air lap exclusion
    - temperature normalisation
    - σ-clip outliers
    """
    clean = stint_laps[
        (stint_laps['IsAccurate'] == True) &
        (stint_laps['TrackStatus'].astype(str).isin(['1', '2'])) &
        stint_laps['LapTime'].notna() &
        stint_laps['TyreLife'].notna() &
        stint_laps['PitOutTime'].isna() &
        stint_laps['PitInTime'].isna()
    ].copy()

    # exclude dirty air laps — they inflate deg artificially
    dirty_mask = clean['LapNumber'].isin(dirty_air_laps)
    clean_no_dirty = clean[~dirty_mask]

    # if removing dirty air leaves too little, keep them but flag it
    dirty_air_excluded = int(dirty_mask.sum())
    use = clean_no_dirty if len(clean_no_dirty) >= 4 else clean

    if len(use) < 4:
        return None

    x   = use['TyreLife'].to_numpy(dtype=float)
    y   = use['LapTime'].dt.total_seconds().to_numpy()

    # build per-lap push weights
    # managed lap (push < 70) gets weight 0.4, full push (>90) gets 1.0
    weights = []
    lap_nums = use['LapNumber'].tolist()
    all_laps = stint_laps['LapNumber'].tolist()
    for ln in lap_nums:
        try:
            idx = all_laps.index(ln)
            ps  = push_scores[idx] if idx < len(push_scores) else None
        except (ValueError, IndexError):
            ps = None
        if ps is None:
            weights.append(0.7)          # unknown — moderate weight
        elif ps >= 90:
            weights.append(1.0)          # full push
        elif ps >= 75:
            weights.append(0.8)          # moderate push
        else:
            weights.append(0.4)          # managing / saving tyres
    w = np.array(weights)

    # σ-clip outliers on unweighted residuals first
    mask = np.abs(y - y.mean()) < 2 * y.std()
    x, y, w = x[mask], y[mask], w[mask]
    if len(x) < 4:
        return None

    # weighted least squares
    W      = np.diag(w)
    X_mat  = np.column_stack([np.ones_like(x), x])
    try:
        coeffs = np.linalg.lstsq(W @ X_mat, W @ y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None

    intercept, slope = float(coeffs[0]), float(coeffs[1])

    y_pred = intercept + slope * x
    ss_res = float(np.sum(w * (y - y_pred) ** 2))
    ss_tot = float(np.sum(w * (y - y.mean()) ** 2))
    r2     = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # temperature-normalised deg rate
    temp_adj    = _temp_correction(mean_track_temp, compound)
    norm_slope  = slope - temp_adj

    # avg push score for this stint (managed vs pushed flag)
    valid_ps = [p for p in push_scores if p is not None]
    avg_push = float(np.mean(valid_ps)) if valid_ps else None

    return {
        "deg_rate":          slope,        # raw observed
        "deg_rate_normalised": norm_slope, # temp-adjusted, comparable cross-race
        "base_pace":         intercept,
        "n_laps":            int(len(x)),
        "stint_length":      int(len(clean)),
        "dirty_air_excluded": dirty_air_excluded,
        "avg_push_score":    avg_push,
        "mean_track_temp":   mean_track_temp,
        "temp_correction":   temp_adj,
        "r2":                r2,
        "points": [
            {"tyre_life": float(xi), "lap_time": float(yi)}
            for xi, yi in zip(x, y)
        ],
    }

@lru_cache(maxsize=16)
def _get_tyre_degradation(year: int, race, telemetry: bool = True) -> dict:
    session = _get_safe_session(year, race, 'R', telemetry=telemetry)
    laps    = session.laps
    results = session.results

    drv_team   = {str(r['DriverNumber']): str(r['TeamName'])   for _, r in results.iterrows()}
    drv_abbrev = {str(r['DriverNumber']): str(r['Abbreviation']) for _, r in results.iterrows()}

    # compute session-level context once
    push_scores_all = _compute_push_scores(session) if telemetry else {}
    dirty_air_all   = _compute_dirty_air_laps(session)
    downforce_idx   = _compute_downforce_index(session) if telemetry else {}

    # mean track temp for the session
    mean_track_temp = REF_TRACK_TEMP
    try:
        wx = session.weather_data
        if wx is not None and not wx.empty and 'TrackTemp' in wx.columns:
            mean_track_temp = float(wx['TrackTemp'].mean())
    except Exception:
        pass

    stints = []

    for drv in session.drivers:
        drv_laps = laps.pick_drivers(drv)
        if drv_laps.empty:
            continue

        team   = drv_team.get(drv, "Unknown")
        abbrev = drv_abbrev.get(drv, drv)
        drv_push = push_scores_all.get(drv, [])
        drv_dirty = dirty_air_all.get(str(drv), set())
        df_idx = downforce_idx.get(str(drv), 1.0)

        for stint_num, sl in drv_laps.groupby("Stint"):
            compound = str(sl["Compound"].iloc[0])
            if compound in ("UNKNOWN", "nan", ""):
                continue

            # slice push scores to this stint's lap indices
            stint_lap_nums  = sl['LapNumber'].tolist()
            all_lap_nums    = drv_laps['LapNumber'].tolist()
            stint_push = []
            for ln in stint_lap_nums:
                try:
                    idx = all_lap_nums.index(ln)
                    stint_push.append(drv_push[idx] if idx < len(drv_push) else None)
                except (ValueError, IndexError):
                    stint_push.append(None)

            deg = _stint_degradation(
                sl, stint_push, drv_dirty,
                mean_track_temp, compound,
            )
            if deg is None:
                continue

            stints.append({
                "driver":          drv,
                "abbreviation":    abbrev,
                "team":            team,
                "stint":           int(stint_num),
                "compound":        compound,
                "downforce_index": round(df_idx, 3),
                **deg,
            })

    # per-race tyre preservation ranking
    preservation = _compute_preservation_ranking(stints)

    return {
        "year":         year,
        "race":         race,
        "track_temp":   mean_track_temp,
        "stints":       stints,
        "preservation": preservation,
    }


@lru_cache(maxsize=8)
def _get_season_performance(year: int) -> dict:
    schedule = fastf1.get_event_schedule(year)
    now = pd.Timestamp.now()
    completed = schedule[(schedule['EventDate'] < now) & (schedule['EventFormat'] != 'testing')]

    drivers = {}
    teams = {}
    races = []

    

    def calc_std_dev(values):
        valid = [v for v in values if v is not None]
        if not valid or len(valid) < 2: return 0.0
        mean = sum(valid) / len(valid)
        return round(math.sqrt(sum((x - mean) ** 2 for x in valid) / len(valid)), 2)

    for race_idx, event in enumerate(completed.iterrows()):
        _, event_data = event
        try:
            session = fastf1.get_session(year, event_data['RoundNumber'], 'R')
            session.load(telemetry=False, weather=False, messages=False, laps=False)
            
            race_name = str(event_data['EventName']).replace(" Grand Prix", "")
            loc = str(event_data.get('Location', ''))
            t_type = TRACK_CATEGORIES.get(loc, 'Mixed')
            races.append(race_name)
            
            for d in drivers.values():
                while len(d["grids"]) < len(races): d["grids"].append(None)
                while len(d["finishes"]) < len(races): d["finishes"].append(None)
                while len(d["points_timeline"]) < len(races): d["points_timeline"].append(d["points_timeline"][-1] if d["points_timeline"] else 0)
            
            for t in teams.values():
                while len(t["points_timeline"]) < len(races): t["points_timeline"].append(t["points_timeline"][-1] if t["points_timeline"] else 0)

            for _, row in session.results.iterrows():
                drv = str(row['Abbreviation'])
                team = str(row['TeamName'])
                grid = safe_float(row.get('GridPosition'), 0)
                finish = safe_float(row.get('Position'), 0)
                pts = safe_float(row.get('Points'), 0)
                status = str(row.get('Status', ''))

                if drv not in drivers:
                    drivers[drv] = {
                        "team": team, "points": 0, "wins": 0, "podiums": 0, "poles": 0,
                        "dnfs": 0, "grids": [None] * (len(races) - 1), "finishes": [None] * (len(races) - 1), 
                        "gained_total": 0, "points_timeline": [0] * (len(races) - 1), 
                        "cat_perf": {c: {"points": 0, "positions": [], "races": 0} for c in CATEGORIES}
                    }
                
                if team not in teams:
                    teams[team] = {
                        "points": 0, "wins": 0, "podiums": 0, "poles": 0, "dnfs": 0,
                        "grids": [], "finishes": [], "drivers": set(),
                        "points_timeline": [0] * (len(races) - 1), 
                        "cat_perf": {c: {"points": 0, "positions": [], "races": 0} for c in CATEGORIES}
                    }
                
                d = drivers[drv]
                t = teams[team]
                
                d["points"] += pts
                t["points"] += pts
                t["drivers"].add(drv)
                
                if grid > 0: 
                    d["grids"].append(grid)
                    t["grids"].append(grid)
                else: 
                    d["grids"].append(None)
                
                if finish > 0: 
                    d["finishes"].append(finish)
                    t["finishes"].append(finish)
                else: 
                    d["finishes"].append(None)
                
                if grid == 1.0: 
                    d["poles"] += 1
                    t["poles"] += 1
                if finish == 1.0: 
                    d["wins"] += 1
                    t["wins"] += 1
                if finish in [1.0, 2.0, 3.0]: 
                    d["podiums"] += 1
                    t["podiums"] += 1
                if not status.startswith("Finished") and not status.startswith("+"): 
                    d["dnfs"] += 1
                    t["dnfs"] += 1
                
                if finish and finish > 0:
                    d["cat_perf"][t_type]["positions"].append(finish)
                    d["cat_perf"][t_type]["races"] += 1
                    t["cat_perf"][t_type]["positions"].append(finish)
                    t["cat_perf"][t_type]["races"] += 1

                d["cat_perf"][t_type]["points"] += pts
                t["cat_perf"][t_type]["points"] += pts

                d["points_timeline"].append(d["points"])
                t["points_timeline"].append(t["points"])

        except Exception:
            continue

    driver_list = []
    for drv, data in drivers.items():
        valid_grids = [x for x in data["grids"] if x is not None]
        valid_finishes = [x for x in data["finishes"] if x is not None]
        
        avg_grid = sum(valid_grids) / len(valid_grids) if valid_grids else 0
        avg_finish = sum(valid_finishes) / len(valid_finishes) if valid_finishes else 0
        
        driver_list.append({
            "abbreviation": drv, "team": data["team"], "points": data["points"],
            "poles": data["poles"], "wins": data["wins"], "podiums": data["podiums"], "dnfs": data["dnfs"],
            "avg_grid": round(avg_grid, 2), "avg_finish": round(avg_finish, 2), "net_gained": int(data["gained_total"]),
            "quali_consistency": calc_std_dev(data["grids"]),
            "race_consistency": calc_std_dev(data["finishes"]),
            "grid_history": data["grids"],
            "finish_history": data["finishes"],
            "points_timeline": data["points_timeline"],
           "cat_perf": {
                c: {
                    "avg_points": round(data["cat_perf"][c]["points"] / data["cat_perf"][c]["races"], 2)
                                if data["cat_perf"][c]["races"] > 0 else 0,
                    "avg_finish": round(sum(data["cat_perf"][c]["positions"]) / len(data["cat_perf"][c]["positions"]), 1)
                                if data["cat_perf"][c]["positions"] else None,
                    "races": data["cat_perf"][c]["races"],
                    "total_points": data["cat_perf"][c]["points"],
                }
                for c in CATEGORIES
            },
        })

    team_list = []
    for tm, data in teams.items():
        avg_grid = sum(data["grids"]) / len(data["grids"]) if data["grids"] else 0
        avg_finish = sum(data["finishes"]) / len(data["finishes"]) if data["finishes"] else 0

        team_list.append({
            "team": tm, "points": data["points"], "points_timeline": data["points_timeline"],
            "wins": data["wins"], "podiums": data["podiums"], "poles": data["poles"], "dnfs": data["dnfs"],
            "avg_grid": round(avg_grid, 2), "avg_finish": round(avg_finish, 2),
            "drivers": list(data["drivers"]),
            # "track_perf_pts": {
            #     "High-Speed": sum(data["track_perf"]["High-Speed"]),
            #     "Technical": sum(data["track_perf"]["Technical"]),
            #     "Mixed": sum(data["track_perf"]["Mixed"])
            # },
            "cat_perf": {
                c: {
                    "avg_points": round(data["cat_perf"][c]["points"] / data["cat_perf"][c]["races"], 2)
                                if data["cat_perf"][c]["races"] > 0 else 0,
                    "avg_finish": round(sum(data["cat_perf"][c]["positions"]) / len(data["cat_perf"][c]["positions"]), 1)
                                if data["cat_perf"][c]["positions"] else None,
                    "races": data["cat_perf"][c]["races"],
                    "total_points": data["cat_perf"][c]["points"],
                }
                for c in CATEGORIES
            },
        })

    return {
        "races": races,
        "drivers": sorted(driver_list, key=lambda x: x['points'], reverse=True),
        "categories": CATEGORIES,
        "year": year,
        "teams": sorted(team_list, key=lambda x: x['points'], reverse=True)
    }


@lru_cache(maxsize=64)
def _get_circuit_data(year: int, race: str) -> dict:
    session = fastf1.get_session(year, race, 'R')
    session.load(laps=True, telemetry=True, weather=False, messages=False)

    fastest = session.laps.pick_fastest()
    pos = fastest.get_pos_data()
    if pos.empty:
        raise ValueError("no position data for fastest lap")

    pos_t = pos['SessionTime'].dt.total_seconds().to_numpy()
    xs = pos['X'].to_numpy()
    ys = pos['Y'].to_numpy()

    def time_to_idx(t):
        if pd.isna(t):
            return None
        sec = t.total_seconds() if hasattr(t, 'total_seconds') else float(t)
        idx = int(np.searchsorted(pos_t, sec))
        return max(0, min(idx, len(pos_t) - 1))

    s1_idx = time_to_idx(fastest['Sector1SessionTime']) or len(pos_t) // 3
    s2_idx = time_to_idx(fastest['Sector2SessionTime']) or 2 * len(pos_t) // 3

    corners = []
    rotation = 0.0
    try:
        ci = session.get_circuit_info()
        if ci is not None:
            rotation = float(getattr(ci, 'rotation', 0) or 0)
            for _, c in ci.corners.iterrows():
                corners.append({
                    "number": int(c['Number']),
                    "letter": str(c.get('Letter', '') or ''),
                    "x": float(c['X']),
                    "y": float(c['Y']),
                    "angle": float(c['Angle']) if pd.notna(c.get('Angle')) else 0.0,
                })
    except Exception:
        pass

    points = [{"x": float(x), "y": float(y)} for x, y in zip(xs, ys)]
    return {
        "points": points,
        "sector_breaks": [s1_idx, s2_idx],
        "corners": corners,
        "rotation": rotation,
        "bounds": {
            "min_x": float(xs.min()), "max_x": float(xs.max()),
            "min_y": float(ys.min()), "max_y": float(ys.max()),
        },
    }


@lru_cache(maxsize=16)
def _get_season_races(year: int) -> list[dict]:
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    out = []
    for _, ev in schedule.iterrows():
        out.append({
            "round": int(ev["RoundNumber"]),
            "name": str(ev["EventName"]).replace(" Grand Prix", "").strip(),
            "full_name": str(ev["EventName"]),
            "country": str(ev.get("Country", "") or ""),
            "location": str(ev.get("Location", "") or ""),
            "date": str(ev.get("EventDate", ""))[:10],
            "format": str(ev.get("EventFormat", "conventional")),
        })
    return out

@lru_cache(maxsize=4)
def _get_season_tyre_analysis(year: int) -> dict:
    """
    Aggregates per-race tyre preservation scores across the season.
    Per team: overall preservation, per compound, per track category.
    """
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    now      = pd.Timestamp.now()
    done     = schedule[schedule['EventDate'] < now]

    # {team: {compound: {category: [scores]}}}
    team_data: dict = {}

    races_processed = []

    for _, ev in done.iterrows():
        race_name = str(ev['EventName']).replace(" Grand Prix", "")
        loc       = str(ev.get('Location', ''))
        category  = TRACK_CATEGORIES.get(loc, 'Mixed')

        try:
            data = _get_tyre_degradation(year, int(ev['RoundNumber']), telemetry=False)
        except Exception:
            continue

        races_processed.append(race_name)

        for entry in data.get("preservation", []):
            team     = entry["team"]
            score    = entry["preservation_score"]
            push     = entry["avg_push_score"]
            by_comp  = entry["by_compound"]

            if team not in team_data:
                team_data[team] = {
                    "overall":    [],
                    "by_compound": {},
                    "by_category": {c: [] for c in CATEGORIES},
                    "by_cat_compound": {},
                }

            td = team_data[team]
            td["overall"].append(score)
            td["by_category"][category].append(score)

            for comp, comp_score in by_comp.items():
                td["by_compound"].setdefault(comp, []).append(comp_score)

                cat_key = f"{category}_{comp}"
                td["by_cat_compound"].setdefault(cat_key, []).append(comp_score)

    # summarise
    def _summarise(scores: list[float]) -> Optional[dict]:
        if not scores:
            return None
        arr = np.array(scores)
        return {
            "mean":   round(float(arr.mean()), 4),
            "std":    round(float(arr.std()), 4),
            "n":      len(scores),
            "best":   round(float(arr.max()), 4),
            "worst":  round(float(arr.min()), 4),
        }

    summary = []
    for team, td in team_data.items():
        by_comp_summary = {
            comp: _summarise(scores)
            for comp, scores in td["by_compound"].items()
        }
        by_cat_summary = {
            cat: _summarise(scores)
            for cat, scores in td["by_category"].items()
            if scores
        }
        by_cat_comp_summary = {
            key: _summarise(scores)
            for key, scores in td["by_cat_compound"].items()
            if scores
        }
        overall = _summarise(td["overall"])
        if not overall:
            continue

        summary.append({
            "team":             team,
            "overall":          overall,
            "by_compound":      by_comp_summary,
            "by_category":      by_cat_summary,
            "by_cat_compound":  by_cat_comp_summary,
        })

    # global rankings
    ranked_overall = sorted(
        [s for s in summary if s["overall"]],
        key=lambda x: x["overall"]["mean"],
        reverse=True,
    )

    # best compound per team — which compound do they manage best
    best_compound_per_team = {}
    for s in summary:
        if not s["by_compound"]:
            continue
        best = max(s["by_compound"].items(),
                   key=lambda x: x[1]["mean"] if x[1] else -999)
        best_compound_per_team[s["team"]] = {
            "compound": best[0],
            "score":    best[1]["mean"] if best[1] else None,
        }

    # best category per team — which track type suits their tyre management
    best_category_per_team = {}
    for s in summary:
        if not s["by_category"]:
            continue
        valid = {k: v for k, v in s["by_category"].items() if v and v["n"] >= 2}
        if not valid:
            continue
        best = max(valid.items(), key=lambda x: x[1]["mean"])
        best_category_per_team[s["team"]] = {
            "category": best[0],
            "score":    best[1]["mean"],
        }

    return {
        "year":                   year,
        "races":                  races_processed,
        "team_summary":           ranked_overall,
        "best_compound_per_team": best_compound_per_team,
        "best_category_per_team": best_category_per_team,
    }


# --- Routes ---

@router.get("/{year}/{race}/overview/{session_type}")
async def historic_session_overview(year: int, race: str, session_type: str):
    loop = asyncio.get_running_loop()
    return {"data": await loop.run_in_executor(None, _get_session_overview, year, race, session_type)}

@router.get("/{year}/season-performance")
async def historic_season_performance(year: int):
    loop = asyncio.get_running_loop()
    return {"data": await loop.run_in_executor(None, _get_season_performance, year)}


@router.get("/{year}/{race}/driver/{driver}/laps")
async def historic_driver_laps(year: int, race: str, driver: str):
    loop = asyncio.get_running_loop()
    return {"data": await loop.run_in_executor(None, _get_driver_laps, year, race, driver)}


@router.get("/{year}/{race}/circuit")
async def get_circuit(year: int, race: str):
    loop = asyncio.get_running_loop()
    try:
        data = await loop.run_in_executor(None, _get_circuit_data, year, race)
    except Exception as e:
        raise HTTPException(500, f"circuit load failed: {e}")
    return {"data": data}


@router.get("/seasons/{year}/races")
async def get_season_races(year: int):
    loop = asyncio.get_running_loop()
    try:
        data = await loop.run_in_executor(None, _get_season_races, year)
    except Exception as e:
        raise HTTPException(500, f"schedule load failed: {e}")
    return {"data": data}   

# @router.get("/{year}/{race}/tyre-degradation")
# async def tyre_degradation(year: int, race: str):
#     loop = asyncio.get_running_loop()
#     try:
#         return {"data": await loop.run_in_executor(None, _get_tyre_degradation, year, race)}
#     except Exception as e:
#         raise HTTPException(500, f"deg calc failed: {e}")
    
@router.get("/{year}/{race}/tyre-degradation")
async def tyre_degradation(year: int, race: str):
    loop = asyncio.get_running_loop()
    try:
        # telemetry=True for single race — full analysis
        return {"data": await loop.run_in_executor(None, _get_tyre_degradation, year, race, True)}
    except Exception as e:
        raise HTTPException(500, f"tyre deg failed: {e}")


@router.get("/{year}/season-tyre-analysis")
async def season_tyre_analysis(year: int):
    loop = asyncio.get_running_loop()
    try:
        return {"data": await loop.run_in_executor(None, _get_season_tyre_analysis, year)}
    except Exception as e:
        raise HTTPException(500, f"season tyre analysis failed: {e}")