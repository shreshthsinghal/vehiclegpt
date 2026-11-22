"""The simulated world: calendar, weather, traffic, trips, charging and the planted faults.

All times are India Standard Time (IST, UTC+05:30). Day 1 = Monday 5 Oct 2026, day 56 = Sunday 29 Nov 2026.
The fault timings here implement twin/fault_catalogue.yaml.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
START = dt.datetime(2026, 10, 5, tzinfo=IST)
DAYS = 56

HIGHWAY_SATURDAYS = {6, 13, 27, 34, 41, 48, 55}    # day 20 skipped: tyre problem week
RAIN_DAYS = {3: 0.5, 9: 0.4, 17: 0.5, 30: 0.4, 38: 0.5, 55: 0.0}   # day 55: rain only in the evening
DC_CHARGE_DAYS = {41, 48}
HEAT_FROM_DAY = 43              # S01: late-season heat
TRAFFIC_FROM_DAY = 43           # S01: wedding/festive season traffic on weekdays
WINDOW_A = range(29, 43)        # S01 comparison windows
WINDOW_B = range(43, 57)

# S02 slow puncture, rear-left tyre
PUNCTURE_START = (20, 10.0)     # (day, hour)
PUNCTURE_END = (26, 18.0)
PUNCTURE_FIXED = (27, 9.0)
PUNCTURE_MIN_KPA = 175.0
TPMS_WARN_KPA = 180.0

# S07 12 V drain nights; S08 logger gap
LV_DRAIN_DAYS = {33: 12.45, 34: 12.25, 35: 12.05, 36: 11.9}
GAP_DAY, GAP_MINUTES = 15, 20

# S05 overheat: warning time on day 48 (T0); charge from T-25 to T-5 min
OVERHEAT_DAY, OVERHEAT_T0 = 48, (16, 3)
# S03/S04 rain disengagements on day 55 return trip (local clock times) + camera fault
RAIN_DISENGAGE_TIMES = [(18, 4, 30), (18, 15, 10), (18, 26, 40), (18, 34, 5)]
CAMERA_FAULT_TIME = (18, 42, 0)


def day_start(day: int) -> dt.datetime:
    return START + dt.timedelta(days=day - 1)


def ts(day: int, h: int, m: int = 0, s: float = 0.0) -> float:
    return (day_start(day) + dt.timedelta(hours=h, minutes=m, seconds=s)).timestamp()


def day_of(t: float) -> int:
    return int((t - START.timestamp()) // 86400) + 1


def local_hour(t: np.ndarray | float) -> np.ndarray | float:
    return ((np.asarray(t) - START.timestamp()) % 86400) / 3600.0


def weekday(day: int) -> int:          # 0 = Monday
    return (day - 1) % 7


# ------------------------------------------------------------------ weather
def _daily_mean(day: int, heat: bool = True) -> float:
    base = 29.0 if day < 29 else 29.5
    if heat and day >= HEAT_FROM_DAY:
        base = 33.5
    return base


def ambient_c(t: np.ndarray, heat: bool = True, rng_seed: int = 7) -> np.ndarray:
    """Outside temperature: daily cycle (coolest 06:00, hottest 15:00) + day-to-day variation + rain cooling."""
    t = np.asarray(t, dtype=float)
    days = ((t - START.timestamp()) // 86400 + 1).astype(int)
    hour = local_hour(t)
    noise = np.random.default_rng(rng_seed).normal(0, 0.6, DAYS + 2)
    mean = np.array([_daily_mean(d, heat) for d in range(0, DAYS + 2)])[np.clip(days, 0, DAYS + 1)]
    temp = mean + noise[np.clip(days, 0, DAYS + 1)] + 3.5 * np.cos((hour - 15.0) / 24 * 2 * np.pi)
    return temp - 3.0 * rain_at(t)


def rain_at(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    days = ((t - START.timestamp()) // 86400 + 1).astype(int)
    hour = local_hour(t)
    out = np.zeros_like(t)
    for d, r in RAIN_DAYS.items():
        out = np.where(days == d, r, out)
    out = np.where((days == 55) & (hour >= 17.5), 0.85, out)       # heavy evening rain, day 55
    return out


# ------------------------------------------------------------------ tyres & 12 V
def tyre_pressures(t: np.ndarray, amb: np.ndarray, puncture: bool = True) -> np.ndarray:
    """(n, 4) pressures FL, FR, RL, RR in kPa. Warm tyres read a little higher."""
    t = np.asarray(t, dtype=float)
    base = 240.0 + (amb - 25.0) / 3.0
    p = np.stack([base + 1.0, base - 0.5, base, base + 0.5], axis=1)
    a, b, c = ts(*_dh(PUNCTURE_START)), ts(*_dh(PUNCTURE_END)), ts(*_dh(PUNCTURE_FIXED))
    frac = np.clip((t - a) / (b - a), 0, 1)
    rl = p[:, 2] - frac * (240.0 - PUNCTURE_MIN_KPA)
    if puncture:
        p[:, 2] = np.where((t >= a) & (t < c), rl, p[:, 2])
    return p


def _dh(x):
    day, hour = x
    return day, int(hour), int(round((hour % 1) * 60))


def lv_resting_voltage(day: int) -> float:
    return LV_DRAIN_DAYS.get(day, 12.62)


# ------------------------------------------------------------------ trips
@dataclass
class PlannedTrip:
    day: int
    start: float
    kind: str
    distance_km: float
    traffic: float
    purpose: str
    tags: list = field(default_factory=list)


def plan_trips(seed: int = 42) -> list[PlannedTrip]:
    rng = np.random.default_rng(seed)
    trips: list[PlannedTrip] = []
    for day in range(1, DAYS + 1):
        wd = weekday(day)
        traffic = 1.2 if (day >= TRAFFIC_FROM_DAY and wd < 5) else 1.0
        if wd < 5:                                         # weekday commutes
            trips.append(PlannedTrip(day, ts(day, 8, 30) + rng.uniform(0, 1200), "city",
                                     rng.uniform(13, 17), traffic, "commute_am"))
            if wd in (1, 3):                               # Tue/Thu afternoon errand (hot!)
                trips.append(PlannedTrip(day, ts(day, 14, 20) + rng.uniform(0, 900), "city",
                                         rng.uniform(5, 8), traffic, "errand"))
            trips.append(PlannedTrip(day, ts(day, 18, 0) + rng.uniform(0, 1200), "city",
                                     rng.uniform(13, 17), traffic, "commute_pm"))
            if day == 24:                                  # S02 distractor: weekday highway run
                trips.append(PlannedTrip(day, ts(day, 11, 0), "highway", 60, 1.0, "highway_out", ["S02_distractor"]))
                trips.append(PlannedTrip(day, ts(day, 15, 0), "highway", 60, 1.0, "highway_back", ["S02_distractor"]))
        elif day in HIGHWAY_SATURDAYS:
            back = ts(day, 15, 58) if day == OVERHEAT_DAY else ts(day, 16, 0)
            dist_back = 75.0
            if day == 55:
                back, dist_back = ts(day, 17, 45), 90.0    # evening return in heavy rain
            trips.append(PlannedTrip(day, ts(day, 10, 30), "highway", 75.0, 1.0, "highway_out"))
            trips.append(PlannedTrip(day, back, "highway", dist_back, 1.0, "highway_back",
                                     ["S05"] if day == OVERHEAT_DAY else (["S03", "S04"] if day == 55 else [])))
        else:                                              # weekend errand
            trips.append(PlannedTrip(day, ts(day, 11, 0) + rng.uniform(0, 1800), "city",
                                     rng.uniform(6, 12), 1.0, "weekend_errand"))
    return sorted(trips, key=lambda tr: tr.start)          # strict time order (bug found by SENSE, Phase 2)


def dc_charge_window(day: int) -> tuple[float, float] | None:
    """DC fast charge on the highway, just before the return trip."""
    if day == OVERHEAT_DAY:
        t0 = ts(day, *OVERHEAT_T0)
        return t0 - 25 * 60, t0 - 5 * 60
    if day in DC_CHARGE_DAYS:
        return ts(day, 15, 38), ts(day, 15, 58)
    return None
