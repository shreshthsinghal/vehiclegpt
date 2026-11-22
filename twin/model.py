"""Physics of the simulated Nexus EV, one sample per second.

Everything here is deliberately simple, textbook vehicle dynamics, so the numbers are explainable:
  wheel power = (mass * accel + rolling resistance + aerodynamic drag) * speed
  battery power = wheel power / drivetrain efficiency   (when driving)
                = wheel power * regen efficiency        (when braking, energy flows back)
                + auxiliary loads (AC compressor, 12 V systems)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

G = 9.81
RHO_AIR = 1.2


@dataclass
class EVParams:
    mass_kg: float = 1900.0
    cda_m2: float = 0.78            # drag coefficient x frontal area
    crr: float = 0.0095             # rolling resistance coefficient at nominal tyre pressure
    tyre_nominal_kpa: float = 240.0
    drive_eff: float = 0.90         # battery -> wheels
    regen_eff: float = 0.72         # wheels -> battery
    regen_max_kw: float = 60.0
    aux_base_kw: float = 0.22       # 12 V systems, infotainment, pumps
    usable_kwh: float = 60.0
    cells_series: int = 96
    wheel_radius_m: float = 0.34
    gear_ratio: float = 9.0


P = EVParams()


# ------------------------------------------------------------------ speed profiles
@dataclass
class TripSpec:
    kind: str                       # "city" or "highway"
    distance_km: float
    start_ts: float                 # epoch seconds
    traffic: float = 1.0            # stop-go intensity (1.0 normal, 1.6 festival traffic)
    rain: float = 0.0               # 0 dry .. 1 heavy rain
    seed: int = 0
    extra: dict = field(default_factory=dict)


def _ramp(v0, v1, a, rng):
    """Speeds (m/s) going from v0 to v1 with roughly constant acceleration a (m/s^2)."""
    n = max(1, int(abs(v1 - v0) / a))
    prof = np.linspace(v0, v1, n + 1)[1:]
    return prof + rng.normal(0, 0.05, prof.size)


def city_profile(dist_km: float, traffic: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Stop-and-go urban driving. Returns (speed m/s per second, road_type array: 0=city)."""
    v_out, target = [], dist_km * 1000
    travelled = 0.0
    while travelled < target:
        cruise = rng.uniform(28, 50) / 3.6 / (0.8 + 0.25 * traffic)     # heavier traffic -> slower
        seg = list(_ramp(0, cruise, rng.uniform(1.4, 2.2), rng))
        seg += list(cruise + rng.normal(0, 0.6, int(rng.uniform(15, 70) / traffic)))
        seg += list(_ramp(cruise, 0, rng.uniform(1.8, 2.8), rng))
        stop = int(rng.uniform(4, 30) * traffic ** 1.8)                # festival traffic: longer waits
        seg += [0.0] * stop
        seg = np.clip(np.array(seg), 0, None)
        v_out.append(seg)
        travelled += seg.sum()
    v = np.concatenate(v_out)
    # trim to the exact distance
    cut = np.searchsorted(np.cumsum(v), target)
    v = v[: cut + 1]
    return v, np.zeros(v.size, dtype=np.int8)


def highway_profile(dist_km: float, rain: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """City exit (4 km) -> highway cruise -> city entry (4 km). road_type: 0=city, 1=highway."""
    a, ra = city_profile(4, 1.0, rng)
    b_target = (dist_km - 8) * 1000
    v_set = rng.uniform(100, 115) / 3.6 * (1 - 0.15 * rain)
    seg = list(_ramp(a[-1] if a.size else 0, v_set, 1.2, rng))
    v = v_set
    travelled = sum(seg)
    while travelled < b_target:                                        # slowly wandering cruise speed
        v += 0.02 * (v_set - v) + rng.normal(0, 0.12)
        if rng.random() < 0.0015:                                      # occasional slowdown
            dip = list(_ramp(v, v * 0.6, 1.5, rng)) + list(_ramp(v * 0.6, v, 1.0, rng))
            seg += dip
            travelled += sum(dip)
        seg.append(v)
        travelled += v
    seg += list(_ramp(v, 0, 2.0, rng))
    b = np.clip(np.array(seg), 0, None)
    c, rc = city_profile(4, 1.0, rng)
    speed = np.concatenate([a, b, c])
    road = np.concatenate([ra, np.ones(b.size, dtype=np.int8), rc])
    return speed, road


# ------------------------------------------------------------------ energy
def hvac_power_kw(ambient_c: np.ndarray, hour: np.ndarray, t_into_trip_s: np.ndarray, ac_on: bool) -> np.ndarray:
    """AC compressor power. Hotter outside -> more power; extra in afternoon sun and while cooling down."""
    if not ac_on:
        return np.zeros_like(ambient_c)
    base = 0.25 + 0.11 * np.clip(ambient_c - 22.0, 0, None)
    sun = np.where((hour >= 12) & (hour <= 16), 0.25, 0.0)
    pulldown = 0.9 * np.exp(-t_into_trip_s / 240.0)                  # first minutes: hot cabin
    return np.clip(base + sun + pulldown, 0, 4.0)


def rolling_factor(tyre_kpa: np.ndarray) -> np.ndarray:
    """Rolling resistance multiplier from 4 tyre pressures (shape (n, 4)). Under-inflation costs energy."""
    per_wheel = (P.tyre_nominal_kpa / np.clip(tyre_kpa, 80, None)) ** 2.5
    return per_wheel.mean(axis=1)


def battery_power_kw(speed: np.ndarray, accel: np.ndarray, roll_mult: np.ndarray, aux_kw: np.ndarray,
                     power_limit_kw: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Battery power (kW, + = discharging) and its parts, one value per second."""
    f_roll = P.mass_kg * G * P.crr * roll_mult * (speed > 0.1)
    f_aero = 0.5 * RHO_AIR * P.cda_m2 * speed ** 2
    f_inert = P.mass_kg * accel
    wheel_kw = (f_roll + f_aero + f_inert) * speed / 1000.0
    if power_limit_kw is not None:                                     # derating caps traction power
        wheel_kw = np.minimum(wheel_kw, power_limit_kw * P.drive_eff)
    drive = np.where(wheel_kw > 0, wheel_kw / P.drive_eff, 0.0)
    regen = np.where(wheel_kw < 0, np.maximum(wheel_kw, -P.regen_max_kw) * P.regen_eff, 0.0)
    total = drive + regen + aux_kw
    return {"total": total, "drive": drive, "regen": -regen, "wheel": wheel_kw,
            "roll": f_roll * speed / 1000, "aero": f_aero * speed / 1000}


def pack_voltage(soc_pct: np.ndarray, current_a: np.ndarray) -> np.ndarray:
    """Open-circuit voltage curve of a 96s NMC pack, minus internal-resistance sag."""
    ocv_cell = 3.40 + 0.0085 * soc_pct - 0.00003 * (soc_pct - 50) ** 2
    return ocv_cell * P.cells_series - 0.08 * current_a
