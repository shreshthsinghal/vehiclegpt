"""Simulate one session (a trip or a charging stop), second by second, and return every signal.

A session result is a dict of numpy arrays (one value per second) plus a list of discrete events
(warnings, DTCs, ADAS disengagements, log lines). generate.py turns these into wire-format recordings.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import world as W
from .model import P, battery_power_kw, city_profile, highway_profile, hvac_power_kw, pack_voltage, rolling_factor


@dataclass
class CarState:
    soc: float = 85.0
    odometer_km: float = 12480.0
    batt_temp: float = 28.0
    motor_temp: float = 30.0
    last_t: float = W.START.timestamp()
    cons_ema: float = 15.5            # kWh/100 km used by the range estimate
    soh: float = 99.0
    tpms_dtc_logged: bool = False
    lv_dtc_logged: bool = False


@dataclass
class Session:
    kind: str                          # "trip" | "dc_charge" | "ac_charge"
    t: np.ndarray
    sig: dict
    events: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)


def _relax(state: CarState, t0: float, amb: float):
    """While parked, battery and motor drift back toward the outside temperature."""
    gap = max(0.0, t0 - state.last_t)
    k = np.exp(-gap / (3 * 3600))
    state.batt_temp = amb + (state.batt_temp - amb) * k
    state.motor_temp = amb + (state.motor_temp - amb) * np.exp(-gap / 1800)


def _thermal(t, p_batt_kw, amb, state: CarState, pump_frac=None):
    """Battery temperature (deg C) with I^2R heating and liquid cooling."""
    n = t.size
    temp = np.empty(n)
    pump = np.empty(n)
    T = state.batt_temp
    for i in range(n):
        chiller = T > 30.0
        pf = pump_frac[i] if pump_frac is not None else (1.0 if chiller else 0.6)
        target = 22.0 if chiller else amb[i]
        current = p_batt_kw[i] * 1000 / 360.0
        losses_kw = 0.08 * current ** 2 / 1000
        T += (losses_kw - 0.5 * pf * (T - target)) / 400.0
        temp[i], pump[i] = T, pf
    state.batt_temp = T
    return temp, pump


def simulate_trip(trip: W.PlannedTrip, state: CarState, seed: int, heat: bool = True,
                  traffic: float | None = None, counterfactual: bool = False, puncture: bool = True) -> Session:
    rng = np.random.default_rng(seed)
    tr = trip.traffic if traffic is None else traffic
    rain_now = float(W.rain_at(np.array([trip.start]))[0])
    if trip.kind == "city":
        speed, road = city_profile(trip.distance_km, tr, rng)
    else:
        speed, road = highway_profile(trip.distance_km, rain_now, rng)
    n = speed.size
    t = trip.start + np.arange(n, dtype=float)
    amb = W.ambient_c(t, heat=heat)
    rain = W.rain_at(t)
    hour = W.local_hour(t)
    accel = np.gradient(speed)
    tyres = W.tyre_pressures(t, amb, puncture)
    ac = hvac_power_kw(amb, hour, np.arange(n), ac_on=True)
    aux = P.aux_base_kw + ac
    limit = np.full(n, 150.0)
    pw = battery_power_kw(speed, accel, rolling_factor(tyres), aux, limit)
    energy_kwh = pw["total"].sum() / 3600.0
    dist_km = speed.sum() / 1000.0
    if counterfactual:                                   # truth computation only needs energy & distance
        return Session("trip", t, {"energy_kwh": energy_kwh, "dist_km": dist_km,
                                   "road": road, "p": pw["total"], "speed": speed})

    _relax(state, trip.start, float(amb[0]))
    day = trip.day
    events: list[dict] = []

    # ---- battery thermal (+ scripted S05 coolant-pump failure)
    pump_frac = None
    if "S05" in trip.tags:
        pump_frac = np.full(n, 0.28)
    temp_avg, pump = _thermal(t, pw["total"], amb, state, pump_frac)
    temp_max = temp_avg + 1.5 + 0.02 * np.abs(pw["total"])
    if "S05" in trip.tags:
        t0 = W.ts(day, *W.OVERHEAT_T0)
        k = np.clip((t - (t0 - 300)) / 180.0, 0, 1)      # continues the rise started while charging
        temp_max = np.maximum(temp_max, 46.0 + 6.5 * k - 2.0 * np.clip((t - t0 - 120) / 600, 0, 1))
        temp_avg = np.maximum(temp_avg, temp_max - 4.0)
        limit = np.where(temp_max >= 50.0, 60.0, 150.0)
        pw = battery_power_kw(speed, accel, rolling_factor(tyres), aux, limit)
        first_derate = t[np.argmax(temp_max >= 50.0)]
        events += [
            {"t": first_derate, "type": "dtc", "code": "P1A21", "ecu": "BMS", "status": 0x2F},
            {"t": t[-1], "type": "dtc", "code": "P1A21", "ecu": "BMS", "status": 0x28},
            {"t": first_derate, "type": "warning", "code": "REDUCED_POWER", "duration": 600},
            {"t": t0, "type": "warning", "code": "BATTERY_TEMP_HIGH", "duration": 600},
            {"t": first_derate, "type": "log", "proc": "bms_mon", "sev": 3,
             "msg": f"pack Tmax {temp_max[np.argmax(temp_max >= 50.0)]:.1f}C >= 50C, discharge limit 150->60 kW"},
            {"t": t0, "type": "log", "proc": "ipc_hmi", "sev": 4, "msg": "telltale BATTERY_TEMP_HIGH on"},
        ]
    temp_min = temp_avg - 1.0

    # ---- SoC, voltage, current, range
    soc = state.soc - np.cumsum(pw["total"]) / 3600.0 / P.usable_kwh * 100.0
    current = pw["total"] * 1000.0 / 360.0
    volt = pack_voltage(soc, current)
    current = pw["total"] * 1000.0 / volt
    odo = state.odometer_km + np.cumsum(speed) / 1000.0
    range_km = soc / 100.0 * P.usable_kwh * state.soh / 100.0 / (state.cons_ema / 100.0)

    # ---- motor
    rpm = speed / P.wheel_radius_m * P.gear_ratio * 60 / (2 * np.pi)
    torque = pw["wheel"] * 1000 / np.maximum(speed, 0.5) * P.wheel_radius_m / P.gear_ratio
    mt = np.empty(n)
    T = state.motor_temp
    for i in range(n):
        T += ((40 + 0.5 * max(pw["drive"][i], 0)) - T) / 600.0
        mt[i] = T
    state.motor_temp = T

    # ---- driver inputs & chassis
    accel_pedal = np.clip(pw["drive"] / 1.2, 0, 100)
    brake_pedal = np.clip((-accel - 0.8) * 45, 0, 100)
    steer = np.cumsum(rng.normal(0, 1.2 if trip.kind == "city" else 0.2, n))
    steer = np.clip(steer - np.convolve(steer, np.ones(60) / 60, "same"), -90, 90)
    lat = speed ** 2 * np.tan(np.radians(steer / 16)) / 2.8 * 0.2

    # ---- HVAC cabin
    k = np.exp(-np.arange(n) / 300.0)
    setpoint = np.full(n, 22.5)
    cabin = setpoint + 1.0 + (amb[0] + 6.0 - setpoint - 1.0) * k
    fan = 30 + 50 * k

    # ---- body & environment
    wiper = np.select([rain > 0.6, rain > 0.3, rain > 0.05], [3, 2, 1], 0)
    lowbeam = ((hour >= 18.2) | (hour < 6.5) | (rain > 0.6)).astype(int)
    lv = np.full(n, 14.1) + rng.normal(0, 0.03, n)
    first_start = trip.purpose in ("commute_am", "highway_out", "weekend_errand")   # first wake after the night
    rest_v = W.lv_resting_voltage(day) if first_start else 12.62
    lv[0] = rest_v
    if first_start and day in W.LV_DRAIN_DAYS:
        events.append({"t": t[0], "type": "log", "proc": "pmic", "sev": 3 if rest_v < 12.3 else 4,
                       "msg": f"12V battery resting voltage at wake {rest_v:.2f} V"})
        if rest_v < 12.0 and not state.lv_dtc_logged:
            state.lv_dtc_logged = True
            events += [{"t": t[0], "type": "dtc", "code": "P1B05", "ecu": "BCM", "status": 0x2F},
                       {"t": t[0], "type": "warning", "code": "LOW_12V_BATTERY", "duration": 60}]

    # ---- DTCs clearing ("test passed") after the problem is gone
    if state.tpms_dtc_logged and not getattr(state, "tpms_passed", False) and tyres[0, 2] > 230:
        state.tpms_passed = True
        events.append({"t": t[0] + 2, "type": "dtc", "code": "C1A10", "ecu": "TPMS", "status": 0x28})
    if state.lv_dtc_logged and not getattr(state, "lv_passed", False) and first_start and rest_v > 12.4:
        state.lv_passed = True
        events.append({"t": t[0] + 2, "type": "dtc", "code": "P1B05", "ecu": "BCM", "status": 0x28})
    # ---- tyres: TPMS warning + DTC (S02)
    low = tyres[:, 2] < W.TPMS_WARN_KPA
    if low.any():
        i = int(np.argmax(low))
        events.append({"t": t[i], "type": "warning", "code": "CHECK_TYRE_PRESSURE", "duration": n - i})
        if not state.tpms_dtc_logged:
            state.tpms_dtc_logged = True
            events += [{"t": t[i], "type": "dtc", "code": "C1A10", "ecu": "TPMS", "status": 0x2F},
                       {"t": t[i], "type": "log", "proc": "tpms_svc", "sev": 3,
                        "msg": f"RL pressure {tyres[i, 2]:.0f} kPa below {W.TPMS_WARN_KPA:.0f} kPa"}]

    # ---- ADAS & camera
    vis = np.clip(95 - 60 * rain + rng.normal(0, 2, n), 5, 100)
    conf = np.clip(92 - 40 * rain + np.convolve(rng.normal(0, 3, n), np.ones(5) / 5, "same"), 0, 100)
    cam_status = np.where(vis < 50, 1, 0)                        # 0 OK, 1 DEGRADED, 2 BLOCKED, 3 OFFLINE
    if "S03" in trip.tags:
        for (h, m, s) in W.RAIN_DISENGAGE_TIMES:
            i0 = int(W.ts(day, h, m, s) - t[0])
            if 0 <= i0 < n:
                conf[max(0, i0 - 20):i0] = np.linspace(conf[max(0, i0 - 20)], 34, i0 - max(0, i0 - 20))
                conf[i0:i0 + 8] = rng.uniform(25, 34, min(8, n - i0))
    cam_fault_t = None
    if "S04" in trip.tags:
        cam_fault_t = W.ts(day, *W.CAMERA_FAULT_TIME)
        i0 = int(cam_fault_t - t[0])
        cam_status[i0:i0 + 25] = 3
        vis[i0:i0 + 25] = 0
        events += [
            {"t": cam_fault_t + 1, "type": "log", "proc": "camerad", "sev": 2, "msg": "CAM0 GMSL link lost (no frames 500 ms)"},
            {"t": cam_fault_t + 2, "type": "log", "proc": "camerad", "sev": 3, "msg": "restarting camera service (1/3)"},
            {"t": cam_fault_t + 9, "type": "log", "proc": "camerad", "sev": 3, "msg": "restarting camera service (2/3)"},
            {"t": cam_fault_t + 17, "type": "log", "proc": "camerad", "sev": 3, "msg": "restarting camera service (3/3)"},
            {"t": cam_fault_t + 22, "type": "log", "proc": "camerad", "sev": 5, "msg": "CAM0 link up, streaming 30 fps"},
            {"t": cam_fault_t + 5, "type": "dtc", "code": "U1C01", "ecu": "ADAS", "status": 0x2F},
            {"t": cam_fault_t + 90, "type": "dtc", "code": "U1C01", "ecu": "ADAS", "status": 0x28},
            {"t": cam_fault_t, "type": "warning", "code": "CAMERA_UNAVAILABLE", "duration": 25},
        ]
    objs = np.clip(rng.poisson(6 if trip.kind == "city" else 3, n), 0, 30)

    lka = np.zeros(n, dtype=int)
    cc = np.zeros(n, dtype=int)
    engaged, ok_run, low_run = False, 0, 0
    override_at = None
    if trip.kind == "highway" and day in (13, 34) and trip.purpose == "highway_out":
        override_at = int(n * 0.5)
    for i in range(n):
        v = speed[i] * 3.6
        eligible = road[i] == 1 and v > 50 and cam_status[i] != 3
        if not engaged:
            ok_run = ok_run + 1 if (eligible and conf[i] >= 50) else 0
            if ok_run >= 10:
                engaged = True
        else:
            low_run = low_run + 1 if conf[i] < 40 else 0
            reason = None
            if cam_status[i] == 3:
                reason = "CAMERA_UNAVAILABLE"
            elif low_run >= 2:
                reason = "LANE_MARKINGS_LOST"
            elif override_at is not None and i == override_at:
                reason = "DRIVER_OVERRIDE"
            elif road[i] == 0 or v <= 50:
                events.append({"t": t[i], "type": "adas", "event": "LKA_OFF", "reason": "ODD_EXIT",
                               "speed": v, "conf": conf[i]})
                engaged, ok_run = False, 0
            if reason:
                events.append({"t": t[i], "type": "adas", "event": "LKA_DISENGAGED", "reason": reason,
                               "speed": v, "conf": conf[i]})
                if reason == "LANE_MARKINGS_LOST":
                    events.append({"t": t[i], "type": "log", "proc": "adas_lka", "sev": 4,
                                   "msg": f"LKA handover: lane confidence {conf[i]:.0f}% < 40%"})
                engaged, ok_run, low_run = False, 0, 0
        lka[i] = int(engaged)
        cc[i] = int(road[i] == 1 and v > 60)
    ldw_warn = ((np.abs(steer) > 25) & (road == 1) & (rng.random(n) < 0.02)).astype(int)
    esc = ((rain > 0.3) & (road == 0) & (rng.random(n) < 0.0008)).astype(int)

    # ---- data-quality gap (S08): CAN logger drops frames
    can_ok = np.ones(n, dtype=bool)
    if day == W.GAP_DAY and trip.purpose == "commute_pm":
        a = 8 * 60
        b = min(n, a + W.GAP_MINUTES * 60)
        can_ok[a:b] = False
        events += [{"t": t[a], "type": "log", "proc": "canlogger", "sev": 2, "msg": "ring buffer overrun on can0, dropping frames"},
                   {"t": t[b - 1], "type": "log", "proc": "canlogger", "sev": 5, "msg": f"can0 capture resumed after {(b - a) // 60} min"}]

    # ---- pump telemetry for SOME/IP thermal service
    pump_rpm = 3200 * pump + rng.normal(0, 15, n)
    coolant = np.where(temp_avg > 30, 22 + 0.4 * (temp_avg - 30), amb - 1)

    # ---- update state, range estimate learns this trip's consumption
    trip_cons = energy_kwh / max(dist_km, 0.1) * 100
    w = min(1.0, dist_km / 100.0)
    state.cons_ema = (1 - w) * state.cons_ema + w * trip_cons
    state.soc, state.odometer_km, state.last_t = float(soc[-1]), float(odo[-1]), float(t[-1])
    state.batt_temp = float(temp_avg[-1])

    sig = dict(
        speed_kmh=speed * 3.6, accel=accel, accel_pedal=accel_pedal, brake_pedal=brake_pedal, gear=np.full(n, 3),
        power_state=np.full(n, 2), steer=steer, accel_lat=lat, odometer=odo, motor_rpm=rpm, motor_temp=mt,
        motor_torque=torque, regen_kw=pw["regen"], pack_v=volt, pack_a=current, pack_w=pw["total"] * 1000, soc=soc,
        temp_avg=temp_avg, temp_max=temp_max, temp_min=temp_min, soh=np.full(n, state.soh),
        soc_disp=np.clip(soc * 1.02 - 2, 0, 100), dis_limit=limit, chg_limit=np.full(n, 120.0), range_km=range_km,
        is_charging=np.zeros(n), dc_a=np.zeros(n), ac_a=np.zeros(n), ac_kw=ac, heater_kw=np.zeros(n),
        ac_on=np.ones(n), cabin=cabin, setpoint=setpoint, fan=fan, ambient=amb, wiper=wiper, lowbeam=lowbeam, lv=lv,
        tyres=tyres, abs=np.zeros(n), esc=esc, eba=np.zeros(n), road=road, rain=rain, lane_conf=conf, lka=lka, cc=cc,
        ldw_on=np.ones(n), ldw_warn=ldw_warn, cam_status=cam_status, cam_vis=vis, cam_objs=objs, pump_rpm=pump_rpm,
        pump_a=pump * 6.5, coolant=coolant, chiller=(temp_avg > 30).astype(int), can_ok=can_ok,
    )
    meta = {"day": day, "purpose": trip.purpose, "kind": trip.kind, "dist_km": dist_km, "energy_kwh": energy_kwh,
            "traffic": tr, "duration_s": n, "tags": trip.tags}
    return Session("trip", t, sig, events, meta)


def simulate_charge(kind: str, t0: float, t1: float | None, state: CarState, day: int, target_soc: float = 90.0) -> Session:
    """DC fast charge (car ON, 1 Hz) or overnight AC charge (car asleep, BMS wakes every 10 s)."""
    step = 1.0 if kind == "dc_charge" else 10.0
    if t1 is None:
        hours = max(0.2, (target_soc - state.soc) / 100 * P.usable_kwh / 7.0)
        t1 = t0 + hours * 3600
    t = np.arange(t0, t1, step)
    n = t.size
    amb = W.ambient_c(t)
    _relax(state, t0, float(amb[0]))
    if kind == "dc_charge":
        soc_guess = state.soc + np.arange(n) * step * 80 / 3600 / P.usable_kwh * 100
        p_kw = np.where(soc_guess < 60, 90.0, np.clip(90 - (soc_guess - 60) * 2.5, 40, 90))
    else:
        p_kw = np.full(n, 7.0)
    p_kw = np.where(np.cumsum(p_kw * step / 3600 / P.usable_kwh * 100) + state.soc > target_soc + 5, 0.0, p_kw)
    soc = state.soc + np.cumsum(p_kw * step) / 3600 / P.usable_kwh * 100
    events = []
    pump_frac = None
    if kind == "dc_charge" and day == W.OVERHEAT_DAY:
        tpump = W.ts(day, *W.OVERHEAT_T0) - 12 * 60
        pump_frac = np.where(t >= tpump, 0.28, 1.0)
        events += [{"t": tpump, "type": "dtc", "code": "P1A20", "ecu": "BMS", "status": 0x2F},
                   {"t": tpump, "type": "log", "proc": "thermal_svc", "sev": 3,
                    "msg": "coolant pump speed deviation: 900 rpm measured / 3200 rpm requested"}]
    temp_avg = np.empty(n)
    T = state.batt_temp
    for i in range(n):
        chiller = T > 30.0
        pf = pump_frac[i] if pump_frac is not None else (1.0 if chiller else 0.6)
        target = 22.0 if chiller else amb[i]
        cur = p_kw[i] * 1000 / 380.0
        T += step * (0.08 * cur ** 2 / 1000 - 0.5 * pf * (T - target)) / 400.0
        temp_avg[i] = T
    if pump_frac is not None:                              # S05: hot spot rises once the pump degrades
        t0w = W.ts(day, *W.OVERHEAT_T0)
        k = np.clip((t - (t0w - 12 * 60)) / (7 * 60), 0, 1)
        temp_avg = np.maximum(temp_avg, 36 + 6 * k)
    state.batt_temp = float(temp_avg[-1])
    temp_max = temp_avg + 1.5 + (2.5 * np.clip((temp_avg - 36) / 6, 0, 1) if pump_frac is not None else 0)
    current = -p_kw * 1000 / 380.0
    volt = pack_voltage(soc, current)
    state.soc, state.last_t = float(soc[-1]), float(t[-1])
    range_km = soc / 100.0 * P.usable_kwh * state.soh / 100.0 / (state.cons_ema / 100.0)
    z = np.zeros(n)
    pump = pump_frac if pump_frac is not None else np.where(temp_avg > 30, 1.0, 0.6)
    sig = dict(
        speed_kmh=z, accel=z, accel_pedal=z, brake_pedal=z, gear=z, power_state=np.full(n, 3), steer=z, accel_lat=z,
        odometer=np.full(n, state.odometer_km), motor_rpm=z, motor_temp=np.full(n, state.motor_temp), motor_torque=z,
        regen_kw=z, pack_v=volt, pack_a=current, pack_w=-p_kw * 1000, soc=soc, temp_avg=temp_avg, temp_max=temp_max,
        temp_min=temp_avg - 1, soh=np.full(n, state.soh), soc_disp=np.clip(soc * 1.02 - 2, 0, 100),
        dis_limit=np.full(n, 150.0), chg_limit=np.where(temp_max > 45, 60.0, 120.0), range_km=range_km,
        is_charging=(p_kw > 0).astype(int), dc_a=(-current if kind == "dc_charge" else z),
        ac_a=(np.where(p_kw > 0, 31.3, 0) if kind == "ac_charge" else z), ac_kw=z, heater_kw=z, ac_on=z,
        cabin=amb + 2, setpoint=np.full(n, 22.5), fan=z, ambient=amb, wiper=z, lowbeam=z, lv=np.full(n, 13.6),
        tyres=W.tyre_pressures(t, amb), abs=z, esc=z, eba=z, road=z.astype(np.int8), rain=W.rain_at(t), lane_conf=z,
        lka=z, cc=z, ldw_on=z, ldw_warn=z, cam_status=z, cam_vis=z, cam_objs=z, pump_rpm=3200 * pump, pump_a=pump * 6.5,
        coolant=np.where(temp_avg > 30, 22 + 0.4 * (temp_avg - 30), amb - 1), chiller=(temp_avg > 30).astype(int),
        can_ok=np.ones(n, dtype=bool),
    )
    meta = {"day": day, "kind": kind, "kwh_added": float(p_kw.sum() * step / 3600), "duration_s": float(t1 - t0),
            "peak_temp_max": float(temp_max.max()), "ambient_c": float(amb.mean())}
    return Session(kind, t, sig, events, meta)
