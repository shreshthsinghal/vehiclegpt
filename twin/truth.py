"""Answer key. Computes the correct answers to the golden questions from the simulation itself.

For the range question (S01) the causes are measured with COUNTERFACTUALS: we re-drive the exact same
trips in the last two weeks with (a) normal weather, (b) normal traffic, (c) both, and split the extra
energy fairly between the causes (a 2-player Shapley split). This is why the answer key is trustworthy.
VehicleGPT never sees this file; it must find the causes from the recorded data alone.
"""
from __future__ import annotations

import datetime as dt

import numpy as np

from . import world as W
from .simulate import CarState, simulate_trip


def _loc(t: float) -> str:
    return f"{dt.datetime.fromtimestamp(t, W.IST):%Y-%m-%d %H:%M:%S}"


def _cons(trips) -> float:
    e = sum(t["energy_kwh"] for t in trips)
    d = sum(t["dist_km"] for t in trips)
    return e / d * 100


def _s01(trips_meta, days):
    from .generate import SEED
    if days < max(W.WINDOW_B):
        return {"note": "needs all 56 days"}
    A = [t for t in trips_meta if t["day"] in W.WINDOW_A]
    B = [t for t in trips_meta if t["day"] in W.WINDOW_B]
    plan = W.plan_trips(SEED)
    idx = {(tr.day, tr.purpose, round(tr.start)): k for k, tr in enumerate(plan)}

    def window_b(heat: bool, normal_traffic: bool) -> float:
        e = d = 0.0
        for k, tr in enumerate(plan):
            if tr.day not in W.WINDOW_B:
                continue
            s = simulate_trip(tr, CarState(), seed=SEED + k, heat=heat,
                              traffic=1.0 if normal_traffic else None, counterfactual=True)
            e += s.sig["energy_kwh"]
            d += s.sig["dist_km"]
        return e / d * 100

    actual = window_b(True, False)
    no_heat = window_b(False, False)
    no_traffic = window_b(True, True)
    neither = window_b(False, True)
    heat_c = 0.5 * ((actual - no_heat) + (no_traffic - neither))
    traffic_c = 0.5 * ((actual - no_traffic) + (no_heat - neither))
    cons_a, cons_b = _cons(A), _cons(B)
    delta = cons_b - cons_a
    other = delta - heat_c - traffic_c
    rps_a = np.median([t["range_per_soc"] for t in A])
    rps_b = np.median([t["range_per_soc"] for t in B])
    return {
        "window_a": f"days {min(W.WINDOW_A)}-{max(W.WINDOW_A)}", "window_b": f"days {min(W.WINDOW_B)}-{max(W.WINDOW_B)}",
        "consumption_a_kwh_100km": round(cons_a, 2), "consumption_b_kwh_100km": round(cons_b, 2),
        "consumption_increase_pct": round(delta / cons_a * 100, 1),
        "displayed_range_change_pct": round((rps_b / rps_a - 1) * 100, 1),
        "contributions_kwh_100km": {"ac_heat": round(heat_c, 2), "stop_go_traffic": round(traffic_c, 2),
                                    "other": round(other, 2)},
        "shares_pct": {"ac_heat": round(heat_c / delta * 100), "stop_go_traffic": round(traffic_c / delta * 100),
                       "other": round(other / delta * 100)},
        "explained_shares_pct": {"ac_heat": round(heat_c / (heat_c + traffic_c) * 100),
                                 "stop_go_traffic": round(traffic_c / (heat_c + traffic_c) * 100)},
        "note": "'other' is normal trip-to-trip variation (e.g. the rain-slowed highway trip on day 55 used less energy)",
        "avg_ac_kwh_per_trip": {"a": round(np.mean([t["ac_kwh"] for t in A]), 3), "b": round(np.mean([t["ac_kwh"] for t in B]), 3)},
        "stop_go_pct": {"a": round(np.mean([t["stop_go_pct"] for t in A]), 1), "b": round(np.mean([t["stop_go_pct"] for t in B]), 1)},
        "tyres_normal_in_window_b": bool(min(t["min_rl_kpa"] for t in B) > 225),
    }


def _s02(trips_meta):
    from .generate import SEED
    plan = W.plan_trips(SEED)
    penalty = {}
    for k, tr in enumerate(plan):
        if not (W.PUNCTURE_START[0] <= tr.day <= W.PUNCTURE_FIXED[0]):
            continue
        a = simulate_trip(tr, CarState(), seed=SEED + k, counterfactual=True)
        b = simulate_trip(tr, CarState(), seed=SEED + k, counterfactual=True, puncture=False)
        e = penalty.setdefault(tr.day, [0.0, 0.0])
        e[0] += a.sig["energy_kwh"] - b.sig["energy_kwh"]
        e[1] += b.sig["energy_kwh"]
    tyre_pct = {d: round(v[0] / v[1] * 100, 1) for d, v in sorted(penalty.items())}
    city = [t for t in trips_meta if t["kind"] == "city"]
    by_day = {}
    for t in city:
        by_day.setdefault(t["day"], []).append(t)
    daily = {d: _cons(v) for d, v in sorted(by_day.items())}
    base = np.median([v for d, v in daily.items() if d < W.PUNCTURE_START[0]])
    affected = {d: round((v / base - 1) * 100, 1) for d, v in daily.items() if W.PUNCTURE_START[0] <= d <= W.PUNCTURE_FIXED[0]}
    worst = max(tyre_pct, key=tyre_pct.get) if tyre_pct else None
    return {"pressure_loss_start": _loc(W.ts(*W._dh(W.PUNCTURE_START))), "repaired": _loc(W.ts(*W._dh(W.PUNCTURE_FIXED))),
            "baseline_city_kwh_100km": round(base, 2), "observed_city_change_pct_by_day": affected,
            "tyre_energy_penalty_pct_by_day": tyre_pct,
            "worst_day": worst, "worst_day_date": f"{W.day_start(worst):%Y-%m-%d}" if worst else None,
            "cause": "rear-left tyre under-inflated (slow puncture), min 175 kPa vs 240 kPa nominal"}


def build_truth(trips_meta, charges, events, days) -> dict:
    disen = [e for e in events if e["type"] == "adas" and e["event"] == "LKA_DISENGAGED"]
    s03 = [{"time": _loc(e["t"]), "reason": e["reason"], "speed_kmh": round(e["speed"], 1),
            "lane_confidence": round(e["conf"])} for e in disen if e["day"] == 55]
    s05_ev = sorted([e for e in events if e["day"] == W.OVERHEAT_DAY and e["type"] in ("dtc", "warning", "log")
                     and e.get("status", 0x2F) == 0x2F],
                    key=lambda e: e["t"])
    dc = [c for c in charges if c["kind"] == "dc_charge"]
    city_t = [t for t in trips_meta]
    city_e = sum(t["city_kwh"] for t in city_t); city_d = sum(t["city_km"] for t in city_t)
    hwy_e = sum(t["hwy_kwh"] for t in city_t); hwy_d = sum(t["hwy_km"] for t in city_t)
    gap = [e for e in events if e["type"] == "log" and e.get("proc") == "canlogger"]
    lv = [e for e in events if e["type"] == "dtc" and e["code"] == "P1B05"]
    return {
        "today": f"{W.day_start(min(days, 56)):%Y-%m-%d} (day {min(days, 56)})",
        "S01_range_drop": _s01(trips_meta, days),
        "S02_efficiency_drop": _s02(trips_meta),
        "S03_S04_adas_yesterday": {"date": f"{W.day_start(55):%Y-%m-%d}", "count": len(s03), "events": s03},
        "S05_overheat": {"warning_time": _loc(W.ts(W.OVERHEAT_DAY, *W.OVERHEAT_T0)),
                         "root_cause": "coolant pump performance low (P1A20) during DC fast charge",
                         "sequence": [{"time": _loc(e["t"]), "type": e["type"],
                                       "detail": e.get("code") or e.get("msg")} for e in s05_ev],
                         "dc_charges": [{"time": _loc(c["start"]), "peak_temp_max_c": round(c["peak_temp_max"], 1),
                                         "ambient_c": round(c["ambient_c"], 1)} for c in dc]},
        "S06_city_vs_highway": {"city_kwh_100km": round(city_e / max(city_d, 0.1) * 100, 2),
                                "highway_kwh_100km": round(hwy_e / max(hwy_d, 0.1) * 100, 2),
                                "city_km": round(city_d), "highway_km": round(hwy_d)},
        "S07_12v": {"dtc": [_loc(e["t"]) for e in lv], "resting_voltage_by_day": W.LV_DRAIN_DAYS},
        "S08_data_gap": {"log_lines": [{"time": _loc(e["t"]), "msg": e["msg"]} for e in gap]},
        "S09_baseline": {"days": "1-14", "faults": [e["code"] for e in events if e["type"] == "dtc" and e["day"] <= 14]},
        "totals": {"trips": len(trips_meta), "km": round(sum(t["dist_km"] for t in trips_meta)),
                   "dc_charges": len(dc), "ac_charges": len(charges) - len(dc)},
    }
