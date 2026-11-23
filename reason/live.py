"""Present state, what-if scenarios, quantified advice, the safety log and the emergency guide.

Same rules as every playbook: numbers come from the store or from a model fitted on this car's own trips, and each is
registered as a fact for the verifier. Emergency steps are fixed text from cited sources, never generated.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

from .answer import Answer
from .catalogue import CATALOGUE, SNAPSHOT, TYRES, find, label
from .dates import Window, fmt_date, fmt_time, phrase

HERE = Path(__file__).parent
V = "Vehicle."
TREND_WORDS = (r"trend|over time|history|changed|change|lately|recently|past|last (week|month|\d+)|average|highest|lowest|since|"
               r"how many times|how often|getting|worse|better|week|month|yesterday|today|\bago\b|during|when|\bdid\b|\bwas\b|\bwere\b|"
               r"declin|deteriorat|degrad|dropping|falling|improving|going (down|up)|\d{1,2} (oct|nov)|monday|tuesday|wednesday|thursday|friday|saturday|sunday|why|drop|lower|less|loss|compare")
NOW_WORDS = (r"\bnow\b|right now|currently|current|present|at the moment|is (it|the|my) .* (fine|ok|okay|working|good|healthy|normal)|"
             r"\bare (my|the)\b|status|^(what'?s|whats|what is|what are|how'?s|how is|how are|is|are) (the |my |our )?")


def r1(x):
    v = round(float(x), 1)
    return int(v) if v.is_integer() else v


# ------------------------------------------------------------------ helpers
def latest(ctx, key, look_h=6):
    """Latest stored value of a catalogue signal (WARM tier), or COLD minute mean if older."""
    path = CATALOGUE[key][0]
    ts, v = ctx.st.series(path, ctx.now - look_h * 3600, ctx.now + 1)
    if v.size:
        return float(v[-1]), float(ts[-1])
    r = ctx.st.rollup(path, ctx.first - 60, ctx.now + 1)
    if r["minute"].size:
        return float(r["mean"][-1]), float(r["minute"][-1] * 60)
    return None, None


def active_dtcs(ctx):
    sets, passes = ctx.st.events(["DTC_SET"]), ctx.st.events(["DTC_PASSED"])
    return [e for e in sets if not any(p["code"] == e["code"] and p["ts"] > e["ts"] for p in passes)]


def dtc_desc(ctx, code):
    row = ctx.st.db.execute("SELECT description FROM dtc_dict WHERE code=?", (code,)).fetchone()
    return row[0] if row else code


# ------------------------------------------------------------------ present state
def vehicle_status_now(ctx, q: str, plan: dict) -> Answer:
    a = Answer(q, "vehicle_status_now")
    f = a.fact
    ql = q.lower()
    keys = [plan["signal"]] if plan.get("signal") in CATALOGUE else find(q)
    snapshot = not keys or bool(re.search(r"everything|overall|status of (the|my) car|car('s)? (status|condition)|how is (the|my) car|"
                                          r"full (status|report)|all (systems|signals)", ql))
    if snapshot:
        keys = SNAPSHOT + TYRES
    a.plan["signals"], a.plan["snapshot"] = keys, bool(snapshot)
    a.evidence.append("WARM tier: latest value of each signal (1 Hz history), COLD tier if older")
    vals, newest = {}, 0.0
    for k in keys:
        v, t = latest(ctx, k)
        if v is None:
            continue
        vals[k] = v
        newest = max(newest, t)
        f(k, r1(v) if CATALOGUE[k][4] is None else label(k, v), CATALOGUE[k][2], CATALOGUE[k][0])
    # legacy fact names used by the status strip / UI
    for k, legacy in (("soc", "state_of_charge"), ("range", "range"), ("batt_temp", "battery_temp")):
        if k in vals:
            f(legacy, r1(vals[k]))
    for k, col in zip(TYRES, ["tyre_fl_min", "tyre_fr_min", "tyre_rl_min", "tyre_rr_min"]):
        if k in vals:
            f(col, r1(vals[k]))
    f("at", fmt_time(newest or ctx.now))
    rear = re.search(r"\b(rear|reverse|back(up)?|parking) cam", ql)
    if rear:
        a.why.append("This car records only a front camera; there is no rear or parking camera signal, so I can't check it.")
    if not vals:
        a.what = "I have no recent reading for that signal."
        return a
    parts = []
    tyre_keys = [k for k in TYRES if k in vals]
    for k in keys:
        if k not in vals or (k in TYRES and len(tyre_keys) > 1):
            continue
        parts.append(f"{CATALOGUE[k][1]} {label(k, vals[k])}")
    if len(tyre_keys) > 1:
        tl = ", ".join(f"{CATALOGUE[k][1].split(' tyre')[0]} {label(k, vals[k])}" for k in tyre_keys)
        ok = min(vals[k] for k in tyre_keys) >= 225
        f("tyre_ok_limit", 225, "kPa", "low-pressure limit used by VehicleGPT")
        parts.append(f"tyres {tl} ({'all normal' if ok else 'LOW: check them'})")
    when = fmt_time(newest)
    if snapshot:
        a.what = f"Your car right now (last reading {when}): " + "; ".join(parts) + "."
    elif rear:
        a.what = f"I can't see a rear camera on this car. The front camera is {label('camera', vals['camera'])} (reading at {when})." \
            if "camera" in vals else "I can't see a rear camera on this car."
    else:
        a.what = f"Right now ({when}): " + "; ".join(parts) + "."
    # judgements that make a value useful
    if "soh" in vals:
        a.why.append(f"Battery health {r1(vals['soh'])}% means the battery still holds {r1(vals['soh'])}% of its original "
                     f"capacity" + (": excellent." if vals["soh"] >= 95 else "."))
    if "camera" in vals and not rear:
        a.why.append("The front camera is working normally." if vals["camera"] == 0 else
                     "The front camera is not fully working: lane assist may switch off.")
    if "lv_voltage" in vals:
        f("lv_ok", 12.3, "V", "normal minimum at rest")
        a.why.append("12 V battery is normal." if vals["lv_voltage"] >= 12.3 else "12 V battery is low; have it checked.")
    act = active_dtcs(ctx)
    warns_on = [e for e in ctx.st.events(["WARNING_ON", "WARNING_OFF"], ctx.now - 86400, ctx.now + 1)]
    if snapshot:
        for e in act:
            f(f"adtc{e['id']}", e["code"])
        a.why.append("Active fault: " + ", ".join(f"{e['code']} ({dtc_desc(ctx, e['code']).lower()})" for e in act) + "."
                     if act else "No active faults.")
        a.next = "Ask \"show my safety log\" for everything safety-related."
    return a


# ------------------------------------------------------------------ consumption model (fitted on this car's trips)
def model(ctx) -> dict:
    if getattr(ctx, "_model", None):
        return ctx._model
    T = [t for t in ctx.st.table("trips") if t["distance_km"] > 3 and (t["data_coverage_pct"] or 100) > 95]
    D = np.array([t["distance_km"] for t in T])
    y = np.array([(t["energy_kwh"] - t["ac_kwh"]) for t in T]) / D * 100
    vs = np.array([t["avg_speed_kmh"] for t in T]); sg = np.array([t["stop_go_pct"] for t in T])
    X = np.c_[np.ones(len(T)), vs ** 2 / 1000, sg]
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    H = np.array([(t["end_ts"] - t["start_ts"]) / 3600 for t in T]); P = np.array([t["ac_kwh"] for t in T]) / H
    A = np.array([t["ambient_avg_c"] for t in T])
    ba, *_ = np.linalg.lstsq(np.c_[np.ones(len(T)), A], P, rcond=None)
    ch = ctx.st.table("charges")
    cap = [c["energy_added_kwh"] / ((c["soc_end"] - c["soc_start"]) / 100) for c in ch if c["soc_end"] - c["soc_start"] > 10]
    power = {}
    for kind in ("AC_HOME", "DC_FAST"):
        s = [c["energy_added_kwh"] / (c["duration_min"] / 60) for c in ch if c["charger"] == kind and c["duration_min"] > 5]
        if s:
            power[kind] = float(np.median(s))
    recent = [t for t in T if t["start_ts"] > ctx.now - 14 * 86400]
    base = sum(t["energy_kwh"] for t in recent) / sum(t["distance_km"] for t in recent) * 100
    ac_base = sum(t["ac_kwh"] for t in recent) / sum(t["distance_km"] for t in recent) * 100
    hw = [t for t in T if t["highway_km"] / t["distance_km"] > 0.5]
    m = {"b": b, "r2": float(r2), "n": len(T), "ac": ba, "cap": float(np.median(cap)) if cap else 60.0, "power": power,
         "base": float(base), "ac_base": float(ac_base), "amb": float(np.mean([t["ambient_avg_c"] for t in recent])),
         "hw_speed": float(np.mean([t["avg_speed_kmh"] for t in hw])) if hw else 80.0,
         "city_sg": float(np.median([t["stop_go_pct"] for t in T if t["highway_km"] == 0] or [20]))}
    ctx._model = m
    return m


def kwh100(m, speed, sg, ac_on=True, amb=None):
    drive = m["b"][0] + m["b"][1] * speed ** 2 / 1000 + m["b"][2] * sg
    ac = max(0.0, m["ac"][0] + m["ac"][1] * (amb if amb is not None else m["amb"])) if ac_on else 0.0
    return float(drive + ac / speed * 100)


NUM = r"(\d+(?:\.\d+)?)"


def parse_scenario(q: str, plan: dict) -> dict:
    ql = q.lower()
    p = {"distance": plan.get("distance_km"), "speed": plan.get("speed_kmh"), "ac": plan.get("ac"),
         "target": plan.get("target_soc"), "charger": None, "road": None, "amb": None}
    p["miles"] = bool(re.search(r"\bmiles?\b|\bmph\b|\bmi\b", ql))
    if not p["distance"]:
        m = re.search(NUM + r"\s*(km|kms|kilomet\w*)\b(?!\s*/\s*h|\s*per hour|ph)", ql)
        mm = re.search(NUM + r"\s*(miles?|mi)\b(?!\s*/\s*h|\s*per hour)", ql)
        p["distance"] = float(m.group(1)) if m else (float(mm.group(1)) * MILE if mm else None)
    if p["distance"] and re.search(r"and back|round.?trip|return trip|there and back", ql):
        p["distance"] *= 2
        p["round"] = True
    if not p["speed"]:
        pair = re.search(NUM + r"\s*(?:km/?h|kmph|kph)?\s*(?:vs\.?|versus|or|and)\s*" + NUM + r"\s*(?:km/?h|kmph|kph)", ql)
        m = re.findall(NUM + r"\s*(?:km/?h|kmph|kph)", ql)
        p["speed"] = [float(pair.group(1)), float(pair.group(2))] if pair else ([float(x) for x in m] if m else None)
        mph = re.findall(NUM + r"\s*mph", ql)
        if mph and not p["speed"]:
            p["speed"] = [float(x) * MILE for x in mph]
    elif not isinstance(p["speed"], list):
        p["speed"] = [float(p["speed"])]
    p["place"] = None
    mp = re.search(r"\b(?:to|reach|get to|make it to)\s+(?:the\s+)?(?:nearest\s+|closest\s+|my\s+)?([a-z']+)", ql)
    if mp and not p["distance"] and mp.group(1) not in ("charge", "charging", "full", "drive", "go", "my", "a", "an", "work", "office", "home"):
        p["place"] = mp.group(1)
    elif mp and not p["distance"] and mp.group(1) in ("work", "office", "home"):
        p["place"] = mp.group(1)
    if p["ac"] in (None, "default"):
        p["ac"] = ("off" if re.search(r"(without|no|turn(ing)? off|switch(ing)? off|skip)\s+(the\s+)?(ac|a/c|air.?con\w*)|(ac|air.?con\w*)\s+off", ql)
                   else "on")
    m = re.search(r"(?:to|reach|until)\s+" + NUM + r"\s*%", ql)
    p["target"] = float(m.group(1)) if m else (100.0 if re.search(r"\bfull\b|100 ?%", ql) else p["target"])
    m = re.search(r"(?:i'?m at|i am at|down to|i'?m down to|at|with|have|leave (?:now )?at)\s+" + NUM + r"\s*%(?:\s*(?:battery|charge|soc|right now|now))?", ql)
    p["soc_given"] = float(m.group(1)) if m and not re.search(r"(at least|reach|charge to|charging to|get to|limit to)\s+" + m.group(1) + r"\s*%", ql[:m.end()]) else None
    m = re.search(r"at least\s+" + NUM + r"\s*%", ql)
    p["reserve"] = float(m.group(1)) if m else None
    m = re.search(NUM + r"\s*kw\b", ql)
    p["charger_kw"] = float(m.group(1)) if m else None
    p["charger"] = "DC_FAST" if re.search(r"fast|\bdc\b|rapid|supercharg", ql) else ("AC_HOME" if re.search(r"home|\bac charger|wall", ql) else None)
    p["road"] = "highway" if re.search(r"highway|motorway|expressway", ql) else ("city" if re.search(r"\bcity|town|traffic", ql) else None)
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:°|deg\w*)\s*(f\b|fahrenheit)?", ql)
    p["amb"] = None
    if m:
        p["amb"] = (float(m.group(1)) - 32) * 5 / 9 if m.group(2) else float(m.group(1))
    elif re.search(r"freezing|below zero|snow|icy", ql):
        p["amb"] = 0.0
    return p


MILE = 1.609344


def what_if(ctx, q: str, plan: dict) -> Answer:
    a = Answer(q, "what_if")
    f = a.fact
    ql = q.lower()
    m = model(ctx)
    p = parse_scenario(q, plan)
    soc, _ = latest(ctx, "soc")
    soc = soc if soc is not None else 80.0
    if p.get("soc_given") is not None:
        soc = p["soc_given"]
        f("soc_given", r1(soc))
        a.why.append(f"Using the {r1(soc)}% you gave me instead of the current reading.")
    cap = m["cap"]
    f("soc_now", r1(soc)); f("cap", r1(cap), "kWh", "median of your charging sessions"); f("n_model", m["n"])
    f("r2", round(m["r2"], 2)); f("r2pct", round(m["r2"] * 100)); f("base", r1(m["base"]), "kWh/100km", "your last 2 weeks")
    a.evidence += [f"consumption model fitted on {m['n']} of your trips (R² {round(m['r2'], 2)}): speed², stop-go, AC vs outside temperature",
                   "usable battery from your charging sessions", "WARM tier: current state of charge"]
    reserve = p.get("reserve") or 10.0
    f("reserve", r1(reserve), "%", "safety reserve")
    amb_seen = [float(t["ambient_avg_c"]) for t in ctx.st.table("trips") if t["ambient_avg_c"] is not None]
    lo, hi = min(amb_seen), max(amb_seen)
    if p.get("amb") is not None and not lo - 3 <= p["amb"] <= hi + 3:
        f("amb_lo", r1(lo)); f("amb_hi", r1(hi)); f("amb_q", r1(p["amb"]))
        a.why.append(f"Caution: your car has only been driven at {r1(lo)}–{r1(hi)} °C outside, so a prediction for "
                     f"{r1(p['amb'])} °C is uncertain. Cold weather adds cabin and battery heating that this car's data "
                     f"cannot measure yet; expect less range than shown.")
        a.confidence = "low"
    unit_mi = p.get("miles")
    cold = p.get("amb") is not None and p["amb"] < lo - 3
    if p.get("amb") is not None:            # never extrapolate beyond the temperatures this car has seen
        p["amb"] = min(max(p["amb"], lo), hi)

    # ---- how long to charge
    if re.search(r"how long|how much time|minutes|hours|time to charge|will it take|would it take", ql) and re.search(r"charg", ql):
        target = p["target"] or 80.0
        kind = p["charger"] or ("DC_FAST" if (p.get("charger_kw") or 0) > 22 else "AC_HOME")
        pw = m["power"].get(kind) or next(iter(m["power"].values()))
        if p.get("charger_kw"):
            seen = max([c["energy_added_kwh"] / (c["duration_min"] / 60) for c in ctx.st.table("charges")
                        if c["charger"] == kind and c["duration_min"] > 5] or [pw])
            f("charger_kw", r1(p["charger_kw"])); f("seen_kw", r1(seen)); f("pw_typ", r1(pw))
            if p["charger_kw"] < pw:
                pw = p["charger_kw"]
            else:
                a.why.append(f"The charger offers {r1(p['charger_kw'])} kW, but your car's sessions averaged about "
                             f"{r1(pw)} kW (best {r1(seen)} kW), so I used your car's real rate.")
        need = max(0.0, (target - soc) / 100 * cap)
        hrs = need / pw
        if need <= 0:
            f("target", r1(target))
            a.what = f"You are already at {r1(soc)}%, above {r1(target)}%: no charging needed."
            a.plan["scenario"] = "charge_time"
            return a
        f("target", r1(target)); f("pw", r1(pw), "kW", "median power of your sessions"); f("need", r1(need))
        f("hrs", r1(hrs)); f("mins", round(hrs * 60))
        name = "a DC fast charger" if kind == "DC_FAST" else "your home charger"
        a.what = (f"From {r1(soc)}% to {r1(target)}% needs {r1(need)} kWh: about {round(hrs * 60)} min "
                  f"({r1(hrs)} h) on {name}, at the {r1(pw)} kW your sessions usually reach.")
        if kind == "DC_FAST" and target > 80:
            f("taper", 80)
            a.why.append("Fast charging slows down above 80%, so the last part takes longer than this estimate.")
        a.plan["scenario"] = "charge_time"
        return a

    # ---- consumption for the scenario
    speeds = p["speed"]
    amb = p["amb"]
    if speeds:
        sg = 2.0 if (p["road"] == "highway" or speeds[0] >= 60) else m["city_sg"]
        cons = [kwh100(m, s, sg, p["ac"] == "on", amb) for s in speeds]
    elif p["road"] == "highway":
        cons = [kwh100(m, m["hw_speed"], 2.0, p["ac"] == "on", amb)]
        speeds = [m["hw_speed"]]
    elif p["road"] == "city":
        cons = [kwh100(m, 25.0, m["city_sg"], p["ac"] == "on", amb)]
        speeds = [25.0]
    else:
        c = m["base"] - (m["ac_base"] if p["ac"] == "off" else 0.0)
        if amb is not None and p["ac"] == "on":
            c += (max(0, m["ac"][0] + m["ac"][1] * amb) - max(0, m["ac"][0] + m["ac"][1] * m["amb"])) / 36 * 100
        cons = [c]
    if cold:                                # colder than any recorded drive: never more optimistic than usual
        cons = [max(c, m["base"]) for c in cons]
    for i, c in enumerate(cons):
        f(f"cons{i}", r1(c), "kWh/100km", "model")
    if speeds:
        for i, s in enumerate(speeds):
            f(f"spd{i}", r1(s))
    if amb is not None:
        f("amb", r1(amb))
    ac_txt = "with the AC off" if p["ac"] == "off" else "with the AC on"
    usable_now = soc / 100 * cap

    # ---- how slow must I drive to make it
    if p["distance"] and re.search(r"how (slow|fast) (do|should|must|can) i", ql):
        d = p["distance"]
        avail = (soc - reserve) / 100 * cap
        best = None
        for v in range(130, 29, -5):
            if d * kwh100(m, v, 2.0 if v >= 60 else m["city_sg"], p["ac"] == "on", amb) / 100 <= avail:
                best = v
                break
        f("dist", r1(d)); f("avail", r1(avail))
        if best == 130:
            f("vmax", best)
            a.what = (f"You can cover {r1(d)} km" + (f" ({r1(d / MILE)} miles)" if unit_mi else "") + f" with {r1(soc)}% at "
                      f"any normal speed, even averaging {best} km/h, and still keep {r1(reserve)}% in reserve.")
            if unit_mi:
                f("dist_mi", r1(d / MILE))
        elif best:
            f("vmax", best)
            a.what = (f"To cover {r1(d)} km with {r1(soc)}% and keep {r1(reserve)}% in reserve, keep your average speed "
                      f"at or below about {best} km/h" + (f" (about {round(best / MILE)} mph)" if unit_mi else "") + ".")
            if unit_mi:
                f("vmax_mph", round(best / MILE))
        else:
            a.what = f"Even at 30 km/h, {r1(d)} km needs more than the {r1(avail)} kWh you can use: charge first."
            f("v30", 30)
        a.why.append("Switch the AC off or to eco to stretch it further.")
        a.plan["scenario"] = "speed_cap"
        return a

    # ---- trip feasibility
    if p["distance"]:
        d = p["distance"]
        if unit_mi:
            f("dist_mi", r1(d / MILE))
        if p.get("round"):
            f("one_way", r1(d / 2))
        need = d * cons[0] / 100
        pct = need / cap * 100
        arrive = soc - pct
        f("dist", r1(d)); f("need", r1(need)); f("pct", r1(pct)); f("arrive", r1(arrive))
        how = (f" at {r1(speeds[0])} km/h" if speeds else "") + f" {ac_txt}"
        if arrive >= 20 and pct + reserve <= 95:
            verdict = f"Yes, comfortably. You would arrive with about {r1(arrive)}%."
        elif arrive >= reserve and pct + reserve <= 95:
            verdict = f"Yes, but it's tight: you would arrive with about {r1(arrive)}%."
        elif pct + reserve > 95:
            full_km = (95 - reserve) / 100 * cap / (cons[0] / 100)
            short = max(0.0, (pct + reserve - soc) / 100 * cap)
            pw = m["power"].get("DC_FAST", 0) or m["power"].get("AC_HOME", 7)
            f("full_km", round(full_km)); f("stop_kwh", r1(short)); f("stop_min", round(short / pw * 60)); f("lim", 95)
            verdict = (f"Not on one charge: even a full battery covers only about {round(full_km)} km this way. Plan one "
                       f"fast-charging stop of about {round(short / pw * 60)} min ({r1(short)} kWh) on the way.")
        else:
            add = reserve + 5 - arrive
            f("add", r1(add)); f("start", r1(min(100, soc + add)))
            pw = m["power"].get("DC_FAST", 0) or m["power"].get("AC_HOME", 7)
            f("addmin", round(add / 100 * cap / pw * 60))
            short_km = max(0.0, -arrive / 100 * cap / (cons[0] / 100))
            f("short_km", round(short_km))
            verdict = ((f"No: you would run out about {round(short_km)} km short. " if arrive < 0 else
                        f"Not safely: you would arrive with about {r1(arrive)}%. ") + "Charge to at least "
                       f"{r1(min(100, soc + add))}% first (about {round(add / 100 * cap / pw * 60)} min on a fast charger)."
                       + (" Drive gently with the AC off until then." if arrive < 0 else ""))
        a.what = (f"{r1(d)} km" + (f" ({r1(d / MILE)} miles)" if unit_mi else "") + (" there and back" if p.get("round") else "")
                  + f"{how} needs about {r1(need)} kWh ({r1(pct)}% of the battery); you have {r1(soc)}%. {verdict}")
        a.why.append(f"Estimated consumption {r1(cons[0])} kWh/100 km, from a model fitted on {m['n']} of your trips "
                     f"(explains {round(m['r2'] * 100)}% of trip-to-trip variation). Keep at least {r1(reserve)}% in reserve.")
        if p["ac"] == "on" and m["ac_base"] > 0.5:
            gain = d * min(m["ac_base"], cons[0]) / 100 / cap * 100
            f("ac_gain", r1(gain))
            a.why.append(f"Turning the AC off would save about {r1(gain)}% of battery on this trip.")
        a.plan["scenario"] = "trip"
        return a

    # ---- range under the scenario (incl. speed comparisons)
    ranges = [usable_now / (c / 100) for c in cons]
    full = [cap / (c / 100) for c in cons]
    base_range = usable_now / (m["base"] / 100)
    for i, (r_, fr) in enumerate(zip(ranges, full)):
        f(f"rng{i}", round(r_)); f(f"full{i}", round(fr))
    f("base_rng", round(base_range))
    if len(ranges) >= 2:
        a.what = (f"At {r1(speeds[0])} km/h you'd use {r1(cons[0])} kWh/100 km (about {round(ranges[0])} km of range left), "
                  f"at {r1(speeds[1])} km/h {r1(cons[1])} kWh/100 km (about {round(ranges[1])} km), {ac_txt}.")
        diff = (cons[1] - cons[0]) / cons[0] * 100
        f("diff", r1(abs(diff)))
        a.why.append(f"That is {r1(abs(diff))}% {'more' if diff > 0 else 'less'} energy per km at the higher speed: air drag "
                     f"grows with the square of speed.")
    elif not speeds and p["ac"] == "on" and amb is None:
        disp, _ = latest(ctx, "range")
        if disp is not None:
            f("disp", r1(disp))
        a.what = (f"With {r1(soc)}% charge you have about {round(base_range)} km at your usual driving "
                  f"(about {round(full[0])} km on a full battery)" + (f"; the car's own estimate shows {r1(disp)} km." if disp else "."))
        if p.get("place"):
            f("example_km", 150)
            a.why.append(f"I don't know how far \"{p['place']}\" is: VehicleGPT has no maps and works offline. Tell me the "
                         f"distance, e.g. \"can I drive 150 km?\", and I'll check the trip.")
    else:
        what = ((f"at {r1(speeds[0])} km/h " + (f"({round(speeds[0] / MILE)} mph) " if unit_mi else "")) if speeds else "") + ac_txt
        if speeds and unit_mi:
            f("spd_mph", round(speeds[0] / MILE)); f("rng_mi", round(ranges[0] / MILE))
        a.what = (f"With {r1(soc)}% charge, {what} you'd have about {round(ranges[0])} km of range "
                  + (f"({round(ranges[0] / MILE)} miles) " if unit_mi else "")
                  + f"(about {round(full[0])} km on a full battery); with your usual driving it is about {round(base_range)} km.")
        if p["ac"] == "off":
            f("gain_km", round(ranges[0] - base_range))
            a.why.append(f"Switching the AC off adds about {round(ranges[0] - base_range)} km at today's charge.")
    a.why.append(f"Based on a consumption model fitted on {m['n']} of your trips (R² {round(m['r2'], 2)}).")
    a.plan["scenario"] = "range"
    return a


# ------------------------------------------------------------------ quantified advice
def improve(ctx, q: str, plan: dict) -> Answer:
    a = Answer(q, "improve")
    f = a.fact
    m = model(ctx)
    cap = m["cap"]
    base_full = cap / (m["base"] / 100)
    f("base_full", round(base_full)); f("base", r1(m["base"])); f("n_model", m["n"])
    tips = []
    # AC: halve AC energy (eco set point / pre-cooling)
    no_ac = cap / ((m["base"] - m["ac_base"] / 2) / 100) - base_full
    f("g_ac", round(no_ac)); f("ac_base", r1(m["ac_base"])); f("adv_lo", 1); f("adv_hi", 2)
    tips.append((no_ac, f"Use the AC more gently (pre-cool while plugged in, 1–2 °C warmer set point, recirculation): halving "
                        f"its {r1(m['ac_base'])} kWh/100 km adds about {round(no_ac)} km per full charge."))
    # highway speed -10 km/h
    hw = m["hw_speed"]
    c_hi, c_lo = kwh100(m, hw, 2.0), kwh100(m, hw - 10, 2.0)
    save = (c_hi - c_lo) / c_hi * 100
    f("hw", r1(hw)); f("hw_lo", r1(hw - 10)); f("g_hw", r1(save))
    tips.append((save * base_full / 100 * 0.4, f"On the highway, average {r1(hw - 10)} instead of {r1(hw)} km/h: {r1(save)}% "
                                               f"less energy per highway km."))
    # smoother traffic driving
    c_sg, c_sg2 = kwh100(m, 25, m["city_sg"]), kwh100(m, 25, max(0, m["city_sg"] - 5))
    g_sg = (c_sg - c_sg2) / c_sg * 100
    f("g_sg", r1(g_sg)); f("sg_pts", 5)
    tips.append((g_sg * base_full / 100 * 0.6, f"In traffic, anticipate and coast instead of stop-start: 5 points less "
                                               f"stopped time saves about {r1(g_sg)}% in the city."))
    # tyres now
    lows = [(k, latest(ctx, k)[0]) for k in TYRES]
    low = [(k, v) for k, v in lows if v is not None and v < 225]
    if low:
        tips.append((99, "Inflate your tyres now: " + ", ".join(f"{CATALOGUE[k][1]} {r1(v)} kPa" for k, v in low)
                         + ". Low tyres were your biggest single efficiency loss in October."))
    titles = {0: ("Gentler AC", f"+{round(no_ac)} km per charge"), 1: ("Slower highway cruising", f"{r1(save)}% less per highway km"),
              2: ("Smoother in traffic", f"{r1(g_sg)}% less in the city")}
    cards = [(tips[i][0], *titles[i]) for i in range(3)]
    if low:
        cards.append((99, "Inflate tyres now", ", ".join(f"{r1(v)} kPa" for _, v in low)))
    cards.sort(key=lambda t: -t[0])
    a.plan["tips"] = [{"title": t, "value": v} for _, t, v in cards] + [{"title": "Battery care", "value": "charge to 80% daily"}]
    tips.sort(key=lambda t: -t[0])
    a.what = (f"Your usual driving gives about {round(base_full)} km per full charge ({r1(m['base'])} kWh/100 km). "
              f"Biggest gains first:")
    a.why = [t for _, t in tips]
    a.why.append("Battery care (general guidance): for daily driving, charge to about 80% and keep fast charging for trips, "
                 "especially when the battery is already hot.")
    f("care_pct", 80)
    a.evidence.append(f"consumption model fitted on {m['n']} of your trips; current tyre pressures")
    a.next = a.why[0]
    return a


# ------------------------------------------------------------------ safety log
SEV = {"WARNING_ON": "high", "DTC_SET": "high", "ESC": "high", "HARSH": "medium", "ANOMALY": "medium",
       "ADAS_DISENGAGEMENT": "medium", "DATA_GAP": "low"}
NICE = {"TYRE_PRESSURE_LOW": "tyre pressure dropping", "COOLANT_PUMP_LOW": "coolant pump running slow",
        "BATTERY_TEMP_ABOVE_CONTEXT": "battery hotter than normal", "LV_LOW_AT_WAKE": "12 V battery low at start-up"}


def _episodes(minutes, gap=3):
    out = []
    for mnt in minutes:
        if out and mnt - out[-1][1] <= gap:
            out[-1][1] = mnt
        else:
            out.append([mnt, mnt])
    return out


def safety_log(ctx, q: str, plan: dict) -> Answer:
    from .playbooks import WARN_NAMES
    a = Answer(q, "safety_log")
    f = a.fact
    w = ctx.window(q, plan) or ctx.all_history()
    rows = []
    for e in ctx.st.events(["WARNING_ON", "DTC_SET", "ANOMALY", "ADAS_DISENGAGEMENT", "DATA_GAP"], w.start, w.end):
        if e["type"] == "WARNING_ON":
            txt = f"warning: {WARN_NAMES.get(e['code'], e['code'])}"
        elif e["type"] == "DTC_SET":
            txt = f"fault {e['code']}: {dtc_desc(ctx, e['code']).lower()}"
        elif e["type"] == "ANOMALY":
            txt = f"early sign: {NICE.get(e['code'], e['code'].lower())}"
        elif e["type"] == "ADAS_DISENGAGEMENT":
            txt = f"lane assist switched off ({(e.get('reason') or '').replace('_', ' ').lower()})"
        else:
            txt = "recording gap"
        rows.append((e["ts"], SEV[e["type"]], txt, e))
        if e.get("code"):
            f(f"sc{e['id']}", e["code"])
    esc = ctx.st.rollup(CATALOGUE["esc"][0], w.start, w.end)
    for s, e_ in _episodes(list(esc["minute"][esc["max"] > 0])):
        rows.append((s * 60, "high", "stability control (ESC) intervened", None))
    acc = ctx.st.rollup(V + "Acceleration.Longitudinal", w.start, w.end)
    for s, e_ in _episodes(list(acc["minute"][acc["min"] < -4.0])):
        rows.append((s * 60, "medium", "very hard braking (over 4 m/s²)", None))
    rows.sort(key=lambda r: r[0], reverse=True)
    n = {k: sum(1 for r in rows if r[1] == k) for k in ("high", "medium", "low")}
    for k, v in n.items():
        f(f"n_{k}", v)
    f("n_all", len(rows)); f("hb_thr", 4.0); f("w", w.fmt())
    act = active_dtcs(ctx)
    for e in act:
        f(f"act{e['id']}", e["code"])
    a.evidence += [f"events table: warnings, faults, early signs, lane-assist events in {w.fmt()}",
                   "COLD tier: ESC interventions and strongest braking per minute"]
    if not rows:
        a.what = f"No safety-relevant events {phrase(w)}."
        return a
    a.what = (f"Safety log {phrase(w)}: {len(rows)} events, {n['high']} high, {n['medium']} medium and {n['low']} low "
              f"importance. " + ("Still open: " + ", ".join(e["code"] for e in act) + "." if act else "Nothing is open now."))
    for ts, sev, txt, e in rows[:14]:
        f(f"sl{int(ts)}", fmt_time(ts))
        a.details.append(f"{fmt_time(ts)} [{sev}] {txt}")
    high = [r for r in rows if r[1] == "high"][:3]
    a.why = [f"{fmt_time(ts)}: {txt}" for ts, _, txt, _ in high]
    a.next = ("Have the open fault checked: " + ", ".join(f"{e['code']} ({dtc_desc(ctx, e['code']).lower()})" for e in act) + "."
              if act else "No action needed.")
    a.plan["log"] = [{"t": fmt_time(ts), "sev": sev, "text": txt} for ts, sev, txt, _ in rows[:30]]
    return a


# ------------------------------------------------------------------ emergency guide
@lru_cache(maxsize=1)
def guide():
    return yaml.safe_load(open(HERE / "emergency.yaml", encoding="utf-8"))


EMERG = [
    ("brake_failure", r"brakes? (fail\w*|(are |is )?not working|aren'?t working|don'?t work|gone|went out|stopped working|feels? spongy|not responding)|no brakes|brake failure|can'?t stop the car"),
    ("stuck_accelerator", r"accelerat\w* (is )?(stuck|jammed)|pedal (is )?stuck|(won'?t|can'?t|doesn'?t) stop accelerating|unintended acceleration|speeding up (by|on) its own|keeps accelerating"),
    ("fire", r"\bon fire\b|\bfire\b|smoke|burning smell|smell\w* (of )?burning|flames?"),
    ("overheat", r"batter\w* (is )?(overheating|too hot|getting hot)|overheat\w* (warning|now|while driving)|temperature warning|battery (temp|heat) (warning|light)"),
    ("blowout", r"(tyre|tire)s? (\w+ )?(burst|blew|blown|blow ?out|exploded|popped)|blow ?out|burst (tyre|tire)|flat (tyre|tire) while driving"),
    ("flood", r"flood|water.?logged|deep water|stalled in water|submerged"),
    ("crash", r"\baccident\b|\bcrash(ed)?\b(?! ahead)|collision|collided|rear.?ended|hit (a|an|another|something|someone)"),
    ("drowsy", r"sleepy|drowsy|falling asleep|can'?t keep my eyes|dozing|so tired|exhausted"),
    ("power_loss", r"(lost|losing|no|loss of) power|car (is )?slowing down (by|on) itself|limp mode|turtle mode"),
]
HELP = r"\bjust\b|i have a|i'?ve got a|we have a|feeling|i feel|\bis stuck\b|(are|is) not working|is on fire|!|what (should|do|can|must) (i|we)|what to do|how (to|do i|should i|can i) (handle|deal|stop|react)|help|emergency|sos|\bnow\b|right now|suddenly|i('m| am)|my (brakes?|car|tyre|tire|accelerator|battery)|while (i'?m )?driving|is on fire|are on fire|if (my|the)"
HISTORY = r"\b(why did|when did|was there|were there|did (the|my) .* (ever|last)|how many|history|last (week|month)|yesterday|on \d{1,2} (oct|nov))\b"


def emergency_topic(q: str) -> str | None:
    ql = q.lower()
    if re.search(HISTORY, ql) or not re.search(HELP, ql):
        return None
    for topic, pat in EMERG:
        if re.search(pat, ql):
            return topic
    if re.search(r"\b(emergency|sos)\b(?! (room|braking|brake|door|services|vehicle|lane|stop))", ql) and not re.search(
            r"make it to|route|navigate|elevation|station|reroute", ql):
        return "general"
    return None


def emergency(ctx, q: str, plan: dict) -> Answer:
    a = Answer(q, "emergency")
    f = a.fact
    g = guide()
    topic = plan.get("topic") or emergency_topic(q) or "general"
    for k, v in g["numbers"].items():
        f(f"num_{k}", v, "", "Indian emergency numbers")
    if topic == "general":
        a.what = ("Tell me what is happening: brakes, fire or smoke, battery warning, tyre burst, stuck accelerator, "
                  "flood water, a crash or loss of power. In any danger, stop safely and call 112.")
        a.plan["emergency"] = {"title": "Emergency", "steps": [], "topics": [t["title"] for t in g["topics"].values()]}
        return a
    t = g["topics"][topic]
    for i, s in enumerate(t["steps"], 1):
        f(f"step{i}", i)
        for n in re.findall(r"\d+(?:\.\d+)?", s):
            f("guide_n", float(n), "", "emergency guide (cited source)")
    a.what = f"{t['title']}: do this now. Call 112 if anyone is in danger."
    a.details = [f"{i}. {s}" for i, s in enumerate(t["steps"], 1)]
    # live state, so the guidance fits the car right now
    live = []
    for k in t.get("context", []):
        v, ts = latest(ctx, k, look_h=1)
        if v is None:
            continue
        f(k, r1(v) if CATALOGUE[k][4] is None else label(k, v))
        live.append(f"{CATALOGUE[k][1]} {label(k, v)}")
    if live:
        a.why.append("Your car right now: " + "; ".join(live) + ".")
    act = active_dtcs(ctx)
    rel = {"overheat": ("P1A20", "P1A21"), "fire": ("P1A20", "P1A21"), "power_loss": ("P1A20", "P1A21"),
           "brake_failure": (), "blowout": ("C1A10",)}.get(topic, ())
    for e in act:
        if e["code"] in rel:
            f(f"rdtc{e['id']}", e["code"])
            a.why.append(f"Related open fault: {e['code']} ({dtc_desc(ctx, e['code']).lower()}).")
    if topic == "brake_failure":
        a.why.append("No brake fault code is stored on this car. VehicleGPT can read the brake pedal and ABS signals "
                     "but cannot test or control the brakes.")
    a.why.append("VehicleGPT never operates the car: these steps are for you. Your owner's manual takes precedence.")
    a.evidence += [f"emergency guide '{topic}' (fixed text, not generated)"] + [f"source: {s}" for s in t["sources"]]
    a.next = "When you are safe, ask me \"what happened before the warning?\" to find the cause."
    a.plan["emergency"] = {"title": t["title"], "steps": t["steps"], "sources": t["sources"]}
    return a


CAPS = r"what (do|can) you (record|know|answer|do|track|tell)|what (data|signals) (do you|does the car|are)|what can i ask|help me with|what are you"


def capabilities(ctx, q: str, plan: dict) -> Answer:
    a = Answer(q, "capabilities")
    a.fact("n_sig", len(CATALOGUE))
    for n in (8, 12, 150, 80, 110):
        a.fact("example", n, "", "example values in the help text")
    a.what = (f"I can answer from your car's own recorded data ({len(CATALOGUE)} named signals, 8 weeks of history), "
              "fully offline. Try any of these:")
    a.why = ["Right now: tyre pressures, battery charge and health, temperatures, gear, camera, 12 V battery, AC.",
             "History: why range dropped, efficiency changes, charging, trips, warnings and faults, lane assist, rain.",
             "What if: can I drive 150 km, range with the AC off, 80 vs 110 km/h, time to charge to 80%.",
             "Advice: how to improve range, repairs, driving habits. Safety: safety log and emergency steps.",
             "I never control the car, and I say so when something isn't recorded."]
    return a


PLAYBOOKS_LIVE = {"capabilities": capabilities, "vehicle_status_now": vehicle_status_now, "what_if": what_if, "improve": improve,
                  "safety_log": safety_log, "emergency": emergency}
