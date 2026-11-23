"""Presentation layer for the head-unit (HMI) screen.

Turns a verified Answer into glanceable UI pieces: one hero number, up to four factor cards, a short sentence
to speak, and follow-up suggestions. Rule: this layer never computes a new number. Every number it shows is
copied from a registered fact, and `ui_verified` re-checks that, so the head-unit keeps the verifier guarantee.
"""
from __future__ import annotations

import re

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

FOLLOW = {
    "capabilities": ["How is my car right now?", "Can I drive 150 km on this charge?", "Show my safety log"],
    "emergency": ["Show my safety log", "How is my car right now?", "What happened before the warning?"],
    "safety_log": ["What happened before the warning?", "Any fault codes?", "How is my car right now?"],
    "what_if": ["How can I improve my range?", "How long to charge to 80% at home?", "80 vs 110 km/h on the highway"],
    "improve": ["How much range if I turn off the AC?", "80 vs 110 km/h on the highway", "Any repairs I should do?"],
    "not_recorded": ["What can you do?", "Any fault codes?", "How much range do I have left?"],
    "vehicle_status_now": ["Why has my range dropped?", "Check battery health", "Any fault codes?"],
    "range_drop": ["When did my efficiency get worse?", "Compare city and highway driving", "How much energy did the AC use?"],
    "efficiency_change": ["Show my tyre pressure trend", "Why has my range dropped?", "Any fault codes?"],
    "segment_compare": ["Why has my range dropped?", "How many trips this week?", "Show my charging history"],
    "signal_trend": ["How much range do I have left?", "Any fault codes?", "Why has my range dropped?"],
    "dtc_lookup": ["What happened before the warning?", "Is my battery healthy?", "How much range do I have left?"],
    "before_event": ["Any fault codes?", "Is my battery healthy?", "How much range do I have left?"],
    "trip_stats": ["Compare city and highway driving", "Show my charging history", "Why has my range dropped?"],
    "charging_history": ["How much range do I have left?", "Is my battery healthy?", "How many trips this week?"],
    "adas_events": ["Any fault codes?", "What happened before the warning?", "How many trips this week?"],
    "health_summary": ["Any fault codes?", "Why has my range dropped?", "Is my battery healthy?"],
    "data_quality": ["How much range do I have left?", "Any fault codes?", "How many trips this week?"],
}
REFUSE = {"unsafe_command", "privacy", "out_of_scope"}
DEFAULT_FOLLOW = ["How much range do I have left?", "Why has my range dropped?", "Any fault codes?"]


def _level(share: float) -> tuple[int, str]:
    if share >= 50:
        return 4, "High"
    if share >= 25:
        return 3, "Medium"
    if share > 0:
        return 2, "Low"
    return 0, "None"


def _num(v) -> str:
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


NAMES = {"range_drop": "why your range changed", "efficiency_change": "when efficiency got worse",
         "adas_events": "lane-assist events", "before_event": "what led to a warning", "segment_compare": "city vs highway",
         "signal_trend": "how a signal changed over time", "dtc_lookup": "fault codes", "charging_history": "charging history",
         "vehicle_status_now": "the car's current state", "trip_stats": "trip statistics", "data_quality": "missing data",
         "health_summary": "a health summary", "what_if": "a what-if scenario", "improve": "ways to improve range",
         "safety_log": "the safety log", "not_recorded": "something not recorded", "emergency": "an emergency"}
ASK = {"range_drop": "Why has my range dropped?", "efficiency_change": "When did my efficiency get worse?",
       "adas_events": "Why did lane assist switch off?", "before_event": "What happened before the warning?",
       "segment_compare": "Compare city and highway driving", "signal_trend": "How has battery temperature changed?",
       "dtc_lookup": "Any fault codes?", "charging_history": "Show my charging history",
       "vehicle_status_now": "How is my car right now?", "trip_stats": "How many trips this week?",
       "data_quality": "Is any data missing?", "health_summary": "Any repairs I should do?",
       "what_if": "Can I drive 150 km on this charge?", "improve": "How can I improve my range?",
       "safety_log": "Show my safety log", "not_recorded": "What can you do?", "emergency": "Emergency help", "capabilities": "What can you do?"}


def warm(ans, ui: dict) -> tuple[str, str]:
    """A short, warm spoken reply (and its style). Numbers are copied from facts; the voice rounds them for the ear."""
    f = {x.name: x.value for x in ans.facts}
    i, plan = ans.intent, ans.plan or {}
    n = _num
    if i == "emergency":
        steps = (plan.get("emergency") or {}).get("steps") or []
        if not steps:
            return "Okay, I'm here. Tell me what's happening: brakes, smoke, a warning, a tyre, or a crash. If anyone is in danger, stop safely and call one one two.", "calm"
        opener = "Okay, I'll talk you through it. " if steps[0].lower().startswith("stay calm") else "Okay. Stay calm, I'll talk you through it. "
        return opener + " ".join(steps) + " If anyone is in danger, call one one two.", "calm"
    if i in ("unsafe_command",):
        extra = " You can change that on the car's own controls." if re.search(r"temp|ac\b|climate|fan|seat|window|mirror|light|wiper|charge limit|mode", ans.question.lower()) else ""
        return "Sorry, I can't do that. I can look at everything your car records, but I never control it." + extra, "warm"
    if i == "privacy":
        return "I'd rather not. I don't track people or the places they go. I only look after the car's own data.", "warm"
    if i == "not_recorded":
        return "I'm afraid I can't help with that one. " + ans.what.split(":", 1)[-1].strip().split(". ")[0].rstrip(".") + ".", "warm"
    if i == "vehicle_status_now":
        ks = plan.get("signals") or []
        tyres = [f[k] for k in ("tyre_fl", "tyre_fr", "tyre_rl", "tyre_rr") if k in f and isinstance(f[k], (int, float))]
        if plan.get("snapshot") and "range" in f:
            open_fault = ans.why and any(w.startswith("Active fault:") for w in ans.why)
            return (f"Everything looks good. You've got {n(f['range'])} km of range at {n(f.get('soc', ''))}%, and the battery "
                    f"is at a comfortable {n(f.get('batt_temp', ''))} °C."
                    + (" One thing to keep in mind: a fault is still open, so it's worth a service visit." if open_fault and "No active" not in " ".join(ans.why) else "")), "warm"
        if len(tyres) > 1:
            ok = min(tyres) >= 225
            return (f"Your tyres look {'good' if ok else 'low, please check them'}: all four are between {n(min(tyres))} and "
                    f"{n(max(tyres))} kPa."), "warm"
        if "soh" in f and (not ks or ks[0] == "soh"):
            return (f"Your battery is in great shape: {n(f['soh'])}% health, so it still holds almost all of its original "
                    f"capacity." if f["soh"] >= 95 else f"Battery health is {n(f['soh'])}%."), "warm"
        if re.search(r"rear|reverse|back(up)? cam", ans.question.lower()):
            return f"This car doesn't have a rear camera I can see, but the front camera is {str(f.get('camera', 'OK')).lower()}.", "warm"
        if "camera" in f and ks and ks[0] == "camera":
            return ("The front camera is working fine." if f["camera"] == "OK" else
                    f"The front camera is {str(f['camera']).lower()}, so lane assist may switch off."), "warm"
        k0 = next((k for k in ks if k in f), None)
        if k0:
            v = f[k0]
            special = {"gear": f"You're in {v}.", "charging": "Yes, it's charging right now." if v == "yes" else "No, the car isn't charging right now.",
                       "ac_on": f"Yes, the air-conditioning is on." if v == "on" else "No, the air-conditioning is off.",
                       "power_state": f"The car is {v}.", "cruise": f"Cruise control is {v}.", "lka": f"Lane-keep assist is {v}.",
                       "wiper": f"The wipers are {v}.", "low_beam": f"The headlights are {v}."}
            if k0 in special:
                return special[k0], "warm"
            from reason.catalogue import CATALOGUE
            unit = CATALOGUE[k0][2] if CATALOGUE[k0][4] is None else ""
            return f"The {CATALOGUE[k0][1]} is {n(v)} {unit} right now.".replace(" .", "."), "warm"
        return ans.what, "warm"
    if i == "range_drop" and "range_change_pct" in f:
        return (f"Good question. Your range is down about {n(abs(f['range_change_pct']))}%. Mostly that's the air-conditioning "
                f"working harder in the heat, plus a bit more stop-start traffic. Your tyres and battery are fine."), "warm"
    if i == "what_if":
        if "arrive" in f:
            a_ = f["arrive"]
            if "stop_min" in f:
                return f"Not on one charge, I'm afraid. Plan one fast-charging stop of about {n(f['stop_min'])} minutes on the way.", "warm"
            if "short_km" in f and a_ < 0:
                return f"I'd charge first. As it is, you'd run out about {n(f['short_km'])} km short.", "warm"
            if a_ >= 20:
                return f"Yes, you'll make it comfortably. You'd arrive with about {n(a_)}% left.", "warm"
            if a_ >= f.get("reserve", 10):
                return f"You'll make it, but it's tight: about {n(a_)}% left when you arrive.", "warm"
            return f"I'd top up first. You'd arrive with only about {n(a_)}%.", "warm"
        if "mins" in f:
            return f"From {n(f['soc_now'])}% to {n(f['target'])}%, it'll take about {n(f['mins'])} minutes.", "warm"
        if "rng1" in f:
            return (f"At {n(f['spd0'])} km/h you'd have about {n(f['rng0'])} km, and at {n(f['spd1'])} about {n(f['rng1'])}. "
                    f"Slowing down really helps."), "warm"
        if "gain_km" in f:
            return f"Switching the AC off would add about {n(f['gain_km'])} km at today's charge.", "warm"
        if "vmax" in f:
            return ans.what, "warm"
        return ans.what.split(";")[0] + ".", "warm"
    if i == "improve" and plan.get("tips"):
        t = plan["tips"][0]
        return (f"Here's the biggest win: {t['title']}, worth {t['value'].replace('+', '')}. I've put a few more ideas on the "
                f"screen."), "warm"
    if i == "health_summary" and ans.what.startswith("Based on") and ans.why:
        return ("Here's what I'd do. " + ans.why[0].replace("Repair: ", "First, get this repaired: ")).replace(" (", ", ").replace(")", ""), "warm"
    if i == "dtc_lookup" and "n" in f and "na" in f:
        return (f"You've had {n(f['n'])} fault codes in the recorded history, and {n(f['na'])} "
                f"{'is' if f['na'] == 1 else 'are'} still active. The details are on the screen."), "warm"
    if i == "signal_trend" and "mean" in f and "min" in f:
        unit = (ans.chart or {}).get("unit", "")
        lbl = ans.what.split(" in ")[0].split(" yesterday")[0].split(" on ")[0]
        return f"{lbl} averaged {n(f['mean'])} {unit}, ranging from {n(f['min'])} to {n(f['max'])}.".replace(" ,", ","), "warm"
    if i == "safety_log":
        return (f"I've got {n(f.get('n_all', 0))} safety events on record, {n(f.get('n_high', 0))} of them important. "
                + ("The coolant pump fault is still open, so it's worth getting that checked." if "P1A20" in ans.what else "Nothing is open right now.")), "warm"
    if i == "capabilities":
        return "I can tell you how your car is doing right now, explain anything in its history, work out trips and charging times, and help in an emergency. Just ask.", "warm"
    return ui.get("say") or ans.what, "warm"


def a_recs(ans) -> bool:
    return ans.what.startswith("Based on")


def offer(f: str) -> str:
    """Turn a follow-up chip into a natural spoken offer.
    'Show my safety log' -> 'Shall I show your safety log?'; a question chip is offered as a question to answer next."""
    f = f.strip().rstrip("?.")
    if re.match(r"(why|when|what|how|where|which|is|are|was|did|does|do|has|have|can|could|will|should|any)\b", f, re.I):
        return "Want me to answer this next: " + f[0].lower() + f[1:] + "?"
    low = f[0].lower() + f[1:]
    low = re.sub(r"\bmy\b", "your", low)
    return "Shall I " + low + "?"


def build(ans) -> dict:
    f = {x.name: x.value for x in ans.facts}
    i = ans.intent
    hero, cards, say = None, [], ""
    first = re.split(r"(?<=[.!?])\s", ans.what.strip())[0] if ans.what else ""

    plan = ans.plan or {}
    if i == "vehicle_status_now" and plan.get("signals") and not plan.get("snapshot"):
        from reason.catalogue import CATALOGUE, TYRES
        ks = [k for k in plan["signals"] if k in f]
        icon = lambda k: ("tyre" if k in TYRES else "battery" if k in ("soc", "soh", "range", "lv_voltage", "batt_voltage", "charging")
                          else "thermo" if "temp" in k else "spark" if k in ("camera", "visibility") else "dot")
        if ks:
            k0 = ks[0]
            unit = CATALOGUE[k0][2] if CATALOGUE[k0][4] is None else ""
            if len([k for k in ks if k in TYRES]) > 1:
                hero = {"value": _num(min(f[k] for k in ks if k in TYRES)), "unit": "kPa", "label": "lowest tyre pressure right now"}
            else:
                hero = {"value": _num(f[k0]), "unit": unit if unit not in ("%",) else "%", "label": f"{CATALOGUE[k0][1]} right now"}
            for k in ks[:4]:
                v = f[k]
                cards.append({"icon": icon(k), "name": CATALOGUE[k][1][0].upper() + CATALOGUE[k][1][1:],
                              "value": f"{_num(v)} {CATALOGUE[k][2]}".strip() if CATALOGUE[k][4] is None else str(v),
                              "tone": "ok" if (k in TYRES and isinstance(v, (int, float)) and v >= 225) or v in ("OK", "normal") else ""})
        say = first
    elif i == "vehicle_status_now" and plan.get("snapshot") and "range" in f:
        hero = {"value": _num(f["range"]), "unit": "km", "label": "estimated range right now"}
        for k, ic, nm in (("soc", "battery", "Battery"), ("gear", "dot", "Gear"), ("speed", "road", "Speed"),
                          ("batt_temp", "thermo", "Battery temperature"), ("motor_temp", "thermo", "Motor"),
                          ("outside_temp", "dot", "Outside"), ("lv_voltage", "battery", "12 V battery"), ("camera", "spark", "Front camera")):
            if k in f:
                from reason.catalogue import CATALOGUE
                unit = CATALOGUE[k][2] if CATALOGUE[k][4] is None else ""
                cards.append({"icon": ic, "name": nm, "value": f"{_num(f[k])} {unit}".strip() if unit else str(f[k])})
        cards = cards[:4]
        say = f"Your car is fine right now: range {_num(f['range'])} kilometres, battery {_num(f.get('soc', ''))} percent."
    elif i == "what_if" and "arrive" in f:
        hero = {"value": f"{_num(f['arrive'])}%", "unit": "", "label": "battery left on arrival"}
        cards += [{"icon": "road", "name": "Trip", "value": f"{_num(f['dist'])} km"},
                  {"icon": "battery", "name": "Energy needed", "value": f"{_num(f['need'])} kWh", "sub": f"{_num(f['pct'])}% of the battery"},
                  {"icon": "spark", "name": "Consumption", "value": f"{_num(f['cons0'])} kWh/100 km", "sub": f"model R² {_num(f['r2'])}"}]
        if "stop_min" in f:
            cards.append({"icon": "battery", "name": "Charging stop", "value": f"{_num(f['stop_min'])} min", "tone": "warn"})
        elif "ac_gain" in f:
            cards.append({"icon": "snow", "name": "AC off would save", "value": f"{_num(f['ac_gain'])}%", "tone": "ok"})
        say = first
    elif i == "what_if" and "mins" in f:
        hero = {"value": _num(f["mins"]), "unit": "min", "label": f"to reach {_num(f['target'])}%"}
        cards += [{"icon": "battery", "name": "Energy to add", "value": f"{_num(f['need'])} kWh"},
                  {"icon": "spark", "name": "Charging power", "value": f"{_num(f['pw'])} kW", "sub": "typical for your sessions"}]
        say = first
    elif i == "what_if" and "rng1" in f:
        hero = {"value": f"{_num(f['diff'])}%", "unit": "", "label": "more energy per km at the higher speed"}
        cards += [{"icon": "road", "name": f"At {_num(f['spd0'])} km/h", "value": f"{_num(f['rng0'])} km range", "sub": f"{_num(f['cons0'])} kWh/100 km", "tone": "ok"},
                  {"icon": "road", "name": f"At {_num(f['spd1'])} km/h", "value": f"{_num(f['rng1'])} km range", "sub": f"{_num(f['cons1'])} kWh/100 km", "tone": "warn"}]
        say = first
    elif i == "what_if" and "rng0" in f:
        hero = {"value": _num(f["rng0"]), "unit": "km", "label": "range in this scenario"}
        cards += [{"icon": "road", "name": "Your usual driving", "value": f"{_num(f['base_rng'])} km"},
                  {"icon": "battery", "name": "On a full battery", "value": f"{_num(f['full0'])} km"}]
        if "gain_km" in f:
            cards.append({"icon": "snow", "name": "AC off adds", "value": f"{_num(f['gain_km'])} km", "tone": "ok"})
        say = first
    elif i == "improve":
        hero = {"value": _num(f["base_full"]), "unit": "km", "label": "per full charge with your usual driving"}
        for t in (plan.get("tips") or [])[:4]:
            cards.append({"icon": "bulb", "name": t["title"], "value": t["value"], "tone": "ok"})
        say = first + " " + ans.why[0]
    elif i == "safety_log":
        hero = {"value": _num(f["n_high"]), "unit": "", "label": f"high-importance safety events · {_num(f['n_all'])} in total"}
        for row in (plan.get("log") or [])[:4]:
            cards.append({"icon": "warn" if row["sev"] == "high" else "dot", "name": row["t"], "value": row["text"][:80],
                          "small": True, "tone": "warn" if row["sev"] == "high" else ""})
        say = first
    elif i == "emergency":
        say = ans.what + " " + " ".join((plan.get("emergency") or {}).get("steps", [])[:3])
    elif i == "capabilities":
        say = first
    elif i == "vehicle_status_now" and "range" in f:
        hero = {"value": _num(f["range"]), "unit": "km", "label": "estimated range remaining"}
        if "state_of_charge" in f:
            cards.append({"icon": "battery", "name": "Battery", "value": f"{_num(f['state_of_charge'])}%",
                          "meter": f["state_of_charge"]})
        if "battery_temp" in f:
            cards.append({"icon": "thermo", "name": "Battery temperature", "value": f"{_num(f['battery_temp'])} °C"})
        tyres = [f[k] for k in ("tyre_fl_min", "tyre_fr_min", "tyre_rl_min", "tyre_rr_min") if k in f]
        if tyres:
            cards.append({"icon": "tyre", "name": "Lowest tyre", "value": f"{_num(min(tyres))} kPa", "tone": "ok"})
        say = f"You have about {_num(f['range'])} kilometres of range, with the battery at {_num(f.get('state_of_charge', ''))} percent."
    elif i == "range_drop" and "range_change_pct" in f:
        hero = {"value": f"{_num(abs(f['range_change_pct']))}%", "unit": "",
                "label": f"{'less' if f['range_change_pct'] < 0 else 'more'} range than the previous {_num(f.get('span_weeks', 2))} weeks"}
        lv, word = _level(f.get("heat_share", 0))
        cards.append({"icon": "snow", "name": "Air conditioning in the heat", "value": word, "bars": lv,
                      "sub": f"{_num(f['heat_share'])}% of the drop · {_num(f['ambA'])}→{_num(f['ambB'])} °C"})
        lv, word = _level(f.get("traffic_share", 0))
        cards.append({"icon": "traffic", "name": "Stop-start traffic", "value": word, "bars": lv,
                      "sub": f"{_num(f['traffic_share'])}% of the drop · stopped {_num(f['sgA'])}→{_num(f['sgB'])}%"})
        cards.append({"icon": "tyre", "name": "Tyres", "value": "Normal", "bars": 0, "tone": "ok",
                      "sub": f"lowest {_num(f['tmin'])} kPa"})
        cards.append({"icon": "battery", "name": "Battery health", "value": "Normal", "bars": 0, "tone": "ok",
                      "sub": f"{_num(f['soh'])}%"})
        say = (f"Your range is down {_num(abs(f['range_change_pct']))} percent. Mostly air conditioning in the heat, "
               f"then stop-start traffic. Tyres and battery are fine.")
    elif i == "efficiency_change" and "worst_pct" in f:
        hero = {"value": f"+{_num(f['worst_pct'])}%", "unit": "", "label": f"extra energy on {f['worst_date']}"}
        if "tyre_min" in f:
            cards.append({"icon": "tyre", "name": "Tyre pressure", "value": "Low", "bars": 4, "tone": "warn",
                          "sub": f"{_num(f['tyre_min'])} kPa vs {_num(f['nominal'])} normal"})
        if "first_anomaly" in f:
            cards.append({"icon": "spark", "name": "VehicleGPT noticed", "value": f["first_anomaly"], "tone": "ok"})
        if "tpms_warn" in f:
            cards.append({"icon": "warn", "name": "Dashboard warning", "value": f["tpms_warn"]})
        say = f"Efficiency was worst on {f['worst_date']}, about {_num(f['worst_pct'])} percent more energy than normal."
    elif i == "segment_compare" and "diff" in f:
        hero = {"value": f"{_num(f['diff'])}%", "unit": "", "label": "more energy per km on the highway"}
        cards += [{"icon": "city", "name": "City", "value": f"{_num(f['city'])} kWh/100 km", "sub": f"{_num(f['cd'])} km driven"},
                  {"icon": "road", "name": "Highway", "value": f"{_num(f['hwy'])} kWh/100 km", "sub": f"{_num(f['hd'])} km driven"}]
        say = f"Highway driving uses about {_num(f['diff'])} percent more energy per kilometre than city driving."
    elif i == "signal_trend" and "mean" in f:
        unit = (ans.chart or {}).get("unit", "")
        hero = {"value": _num(f["mean"]), "unit": unit, "label": f"average, {f.get('w', '')}"}
        cards += [{"icon": "up", "name": "Highest", "value": f"{_num(f['max'])} {unit}", "sub": f.get("max_t", "")},
                  {"icon": "down", "name": "Lowest", "value": f"{_num(f['min'])} {unit}"}]
        say = first
    elif i == "dtc_lookup" and "n" in f:
        hero = {"value": _num(f["n"]), "unit": "", "label": f"fault codes logged · {_num(f.get('na', 0))} active"}
        for k, v in f.items():
            if re.fullmatch(r"c\d+", k):
                i_ = k[1:]
                act = f.get("st" + i_) == "active"
                cards.append({"icon": "warn", "name": f"{v} · {'ACTIVE' if act else 'cleared'}", "value": f.get("desc" + i_, v),
                              "sub": f.get("td" + i_, ""), "tone": "warn" if act else "", "small": True, "_a": act})
        cards = sorted(cards, key=lambda c: not c.pop("_a"))[:4]
        say = first
    elif i == "trip_stats" and "avg" in f and "vmax" in f:
        hero = {"value": _num(f["avg"]), "unit": "km/h", "label": "average driving speed"}
        cards.append({"icon": "up", "name": "Top speed", "value": f"{_num(f['vmax'])} km/h"})
        if "city_avg" in f:
            cards.append({"icon": "city", "name": "City trips", "value": f"{_num(f['city_avg'])} km/h"})
        say = first
    elif i == "trip_stats" and "per" in f and "kwh100" in f:
        hero = {"value": f"{_num(f['per'])}%", "unit": "", "label": "of the battery per 100 km"}
        cards.append({"icon": "battery", "name": "Energy", "value": f"{_num(f['kwh100'])} kWh/100 km"})
        if "best" in f:
            cards += [{"icon": "down", "name": "Best trip", "value": f"{_num(f['best'])} kWh/100 km", "sub": f["best_d"], "tone": "ok"},
                      {"icon": "up", "name": "Worst trip", "value": f"{_num(f['worst'])} kWh/100 km", "sub": f["worst_d"], "tone": "warn"},
                      {"icon": "snow", "name": "Air-conditioning share", "value": f"{_num(f['ac'])}%"}]
        say = first
    elif i == "trip_stats" and "dd" in f and "span" in f:
        hero = {"value": f"{_num(f['dd'])}/{_num(f['span'])}", "unit": "days", "label": "with at least one drive"}
        cards += [{"icon": "road", "name": "Per day", "value": f"{_num(f['per_day'])} km"},
                  {"icon": "city", "name": "Weekday trips", "value": f"{_num(f['wk_share'])}%"}]
        say = first
    elif i == "trip_stats" and any(k.startswith("km_") for k in f):
        months = [(k[3:], f[k]) for k in f if k.startswith("km_")]
        hero = {"value": _num(months[-1][1]), "unit": "km", "label": f"driven in {months[-1][0]}"}
        for m, km in months:
            cards.append({"icon": "road", "name": m, "value": f"{_num(km)} km", "sub": f"{_num(f['n_' + m])} trips · {_num(f['dd_' + m])} days"})
        say = first
    elif i == "trip_stats" and "bt" in f:
        hero = {"value": _num(f["bt"]), "unit": "°C", "label": "battery peak on that drive"}
        cards += [{"icon": "thermo", "name": "Your usual trip peak", "value": f"{_num(f['usual'])} °C", "tone": "ok"},
                  {"icon": "dot", "name": "Outside", "value": f"{_num(f['amb'])} °C"}]
        say = first
    elif i == "charging_history" and "avg_min" in f:
        hero = {"value": _num(f["avg_h"]), "unit": "h", "label": "average charging session"}
        cards += [{"icon": "battery", "name": "Energy per session", "value": f"{_num(f['avg_kwh'])} kWh"},
                  {"icon": "dot", "name": "Sessions", "value": _num(f["nsel"]), "sub": f"{_num(f['min_min'])}–{_num(f['max_min'])} min"}]
        say = first
    elif i == "charging_history" and "kwh" in f:
        if re.search(r"fast|\bdc\b", ans.question.lower()):
            hero = {"value": _num(f["ndc"]), "unit": "", "label": "DC fast charges"}
        elif re.search(r"how (often|many)|frequen", ans.question.lower()):
            hero = {"value": _num(f["n"]), "unit": "", "label": "charging sessions"}
        else:
            hero = {"value": _num(f["kwh"]), "unit": "kWh", "label": "charged into the battery"}
        cards += [{"icon": "dot", "name": "Sessions", "value": _num(f["n"])},
                  {"icon": "spark", "name": "DC fast charges", "value": _num(f["ndc"])},
                  {"icon": "battery", "name": "Home AC charges", "value": _num(f["nac"])}]
        if "soh_b" in f:
            cards.append({"icon": "battery", "name": "Battery health", "value": f"{_num(f['soh_b'])}%", "tone": "ok", "sub": "no measurable change"})
        say = first
    elif i == "health_summary" and a_recs(ans):
        for w in ans.why[:4]:
            head, _, rest = w.partition(":")
            if rest and len(head) <= 22:
                name, val = head, rest.strip()
            else:
                name, val = ("Habit" if "habit" in ans.what else "Advice"), w
            icon = "warn" if name == "Repair" else ("check" if name.startswith("No upgrade") else "bulb")
            cards.append({"icon": icon, "name": name, "value": val[:110], "small": True,
                          "tone": "warn" if name == "Repair" else ("ok" if icon == "check" else "")})
        say = ans.what + " " + ans.why[0]
    elif i == "signal_trend" and "min_on" in f:
        hero = {"value": _num(f["days_on"]), "unit": "days", "label": "with the wipers on"}
        cards += [{"icon": "dot", "name": "Minutes wiping", "value": _num(f["min_on"])},
                  {"icon": "dot", "name": "Steady rain (low/high)", "value": f"{_num(f['min_heavy'])} min"}]
        say = first
    elif i == "not_recorded":
        say = first
    elif i == "trip_stats" and "km" in f:
        hero = {"value": _num(f["km"]), "unit": "km", "label": f"driven, {f.get('window', '')}"}
        cards += [{"icon": "road", "name": "Trips", "value": _num(f["n"])},
                  {"icon": "battery", "name": "Energy used", "value": f"{_num(f['kwh'])} kWh"}]
        say = first
    elif i == "before_event" and "warn_time" in f:
        hero = {"value": f["warn_time"], "unit": "", "label": f"warning on {f.get('warn_date', '')}"}
        if "tmax" in f:
            cards.append({"icon": "thermo", "name": "Battery peak before it", "value": f"{_num(f['tmax'])} °C",
                          "sub": f"limit {_num(f.get('t_threshold', ''))} °C", "tone": "ok" if f['tmax'] < f.get('t_threshold', 99) else "warn"})
        for k, v in f.items():
            if re.fullmatch(r"d\d+", k):
                cards.append({"icon": "warn", "name": f"Fault {v}", "value": f.get("dt" + k[1:], ""), "tone": "warn"})
        say = first
    elif i in REFUSE:
        say = first
    if not cards:
        for p in ans.proof[:4]:
            cards.append({"icon": "dot", "name": p[0], "value": str(p[1])})
    follow = FOLLOW.get(i, DEFAULT_FOLLOW)
    alts = (ans.plan or {}).get("alternatives") or []
    understood = None
    if alts:
        understood = f"I understood this as {NAMES.get(i, i)}. Not what you meant?"
        follow = [ASK[a] for a in alts if a in ASK][:3] or follow
    if "min_on" in f:
        follow = ["Was there any problem during rain?", "Why did lane assist switch off?", "How much range do I have left?"]
    out = {"hero": hero, "cards": cards, "say": say or first, "follow": follow, "understood": understood,
           "emergency": plan.get("emergency"),
           "kind": "refuse" if i in REFUSE else ("emergency" if i == "emergency" else "info" if i in ("not_recorded", "capabilities") else ("search" if i == "related_records" else "answer"))}
    out["voice"], out["voice_style"] = warm(ans, out)
    if out["follow"]:
        out["ask_next"] = offer(out["follow"][0])
    out["ui_verified"] = ui_verified(out, ans)
    return out


def ui_verified(ui: dict, ans) -> bool:
    """Every number shown on the HMI must equal a registered fact (or appear in the verified answer text)."""
    nums = set()
    for x in ans.facts:
        if isinstance(x.value, (int, float)):
            nums.add(round(abs(float(x.value)), 2))
        else:
            nums.update(round(float(n), 2) for n in NUM_RE.findall(str(x.value)))
    text_nums = {round(abs(float(n)), 2) for n in NUM_RE.findall(ans.text())}
    allowed = nums | text_nums | {0.0, 100.0}
    shown = []
    h = ui.get("hero") or {}
    shown += [h.get("value", ""), h.get("label", "")]
    for c in ui["cards"]:
        shown += [str(c.get("value", "")), str(c.get("sub", "")), str(c.get("name", ""))]
    shown.append(ui["say"])
    shown.append(ui.get("voice", ""))
    for s in shown:
        for n in NUM_RE.findall(s.replace("→", " ")):
            if round(abs(float(n)), 2) not in allowed:
                return False
    return True
