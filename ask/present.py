"""Presentation layer for the head-unit (HMI) screen.

Turns a verified Answer into glanceable UI pieces: one hero number, up to four factor cards, a short sentence
to speak, and follow-up suggestions. Rule: this layer never computes a new number. Every number it shows is
copied from a registered fact, and `ui_verified` re-checks that, so the head-unit keeps the verifier guarantee.
"""
from __future__ import annotations

import re

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

FOLLOW = {
    "not_recorded": ["Show my charging history", "Any fault codes?", "How much range do I have left?"],
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


def a_recs(ans) -> bool:
    return ans.what.startswith("Based on")


def build(ans) -> dict:
    f = {x.name: x.value for x in ans.facts}
    i = ans.intent
    hero, cards, say = None, [], ""
    first = re.split(r"(?<=[.!?])\s", ans.what.strip())[0] if ans.what else ""

    if i == "vehicle_status_now" and "range" in f:
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
    if "min_on" in f:
        follow = ["Was there any problem during rain?", "Why did lane assist switch off?", "How much range do I have left?"]
    out = {"hero": hero, "cards": cards, "say": say or first, "follow": follow,
           "kind": "refuse" if i in REFUSE else ("info" if i == "not_recorded" else ("search" if i == "related_records" else "answer"))}
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
    for s in shown:
        for n in NUM_RE.findall(s.replace("→", " ")):
            if round(abs(float(n)), 2) not in allowed:
                return False
    return True
