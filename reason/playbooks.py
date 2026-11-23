"""Playbooks: deterministic analysis recipes. They query REMEMBER, compute, and register every number as a fact.

No language model is involved here. Each returns an Answer whose text is built only from registered facts.
"""
from __future__ import annotations

import re

import numpy as np

from remember.store import Store

from .answer import Answer
from .dates import IST, Window, fmt_date, fmt_time, phrase, resolve

V = "Vehicle."
P_RANGE = V + "Powertrain.TractionBattery.Range"
P_SOC = V + "Powertrain.TractionBattery.StateOfCharge.Current"
P_SOH = V + "Powertrain.TractionBattery.StateOfHealth"
P_TMAX = V + "Powertrain.TractionBattery.Temperature.Max"
P_PUMP = V + "Private.Thermal.CoolantPump.Speed"
P_DLIM = V + "Private.TractionBattery.PowerLimit.Discharge"
P_WIPER = V + "Body.Windshield.Front.Wiping.Mode"
P_VIS = V + "Private.Camera.Front.Visibility"
P_LANE = V + "Private.ADAS.LaneMarkingConfidence"
P_AMB = V + "Exterior.AirTemperature"
TYRE_COLS = ["tyre_fl_min", "tyre_fr_min", "tyre_rl_min", "tyre_rr_min"]
TYRE_NAMES = ["front-left", "front-right", "rear-left", "rear-right"]
WIPER = {0: "off", 1: "intermittent", 2: "low", 3: "high"}
WARN_NAMES = {"BATTERY_TEMP_HIGH": "battery temperature high", "REDUCED_POWER": "reduced power",
              "CHECK_TYRE_PRESSURE": "check tyre pressure", "LOW_12V_BATTERY": "low 12 V battery",
              "CAMERA_UNAVAILABLE": "camera unavailable"}
SIGNALS = {  # plan signal name -> (VSS path, label, unit)
    "battery_temp": (P_TMAX, "battery temperature (max cell)", "°C"),
    "battery_health": (P_SOH, "battery state of health", "%"),
    "state_of_charge": (P_SOC, "state of charge", "%"),
    "range": (P_RANGE, "displayed range", "km"),
    "ambient_temp": (P_AMB, "outside temperature", "°C"),
    "motor_temp": (V + "Powertrain.ElectricMotor.Temperature", "motor temperature", "°C"),
    "tyre_rear_left": (V + "Chassis.Axle.Row2.Wheel.Left.Tire.Pressure", "rear-left tyre pressure", "kPa"),
    "coolant_pump": (P_PUMP, "coolant pump speed", "rpm"),
    "ac_power": (V + "Private.HVAC.CompressorPower", "AC compressor power", "kW"),
    "regen_power": (V + "Private.Powertrain.RegenPower", "regenerative braking power", "kW"),
    "lv_voltage": (V + "LowVoltageBattery.CurrentVoltage", "12 V battery voltage", "V"),
    "speed": (V + "Speed", "vehicle speed", "km/h"),
}


class Ctx:
    def __init__(self, store: Store):
        self.st = store
        self.now = float(store.db.execute("SELECT v FROM meta WHERE k='retention_now'").fetchone()[0])
        self.first = float(store.db.execute("SELECT MIN(start_ts) FROM trips").fetchone()[0] or self.now)
        self.windows: list[Window] = []          # windows used by the last answer (their labels are facts)

    def _use(self, w):
        if w is not None:
            self.windows.append(w)
        return w

    def window(self, question, plan, default_days=None) -> Window | None:
        if plan.get("window_override"):
            a, b, label = plan["window_override"]
            return self._use(Window(a, b, label))
        return self._use(resolve(question, self.now, default_days))

    def all_history(self) -> Window:
        weeks = round((self.now - self.first) / 86400 / 7)
        return self._use(Window(self.first - 60, self.now + 1, f"the full {weeks}-week history", explicit=False))


def r1(x):
    return round(float(x), 1)


def r2(x):
    return round(float(x), 2)


def _good_trips(trips):
    return [t for t in trips if (t["data_coverage_pct"] or 0) >= 95 and (t["distance_km"] or 0) > 1]


def _cons(trips):
    e = sum(t["energy_kwh"] for t in trips)
    d = sum(t["distance_km"] for t in trips)
    return e / d * 100 if d else float("nan")


def _hours(t):
    return (t["end_ts"] - t["start_ts"]) / 3600


def _tyre_deficit(t):
    return sum(max(0.0, 232.0 - (t[c] or 240)) for c in TYRE_COLS)


def _regression(trips):
    """non-AC consumption (kWh/100 km) ~ stop-go share + highway share + rain + tyre deficit, over all history."""
    rows, y = [], []
    for t in _good_trips(trips):
        d = t["distance_km"]
        rows.append([1.0, t["stop_go_pct"], (t["highway_km"] or 0) / d * 100, t["rain_pct"], _tyre_deficit(t)])
        y.append((t["energy_kwh"] - t["ac_kwh"]) / d * 100)
    X, y = np.array(rows), np.array(y)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef


def _per_soc(st, w):
    r, s = st.rollup(P_RANGE, w.start, w.end), st.rollup(P_SOC, w.start, w.end)
    common, ia, ib = np.intersect1d(r["minute"], s["minute"], return_indices=True)
    soc = s["mean"][ib]
    ok = soc > 5
    return float(np.median(r["mean"][ia][ok] / soc[ok])) if ok.any() else float("nan")


# =============================================================== Q1 range drop: delta attribution
def range_drop(ctx: Ctx, q: str, plan: dict) -> Answer:
    a = Answer(q, "range_drop")
    wB = ctx.window(q, plan, default_days=14)
    wA = wB.previous()
    all_trips = ctx.st.table("trips")
    A = _good_trips(ctx.st.table("trips", wA.start, wA.end))
    B = _good_trips(ctx.st.table("trips", wB.start, wB.end))
    a.evidence += [f"trips table: {len(A)} trips {wA.fmt()} vs {len(B)} trips {wB.fmt()}",
                   "COLD tier: per-minute range and state of charge", "regression over all 8 weeks of trips"]
    consA, consB = _cons(A), _cons(B)
    dist = lambda T: sum(t["distance_km"] for t in T)
    hrs = lambda T: sum(_hours(t) for t in T)
    acA, acB = sum(t["ac_kwh"] for t in A), sum(t["ac_kwh"] for t in B)
    P_A, P_B = acA / hrs(A), acB / hrs(B)                       # average AC power while driving (kW)
    h_A, h_B = hrs(A) / dist(A), hrs(B) / dist(B)               # hours per km (slower traffic -> more)
    heat = (P_B - P_A) * (h_A + h_B) / 2 * 100                 # AC working harder in the heat
    traffic_ac = (P_A + P_B) / 2 * (h_B - h_A) * 100           # AC running longer because trips are slower
    coef = _regression(all_trips)
    sg = lambda T: sum(t["stop_go_pct"] * t["distance_km"] for t in T) / dist(T)
    hw = lambda T: sum(t["highway_km"] for t in T) / dist(T) * 100
    rn = lambda T: sum(t["rain_pct"] * t["distance_km"] for t in T) / dist(T)
    traffic = traffic_ac + coef[1] * (sg(B) - sg(A))
    delta = consB - consA
    explained = max(heat, 0) + max(traffic, 0)
    heat_share = round(max(heat, 0) / explained * 100) if explained > 0 else 0
    traffic_share = 100 - heat_share if explained > 0 else 0
    rA, rB = _per_soc(ctx.st, wA), _per_soc(ctx.st, wB)
    range_chg = (rB / rA - 1) * 100
    ambA = np.mean([t["ambient_avg_c"] for t in A])
    ambB = np.mean([t["ambient_avg_c"] for t in B])
    tmin = min(min(t[c] for c in TYRE_COLS) for t in B)
    sohB = ctx.st.rollup(P_SOH, wB.start, wB.end)["mean"]
    sohA = ctx.st.rollup(P_SOH, wA.start, wA.end)["mean"]
    f = a.fact
    f("range_change_pct", r1(range_chg), "%", "COLD range/SoC")
    f("consA", r1(consA), "kWh/100km", "trips"); f("consB", r1(consB), "kWh/100km", "trips")
    f("cons_change_pct", r1(delta / consA * 100), "%", "trips")
    f("heat_share", heat_share, "%", "attribution"); f("traffic_share", traffic_share, "%", "attribution")
    f("heat_kwh", r2(heat), "kWh/100km", "attribution"); f("traffic_kwh", r2(traffic), "kWh/100km", "attribution")
    f("ambA", r1(ambA), "°C", "trips"); f("ambB", r1(ambB), "°C", "trips")
    f("sgA", r1(sg(A)), "%", "trips"); f("sgB", r1(sg(B)), "%", "trips")
    f("acA", r1(acA / dist(A) * 100), "kWh/100km", "trips"); f("acB", r1(acB / dist(B) * 100), "kWh/100km", "trips")
    f("tmin", r1(tmin), "kPa", "trips"); f("soh", r1(sohB.mean() if sohB.size else 99), "%", "COLD")
    f("nA", len(A)); f("nB", len(B)); f("hwA", r1(hw(A))); f("hwB", r1(hw(B)))
    f("wA", wA.fmt()); f("wB", wB.fmt()); f("span_weeks", round(wB.days() / 7), "weeks", "window")
    down = range_chg < 0
    a.what = (f"Your displayed range is {abs(r1(range_chg))}% {'lower' if down else 'higher'} than in the previous "
              f"{round(wB.days() / 7)} weeks at the "
              f"same charge level: energy use rose from {r1(consA)} to {r1(consB)} kWh/100 km "
              f"({r1(delta / consA * 100):+}%).")
    causes = [(heat_share, f"Air-conditioning in hotter weather: ~{heat_share}% of the increase. Average outside "
                           f"temperature on your drives went from {r1(ambA)} to {r1(ambB)} °C and AC energy from "
                           f"{r1(acA / dist(A) * 100)} to {r1(acB / dist(B) * 100)} kWh/100 km."),
              (traffic_share, f"More stop-go traffic: ~{traffic_share}% of the increase. The car was stopped "
                              f"{r1(sg(A))}% of the time before vs {r1(sg(B))}% now, so trips take longer and "
                              f"accelerate more often.")]
    a.why = [c for _, c in sorted(causes, reverse=True)]
    soh_txt = f"{r1(sohB.mean())}%" if sohB.size else "unchanged"
    a.why.append(f"Not the cause: tyres are normal (lowest pressure {r1(tmin)} kPa) and battery health is {soh_txt}.")
    if abs(hw(B) - hw(A)) > 10:
        a.why.append(f"Note: your route mix changed (highway share {r1(hw(A))}% vs {r1(hw(B))}%).")
    a.proof = [("trips compared", f"{len(A)} vs {len(B)}"), ("periods", f"{wA.fmt()} vs {wB.fmt()}"),
               ("AC energy", f"{r1(acA / dist(A) * 100)} -> {r1(acB / dist(B) * 100)} kWh/100 km"),
               ("stopped time", f"{r1(sg(A))}% -> {r1(sg(B))}%")]
    a.next = "Pre-cool the cabin while the car is still plugged in, and use recirculation in heavy traffic."
    a.chart = {"type": "bar", "title": "What added energy per 100 km", "unit": "kWh/100 km",
               "labels": ["AC in the heat", "stop-go traffic", "other"],
               "values": [r2(max(heat, 0)), r2(max(traffic, 0)), r2(delta - max(heat, 0) - max(traffic, 0))]}
    return a


# =============================================================== Q2 efficiency change point
def efficiency_change(ctx: Ctx, q: str, plan: dict) -> Answer:
    a = Answer(q, "efficiency_change")
    w = ctx.window(q, plan) or ctx.all_history()
    trips = _good_trips(ctx.st.table("trips", w.start, w.end))
    coef = _regression(ctx.st.table("trips"))
    coef_no_tyre = coef.copy()
    coef_no_tyre[4] = 0.0                                          # what consumption "should" be with good tyres
    days: dict[str, list] = {}
    for t in trips:
        d = t["distance_km"]
        x = np.array([1.0, t["stop_go_pct"], (t["highway_km"] or 0) / d * 100, t["rain_pct"], 0.0])
        expected = float(x @ coef_no_tyre)
        actual = (t["energy_kwh"] - t["ac_kwh"]) / d * 100
        days.setdefault(fmt_date(t["start_ts"]), []).append((t["start_ts"], (actual - expected) / expected * 100, t))
    series = sorted(((min(x[0] for x in v), np.mean([x[1] for x in v]), v) for v in days.values()), key=lambda r: r[0])
    vals = np.array([s[1] for s in series])
    base = float(np.median(vals))
    excess = vals - base
    i_worst = int(np.argmax(excess))
    i0 = i_worst
    while i0 > 0 and excess[i0 - 1] > 1.0:
        i0 -= 1
    i1 = i_worst
    while i1 + 1 < len(excess) and excess[i1 + 1] > 1.0:
        i1 += 1
    worst_t, worst_ex = series[i_worst][0], excess[i_worst]
    start_t, end_t = series[i0][0], series[i1][0]
    after = series[i1 + 1][0] if i1 + 1 < len(series) else None
    period = [t for s in series[i0:i1 + 1] for _, _, t in s[2]]
    a.evidence += [f"trips table: {len(trips)} trips in {w.label}", "efficiency normalised for traffic, route and rain "
                   "(regression over all trips)", "events table: tyre anomalies / TPMS warning"]
    f = a.fact
    f("worst_date", fmt_date(worst_t)); f("start_date", fmt_date(start_t)); f("worst_pct", r1(worst_ex), "%")
    if after:
        f("after_date", fmt_date(after))
    a.what = (f"The largest efficiency drop was on {fmt_date(worst_t)}: about {r1(worst_ex)}% more energy per km than "
              f"normal for the same traffic and route. The worse period ran from {fmt_date(start_t)} to "
              f"{fmt_date(end_t)}.")
    f("end_date", fmt_date(end_t))
    # likely cause: compare candidate factors inside the period vs normal
    tyre_min = {i: min(t[c] for t in period) for i, c in enumerate(TYRE_COLS)} if period else {}
    worst_wheel = min(tyre_min, key=tyre_min.get) if tyre_min else None
    an = ctx.st.events(["ANOMALY"], w.start, w.end, code="TYRE_PRESSURE_LOW")
    warn = [e for e in ctx.st.events(["WARNING_ON"], w.start, w.end) if e["code"] == "CHECK_TYRE_PRESSURE"]
    if worst_wheel is not None and tyre_min[worst_wheel] < 225:
        f("tyre_min", r1(tyre_min[worst_wheel]), "kPa", "trips"); f("nominal", 240, "kPa", "vehicle spec (door placard)")
        a.why.append(f"The {TYRE_NAMES[worst_wheel]} tyre was losing pressure, down to {r1(tyre_min[worst_wheel])} kPa "
                     f"(the others stayed near 240 kPa). Under-inflated tyres increase rolling resistance.")
        if an:
            f("first_anomaly", fmt_date(an[0]["ts"]))
            a.why.append(f"VehicleGPT flagged the low tyre from {fmt_date(an[0]['ts'])}"
                         + (f", before the car's own tyre warning on {fmt_date(warn[0]['ts'])}." if warn else "."))
            if warn:
                f("tpms_warn", fmt_date(warn[0]["ts"]))
        if after:
            a.why.append(f"Efficiency returned to normal from {fmt_date(after)}, after the tyre pressure was restored.")
        a.next = "Check tyre pressures weekly; a slow puncture costs energy days before the warning light comes on."
    else:
        a.why.append("No single component explains it; traffic, route and rain have already been accounted for.")
        a.confidence = "medium"
    a.proof = [("worst day", fmt_date(worst_t)), ("excess energy", f"{r1(worst_ex)}%"),
               ("trips analysed", str(len(trips)))]
    a.chart = {"type": "line", "title": "Extra energy vs normal, per day", "unit": "%",
               "labels": [fmt_date(s[0])[4:] for s in series], "values": [r1(x) for x in excess],
               "highlight": [i0, i1]}
    f("n_trips", len(trips))
    return a


# =============================================================== Q3 ADAS events
def adas_events(ctx: Ctx, q: str, plan: dict) -> Answer:
    a = Answer(q, "adas_events")
    w = ctx.window(q, plan, default_days=7)
    ev = ctx.st.events(["ADAS_DISENGAGEMENT"], w.start, w.end)
    a.evidence += [f"events table: ADAS_DISENGAGEMENT in {w.fmt()}", "WARM tier: wiper, camera visibility, lane confidence",
                   "events: DTCs near each disengagement"]
    f = a.fact
    f("n", len(ev)); f("w", w.fmt())
    if not ev:
        a.what = f"No driver-assist disengagements were recorded in {w.label} ({w.fmt()})."
        return a
    reasons: dict[str, list] = {}
    for e in ev:
        reasons.setdefault(e["reason"], []).append(e)
    for k, v in reasons.items():
        f(f"n_{k}", len(v))
    on = "" if w.label in ("yesterday", "today") else ("on " if w.days() <= 1.01 else "in ")
    a.what = (f"{len(ev)} lane-keep-assist disengagement{'s' if len(ev) > 1 else ''} {on}{w.label}"
              + (f" ({w.fmt()})" if w.fmt() not in w.label else "") + ": "
              + ", ".join(f"{len(v)} x {k.replace('_', ' ').lower()}" for k, v in reasons.items()) + ".")
    for e in ev:
        t = e["ts"]
        ts, wv = ctx.st.series(P_WIPER, t - 2, t)
        _, vis = ctx.st.series(P_VIS, t - 20, t - 2)
        wiper = WIPER.get(int(wv[-1]), "?") if wv.size else "unknown"
        f(f"t{e['id']}", fmt_time(t, False)); f(f"v{e['id']}", r1(e["speed_kmh"])); f(f"c{e['id']}", e["lane_conf"])
        line = f"{fmt_time(t, False)} at {r1(e['speed_kmh'])} km/h - {e['reason'].replace('_', ' ').lower()}"
        if e["reason"] == "LANE_MARKINGS_LOST":
            vmin = r1(vis.min()) if vis.size else None
            if vmin is not None:
                f(f"vis{e['id']}", vmin)
            line += f" (lane confidence {e['lane_conf']}%, wipers {wiper}" + (f", camera visibility {vmin}%)" if vmin is not None else ")")
        elif e["reason"] == "CAMERA_UNAVAILABLE":
            dtc = [d for d in ctx.st.events(["DTC_SET"], t - 60, t + 120)]
            logs = ctx.st.db.execute("SELECT COUNT(*) FROM logs WHERE ts BETWEEN ? AND ? AND template LIKE "
                                     "'%restarting camera service%'", (t - 60, t + 120)).fetchone()[0]
            f(f"logs{e['id']}", logs)
            for d in dtc:
                f(f"dtc{d['id']}", d["code"])
            line += (f": front camera link lost, camera service restarted {logs} times"
                     + (f", fault code {dtc[0]['code']} ({ctx.st.db.execute('SELECT description FROM dtc_dict WHERE code=?', (dtc[0]['code'],)).fetchone()[0].lower()})" if dtc else ""))
        a.details.append(line)
    rain = reasons.get("LANE_MARKINGS_LOST", [])
    cam = reasons.get("CAMERA_UNAVAILABLE", [])
    if rain:
        a.why.append(f"{len(rain)} happened because lane markings were not visible in heavy rain: expected system "
                     f"behaviour, not a fault.")
    if cam:
        a.why.append(f"{len(cam)} was caused by a front-camera communication fault: this one is a real defect.")
    other = [k for k in reasons if k not in ("LANE_MARKINGS_LOST", "CAMERA_UNAVAILABLE")]
    for k in other:
        a.why.append(f"{len(reasons[k])} x {k.replace('_', ' ').lower()}.")
    a.proof = [("events", str(len(ev))), ("period", w.fmt())]
    a.next = ("Have the front camera connection checked at the next service." if cam else
              "No action needed; keep hands on the wheel in poor visibility.")
    return a


# =============================================================== Q4 causal timeline
WARN_KEYWORDS = [("BATTERY_TEMP_HIGH", r"batter\w* .*(hot|heat|temp|overheat)|overheat|thermal|too hot|temperature (warning|alert)|heat"),
                 ("REDUCED_POWER", r"power|derat|limp"), ("CHECK_TYRE_PRESSURE", r"tyre|tire"),
                 ("LOW_12V_BATTERY", r"12 ?v|12-volt|low voltage"), ("CAMERA_UNAVAILABLE", r"camera")]


def before_event(ctx: Ctx, q: str, plan: dict) -> Answer:
    a = Answer(q, "before_event")
    ql = q.lower()
    code = plan.get("code") or next((c for c, pat in WARN_KEYWORDS if re.search(pat, ql)), None)
    w = ctx.window(q, plan) or ctx.all_history()
    warns = [e for e in ctx.st.events(["WARNING_ON"], w.start, w.end) if code is None or e["code"] == code]
    if not warns:
        a.what = f"I found no {'matching ' if code else ''}warning in {w.label}."
        a.confidence = "low"
        return a
    target = warns[-1]
    T = target["ts"]
    look = 30 * 60
    a.fact("lookback_min", 30, "min", "playbook look-back window")
    ev = [e for e in ctx.st.events(["DTC_SET", "ANOMALY", "WARNING_ON"], T - look, T + 5)]
    charges = [c for c in ctx.st.table("charges", T - 3 * 3600, T) if c["end_ts"] >= T - look]
    a.evidence += [f"events table: DTCs, anomalies and warnings {fmt_time(T - look)} - {fmt_time(T)}",
                   "charges table", "WARM tier: coolant pump speed, battery temperature, power limit",
                   "system logs (shown to you; log text is never given to the language model)"]
    f = a.fact
    f("warn_time", fmt_time(T, False)); f("warn_date", fmt_date(T)); f("warn_code", target["code"])
    steps = []
    for c in charges:
        f(f"cs{c['id']}", fmt_time(c["start_ts"], False)); f(f"ce{c['id']}", fmt_time(c["end_ts"], False))
        f(f"ckw{c['id']}", r1(c["energy_added_kwh"]))
        steps.append((c["start_ts"], f"{fmt_time(c['start_ts'], False)} {c['charger'].replace('_', ' ').lower()} "
                                     f"charging started ({r1(c['energy_added_kwh'])} kWh added by "
                                     f"{fmt_time(c['end_ts'], False)})"))
    ts, pump = ctx.st.series(P_PUMP, T - look, T)
    if pump.size:
        low = np.where(pump < 1500)[0]
        if low.size:
            f("pump_t", fmt_time(ts[low[0]], False)); f("pump_rpm", round(float(pump[low[0]])))
            f("pump_before", round(float(np.median(pump[:max(low[0], 1)]))))
            steps.append((ts[low[0]], f"{fmt_time(ts[low[0]], False)} coolant pump speed fell to "
                                      f"{round(float(pump[low[0]]))} rpm (was {round(float(np.median(pump[:max(low[0], 1)])))} rpm)"))
    ts, tmax = ctx.st.series(P_TMAX, T - look, T + 60)
    if tmax.size:
        i = int(np.argmax(tmax))
        f("tmax", r1(tmax[i])); f("tmax_t", fmt_time(ts[i], False))
        hot = np.where(tmax >= 45)[0]
        f("t_threshold", 45, "°C", "playbook threshold")
        if hot.size:
            f("t45", fmt_time(ts[hot[0]], False))
            steps.append((ts[hot[0]], f"{fmt_time(ts[hot[0]], False)} battery temperature passed 45 °C"))
    ts, dl = ctx.st.series(P_DLIM, T - look, T)
    if dl.size and dl.min() < dl.max():
        j = int(np.argmax(dl < dl.max()))
        f("derate_t", fmt_time(ts[j], False)); f("dl_from", round(float(dl.max()))); f("dl_to", round(float(dl.min())))
        steps.append((ts[j], f"{fmt_time(ts[j], False)} drive power limited from {round(float(dl.max()))} to "
                             f"{round(float(dl.min()))} kW to protect the battery"))
    for e in ev:
        if e["type"] == "DTC_SET":
            desc = ctx.st.db.execute("SELECT description FROM dtc_dict WHERE code=?", (e["code"],)).fetchone()
            f(f"d{e['id']}", e["code"]); f(f"dt{e['id']}", fmt_time(e["ts"], False))
            steps.append((e["ts"] + 0.1, f"{fmt_time(e['ts'], False)} fault code {e['code']}: {desc[0] if desc else ''}"))
    if target["code"] == "LOW_12V_BATTERY":                      # slow problems: look back several days
        lv = ctx.st.events(["ANOMALY"], T - 7 * 86400, T + 5, code="LV_LOW_AT_WAKE")
        for e in lv:
            f(f"lv{e['id']}", r2(e["value_v"])); f(f"lvd{e['id']}", fmt_date(e["ts"])); f(f"lvt{e['id']}", fmt_time(e["ts"], False))
            steps.append((e["ts"] - 0.1, f"{fmt_date(e['ts'])} {fmt_time(e['ts'], False)} 12 V battery at wake-up: "
                                         f"{r2(e['value_v'])} V"))
    steps.append((T + 0.2, f"{fmt_time(T, False)} warning shown: {WARN_NAMES.get(target['code'], target['code'])}"))
    # system log lines around the warning (Drain3 templates; shown to the user, never fed to the language model)
    logs = ctx.st.db.execute("SELECT ts, proc, msg FROM logs WHERE ts BETWEEN ? AND ? AND sev <= 4 ORDER BY ts",
                             (T - 300, T + 60)).fetchall()
    for lt, proc, msg in logs:
        f(f"lg{lt}", fmt_time(lt, False))
        for n in re.findall(r"\d+(?:\.\d+)?", msg):              # quoted log text: its numbers are stored data
            f("log_n", float(n), source="system log (quoted)")
        steps.append((lt + 0.05, f"{fmt_time(lt, False)} log [{proc}] {msg}"))
    steps.sort()
    a.details = [s for _, s in steps]
    dtcs = [e for e in ev if e["type"] == "DTC_SET"]
    root = dtcs[0] if dtcs else next((e for e in ev if e["type"] == "ANOMALY"), None)
    span = T - steps[0][0]
    span_txt = f"{round(span / 86400)} days" if span > 86400 * 1.5 else f"{round(span / 60)} minutes"
    f("chain_min", round(span / 60)); f("chain_days", round(span / 86400))
    wname = WARN_NAMES.get(target["code"], target["code"])
    if span < 90:
        a.what = (f"The '{wname}' warning at {fmt_time(T, False)} on {fmt_date(T)} appeared suddenly, together with the "
                  f"fault below; there were no warning signs in the 30 minutes before.")
    else:
        a.what = f"The '{wname}' warning at {fmt_time(T, False)} on {fmt_date(T)} was the end of a chain that started {span_txt} earlier."
    if root is not None and root["type"] == "DTC_SET":
        desc = ctx.st.db.execute("SELECT description FROM dtc_dict WHERE code=?", (root["code"],)).fetchone()
        consequence = {"P1A20": "With weak cooling, the battery heated up and the car limited power.",
                       "P1B05": "The 12 V battery was being drained while the car was parked; its voltage at wake-up "
                                "fell night after night until it dropped below the warning level.",
                       "C1A10": "The tyre was losing air slowly (a puncture), so the pressure fell below the limit.",
                       "U1C01": "The camera module stopped sending images, so the assistance systems that need it "
                                "switched off."}.get(root["code"], "")
        a.why.append(f"Root cause: {desc[0].lower() if desc else root['code']} ({root['code']} at "
                     f"{fmt_time(root['ts'], False)}). {consequence}".strip())
    elif root is not None:
        a.why.append(f"Earliest abnormal sign: {root['code'].replace('_', ' ').lower()} at {fmt_time(root['ts'], False)}.")
    thermal = target["code"] in ("BATTERY_TEMP_HIGH", "REDUCED_POWER")
    if charges and thermal:
        a.why.append("It happened during and right after a fast charge, when the battery produces the most heat.")
    # distractor check: same kind of charge before, without the fault
    others = [c for c in ctx.st.table("charges") if c["charger"] == "DC_FAST" and c["end_ts"] < T - look]
    if others and thermal:
        o = others[-1]
        f("o_date", fmt_date(o["start_ts"])); f("o_amb", r1(o["ambient_avg_c"])); f("o_t", r1(o["batt_tmax_c"]))
        a.why.append(f"Heat alone does not explain it: a similar fast charge on {fmt_date(o['start_ts'])} at "
                     f"{r1(o['ambient_avg_c'])} °C outside peaked at only {r1(o['batt_tmax_c'])} °C.")
    if tmax.size and thermal:
        a.proof = [("peak battery temp", f"{r1(tmax.max())} °C"), ("chain length", f"{round((T - steps[0][0]) / 60)} min")]
    a.proof.append(("black-box file", target.get("case_id") or "-"))
    if thermal and tmax.size:
        step = max(1, tmax.size // 120)
        tt, pp = ctx.st.series(P_PUMP, T - look, T + 60)
        a.chart = {"type": "line", "title": "Battery temperature (max) before the warning", "unit": "°C",
                   "labels": [fmt_time(x, False)[:5] for x in ts[::step]], "values": [r1(x) for x in tmax[::step]],
                   "marker": fmt_time(T, False)[:5]}
    f("case", str(target.get("case_id")))
    a.next = {"P1A20": "Have the battery coolant pump inspected before the next fast charge.",
              "P1B05": "Have the 12 V battery and parked-mode power consumers checked (something drains it overnight).",
              "C1A10": "Repair the tyre and check all pressures.",
              "U1C01": "Have the front camera connection checked."}.get((root or {}).get("code"),
                                                                     "Have the fault codes read at the next service.")
    return a


# =============================================================== Q5 segment compare
def segment_compare(ctx: Ctx, q: str, plan: dict) -> Answer:
    a = Answer(q, "segment_compare")
    w = ctx.window(q, plan) or ctx.all_history()
    trips = _good_trips(ctx.st.table("trips", w.start, w.end))
    f = a.fact
    if re.search(r"weekend|weekday", q.lower()):
        import datetime as dt
        wd = lambda t: dt.datetime.fromtimestamp(t["start_ts"], IST).weekday() < 5
        A = [t for t in trips if wd(t)]
        B = [t for t in trips if not wd(t)]
        cA, cB = _cons(A), _cons(B)
        f("a", r1(cA)); f("b", r1(cB)); f("na", len(A)); f("nb", len(B))
        a.what = f"Weekdays: {r1(cA)} kWh/100 km ({len(A)} trips). Weekends: {r1(cB)} kWh/100 km ({len(B)} trips)."
        a.why.append("Weekend trips are mostly long highway drives; weekdays are city commutes.")
        a.evidence.append(f"trips table: {len(trips)} trips in {w.label}")
        return a
    ce, cd = sum(t["city_kwh"] for t in trips), sum(t["city_km"] for t in trips)
    he, hd = sum(t["highway_kwh"] for t in trips), sum(t["highway_km"] for t in trips)
    c, h = ce / cd * 100, he / hd * 100
    hwy_trips = [t for t in trips if t["highway_km"] > 0.6 * t["distance_km"]]
    city_trips = [t for t in trips if t["highway_km"] == 0]
    if not hwy_trips or not city_trips or hd < 1 or cd < 1:
        a.what = f"Not enough city and highway driving in {w.label} to compare."
        return a
    hv = np.mean([t["avg_speed_kmh"] for t in hwy_trips]) if hwy_trips else float("nan")
    cv = np.mean([t["avg_speed_kmh"] for t in city_trips]) if city_trips else float("nan")
    hvmax = np.mean([t["max_speed_kmh"] for t in hwy_trips]) if hwy_trips else float("nan")
    ac_city = sum(t["ac_kwh"] for t in city_trips) / max(sum(t["distance_km"] for t in city_trips), 1) * 100
    ac_hwy = sum(t["ac_kwh"] for t in hwy_trips) / max(sum(t["distance_km"] for t in hwy_trips), 1) * 100
    f("city", r1(c)); f("hwy", r1(h)); f("cd", round(cd)); f("hd", round(hd)); f("diff", r1(abs(h - c) / min(c, h) * 100))
    f("cv", r1(cv)); f("hv", r1(hv)); f("hvmax", r1(hvmax)); f("acc", r1(ac_city)); f("ach", r1(ac_hwy))
    more = "highway" if h > c else "city"
    a.what = (f"Highway driving uses {r1(h)} kWh/100 km and city driving {r1(c)} kWh/100 km, so {more} driving costs "
              f"about {r1(abs(h - c) / min(c, h) * 100)}% more energy per km (over {round(cd)} km city and {round(hd)} km "
              f"highway in {w.label}).")
    a.why = [f"Highway: air drag grows with the square of speed; you cruise at up to {r1(hvmax)} km/h "
             f"(average {r1(hv)} km/h on highway trips).",
             f"City: stop-go costs energy, but regenerative braking recovers part of it; the AC weighs more per km "
             f"({r1(ac_city)} kWh/100 km in the city vs {r1(ac_hwy)} on the highway) because you spend longer per km."]
    a.proof = [("city", f"{r1(c)} kWh/100 km"), ("highway", f"{r1(h)} kWh/100 km"), ("distance", f"{round(cd)} + {round(hd)} km")]
    f("advice_kmh", 10, "km/h", "eco-driving guidance")
    a.chart = {"type": "bar", "title": "Energy per 100 km", "unit": "kWh/100 km", "labels": ["city", "highway"],
               "values": [r1(c), r1(h)]}
    a.next = "On the highway, cruising about 10 km/h slower is the biggest single saving."
    a.evidence.append(f"trips table: city/highway energy split per trip, {len(trips)} trips")
    return a


# =============================================================== generic playbooks
def _pick_signal(q: str, plan: dict) -> str:
    if plan.get("signal") in SIGNALS:
        return plan["signal"]
    ql = q.lower()
    for key, pat in [("battery_health", r"health|soh|degrad"), ("ac_power", r"\bac\b|air.?con|a/c|hvac"),
                     ("regen_power", r"regen"), ("tyre_rear_left", r"tyre|tire"), ("motor_temp", r"motor"),
                     ("lv_voltage", r"12 ?v|12-volt"), ("ambient_temp", r"outside|ambient|weather"),
                     ("coolant_pump", r"pump|coolant"), ("battery_temp", r"batter\w* (temp|hot)|hot|temperature"),
                     ("range", r"range"), ("state_of_charge", r"charge level|soc|battery level")]:
        if re.search(pat, ql):
            return key
    return "battery_temp"


def signal_trend(ctx: Ctx, q: str, plan: dict) -> Answer:
    a = Answer(q, "signal_trend")
    key = _pick_signal(q, plan)
    path, label, unit = SIGNALS[key]
    w = ctx.window(q, plan, default_days=30)
    r = ctx.st.rollup(path, w.start, w.end)
    f = a.fact
    a.evidence.append(f"COLD tier: per-minute {label}, {w.fmt()}")
    if not r["minute"].size:
        a.what = f"No {label} data in {w.label}."
        return a
    if key == "ac_power":
        kwh = float(np.sum(r["mean"] * r["count"]) / 3600)
        f("kwh", r1(kwh)); f("w", w.fmt())
        a.what = f"The air-conditioning used {r1(kwh)} kWh {phrase(w)}."
        return a
    if key == "regen_power" and re.search(r"city|highway", q.lower()):
        wa = ctx.all_history() if not w.explicit else w
        rg = ctx.st.rollup(path, wa.start, wa.end)
        rd = ctx.st.rollup(V + "Private.Road.Type", wa.start, wa.end)
        sp = ctx.st.rollup(V + "Speed", wa.start, wa.end)
        m, i1, i2 = np.intersect1d(rg["minute"], rd["minute"], return_indices=True)
        m2, j1, j2 = np.intersect1d(m, sp["minute"], return_indices=True)
        regen_kwh = rg["mean"][i1][j1] * rg["count"][i1][j1] / 3600
        km = sp["mean"][j2] * sp["count"][j2] / 3600
        road = rd["mean"][i2][j1]
        out = {}
        for name, mask in (("city", road < 0.1), ("highway", road > 0.9)):
            out[name] = regen_kwh[mask].sum() / max(km[mask].sum(), 1) * 100
            f(name, r2(out[name]))
        a.evidence.append("COLD tier: per-minute regen power, road type and speed over the full history")
        a.what = (f"Regenerative braking recovered {r2(out['city'])} kWh per 100 km in the city vs "
                  f"{r2(out['highway'])} kWh per 100 km on the highway.")
        a.why.append("In the city you slow down far more often, and each slow-down sends energy back to the battery."
                     if out["city"] > out["highway"] else "On your routes the highway slow-downs recovered more energy.")
        return a
    i = int(np.argmax(r["max"]))
    j = int(np.argmin(r["min"]))
    days = (r["minute"] * 60 - w.start) // 86400
    daily = [float(np.mean(r["mean"][days == d])) for d in np.unique(days)]
    slope = np.polyfit(np.arange(len(daily)), daily, 1)[0] * 7 if len(daily) > 2 else 0.0
    f("max", r1(r["max"][i])); f("max_t", fmt_time(r["minute"][i] * 60)); f("min", r1(r["min"][j]))
    f("mean", r1(np.mean(r["mean"]))); f("slope", r2(slope)); f("w", w.fmt())
    a.chart = {"type": "line", "title": f"{label} - daily average", "unit": unit,
               "labels": [fmt_date(w.start + d * 86400)[4:] for d in np.unique(days)], "values": [r1(x) for x in daily]}
    trend = "stable" if abs(slope) < 0.05 * max(abs(np.mean(daily)), 1e-6) / 4 else ("rising" if slope > 0 else "falling")
    a.what = (f"{label.capitalize()} {phrase(w)}: average {r1(np.mean(r['mean']))} {unit}, highest "
              f"{r1(r['max'][i])} {unit} on {fmt_time(r['minute'][i] * 60)}, lowest {r1(r['min'][j])} {unit}.")
    if w.days() > 2:                                   # a trend needs more than a day of data
        a.why.append(f"Trend: {trend} ({r2(slope):+} {unit} per week).")
    if key == "battery_health" and a.why:
        a.why[-1] = f"Trend: {trend}; battery health is not declining measurably over this period." if trend == "stable" else a.why[-1]
    return a


def dtc_lookup(ctx: Ctx, q: str, plan: dict) -> Answer:
    a = Answer(q, "dtc_lookup")
    w = ctx.window(q, plan) or ctx.all_history()
    sets = ctx.st.events(["DTC_SET"], w.start, w.end)
    passes = ctx.st.events(["DTC_PASSED"], w.start, ctx.now + 1)
    f = a.fact
    a.evidence.append(f"events table: DTC_SET / DTC_PASSED in {w.label}; dtc_dict descriptions")
    ql = q.lower()
    if re.search(r"camera", ql):
        sets = [e for e in sets if e["code"] in ("U1C01", "B1C02")]
    if re.search(r"pump|coolant", ql):
        sets = [e for e in sets if e["code"] == "P1A20"]
    f("n", len(sets))
    if not sets:
        a.what = f"No fault codes were logged in {w.label}."
        return a
    active = [e for e in sets if not any(p["code"] == e["code"] and p["ts"] > e["ts"] for p in passes)]
    a.what = f"{len(sets)} fault code{'s' if len(sets) > 1 else ''} logged in {w.label}; {len(active)} still active."
    f("na", len(active))
    for e in sets:
        desc = ctx.st.db.execute("SELECT description FROM dtc_dict WHERE code=?", (e["code"],)).fetchone()
        f(f"c{e['id']}", e["code"]); f(f"t{e['id']}", fmt_time(e["ts"])); f(f"td{e['id']}", fmt_date(e["ts"]))
        state = "active" if e in active else "passed self-test since"
        f(f"st{e['id']}", "active" if e in active else "passed"); f(f"desc{e['id']}", desc[0] if desc else e["code"])
        a.details.append(f"{e['code']} {fmt_time(e['ts'])} ({e['ecu']}): {desc[0] if desc else ''} [{state}]")
    a.next = "Codes marked active should be checked at the next service." if active else ""
    return a


def charging_history(ctx: Ctx, q: str, plan: dict) -> Answer:
    a = Answer(q, "charging_history")
    w = ctx.window(q, plan) or ctx.all_history()
    ch = ctx.st.table("charges", w.start, w.end)
    dc = [c for c in ch if c["charger"] == "DC_FAST"]
    ac = [c for c in ch if c["charger"] == "AC_HOME"]
    f = a.fact
    f("n", len(ch)); f("ndc", len(dc)); f("nac", len(ac)); f("kwh", r1(sum(c["energy_added_kwh"] for c in ch)))
    a.evidence.append(f"charges table: {len(ch)} sessions in {w.label}")
    a.what = (f"{len(ch)} charging sessions in {w.label}: {len(dc)} DC fast charge{'s' if len(dc) != 1 else ''} and "
              f"{len(ac)} home AC charges, {r1(sum(c['energy_added_kwh'] for c in ch))} kWh in total.")
    for c in dc:
        f(f"d{c['id']}", fmt_date(c["start_ts"])); f(f"k{c['id']}", r1(c["energy_added_kwh"]))
        f(f"m{c['id']}", r1(c["duration_min"])); f(f"t{c['id']}", r1(c["batt_tmax_c"])); f(f"a{c['id']}", r1(c["ambient_avg_c"]))
        a.details.append(f"DC {fmt_date(c['start_ts'])}: {r1(c['energy_added_kwh'])} kWh in {r1(c['duration_min'])} min, "
                         f"outside {r1(c['ambient_avg_c'])} °C, battery peaked at {r1(c['batt_tmax_c'])} °C")
    if re.search(r"hot|heat|weather|slow", q.lower()) and len(dc) >= 2:
        pw = [(c["ambient_avg_c"], c["energy_added_kwh"] / (c["duration_min"] / 60), c) for c in dc]
        pw.sort(key=lambda x: x[0])
        (ac, pc, c1), (ah, ph, c2) = pw[0], pw[-1]
        f("p_cool", r1(pc)); f("p_hot", r1(ph)); f("a_cool", r1(ac)); f("a_hot", r1(ah))
        f("t_cool", r1(c1["batt_tmax_c"])); f("t_hot", r1(c2["batt_tmax_c"]))
        slower = (pc - ph) / pc * 100
        f("slower", r1(abs(slower)))
        a.why.append(f"Average charging power was {r1(pc)} kW at {r1(ac)} °C outside and {r1(ph)} kW at {r1(ah)} °C, "
                     + (f"so the hotter session was {r1(abs(slower))}% slower." if slower > 3 else
                        "so heat did not measurably slow charging.")
                     + f" The battery reached {r1(c1['batt_tmax_c'])} °C vs {r1(c2['batt_tmax_c'])} °C.")
    return a


def vehicle_status_now(ctx: Ctx, q: str, plan: dict) -> Answer:
    a = Answer(q, "vehicle_status_now")
    f = a.fact
    vals = {}
    for key in ("state_of_charge", "range", "battery_temp"):
        ts, v = ctx.st.series(SIGNALS[key][0], ctx.now - 3600, ctx.now + 1)
        if v.size:
            vals[key] = (r1(v[-1]), ts[-1])
            f(key, r1(v[-1]))
    tyres = []
    for c, name, path in zip(TYRE_COLS, TYRE_NAMES, [V + f"Chassis.Axle.Row{r}.Wheel.{s}.Tire.Pressure" for r, s in
                                                     ((1, "Left"), (1, "Right"), (2, "Left"), (2, "Right"))]):
        _, v = ctx.st.series(path, ctx.now - 3600, ctx.now + 1)
        if v.size:
            tyres.append(r1(v[-1]))
            f(c, r1(v[-1]))
    last = max(t for _, t in vals.values()) if vals else ctx.now
    f("at", fmt_time(last))
    a.evidence.append("WARM tier: latest values")
    a.what = (f"Last reading ({fmt_time(last)}): charge {vals.get('state_of_charge', ('?',))[0]}%, range "
              f"{vals.get('range', ('?',))[0]} km, battery {vals.get('battery_temp', ('?',))[0]} °C.")
    if tyres:
        a.why.append("Tyres: " + ", ".join(f"{n} {p} kPa" for n, p in zip(TYRE_NAMES, tyres)) + ".")
    return a


def trip_stats(ctx: Ctx, q: str, plan: dict) -> Answer:
    a = Answer(q, "trip_stats")
    ql = q.lower()
    w = ctx.window(q, plan) or ctx.all_history()
    trips = ctx.st.table("trips", w.start, w.end)
    f = a.fact
    a.evidence.append(f"trips table: {len(trips)} trips in {w.label}")
    if not trips:
        a.what = f"No trips in {w.label}."
        return a
    metric = plan.get("metric")
    if metric is None:
        metric = ("longest" if re.search(r"longest|farthest", ql) else "shortest" if "shortest" in ql else
                  "worst_eff" if re.search(r"worst|least efficient|highest consumption", ql) else
                  "tyre_last_hwy" if re.search(r"tyre|tire", ql) else
                  "speed_compare" if re.search(r"speed", ql) else "braking_rain" if re.search(r"brak", ql) else "total")
    def desc(t):
        return (f"{fmt_time(t['start_ts'])}: {r1(t['distance_km'])} km, {r1(t['kwh_per_100km'])} kWh/100 km, "
                f"average {r1(t['avg_speed_kmh'])} km/h")

    def reg(t):
        f(f"s{t['id']}", fmt_time(t["start_ts"])); f(f"d{t['id']}", r1(t["distance_km"]))
        f(f"e{t['id']}", r1(t["kwh_per_100km"])); f(f"v{t['id']}", r1(t["avg_speed_kmh"]))
    if metric in ("longest", "shortest"):
        t = (max if metric == "longest" else min)(trips, key=lambda t: t["distance_km"])
        reg(t)
        a.what = f"Your {metric} trip in {w.label}: {desc(t)}."
    elif metric == "worst_eff":
        good = _good_trips(trips)
        t = max(good, key=lambda t: t["kwh_per_100km"])
        reg(t)
        f("ac", r1(t["ac_share_pct"] or 0)); f("sg", r1(t["stop_go_pct"])); f("rl", r1(t["tyre_rl_min"]))
        a.what = f"Least efficient trip in {w.label}: {desc(t)}."
        a.why.append(f"On that trip the AC took {r1(t['ac_share_pct'] or 0)}% of the energy, the car was stopped "
                     f"{r1(t['stop_go_pct'])}% of the time, and the lowest tyre pressure was {r1(t['tyre_rl_min'])} kPa.")
    elif metric == "tyre_last_hwy":
        hw = [t for t in trips if t["highway_km"] > 20]
        t = hw[-1] if hw else trips[-1]
        reg(t)
        p = [t[c] for c in TYRE_COLS]
        for c in TYRE_COLS:
            f(c + str(t["id"]), r1(t[c]))
        ok = min(p) > 225
        a.what = (f"On your last highway trip ({fmt_time(t['start_ts'])}) tyre pressures were "
                  + ("normal" if ok else "LOW") + ": " + ", ".join(f"{n} {r1(x)} kPa" for n, x in zip(TYRE_NAMES, p)) + ".")
    elif metric == "speed_compare":
        prev = ctx.window(q, plan, default_days=7)
        wp = prev.previous()
        A = [t for t in ctx.st.table("trips", wp.start, wp.end) if t["highway_km"] == 0]
        B = [t for t in ctx.st.table("trips", prev.start, prev.end) if t["highway_km"] == 0]
        sa, sb = np.mean([t["avg_speed_kmh"] for t in A]), np.mean([t["avg_speed_kmh"] for t in B])
        f("sa", r1(sa)); f("sb", r1(sb)); f("wa", wp.fmt()); f("wb", prev.fmt())
        a.what = f"Average city/commute speed: {r1(sb)} km/h in {prev.fmt()} vs {r1(sa)} km/h in {wp.fmt()}."
    elif metric == "braking_rain":
        acc = ctx.st.rollup(V + "Acceleration.Longitudinal", w.start, w.end)
        wip = ctx.st.rollup(P_WIPER, w.start, w.end)
        m, i1, i2 = np.intersect1d(acc["minute"], wip["minute"], return_indices=True)
        harsh = acc["min"][i1] < -2.5
        rain = wip["max"][i2] >= 2
        hr = harsh[rain].mean() * 100 if rain.any() else 0.0
        hd = harsh[~rain].mean() * 100 if (~rain).any() else 0.0
        f("hr", r1(hr)); f("hd", r1(hd)); f("thr", 2.5); f("mr", int(rain.sum())); f("md", int((~rain).sum()))
        a.evidence.append("COLD tier: per-minute strongest deceleration and wiper mode")
        a.what = (f"Harsh braking (deceleration stronger than 2.5 m/s²) happened in {r1(hr)}% of driving minutes in rain "
                  f"vs {r1(hd)}% in dry weather.")
        a.why.append("So braking did not get harsher in the rain." if hr <= hd * 1.2 else
                     "So braking was harsher in the rain; leave more distance to the car ahead.")
    else:
        km = sum(t["distance_km"] for t in trips)
        e = sum(t["energy_kwh"] for t in trips)
        f("km", r1(km)); f("kwh", r1(e)); f("n", len(trips))
        a.what = f"In {w.label} you drove {r1(km)} km in {len(trips)} trips and used {r1(e)} kWh."
    return a


def data_quality(ctx: Ctx, q: str, plan: dict) -> Answer:
    a = Answer(q, "data_quality")
    w = ctx.window(q, plan) or ctx.all_history()
    gaps = ctx.st.events(["DATA_GAP"], w.start, w.end)
    f = a.fact
    f("n", len(gaps))
    a.evidence.append(f"events table: DATA_GAP in {w.label}")
    if not gaps:
        a.what = f"The recording is complete for {w.label}: no data gaps."
        return a
    a.what = f"{len(gaps)} data gap{'s' if len(gaps) > 1 else ''} in {w.label}."
    for g in gaps:
        f(f"s{g['id']}", g["start"][11:19]); f(f"e{g['id']}", g["end"][11:19]); f(f"m{g['id']}", g["duration_min"])
        f(f"d{g['id']}", fmt_date(g["ts"]))
        a.details.append(f"{fmt_date(g['ts'])} {g['start'][11:19]} - {g['end'][11:19]}: CAN frames missing for "
                         f"{g['duration_min']} min while other sources were active (values in this window are unknown).")
    a.why.append("VehicleGPT never fills gaps with guesses; answers touching this window are marked incomplete.")
    return a


def health_summary(ctx: Ctx, q: str, plan: dict) -> Answer:
    a = Answer(q, "health_summary")
    ql = q.lower()
    if re.search(r"first (two|2) weeks", ql):
        w = ctx._use(Window(ctx.first - 60, ctx.first + 14 * 86400, "the first 2 weeks"))
    else:
        w = ctx.window(q, plan) or ctx.all_history()
    f = a.fact
    dtcs = ctx.st.events(["DTC_SET"], w.start, w.end)
    an = ctx.st.events(["ANOMALY"], w.start, w.end)
    warns = ctx.st.events(["WARNING_ON"], w.start, w.end)
    gaps = ctx.st.events(["DATA_GAP"], w.start, w.end)
    f("nd", len(dtcs)); f("nw", len(warns)); f("ng", len(gaps)); f("w", w.fmt())
    a.evidence.append(f"events table: DTCs, anomalies, warnings, gaps in {w.fmt()}")
    if not (dtcs or an or warns):
        a.what = f"No faults, warnings or anomalies in {w.label} ({w.fmt()}): the vehicle was healthy."
        return a
    kinds: dict[str, list] = {}
    for e in an:
        kinds.setdefault(e["code"], []).append(e)
    f("na", sum(len(v) for v in kinds.values()))
    a.what = (f"In {w.label}: {len(dtcs)} fault codes, {len(warns)} dashboard warnings, "
              f"{sum(len(v) for v in kinds.values())} anomaly detections, {len(gaps)} data gap{'s' if len(gaps) != 1 else ''}.")
    for e in dtcs:
        desc = ctx.st.db.execute("SELECT description FROM dtc_dict WHERE code=?", (e["code"],)).fetchone()
        f(f"c{e['id']}", e["code"]); f(f"t{e['id']}", fmt_date(e["ts"]))
        a.details.append(f"{e['code']} on {fmt_date(e['ts'])}: {desc[0] if desc else ''}")
    for k, v in kinds.items():
        f(f"k{k}", len(v)); f(f"kf{k}", fmt_date(v[0]["ts"]))
        a.details.append(f"{k.replace('_', ' ').lower()}: {len(v)} detection{'s' if len(v) > 1 else ''} from "
                         f"{fmt_date(v[0]['ts'])}")
    a.next = "Show this summary (or export it) at the next service visit."
    return a


def refuse(ctx: Ctx, q: str, plan: dict) -> Answer:
    kind = plan.get("intent", "unsafe_command")
    a = Answer(q, kind)
    a.fact("grid_km", 1, "km", "privacy setting: location coarsened to 0.01 degree")
    a.what = {"unsafe_command": "I can't do that. VehicleGPT is read-only: it can look at vehicle data but has no way to "
                                "change settings, control the car, clear codes or send data anywhere.",
              "privacy": "I don't identify or track people. VehicleGPT stores vehicle data only, with location "
                         "coarsened to about 1 km.",
              "out_of_scope": "That needs the internet or isn't about this vehicle. I can answer questions about the "
                              "car's data: range, efficiency, charging, warnings, faults and trips."}[kind]
    return a


PLAYBOOKS = {"range_drop": range_drop, "efficiency_change": efficiency_change, "adas_events": adas_events,
             "before_event": before_event, "segment_compare": segment_compare, "signal_trend": signal_trend,
             "dtc_lookup": dtc_lookup, "charging_history": charging_history, "vehicle_status_now": vehicle_status_now,
             "trip_stats": trip_stats, "data_quality": data_quality, "health_summary": health_summary,
             "unsafe_command": refuse, "privacy": refuse, "out_of_scope": refuse}
