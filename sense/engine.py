"""SENSE engine: normalise -> 1 Hz snapshot -> event miner -> black-box recorder -> sinks.

Normaliser : every source already speaks VSS (adapters); here we merge them into one current state.
             If two sources report the same signal (e.g. speed from CAN and from Android VHAL), CAN wins
             unless its value is stale (>3 s old).
Snapshot   : once per second, the latest fresh value (<=15 s old) of every signal.
Miner      : turns snapshots + discrete records into typed events (trips, charges, warnings, DTCs,
             ADAS disengagements, data gaps, contextual anomalies, log alerts).
Black box  : keeps the last 60 s in memory; any trigger event freezes -60 s / +10 s into a case file.
"""
from __future__ import annotations

import datetime as dt
import gzip
import json
import math
from collections import Counter, deque
from pathlib import Path

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig

from .config import POWER_STATE, WARNINGS

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
V = "Vehicle."
PS = V + "Private.Vehicle.PowerState"
SPEED, ODO = V + "Speed", V + "TraveledDistance"
PACKW, SOC = V + "Powertrain.TractionBattery.CurrentPower", V + "Powertrain.TractionBattery.StateOfCharge.Current"
TMAX = V + "Powertrain.TractionBattery.Temperature.Max"
AMB, COMP = V + "Exterior.AirTemperature", V + "Private.HVAC.CompressorPower"
ROAD, WIPER = V + "Private.Road.Type", V + "Body.Windshield.Front.Wiping.Mode"
LV, WARN = V + "LowVoltageBattery.CurrentVoltage", V + "Private.Cluster.WarningCode"
DCA = V + "Powertrain.TractionBattery.Charging.ChargeCurrent.DC"
PUMP, CHILLER = V + "Private.Thermal.CoolantPump.Speed", V + "Private.Thermal.ChillerActive"
TYRES = [V + f"Chassis.Axle.Row{r}.Wheel.{s}.Tire.Pressure" for r, s in ((1, "Left"), (1, "Right"), (2, "Left"), (2, "Right"))]
WHEEL_NAMES = ["front-left", "front-right", "rear-left", "rear-right"]

TRIGGERS = {"WARNING_ON", "DTC_SET", "ADAS_DISENGAGEMENT", "ANOMALY"}
# Signals published only when they change (Android VHAL on-change properties): valid until replaced.
STICKY = {V + "Private.Road.Type", V + "Powertrain.Transmission.SelectedGear", V + "Cabin.HVAC.IsAirConditioningActive",
          V + "Cabin.HVAC.Station.Row1.Driver.Temperature"}
STICKY_S = 1800.0
PRIO = {"can": 2, "someip": 2, "vhal": 1}
FRESH_S = 15.0


def clock(t: float) -> str:
    return dt.datetime.fromtimestamp(t, IST).strftime("%Y-%m-%d %H:%M:%S")


# ============================================================================ miner
class Miner:
    def __init__(self, emit):
        self.emit = emit
        self.trip = None
        self.charge = None
        self.warn_code = 0
        self.warn_since = None
        self.prev_sec = None
        self.gap_from = None
        self.base: dict = {}            # contextual baselines: key -> [n, mean, M2]
        self.run: Counter = Counter()   # consecutive-second counters for sustained conditions
        self.fired: set = set()         # one anomaly of each kind per trip/charge session
        cfg = TemplateMinerConfig()
        cfg.profiling_enabled = False
        self.tm = TemplateMiner(config=cfg)

    # ---------------------------------------------------------------- discrete records
    def on_event(self, ts, src, e):
        if e["type"] == "DTC":
            kind = "DTC_SET" if (e["confirmed"] and e["test_failed"]) else "DTC_PASSED"
            self.emit({"ts": ts, "type": kind, "source": src, "severity": "warning" if kind == "DTC_SET" else "info",
                       "code": e["code"], "ecu": e["ecu"], "dtc_hex": e["dtc_hex"], "status": f"0x{e['status']:02X}"})
        elif e["type"] in ("ADAS_DISENGAGEMENT", "ADAS_LKA_OFF"):
            self.emit({"ts": ts, "source": src, "severity": "warning" if e["type"] == "ADAS_DISENGAGEMENT" else "info",
                       **e})

    def on_log(self, ts, rec) -> dict:
        r = self.tm.add_log_message(rec["msg"])
        out = {"ts": ts, **rec, "template_id": r["cluster_id"], "template": r["template_mined"]}
        if rec["sev"] <= 3:
            self.emit({"ts": ts, "type": "LOG_ALERT", "source": "qnx", "severity": "warning" if rec["sev"] <= 2 else "info",
                       "proc": rec["proc"], "log_sev": rec["sev"], "template_id": r["cluster_id"],
                       "template": r["template_mined"], "untrusted_text": True})
        return out

    # ---------------------------------------------------------------- per-second snapshot
    def on_snapshot(self, sec, s, last):
        if self.prev_sec is not None and sec - self.prev_sec > 120:
            self.close_all(self.prev_sec)
        ps = s.get(PS)
        mode = {2: "ON", 3: "CHARGING"}.get(int(ps)) if ps is not None else None
        if mode == "ON":
            if self.charge:
                self._close_charge(sec)
            if not self.trip:
                self._open_trip(sec, s)
            self._update_trip(sec, s, last)
        elif mode == "CHARGING":
            if self.trip:
                self._close_trip(sec)
            if not self.charge:
                self._open_charge(sec, s)
            self._update_charge(sec, s)
        elif self.trip and self.gap_from is not None:
            self.trip["gap_s"] += sec - (self.prev_sec or sec)
            v = s.get(SPEED)
            if v is not None:
                self.trip["dist_gap"] += v / 3600

        wc = int(s.get(WARN, self.warn_code))
        if wc != self.warn_code:
            if self.warn_code:
                self.emit({"ts": sec, "type": "WARNING_OFF", "source": "can", "severity": "info",
                           "code": WARNINGS.get(self.warn_code), "duration_s": sec - self.warn_since})
            if wc:
                self.emit({"ts": sec, "type": "WARNING_ON", "source": "can",
                           "severity": "critical" if wc in (1, 2) else "warning", "code": WARNINGS.get(wc)})
            self.warn_code, self.warn_since = wc, sec

        can_age = sec - last["can"]
        other_age = min(sec - last["someip"], sec - last["vhal"])
        if self.gap_from is None and can_age > FRESH_S and other_age < 3:
            self.gap_from = last["can"]
        elif self.gap_from is not None and can_age < 2:
            self.emit({"ts": self.gap_from, "type": "DATA_GAP", "source": "can", "severity": "warning",
                       "start": clock(self.gap_from), "end": clock(last["can"]),
                       "duration_min": round((last["can"] - self.gap_from) / 60, 1),
                       "note": "CAN frames missing while other sources were active; values in this window are unknown"})
            self.gap_from = None

        if mode:
            self._anomalies(sec, s, mode)
        self.prev_sec = sec

    # ---------------------------------------------------------------- anomalies
    def _anomaly(self, sec, kind, **attrs):
        key = (kind, attrs.get("wheel"), (self.trip or self.charge or {}).get("start"))
        if key in self.fired:
            return
        self.fired.add(key)
        self.emit({"ts": sec, "type": "ANOMALY", "kind": kind, "source": "miner", "severity": "warning", **attrs})

    def _anomalies(self, sec, s, mode):
        tyres = [s.get(p) for p in TYRES]
        if mode == "ON" and None not in tyres:
            for i, p in enumerate(tyres):
                others = sorted(tyres[:i] + tyres[i + 1:])
                k = f"tyre{i}"
                self.run[k] = self.run[k] + 1 if others[1] - p > 20 else 0
                if self.run[k] == 30:
                    self._anomaly(sec, "TYRE_PRESSURE_LOW", wheel=WHEEL_NAMES[i], value_kpa=round(p, 1),
                                  others_median_kpa=round(others[1], 1), signal=TYRES[i])
        pump, chill = s.get(PUMP), s.get(CHILLER)
        self.run["pump"] = self.run["pump"] + 1 if (chill == 1 and pump is not None and pump < 1500) else 0
        if self.run["pump"] == 10:
            self._anomaly(sec, "COOLANT_PUMP_LOW", value_rpm=round(pump), expected_rpm=3200, signal=PUMP,
                          context=mode)
        t, amb = s.get(TMAX), s.get(AMB)
        if t is not None and amb is not None:
            rise = t - amb                      # battery heating above the outside air: what really matters
            sub_mode = "DC" if s.get(DCA, 0) > 1 else ("AC" if mode == "CHARGING" else "DRIVE")
            ctx_fine = f"{mode}|{sub_mode}"
            for ctx in (ctx_fine, mode):
                n, mean, m2 = self.base.get(ctx, (0, 0.0, 0.0))
                if n >= 300:
                    std = max(math.sqrt(m2 / (n - 1)), 1.5)
                    z = (rise - mean) / std
                    if z > 4 and rise > 6:
                        self._anomaly(sec, "BATTERY_TEMP_ABOVE_CONTEXT", value_c=round(t, 1), ambient_c=round(amb, 1),
                                      rise_c=round(rise, 1), normal_rise_c=round(mean, 1), std_c=round(std, 2),
                                      z=round(z, 1), context=ctx, signal=TMAX)
                        return                  # do not learn from anomalous values
                    break
            t = rise
            for ctx in (ctx_fine, mode):        # Welford running mean/variance
                n, mean, m2 = self.base.get(ctx, (0, 0.0, 0.0))
                n += 1
                d = t - mean
                mean += d / n
                m2 += d * (t - mean)
                self.base[ctx] = (n, mean, m2)

    # ---------------------------------------------------------------- trips
    def _open_trip(self, sec, s):
        lv = s.get(LV)
        self.trip = {"start": sec, "odo0": s.get(ODO), "soc0": s.get(SOC), "energy": 0.0, "cov_s": 0, "dist_cov": 0.0,
                     "n": 0, "stop": 0, "ac": 0.0, "amb": 0.0, "amb_n": 0, "amb_max": -99, "city_km": 0.0,
                     "hwy_km": 0.0, "city_kwh": 0.0, "hwy_kwh": 0.0, "rain_s": 0, "tyre_min": [999.0] * 4,
                     "tmax": -99, "lv_wake": lv, "gap_s": 0, "dist_gap": 0.0, "vmax": 0.0, "last": s}
        if lv is not None and lv < 12.3:
            self._anomaly(sec, "LV_LOW_AT_WAKE", value_v=round(lv, 2), threshold_v=12.3, signal=LV)

    def _update_trip(self, sec, s, last):
        tr = self.trip
        tr["n"] += 1
        tr["last"], tr["end"] = s, sec
        v = s.get(SPEED, 0.0)
        tr["vmax"] = max(tr["vmax"], v)
        tr["stop"] += v < 5
        can_fresh = sec - last["can"] < 2
        pw = s.get(PACKW)
        if pw is not None and can_fresh:
            kwh = pw / 3.6e6
            tr["energy"] += kwh
            tr["cov_s"] += 1
            tr["dist_cov"] += v / 3600
            road = s.get(ROAD, 0)
            if road == 1:
                tr["hwy_km"] += v / 3600
                tr["hwy_kwh"] += kwh
            else:
                tr["city_km"] += v / 3600
                tr["city_kwh"] += kwh
            tr["ac"] += s.get(COMP, 0.0) / 3600
        a = s.get(AMB)
        if a is not None:
            tr["amb"] += a
            tr["amb_n"] += 1
            tr["amb_max"] = max(tr["amb_max"], a)
        tr["rain_s"] += s.get(WIPER, 0) >= 2
        for i, p in enumerate(TYRES):
            if p in s:
                tr["tyre_min"][i] = min(tr["tyre_min"][i], s[p])
        if TMAX in s:
            tr["tmax"] = max(tr["tmax"], s[TMAX])

    def _close_trip(self, sec):
        tr, self.trip = self.trip, None
        last = tr["last"]
        odo_km = (last.get(ODO, 0) - tr["odo0"]) if tr["odo0"] is not None and ODO in last else None
        dist = odo_km if odo_km and odo_km > 0 else tr["dist_cov"] + tr["dist_gap"]
        cons = tr["energy"] / tr["dist_cov"] * 100 if tr["dist_cov"] > 0.5 else None
        self.emit({"ts": tr["start"], "type": "TRIP", "source": "miner", "severity": "info",
                   "end_ts": tr.get("end", sec), "start": clock(tr["start"]), "end": clock(tr.get("end", sec)),
                   "duration_min": round((tr.get("end", sec) - tr["start"]) / 60, 1), "distance_km": round(dist, 2),
                   "energy_kwh": round(tr["energy"], 3), "kwh_per_100km": round(cons, 2) if cons else None,
                   "avg_speed_kmh": round(tr["dist_cov"] / max(tr["cov_s"], 1) * 3600, 1),
                   "max_speed_kmh": round(tr["vmax"], 1),
                   "stop_go_pct": round(tr["stop"] / max(tr["n"], 1) * 100, 1), "ac_kwh": round(tr["ac"], 3),
                   "ac_share_pct": round(tr["ac"] / tr["energy"] * 100, 1) if tr["energy"] > 0 else None,
                   "ambient_avg_c": round(tr["amb"] / tr["amb_n"], 1) if tr["amb_n"] else None,
                   "ambient_max_c": round(tr["amb_max"], 1),
                   "city_km": round(tr["city_km"], 2), "highway_km": round(tr["hwy_km"], 2),
                   "city_kwh": round(tr["city_kwh"], 3), "highway_kwh": round(tr["hwy_kwh"], 3),
                   "rain_pct": round(tr["rain_s"] / max(tr["n"], 1) * 100, 1),
                   "tyre_min_kpa": [round(x, 1) for x in tr["tyre_min"]], "batt_tmax_c": round(tr["tmax"], 1),
                   "soc_start": tr["soc0"], "soc_end": last.get(SOC), "lv_at_wake_v": tr["lv_wake"],
                   "data_coverage_pct": round(tr["cov_s"] / max(tr["n"] + tr["gap_s"], 1) * 100, 1),
                   "data_gap_min": round(tr["gap_s"] / 60, 1)})

    # ---------------------------------------------------------------- charging
    def _open_charge(self, sec, s):
        self.charge = {"start": sec, "soc0": s.get(SOC), "kwh": 0.0, "dc": False, "tmax": -99, "amb": 0.0, "amb_n": 0,
                       "prev": sec, "last": s}

    def _update_charge(self, sec, s):
        c = self.charge
        dt_s = min(sec - c["prev"], 15) or 1
        c["prev"], c["last"], c["end"] = sec, s, sec
        pw = s.get(PACKW, 0.0)
        if pw < 0:
            c["kwh"] += -pw * dt_s / 3.6e6
        c["dc"] |= s.get(DCA, 0) > 1
        c["tmax"] = max(c["tmax"], s.get(TMAX, -99))
        if AMB in s:
            c["amb"] += s[AMB]
            c["amb_n"] += 1

    def _close_charge(self, sec):
        c, self.charge = self.charge, None
        self.emit({"ts": c["start"], "type": "CHARGE", "source": "miner", "severity": "info",
                   "end_ts": c.get("end", sec), "charger": "DC_FAST" if c["dc"] else "AC_HOME", "start": clock(c["start"]), "end": clock(c.get("end", sec)),
                   "duration_min": round((c.get("end", sec) - c["start"]) / 60, 1), "energy_added_kwh": round(c["kwh"], 2),
                   "soc_start": c["soc0"], "soc_end": c["last"].get(SOC), "batt_tmax_c": round(c["tmax"], 1),
                   "ambient_avg_c": round(c["amb"] / c["amb_n"], 1) if c["amb_n"] else None})

    def close_all(self, sec):
        if self.warn_code:                      # a new drive cycle starts with a clean cluster
            self.emit({"ts": sec, "type": "WARNING_OFF", "source": "can", "severity": "info",
                       "code": WARNINGS.get(self.warn_code), "duration_s": sec - self.warn_since})
            self.warn_code = 0
        if self.trip:
            self._close_trip(sec)
        if self.charge:
            self._close_charge(sec)
        self.fired = {k for k in self.fired if k[2] is None}
        self.run.clear()


# ============================================================================ black box
class BlackBox:
    def __init__(self, out_dir: Path | None, pre: int = 60, post: int = 10, writer=None):
        self.writer = writer            # optional callback(case_dict) -> e.g. encrypted store (Phase 3)
        self.dir = out_dir
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
        self.pre, self.post = pre, post
        self.buf: deque = deque(maxlen=pre + 1)
        self.logs: deque = deque(maxlen=400)
        self.events: deque = deque(maxlen=400)
        self.open: list = []
        self.written: list = []

    def push(self, sec, snap):
        self.buf.append((sec, snap))
        for c in self.open:
            if sec > c["t0"]:
                c["post"].append((sec, snap))
        done = [c for c in self.open if sec >= c["t0"] + self.post]
        for c in done:
            self._write(c)
        self.open = [c for c in self.open if c not in done]

    def note_log(self, rec):
        self.logs.append(rec)

    def note_event(self, e):
        self.events.append(e)

    def trigger(self, e):
        for c in self.open:
            if e["ts"] <= c["t0"] + self.post:
                c["triggers"].append(e)
                e["case_id"] = c["id"]
                return
        cid = f"case_{dt.datetime.fromtimestamp(e['ts'], IST):%Y%m%d_%H%M%S}_{e.get('kind') or e.get('code') or e['type']}"
        c = {"id": cid, "t0": e["ts"], "pre": [x for x in self.buf if x[0] >= e["ts"] - self.pre], "post": [],
             "triggers": [e]}
        e["case_id"] = cid
        self.open.append(c)

    def flush(self):
        for c in self.open:
            self._write(c)
        self.open = []

    def _write(self, c):
        rows = c["pre"] + c["post"]
        lo, hi = c["t0"] - self.pre, c["t0"] + self.post
        paths = sorted({p for _, s in rows for p in s})
        case = {"id": c["id"], "trigger_time": clock(c["t0"]), "window": [clock(lo), clock(hi)],
                "triggers": [{k: v for k, v in t.items() if k != "case_id"} for t in c["triggers"]],
                "events_in_window": [e for e in self.events if lo <= e["ts"] <= hi and e not in c["triggers"]],
                "logs_in_window": [dict(l, untrusted_text=True) for l in self.logs if lo <= l["ts"] <= hi],
                "signals": {"t": [r[0] for r in rows], **{p: [s.get(p) for _, s in rows] for p in paths}}}
        self.written.append(c["id"])
        if self.writer:
            self.writer(case, c["t0"])
        elif self.dir:
            with gzip.open(self.dir / f"{c['id']}.json.gz", "wt", encoding="utf-8") as f:
                json.dump(case, f, separators=(",", ":"), default=float)


# ============================================================================ engine
class Engine:
    def __init__(self, sinks=(), cases_dir: Path | None = None, case_writer=None):
        self.state: dict = {}
        self.sec = None
        self.last = {"can": -1e12, "someip": -1e12, "vhal": -1e12}
        self.sinks = list(sinks)
        self.miner = Miner(self._emit)
        self.bb = BlackBox(cases_dir, writer=case_writer)
        self.counts: Counter = Counter()
        self.events_by_type: Counter = Counter()
        self.paths: set = set()
        self.snapshots = 0
        self._eid = 0

    def _emit(self, e):
        self._eid += 1
        e = {"id": self._eid, **e}
        e.setdefault("time", clock(e["ts"]))
        if e["type"] in TRIGGERS:
            self.bb.trigger(e)
        self.bb.note_event(e)
        self.events_by_type[e["type"] if e["type"] != "ANOMALY" else f"ANOMALY:{e['kind']}"] += 1
        for k in self.sinks:
            k.on_event(e)

    def feed(self, rec):
        ts, src, kind, payload = rec
        sec = int(ts)
        if self.sec is None:
            self.sec = sec
        if sec > self.sec:
            self._tick(self.sec)
            self.sec = sec
        self.counts[src] += 1
        if kind == "sig":
            self.last[src] = ts
            pr = PRIO[src]
            for path, v in payload:
                cur = self.state.get(path)
                if cur is None or pr >= cur[2] or ts - cur[1] > 3:
                    self.state[path] = (v, ts, pr)
        elif kind == "evt":
            self.miner.on_event(ts, src, payload)
        elif kind == "log":
            out = self.miner.on_log(ts, payload)
            self.bb.note_log(out)
            for k in self.sinks:
                k.on_log(out)

    def _tick(self, sec):
        if max(self.last.values()) < sec:
            return
        snap = {p: v for p, (v, t, _) in self.state.items()
                if sec - t <= (STICKY_S if p in STICKY else FRESH_S)}
        if not snap:
            return
        self.snapshots += 1
        self.paths.update(snap)
        self.miner.on_snapshot(sec, snap, self.last)
        self.bb.push(sec, snap)
        for k in self.sinks:
            k.on_snapshot(sec, snap)

    def finish(self):
        if self.sec is not None:
            self._tick(self.sec)
        self.miner.close_all(self.sec or 0)
        self.bb.flush()
        for k in self.sinks:
            k.close()


# ============================================================================ simple sinks
class JsonlSink:
    """Phase 2 sink: events.jsonl + logs.jsonl. Phase 3 adds the tiered on-device store."""

    def __init__(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.ev = open(out_dir / "events.jsonl", "w", encoding="utf-8")
        self.lg = open(out_dir / "logs.jsonl", "w", encoding="utf-8")

    def on_snapshot(self, sec, snap):
        pass

    def on_event(self, e):
        self.ev.write(json.dumps(e, default=float) + "\n")

    def on_log(self, rec):
        self.lg.write(json.dumps(rec) + "\n")

    def close(self):
        self.ev.close()
        self.lg.close()


def describe_power_state(v) -> str:
    return POWER_STATE.get(int(v), "?")
