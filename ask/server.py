"""VehicleGPT HMI server. Offline, loopback-only, standard-library HTTP + one static page.

  python -m ask.server            # then open http://127.0.0.1:8765
Roles (role-based access): driver (no PIN), service, fleet, validation (PINs printed at start-up,
stored in data/store/roles.json). The driver sees answers; only service sees raw timelines, black boxes and exports.
"""
from __future__ import annotations

import datetime as dt
import io
import json
import os
import secrets
import threading
import time
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from reason.dates import IST, fmt_date, fmt_time
from reason.engine import VehicleGPT
from reason.playbooks import SIGNALS, TYRE_COLS, WARN_NAMES
from remember.build import STORE
from safety.guards import COUNTERS
from ask import present, speech
import yaml
DTC_NAMES = yaml.safe_load((Path(__file__).resolve().parents[1] / "sense" / "dtc_dictionary.yaml").read_text())

STATIC = Path(__file__).with_name("static")
ROLES_FILE = STORE.parent / "roles.json"
PORT = int(os.environ.get("VGPT_PORT", "8765"))
SERVICE_SIGNALS = ["speed", "battery_temp", "state_of_charge", "coolant_pump", "ac_power", "ambient_temp",
                   "tyre_rear_left", "lv_voltage"]
EXTRA_SIGNALS = {"lane_confidence": ("Vehicle.Private.ADAS.LaneMarkingConfidence", "lane-marking confidence", "%"),
                 "camera_visibility": ("Vehicle.Private.Camera.Front.Visibility", "camera visibility", "%"),
                 "power_limit": ("Vehicle.Private.TractionBattery.PowerLimit.Discharge", "drive power limit", "kW")}
ALL_SIGNALS = {**SIGNALS, **EXTRA_SIGNALS}


def load_roles() -> dict:
    if ROLES_FILE.exists():
        return json.loads(ROLES_FILE.read_text())
    roles = {"driver": "", "service": f"{secrets.randbelow(10000):04d}", "fleet": f"{secrets.randbelow(10000):04d}",
             "validation": f"{secrets.randbelow(10000):04d}", "fleet_sharing_opt_in": False}
    ROLES_FILE.write_text(json.dumps(roles, indent=2))
    return roles


def llm_up(url="http://127.0.0.1:8080") -> bool:
    try:
        return urllib.request.urlopen(url + "/health", timeout=1).status == 200
    except Exception:          # noqa: BLE001
        return False


class App:
    def __init__(self, store_path: Path = STORE):
        self.lock = threading.Lock()
        planner = None
        if llm_up():
            from reason.planner import SlmPlanner
            planner = SlmPlanner()
        self.vg = VehicleGPT(store_path, planner=planner)
        self.st = self.vg.store
        self.roles = load_roles()
        for q in ("why has my range dropped", "what happened before the warning"):   # warm caches before serving
            self.vg.ask(q)
        self.started = time.time()

    # ------------------------------------------------------------------ helpers
    def counters(self) -> dict:
        ext = None
        rss = None
        try:
            import psutil
            p = psutil.Process()
            rss = round(p.memory_info().rss / 1e6, 1)
            conns = p.net_connections(kind="inet") if hasattr(p, "net_connections") else p.connections(kind="inet")
            ext = sum(1 for c in conns if c.raddr and not str(c.raddr.ip).startswith(("127.", "::1")))
        except Exception:      # noqa: BLE001
            pass
        return {"can_tx_attempts": COUNTERS["tx_attempts"], "external_connections": ext, "egress_bytes": 0 if ext == 0 else None,
                "process_rss_mb": rss, "model_on": self.vg.planner is not None}

    def status(self) -> dict:
        with self.lock:
            a = self.vg.ask("how is my car right now")
            now = self.vg.ctx.now
            warns = self.st.events(["WARNING_ON", "WARNING_OFF"], now - 3 * 86400, now + 1)
            active = {}
            for e in warns:
                if e["type"] == "WARNING_ON":
                    active[e["code"]] = e
                else:
                    active.pop(e["code"], None)
            sets = self.st.events(["DTC_SET"])
            passes = self.st.events(["DTC_PASSED"])
            dtcs = [e["code"] for e in sets if not any(p["code"] == e["code"] and p["ts"] > e["ts"] for p in passes)]
            _, spd = self.st.series(SIGNALS["speed"][0], now - 120, now + 1)
        vals = {f.name: f.value for f in a.facts}
        names = DTC_NAMES
        return {"vehicle_clock": fmt_time(now), "soc": vals.get("state_of_charge"), "range_km": vals.get("range"),
                "battery_temp": vals.get("battery_temp"), "tyres": [vals.get(c) for c in TYRE_COLS],
                "warnings": [WARN_NAMES.get(c, c) for c in active], "active_dtcs": sorted(set(dtcs)),
                "active_dtc_names": {c: names.get(c, c) for c in set(dtcs)},
                "moving": bool(spd.size and spd[-1] > 2),
                "stt": speech.available(), **self.counters()}

    def ask(self, question: str) -> dict:
        with self.lock:
            ans = self.vg.ask(question[:300])
            d = ans.to_dict()
            d["ui"] = present.build(ans)
            return d

    def days(self) -> list:
        trips = self.st.table("trips")
        out = {}
        for t in trips:
            d = dt.datetime.fromtimestamp(t["start_ts"], IST).date().isoformat()
            out.setdefault(d, {"date": d, "trips": 0, "km": 0.0, "events": 0})
            out[d]["trips"] += 1
            out[d]["km"] = round(out[d]["km"] + t["distance_km"], 1)
        for e in self.st.events(["DTC_SET", "WARNING_ON", "ANOMALY", "ADAS_DISENGAGEMENT", "DATA_GAP"]):
            d = dt.datetime.fromtimestamp(e["ts"], IST).date().isoformat()
            if d in out:
                out[d]["events"] += 1
        return sorted(out.values(), key=lambda r: r["date"])

    def day(self, date: str, signals: list[str]) -> dict:
        d0 = dt.datetime.fromisoformat(date).replace(tzinfo=IST).timestamp()
        d1 = d0 + 86400
        with self.lock:
            trips = self.st.table("trips", d0, d1)
            ev = [e for e in self.st.events(None, d0, d1) if e["type"] not in ("TRIP", "CHARGE", "WARNING_OFF", "LOG_ALERT")]
            charges = self.st.table("charges", d0, d1)
            series = {}
            tier = "WARM (1 Hz)"
            for key in signals:
                if key not in ALL_SIGNALS:
                    continue
                path, label, unit = ALL_SIGNALS[key]
                ts, v = self.st.series(path, d0, d1)
                if not ts.size:
                    r = self.st.rollup(path, d0, d1)
                    ts, v, tier = r["minute"] * 60, r["mean"], "COLD (1-minute averages)"
                step = max(1, ts.size // 1500)
                series[key] = {"label": label, "unit": unit, "t": [int(x) for x in ts[::step]],
                               "v": [round(float(x), 2) for x in v[::step]]}
            logs = self.st.db.execute("SELECT ts, proc, sev, msg FROM logs WHERE ts BETWEEN ? AND ? AND sev <= 4 "
                                      "ORDER BY ts", (d0, d1)).fetchall()
        def ev_row(e):
            what = e.get("code") or e.get("reason") or ""
            return {"t": e["ts"], "time": fmt_time(e["ts"], False), "type": e["type"], "what": what,
                    "severity": e["severity"], "case_id": e.get("case_id"),
                    "detail": {k: v for k, v in e.items() if k in ("value_c", "rise_c", "value_kpa", "value_rpm", "value_v",
                                                                   "speed_kmh", "lane_conf", "duration_min", "ecu", "wheel")}}
        return {"date": date, "tier": tier, "series": series, "events": [ev_row(e) for e in ev],
                "trips": [{"start": fmt_time(t["start_ts"], False), "end": fmt_time(t["end_ts"], False),
                           "km": round(t["distance_km"], 1), "kwh_100km": t["kwh_per_100km"],
                           "coverage": t["data_coverage_pct"]} for t in trips],
                "charges": [{"start": fmt_time(c["start_ts"], False), "charger": c["charger"], "kwh": c["energy_added_kwh"],
                             "tmax": c["batt_tmax_c"]} for c in charges],
                "logs": [{"time": fmt_time(t, False), "proc": p, "sev": s, "msg": m} for t, p, s, m in logs]}

    def case(self, cid: str) -> dict:
        with self.lock:
            c = self.st.case(cid)
        keep = {k: c["signals"][v[0]] for k, v in ALL_SIGNALS.items() if v[0] in c["signals"]}
        return {"id": c["id"], "trigger_time": c["trigger_time"], "window": c["window"], "triggers": c["triggers"],
                "logs": c["logs_in_window"], "t": c["signals"]["t"], "signals": keep,
                "labels": {k: [ALL_SIGNALS[k][1], ALL_SIGNALS[k][2]] for k in keep},
                "n_signals_recorded": len(c["signals"]) - 1}

    def mdf4(self, cid: str) -> bytes:
        from asammdf import MDF, Signal
        with self.lock:
            c = self.st.case(cid)
        t = np.array(c["signals"]["t"], dtype=float)
        t = t - t[0]
        sigs = []
        for path, vals in c["signals"].items():
            if path == "t":
                continue
            v = np.array([np.nan if x is None else x for x in vals], dtype=float)
            if np.isnan(v).all():
                continue
            sigs.append(Signal(np.nan_to_num(v), t, name=path, comment=f"VehicleGPT case {cid}"))
        m = MDF(version="4.10")
        m.append(sigs, comment=f"VehicleGPT black-box {cid}, trigger {c['trigger_time']}")
        buf = io.BytesIO()
        tmp = STORE.parent / f"_{cid}.mf4"
        m.save(tmp, overwrite=True)
        data = tmp.read_bytes()
        tmp.unlink()
        return data

    def fleet(self, end: str | None) -> dict:
        now = self.vg.ctx.now
        end_day = dt.datetime.fromisoformat(end).replace(tzinfo=IST) if end else \
            dt.datetime.fromtimestamp(now, IST).replace(hour=0, minute=0, second=0, microsecond=0)
        e1 = end_day.timestamp() + 86400
        e0 = e1 - 7 * 86400
        with self.lock:
            trips = self.st.table("trips", e0, e1)
            ch = self.st.table("charges", e0, e1)
            ev = self.st.events(["WARNING_ON", "DTC_SET", "ANOMALY", "ADAS_DISENGAGEMENT", "DATA_GAP"], e0, e1)
        km = sum(t["distance_km"] for t in trips)
        kwh = sum(t["energy_kwh"] for t in trips)
        cnt = {}
        for e in ev:
            cnt[e["type"]] = cnt.get(e["type"], 0) + 1
        return {"week": f"{fmt_date(e0)} - {fmt_date(e1 - 1)}", "trips": len(trips), "km": round(km, 1),
                "energy_kwh": round(kwh, 1), "kwh_per_100km": round(kwh / km * 100, 1) if km else None,
                "charges": len(ch), "dc_fast_charges": sum(c["charger"] == "DC_FAST" for c in ch),
                "event_counts": cnt, "dtcs": sorted({e["code"] for e in ev if e["type"] == "DTC_SET"}),
                "data_coverage_pct": round(np.mean([t["data_coverage_pct"] for t in trips]), 1) if trips else None,
                "contains": "aggregates only: no location, no timestamps finer than a day, no driver identity",
                "sharing_opt_in": self.roles.get("fleet_sharing_opt_in", False)}

    def validation(self) -> dict:
        from eval.run_golden import main
        with self.lock:
            return main(quiet=True, store_path=self.st.path, out=STORE.parent / "golden_hmi.json")


# =========================================================================== HTTP
PROTECTED = {"/api/day": "service", "/api/days": "service", "/api/case": "service", "/api/export": "service",
             "/api/fleet": "fleet", "/api/validation": "validation"}


def make_handler(app: App):
    class H(BaseHTTPRequestHandler):
        server_version = "VehicleGPT/1.0"

        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json", extra=None):
            data = body if isinstance(body, bytes) else json.dumps(body, default=float).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; "
                             "script-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; media-src 'self'")
            self.send_header("Permissions-Policy", "microphone=(self)")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def _authorised(self, path) -> bool:
            need = PROTECTED.get(path)
            if not need:
                return True
            role, pin = self.headers.get("X-Role", ""), self.headers.get("X-Pin", "")
            return role == need and pin == app.roles.get(need)

        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
            if u.path in ("/", "/hmi", "/hmi.html"):
                return self._send(200, (STATIC / "hmi.html").read_bytes(), "text/html; charset=utf-8")
            if u.path in ("/console", "/index.html"):
                return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            if not self._authorised(u.path):
                return self._send(403, {"error": "this view needs the right role and PIN"})
            try:
                if u.path == "/api/status":
                    return self._send(200, app.status())
                if u.path == "/api/days":
                    return self._send(200, app.days())
                if u.path == "/api/day":
                    return self._send(200, app.day(q["date"], q.get("signals", ",".join(SERVICE_SIGNALS[:4])).split(",")))
                if u.path == "/api/case":
                    return self._send(200, app.case(q["id"]))
                if u.path == "/api/export":
                    data = app.mdf4(q["case"])
                    return self._send(200, data, "application/octet-stream",
                                      {"Content-Disposition": f'attachment; filename="{q["case"]}.mf4"'})
                if u.path == "/api/fleet":
                    return self._send(200, app.fleet(q.get("week_end")))
                if u.path == "/api/validation":
                    return self._send(200, app.validation())
                if u.path == "/api/signals":
                    return self._send(200, {k: [v[1], v[2]] for k, v in ALL_SIGNALS.items()})
            except Exception as exc:          # noqa: BLE001
                return self._send(500, {"error": f"{type(exc).__name__}: {exc}"})
            return self._send(404, {"error": "not found"})

        def do_POST(self):
            u = urllib.parse.urlparse(self.path)
            if u.path == "/api/listen":                       # on-device speech to text (audio never stored)
                n = int(self.headers.get("Content-Length", 0))
                if n > speech.MAX_BYTES:
                    return self._send(413, {"error": "recording too long"})
                try:
                    return self._send(200, speech.transcribe(self.rfile.read(n)))
                except Exception as exc:          # noqa: BLE001
                    return self._send(400, {"error": str(exc), "text": ""})
            n = min(int(self.headers.get("Content-Length", 0)), 4096)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._send(400, {"error": "bad json"})
            if u.path == "/api/ask":
                return self._send(200, app.ask(str(body.get("question", ""))))
            if u.path == "/api/login":
                role, pin = body.get("role"), body.get("pin", "")
                ok = role == "driver" or (role in app.roles and pin == app.roles.get(role))
                return self._send(200 if ok else 403, {"ok": ok})
            return self._send(404, {"error": "not found"})
    return H


def serve(port: int = PORT, store_path: Path = STORE):
    app = App(store_path)
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))
    print(f"VehicleGPT HMI on http://127.0.0.1:{port}  (loopback only; language model: "
          f"{'ON' if app.vg.planner else 'off'})")
    print(f"PINs  service {app.roles['service']}  fleet {app.roles['fleet']}  validation {app.roles['validation']}"
          f"   (stored in {ROLES_FILE})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return app


if __name__ == "__main__":
    serve()
