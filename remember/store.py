"""REMEMBER: the on-device vehicle memory. One SQLite file, four tiers.

  HOT     last 10 min of 1 Hz snapshots, in RAM (a ring buffer)          -> "right now" questions
  WARM    1 Hz history, 10-minute compressed blocks per signal, 14 days -> detailed recent questions
  COLD    1-minute min/max/mean/count per signal, kept 400 days         -> trends over weeks
  PINNED  black-box case files (-60 s/+10 s), never expire               -> "what happened before X?"
plus small tables that are kept for the life of the vehicle: events, trips, charges, logs (+ FTS5 keyword
search), DTC descriptions. Time-series blocks and case files are encrypted with AES-256-GCM.

Key-off flush: data still waiting in RAM is written to flash as soon as the car goes quiet (>120 s without
data) or the process stops, so nothing is lost at power-down.
"""
from __future__ import annotations

import gzip
import json
import os
import sqlite3
import time
from collections import deque
from pathlib import Path

import numpy as np
import yaml

from sense.config import ROOT, dbc, mapping

from .codec import Cipher, decode, decode_rollup, encode, encode_rollup

# Plain-language descriptions added to the search text of events (warnings, anomaly kinds, ADAS reasons).
EVENT_WORDS = {
    "BATTERY_TEMP_HIGH": "battery temperature high overheating hot thermal", "REDUCED_POWER": "reduced power limited derated",
    "CHECK_TYRE_PRESSURE": "tyre tire pressure low flat puncture", "LOW_12V_BATTERY": "12 volt battery low weak",
    "CAMERA_UNAVAILABLE": "front camera unavailable not working", "BATTERY_TEMP_ABOVE_CONTEXT": "battery hotter than normal overheating",
    "COOLANT_PUMP_LOW": "coolant pump slow cooling weak", "TYRE_PRESSURE_LOW": "tyre tire pressure low flat puncture",
    "LV_LOW_AT_WAKE": "12 volt battery weak at start", "LANE_MARKINGS_LOST": "lane markings not visible rain",
    "DRIVER_OVERRIDE": "driver steered", "DATA_GAP": "missing data recording gap", "TRIP": "drive journey",
    "CHARGE": "charging session"}
BLOCK_S = 600
WARM_DAYS = 14
COLD_DAYS = 400
HOT_S = 600

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS signals(id INTEGER PRIMARY KEY, path TEXT UNIQUE, quantum REAL);
CREATE TABLE IF NOT EXISTS warm(sid INTEGER, t0 INTEGER, t1 INTEGER, n INTEGER, data BLOB, PRIMARY KEY(sid, t0)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS cold(sid INTEGER, t0 INTEGER, t1 INTEGER, n INTEGER, data BLOB, PRIMARY KEY(sid, t0)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, ts INTEGER, type TEXT, severity TEXT, code TEXT,
                                  case_id TEXT, attrs TEXT);
CREATE INDEX IF NOT EXISTS events_type_ts ON events(type, ts);
CREATE INDEX IF NOT EXISTS events_ts ON events(ts);
CREATE VIRTUAL TABLE IF NOT EXISTS events_fts USING fts5(text, tokenize='porter');
CREATE TABLE IF NOT EXISTS trips(id INTEGER PRIMARY KEY, start_ts INTEGER, end_ts INTEGER, distance_km REAL,
    energy_kwh REAL, kwh_per_100km REAL, avg_speed_kmh REAL, max_speed_kmh REAL, stop_go_pct REAL, ac_kwh REAL,
    ac_share_pct REAL, ambient_avg_c REAL, ambient_max_c REAL, city_km REAL, highway_km REAL, city_kwh REAL,
    highway_kwh REAL, rain_pct REAL, tyre_fl_min REAL, tyre_fr_min REAL, tyre_rl_min REAL, tyre_rr_min REAL,
    batt_tmax_c REAL, soc_start REAL, soc_end REAL, lv_at_wake_v REAL, data_coverage_pct REAL, data_gap_min REAL);
CREATE INDEX IF NOT EXISTS trips_ts ON trips(start_ts);
CREATE TABLE IF NOT EXISTS charges(id INTEGER PRIMARY KEY, start_ts INTEGER, end_ts INTEGER, charger TEXT,
    energy_added_kwh REAL, soc_start REAL, soc_end REAL, batt_tmax_c REAL, ambient_avg_c REAL, duration_min REAL);
CREATE TABLE IF NOT EXISTS logs(id INTEGER PRIMARY KEY, ts INTEGER, proc TEXT, sev INTEGER, template_id INTEGER,
                                template TEXT, msg TEXT);
CREATE INDEX IF NOT EXISTS logs_ts ON logs(ts);
CREATE VIRTUAL TABLE IF NOT EXISTS logs_fts USING fts5(msg, template, proc, content='logs', content_rowid='id',
                                                       tokenize='porter');
CREATE TABLE IF NOT EXISTS cases(id TEXT PRIMARY KEY, t0 INTEGER, triggers TEXT, bytes INTEGER, file TEXT,
                                 pinned INTEGER DEFAULT 1);
CREATE TABLE IF NOT EXISTS dtc_dict(code TEXT PRIMARY KEY, description TEXT);
"""


def _quanta() -> dict:
    """Storage resolution of every signal = the resolution its source actually has (lossless)."""
    q = {}
    sigmap = mapping()["can"]["signals"]
    for m in dbc().messages:
        for s in m.signals:
            if s.name in sigmap:
                q[sigmap[s.name]] = float(s.scale) if s.scale and s.scale < 1 else 1.0
    for evs in mapping()["someip"].values():
        for spec in evs.values():
            for path, scale in spec.get("fields", []):
                q.setdefault(path, float(scale) if scale < 1 else 1.0)
    for path, _scale in mapping()["vhal"].values():
        q.setdefault(path, 0.01)
    return q


def load_key(path: Path) -> bytes:
    """Stand-in for an HSM/TPM-sealed key: created once, kept beside the store, never in git."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(os.urandom(32))
    return path.read_bytes()


class Store:
    def __init__(self, path: Path, key: bytes | None = None, fresh: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if fresh:
            for p in self.path.parent.glob(self.path.name + "*"):
                p.unlink()
        self.db = sqlite3.connect(str(self.path), check_same_thread=False)
        self.db.executescript(SCHEMA)
        self.cipher = Cipher(key)
        self.cases_dir = self.path.parent / "cases"
        self.cases_dir.mkdir(exist_ok=True)
        self._sid: dict[str, int] = dict(self.db.execute("SELECT path, id FROM signals"))
        self._q = _quanta()
        if not self.db.execute("SELECT 1 FROM dtc_dict LIMIT 1").fetchone():
            d = yaml.safe_load((ROOT / "sense" / "dtc_dictionary.yaml").read_text(encoding="utf-8"))
            self.db.executemany("INSERT INTO dtc_dict VALUES (?,?)", d.items())
        self.db.execute("INSERT OR REPLACE INTO meta VALUES ('encrypted', ?)", ("1" if key else "0",))
        self.db.commit()

    # ------------------------------------------------------------------ registry
    def sid(self, path: str) -> int:
        i = self._sid.get(path)
        if i is None:
            cur = self.db.execute("INSERT INTO signals(path, quantum) VALUES (?,?)", (path, self._q.get(path, 0.001)))
            i = self._sid[path] = cur.lastrowid
        return i

    def quantum(self, path: str) -> float:
        return self._q.get(path, 0.001)

    # ------------------------------------------------------------------ writes
    def write_block(self, path: str, ts: np.ndarray, vals: np.ndarray):
        sid = self.sid(path)
        q = self.quantum(path)
        aad = f"{sid}:{int(ts[0])}".encode()
        self.db.execute("INSERT OR REPLACE INTO warm VALUES (?,?,?,?,?)",
                        (sid, int(ts[0]), int(ts[-1]), int(ts.size), self.cipher.seal(encode(ts, vals, q), aad)))
        minutes = ts // 60
        um, idx = np.unique(minutes, return_index=True)
        ends = np.append(idx[1:], ts.size)
        mn = np.array([vals[a:b].min() for a, b in zip(idx, ends)])
        mx = np.array([vals[a:b].max() for a, b in zip(idx, ends)])
        mean = np.array([vals[a:b].mean() for a, b in zip(idx, ends)])
        cnt = (ends - idx).astype(float)
        self.db.execute("INSERT OR REPLACE INTO cold VALUES (?,?,?,?,?)",
                        (sid, int(ts[0]), int(ts[-1]), int(um.size),
                         self.cipher.seal(encode_rollup(um, mn, mx, mean, cnt), aad + b":c")))

    def add_event(self, e: dict):
        attrs = {k: v for k, v in e.items() if k not in ("id", "ts", "type", "severity", "case_id")}
        code = e.get("code") or e.get("kind") or e.get("reason")
        self.db.execute("INSERT OR REPLACE INTO events VALUES (?,?,?,?,?,?,?)",
                        (e["id"], int(e["ts"]), e["type"], e["severity"], code, e.get("case_id"),
                         json.dumps(attrs, default=float)))
        desc = self.db.execute("SELECT description FROM dtc_dict WHERE code=?", (code,)).fetchone() if code else None
        words = [e["type"].replace("_", " "), str(code or "").replace("_", " "), desc[0] if desc else "",
                 EVENT_WORDS.get(str(code), ""), EVENT_WORDS.get(e["type"], ""),
                 str(e.get("wheel", "")), str(e.get("charger", "")).replace("_", " ")]
        if e["type"] == "LOG_ALERT":
            words.append(str(e.get("template", "")))
        self.db.execute("INSERT INTO events_fts(rowid, text) VALUES (?,?)", (e["id"], " ".join(w for w in words if w)))
        if e["type"] == "TRIP":
            t = e["tyre_min_kpa"]
            self.db.execute("INSERT INTO trips VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            (e["id"], int(e["ts"]), int(e["end_ts"]), e["distance_km"], e["energy_kwh"],
                             e["kwh_per_100km"], e["avg_speed_kmh"], e["max_speed_kmh"], e["stop_go_pct"], e["ac_kwh"],
                             e["ac_share_pct"], e["ambient_avg_c"], e["ambient_max_c"], e["city_km"], e["highway_km"],
                             e["city_kwh"], e["highway_kwh"], e["rain_pct"], *t, e["batt_tmax_c"], e["soc_start"],
                             e["soc_end"], e["lv_at_wake_v"], e["data_coverage_pct"], e["data_gap_min"]))
        elif e["type"] == "CHARGE":
            self.db.execute("INSERT INTO charges VALUES (?,?,?,?,?,?,?,?,?,?)",
                            (e["id"], int(e["ts"]), int(e["end_ts"]), e["charger"], e["energy_added_kwh"],
                             e["soc_start"], e["soc_end"], e["batt_tmax_c"], e["ambient_avg_c"], e["duration_min"]))

    def add_log(self, rec: dict):
        cur = self.db.execute("INSERT INTO logs(ts, proc, sev, template_id, template, msg) VALUES (?,?,?,?,?,?)",
                              (int(rec["ts"]), rec["proc"], rec["sev"], rec["template_id"], rec["template"], rec["msg"]))
        self.db.execute("INSERT INTO logs_fts(rowid, msg, template, proc) VALUES (?,?,?,?)",
                        (cur.lastrowid, rec["msg"], rec["template"], rec["proc"]))

    def add_case(self, case: dict, t0: float):
        raw = gzip.compress(json.dumps(case, separators=(",", ":"), default=float).encode())
        blob = self.cipher.seal(raw, case["id"].encode())
        f = self.cases_dir / f"{case['id']}.bin"
        f.write_bytes(blob)
        trig = ",".join(t.get("kind") or t.get("code") or t.get("reason") or t["type"] for t in case["triggers"])
        self.db.execute("INSERT OR REPLACE INTO cases VALUES (?,?,?,?,?,1)", (case["id"], int(t0), trig, len(blob), f.name))

    def commit(self):
        self.db.commit()

    # ------------------------------------------------------------------ retention
    def enforce_retention(self, now: float) -> dict:
        w = self.db.execute("DELETE FROM warm WHERE t1 < ?", (int(now - WARM_DAYS * 86400),)).rowcount
        c = self.db.execute("DELETE FROM cold WHERE t1 < ?", (int(now - COLD_DAYS * 86400),)).rowcount
        self.db.execute("INSERT OR REPLACE INTO meta VALUES ('retention_now', ?)", (str(int(now)),))
        self.db.commit()
        return {"warm_blocks_dropped": w, "cold_blocks_dropped": c}

    # ------------------------------------------------------------------ reads (used by REASON in Phase 4)
    def series(self, path: str, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
        """1 Hz values from the WARM tier (lossless)."""
        sid = self._sid.get(path)
        if sid is None:
            return np.zeros(0), np.zeros(0)
        ts_all, v_all = [], []
        for bt0, blob in self.db.execute("SELECT t0, data FROM warm WHERE sid=? AND t1>=? AND t0<=? ORDER BY t0",
                                         (sid, int(t0), int(t1))):
            ts, v = decode(self.cipher.open(blob, f"{sid}:{bt0}".encode()))
            m = (ts >= t0) & (ts <= t1)
            ts_all.append(ts[m])
            v_all.append(v[m])
        if not ts_all:
            return np.zeros(0), np.zeros(0)
        return np.concatenate(ts_all), np.concatenate(v_all)

    def rollup(self, path: str, t0: float, t1: float) -> dict:
        """Per-minute min/max/mean/count from the COLD tier (covers the full history)."""
        sid = self._sid.get(path)
        out = {"minute": [], "min": [], "max": [], "mean": [], "count": []}
        if sid is None:
            return {k: np.zeros(0) for k in out}
        for bt0, blob in self.db.execute("SELECT t0, data FROM cold WHERE sid=? AND t1>=? AND t0<=? ORDER BY t0",
                                         (sid, int(t0), int(t1))):
            m, mn, mx, mean, cnt = decode_rollup(self.cipher.open(blob, f"{sid}:{bt0}:c".encode()))
            keep = (m * 60 >= t0 - 59) & (m * 60 <= t1)
            for k, arr in zip(out, (m, mn, mx, mean, cnt)):
                out[k].append(arr[keep])
        return {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in out.items()}

    def events(self, types=None, t0=0, t1=4e9, code=None) -> list[dict]:
        sql, args = "SELECT id, ts, type, severity, code, case_id, attrs FROM events WHERE ts BETWEEN ? AND ?", [t0, t1]
        if types:
            sql += f" AND type IN ({','.join('?' * len(types))})"
            args += list(types)
        if code:
            sql += " AND code = ?"
            args.append(code)
        rows = self.db.execute(sql + " ORDER BY ts", args).fetchall()
        return [{"id": r[0], "ts": r[1], "type": r[2], "severity": r[3], "code": r[4], "case_id": r[5],
                 **json.loads(r[6])} for r in rows]

    def table(self, name: str, t0=0, t1=4e9) -> list[dict]:
        assert name in ("trips", "charges")
        cur = self.db.execute(f"SELECT * FROM {name} WHERE start_ts BETWEEN ? AND ? ORDER BY start_ts", (t0, t1))
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r)) for r in cur]

    def search_events(self, query: str, limit: int = 20) -> list[dict]:
        ids = [r[0] for r in self.db.execute("SELECT rowid FROM events_fts WHERE events_fts MATCH ? ORDER BY rank LIMIT ?",
                                             (query, limit))]
        if not ids:
            return []
        return [e for e in self.events() if e["id"] in set(ids)]

    def search_logs(self, query: str, limit: int = 20) -> list[dict]:
        cur = self.db.execute("SELECT l.ts, l.proc, l.sev, l.template_id, l.template, l.msg FROM logs_fts "
                              "JOIN logs l ON l.id = logs_fts.rowid WHERE logs_fts MATCH ? ORDER BY rank LIMIT ?",
                              (query, limit))
        return [dict(zip(("ts", "proc", "sev", "template_id", "template", "msg"), r), untrusted_text=True) for r in cur]

    def case(self, case_id: str) -> dict:
        row = self.db.execute("SELECT file FROM cases WHERE id=?", (case_id,)).fetchone()
        blob = (self.cases_dir / row[0]).read_bytes()
        return json.loads(gzip.decompress(self.cipher.open(blob, case_id.encode())))

    # ------------------------------------------------------------------ meaning search (sqlite-vec)
    def _vec(self):
        import sqlite_vec
        self.db.enable_load_extension(True)
        sqlite_vec.load(self.db)
        self.db.enable_load_extension(False)

    def index_meaning(self, embed_fn, dim: int) -> int:
        """Embed every event's searchable text into a vec0 table (run after ingest)."""
        self._vec()
        self.db.execute("DROP TABLE IF EXISTS events_vec")
        self.db.execute(f"CREATE VIRTUAL TABLE events_vec USING vec0(embedding float[{dim}])")
        rows = self.db.execute("SELECT rowid, text FROM events_fts").fetchall()
        import struct
        for rid, text in rows:
            v = embed_fn(text)
            self.db.execute("INSERT INTO events_vec(rowid, embedding) VALUES (?, ?)", (rid, struct.pack(f"{dim}f", *v)))
        self.db.commit()
        return len(rows)

    def search_meaning(self, vec, k: int = 8) -> list[tuple[int, float]]:
        import struct
        self._vec()
        return self.db.execute("SELECT rowid, distance FROM events_vec WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
                               (struct.pack(f"{len(vec)}f", *vec), k)).fetchall()

    def stats(self) -> dict:
        q = lambda s: self.db.execute(s).fetchone()
        warm = q("SELECT COUNT(*), COALESCE(SUM(n),0), COALESCE(SUM(LENGTH(data)),0), MIN(t0), MAX(t1) FROM warm")
        cold = q("SELECT COUNT(*), COALESCE(SUM(n),0), COALESCE(SUM(LENGTH(data)),0), MIN(t0), MAX(t1) FROM cold")
        cases = q("SELECT COUNT(*), COALESCE(SUM(bytes),0) FROM cases")
        self.db.commit()
        self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        files = sum(p.stat().st_size for p in self.path.parent.glob(self.path.name + "*"))
        return {"db_file_mb": round(files / 1e6, 2),
                "warm": {"blocks": warm[0], "points": warm[1], "payload_mb": round(warm[2] / 1e6, 2),
                         "bytes_per_point": round(warm[2] / max(warm[1], 1), 3), "days": round(((warm[4] or 0) - (warm[3] or 0)) / 86400, 1)},
                "cold": {"blocks": cold[0], "minutes": cold[1], "payload_mb": round(cold[2] / 1e6, 2),
                         "days": round(((cold[4] or 0) - (cold[3] or 0)) / 86400, 1)},
                "pinned_cases": {"count": cases[0], "kb_total": round(cases[1] / 1024, 1)},
                "events": q("SELECT COUNT(*) FROM events")[0], "trips": q("SELECT COUNT(*) FROM trips")[0],
                "charges": q("SELECT COUNT(*) FROM charges")[0], "logs": q("SELECT COUNT(*) FROM logs")[0],
                "signals": q("SELECT COUNT(*) FROM signals")[0],
                "encrypted": q("SELECT v FROM meta WHERE k='encrypted'")[0] == "1"}


# =========================================================================== sink (plugs into SENSE)
class StoreSink:
    """Receives SENSE output. HOT ring in RAM; WARM/COLD blocks flushed every 10 min and at key-off."""

    def __init__(self, store: Store):
        self.store = store
        self.hot: deque = deque(maxlen=HOT_S)
        self.buf: dict[str, tuple[list, list]] = {}
        self.window = None
        self.last_sec = None
        self.flushes = {"block": 0, "key_off": 0}
        self._n = 0

    def on_snapshot(self, sec, snap):
        if self.last_sec is not None and sec - self.last_sec > 120:
            self._flush("key_off")
        w = sec // BLOCK_S
        if self.window is not None and w != self.window:
            self._flush("block")
        self.window, self.last_sec = w, sec
        self.hot.append((sec, snap))
        for p, v in snap.items():
            b = self.buf.get(p)
            if b is None:
                b = self.buf[p] = ([], [])
            b[0].append(sec)
            b[1].append(v)

    def _flush(self, why):
        if not self.buf:
            return
        for p, (ts, vs) in self.buf.items():
            if ts:
                self.store.write_block(p, np.asarray(ts, np.int64), np.asarray(vs, np.float64))
        self.buf = {}
        self.flushes[why] += 1
        self.store.commit()

    def on_event(self, e):
        self.store.add_event(e)
        self._n += 1
        if self._n % 200 == 0:
            self.store.commit()

    def on_log(self, rec):
        self.store.add_log(rec)

    def write_case(self, case, t0):
        self.store.add_case(case, t0)

    def close(self):
        self._flush("key_off")
        self.store.commit()

    def latest(self) -> dict:
        return self.hot[-1][1] if self.hot else {}
