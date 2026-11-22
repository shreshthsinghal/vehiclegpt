"""Listen-only adapters. Each turns one native source into a time-ordered stream of tuples:

    (ts, source, kind, payload)
      kind "sig": payload = list[(vss_path, value)]       measurements
      kind "evt": payload = dict                          discrete events (ADAS event, DTC record)
      kind "log": payload = dict(proc, pid, sev, msg)     system log line (UNTRUSTED text)

File readers are used for backfill (recordings); live readers (CAN bus, SOME/IP over UDP) for the demo.
"""
from __future__ import annotations

import datetime as dt
import gzip
import json
import re
import socket
import struct
from pathlib import Path
from typing import Iterator

import can

from ..config import dbc, mapping

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
SOMEIP_HEADER = struct.Struct(">HHIHHBBBB")


# ----------------------------------------------------------------------------- CAN
class CanDecoder:
    def __init__(self):
        self.db = dbc()
        sigmap = mapping()["can"]["signals"]
        self.table = {}
        for m in self.db.messages:
            self.table[m.frame_id] = (m, [(s.name, sigmap[s.name]) for s in m.signals if s.name in sigmap])
        self.unknown = 0

    def decode(self, msg: can.Message):
        entry = self.table.get(msg.arbitration_id)
        if entry is None:
            self.unknown += 1
            return None
        m, pairs = entry
        vals = m.decode(msg.data, decode_choices=False)
        return [(path, float(vals[name])) for name, path in pairs]


def can_files(paths: list[Path], dec: CanDecoder | None = None) -> Iterator[tuple]:
    """Daily BLF recordings, in order (each file is time-ordered and files do not overlap)."""
    dec = dec or CanDecoder()
    for path in paths:
        for msg in can.BLFReader(str(path)):
            out = dec.decode(msg)
            if out:
                yield (msg.timestamp, "can", "sig", out)


def can_live(bus, dec: CanDecoder | None = None) -> Iterator[tuple]:
    """bus must be a safety.guards.ListenOnlyBus. Live data is stamped with this ECU's clock on arrival,
    so every source shares one time base (as in a real vehicle)."""
    import time
    dec = dec or CanDecoder()
    for msg in bus:
        out = dec.decode(msg)
        if out:
            yield (time.time(), "can", "sig", out)


# ----------------------------------------------------------------------------- SOME/IP
class SomeIpDecoder:
    def __init__(self):
        m = mapping()
        self.svc = m["someip"]
        self.reasons = m["adas_reasons"]
        self.bad = 0

    def decode(self, ts: float, raw: bytes):
        if len(raw) < 16:
            self.bad += 1
            return None
        service, event, length, _client, _session, proto, _iface, mtype, _rc = SOMEIP_HEADER.unpack_from(raw)
        if proto != 1 or length != len(raw) - 8:
            self.bad += 1
            return None
        spec = self.svc.get(service, {}).get(event)
        if spec is None:
            return None
        vals = struct.unpack(spec["format"], raw[16:16 + struct.calcsize(spec["format"])])
        if spec.get("event") == "ADAS_EVENT":
            code, speed, conf = vals
            return (ts, "someip", "evt", {"type": "ADAS_LKA_OFF" if code & 0x80 else "ADAS_DISENGAGEMENT",
                                           "reason": self.reasons.get(code & 0x7F, "UNKNOWN"),
                                           "speed_kmh": speed / 100, "lane_conf": conf})
        return (ts, "someip", "sig", [(p, v * s) for (p, s), v in zip(spec["fields"], vals)])


def someip_file(path: Path) -> Iterator[tuple]:
    dec = SomeIpDecoder()

    def conv(line):
        r = json.loads(line)
        return dec.decode(r["ts"], bytes.fromhex(r["hex"]))
    yield from _stream(path, conv)


def someip_live(port: int = 30501) -> Iterator[tuple]:
    """Receive SOME/IP notifications on UDP (receive-only socket)."""
    dec = SomeIpDecoder()
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.bind(("127.0.0.1", port))
    except OSError as exc:
        raise SystemExit(f"UDP port {port} is busy - is another `sense.ingest --live` already running? ({exc})")
    s.settimeout(1.0)
    while True:
        try:
            raw, _ = s.recvfrom(2048)
        except socket.timeout:
            yield None
            continue
        import time
        out = dec.decode(time.time(), raw)
        if out:
            yield out


# ----------------------------------------------------------------------------- VHAL
GEAR_FROM_VHAL = {8: 3.0, 4: 0.0, 2: 1.0, 1: 2.0}        # VehicleGear -> our gear codes (P0 R1 N2 D3)


def vhal_file(path: Path) -> Iterator[tuple]:
    m = mapping()["vhal"]
    q = mapping()["privacy"]["coarsen_degrees"]

    def conv(line):
        r = json.loads(line)
        spec = m.get(r["id"].lower())
        if spec is None:
            return None
        vss, scale = spec
        v = r["value"]
        if vss.endswith("SelectedGear"):
            v = GEAR_FROM_VHAL.get(int(v), 0.0)
        elif vss.startswith("Vehicle.CurrentLocation"):
            v = round(round(float(v) / q) * q, 4)              # privacy: ~1 km grid
        else:
            v = float(v) * scale
        return (r["ts"], "vhal", "sig", [(vss, float(v))])
    yield from _stream(path, conv)


# ----------------------------------------------------------------------------- QNX logs
SLOG = re.compile(r"^(\w{3} \d{2} \d{2}:\d{2}:\d{2}\.\d{3})\s+(\S+?)\.(\d+)\s+\S+\s+(\d)\s+(.*)$")


def qnx_file(path: Path, year: int = 2026) -> Iterator[tuple]:
    def conv(line):
        mt = SLOG.match(line.rstrip("\n"))
        if not mt:
            return None
        stamp = dt.datetime.strptime(f"{year} {mt.group(1)}", "%Y %b %d %H:%M:%S.%f").replace(tzinfo=IST)
        return (stamp.timestamp(), "qnx", "log", {"proc": mt.group(2), "pid": int(mt.group(3)),
                                                  "sev": int(mt.group(4)), "msg": mt.group(5)})
    yield from _stream(path, conv)


# ----------------------------------------------------------------------------- UDS
def uds_file(path: Path) -> Iterator[tuple]:
    def conv(line):
        r = json.loads(line)
        st = int(r["status"], 16)
        return (r["ts"], "uds", "evt", {"type": "DTC", "code": r["dtc"], "ecu": r["ecu"], "dtc_hex": r["dtc_hex"],
                                         "status": st, "test_failed": bool(st & 1), "confirmed": bool(st & 8)})
    yield from _stream(path, conv)


# ----------------------------------------------------------------------------- helpers
def _stream(path: Path, loader, horizon: float = 3 * 3600) -> Iterator[tuple]:
    """Yield records in time order. Recordings are written session by session, so a few events can be
    slightly out of order (by at most one session); a small re-ordering heap with a 3-hour horizon fixes that."""
    import heapq
    opener = gzip.open if path.suffix == ".gz" else open
    heap, newest, n = [], float("-inf"), 0
    with opener(path, "rt", encoding="utf-8") as f:
        for raw in f:
            rec = loader(raw)
            if rec is None:
                continue
            n += 1
            heapq.heappush(heap, (rec[0], n, rec))
            newest = max(newest, rec[0])
            while heap and heap[0][0] < newest - horizon:
                yield heapq.heappop(heap)[2]
    while heap:
        yield heapq.heappop(heap)[2]
