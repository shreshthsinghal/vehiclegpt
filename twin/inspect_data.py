"""Show what the digital twin produced: one decoded sample from every source, in plain text.

  python -m twin.inspect_data
"""
from __future__ import annotations

import datetime as dt
import gzip
import json
import struct

import can

from . import dbc as DBC
from . import world as W
from .generate import OUT


def _clock(t):
    return dt.datetime.fromtimestamp(t, W.IST).strftime("%Y-%m-%d %H:%M:%S")


def main():
    man = json.loads((OUT / "manifest.json").read_text())
    print("=== MANIFEST ===")
    for k, v in man.items():
        print(f"  {k:<18} {v:,}" if isinstance(v, int) else f"  {k:<18} {v}")

    print("\n=== CAN (day 55, 18:04:20 - just before a rain disengagement) decoded with twin/nexus_ev.dbc ===")
    db = DBC.load()
    target = W.ts(55, 18, 4, 20)
    shown = set()
    for msg in can.BLFReader(str(OUT / "can" / "day55.blf")):
        if msg.timestamp < target or msg.arbitration_id in shown:
            continue
        shown.add(msg.arbitration_id)
        m = db.get_message_by_frame_id(msg.arbitration_id)
        dec = m.decode(msg.data, decode_choices=True)
        preview = ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in list(dec.items())[:5])
        print(f"  0x{msg.arbitration_id:03X} {m.name:<16} {'FD ' if msg.is_fd else '   '}raw={msg.data[:8].hex()}..  {preview}")
        if len(shown) >= 14:
            break

    print("\n=== SOME/IP (ADAS service 0x1200, disengagement events on day 55) ===")
    reasons = {1: "LANE_MARKINGS_LOST", 2: "CAMERA_UNAVAILABLE", 3: "DRIVER_OVERRIDE", 4: "ODD_EXIT"}
    with gzip.open(OUT / "someip" / "someip.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line)
            raw = bytes.fromhex(r["hex"])
            svc, meth, length, client, sess, pv, iv, mt, rc = struct.unpack(">HHIHHBBBB", raw[:16])
            if svc == 0x1200 and meth == 0x8002 and W.day_of(r["ts"]) == 55:
                code, spd, conf = struct.unpack(">BHB", raw[16:20])
                if code & 0x80:
                    continue
                print(f"  {_clock(r['ts'])}  svc=0x{svc:04X} event=0x{meth:04X} len={length} type=0x{mt:02X}  "
                      f"reason={reasons[code & 0x7F]} speed={spd / 100:.1f} km/h lane_conf={conf}%")

    print("\n=== Android VHAL (first 4 events) ===")
    with gzip.open(OUT / "vhal" / "vhal.jsonl.gz", "rt") as f:
        for _, line in zip(range(4), f):
            print("  " + line.strip())

    print("\n=== QNX slog2 (around the camera fault, 28 Nov 18:42) ===")
    with gzip.open(OUT / "qnx" / "slog2.log.gz", "rt") as f:
        for line in f:
            if line.startswith("Nov 28 18:4"):
                print("  " + line.rstrip())

    print("\n=== UDS diagnostic trouble codes ===")
    for line in open(OUT / "uds" / "dtc.jsonl"):
        r = json.loads(line)
        print(f"  {_clock(r['ts'])}  {r['ecu']:<5} {r['dtc']} (0x{r['dtc_hex']}) status {r['status']}")

    print("\nAnswer key (never read by VehicleGPT): data/twin/truth/truth.json")


if __name__ == "__main__":
    main()
