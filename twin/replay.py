"""Replay a recorded day onto a live CAN bus, as if the car were driving right now.

Examples
  python -m twin.replay --day 55 --start 18:00 --minutes 5            # real time
  python -m twin.replay --day 55 --start 17:45 --minutes 60 --speed 20  # 20x faster
Run `python -m twin.monitor` in a second terminal to watch the frames arrive.
"""
from __future__ import annotations

import argparse
import gzip
import heapq
import json
import socket
import time

import can

from . import world as W
from .bus import default_config, open_bus
from .generate import OUT


def replay(day: int, start: str, minutes: float, speed: float, interface=None, channel=None, quiet=False,
           someip_port: int | None = 30501) -> int:
    hms = [int(x) for x in start.split(":")] + [0]
    t0 = W.ts(day, hms[0], hms[1], hms[2])
    t1 = t0 + minutes * 60
    path = OUT / "can" / f"day{day:02d}.blf"
    if not path.exists():
        raise SystemExit(f"{path} not found - run `python -m twin.generate` first")
    bus = open_bus(interface, channel)
    cfg = default_config() if interface is None else {"interface": interface, "channel": channel}
    if not quiet:
        print(f"replaying {path.name} {start} +{minutes:g} min at {speed:g}x onto {cfg['interface']}:{cfg['channel']}")
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if someip_port else None
    sip = []
    if udp:
        with gzip.open(OUT / "someip" / "someip.jsonl.gz", "rt") as f:
            for line in f:
                r = json.loads(line)
                if t0 <= r["ts"] <= t1:
                    sip.append((r["ts"], bytes.fromhex(r["hex"])))
        sip.sort()
    can_iter = ((m.timestamp, "can", m) for m in can.BLFReader(str(path)) if t0 <= m.timestamp <= t1)
    sent, wall0, first = 0, time.perf_counter(), None
    try:
        for ts, kind, item in heapq.merge(can_iter, ((t, "sip", raw) for t, raw in sip), key=lambda x: x[0]):
            first = first if first is not None else ts
            due = (ts - first) / speed
            lag = due - (time.perf_counter() - wall0)
            if lag > 0:
                time.sleep(lag)
            if kind == "can":
                bus.send(item)
            else:   # SOME/IP rides on UDP in real cars too
                udp.sendto(item, ("127.0.0.1", someip_port))
            sent += 1
    except KeyboardInterrupt:
        pass
    finally:
        bus.shutdown()
    if not quiet:
        print(f"sent {sent:,} CAN frames + SOME/IP packets in {time.perf_counter() - wall0:.1f} s")
    return sent


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--day", type=int, default=55)
    ap.add_argument("--start", default="18:00", help="local time HH:MM or HH:MM:SS")
    ap.add_argument("--minutes", type=float, default=5)
    ap.add_argument("--speed", type=float, default=1.0, help="1 = real time, 10 = ten times faster")
    ap.add_argument("--interface")
    ap.add_argument("--channel")
    a = ap.parse_args()
    replay(a.day, a.start, a.minutes, a.speed, a.interface, a.channel)
