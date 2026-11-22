"""Listen to the CAN bus and print a decoded one-line dashboard every second. Receive-only: it never sends.

  python -m twin.monitor
"""
from __future__ import annotations

import argparse
import datetime as dt
import time

from . import dbc as DBC
from . import world as W
from .bus import open_bus


def monitor(seconds: float | None = None, interface=None, channel=None):
    db = DBC.load()
    bus = open_bus(interface, channel)
    state: dict = {}
    frames, last_print, start = 0, 0.0, time.time()
    print("listening (receive-only). Ctrl+C to stop.")
    try:
        while seconds is None or time.time() - start < seconds:
            msg = bus.recv(timeout=0.5)
            if msg is None:
                continue
            frames += 1
            try:
                state.update(db.decode_message(msg.arbitration_id, msg.data, decode_choices=True))
            except KeyError:
                continue
            if time.time() - last_print >= 1.0 and "VehicleSpeed" in state and "SoC" in state:
                last_print = time.time()
                clock = dt.datetime.now(W.IST).strftime("%H:%M:%S")
                warn = state.get("WarningCode", "NONE")
                print(f"{clock} | {state.get('VehicleSpeed', 0):6.1f} km/h | SoC {state.get('SoC', 0):5.1f}% | "
                      f"batt {state.get('TempMax', 0):4.1f}C | AC {state.get('CompressorPower', 0):4.2f} kW | "
                      f"wiper {state.get('WiperMode', '-')!s:<8} | warning {warn} | frames {frames:,}")
    except KeyboardInterrupt:
        pass
    finally:
        bus.shutdown()
    return frames


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float)
    ap.add_argument("--interface")
    ap.add_argument("--channel")
    a = ap.parse_args()
    monitor(a.seconds, a.interface, a.channel)
