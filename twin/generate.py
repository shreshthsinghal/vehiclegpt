"""Generate 8 weeks of vehicle data in native wire formats, plus the truth labels used for grading.

Run:  python -m twin.generate                 (full 56 days, about 2-4 minutes)
      python -m twin.generate --days 7        (quick test)

Output (data/twin/):
  can/dayNN.blf          CAN + CAN-FD frames, Vector BLF format (what real loggers record), DBC: twin/nexus_ev.dbc
  someip/someip.jsonl.gz SOME/IP notifications: real 16-byte SOME/IP header + payload, hex encoded
  vhal/vhal.jsonl.gz     Android Automotive VHAL property events
  qnx/slog2.log.gz       QNX slog2info-style system log
  uds/dtc.jsonl          UDS ReadDTCInformation (0x19 -> 0x59) responses
  truth/                 ANSWER KEY - only the evaluation code may read this, never VehicleGPT itself
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import gzip
import json
import struct
import time
from pathlib import Path

import can
import numpy as np

from . import dbc as DBC
from . import world as W
from .model import P
from .simulate import CarState, Session, simulate_charge, simulate_trip

OUT = Path(__file__).resolve().parents[1] / "data" / "twin"
WARN_CODES = {v: k for k, v in DBC.WARNING.items()}
REASONS = {"LANE_MARKINGS_LOST": 1, "CAMERA_UNAVAILABLE": 2, "DRIVER_OVERRIDE": 3, "ODD_EXIT": 4}
DTC_PREFIX = {"P": 0b00, "C": 0b01, "B": 0b10, "U": 0b11}
SEED = 2026


# ======================================================================= helpers
def dtc_to_hex(code: str) -> str:
    """SAE J2012 DTC (e.g. 'P1A20') -> 3-byte UDS DTC (2 bytes + failure-type byte)."""
    first = (DTC_PREFIX[code[0]] << 14) | (int(code[1], 16) << 12) | int(code[2:], 16)
    return f"{first:04X}00"


def local(t: float) -> dt.datetime:
    return dt.datetime.fromtimestamp(t, W.IST)


def warning_series(sess: Session) -> np.ndarray:
    n = sess.t.size
    out = np.zeros(n, dtype=int)
    for e in sorted((e for e in sess.events if e["type"] == "warning"), key=lambda e: e["t"]):
        i = int(np.searchsorted(sess.t, e["t"]))
        out[i:i + int(e["duration"])] = WARN_CODES[e["code"]]
    return out


# ======================================================================= CAN writer
class CanWriter:
    def __init__(self, out_dir: Path):
        self.db = DBC.load()
        self.m = {m.name: m for m in self.db.messages}
        self.dir = out_dir / "can"
        self.dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(5)
        self.cell_off = rng.normal(0, 0.006, 96)
        self.cell_off[40] -= 0.015                       # cell 41 is slightly weak (realism)
        self.mod_off = rng.normal(0, 0.6, 32)
        self.mod_off[6] = 1.5                            # module 7 is the hottest
        self.writer, self.day, self.frames = None, None, 0

    def _open(self, day: int):
        if self.writer:
            self.writer.stop()
        self.writer = can.BLFWriter(str(self.dir / f"day{day:02d}.blf"))
        self.day = day

    def _send(self, ts: float, name: str, data: dict):
        msg = self.m[name]
        payload = msg.encode(data, strict=False)
        self.writer.on_message_received(can.Message(timestamp=ts, arbitration_id=msg.frame_id, data=payload,
                                                    is_extended_id=False, is_fd=msg.is_fd,
                                                    bitrate_switch=msg.is_fd, channel=0))
        self.frames += 1

    def write(self, sess: Session, day: int):
        if self.day != day:
            self._open(day)
        s, t = sess.sig, sess.t
        warn = warning_series(sess)
        step = 1 if sess.kind != "ac_charge" else 1
        for i in range(0, t.size, step):
            if not s["can_ok"][i]:
                continue
            ts = float(t[i])
            charging_only = sess.kind == "ac_charge"
            self._send(ts, "VCU_Motion", {"VehicleSpeed": max(0, s["speed_kmh"][i]), "AccelLong": s["accel"][i],
                                          "AccelPedal": s["accel_pedal"][i], "BrakePedal": s["brake_pedal"][i],
                                          "Gear": int(s["gear"][i]), "PowerState": int(s["power_state"][i])})
            self._send(ts + 0.002, "BMS_Pack", {"PackVoltage": s["pack_v"][i], "PackCurrent": s["pack_a"][i],
                                                "PackPower": s["pack_w"][i], "SoC": np.clip(s["soc"][i], 0, 100)})
            self._send(ts + 0.003, "BMS_Thermal", {"TempAvg": s["temp_avg"][i], "TempMax": s["temp_max"][i],
                                                   "TempMin": s["temp_min"][i], "SoH": s["soh"][i],
                                                   "SoCDisplayed": s["soc_disp"][i], "DischargeLimit": s["dis_limit"][i],
                                                   "ChargeLimit": s["chg_limit"][i]})
            self._send(ts + 0.004, "BMS_Charge", {"Range": max(0, s["range_km"][i]), "IsCharging": int(s["is_charging"][i]),
                                                  "ChargeCurrentDC": max(0, s["dc_a"][i]), "ChargeCurrentAC": s["ac_a"][i]})
            pack_cell = s["pack_v"][i] / P.cells_series
            if i % 5 == 0 or charging_only:
                cells = pack_cell + self.cell_off
                self._send(ts + 0.005, "BMS_CellStats", {"CellVmin": cells.min(), "CellVmax": cells.max()})
                for b in range(6):
                    self._send(ts + 0.006 + b * 0.0005, f"BMS_Cells_{b}",
                               {f"Cell{b * 16 + k + 1:02d}V": cells[b * 16 + k] for k in range(16)})
                mods = s["temp_avg"][i] + self.mod_off
                mods[6] = s["temp_max"][i]
                self._send(ts + 0.01, "BMS_ModuleTemps", {f"Module{k + 1:02d}T": mods[k] for k in range(32)})
            if charging_only:
                continue
            self._send(ts + 0.012, "VCU_Chassis", {"SteeringAngle": s["steer"][i], "AccelLat": s["accel_lat"][i],
                                                   "Odometer": s["odometer"][i]})
            self._send(ts + 0.014, "MCU_Status", {"MotorSpeed": s["motor_rpm"][i], "MotorTemp": s["motor_temp"][i],
                                                  "MotorTorque": s["motor_torque"][i], "RegenPower": s["regen_kw"][i]})
            self._send(ts + 0.016, "HVAC_Status", {"CompressorPower": s["ac_kw"][i], "HeaterPower": s["heater_kw"][i],
                                                   "ACActive": int(s["ac_on"][i]), "CabinTemp": s["cabin"][i],
                                                   "SetPoint": s["setpoint"][i], "FanSpeed": s["fan"][i]})
            self._send(ts + 0.018, "BCM_Environment", {"ExteriorTemp": s["ambient"][i], "WiperMode": int(s["wiper"][i]),
                                                       "LowBeam": int(s["lowbeam"][i]), "LVBatteryVoltage": s["lv"][i]})
            ty = s["tyres"][i]
            self._send(ts + 0.02, "TPMS_Pressures", {"TyrePressFL": ty[0], "TyrePressFR": ty[1],
                                                     "TyrePressRL": ty[2], "TyrePressRR": ty[3]})
            self._send(ts + 0.022, "ESC_Status", {"ABSActive": int(s["abs"][i]), "ESCActive": int(s["esc"][i]),
                                                  "EBAActive": int(s["eba"][i])})
            self._send(ts + 0.024, "Cluster_Warning", {"WarningCode": int(warn[i]), "WarningActive": int(warn[i] > 0)})

    def close(self):
        if self.writer:
            self.writer.stop()


# ======================================================================= SOME/IP, VHAL, QNX, UDS writers
class SomeIpWriter:
    """SOME/IP notifications. Header: MessageID(service,method) | Length | RequestID | proto 1 | iface 1 | type 0x02 | rc 0."""
    SERVICES = {"thermal": (0x1100, "192.168.10.31:30501"), "adas": (0x1200, "192.168.10.41:30501"),
                "camera": (0x1300, "192.168.10.42:30501")}

    def __init__(self, out_dir: Path):
        (out_dir / "someip").mkdir(parents=True, exist_ok=True)
        self.f = gzip.open(out_dir / "someip" / "someip.jsonl.gz", "wt", encoding="utf-8")
        self.session = 0
        self.count = 0

    def _emit(self, ts: float, svc: str, event_id: int, payload: bytes):
        service_id, src = self.SERVICES[svc]
        self.session = (self.session % 0xFFFF) + 1
        header = struct.pack(">HHIHHBBBB", service_id, event_id, 8 + len(payload), 0x0000, self.session, 0x01, 0x01, 0x02, 0x00)
        self.f.write(json.dumps({"ts": round(ts, 3), "src": src, "hex": (header + payload).hex()}) + "\n")
        self.count += 1

    def write(self, sess: Session):
        s, t = sess.sig, sess.t
        step = 10 if sess.kind == "ac_charge" else 1
        for i in range(0, t.size, step):
            ts = float(t[i])
            if sess.kind == "ac_charge" or i % 5 == 0:
                self._emit(ts, "thermal", 0x8001, struct.pack(">hHHB", int(s["coolant"][i] * 10), int(max(0, s["pump_rpm"][i])),
                                                              int(s["pump_a"][i] * 100), int(s["chiller"][i])))
            if sess.kind == "trip" and i % 2 == 0:
                self._emit(ts + 0.01, "adas", 0x8001, struct.pack(">BBBBB", int(s["cc"][i]), int(s["ldw_on"][i]),
                                                                  int(s["ldw_warn"][i]), int(s["lka"][i]),
                                                                  int(np.clip(s["lane_conf"][i], 0, 100))))
                self._emit(ts + 0.02, "camera", 0x8001, struct.pack(">BBB", int(s["cam_status"][i]),
                                                                    int(s["cam_objs"][i]), int(s["cam_vis"][i])))
        for e in sess.events:
            if e["type"] == "adas":
                code = REASONS[e["reason"]] | (0x80 if e["event"] == "LKA_OFF" else 0)
                self._emit(e["t"] + 0.05, "adas", 0x8002, struct.pack(">BHB", code, int(e["speed"] * 100), int(e["conf"])))

    def close(self):
        self.f.close()


class VhalWriter:
    """Android Automotive VHAL property events (IDs from AOSP VehicleProperty; 0x2... = vendor properties)."""
    PROPS = {"PERF_VEHICLE_SPEED": "0x11600207", "ENV_OUTSIDE_TEMPERATURE": "0x11600703", "HVAC_AC_ON": "0x1520050A",
             "HVAC_TEMPERATURE_SET": "0x15600503", "GEAR_SELECTION": "0x11400400",
             "VENDOR_LOCATION_LAT": "0x21600F01", "VENDOR_LOCATION_LON": "0x21600F02", "VENDOR_ROAD_TYPE": "0x21400F03"}

    def __init__(self, out_dir: Path):
        (out_dir / "vhal").mkdir(parents=True, exist_ok=True)
        self.f = gzip.open(out_dir / "vhal" / "vhal.jsonl.gz", "wt", encoding="utf-8")
        self.lat, self.lon, self.heading = 15.4909, 73.8278, 0.0
        self.rng = np.random.default_rng(11)
        self.count = 0

    def _emit(self, ts, prop, value, area=0):
        self.f.write(json.dumps({"ts": round(ts, 3), "prop": prop, "id": self.PROPS[prop], "area": area,
                                 "value": value}) + "\n")
        self.count += 1

    def write(self, sess: Session):
        if sess.kind != "trip":
            return
        s, t = sess.sig, sess.t
        self._emit(t[0], "GEAR_SELECTION", 8)                       # VehicleGear.GEAR_DRIVE
        self._emit(t[0], "HVAC_AC_ON", True, area=0x31)
        self._emit(t[0], "HVAC_TEMPERATURE_SET", 22.5, area=0x1)
        out = sess.meta.get("purpose", "").endswith(("_out", "_am", "errand"))
        for i in range(0, t.size, 5):
            v = s["speed_kmh"][i] / 3.6
            self.heading += self.rng.normal(0, 0.15 if s["road"][i] == 0 else 0.02)
            d = v * 5 * (1 if out else -1)
            self.lat += d * np.cos(self.heading) / 111_000
            self.lon += d * np.sin(self.heading) / 107_000
            self._emit(t[i], "PERF_VEHICLE_SPEED", round(float(v), 2))
            self._emit(t[i], "VENDOR_LOCATION_LAT", round(self.lat, 5))
            self._emit(t[i], "VENDOR_LOCATION_LON", round(self.lon, 5))
            if i % 30 == 0:
                self._emit(t[i], "ENV_OUTSIDE_TEMPERATURE", round(float(s["ambient"][i]), 1))
                self._emit(t[i], "VENDOR_ROAD_TYPE", int(s["road"][i]))
        self._emit(t[-1], "GEAR_SELECTION", 4)                      # GEAR_PARK

    def close(self):
        self.f.close()


class QnxLogWriter:
    PIDS = {"bootmgr": 4001, "someip_gw": 4097, "camerad": 8211, "adas_lka": 8305, "thermal_svc": 5120, "bms_mon": 5133,
            "pmic": 3002, "tpms_svc": 6140, "canlogger": 2210, "ipc_hmi": 9001}

    def __init__(self, out_dir: Path):
        (out_dir / "qnx").mkdir(parents=True, exist_ok=True)
        self.lines: list[tuple[float, str]] = []

    def add(self, ts, proc, sev, msg):
        self.lines.append((ts, f"{local(ts):%b %d %H:%M:%S}.{int(ts * 1000) % 1000:03d}    "
                               f"{proc}.{self.PIDS.get(proc, 7000):<10} main  {sev}  {msg}"))

    def write(self, sess: Session):
        t0 = float(sess.t[0])
        if sess.kind != "ac_charge":
            self.add(t0, "bootmgr", 5, "startup complete in 1.84 s (QNX 8.0, hypervisor guest qm-vm1)")
            self.add(t0 + 0.4, "someip_gw", 5, "service 0x1100 thermal available; 0x1200 adas available; 0x1300 camera available")
        for e in sess.events:
            if e["type"] == "log":
                self.add(e["t"], e["proc"], e["sev"], e["msg"])
        if sess.kind != "ac_charge":
            self.add(float(sess.t[-1]) + 1, "bootmgr", 5, "shutdown requested (ignition off), flushing logs")

    def close(self, out_dir: Path):
        with gzip.open(out_dir / "qnx" / "slog2.log.gz", "wt", encoding="utf-8") as f:
            for _, line in sorted(self.lines):
                f.write(line + "\n")


class UdsWriter:
    def __init__(self, out_dir: Path):
        (out_dir / "uds").mkdir(parents=True, exist_ok=True)
        self.f = open(out_dir / "uds" / "dtc.jsonl", "w", encoding="utf-8")
        self.count = 0

    def write(self, sess: Session):
        for e in sorted((e for e in sess.events if e["type"] == "dtc"), key=lambda e: e["t"]):
            st = e["status"]
            self.f.write(json.dumps({"ts": round(e["t"], 3), "ecu": e["ecu"], "service": "0x59", "subfunction": "0x02",
                                     "dtc": e["code"], "dtc_hex": dtc_to_hex(e["code"]), "status": f"0x{st:02X}",
                                     "status_bits": {"testFailed": bool(st & 1), "pendingDTC": bool(st & 4),
                                                     "confirmedDTC": bool(st & 8)},
                                     "odometer_km": round(float(sess.sig["odometer"][0]), 1)}) + "\n")
            self.count += 1

    def close(self):
        self.f.close()


# ======================================================================= main loop
def build_sessions(days: int):
    """Yield (day, Session) in time order for the whole simulation."""
    state = CarState()
    trips = [tr for tr in W.plan_trips(SEED) if tr.day <= days]
    by_day: dict[int, list] = {}
    for k, tr in enumerate(trips):
        by_day.setdefault(tr.day, []).append((k, tr))
    for day in range(1, days + 1):
        for k, tr in by_day.get(day, []):
            win = W.dc_charge_window(day)
            if win and tr.purpose == "highway_back":
                yield day, simulate_charge("dc_charge", win[0], win[1], state, day, target_soc=92.0)
            yield day, simulate_trip(tr, state, seed=SEED + k)
        tomorrow_hwy = (day + 1) in W.HIGHWAY_SATURDAYS or day + 1 == 24
        if state.soc < 55 or tomorrow_hwy:
            yield day, simulate_charge("ac_charge", W.ts(day, 22, 0), None, state, day, target_soc=90.0)


def main(days: int = W.DAYS, out: Path = OUT):
    t_start = time.time()
    out.mkdir(parents=True, exist_ok=True)
    DBC.write()
    canw, sip, vh, qnx, uds = CanWriter(out), SomeIpWriter(out), VhalWriter(out), QnxLogWriter(out), UdsWriter(out)
    trips_meta, charges, all_events = [], [], []
    range_samples = []
    for day, sess in build_sessions(days):
        canw.write(sess, day)
        sip.write(sess)
        vh.write(sess)
        qnx.write(sess)
        uds.write(sess)
        all_events += [dict(e, day=day) for e in sess.events]
        if sess.kind == "trip":
            s = sess.sig
            spd = s["speed_kmh"]
            p = s["pack_w"] / 1000
            city, hwy = s["road"] == 0, s["road"] == 1
            trips_meta.append({**sess.meta, "start": float(sess.t[0]),
                               "avg_speed_kmh": float(spd.mean()), "stop_go_pct": float((spd < 5).mean() * 100),
                               "ac_kwh": float(s["ac_kw"].sum() / 3600),
                               "city_km": float(spd[city].sum() / 3600), "hwy_km": float(spd[hwy].sum() / 3600),
                               "city_kwh": float(p[city].sum() / 3600), "hwy_kwh": float(p[hwy].sum() / 3600),
                               "min_rl_kpa": float(s["tyres"][:, 2].min()),
                               "range_per_soc": float(np.median(s["range_km"] / np.maximum(s["soc"], 1)))})
            range_samples.append((day, float(np.median(s["range_km"] / np.maximum(s["soc"], 1)))))
        else:
            charges.append({**sess.meta, "start": float(sess.t[0])})
        print(f"\r  day {day:2d}/{days}  CAN frames {canw.frames:,}", end="", flush=True)
    canw.close(); sip.close(); vh.close(); qnx.close(out); uds.close()
    print()

    from .truth import build_truth
    truth = build_truth(trips_meta, charges, all_events, days)
    tdir = out / "truth"
    tdir.mkdir(exist_ok=True)
    (tdir / "truth.json").write_text(json.dumps(truth, indent=2, default=float), encoding="utf-8")
    with open(tdir / "trips.csv", "w", newline="", encoding="utf-8") as f:
        cols = ["day", "start_local", "purpose", "kind", "dist_km", "energy_kwh", "kwh_per_100km", "traffic",
                "avg_speed_kmh", "stop_go_pct", "ac_kwh", "min_rl_kpa", "city_km", "hwy_km"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for m in trips_meta:
            w.writerow({"day": m["day"], "start_local": f"{local(m['start']):%Y-%m-%d %H:%M}", "purpose": m["purpose"],
                        "kind": m["kind"], "dist_km": round(m["dist_km"], 2), "energy_kwh": round(m["energy_kwh"], 3),
                        "kwh_per_100km": round(m["energy_kwh"] / m["dist_km"] * 100, 2), "traffic": m["traffic"],
                        "avg_speed_kmh": round(m["avg_speed_kmh"], 1), "stop_go_pct": round(m["stop_go_pct"], 1),
                        "ac_kwh": round(m["ac_kwh"], 3), "min_rl_kpa": round(m["min_rl_kpa"], 1),
                        "city_km": round(m["city_km"], 2), "hwy_km": round(m["hwy_km"], 2)})
    sizes = {str(p.relative_to(out)): p.stat().st_size for p in out.rglob("*") if p.is_file()}
    manifest = {"generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "days": days,
                "seed": SEED, "can_frames": canw.frames, "someip_messages": sip.count, "vhal_events": vh.count,
                "qnx_log_lines": len(qnx.lines), "uds_records": uds.count, "trips": len(trips_meta),
                "charge_sessions": len(charges), "total_mb": round(sum(sizes.values()) / 1e6, 1),
                "seconds": round(time.time() - t_start, 1)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=W.DAYS)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    main(a.days, a.out)
