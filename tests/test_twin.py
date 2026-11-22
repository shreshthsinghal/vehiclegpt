"""Phase 1 tests: the digital twin produces valid wire formats and a correct answer key."""
import gzip
import json
import struct

import can
import pytest

from twin import dbc as DBC
from twin import world as W
from twin.generate import dtc_to_hex, main


def test_dbc_roundtrip():
    db = DBC.load()
    m = db.get_message_by_name("VCU_Motion")
    data = m.encode({"VehicleSpeed": 61.5, "AccelLong": -0.3, "AccelPedal": 20, "BrakePedal": 0,
                     "Gear": "D", "PowerState": "ON"})
    out = m.decode(data)
    assert abs(out["VehicleSpeed"] - 61.5) < 0.01 and str(out["Gear"]) == "D"
    assert sum(len(m.signals) for m in db.messages) == 177
    assert db.get_message_by_name("BMS_Cells_0").is_fd


def test_dtc_encoding():
    assert dtc_to_hex("P1A20") == "1A2000"
    assert dtc_to_hex("C1A10") == "5A1000"
    assert dtc_to_hex("U1C01") == "DC0100"


def test_calendar():
    assert f"{W.day_start(1):%a %d %b %Y}" == "Mon 05 Oct 2026"
    assert f"{W.day_start(56):%a %d %b %Y}" == "Sun 29 Nov 2026"


@pytest.fixture(scope="module")
def week(tmp_path_factory):
    out = tmp_path_factory.mktemp("twin7")
    man = main(days=7, out=out)
    return out, man


def test_week_outputs(week):
    out, man = week
    assert man["trips"] == 15 and man["can_frames"] > 300_000
    db = DBC.load()
    n = 0
    for msg in can.BLFReader(str(out / "can" / "day01.blf")):
        db.decode_message(msg.arbitration_id, msg.data)       # every frame must decode
        n += 1
        if n > 2000:
            break
    with gzip.open(out / "someip" / "someip.jsonl.gz", "rt") as f:
        raw = bytes.fromhex(json.loads(f.readline())["hex"])
    svc, meth, length, *_ = struct.unpack(">HHIHHBBBB", raw[:16])
    assert length == len(raw) - 8 and svc in (0x1100, 0x1200, 0x1300)


@pytest.mark.slow
def test_full_answer_key(full_twin):
    t = json.loads((full_twin / "truth" / "truth.json").read_text())
    s01 = t["S01_range_drop"]
    assert s01["explained_shares_pct"]["ac_heat"] > s01["explained_shares_pct"]["stop_go_traffic"]
    assert s01["displayed_range_change_pct"] < -5 and s01["tyres_normal_in_window_b"]
    assert t["S02_efficiency_drop"]["worst_day"] == 26
    ev = t["S03_S04_adas_yesterday"]["events"]
    assert len(ev) == 5 and sum(e["reason"] == "LANE_MARKINGS_LOST" for e in ev) == 4
    seq = [s["detail"] for s in t["S05_overheat"]["sequence"] if s["type"] == "dtc"]
    assert seq == ["P1A20", "P1A21"]
    assert t["S09_baseline"]["faults"] == []
    assert t["S06_city_vs_highway"]["highway_kwh_100km"] > t["S06_city_vs_highway"]["city_kwh_100km"]
