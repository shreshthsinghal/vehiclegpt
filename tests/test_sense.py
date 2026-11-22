"""Phase 2 tests: SENSE turns raw recordings into the right events, graded against the answer key."""
import csv
import json
import struct

import pytest

from safety.guards import COUNTERS, ListenOnlyBus, TransmitBlocked
from sense.adapters.sources import SomeIpDecoder
from sense.ingest import backfill


class _FakeBus:
    def recv(self, timeout=None):
        return None

    def shutdown(self):
        pass


def test_listen_only_blocks_transmit():
    bus = ListenOnlyBus(_FakeBus())
    before = COUNTERS["tx_attempts"]
    with pytest.raises(TransmitBlocked):
        bus.send(object())
    assert COUNTERS["tx_attempts"] == before + 1
    COUNTERS["tx_attempts"] = before                     # restore for the other tests


def test_someip_rejects_malformed():
    d = SomeIpDecoder()
    good = struct.pack(">HHIHHBBBB", 0x1300, 0x8001, 8 + 3, 0, 1, 1, 1, 2, 0) + bytes([0, 5, 90])
    assert d.decode(1.0, good)[2] == "sig"
    assert d.decode(1.0, good[:-1]) is None              # wrong length field
    assert d.decode(1.0, b"\x00" * 5) is None            # truncated


@pytest.fixture(scope="module")
def sensed(full_twin, tmp_path_factory):
    out = tmp_path_factory.mktemp("sense")
    COUNTERS["tx_attempts"] = 0
    rep = backfill(twin=full_twin, out=out, quiet=True)
    ev = [json.loads(l) for l in open(out / "events.jsonl", encoding="utf-8")]
    return rep, ev, out


@pytest.mark.slow
def test_report_is_clean(sensed):
    rep, _, _ = sensed
    assert rep["tx_attempts"] == 0 and rep["egress_bytes"] == 0 and rep["unknown_can_ids"] == 0
    assert rep["vss_paths_seen"] >= 190 and rep["case_files"] > 0


@pytest.mark.slow
def test_events_match_answer_key(sensed, full_twin):
    _, ev, _ = sensed
    by = lambda t: [e for e in ev if e["type"] == t]
    trips = by("TRIP")
    truth = {r["start_local"]: r for r in csv.DictReader(open(full_twin / "truth" / "trips.csv"))}
    assert len(trips) == len(truth) == 121
    for e in trips:                                       # energy per km within 1 % on complete trips
        if e["data_gap_min"] == 0:
            t = float(truth[e["start"][:16]]["kwh_per_100km"])
            assert abs(e["kwh_per_100km"] - t) / t < 0.01
    gaps = by("DATA_GAP")
    assert len(gaps) == 1 and gaps[0]["start"].startswith("2026-10-19 18:1") and 19 <= gaps[0]["duration_min"] <= 21
    dis = [e for e in by("ADAS_DISENGAGEMENT") if e["time"].startswith("2026-11-28")]
    assert [e["reason"] for e in dis] == ["LANE_MARKINGS_LOST"] * 4 + ["CAMERA_UNAVAILABLE"]
    assert sorted(e["code"] for e in by("DTC_SET")) == ["C1A10", "P1A20", "P1A21", "P1B05", "U1C01"]
    an = {}
    for e in by("ANOMALY"):
        an.setdefault(e["kind"], []).append(e["time"])
    assert len(an["COOLANT_PUMP_LOW"]) == 1 and an["COOLANT_PUMP_LOW"][0] < "2026-11-21 16:03"
    assert len(an["BATTERY_TEMP_ABOVE_CONTEXT"]) == 1 and an["BATTERY_TEMP_ABOVE_CONTEXT"][0].startswith("2026-11-21")
    assert len(an["LV_LOW_AT_WAKE"]) == 3
    tpms = next(e["time"] for e in by("WARNING_ON") if e["code"] == "CHECK_TYRE_PRESSURE")
    assert min(an["TYRE_PRESSURE_LOW"]) < tpms           # VehicleGPT notices the puncture before the car warns
