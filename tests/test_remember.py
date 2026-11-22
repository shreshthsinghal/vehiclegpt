"""Phase 3 tests: lossless compression, tiering/retention, search, encryption at rest."""
import datetime as dt

import numpy as np
import pytest

from remember.build import build
from remember.codec import Cipher, decode, encode
from remember.store import WARM_DAYS, Store

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
T = lambda *a: dt.datetime(*a, tzinfo=IST).timestamp()


def test_codec_lossless_irregular():
    rng = np.random.default_rng(0)
    ts = np.sort(rng.choice(np.arange(0, 9000), 800, replace=False)) + 1_790_000_000
    for q in (0.001, 0.01, 0.5, 1.0):
        v = np.round(rng.normal(30, 5, ts.size) / q) * q
        t2, v2 = decode(encode(ts, v, q))
        assert np.array_equal(ts, t2) and np.allclose(v, v2, atol=q / 10)


def test_cipher_rejects_wrong_key():
    a, b = Cipher(b"a" * 32), Cipher(b"b" * 32)
    sealed = a.seal(b"vehicle data", b"aad")
    assert a.open(sealed, b"aad") == b"vehicle data" and b"vehicle data" not in sealed
    with pytest.raises(Exception):
        b.open(sealed, b"aad")


@pytest.fixture(scope="module")
def store(full_twin, tmp_path_factory):
    path = tmp_path_factory.mktemp("store") / "vehicle.db"
    rep = build(twin=full_twin, path=path, quiet=True)
    key = (path.parent / "device.key").read_bytes()
    return rep, Store(path, key=key)


@pytest.mark.slow
def test_tiers_and_size(store):
    rep, st = store
    s = rep["store"]
    assert s["warm"]["days"] <= WARM_DAYS + 0.5 and s["cold"]["days"] >= 54
    assert s["warm"]["bytes_per_point"] < 3.0                  # Round 1 budget: 3 B/point
    assert rep["total_on_disk_mb"] < 100                        # 8 weeks well under the 1 GB target
    assert s["trips"] == 121 and s["pinned_cases"]["count"] >= 20 and s["encrypted"]
    assert rep["sense"]["tx_attempts"] == 0


@pytest.mark.slow
def test_queries(store):
    _, st = store
    ts, v = st.series("Vehicle.Powertrain.TractionBattery.Temperature.Max", T(2026, 11, 21, 15, 50), T(2026, 11, 21, 16, 5))
    assert ts.size > 800 and v.max() >= 52
    assert st.series("Vehicle.Speed", T(2026, 10, 24), T(2026, 10, 25))[0].size == 0      # WARM expired
    r = st.rollup("Vehicle.Chassis.Axle.Row2.Wheel.Left.Tire.Pressure", T(2026, 10, 24), T(2026, 11, 1))
    assert r["minute"].size > 300 and r["min"].min() < 180                                # COLD still has it
    assert {e["code"] for e in st.search_events("coolant pump")} >= {"P1A20", "COOLANT_PUMP_LOW"}
    assert any("camera" in h["template"] for h in st.search_logs("camera restart"))
    cid = st.db.execute("SELECT id FROM cases WHERE triggers LIKE '%BATTERY_TEMP_HIGH%'").fetchone()[0]
    assert len(st.case(cid)["signals"]) > 150
