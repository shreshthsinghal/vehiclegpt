"""Phase 5 tests: HMI server endpoints, role-based access, MDF4 export, privacy counters."""
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest


@pytest.fixture(scope="module")
def hmi(full_store):
    import ask.server as S
    S.ROLES_FILE = full_store[0].parent / "roles.json"
    app = S.App(full_store[0])
    srv = ThreadingHTTPServer(("127.0.0.1", 0), S.make_handler(app))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}", app
    srv.shutdown()


def _get(url, role=None, pin=None, raw=False):
    req = urllib.request.Request(url, headers={"X-Role": role or "", "X-Pin": pin or ""})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read()
    return data if raw else json.loads(data)


@pytest.mark.slow
def test_page_and_status(hmi):
    base, _ = hmi
    html = _get(base + "/", raw=True).decode()
    assert "VehicleGPT" in html and "http://" not in html.replace("http://127.0.0.1", "")   # no external resources
    s = _get(base + "/api/status")
    assert s["can_tx_attempts"] == 0 and s["external_connections"] in (0, None)
    assert s["active_dtcs"] == ["P1A20"]                                                       # others passed after repair


@pytest.mark.slow
def test_ask_endpoint(hmi):
    base, _ = hmi
    req = urllib.request.Request(base + "/api/ask", data=json.dumps({"question": "Compare energy consumption between city and highway driving."}).encode())
    a = json.loads(urllib.request.urlopen(req, timeout=60).read())
    assert a["verified"] and a["chart"]["type"] == "bar" and a["intent"] == "segment_compare"


@pytest.mark.slow
def test_role_based_access_and_mdf4(hmi):
    base, app = hmi
    with pytest.raises(urllib.error.HTTPError) as e:
        _get(base + "/api/day?date=2026-11-21")
    assert e.value.code == 403
    with pytest.raises(urllib.error.HTTPError):
        _get(base + "/api/day?date=2026-11-21", "service", "0000" if app.roles["service"] != "0000" else "1111")
    pin = app.roles["service"]
    d = _get(base + "/api/day?date=2026-11-21&signals=battery_temp", "service", pin)
    assert d["series"]["battery_temp"]["t"] and any(e["case_id"] for e in d["events"])
    cid = next(e["case_id"] for e in d["events"] if e["case_id"])
    blob = _get(base + f"/api/export?case={cid}", "service", pin, raw=True)
    path = app.st.path.parent / "t.mf4"
    path.write_bytes(blob)
    from asammdf import MDF
    assert len(MDF(path).channels_db) > 100
    f = _get(base + "/api/fleet", "fleet", app.roles["fleet"])
    assert "no location" in f["contains"] and f["sharing_opt_in"] is False


@pytest.mark.slow
def test_head_unit_screen_and_ui_layer(hmi):
    """Phase 7: the driver head-unit page is served at /, the console at /console, and every number the
    head-unit shows (hero, factor cards, spoken line) is a registered fact."""
    base, app = hmi
    html = _get(base + "/", raw=True).decode()
    assert 'id="orb"' in html and "http" not in html                     # fully offline page
    assert "VehicleGPT" in _get(base + "/console", raw=True).decode()
    for q in ["How much range do I have left?", "Why has my range dropped?", "When did my efficiency get worse?",
              "Any fault codes?", "Compare city and highway driving", "What happened before the warning?",
              "Is my battery healthy?", "unlock the doors", "what is the owner's home address"]:
        req = urllib.request.Request(base + "/api/ask", data=json.dumps({"question": q}).encode())
        a = json.loads(urllib.request.urlopen(req, timeout=60).read())
        assert a["ui"]["ui_verified"], q
        assert a["ui"]["follow"]
    assert a["intent"] == "privacy" and a["ui"]["kind"] == "refuse"


@pytest.mark.slow
def test_listen_rejects_non_wav(hmi):
    base, _ = hmi
    req = urllib.request.Request(base + "/api/listen", data=b"not audio at all, definitely not a wav file......")
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=10)
    assert e.value.code == 400
