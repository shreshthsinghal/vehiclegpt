"""Load the SENSE mapping (native names -> VSS) and the vehicle DBC."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cantools
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


@lru_cache(maxsize=1)
def mapping() -> dict:
    m = yaml.safe_load((ROOT / "sense" / "mappings.yaml").read_text(encoding="utf-8"))
    m["someip"] = {int(str(s), 0): {int(str(e), 0): v for e, v in evs.items()} for s, evs in m["someip"].items()}
    m["vhal"] = {k.lower(): v for k, v in m["vhal"].items()}
    return m


@lru_cache(maxsize=1)
def dbc():
    return cantools.database.load_file(str(ROOT / mapping()["can"]["dbc"]))


# Numeric enum codes used inside VehicleGPT (kept numeric for compact storage).
POWER_STATE = {0: "OFF", 1: "ACC", 2: "ON", 3: "CHARGING"}
WARNINGS = {0: "NONE", 1: "BATTERY_TEMP_HIGH", 2: "REDUCED_POWER", 3: "CHECK_TYRE_PRESSURE", 4: "LOW_12V_BATTERY",
            5: "CAMERA_UNAVAILABLE"}
WIPER = {0: "OFF", 1: "INTERVAL", 2: "LOW", 3: "HIGH"}
