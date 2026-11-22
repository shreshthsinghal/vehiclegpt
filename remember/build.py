"""Build the on-device store from the recordings (SENSE -> REMEMBER) and report what it costs.

  python -m remember.build            # data/store/vehicle.db + encrypted case files
  python -m remember.build --days 7
  python -m remember.build --no-encrypt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from sense.config import DATA
from sense.ingest import TWIN, backfill, peak_rss_mb

from .store import Store, StoreSink, load_key

STORE = DATA / "store" / "vehicle.db"


def build(twin: Path = TWIN, path: Path = STORE, days: int | None = None, encrypt: bool = True, quiet=False) -> dict:
    t0 = time.time()
    key = load_key(path.parent / "device.key") if encrypt else None
    store = Store(path, key=key, fresh=True)
    sink = StoreSink(store)
    rep = backfill(twin=twin, days=days, extra_sinks=[sink], quiet=True, case_writer=sink.write_case)
    ret = store.enforce_retention(rep["last_ts"])
    store.db.execute("VACUUM")
    st = store.stats()
    recorded_days = days or 56
    drive_h_per_day = rep["snapshots_1hz"] / recorded_days / 3600
    full_8wk_mb = None
    # what 8 weeks would cost if WARM were kept for all 8 weeks (worst case) at this usage and at 8 h/day
    per_day_warm = st["warm"]["payload_mb"] / max(st["warm"]["days"], 1)
    per_day_cold = st["cold"]["payload_mb"] / max(st["cold"]["days"], 1)
    db_mb = st["db_file_mb"] + st["pinned_cases"]["kb_total"] / 1024
    report = {
        "store": st, "total_on_disk_mb": round(db_mb, 1), "retention": ret, "flushes": sink.flushes,
        "usage_h_per_day": round(drive_h_per_day, 2),
        "per_day_mb": {"warm_1hz": round(per_day_warm, 3), "cold_1min": round(per_day_cold, 3)},
        "projection_8_weeks_mb": {f"at {drive_h_per_day:.1f} h/day (measured)": round(db_mb, 1),
                                  "at 2 h/day (Round 1 assumption)": round(db_mb * 2 / drive_h_per_day, 1),
                                  "at 8 h/day (taxi/fleet)": round(db_mb * 8 / drive_h_per_day, 1)},
        "sense": {k: rep[k] for k in ("snapshots_1hz", "case_files", "tx_attempts", "egress_bytes")},
        "seconds": round(time.time() - t0, 1), "peak_rss_mb": round(peak_rss_mb() or 0, 1)}
    (path.parent / "store_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not quiet:
        print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int)
    ap.add_argument("--no-encrypt", action="store_true")
    a = ap.parse_args()
    build(days=a.days, encrypt=not a.no_encrypt)
