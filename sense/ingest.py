"""Run the SENSE layer.

  python -m sense.ingest --backfill            # all recorded twin data -> data/sense/ (events, logs, case files)
  python -m sense.ingest --backfill --days 7   # quick
  python -m sense.ingest --live                # listen to the CAN bus + SOME/IP (UDP) while twin.replay plays
"""
from __future__ import annotations

import argparse
import heapq
import json
import time
from pathlib import Path

from safety.guards import COUNTERS, ListenOnlyBus

from .adapters.sources import CanDecoder, can_files, can_live, qnx_file, someip_file, someip_live, uds_file, vhal_file
from .config import DATA
from .engine import Engine, JsonlSink

TWIN = DATA / "twin"
OUT = DATA / "sense"


def peak_rss_mb() -> float | None:
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except ImportError:
        try:
            import psutil
            return psutil.Process().memory_info().peak_wset / 1e6      # Windows
        except Exception:                                               # noqa: BLE001
            return None


def backfill(twin: Path = TWIN, out: Path = OUT, days: int | None = None, extra_sinks=(), quiet=False,
             case_writer=None) -> dict:
    t0 = time.time()
    blfs = sorted((twin / "can").glob("day*.blf"))[: days or None]
    dec = CanDecoder()
    streams = [can_files(blfs, dec), someip_file(twin / "someip" / "someip.jsonl.gz"),
               vhal_file(twin / "vhal" / "vhal.jsonl.gz"), qnx_file(twin / "qnx" / "slog2.log.gz"),
               uds_file(twin / "uds" / "dtc.jsonl")]
    end_ts = None
    if days:
        import can
        first = next(iter(can.BLFReader(str(blfs[0])))).timestamp
        end_ts = (first // 86400) * 86400 + 86400 * days + 86400   # generous: stop after the last selected day
    import shutil
    shutil.rmtree(out / "cases", ignore_errors=True)
    eng = Engine([JsonlSink(out), *extra_sinks], cases_dir=out / "cases", case_writer=case_writer)
    n = 0
    for rec in heapq.merge(*streams, key=lambda r: r[0]):
        if end_ts and rec[0] > end_ts:
            break
        eng.feed(rec)
        n += 1
        if not quiet and n % 500_000 == 0:
            print(f"\r  {n:,} records  {eng.snapshots:,} snapshots  {sum(eng.events_by_type.values()):,} events",
                  end="", flush=True)
    eng.finish()
    report = {"last_ts": eng.sec, "records": dict(eng.counts), "snapshots_1hz": eng.snapshots, "vss_paths_seen": len(eng.paths),
              "events": dict(sorted(eng.events_by_type.items())), "case_files": len(eng.bb.written),
              "unknown_can_ids": dec.unknown, "tx_attempts": COUNTERS["tx_attempts"],
              "egress_bytes": COUNTERS["egress_bytes"], "seconds": round(time.time() - t0, 1),
              "peak_rss_mb": round(peak_rss_mb() or 0, 1)}
    (out / "ingest_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not quiet:
        print()
        print(json.dumps(report, indent=2))
    return report


def live(interface=None, channel=None, someip_port: int = 30501):
    """Live demo: CAN (listen-only) + SOME/IP over UDP, printing events as they happen."""
    import threading
    import queue

    from twin.bus import open_bus

    class PrintSink:
        def on_snapshot(self, sec, snap):
            pass

        def on_event(self, e):
            if e["type"] in ("TRIP", "CHARGE"):
                return
            what = e.get("kind") or e.get("code") or e.get("reason") or ""
            print(f"  EVENT {e['time']}  {e['type']:<20} {what}" + (f"   -> black box {e['case_id']}" if "case_id" in e else ""))

        def on_log(self, rec):
            pass

        def close(self):
            pass

    q: queue.Queue = queue.Queue()
    bus = ListenOnlyBus(open_bus(interface, channel))

    def pump(gen):
        for rec in gen:
            if rec is not None:
                q.put(rec)

    threading.Thread(target=pump, args=(can_live(bus),), daemon=True).start()
    threading.Thread(target=pump, args=(someip_live(someip_port),), daemon=True).start()
    eng = Engine([PrintSink()], cases_dir=OUT / "live_cases")
    print(f"SENSE live: listening (CAN listen-only, SOME/IP udp/{someip_port}). Ctrl+C to stop.")
    buf: list = []
    try:
        while True:
            try:
                rec = q.get(timeout=1.0)
                heapq.heappush(buf, (rec[0], id(rec), rec))
            except queue.Empty:
                pass
            while buf and (len(buf) > 50 or q.empty()):     # small re-order buffer between the two sources
                eng.feed(heapq.heappop(buf)[2])
    except KeyboardInterrupt:
        pass
    finally:
        eng.finish()
        bus.shutdown()
        print(f"stopped. tx_attempts={COUNTERS['tx_attempts']}  case files: {len(eng.bb.written)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--backfill", action="store_true")
    g.add_argument("--live", action="store_true")
    ap.add_argument("--days", type=int)
    ap.add_argument("--interface")
    ap.add_argument("--channel")
    a = ap.parse_args()
    if a.backfill:
        backfill(days=a.days)
    else:
        live(a.interface, a.channel)
