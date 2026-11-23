"""Blind test: questions written by people outside the build (eval/blind_questions.yaml).

  python -m eval.run_blind            # FAST path only (no language model)
  python -m eval.run_blind --smart    # as deployed: low-confidence questions go to the on-device SLM (needs llama-server)
Automatic routing grade: intent matches / refused when it should / honest "no data" when nothing is recorded.
Answer-quality grading (does it fully answer?) is done by a person and kept in docs/blind_test.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from reason.engine import VehicleGPT

ROOT = Path(__file__).resolve().parents[1]
REFUSE = {"unsafe_command", "privacy"}
NO_DATA = {"out_of_scope", "related_records", "not_recorded"}


def ok(expect: str, intent: str) -> bool:
    if expect == "refuse":
        return intent in REFUSE
    if expect == "no_data":
        return intent in NO_DATA
    return intent == expect.split(":", 1)[1]


def main(smart: bool, tag: str, store_path=None, out_dir=None):
    planner = None
    if smart:
        from reason.planner import SlmPlanner
        planner = SlmPlanner()
    kw = {"store_path": store_path} if store_path else {}
    vg = VehicleGPT(planner=planner, learn=False, **kw)     # never learn from the blind set
    qs = yaml.safe_load(open(ROOT / "eval" / "blind_questions.yaml"))["questions"]
    rows, hit = [], 0
    for i, x in enumerate(qs, 1):
        a = vg.ask(x["q"])
        good = ok(x["expect"], a.intent)
        hit += good
        rows.append({"n": i, "q": x["q"], "expect": x["expect"], "intent": a.intent, "path": a.path, "routing_ok": good,
                     "verified": a.verified, "what": a.what, "ms": round(a.timings_ms.get("total", 0), 1)})
        print(f"{'OK' if good else 'XX'} {i:2d} {a.path:14s} {a.intent:18s} <- {x['q'][:80]}")
    by = {}
    for r in rows:
        k = r["expect"].split(":")[0]
        by.setdefault(k, [0, 0]); by[k][0] += r["routing_ok"]; by[k][1] += 1
    summary = {"mode": "smart" if smart else "fast-only", "routing_ok": f"{hit}/{len(qs)}",
               "by_kind": {k: f"{a}/{b}" for k, (a, b) in by.items()},
               "verified": f"{sum(r['verified'] for r in rows)}/{len(qs)}"}
    out = (out_dir or ROOT / "eval" / "results") / f"blind_{tag}_{summary['mode']}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print("BLIND " + json.dumps(summary))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smart", action="store_true")
    ap.add_argument("--tag", default="baseline")
    a = ap.parse_args()
    main(a.smart, a.tag)
