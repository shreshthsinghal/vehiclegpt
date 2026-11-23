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
NO_DATA = {"out_of_scope", "related_records", "not_recorded", "capabilities"}


def ok(expect: str, intent: str) -> bool:
    """expect: 'refuse', 'no_data', 'intent:a|b', or combinations like 'refuse|intent:a|b' / 'no_data|refuse'."""
    head, _, tail = expect.partition("intent:")
    kinds = [k for k in head.split("|") if k]
    if tail and intent in tail.split("|"):
        return True
    return ("refuse" in kinds and intent in REFUSE) or ("no_data" in kinds and intent in NO_DATA)


def main(smart: bool, tag: str, store_path=None, out_dir=None, file="blind_questions.yaml", split=None, quiet=False):
    planner = None
    if smart:
        from reason.planner import SlmPlanner
        planner = SlmPlanner()
    kw = {"store_path": store_path} if store_path else {}
    vg = VehicleGPT(planner=planner, learn=False, **kw)     # never learn from the blind set
    qs = yaml.safe_load(open(ROOT / "eval" / file))["questions"]
    qs = [x for x in qs if x["expect"] != "skip" and (split is None or x.get("split") == split)]
    rows, hit = [], 0
    for i, x in enumerate(qs, 1):
        a = vg.ask(x["q"])
        good = ok(x["expect"], a.intent)
        hit += good
        rows.append({"n": i, "q": x["q"], "expect": x["expect"], "intent": a.intent, "path": a.path, "routing_ok": good,
                     "verified": a.verified, "what": a.what, "ms": round(a.timings_ms.get("total", 0), 1)})
        if not quiet or not good:
            print(f"{'OK' if good else 'XX'} {i:2d} {a.path:14s} {a.intent:18s} <- {x['q'][:80]}   [{x['expect']}]")
        if not a.verified:
            print(f"   unverified: {a.unverified}")
    by = {}
    for r in rows:
        k = r["expect"].split(":")[0].split("|")[0]
        by.setdefault(k, [0, 0]); by[k][0] += r["routing_ok"]; by[k][1] += 1
    summary = {"mode": "smart" if smart else "fast-only", "routing_ok": f"{hit}/{len(qs)}",
               "by_kind": {k: f"{a}/{b}" for k, (a, b) in by.items()},
               "verified": f"{sum(r['verified'] for r in rows)}/{len(qs)}"}
    out = (out_dir or ROOT / "eval" / "results") / f"{Path(file).stem.split('_')[0]}_{tag}{'_' + split if split else ''}_{summary['mode']}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print("BLIND " + json.dumps(summary))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smart", action="store_true")
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--file", default="blind_questions.yaml")
    ap.add_argument("--split", default=None)
    ap.add_argument("--quiet", action="store_true", help="print only misses")
    a = ap.parse_args()
    main(a.smart, a.tag, file=a.file, split=a.split, quiet=a.quiet)
