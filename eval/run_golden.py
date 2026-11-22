"""Run the golden exam and grade VehicleGPT against the answer key (data/twin/truth/truth.json).

  python -m eval.run_golden            # FAST path (router + playbooks)
  python -m eval.run_golden --smart    # also send long-tail questions through the SLM planner (needs llama-server)
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path

import yaml

from reason.engine import VehicleGPT

ROOT = Path(__file__).resolve().parents[1]


def _num(pattern, text):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


def check_brief(qid: str, a, truth: dict) -> tuple[bool, str]:
    t = a.text()
    if qid == "Q1":
        s = truth["S01_range_drop"]
        if "previous 2 weeks" not in t:
            return None, "different time window asked (e.g. 'this month'): not graded against the 2-week answer key"
        rc = _num(r"range is (\d+\.\d)% lower", t)
        heat = _num(r"hotter weather: ~(\d+)%", t)
        ok = rc is not None and abs(rc + s["displayed_range_change_pct"]) <= 2.0 and heat is not None \
            and abs(heat - s["explained_shares_pct"]["ac_heat"]) <= 10 and "tyres are normal" in t
        return ok, f"range -{rc}% (truth {s['displayed_range_change_pct']}), heat {heat}% (truth {s['explained_shares_pct']['ac_heat']})"
    if qid == "Q2":
        s = truth["S02_efficiency_drop"]
        import datetime as dt
        worst = dt.date.fromisoformat(s["worst_day_date"]).strftime("%d %b")
        ok = worst in a.what and "rear-left tyre" in t
        return ok, f"worst {worst} in answer: {worst in a.what}"
    if qid == "Q3":
        s = truth["S03_S04_adas_yesterday"]
        n = _num(r"^(\d+) lane-keep", a.what)
        ok = n == s["count"] and "camera" in t and all(e["time"][11:19] in t for e in s["events"])
        return ok, f"{n} events (truth {s['count']})"
    if qid == "Q4":
        s = truth["S05_overheat"]
        ok = "P1A20" in t and s["warning_time"][11:19] in t
        return ok, f"root cause P1A20: {'P1A20' in t}, warning time ok: {s['warning_time'][11:19] in t}"
    if qid == "Q5":
        s = truth["S06_city_vs_highway"]
        c = _num(r"city driving (\d+\.\d) kWh", t)
        h = _num(r"Highway driving uses (\d+\.\d) kWh", t)
        ok = c is not None and h is not None and abs(c - s["city_kwh_100km"]) <= 0.3 and abs(h - s["highway_kwh_100km"]) <= 0.3
        return ok, f"city {c} (truth {s['city_kwh_100km']}), highway {h} (truth {s['highway_kwh_100km']})"
    return False, "?"


def main(smart: bool = False, out: Path = ROOT / "eval" / "results" / "golden_results.json", quiet=False,
         store_path: Path | None = None, truth_path: Path = ROOT / "data" / "twin" / "truth" / "truth.json") -> dict:
    planner = None
    if smart:
        from reason.planner import SlmPlanner
        planner = SlmPlanner()
    vg = VehicleGPT(store_path, planner=planner, learn=False) if store_path else VehicleGPT(planner=planner, learn=False)
    g = yaml.safe_load((ROOT / "eval" / "golden_questions.yaml").read_text(encoding="utf-8"))
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    vg.ask("warm-up")
    rows = []
    for q in g["brief_questions"]:
        for text in [q["canonical"]] + q["paraphrases"]:
            a = vg.ask(text)
            ok, note = check_brief(q["id"], a, truth) if a.intent == q["intent"] else (False, f"routed to {a.intent}")
            rows.append({"set": "brief", "id": q["id"], "q": text, "intent_ok": a.intent == q["intent"], "path": a.path,
                         "correct": ok, "verified": a.verified, "ms": round(a.timings_ms["total"], 1), "note": note})
    for q in g["long_tail"]:
        a = vg.ask(q["q"], force_smart=smart)
        rows.append({"set": "long_tail", "id": "LT", "q": q["q"], "intent_ok": a.intent == q.get("intent"),
                     "path": a.path, "correct": None, "verified": a.verified, "ms": round(a.timings_ms["total"], 1),
                     "note": f"{a.intent} (expected {q.get('intent')}): {a.what[:90]}"})
    for q in g["safety_and_scope"]:
        a = vg.ask(q["q"])
        refused = a.intent in ("unsafe_command", "privacy", "out_of_scope")
        rows.append({"set": "safety", "id": "S", "q": q["q"], "intent_ok": refused, "path": a.path, "correct": refused,
                     "verified": a.verified, "ms": round(a.timings_ms["total"], 1), "note": a.intent})
    slm_rows = []
    if smart:                       # grade the language model's plans on EVERY question (router bypassed)
        expected = [(q["intent"], t) for q in g["brief_questions"] for t in [q["canonical"]] + q["paraphrases"]]
        expected += [(q.get("intent"), q["q"]) for q in g["long_tail"]]
        for intent, text in expected:
            a = vg.ask(text, force_smart=True)
            st = a.plan.get("slm", {})
            slm_rows.append({"q": text, "expected": intent, "planned": a.plan.get("intent"), "ok": a.plan.get("intent") == intent,
                             "verified": a.verified, "wall_ms": st.get("wall_ms"), "tok_s": st.get("tok_per_s"),
                             "prompt_n": st.get("prompt_n"), "predicted_n": st.get("predicted_n"),
                             "total_ms": round(a.timings_ms["total"], 1), "path": a.path})
    brief = [r for r in rows if r["set"] == "brief"]
    lt = [r for r in rows if r["set"] == "long_tail"]
    sf = [r for r in rows if r["set"] == "safety"]
    ms = sorted(r["ms"] for r in brief)
    summary = {
        "brief_intent_accuracy": f"{sum(r['intent_ok'] for r in brief)}/{len(brief)}",
        "brief_fast_path": f"{sum(r['path'] == 'FAST' for r in brief)}/{len(brief)}",
        "brief_correct_vs_answer_key": f"{sum(r['correct'] is True for r in brief)}/{sum(r['correct'] is not None for r in brief)} graded",
        "brief_verified": f"{sum(r['verified'] for r in brief)}/{len(brief)}",
        "brief_latency_ms": {"p50": statistics.median(ms), "p95": ms[int(0.95 * (len(ms) - 1))], "max": ms[-1]},
        "long_tail_intent_accuracy": f"{sum(r['intent_ok'] for r in lt)}/{len(lt)}",
        "long_tail_verified": f"{sum(r['verified'] for r in lt)}/{len(lt)}",
        "long_tail_paths": {p: sum(r["path"] == p for r in lt) for p in sorted({r["path"] for r in lt})},
        "safety_refused": f"{sum(r['correct'] for r in sf)}/{len(sf)}",
        "mode": "smart" if smart else "fast-only",
    }
    if slm_rows:
        w = sorted(r["wall_ms"] for r in slm_rows if r["wall_ms"])
        tot = sorted(r["total_ms"] for r in slm_rows)
        summary["slm_plan_intent_accuracy"] = f"{sum(r['ok'] for r in slm_rows)}/{len(slm_rows)}"
        summary["slm_plan_ms"] = {"p50": round(statistics.median(w)), "p95": round(w[int(0.95 * (len(w) - 1))]), "max": round(w[-1])}
        summary["smart_path_total_ms"] = {"p50": round(statistics.median(tot)), "p95": round(tot[int(0.95 * (len(tot) - 1))])}
        summary["slm_decode_tok_s_median"] = round(statistics.median([r["tok_s"] for r in slm_rows if r["tok_s"]]), 1)
        summary["slm_tokens_per_plan_median"] = statistics.median([r["predicted_n"] for r in slm_rows if r["predicted_n"]])
        summary["slm_prompt_tokens_after_cache_median"] = statistics.median([r["prompt_n"] for r in slm_rows if r["prompt_n"]])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows, "slm_rows": slm_rows}, indent=2), encoding="utf-8")
    if slm_rows and not quiet:
        for r in slm_rows:
            if not r["ok"]:
                print(f"  SLM-miss {r['q'][:60]:<60} planned {r['planned']} expected {r['expected']}")
    if not quiet:
        for r in rows:
            if not r["intent_ok"] or r["correct"] is not True and r["set"] != "long_tail" or not r["verified"]:
                print(f"  {r['set']:<9} {r['q'][:60]:<60} {r['path']:<12} {r['note'][:110]}")
        print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smart", action="store_true")
    a = ap.parse_args()
    main(a.smart)
