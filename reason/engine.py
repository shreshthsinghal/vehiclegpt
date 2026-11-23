"""VehicleGPT question answering: route -> plan -> playbook -> verify.

FAST path : the router recognises the question type -> playbook directly (no language model).
SMART path: unclear questions -> the on-device SLM writes a JSON plan (grammar-constrained) -> same playbooks.
"""
from __future__ import annotations

import time
from pathlib import Path

from remember.build import STORE
from remember.store import Store, load_key

from .answer import Answer, verify
from .playbooks import PLAYBOOKS, Ctx
import re

from .router import Router, english

# A second question joined to the first ("..., and did it affect my battery health?") is answered too.
SPLIT = re.compile(r"\?\s+(?=\w)|,?\s+and\s+(?=(?:what|how|did|does|do|is|was|were|which|when|why|where|has|have|are|any)\b)",
                   re.IGNORECASE)
# Second parts the first playbook already answers itself (no second analysis needed).
ABSORBED = {"efficiency_change": r"weather|temperature|rain|hot|cold", "before_event": r"still|active|issue|problem|fixed|safe|mean",
            "trip_stats": r"battery|energy|kwh|charge|percent|%", "charging_history": r"health|degrad|affect|batter",
            "range_drop": r"why|cause|reason", "signal_trend": r"why|normal|unusual"}


class VehicleGPT:
    def __init__(self, store_path: Path = STORE, planner=None, learn: bool = True):
        key_file = store_path.parent / "device.key"
        self.store = Store(store_path, key=key_file.read_bytes() if key_file.exists() else None)
        self.ctx = Ctx(self.store)
        self.router = Router(store_path.parent / "learned_intents.jsonl" if learn else None)
        self.planner = planner
        self.warmup_ms = planner.warmup() if planner is not None else 0.0

    def ask(self, question: str, force_smart: bool = False) -> Answer:
        original = question
        question = english(question)
        parts = [p.strip(" ,?") for p in SPLIT.split(question) if len(p.strip(" ,?").split()) >= 3]
        ans = self._ask_one(question, force_smart)
        if len(parts) >= 2 and ans.intent not in ("unsafe_command", "privacy", "not_recorded", "out_of_scope"):
            for part in parts[1:2]:
                if re.search(ABSORBED.get(ans.intent, r"^$"), part.lower()):
                    continue
                r = self.router.route(part)
                if not r.fast or r.intent in (ans.intent, "out_of_scope", "privacy", "not_recorded", "capabilities") or r.intent not in PLAYBOOKS:
                    continue
                sub = self._ask_one(part, False)
                ans.what = f"{ans.what} {sub.what}".strip()
                ans.why += sub.why
                ans.facts += sub.facts
                ans.evidence += sub.evidence
                ans.details += sub.details
                ans.next = ans.next or sub.next
                ans.plan = {**ans.plan, "second_part": {"text": part, "intent": sub.intent}}
                ans = verify(ans)
        ans.question = original
        return ans

    def _ask_one(self, question: str, force_smart: bool = False) -> Answer:
        t0 = time.perf_counter()
        route = self.router.route(question)
        t1 = time.perf_counter()
        plan = {"intent": route.intent}
        path = "FAST"
        if (not route.fast or force_smart) and route.intent not in ("unsafe_command", "privacy"):
            if self.planner is not None:
                plan = self.planner.plan(question, self.ctx, route.candidates)
                path = "SMART"
                if plan.get("intent") not in PLAYBOOKS:
                    plan = {"intent": route.intent}
                    path = "SMART-fallback"
            else:
                path = "FAST-lowconf"
                if route.score < 0.2:                      # unclear question, no model available: show related records
                    return self._related(question, t0)
        t2 = time.perf_counter()
        self.ctx.windows = []
        try:
            ans = PLAYBOOKS[plan["intent"]](self.ctx, question, plan)
        except Exception as exc:                       # never crash the HMI; say what failed
            ans = Answer(question, plan["intent"], what=f"I could not compute this answer ({type(exc).__name__}).")
            ans.confidence = "low"
        t3 = time.perf_counter()
        import re as _re
        for w in self.ctx.windows:                   # numbers in time-window labels are computed by code
            ans.fact("window", w.fmt(), source="date resolver")
            for n in _re.findall(r"\d+", w.label):
                ans.fact("window_n", float(n), source="date resolver")
        ans = verify(ans)
        t4 = time.perf_counter()
        if path == "FAST-lowconf":
            ans.confidence = "low" if ans.confidence == "high" else ans.confidence
        if path == "SMART" and ans.verified and self.router.learned_path is not None:
            self.router.learn(question, plan["intent"], plan)      # next time this is a FAST question
        ans.path, ans.plan = path, {**plan, **(ans.plan or {})}
        if path in ("FAST-lowconf", "SMART-fallback") or (path == "SMART" and route.intent != plan.get("intent")):
            ans.plan = {**ans.plan, "alternatives": [c for c in route.candidates if c != plan.get("intent")][:3]}
        ans.timings_ms = {"route": (t1 - t0) * 1e3, "plan": (t2 - t1) * 1e3, "query": (t3 - t2) * 1e3,
                          "verify": (t4 - t3) * 1e3, "total": (t4 - t0) * 1e3}
        return ans


    def _related(self, question: str, t0: float) -> Answer:
        import re as _re
        from .dates import fmt_time
        from .embed import embed
        hits = self.store.search_meaning(embed(question), k=5)
        ans = Answer(question, "related_records", path="MEANING-SEARCH", confidence="low")
        ans.what = "I'm not sure what you're asking. These recorded events look related:"
        ids = {rid for rid, _ in hits}
        for e in self.store.events():
            if e["id"] in ids:
                ans.fact(f"t{e['id']}", fmt_time(e["ts"]))
                if e["code"]:
                    ans.fact(f"c{e['id']}", e["code"], source="events table")
        for line in ans.details:                      # quoted stored event values
            for n in _re.findall(r"\d+(?:\.\d+)?", line):
                ans.fact("ev_n", float(n), source="events table (quoted)")
                ans.details.append(f"{fmt_time(e['ts'])} {e['type'].replace('_', ' ').lower()} {e['code'] or ''}".strip())
        ans.evidence.append("meaning search (sqlite-vec) over all event descriptions")
        ans = verify(ans)
        ans.timings_ms = {"total": (time.perf_counter() - t0) * 1e3}
        return ans


if __name__ == "__main__":
    import sys
    vg = VehicleGPT()
    for q in sys.argv[1:] or ["Why has my driving range reduced over the last two weeks?"]:
        print(vg.ask(q).render())
        print()
