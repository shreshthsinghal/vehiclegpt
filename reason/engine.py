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
from .router import Router


class VehicleGPT:
    def __init__(self, store_path: Path = STORE, planner=None, learn: bool = True):
        key_file = store_path.parent / "device.key"
        self.store = Store(store_path, key=key_file.read_bytes() if key_file.exists() else None)
        self.ctx = Ctx(self.store)
        self.router = Router(store_path.parent / "learned_intents.jsonl" if learn else None)
        self.planner = planner

    def ask(self, question: str, force_smart: bool = False) -> Answer:
        t0 = time.perf_counter()
        route = self.router.route(question)
        t1 = time.perf_counter()
        plan = {"intent": route.intent}
        path = "FAST"
        if (not route.fast or force_smart) and route.intent not in ("unsafe_command", "privacy"):
            if self.planner is not None:
                plan = self.planner.plan(question, self.ctx)
                path = "SMART"
                if plan.get("intent") not in PLAYBOOKS:
                    plan = {"intent": route.intent}
                    path = "SMART-fallback"
            else:
                path = "FAST-lowconf"
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
        ans.path, ans.plan = path, plan
        ans.timings_ms = {"route": (t1 - t0) * 1e3, "plan": (t2 - t1) * 1e3, "query": (t3 - t2) * 1e3,
                          "verify": (t4 - t3) * 1e3, "total": (t4 - t0) * 1e3}
        return ans


if __name__ == "__main__":
    import sys
    vg = VehicleGPT()
    for q in sys.argv[1:] or ["Why has my driving range reduced over the last two weeks?"]:
        print(vg.ask(q).render())
        print()
