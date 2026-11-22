"""Ask VehicleGPT questions in the terminal.

  python -m reason.ask                         # interactive; uses the language model if llama-server is running
  python -m reason.ask "why is my range lower?"
"""
from __future__ import annotations

import json
import sys
import urllib.request

from .engine import VehicleGPT
from .planner import SlmPlanner


def _llm_up(url="http://127.0.0.1:8080") -> bool:
    try:
        return urllib.request.urlopen(url + "/health", timeout=1).status == 200
    except Exception:          # noqa: BLE001
        return False


def main():
    planner = SlmPlanner() if _llm_up() else None
    vg = VehicleGPT(planner=planner)
    print(f"VehicleGPT ready (language model: {'ON' if planner else 'off - FAST path only'}). Empty line to quit.\n")
    qs = sys.argv[1:]
    while True:
        q = qs.pop(0) if qs else (input("ask> ").strip() if not sys.argv[1:] else "")
        if not q:
            break
        a = vg.ask(q)
        print(a.render())
        if planner and a.path.startswith("SMART"):
            print("      plan:", json.dumps({k: v for k, v in a.plan.items() if k not in ("slm", "window_override")}))
        print()


if __name__ == "__main__":
    main()
