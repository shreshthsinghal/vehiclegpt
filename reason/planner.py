"""SMART path: the on-device small language model (SLM) turns an unusual question into a JSON PLAN.

The model never sees vehicle data and never writes the answer. It only fills in a small form:
  {"intent": ..., "signal": ..., "metric": ..., "window": ..., "date": ...}
The form is enforced by a JSON-schema GRAMMAR inside llama.cpp, so the model physically cannot produce anything
else (no free text, no code, no commands). The plan is validated again here before a playbook runs it.

Runtime: llama.cpp `llama-server` on 127.0.0.1 (loopback only: no network egress), model Qwen3-0.6B Q4_K_M.
The fixed instruction prefix is cached (cache_prompt), so each question only costs its own ~20 tokens + the plan.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import time
import urllib.request

from .dates import IST, Window
from .catalogue import CATALOGUE
from .playbooks import PLAYBOOKS, SIGNALS

INTENTS = [k for k in PLAYBOOKS if k not in ("unsafe_command", "privacy", "capabilities")]
SIGNAL_NAMES = list(dict.fromkeys([*SIGNALS.keys(), *CATALOGUE.keys()]))
METRICS = ["none", "longest", "shortest", "worst_eff", "total", "speed_compare", "tyre_last_hwy", "braking_rain",
           "per_month", "soc_per_100", "usage", "avg_speed", "efficiency", "trip_heat"]
ALWAYS = ["out_of_scope", "what_if", "emergency", "not_recorded", "vehicle_status_now"]   # rule-first intents the SLM may also pick
TOPICS = ["general", "brake_failure", "stuck_accelerator", "fire", "overheat", "blowout", "flood", "crash", "power_loss"]
# Second, tiny form used only for what-if questions (the regex parser fills the same fields; the SLM covers wording like
# "two hundred kilometres" or "a hundred and ten").
SCENARIO_SCHEMA = {"type": "object", "properties": {
    "distance_km": {"type": "integer", "minimum": 0, "maximum": 2000},
    "speed_kmh": {"type": "integer", "minimum": 0, "maximum": 200},
    "ac": {"enum": ["default", "on", "off"]},
    "target_soc": {"type": "integer", "minimum": 0, "maximum": 100}},
    "required": ["distance_km", "speed_kmh", "ac", "target_soc"], "additionalProperties": False}
TOPIC_SCHEMA = {"type": "object", "properties": {"topic": {"enum": TOPICS}}, "required": ["topic"],
                "additionalProperties": False}
WINDOWS = ["default", "today", "yesterday", "last_7_days", "last_14_days", "last_30_days", "this_month", "all",
           "first_2_weeks", "specific_date"]

SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"enum": INTENTS},
        "signal": {"enum": ["none", *SIGNAL_NAMES]},
        "metric": {"enum": METRICS},
        "window": {"enum": WINDOWS},
        "date": {"type": "string", "maxLength": 10},
    },
    "required": ["intent", "signal", "metric", "window", "date"],
    "additionalProperties": False,
}

SYSTEM = """You convert a driver's or service engineer's question about THEIR OWN VEHICLE into a JSON plan.
You never answer the question and never invent data. Choose:
intent: range_drop (why range fell) | efficiency_change (when efficiency got worse) | adas_events (driver-assist
disengagements) | before_event (what led to a warning or fault) | segment_compare (city vs highway, weekday vs weekend)
| signal_trend (how one measured signal behaved: max, average, trend, energy) | dtc_lookup (fault codes) |
charging_history (charging sessions) | vehicle_status_now (current value of anything: tyres, gear, camera, battery
health now) | trip_stats (trips: longest, worst, totals, speed, per month, usage) | data_quality (missing data) |
health_summary (overall problems, repairs, habits) | what_if (hypothetical: can I drive X km, range at a speed, AC off,
time to charge) | improve (how to get more range) | safety_log (safety-related events) | emergency (the driver is in
danger NOW: brakes, fire, smoke, crash, flood, tyre burst) | not_recorded (something the car does not record: music,
drive mode, horn, accidents) | out_of_scope (not about this vehicle).
signal: which measurement, or none. metric: for trip_stats, else none.
window: the time the question refers to; use specific_date with date YYYY-MM-DD when a calendar date is named,
otherwise date "". Use default when no time is mentioned."""

EXAMPLES = [
    ("how warm did the motor get on the highway last week",
     {"intent": "signal_trend", "signal": "motor_temp", "metric": "none", "window": "last_7_days", "date": ""}),
    ("anything wrong with the car i should tell the garage",
     {"intent": "health_summary", "signal": "none", "metric": "none", "window": "default", "date": ""}),
    ("which drive used the most energy per km this month",
     {"intent": "trip_stats", "signal": "none", "metric": "worst_eff", "window": "this_month", "date": ""}),
    ("what led up to the tyre warning on 30 Oct",
     {"intent": "before_event", "signal": "none", "metric": "none", "window": "specific_date", "date": "2026-10-30"}),
    ("did we lose any recordings yesterday",
     {"intent": "data_quality", "signal": "none", "metric": "none", "window": "yesterday", "date": ""}),
    ("how much juice is left",
     {"intent": "vehicle_status_now", "signal": "state_of_charge", "metric": "none", "window": "default", "date": ""}),
    ("would a hundred and fifty kilometre drive be ok with what's left",
     {"intent": "what_if", "signal": "none", "metric": "none", "window": "default", "date": ""}),
    ("the pedal goes all the way down and the car won't slow",
     {"intent": "emergency", "signal": "none", "metric": "none", "window": "default", "date": ""}),
    ("is the dashcam ok",
     {"intent": "vehicle_status_now", "signal": "camera", "metric": "none", "window": "default", "date": ""}),
    ("what is the best pizza in goa",
     {"intent": "out_of_scope", "signal": "none", "metric": "none", "window": "default", "date": ""}),
]


def _chatml(question: str) -> str:
    shots = "".join(f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                    f"{json.dumps(p, separators=(',', ':'))}<|im_end|>\n" for q, p in EXAMPLES)
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n{shots}"
            f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")


def validate(plan: dict) -> dict:
    """Second line of defence after the grammar: only known values pass."""
    out = {"intent": plan.get("intent") if plan.get("intent") in INTENTS else None,
           "signal": plan.get("signal") if plan.get("signal") in SIGNAL_NAMES else None,
           "metric": plan.get("metric") if plan.get("metric") in METRICS[1:] else None,
           "window": plan.get("window") if plan.get("window") in WINDOWS else "default",
           "date": plan.get("date") or ""}
    if out["window"] == "specific_date":
        try:
            dt.date.fromisoformat(out["date"])
        except ValueError:
            out["window"], out["date"] = "default", ""
    return out


def window_from_plan(plan: dict, now: float) -> Window | None:
    w = plan.get("window", "default")
    today = dt.datetime.fromtimestamp(now, IST).replace(hour=0, minute=0, second=0, microsecond=0)
    day = lambda d, label: Window(d.timestamp(), (d + dt.timedelta(days=1)).timestamp(), label)
    last = lambda n, label: Window((today - dt.timedelta(days=n - 1)).timestamp(), now + 1, label)
    if w == "today":
        return day(today, "today")
    if w == "yesterday":
        return day(today - dt.timedelta(days=1), "yesterday")
    if w in ("last_7_days", "last_14_days", "last_30_days"):
        n = int(w.split("_")[1])
        return last(n, f"the last {n} days")
    if w == "this_month":
        return Window(today.replace(day=1).timestamp(), now + 1, f"{today:%B} so far")
    if w == "specific_date" and plan.get("date"):
        d = dt.datetime.fromisoformat(plan["date"]).replace(tzinfo=IST)
        return day(d, f"{d:%d %b}")
    return None


class SlmPlanner:
    def __init__(self, url: str | None = None, timeout: float = 30.0):
        self.url = (url or os.environ.get("VGPT_LLM_URL", "http://127.0.0.1:8080")).rstrip("/")
        assert self.url.startswith(("http://127.0.0.1", "http://localhost")), "planner must be on-device (loopback)"
        self.timeout = timeout
        self.last: dict = {}

    def warmup(self) -> float:
        """Process the fixed instruction prefix once at start-up so the first real question is fast."""
        t0 = time.perf_counter()
        try:
            self.raw("warm up")
        except Exception:          # noqa: BLE001
            pass
        return (time.perf_counter() - t0) * 1000

    def raw(self, question: str, candidates: list | None = None, schema: dict | None = None,
            prompt: str | None = None) -> tuple[dict, dict]:
        if schema is None:
            schema = SCHEMA
            if candidates:                 # retrieval-augmented planning: router's top candidates + rule-first intents
                allowed = [c for c in candidates if c in INTENTS] + [a for a in ALWAYS if a in INTENTS or a == "out_of_scope"]
                schema = json.loads(json.dumps(SCHEMA))
                schema["properties"]["intent"] = {"enum": list(dict.fromkeys(allowed))}
        body = {"prompt": prompt or _chatml(question[:300]), "json_schema": schema, "n_predict": 64, "temperature": 0.0,
                "cache_prompt": True, "stop": ["<|im_end|>"]}
        req = urllib.request.Request(self.url + "/completion", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            out = json.loads(r.read())
        ms = (time.perf_counter() - t0) * 1000
        timings = out.get("timings", {})
        return json.loads(out["content"]), {"wall_ms": ms, "prompt_n": timings.get("prompt_n"),
                                            "predicted_n": timings.get("predicted_n"),
                                            "prompt_ms": timings.get("prompt_ms"), "predicted_ms": timings.get("predicted_ms"),
                                            "tok_per_s": timings.get("predicted_per_second")}

    def plan(self, question: str, ctx, candidates: list | None = None) -> dict:
        try:
            raw, stats = self.raw(question, candidates)
        except Exception as exc:                                    # model down -> caller falls back to router
            self.last = {"error": f"{type(exc).__name__}: {exc}"}
            return {"intent": None}
        p = validate(raw)
        self.last = {"raw": raw, **stats}
        w = window_from_plan(p, ctx.now)
        if w is not None:
            p["window_override"] = (w.start, w.end, w.label)
        p["slm"] = stats
        try:
            if p["intent"] == "what_if":           # second tiny form: trip distance, speed, AC, charge target
                sc, st2 = self.raw(question, schema=SCENARIO_SCHEMA, prompt=_form(question, SCENARIO_HINT))
                p.update({k: v for k, v in sc.items() if v not in (0, "default")})
                p["slm"]["second_ms"] = st2["wall_ms"]
            elif p["intent"] == "emergency":
                tp, st2 = self.raw(question, schema=TOPIC_SCHEMA, prompt=_form(question, TOPIC_HINT))
                p["topic"] = tp.get("topic") if tp.get("topic") in TOPICS else "general"
                p["slm"]["second_ms"] = st2["wall_ms"]
        except Exception:                          # noqa: BLE001  the regex parser still fills these fields
            pass
        return p


SCENARIO_HINT = ("Extract numbers from a driver's what-if question. distance_km: trip length in km (0 if none). "
                 "speed_kmh: driving speed (0 if none). ac: on/off if mentioned, else default. target_soc: charge "
                 "target in percent (0 if none).")
TOPIC_HINT = "Classify the driver's emergency: " + ", ".join(TOPICS) + "."


def _form(question: str, hint: str) -> str:
    return (f"<|im_start|>system\n{hint}<|im_end|>\n<|im_start|>user\n{question[:300]}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n\n</think>\n\n")
