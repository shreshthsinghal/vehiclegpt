"""Label the external 1,050-scenario EV-assistant dataset with what VEHICLEGPT should do under our constraints.

The dataset was written for a general, cloud-connected, car-controlling assistant. VehicleGPT is offline, read-only and
answers from recorded vehicle data, so each row is mapped to one of:
  intent:<a>|<b>  answer with one of these analyses (any listed one is acceptable)
  refuse          a command to change/operate the car (read-only), or an unsafe / adversarial request
  no_data         needs maps/internet/phone/media/owner's-manual knowledge, or a signal this car does not record
  skip            multi-turn context we do not have (coreference, barge-in), excluded from scoring
Rows are split train/test by utterance TEMPLATE (numbers removed), so near-duplicates never cross the split.

  python -m eval.import_ev1050      -> eval/external/ev1050_labeled.yaml
"""
from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

import yaml

HERE = Path(__file__).parent / "external"

CONTROL = re.compile(r"^(please |yo car, |hey car, )?(set|change|turn|open|close|roll|crack|fold|unfold|switch|activate|enable|disable|"
                     r"start|stop|unlock|lock|max|aim|put|raise|lower|remember|save|reset|schedule|have the car|only charge|pop|"
                     r"crank|make it|unbuckle|rev|ignore|parallel|change lanes|apply|engage|dim|mute|increase|decrease|keep|"
                     r"adjust|cap|limit|precondition|warm|cool|heat|defrost|vent|honk|flash|tilt|move|recline|release|arm|"
                     r"disarm|show me the (dashcam|camera)|record|delete|update|install|pair|connect|end my|cancel|skip|swap|"
                     r"recalculate|factor|use hov|switch the map|add|exclude|prioritize|charge the car)\b", re.I)
NETWORK = re.compile(r"navigat|route|\bfind\b|nearest|closest|near (my|me|destination)|along my|on my way|stalls?\b|congestion|"
                     r"peak .*hours|price|tariff|\$|eta\b|arrival time|minutes (left|until)|elevation|traffic|detour|waypoint|"
                     r"weather forecast|share my|\bcall\b|text |message|whatsapp|play |pause|podcast|spotify|radio|song|music|"
                     r"garage door|thermostat|smart home|security camera|recall|book a|appointment|software update|release notes|"
                     r"plug & charge|apple pay|credit card|network|supercharger|ionity|fastned|electrify|evgo|chargepoint|"
                     r"bp pulse|shell|tata power|rivian|hotel|restaurant|coffee|parking garage|wi-fi|hotspot|bluetooth", re.I)

ANSWER = {  # subcategory -> acceptable intents
    "Range & SOC Query": "what_if|vehicle_status_now", "Range Calculation": "what_if",
    "Charging Time Estimate": "what_if", "Charging Stop Necessity": "what_if", "Arrival Energy Buffer": "what_if",
    "Arrival SOC Forecast": "what_if", "Range Feasibility Check": "what_if",
    "Charging History": "charging_history", "Tire Pressure Monitoring": "vehicle_status_now",
    "Dashboard Warning Lights": "before_event|dtc_lookup|emergency|safety_log|vehicle_status_now",
    "Regen Limitation Explanation": "signal_trend|vehicle_status_now|not_recorded",
    "Cold Weather Range": "range_drop|what_if|not_recorded|improve",
    "12V Battery Check": "vehicle_status_now|dtc_lookup|signal_trend",
    "Collision / Airbag": "emergency", "Emergency Panic Response": "emergency", "Flat Tire Emergency": "emergency",
    "Out of Charge": "emergency|what_if|not_recorded", "Emergency Door Release": "emergency|not_recorded",
    "Implicit Drowsiness": "emergency", "Implicit Low Battery": "what_if|vehicle_status_now",
    "Implicit Range Intent": "what_if", "Efficiency & Trip Meter": "trip_stats",
    "Battery Health & Longevity": "vehicle_status_now|signal_trend|not_recorded|improve",
    "Charging Speed": "charging_history|what_if", "Range Anxiety & Efficiency": "vehicle_status_now|improve|what_if",
    "Charging Troubleshooting": "charging_history|dtc_lookup|not_recorded",
    "Privacy / Data Security": "privacy",
}
SKIP = {"Coreference", "Coreference Resolution", "Barge-In Cancellation"}


def label(row: dict) -> tuple[str, str]:
    q, sub = row["User_Utterance"].strip(), row["Subcategory"]
    ql = q.lower()
    if sub in SKIP:
        return "skip", "needs earlier conversation turns"
    if sub in ANSWER:
        intents = ANSWER[sub]
        if sub == "Implicit Drowsiness" or sub in ("Collision / Airbag", "Emergency Panic Response", "Flat Tire Emergency"):
            return f"intent:{intents}", "emergency guidance (fixed, cited)"
        if CONTROL.search(q) and sub not in ("Dashboard Warning Lights",):
            return f"refuse|intent:{intents}", "command + data question"
        return f"intent:{intents}", "answerable from recorded data / model"
    if re.search(r"privacy|address|call history|api key|system instructions|bypass|interlock|airbag|seatbelt|"
                 r"traction control|stability control|headlights|driver door|trunk while|park right now", ql):
        return "refuse", "safety gate / adversarial"
    if re.search(r"how many miles of juice|juice do i have|stranded before|enough juice|still \d+ miles from", ql):
        return "intent:what_if|vehicle_status_now", "implicit range"
    if CONTROL.search(q) or re.search(r"\b(turn it up|open it|turn that off|make it warmer)\b", ql):
        return "refuse|no_data", "command (VehicleGPT is read-only)"
    if NETWORK.search(q):
        return "no_data|refuse", "needs maps / internet / phone (offline by design)"
    return "no_data|refuse", "not in recorded data (owner's manual / general knowledge / hardware)"


def main():
    rows = list(csv.DictReader(open(HERE / "ev_dashboard_assistant_1050.csv", encoding="utf-8", errors="replace")))
    out = []
    for r in rows:
        exp, note = label(r)
        tmpl = re.sub(r"-?\d+(?:[.,]\d+)?", "#", r["User_Utterance"].lower())
        split = "test" if int(hashlib.sha1(tmpl.encode()).hexdigest(), 16) % 10 < 3 else "train"
        out.append({"id": r["ID"], "q": r["User_Utterance"], "category": r["Category"], "sub": r["Subcategory"],
                    "expect": exp, "why": note, "split": split})
    yaml.safe_dump({"source": "ev_dashboard_assistant_1050_scenarios.csv (shared by the user, 23 Nov 2026)",
                    "questions": out}, open(HERE / "ev1050_labeled.yaml", "w"), sort_keys=False, allow_unicode=True, width=200)
    from collections import Counter
    print(Counter(o["expect"].split(":")[0].split("|")[0] for o in out), Counter(o["split"] for o in out))


if __name__ == "__main__":
    main()
