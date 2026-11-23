"""Intent router: nearest-neighbour search over the intent library. The FAST path.

score(intent) = mean of its top-2 cosine similarities to the question.
If the best score is high enough AND clearly ahead of the runner-up, we trust it (FAST path, no SLM).
Otherwise the question goes to the SLM planner (SMART path). Safety intents are checked by rules first.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .embed import embed, embed_many

HERE = Path(__file__).parent
THRESHOLD = 0.30
MARGIN = 0.04

UNSAFE = re.compile(r"\b(unlock|lock the|open the|close the|turn (on|off)|switch (on|off)|disable|enable|activate|"
                    r"deactivate|start the (car|engine|motor)|(clear|erase|delete|wipe)\b[^.?!]*\b(codes?|dtcs?|faults?|errors?|logs?|data)|reset|change the|set the|"
                    r"send|upload|post|email|transmit|share my|ignore (your|all|previous)|override)\b")
PRIVACY = re.compile(r"\b(who (drove|was driving|is driving)|which person|where was (my|the) (wife|husband|son|daughter|"
                     r"friend|driver)|track (the )?driver|identify the driver|(home|house|office|work) address|where does .* live|"
                     r"(owner|driver)'?s? (name|phone|address|location)|phone number of)\b")


# Driver/vehicle CONTROL requests (VehicleGPT is read-only), beyond the verbs above.
CONTROL = re.compile(r"\b((apply|press|hit|use) the brakes?\b(?! pedal)( for me)?|brake for me|stop the car|slow (the car )?down for me|"
                     r"park the car|drive (for me|the car)|disable (the )?(abs|esc|airbags?|traction)|roll (down|up)|(open|close|lower|raise|crack)\b[^.?!]*\bwindows?|windows? (down|up)|"
                     r"play\b[^.?!]*\b(music|songs?|radio|playlist)|(volume|louder|loud music)|honk|"
                     r"(remote(ly)?|take over|control)\b[^.?!]*\b(steer\w*|brak\w*|accelerat\w*|the car)|steer the car|"
                     r"(silence|mute|hide|stop|suppress)\b[^.?!]*\b(warnings?|chimes?|alerts?|alarms?))\b")
# Location history is personal data: refuse, like identifying people.
PLACES = re.compile(r"\b((which|what) (places|locations)|places (i|we) (visit|went|go|drove)\w*|(where|places) (did|have) (i|we) "
                    r"(go|gone|been|visit\w*|drive|drove|park\w*)|location history|most visited|visited the most|"
                    r"where (was|is) (the car|it) parked)\b")
# Things this vehicle's recorded data does not contain: answer honestly instead of guessing.
NOT_RECORDED = [
    ("navigation", r"(find|nearest|near me|nearby|locate|where (is|are)|show me|suggest)\b[^.?!]*charg\w* (stations?|points?)|"
                   r"charg\w* (stations?|points?)\b[^.?!]*(near|on my way|on the way|nearby|around|route)|"
                   r"route to|directions to|navigate|weather forecast|traffic (ahead|now|on my way)"),
    ("media", r"\b(songs?|music|playlist|podcast|spotify|radio station)\b"),
    ("drive_mode", r"\b(drive|driving) modes?\b|\b(eco|sport|comfort) mode\b"),
    ("collision", r"\bcollision|\baeb\b|emergency brak\w* (system|assist)|forward warning"),
    ("accident", r"\b(accident|crash\w*|airbags?|collided)\b"),
    ("horn", r"\bhorn\b"),
    ("unrecorded", r"coolant level|brake fluid|washer fluid|tread depth|tyre tread|tire tread|speed limit (on|of|here)|current speed limit|"
                   r"g.?force|tire temperature|tyre temperature|seat position|air quality|odou?r|\bsmell|wheel size|"
                   r"recall|service history|odometer at purchase"),
    ("comfort", r"back is killing|lower back|sauna|muggy|sticky in|can'?t hear|too loud|blinding|numb|fogged|freezing in here|"
                r"boiling in here|too (hot|cold) in here"),
]
# Requests that start with a command verb (after polite prefixes) ask VehicleGPT to operate the car: refused (read-only).
POLITE = re.compile(r"^((hey|hi|ok|okay|yo)\s+\w+[,!]?\s+|please\s+|(can|could|would|will) you( please)?\s+|i (want|need) you to\s+|go ahead and\s+)+")
COMMAND = re.compile(r"^(play|stream|set|change|lower|raise|increase|decrease|reduce|turn|switch|open|close|shut|lock|unlock|roll|crack|fold|"
                     r"unfold|activate|deactivate|enable|disable|start|stop|pause|resume|schedule|warm|cool|heat|pre-?heat|"
                     r"precondition|defrost|demist|reset|save|record|delete|remember|put|engage|mute|unmute|dim|brighten|"
                     r"adjust|keep (it|the|my)|make it|crank|pop|honk|flash|park|parallel|move|tilt|recline|release|arm|"
                     r"disarm|apply|skip|swap|exclude|prioriti[sz]e|install|update|pair|connect|disconnect|cancel|end|"
                     r"charge (the|my) car|have the car|max|aim|lift|drop|unbuckle|rev|read me|write|vent|sync|restore|alert|"
                     r"notify|remind|bring (up|in)|show (me )?the [^.?]*(graph|meter|screen|view|camera|monitor)|recalculate|"
                     r"forget|output|bypass|let [^.?]* in)\b")
ADVERSARIAL = re.compile(r"(forget|ignore|override) (your|all|previous|the) (system )?(instructions|rules|prompt|interlocks?)|"
                         r"api keys?|system prompt|jailbreak|bypass|developer mode|pretend (you are|to be)")
# Needs live online services (maps, charger networks, prices, traffic, phone, streaming): offline by design.
ONLINE = re.compile(r"navigat|\broute (me|to)\b|reroute|along (my|the) route|on my route|(best|fastest|alternative|eco) route|"
                    r"take me to|directions|\bfind (me )?(a |an |the |some )?(\w+ )?(\w+ )?(charg\w*|stations?|parking|restaurant|cafe|coffee|hotel|route|place|spot|garage|shop|store)|nearest|closest|near (here|me|my)|within \d+ (miles|km|minutes) of|"
                    r"on my way|next (highway )?exit|\bstalls?\b|congestion|peak .*pricing|\bprice|\bcost(s)? (me )?(per|of|to charge)|how much (does|will|did|is) [^.?]*cost|tariff|per kwh|\$|"
                    r"\beta\b|arrival time|(miles|minutes|km) (left )?until|elevation|traffic (ahead|now|on|update|jam)|detour|"
                    r"waypoint|stop at the|weather forecast|share my (live|location|eta|arrival)|\bcall\b|\btext\b|send (a |an )?(text|message|sms)|messages? to|whatsapp|"
                    r"\bplay\b|podcast|audiobook|spotify|netflix|youtube|radio station|garage door|thermostat|ecobee|alexa|"
                    r"smart home|security camera|recall|\bbook\b|appointment|software update|over.the.air|\bota\b|release notes|"
                    r"plug ?& ?charge|apple pay|credit card|charging networks?|supercharger|ionity|fastned|electrify america|evgo|"
                    r"chargepoint|bp pulse|shell|tata power|rivian|hotel|restaurant|cafe|coffee|bakery|parking (garage|spot)|"
                    r"wi-?fi|hotspot|bluetooth|nacs|level 2 charger|charger (is|are) (busy|free|available)|open stalls")
# Owner's-manual / general EV knowledge (not in the car's recorded data).
KNOWLEDGE = re.compile(r"^(explain how|why does (my car|it) (make|say)|how (do|can|should|would) (i|you|we)|how often should|where (is|are|do|can)|can (a|an|i) (regular|plug|use|tow|leave|jump)|"
                       r"is it (bad|ok|okay|safe|normal|fine|better) (if|to)|should i|do i (still )?need (to|my|a)|what is the (best|right|recommended)|"
                       r"why (does|do) (evs?|electric cars)|what (changes|happens) when i (put|switch)|how long does .* usually)")
DATA_WORDS = re.compile(r"improv|increase|extend|save (battery|energy)|get more|more range|efficien|\bmy (range|battery|car|trips?|charging|efficiency|tyres?|tires?)\b.*\b(now|last|this|history|week|month|yesterday)|"
                        r"\b(last|this) (week|month)|yesterday|history|recorded")
ICON = re.compile(r"(icon|symbol|light|lamp|telltale)\b[^.?]*(dashboard|cluster|lit|came on|showing|popped up)|"
                  r"what does (the|this|that|a) [^.?]*(icon|symbol|light) mean|(turtle|exclamation) (icon|symbol|mark)")


# Common Hinglish words used in questions -> English (before routing and analysis).
HINGLISH = [(r"\bhua tha kya\b|\bhui thi kya\b|\btha kya\b", "was there"), (r"\bkitn[aie]\b", "how much"),
            (r"\bkab\b", "when"), (r"\bky[ou]n\b", "why"), (r"\bgaa?di\b", "car"), (r"\bbaa?rish\b", "rain"),
            (r"\bchalai?ya\b", "drove"), (r"\bkam\b", "lower"), (r"\bzyada\b", "more"), (r"\bkya\b", ""),
            (r"\bhai\b|\bhain\b|\btha\b|\bthi\b", ""), (r"\bmeri\b|\bmera\b", "my"), (r"\bkaisa\b|\bkaisi\b", "how")]


# Component signals: questions about their history always go to the signal-history analysis.
COMPONENT = {"motor_temp", "motor_speed", "torque", "coolant_temp", "coolant_pump", "batt_voltage", "batt_current", "cell_min",
             "cabin_temp", "set_temp", "fan", "lv_voltage", "lane_conf", "visibility", "steering", "accel_pedal",
             "brake_pedal", "power_limit", "batt_power"}
# Intents decided only by explicit rules (never by similarity): commands, emergencies, scenarios, the safety log.
RULES_ONLY = {"unsafe_command", "emergency", "what_if", "safety_log", "capabilities", "privacy"}
# What-if questions: trips, speeds, AC on/off, charging time.
SCENARIO = re.compile(r"\bwhat if\b|\bif i (drive|go|turn|switch|keep|travel|use|charge)|can i (make|reach|drive|do|get to|go)\b|"
                      r"will i (make|reach|have enough|get)|(enough|sufficient) (charge|battery|range) (to|for)|"
                      r"how (long|much time) (will it take |would it take |does it take )?to (charge|fill)|(charge|charging) (to|till|until) \d+ ?%|"
                      r"\bwithout (the )?(ac|a/c|air.?con)|\d+ ?(km|kms) (trip|drive|journey)|trip of \d+|drive \d+ ?(km|kms)\b|"
                      r"\bat \d+ ?(km/?h|kmph)|\d+ ?(km/?h|kmph)? (vs|versus|or) \d+ ?(km/?h|kmph)|\brange if\b|if i (didn'?t|don'?t) use|"
                      r"(left|remaining) when i (reach|get|arrive)|when i (reach|get to|arrive)|get stranded|stranded before|"
                      r"enough juice|still \d+ ?(km|kms|miles|mi)\b.*from|how (slow|fast) (do|should|must|can) i|"
                      r"how many (miles|km|kms|kilomet\w*) can i|how long (to|until|till|will it take to) (reach|get to) \d+|"
                      r"realistic range|range .*at -?\d+ ?°|do i need to charge before|range .* per hour|per hour .* charg|"
                      r"\bi'?m at \d+ ?%|if i leave (now )?at \d+")


def english(question: str) -> str:
    q = question
    for pat, rep in HINGLISH:
        q = re.sub(pat, rep, q, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", q).strip()


def not_recorded(q: str) -> str | None:
    for kind, pat in NOT_RECORDED:
        if re.search(pat, q):
            return kind
    return None


@dataclass
class Route:
    intent: str
    score: float
    margin: float
    fast: bool
    neighbours: list
    candidates: list = None          # top intents, handed to the SLM planner as its only allowed choices


class Router:
    def __init__(self, learned_path: Path | None = None):
        lib = yaml.safe_load((HERE / "intents.yaml").read_text(encoding="utf-8"))
        # commands are blocked by explicit rules only (never by similarity), so they are not kNN classes
        self.examples = [(intent, ex) for intent, exs in lib.items() if intent not in RULES_ONLY for ex in exs]
        self.learned_path = learned_path
        if learned_path and learned_path.exists():
            for line in learned_path.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                self.examples.append((r["intent"], r["question"]))
        self.labels = [i for i, _ in self.examples]
        self.matrix = embed_many([e for _, e in self.examples])

    def learn(self, question: str, intent: str, plan: dict):
        """A validated SLM plan becomes a new FAST-path example."""
        self.examples.append((intent, question))
        self.labels.append(intent)
        self.matrix = np.vstack([self.matrix, embed(question)])
        if self.learned_path:
            with open(self.learned_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"question": question, "intent": intent, "plan": plan}) + "\n")

    def route(self, question: str) -> Route:
        q = english(question).lower()
        if PRIVACY.search(q) or PLACES.search(q):
            return Route("privacy", 1.0, 1.0, True, [])
        from .live import NOW_WORDS, TREND_WORDS, emergency_topic
        from .catalogue import find as find_signals
        if ADVERSARIAL.search(q):
            return Route("unsafe_command", 1.0, 1.0, True, [])
        if emergency_topic(q):
            return Route("emergency", 1.0, 1.0, True, [])
        from .live import CAPS
        if re.search(CAPS, q):
            return Route("capabilities", 1.0, 1.0, True, [])
        if ICON.search(q):
            return Route("before_event", 1.0, 1.0, True, [])
        if KNOWLEDGE.search(q) and not DATA_WORDS.search(q) and not SCENARIO.search(q):
            return Route("not_recorded", 1.0, 1.0, True, [])
        if COMMAND.search(POLITE.sub("", q).strip()) or re.search(r"update (my|the) (range|wheel|tyre|tire)", q):
            return Route("unsafe_command", 1.0, 1.0, True, [])
        if re.search(r"(only|just) getting \d+ ?kw|charging (speed|power|rate) (is )?(slow|low)|why (is|was) (my )?charging (so )?slow", q):
            return Route("charging_history", 1.0, 1.0, True, [])
        if re.search(r"(can|will) i make it|enough (charge|battery|range|juice)|get stranded|how (slow|fast) (do|should|must|can) i", q):
            return Route("what_if", 1.0, 1.0, True, [])
        if ONLINE.search(q) and not re.search(r"\d+ ?(km|kms|miles|mi)\b", q) or re.search(
                r"\bfrom [a-z' ]+ to [a-z' ]+.*(charging stops?|how long|how far)|charging stops? (will|do) i need|mountain|downhill|"
                r"uphill|wallet|paid charger|payment", q):
            if not re.search(r"\bi'?m at \d+ ?%|if i leave (now )?at \d+|how long (to|until|till) (reach|get to) \d+", q):
                return Route("not_recorded", 1.0, 1.0, True, [])
        if SCENARIO.search(q):
            return Route("what_if", 1.0, 1.0, True, [])
        asks_history = re.search(r"\b(why|when|what happened|did .* (turn|switch)|how many|how often|which|was there|were there)\b", q)
        if (UNSAFE.search(q) or CONTROL.search(q)) and not asks_history:
            return Route("unsafe_command", 1.0, 1.0, True, [])
        if ONLINE.search(q):
            return Route("not_recorded", 1.0, 1.0, True, [])
        if re.search(r"\b(rear|reverse|back(up)?|parking) cam", q):
            return Route("vehicle_status_now", 1.0, 1.0, True, [])
        if not_recorded(q):
            return Route("not_recorded", 1.0, 1.0, True, [])
        if re.search(r"safety (log|report|history|events?|record|summary|issues|incidents|problems)|safety.related|any safety|"
                     r"stability control|\besc\b|kick(ed)? in|\babs\b (kick|activ|engag)", q):
            return Route("safety_log", 1.0, 1.0, True, [])
        if SCENARIO.search(q):
            return Route("what_if", 1.0, 1.0, True, [])
        sig = find_signals(q)
        if sig and re.search(NOW_WORDS, q) and not re.search(TREND_WORDS, q):
            return Route("vehicle_status_now", 1.0, 1.0, True, [])
        if re.search(r"\b(ac|a/c|air.?con\w*)\b[^.?]*\b(energy|kwh|use|used|consum\w*)", q) and not re.search(r"charg", q):
            return Route("signal_trend", 1.0, 1.0, True, [])
        if sig and sig[0] in COMPONENT and not re.search(r"warning|fault|code|dtc|error|alert|light|\bwhy\b", q) and re.search(TREND_WORDS + r"|highest|max|lowest|minimum|peak|behave", q):
            return Route("signal_trend", 1.0, 1.0, True, [])
        sims = self.matrix @ embed(q)
        best: dict[str, list] = {}
        for i in np.argsort(-sims)[:25]:
            best.setdefault(self.labels[i], []).append(float(sims[i]))
        scores = sorted(((np.mean(v[:2]), k) for k, v in best.items()), reverse=True)
        (s1, top), (s2, _) = scores[0], (scores[1] if len(scores) > 1 else (0.0, None))
        nb = [(self.labels[i], self.examples[i][1], round(float(sims[i]), 3)) for i in np.argsort(-sims)[:3]]
        cands = [k for _, k in scores[:4]]
        if top == "vehicle_status_now" and re.search(TREND_WORDS, q):      # a time-bounded question is not "right now"
            alt = [k for k in cands if k != "vehicle_status_now"]
            top = "signal_trend" if "signal_trend" in alt[:3] or find_signals(q) else alt[0]
            s1, s2 = s1, s1
        return Route(top, float(s1), float(s1 - s2), bool(s1 >= THRESHOLD and s1 - s2 >= MARGIN), nb, cands)
