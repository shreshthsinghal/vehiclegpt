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
CONTROL = re.compile(r"\b(roll (down|up)|(open|close|lower|raise|crack)\b[^.?!]*\bwindows?|windows? (down|up)|"
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
]
# Common Hinglish words used in questions -> English (before routing and analysis).
HINGLISH = [(r"\bhua tha kya\b|\bhui thi kya\b|\btha kya\b", "was there"), (r"\bkitn[aie]\b", "how much"),
            (r"\bkab\b", "when"), (r"\bky[ou]n\b", "why"), (r"\bgaa?di\b", "car"), (r"\bbaa?rish\b", "rain"),
            (r"\bchalai?ya\b", "drove"), (r"\bkam\b", "lower"), (r"\bzyada\b", "more"), (r"\bkya\b", ""),
            (r"\bhai\b|\bhain\b|\btha\b|\bthi\b", ""), (r"\bmeri\b|\bmera\b", "my"), (r"\bkaisa\b|\bkaisi\b", "how")]


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
        self.examples = [(intent, ex) for intent, exs in lib.items() if intent != "unsafe_command" for ex in exs]
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
        asks_history = re.search(r"\b(why|when|what happened|did .* (turn|switch)|how many|how often|which|was there|were there)\b", q)
        if (UNSAFE.search(q) or CONTROL.search(q)) and not asks_history:
            return Route("unsafe_command", 1.0, 1.0, True, [])
        if not_recorded(q):
            return Route("not_recorded", 1.0, 1.0, True, [])
        sims = self.matrix @ embed(q)
        best: dict[str, list] = {}
        for i in np.argsort(-sims)[:25]:
            best.setdefault(self.labels[i], []).append(float(sims[i]))
        scores = sorted(((np.mean(v[:2]), k) for k, v in best.items()), reverse=True)
        (s1, top), (s2, _) = scores[0], (scores[1] if len(scores) > 1 else (0.0, None))
        nb = [(self.labels[i], self.examples[i][1], round(float(sims[i]), 3)) for i in np.argsort(-sims)[:3]]
        cands = [k for _, k in scores[:4]]
        return Route(top, float(s1), float(s1 - s2), bool(s1 >= THRESHOLD and s1 - s2 >= MARGIN), nb, cands)
