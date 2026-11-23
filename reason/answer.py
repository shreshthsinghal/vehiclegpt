"""Answer card (WHAT / WHY / PROOF / NEXT) + the verifier.

Every number a playbook wants to show is first registered as a FACT (value + where it came from).
The narrator writes text using those facts. The VERIFIER then reads the finished text, finds every number,
date and time in it, and checks each one against the registered facts. Anything it cannot trace is reported;
an answer with untraceable numbers is never shown as-is (this is what stops "hallucinated" numbers).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

DTC_RE = re.compile(r"\b[PCBU][0-9A-F]{4}\b")
TIME_RE = re.compile(r"\b\d{2}:\d{2}(?::\d{2})?\b")
DATE_RE = re.compile(r"\b\d{1,2} (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b")
NUM_RE = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?![\w])")
# Numbers that are part of names/units, not claims.
SAFE_PHRASES = re.compile(r"kWh(/| per )100 ?km|\b12 ?V\b|0x[0-9A-Fa-f]+|-60 s|\+10 s|\b2 s glance\b|AES-256|top-\d|Q[1-5]\b",
                          re.IGNORECASE)


@dataclass
class Fact:
    name: str
    value: float | str
    unit: str = ""
    source: str = ""


@dataclass
class Answer:
    question: str
    intent: str
    what: str = ""
    why: list[str] = field(default_factory=list)
    proof: list[tuple[str, str]] = field(default_factory=list)       # (label, value text)
    next: str = ""
    details: list[str] = field(default_factory=list)                  # optional timeline / table lines
    facts: list[Fact] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)                 # which queries produced the facts
    path: str = "FAST"
    plan: dict = field(default_factory=dict)
    timings_ms: dict = field(default_factory=dict)
    verified: bool = False
    unverified: list[str] = field(default_factory=list)
    confidence: str = "high"
    chart: dict | None = None                                         # {"type": "bar"|"line", ...} for the HMI

    # ---------------------------------------------------------------- facts
    def fact(self, name, value, unit="", source=""):
        self.facts.append(Fact(name, value, unit, source))
        return value

    def text(self) -> str:
        return "\n".join([self.what, *self.why, *(f"{a}: {b}" for a, b in self.proof), self.next, *self.details])

    def render(self) -> str:
        lines = [f"Q: {self.question}", f"[{self.path} path | intent {self.intent} | "
                 f"{self.timings_ms.get('total', 0):.0f} ms | numbers verified: {'yes' if self.verified else 'NO'}]",
                 f"WHAT  {self.what}"]
        lines += [("WHY   " if i == 0 else "      ") + w for i, w in enumerate(self.why)]
        if self.proof:
            lines.append("PROOF " + " | ".join(f"{a}: {b}" for a, b in self.proof))
        if self.next:
            lines.append(f"NEXT  {self.next}")
        lines += ["      " + d for d in self.details]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"question": self.question, "intent": self.intent, "path": self.path, "what": self.what,
                "why": self.why, "proof": [{"label": a, "value": b} for a, b in self.proof], "next": self.next,
                "details": self.details, "evidence": self.evidence, "verified": self.verified,
                "unverified": self.unverified, "confidence": self.confidence,
                "plan": {k: v for k, v in self.plan.items() if k != "window_override"},
                "timings_ms": self.timings_ms, "chart": self.chart}


# ==================================================================== verifier
def _decimals(tok: str) -> int:
    return len(tok.split(".")[1]) if "." in tok else 0


def verify(ans: Answer) -> Answer:
    text = ans.text()
    nums, strs = [], set()
    for f in ans.facts:
        if isinstance(f.value, (int, float)):
            nums.append(float(f.value))
        else:
            strs.add(str(f.value))
    fact_text = " ".join(strs)
    bad = []
    for m in TIME_RE.finditer(text):
        if m.group(0) not in fact_text:
            bad.append(m.group(0))
    for m in DATE_RE.finditer(text):
        if m.group(0) not in fact_text:
            bad.append(m.group(0))
    for m in DTC_RE.finditer(text):
        if m.group(0) not in fact_text:
            bad.append(m.group(0))
    clean = SAFE_PHRASES.sub(" ", text)
    clean = TIME_RE.sub(" ", DATE_RE.sub(" ", DTC_RE.sub(" ", clean)))
    for m in NUM_RE.finditer(clean):
        tok = m.group(0)
        x = float(tok)
        d = _decimals(tok)
        tol = 0.5 * 10 ** (-d) + 1e-9
        if not any(abs(abs(x) - abs(v)) <= tol or abs(abs(x) - round(abs(v), d)) < 1e-9 for v in nums):
            bad.append(tok)
    ans.unverified = sorted(set(bad))
    ans.verified = not bad
    return ans
