"""Tiny, dependency-free text embedder used by the intent router and the meaning search.

Feature hashing of word unigrams/bigrams + character 3-5-grams, after a small automotive synonym lexicon,
projected into a fixed 512-dim vector and L2-normalised. ~0.1 ms per sentence, ~0 MB model.
A neural encoder (e.g. MiniLM int8 ONNX) can replace it through the same `embed()` interface (see Encoder).
"""
from __future__ import annotations

import hashlib
import re

import numpy as np

DIM = 512
STOP = set("the a an my is are was were did do does of for to in on at and or it its this that me i you your be "
           "been has have had with by from show tell give please can could would".split())   # why/when/how kept: they carry intent

# Domain lexicon: many ways people say the same thing -> one canonical token (added, originals kept).
SYN = {
    r"\b(range|kilometres?|kms?|distance|far|mileage left|kilometers?)\b": " range ",
    r"\b(efficien\w*|consumption|kwh|energy use|energy per|economy|mileage)\b": " efficiency ",
    r"\b(fuel)\b": " efficiency ",
    r"\b(adas|lane ?keep\w*|lane assist|lka|driver assist\w*|autopilot|assistance|self.?driving|cruise)\b": " adas ",
    r"\b(disengag\w*|handover|takeover|cut(s|ting)? out|drop(ped)? out|switch(ed)? off|turn(ed)? (itself )?off|stopp?ed working|interrupt\w*)\b": " disengage ",
    r"\b(overheat\w*|too hot|thermal|hot|heat)\b": " hot ",
    r"\b(warning|alert|telltale|message|light)\b": " warning ",
    r"\b(before|led to|lead(ing)? up|leading to|caused?|triggered|root cause|sequence|timeline|preceded)\b": " before ",
    r"\b(city|town|urban|traffic)\b": " city ",
    r"\b(cool\w*|chill\w*|radiator)\b": " coolant cooling ",
    r"\b(highway|motorway|expressway|freeway)\b": " highway ",
    r"\b(compare|comparison|versus|vs|difference|between)\b": " compare ",
    r"\b(dtcs?|fault codes?|error codes?|trouble codes?|diagnostic\w*|faults?)\b": " dtc ",
    r"\b(charg\w*)\b": " charge ",
    r"\b(trip|drive|journey|commute)s?\b": " trip ",
    r"\b(missing|gap|gaps|dropp?ed data|incomplete|lost data)\b": " gap ",
    r"\b(dropp?ed|lower|reduc\w*|decreas\w*|worse|less|fewer|fall|fell|loss|down)\b": " down ",
    r"\b(when|date|day)\b": " when ",
    r"\b(why|reason|explain|cause|caus\w+|hurt|eating|changed)\b": " why ",
    r"\b(now|current\w*|right now|at the moment|left|remaining)\b": " now ",
    r"\b(power (limit|reduc|cut|derat)\w*|reduced power|limp|derat\w*)\b": " derate warning ",
    r"\b(largest|biggest|most|worst|sudden\w*|sharp\w*)\b": " most ",
}


def normalise(text: str) -> str:
    t = " " + text.lower().replace("’", "'") + " "
    extra = []
    for pat, canon in SYN.items():
        if re.search(pat, t):
            extra.append(canon.strip())
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    return t + " " + " ".join(extra)


def _h(s: str) -> tuple[int, float]:
    d = hashlib.blake2b(s.encode(), digest_size=8).digest()
    v = int.from_bytes(d, "little")
    return v % DIM, (1.0 if (v >> 63) & 1 else -1.0)


def features(text: str) -> dict[str, float]:
    t = normalise(text)
    words = [w for w in t.split() if w not in STOP]
    f: dict[str, float] = {}
    for w in words:
        f["w:" + w] = f.get("w:" + w, 0) + 1.0
        ww = f"<{w}>"
        for n in (3, 4, 5):
            for i in range(max(1, len(ww) - n + 1)):
                g = "c:" + ww[i:i + n]
                f[g] = f.get(g, 0) + 0.35
    for a, b in zip(words, words[1:]):
        f[f"b:{a}_{b}"] = f.get(f"b:{a}_{b}", 0) + 0.7
    return f


def embed(text: str) -> np.ndarray:
    v = np.zeros(DIM, np.float32)
    for k, w in features(text).items():
        i, s = _h(k)
        v[i] += s * w
    n = np.linalg.norm(v)
    return v / n if n else v


def embed_many(texts) -> np.ndarray:
    return np.stack([embed(t) for t in texts]) if texts else np.zeros((0, DIM), np.float32)
