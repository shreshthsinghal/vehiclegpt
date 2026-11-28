"""Spoken form -> written form for car questions ("inverse text normalisation").

Speech models write what they hear: "can I do one hundred fifty kilometres at sixty percent". The analyses
read digits and units, so we turn that into "can I do 150 km at 60%". Small, rule-based, no model.
"""
from __future__ import annotations

import re

UNITS = {"zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
         "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
         "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
        "eighty": 80, "ninety": 90}
SCALE = {"hundred": 100, "thousand": 1000}
NUMWORD = set(UNITS) | set(TENS) | set(SCALE) | {"and", "a", "point"}

# after numbers have become digits
UNIT_RULES = [
    (r"(\d)\s*(kilometers?|kilometres?|km)\s*(per|an|a|/)\s*hour\b", r"\1 km/h"),
    (r"(\d)\s*(kmph|k\.?m\.?p\.?h\.?|kph)\b", r"\1 km/h"),
    (r"(\d)\s*miles?\s*(per|an|a)\s*hour\b", r"\1 mph"),
    (r"(\d)\s*(kilometers?|kilometres?|kms|k\.?m\.?s?)\b", r"\1 km"),
    (r"(\d)\s*(percent|per cent|percentage)\b", r"\1%"),
    (r"(\d)\s*(kilowatt[- ]hours?|kilo watt hours?|kwh|k w h)\b", r"\1 kWh"),
    (r"(\d)\s*(kilowatts?|kw)\b", r"\1 kW"),
    (r"(\d)\s*(degrees?( celsius| centigrade| c)?)\b", r"\1 °C"),
    (r"(\d)\s*(volts?|v)\b", r"\1 V"),
    (r"\b(twelve|12)[- ]volt\b", "12 V"),
    (r"\bkm\s*(slash|/)\s*h\b", "km/h"),
    (r"\b(kilowatt[- ]hours?|kilo watt hours?)\b", "kWh"),
    (r"\bstate of charge\b", "SoC"),
    (r"\b(versus|vs\.?)\b", "vs"),
]


def _value(words: list[str]) -> float | None:
    """'one hundred and ten' -> 110, 'twenty five' -> 25, 'seven point five' -> 7.5"""
    if "point" in words:
        i = words.index("point")
        whole = _value(words[:i]) if i else 0
        dec = "".join(str(UNITS.get(w, "")) for w in words[i + 1:])
        return None if whole is None or not dec else float(f"{int(whole)}.{dec}")
    total = cur = 0
    seen = False
    for w in words:
        if w in ("and",):
            continue
        if w == "a":
            cur = 1; continue
        if w in UNITS:
            cur += UNITS[w]; seen = True
        elif w in TENS:
            cur += TENS[w]; seen = True
        elif w == "hundred":
            cur = (cur or 1) * 100; seen = True
        elif w == "thousand":
            total += (cur or 1) * 1000; cur = 0; seen = True
    return total + cur if seen else None


def numbers(text: str) -> str:
    toks = re.findall(r"[A-Za-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?|[^\sA-Za-z\d]+|\s+", text)
    out, i = [], 0
    while i < len(toks):
        t = toks[i].lower()
        if t in UNITS or t in TENS or t in SCALE:
            j, run, ends = i, [], []
            while j < len(toks):
                w = toks[j].lower()
                if w.isspace() or w == "-":
                    j += 1; continue
                if w == "and" and (not run or run[-1] not in ("hundred", "thousand")):
                    break
                if w in NUMWORD and not (w in ("and", "a", "point") and not run):
                    run.append(w); j += 1; ends.append(j); continue
                break
            while run and run[-1] in ("and", "a", "point"):           # "...sixty and" -> keep the "and" as text
                run.pop(); ends.pop()
            j = ends[-1] if ends else i + 1
            # single small words used as words ("one more", "no one", "oh") stay as words
            if len(run) == 1 and run[0] in ("one", "oh", "zero") or (run and _value(run) is None):
                out.append(toks[i]); i += 1; continue
            v = _value(run)
            out.append(str(int(v)) if float(v).is_integer() else str(v))
            i = j
            continue
        out.append(toks[i]); i += 1
    return "".join(out)


def written(text: str) -> str:
    t = numbers(text)
    for pat, rep in UNIT_RULES:
        t = re.sub(pat, rep, t, flags=re.I)
    t = re.sub(r"(\d+)\s*(st|nd|rd|th)\b", r"\1\2", t)
    return re.sub(r"\s+", " ", t).strip()
