"""Turn time phrases into exact windows. Done by CODE, never by the language model (small models are bad at
calendar arithmetic). "now" is the vehicle clock; for recorded data it is the time of the last stored sample.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
NUM = {"a": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "ten": 10,
       "couple of": 2, "few": 3}


@dataclass
class Window:
    start: float
    end: float
    label: str
    explicit: bool = True            # False = default chosen by us, not said by the user

    def days(self) -> float:
        return (self.end - self.start) / 86400

    def previous(self) -> "Window":
        span = self.end - self.start
        return Window(self.start - span, self.start, f"the {self.label.replace('the ', '')} before", self.explicit)

    def fmt(self) -> str:
        a = dt.datetime.fromtimestamp(self.start, IST)
        b = dt.datetime.fromtimestamp(self.end - 1, IST)
        return f"{a:%d %b}" if a.date() == b.date() else f"{a:%d %b} - {b:%d %b}"


def _midnight(t: float) -> dt.datetime:
    d = dt.datetime.fromtimestamp(t, IST)
    return d.replace(hour=0, minute=0, second=0, microsecond=0)


def _day_window(d: dt.datetime, label: str) -> Window:
    return Window(d.timestamp(), (d + dt.timedelta(days=1)).timestamp(), label)


def _last_days(today: dt.datetime, now: float, days: int, label: str, explicit=True) -> Window:
    """Calendar-aligned: 'last 14 days' = 14 whole days ending today (midnight start)."""
    return Window((today - dt.timedelta(days=days - 1)).timestamp(), now + 1, label, explicit)


def resolve(text: str, now: float, default_days: float | None = None) -> Window | None:
    """Return the time window a question refers to, or a default window (explicit=False), or None."""
    q = " " + text.lower().replace("’", "'") + " "
    today = _midnight(now)
    end_now = now + 1

    if re.search(r"\byesterday\b|\blast (night|evening)\b", q):
        return _day_window(today - dt.timedelta(days=1), "yesterday")
    if re.search(r"\btoday\b|\bthis morning\b", q):
        return _day_window(today, "today")
    m = re.search(r"\b(?:on |for )?(?:the )?(\d{1,2})(?:st|nd|rd|th)?\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b", q) or \
        re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(\d{1,2})(?:st|nd|rd|th)?\b", q)
    if m:
        a, b = m.groups()
        day, mon = (int(a), MONTHS[b]) if a.isdigit() else (int(b), MONTHS[a])
        d = today.replace(month=mon, day=day)
        if d > today:
            d = d.replace(year=d.year - 1)
        return _day_window(d, f"{d:%d %b}")
    for i, wd in enumerate(WEEKDAYS):
        if re.search(rf"\b(last |on |this past )?{wd}\b", q):
            back = (today.weekday() - i) % 7 or 7
            d = today - dt.timedelta(days=back)
            return _day_window(d, f"last {wd.title()} ({d:%d %b})")
    m = re.search(r"\b(?:last|past|previous|over the last|over the past|in the last)\s+(\d+|a|one|two|three|four|five|six|seven|eight|ten|couple of|few)\s+(day|week|month)s?\b", q)
    if m:
        n = int(m.group(1)) if m.group(1).isdigit() else NUM[m.group(1)]
        days = n * {"day": 1, "week": 7, "month": 30}[m.group(2)]
        return _last_days(today, now, days, f"the last {n} {m.group(2)}{'s' if n > 1 else ''}")
    if re.search(r"\b(fortnight|two weeks|2 weeks)\b", q):
        return _last_days(today, now, 14, "the last 2 weeks")
    if re.search(r"\blast week\b|\bpast week\b|\bthis week\b", q):
        return _last_days(today, now, 7, "the last 7 days")
    if re.search(r"\bthis month\b|\bin (november|october)\b", q):
        mon = "november" if "november" in q else ("october" if "october" in q else None)
        if mon == "october":
            s = today.replace(month=10, day=1)
            return Window(s.timestamp(), today.replace(month=11, day=1).timestamp(), "October")
        s = today.replace(day=1)
        return Window(s.timestamp(), end_now, f"{today:%B} so far")
    if re.search(r"\blast month\b", q):
        return _last_days(today, now, 30, "the last 30 days")
    if re.search(r"\bfirst (two|2) weeks\b", q):
        return None                       # handled by the planner ("first two weeks" of the history)
    if re.search(r"\b(lately|recently|these days|nowadays|than before|used to|anymore)\b", q):
        return _last_days(today, now, 14, "the last 2 weeks", explicit=False)
    if default_days:
        return _last_days(today, now, int(default_days), f"the last {int(default_days)} days", explicit=False)
    return None


def phrase(w: "Window") -> str:
    """'yesterday (28 Nov)', 'on 21 Nov', 'in the last 7 days (23 Nov - 29 Nov)'."""
    if w.label in ("yesterday", "today"):
        return f"{w.label} ({w.fmt()})"
    if w.days() <= 1.01:
        return f"on {w.fmt()}"
    return f"in {w.label}" + (f" ({w.fmt()})" if w.explicit and w.fmt() not in w.label else "")


def fmt_time(t: float, with_date: bool = True) -> str:
    d = dt.datetime.fromtimestamp(t, IST)
    return f"{d:%d %b %H:%M}" if with_date else f"{d:%H:%M:%S}"


def fmt_date(t: float) -> str:
    return f"{dt.datetime.fromtimestamp(t, IST):%a %d %b}"
