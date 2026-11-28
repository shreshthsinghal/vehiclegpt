"""Noisy-cabin speech-to-text benchmark: which offline model hears a driver best in traffic?

  python -m eval.stt_speech                       # 1. speak the questions (data/stt_eval/clean)
  python -m eval.stt_bench mix --noise DIR        # 2. mix with cabin/traffic noise (data/stt_eval/<condition>)
  python -m eval.stt_bench run ENGINE [--front none|gtcrn] [--dir models/stt]   # 3. score one engine
  python -m eval.stt_bench table                  # 4. summary table

Conditions (SNR = speech power vs noise power, in dB; lower is louder noise):
  clean                 no noise
  cabin_10/5/0          steady road + tyre rumble and AC fan, like a closed car at speed
  traffic_5/0           cabin noise plus horns, sirens, engines and trains from the ESC-50 recordings
  rain_wind_5           cabin noise plus rain and wind recordings (ESC-50)
  radio_5               cabin noise plus someone else talking (radio / passenger), the hardest case
  car_5/0, street_5     REAL recordings from the DEMAND corpus (inside a moving car; a busy street), if
                        downloaded (CI does: --demand DIR with TCAR and STRAFFIC folders, CC BY-SA 3.0)
Scores: word error rate (WER) and "same answer": does the transcript route to the same VehicleGPT analysis as
the exact question text? That second number is what the driver feels.
Noise clips (ESC-50, CC BY-NC 3.0) are used only for this evaluation and are not shipped.
"""
from __future__ import annotations

import json
import os
import random
import re
import resource
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "data" / "stt_eval"
SR = 16000
CONDITIONS = {"clean": None, "cabin_10": ("cabin", 10), "cabin_5": ("cabin", 5), "cabin_0": ("cabin", 0),
              "traffic_5": ("traffic", 5), "traffic_0": ("traffic", 0), "rain_wind_5": ("rain_wind", 5),
              "radio_5": ("radio", 5)}
ESC = {"traffic": ("car_horn", "engine", "siren", "train"), "rain_wind": ("rain", "wind")}


# ------------------------------------------------------------------ noise
def colored(n: int, rnd: np.random.Generator, power: float) -> np.ndarray:
    """1/f^power noise (power 1 = pink, 2 = brown)."""
    f = np.fft.rfftfreq(n, 1 / SR)
    spec = (rnd.standard_normal(len(f)) + 1j * rnd.standard_normal(len(f))) / np.maximum(f, 20) ** (power / 2)
    x = np.fft.irfft(spec, n)
    return (x / (np.abs(x).max() + 1e-9)).astype(np.float32)


def cabin(n: int, rnd) -> np.ndarray:
    road = colored(n, rnd, 2.0)                                   # tyre/road rumble, mostly below 300 Hz
    fan = colored(n, rnd, 1.0) * 0.35                             # AC blower, broadband
    hum = 0.08 * np.sin(2 * np.pi * rnd.uniform(90, 140) * np.arange(n) / SR)   # motor/inverter whine
    return (road + fan + hum).astype(np.float32)


def load_esc(noise_dir: Path) -> dict[str, list[np.ndarray]]:
    import csv
    want = {c for v in ESC.values() for c in v}
    out = {c: [] for c in want}
    for r in csv.DictReader(open(noise_dir / "meta" / "esc50.csv")):
        if r["category"] in want and (noise_dir / "audio" / r["filename"]).exists():
            x, sr = sf.read(noise_dir / "audio" / r["filename"], dtype="float32")
            x = np.interp(np.arange(0, len(x) * SR / sr) * sr / SR, np.arange(len(x)), x).astype(np.float32)
            if np.abs(x).max() > 1e-3:
                out[r["category"]].append(x)
    return out


def power(x: np.ndarray, active: bool = False) -> float:
    if active:                                                    # speech power over the voiced part only
        e = np.convolve(x ** 2, np.ones(320) / 320, "same")
        x = x[e > 0.05 * e.max()]
    return float(np.mean(x ** 2) + 1e-12)


def events(kind: str, n: int, esc, rnd) -> np.ndarray:
    """A few loud traffic events dropped at random times (they overlap the speech on purpose)."""
    y = np.zeros(n, np.float32)
    for _ in range(rnd.integers(2, 4)):
        cat = rnd.choice(ESC[kind])
        clip = esc[cat][rnd.integers(len(esc[cat]))]
        start = int(rnd.integers(0, max(1, n - SR)))
        seg = clip[: n - start]
        y[start:start + len(seg)] += seg / (np.abs(seg).max() + 1e-9) * rnd.uniform(0.6, 1.0)
    return y


def load_demand(d: Path | None) -> dict[str, np.ndarray]:
    out = {}
    if not d:
        return out
    for env in ("TCAR", "STRAFFIC"):
        chans = sorted((d / env).glob("ch*.wav"))
        if chans:
            x, sr = sf.read(chans[0], dtype="float32")
            out[env] = x if sr == SR else np.interp(np.arange(0, len(x) * SR / sr) * sr / SR, np.arange(len(x)), x).astype(np.float32)
    return out


def conditions() -> list[str]:
    extra = [c for c in ("car_5", "car_0", "street_5") if (EVAL / c / "manifest.ok").exists()]
    return list(CONDITIONS) + extra


def mix(noise_dir: str, demand_dir: str | None = None):
    clean = EVAL / "clean"
    man = json.load(open(clean / "manifest.json"))
    esc = load_esc(Path(noise_dir))
    print({k: len(v) for k, v in esc.items()})
    talk = [sf.read(clean / m["file"], dtype="float32")[0] for m in man]
    for cond, spec in CONDITIONS.items():
        d = EVAL / cond
        d.mkdir(parents=True, exist_ok=True)
        rnd = np.random.default_rng(abs(hash(cond)) % 2**32)
        for i, m in enumerate(man):
            s, _ = sf.read(clean / m["file"], dtype="float32")
            if spec is None:
                y = s
            else:
                kind, snr = spec
                n = len(s)
                noise = cabin(n, rnd)
                if kind in ESC:                                   # steady cabin noise + loud events, equal power
                    ev = events(kind, n, esc, rnd)
                    noise = noise / np.sqrt(power(noise)) + ev / np.sqrt(power(ev))
                if kind == "radio":                               # another voice, different question
                    other = talk[(i + 37) % len(talk)]
                    other = np.resize(other, n)
                    noise = noise / np.sqrt(power(noise)) * 0.5 + other / np.sqrt(power(other, True))
                noise = noise * np.sqrt(power(s, True) / power(noise) / 10 ** (snr / 10))
                y = s + noise
            y = y / max(1.0, np.abs(y).max() / 0.95)              # no clipping
            sf.write(d / m["file"], y, SR, subtype="PCM_16")
        print(f"{cond}: {len(man)} clips")
    demand = load_demand(Path(demand_dir) if demand_dir else None)
    for cond, (env, snr) in {"car_5": ("TCAR", 5), "car_0": ("TCAR", 0), "street_5": ("STRAFFIC", 5)}.items():
        if env not in demand:
            continue
        d = EVAL / cond; d.mkdir(parents=True, exist_ok=True)
        rnd = np.random.default_rng(len(cond) * 7919)
        rec = demand[env]
        for m in man:
            s, _ = sf.read(clean / m["file"], dtype="float32")
            start = int(rnd.integers(0, len(rec) - len(s)))
            noise = rec[start:start + len(s)].copy()
            noise = noise * np.sqrt(power(s, True) / power(noise) / 10 ** (snr / 10))
            y = s + noise
            sf.write(d / m["file"], y / max(1.0, np.abs(y).max() / 0.95), SR, subtype="PCM_16")
        (d / "manifest.ok").write_text("ok")
        print(f"{cond}: {len(man)} clips (real {env} recording)")


# ------------------------------------------------------------------ scoring
def norm(t: str) -> list[str]:
    t = t.lower().replace("’", "'")
    t = re.sub(r"\btires?\b", lambda m: m.group(0).replace("ire", "yre"), t)     # US/UK spelling is not an error
    t = re.sub(r"[^a-z0-9%' ]+", " ", t)
    return t.split()


def wer(ref: str, hyp: str) -> tuple[int, int]:
    r, h = norm(ref), norm(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)], len(r)


def run(engine: str, front: str, model_dir: str | None, threads: int = 2):
    if model_dir:
        os.environ["VGPT_STT_DIR"] = model_dir
    os.environ["VGPT_STT_ENGINE"] = engine
    os.environ["VGPT_STT_FRONT"] = front
    from ask import stt
    from reason.router import Router
    rt = Router(None)
    t = time.perf_counter(); stt.recognizer(); load = time.perf_counter() - t
    man = json.load(open(EVAL / "clean" / "manifest.json"))
    res = {"engine": engine, "front": front, "load_s": round(load, 2), "conditions": {}}
    for cond in conditions():
        errs = words = same = 0; ms = []; bad = []
        for m in man:
            x, _ = sf.read(EVAL / cond / m["file"], dtype="float32")
            r = stt.transcribe_samples(x, threads=threads)
            e, w = wer(m["text"], r["text"]); errs += e; words += w; ms.append(r["ms"])
            ok = rt.route(r["text"]).intent == rt.route(m["text"]).intent if r["text"] else False
            same += ok
            if not ok and len(bad) < 4:
                bad.append([m["text"], r["text"]])
        res["conditions"][cond] = {"wer": round(100 * errs / words, 1), "same_answer": f"{same}/{len(man)}",
                                   "same_pct": round(100 * same / len(man), 1),
                                   "ms_p50": int(np.median(ms)), "ms_p95": int(np.percentile(ms, 95)), "misses": bad}
        c = res["conditions"][cond]
        print(f"{engine:>16} {front:>6} {cond:>12}  WER {c['wer']:5.1f}%  same answer {c['same_answer']:>7}  "
              f"p50 {c['ms_p50']} ms", flush=True)
    res["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    out = EVAL / "results"; out.mkdir(exist_ok=True)
    json.dump(res, open(out / f"{engine}_{front}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "conditions"}))


def table():
    rows = [json.load(open(f)) for f in sorted((EVAL / "results").glob("*.json"))]
    order = list(CONDITIONS) + ["car_5", "car_0", "street_5"]
    conds = [c for c in order if rows and all(c in r["conditions"] for r in rows)]
    print("| engine | front end | " + " | ".join(conds) + " | p50 ms (clean) | RAM MB |")
    print("|---|---|" + "---|" * len(conds) + "---|---|")
    for r in rows:
        c = r["conditions"]
        print(f"| {r['engine']} | {r['front']} | " + " | ".join(f"{c[k]['same_pct']:.0f}% / {c[k]['wer']:.0f}" for k in conds)
              + f" | {c['clean']['ms_p50']} | {r['peak_rss_mb']} |")
    print("\ncells: same answer as the typed question / word error rate %")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "mix":
        mix(a[a.index("--noise") + 1], a[a.index("--demand") + 1] if "--demand" in a else None)
    elif a[0] == "run":
        run(a[1], a[a.index("--front") + 1] if "--front" in a else "none", a[a.index("--dir") + 1] if "--dir" in a else None)
    elif a[0] == "table":
        table()
