"""On-device text-to-speech with a warm, natural voice (no cloud).

Engines (picked automatically, best first):
  kokoro : Kokoro-82M (Apache-2.0) via kokoro-onnx. Natural, warm voices (default "af_heart"). Plus/Pro and laptops.
  piper  : Piper (fast, light) for Lite boards. Clear but less expressive.
  none   : the browser falls back to an on-device system voice.

The text is first rewritten for the ear (units spelled out, numbers rounded) and synthesised one short sentence at a
time, so the first words play after a fraction of a second while the rest is prepared.

Setup (one time): python scripts/get_model.py --voice          (Kokoro, ~350 MB)
                  python scripts/get_model.py --voice piper    (Piper, ~60 MB)
"""
from __future__ import annotations

import io
import os
import re
import threading
import time
import wave
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
VOICE = os.environ.get("VGPT_VOICE", "af_heart")          # warm, friendly
STYLES = {"warm": 0.95, "calm": 0.88, "brisk": 1.05}        # speaking speed per style (emergencies: calm)

_lock = threading.Lock()


# ------------------------------------------------------------------ text for the ear
UNITS = [(r"kWh\s*/\s*100\s*km", "kilowatt-hours per hundred kilometres"), (r"\bkWh\b", "kilowatt-hours"),
         (r"\bkW\b", "kilowatts"), (r"\bkm/h\b", "kilometres per hour"), (r"\bkm\b", "kilometres"), (r"\bkPa\b", "kilopascals"),
         (r"°C", " degrees"), (r"\bmin\b", "minutes"), (r"\bh\b", "hours"), (r"\bmph\b", "miles per hour"),
         (r"(\d)\s*%", r"\1 percent"), (r"\b12 V\b", "twelve-volt"), (r"\bV\b", "volts"), (r"\bAC\b", "A C"),
         (r"\bDC\b", "D C"), (r"\bSoC\b", "state of charge"), (r"\bADAS\b", "driver assistance"), (r"→", " to "),
         (r"\bvs\b", "versus"), (r"\be\.g\.", "for example"), (r"&", " and ")]


def for_ear(text: str) -> str:
    t = text
    t = re.sub(r"(\d+)\.(\d)\d*\s*(?=(km|kilomet|kWh|kW|%|percent| degrees|°C|kPa))",
               lambda m: str(int(float(f"{m.group(1)}.{m.group(2)}") + 0.5)) + " ", t)   # 294.5 km -> 295 km
    t = re.sub(r"\b(\d{2,})\.(\d)\d*\b(?=\s+(and|to|or)\s+\d)",
               lambda m: str(int(float(f"{m.group(1)}.{m.group(2)}") + 0.5)), t)          # "243.1 and 245 kPa": round both ends
    for pat, rep in UNITS:
        t = re.sub(pat, rep, t)
    t = re.sub(r"\b([PCBU])([0-9A-F]{4})\b", lambda m: " ".join(m.group(0)), t)       # P1A20 -> P 1 A 2 0
    t = re.sub(r"\s+", " ", t).strip()
    return t


def sentences(text: str, max_words: int = 14) -> list[str]:
    """Short chunks: sentence ends first, then commas/semicolons for long sentences."""
    out = []
    for s in re.split(r"(?<=[.!?])\s+", text.strip()):
        words = s.split()
        if len(words) <= max_words:
            if s:
                out.append(s)
            continue
        cur = []
        for part in re.split(r"(?<=[,;:])\s+", s):
            if cur and len(" ".join(cur + [part]).split()) > max_words:
                out.append(" ".join(cur)); cur = []
            cur.append(part)
        if cur:
            out.append(" ".join(cur))
    # the first chunk decides how soon the driver hears anything: keep it short (split at the first , : ;)
    if out and len(out[0].split()) > 7:
        head = re.split(r"(?<=[,;:])\s+", out[0], maxsplit=1)
        if len(head) == 2 and len(head[0].split()) >= 2:
            out[:1] = head
    return out


# ------------------------------------------------------------------ engines
@lru_cache(maxsize=1)
def engine():
    """Return (name, synth(text, speed) -> (float32 samples, rate)) for the best available engine."""
    pref = os.environ.get("VGPT_TTS", "auto")
    kdir = MODELS / "kokoro"
    if pref in ("auto", "kokoro") and (kdir / "voices-v1.0.bin").exists():
        model = next((p for p in (kdir / "kokoro-v1.0.onnx", kdir / "kokoro-v1.0.fp16.onnx", kdir / "kokoro-v1.0.int8.onnx")
                      if p.exists()), None)
        if model:
            try:
                from kokoro_onnx import Kokoro
                k = Kokoro(str(model), str(kdir / "voices-v1.0.bin"))
                voice = VOICE if VOICE in k.get_voices() else "af_heart"
                return "kokoro", lambda text, speed: k.create(text, voice=voice, speed=speed, lang="en-us")
            except Exception:                       # noqa: BLE001
                pass
    pdir = MODELS / "piper"
    onnx = next(iter(sorted(pdir.glob("*.onnx"))), None) if pdir.exists() else None
    if pref in ("auto", "piper") and onnx:
        try:
            from piper import PiperVoice, SynthesisConfig
            pv = PiperVoice.load(str(onnx))

            def synth(text, speed):
                chunks = list(pv.synthesize(text, syn_config=SynthesisConfig(length_scale=1.0 / speed)))
                audio = (np.concatenate([c.audio_int16_array for c in chunks]).astype(np.float32) / 32768.0
                         if chunks else np.zeros(1, np.float32))
                return audio, pv.config.sample_rate
            return "piper", synth
        except Exception:                           # noqa: BLE001
            pass
    return "none", None


def available() -> str:
    return engine()[0]


def to_wav(samples, rate: int) -> bytes:
    pcm = (np.clip(np.asarray(samples, dtype=np.float32), -1, 1) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(int(rate)); w.writeframes(pcm.tobytes())
    return buf.getvalue()


@lru_cache(maxsize=64)
def speak(text: str, style: str = "warm") -> tuple[bytes, dict]:
    """One chunk of speech as WAV (cached: follow-up chips and repeated phrases play instantly)."""
    name, synth = engine()
    if synth is None:
        raise RuntimeError("no on-device voice installed")
    t0 = time.perf_counter()
    with _lock:                                       # one synthesis at a time keeps CPU use bounded
        audio, rate = synth(for_ear(text)[:400], STYLES.get(style, 0.95))
    ms = (time.perf_counter() - t0) * 1000
    secs = len(audio) / rate
    return to_wav(audio, rate), {"engine": name, "ms": round(ms), "audio_s": round(secs, 2), "rtf": round(ms / 1000 / max(secs, 1e-3), 2)}
