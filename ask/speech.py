"""On-device speech-to-text with whisper.cpp (no cloud, no network).

The browser records the driver's voice and posts a 16 kHz mono WAV to /api/listen on 127.0.0.1. We run the
whisper.cpp command-line program on it and return the text. The model is loaded only for the few hundred
milliseconds it takes to transcribe, so it uses no memory while nobody is speaking.

Setup (one time):  python scripts/get_model.py --speech   (downloads ggml-tiny.en.bin, ~75 MB, and on Windows
                   the whisper.cpp program). Afterwards everything runs offline.
"""
from __future__ import annotations

import os
import re
import shutil
import struct
import subprocess
import tempfile
import time
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
BIN_NAMES = ("whisper-cli", "whisper-cli.exe", "main", "main.exe")
MAX_BYTES = 16000 * 2 * 12 + 44          # 12 s of 16 kHz 16-bit mono
# Words the car cares about, given to whisper as context so "SoC", "DTC" or "kWh" are heard correctly.
PROMPT = ("Questions about an electric car: range, battery, state of charge, charging, kWh, tyre pressure, "
          "fault codes, DTC, warning light, efficiency, highway, city, air conditioning, last week, yesterday.")


@lru_cache(maxsize=1)
def model_path() -> Path | None:
    env = os.environ.get("VGPT_WHISPER_MODEL")
    if env and Path(env).exists():
        return Path(env)
    for name in ("ggml-base.en.bin", "ggml-tiny.en.bin", "ggml-base.en-q5_1.bin", "ggml-tiny.en-q5_1.bin"):
        if (MODEL_DIR / name).exists():
            return MODEL_DIR / name
    return None


@lru_cache(maxsize=1)
def binary() -> str | None:
    env = os.environ.get("VGPT_WHISPER_BIN")
    if env and Path(env).exists():
        return env
    for d in (ROOT / "whisper", ROOT / "whisper" / "build" / "bin", ROOT / "whisper" / "Release", ROOT / "whisper.cpp" / "build" / "bin"):
        for n in BIN_NAMES:
            if (d / n).exists():
                return str(d / n)
    for root in (ROOT / "whisper", ROOT / "whisper.cpp"):
        if root.exists():
            for n in ("whisper-cli.exe", "whisper-cli"):
                hit = next((p for p in root.rglob(n) if p.is_file()), None)
                if hit:
                    return str(hit)
    return shutil.which("whisper-cli")


def available() -> bool:
    return model_path() is not None and binary() is not None


def check_wav(data: bytes) -> None:
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a WAV file")
    if len(data) > MAX_BYTES:
        raise ValueError("recording too long")
    ch, rate = struct.unpack("<HI", data[22:28])
    bits = struct.unpack("<H", data[34:36])[0]
    if (ch, rate, bits) != (1, 16000, 16):
        raise ValueError("expected 16 kHz mono 16-bit audio")


def clean(text: str) -> str:
    text = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", text)            # [BLANK_AUDIO], (wind noise), timestamps
    text = re.sub(r"\s+", " ", text).strip()
    return text[:300]


def transcribe(data: bytes, threads: int = 2) -> dict:
    check_wav(data)
    if not available():
        return {"text": "", "error": "speech model not installed"}
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:                   # audio is deleted as soon as it is transcribed
        wav = Path(tmp) / "q.wav"
        wav.write_bytes(data)
        cmd = [binary(), "-m", str(model_path()), "-f", str(wav), "-l", "en", "-t", str(threads),
               "-nt", "-np", "--prompt", PROMPT]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    text = clean(out.stdout)
    return {"text": text, "ms": round((time.perf_counter() - t0) * 1e3), "model": model_path().name}
