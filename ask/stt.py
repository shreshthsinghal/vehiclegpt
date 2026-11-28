"""On-device speech-to-text for a noisy car cabin (sherpa-onnx; no cloud, no network).

Pipeline for each spoken question (16 kHz mono):
  1. noise suppression (optional, GTCRN, 0.5 MB): cleans steady road/fan noise
  2. voice activity detection (Silero VAD, 0.6 MB): keeps only the parts where someone speaks, so horns,
     silence and engine noise before/after the question are never transcribed (no invented words)
  3. speech recognition: the selected offline model (see ENGINES)
Everything is loaded from models/stt/ and runs on the CPU. Setup once: python scripts/get_model.py --speech
"""
from __future__ import annotations

import io
import os
import re
import struct
import threading
import time
import wave
from functools import lru_cache
from pathlib import Path

import numpy as np

from ask.itn import written

ROOT = Path(__file__).resolve().parents[1]
SR = 16000
MAX_BYTES = SR * 2 * 12 + 44            # 12 s of 16 kHz 16-bit mono

# name -> (release asset, folder). All English models; licences: Parakeet CC-BY-4.0 (NVIDIA), Moonshine MIT,
# Whisper MIT. Front-end models: Silero VAD MIT, GTCRN MIT.
ENGINES = {
    "parakeet-110m": "sherpa-onnx-nemo-parakeet_tdt_transducer_110m-en-36000-int8",
    "parakeet-0.6b": "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8",
    "moonshine-base": "sherpa-onnx-moonshine-base-en-quantized-2026-02-27",
    "moonshine-tiny": "sherpa-onnx-moonshine-tiny-en-int8",
    "whisper-base": "sherpa-onnx-whisper-base.en",
}
DEFAULT = "parakeet-110m"
_lock = threading.Lock()


def model_dir() -> Path:
    return Path(os.environ.get("VGPT_STT_DIR", ROOT / "models" / "stt"))


def engine_name() -> str | None:
    want = os.environ.get("VGPT_STT_ENGINE")
    names = [want] if want else [DEFAULT] + [n for n in ENGINES if n != DEFAULT]
    for n in names:
        if n in ENGINES and (model_dir() / ENGINES[n] / "tokens.txt").exists() or \
                n == "whisper-base" and (model_dir() / ENGINES[n] / "base.en-tokens.txt").exists():
            return n
    return None


def available() -> bool:
    try:
        import sherpa_onnx  # noqa: F401
    except ImportError:
        return False
    return engine_name() is not None and (model_dir() / "silero_vad.onnx").exists()


@lru_cache(maxsize=1)
def recognizer():
    import sherpa_onnx as so
    name = engine_name()
    if not name:
        raise RuntimeError("no speech model installed (python scripts/get_model.py --speech)")
    d = model_dir() / ENGINES[name]
    t = int(os.environ.get("VGPT_STT_THREADS", "2"))
    if name.startswith("parakeet"):
        r = so.OfflineRecognizer.from_transducer(str(d / "encoder.int8.onnx"), str(d / "decoder.int8.onnx"),
                                                 str(d / "joiner.int8.onnx"), str(d / "tokens.txt"),
                                                 num_threads=t, model_type="nemo_transducer")
    elif name == "moonshine-base":
        r = so.OfflineRecognizer.from_moonshine_v2(str(d / "encoder_model.ort"), str(d / "decoder_model_merged.ort"),
                                                   str(d / "tokens.txt"), num_threads=t)
    elif name == "moonshine-tiny":
        r = so.OfflineRecognizer.from_moonshine(str(d / "preprocess.onnx"), str(d / "encode.int8.onnx"),
                                                str(d / "uncached_decode.int8.onnx"), str(d / "cached_decode.int8.onnx"),
                                                str(d / "tokens.txt"), num_threads=t)
    else:
        r = so.OfflineRecognizer.from_whisper(str(d / "base.en-encoder.int8.onnx"), str(d / "base.en-decoder.int8.onnx"),
                                              str(d / "base.en-tokens.txt"), num_threads=t)
    return name, r


@lru_cache(maxsize=1)
def denoiser():
    front = os.environ.get("VGPT_STT_FRONT", "gtcrn")
    p = model_dir() / "gtcrn_simple.onnx"
    if front != "gtcrn" or not p.exists():
        return None
    import sherpa_onnx as so
    cfg = so.OfflineSpeechDenoiserConfig(model=so.OfflineSpeechDenoiserModelConfig(
        gtcrn=so.OfflineSpeechDenoiserGtcrnModelConfig(model=str(p)), num_threads=1))
    return so.OfflineSpeechDenoiser(cfg)


def voiced(x: np.ndarray) -> np.ndarray:
    """Keep only speech (Silero VAD), with a little padding. Returns empty if nobody spoke."""
    import sherpa_onnx as so
    cfg = so.VadModelConfig()
    cfg.silero_vad.model = str(model_dir() / "silero_vad.onnx")
    cfg.silero_vad.threshold = float(os.environ.get("VGPT_VAD_THRESHOLD", "0.4"))
    cfg.silero_vad.min_silence_duration = 0.5
    cfg.silero_vad.min_speech_duration = 0.2
    cfg.sample_rate = SR
    vad = so.VoiceActivityDetector(cfg, buffer_size_in_seconds=15)
    w = cfg.silero_vad.window_size
    for i in range(0, len(x), w):
        vad.accept_waveform(x[i:i + w])
    vad.flush()
    pad = int(0.15 * SR)
    segs = []
    while not vad.empty():
        s = vad.front.start; seg = np.asarray(vad.front.samples, np.float32); vad.pop()
        a, b = max(0, s - pad), min(len(x), s + len(seg) + pad)
        segs.append(x[a:b])
    return np.concatenate(segs) if segs else np.zeros(0, np.float32)


def clean(text: str) -> str:
    text = re.sub(r"\[[^\]]*\]|\([^)]*\)|<[^>]*>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text and text == text.upper() and any(c.isalpha() for c in text):
        text = text.capitalize()
    return text[:300]


def transcribe_samples(x: np.ndarray, threads: int = 2) -> dict:
    t0 = time.perf_counter()
    x = np.asarray(x, np.float32)
    peak = float(np.abs(x).max()) if len(x) else 0.0
    if peak > 0:
        x = x / peak * 0.9                                            # level the input (car mics vary a lot)
    with _lock:
        dn = denoiser()
        if dn is not None and len(x):
            x = np.asarray(dn.run(x, SR).samples, np.float32)
        speech = voiced(x)
        if len(speech) < int(0.25 * SR):
            return {"text": "", "ms": round((time.perf_counter() - t0) * 1e3), "model": engine_name(), "heard": False}
        name, rec = recognizer()
        s = rec.create_stream()
        s.accept_waveform(SR, np.concatenate([np.zeros(int(0.1 * SR), np.float32), speech, np.zeros(int(0.3 * SR), np.float32)]))
        rec.decode_stream(s)
        raw = clean(s.result.text)
        text = written(raw)
    return {"text": text, "raw": raw, "ms": round((time.perf_counter() - t0) * 1e3), "model": name, "heard": True,
            "speech_s": round(len(speech) / SR, 2)}


def check_wav(data: bytes) -> None:
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a WAV file")
    if len(data) > MAX_BYTES:
        raise ValueError("recording too long")
    ch, rate = struct.unpack("<HI", data[22:28])
    bits = struct.unpack("<H", data[34:36])[0]
    if (ch, rate, bits) != (1, SR, 16):
        raise ValueError("expected 16 kHz mono 16-bit audio")


def transcribe(data: bytes) -> dict:
    """WAV bytes from the head unit -> text. The audio lives only in memory and is never stored."""
    check_wav(data)
    if not available():
        return {"text": "", "error": "speech model not installed"}
    with wave.open(io.BytesIO(data)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    return transcribe_samples(x)


# ------------------------------------------------------------------ live listening (end-of-speech by Silero VAD)
class Session:
    """One spoken question, streamed from the head unit in small chunks.

    A loudness threshold can't tell a driver from a horn or road noise, so the end of the question is decided by
    the neural voice detector: we stop after `END_SILENCE` seconds without speech once the driver has spoken.
    The audio exists only in memory and is dropped after transcription.
    """
    END_SILENCE = 0.8
    MAX_S = 10.0

    def __init__(self):
        import sherpa_onnx as so
        cfg = so.VadModelConfig()
        cfg.silero_vad.model = str(model_dir() / "silero_vad.onnx")
        cfg.silero_vad.threshold = float(os.environ.get("VGPT_VAD_THRESHOLD", "0.4"))
        cfg.silero_vad.min_silence_duration = 0.3
        cfg.silero_vad.min_speech_duration = 0.15
        cfg.sample_rate = SR
        self.vad = so.VoiceActivityDetector(cfg, buffer_size_in_seconds=20)
        self.win = cfg.silero_vad.window_size
        self.audio: list[np.ndarray] = []
        self.pending = np.zeros(0, np.float32)
        self.n = 0
        self.speech_s = 0.0
        self.last_speech = None
        self.created = time.time()

    def feed(self, pcm16: bytes) -> dict:
        x = np.frombuffer(pcm16[: len(pcm16) // 2 * 2], np.int16).astype(np.float32) / 32768
        self.audio.append(x)
        buf = np.concatenate([self.pending, x])
        k = len(buf) // self.win * self.win
        for i in range(0, k, self.win):
            self.vad.accept_waveform(buf[i:i + self.win])
            self.n += self.win
            if self.vad.is_speech_detected():
                self.speech_s += self.win / SR
                self.last_speech = self.n / SR
        self.pending = buf[k:]
        while not self.vad.empty():
            self.vad.pop()
        t = self.n / SR
        silence = t - self.last_speech if self.last_speech is not None else t
        ended = (self.speech_s >= 0.3 and silence >= self.END_SILENCE) or t >= self.MAX_S
        return {"t": round(t, 2), "speech_s": round(self.speech_s, 2), "silence_s": round(silence, 2), "ended": ended}

    def finish(self) -> dict:
        if self.speech_s < 0.2:
            return {"text": "", "heard": False, "ms": 0, "model": engine_name()}
        return transcribe_samples(np.concatenate(self.audio) if self.audio else np.zeros(0, np.float32))


_sessions: dict[str, Session] = {}


def session_start() -> str:
    import secrets
    for k in [k for k, s in _sessions.items() if time.time() - s.created > 30]:
        _sessions.pop(k, None)                                      # forget abandoned recordings
    sid = secrets.token_hex(8)
    _sessions[sid] = Session()
    return sid


def session_feed(sid: str, pcm16: bytes) -> dict:
    s = _sessions.get(sid)
    if s is None:
        raise KeyError("unknown or expired listening session")
    return s.feed(pcm16)


def session_end(sid: str, cancel: bool = False) -> dict:
    s = _sessions.pop(sid, None)
    if s is None or cancel:
        return {"text": "", "heard": False}
    return s.finish()
