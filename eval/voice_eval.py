"""Voice test: speak real questions with a synthetic voice, transcribe on-device with whisper.cpp,
and check VehicleGPT picks the same analysis as for the typed question.

  python -m eval.voice_eval            (needs espeak-ng + ffmpeg + whisper.cpp + models/ggml-*.en.bin)
Prints one RESULT {json} line for the CI log.
"""
from __future__ import annotations

import json
import re
import resource
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

from ask import speech
from reason.engine import VehicleGPT

QUESTIONS = [
    "How much range do I have left?",
    "Why has my range dropped over the last two weeks?",
    "When did my efficiency get worse?",
    "What happened before the battery temperature warning?",
    "Compare my energy use in the city and on the highway.",
    "Any fault codes?",
    "Is my battery healthy?",
    "Show my charging history.",
    "How many trips did I make this week?",
    "Why did lane assist switch off?",
    "What is the tyre pressure trend?",
    "Unlock the doors.",
]


def words(t: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", t.lower())


def wer(ref: str, hyp: str) -> float:
    r, h = words(ref), words(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev = cur
    return d[len(h)] / max(1, len(r))


def synth(text: str, out: Path):
    raw = out.with_suffix(".raw.wav")
    subprocess.run(["espeak-ng", "-v", "en-us", "-s", "150", "-w", str(raw), text], check=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(raw), "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", str(out)], check=True)


def main():
    if not speech.available():
        print("speech model or whisper.cpp missing"); return 1
    vg = VehicleGPT()
    rows, lat, errs, same = [], [], [], 0
    with tempfile.TemporaryDirectory() as tmp:
        for i, q in enumerate(QUESTIONS):
            wav = Path(tmp) / f"q{i}.wav"
            synth(q, wav)
            r = speech.transcribe(wav.read_bytes())
            typed, spoken = vg.ask(q), vg.ask(r["text"] or "?")
            ok = typed.intent == spoken.intent
            same += ok
            lat.append(r["ms"]); errs.append(wer(q, r["text"]))
            rows.append({"q": q, "heard": r["text"], "ms": r["ms"], "wer": round(errs[-1], 2),
                         "intent": spoken.intent, "same_as_typed": ok, "verified": spoken.verified})
            print(f"{'OK ' if ok else 'XX '} {r['ms']:5d} ms  {q!r} -> {r['text']!r} [{spoken.intent}]")
    peak_child_mb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024
    import os
    res = {"model": speech.model_path().name, "settings": "full 30 s window, beam 5" if os.environ.get("VGPT_WHISPER_FULL") == "1" else "clip-sized window, greedy", "questions": len(QUESTIONS), "same_analysis_as_typed": f"{same}/{len(QUESTIONS)}",
           "mean_word_error_rate": round(statistics.mean(errs), 3), "stt_ms_p50": round(statistics.median(lat)),
           "stt_ms_max": max(lat), "whisper_peak_rss_mb": round(peak_child_mb, 1), "rows": rows}
    print("RESULT " + json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
