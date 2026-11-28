"""Step 1 of the noisy-cabin speech test: speak the driver questions in several voices/accents.

Writes 16 kHz mono WAVs + a manifest to data/stt_eval/clean/. Uses the on-device Kokoro voice (no cloud).
Voices: US female/male, British female/male, and two Indian-English voices; speaking speed varies.
  python -m eval.stt_speech [--n 60]
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "stt_eval" / "clean"
VOICES = [("af_heart", "en-us"), ("am_michael", "en-us"), ("bf_emma", "en-gb"), ("bm_george", "en-gb"),
          ("hf_alpha", "en-us"), ("hm_omega", "en-us")]


def questions(n: int) -> list[str]:
    qs = [x["q"] for x in yaml.safe_load(open(ROOT / "eval/capability_questions.yaml"))["questions"]]
    qs += [x["q"] for x in yaml.safe_load(open(ROOT / "eval/blind_questions.yaml"))["questions"]
           if x["q"].isascii() and not any(w in x["q"].lower() for w in (" hai", " kya", " kitna", " mera", "kaisa"))]
    seen, out = set(), []
    for q in qs:
        if q.lower() not in seen and len(q.split()) <= 22:
            seen.add(q.lower()); out.append(q)
    return out[:n]


def resample(x: np.ndarray, sr: int, to: int = 16000) -> np.ndarray:
    t = np.arange(0, len(x) * to / sr) * sr / to
    return np.interp(t, np.arange(len(x)), x).astype(np.float32)


def main(n: int = 60):
    from kokoro_onnx import Kokoro
    from ask.tts import for_ear                                # say "kilometres per hour", not "km slash h"
    k = Kokoro(str(ROOT / "models/kokoro/kokoro-v1.0.onnx"), str(ROOT / "models/kokoro/voices-v1.0.bin"))
    OUT.mkdir(parents=True, exist_ok=True)
    rnd = random.Random(7)
    man = []
    for i, q in enumerate(questions(n)):
        for j in range(2):                                     # two different speakers per question
            voice, lang = VOICES[(2 * i + j) % len(VOICES)]
            speed = round(rnd.uniform(0.9, 1.15), 2)
            f = OUT / f"q{i:03d}_{voice}.wav"
            if not f.exists():
                x, sr = k.create(for_ear(q), voice=voice, speed=speed, lang=lang)
                x = resample(x, sr)
                x = np.concatenate([np.zeros(int(0.4 * 16000)), x, np.zeros(int(0.5 * 16000))]).astype(np.float32)
                sf.write(f, x / max(1e-6, np.abs(x).max()) * 0.5, 16000, subtype="PCM_16")
            man.append({"file": f.name, "text": q, "voice": voice, "speed": speed})
        print(f"\r{i + 1} questions", end="", flush=True)
    json.dump(man, open(OUT / "manifest.json", "w"), indent=1)
    print(f"\n{len(man)} clips in {OUT}")


if __name__ == "__main__":
    main(int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 60)
