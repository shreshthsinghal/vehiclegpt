"""Warm voice benchmark: how soon does the driver hear the first words, and how heavy is the voice?

Takes the real spoken lines the head unit produces for a set of questions, splits them the way the screen
does (short first chunk, then sentences) and times the first chunk, the real-time factor and peak memory.
  python -m eval.tts_eval            (run pinned to 2 cores in CI: taskset -c 0,1)
"""
import json
import os
import resource
import statistics
import time

from ask import present, tts
from reason.engine import VehicleGPT

QUESTIONS = ["What's my tyre pressure right now?", "How is battery health right now?", "Why has my range dropped?",
             "Can I reach 180 km if I leave at 60%?", "my brakes failed what do I do", "How can I improve my range?",
             "Any fault codes?", "Show my safety log", "unlock the doors", "Is the rear camera fine?"]


def main():
    name = tts.available()
    if name == "none":
        print("no voice model installed (python scripts/get_model.py --voice)")
        return 1
    t = time.perf_counter(); tts.speak("Hello.", "warm"); load = time.perf_counter() - t
    vg = VehicleGPT()
    first, rtf, rows = [], [], []
    for q in QUESTIONS:
        u = present.build(vg.ask(q))
        chunks = tts.sentences(tts.for_ear(u["voice"]))
        t = time.perf_counter(); _, m = tts.speak(chunks[0], u["voice_style"]); f = time.perf_counter() - t
        first.append(f); rtf.append(m["rtf"])
        rows.append({"q": q, "style": u["voice_style"], "first_chunk": chunks[0], "first_audio_s": round(f, 2), "rtf": m["rtf"]})
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    for r in rows:
        print(f"{r['first_audio_s']:5.2f}s  rtf {r['rtf']:.2f}  [{r['style']}]  {r['first_chunk']}")
    out = {"engine": name, "voice": tts.VOICE, "cores": len(os.sched_getaffinity(0)), "load_s": round(load, 2),
           "first_audio_median_s": round(statistics.median(first), 2), "first_audio_max_s": round(max(first), 2),
           "rtf_median": round(statistics.median(rtf), 2), "peak_rss_mb": round(rss)}
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
