# Voice input in a noisy car (Phase 7, 28 Nov)

**The problem.** The driver must be able to *speak* a question in traffic, and nothing may go to the cloud.
The first version (whisper.cpp) needed a separate program download on Windows, cut off the recording with a
loudness threshold (which never stops in traffic and can't tell a horn from a voice), and was never tested in noise.

## What runs now (all on the car, nothing sent anywhere)

```
mic --(echo cancellation on; browser noise suppression off)--> 16 kHz chunks every ~45 ms --> 127.0.0.1
    --> Silero VAD: is someone speaking? stop 0.8 s after the driver stops (horns and road noise don't count)
    --> Parakeet-110M (NVIDIA, int8) speech-to-text via sherpa-onnx
    --> spoken numbers to digits ("sixty percent" -> "60%", "one hundred ten kilometres per hour" -> "110 km/h")
    --> the same question pipeline as typing; the screen shows "You said ..." so a mishearing is visible
```

| Part | What it is | Licence | Size |
|---|---|---|---|
| sherpa-onnx | Speech toolkit, pip wheel for Windows/Linux/macOS, x86 and ARM | Apache-2.0 | 2.5 MB wheel |
| Parakeet TDT-CTC 110M (int8) | Speech recognition, trained on 36,000 h of English | CC-BY-4.0 (NVIDIA NeMo + Suno.ai) | 105 MB |
| Silero VAD | Neural voice detector | MIT | 0.6 MB |

Setup is one command (no compiler, no separate program): `python scripts/get_model.py --speech`.
Profiles: Lite uses Parakeet-110M; Plus/Pro use Parakeet-0.6B (more accurate in the loudest traffic, 1.2 GB).

## How we tested it

- **Speech:** 60 driver questions (capability + blind sets) spoken in 6 voices: US, British and Indian-English,
  male and female, at varied speed → 120 clips (on-device Kokoro voice; `eval/stt_speech.py`).
- **Noise** (`eval/stt_bench.py`), at a given SNR (speech vs noise power; 0 dB = noise as loud as the voice):
  - *cabin*: road/tyre rumble + AC fan + motor whine (synthetic), 10/5/0 dB
  - *traffic*: cabin + real horns, sirens, engines, trains (ESC-50 recordings), 5/0 dB
  - *rain/wind*: cabin + real rain and wind (ESC-50), 5 dB
  - *radio*: cabin + another person talking, 5 dB
  - *car / street*: **real recordings from inside a moving car and from a busy street** (DEMAND corpus), 5/0 dB
- **Score:** "same answer" = the transcript leads to the same VehicleGPT analysis as the typed question (what
  the driver feels), plus word error rate (WER).
- **Hardware:** GitHub ARM64 runner pinned to **2 cores** (`.github/workflows/stt_noise.yml`, run 46143968718).

## Results (2 ARM cores; cell = same answer / WER %)

| Engine | clean | cabin 5 dB | cabin 0 dB | traffic 5 dB | traffic 0 dB | rain/wind 5 dB | radio 5 dB | real car 0 dB | real street 5 dB | time per question | RAM |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Parakeet-110M (chosen)** | **98% / 4** | **98% / 5** | 91% / 9 | **93% / 8** | **88% / 18** | **92% / 9** | 58% / 49 | **98% / 4** | **98% / 5** | **97 ms** | 382 MB |
| Parakeet-0.6B (Plus/Pro) | 98% / 3 | 98% / 6 | 96% / 6 | 96% / 7 | 90% / 13 | 94% / 8 | 62% / 39 | 98% / 3 | 98% / 4 | 232 ms | 1234 MB |
| Moonshine-base | 96% / 4 | 94% / 7 | 92% / 10 | 88% / 10 | 79% / 25 | 88% / 11 | 53% / 65 | 96% / 5 | 92% / 6 | 126 ms | 411 MB |
| Moonshine-tiny | 97% / 4 | 95% / 10 | 88% / 16 | 82% / 17 | 69% / 33 | 88% / 13 | 47% / 68 | 96% / 5 | 89% / 8 | 114 ms | 284 MB |
| Whisper-base (≈ the old engine) | 96% / 4 | 92% / 5 | 92% / 9 | 88% / 8 | 77% / 21 | 90% / 11 | 61% / 40 | 96% / 4 | 94% / 5 | 395 ms | 487 MB |

What this means:
- In a **real moving car** (DEMAND recording) the chosen model gets **98% of questions right even when the
  noise is as loud as the voice**, and answers in about **0.1 s** after the driver stops speaking (+0.8 s wait to
  be sure they have finished).
- With **horns and sirens as loud as the voice** it still gets **88%** right, compared with 77% for the old Whisper engine.
- **Denoising first made it worse** (GTCRN: traffic 0 dB WER 18.8 → 23.6%). The model is trained on noisy speech
  and the cleaning step leaves artefacts, so it is off by default. The browser's own noise suppression is
  also off, and its echo cancellation is on.
- **Neural end-of-speech works in noise.** With steady road noise continuing after the question, recording
  stopped 0.83–0.86 s after the last word. Pure road noise is never taken as speech (`tests/test_ask.py`).
- **Another voice is the weak spot (radio 5 dB: 58%).** A microphone can't tell whose voice is whose. In a real
  car this is handled the standard way:
  - the head unit **mutes or ducks its own media while listening**, and **echo cancellation** removes what the
    car itself plays;
  - a **driver-facing beamforming mic array** (hardware) favours the driver's seat.

  We name these as production requirements rather than claim we solved them.

## Limits (honest)
- The test voices are synthetic (TTS), not real drivers. The noise recordings are real (ESC-50, DEMAND).
  Before the final demo, a few questions recorded by team members in a real car will be added.
- English only. Hinglish questions are not handled by this speech model (typed Hinglish still works).
- ESC-50 (CC BY-NC 3.0) and DEMAND (CC BY-SA 3.0) are used only for this evaluation and are not shipped.
