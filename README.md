# VehicleGPT — ask the vehicle, get evidence, not guesses

**Team Nexus · BITS Pilani, K K Birla Goa Campus** — Shreshth Singhal (Team Leader), Shaurya Awasthi, Subhranshu Mitra
Telipoort Season 4 (Tata Elxsi) · Case Study 5: Natural Language Vehicle Information using an On-Device SLM/LLM

An on-device assistant that keeps a searchable memory of everything a vehicle does (CAN, SOME/IP, Android VHAL, QNX logs,
ADAS, camera metadata, diagnostics) and answers plain-language questions such as *"Why has my range dropped?"* in under
2 seconds on a 2-core / 2 GB ARM target, with **no cloud** and **no path to the vehicle's safety domain**.

> The small language model interprets the question; a deterministic data engine computes the answer; a verifier
> checks every number against the data before it is shown.

## Layout
| Folder | Layer |
|---|---|
| `twin/` | Digital twin: simulated and recorded vehicle data with injected faults |
| `sense/` | Listen-only data collection and normalisation to COVESA VSS |
| `remember/` | Tiered on-device storage (weeks of history) |
| `reason/` | Intent router, SLM planner, analysis playbooks, verifier |
| `ask/` | User interfaces for drivers, service engineers, fleets and validation teams |
| `safety/` | Isolation and privacy checks |
| `eval/` | Golden questions, accuracy and performance benchmarks |
| `profiles/` | Hardware profiles (Lite / Plus / Pro) |
| `docker/` | Containers, including the 2 CPU / 2 GB constrained run |

## Quick start (Windows, macOS or Linux)
```
git clone https://github.com/shreshthsinghal/vehiclegpt
cd vehiclegpt
python scripts/check_setup.py
```

## Status
Phase 0 (setup). See the project logbook for the plan.
