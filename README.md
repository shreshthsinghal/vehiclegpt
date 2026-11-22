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

## Phase 1: digital twin (simulated vehicle data)
```
pip install -r requirements.txt
python -m twin.generate          # 56 days of data in ~1 minute -> data/twin/
python -m twin.inspect_data      # see one decoded sample from every source
```
Live demo (two terminals):
```
python -m twin.monitor                                   # terminal 1: receive-only dashboard
python -m twin.replay --day 55 --start 18:00 --minutes 5 # terminal 2: replay yesterday's rainy drive
```
| Output | Format |
|---|---|
| `data/twin/can/dayNN.blf` | CAN + CAN-FD frames (Vector BLF), decoded with `twin/nexus_ev.dbc` (19 messages, 177 signals) |
| `data/twin/someip/` | SOME/IP notifications (thermal, ADAS, camera services) with real 16-byte headers |
| `data/twin/vhal/` | Android Automotive VHAL property events |
| `data/twin/qnx/` | QNX slog2-style system log |
| `data/twin/uds/` | UDS diagnostic trouble codes (service 0x19/0x59) |
| `data/twin/truth/` | Answer key for grading - never read by VehicleGPT |

Scenarios and planted faults: `twin/fault_catalogue.yaml`. Exam questions: `eval/golden_questions.yaml`.

## Phase 2: SENSE (listen-only collection, normalisation, events, black box)
```
python -m sense.ingest --backfill     # all recordings -> data/sense/ (events, logs, black-box case files) in ~45 s
```
Live (two terminals): `python -m sense.ingest --live`, then `python -m twin.replay --day 55 --start 18:03 --minutes 3`.

| Piece | What it does |
|---|---|
| `sense/adapters/sources.py` | One listen-only adapter per source: CAN/CAN-FD (DBC), SOME/IP (header-validated), Android VHAL, QNX slog2, UDS DTCs |
| `sense/mappings.yaml` | Every native name -> COVESA VSS path (192 paths seen); location coarsened to ~1 km |
| `sense/engine.py` | 1 Hz normalised snapshot; event miner (trips, charges, warnings, DTCs, ADAS, data gaps, contextual anomalies, Drain3 log templates); black box (-60 s / +10 s case files) |
| `safety/guards.py` | `ListenOnlyBus`: any CAN transmit raises and is counted (must stay 0) |

## Phase 3: REMEMBER (tiered on-device memory)
```
python -m remember.build      # SENSE -> data/store/vehicle.db (+ encrypted case files) and a size report
```
| Tier | Content | Kept | Measured (8 weeks, 1.4 h/day) |
|---|---|---|---|
| HOT | last 10 min of 1 Hz snapshots, RAM ring | 10 min | in memory |
| WARM | 1 Hz, all 192 signals, 10-min compressed blocks (delta-of-delta + quantised deltas + zlib, lossless) | 14 days | 0.49 bytes/reading |
| COLD | 1-minute min/max/mean/count per signal | 400 days | 0.41 MB/day |
| PINNED | black-box case files | forever | ~5 KB each |

Plus trips, charges, events, logs (FTS5 keyword search), DTC descriptions. Time-series blocks and case files are
AES-256-GCM encrypted (key file stands in for the vehicle HSM). Total on disk: ~36 MB for 8 weeks.

## Status
Phase 3 (REMEMBER) complete. Tests: `python -m pytest -q` (13 tests, ~4 min).
