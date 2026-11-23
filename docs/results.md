# VehicleGPT - measured results (evidence sheet)

All numbers below are **measured**, not targets. Source: GitHub Actions runs on native ARM64 (Neoverse-N2) and x86-64,
Linux, plus the team's Windows laptop. Data: 8 weeks of simulated vehicle recordings (digital twin, seed 2026).

## 1. Edge performance - whole stack in one container, hard limits, no network
`docker run --cpus=2 --memory=2g --network none` (Lite profile), ARM64. Container had only the `lo` interface.

| Metric | Round 1 target | Measured (Lite: 2 CPU / 2 GB) | Measured (Plus: 4 CPU / 4 GB, 1.7B model) |
|---|---|---|---|
| Common-question latency (FAST path, through the HTTP server, 81 questions) | p95 ≤ 2 s | **p50 3.0 ms, p95 27 ms** | p50 2.8 ms, p95 30 ms |
| Unusual-question latency (SMART path, on-device SLM plans, 75 questions) | ≈3 s Lite | **p50 1.36 s, p95 1.77 s, max 1.90 s** | p50 1.26 s, p95 1.71 s |
| SLM plan accuracy (router bypassed, grammar-constrained) | - | **63/75 (84%)**, Qwen3-0.6B | **65/75 (87%)**, Qwen3-1.7B |
| Answers whose numbers were all verified | 100% | **156/156** | 156/156 |
| Memory, whole container peak (cgroup) | ≤ 1.3 GB | **290 MB** (+ model file pages ≈ 456 MB that were already in the host page cache; worst case ≈ 750 MB) | 1.40 GB (weights repacked for speed) |
| CPU while idle (everything running, no questions) | ≤ 25% | **0.0 cores** | 0.0 cores |
| SENSE + REMEMBER cost while the car drives | - | **0.039% of one core** (110.7 CPU-s for 280,637 vehicle-seconds) | 0.038% |
| Ingest of the full 8 weeks (backfill) | - | 122 s, peak 235 MB | 117 s |
| Start-up: SLM ready / app ready incl. warm-up | - | 0.5 s / 9.0 s | 1.5 s / 7.3 s |
| Network egress | 0 B | **0 external connections; container had no network** | same |
| CAN transmit attempts by the AI | 0 | **0** | 0 |
| Image size (app + llama.cpp) | - | 435 MB | 435 MB |

## 2. Safety-domain isolation (jitter test)
`cyclictest` (Linux real-time latency tool) on CPU 3 at priority 80, 1 ms period, 20,000 cycles, while VehicleGPT is
confined to CPUs 0-1 (`--cpuset-cpus`) and running the SMART load at ~2 cores.

| | avg latency | max latency |
|---|---|---|
| VehicleGPT not running | 3 µs | 117 µs |
| VehicleGPT at full load | 4 µs | 69 µs |
No degradation: the AI partition cannot starve the safety partition. (Production: hypervisor partition + cgroups.)

## 3. Storage (8 weeks, 192 signals, encrypted)
| Metric | Round 1 target | Measured |
|---|---|---|
| WARM 1 Hz history | 3 B/reading | **0.49 B/reading** (lossless), 7.3 MB for 14 days |
| COLD 1-minute rollups | 0.4 MB/day | **0.41 MB/day** |
| Black-box case files | ~126 KB each | **~5 KB each** (gzip + AES-256-GCM) |
| Total for 8 weeks | ≈0.4 GB @ 2 h/day | **38 MB measured @ 1.4 h/day → ~52 MB @ 2 h/day, ~209 MB @ 8 h/day** |
| Query latency (series / rollup / keyword / meaning search) | - | 1-2 ms |

## 4. Accuracy vs the answer key (counterfactual truth from the digital twin)
| Question (from the brief) | VehicleGPT | Answer key |
|---|---|---|
| Q1 Why has range dropped (2 weeks)? | range -12.8%; AC/heat 60%, traffic 40%; tyres normal | -12.4%; 64% / 36% |
| Q2 When was the largest efficiency drop? | Fri 30 Oct; period 25-30 Oct; rear-left tyre | worst 30 Oct (+8.2%), onset 25 Oct |
| Q3 ADAS disengagements yesterday | 5: 4 lane markings lost (rain), 1 camera fault (U1C01) | identical |
| Q4 What happened before the overheat warning? | fast charge → pump P1A20 15:51 → 52.5 °C → derate 15:59:51 → warning 16:03; heat alone ruled out | identical chain, root cause P1A20 |
| Q5 City vs highway | 18.2 vs 19.8 kWh/100 km | 18.17 vs 19.84 |
Golden exam: 55/55 brief variants routed, 54/54 graded correct, 20/20 long-tail routed, 6/6 unsafe requests refused.
Early warning: tyre anomaly flagged **26 Oct, 4 days before the car's own TPMS warning (30 Oct)**.

## Honest caveats
- Neoverse-N2 cores are faster than Cortex-A55. Estimated SMART p95 on a real A55 (Lite): ~3 s; FAST path (common questions) stays in milliseconds. Plus/Pro profiles meet 2 s for SMART too.
- Router accuracy on the golden set is optimistic (tuned once on its misses); blind accuracy will be reported separately.
- The digital twin is synthetic (no CARLA: no GPU); real recorded datasets can be fed through the same adapters.


## Phase 7: on-device voice (CI run 45134481754, GitHub ARM64, `taskset` 2 cores)
12 questions (golden phrasings plus a safety command) were spoken by a synthetic voice (espeak-ng), transcribed by
whisper.cpp, and answered by VehicleGPT. "Same analysis" means the spoken question routed to the same playbook as
the typed one.

| Model · settings | Same analysis as typed | Word error rate | Median time | Peak memory (only while transcribing) |
|---|---|---|---|---|
| **base.en · clip-sized window, greedy (default)** | **12/12** | **6.3%** | **1.36 s** | **222 MB** |
| base.en · whisper defaults (30 s window, beam 5) | 12/12 | 7.9% | 8.88 s | 295 MB |
| tiny.en · clip-sized window, greedy | 11/12 | 119% (one hallucinated run-on) | 0.69 s | 139 MB |
| tiny.en · whisper defaults | 10/12 | 16.7% | 4.06 s | 187 MB |

End to end through the head-unit server, "why has my range dropped" was heard as "Wifi's my range dropped" and still
routed to `range_drop`, verified, with the screen's big number at 12.8%. The router tolerates mishearings because it
matches meaning, not exact words.
Caveat: a synthetic voice in a quiet room. Real cabins add road noise and accents; next step is to test with recordings of real voices.
