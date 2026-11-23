# External dataset: 1,050 EV-assistant scenarios (shared by the user, 23 Nov 2026)

`eval/external/ev_dashboard_assistant_1050.csv` was written for a general, cloud-connected assistant that also
**operates the car**. VehicleGPT is deliberately offline, read-only and answers from recorded vehicle data, so we did
not copy that behaviour. Each row was mapped to what VehicleGPT *should* do (`eval/import_ev1050.py`):

| Label | Rows | Meaning |
|---|---|---|
| answer (intent) | 146 | answerable from recorded data or the car's own consumption model: range, trip feasibility, charging time, battery health, tyres, warning lights, emergencies |
| refuse | 275 | commands to operate the car (climate, seats, windows, charge limit, drive mode, ADAS), safety gates, prompt injection |
| no_data | 623 | needs maps/internet/phone/media (routing, charger networks, prices, calls, streaming) or owner's-manual knowledge or an unrecorded signal: an honest "not available" answer, never a made-up one |
| skip | 6 | needs earlier conversation turns (coreference, barge-in) |

**What we excluded on purpose (constraint violations):** executing commands, navigation and live charger data, calls
and media, smart home, cloud recall/OTA, and any answer about a signal the car does not record.

## Method
- Split **by utterance template** (numbers removed) into train (702) and test (342), so near-duplicate rows never cross.
- Tuning used **train only**: general rules (leading command verbs, online services, owner's-manual questions,
  unrecorded signals, dashboard icons, scenario phrasings, prompt injection) and new abilities (miles/mph/°F,
  stated charge level, charger kW, round trips, "how slow must I drive", drowsy-driving guide, conservative
  cold-weather estimates).
- The test split was run **once** after tuning. One bug found there was fixed and re-run: a charger-power number
  that the verifier flagged. No label was changed after seeing test results.

## Results (FAST path, no language model)
| Split | Before tuning | After tuning |
|---|---|---|
| Train (702) | 376 (53.6%) | **693/702** |
| **Test, held out (342)** | **163 (47.7%)** | **321/342** · data questions 56/58 · refusals 87/88 · honest "not available" 178/196 |
| Numbers verified (test) | 342/342 | 342/342 |

Golden exam, blind set and capability set unchanged (55/55, 40/41, 40/40). Remaining test misses are listed by
`python -m eval.run_blind --file external/ev1050_labeled.yaml --split test --quiet`; they are mostly unseen
command verbs ("dip the mirrors") and charger-hardware questions.
