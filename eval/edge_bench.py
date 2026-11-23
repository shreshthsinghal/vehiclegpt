"""Edge proof: run the whole on-device stack inside its resource limits and measure it.

Meant to run INSIDE the container started with e.g.
  docker run --cpus=2 --memory=2g --network none -v ./models:/app/models:ro -v ./data:/app/data vehiclegpt \
      python -m eval.edge_bench --profile lite
Measures (from the container's own cgroup accounting, so it covers EVERY process in the box):
  ingest of 8 weeks (time, CPU, SENSE cost per vehicle-second), memory peak/current/anon, CPU idle vs busy,
  FAST answers through the HTTP server, SMART plans through the on-device SLM, network isolation.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CG = Path("/sys/fs/cgroup")


def cg_read(name):
    try:
        return (CG / name).read_text().strip()
    except OSError:
        return None


def mem_now():
    cur = cg_read("memory.current")
    stat = dict(line.split() for line in (cg_read("memory.stat") or "").splitlines())
    return {"current_mb": round(int(cur) / 1e6, 1) if cur else None,
            "anon_mb": round(int(stat.get("anon", 0)) / 1e6, 1), "file_mb": round(int(stat.get("file", 0)) / 1e6, 1)}


def cpu_usec():
    stat = dict(line.split() for line in (cg_read("cpu.stat") or "usage_usec 0").splitlines())
    return int(stat.get("usage_usec", 0))


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.max = {"current_mb": 0, "anon_mb": 0, "file_mb": 0}
        self.stop = False

    def run(self):
        while not self.stop:
            m = mem_now()
            for k in self.max:
                if m[k] is not None:
                    self.max[k] = max(self.max[k], m[k])
            time.sleep(0.25)


def pct(xs, q):
    xs = sorted(xs)
    return round(xs[int(q * (len(xs) - 1))], 1) if xs else None


def cpu_cores_during(fn):
    c0, t0 = cpu_usec(), time.time()
    out = fn()
    return out, round((cpu_usec() - c0) / 1e6 / (time.time() - t0), 2)


def main(profile: str, port: int, skip_ingest: bool):
    prof = yaml.safe_load((ROOT / "profiles" / f"{profile}.yaml").read_text())
    res = {"profile": profile, "limits": {"cpu_max": cg_read("cpu.max"), "memory_max_mb":
           round(int(cg_read("memory.max")) / 1e6) if (cg_read("memory.max") or "max").isdigit() else "max"},
           "network_interfaces": sorted(os.listdir("/sys/class/net")) if Path("/sys/class/net").exists() else None}
    smp = Sampler()
    smp.start()

    # ---------------------------------------------------------------- 1. ingest 8 weeks (SENSE -> REMEMBER)
    from remember.build import build
    if not skip_ingest:
        print("PHASE INGEST", flush=True)
        r0 = resource.getrusage(resource.RUSAGE_SELF)
        rep, cores = cpu_cores_during(lambda: build(quiet=True))
        r1 = resource.getrusage(resource.RUSAGE_SELF)
        cpu_s = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
        veh_s = rep["sense"]["snapshots_1hz"]
        res["ingest"] = {"wall_s": rep["seconds"], "cpu_s": round(cpu_s, 1), "vehicle_seconds": veh_s,
                         "sense_remember_cpu_pct_of_one_core_at_realtime": round(cpu_s / veh_s * 100, 3),
                         "store_mb": rep["total_on_disk_mb"], "peak_mem_mb_during_ingest": smp.max["current_mb"]}
        print(json.dumps(res["ingest"]), flush=True)

    # ---------------------------------------------------------------- 2. start the SLM server
    print("PHASE START_LLM", flush=True)
    model = ROOT / "models" / prof["model"]
    t0 = time.time()
    llm = subprocess.Popen([os.environ.get("VGPT_LLAMA_SERVER", "llama-server"), "-m", str(model), *prof["llama_args"],
                            "--host", "127.0.0.1", "--port", "8080"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(300):
        try:
            if urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=1).status == 200:
                break
        except Exception:      # noqa: BLE001
            time.sleep(0.5)
    res["llm_ready_s"] = round(time.time() - t0, 1)

    # ---------------------------------------------------------------- 3. HMI server (same process) + warm-up
    import ask.server as S
    from http.server import ThreadingHTTPServer
    t0 = time.time()
    app = S.App()
    srv = ThreadingHTTPServer(("127.0.0.1", port), S.make_handler(app))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    res["app_ready_s_incl_slm_warmup"] = round(time.time() - t0, 1)
    res["slm_warmup_ms"] = round(app.vg.warmup_ms)

    # ---------------------------------------------------------------- 4. idle CPU (everything running, nobody asking)
    _, res["cpu_cores_idle"] = cpu_cores_during(lambda: time.sleep(20))
    res["mem_idle"] = mem_now()

    # ---------------------------------------------------------------- 5. FAST answers through HTTP
    g = yaml.safe_load((ROOT / "eval" / "golden_questions.yaml").read_text())
    fast_qs = [(q["intent"], t) for q in g["brief_questions"] for t in [q["canonical"]] + q["paraphrases"]]
    fast_qs += [(q.get("intent"), q["q"]) for q in g["long_tail"]] + [(None, q["q"]) for q in g["safety_and_scope"]]

    def run_fast():
        lat, ok, ver = [], 0, 0
        for intent, q in fast_qs:
            t = time.perf_counter()
            req = urllib.request.Request(f"http://127.0.0.1:{port}/api/ask", data=json.dumps({"question": q}).encode())
            a = json.loads(urllib.request.urlopen(req, timeout=60).read())
            lat.append((time.perf_counter() - t) * 1000)
            ok += intent is None or a["intent"] == intent
            ver += a["verified"]
        return lat, ok, ver
    print("PHASE FAST", flush=True)
    (lat, ok, ver), cores = cpu_cores_during(run_fast)
    res["fast_http"] = {"n": len(fast_qs), "intent_ok": ok, "verified": ver, "p50_ms": pct(lat, .5), "p95_ms": pct(lat, .95),
                        "max_ms": pct(lat, 1.0), "cpu_cores": cores}

    # ---------------------------------------------------------------- 6. SMART plans through the SLM (router bypassed)
    smart_qs = [(q["intent"], t) for q in g["brief_questions"] for t in [q["canonical"]] + q["paraphrases"]]
    smart_qs += [(q.get("intent"), q["q"]) for q in g["long_tail"]]

    def run_smart():
        lat, ok, ver = [], 0, 0
        for intent, q in smart_qs:
            with app.lock:
                a = app.vg.ask(q, force_smart=True)
            lat.append(a.timings_ms["total"])
            ok += a.plan.get("intent") == intent
            ver += a.verified
        return lat, ok, ver
    print("PHASE SMART", flush=True)
    (lat, ok, ver), cores = cpu_cores_during(run_smart)
    print("PHASE SMART_DONE", flush=True)
    res["smart"] = {"n": len(smart_qs), "plan_intent_ok": ok, "verified": ver, "p50_ms": pct(lat, .5),
                    "p95_ms": pct(lat, .95), "max_ms": pct(lat, 1.0), "cpu_cores": cores}

    # ---------------------------------------------------------------- 7. totals + isolation evidence
    smp.stop = True
    res["mem_peak_sampled"] = smp.max
    peak = cg_read("memory.peak")
    res["mem_peak_cgroup_mb"] = round(int(peak) / 1e6, 1) if peak else None
    res["privacy"] = app.counters()
    res["fits_budget"] = {"memory": (res["mem_peak_cgroup_mb"] or smp.max["current_mb"]) < prof["ram_gb"] * 1000,
                          "smart_p95_under_2s": res["smart"]["p95_ms"] < 2000, "fast_p95_under_2s": res["fast_http"]["p95_ms"] < 2000}
    llm.terminate()
    out = ROOT / "eval" / "results" / f"edge_bench_{profile}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print("RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="lite")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--skip-ingest", action="store_true")
    a = ap.parse_args()
    main(a.profile, a.port, a.skip_ingest)
