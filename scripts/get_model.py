"""Download the on-device language model (one-time, ~0.5 GB) into models/.

  python scripts/get_model.py
The download is the ONLY internet use; afterwards VehicleGPT runs fully offline.
"""
import json
import sys
import urllib.request
from pathlib import Path

REPOS = {"lite": "bartowski/Qwen_Qwen3-0.6B-GGUF", "plus": "bartowski/Qwen_Qwen3-1.7B-GGUF",
         "pro": "bartowski/Qwen_Qwen3-4B-GGUF"}
PROFILE = sys.argv[sys.argv.index("--profile") + 1] if "--profile" in sys.argv else "lite"
REPO = REPOS[PROFILE]
WANT = "Q4_K_M"
OUT = Path(__file__).resolve().parents[1] / "models"


def main():
    OUT.mkdir(exist_ok=True)
    files = [s["rfilename"] for s in json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{REPO}"))["siblings"]]
    name = next(f for f in files if f.endswith(f"{WANT}.gguf"))
    dest = OUT / name
    if dest.exists() and dest.stat().st_size > 100_000_000:
        print(f"already present: {dest}")
        return dest
    url = f"https://huggingface.co/{REPO}/resolve/main/{name}"
    print(f"downloading {url}")
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print(f"\nsaved {dest}")
    return dest


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
