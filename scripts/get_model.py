"""Download the on-device language model (one-time, ~0.5 GB) into models/.

  python scripts/get_model.py
  python scripts/get_model.py --speech        # on-device speech-to-text: whisper.cpp + ggml-base.en (~142 MB)
  python scripts/get_model.py --speech tiny   # smaller, less accurate speech model (~75 MB)
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


def _fetch(url: str, dest: Path):
    print(f"downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "vehiclegpt-setup"})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print(f"\nsaved {dest}")


def speech():
    """whisper.cpp program + English speech model. Afterwards voice input works with no internet."""
    import platform
    import zipfile
    size = sys.argv[sys.argv.index("--speech") + 1] if len(sys.argv) > sys.argv.index("--speech") + 1 else "base"
    size = size if size in ("tiny", "base") else "base"
    OUT.mkdir(exist_ok=True)
    model = OUT / f"ggml-{size}.en.bin"
    if not model.exists():
        _fetch(f"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{model.name}", model)
    else:
        print(f"already present: {model}")
    root = OUT.parent / "whisper"
    if platform.system() == "Windows":
        if not any(root.rglob("whisper-cli.exe")):
            rel = json.load(urllib.request.urlopen(urllib.request.Request(
                "https://api.github.com/repos/ggml-org/whisper.cpp/releases/latest", headers={"User-Agent": "vehiclegpt-setup"})))
            asset = next(a for a in rel["assets"] if a["name"] == "whisper-bin-x64.zip")
            zpath = OUT / asset["name"]
            _fetch(asset["browser_download_url"], zpath)
            zipfile.ZipFile(zpath).extractall(root)
            zpath.unlink()
        exe = next(root.rglob("whisper-cli.exe"))
        print(f"whisper.cpp ready: {exe}")
    else:
        if not (any(root.rglob("whisper-cli"))):
            print("Build whisper.cpp once (Linux/macOS):\n"
                  "  git clone --depth 1 https://github.com/ggml-org/whisper.cpp whisper\n"
                  "  cmake -S whisper -B whisper/build -DCMAKE_BUILD_TYPE=Release && cmake --build whisper/build -j --target whisper-cli")
    print("Voice input is ready after a server restart: python -m ask.server")
    return True


if __name__ == "__main__":
    if "--speech" in sys.argv:
        sys.exit(0 if speech() else 1)
    sys.exit(0 if main() else 1)
