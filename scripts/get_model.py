"""Download the on-device language model (one-time, ~0.5 GB) into models/.

  python scripts/get_model.py
  python scripts/get_model.py --speech        # voice input: Parakeet-110M + Silero VAD via sherpa-onnx (~105 MB, works in traffic)
  python scripts/get_model.py --speech moonshine-base   # alternative engines: parakeet-0.6b, moonshine-base,
                                                        # moonshine-tiny, whisper-base (see ask/stt.py)
  python scripts/get_model.py --speech whisper.cpp      # the older whisper.cpp path (base.en)
  python scripts/get_model.py --voice         # warm on-device voice: Kokoro-82M (~355 MB, Apache-2.0)
  python scripts/get_model.py --voice piper   # lighter on-device voice: Piper en_US-amy (~60 MB)
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


STT_RELEASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download"


def speech():
    """Voice input: sherpa-onnx (pip wheel for Windows/Linux/macOS, x86 and ARM) + an English speech model +
    the Silero voice detector, all from GitHub releases. Afterwards voice input works with no internet."""
    import subprocess
    import tarfile
    sys.path.insert(0, str(OUT.parent))
    from ask.stt import ENGINES, DEFAULT
    arg = sys.argv[sys.argv.index("--speech") + 1] if len(sys.argv) > sys.argv.index("--speech") + 1 else DEFAULT
    if arg in ("whisper.cpp", "tiny", "base"):
        return speech_whisper_cpp("tiny" if arg == "tiny" else "base")
    name = arg if arg in ENGINES else DEFAULT
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "sherpa-onnx>=1.12"])
    d = OUT / "stt"; d.mkdir(parents=True, exist_ok=True)
    for f in ("silero_vad.onnx",):
        if not (d / f).exists():
            _fetch(f"{STT_RELEASE}/asr-models/{f}", d / f)
    folder = ENGINES[name]
    if not (d / folder).exists():
        tar = d / f"{folder}.tar.bz2"
        _fetch(f"{STT_RELEASE}/asr-models/{folder}.tar.bz2", tar)
        print("unpacking…")
        with tarfile.open(tar) as t:
            t.extractall(d, filter="data")
        tar.unlink()
    else:
        print(f"already present: {d / folder}")
    print(f"Voice input ready ({name}). Restart the server: python -m ask.server")
    return True


def speech_whisper_cpp(size: str = "base"):
    """whisper.cpp program + English speech model. Afterwards voice input works with no internet."""
    import platform
    import zipfile
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


def voice():
    """Text-to-speech model + Python package. Afterwards the assistant speaks with no internet."""
    import subprocess
    kind = sys.argv[sys.argv.index("--voice") + 1] if len(sys.argv) > sys.argv.index("--voice") + 1 else "kokoro"
    if kind == "piper":
        subprocess.check_call([sys.executable, "-m", "pip", "install", "piper-tts>=1.3"])
        d = OUT / "piper"; d.mkdir(parents=True, exist_ok=True)
        base = "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx"
        for url in (base, base + ".json"):
            dest = d / url.rsplit("/", 1)[1]
            _fetch(url, dest) if not dest.exists() else print(f"already present: {dest}")
    else:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kokoro-onnx>=0.4", "soundfile>=0.12"])
        d = OUT / "kokoro"; d.mkdir(parents=True, exist_ok=True)
        rel = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
        for name in ("kokoro-v1.0.onnx", "voices-v1.0.bin"):
            dest = d / name
            _fetch(rel + name, dest) if not dest.exists() else print(f"already present: {dest}")
    print("Warm voice is ready after a server restart: python -m ask.server")
    return True


if __name__ == "__main__":
    if "--voice" in sys.argv:
        sys.exit(0 if voice() else 1)
    if "--speech" in sys.argv:
        sys.exit(0 if speech() else 1)
    sys.exit(0 if main() else 1)
