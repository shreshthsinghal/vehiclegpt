"""Start the on-device language model server (llama.cpp) with VehicleGPT's memory-saving settings.

  python scripts/start_llm.py                    # looks for llama-server in ./llama/ or $VGPT_LLAMA_SERVER
Leave it running in its own terminal. It listens on 127.0.0.1 only (no network exposure).
"""
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
exe = "llama-server.exe" if platform.system() == "Windows" else "llama-server"
cands = [os.environ.get("VGPT_LLAMA_SERVER"), ROOT / "llama" / exe, ROOT / "llama.cpp" / "build" / "bin" / exe,
         shutil.which("llama-server")]
server = next((Path(c) for c in cands if c and Path(c).exists()), None)
import yaml
prof_name = sys.argv[sys.argv.index("--profile") + 1] if "--profile" in sys.argv else "lite"
profile = yaml.safe_load((ROOT / "profiles" / f"{prof_name}.yaml").read_text())
model = ROOT / "models" / profile["model"]
if not model.exists():
    model = next(iter(sorted((ROOT / "models").glob("*.gguf"))), None)
if not server:
    sys.exit(f"llama-server not found. Unzip the llama.cpp release into {ROOT / 'llama'} or set VGPT_LLAMA_SERVER.")
if not model:
    sys.exit("No model in models/. Run: python scripts/get_model.py")
args = [a for a in profile["llama_args"] if a != "--no-repack"]
cmd = [str(server), "-m", str(model), *args, "--host", "127.0.0.1", "--port", "8080"]
# Weight repacking: on ARM (the vehicle target) the lite profile disables it to save ~380 MB; on x86 laptops it is
# ~30% faster, so we keep it there unless --low-mem is given.
if "--no-repack" in profile["llama_args"] and (platform.machine().lower() in ("arm64", "aarch64") or "--low-mem" in sys.argv):
    cmd.append("--no-repack")
print(" ".join(cmd))
sys.exit(subprocess.call(cmd))
