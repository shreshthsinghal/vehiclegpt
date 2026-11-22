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
model = next(iter(sorted((ROOT / "models").glob("*.gguf"))), None)
if not server:
    sys.exit(f"llama-server not found. Unzip the llama.cpp release into {ROOT / 'llama'} or set VGPT_LLAMA_SERVER.")
if not model:
    sys.exit("No model in models/. Run: python scripts/get_model.py")
cmd = [str(server), "-m", str(model), "-t", "2", "-c", "1024", "-np", "1", "--cache-ram", "0", "-b", "256", "-ub", "128", "-fa", "on",
       "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "--host", "127.0.0.1", "--port", "8080"]
print(" ".join(cmd))
sys.exit(subprocess.call(cmd))
