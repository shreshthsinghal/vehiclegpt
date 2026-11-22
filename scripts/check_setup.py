"""Phase 0 setup check. Run:  python scripts/check_setup.py

Prints a pass/fail line for each tool the project needs on your machine.
Nothing here changes your system; it only looks.
"""
import platform
import shutil
import subprocess
import sys


def run(cmd):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        return (out.stdout or out.stderr).strip().splitlines()[0]
    except Exception as exc:  # noqa: BLE001
        return f"error: {exc}"


def line(ok, name, detail):
    mark = "PASS" if ok else ("WARN" if ok is None else "FAIL")
    print(f"[{mark}] {name:<16} {detail}")


print("VehicleGPT setup check\n")
v = sys.version_info
line((3, 11) <= (v.major, v.minor) <= (3, 12), "Python", f"{platform.python_version()} (need 3.11 or 3.12)")
line(True, "OS / CPU", f"{platform.system()} {platform.release()} / {platform.machine()}")

git = shutil.which("git")
line(bool(git), "Git", run(["git", "--version"]) if git else "not found on PATH")

docker = shutil.which("docker")
if docker:
    line(True, "Docker CLI", run(["docker", "--version"]))
    info = run(["docker", "info", "--format", "{{.ServerVersion}}"])
    running = info and "error" not in info.lower() and "cannot" not in info.lower()
    line(True if running else None, "Docker engine", f"running (server {info})" if running else "installed but not running (open Docker Desktop) - OK for now")
else:
    line(None, "Docker", "not installed yet - needed from Phase 6, OK for now")

code = shutil.which("code")
line(True if code else None, "VS Code", "found" if code else "not on PATH (fine if installed)")

print("\nIf every line is PASS or WARN you are ready for Phase 1.")
