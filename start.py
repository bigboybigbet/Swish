"""
One-command launcher for Swish.

    python start.py              (or double-click start.bat / run ./start.sh)
    python start.py --port 8010  (any `python -m server` option is passed on)

First run: creates the virtual environment in .venv and installs
requirements.txt.  Later runs skip the install unless requirements.txt has
changed since the last one (a hash of it is kept in .venv), then start the
web app.  Run it with any Python 3.11+; it uses the .venv one for the app.
"""
import hashlib
import os
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
PYTHON = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
REQUIREMENTS = ROOT / "requirements.txt"
STAMP = VENV / ".requirements.sha256"


def requirements_hash() -> str:
    return hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()


def ensure_environment() -> None:
    if not PYTHON.exists():
        print("First run: creating the virtual environment in .venv …", flush=True)
        venv.EnvBuilder(with_pip=True).create(VENV)
    want = requirements_hash()
    have = STAMP.read_text().strip() if STAMP.exists() else ""
    if want == have:
        print("Dependencies are up to date.", flush=True)
        return
    print("Installing dependencies (first run, or requirements.txt changed). "
          "This takes a few minutes once …", flush=True)
    subprocess.check_call([str(PYTHON), "-m", "pip", "install", "--upgrade", "pip", "-q"])
    subprocess.check_call([str(PYTHON), "-m", "pip", "install", "-r", str(REQUIREMENTS)])
    STAMP.write_text(want)


def main(argv: list) -> int:
    if sys.version_info < (3, 11):
        print(f"Swish needs Python 3.11 or newer; this is {sys.version.split()[0]}.")
        return 1
    try:
        ensure_environment()
    except subprocess.CalledProcessError:
        print("\nInstalling the dependencies failed; see the messages above. "
              "Check your internet connection and run start again.")
        return 1
    try:
        return subprocess.call([str(PYTHON), "-m", "server", *argv], cwd=ROOT)
    except KeyboardInterrupt:          # Ctrl+C stops the server, not an error
        return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
