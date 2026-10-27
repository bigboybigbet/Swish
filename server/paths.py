"""
Absolute paths for everything the server reads or writes.

config.py and a few scripts use paths relative to the working directory,
so the server always runs with cwd = ROOT (see chdir_root). Server code
should use the absolute paths below instead of the config names.
"""
import mimetypes
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # noqa: E402  (needs ROOT on sys.path)

# Module scripts are rejected by browsers when served as octet-stream, and
# some Python/OS combinations don't know these types.
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("font/woff2", ".woff2")

# ── web ───────────────────────────────────────────────────────────────────────
WEB_DIR       = ROOT / "web"
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR    = WEB_DIR / "static"
VENDOR_DIR    = STATIC_DIR / "vendor"
VENDOR_LOCK   = WEB_DIR / "vendor.lock.json"
TEAMS_JSON    = STATIC_DIR / "data" / "teams.json"
GLOSSARY_JSON = STATIC_DIR / "data" / "glossary.json"

# ── existing state files (unchanged locations) ───────────────────────────────
MODEL_FILE         = ROOT / config.MODEL_FILE
DATASET_FILE       = ROOT / config.DATASET_FILE
ELO_FILE           = ROOT / config.ELO_FILE
HISTORY_FILE       = ROOT / config.HISTORY_FILE
OPTUNA_PARAMS_FILE = ROOT / config.OPTUNA_PARAMS_FILE
PLAYER_IMPACT_FILE = ROOT / config.PLAYER_IMPACT_FILE
CACHE_DB           = ROOT / config.CACHE_DB
BLEND_FILE         = ROOT / config.BLEND_FILE
AUTOBACKTEST_LOG   = ROOT / "autobacktest_log.json"
AUTOBACKTEST_CSV   = ROOT / "autobacktest_results.csv"
PLAYER_IMPACT_UPDATER = ROOT / "player_impact_updater.py"

# ── new state (gitignored) ────────────────────────────────────────────────────
ENV_FILE           = ROOT / ".env"
SETTINGS_FILE      = ROOT / "settings.json"
MODEL_METRICS_FILE = ROOT / "model_metrics.json"
RATINGS_STATE_FILE = ROOT / "ratings_state.joblib"
RUNS_DIR           = ROOT / "runs"
JOBS_FILE          = RUNS_DIR / "jobs.json"
RESULTS_DIR        = RUNS_DIR / "results"
SCANS_DIR          = RUNS_DIR / "scans"
SNAPSHOTS_DIR      = RUNS_DIR / "ratings_snapshots"


def chdir_root():
    """Make relative paths in config.py & co. resolve against the project root."""
    os.chdir(ROOT)
