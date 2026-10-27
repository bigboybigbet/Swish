"""
Training worker, run by the `train` job in its own (niced) process:

    python -m server.workers.train [--force-rebuild] [--optuna]

It runs engine.models.train_model and prints one JSON object per line on
stdout for the job manager:

    {"type": "stage", "i": 3, "n": 7, "msg": "Replaying Elo ratings"}
    {"type": "log", "msg": "Paired 3,690 matchup rows from 7,380 team rows"}
    {"type": "result", "data": {"rows": 3690, "metrics": {...}}}
    {"type": "error", "msg": "..."}

On success it writes model_metrics.json (atomically) so the Model page's
metrics, feature importance and reliability curve survive a refresh. Model,
dataset and Elo files are written atomically by engine.models, so a
cancelled run leaves the previous ones in place.

Demo mode (NBA_QUANT_DEMO_DATE): the game logs come from the 2010-24 history
files (the 3 seasons up to the demo date) instead of stats.nba.com.
"""
import argparse
import json
import os
import signal
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# (stage number, message prefixes from train_model's status_cb, label)
STAGE_RULES = [
    (1, ("Downloading",), "Downloading game logs"),
    (2, ("Paired",), "Pairing home and away rows"),
    (3, ("Elo ratings replayed",), "Replaying Elo ratings"),
    (4, ("Running RandomizedSearchCV", "Running Optuna"), "Searching hyperparameters"),
    (5, ("Training XGBoost",), "Fitting XGBoost"),
    (6, ("Calibrating", "Not enough calibration"), "Calibrating probabilities"),
    (7, ("Model saved",), "Saving the model"),
]
N_STAGES = len(STAGE_RULES)


def emit(obj: dict):
    print(json.dumps(obj, default=str), flush=True)


def stage_for(msg: str):
    """(i, label) when a status message starts a new stage, else None."""
    for i, prefixes, label in STAGE_RULES:
        if msg.startswith(prefixes):
            return i, label
    return None


def use_history_logs(demo):
    """Demo mode: train_model downloads nothing; it reads the history files instead."""
    import config
    import engine.models as models
    from server.services.model_jobs import history_game_logs

    seasons = config.recent_seasons(config.N_TRAIN_SEASONS, demo)

    def fetch_game_logs(seasons_arg=None, report=print):
        df = history_game_logs(seasons_arg or seasons, before=demo)
        report(f"Read {len(df):,} team rows for {', '.join(seasons_arg or seasons)} "
               f"from the history files (demo mode)")
        return df if not df.empty else None

    models.fetch_game_logs = fetch_game_logs
    models.TRAIN_SEASONS = seasons
    return seasons


def metrics_payload(rows: int, metrics: dict, params: dict, seasons: list) -> dict:
    """What model_metrics.json holds (read by /api/model and /api/status)."""
    from server import clock
    from server.services.metrics import calibration_bins, feature_importance
    keep = ("val_acc", "val_loss", "n_train", "n_val", "best_iter", "best_params")
    return {
        "version": "v5",
        "trained_at": clock.iso_utc(),
        "rows": int(rows),
        "seasons": list(seasons),
        "params": params,
        "metrics": {k: metrics.get(k) for k in keep},
        "feature_importance": feature_importance(metrics.get("feat_imp")),
        "calibration": {"bins": calibration_bins(metrics.get("val_proba") or [],
                                                 metrics.get("val_y") or [])},
    }


def run(force_rebuild: bool = False, use_optuna: bool = False, train=None, out=emit) -> int:
    """Train, report, write model_metrics.json. Returns the exit code."""
    from data.atomic import atomic_write_json
    from server import clock, jsonutil, paths

    import config
    seasons = list(config.TRAIN_SEASONS)
    demo = clock.demo_date()
    if demo is not None:
        seasons = use_history_logs(demo)
        if not force_rebuild:
            # a saved dataset may hold games after the demo date; rebuilding from local files is quick
            out({"type": "log", "msg": "Demo mode: rebuilding the dataset from the history files."})
            force_rebuild = True
    if train is None:
        from engine.models import train_model as train

    def status_cb(msg):
        msg = str(msg)
        st = stage_for(msg)
        if st:
            out({"type": "stage", "i": st[0], "n": N_STAGES, "msg": st[1]})
        out({"type": "log", "msg": msg})

    if not force_rebuild and paths.DATASET_FILE.exists():
        out({"type": "log", "msg": "Using the saved dataset (Force full rebuild downloads it again)."})
    try:
        model, rows, metrics = train(force_rebuild=force_rebuild, use_optuna=use_optuna,
                                     status_cb=status_cb)
    except Exception as e:                                   # noqa: BLE001
        for line in traceback.format_exc().splitlines()[-6:]:
            out({"type": "log", "msg": line})
        out({"type": "error", "msg": f"Training failed: {type(e).__name__}: {e}"})
        return 1
    if model is None:
        out({"type": "error", "msg": "Training failed: no game data (check the connection to stats.nba.com)."})
        return 1

    payload = metrics_payload(rows, metrics, {"force_rebuild": force_rebuild, "use_optuna": use_optuna},
                              seasons)
    atomic_write_json(paths.MODEL_METRICS_FILE, jsonutil.clean(payload), indent=2)
    out({"type": "result", "data": jsonutil.clean({"rows": rows, "metrics": payload["metrics"],
                                                   "trained_at": payload["trained_at"]})})
    return 0


def _graceful_term(signum, frame):
    raise SystemExit(128 + signum)        # runs finally blocks: temp files are removed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m server.workers.train")
    ap.add_argument("--force-rebuild", action="store_true", help="download the seasons again")
    ap.add_argument("--optuna", action="store_true", help="tune hyperparameters with Optuna")
    args = ap.parse_args(argv)

    os.chdir(ROOT)                        # config.py paths are relative to the project root
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if hasattr(os, "nice"):
        try:
            os.nice(10)                   # keep the web server responsive
        except OSError:
            pass
    signal.signal(signal.SIGTERM, _graceful_term)
    return run(force_rebuild=args.force_rebuild, use_optuna=args.optuna)


if __name__ == "__main__":
    sys.exit(main())
