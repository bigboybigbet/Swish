"""
Model-side background jobs (all writers, one at a time):

  train          subprocess `python -m server.workers.train`; JSON-line stages;
                 writes model_metrics.json; the server reloads the model after
  autobacktest   thread; grades new finished games and updates Elo and the
                 blend calibrator (needs a trained model)
  player_impact  subprocess `player_impact_updater.py` with cwd=ROOT, 120 s timeout

Registered with server.jobs when this module is imported (routers/model.py
imports it). Demo mode (NBA_QUANT_DEMO_DATE) reads game logs from the
2010-24 history files instead of stats.nba.com.
"""
import json
import re
import sys
from contextlib import contextmanager

from pydantic import BaseModel, ConfigDict, Field, field_validator

import config
from server import clock, jobs, model_store, paths
from server.routers.meta import has_module

TRAIN_STAGES = ["Download", "Pair games", "Elo", "Search", "Fit", "Calibrate", "Save"]
TRAIN_TYPICAL_S = 594          # measured on 4 cores (README benchmark)
PLAYER_IMPACT_TIMEOUT_S = 120


# ── history game logs (demo mode) ─────────────────────────────────────────────

def history_game_logs(seasons: list | None = None, before=None):
    """Team game logs (leaguegamefinder columns) from NBA_HISTORY_DIR, before a date."""
    import pandas as pd
    from research.benchmark import load_history
    raw = load_history()
    if seasons:
        raw = raw[raw["SEASON_YEAR"].isin(seasons)]
    if before is not None:
        raw = raw[raw["GAME_DATE"] < pd.Timestamp(before)]
    return raw.reset_index(drop=True)


class _HistoryFinder:
    """Stands in for nba_api's leaguegamefinder module in demo mode."""

    def __init__(self, before):
        self.before = before

    def LeagueGameFinder(self, season_nullable=None, **_):      # noqa: N802 (nba_api's name)
        df = history_game_logs([season_nullable] if season_nullable else None, self.before)

        class _Result:
            def get_data_frames(self):
                return [df]
        return _Result()


@contextmanager
def demo_autobacktest_source():
    """In demo mode, auto-backtest grades history games before the demo date."""
    demo = clock.demo_date()
    if demo is None:
        yield
        return
    import data.autobacktest as ab
    saved = ab.leaguegamefinder, ab.SEASONS
    ab.leaguegamefinder = _HistoryFinder(demo)
    ab.SEASONS = config.recent_seasons(2, demo)
    try:
        yield
    finally:
        ab.leaguegamefinder, ab.SEASONS = saved


# ── train ─────────────────────────────────────────────────────────────────────

class TrainParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    force_rebuild: bool = False
    use_optuna: bool = False

    @field_validator("use_optuna")
    @classmethod
    def _optuna_installed(cls, v):
        if v and not has_module("optuna"):
            raise ValueError("Optuna is not installed (pip install optuna)")
        return v


def train_command(params: dict) -> list:
    cmd = [sys.executable, "-m", "server.workers.train"]
    if params.get("force_rebuild"):
        cmd.append("--force-rebuild")
    if params.get("use_optuna"):
        cmd.append("--optuna")
    return cmd


def run_train(ctx: jobs.JobContext):
    n = len(TRAIN_STAGES)
    ctx.progress(done=0, total=n, stage="Starting the training process…", progress=None)
    result, errors = {}, []

    def on_line(line: str):
        try:
            msg = json.loads(line)
        except ValueError:
            msg = None
        if not isinstance(msg, dict) or "type" not in msg:
            ctx.log(line)                      # library output, warnings, tracebacks
            return
        kind = msg["type"]
        if kind == "stage":
            ctx.progress(done=msg.get("i"), total=msg.get("n") or n,
                         stage=msg.get("msg") or "", progress=None)
        elif kind == "log":
            ctx.log(msg.get("msg", ""))
        elif kind == "result":
            result.update(msg.get("data") or {})
        elif kind == "error":
            errors.append(str(msg.get("msg") or "Training failed"))
            ctx.log(errors[-1])

    rc = ctx.run_subprocess(train_command(ctx.params), on_line=on_line)
    if rc != 0 or not result:
        raise RuntimeError(errors[-1] if errors else f"The training process exited with code {rc}.")
    ctx.progress(stage="Loading the new model…", progress=None)
    model_store.reload()
    return result


# ── auto-backtest ─────────────────────────────────────────────────────────────

_COUNT = re.compile(r"(\d+)/(\d+)")


class NoParams(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _calibrator_keys(summary: dict | None) -> dict | None:
    if not summary:
        return None
    w = summary.get("weights") or {}
    return {"weights": {"xgb": w.get("XGBoost"), "mc": w.get("Monte Carlo"), "elo": w.get("ELO diff")},
            "bias": summary.get("bias"), "n_updates": summary.get("n_updates")}


def run_autobacktest(ctx: jobs.JobContext):
    from data.autobacktest import run_auto_backtest
    model = model_store.get_model()
    if model is None:
        raise RuntimeError("No trained model: train one on the Model page first.")
    ctx.progress(stage="Looking for finished games…", progress=None)

    def on_status(msg: str):
        ctx.log(msg)
        m = _COUNT.search(msg)
        if m and int(m.group(2)):
            ctx.progress(done=int(m.group(1)), total=int(m.group(2)))
        elif msg.startswith("Fetching"):
            ctx.progress(stage=msg.strip(), progress=None)

    def on_progress(done: int, total: int):
        ctx.progress(done=done, total=total, stage=f"Grading game {done + 1} of {total}")

    with demo_autobacktest_source():
        out = run_auto_backtest(model, status_cb=on_status,
                                should_stop=lambda: ctx.cancelled, progress_cb=on_progress)
    return {"new": out.get("new", 0), "correct": out.get("correct", 0), "acc": out.get("acc"),
            "calibrator": _calibrator_keys(out.get("calibrator")),
            "stopped": bool(out.get("stopped"))}


# ── player impact ─────────────────────────────────────────────────────────────

class PlayerImpactParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    season: str | None = Field(None, pattern=r"^\d{4}-\d{2}$", description="e.g. 2025-26")


def run_player_impact(ctx: jobs.JobContext):
    if clock.is_demo():
        raise RuntimeError("Player impact needs a live connection to stats.nba.com, "
                           "so it is not available in demo mode.")
    season = ctx.params.get("season")
    ctx.progress(stage=f"Fetching {season or 'current'} season stats from NBA.com…", progress=None)
    cmd = [sys.executable, str(paths.PLAYER_IMPACT_UPDATER)] + ([season] if season else [])
    rc = ctx.run_subprocess(cmd, cwd=paths.ROOT, timeout=PLAYER_IMPACT_TIMEOUT_S)
    if rc != 0:
        last = next((line for line in reversed(ctx.job.log) if line.strip()), "")
        raise RuntimeError("Update failed (check the connection to stats.nba.com)"
                           + (f": {last}" if last else "."))
    from server.routers.meta import player_impact_info
    info = player_impact_info()
    return {"season": info["season"], "updated": info["updated"], "count": info["count"]}


# ── registration ──────────────────────────────────────────────────────────────

jobs.register_kind(jobs.JobKind("train", run_train, writer=True, params=TrainParams,
                                label="Training"))
jobs.register_kind(jobs.JobKind("autobacktest", run_autobacktest, writer=True, params=NoParams,
                                requires_model=True, label="Auto-backtest"))
jobs.register_kind(jobs.JobKind("player_impact", run_player_impact, writer=True,
                                params=PlayerImpactParams, label="Player impact update"))
