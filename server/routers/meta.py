"""GET /api/health and GET /api/status."""
import importlib
import importlib.util
import json
from datetime import datetime

from fastapi import APIRouter

import config
from server import __version__, clock, config_store, paths

router = APIRouter(prefix="/api", tags=["meta"])
ORDER = 10


@router.get("/health")
def health():
    return {"ok": True, "version": __version__}


@router.get("/status")
def status():
    today = clock.today_et()
    settings = config_store.load_settings()
    model = model_info()
    demo = clock.demo_date()
    return {
        "version": __version__,
        "server_time_utc": clock.iso_utc(),
        "today_et": today.isoformat(),
        "demo": {"date": demo.isoformat()} if demo else None,
        "model": model,
        "dataset": dataset_info(),
        "elo": elo_info(),
        "calibrator": calibrator_info(),
        "autobacktest": autobacktest_info(),
        "player_impact": player_impact_info(),
        "cache": cache_info(),
        "capabilities": capabilities(model["exists"]),
        "secrets": config_store.secrets_status(),
        "thresholds": settings["thresholds"],
        "kelly": settings["kelly"],
        "seasons": {"train": list(config.TRAIN_SEASONS),
                    "current": config.current_season(today)},
        "active_jobs": active_jobs(),
    }


# ── pieces (also used by other routers) ──────────────────────────────────────

def _read_json(path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def model_info() -> dict:
    """v5 model file + metrics saved by the train job (model_metrics.json)."""
    exists = paths.MODEL_FILE.exists()
    saved = (_read_json(paths.MODEL_METRICS_FILE) or {}) if exists else {}
    metrics = saved.get("metrics", saved)
    trained_at = saved.get("trained_at") or (clock.iso_mtime(paths.MODEL_FILE) if exists else None)
    age_days = None
    if trained_at:
        try:
            t = datetime.fromisoformat(trained_at.replace("Z", "+00:00"))
            age_days = round((clock.now_utc() - t).total_seconds() / 86400, 1)
        except ValueError:
            pass
    return {
        "exists": exists, "version": "v5", "trained_at": trained_at,
        "val_acc": metrics.get("val_acc"), "val_loss": metrics.get("val_loss"),
        "n_train": metrics.get("n_train"), "age_days": age_days,
    }


_rows_cache: dict = {}


def dataset_info() -> dict:
    path = paths.DATASET_FILE
    if not path.exists():
        return {"exists": False, "rows": None, "mtime": None}
    st = path.stat()
    key = (st.st_mtime, st.st_size)
    if _rows_cache.get("key") != key:
        with open(path, "rb") as f:
            lines = sum(1 for _ in f)
        _rows_cache.update(key=key, rows=max(lines - 1, 0))
    return {"exists": True, "rows": _rows_cache["rows"], "mtime": clock.iso_mtime(path)}


def elo_info() -> dict:
    data = _read_json(paths.ELO_FILE)
    return {"exists": data is not None,
            "n_teams": len(data) if isinstance(data, dict) else None,
            "mtime": clock.iso_mtime(paths.ELO_FILE)}


def calibrator_info() -> dict:
    """Blend calibrator weights with API keys (xgb, mc, elo)."""
    from engine.models import get_calibrator
    s = get_calibrator().summary()
    w = s["weights"]
    return {"weights": {"xgb": w["XGBoost"], "mc": w["Monte Carlo"], "elo": w["ELO diff"]},
            "bias": s["bias"], "n_updates": s["n_updates"]}


def autobacktest_info() -> dict:
    data = _read_json(paths.AUTOBACKTEST_LOG) or {}
    return {"total": int(data.get("total", 0))}


def player_impact_info() -> dict:
    from data.injuries import load_player_impact_meta
    meta = load_player_impact_meta()
    count = meta.get("count") or 0
    fallback = not paths.PLAYER_IMPACT_FILE.exists() or count == 0
    return {
        "season": None if meta.get("season") in (None, "—") else meta["season"],
        "updated": None if meta.get("updated") in (None, "—") else meta["updated"],
        "count": count if not fallback else len(config.PLAYER_IMPACT_FALLBACK),
        "fallback": fallback,
    }


def cache_info() -> dict:
    from data import cache
    return cache.stats()


def has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def ratings_available() -> dict:
    """engine.ratings.available() once the v6 engine exists; all False until then."""
    avail = {"kalman": False, "adjusted": False, "players": False, "stacked": False}
    if has_module("engine.ratings"):
        ratings = importlib.import_module("engine.ratings")
        avail.update({k: bool(v) for k, v in ratings.available().items() if k in avail})
    return avail


def capabilities(model_exists: bool) -> dict:
    r = ratings_available()
    return {
        "predict_mode": "blend_v5" if model_exists else "elo_only",
        "shap": has_module("shap"),
        "optuna": has_module("optuna"),
        "ratings": {"elo": True, "kalman": r["kalman"], "adjusted": r["adjusted"],
                    "players": r["players"]},
        "stacked_ensemble": r["stacked"],
    }


def active_jobs() -> list:
    """[{id, kind, progress, stage}] from server.jobs (M2) once it exists."""
    if not has_module("server.jobs"):
        return []
    jobs = importlib.import_module("server.jobs")
    fn = getattr(jobs, "active_summary", None)
    return list(fn()) if fn else []
