"""The v5 model: metrics, reload (no refit), delete, and the global ensemble weights.

Importing this module registers the train / autobacktest / player_impact jobs.
"""
import json

from fastapi import APIRouter, Depends

from server import clock, jobs, model_store, paths
from server.errors import ApiError
from server.jsonutil import r4
from server.routers import meta
from server.security import require_loopback
from server.services import model_jobs  # noqa: F401  (registers the model job kinds)

router = APIRouter(prefix="/api/model", tags=["model"])
ORDER = 30

COMPONENTS = [("elo", "Elo", 2), ("mc", "Monte Carlo", 1), ("xgb", "XGBoost", 0)]   # ladder order
X0 = {"xgb": 0.5, "mc": 0.5, "elo": 0.0}


def _read_json(path) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def require_confirm(confirm: bool = False):
    if not confirm:
        raise ApiError("confirm_required", "This deletes a file. Repeat the request with ?confirm=true.")


@router.get("")
def get_model():
    """Model status plus the metrics, feature importance and calibration saved by the train job."""
    info = meta.model_info()
    saved = _read_json(paths.MODEL_METRICS_FILE) if info["exists"] else {}
    importance = saved.get("feature_importance") or []
    if info["exists"] and not importance:
        importance = _importance_from_model()
    return {
        "exists": info["exists"], "version": info["version"], "trained_at": info["trained_at"],
        "age_days": info["age_days"], "rows": saved.get("rows"), "seasons": saved.get("seasons"),
        "params": saved.get("params"), "metrics": saved.get("metrics", {}) if info["exists"] else None,
        "feature_importance": importance,
        "calibration": saved.get("calibration") or {"bins": []},
        "has_saved_metrics": bool(saved),
        "train_eta_s": model_jobs.TRAIN_TYPICAL_S,
        "train_stages": model_jobs.TRAIN_STAGES,
    }


def _importance_from_model() -> list:
    """Feature importance straight from the loaded model (no metrics file yet)."""
    from engine.models import get_feature_importance
    from server.services.metrics import feature_importance
    try:
        model = model_store.get_model()
    except Exception:                               # noqa: BLE001  (unreadable model file)
        return []
    return feature_importance(get_feature_importance(model)) if model is not None else []


@router.post("/reload")
def reload_model():
    """Load nba_brain_v5.joblib again, with no refit."""
    if not paths.MODEL_FILE.exists():
        raise ApiError("no_model", "There is no model file to load.", hint="Train a model first.")
    try:
        model_store.reload()
    except Exception as e:                          # noqa: BLE001
        raise ApiError("internal", f"Could not load the model: {type(e).__name__}: {e}") from None
    info = meta.model_info()
    return {"trained_at": info["trained_at"], "loaded_at": clock.iso_utc()}


@router.delete("", dependencies=[Depends(require_loopback), Depends(require_confirm)])
def delete_model():
    """Remove the model file (and its saved metrics). Takes the writer lock."""
    with jobs.manager().writer_lock("Deleting the model"):
        removed = _unlink(paths.MODEL_FILE)
        _unlink(paths.MODEL_METRICS_FILE)
    model_store.get_model()                         # drops the cached model
    return {"ok": True, "removed": removed, "file": paths.MODEL_FILE.name}


def _unlink(path) -> bool:
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False


# ── global ensemble weights ───────────────────────────────────────────────────

_contrib_cache: dict = {}


def _mean_abs_contributions(weights: dict) -> dict | None:
    """Mean |w_i (x_i − x0_i)| over graded auto-backtest games (current weights)."""
    path = paths.AUTOBACKTEST_CSV
    try:
        st = path.stat()
    except OSError:
        return None
    key = (st.st_mtime_ns, st.st_size, tuple(sorted(weights.items())))
    if _contrib_cache.get("key") != key:
        import pandas as pd
        try:
            df = pd.read_csv(path, usecols=["XGB_P", "MC_P", "ELO_DIFF"]).dropna()
        except (OSError, ValueError):
            return None
        if df.empty:
            return None
        x = {"xgb": df["XGB_P"], "mc": df["MC_P"], "elo": df["ELO_DIFF"] / 400.0}
        _contrib_cache.update(key=key, value={
            k: r4((weights[k] * (x[k] - X0[k])).abs().mean()) for k in X0})
    return _contrib_cache["value"]


@router.get("/ensemble")
def ensemble():
    """How the final probability is combined: blend_v5 (calibrator) or elo_only (no model)."""
    from engine.models import get_calibrator
    cal = get_calibrator()
    w = {"xgb": float(cal.weights[0]), "mc": float(cal.weights[1]), "elo": float(cal.weights[2])}
    calibrator = {"weights": {k: r4(v) for k, v in w.items()}, "bias": r4(cal.bias),
                  "n_updates": cal.n_updates}
    updated_at = clock.iso_mtime(paths.BLEND_FILE)
    if not paths.MODEL_FILE.exists():
        return {"method": "elo_only", "bias": 0.0, "n_updates": cal.n_updates, "updated_at": updated_at,
                "components": [{"key": "elo", "label": "Elo", "coef": 1.0,
                                "mean_abs_contribution": None, "oos_log_loss": None}],
                "calibrator": calibrator}
    contrib = _mean_abs_contributions(w) or {}
    return {
        "method": "blend_v5", "bias": r4(cal.bias), "n_updates": cal.n_updates, "updated_at": updated_at,
        "components": [{"key": key, "label": label, "coef": r4(w[key]),
                        "mean_abs_contribution": contrib.get(key), "oos_log_loss": None}
                       for key, label, _ in COMPONENTS],
        "calibrator": calibrator,
    }
