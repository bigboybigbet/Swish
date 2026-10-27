"""
The v5 XGBoost model, loaded once and reloaded when the file changes, plus a
cached SHAP explainer.

get_model() returns None when nba_brain_v5.joblib is missing; predictions
then run in elo_only mode. M2 adds an explicit reload (POST /api/model/reload).
"""
import logging
import threading

from joblib import load

from config import FEATURE_COLS
from server import paths
from server.jsonutil import r4

log = logging.getLogger("server")

_lock = threading.Lock()
_state = {"key": None, "model": None, "explainer": None}


def get_model():
    """The trained model, or None if there is no model file."""
    path = paths.MODEL_FILE
    try:
        st = path.stat()
    except OSError:
        with _lock:
            _state.update(key=None, model=None, explainer=None)
        return None
    key = (str(path), st.st_mtime_ns, st.st_size)
    with _lock:
        if _state["key"] != key:
            log.info("Loading model %s", path.name)
            _state.update(model=load(path), key=key, explainer=None)
        return _state["model"]


def _explainer(model):
    """SHAP TreeExplainer for the raw XGBoost inside the calibrated model (cached)."""
    with _lock:
        if _state["explainer"] is not None and _state["model"] is model:
            return _state["explainer"]
    import shap
    from engine.models import unwrap_xgb
    explainer = shap.TreeExplainer(unwrap_xgb(model))
    with _lock:
        if _state["model"] is model:
            _state["explainer"] = explainer
    return explainer


def shap_explain(model, row) -> dict | None:
    """{base_value, features:[{name, value, shap}]} for one matchup row, largest |shap| first.

    Values are in the raw XGBoost log-odds space. None without shap or a model.
    """
    if model is None:
        return None
    try:
        explainer = _explainer(model)
    except ImportError:
        return None
    sv = explainer.shap_values(row)
    if isinstance(sv, list):              # older shap: one array per class
        sv = sv[-1]
    values = sv[0] if getattr(sv, "ndim", 1) > 1 else sv
    base = explainer.expected_value
    try:
        base = float(base)
    except TypeError:
        base = float(list(base)[-1])
    feats = [{"name": name, "value": r4(row.iloc[0][name]), "shap": r4(values[i])}
             for i, name in enumerate(FEATURE_COLS)]
    feats.sort(key=lambda f: abs(f["shap"] or 0), reverse=True)
    return {"base_value": r4(base), "features": feats}


def reload():
    """Load the model file again now, with no refit (POST /api/model/reload).

    Returns the model, or None when there is no model file.
    """
    with _lock:
        _state.update(key=None, model=None, explainer=None)
    return get_model()
