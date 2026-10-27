"""
JSON helpers: numpy/pandas → plain Python, NaN/inf → None, rounding.

Routes may return numpy scalars or arrays; pass them through clean() (the
default response class does this too, but FastAPI's own encoder runs first
and rejects numpy ints, so call clean() on anything that came from pandas).
"""
import json
import math
from datetime import date, datetime
from pathlib import PurePath

import numpy as np
from fastapi.responses import JSONResponse


def clean(obj):
    """Recursively convert to JSON-safe Python; non-finite floats become None."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if type(obj).__name__ in ("NAType", "NaTType"):   # pandas missing values (NaT is a datetime)
        return None
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [clean(v) for v in obj.tolist()]
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, PurePath):
        return str(obj)
    if hasattr(obj, "isoformat"):     # e.g. pandas Timestamp, without importing pandas
        return obj.isoformat()
    if hasattr(obj, "item"):          # other numpy scalars
        return clean(obj.item())
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def rnd(x, ndigits: int):
    """Round a number, or return None for None/NaN/inf."""
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return round(f, ndigits) if math.isfinite(f) else None


def r4(x):
    """Probabilities: 4 decimal places."""
    return rnd(x, 4)


def r1(x):
    """Points: 1 decimal place."""
    return rnd(x, 1)


def dumps(obj) -> str:
    return json.dumps(clean(obj), ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"))


class SafeJSONResponse(JSONResponse):
    """Default response class: never emits NaN/Infinity."""

    def render(self, content) -> bytes:
        return dumps(content).encode("utf-8")
