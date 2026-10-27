"""Settings (thresholds, Kelly), secrets in .env (write-only) and the Telegram test."""
import platform
from importlib import metadata

from fastapi import APIRouter, Body, Depends, Request
from pydantic import BaseModel, ConfigDict, Field

from server import config_store, paths
from server.errors import ApiError
from server.security import require_loopback
from server.services import telegram

router = APIRouter(prefix="/api", tags=["settings"])
ORDER = 40

# label → distribution name
PACKAGES = {"fastapi": "fastapi", "uvicorn": "uvicorn", "pandas": "pandas", "numpy": "numpy",
            "xgboost": "xgboost", "sklearn": "scikit-learn", "nba_api": "nba_api",
            "shap": "shap", "optuna": "optuna"}


class ThresholdsIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pick: float | None = Field(None, ge=0.5, le=0.9)
    safe: float | None = Field(None, ge=0.5, le=0.9)
    vip: float | None = Field(None, ge=0.5, le=0.9)
    value_edge: float | None = Field(None, ge=0.0, le=0.2)


class KellyIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fraction: float | None = Field(None, ge=0.0, le=1.0)
    cap_pct: float | None = Field(None, ge=0.0, le=25.0)


class SettingsIn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    thresholds: ThresholdsIn | None = None
    kelly: KellyIn | None = None


class SecretsIn(BaseModel):
    """Absent or null: unchanged. Empty string: removed from .env."""
    model_config = ConfigDict(extra="forbid")
    odds_api_key: str | None = Field(None, max_length=config_store.SECRET_MAX_LEN)
    telegram_token: str | None = Field(None, max_length=config_store.SECRET_MAX_LEN)
    telegram_chat_id: str | None = Field(None, max_length=config_store.SECRET_MAX_LEN)


def package_versions() -> dict:
    out = {}
    for label, dist in PACKAGES.items():
        try:
            out[label] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out[label] = None
    return out


def settings_payload(request: Request) -> dict:
    host, port = request.app.state.bind
    settings = config_store.load_settings()
    return {
        "secrets": config_store.secrets_detail(),
        "thresholds": settings["thresholds"],
        "kelly": settings["kelly"],
        "limits": {"thresholds": config_store.THRESHOLD_LIMITS, "kelly": config_store.KELLY_LIMITS},
        "defaults": {"thresholds": config_store.DEFAULT_THRESHOLDS, "kelly": config_store.DEFAULT_KELLY},
        "files": {"model": paths.MODEL_FILE.name, "dataset": paths.DATASET_FILE.name,
                  "elo": paths.ELO_FILE.name, "cache": paths.CACHE_DB.name, "env": paths.ENV_FILE.name},
        "server": {"host": host, "port": port, "root": str(paths.ROOT),
                   "python": platform.python_version(), "packages": package_versions()},
    }


@router.get("/settings")
def get_settings(request: Request):
    return settings_payload(request)


@router.put("/settings")
def put_settings(body: SettingsIn, request: Request):
    """Thresholds 0.5–0.9 (safe and vip ≥ pick), value_edge 0–0.2, fraction 0–1, cap 0–25."""
    try:
        config_store.save_settings(
            body.thresholds.model_dump(exclude_none=True) if body.thresholds else None,
            body.kelly.model_dump(exclude_none=True) if body.kelly else None)
    except ValueError as e:
        raise ApiError("validation", str(e)) from None
    return settings_payload(request)


@router.put("/settings/secrets", dependencies=[Depends(require_loopback)])
def put_secrets(body: SecretsIn, request: Request):
    """Write keys to .env (mode 0600). Values are never returned."""
    try:
        config_store.set_secrets(body.model_dump(exclude_none=True))
    except ValueError as e:
        raise ApiError("validation", str(e)) from None
    return settings_payload(request)


@router.post("/telegram/test")
def telegram_test(_body: dict | None = Body(default=None)):
    """Send a test message with the configured bot: {ok, detail} (redacted)."""
    ok, detail = telegram.send_test()
    return {"ok": ok, "detail": detail}
