"""
Settings and secrets.

Secrets live in `.env` at the project root; process env vars take
precedence and are reported as source "env". Values are never returned to
the browser: callers get `configured` booleans, and security.redact() masks
the values wherever text leaves the server. Business thresholds and Kelly
sizing live in settings.json (defaults below).

M0 ships the read side; the write side (PUT /api/settings, secrets) is M2.
"""
import json
import os

from dotenv import dotenv_values

from server import paths

SECRET_ENV = {
    "odds_api_key":     "ODDS_API_KEY",
    "telegram_token":   "TELEGRAM_BOT_TOKEN",
    "telegram_chat_id": "TELEGRAM_CHAT_ID",
}

DEFAULT_THRESHOLDS = {"pick": 0.55, "safe": 0.65, "vip": 0.65, "value_edge": 0.04}
DEFAULT_KELLY      = {"fraction": 0.25, "cap_pct": 5.0}
DEFAULT_HOST       = "127.0.0.1"
DEFAULT_PORT       = 8000

_env_cache: dict = {"key": None, "values": {}}


def env_file_values() -> dict:
    """Key/values from .env (cached until the file changes)."""
    path = paths.ENV_FILE
    try:
        st = os.stat(path)
    except OSError:
        return {}
    key = (str(path), st.st_mtime_ns, st.st_size)
    if key != _env_cache["key"]:
        values = dotenv_values(path)
        _env_cache.update(key=key, values={k: v for k, v in values.items() if v})
    return _env_cache["values"]


def get_env(name: str) -> tuple:
    """(value, source) for a setting: process env first, then .env; (None, None) if unset."""
    value = os.environ.get(name, "").strip()
    if value:
        return value, "env"
    value = (env_file_values().get(name) or "").strip()
    if value:
        return value, "file"
    return None, None


def secret(key: str) -> str | None:
    """A secret's value by its short key (odds_api_key, telegram_token, telegram_chat_id)."""
    return get_env(SECRET_ENV[key])[0]


def secret_values() -> list:
    """Every configured secret value (for redaction)."""
    return [v for v in (secret(k) for k in SECRET_ENV) if v]


def secrets_status() -> dict:
    """Booleans only: which integrations are configured."""
    return {
        "odds_api": bool(secret("odds_api_key")),
        "telegram": bool(secret("telegram_token") and secret("telegram_chat_id")),
    }


def server_address() -> tuple:
    """(host, port) from NBA_HOST / NBA_PORT, falling back to the defaults."""
    host = get_env("NBA_HOST")[0] or DEFAULT_HOST
    port = get_env("NBA_PORT")[0]
    try:
        port = int(port) if port else DEFAULT_PORT
    except ValueError:
        port = DEFAULT_PORT
    return host, port


def load_settings() -> dict:
    """Thresholds and Kelly sizing: settings.json merged over the defaults."""
    data = {}
    try:
        with open(paths.SETTINGS_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        pass
    thresholds = {**DEFAULT_THRESHOLDS, **{k: v for k, v in (data.get("thresholds") or {}).items()
                                           if k in DEFAULT_THRESHOLDS}}
    kelly = {**DEFAULT_KELLY, **{k: v for k, v in (data.get("kelly") or {}).items()
                                 if k in DEFAULT_KELLY}}
    return {"thresholds": thresholds, "kelly": kelly}
