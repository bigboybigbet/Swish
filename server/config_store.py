"""
Settings and secrets.

Secrets live in `.env` at the project root; process env vars take
precedence and are reported as source "env". Values are never returned to
the browser: callers get `configured` booleans, and security.redact() masks
the values wherever text leaves the server. Business thresholds and Kelly
sizing live in settings.json (defaults below).

Writes: save_settings() validates and writes settings.json atomically;
set_secrets() writes .env with python-dotenv and keeps it at mode 0600.
"""
import json
import os
import threading

from dotenv import dotenv_values, set_key, unset_key

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

# Allowed ranges (inclusive) for PUT /api/settings.
THRESHOLD_LIMITS = {"pick": (0.5, 0.9), "safe": (0.5, 0.9), "vip": (0.5, 0.9), "value_edge": (0.0, 0.2)}
KELLY_LIMITS     = {"fraction": (0.0, 1.0), "cap_pct": (0.0, 25.0)}
SECRET_MAX_LEN   = 512
MASK             = "•••"

_write_lock = threading.Lock()

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


# ── writes (M2) ───────────────────────────────────────────────────────────────

def _read_settings_file() -> dict:
    try:
        with open(paths.SETTINGS_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def validate_settings(thresholds: dict, kelly: dict):
    """Raise ValueError unless every value is in range and the tiers are ordered."""
    for group, limits in ((thresholds, THRESHOLD_LIMITS), (kelly, KELLY_LIMITS)):
        for key, (lo, hi) in limits.items():
            v = group.get(key)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
                raise ValueError(f"{key} must be a number from {lo:g} to {hi:g}")
    if thresholds["safe"] < thresholds["pick"]:
        raise ValueError("safe must be at least pick")
    if thresholds["vip"] < thresholds["pick"]:
        raise ValueError("vip must be at least pick")


def save_settings(thresholds: dict | None = None, kelly: dict | None = None) -> dict:
    """Merge the given values over the current settings, validate, write settings.json."""
    from data.atomic import atomic_write_json
    with _write_lock:
        current = load_settings()
        new_t = {**current["thresholds"], **(thresholds or {})}
        new_k = {**current["kelly"], **(kelly or {})}
        validate_settings(new_t, new_k)
        data = _read_settings_file()            # keep keys other milestones may add
        data.update(thresholds={k: float(new_t[k]) for k in DEFAULT_THRESHOLDS},
                    kelly={k: float(new_k[k]) for k in DEFAULT_KELLY})
        atomic_write_json(paths.SETTINGS_FILE, data, indent=2)
    return load_settings()


def mask_tail(value: str | None, keep: int = 4) -> str | None:
    """'•••4821' for a configured value, None when unset."""
    if not value:
        return None
    return MASK + (value[-keep:] if len(value) > keep + 2 else "")


def secrets_detail() -> dict:
    """GET /api/settings view of the secrets: configured flags and sources, never values."""
    odds, odds_src = get_env(SECRET_ENV["odds_api_key"])
    token, token_src = get_env(SECRET_ENV["telegram_token"])
    chat, chat_src = get_env(SECRET_ENV["telegram_chat_id"])
    return {
        "odds_api": {"configured": bool(odds), "source": odds_src},
        "telegram": {"configured": bool(token and chat), "source": token_src or chat_src,
                     "token_configured": bool(token), "token_source": token_src,
                     "chat_id_configured": bool(chat), "chat_id_source": chat_src,
                     "chat_id_masked": mask_tail(chat)},
    }


def _private_file(path):
    """Create `path` with mode 0600 if missing; tighten it to 0600 on POSIX."""
    if not path.exists():
        fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o600)
        os.close(fd)
    if os.name == "posix":
        os.chmod(path, 0o600)


def set_secrets(values: dict):
    """Write secrets to .env. values: short key → str; "" removes the key; others untouched.

    Process environment variables still take precedence over .env.
    """
    clean = {}
    for key, value in values.items():
        if key not in SECRET_ENV:
            raise ValueError(f"unknown secret {key!r}")
        if value is None:
            continue
        value = str(value).strip()
        if len(value) > SECRET_MAX_LEN or any(c in value for c in "\r\n\0"):
            raise ValueError(f"{key} must be a single line of at most {SECRET_MAX_LEN} characters")
        clean[key] = value
    if not clean:
        return
    path = paths.ENV_FILE
    with _write_lock:
        _private_file(path)
        present = dotenv_values(path)
        for key, value in clean.items():
            name = SECRET_ENV[key]
            if value:
                set_key(path, name, value, quote_mode="always")
            elif name in present:
                unset_key(path, name, quote_mode="always")
        _private_file(path)
        _env_cache["key"] = None
