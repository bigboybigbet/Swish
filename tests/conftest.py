"""Shared fixtures for the web app tests.

`client` is a TestClient for a fresh app bound to 127.0.0.1:8000 (so the
Host guard accepts it) whose requests come from a loopback client address.
`isolated` points .env / settings.json at a temp dir and clears secret and
demo env vars, so a developer's real keys never leak into (or break) tests.
"""
import pytest

BASE_URL = "http://127.0.0.1:8000"
SECRET_VARS = ("ODDS_API_KEY", "TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID")


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    from server import paths
    monkeypatch.setattr(paths, "ENV_FILE", tmp_path / ".env")
    monkeypatch.setattr(paths, "SETTINGS_FILE", tmp_path / "settings.json")
    for var in (*SECRET_VARS, "NBA_QUANT_DEMO_DATE", "NBA_HOST", "NBA_PORT"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


@pytest.fixture
def app(isolated):
    from server.app import create_app
    return create_app(host="127.0.0.1", port=8000)


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient
    with TestClient(app, base_url=BASE_URL, client=("127.0.0.1", 50000)) as c:
        yield c
