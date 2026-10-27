"""M2 settings: thresholds/Kelly validation, write-only secrets in .env (0600), Telegram test, redaction."""
import json
import os
import stat

import pytest
import requests
from fastapi.testclient import TestClient

from server import config_store
from tests.conftest import BASE_URL

ODDS_KEY = "odds-0123456789abcdef"
TOKEN = "123456789:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw"
CHAT_ID = "-1001234564821"


def test_get_settings_shape(client):
    d = client.get("/api/settings").json()
    assert d["secrets"]["odds_api"] == {"configured": False, "source": None}
    tg = d["secrets"]["telegram"]
    assert tg["configured"] is False and tg["source"] is None and tg["chat_id_masked"] is None
    assert d["thresholds"] == {"pick": 0.55, "safe": 0.65, "vip": 0.65, "value_edge": 0.04}
    assert d["kelly"] == {"fraction": 0.25, "cap_pct": 5.0}
    server = d["server"]
    assert server["host"] == "127.0.0.1" and server["port"] == 8000 and server["root"]
    assert server["python"].count(".") == 2
    assert {"fastapi", "xgboost", "sklearn", "nba_api", "shap", "optuna"} <= set(server["packages"])
    assert server["packages"]["fastapi"]


def test_put_settings_validates_and_persists(client, isolated):
    r = client.put("/api/settings", json={"thresholds": {"pick": 0.6, "safe": 0.7}, "kelly": {"cap_pct": 2.5}})
    assert r.status_code == 200
    assert r.json()["thresholds"] == {"pick": 0.6, "safe": 0.7, "vip": 0.65, "value_edge": 0.04}
    saved = json.loads((isolated / "settings.json").read_text())
    assert saved["kelly"] == {"fraction": 0.25, "cap_pct": 2.5}
    status = client.get("/api/status").json()
    assert status["thresholds"]["pick"] == 0.6 and status["kelly"]["cap_pct"] == 2.5


@pytest.mark.parametrize("body", [
    {"thresholds": {"pick": 0.95}},
    {"thresholds": {"value_edge": 0.3}},
    {"kelly": {"fraction": 1.5}},
    {"kelly": {"cap_pct": 30}},
    {"thresholds": {"pick": 0.7, "safe": 0.6}},       # safe below pick
    {"thresholds": {"pick": "high"}},
    {"other": 1},
])
def test_put_settings_rejects_bad_values(client, isolated, body):
    r = client.put("/api/settings", json=body)
    assert r.status_code == 422 and r.json()["error"] == "validation"
    assert not (isolated / "settings.json").exists()


def test_secrets_are_written_0600_and_never_returned(client, isolated):
    r = client.put("/api/settings/secrets", json={"odds_api_key": ODDS_KEY, "telegram_token": TOKEN,
                                                  "telegram_chat_id": CHAT_ID})
    assert r.status_code == 200
    env = isolated / ".env"
    if os.name == "posix":
        assert stat.S_IMODE(env.stat().st_mode) == 0o600
    text = env.read_text()
    assert ODDS_KEY in text and TOKEN in text
    for resp in (r, client.get("/api/settings"), client.get("/api/status")):
        raw = resp.text
        assert ODDS_KEY not in raw and TOKEN not in raw and CHAT_ID not in raw
    s = client.get("/api/settings").json()["secrets"]
    assert s["odds_api"] == {"configured": True, "source": "file"}
    assert s["telegram"]["configured"] is True and s["telegram"]["source"] == "file"
    assert s["telegram"]["chat_id_masked"] == "•••4821"
    assert config_store.secret("odds_api_key") == ODDS_KEY


def test_empty_string_clears_and_absent_keys_are_kept(client, isolated):
    client.put("/api/settings/secrets", json={"odds_api_key": ODDS_KEY, "telegram_chat_id": CHAT_ID})
    r = client.put("/api/settings/secrets", json={"odds_api_key": ""})
    assert r.json()["secrets"]["odds_api"] == {"configured": False, "source": None}
    assert r.json()["secrets"]["telegram"]["chat_id_configured"] is True
    assert "ODDS_API_KEY" not in (isolated / ".env").read_text()


def test_env_var_takes_precedence(client, monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", ODDS_KEY)
    client.put("/api/settings/secrets", json={"odds_api_key": "file-value-123456"})
    s = client.get("/api/settings").json()["secrets"]
    assert s["odds_api"] == {"configured": True, "source": "env"}
    assert config_store.secret("odds_api_key") == ODDS_KEY


def test_existing_env_file_is_tightened_to_0600(client, isolated):
    env = isolated / ".env"
    env.write_text("NBA_PORT=8000\n")
    os.chmod(env, 0o644)
    client.put("/api/settings/secrets", json={"telegram_chat_id": CHAT_ID})
    if os.name == "posix":
        assert stat.S_IMODE(env.stat().st_mode) == 0o600
    assert "NBA_PORT=8000" in env.read_text()


def test_secrets_endpoint_is_loopback_and_same_origin_only(app, client, isolated):
    remote = TestClient(app, base_url=BASE_URL, client=("10.0.0.7", 1))
    r = remote.put("/api/settings/secrets", json={"odds_api_key": ODDS_KEY})
    assert r.status_code == 403 and r.json()["error"] == "loopback_only"
    r = client.put("/api/settings/secrets", json={"odds_api_key": ODDS_KEY},
                   headers={"Origin": "http://evil.example"})
    assert r.status_code == 403 and r.json()["error"] == "forbidden_origin"
    assert not (isolated / ".env").exists()


@pytest.mark.parametrize("body", [{"odds_api_key": "abc\nTELEGRAM_CHAT_ID=1"}, {"odds_api_key": "x" * 600},
                                  {"unknown_key": "x"}])
def test_bad_secret_values_are_rejected(client, isolated, body):
    r = client.put("/api/settings/secrets", json=body)
    assert r.status_code == 422
    assert not (isolated / ".env").exists() or "TELEGRAM_CHAT_ID" not in (isolated / ".env").read_text()


# ── Telegram ──────────────────────────────────────────────────────────────────

class _Resp:
    def __init__(self, status, body):
        self.status_code, self._body, self.ok = status, body, status < 400

    def json(self):
        return self._body


def test_telegram_test_not_configured(client):
    assert client.post("/api/telegram/test", json={}).json() == {
        "ok": False, "detail": "Telegram is not configured: add the bot token and chat ID in Settings."}


def test_telegram_test_sends_with_configured_bot(client, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", CHAT_ID)
    sent = {}

    def fake_post(url, json=None, timeout=None):
        sent.update(url=url, json=json, timeout=timeout)
        return _Resp(200, {"ok": True, "result": {}})

    monkeypatch.setattr(requests, "post", fake_post)
    assert client.post("/api/telegram/test", json={}).json() == {"ok": True, "detail": "Message sent."}
    assert sent["url"].endswith(f"/bot{TOKEN}/sendMessage") and sent["json"]["chat_id"] == CHAT_ID
    assert sent["json"]["parse_mode"] == "HTML" and sent["timeout"] == 8


def test_telegram_errors_are_redacted(client, monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", CHAT_ID)

    def failing_post(url, json=None, timeout=None):
        raise requests.ConnectionError(f"HTTPSConnectionPool: Max retries exceeded with url: {url}")

    monkeypatch.setattr(requests, "post", failing_post)
    d = client.post("/api/telegram/test", json={}).json()
    assert d["ok"] is False and "Could not reach Telegram" in d["detail"]
    assert TOKEN not in d["detail"] and "bot•••" in d["detail"]

    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(401, {"ok": False, "description": "Unauthorized"}))
    assert client.post("/api/telegram/test", json={}).json() == {"ok": False, "detail": "Unauthorized"}
