"""Host/Origin guard, loopback-only dependency and secret redaction."""
import logging

import pytest
from fastapi import Depends
from fastapi.testclient import TestClient

from server import security
from tests.conftest import BASE_URL

SECRET = "sk_live_abc123SECRETvalue"
TOKEN = "123456:AAH-telegram/token+x"


# ── Host guard ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("host", ["evil.example:8000", "127.0.0.1:9999", "127.0.0.1",
                                  "localhost.evil.example:8000", ""])
def test_bad_host_is_rejected(client, host):
    r = client.get("/api/health", headers={"host": host})
    assert r.status_code == 403
    assert r.json()["error"] == "forbidden_host"


@pytest.mark.parametrize("host", ["127.0.0.1:8000", "localhost:8000", "LOCALHOST:8000", "[::1]:8000"])
def test_loopback_hosts_are_allowed(client, host):
    r = client.get("/api/health", headers={"host": host})
    assert r.status_code == 200 and r.json()["ok"] is True


@pytest.mark.parametrize("path", ["/", "/static/css/app.css", "/api/status"])
def test_guard_covers_pages_static_and_api(client, path):
    assert client.get(path, headers={"host": "rebind.example:8000"}).status_code == 403


def test_allowed_hosts_for_non_loopback_bind():
    assert "192.168.1.5:8000" in security.allowed_hosts("192.168.1.5", 8000)
    assert "0.0.0.0:8000" not in security.allowed_hosts("0.0.0.0", 8000)
    assert "localhost" in security.allowed_hosts("127.0.0.1", 80)


# ── Origin guard ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("origin", ["http://evil.example", "null", "http://localhost:8000",
                                    "https://127.0.0.1:8000"])
def test_cross_origin_write_is_rejected(client, origin):
    for method in ("post", "put", "patch", "delete"):
        r = getattr(client, method)("/api/health", headers={"origin": origin})
        assert r.status_code == 403, (method, origin)
        assert r.json()["error"] == "forbidden_origin"


def test_same_origin_write_passes_the_guard(client):
    r = client.post("/api/health", headers={"origin": BASE_URL})
    assert r.status_code == 405                      # reached the router
    assert r.json()["error"] == "method_not_allowed"


def test_write_without_origin_passes_the_guard(client):
    assert client.post("/api/health").status_code == 405


def test_cross_origin_get_is_allowed(client):
    assert client.get("/api/health", headers={"origin": "http://evil.example"}).status_code == 200


# ── loopback-only dependency ──────────────────────────────────────────────────

def test_require_loopback(app):
    app.add_api_route("/api/_test/loopback", lambda: {"ok": True},
                      dependencies=[Depends(security.require_loopback)], methods=["POST"])
    with TestClient(app, base_url=BASE_URL, client=("127.0.0.1", 1)) as local:
        assert local.post("/api/_test/loopback").status_code == 200
    with TestClient(app, base_url=BASE_URL, client=("::1", 1)) as local6:
        assert local6.post("/api/_test/loopback").status_code == 200
    with TestClient(app, base_url=BASE_URL, client=("10.0.0.7", 1)) as remote:
        r = remote.post("/api/_test/loopback")
        assert r.status_code == 403 and r.json()["error"] == "loopback_only"


# ── redaction ─────────────────────────────────────────────────────────────────

@pytest.fixture
def secrets(isolated, monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", SECRET)
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", TOKEN)
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "987654321")


def test_redact_masks_plain_and_url_encoded_values(secrets):
    from urllib.parse import quote
    text = (f"GET https://api.the-odds-api.com/v4/odds?apiKey={SECRET} failed; "
            f"https://api.telegram.org/bot{quote(TOKEN, safe='')}/sendMessage chat 987654321")
    out = security.redact(text)
    assert SECRET not in out and TOKEN not in out and quote(TOKEN, safe="") not in out
    assert "987654321" not in out
    assert out.count(security.MASK) == 3


def test_redact_reads_dotenv_file(isolated):
    (isolated / ".env").write_text(f"ODDS_API_KEY={SECRET}\n")
    assert security.redact(f"key={SECRET}") == f"key={security.MASK}"


def test_redact_without_secrets_is_identity(isolated):
    assert security.redact("nothing to hide") == "nothing to hide"
    assert security.redact(None) is None


def test_secret_never_in_status(secrets, client):
    r = client.get("/api/status")
    assert r.status_code == 200
    assert r.json()["secrets"] == {"odds_api": True, "telegram": True}
    for value in (SECRET, TOKEN, "987654321"):
        assert value not in r.text


def test_secret_never_in_error_detail(secrets, app):
    def boom():
        raise RuntimeError(f"upstream said no for apiKey={SECRET}")
    app.add_api_route("/api/_test/boom", boom)
    with TestClient(app, base_url=BASE_URL, raise_server_exceptions=False) as c:
        r = c.get("/api/_test/boom")
    assert r.status_code == 500
    body = r.json()
    assert body["error"] == "internal"
    assert SECRET not in r.text and security.MASK in body["detail"]


def test_log_filter_redacts_message_and_traceback(secrets):
    f = security.RedactFilter()
    rec = logging.LogRecord("x", logging.ERROR, __file__, 1, "token %s", (TOKEN,), None)
    f.filter(rec)
    assert TOKEN not in rec.getMessage()
    try:
        raise ValueError(f"bad key {SECRET}")
    except ValueError:
        import sys
        rec = logging.LogRecord("x", logging.ERROR, __file__, 1, "failed", (), sys.exc_info())
    f.filter(rec)
    assert SECRET not in logging.Formatter().format(rec)
