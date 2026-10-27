"""M0 scaffold: health, status, teams, error envelope, JSON cleaning, demo flag, entry point."""
import json
import math
from datetime import date

import numpy as np
import pytest
from fastapi.testclient import TestClient

from server import __version__, clock, config_store, jsonutil, paths, routers
from server.errors import ApiError
from server.services import teams
from tests.conftest import BASE_URL

STATUS_KEYS = {"version", "server_time_utc", "today_et", "demo", "model", "dataset", "elo",
               "calibrator", "autobacktest", "player_impact", "cache", "capabilities", "secrets",
               "thresholds", "kelly", "seasons", "active_jobs"}


# ── health / status ───────────────────────────────────────────────────────────

def test_health(client):
    assert client.get("/api/health").json() == {"ok": True, "version": __version__}


def test_status_without_model(client, monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "MODEL_FILE", tmp_path / "missing.joblib")
    d = client.get("/api/status").json()
    assert STATUS_KEYS <= set(d)
    assert d["model"]["exists"] is False and d["model"]["trained_at"] is None
    assert d["capabilities"]["predict_mode"] == "elo_only"
    assert d["capabilities"]["ratings"]["elo"] is True
    assert set(d["calibrator"]["weights"]) == {"xgb", "mc", "elo"}
    assert d["thresholds"] == {"pick": 0.55, "safe": 0.65, "vip": 0.65, "value_edge": 0.04}
    assert d["kelly"] == {"fraction": 0.25, "cap_pct": 5.0}
    assert d["secrets"] == {"odds_api": False, "telegram": False}
    assert d["demo"] is None and d["active_jobs"] == []
    assert d["server_time_utc"].endswith("Z")
    date.fromisoformat(d["today_et"])


def test_status_with_model_and_metrics(client, monkeypatch, tmp_path):
    model = tmp_path / "model.joblib"
    model.write_bytes(b"x")
    metrics = tmp_path / "model_metrics.json"
    metrics.write_text(json.dumps({"trained_at": "2026-10-20T10:00:00Z",
                                   "metrics": {"val_acc": 0.64, "val_loss": 0.63, "n_train": 3000}}))
    monkeypatch.setattr(paths, "MODEL_FILE", model)
    monkeypatch.setattr(paths, "MODEL_METRICS_FILE", metrics)
    d = client.get("/api/status").json()
    assert d["capabilities"]["predict_mode"] == "blend_v5"
    assert d["model"]["val_acc"] == 0.64 and d["model"]["n_train"] == 3000
    assert d["model"]["age_days"] > 0


def test_status_settings_file_overrides_defaults(client, isolated):
    (isolated / "settings.json").write_text(json.dumps({"thresholds": {"pick": 0.6}, "kelly": {"cap_pct": 2}}))
    d = client.get("/api/status").json()
    assert d["thresholds"]["pick"] == 0.6 and d["thresholds"]["safe"] == 0.65
    assert d["kelly"] == {"fraction": 0.25, "cap_pct": 2}


def test_status_demo_mode(client, monkeypatch):
    monkeypatch.setenv("NBA_QUANT_DEMO_DATE", "2024-03-15")
    d = client.get("/api/status").json()
    assert d["today_et"] == "2024-03-15"
    assert d["demo"] == {"date": "2024-03-15"}
    assert d["seasons"]["current"] == "2023-24"


def test_bad_demo_date_fails_fast(isolated, monkeypatch):
    from server.app import create_app
    monkeypatch.setenv("NBA_QUANT_DEMO_DATE", "15/03/2024")
    with pytest.raises(ValueError, match="NBA_QUANT_DEMO_DATE"):
        create_app(host="127.0.0.1", port=8000)


def test_clock(monkeypatch):
    monkeypatch.delenv("NBA_QUANT_DEMO_DATE", raising=False)
    assert clock.demo_date() is None and not clock.is_demo()
    monkeypatch.setenv("NBA_QUANT_DEMO_DATE", "2024-03-15")
    assert clock.today_et() == date(2024, 3, 15)
    assert clock.now_et().date() == date(2024, 3, 15)
    assert clock.iso_utc().endswith("Z")


# ── teams ─────────────────────────────────────────────────────────────────────

def test_teams_endpoint(client):
    rows = client.get("/api/teams").json()
    assert len(rows) == 30
    assert set(rows[0]) == {"id", "abbr", "name", "nickname", "city", "espn_name", "colors"}
    lac = next(r for r in rows if r["abbr"] == "LAC")
    assert lac["espn_name"] == "LA Clippers" and lac["nickname"] == "Clippers"
    assert set(lac["colors"]) == {"primary", "secondary"}


@pytest.mark.parametrize("name,abbr", [
    ("LA Clippers", "LAC"), ("Los Angeles Clippers", "LAC"), ("L.A. Clippers", "LAC"),
    ("GS", "GSW"), ("golden state warriors", "GSW"), ("Charlotte Bobcats", "CHA"),
    ("New Jersey Nets", "BKN"), ("NOH", "NOP"), ("UTAH", "UTA"), ("WSH", "WAS"), ("Lakers", "LAL"),
])
def test_resolve_team_names(name, abbr):
    assert teams.resolve(name)["abbr"] == abbr


def test_resolve_unknown_and_lookup():
    assert teams.resolve("Seattle Storm") is None and teams.resolve("") is None
    lac = teams.resolve("LAC")
    assert teams.lookup({"LA Clippers": 1601.2}, lac) == 1601.2          # Elo file key
    assert teams.lookup({}, lac, default=1500) == 1500
    with pytest.raises(ApiError) as e:
        teams.require(42)
    assert e.value.code == "team_not_found" and e.value.status == 404


# ── error envelope ────────────────────────────────────────────────────────────

def test_error_envelope(app):
    def busy():
        raise ApiError("busy", "A training job is running.", job_id="abc123")

    app.add_api_route("/api/_test/busy", busy, methods=["POST"])
    with TestClient(app, base_url=BASE_URL) as c:
        r = c.post("/api/_test/busy")
        assert r.status_code == 409
        assert r.json() == {"error": "busy", "detail": "A training job is running.", "job_id": "abc123"}
        assert c.get("/api/nope").json() == {"error": "not_found", "detail": "Not Found"}


def test_validation_envelope(app):
    def needs_int(team_id: int):
        return {"team_id": team_id}

    app.add_api_route("/api/_test/int", needs_int)
    with TestClient(app, base_url=BASE_URL) as c:
        r = c.get("/api/_test/int", params={"team_id": "abc"})
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "validation" and isinstance(body["detail"], str)
    assert "team_id" in body["detail"] and body["errors"][0]["loc"] == ["query", "team_id"]


def test_html_404_page(client):
    r = client.get("/no-such-page", headers={"accept": "text/html"})
    assert r.status_code == 404 and "Page not found" in r.text


# ── JSON cleaning ─────────────────────────────────────────────────────────────

def test_clean_converts_numpy_and_non_finite():
    out = jsonutil.clean({"a": np.float64("nan"), "b": float("inf"), "c": np.int64(3),
                          "d": np.float32(0.5), "e": np.array([1.0, np.nan]), "f": np.bool_(True),
                          "g": (1, None), "h": date(2024, 3, 15), 7: -float("inf")})
    assert out == {"a": None, "b": None, "c": 3, "d": 0.5, "e": [1.0, None], "f": True,
                   "g": [1, None], "h": "2024-03-15", "7": None}
    assert type(out["c"]) is int and type(out["f"]) is bool
    json.dumps(out, allow_nan=False)


def test_clean_pandas_values():
    import pandas as pd
    out = jsonutil.clean({"ts": pd.Timestamp("2024-03-15"), "nat": pd.NaT, "na": pd.NA,
                          "s": pd.Series([1.5, np.nan]).tolist()})
    assert out == {"ts": "2024-03-15T00:00:00", "nat": None, "na": None, "s": [1.5, None]}


def test_rounding_helpers():
    assert jsonutil.r4(0.123456) == 0.1235 and jsonutil.r1(4.56) == 4.6
    assert jsonutil.r4(float("nan")) is None and jsonutil.r1(None) is None


def test_nan_becomes_null_in_responses(app):
    app.add_api_route("/api/_test/nan", lambda: {"p": float("nan"), "q": [float("inf"), 1.0]})
    with TestClient(app, base_url=BASE_URL) as c:
        r = c.get("/api/_test/nan")
    assert r.status_code == 200
    assert "NaN" not in r.text and "Infinity" not in r.text
    assert r.json() == {"p": None, "q": [None, 1.0]}
    assert not any(isinstance(v, float) and math.isnan(v) for v in r.json().values())


# ── wiring ────────────────────────────────────────────────────────────────────

def test_router_discovery_order():
    names = [m.__name__.rsplit(".", 1)[1] for m in routers.discover()]
    assert {"meta", "teams", "pages"} <= set(names)
    assert names[-1] == "pages"


def test_docs_moved_under_api(client):
    assert client.get("/api/docs").status_code == 200
    assert client.get("/api/openapi.json").json()["info"]["version"] == __version__
    assert client.get("/docs").status_code == 404


def test_server_address_from_env(isolated, monkeypatch):
    assert config_store.server_address() == ("127.0.0.1", 8000)
    (isolated / ".env").write_text("NBA_PORT=8123\n")
    assert config_store.server_address() == ("127.0.0.1", 8123)
    monkeypatch.setenv("NBA_PORT", "8124")
    assert config_store.server_address()[1] == 8124


def test_main_runs_uvicorn_factory(isolated, monkeypatch):
    import uvicorn
    from server.__main__ import main
    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda *a, **kw: calls.append((a, kw)))
    main(["--port", "8123", "--no-browser"])
    (target,), kw = calls[0]
    assert target == "server.app:create_app" and kw["factory"] is True
    assert kw["workers"] == 1 and kw["port"] == 8123 and kw["host"] == "127.0.0.1"
    import os
    assert os.environ["NBA_PORT"] == "8123"


def test_main_rejects_bad_demo_date(isolated, monkeypatch):
    from server.__main__ import main
    monkeypatch.setenv("NBA_QUANT_DEMO_DATE", "nope")
    with pytest.raises(SystemExit) as e:
        main(["--no-browser"])
    assert e.value.code == 2
