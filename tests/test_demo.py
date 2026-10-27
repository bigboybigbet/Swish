"""Demo mode: NBA_QUANT_DEMO_DATE replays a past date from the historical CSVs, offline."""
import os
from datetime import date

import pytest
import requests

from server.services import demo, espn, market, predict

pytestmark = pytest.mark.skipif(not os.path.isdir(demo.history_dir()),
                                reason="historical CSVs not available (NBA_HISTORY_DIR)")

DEMO = "2024-03-15"
CHA, PHX = 1610612766, 1610612756


@pytest.fixture
def demo_client(isolated, monkeypatch):
    """A demo-mode app with every network call blocked."""
    from fastapi.testclient import TestClient

    from server.app import create_app

    def no_network(*a, **k):
        raise AssertionError("demo mode must not touch the network")
    monkeypatch.setattr(requests, "get", no_network)
    monkeypatch.setattr(requests.Session, "request", no_network)
    monkeypatch.setenv("NBA_QUANT_DEMO_DATE", DEMO)
    app = create_app(host="127.0.0.1", port=8000)
    with TestClient(app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 50000)) as c:
        yield c


def test_install_swaps_and_restores_sources(demo_client):
    from engine import features
    assert demo.active()
    assert espn._fetch_scoreboard is demo.fetch_scoreboard
    assert features.leaguegamefinder is demo._FinderModule
    assert isinstance(predict._cache, demo.MemoryCache)
    assert demo_client.get("/api/status").json()["today_et"] == DEMO
    assert demo_client.get("/api/demo").json()["date"] == DEMO


def test_sources_restored_after_shutdown(isolated, monkeypatch):
    from fastapi.testclient import TestClient

    from data import cache, schedule
    from engine import features
    from server.app import create_app
    monkeypatch.setenv("NBA_QUANT_DEMO_DATE", DEMO)
    with TestClient(create_app(host="127.0.0.1", port=8000), base_url="http://127.0.0.1:8000"):
        assert demo.active()
    assert not demo.active()
    assert espn._fetch_scoreboard is schedule.fetch_scoreboard
    assert espn._cache is cache and predict._cache is cache and market._cache is cache
    assert features.leaguegamefinder.__name__ == "nba_api.stats.endpoints.leaguegamefinder"


def test_scoreboard_replays_finals_and_schedule(demo_client):
    today = espn.scoreboard(date(2024, 3, 15))
    pairs = {(e["away"]["abbr"], e["home"]["abbr"]) for e in today["events"]}
    assert ("PHX", "CHA") in pairs and len(today["events"]) == 6
    assert all(e["status"] == "pre" and e["home"]["score"] is None for e in today["events"])
    assert today["events"][0]["tipoff_utc"] == "2024-03-15T23:00:00Z"
    assert all(e["odds"] is None for e in today["events"])
    past = espn.scoreboard(date(2024, 3, 14))
    assert past["events"] and all(e["status"] == "post" and e["home"]["score"] for e in past["events"])
    assert espn.scoreboard(date(2024, 7, 20))["events"] == []          # offseason


def test_team_logs_stop_before_demo_date(monkeypatch):
    monkeypatch.setenv("NBA_QUANT_DEMO_DATE", DEMO)
    df = demo._LeagueGameFinder(team_id_nullable=CHA).get_data_frames()[0]
    assert len(df) > 1000 and df["GAME_DATE"].max() == "2024-03-13"
    assert set(df["TEAM_ID"]) == {CHA}


def test_predict_in_demo_mode(demo_client):
    r = demo_client.post("/api/predict", json={"home_id": CHA, "away_id": PHX,
                                                "include": ["hist", "poisson", "shap"]})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["game"]["date"] == DEMO and d["game"]["event_id"] == "0022300961"
    assert d["market"]["status"] == "no_odds_yet" and d["kelly"] is None
    assert d["availability"]["home"]["players"] == [] and d["availability"]["home"]["status"] == "ok"
    assert "demo_mode" in d["caveats"] and "past_date_current_elo" not in d["caveats"]
    # Elo replayed to the demo date: Phoenix (playoff team) above Charlotte (lottery)
    assert d["ratings"]["away"]["elo"] > d["ratings"]["home"]["elo"]
    assert d["win_prob"]["away"] > 0.5
    # a past date in the replay: the actual final is attached
    r = demo_client.post("/api/predict", json={"home_id": 1610612738, "away_id": PHX, "date": "2024-03-14"})
    d = r.json()
    assert d["game"]["result"] == {"home_score": 127, "away_score": 112}
    # team strength rates teams as of the game date, so no "current Elo" caveat
    assert d["mode"] == "team_strength"
    assert "past_date_current_elo" not in d["caveats"]
    assert d["ratings"]["home"]["kalman"]["rating"] > d["ratings"]["away"]["kalman"]["rating"]


def test_demo_form_and_odds(demo_client):
    form = demo_client.get(f"/api/teams/{CHA}/form").json()
    assert form["date"] == DEMO and form["days_rest"] >= 1
    o = demo_client.get("/api/odds", params={"home_id": CHA, "away_id": PHX}).json()
    assert o["status"] == "no_odds_yet" and o["home_ml"] is None
    assert demo_client.get("/api/odds", params={"home_id": PHX, "away_id": CHA}).json()["status"] == "not_found"


def test_install_without_data_fails_clearly(monkeypatch, tmp_path):
    monkeypatch.setenv("NBA_QUANT_DEMO_DATE", DEMO)
    monkeypatch.setenv("NBA_HISTORY_DIR", str(tmp_path / "nope"))
    with pytest.raises(RuntimeError, match="NBA_HISTORY_DIR"):
        demo.install()
    assert not demo.active()
