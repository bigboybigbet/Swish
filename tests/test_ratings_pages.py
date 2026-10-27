"""Ratings and Accuracy APIs (demo mode, offline)."""
import os

import pytest

from server.services import demo
from tests.test_demo import demo_client  # noqa: F401  (fixture)

pytestmark = pytest.mark.skipif(not os.path.isdir(demo.history_dir()),
                                reason="historical CSVs not available (NBA_HISTORY_DIR)")


def test_team_ratings_rank_all_thirty(demo_client):
    t = demo_client.get("/api/ratings/teams").json()
    assert t["season"] == "2023-24" and t["last_game_date"] == "2024-03-14"
    assert len(t["teams"]) == 30 and [r["rank"] for r in t["teams"]] == list(range(1, 31))
    top = t["teams"][0]
    assert top["abbr"] == "BOS" and top["wins"] == 52 and top["losses"] == 14     # real record that day
    assert all(t["teams"][i]["rating"] >= t["teams"][i + 1]["rating"] for i in range(29))
    assert top["path"] and top["sd"] > 0


def test_accuracy_windows(demo_client):
    season = demo_client.get("/api/accuracy").json()
    assert season["window"] == "season" and season["n"] > 900
    assert 0.6 < season["acc"] < 0.7 and season["log_loss"] < 0.65 and season["brier"] < 0.25
    assert season["acc"] > season["baseline_home"]
    assert sum(c["n"] for c in season["calibration"]) == season["n"]
    assert sum(t["n"] for t in season["tiers"]) == season["n"]
    tiers = {t["tier"]: t["acc"] for t in season["tiers"]}
    assert tiers["high"] > tiers["low"]
    assert demo_client.get("/api/accuracy", params={"window": "30d"}).json()["n"] < season["n"]
    assert demo_client.get("/api/accuracy", params={"window": "all"}).json()["n"] > season["n"]
    assert demo_client.get("/api/accuracy", params={"window": "bad"}).status_code == 422


def test_player_impact_table(demo_client):
    p = demo_client.get("/api/ratings/players").json()
    assert p["players"] and p["players"][0]["impact"] >= p["players"][-1]["impact"]
