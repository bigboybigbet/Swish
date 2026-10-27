"""Engine/data changes for M1: E2 injury feed, E3 scoreboard, E4 Odds API, E5 cache age, E7 adjusted rows."""
import sqlite3
import time

import numpy as np
import pytest

import data.cache as cache
from data import injuries, odds, schedule
from engine import models

BOARD = {"events": [
    {"id": "401", "date": "2026-10-28T02:30Z",
     "status": {"period": 0, "type": {"state": "pre", "completed": False, "name": "STATUS_SCHEDULED"}},
     "competitions": [{
         "competitors": [
             {"homeAway": "home", "score": "0", "team": {"displayName": "Los Angeles Lakers", "abbreviation": "LAL"}},
             {"homeAway": "away", "score": "0", "team": {"displayName": "Golden State Warriors", "abbreviation": "GS"}}],
         "odds": [
             {"provider": {"name": "Book A"}, "details": "GS -2.5", "overUnder": 230.5, "spread": 2.5},
             {"provider": {"name": "Book B"}, "details": "GS -3", "overUnder": 231,
              "homeTeamOdds": {"moneyLine": 130}, "awayTeamOdds": {"current": {"moneyLine": "-150"}}}]}]},
    {"id": "402", "date": "2026-10-27T23:00Z",
     "status": {"period": 5, "type": {"state": "post", "completed": True, "name": "STATUS_FINAL"}},
     "competitions": [{
         "competitors": [
             {"homeAway": "home", "score": "101", "team": {"displayName": "Boston Celtics", "abbreviation": "BOS"}},
             {"homeAway": "away", "score": "99", "team": {"displayName": "New York Knicks", "abbreviation": "NY"}}],
         "odds": [{"provider": {"name": "ESPN BET"},
                   "moneyline": {"home": {"close": {"odds": "-140"}}, "away": {"close": {"odds": "+120"}}},
                   "pointSpread": {"home": {"close": {"line": "-3.5"}}},
                   "total": {"over": {"close": {"line": "o215.5"}}}}]}]},
    {"id": "bad", "competitions": []},
]}


@pytest.fixture
def fake_scoreboard(monkeypatch):
    calls = []

    def fetch(d):
        calls.append(d)
        return BOARD
    monkeypatch.setattr(schedule, "fetch_scoreboard", fetch)
    monkeypatch.setattr(odds, "fetch_scoreboard", fetch)
    return calls


# ── E3: scoreboard ───────────────────────────────────────────────────────────

def test_parse_scoreboard_keeps_ids_status_scores_and_odds():
    events = schedule.parse_scoreboard(BOARD)
    assert [e["event_id"] for e in events] == ["401", "402"]          # malformed event skipped
    pre, post = events
    assert pre["date"] == "2026-10-28T02:30Z" and pre["status"] == "pre"
    assert pre["home"] == {"name": "Los Angeles Lakers", "abbr": "LAL", "score": None}
    # the entry with moneylines wins; GS (away) is the favourite → home +3
    assert pre["odds"] == {"home_ml": 130, "away_ml": -150, "spread": 3.0, "total": 231.0,
                           "provider": "Book B"}
    assert post["status"] == "post" and post["completed"] and post["period"] == 5
    assert post["home"]["score"] == 101 and post["away"]["score"] == 99
    # newer nested odds fields
    assert post["odds"] == {"home_ml": -140, "away_ml": 120, "spread": -3.5, "total": 215.5,
                            "provider": "ESPN BET"}


@pytest.mark.parametrize("details,spread,expected", [
    ("LAL -4.5", None, -4.5), ("GS -4.5", None, 4.5), ("EVEN", 1.0, 0.0), ("", -6.0, -6.0), ("", None, None)])
def test_home_spread(details, spread, expected):
    assert schedule._home_spread({"details": details, "spread": spread}, "LAL", "GS") == expected


def test_parse_scoreboard_odds_are_nullable():
    board = {"events": [dict(BOARD["events"][0])]}
    board["events"][0]["competitions"] = [{**BOARD["events"][0]["competitions"][0], "odds": []}]
    assert schedule.parse_scoreboard(board)[0]["odds"] is None
    assert schedule.parse_scoreboard({}) == []


def test_old_helpers_rebuilt_on_scoreboard(fake_scoreboard):
    from datetime import date
    df = schedule.get_schedule_by_date(date(2026, 10, 27))
    assert list(df.columns) == ["Date", "Home Team", "Away Team"] and len(df) == 2
    results = schedule.get_past_results_espn(date(2026, 10, 27))
    assert results == [{"Home Team": "Boston Celtics", "Away Team": "New York Knicks", "H_Score": 101,
                        "A_Score": 99, "Actual_Winner": 1, "OT": True, "Periods": 5}]
    assert odds.fetch_odds_espn("Lakers", "Warriors", date(2026, 10, 27)) == (130, -150, "Book B")
    assert odds.fetch_odds_espn("Celtics", "Knicks") == (-140, 120, "ESPN BET")
    assert odds.fetch_odds_espn("Heat", "Bulls") is None


def test_fetch_scoreboard_raises_when_all_mirrors_fail(monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("offline")
    monkeypatch.setattr(schedule.requests, "get", boom)
    with pytest.raises(ConnectionError):
        from datetime import date
        schedule.fetch_scoreboard(date(2026, 10, 27))
    assert schedule.get_schedule_by_date(__import__("datetime").date(2026, 10, 27)) is None
    assert schedule.get_past_results_espn(__import__("datetime").date(2026, 10, 27)) == []


# ── E2: injury feed ──────────────────────────────────────────────────────────

FEED = {"injuries": [
    {"displayName": "Denver Nuggets", "injuries": [
        {"athlete": {"displayName": "Nikola Jokić"}, "status": "Day-To-Day", "shortComment": "Wrist"},
        {"athlete": {"displayName": "Nikola Jokić"}, "status": "Out"},
        {"athlete": {"displayName": "Bench Guy"}, "status": "Active"}]},
    {"displayName": "Boston Celtics", "injuries": [
        {"athlete": {"displayName": "Someone"}, "status": "Out"}]},
]}


def test_parse_injury_report():
    assert injuries.parse_injury_report(FEED, "Denver Nuggets") == [
        {"name": "Nikola Jokić", "status": "day-to-day", "detail": "Wrist"}]
    assert injuries.parse_injury_report(FEED, "Miami Heat") == []
    assert injuries.parse_injury_report({}, "Denver Nuggets") == []


def test_feed_errors_raise_but_wrapper_returns_empty(monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("offline")
    monkeypatch.setattr(injuries.requests, "get", boom)
    with pytest.raises(ConnectionError):
        injuries.fetch_injury_feed()
    assert injuries.get_injury_report("Denver Nuggets") == []
    monkeypatch.setattr(injuries, "fetch_injury_feed", lambda: FEED)
    assert injuries.get_injury_report("Boston Celtics")[0]["name"] == "Someone"


# ── E4: The Odds API, one call for all games ─────────────────────────────────

GAMES = [{"home_team": "Los Angeles Lakers", "away_team": "Golden State Warriors", "bookmakers": [
    {"title": "FanDuel", "markets": [{"key": "spreads", "outcomes": []}, {"key": "h2h", "outcomes": [
        {"name": "Golden State Warriors", "price": 140}, {"name": "Los Angeles Lakers", "price": -165}]}]}]}]


def test_odds_api_all_and_find(monkeypatch):
    calls = []

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return GAMES
    monkeypatch.setattr(odds.requests, "get", lambda url, params, timeout: calls.append(params) or Resp())
    games = odds.fetch_odds_api_all("KEY")
    assert games == GAMES and calls[0]["markets"] == "h2h" and calls[0]["apiKey"] == "KEY"
    assert odds.find_odds_api_game(games, "Lakers", "Warriors") == (-165, 140, "FanDuel")
    assert odds.find_odds_api_game(games, "Warriors", "Lakers") is None
    assert odds.fetch_odds_api("Lakers", "Warriors", "KEY") == (-165, 140, "FanDuel")
    assert len(calls) == 2


def test_odds_api_errors(monkeypatch):
    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": "quota"}
    monkeypatch.setattr(odds.requests, "get", lambda *a, **k: Resp())
    with pytest.raises(ValueError):
        odds.fetch_odds_api_all("KEY")
    assert odds.fetch_odds_api("Lakers", "Warriors", "KEY") is None


# ── E5: cache with age ───────────────────────────────────────────────────────

@pytest.fixture
def tmp_cache(tmp_path, monkeypatch):
    db = tmp_path / "cache.db"
    monkeypatch.setattr(cache, "CACHE_DB", str(db))
    cache.init()
    return db


def test_get_with_age(tmp_cache):
    before = time.time()
    cache.set("k", {"a": 1}, ttl=60)
    value, stored = cache.get_with_age("k")
    assert value == {"a": 1} and before <= stored <= time.time()
    cache.set("old", [1, 2], ttl=-1, stored_at=1000.0)
    assert cache.get("old") is None and cache.get_with_age("old") is None
    assert cache.get_with_age("old", allow_stale=True) == ([1, 2], 1000.0)
    assert cache.get_with_age("missing", allow_stale=True) is None


def test_cache_migrates_old_schema(tmp_path, monkeypatch):
    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE api_cache (key TEXT PRIMARY KEY, payload BLOB NOT NULL, expires REAL NOT NULL)")
    import pickle
    con.execute("INSERT INTO api_cache VALUES (?, ?, ?)", ("legacy", pickle.dumps("v"), time.time() + 60))
    con.commit()
    con.close()
    monkeypatch.setattr(cache, "CACHE_DB", str(db))
    cache.init()
    assert cache.get("legacy") == "v"
    assert cache.get_with_age("legacy") == ("v", None)
    cache.set("new", 1)
    assert cache.get_with_age("new")[1] is not None
    assert cache.stats()["total"] == 2


# ── E7: one injury adjustment, shared by every engine and SHAP ───────────────

def _team(strength):
    return {"ELO": 1550.0 + strength, "NET5": 2.0 + strength, "NET15": 1.0, "PTS10": 112.0,
            "PA10": 110.0, "EFG10": 0.54, "TOV10": 0.13, "ORB10": 0.25, "FTR10": 0.2, "PACE10": 99.0,
            "WPCT10": 0.6, "WIN_STREAK": 1, "DAYS_REST": 2, "IS_B2B": 0, "HOME_WPCT": 0.6,
            "AWAY_WPCT": 0.5, "PLUS_MINUS": 2.0, "REB": 44.0, "eFG_PCT": 0.54, "AST_TOV": 1.7,
            "TS_PCT": 0.57, "OREB_PCT": 0.25, "PTS": 112.0, "PTS_STD": 10.0, "PTS_ALLOWED": 110.0,
            "PTS_ALLOWED_STD": 10.0}


class Recorder:
    def __init__(self):
        self.rows = []

    def predict_proba(self, row):
        self.rows.append(row.copy())
        return np.array([[0.4, 0.6]])


def test_predict_detail_uses_adjusted_rows():
    hd, ad = _team(50), _team(0)
    inj = [{"name": "LeBron James", "status": "out"}]
    model = Recorder()
    d = models.predict_detail(hd, ad, model, inj, None)
    hs, as_, hf, af = models.injury_adjusted(hd, ad, inj, None)
    assert hf < 1.0 and af == 1.0 and d["hf"] == hf
    assert d["hs"]["ELO"] == hs["ELO"] < hd["ELO"]
    expected = models.build_matchup_row(hs, as_)
    assert d["row"].equals(expected) and model.rows[0].equals(expected)
    assert d["elo_diff"] == pytest.approx(hs["ELO"] - as_["ELO"])
    assert d["fp"] == pytest.approx(models.get_calibrator().predict(0.6, d["mc_p"], d["elo_diff"]))


def test_predict_hybrid_tuple_unchanged():
    out = models.predict_hybrid(_team(50), _team(0), Recorder())
    fp, spread, total, xgb_p, mc_p, conflict, mc, hf, af = out
    assert xgb_p == 0.6 and hf == af == 1.0 and spread == mc["spread_margin"] and total == mc["median_total"]
    assert conflict == ((xgb_p > 0.5) != (mc_p > 0.5))


def test_predict_detail_without_model():
    d = models.predict_detail(_team(50), _team(0), None)
    assert d["xgb_p"] is None and d["fp"] is None and d["conflict"] is None
    assert 0 < d["mc_p"] < 1
